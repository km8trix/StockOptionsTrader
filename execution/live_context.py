"""Canonical composition root for live E*TRADE execution.

The individual live components remain independently injectable and testable,
but production must never discover them through unrelated registries.  One
``LiveExecutionContext`` binds an auth/client generation and configured account
to exactly one broker, persistent local book, patient executor, optional trading
session, and scheduler.

Construction performs no network call and starts no thread.  Automation remains
explicit: ``configure_session`` builds a scheduler but never starts it.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from brokers.live_trader import LiveEtradeBroker
from brokers.local_book import LocalBook
from brokers.reconcile import CASH_TOLERANCE, reconcile
from core.models import AssetType
from execution.patient_executor import PatientExecutor
from execution.live_risk_gate import LiveRiskGate
from portfolio.manager import PortfolioManager
from portfolio.risk_reservations import RiskReservationLedger
from utils.live_session import LiveTradingSession
from utils.scheduler import LiveScheduler

logger = logging.getLogger(__name__)

_NONTERMINAL_ORDER_STATES = frozenset({
    'OPEN', 'PARTIAL', 'PARTIALLY_FILLED', 'CANCEL_REQUESTED',
    'PENDING', 'PENDING_REVIEW', 'QUEUED',
})
_TERMINAL_TRACKED_ORDER_STATES = frozenset({
    'CANCELLED', 'CANCELED', 'EXECUTED', 'FILLED', 'REJECTED', 'EXPIRED',
})


@dataclass(frozen=True)
class LiveContextIdentity:
    """Immutable identity of one process-local execution graph."""

    db_path: str
    env: str
    account_id_key: str
    auth_manager_id: int


class LiveContextClosed(RuntimeError):
    """Raised when new work reaches a closing/closed context."""


class PlacedOrderTrackingError(RuntimeError):
    """The broker accepted an order that the durable book could not track."""

    def __init__(self, order_id, cause: Exception):
        self.order_id = order_id
        super().__init__(
            f'placed order {order_id} could not be persisted: '
            f'{type(cause).__name__}: {cause}')


class LiveExecutionContext:
    """Own the complete live execution graph for one account.

    The context uses short operation leases so shutdown cannot invalidate
    components between a route's readiness check and its network operation.
    Network calls never run while the lifecycle lock is held.
    """

    def __init__(self, *, auth_manager, client, kill_switch, audit,
                 account_id_key: str, db_path: Optional[str] = None,
                 broker=None, local_book=None, executor=None,
                 reservation_ledger=None, reservation_gate=None,
                 quote_fn=None):
        account = str(account_id_key or '').strip()
        if not account:
            raise ValueError('account_id_key is required')
        resolved_db = os.path.realpath(
            db_path or getattr(auth_manager, 'db_path', None)
            or os.environ.get('TRADING_DB_PATH') or 'trading_data.db')
        env = str(getattr(auth_manager, 'env', None)
                  or os.environ.get('ETRADE_ENV', 'sandbox')).strip().lower()
        self.identity = LiveContextIdentity(
            db_path=resolved_db,
            env=env,
            account_id_key=account,
            auth_manager_id=id(auth_manager),
        )
        self.auth_manager = auth_manager
        self.client = client
        self.kill_switch = kill_switch
        self.audit = audit
        self.account_id_key = account
        self.broker = broker or LiveEtradeBroker(client, account)
        if local_book is None:
            try:
                local_book = LocalBook(
                    resolved_db, env=env, account_id_key=account)
            except TypeError:
                # Compatibility while migrating older injected LocalBook
                # implementations that predate account scoping.
                local_book = LocalBook(resolved_db, env=env)
        self.local_book = local_book
        self.reservation_ledger = (
            reservation_ledger or RiskReservationLedger(
                resolved_db, env=env, account_id_key=account))
        existing_gate = getattr(client, 'reservation_gate', None)
        if reservation_gate is None and existing_gate is not None:
            reservation_gate = existing_gate
        self.reservation_gate = (
            reservation_gate or LiveRiskGate(
                client, self.reservation_ledger, kill_switch, audit, account,
                local_book=self.local_book))
        if existing_gate is not None and existing_gate is not self.reservation_gate:
            raise ValueError(
                'client is already bound to a different reservation gate')
        client.reservation_gate = self.reservation_gate
        self.executor = executor or PatientExecutor(
            self.broker, quote_fn or self._equity_quote,
            kill_switch=kill_switch)

        self.session: Optional[LiveTradingSession] = None
        self.scheduler: Optional[LiveScheduler] = None
        self.verified_deployment = None
        self.last_reconciliation: Optional[Dict] = None

        self._condition = threading.Condition(threading.RLock())
        self._state = 'ready'
        self._active_operations = 0

    # ------------------------------------------------------------------
    # Lifecycle / operation leases
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @contextmanager
    def operation(self, name: str) -> Iterator[None]:
        """Lease the context for one route/session operation."""
        with self._condition:
            if self._state != 'ready':
                raise LiveContextClosed(
                    f'live execution context is {self._state}; refusing {name}')
            self._active_operations += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_operations -= 1
                self._condition.notify_all()

    def shutdown(self, timeout: float = 10.0) -> bool:
        """Stop owned workers and permanently close this context.

        Idempotent.  New leases are refused as soon as ``closing`` is set;
        existing short route operations are allowed to drain.
        """
        with self._condition:
            if self._state == 'closed':
                return False
            if self._state == 'closing':
                self._condition.wait_for(
                    lambda: self._state == 'closed', timeout=timeout)
                return False
            self._state = 'closing'
            scheduler = self.scheduler
            executor = self.executor

        # Signal execution before joining the scheduler: evaluate_once may be
        # blocked inside the executor, and must wake before scheduler.stop joins.
        close_executor = getattr(executor, 'close', None)
        if callable(close_executor):
            close_executor(timeout=timeout)
        else:
            executor.stop()
        if scheduler is not None:
            scheduler.stop(join_timeout=timeout)

        with self._condition:
            self._condition.wait_for(
                lambda: self._active_operations == 0, timeout=timeout)
            if self._active_operations:
                logger.error(
                    'Live context closed with %d operation(s) still active',
                    self._active_operations)
            self._state = 'closed'
            self._condition.notify_all()
        return True

    # ------------------------------------------------------------------
    # Construction of optional automation
    # ------------------------------------------------------------------
    def configure_session(self, *, portfolio: PortfolioManager, data_fn,
                          desk=None, orchestrator=None,
                          interval_minutes: float = 15,
                          market_hours=None,
                          verified_deployment=None) -> LiveTradingSession:
        """Build (but never start) a session and scheduler over owned parts."""
        if (desk is None) == (orchestrator is None):
            raise ValueError('provide exactly one of desk or orchestrator')

        execution_guard = None
        if self.identity.env == 'production' and verified_deployment is None:
            raise ValueError(
                'production automation requires a verified deployment')
        if verified_deployment is not None:
            from deployment.live import VerifiedFoundationDeployment
            if not isinstance(verified_deployment,
                              VerifiedFoundationDeployment):
                raise ValueError('invalid verified deployment capability')
            if orchestrator is not None:
                raise ValueError(
                    'verified Foundation deployment cannot bind an orchestrator')
            execution_guard = verified_deployment.bind(
                self, desk, interval_minutes)
            envelope_binder = getattr(
                execution_guard, 'bind_executor', None)
            if not callable(envelope_binder):
                raise ValueError(
                    'verified deployment guard cannot bind notional envelopes')
            envelope_binder(self.executor)

        with self.operation('configure_session'):
            previous = self.scheduler
            if previous is not None:
                previous.stop()
            session = LiveTradingSession(
                desk=desk,
                orchestrator=orchestrator,
                broker=self.broker,
                portfolio=portfolio,
                data_fn=data_fn,
                executor=self.executor,
                audit=self.audit,
                kill_switch=self.kill_switch,
                auth_manager=self.auth_manager,
                local_book=self.local_book,
                enforce_market_hours=True,
                execution_guard=execution_guard,
            )
            scheduler_kwargs = {
                'market_hours': market_hours,
                'interval_minutes': interval_minutes,
                'audit': self.audit,
            }
            if verified_deployment is not None:
                # A controlled production deployment gets one attempt during
                # its ARMED activation handshake.  Retrying an exception five
                # times or beginning a second cycle before the controller's
                # post-cycle reconciliation would leave ambiguous permission.
                scheduler_kwargs['max_consecutive_errors'] = 1
                scheduler_kwargs['hold_after_first_cycle'] = True
            scheduler = LiveScheduler(session, **scheduler_kwargs)
            with self._condition:
                self.session = session
                self.scheduler = scheduler
                self.verified_deployment = verified_deployment
            return session

    # ------------------------------------------------------------------
    # Quotes / orders
    # ------------------------------------------------------------------
    def _equity_quote(self, instrument) -> Dict[str, float]:
        """Bid/ask adapter for patient EQUITY execution only.

        The installed E*TRADE quote surface is symbol/underlying based and is
        not a valid option-premium or multi-leg package quote source.  Those
        instruments fail closed until the atomic-structure phase supplies a
        contract/package adapter.
        """
        if isinstance(instrument, (list, tuple)) or getattr(
                instrument, 'asset_type', None) is not AssetType.STOCK:
            raise ValueError(
                'patient execution requires a real option/package quote '
                'adapter; underlying quotes are forbidden')
        quote = self.client.get_quotes([instrument.symbol]).get(
            instrument.symbol, {})
        if self.verified_deployment is not None:
            from deployment.live import validate_realtime_equity_quote
            market = validate_realtime_equity_quote(
                quote, symbol=instrument.symbol,
                now=datetime.now(timezone.utc))
            return {"bid": market["bid"], "ask": market["ask"]}
        bid, ask = quote.get('bid'), quote.get('ask')
        if bid is None or ask is None:
            raise ValueError(f'no bid/ask quote for {instrument.symbol}')
        bid, ask = float(bid), float(ask)
        if (not math.isfinite(bid) or not math.isfinite(ask)
                or bid <= 0 or ask <= 0 or bid > ask):
            raise ValueError(f'invalid bid/ask quote for {instrument.symbol}')
        return {'bid': bid, 'ask': ask}

    @staticmethod
    def _normalise_order(order: Dict) -> Optional[Dict]:
        details = order.get('OrderDetail') or []
        detail = details[0] if details else {}
        status = str(detail.get('status') or order.get('status') or '').upper()
        if status not in _NONTERMINAL_ORDER_STATES:
            return None
        instruments = [inst for block in details
                       for inst in (block.get('Instrument') or [])]
        if not instruments:
            instruments = detail.get('Instrument') or []
        items = []
        ordered = filled = 0.0
        actions = []
        for inst in instruments:
            product = inst.get('Product') or {}
            quantity = float(inst.get('orderedQuantity', inst.get(
                'quantity', 0)) or 0)
            filled_qty = float(inst.get('filledQuantity', 0) or 0)
            ordered = max(ordered, quantity)
            filled = max(filled, filled_qty)
            action = inst.get('orderAction')
            if action:
                actions.append(str(action))
            items.append({
                'symbol': product.get('symbol'),
                'security_type': product.get('securityType'),
                'action': action,
                'quantity': quantity,
                'filled_quantity': filled_qty,
            })
        order_id = order.get('orderId') or detail.get('orderId')
        return {
            'order_id': str(order_id) if order_id is not None else None,
            'client_order_id': (order.get('clientOrderId')
                                or detail.get('clientOrderId')),
            'status': status,
            'side': '/'.join(actions) if actions else None,
            'quantity': ordered,
            'filled_quantity': filled,
            'remaining_quantity': max(0.0, ordered - filled),
            'limit_price': (detail.get('limitPrice')
                            if detail.get('limitPrice') is not None
                            else order.get('limitPrice')),
            'instruments': items,
            'recovered': True,
        }

    def working_orders(self) -> list[Dict]:
        """Broker-authoritative nonterminal orders, enriched by the executor."""
        with self.operation('working_orders'):
            self._sync_tracked_orders()
            raw = self.client.list_orders(self.account_id_key, count=100)
            normalised = [self._normalise_order(order) for order in raw]
            by_id = {item['order_id']: item for item in normalised
                     if item is not None}
            getter = getattr(self.executor, 'working_orders', None)
            for active in list(getter() if callable(getter) else []):
                order_id = active.get('order_id')
                if order_id in by_id:
                    by_id[order_id].update(active)
                    by_id[order_id]['recovered'] = False
                else:
                    copy = dict(active)
                    copy.setdefault('recovered', False)
                    by_id[order_id] = copy
            for order_id, item in by_id.items():
                reservation = self.reservation_ledger.reservation_for_order(
                    order_id) if order_id is not None else None
                if reservation is not None:
                    item['reservation'] = reservation
            return sorted(by_id.values(), key=lambda item: str(
                item.get('order_id') or ''))

    def risk_estimate(self, order_request: Dict) -> Dict:
        """Fresh, non-consuming reservation estimate for an order preview."""
        with self.operation('risk_estimate'):
            return self.reservation_gate.estimate(order_request)

    def reservation_snapshot(self) -> Dict:
        """Read-only durable pending-risk state for status and operations."""
        with self.operation('reservation_snapshot'):
            return self.reservation_gate.snapshot()

    def cancel_order(self, order_id: str) -> bool:
        """Cancel through the context-bound account; callable under kill switch."""
        with self.operation('cancel_order'):
            cancel_current = getattr(self.executor, 'cancel_current', None)
            if callable(cancel_current) and cancel_current(str(order_id)):
                # The executor remains the sole owner of cancel confirmation
                # and fill-race banking for its active order.
                return True
            return self.broker.cancel_order(str(order_id))

    def preview_order(self, order_request: Dict) -> Dict:
        """Validate and preview on the context-bound account generation."""
        with self.operation('preview_order'):
            self.broker.validate_order_request(order_request)
            return self.client.preview_order(
                self.account_id_key, order_request)

    def place_order(self, order_request: Dict, preview_ids: list) -> Dict:
        """Place and durably register before returning accepted success."""
        with self.operation('place_order'):
            result = self.client.place_order(
                self.account_id_key, order_request, preview_ids)
            order_id = (result.get('order_id')
                        if isinstance(result, dict) else None)
            try:
                self._track_placed_order(order_id, order_request)
            except Exception as exc:
                if self.kill_switch is not None:
                    self.kill_switch.engage(
                        f'Placed order {order_id} could not be persisted: '
                        f'{type(exc).__name__}: {exc}',
                        'live_context',
                    )
                raise PlacedOrderTrackingError(order_id, exc) from exc
            return result

    def track_placed_order(self, order_id, order_request: Dict) -> None:
        """Persist a newly accepted GUI order for restart-safe fill booking."""
        with self.operation('track_placed_order'):
            self._track_placed_order(order_id, order_request)

    def _track_placed_order(self, order_id, order_request: Dict) -> None:
        tracker = getattr(self.local_book, 'track_order', None)
        if not callable(tracker):
            raise RuntimeError('LocalBook does not support order tracking')
        tracker(str(order_id), order_request)

    def sync_tracked_orders(self) -> None:
        """Poll durable nonterminal orders and bank unseen cumulative fills."""
        with self.operation('sync_tracked_orders'):
            self._sync_tracked_orders()

    def _sync_tracked_orders(self) -> None:
        getter = getattr(self.local_book, 'tracked_orders', None)
        applier = getattr(self.local_book, 'apply_order_status', None)
        if not callable(getter) or not callable(applier):
            raise RuntimeError(
                'LocalBook does not support durable order synchronization')
        for tracked in getter():
            if str(tracked.get('status') or '').upper() in (
                    _TERMINAL_TRACKED_ORDER_STATES):
                continue
            order_id = tracked['order_id']
            status = self.broker.order_status(order_id)
            if status is None:
                continue
            applier(order_id, status)

    # ------------------------------------------------------------------
    # Persistent book / reconciliation
    # ------------------------------------------------------------------
    def bootstrap_local_book(self) -> Dict:
        """Explicitly adopt the broker snapshot into an uninitialized book."""
        with self.operation('bootstrap_local_book'):
            status = self.broker.get_portfolio_status()
            positions = {row['symbol']: float(row['quantity'])
                         for row in status.get('positions', [])}
            bootstrap = getattr(self.local_book, 'bootstrap', None)
            if not callable(bootstrap):
                raise RuntimeError('LocalBook does not support safe bootstrap')
            created = bool(bootstrap(
                positions, float(status.get('cash', 0.0))))
            if created and self.audit is not None:
                self.audit.append('live_context', 'local_book_bootstrapped', {
                    'account_id_key': self.account_id_key,
                    'position_count': len(positions),
                })
            snapshot = self.local_book.snapshot()
            snapshot['bootstrapped'] = created
            return snapshot

    def run_reconciliation(self,
                           cash_tolerance: float = CASH_TOLERANCE) -> Dict:
        """Sync tracked fills, atomically snapshot the book, then reconcile."""
        with self.operation('reconciliation'):
            try:
                self._sync_tracked_orders()
                snapshotter = getattr(
                    self.local_book, 'reconciliation_snapshot', None)
                if callable(snapshotter):
                    snapshot = snapshotter()
                    if not snapshot.get('initialized', True):
                        raise RuntimeError(
                            'local book is uninitialized; explicit bootstrap '
                            'is required before reconciliation')
                    local_positions = snapshot['positions']
                    local_cash = snapshot['cash']
                else:
                    local_positions = self.local_book.positions()
                    local_cash = self.local_book.cash()
                result = reconcile(
                    local_positions, local_cash, self.broker,
                    cash_tolerance=cash_tolerance)
            except Exception as exc:
                if self.kill_switch is not None:
                    self.kill_switch.engage(
                        f'reconciliation could not run: '
                        f'{type(exc).__name__}: {exc}',
                        'live_context')
                raise

            self.last_reconciliation = result
            if self.audit is not None:
                self.audit.append('live_context', 'reconciliation', {
                    'ok': result['ok'],
                    'mismatches': result['mismatches'],
                    'checked_at': result['checked_at'],
                })
            if not result['ok'] and self.kill_switch is not None:
                self.kill_switch.engage(
                    f"reconciliation mismatch: "
                    f"{len(result['mismatches'])} difference(s) vs broker",
                    'live_context')
            return result
