"""
API routes for the Live Trading page (Phase 9, contract C20).

Frontend-owned routes over the backend Phase 9 surfaces:

* C16  brokers/etrade_auth.EtradeAuthManager  (OAuth 1.0a lifecycle)
* C17  utils/kill_switch.KillSwitch           (persistent, audit-logged)
* C18  utils/audit.AuditLog                   (append-only sha256 chain)
* C19  brokers/reconcile.reconcile            (local book vs broker)

Every backend import is guarded with the 503-with-reason pattern
(precedent: api_trading.LIVE_BROKER_IMPORT_ERROR): a missing or broken
surface returns ``503 {'error': ..., 'reason': <import error>}`` instead of
crashing the app, so this blueprint registers cleanly while the parallel
backend task lands.

NO route here performs network I/O itself — the auth manager and broker own
their (injected) transports, and nothing in this module ever constructs an
HTTP session.

Response shapes consumed by gui/static/js/live.js:

  GET  /api/live/status      -> {'auth': <C16 status>, 'env': str|None,
                                 'kill_switch': {'engaged': bool},
                                 'reconciliation': <last result>|None}
  POST /api/live/auth/start  -> {'authorize_url': str, 'auth': <C16 status>}
  POST /api/live/auth/verifier {code}      -> {'auth': <C16 status>}
  POST /api/live/auth/renew  (additive)    -> {'renewed': bool, 'auth': ...}
  POST /api/live/auth/disconnect (additive)-> {'auth': <C16 status>}
  POST /api/live/killswitch {engaged, reason} -> {'engaged': bool}
  GET  /api/live/audit?limit=&offset=&event_type=&verify=
        -> {'entries': [...], 'limit', 'offset', 'event_type'[, 'verify']}
  POST /api/live/reconcile   -> <C19 result> + {'kill_switch_engaged': bool}
  GET  /api/live/orders (additive)         -> {'orders': [...], 'count': int}
  POST /api/live/orders/<id>/cancel (additive)
  GET  /api/live/scheduler   -> <LiveScheduler.status()> + interval_minutes
  POST /api/live/scheduler {action: 'start'|'stop', interval_minutes?}
        -> same shape as GET (503 until a scheduler is configured)
  GET  /api/live/keepalive   -> <TokenKeepAliveScheduler.status()>
  POST /api/live/keepalive {action: 'start'|'stop', interval_minutes?}
        -> same shape as GET (renew-only loop; auto-starts on connect)
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request
import logging
import os
import secrets
import threading
import time as _time
from datetime import datetime, timezone

from gui.routes import live_order_support as _order_support

try:
    from brokers.circuit_breaker import (DailyLossCircuitBreaker,
                                         DailyLossGate, extract_account_value)
except Exception:  # noqa: BLE001 - daily-loss gate is optional on the GUI client
    DailyLossCircuitBreaker = DailyLossGate = extract_account_value = None

try:
    from utils.market_hours import MarketHours
except Exception:  # noqa: BLE001 - market-hours guard is optional
    MarketHours = None

logger = logging.getLogger(__name__)

# ==================== GUARDED BACKEND IMPORTS (503 pattern) ====================

ETRADE_AUTH_IMPORT_ERROR: str | None
try:
    from brokers.etrade_auth import (EtradeAuthError, EtradeAuthManager,
                                      EtradeNotConfigured)
    ETRADE_AUTH_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - any import failure disables the surface
    EtradeAuthManager = None
    ETRADE_AUTH_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import EtradeAuthManager — live auth is unavailable: %s',
        ETRADE_AUTH_IMPORT_ERROR,
    )

    class EtradeAuthError(Exception):
        """Placeholder base so except-clauses stay valid while the backend
        surface is absent (never raised through this fallback)."""

    class EtradeNotConfigured(EtradeAuthError):
        """Placeholder so except-clauses stay valid while the backend
        surface is absent (never raised through this fallback)."""

KILL_SWITCH_IMPORT_ERROR: str | None
try:
    from utils.kill_switch import KillSwitch
    KILL_SWITCH_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    KillSwitch = None
    KILL_SWITCH_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import KillSwitch — kill switch is unavailable: %s',
        KILL_SWITCH_IMPORT_ERROR,
    )

AUDIT_IMPORT_ERROR: str | None
try:
    from utils.audit import AuditLog
    AUDIT_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    AuditLog = None
    AUDIT_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import AuditLog — audit log is unavailable: %s',
        AUDIT_IMPORT_ERROR,
    )

RECONCILE_IMPORT_ERROR: str | None
try:
    from brokers.reconcile import reconcile
    RECONCILE_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    reconcile = None
    RECONCILE_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import reconcile — reconciliation is unavailable: %s',
        RECONCILE_IMPORT_ERROR,
    )

PARITY_IMPORT_ERROR: str | None
try:
    from backtesting.parity_harness import parity_report
    PARITY_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    parity_report = None
    PARITY_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import parity_harness — parity is unavailable: %s',
        PARITY_IMPORT_ERROR,
    )

SCHEDULER_IMPORT_ERROR: str | None
try:
    from utils.scheduler import LiveScheduler
    SCHEDULER_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    LiveScheduler = None
    SCHEDULER_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import LiveScheduler — the scheduler is unavailable: %s',
        SCHEDULER_IMPORT_ERROR,
    )

KEEPALIVE_IMPORT_ERROR: str | None
try:
    from utils.token_keepalive import (
        KEEPALIVE_INTERVAL_MAX,
        KEEPALIVE_INTERVAL_MIN,
        TokenKeepAliveScheduler,
    )
    KEEPALIVE_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    TokenKeepAliveScheduler = None
    KEEPALIVE_INTERVAL_MIN = 1
    KEEPALIVE_INTERVAL_MAX = 60  # mirror the real constant if the import fails
    KEEPALIVE_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import TokenKeepAliveScheduler — keep-alive is '
        'unavailable: %s', KEEPALIVE_IMPORT_ERROR,
    )

# The trading client + its typed exceptions and order-request builders.
# A single guarded import block: if any of these is missing the whole
# trade surface (accounts/quotes/order ticket) answers 503-with-reason,
# exactly like the surfaces above.
CLIENT_IMPORT_ERROR: str | None
try:
    from brokers.etrade_client import (
        EtradeClient,
        EtradeApiError,
        EtradeOrderRejected,
        EtradeRateLimited,
        EtradeUnavailable,
        build_equity_order,
        build_option_order,
        build_spread_order,
    )
    from brokers.etrade_auth import EtradeAuthExpired
    from brokers.live_trader import LiveEtradeBroker, PriceSanityError
    from execution.live_context import (LiveContextClosed,
                                        LiveExecutionContext,
                                        PlacedOrderTrackingError)
    from execution.live_risk_gate import (
        ReservationPersistenceError,
        ReservationStateUnknown,
        RiskValuationUnavailable,
        UnsupportedRiskOrder,
    )
    from portfolio.risk_reservations import RiskCapacityExceeded
    from utils.kill_switch import KillSwitchEngaged
    CLIENT_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    EtradeClient = None
    LiveEtradeBroker = None
    LiveExecutionContext = None
    build_equity_order = build_option_order = build_spread_order = None
    CLIENT_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import EtradeClient — live trading routes are '
        'unavailable: %s', CLIENT_IMPORT_ERROR,
    )

    class EtradeApiError(Exception):
        """Placeholder so except-clauses stay valid while the client
        surface is absent (never raised through this fallback)."""

    class EtradeOrderRejected(EtradeApiError):
        pass

    class EtradeRateLimited(EtradeApiError):
        pass

    class EtradeUnavailable(EtradeApiError):
        pass

    class EtradeAuthExpired(Exception):
        pass

    class KillSwitchEngaged(Exception):
        pass

    class PriceSanityError(ValueError):
        pass

    class LiveContextClosed(RuntimeError):
        pass

    class PlacedOrderTrackingError(RuntimeError):
        order_id = None

    class ReservationPersistenceError(RuntimeError):
        pass

    class ReservationStateUnknown(RuntimeError):
        pass

    class RiskValuationUnavailable(RuntimeError):
        pass

    class UnsupportedRiskOrder(RuntimeError):
        pass

    class RiskCapacityExceeded(RuntimeError):
        breaches = []

live_bp = Blueprint('live', __name__, url_prefix='/api/live')

# ==================== SHARED SINGLETONS ====================
# One auth manager / kill switch / audit log per GUI process. The kill
# switch and audit log share the trading SQLite file (TRADING_DB_PATH,
# same resolution as gui.globals.get_db) so state survives restarts.

# RLock (not Lock): get_client() holds this while calling get_kill_switch()
# and get_audit_log(), which re-acquire it to lazily build their own
# singletons — a plain Lock self-deadlocks on that first nested call.
_singleton_lock = threading.RLock()
_auth_manager = None
_kill_switch = None
_audit_log = None
# Construction-time failure (e.g. EtradeNotConfigured for a production env
# without the explicit ack gate) — preserved as the 503 reason.
_auth_construct_error: str | None = None

# Last reconciliation result for GET /api/live/status (per-process; the
# durable record is the audit log, which the backend live session writes).
_last_reconciliation: dict | None = None


def _live_db_path() -> str:
    """SQLite path for kill-switch/audit state (mirrors gui.globals)."""
    return os.environ.get('TRADING_DB_PATH') or 'trading_data.db'


def get_auth_manager():
    """Shared EtradeAuthManager, or None when the surface is unavailable.

    The manager reads its own ETRADE_* configuration from the environment
    (C16: status() reports 'unconfigured' when consumer keys are absent) —
    this module never reads or logs key material itself.
    """
    global _auth_manager, _auth_construct_error
    if EtradeAuthManager is None:
        return None
    if _auth_manager is None:
        with _singleton_lock:
            if _auth_manager is None:
                try:
                    _auth_manager = EtradeAuthManager(
                        db_path=_live_db_path())
                except Exception as e:
                    _auth_construct_error = f'{type(e).__name__}: {e}'
                    logger.error('EtradeAuthManager construction failed',
                                 exc_info=True)
                    return None
    return _auth_manager


def get_kill_switch():
    """Shared KillSwitch, or None when the surface is unavailable."""
    global _kill_switch
    if KillSwitch is None:
        return None
    if _kill_switch is None:
        with _singleton_lock:
            if _kill_switch is None:
                try:
                    _kill_switch = KillSwitch(_live_db_path())
                except Exception:
                    logger.error('KillSwitch construction failed',
                                 exc_info=True)
                    return None
    return _kill_switch


def get_audit_log():
    """Shared AuditLog, or None when the surface is unavailable."""
    global _audit_log
    if AuditLog is None:
        return None
    if _audit_log is None:
        with _singleton_lock:
            if _audit_log is None:
                try:
                    _audit_log = AuditLog(_live_db_path())
                except Exception:
                    logger.error('AuditLog construction failed', exc_info=True)
                    return None
    return _audit_log


def auth_unavailable_reason() -> str:
    """Reason string for 503s when the auth surface is unusable."""
    return (ETRADE_AUTH_IMPORT_ERROR or _auth_construct_error
            or 'EtradeAuthManager construction failed')


# Cached EtradeClient, plus the auth-manager identity it was built over.
# The client binds to ONE auth manager (its signing session + base URL);
# if the auth manager is ever replaced (e.g. a reconnect rebuilds it),
# the cached client is stale and must be rebuilt over the new manager.
_client = None
_client_auth_manager = None
_execution_context = None


def get_client():
    """Shared EtradeClient over the live auth manager + kill switch + audit,
    or None when the client surface or the auth manager is unavailable.

    The client is cached and reused, but rebuilt whenever the underlying
    auth manager identity changes (a reconnect) so it never signs with a
    dead session. The kill switch is wired in so preview/place raise
    KillSwitchEngaged FIRST — the route layer never bypasses it, and a
    daily-loss gate is wired so the -2% rail PRE-blocks orders (not only
    retroactively once a breach engages the kill switch).
    """
    global _client, _client_auth_manager
    if EtradeClient is None:
        return None
    manager = get_auth_manager()
    if manager is None:
        return None
    with _singleton_lock:
        if _client is None or _client_auth_manager is not manager:
            try:
                client = EtradeClient(
                    manager,
                    kill_switch=get_kill_switch(),
                    audit=get_audit_log(),
                )
                _wire_daily_loss_gate(client)
                _client = client
                _client_auth_manager = manager
            except Exception:
                logger.error('EtradeClient construction failed', exc_info=True)
                _client = None
                _client_auth_manager = None
                return None
    return _client


def get_execution_context(create: bool = True):
    """Canonical live execution graph for the configured account.

    Construction is lazy, performs no network I/O, and starts no thread.  The
    identity includes the auth-manager object, account, environment, and DB
    path; changing any of them closes the old graph before a replacement can
    be returned.
    """
    global _execution_context
    if LiveExecutionContext is None:
        return None
    if not create:
        return _execution_context

    manager = get_auth_manager()
    client = get_client()
    account = str(os.environ.get('ETRADE_ACCOUNT_ID_KEY') or '').strip()
    if manager is None or client is None or not account:
        return None
    identity = (
        os.path.realpath(_live_db_path()),
        str(getattr(manager, 'env', 'sandbox')).strip().lower(),
        account,
        id(manager),
    )
    with _singleton_lock:
        current = _execution_context
        current_identity = None
        if current is not None:
            value = current.identity
            current_identity = (
                value.db_path, value.env, value.account_id_key,
                value.auth_manager_id)
        if current is not None and current_identity != identity:
            current.shutdown()
            _execution_context = None
        if _execution_context is None:
            try:
                _execution_context = LiveExecutionContext(
                    auth_manager=manager,
                    client=client,
                    kill_switch=get_kill_switch(),
                    audit=get_audit_log(),
                    account_id_key=account,
                    db_path=_live_db_path(),
                )
            except Exception:
                logger.error('LiveExecutionContext construction failed',
                             exc_info=True)
                _execution_context = None
        return _execution_context


def reset_execution_context(shutdown: bool = True) -> None:
    """Clear the process context (tests/config rotation/shutdown)."""
    global _execution_context
    with _singleton_lock:
        current = _execution_context
        _execution_context = None
    if shutdown and current is not None:
        current.shutdown()


def execution_context_unavailable_reason() -> str:
    if not os.environ.get('ETRADE_ACCOUNT_ID_KEY'):
        return 'ETRADE_ACCOUNT_ID_KEY is required'
    return (CLIENT_IMPORT_ERROR or 'LiveExecutionContext construction failed')


def _wire_daily_loss_gate(client) -> None:
    """Attach a -2% daily-loss gate to the shared GUI client so preview/place
    are PRE-blocked by the rail (each call reads balances and engages the kill
    switch on a breach), bound to the configured ETRADE_ACCOUNT_ID_KEY account.

    No-op when the breaker import, the kill switch, or the account id is
    unavailable — read-only GETs are never gated, and the rail FAILS CLOSED:
    if value_fn raises (balances unreadable), the exception propagates and the
    gated order is refused. Mirrors LiveEtradeBroker._build_daily_loss_gate.
    """
    if DailyLossGate is None:
        return
    kill_switch = get_kill_switch()
    account_id_key = os.environ.get("ETRADE_ACCOUNT_ID_KEY")
    if kill_switch is None or not account_id_key:
        return
    breaker = DailyLossCircuitBreaker(kill_switch, audit=get_audit_log())
    client.circuit_breaker = DailyLossGate(
        breaker,
        value_fn=lambda: extract_account_value(
            client.get_balances(account_id_key)))


def client_unavailable_reason() -> str:
    """Reason string for 503s when the trading-client surface is unusable."""
    return (CLIENT_IMPORT_ERROR or auth_unavailable_reason()
            or 'EtradeClient construction failed')


# The process-wide LiveScheduler. Deliberately NOT auto-constructed: a
# scheduler needs a LiveTradingSession (desk + broker + portfolio +
# data_fn), which only the operator's live wiring can provide. Whoever
# builds that session installs its scheduler here via set_scheduler();
# until then the /scheduler routes answer 503 'unconfigured' — automation
# that materializes by itself is exactly what Phase 9 ruled out.
_scheduler = None


def set_scheduler(scheduler) -> None:
    """Install (or clear, with None) the shared LiveScheduler."""
    global _scheduler
    with _singleton_lock:
        _scheduler = scheduler
        context = _execution_context
        if context is not None:
            context.scheduler = scheduler


def get_scheduler():
    """The installed LiveScheduler, or None while unconfigured."""
    if _scheduler is not None:
        return _scheduler
    context = get_execution_context(create=False)
    return getattr(context, 'scheduler', None) if context is not None else None


def scheduler_unavailable_reason() -> str:
    """Reason string for 503s when the scheduler surface is unusable."""
    return (SCHEDULER_IMPORT_ERROR
            or 'No live scheduler is configured — create a live trading '
               'session (which installs its scheduler) first.')


# The process-wide token keep-alive loop. UNLIKE the trading scheduler
# above, this one self-constructs lazily: it needs ONLY the shared auth
# manager (no desk / broker / portfolio / data feed) and NEVER trades, so
# "instantiate in prod" is safe to do on first use. It calls renew() on a
# timer during market hours to clear E*TRADE's ~2h idle timeout — it cannot
# place an order, resurrect a dead token, or bypass the midnight re-auth.
_keepalive_scheduler = None


def set_keepalive_scheduler(scheduler) -> None:
    """Install (or clear, with None) the shared keep-alive scheduler.
    Production builds it lazily via get_keepalive_scheduler(); tests use
    this to inject a fake or reset between cases."""
    global _keepalive_scheduler
    with _singleton_lock:
        _keepalive_scheduler = scheduler


def get_keepalive_scheduler():
    """Shared TokenKeepAliveScheduler over the live auth manager, built
    lazily, or None when the surface/auth manager is unavailable.

    Bound to the process-wide auth-manager singleton (which is constructed
    once and never replaced), so a single long-lived keep-alive loop is
    correct for the GUI process.
    """
    global _keepalive_scheduler
    if TokenKeepAliveScheduler is None:
        return None
    if _keepalive_scheduler is None:
        with _singleton_lock:
            if _keepalive_scheduler is None:
                # Resolve the manager INSIDE the lock so the binding and the
                # double-checked construction are atomic (the lock is an
                # RLock, so get_auth_manager()'s own acquire is fine).
                manager = get_auth_manager()
                if manager is None:
                    return None
                try:
                    _keepalive_scheduler = TokenKeepAliveScheduler(
                        manager, audit=get_audit_log())
                except Exception:
                    logger.error('TokenKeepAliveScheduler construction failed',
                                 exc_info=True)
                    return None
    return _keepalive_scheduler


def keepalive_unavailable_reason() -> str:
    """Reason string for 503s when the keep-alive surface is unusable."""
    return (KEEPALIVE_IMPORT_ERROR or auth_unavailable_reason()
            or 'Token keep-alive is unavailable')


def shutdown_background_workers() -> None:
    """Best-effort: stop the keep-alive and live-scheduler threads on process
    shutdown so a ``docker stop`` / Ctrl-C tears them down cleanly instead of
    orphaning daemon threads mid-cycle. Never raises, and never constructs a
    scheduler that was not already running (reads the singletons directly)."""
    with _singleton_lock:
        workers = [w for w in (_keepalive_scheduler, _scheduler)
                   if w is not None]
    for worker in workers:
        try:
            worker.stop()
        except Exception:  # noqa: BLE001 - shutdown must never raise
            logger.warning('Error stopping %s during shutdown',
                           type(worker).__name__, exc_info=True)
    try:
        reset_execution_context()
    except Exception:  # noqa: BLE001 - shutdown must never raise
        logger.warning('Error closing live execution context during shutdown',
                       exc_info=True)


def _start_keepalive_best_effort() -> None:
    """Begin (or resume after a pause) the token keep-alive loop following a
    successful connect. Best-effort and silent on failure — keeping the
    session warm must NEVER break the connect flow."""
    try:
        scheduler = get_keepalive_scheduler()
        if scheduler is not None:
            scheduler.start()
    except Exception:  # noqa: BLE001 - keep-alive must never break connect
        logger.warning('Could not start token keep-alive after connect',
                       exc_info=True)


def _stop_keepalive_best_effort() -> None:
    """Stop the keep-alive loop when the session is torn down (disconnect).
    Best-effort and silent — never breaks the disconnect flow."""
    try:
        scheduler = get_keepalive_scheduler()
        if scheduler is not None:
            scheduler.stop()
    except Exception:  # noqa: BLE001 - never break disconnect
        logger.warning('Could not stop token keep-alive after disconnect',
                       exc_info=True)


def _keepalive_desired_running() -> bool:
    """Was keep-alive's LAST EXPLICIT intent 'start' (not 'stop')?

    Derived from the persistent audit log so the intent survives a process
    restart: compare the most-recent 'keepalive_started' vs 'keepalive_stopped'
    sequence. A 'keepalive_paused' (token_expired / renew_failure_storm) is NOT
    an explicit stop — it does not flip the intent (we just can't act on it
    until the token reconnects, which the caller's connected-guard enforces).
    """
    audit = get_audit_log()
    if audit is None:
        return False
    try:
        started = audit.entries(limit=1, event_type='keepalive_started')
        stopped = audit.entries(limit=1, event_type='keepalive_stopped')
    except Exception:  # noqa: BLE001 - best-effort; absence means "not desired"
        logger.warning('Could not read keep-alive intent from audit',
                       exc_info=True)
        return False
    if not started:
        return False
    started_seq = started[0].get('seq')
    stopped_seq = stopped[0].get('seq') if stopped else None
    if started_seq is None:
        return False
    return stopped_seq is None or started_seq > stopped_seq


def resume_keepalive_if_desired() -> bool:
    """Restart recovery: after a process restart, resume the keep-alive loop
    IFF its last explicit intent was 'start' AND the token is currently
    connected (same ET day, still active in the DB). Returns True if it
    (re)started the loop.

    Conservative by construction: it can only ever resume a RENEW-ONLY loop
    the operator had already started, for a token that is still live — it
    never starts trading and never resurrects a dead token. Best-effort:
    a failure here must never break app startup.
    """
    try:
        # Never CREATE a db file just to look for a recovery intent: recovery
        # reads PRIOR audit rows, so if the db file does not already exist
        # there is nothing to recover. Guarding on existence (not just
        # TRADING_DB_PATH) keeps app construction side-effect free even when a
        # path is configured but no file has been written yet.
        if not os.path.exists(_live_db_path()):
            return False
        if not _keepalive_desired_running():
            return False
        # The connected-guard binds recovery to THIS env's live token: even if
        # the most-recent 'started' intent came from a different env sharing
        # the db, we only ever resume the renew-only loop bound to the current
        # auth manager, and only when its own token is connected — so a
        # cross-env intent can at worst keep a token the operator is already
        # holding warm. It can never trade or resurrect a dead token.
        manager = get_auth_manager()
        if manager is None or manager.status().get('state') != 'connected':
            return False
        scheduler = get_keepalive_scheduler()
        if scheduler is None:
            return False
        resumed = scheduler.start()
        if resumed:
            logger.info('Keep-alive resumed after restart (token connected, '
                        'last intent was start)')
        return resumed
    except Exception:  # noqa: BLE001 - never break app startup
        logger.warning('Keep-alive restart-recovery failed', exc_info=True)
        return False


def kill_switch_engaged() -> bool:
    """Best-effort engaged state for the base-template banner.

    Called by the app-wide context processor on every page render, so it
    must never raise and must stay cheap. It reads through the shared
    singleton when one already exists; otherwise it constructs one when a
    persistent DB is in play — TRADING_DB_PATH explicitly configured (the
    Docker/production case) OR the default db file already existing on
    disk (a dev server that has traded before). That keeps the 'red banner
    on EVERY page' guarantee across process restarts in both cases, while
    a page render never CREATES a db file: with no configured path and no
    existing default file there is nothing persisted to read, and the Live
    page's own status poll (the only poller) lights the banner up.
    """
    try:
        if _kill_switch is not None:
            return bool(_kill_switch.engaged())
        if KillSwitch is None:
            return False
        if (not os.environ.get('TRADING_DB_PATH')
                and not os.path.exists(_live_db_path())):
            return False
        ks = get_kill_switch()
        return bool(ks.engaged()) if ks is not None else False
    except Exception:
        logger.error('Kill-switch banner state check failed', exc_info=True)
        return False


def _find_live_broker():
    """The broker owned by the canonical account-bound live context."""
    context = get_execution_context()
    return context.broker if context is not None else None


def _local_state(broker) -> tuple[dict, float]:
    """Local (system-side) positions/cash for reconciliation (C19).

    The live session owns the canonical local book; the broker wrapper
    exposes it as ``local_positions`` / ``local_cash``. A session that
    tracks nothing locally reconciles as an empty book — conservative in
    the fail-safe direction (any broker-side position becomes a mismatch
    that engages the kill switch rather than passing silently).
    """
    context = get_execution_context(create=False)
    if context is not None and context.broker is broker:
        snapshot = context.local_book.reconciliation_snapshot()
        if not snapshot.get('initialized'):
            raise RuntimeError(
                'local book is uninitialized; explicit bootstrap is required')
        return dict(snapshot['positions']), float(snapshot['cash'])
    positions = getattr(broker, 'local_positions', None) or {}
    cash = getattr(broker, 'local_cash', None)
    return dict(positions), float(cash if cash is not None else 0.0)


def _unavailable(what: str, reason: str | None):
    return jsonify({'error': f'{what} unavailable', 'reason': reason}), 503


def _int_arg(name: str, default: int, lo: int, hi: int):
    """Parse a bounded int query param; None signals a 400 to the caller."""
    raw = request.args.get(name)
    if raw is None or raw == '':
        return default
    try:
        value = int(raw)
    except ValueError:
        return None
    if not lo <= value <= hi:
        return None
    return value


# ==================== ACCOUNT-NUMBER REDACTION ====================
# accountIdKey is the opaque, non-sensitive handle E*TRADE wants on every
# subsequent call — safe to expose. The raw account NUMBER (accountId) is
# sensitive and is NEVER returned whole: it surfaces only as a last-4 mask.

_ACCOUNT_NUMBER_FIELDS = ('accountId', 'accountNumber', 'acctNumber')


def _mask_account_number(value) -> str | None:
    """Last-4 mask for an account number ('••••1234'), or None for empties."""
    if value is None:
        return None
    digits = ''.join(ch for ch in str(value) if ch.isalnum())
    if not digits:
        return None
    return '••••' + digits[-4:]


def _redact_account_numbers(obj):
    """Recursively drop every raw account-number field from a balances /
    portfolio payload, replacing each with a '<field>_masked' last-4.

    Defensive by construction: E*TRADE sprinkles accountId into nested
    balance/position blocks, so a shallow scrub could leak one. accountIdKey
    is deliberately left untouched (it is the safe handle).
    """
    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            if key in _ACCOUNT_NUMBER_FIELDS:
                out[f'{key}_masked'] = _mask_account_number(val)
            else:
                out[key] = _redact_account_numbers(val)
        return out
    if isinstance(obj, list):
        return [_redact_account_numbers(item) for item in obj]
    return obj


# ==================== ORDER-REF CACHE (preview -> place handshake) ====================
# A previewed order_request is cached server-side, keyed by an opaque
# order_ref handed back to the client. PLACE works ONLY from a cached ref:
# the client can never post an arbitrary order payload to place. Refs are
# single-use (consumed on a successful place), TTL-expired, and the cache
# is size-capped (oldest evicted) so it cannot grow without bound.

_ORDER_REF_TTL_S = _order_support.ORDER_REF_TTL_S
_ORDER_REF_CACHE_MAX = _order_support.ORDER_REF_CACHE_MAX
_ORDER_REF_CACHE: dict = {}     # order_ref -> {request, preview_ids, account_id_key, created}


def _prune_order_refs(now: float | None = None) -> None:
    """Compatibility wrapper; caller holds ``_singleton_lock``."""
    _order_support.prune_order_refs(
        _ORDER_REF_CACHE,
        now=_time.time() if now is None else now,
        ttl_s=_ORDER_REF_TTL_S,
        max_entries=_ORDER_REF_CACHE_MAX,
    )


def _cache_order_ref(account_id_key: str, order_request: dict,
                     preview_ids: list) -> str:
    """Cache a previewed order_request and return its opaque order_ref."""
    return _order_support.cache_order_ref(
        _ORDER_REF_CACHE,
        _singleton_lock,
        account_id_key,
        order_request,
        preview_ids,
        clock=_time.time,
        ref_factory=secrets.token_urlsafe,
        ttl_s=_ORDER_REF_TTL_S,
        max_entries=_ORDER_REF_CACHE_MAX,
    )


def _consume_order_ref(ref: str):
    """Pop a non-expired cached order_ref (single-use), or None if missing
    or expired. Caller resolves None into a 404."""
    return _order_support.consume_order_ref(
        _ORDER_REF_CACHE,
        _singleton_lock,
        ref,
        clock=_time.time,
        ttl_s=_ORDER_REF_TTL_S,
        max_entries=_ORDER_REF_CACHE_MAX,
    )


# ==================== STATUS ====================

@live_bp.route('/status', methods=['GET'])
def live_status():
    """Auth status + env + kill switch + last reconciliation (C20).

    The live page is the ONLY page that polls this endpoint; an ambiguous
    kill-switch state must never read as fine, so an unavailable kill
    switch is a loud 503 here rather than a null field.
    """
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    ks = get_kill_switch()
    if ks is None:
        return _unavailable('Kill switch', KILL_SWITCH_IMPORT_ERROR)
    try:
        auth = manager.status()
        context = get_execution_context(create=False)
        last_reconciliation = (
            context.last_reconciliation if context is not None
            else _last_reconciliation)
        reservation_health = None
        if context is not None:
            snapshotter = getattr(context, 'reservation_snapshot', None)
            if callable(snapshotter):
                snapshot = snapshotter()
                reservations = snapshot.get('reservations') or []
                reservation_health = {
                    'active_count': sum(
                        row.get('status') == 'ACTIVE'
                        for row in reservations),
                    'active_totals': snapshot.get('active_totals'),
                }
        return jsonify({
            'auth': auth,
            'env': auth.get('env'),
            'kill_switch': {'engaged': bool(ks.engaged())},
            'reconciliation': last_reconciliation,
            'reservations': reservation_health,
        })
    except Exception:
        logger.error('Failed to retrieve live status', exc_info=True)
        return jsonify({'error': 'Failed to retrieve live status'}), 500


# ==================== AUTH LIFECYCLE (C16) ====================

@live_bp.route('/auth/start', methods=['POST'])
def auth_start():
    """Begin the OAuth flow. ONLY ever called from an explicit user click —
    the live page never auto-starts auth on load."""
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    try:
        authorize_url = manager.start_auth()
    except EtradeNotConfigured as e:
        return jsonify({'error': 'E*TRADE not configured',
                        'reason': str(e)}), 503
    except EtradeAuthError as e:
        # E*TRADE rejected the request_token call (e.g. bad / not-yet-approved
        # production keys, or a key/secret mismatch). The message redacts the
        # consumer key; surface it so the operator sees the actual oauth_problem
        # instead of a blind 500.
        logger.error('E*TRADE authorization failed: %s', e)
        return jsonify({'error': 'Failed to start authorization',
                        'reason': str(e)}), 502
    except Exception:
        logger.error('Failed to start E*TRADE authorization', exc_info=True)
        return jsonify({'error': 'Failed to start authorization'}), 500
    return jsonify({'authorize_url': authorize_url, 'auth': manager.status()})


@live_bp.route('/auth/verifier', methods=['POST'])
def auth_verifier():
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    data = request.get_json(silent=True) or {}
    code = str(data.get('code') or '').strip()
    if not code:
        return jsonify({'error': 'Verifier code is required'}), 400
    try:
        status = manager.submit_verifier(code)
    except EtradeNotConfigured as e:
        return jsonify({'error': 'E*TRADE not configured',
                        'reason': str(e)}), 503
    except EtradeAuthError as e:
        logger.error('E*TRADE verifier exchange failed: %s', e)
        return jsonify({'error': 'Failed to submit verifier code',
                        'reason': str(e)}), 502
    except Exception:
        logger.error('Failed to submit verifier code', exc_info=True)
        return jsonify({'error': 'Failed to submit verifier code'}), 500
    # A successful connect starts the keep-alive loop so the freshly issued
    # token stays warm through the trading day with no manual step (and
    # clears any prior token_expired pause). Best-effort: never fails connect.
    _start_keepalive_best_effort()
    return jsonify({'auth': status})


@live_bp.route('/auth/renew', methods=['POST'])
def auth_renew():
    """Additive beyond C20's minimum: the connected-state Renew button."""
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    try:
        renewed = bool(manager.renew())
        return jsonify({'renewed': renewed, 'auth': manager.status()})
    except EtradeNotConfigured as e:
        return jsonify({'error': 'E*TRADE not configured',
                        'reason': str(e)}), 503
    except EtradeAuthError as e:
        logger.error('E*TRADE token renew failed: %s', e)
        return jsonify({'error': 'Failed to renew token',
                        'reason': str(e)}), 502
    except Exception:
        logger.error('Failed to renew E*TRADE token', exc_info=True)
        return jsonify({'error': 'Failed to renew token'}), 500


@live_bp.route('/auth/disconnect', methods=['POST'])
def auth_disconnect():
    """Additive beyond C20's minimum: the connected-state Disconnect button."""
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    try:
        # Close/cancel account-bound execution while the OAuth token is still
        # usable; only then tear down the signing session.
        reset_execution_context()
        manager.disconnect()
        # The session is gone — stop keeping a dead token warm.
        _stop_keepalive_best_effort()
        return jsonify({'auth': manager.status()})
    except Exception:
        logger.error('Failed to disconnect E*TRADE session', exc_info=True)
        return jsonify({'error': 'Failed to disconnect'}), 500


# ==================== KILL SWITCH (C17) ====================

@live_bp.route('/killswitch', methods=['POST'])
def set_kill_switch():
    """Engage/disengage the kill switch. Engaging REQUIRES a reason; every
    flip is audit-logged by the KillSwitch itself (C17)."""
    ks = get_kill_switch()
    if ks is None:
        return _unavailable('Kill switch', KILL_SWITCH_IMPORT_ERROR)
    data = request.get_json(silent=True) or {}
    engaged = data.get('engaged')
    if not isinstance(engaged, bool):
        return jsonify({'error': "'engaged' must be a boolean"}), 400
    reason = str(data.get('reason') or '').strip()
    try:
        if engaged:
            if not reason:
                return jsonify(
                    {'error': 'A reason is required to engage the '
                              'kill switch'}), 400
            ks.engage(reason, 'gui')
        else:
            ks.disengage('gui')
        return jsonify({'engaged': bool(ks.engaged())})
    except Exception:
        logger.error('Failed to flip kill switch', exc_info=True)
        return jsonify({'error': 'Failed to update kill switch'}), 500


# ==================== AUDIT LOG (C18) ====================

@live_bp.route('/audit', methods=['GET'])
def audit_entries():
    """Paged audit entries; '?verify=1' is the additive query param that
    also runs the hash-chain verification (C20 allows this addition)."""
    log = get_audit_log()
    if log is None:
        return _unavailable('Audit log', AUDIT_IMPORT_ERROR)

    limit = _int_arg('limit', default=50, lo=1, hi=500)
    if limit is None:
        return jsonify(
            {'error': "'limit' must be an integer between 1 and 500"}), 400
    offset = _int_arg('offset', default=0, lo=0, hi=10_000_000)
    if offset is None:
        return jsonify({'error': "'offset' must be a non-negative "
                                 'integer'}), 400
    event_type = (request.args.get('event_type') or '').strip() or None

    try:
        payload = {
            'entries': log.entries(limit=limit, offset=offset,
                                   event_type=event_type),
            'limit': limit,
            'offset': offset,
            'event_type': event_type,
        }
        if (request.args.get('verify') or '').lower() in ('1', 'true'):
            payload['verify'] = log.verify_chain()
        return jsonify(payload)
    except Exception:
        logger.error('Failed to read audit log', exc_info=True)
        return jsonify({'error': 'Failed to read audit log'}), 500


# ==================== EXECUTION PARITY (Phase 3 Step 5) ====================

@live_bp.route('/parity', methods=['GET'])
def execution_parity():
    """Read-only live-vs-backtest execution parity over audited fills.

    Replays the audit log's execution_report rows through the backtest cost
    model and reports slippage/spread drift (+ modeled commission). Pure read:
    no orders, no broker calls, no audit writes.
    """
    if parity_report is None:
        return _unavailable('Execution parity', PARITY_IMPORT_ERROR)
    log = get_audit_log()
    if log is None:
        return _unavailable('Audit log', AUDIT_IMPORT_ERROR)

    limit = _int_arg('limit', default=10_000, lo=1, hi=100_000)
    if limit is None:
        return jsonify(
            {'error': "'limit' must be an integer between 1 and 100000"}), 400
    try:
        return jsonify(parity_report(log, limit=limit))
    except Exception:
        logger.error('Failed to compute execution parity', exc_info=True)
        return jsonify({'error': 'Failed to compute execution parity'}), 500


# ==================== RECONCILIATION (C19) ====================

@live_bp.route('/reconcile', methods=['POST'])
def run_reconcile():
    """Reconcile the persistent account-scoped book against the broker."""
    global _last_reconciliation
    if reconcile is None:
        return _unavailable('Reconciliation', RECONCILE_IMPORT_ERROR)
    _client, err = _require_client()
    if err is not None:
        return err
    context = get_execution_context()
    if context is None:
        return _unavailable('Live execution context',
                            execution_context_unavailable_reason())

    try:
        result = dict(context.run_reconciliation())
        result['kill_switch_engaged'] = not bool(result.get('ok'))
        context.last_reconciliation = result
        _last_reconciliation = result
        return jsonify(result)
    except LiveContextClosed as exc:
        return jsonify({'error': 'Live execution context unavailable',
                        'detail': str(exc)}), 409
    except RuntimeError as exc:
        if 'uninitialized' in str(exc).lower():
            return jsonify({
                'error': 'local book uninitialized',
                'detail': str(exc),
            }), 409
        logger.error('Reconciliation failed', exc_info=True)
        return jsonify({'error': 'Reconciliation failed'}), 500
    except Exception:
        logger.error('Reconciliation failed', exc_info=True)
        return jsonify({'error': 'Reconciliation failed'}), 500


@live_bp.route('/reconcile/bootstrap', methods=['POST'])
def bootstrap_reconciliation_book():
    """Explicitly initialize this account's local book from the broker."""
    _client, err = _require_client()
    if err is not None:
        return err
    context = get_execution_context()
    if context is None:
        return _unavailable('Live execution context',
                            execution_context_unavailable_reason())
    try:
        return jsonify(context.bootstrap_local_book())
    except LiveContextClosed as exc:
        return jsonify({'error': 'Live execution context unavailable',
                        'detail': str(exc)}), 409
    except Exception:
        logger.error('Local-book bootstrap failed', exc_info=True)
        return jsonify({'error': 'Local-book bootstrap failed'}), 500


# ==================== WORKING ORDERS (patient executor) ====================

@live_bp.route('/orders', methods=['GET'])
def working_orders():
    """Broker-authoritative working orders plus in-process executor detail."""
    _client, err = _require_client()
    if err is not None:
        return err
    context = get_execution_context()
    if context is None:
        return _unavailable('Live execution context',
                            execution_context_unavailable_reason())
    try:
        orders = list(context.working_orders())
    except LiveContextClosed as exc:
        return jsonify({'error': 'Live execution context unavailable',
                        'detail': str(exc)}), 409
    except Exception:
        logger.error('Failed to list working orders', exc_info=True)
        return jsonify({'error': 'Failed to list working orders'}), 500
    return jsonify({'orders': orders, 'count': len(orders)})


@live_bp.route('/risk/reservations', methods=['GET'])
def risk_reservations():
    """Durable pending-order capacity for the configured live account."""
    _client, err = _require_client()
    if err is not None:
        return err
    context = get_execution_context()
    if context is None:
        return _unavailable('Live execution context',
                            execution_context_unavailable_reason())
    try:
        return jsonify(context.reservation_snapshot())
    except LiveContextClosed as exc:
        return jsonify({'error': 'Live execution context unavailable',
                        'detail': str(exc)}), 409
    except Exception:
        logger.error('Failed to read risk reservations', exc_info=True)
        return jsonify({'error': 'Failed to read risk reservations'}), 500


@live_bp.route('/orders/<order_id>/cancel', methods=['POST'])
def cancel_working_order(order_id):
    """Cancel through the canonical context's server-configured account."""
    data = request.get_json(silent=True) or {}
    account_id_key = str(data.get('account_id_key')
                         or request.args.get('account_id_key') or '').strip()
    configured = str(os.environ.get('ETRADE_ACCOUNT_ID_KEY') or '').strip()
    if account_id_key and account_id_key != configured:
        return jsonify({'error': 'account mismatch'}), 409
    _client, err = _require_client()
    if err is not None:
        return err
    context = get_execution_context()
    if context is None:
        return _unavailable('Live execution context',
                            execution_context_unavailable_reason())
    try:
        if context.cancel_order(order_id):
            return jsonify({'message': 'Cancellation requested',
                            'order_id': order_id})
        return jsonify({'error': f'Order {order_id} not found'}), 404
    except Exception as exc:  # noqa: BLE001
        mapped = _client_error_response(exc)
        if mapped is not None:
            return mapped
        logger.error('Failed to cancel working order %s', order_id,
                     exc_info=True)
        return jsonify({'error': 'Failed to cancel order'}), 500


# ==================== MARKET-HOURS SCHEDULER ====================

#: Bounds for the GUI-settable evaluation interval (minutes).
SCHEDULER_INTERVAL_MIN = 1
SCHEDULER_INTERVAL_MAX = 240


def _scheduler_payload(scheduler) -> dict:
    """LiveScheduler.status() plus the configured interval (one shape for
    GET and POST so live.js renders both identically)."""
    payload = dict(scheduler.status())
    payload['interval_minutes'] = getattr(scheduler, 'interval_minutes', None)
    return payload


@live_bp.route('/scheduler', methods=['GET'])
def scheduler_status():
    """Scheduler state for the Live page card.

    503 until a scheduler is configured (set_scheduler) — the GUI never
    constructs a trading loop on its own.
    """
    scheduler = get_scheduler()
    if scheduler is None:
        return _unavailable('Scheduler', scheduler_unavailable_reason())
    try:
        return jsonify(_scheduler_payload(scheduler))
    except Exception:
        logger.error('Failed to read scheduler status', exc_info=True)
        return jsonify({'error': 'Failed to read scheduler status'}), 500


@live_bp.route('/scheduler', methods=['POST'])
def scheduler_action():
    """Start/stop the scheduler ({action: 'start'|'stop'}).

    'start' optionally takes interval_minutes (an int, 1-240) applied
    before the loop starts. The scheduler itself audits every start/stop/
    pause; pause states (kill switch, circuit breaker, error storm) can
    ONLY be cleared by an explicit 'start' — there is no 'resume'.
    """
    scheduler = get_scheduler()
    if scheduler is None:
        return _unavailable('Scheduler', scheduler_unavailable_reason())
    data = request.get_json(silent=True) or {}
    action = data.get('action')
    if action not in ('start', 'stop'):
        return jsonify(
            {'error': "'action' must be 'start' or 'stop'"}), 400
    interval = data.get('interval_minutes')
    if interval is not None:
        if action != 'start':
            return jsonify({'error': "'interval_minutes' only applies to "
                                     "action 'start'"}), 400
        if (not isinstance(interval, int) or isinstance(interval, bool)
                or not SCHEDULER_INTERVAL_MIN <= interval
                <= SCHEDULER_INTERVAL_MAX):
            return jsonify(
                {'error': f"'interval_minutes' must be an integer between "
                          f'{SCHEDULER_INTERVAL_MIN} and '
                          f'{SCHEDULER_INTERVAL_MAX}'}), 400
    try:
        if action == 'start':
            if interval is not None:
                scheduler.interval_minutes = interval
            scheduler.start()
        else:
            scheduler.stop()
        return jsonify(_scheduler_payload(scheduler))
    except Exception:
        logger.error('Scheduler %s failed', action, exc_info=True)
        return jsonify({'error': f'Scheduler {action} failed'}), 500


# ==================== TOKEN KEEP-ALIVE ====================
# A renew-only loop that keeps the E*TRADE OAuth session warm through the
# trading day (clears the ~2h idle timeout). Self-constructs over the auth
# manager, so — unlike /scheduler — it is available as soon as live auth
# imports cleanly. It NEVER trades and cannot bypass the midnight re-auth.

@live_bp.route('/keepalive', methods=['GET'])
def keepalive_status():
    """Token keep-alive loop state for the Live page."""
    scheduler = get_keepalive_scheduler()
    if scheduler is None:
        return _unavailable('Keep-alive', keepalive_unavailable_reason())
    try:
        return jsonify(scheduler.status())
    except Exception:
        logger.error('Failed to read keep-alive status', exc_info=True)
        return jsonify({'error': 'Failed to read keep-alive status'}), 500


@live_bp.route('/keepalive', methods=['POST'])
def keepalive_action():
    """Start/stop the keep-alive loop ({action: 'start'|'stop',
    interval_minutes?}). A 'start' also clears a pause (token_expired /
    renew_failure_storm) once the operator has re-authed."""
    scheduler = get_keepalive_scheduler()
    if scheduler is None:
        return _unavailable('Keep-alive', keepalive_unavailable_reason())
    data = request.get_json(silent=True) or {}
    action = data.get('action')
    if action not in ('start', 'stop'):
        return jsonify({'error': "'action' must be 'start' or 'stop'"}), 400
    interval = data.get('interval_minutes')
    if interval is not None:
        if action != 'start':
            return jsonify({'error': "'interval_minutes' only applies to "
                                     "action 'start'"}), 400
        if (not isinstance(interval, int) or isinstance(interval, bool)
                or not KEEPALIVE_INTERVAL_MIN <= interval
                <= KEEPALIVE_INTERVAL_MAX):
            return jsonify(
                {'error': f"'interval_minutes' must be an integer between "
                          f'{KEEPALIVE_INTERVAL_MIN} and '
                          f'{KEEPALIVE_INTERVAL_MAX}'}), 400
    try:
        if action == 'start':
            if interval is not None:
                scheduler.interval_minutes = interval
            scheduler.start()
        else:
            scheduler.stop()
        return jsonify(scheduler.status())
    except Exception:
        logger.error('Keep-alive %s failed', action, exc_info=True)
        return jsonify({'error': f'Keep-alive {action} failed'}), 500


# ==================== LIVE TRADING (accounts / quotes / order ticket) ====================
# Every route below requires a CONNECTED auth state. The flow is:
#   1. _require_client()  -> 503 when the client surface is unavailable.
#   2. connected state    -> 409 {'error':'not connected','state':<state>}.
#   3. typed exceptions on the call are mapped by _client_error_response:
#        EtradeAuthExpired   -> 401 {'error':'reauthorize'}
#        KillSwitchEngaged   -> 409 {'error':'kill switch engaged'|'circuit breaker'}
#        EtradeOrderRejected -> 422 {'error':..., 'reason':...}
#        EtradeRateLimited   -> 503 (typed reason)
#        EtradeUnavailable   -> 503 (typed reason)
#        EtradeApiError      -> 502 (typed reason)
# Tracebacks and secrets NEVER leak: only the typed reason strings (which
# the client builds from E*TRADE's own Error.message) reach the client.

#: Max symbols per /quotes request (E*TRADE multi-quote practical cap).
QUOTE_SYMBOL_CAP = 25


def _require_client():
    """(client, None) when ready; (None, error_response) otherwise.

    Resolves the 503 (surface unavailable) and 409 (not connected) guards
    in one place so each route stays a thin happy-path.
    """
    if EtradeClient is None:
        return None, _unavailable('Live trading', client_unavailable_reason())
    manager = get_auth_manager()
    if manager is None:
        return None, _unavailable('Live trading', client_unavailable_reason())
    try:
        state = (manager.status() or {}).get('state')
    except Exception:
        logger.error('Failed to read auth state for live trade route',
                     exc_info=True)
        return None, (jsonify({'error': 'Failed to read auth state'}), 500)
    if state != 'connected':
        return None, (jsonify({'error': 'not connected', 'state': state}), 409)
    client = get_client()
    if client is None:
        return None, _unavailable('Live trading', client_unavailable_reason())
    return client, None


def _client_error_response(exc: Exception):
    """Map a typed client exception to its (json, status). Returns None when
    the exception is not one we translate (caller raises/500s)."""
    if isinstance(exc, EtradeAuthExpired):
        return jsonify({'error': 'reauthorize'}), 401
    if isinstance(exc, RiskCapacityExceeded):
        return jsonify({
            'error': 'risk limit exceeded',
            'code': 'RISK_LIMIT_EXCEEDED',
            'breaches': exc.breaches,
        }), 409
    if isinstance(exc, UnsupportedRiskOrder):
        return jsonify({'error': 'unsupported risk order',
                        'reason': str(exc)}), 422
    if isinstance(exc, ReservationPersistenceError):
        return jsonify({'error': 'reservation persistence failed',
                        'reason': str(exc),
                        'kill_switch_engaged': True}), 500
    if isinstance(exc, ReservationStateUnknown):
        return jsonify({'error': 'reservation state unknown',
                        'reason': str(exc),
                        'kill_switch_engaged': True}), 503
    if isinstance(exc, RiskValuationUnavailable):
        return jsonify({'error': 'risk valuation unavailable',
                        'reason': str(exc)}), 503
    if isinstance(exc, KillSwitchEngaged):
        reason = str(exc)
        label = ('circuit breaker' if 'circuit breaker' in reason.lower()
                 else 'kill switch engaged')
        return jsonify({'error': label, 'reason': reason}), 409
    if isinstance(exc, EtradeOrderRejected):
        return jsonify({'error': 'order rejected', 'reason': str(exc)}), 422
    if isinstance(exc, EtradeRateLimited):
        return jsonify({'error': 'rate limited', 'reason': str(exc)}), 503
    if isinstance(exc, EtradeUnavailable):
        return jsonify({'error': 'E*TRADE unavailable',
                        'reason': str(exc)}), 503
    if isinstance(exc, EtradeApiError):
        return jsonify({'error': 'E*TRADE API error', 'reason': str(exc)}), 502
    return None


# -------------------- R1: accounts --------------------

@live_bp.route('/accounts', methods=['GET'])
def live_accounts():
    """List E*TRADE accounts (account numbers masked to last-4)."""
    client, err = _require_client()
    if err is not None:
        return err
    try:
        raw = client.list_accounts()
    except Exception as exc:  # noqa: BLE001
        mapped = _client_error_response(exc)
        if mapped is not None:
            return mapped
        logger.error('Failed to list accounts', exc_info=True)
        return jsonify({'error': 'Failed to list accounts'}), 500
    accounts = [{
        'accountIdKey': acct.get('accountIdKey'),
        'accountType': acct.get('accountType'),
        'accountDesc': acct.get('accountDesc'),
        'accountStatus': acct.get('accountStatus'),
        'accountId_masked': _mask_account_number(acct.get('accountId')),
    } for acct in (raw or [])]
    return jsonify({'accounts': accounts})


# -------------------- R2: balances --------------------

@live_bp.route('/accounts/<id_key>/balances', methods=['GET'])
def live_balances(id_key):
    """Account balances, with every account-number field masked."""
    client, err = _require_client()
    if err is not None:
        return err
    try:
        balances = client.get_balances(id_key)
    except Exception as exc:  # noqa: BLE001
        mapped = _client_error_response(exc)
        if mapped is not None:
            return mapped
        logger.error('Failed to read balances', exc_info=True)
        return jsonify({'error': 'Failed to read balances'}), 500
    return jsonify({'balances': _redact_account_numbers(balances or {})})


# -------------------- R3: portfolio --------------------

@live_bp.route('/accounts/<id_key>/portfolio', methods=['GET'])
def live_portfolio(id_key):
    """Account positions, with every account-number field masked."""
    client, err = _require_client()
    if err is not None:
        return err
    try:
        positions = client.get_portfolio(id_key)
    except Exception as exc:  # noqa: BLE001
        mapped = _client_error_response(exc)
        if mapped is not None:
            return mapped
        logger.error('Failed to read portfolio', exc_info=True)
        return jsonify({'error': 'Failed to read portfolio'}), 500
    return jsonify({'positions': _redact_account_numbers(positions or [])})


@live_bp.route('/positions/<symbol>', methods=['GET'])
def live_symbol_positions(symbol):
    """Configured-account positions for ONE symbol — the Charts entry-price
    overlay. Scoped server-side to ETRADE_ACCOUNT_ID_KEY so the chart page
    never needs the (secret) id_key and only ever sees the charted symbol; no
    account-number fields are returned. Gated/fail-closed like every live route
    (409 when not connected); returns [] when flat in that symbol.
    """
    client, err = _require_client()
    if err is not None:
        return err
    account_id_key = os.environ.get("ETRADE_ACCOUNT_ID_KEY")
    if not account_id_key:
        return jsonify({'error': 'ETRADE_ACCOUNT_ID_KEY not configured'}), 503
    want = (symbol or '').strip().upper()
    if not want:
        return jsonify({'error': 'symbol is required'}), 400
    try:
        raw = client.get_portfolio(account_id_key)
    except Exception as exc:  # noqa: BLE001
        mapped = _client_error_response(exc)
        if mapped is not None:
            return mapped
        logger.error('Failed to read portfolio for chart overlay',
                     exc_info=True)
        return jsonify({'error': 'Failed to read portfolio'}), 500
    out = []
    for pos in (raw or []):
        sym = ((pos.get('Product') or {}).get('symbol')
               or pos.get('symbolDescription') or '').strip().upper()
        if sym != want:
            continue
        out.append({
            'symbol': want,
            'quantity': pos.get('quantity'),
            'price_paid': pos.get('pricePaid'),
            'position_type': pos.get('positionType'),
        })
    return jsonify({'positions': out})


# -------------------- R4: quotes --------------------

@live_bp.route('/quotes', methods=['GET'])
def live_quotes():
    """Multi-symbol quotes (bid/ask/last). Capped at QUOTE_SYMBOL_CAP;
    over-cap requests are a 400 (the UI never sends more)."""
    client, err = _require_client()
    if err is not None:
        return err
    raw = (request.args.get('symbols') or '').strip()
    requested = [s.strip().upper() for s in raw.split(',') if s.strip()]
    # De-dupe while preserving order.
    seen: dict = {}
    for sym in requested:
        seen.setdefault(sym, None)
    requested = list(seen.keys())
    if not requested:
        return jsonify({'error': "'symbols' is required (comma-separated)"}), 400
    if len(requested) > QUOTE_SYMBOL_CAP:
        return jsonify({'error': f'Too many symbols (max '
                                 f'{QUOTE_SYMBOL_CAP})'}), 400
    try:
        quotes = client.get_quotes(requested)
    except Exception as exc:  # noqa: BLE001
        mapped = _client_error_response(exc)
        if mapped is not None:
            return mapped
        logger.error('Failed to read quotes', exc_info=True)
        return jsonify({'error': 'Failed to read quotes'}), 500
    return jsonify({'quotes': quotes or {}, 'requested': requested})


# -------------------- R5/R6: order ticket --------------------

def _to_float(value):
    """Compatibility wrapper for the extracted finite-float parser."""
    return _order_support.to_float(value)


def _to_int(value):
    """Compatibility wrapper for the extracted exact-integer parser."""
    return _order_support.to_int(value)


def _optional_limit_price(params: dict, key: str = 'limit_price'):
    """Compatibility wrapper preserving the route's parser monkeypatch seam."""
    return _order_support.optional_limit_price(
        params, key, parse_float=_to_float)


def _build_order_request(kind: str, params: dict):
    """Compatibility wrapper around pure ticket validation/building."""
    return _order_support.build_order_request(
        kind,
        params,
        equity_builder=build_equity_order,
        option_builder=build_option_order,
        spread_builder=build_spread_order,
        parse_float=_to_float,
        parse_int=_to_int,
        parse_limit=_optional_limit_price,
    )


def _preview_summary(preview: dict) -> dict:
    """Compatibility wrapper around pure preview summarization."""
    return _order_support.preview_summary(preview)


def _market_hours_block(data: dict):
    """409 response when a PRODUCTION order must not transmit outside the NYSE
    regular session, else None.

    Sandbox is a test venue with no real market, so it is never blocked (live
    rehearsal stays unimpeded). Override per request with allow_after_hours=true
    for deliberate extended-hours orders.
    """
    if MarketHours is None or data.get('allow_after_hours') is True:
        return None
    manager = get_auth_manager()
    env = getattr(manager, 'env', 'sandbox') if manager is not None \
        else 'sandbox'
    if env != 'production':
        return None
    try:
        if MarketHours().is_market_open(datetime.now(timezone.utc)):
            return None
    except Exception:  # noqa: BLE001 - a calendar failure fails CLOSED (block)
        logger.warning('Market-hours check failed; blocking order off-hours',
                       exc_info=True)
    return jsonify({
        'error': 'market closed',
        'detail': 'The NYSE regular session is closed. Pass '
                  'allow_after_hours=true to override.',
    }), 409


def _configured_order_account():
    """Return the only account live order routes may target, or a 503.

    The GUI client's daily-loss gate is bound to ETRADE_ACCOUNT_ID_KEY.  An
    order route must therefore fail closed when it is absent instead of
    accepting an arbitrary request account that the rail is not monitoring.
    """
    account_id_key = str(os.environ.get('ETRADE_ACCOUNT_ID_KEY') or '').strip()
    if account_id_key:
        return account_id_key, None
    return None, (jsonify({
        'error': 'Live order routing is not configured',
        'reason': 'ETRADE_ACCOUNT_ID_KEY is required for preview and place',
    }), 503)


@live_bp.route('/order/preview', methods=['POST'])
def live_order_preview():
    """Build + preview an order, caching the built request behind an opaque
    order_ref. PLACE only ever works from a cached, previewed ref."""
    configured_account, account_err = _configured_order_account()
    if account_err is not None:
        return account_err
    client, err = _require_client()
    if err is not None:
        return err
    context = get_execution_context()
    if context is None:
        return _unavailable('Live execution context',
                            execution_context_unavailable_reason())
    data = request.get_json(silent=True) or {}
    account_id_key = str(data.get('account_id_key') or '').strip()
    kind = str(data.get('kind') or '').strip()
    if not account_id_key:
        return jsonify({'error': "'account_id_key' is required"}), 400
    # The daily-loss rail (get_client) monitors the configured
    # ETRADE_ACCOUNT_ID_KEY account; refuse an order for any OTHER account so
    # the rail and the order can never diverge. Missing configuration already
    # failed closed above. Place inherits this binding via the cached order_ref.
    if account_id_key != configured_account:
        return jsonify({
            'error': 'account mismatch',
            'detail': 'Live orders must target the configured '
                      'ETRADE_ACCOUNT_ID_KEY account — the one the daily-loss '
                      'rail monitors.',
        }), 409
    if kind not in ('equity', 'option', 'spread'):
        return jsonify({'error': "'kind' must be 'equity', 'option' or "
                                 "'spread'"}), 400
    blocked = _market_hours_block(data)
    if blocked is not None:
        return blocked

    order_request, build_err = _build_order_request(kind, data)
    if build_err is not None:
        return jsonify({'error': build_err}), 400

    try:
        risk_estimate = context.risk_estimate(order_request)
        preview = context.preview_order(order_request)
    except PriceSanityError as exc:
        return jsonify({
            'error': 'price sanity check failed',
            'detail': str(exc),
        }), 400
    except Exception as exc:  # noqa: BLE001
        mapped = _client_error_response(exc)
        if mapped is not None:
            return mapped
        logger.error('Order preview failed', exc_info=True)
        return jsonify({'error': 'Order preview failed'}), 500

    preview_ids = preview.get('PreviewIds', []) if isinstance(preview, dict) \
        else []
    order_ref = _cache_order_ref(account_id_key, order_request, preview_ids)
    return jsonify({
        'preview': _preview_summary(preview if isinstance(preview, dict)
                                    else {}),
        'risk': risk_estimate,
        'order_ref': order_ref,
    })


@live_bp.route('/order/place', methods=['POST'])
def live_order_place():
    """Place a previously-previewed order, addressed ONLY by order_ref.

    Looks up the cached previewed request (404 if missing/expired), places
    it, and consumes the ref on success (single-use). The cached request is
    consumed BEFORE the place call so a successful placement can never be
    replayed; a failed place re-caches nothing (the operator re-previews).
    """
    configured_account, account_err = _configured_order_account()
    if account_err is not None:
        return account_err
    client, err = _require_client()
    if err is not None:
        return err
    context = get_execution_context()
    if context is None:
        return _unavailable('Live execution context',
                            execution_context_unavailable_reason())
    data = request.get_json(silent=True) or {}
    order_ref = str(data.get('order_ref') or '').strip()
    if not order_ref:
        return jsonify({'error': "'order_ref' is required"}), 400
    blocked = _market_hours_block(data)
    if blocked is not None:
        return blocked

    record = _consume_order_ref(order_ref)
    if record is None:
        return jsonify({'error': 'order_ref not found or expired'}), 404

    account_id_key = record['account_id_key']
    if account_id_key != configured_account:
        return jsonify({
            'error': 'account mismatch',
            'detail': 'The previewed order does not target the currently '
                      'configured ETRADE_ACCOUNT_ID_KEY account.',
        }), 409
    order_request = record['request']
    preview_ids = record['preview_ids']
    try:
        result = context.place_order(order_request, preview_ids)
    except PlacedOrderTrackingError as exc:
        logger.critical('Order %s placed but local tracking failed',
                        exc.order_id, exc_info=True)
        return jsonify({
            'error': 'order placed but local tracking failed',
            'orderId': exc.order_id,
            'kill_switch_engaged': True,
        }), 500
    except Exception as exc:  # noqa: BLE001
        mapped = _client_error_response(exc)
        if mapped is not None:
            return mapped
        logger.error('Order placement failed', exc_info=True)
        return jsonify({'error': 'Order placement failed'}), 500

    order_id = result.get('order_id') if isinstance(result, dict) else None
    placed = result.get('response', {}) if isinstance(result, dict) else {}
    status = None
    if isinstance(placed, dict):
        order_ids = placed.get('OrderIds') or []
        if order_ids and isinstance(order_ids[0], dict):
            status = order_ids[0].get('status')
        status = status or placed.get('status')
    return jsonify({'order': {'orderId': order_id, 'status': status}})
