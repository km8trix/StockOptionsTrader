"""Contract tests for the canonical live execution composition root.

These tests keep network and worker startup fully injected while exercising
the real account-scoped SQLite ledger.  The point of the context is not merely
convenient construction: it must make account identity, restart recovery,
cancellation ownership, reconciliation, and shutdown one coherent boundary.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import execution.live_context as live_context_module
from brokers.etrade_client import build_equity_order
from brokers.local_book import LocalBook
from core.models import Asset, AssetType
from deployment.live import FoundationExecutionGuard
from execution.live_context import LiveContextClosed, LiveExecutionContext
from execution.patient_executor import (
    ControlledNotionalEnvelopeError,
    PatientExecutor,
)
from portfolio.manager import PortfolioManager


SPY = Asset("SPY", AssetType.STOCK)
SPY_CALL = Asset("SPY", AssetType.CALL, 500.0, "2026-07-17")


class FakeAuthManager:
    def __init__(self, db_path, env="sandbox"):
        self.db_path = str(db_path)
        self.env = env


class FakeClient:
    """No method performs I/O; calls are retained for routing assertions."""

    def __init__(self):
        self.quotes = {"SPY": {"bid": 100.0, "ask": 100.2}}
        self.orders = []
        self.quote_calls = []
        self.list_calls = []

    # LiveEtradeBroker recognizes a prebuilt client through this surface.
    def preview_order(self, *_args, **_kwargs):  # pragma: no cover - marker
        raise AssertionError("construction must not preview an order")

    def get_quotes(self, symbols):
        self.quote_calls.append(list(symbols))
        return self.quotes

    def list_orders(self, account_id_key, **params):
        self.list_calls.append((account_id_key, params))
        return list(self.orders)


class FakeKillSwitch:
    def __init__(self):
        self.engagements = []

    def engaged(self):
        return bool(self.engagements)

    def engage(self, reason, actor):
        self.engagements.append((reason, actor))


class FakeAudit:
    def __init__(self):
        self.events = []

    def append(self, actor, event_type, payload):
        self.events.append((actor, event_type, payload))
        return len(self.events)


class FakeBroker:
    def __init__(self, portfolio=None):
        self.portfolio = portfolio or {"cash": 0.0, "positions": []}
        self.statuses = {}
        self.status_calls = []
        self.cancel_calls = []

    def get_portfolio_status(self):
        return {
            "cash": self.portfolio["cash"],
            "positions": [dict(row) for row in self.portfolio["positions"]],
        }

    def order_status(self, order_id):
        self.status_calls.append(str(order_id))
        status = self.statuses.get(str(order_id))
        return dict(status) if status is not None else None

    def cancel_order(self, order_id):
        self.cancel_calls.append(str(order_id))
        return True


class FakeExecutor:
    def __init__(self, working=None, owned_order_ids=()):
        self._working = list(working or [])
        self.owned_order_ids = {str(value) for value in owned_order_ids}
        self.cancel_requests = []
        self.close_calls = []
        self.stop_calls = 0

    def working_orders(self):
        return [dict(order) for order in self._working]

    def cancel_current(self, order_id=None):
        order_id = str(order_id) if order_id is not None else None
        self.cancel_requests.append(order_id)
        return order_id in self.owned_order_ids

    def close(self, timeout=None):
        self.close_calls.append(timeout)
        return True

    def stop(self):
        self.stop_calls += 1


class NotionalBroker:
    """Small limit-order fake for controlled notional envelope tests."""

    def __init__(self, *, filled_quantity=0, fill_price=None):
        self.filled_quantity = filled_quantity
        self.fill_price = fill_price
        self.placements = []
        self.cancelled = set()

    def place_order(self, asset, order_type, quantity, limit_price):
        order_id = f"ORDER-{len(self.placements) + 1}"
        self.placements.append({
            "order_id": order_id,
            "asset": asset,
            "order_type": order_type,
            "quantity": quantity,
            "limit_price": limit_price,
        })
        return order_id

    def order_status(self, order_id):
        quantity = self.placements[int(order_id.split("-")[-1]) - 1][
            "quantity"]
        status = "CANCELLED" if order_id in self.cancelled else "OPEN"
        if self.filled_quantity >= quantity:
            status = "EXECUTED"
        return {
            "status": status,
            "filled_quantity": min(self.filled_quantity, quantity),
            "avg_fill_price": self.fill_price,
        }

    def cancel_order(self, order_id):
        self.cancelled.add(order_id)
        return True


def make_context(tmp_path, *, client=None, broker=None, executor=None,
                 book=None, kill_switch=None, audit=None):
    db_path = tmp_path / "live.db"
    client = client or FakeClient()
    broker = broker or FakeBroker()
    executor = executor or FakeExecutor()
    kill_switch = kill_switch or FakeKillSwitch()
    audit = audit or FakeAudit()
    book = book or LocalBook(
        str(db_path), env="sandbox", account_id_key="ACCOUNT-1")
    context = LiveExecutionContext(
        auth_manager=FakeAuthManager(db_path),
        client=client,
        kill_switch=kill_switch,
        audit=audit,
        account_id_key="ACCOUNT-1",
        broker=broker,
        local_book=book,
        executor=executor,
    )
    return context, client, broker, executor, book, kill_switch, audit


class TestConstructionAndQuotes:
    def test_default_construction_binds_one_account_without_io_or_workers(
            self, tmp_path):
        db_path = tmp_path / "canonical.db"
        client = FakeClient()
        context = LiveExecutionContext(
            auth_manager=FakeAuthManager(db_path, env="production"),
            client=client,
            kill_switch=FakeKillSwitch(),
            audit=FakeAudit(),
            account_id_key="ACCOUNT-9",
        )

        assert context.identity.db_path == str(db_path.resolve())
        assert context.identity.env == "production"
        assert context.identity.account_id_key == "ACCOUNT-9"
        assert context.broker.client is client
        assert context.broker.account_id_key == "ACCOUNT-9"
        assert context.local_book.account_id_key == "ACCOUNT-9"
        assert context.local_book.env == "production"
        assert context.executor.broker is context.broker
        assert context.session is None
        assert context.scheduler is None
        assert context.state == "ready"
        assert client.quote_calls == []
        assert client.list_calls == []

        assert context.shutdown() is True

    def test_account_id_is_mandatory(self, tmp_path):
        with pytest.raises(ValueError, match="account_id_key is required"):
            LiveExecutionContext(
                auth_manager=FakeAuthManager(tmp_path / "bad.db"),
                client=FakeClient(),
                kill_switch=FakeKillSwitch(),
                audit=FakeAudit(),
                account_id_key="  ",
            )

    def test_equity_quote_uses_exact_symbol_bid_and_ask(self, tmp_path):
        context, client, *_ = make_context(tmp_path)
        client.quotes = {"SPY": {"bid": "499.90", "ask": "500.10"}}

        assert context._equity_quote(SPY) == {"bid": 499.9, "ask": 500.1}
        assert client.quote_calls == [["SPY"]]

    @pytest.mark.parametrize("instrument", [SPY_CALL, [SPY, SPY_CALL]])
    def test_option_and_package_quotes_fail_closed_before_underlying_lookup(
            self, tmp_path, instrument):
        context, client, *_ = make_context(tmp_path)

        with pytest.raises(ValueError, match="option/package quote adapter"):
            context._equity_quote(instrument)
        assert client.quote_calls == []

    @pytest.mark.parametrize("quote", [
        {},
        {"bid": 0, "ask": 1},
        {"bid": 2, "ask": 1},
        {"bid": float("nan"), "ask": 1},
    ])
    def test_invalid_equity_market_is_refused(self, tmp_path, quote):
        context, client, *_ = make_context(tmp_path)
        client.quotes = {"SPY": quote}
        with pytest.raises(ValueError, match="quote for SPY"):
            context._equity_quote(SPY)


class TestControlledNotionalEnvelope:
    def test_controlled_executor_requires_exact_one_shot_authorization(self):
        broker = NotionalBroker()
        executor = PatientExecutor(
            broker, lambda _asset: {"bid": 99.0, "ask": 100.0},
            sleep_fn=lambda _seconds: None,
        )
        executor.require_controlled_notional_envelopes()

        with pytest.raises(
                ControlledNotionalEnvelopeError,
                match="no durable notional envelope"):
            executor.execute(
                "BUY", SPY, 10, execution_id="not-authorized")

        assert broker.placements == []

    def test_guard_reservation_caps_later_patient_quote(self):
        now = datetime(2026, 7, 13, 14, tzinfo=timezone.utc)

        class Store:
            def __init__(self):
                self.calls = []

            def authorize_order(self, _manifest_hash, **economics):
                self.calls.append(economics)
                return {
                    "authorized": True,
                    "notional": (economics["quantity"]
                                 * economics["reference_price"]),
                }

        class QuoteBroker(NotionalBroker):
            def get_current_quote(self, _symbol):
                return {
                    "bid": 99.9,
                    "ask": 100.0,
                    "last": 99.95,
                    "observed_at": now.isoformat(),
                    "quote_status": "REALTIME",
                }

        broker = QuoteBroker()
        store = Store()
        executor = PatientExecutor(
            broker, lambda _asset: {"bid": 120.0, "ask": 122.0},
            clock=lambda: now, sleep_fn=lambda _seconds: None,
        )
        guard = FoundationExecutionGuard(
            SimpleNamespace(manifest_hash="a" * 64), store, broker)
        guard.bind_executor(executor)
        guard(
            intent=SimpleNamespace(
                asset=SPY, action="BUY", intent_id="opening-1"),
            side="BUY", quantity=10, now=now,
        )

        report = executor.execute(
            "BUY", SPY, 10, execution_id="opening-1")

        assert store.calls[0]["reference_price"] == 100.0
        assert report["status"] == "error"
        assert report["error_type"] == "ControlledNotionalEnvelopeError"
        assert "persistently authorized notional" in report["error"]
        assert broker.placements == []

    def test_partial_fill_and_replacement_share_one_aggregate_budget(self):
        quotes = [
            {"bid": 99.8, "ask": 100.0},
            {"bid": 103.0, "ask": 104.0},
        ]
        broker = NotionalBroker(filled_quantity=2, fill_price=99.9)
        executor = PatientExecutor(
            broker,
            lambda _asset: quotes.pop(0) if len(quotes) > 1 else quotes[0],
            sleep_fn=lambda _seconds: None,
        )
        executor.require_controlled_notional_envelopes()
        executor.arm_controlled_notional_envelope(
            execution_id="opening-2", side="BUY", symbol="SPY",
            quantity=10, opening=True, max_notional=1_000.0,
        )

        report = executor.execute(
            "BUY", SPY, 10, max_minutes=1, step_interval_s=0,
            execution_id="opening-2",
        )

        assert report["status"] == "error"
        assert report["fills"][0]["qty"] == 2
        assert "1007.24 > 1000.00" in report["error"]
        assert len(broker.placements) == 1
        assert broker.cancelled == {"ORDER-1"}

    def test_exit_is_not_blocked_by_opening_notional_capacity(self):
        broker = NotionalBroker(filled_quantity=10, fill_price=200.5)
        executor = PatientExecutor(
            broker, lambda _asset: {"bid": 200.0, "ask": 201.0},
            sleep_fn=lambda _seconds: None,
        )
        executor.require_controlled_notional_envelopes()
        executor.arm_controlled_notional_envelope(
            execution_id="exit-1", side="SELL", symbol="SPY",
            quantity=10, opening=False, max_notional=None,
        )

        report = executor.execute(
            "SELL", SPY, 10, execution_id="exit-1")

        assert report["status"] == "filled"
        assert broker.placements[0]["limit_price"] == 200.5


class TestWorkingOrdersAndCancellation:
    def test_broker_orders_are_authoritative_and_executor_enriches_active_one(
            self, tmp_path):
        client = FakeClient()
        client.orders = [
            {
                "orderId": "100",
                "clientOrderId": "browser-1",
                "OrderDetail": [{
                    "status": "OPEN",
                    "limitPrice": 100.15,
                    "Instrument": [{
                        "Product": {"symbol": "SPY", "securityType": "EQ"},
                        "orderAction": "BUY",
                        "orderedQuantity": 10,
                        "filledQuantity": 3,
                    }],
                }],
            },
            {
                "orderId": "terminal",
                "OrderDetail": [{"status": "EXECUTED", "Instrument": []}],
            },
        ]
        executor = FakeExecutor(working=[
            {
                "order_id": "100",
                "status": "cancel_requested",
                "limit_price": 100.16,
                "remaining_seconds": 12.0,
            },
            {
                "order_id": "starting",
                "status": "starting",
                "quantity": 2,
            },
        ])
        context, *_ = make_context(
            tmp_path, client=client, executor=executor)

        orders = context.working_orders()

        assert [order["order_id"] for order in orders] == ["100", "starting"]
        recovered = orders[0]
        assert recovered["client_order_id"] == "browser-1"
        assert recovered["filled_quantity"] == 3.0
        assert recovered["remaining_quantity"] == 7.0
        assert recovered["instruments"][0]["symbol"] == "SPY"
        assert recovered["status"] == "cancel_requested"
        assert recovered["limit_price"] == 100.16
        assert recovered["remaining_seconds"] == 12.0
        assert recovered["recovered"] is False
        assert orders[1]["recovered"] is False
        assert client.list_calls == [("ACCOUNT-1", {"count": 100})]

    def test_executor_owns_cancellation_of_its_active_order(self, tmp_path):
        broker = FakeBroker()
        executor = FakeExecutor(owned_order_ids={"ACTIVE"})
        context, *_ = make_context(
            tmp_path, broker=broker, executor=executor)

        assert context.cancel_order("ACTIVE") is True
        assert executor.cancel_requests == ["ACTIVE"]
        assert broker.cancel_calls == []

        assert context.cancel_order("RECOVERED") is True
        assert executor.cancel_requests == ["ACTIVE", "RECOVERED"]
        assert broker.cancel_calls == ["RECOVERED"]


class TestDurableTrackingAndReconciliation:
    def test_tracking_and_sync_bank_each_cumulative_fill_once_across_restart(
            self, tmp_path):
        broker = FakeBroker()
        context, _, _, _, book, *_ = make_context(
            tmp_path, broker=broker)
        assert book.bootstrap({}, 1_000.0) is True
        request = build_equity_order(
            "SPY", "BUY", 2, limit_price=100.0,
            client_order_id="durable-1")
        context.track_placed_order("ORDER-1", request)
        broker.statuses["ORDER-1"] = {
            "status": "EXECUTED",
            "filled_quantity": 2,
            "avg_fill_price": 100.0,
        }

        context.sync_tracked_orders()
        context.sync_tracked_orders()

        assert broker.status_calls == ["ORDER-1"]
        assert book.positions() == {"SPY": 2.0}
        assert book.cash() == pytest.approx(800.0)
        tracked = LocalBook(
            book.db_path, env="sandbox",
            account_id_key="ACCOUNT-1").tracked_order("ORDER-1")
        assert tracked is not None
        assert tracked["status"] == "EXECUTED"
        assert tracked["cumulative_booked_quantity"] == 2.0
        assert tracked["request"] == request

    def test_bootstrap_explicitly_adopts_broker_snapshot_only_once(
            self, tmp_path):
        broker = FakeBroker({
            "cash": 900.0,
            "positions": [{"symbol": "SPY", "quantity": 1}],
        })
        context, _, _, _, book, _, audit = make_context(
            tmp_path, broker=broker)

        first = context.bootstrap_local_book()
        broker.portfolio = {
            "cash": 1.0,
            "positions": [{"symbol": "QQQ", "quantity": 99}],
        }
        second = context.bootstrap_local_book()

        assert first["bootstrapped"] is True
        assert second["bootstrapped"] is False
        assert book.positions() == {"SPY": 1.0}
        assert book.cash() == 900.0
        bootstrap_events = [event for event in audit.events
                            if event[1] == "local_book_bootstrapped"]
        assert len(bootstrap_events) == 1
        assert bootstrap_events[0][2]["account_id_key"] == "ACCOUNT-1"

    def test_uninitialized_reconciliation_fails_closed_and_engages_switch(
            self, tmp_path):
        context, _, broker, _, book, kill_switch, audit = make_context(tmp_path)
        assert book.reconciliation_snapshot()["initialized"] is False

        with pytest.raises(RuntimeError, match="explicit bootstrap"):
            context.run_reconciliation()

        assert broker.status_calls == []
        assert len(kill_switch.engagements) == 1
        reason, actor = kill_switch.engagements[0]
        assert "local book is uninitialized" in reason
        assert actor == "live_context"
        assert not [event for event in audit.events
                    if event[1] == "reconciliation"]

    def test_reconciliation_mismatch_is_audited_and_engages_switch(
            self, tmp_path):
        broker = FakeBroker({
            "cash": 800.0,
            "positions": [{"symbol": "SPY", "quantity": 2}],
        })
        context, _, _, _, book, kill_switch, audit = make_context(
            tmp_path, broker=broker)
        assert book.bootstrap({"SPY": 1}, 900.0) is True

        result = context.run_reconciliation()

        assert result["ok"] is False
        assert {(item["kind"], item["symbol"])
                for item in result["mismatches"]} == {
                    ("position", "SPY"), ("cash", None)}
        assert context.last_reconciliation == result
        assert len(kill_switch.engagements) == 1
        assert kill_switch.engagements[0][1] == "live_context"
        events = [event for event in audit.events
                  if event[1] == "reconciliation"]
        assert len(events) == 1
        assert events[0][2]["ok"] is False


class TestLifecycleAndAutomation:
    def test_shutdown_refuses_new_leases_and_waits_for_active_lease(
            self, tmp_path):
        context, _, _, executor, *_ = make_context(tmp_path)
        entered = threading.Event()
        release = threading.Event()

        def hold_operation():
            with context.operation("long-read"):
                entered.set()
                assert release.wait(timeout=2.0)

        operation_thread = threading.Thread(target=hold_operation)
        operation_thread.start()
        assert entered.wait(timeout=1.0)

        result = []
        shutdown_thread = threading.Thread(
            target=lambda: result.append(context.shutdown(timeout=2.0)))
        shutdown_thread.start()
        deadline = time.monotonic() + 1.0
        while context.state == "ready" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert context.state == "closing"
        assert shutdown_thread.is_alive()
        with pytest.raises(LiveContextClosed, match="closing"):
            with context.operation("late-work"):
                pass

        release.set()
        operation_thread.join(timeout=1.0)
        shutdown_thread.join(timeout=1.0)
        assert not operation_thread.is_alive()
        assert not shutdown_thread.is_alive()
        assert result == [True]
        assert context.state == "closed"
        assert executor.close_calls == [2.0]
        assert context.shutdown() is False

    def test_configure_session_builds_scheduler_but_never_starts_it(
            self, tmp_path, monkeypatch):
        built = []

        class NeverAutoStartScheduler:
            def __init__(self, session, **kwargs):
                self.session = session
                self.kwargs = kwargs
                self.start_calls = 0
                self.stop_calls = []
                built.append(self)

            def start(self):
                self.start_calls += 1
                raise AssertionError("configure_session must not auto-start")

            def stop(self, join_timeout=10.0):
                self.stop_calls.append(join_timeout)
                return False

        monkeypatch.setattr(
            live_context_module, "LiveScheduler", NeverAutoStartScheduler)
        context, _, broker, executor, book, kill_switch, audit = make_context(
            tmp_path)
        portfolio = PortfolioManager(10_000.0)
        desk = object()
        market_hours = object()

        session = context.configure_session(
            portfolio=portfolio,
            data_fn=lambda: {},
            desk=desk,
            interval_minutes=7,
            market_hours=market_hours,
        )

        assert context.session is session
        assert session.desk is desk
        assert session.broker is broker
        assert session.portfolio is portfolio
        assert session.executor is executor
        assert session.local_book is book
        assert session.kill_switch is kill_switch
        assert session.audit is audit
        assert session.enforce_market_hours is True
        assert len(built) == 1
        assert context.scheduler is built[0]
        assert built[0].start_calls == 0
        assert built[0].kwargs["interval_minutes"] == 7
        assert built[0].kwargs["market_hours"] is market_hours
        assert "max_consecutive_errors" not in built[0].kwargs

        context.shutdown(timeout=0.25)
        assert built[0].stop_calls == [0.25]

    def test_verified_production_session_pauses_after_one_scheduler_error(
            self, tmp_path, monkeypatch):
        built = []

        class CapturingScheduler:
            def __init__(self, session, **kwargs):
                self.session = session
                self.kwargs = kwargs
                built.append(self)

            def stop(self, join_timeout=10.0):
                return False

        class FakeVerifiedDeployment:
            def __init__(self):
                self.bound_executor = None

            def bind(self, _context, _desk, _interval_minutes):
                deployment = self

                class Guard:
                    def __call__(self, **_kwargs):
                        return None

                    def bind_executor(self, executor):
                        deployment.bound_executor = executor

                return Guard()

        import deployment.live as deployment_live
        monkeypatch.setattr(
            live_context_module, "LiveScheduler", CapturingScheduler)
        monkeypatch.setattr(
            deployment_live, "VerifiedFoundationDeployment",
            FakeVerifiedDeployment)

        db_path = tmp_path / "production-live.db"
        book = LocalBook(
            str(db_path), env="production", account_id_key="ACCOUNT-1")
        context = LiveExecutionContext(
            auth_manager=FakeAuthManager(db_path, env="production"),
            client=FakeClient(),
            kill_switch=FakeKillSwitch(),
            audit=FakeAudit(),
            account_id_key="ACCOUNT-1",
            broker=FakeBroker(),
            local_book=book,
            executor=FakeExecutor(),
        )

        verified = FakeVerifiedDeployment()
        context.configure_session(
            portfolio=PortfolioManager(10_000.0),
            data_fn=lambda: {},
            desk=object(),
            verified_deployment=verified,
        )

        assert len(built) == 1
        assert verified.bound_executor is context.executor
        assert built[0].kwargs["max_consecutive_errors"] == 1
        assert built[0].kwargs["hold_after_first_cycle"] is True
        context.shutdown()
