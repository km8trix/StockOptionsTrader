"""LiveTradingSession (E5): intents flow to the executor, risk-blocked
intents never reach it, every step is audited in order, the kill switch
halts cleanly, paper-mode parity, and the C19 reconciliation wiring."""

from __future__ import annotations

import pytest

from brokers.etrade_client import build_equity_order
from core.models import Asset, AssetType, OrderType
from desks.base import DeskIntent
from desks.orchestrator import FundOrchestrator
from tests.test_fund_orchestrator import ScriptedDesk
from tests.test_fund_orchestrator import intent as fund_intent
from tests.test_fund_orchestrator import stock as fund_stock
from portfolio.manager import PortfolioManager
from portfolio.targets import TargetPosition
from portfolio.structures import LegAction, StructureIntent, StructureLeg
from utils.audit import AuditLog
from utils.kill_switch import KillSwitch
from utils.live_session import LiveTradingSession

SPY = Asset("SPY", AssetType.STOCK)
QQQ = Asset("QQQ", AssetType.STOCK)


class FakeDesk:
    key = "fake"
    capital_allocation = 0.5

    def __init__(self, intents, approved=None):
        self.intents = intents
        self.approved = approved if approved is not None else list(intents)
        self.generate_called = False

    def set_clock(self, now):
        self.clock = now

    def generate_intents(self, all_data, date, portfolio):
        self.generate_called = True
        return list(self.intents)

    def apply_risk(self, intents, portfolio, all_data, date):
        return [i for i in intents if i in self.approved]


class FakePortfolio:
    def get_portfolio_value(self):
        return 100_000.0


class FakeBroker:
    def __init__(self, price=50.0):
        self.price = price
        self.orders = []

    def place_order(self, asset, order_type, quantity, limit_price):
        self.orders.append((asset, order_type, quantity, limit_price))
        return f"ORD-{len(self.orders)}"

    def get_current_price(self, symbol):
        return self.price

    def get_portfolio_status(self):
        return {"cash": 100_000.0, "positions": []}


class FakeExecutor:
    def __init__(self, on_execute=None):
        self.calls = []
        self.on_execute = on_execute

    def execute(self, side, instrument, quantity, **kwargs):
        self.calls.append((side, instrument, quantity))
        if self.on_execute is not None:
            self.on_execute(len(self.calls))
        return {"status": "filled", "avg_fill": 50.0,
                "shortfall_per_unit": 0.0, "fills": []}


@pytest.fixture
def rails(tmp_path):
    db_path = str(tmp_path / "session.db")
    audit = AuditLog(db_path, env="sandbox")
    switch = KillSwitch(db_path, audit=audit)
    return audit, switch


def make_session(desk, broker, executor, audit, switch, **kwargs):
    return LiveTradingSession(
        desk=desk, broker=broker, portfolio=FakePortfolio(),
        data_fn=lambda: {"SPY": None}, executor=executor, audit=audit,
        kill_switch=switch, **kwargs)


def test_evaluate_once_skips_off_hours_when_enforced(rails):
    """Gap 3: with enforce_market_hours, a manual evaluate_once outside the
    NYSE session does not trade (the scheduler gates the autonomous loop, but a
    direct call otherwise transmits)."""
    from datetime import datetime, timezone
    audit, switch = rails
    desk = FakeDesk([])
    saturday = datetime(2026, 6, 13, 15, 0, tzinfo=timezone.utc)  # weekend
    session = make_session(desk, FakeBroker(), None, audit, switch,
                           enforce_market_hours=True,
                           clock=lambda: saturday)
    result = session.evaluate_once()
    assert result["status"] == "market_closed"


class TestOrchestratorMode:
    """Fund mode: a FundOrchestrator drives the live session in place of one
    desk — the exactly-one-of guard, and netted/account-approved intents reach
    the executor through the same path (single-desk path is unchanged)."""

    def test_exactly_one_of_desk_or_orchestrator_required(self, rails):
        audit, switch = rails
        orch = FundOrchestrator([ScriptedDesk('s', 1.0)])
        with pytest.raises(ValueError, match='exactly one'):
            LiveTradingSession(desk=FakeDesk([]), broker=FakeBroker(),
                               portfolio=FakePortfolio(),
                               data_fn=lambda: {}, orchestrator=orch)
        with pytest.raises(ValueError, match='exactly one'):
            LiveTradingSession(desk=None, broker=FakeBroker(),
                               portfolio=FakePortfolio(),
                               data_fn=lambda: {})

    def test_fund_mode_executes_netted_intents(self, rails):
        audit, switch = rails
        # The orchestrator runs the REAL Desk.apply_risk (daily-loss circuit
        # reads portfolio_history), so use a real PortfolioManager here.
        from portfolio.manager import PortfolioManager
        orch = FundOrchestrator([ScriptedDesk(
            's', 1.0, intents=[fund_intent(fund_stock('SPY'), 'BUY', 0.1)])])
        executor = FakeExecutor()
        session = LiveTradingSession(
            desk=None, broker=FakeBroker(), portfolio=PortfolioManager(1e5),
            data_fn=lambda: {'SPY': None}, executor=executor, audit=audit,
            kill_switch=switch, orchestrator=orch)
        result = session.evaluate_once()
        assert result['status'] == 'ok'
        assert result['approved'] == 1
        assert len(executor.calls) == 1
        side, asset, qty = executor.calls[0]
        assert side == 'BUY' and asset.symbol == 'SPY' and qty > 0


class TestIntentGenerationFailClosed:
    """A raise during intent generation/risk (data fetch, a desk's
    generate_intents, the orchestrator's net/apply_risk/aggregator) becomes a
    clean AUDITED session_halted — never a bare exception out of
    evaluate_once leaving the audit trail open."""

    def test_fund_mode_desk_error_halts_cleanly(self, rails):
        audit, switch = rails
        from portfolio.manager import PortfolioManager

        class Boom(ScriptedDesk):
            def generate_intents(self, all_data, date, portfolio):
                raise RuntimeError("desk boom")

        orch = FundOrchestrator([Boom('boom', 1.0)])
        executor = FakeExecutor()
        session = LiveTradingSession(
            desk=None, broker=FakeBroker(), portfolio=PortfolioManager(1e5),
            data_fn=lambda: {'SPY': None}, executor=executor, audit=audit,
            kill_switch=switch, orchestrator=orch)
        result = session.evaluate_once()  # must NOT raise
        assert result["status"] == "halted"
        assert result["reason"] == "intent_generation_error"
        assert executor.calls == []  # nothing executed

    def test_desk_mode_generate_error_halts_cleanly(self, rails):
        audit, switch = rails

        class BoomDesk:
            key = "boom"
            capital_allocation = 1.0

            def set_clock(self, now):
                pass

            def generate_intents(self, all_data, date, portfolio):
                raise RuntimeError("desk boom")

            def apply_risk(self, intents, portfolio, all_data, date):
                return intents

        executor = FakeExecutor()
        session = make_session(BoomDesk(), FakeBroker(), executor, audit,
                               switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "intent_generation_error"
        assert executor.calls == []


class TestIntentFlow:
    def test_approved_intents_reach_executor_blocked_never_do(self, rails):
        audit, switch = rails
        buy_spy = DeskIntent(SPY, "BUY", 0.05, "signal", quantity=10)
        buy_qqq = DeskIntent(QQQ, "BUY", 0.50, "too big", quantity=5)
        desk = FakeDesk([buy_spy, buy_qqq], approved=[buy_spy])
        executor = FakeExecutor()
        session = make_session(desk, FakeBroker(), executor, audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "ok"
        assert result["generated"] == 2 and result["approved"] == 1
        # ONLY the approved intent was executed; the risk-blocked one
        # never reached the executor.
        assert executor.calls == [("BUY", SPY, 10)]
        assert result["reports"][0]["status"] == "filled"

    def test_every_step_audited_in_order(self, rails):
        audit, switch = rails
        buy_spy = DeskIntent(SPY, "BUY", 0.05, "signal", quantity=10)
        sell_qqq = DeskIntent(QQQ, "SELL", 0.05, "exit", quantity=5)
        desk = FakeDesk([buy_spy, sell_qqq])
        session = make_session(desk, FakeBroker(), FakeExecutor(), audit,
                               switch)
        session.evaluate_once()
        events = [e["event_type"] for e in audit.entries(limit=50,
                                                         ascending=True)]
        assert events == [
            "session_evaluate",
            "desk_intent", "desk_intent",
            "execution_start", "execution_report",
            "execution_start", "execution_report",
        ]
        intent_rows = audit.entries(event_type="desk_intent",
                                    ascending=True)
        assert intent_rows[0]["payload"]["symbol"] == "SPY"
        assert intent_rows[1]["payload"]["action"] == "SELL"

    def test_sizing_fallback_when_intent_has_no_quantity(self, rails):
        audit, switch = rails
        intent = DeskIntent(SPY, "BUY", 0.1, "signal")  # no quantity
        desk = FakeDesk([intent])
        executor = FakeExecutor()
        session = make_session(desk, FakeBroker(price=50.0), executor,
                               audit, switch)
        session.evaluate_once()
        # 100k * 0.5 allocation * 0.1 fraction = $5k at $50 -> 100 shares.
        assert executor.calls == [("BUY", SPY, 100)]

    def test_unsizable_intent_skipped_and_audited(self, rails):
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.1, "signal")])
        broker = FakeBroker(price=50.0)
        broker.price = None  # quotes dark
        executor = FakeExecutor()
        session = make_session(desk, broker, executor, audit, switch)
        result = session.evaluate_once()
        assert executor.calls == []
        assert result["reports"] == []
        assert audit.entries(event_type="execution_skipped")


class TestTargetNativeFlow:
    """Opt-in desired-position construction stays idempotent around work."""

    class TargetDesk:
        key = "target"
        capital_allocation = 1.0
        target_native_enabled = True

        def __init__(self, quantity):
            self.quantity = quantity

        def set_clock(self, now):
            self.clock = now

        def generate_targets(self, all_data, date, portfolio, snapshot):
            return [TargetPosition(
                SPY,
                self.quantity,
                owner=self.key,
                strategy=self.key,
                reason="desired SPY position",
                metadata={"size_fraction": 0.10,
                          "deployment_id": "test-v1"},
            )]

        def apply_risk(self, intents, portfolio, all_data, date):
            return list(intents)

    @staticmethod
    def _active_reservation(quantity=100):
        request = build_equity_order(
            "SPY", "BUY", quantity, 50.0, "tp123456789012345678")
        return {
            "reservations": [{
                "reservation_id": "tp123456789012345678",
                "status": "ACTIVE",
                "units": quantity,
                "metadata": {"order_request": request},
                "orders": [{
                    "order_id": "ORDER-1",
                    "status": "OPEN",
                    "cumulative_filled_units": 0,
                }],
            }],
        }

    def test_delta_identity_reaches_patient_executor(self, rails):
        audit, switch = rails

        class IdentityExecutor(FakeExecutor):
            def execute(self, side, instrument, quantity, **kwargs):
                self.calls.append((side, instrument, quantity, kwargs))
                return {"status": "filled", "avg_fill": 50.0,
                        "shortfall_per_unit": 0.0, "fills": []}

        executor = IdentityExecutor()
        session = LiveTradingSession(
            desk=self.TargetDesk(100), broker=FakeBroker(),
            portfolio=PortfolioManager(100_000.0), data_fn=lambda: {},
            executor=executor, audit=audit, kill_switch=switch)

        result = session.evaluate_once()

        assert result["status"] == "ok"
        assert executor.calls[0][:3] == ("BUY", SPY, 100)
        execution_id = executor.calls[0][3]["execution_id"]
        assert execution_id.startswith("tp") and len(execution_id) == 20
        assert audit.entries(event_type="target_position")
        assert audit.entries(event_type="target_order_delta")

    def test_active_reservation_makes_repeated_target_a_noop(self, rails):
        audit, switch = rails

        class Gate:
            def snapshot(self):
                return TestTargetNativeFlow._active_reservation()

        broker = FakeBroker()
        broker.client = type("Client", (), {"reservation_gate": Gate()})()
        executor = FakeExecutor()
        session = LiveTradingSession(
            desk=self.TargetDesk(100), broker=broker,
            portfolio=PortfolioManager(100_000.0), data_fn=lambda: {},
            executor=executor, audit=audit, kill_switch=switch)

        result = session.evaluate_once()

        assert result["status"] == "ok"
        assert result["generated"] == 0
        assert executor.calls == []

    def test_changed_target_cancels_work_before_reversal(self, rails):
        audit, switch = rails

        class Gate:
            def snapshot(self):
                return TestTargetNativeFlow._active_reservation()

        class CancelBroker(FakeBroker):
            def __init__(self):
                super().__init__()
                self.cancelled = []
                self.client = type(
                    "Client", (), {"reservation_gate": Gate()})()

            def cancel_order(self, order_id):
                self.cancelled.append(order_id)
                return True

            def order_status(self, order_id):
                return {"status": "CANCEL_REQUESTED",
                        "filled_quantity": 0}

        broker = CancelBroker()
        executor = FakeExecutor()
        session = LiveTradingSession(
            desk=self.TargetDesk(0), broker=broker,
            portfolio=PortfolioManager(100_000.0), data_fn=lambda: {},
            executor=executor, audit=audit, kill_switch=switch)

        result = session.evaluate_once()

        assert result["status"] == "pending"
        assert result["reason"] == "target_order_cancellation_requested"
        assert broker.cancelled == ["ORDER-1"]
        assert executor.calls == []
        assert audit.entries(event_type="target_order_cancel")


class TestAtomicStructureFlow:
    """Canonical packages reach the executor once, never leg by leg."""

    class StructureDesk(FakeDesk):
        def __init__(self, structure):
            super().__init__([])
            self.structure = structure

        def generate_structure_intents(self, all_data, date, portfolio):
            return [self.structure]

    @staticmethod
    def _package(*, closing=False):
        expiry = "2026-08-21"
        actions = ((LegAction.BUY_CLOSE, LegAction.SELL_CLOSE)
                   if closing else
                   (LegAction.SELL_OPEN, LegAction.BUY_OPEN))
        legs = (
            StructureLeg(
                Asset("SPY", AssetType.CALL, 450.0, expiry), actions[0]),
            StructureLeg(
                Asset("SPY", AssetType.CALL, 455.0, expiry), actions[1]),
        )
        return StructureIntent(
            legs=legs, quantity=2,
            net_price=-0.50 if closing else 1.0,
            max_loss=0.0 if closing else 800.0,
            greeks={"delta": -2.0, "vega": -15.0},
            reason="atomic vertical")

    def test_open_package_is_one_execution_with_explicit_actions(self, rails):
        audit, switch = rails

        class PackageExecutor(FakeExecutor):
            def execute(self, side, instrument, quantity, **kwargs):
                self.calls.append((side, instrument, quantity, kwargs))
                return {"status": "filled", "avg_fill": 1.0,
                        "shortfall_per_unit": 0.0,
                        "fills": [{"qty": 2, "price": 1.0}]}

        package = self._package()
        executor = PackageExecutor()
        session = LiveTradingSession(
            desk=self.StructureDesk(package), broker=FakeBroker(),
            portfolio=PortfolioManager(100_000.0), data_fn=lambda: {},
            executor=executor, audit=audit, kill_switch=switch)

        result = session.evaluate_once()

        assert result["status"] == "ok"
        assert result["generated"] == 0
        assert result["generated_structures"] == 1
        assert len(executor.calls) == 1
        side, legs, quantity, kwargs = executor.calls[0]
        assert side == "SELL" and quantity == 2
        assert [leg["action"] for leg in legs] == ["SELL_OPEN", "BUY_OPEN"]
        assert kwargs == {"execution_id": package.intent_id,
                          "closing": False}
        assert audit.entries(event_type="structure_execution_report")

    def test_close_package_preserves_close_lifecycle(self, rails):
        audit, switch = rails

        class PackageExecutor(FakeExecutor):
            def execute(self, side, instrument, quantity, **kwargs):
                self.calls.append((side, instrument, quantity, kwargs))
                return {"status": "filled", "fills": []}

        package = self._package(closing=True)
        executor = PackageExecutor()
        session = LiveTradingSession(
            desk=self.StructureDesk(package), broker=FakeBroker(),
            portfolio=PortfolioManager(100_000.0), data_fn=lambda: {},
            executor=executor, audit=audit, kill_switch=switch)

        assert session.evaluate_once()["status"] == "ok"
        side, legs, _, kwargs = executor.calls[0]
        assert side == "BUY"
        assert [leg["action"] for leg in legs] == [
            "BUY_CLOSE", "SELL_CLOSE"]
        assert kwargs["closing"] is True


class TestKillSwitchHalts:
    def test_engaged_before_evaluation_halts_cleanly(self, rails):
        audit, switch = rails
        switch.engage("manual", actor="ops")
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=1)])
        executor = FakeExecutor()
        session = make_session(desk, FakeBroker(), executor, audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert desk.generate_called is False  # nothing ran
        assert executor.calls == []
        halted = audit.entries(event_type="session_halted")
        assert halted[0]["payload"]["reason"] == "kill_switch_engaged"

    def test_mid_loop_engagement_stops_remaining_intents(self, rails):
        audit, switch = rails
        intents = [DeskIntent(SPY, "BUY", 0.05, "a", quantity=1),
                   DeskIntent(QQQ, "BUY", 0.05, "b", quantity=2)]
        desk = FakeDesk(intents)
        # First execution trips the breaker/operator: switch engages.
        executor = FakeExecutor(
            on_execute=lambda _n: switch.engage("breaker", "circuit"))
        session = make_session(desk, FakeBroker(), executor, audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert len(executor.calls) == 1  # the second intent never ran


class TestExecutionErrorHalts:
    """A broker/auth failure during execution (midnight-ET token expiry
    mid-order is THE case) is a clean, audited stop: execution_report
    with the typed reason, session_halted, remaining intents never run —
    and evaluate_once returns a halted result instead of raising."""

    def _two_intents(self):
        return [DeskIntent(SPY, "BUY", 0.05, "a", quantity=10),
                DeskIntent(QQQ, "BUY", 0.05, "b", quantity=5)]

    def test_executor_raising_auth_expired_halts_and_audits(self, rails):
        from brokers.etrade_auth import EtradeAuthExpired

        audit, switch = rails

        class RaisingExecutor:
            calls = 0

            def execute(self, side, instrument, quantity, **kwargs):
                RaisingExecutor.calls += 1
                raise EtradeAuthExpired(
                    "E*TRADE access token hit the midnight-ET hard "
                    "expiry; re-authorization required")

        desk = FakeDesk(self._two_intents())
        session = make_session(desk, FakeBroker(), RaisingExecutor(),
                               audit, switch)
        result = session.evaluate_once()  # must NOT raise
        assert result["status"] == "halted"
        assert result["reason"] == "execution_error"
        assert RaisingExecutor.calls == 1  # the second intent never ran
        assert result["reports"][0]["status"] == "error"
        assert result["reports"][0]["error_type"] == "EtradeAuthExpired"
        events = [e["event_type"] for e in audit.entries(limit=50,
                                                         ascending=True)]
        # The trail CLOSES: no dangling execution_start.
        assert events == [
            "session_evaluate", "desk_intent", "desk_intent",
            "execution_start", "execution_report", "session_halted",
        ]
        report_row = audit.entries(event_type="execution_report")[0]
        assert report_row["payload"]["status"] == "error"
        assert report_row["payload"]["error_type"] == "EtradeAuthExpired"
        halted_row = audit.entries(event_type="session_halted")[0]
        assert halted_row["payload"]["reason"] == "execution_error"
        assert "midnight-ET" in halted_row["payload"]["error"]

    def test_executor_error_report_halts_remaining_intents(self, rails):
        """The PatientExecutor's own conversion path: a terminal 'error'
        report (no exception) must halt the remaining intents too."""
        audit, switch = rails

        class ErrorReportExecutor:
            calls = 0

            def execute(self, side, instrument, quantity, **kwargs):
                ErrorReportExecutor.calls += 1
                return {"status": "error", "fills": [],
                        "error": "transport died mid-poll",
                        "error_type": "EtradeUnavailable"}

        desk = FakeDesk(self._two_intents())
        session = make_session(desk, FakeBroker(), ErrorReportExecutor(),
                               audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "execution_error"
        assert ErrorReportExecutor.calls == 1
        halted_row = audit.entries(event_type="session_halted")[0]
        assert halted_row["payload"]["error_type"] == "EtradeUnavailable"

    def test_paper_path_place_order_raising_halts_and_audits(self, rails):
        audit, switch = rails

        class BrokenBroker(FakeBroker):
            def place_order(self, asset, order_type, quantity,
                            limit_price):
                raise ConnectionError("broker socket died")

        desk = FakeDesk(self._two_intents())
        session = make_session(desk, BrokenBroker(), None, audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "execution_error"
        report_row = audit.entries(event_type="execution_report")[0]
        assert report_row["payload"]["status"] == "error"
        assert report_row["payload"]["error_type"] == "ConnectionError"


class TestExecutionReportEnrichment:
    """Step 5: execution_report rows additionally record arrival_mid,
    asset_type and quantity so the parity harness can replay them exactly.
    Additive — the existing keys and the hash chain are unaffected."""

    def test_report_payload_carries_parity_fields(self, rails):
        audit, switch = rails

        class MidExecutor:
            def execute(self, side, instrument, quantity, **kwargs):
                return {"status": "filled", "avg_fill": 100.10,
                        "arrival_mid": 100.0, "shortfall_per_unit": 0.10,
                        "fills": [{"qty": 10, "price": 100.10}]}

        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=10)])
        session = make_session(desk, FakeBroker(), MidExecutor(), audit, switch)
        session.evaluate_once()

        reports = audit.entries(event_type="execution_report")
        assert reports, "expected an execution_report row"
        payload = reports[0]["payload"]
        assert payload["arrival_mid"] == pytest.approx(100.0)
        assert payload["asset_type"] == "stock"
        assert payload["quantity"] == 10
        assert payload["filled_quantity"] == 10  # actual filled size
        assert payload["commission"] is None  # broker does not surface it yet
        # Legacy keys still present; additive fields do not break the chain.
        assert payload["avg_fill"] == pytest.approx(100.10)
        assert payload["shortfall_per_unit"] == pytest.approx(0.10)
        assert audit.verify_chain()["ok"] is True


class FakeGate:
    """Zero-arg daily-loss gate double: scripted results (last sticky);
    exceptions in the queue are raised."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        item = (self.results.pop(0) if len(self.results) > 1
                else self.results[0])
        if isinstance(item, Exception):
            raise item
        return item


class TestCircuitBreakerHalts:
    """E5: the daily-loss gate is evaluated at the top of EVERY
    evaluate_once(), before any intent is generated or executed."""

    def test_breach_halts_before_any_intent_runs(self, rails):
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=1)])
        executor = FakeExecutor()
        gate = FakeGate({"breached": True, "loss_pct": 0.025,
                         "limit_pct": 0.02})
        session = make_session(desk, FakeBroker(), executor, audit, switch,
                               circuit_breaker=gate)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "daily_loss_circuit_breaker"
        assert desk.generate_called is False  # halted BEFORE intents
        assert executor.calls == []
        halted = audit.entries(event_type="session_halted")[0]
        assert halted["payload"]["reason"] == "daily_loss_circuit_breaker"
        assert halted["payload"]["loss_pct"] == 0.025

    def test_gate_error_fails_closed(self, rails):
        """A rail that cannot be read must halt the cycle, never wave it
        through."""
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=1)])
        executor = FakeExecutor()
        session = make_session(
            desk, FakeBroker(), executor, audit, switch,
            circuit_breaker=FakeGate(RuntimeError("balances dark")))
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "circuit_breaker_error"
        assert executor.calls == []
        halted = audit.entries(event_type="session_halted")[0]
        assert "balances dark" in halted["payload"]["error"]

    def test_clean_gate_runs_once_and_trading_proceeds(self, rails):
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=1)])
        executor = FakeExecutor()
        gate = FakeGate({"breached": False, "loss_pct": 0.001})
        session = make_session(desk, FakeBroker(), executor, audit, switch,
                               circuit_breaker=gate)
        result = session.evaluate_once()
        assert result["status"] == "ok"
        assert gate.calls == 1
        assert executor.calls == [("BUY", SPY, 1)]

    def test_gate_auto_discovered_from_broker(self, rails):
        """LiveEtradeBroker exposes its auto-wired gate as
        broker.circuit_breaker; the session picks it up by default."""
        audit, switch = rails
        broker = FakeBroker()
        broker.circuit_breaker = FakeGate({"breached": True,
                                           "loss_pct": 0.03,
                                           "limit_pct": 0.02})
        session = make_session(FakeDesk([]), broker, FakeExecutor(), audit,
                               switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "daily_loss_circuit_breaker"
        assert broker.circuit_breaker.calls == 1

    def test_explicit_none_disables_discovery(self, rails):
        audit, switch = rails
        broker = FakeBroker()
        broker.circuit_breaker = FakeGate({"breached": True})
        session = make_session(FakeDesk([]), broker, FakeExecutor(), audit,
                               switch, circuit_breaker=None)
        assert session.evaluate_once()["status"] == "ok"
        assert broker.circuit_breaker.calls == 0


class TestPaperParityAndRenew:
    def test_no_executor_routes_direct_market_orders(self, rails):
        """Paper-mode parity: the same session drives PaperTrader-style
        brokers with direct place_order calls."""
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=7)])
        broker = FakeBroker()
        session = make_session(desk, broker, None, audit, switch)
        result = session.evaluate_once()
        assert broker.orders == [(SPY, OrderType.BUY, 7, None)]
        assert result["reports"][0] == {"status": "submitted",
                                        "order_id": "ORD-1"}

    def test_opportunistic_renew_called(self, rails):
        audit, switch = rails

        class FakeAuth:
            renewed = 0

            def renew(self):
                FakeAuth.renewed += 1
                return True

        desk = FakeDesk([])
        session = make_session(desk, FakeBroker(), FakeExecutor(), audit,
                               switch, auth_manager=FakeAuth())
        session.evaluate_once()
        assert FakeAuth.renewed == 1


class TestReconciliationWiring:
    def test_mismatch_engages_kill_switch_and_audits(self, rails):
        audit, switch = rails
        desk = FakeDesk([])
        broker = FakeBroker()  # broker says: no positions, 100k cash
        session = make_session(desk, broker, FakeExecutor(), audit, switch)
        result = session.run_reconciliation({"AAPL": 100}, 100_000.0)
        assert result["ok"] is False
        assert switch.engaged() is True  # C19: caller engages the switch
        assert session.last_reconciliation == result
        rows = audit.entries(event_type="reconciliation")
        assert rows and rows[0]["payload"]["ok"] is False
        # And the halted book stays halted: next evaluation refuses.
        assert session.evaluate_once()["status"] == "halted"

    def test_clean_reconciliation_leaves_switch_alone(self, rails):
        audit, switch = rails
        session = make_session(FakeDesk([]), FakeBroker(), FakeExecutor(),
                               audit, switch)
        result = session.run_reconciliation({}, 100_000.0)
        assert result["ok"] is True
        assert switch.engaged() is False

    def test_single_arg_is_a_caller_bug_not_a_halt(self, rails, tmp_path):
        """Exactly one of local_positions/local_cash is ambiguous (which
        half comes from the book?) — ValueError, and NO kill switch:
        nothing was reconciled, nothing traded on a wrong book."""
        from brokers.local_book import LocalBook
        audit, switch = rails
        book = LocalBook(str(tmp_path / "book.db"), env="sandbox")
        session = make_session(FakeDesk([]), FakeBroker(), None, audit,
                               switch, local_book=book)
        with pytest.raises(ValueError, match="provide both or neither"):
            session.run_reconciliation(local_positions={})
        with pytest.raises(ValueError, match="provide both or neither"):
            session.run_reconciliation(local_cash=100_000.0)
        assert switch.engaged() is False

    def test_book_read_failure_fails_closed(self, rails):
        """FAIL-CLOSED rail: a local_book whose reads raise engages the
        kill switch BEFORE the exception propagates — an unreadable book
        is indistinguishable from drift (mirrors _check_circuit_breaker)."""
        audit, switch = rails

        class ExplodingBook:
            def positions(self):
                raise RuntimeError("book db gone")

            def cash(self):
                return 0.0

        session = make_session(FakeDesk([]), FakeBroker(), None, audit,
                               switch, local_book=ExplodingBook())
        with pytest.raises(RuntimeError, match="book db gone"):
            session.run_reconciliation()
        assert switch.engaged() is True

    def test_reconcile_fn_failure_fails_closed(self, rails):
        """Same rail for the reconcile itself (broker dark mid-compare):
        engage the switch, then re-raise."""
        audit, switch = rails

        def dark_broker_reconcile(local_positions, local_cash, broker):
            raise ConnectionError("broker unreachable")

        session = make_session(FakeDesk([]), FakeBroker(), None, audit,
                               switch, reconcile_fn=dark_broker_reconcile)
        with pytest.raises(ConnectionError, match="broker unreachable"):
            session.run_reconciliation({}, 100_000.0)
        assert switch.engaged() is True

    def test_cash_tolerance_passes_through(self, rails, tmp_path):
        """Fee-aware operational reconcile: a $3 unreported-fee cash
        drift passes at cash_tolerance=5.00 but fails (and halts) at the
        strict default — positions unaffected either way."""
        from brokers.local_book import LocalBook
        audit, switch = rails
        book = LocalBook(str(tmp_path / "book.db"), env="sandbox")
        book.set_cash(100_000.0 - 3.0)  # FakeBroker reports 100k
        session = make_session(FakeDesk([]), FakeBroker(), None, audit,
                               switch, local_book=book)
        assert session.run_reconciliation(cash_tolerance=5.00)["ok"] is True
        assert switch.engaged() is False
        # cash_tolerance=None (the default) keeps reconcile's strict $0.01.
        result = session.run_reconciliation()
        assert result["ok"] is False
        assert switch.engaged() is True


class TestLocalBookWiring:
    """Step 5 opt-in persistent book: local_book=None is byte-identical
    (pinned — the session never even touches order_status); when set,
    CONFIRMED fills land in the book and no-arg run_reconciliation reads
    book.positions()/cash()."""

    @staticmethod
    def _book(tmp_path):
        from brokers.local_book import LocalBook
        return LocalBook(str(tmp_path / "book.db"), env="sandbox")

    def test_default_none_never_touches_order_status(self, rails):
        """Byte-identity pin: without a local_book the direct path is
        exactly as before — a broker with a booby-trapped order_status
        (and get_portfolio_status) still evaluates cleanly."""
        audit, switch = rails

        class TrappedBroker(FakeBroker):
            def order_status(self, order_id):
                raise AssertionError("order_status must not be called")

            def get_portfolio_status(self):
                raise AssertionError(
                    "get_portfolio_status must not be called")

        intent = DeskIntent(SPY, "BUY", 0.1, "t", quantity=4)
        desk = FakeDesk([intent])
        broker = TrappedBroker()
        session = make_session(desk, broker, None, audit, switch)
        assert session.local_book is None
        result = session.evaluate_once()
        assert result["status"] == "ok"
        assert len(broker.orders) == 1

    def test_direct_path_records_confirmed_fill(self, rails, tmp_path):
        """The direct-market path confirms via order_status; a pending
        order gets ONE get_portfolio_status nudge (paper-parity: that is
        where PaperTrader processes fills) before the second poll."""
        audit, switch = rails
        book = self._book(tmp_path)
        book.set_cash(1_000.0)

        class StatusBroker(FakeBroker):
            def __init__(self):
                super().__init__()
                self.processed = False

            def get_portfolio_status(self):
                self.processed = True
                return super().get_portfolio_status()

            def order_status(self, order_id):
                if not self.processed:
                    return {"status": "OPEN", "filled_quantity": 0,
                            "avg_fill_price": None}
                return {"status": "FILLED", "filled_quantity": 4,
                        "avg_fill_price": 50.0}

        intent = DeskIntent(SPY, "BUY", 0.1, "t", quantity=4)
        session = make_session(FakeDesk([intent]), StatusBroker(), None,
                               audit, switch, local_book=book)
        result = session.evaluate_once()
        assert result["status"] == "ok"
        assert book.positions() == {"SPY": 4.0}
        assert book.cash() == pytest.approx(1_000.0 - 4 * 50.0)

    def test_direct_path_books_partial_then_cancelled_fill(self, rails,
                                                           tmp_path):
        """A partial-then-cancelled order (terminal CANCELLED with real
        fills attached) books the CONFIRMED partial quantity — those 2
        units are a real position change, whatever the status says."""
        audit, switch = rails
        book = self._book(tmp_path)
        book.set_cash(1_000.0)

        class PartialCancelBroker(FakeBroker):
            def order_status(self, order_id):
                return {"status": "CANCELLED", "filled_quantity": 2,
                        "avg_fill_price": 50.0}

        intent = DeskIntent(SPY, "BUY", 0.1, "t", quantity=4)
        session = make_session(FakeDesk([intent]), PartialCancelBroker(),
                               None, audit, switch, local_book=book)
        assert session.evaluate_once()["status"] == "ok"
        assert book.positions() == {"SPY": 2.0}
        assert book.cash() == pytest.approx(1_000.0 - 2 * 50.0)

    def test_direct_path_books_working_partial_after_nudge(self, rails,
                                                           tmp_path):
        """E*TRADE 'PARTIAL' (still working): the second poll's confirmed
        partial quantity is booked; anything filling after the two-poll
        window is reconciliation's job (known bound, see docstring)."""
        audit, switch = rails
        book = self._book(tmp_path)
        book.set_cash(1_000.0)

        class PartialAfterNudge(FakeBroker):
            def __init__(self):
                super().__init__()
                self.nudged = False

            def get_portfolio_status(self):
                self.nudged = True
                return super().get_portfolio_status()

            def order_status(self, order_id):
                if not self.nudged:
                    return {"status": "OPEN", "filled_quantity": 0,
                            "avg_fill_price": None}
                return {"status": "PARTIAL", "filled_quantity": 3,
                        "avg_fill_price": 50.0}

        intent = DeskIntent(SPY, "BUY", 0.1, "t", quantity=4)
        session = make_session(FakeDesk([intent]), PartialAfterNudge(),
                               None, audit, switch, local_book=book)
        assert session.evaluate_once()["status"] == "ok"
        assert book.positions() == {"SPY": 3.0}
        assert book.cash() == pytest.approx(1_000.0 - 3 * 50.0)

    def test_direct_path_unconfirmed_fill_records_nothing(self, rails,
                                                          tmp_path):
        """An order the broker never confirms FILLED must NOT be booked —
        reconciliation, not optimism, resolves it."""
        audit, switch = rails
        book = self._book(tmp_path)
        book.set_cash(500.0)

        class OpenForeverBroker(FakeBroker):
            def order_status(self, order_id):
                return {"status": "OPEN", "filled_quantity": 0,
                        "avg_fill_price": None}

        intent = DeskIntent(SPY, "BUY", 0.1, "t", quantity=4)
        session = make_session(FakeDesk([intent]), OpenForeverBroker(), None,
                               audit, switch, local_book=book)
        assert session.evaluate_once()["status"] == "ok"
        assert book.positions() == {}
        assert book.cash() == 500.0

    def test_executor_path_records_banked_fills_signed(self, rails,
                                                       tmp_path):
        """Executor reports carry per-slice fills; a SELL books negative
        quantity and raises cash."""
        audit, switch = rails
        book = self._book(tmp_path)
        book.record_fill("SPY", 5, 40.0)          # existing long, cash -200

        class SellExecutor(FakeExecutor):
            def execute(self, side, instrument, quantity, **kwargs):
                self.calls.append((side, instrument, quantity))
                return {"status": "filled", "avg_fill": 50.0,
                        "shortfall_per_unit": 0.0,
                        "fills": [{"qty": 3, "price": 50.0, "ts": "t"}]}

        intent = DeskIntent(SPY, "SELL", 0.1, "t", quantity=3)
        session = make_session(FakeDesk([intent]), FakeBroker(),
                               SellExecutor(), audit, switch,
                               local_book=book)
        assert session.evaluate_once()["status"] == "ok"
        snap = book.snapshot()
        assert snap["positions"]["SPY"]["quantity"] == 2.0
        assert snap["cash"] == pytest.approx(-200.0 + 3 * 50.0)

    def test_reservation_gate_is_sole_live_fill_booker(self, rails,
                                                       tmp_path):
        """A reservation-aware client books each broker cumulative update;
        the terminal executor report must not apply those fills twice."""
        audit, switch = rails
        book = self._book(tmp_path)
        book.set_cash(1_000.0)

        class FilledExecutor(FakeExecutor):
            def execute(self, side, instrument, quantity, **kwargs):
                return {"status": "filled", "avg_fill": 50.0,
                        "shortfall_per_unit": 0.0,
                        "fills": [{"qty": 2, "price": 50.0, "ts": "t"}]}

        class ReservationGate:
            books_fills = True

        broker = FakeBroker()
        broker.client = type("Client", (), {
            "reservation_gate": ReservationGate(),
        })()
        intent = DeskIntent(SPY, "BUY", 0.1, "t", quantity=2)
        session = make_session(FakeDesk([intent]), broker,
                               FilledExecutor(), audit, switch,
                               local_book=book)
        assert session.evaluate_once()["status"] == "ok"
        assert book.positions() == {}
        assert book.cash() == 1_000.0

    def test_option_fill_books_contracts_and_multiplied_cash(self, rails,
                                                             tmp_path):
        """Options: quantity in native CONTRACTS under the canonical
        str(Asset) key; cash moves fill_price x100."""
        audit, switch = rails
        book = self._book(tmp_path)
        call = Asset("SPY", AssetType.CALL, strike_price=440.0,
                     expiration_date="2026-07-17")

        class OptionExecutor(FakeExecutor):
            def execute(self, side, instrument, quantity, **kwargs):
                return {"status": "filled", "avg_fill": 1.5,
                        "shortfall_per_unit": 0.0,
                        "fills": [{"qty": 2, "price": 1.5, "ts": "t"}]}

        intent = DeskIntent(call, "BUY", 0.1, "t", quantity=2)
        session = make_session(FakeDesk([intent]), FakeBroker(),
                               OptionExecutor(), audit, switch,
                               local_book=book)
        assert session.evaluate_once()["status"] == "ok"
        assert book.positions() == {str(call): 2.0}
        assert book.cash() == pytest.approx(-2 * 1.5 * 100)

    def test_book_failure_never_halts_the_session(self, rails, tmp_path):
        """A book that raises must not stop trading — the drift surfaces
        at reconciliation instead."""
        audit, switch = rails

        class BrokenBook:
            def record_fill(self, *a, **k):
                raise RuntimeError("disk full")

        class FilledExecutor(FakeExecutor):
            def execute(self, side, instrument, quantity, **kwargs):
                return {"status": "filled", "avg_fill": 50.0,
                        "shortfall_per_unit": 0.0,
                        "fills": [{"qty": 1, "price": 50.0, "ts": "t"}]}

        intent = DeskIntent(SPY, "BUY", 0.1, "t", quantity=1)
        session = make_session(FakeDesk([intent]), FakeBroker(),
                               FilledExecutor(), audit, switch,
                               local_book=BrokenBook())
        assert session.evaluate_once()["status"] == "ok"

    def test_no_arg_reconciliation_reads_the_book(self, rails, tmp_path):
        audit, switch = rails
        book = self._book(tmp_path)
        book.set_cash(100_000.0)  # matches FakeBroker: no positions, 100k
        session = make_session(FakeDesk([]), FakeBroker(), None, audit,
                               switch, local_book=book)
        result = session.run_reconciliation()
        assert result["ok"] is True
        assert switch.engaged() is False
        # Now drift the book: reconcile fails and engages the switch.
        book.record_fill("SPY", 5, 10.0)
        result = session.run_reconciliation()
        assert result["ok"] is False
        assert switch.engaged() is True

    def test_no_arg_reconciliation_without_book_raises(self, rails):
        audit, switch = rails
        session = make_session(FakeDesk([]), FakeBroker(), None, audit,
                               switch)
        with pytest.raises(ValueError, match="local_book"):
            session.run_reconciliation()

    def test_explicit_args_still_win_over_the_book(self, rails, tmp_path):
        """Existing callers (GUI routes) pass positions/cash explicitly;
        the book must not shadow them."""
        audit, switch = rails
        book = self._book(tmp_path)
        book.record_fill("SPY", 99, 1.0)       # book is WRONG on purpose
        session = make_session(FakeDesk([]), FakeBroker(), None, audit,
                               switch, local_book=book)
        result = session.run_reconciliation({}, 100_000.0)
        assert result["ok"] is True            # explicit args used, not book
