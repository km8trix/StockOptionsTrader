"""PatientExecutor (E4): scripted quote/fill sequences over virtual time —
immediate mid fill, the exact 25%-of-remaining repricing path, the
far-touch cap under a moving quote, edge-decay, timeout, partial fills,
kill-switch mid-work, the cancel-confirm race (a fill landing between
the cancel request and confirmation must be banked before the
replacement is sized), and the shortfall sign convention on both
sides."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from core.models import Asset, AssetType, OrderType
from execution.patient_executor import (CANCEL_CONFIRM_ATTEMPTS,
                                        CANCEL_POLL_S, PatientExecutor,
                                        round_tick)
from utils.kill_switch import KillSwitch

SPY = Asset("SPY", AssetType.STOCK)


def S(filled=0.0, avg=None, status="OPEN"):
    return {"status": status, "filled_quantity": filled,
            "avg_fill_price": avg}


class VirtualTime:
    """Frozen clock advanced only by the executor's sleep calls."""

    def __init__(self):
        self.now = datetime(2026, 6, 12, 9, 30, tzinfo=timezone.utc)
        self.sleeps = []
        self.on_sleep = None  # optional hook(sleep_index)

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)
        if self.on_sleep is not None:
            self.on_sleep(len(self.sleeps))


class FakeBroker:
    """Orders get ids ORD-1, ORD-2, ...; order_status for placement i is
    scripted by scripts[i] (a queue; the last entry is sticky)."""

    def __init__(self, scripts=None):
        self.scripts = scripts or []
        self.placements = []
        self.cancels = []
        self._counter = 0

    def _new_id(self):
        self._counter += 1
        return f"ORD-{self._counter}"

    def place_order(self, asset, order_type, quantity, limit_price):
        order_id = self._new_id()
        self.placements.append({"order_id": order_id, "asset": asset,
                                "order_type": order_type, "qty": quantity,
                                "limit": limit_price})
        return order_id

    def place_structure(self, legs, net_price, contracts, closing=False):
        order_id = self._new_id()
        self.placements.append({"order_id": order_id, "legs": legs,
                                "net": net_price, "qty": contracts,
                                "closing": closing})
        return order_id

    def cancel_order(self, order_id):
        self.cancels.append(order_id)
        return True

    def order_status(self, order_id):
        index = int(order_id.split("-")[1]) - 1
        if index >= len(self.scripts) or not self.scripts[index]:
            status = S()
        else:
            queue = self.scripts[index]
            status = queue.pop(0) if len(queue) > 1 else queue[0]
        if order_id in self.cancels and status["status"] == "OPEN":
            # Cancel is synchronous on the fake: a still-working order
            # confirms CANCELLED on the first post-cancel poll.
            status = dict(status, status="CANCELLED")
        return status


class ClientIdBroker(FakeBroker):
    """Capability-aware fake that records explicit client order ids."""

    def place_order_with_client_id(
            self, asset, order_type, quantity, limit_price,
            client_order_id):
        order_id = self.place_order(
            asset, order_type, quantity, limit_price)
        self.placements[-1]["client_order_id"] = client_order_id
        return order_id

    def place_structure_with_client_id(
            self, legs, net_price, contracts, client_order_id,
            closing=False):
        order_id = self.place_structure(
            legs, net_price, contracts, closing=closing)
        self.placements[-1]["client_order_id"] = client_order_id
        return order_id


def make_executor(broker, quotes, vt, kill_switch=None):
    """quotes: a dict (constant) or a list consumed per quote_fn call
    (last sticky)."""
    if isinstance(quotes, dict):
        quote_fn = lambda _instrument: quotes  # noqa: E731
    else:
        queue = list(quotes)
        def quote_fn(_instrument):
            return queue.pop(0) if len(queue) > 1 else queue[0]
    return PatientExecutor(broker, quote_fn, clock=vt.clock,
                           sleep_fn=vt.sleep, kill_switch=kill_switch)


class TestRoundTick:
    def test_half_up_with_float_dust(self):
        assert round_tick(1.125) == 1.13  # half rounds UP
        assert round_tick(1.1249999999999998) == 1.13  # dust-tolerant
        assert round_tick(1.1625) == 1.16  # 116.25 floors to 116
        assert round_tick((1.00 + 1.20) / 2) == 1.10


class TestImmediateFill:
    def test_mid_fill_at_t0(self):
        vt = VirtualTime()
        broker = FakeBroker(scripts=[[S(10, 1.10, "EXECUTED")]])
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10)
        assert report["status"] == "filled"
        assert report["arrival_mid"] == 1.10
        assert report["avg_fill"] == 1.10
        assert report["shortfall_per_unit"] == pytest.approx(0.0)
        assert report["fills"] == [{"qty": 10.0, "price": 1.10,
                                    "ts": vt.now.isoformat()}]
        assert [s["limit"] for s in report["steps"]] == [1.10]
        assert vt.sleeps == []  # filled before any waiting
        assert broker.placements[0]["limit"] == 1.10
        assert broker.placements[0]["order_type"] == OrderType.BUY


class TestExecutionIdentity:
    def test_same_execution_path_derives_stable_client_order_ids(self):
        def run_once():
            vt = VirtualTime()
            broker = ClientIdBroker(
                scripts=[[S(5, 1.10, "EXECUTED")]])
            executor = make_executor(
                broker, {"bid": 1.00, "ask": 1.20}, vt)
            report = executor.execute(
                "BUY", SPY, 5, execution_id="desk-intent-42")
            assert report["status"] == "filled"
            return [p["client_order_id"] for p in broker.placements]

        first = run_once()
        second = run_once()
        assert first == second
        assert len(first[0]) == 20
        assert first[0].isalnum()

    def test_replacement_children_receive_distinct_ids(self):
        vt = VirtualTime()
        broker = ClientIdBroker()
        executor = make_executor(
            broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute(
            "BUY", SPY, 5, max_minutes=1, step_interval_s=30,
            execution_id="desk-intent-42")
        assert report["status"] == "timeout"
        ids = [p["client_order_id"] for p in broker.placements]
        assert len(ids) == 2
        assert len(set(ids)) == 2
        assert all(len(value) <= 20 and value.isalnum() for value in ids)

    def test_legacy_broker_keeps_original_place_order_contract(self):
        vt = VirtualTime()
        broker = FakeBroker(scripts=[[S(5, 1.10, "EXECUTED")]])
        executor = make_executor(
            broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute(
            "BUY", SPY, 5, execution_id="desk-intent-42")
        assert report["status"] == "filled"
        assert "client_order_id" not in broker.placements[0]

    def test_package_replacement_is_atomic_and_has_stable_child_ids(self):
        legs = [
            {"asset": Asset("IWM", AssetType.PUT, 195.0, "2026-07-17"),
             "action": "SHORT"},
            {"asset": Asset("IWM", AssetType.PUT, 190.0, "2026-07-17"),
             "action": "BUY"},
        ]

        def run_once():
            vt = VirtualTime()
            broker = ClientIdBroker(scripts=[[
                S(), S(1, -1.10, "OPEN"), S(1, -1.10, "CANCELLED")],
                [S(2, -1.08, "EXECUTED")],
            ])
            executor = make_executor(
                broker, {"bid": 1.00, "ask": 1.20}, vt)
            report = executor.execute(
                "SELL", legs, 3, max_minutes=1, step_interval_s=30,
                execution_id="spread-intent-42")
            return report, broker

        first_report, first_broker = run_once()
        second_report, second_broker = run_once()
        assert first_report["status"] == "filled"
        assert [fill["qty"] for fill in first_report["fills"]] == [1.0, 2.0]
        assert [fill["price"] for fill in first_report["fills"]] == [1.10, 1.08]
        assert first_report["avg_fill"] == pytest.approx(1.0866666667)
        assert [placement["qty"] for placement in first_broker.placements] == [
            3, 2]
        assert all(placement["legs"] == legs
                   for placement in first_broker.placements)
        first_ids = [placement["client_order_id"]
                     for placement in first_broker.placements]
        second_ids = [placement["client_order_id"]
                      for placement in second_broker.placements]
        assert first_ids == second_ids
        assert len(set(first_ids)) == 2
        assert all(len(value) == 20 and value.isalnum()
                   for value in first_ids)
        assert second_report["status"] == "filled"

    def test_package_close_lifecycle_survives_patient_placement(self):
        vt = VirtualTime()
        broker = ClientIdBroker(scripts=[[S(2, 0.60, "EXECUTED")]])
        legs = [
            {"asset": Asset("IWM", AssetType.PUT, 195.0, "2026-07-17"),
             "action": "SHORT"},
            {"asset": Asset("IWM", AssetType.PUT, 190.0, "2026-07-17"),
             "action": "BUY"},
        ]
        report = make_executor(
            broker, {"bid": 0.50, "ask": 0.70}, vt).execute(
                "BUY", legs, 2, execution_id="spread-close-42",
                closing=True)
        assert report["status"] == "filled"
        assert broker.placements[0]["closing"] is True
        assert broker.placements[0]["net"] == -0.60
        assert broker.placements[0]["qty"] == 2

    def test_closing_flag_is_rejected_for_single_instrument(self):
        vt = VirtualTime()
        executor = make_executor(
            FakeBroker(), {"bid": 1.00, "ask": 1.20}, vt)
        with pytest.raises(ValueError, match="leg packages"):
            executor.execute("BUY", SPY, 1, closing=True)


class TestRepricingPath:
    def test_buy_25pct_of_remaining_over_5_steps_then_timeout(self):
        """Hand-computed limits, bid 1.00 / ask 1.20 constant:
        mid 1.10 -> 1.13 -> 1.15 -> 1.16 -> 1.17 -> 1.18; the deadline
        (3 min, 30s steps) then cancels: 'timeout', NO market fallback."""
        vt = VirtualTime()
        broker = FakeBroker()  # never fills
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=3,
                                  step_interval_s=30)
        assert report["status"] == "timeout"
        assert [s["limit"] for s in report["steps"]] == [
            1.10, 1.13, 1.15, 1.16, 1.17, 1.18]
        assert report["fills"] == []
        assert report["avg_fill"] is None
        assert report["shortfall_per_unit"] is None
        # 6 placements (initial + 5 reprices), all cancelled in the end:
        # 5 cancel-and-replace + 1 final deadline cancel.
        assert len(broker.placements) == 6
        assert len(broker.cancels) == 6
        # Every placement worked the full remaining quantity.
        assert all(p["qty"] == 10 for p in broker.placements)

    def test_sell_reprices_down_toward_bid(self):
        vt = VirtualTime()
        broker = FakeBroker()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("SELL", SPY, 5, max_minutes=2,
                                  step_interval_s=30)
        # mid 1.10 -> 1.08 (1.075 rounds half-up) -> 1.06 -> 1.05, then
        # the 2-minute deadline cancels.
        assert [s["limit"] for s in report["steps"]] == [
            1.10, 1.08, 1.06, 1.05]
        assert report["status"] == "timeout"

    def test_never_beyond_current_far_touch_when_quote_moves(self):
        """Ask collapses below our working limit: the reprice is capped
        AT the new touch (1.05), never through it."""
        vt = VirtualTime()
        broker = FakeBroker()
        quotes = [{"bid": 1.00, "ask": 1.20},   # t0: mid 1.10
                  {"bid": 0.95, "ask": 1.05}]   # step 1 re-fetch (sticky)
        executor = make_executor(broker, quotes, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=1,
                                  step_interval_s=30)
        # 25% step would target 1.09; cap = min(1.09, ask 1.05) = 1.05.
        assert [s["limit"] for s in report["steps"]] == [1.10, 1.05]
        assert broker.placements[1]["limit"] == 1.05


class TestEdgeDecay:
    def test_edge_decay_cancels_at_step_3(self):
        vt = VirtualTime()
        broker = FakeBroker()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        edge_results = iter([True, True, False])
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30,
                                  edge_check=lambda: next(edge_results))
        assert report["status"] == "edge_decayed"
        # Two reprices happened before the edge died at step 3.
        assert [s["limit"] for s in report["steps"]] == [1.10, 1.13, 1.15]
        # Final working order was cancelled, nothing replaced it.
        assert len(broker.placements) == 3
        assert len(broker.cancels) == 3


class TestPartialFills:
    def test_partial_fill_continues_then_completes(self):
        vt = VirtualTime()
        broker = FakeBroker(scripts=[
            [S(0), S(4, 1.10), S(4, 1.10)],   # order 1: 4 lots at mid
            [S(0), S(6, 1.13)],               # order 2: remainder fills
        ])
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "filled"
        assert [(f["qty"], f["price"]) for f in report["fills"]] == [
            (4.0, 1.10), (6.0, 1.13)]
        assert report["avg_fill"] == pytest.approx(1.118)
        assert report["shortfall_per_unit"] == pytest.approx(0.018)
        # The replacement order worked ONLY the remaining 6.
        assert broker.placements[1]["qty"] == 6
        assert broker.placements[1]["limit"] == 1.13

    def test_deadline_with_partials_reports_partial(self):
        vt = VirtualTime()
        broker = FakeBroker(scripts=[[S(0), S(4, 1.10), S(4, 1.10)]])
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=1,
                                  step_interval_s=30)
        assert report["status"] == "partial"
        assert sum(f["qty"] for f in report["fills"]) == 4.0

    def test_deadline_with_no_fills_reports_timeout(self):
        vt = VirtualTime()
        broker = FakeBroker()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=1,
                                  step_interval_s=30)
        assert report["status"] == "timeout"


class TestCancelRace:
    """cancel_order is only a REQUEST: a fill can land between the
    request and the broker confirming CANCELLED. The replacement must
    be sized only AFTER confirmation, with the racing fill banked."""

    def test_fill_racing_cancel_is_banked_before_replacement_is_sized(self):
        vt = VirtualTime()

        class RacingBroker(FakeBroker):
            """Cancel does not confirm immediately: the first poll after
            the cancel request shows 4 lots having filled while the
            cancel works (CANCEL_REQUESTED); the next poll confirms
            CANCELLED. The replacement then fills in full."""

            def order_status(self, order_id):
                if order_id == "ORD-2":
                    return S(6, 1.13, "EXECUTED")
                if "ORD-1" not in self.cancels:
                    return S()
                self.post_cancel_polls = getattr(
                    self, "post_cancel_polls", 0) + 1
                if self.post_cancel_polls == 1:
                    return S(4, 1.10, "CANCEL_REQUESTED")
                return S(4, 1.10, "CANCELLED")

        broker = RacingBroker()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "filled"
        # The racing 4 lots were banked BEFORE the replacement was
        # sized: it works 6, not the stale 10 (= double-execution).
        assert [p["qty"] for p in broker.placements] == [10, 6]
        assert [(f["qty"], f["price"]) for f in report["fills"]] == [
            (4.0, 1.10), (6.0, 1.13)]
        assert report["avg_fill"] == pytest.approx(1.118)
        # One step wait, then exactly one confirmation wait.
        assert vt.sleeps == [30, CANCEL_POLL_S]
        assert broker.cancels == ["ORD-1"]

    def test_full_fill_racing_cancel_places_no_replacement(self):
        vt = VirtualTime()

        class FillsOnCancel(FakeBroker):
            def order_status(self, order_id):
                if order_id in self.cancels:
                    # The whole order filled before the cancel landed.
                    return S(10, 1.10, "EXECUTED")
                return S()

        broker = FillsOnCancel()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "filled"
        assert len(broker.placements) == 1  # nothing was replaced
        assert [(f["qty"], f["price"]) for f in report["fills"]] == [
            (10.0, 1.10)]

    def test_unconfirmable_cancel_errors_instead_of_replacing(self):
        vt = VirtualTime()

        class StuckBroker(FakeBroker):
            def order_status(self, order_id):
                return S()  # forever OPEN, even after the cancel request

        broker = StuckBroker()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "error"
        assert report["error_type"] == "RuntimeError"
        assert "not confirmed" in report["error"]
        # NEVER a blind replacement over an order of unknown state.
        assert len(broker.placements) == 1
        # The confirm loop spent its full poll budget standing down.
        assert vt.sleeps == [30] + [CANCEL_POLL_S] * CANCEL_CONFIRM_ATTEMPTS


class TestKillSwitchAndStop:
    def test_kill_switch_mid_work_cancels(self, tmp_path):
        vt = VirtualTime()
        switch = KillSwitch(str(tmp_path / "ks.db"))
        broker = FakeBroker()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt,
                                 kill_switch=switch)
        # Engage during the SECOND wait: step 1 reprices normally, step 2
        # sees the switch and stands down.
        vt.on_sleep = (lambda n: switch.engage("halt", "ops")
                       if n == 2 else None)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "killed"
        assert [s["limit"] for s in report["steps"]] == [1.10, 1.13]
        assert broker.cancels  # working order was pulled

    def test_kill_switch_already_engaged_places_nothing(self, tmp_path):
        """t0 gate: an ALREADY-engaged switch means the initial order is
        never placed — the documented 'killed' report comes back instead
        of an exception (live broker) or a stray order (non-gating
        broker)."""
        vt = VirtualTime()
        switch = KillSwitch(str(tmp_path / "ks.db"))
        switch.engage("pre-engaged", actor="ops")
        broker = FakeBroker()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt,
                                 kill_switch=switch)
        report = executor.execute("BUY", SPY, 10)
        assert report["status"] == "killed"
        assert broker.placements == []  # the t0 order never went out
        assert broker.cancels == []
        assert report["fills"] == [] and report["steps"] == []
        assert report["arrival_mid"] == 1.10
        assert vt.sleeps == []  # stood down immediately

    def test_stop_event_kills_in_flight(self):
        vt = VirtualTime()
        broker = FakeBroker()
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        vt.on_sleep = (lambda n: executor.stop() if n == 1 else None)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "killed"
        assert [s["limit"] for s in report["steps"]] == [1.10]


class TestShortfallSignConvention:
    """POSITIVE shortfall = WORSE than arrival mid, on BOTH sides."""

    def test_buy_paying_up_is_positive(self):
        vt = VirtualTime()
        broker = FakeBroker(scripts=[[S(0)], [S(0), S(10, 1.13)]])
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10)
        assert report["status"] == "filled"
        assert report["shortfall_per_unit"] == pytest.approx(0.03)

    def test_sell_receiving_less_is_positive(self):
        vt = VirtualTime()
        broker = FakeBroker(scripts=[[S(0)], [S(0), S(10, 1.08)]])
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("SELL", SPY, 10)
        assert report["status"] == "filled"
        assert report["shortfall_per_unit"] == pytest.approx(0.02)

    def test_sell_price_improvement_is_negative(self):
        vt = VirtualTime()
        broker = FakeBroker(scripts=[[S(0), S(10, 1.12)]])
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("SELL", SPY, 10)
        assert report["shortfall_per_unit"] == pytest.approx(-0.02)


class TestMidWorkBrokerFailure:
    """A broker/auth exception mid-work (the midnight-ET expiry case) is
    a terminal 'error' report after a best-effort cancel — never an
    exception with a working order left in an unknown state."""

    def _expiring_broker(self, fail_on_poll=2, cancel_raises=False):
        from brokers.etrade_auth import EtradeAuthExpired

        broker = FakeBroker()
        polls = {"n": 0}

        def order_status(order_id):
            polls["n"] += 1
            if polls["n"] >= fail_on_poll:
                raise EtradeAuthExpired(
                    "E*TRADE access token hit the midnight-ET hard "
                    "expiry; re-authorization required")
            return S()

        broker.order_status = order_status
        if cancel_raises:
            def cancel_order(order_id):
                broker.cancels.append(order_id)
                raise EtradeAuthExpired("expired on cancel too")
            broker.cancel_order = cancel_order
        return broker

    def test_auth_expiry_mid_poll_cancels_and_reports_error(self):
        vt = VirtualTime()
        broker = self._expiring_broker(fail_on_poll=2)
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "error"
        assert report["error_type"] == "EtradeAuthExpired"
        assert "midnight-ET" in report["error"]
        # The working order was pulled (best-effort cancel attempted).
        assert broker.cancels == ["ORD-1"]
        # Report shape stays E4-complete for downstream consumers.
        assert report["arrival_mid"] == 1.10
        assert report["fills"] == []
        assert [s["limit"] for s in report["steps"]] == [1.10]

    def test_cancel_failing_too_still_returns_error_report(self):
        vt = VirtualTime()
        broker = self._expiring_broker(fail_on_poll=2, cancel_raises=True)
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "error"
        assert report["error_type"] == "EtradeAuthExpired"
        assert broker.cancels == ["ORD-1"]  # the attempt was made

    def test_fills_banked_before_failure_are_kept(self):
        from brokers.etrade_auth import EtradeAuthExpired

        vt = VirtualTime()
        broker = FakeBroker()
        polls = {"n": 0}

        def order_status(order_id):
            polls["n"] += 1
            if polls["n"] == 1:
                return S()
            if polls["n"] == 2:
                return S(4, 1.10)  # partial banked at step 1
            raise EtradeAuthExpired("midnight ET")

        broker.order_status = order_status
        executor = make_executor(broker, {"bid": 1.00, "ask": 1.20}, vt)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "error"
        assert [(f["qty"], f["price"]) for f in report["fills"]] == [
            (4.0, 1.10)]
        assert report["avg_fill"] == pytest.approx(1.10)

    def test_quote_failure_during_reprice_reports_error(self):
        vt = VirtualTime()
        broker = FakeBroker()
        quotes = iter([{"bid": 1.00, "ask": 1.20}])

        def quote_fn(_instrument):
            try:
                return next(quotes)
            except StopIteration:
                raise ConnectionError("quote feed died")

        executor = PatientExecutor(broker, quote_fn, clock=vt.clock,
                                   sleep_fn=vt.sleep)
        report = executor.execute("BUY", SPY, 10, max_minutes=10,
                                  step_interval_s=30)
        assert report["status"] == "error"
        assert report["error_type"] == "ConnectionError"
        assert broker.cancels == ["ORD-1"]


class TestMultiLegAndGuards:
    def test_multi_leg_package_routes_to_place_structure(self):
        vt = VirtualTime()
        broker = FakeBroker(scripts=[[S(2, 2.40, "EXECUTED")]])
        executor = make_executor(broker, {"bid": 2.30, "ask": 2.50}, vt)
        legs = [{"asset": Asset("SPY", AssetType.PUT, 440.0, "2026-07-17"),
                 "action": "SHORT"}]
        report = executor.execute("SELL", legs, 2)
        assert report["status"] == "filled"
        placement = broker.placements[0]
        assert placement["legs"] == legs
        assert placement["net"] == 2.40  # SELL works a positive credit
        assert placement["qty"] == 2

    def test_invalid_args_rejected(self):
        vt = VirtualTime()
        executor = make_executor(FakeBroker(), {"bid": 1, "ask": 2}, vt)
        with pytest.raises(ValueError):
            executor.execute("HOLD", SPY, 10)
        with pytest.raises(ValueError):
            executor.execute("BUY", SPY, 0)


class TestLifecycleAndObservability:
    def test_snapshot_cancel_await_idle_and_reuse(self):
        entered_sleep = threading.Event()
        release_sleep = threading.Event()
        broker = FakeBroker(scripts=[[], [S(5, 1.10, "EXECUTED")]])

        def blocking_sleep(_seconds):
            entered_sleep.set()
            assert release_sleep.wait(2)

        executor = PatientExecutor(
            broker, lambda _instrument: {"bid": 1.00, "ask": 1.20},
            sleep_fn=blocking_sleep)
        result = {}
        worker = threading.Thread(target=lambda: result.update(
            report=executor.execute("BUY", SPY, 5, step_interval_s=30)))
        worker.start()
        assert entered_sleep.wait(1)

        assert executor.await_idle(0) is False
        orders = executor.working_orders()
        assert len(orders) == 1
        assert orders[0] == {
            "order_id": "ORD-1",
            "instrument": "SPY",
            "side": "BUY",
            "quantity": 5,
            "filled_quantity": 0.0,
            "remaining_quantity": 5,
            "limit_price": 1.10,
            "status": "working",
            "steps": [{"ts": orders[0]["steps"][0]["ts"], "limit": 1.10}],
            "started_at": orders[0]["started_at"],
            "expires_at": orders[0]["expires_at"],
            "remaining_seconds": pytest.approx(orders[0]["remaining_seconds"]),
        }
        # Returned data is a deep snapshot, never the mutable live state.
        orders[0]["steps"].append({"ts": "tampered", "limit": 999})
        assert len(executor.working_orders()[0]["steps"]) == 1

        assert executor.cancel_current("wrong-id") is False
        assert executor.cancel_current("ORD-1") is True
        assert executor.working_orders()[0]["status"] == "cancel_requested"
        release_sleep.set()
        worker.join(2)
        assert not worker.is_alive()
        assert result["report"]["status"] == "killed"
        assert executor.await_idle(0) is True
        assert executor.working_orders() == []
        assert executor.cancel_current() is False

        # Per-order cancel is cleared at the terminal report; the worker is
        # reusable unless its sticky lifecycle stop was requested.
        second = executor.execute("BUY", SPY, 5)
        assert second["status"] == "filled"

    def test_close_interrupts_default_sleep_and_is_idempotent(self):
        placed = threading.Event()

        class NotifyingBroker(FakeBroker):
            def place_order(self, asset, order_type, quantity, limit_price):
                order_id = super().place_order(
                    asset, order_type, quantity, limit_price)
                placed.set()
                return order_id

        broker = NotifyingBroker()
        executor = PatientExecutor(
            broker, lambda _instrument: {"bid": 1.00, "ask": 1.20})
        result = {}
        worker = threading.Thread(target=lambda: result.update(
            report=executor.execute("BUY", SPY, 5, step_interval_s=60)))
        worker.start()
        assert placed.wait(1)

        started = time.monotonic()
        assert executor.close(timeout=1) is True
        assert time.monotonic() - started < 1
        worker.join(1)
        assert result["report"]["status"] == "killed"
        assert broker.cancels == ["ORD-1"]
        assert executor.close(timeout=0) is True  # idempotent

        # execute() cannot erase a prior stop/close and place a new order.
        after_close = executor.execute("BUY", SPY, 5)
        assert after_close["status"] == "killed"
        assert len(broker.placements) == 1

    def test_await_idle_rejects_negative_timeout(self):
        executor = PatientExecutor(
            FakeBroker(), lambda _instrument: {"bid": 1.00, "ask": 1.20})
        with pytest.raises(ValueError, match="non-negative"):
            executor.await_idle(-0.01)


class TestQuoteIsolation:
    def test_option_and_package_use_exact_injected_quote_subject(self):
        option = Asset("SPY", AssetType.CALL, 500.0, "2026-07-17")
        legs = [{"asset": option, "action": "LONG"}]
        seen = []

        def quote_fn(subject):
            seen.append(subject)
            return {"bid": 2.30, "ask": 2.50}

        option_broker = FakeBroker(scripts=[[S(1, 2.40, "EXECUTED")]])
        option_executor = PatientExecutor(option_broker, quote_fn)
        assert option_executor.execute("BUY", option, 1)["status"] == "filled"
        assert seen.pop() is option

        package_broker = FakeBroker(scripts=[[S(1, 2.40, "EXECUTED")]])
        package_executor = PatientExecutor(package_broker, quote_fn)
        assert package_executor.execute("SELL", legs, 1)["status"] == "filled"
        assert seen.pop() is legs
        assert package_executor.working_orders() == []

    @pytest.mark.parametrize("quote", [
        None,
        {"bid": 1.0},
        {"bid": float("nan"), "ask": 1.0},
        {"bid": 2.0, "ask": 1.0},
    ])
    def test_invalid_option_quote_never_places(self, quote):
        option = Asset("SPY", AssetType.PUT, 450.0, "2026-07-17")
        broker = FakeBroker()
        executor = PatientExecutor(broker, lambda _subject: quote)
        with pytest.raises(ValueError, match="option/package"):
            executor.execute("BUY", option, 1)
        assert broker.placements == []
        assert executor.working_orders() == []

    def test_quote_fn_is_required_and_must_be_callable(self):
        with pytest.raises(TypeError, match="quote_fn"):
            PatientExecutor(FakeBroker(), None)
