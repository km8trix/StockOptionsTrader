"""Tests for the canonical target-position contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from brokers.etrade_client import build_equity_order, build_option_order
from core.models import Asset, AssetType, Position
from portfolio.manager import PortfolioManager
from portfolio.targets import (
    ConflictingTargetError,
    DeltaPhase,
    PortfolioSnapshot,
    PositionFlipNotAllowed,
    TargetPosition,
    build_order_deltas,
    reserved_deltas_from_risk_snapshot,
)


NOW = datetime(2026, 7, 12, 14, 30, tzinfo=timezone.utc)
AAPL = Asset("AAPL", AssetType.STOCK)
MSFT = Asset("MSFT", AssetType.STOCK)
AAPL_CALL = Asset("AAPL", AssetType.CALL, 225.0, "2026-07-17")


def snapshot(
    filled=None,
    reserved=None,
    *,
    version=1,
    as_of=NOW,
):
    return PortfolioSnapshot(
        filled_quantities=filled or {},
        reserved_deltas=reserved or {},
        version=version,
        as_of=as_of,
    )


def reservation_row(request, *, units, filled=0, status="ACTIVE"):
    return {
        "reservation_id": "intent-1",
        "units": units,
        "status": status,
        "metadata": {"order_request": request},
        "orders": [
            {
                "order_id": "order-1",
                "cumulative_filled_units": filled,
                "status": "OPEN",
            }
        ],
    }


class TestTargetAndSnapshotValidation:
    def test_target_and_metadata_are_immutable_canonical_copies(self):
        metadata = {"model": {"features": ["value", "quality"]}}
        target = TargetPosition(
            Asset(" aapl ", AssetType.STOCK),
            10,
            owner=" desk-a ",
            metadata=metadata,
        )
        metadata["model"]["features"].append("momentum")

        assert target.asset.symbol == "AAPL"
        assert target.owner == "desk-a"
        assert target.metadata["model"]["features"] == ("value", "quality")
        with pytest.raises(FrozenInstanceError):
            target.target_quantity = 20
        with pytest.raises(TypeError):
            target.metadata["new"] = True

    @pytest.mark.parametrize("quantity", [True, 1.5, float("nan"), float("inf")])
    def test_target_rejects_non_integer_or_nonfinite_quantity(self, quantity):
        with pytest.raises(ValueError, match="finite integer"):
            TargetPosition(AAPL, quantity)

    def test_snapshot_requires_a_generation_and_timezone(self):
        with pytest.raises(ValueError, match="version"):
            snapshot(version=" ")
        with pytest.raises(ValueError, match="timezone-aware"):
            snapshot(as_of=datetime(2026, 7, 12))
        with pytest.raises(ValueError, match="finite integer"):
            snapshot(filled={AAPL: 0.25})


class TestDeterministicDeltas:
    def test_exact_repeat_is_stable_and_duplicate_target_is_coalesced(self):
        target = TargetPosition(
            AAPL,
            100,
            strategy="quality",
            metadata={"score": 0.9},
        )
        state = snapshot()

        first = build_order_deltas([target, target], state)
        second = build_order_deltas([target], state)

        assert first == second
        assert len(first) == 1
        assert first[0].signed_quantity == 100
        assert first[0].side == "BUY"
        assert first[0].quantity == 100
        assert len(first[0].intent_id) == 20

    def test_generation_change_changes_intent_id(self):
        target = TargetPosition(AAPL, 100)
        first = build_order_deltas([target], snapshot(version=1))[0]
        second = build_order_deltas([target], snapshot(version=2))[0]
        assert first.intent_id != second.intent_id

    def test_partial_fill_plus_remaining_reservation_needs_no_more(self):
        target = TargetPosition(AAPL, 100)
        assert build_order_deltas(
            [target], snapshot(filled={AAPL: 40}, reserved={AAPL: 60})
        ) == ()

    def test_released_reservation_emits_only_unfilled_remainder(self):
        target = TargetPosition(AAPL, 100)
        deltas = build_order_deltas([target], snapshot(filled={AAPL: 40}))

        assert len(deltas) == 1
        assert deltas[0].signed_quantity == 60
        assert deltas[0].effective_quantity == 40
        assert deltas[0].phase is DeltaPhase.REBALANCE

    def test_flip_fails_closed_by_default(self):
        with pytest.raises(PositionFlipNotAllowed, match="flips"):
            build_order_deltas(
                [TargetPosition(AAPL, -25)],
                snapshot(filled={AAPL: 100}),
            )

    def test_authorized_flip_is_split_close_first_then_open(self):
        deltas = build_order_deltas(
            [TargetPosition(AAPL, -25)],
            snapshot(filled={AAPL: 100}),
            allow_flip=True,
        )

        assert [(item.phase, item.signed_quantity) for item in deltas] == [
            (DeltaPhase.CLOSE, -100),
            (DeltaPhase.OPEN, -25),
        ]
        assert deltas[0].intent_id != deltas[1].intent_id

    def test_output_order_is_independent_of_target_input_order(self):
        aapl = TargetPosition(AAPL, 10)
        msft = TargetPosition(MSFT, 20)
        state = snapshot()
        forward = build_order_deltas([aapl, msft], state)
        reverse = build_order_deltas([msft, aapl], state)
        assert forward == reverse
        assert [delta.asset.symbol for delta in forward] == ["AAPL", "MSFT"]

    def test_conflicting_duplicate_targets_fail_closed(self):
        with pytest.raises(ConflictingTargetError, match="conflicting targets"):
            build_order_deltas(
                [TargetPosition(AAPL, 10), TargetPosition(AAPL, 20)], snapshot()
            )

    def test_option_quantities_remain_contracts_not_share_equivalents(self):
        delta = build_order_deltas(
            [TargetPosition(AAPL_CALL, 3)],
            snapshot(filled={AAPL_CALL: 1}),
        )[0]
        assert delta.signed_quantity == 2
        assert delta.quantity == 2
        assert delta.asset.multiplier == 100


class TestExistingSystemAdapters:
    def test_portfolio_manager_adapter_preserves_signed_native_quantities(self):
        portfolio = PortfolioManager(100_000.0)
        portfolio.add_position(Position(AAPL, -40, 200.0, 195.0, NOW))
        portfolio.add_position(Position(AAPL_CALL, 2, 3.0, 4.0, NOW))

        state = PortfolioSnapshot.from_portfolio_manager(
            portfolio,
            version="book-7",
            as_of=NOW,
        )

        assert state.filled_quantities[AAPL] == -40
        assert state.filled_quantities[AAPL_CALL] == 2

    def test_reservation_adapter_subtracts_cumulative_partial_fills(self):
        request = build_equity_order("AAPL", "BUY", 100, 200.0, "intent-1")
        risk_snapshot = {
            "reservations": [reservation_row(request, units=100, filled=40)]
        }

        reserved = reserved_deltas_from_risk_snapshot(risk_snapshot)
        state = snapshot(filled={AAPL: 40}, reserved=reserved)

        assert reserved[AAPL] == 60
        assert build_order_deltas([TargetPosition(AAPL, 100)], state) == ()

    def test_inactive_reservation_is_released(self):
        request = build_equity_order("AAPL", "BUY", 100, 200.0, "intent-1")
        risk_snapshot = {
            "reservations": [
                reservation_row(request, units=100, filled=40, status="RELEASED")
            ]
        }
        reserved = reserved_deltas_from_risk_snapshot(risk_snapshot)
        assert dict(reserved) == {}

    def test_option_reservation_adapter_keeps_contract_units(self):
        request = build_option_order(
            "AAPL",
            "CALL",
            225.0,
            "2026-07-17",
            "BUY_OPEN",
            3,
            2.50,
            "intent-1",
        )
        risk_snapshot = {
            "reservations": [reservation_row(request, units=3, filled=1)]
        }
        reserved = reserved_deltas_from_risk_snapshot(risk_snapshot)
        assert reserved[AAPL_CALL] == 2

    def test_active_reservation_without_order_metadata_fails_closed(self):
        risk_snapshot = {
            "reservations": [
                {
                    "status": "ACTIVE",
                    "units": 1,
                    "metadata": {},
                    "orders": [],
                }
            ]
        }
        with pytest.raises(ValueError, match="order_request"):
            reserved_deltas_from_risk_snapshot(risk_snapshot)
