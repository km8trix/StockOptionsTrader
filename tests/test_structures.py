"""Tests for desks.structures — builders, max-loss math, sizing, exits.

The max-loss formula is hand-verified (including the asymmetric-wing
case where the wider side governs) and the exit triggers are pinned at
their EXACT boundaries (inclusive semantics). Offline, deterministic.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.models import Asset, AssetType
from desks.structures import (StructureTracker, build_call_credit_spread,
                              build_iron_condor, build_put_credit_spread,
                              max_loss_per_contract, size_contracts,
                              strike_increment, wing_width)


def legs_by(specs, action, right):
    return [leg for leg in specs
            if leg['action'] == action and leg['right'] == right]


class TestBuilders:
    def test_condor_strikes_spot_100_increment_5(self):
        # spot=100 -> $5 strikes; sigma=0.2, dte=35:
        # std move = 100 * 0.2 * sqrt(35/365) = 6.1930...
        specs = build_iron_condor(100.0, 0.2, 35)
        std = 100.0 * 0.2 * math.sqrt(35 / 365.0)
        assert round((100 - std) / 5) * 5 == 95.0
        assert [(s['action'], s['right'], s['strike']) for s in specs] == [
            ('SHORT', 'put', 95.0), ('BUY', 'put', 90.0),
            ('SHORT', 'call', 105.0), ('BUY', 'call', 110.0)]

    def test_condor_strikes_spot_50_increment_1(self):
        specs = build_iron_condor(50.0, 0.2, 35)
        # std move = 3.0965: short put round(46.90)=47, long round(45.36)=45
        assert [(s['action'], s['right'], s['strike']) for s in specs] == [
            ('SHORT', 'put', 47.0), ('BUY', 'put', 45.0),
            ('SHORT', 'call', 53.0), ('BUY', 'call', 55.0)]

    def test_increment_boundary(self):
        assert strike_increment(99.99) == 1.0
        assert strike_increment(100.0) == 5.0

    def test_verticals_match_the_condor_sides(self):
        condor = build_iron_condor(100.0, 0.2, 35)
        assert build_put_credit_spread(100.0, 0.2, 35) == condor[:2]
        assert build_call_credit_spread(100.0, 0.2, 35) == condor[2:]

    def test_collapsed_wing_is_pushed_one_increment_out(self):
        # Tiny sigma: both strikes round to the same value; the wing
        # must be pushed out (a zero-width wing has no defined risk).
        specs = build_put_credit_spread(100.0, 0.01, 30)
        short, long = specs[0]['strike'], specs[1]['strike']
        assert short - long == 5.0
        specs = build_call_credit_spread(100.0, 0.01, 30)
        assert specs[1]['strike'] - specs[0]['strike'] == 5.0


class TestMaxLossMath:
    def test_symmetric_condor_hand_check(self):
        specs = build_iron_condor(100.0, 0.2, 35)  # both wings $5 wide
        assert wing_width(specs) == 5.0
        # max_loss = width*100 - credit: $5 wings, $180 credit -> $320.
        assert max_loss_per_contract(specs, 180.0) == pytest.approx(320.0)

    def test_asymmetric_wings_wider_side_governs(self):
        specs = [
            {'action': 'SHORT', 'right': 'put', 'strike': 95.0},
            {'action': 'BUY', 'right': 'put', 'strike': 90.0},   # $5 side
            {'action': 'SHORT', 'right': 'call', 'strike': 105.0},
            {'action': 'BUY', 'right': 'call', 'strike': 115.0},  # $10 side
        ]
        assert wing_width(specs) == 10.0
        # Worst case is the underlying blowing through the CALL side:
        # lose $10 x 100, keep the $250 credit -> $750, NOT $250.
        assert max_loss_per_contract(specs, 250.0) == pytest.approx(750.0)

    def test_single_vertical(self):
        specs = build_put_credit_spread(100.0, 0.2, 35)  # 95/90
        assert max_loss_per_contract(specs, 120.0) == pytest.approx(380.0)


class TestSizing:
    def test_floor_of_budget_over_max_loss(self):
        # risk_budget = 0.02 * desk_capital * book_weight, the rule.
        assert size_contracts(0.02 * 100_000 * 0.5, 320.0) == 3  # 1000/320
        assert size_contracts(960.0, 320.0) == 3  # exact multiple
        assert size_contracts(959.99, 320.0) == 2

    def test_zero_contract_skip(self):
        assert size_contracts(300.0, 320.0) == 0
        assert size_contracts(0.0, 320.0) == 0
        assert size_contracts(1000.0, 0.0) == 0  # degenerate guard
        assert size_contracts(1000.0, -5.0) == 0


# ----------------------------------------------------------------------
# Tracker lifecycle + exit boundaries
# ----------------------------------------------------------------------
EXPIRY = '2023-06-15'


def condor_legs(symbol='XYZ', expiry=EXPIRY):
    def opt(right, strike):
        return Asset(symbol, AssetType.PUT if right == 'put'
                     else AssetType.CALL, strike_price=strike,
                     expiration_date=expiry)
    return [
        {'asset': opt('put', 95.0), 'action': 'SHORT'},
        {'asset': opt('put', 90.0), 'action': 'BUY'},
        {'asset': opt('call', 105.0), 'action': 'SHORT'},
        {'asset': opt('call', 110.0), 'action': 'BUY'},
    ]


def marks(p_short_put, p_long_put, p_short_call, p_long_call,
          legs=None):
    legs = legs if legs is not None else condor_legs()
    return {legs[0]['asset']: p_short_put, legs[1]['asset']: p_long_put,
            legs[2]['asset']: p_short_call, legs[3]['asset']: p_long_call}


def open_condor(tracker, credit=200.0, contracts=1, opened='2023-04-01'):
    return tracker.open_structure('iron_condor', 'XYZ', condor_legs(),
                                  credit=credit, max_loss=300.0,
                                  contracts=contracts, date=opened)


class TestTracker:
    def test_ids_are_sequential_and_deterministic(self):
        tracker = StructureTracker()
        assert open_condor(tracker) == 'JS-0001'
        assert open_condor(tracker) == 'JS-0002'

    def test_cost_to_close_arithmetic(self):
        tracker = StructureTracker()
        sid = open_condor(tracker, contracts=2)
        # Per contract: buy back shorts (0.80 + 0.70), sell wings
        # (0.20 + 0.10) -> 1.20 net; x100 x2 contracts = $240.
        cost = tracker.cost_to_close(sid, marks(0.80, 0.20, 0.70, 0.10))
        assert cost == pytest.approx(240.0)
        assert tracker.get(sid)['last_cost_to_close'] == pytest.approx(240.0)

    def test_cost_to_close_none_when_a_leg_lacks_a_mark(self):
        tracker = StructureTracker()
        sid = open_condor(tracker)
        leg_prices = marks(0.8, 0.2, 0.7, 0.1)
        del leg_prices[condor_legs()[3]['asset']]
        assert tracker.cost_to_close(sid, leg_prices) is None

    # --- exit boundaries (inclusive semantics, exact values) ----------
    def far_date(self):
        return pd.Timestamp(EXPIRY) - pd.Timedelta(days=30)  # DTE 30

    def test_profit_target_at_exactly_half_the_credit(self):
        tracker = StructureTracker()
        sid = open_condor(tracker, credit=200.0)
        # cost-to-close exactly $100 == 50% of the $200 credit: triggers
        # (binary-exact marks: 0.625 - 0.125 + 0.625 - 0.125 == 1.00).
        assert tracker.check_exits(sid, marks(0.625, 0.125, 0.625, 0.125),
                                   self.far_date()) == 'profit_target'
        # A hair above 50%: no trigger.
        assert tracker.check_exits(sid, marks(0.625, 0.125, 0.6875, 0.125),
                                   self.far_date()) is None

    def test_stop_loss_at_exactly_twice_the_credit(self):
        tracker = StructureTracker()
        sid = open_condor(tracker, credit=200.0)
        # cost-to-close exactly $400 == 2x the credit: triggers (the
        # marks are binary-exact so the sum is exactly 4.00/contract).
        assert tracker.check_exits(sid, marks(2.5, 0.25, 2.0, 0.25),
                                   self.far_date()) == 'stop_loss'
        # A hair below 2x: no trigger.
        assert tracker.check_exits(sid, marks(2.5, 0.25, 2.0, 0.2625),
                                   self.far_date()) is None

    def test_time_exit_at_exactly_dte_21(self):
        tracker = StructureTracker()
        sid = open_condor(tracker, credit=200.0)
        neutral = marks(1.0, 0.3, 1.0, 0.3)  # $140: between 50% and 2x
        on_boundary = pd.Timestamp(EXPIRY) - pd.Timedelta(days=21)
        assert tracker.check_exits(sid, neutral,
                                   on_boundary) == 'time_exit'
        day_before = pd.Timestamp(EXPIRY) - pd.Timedelta(days=22)
        assert tracker.check_exits(sid, neutral, day_before) is None

    def test_closing_intents_and_serialization(self):
        tracker = StructureTracker()
        sid = open_condor(tracker, credit=200.0, contracts=3)
        intents = tracker.build_closing_intents(sid, 'profit_target')
        # SHORT legs COVER, BUY legs SELL, full-size closes.
        assert [(i.action, i.asset.strike_price) for i in intents] == [
            ('COVER', 95.0), ('SELL', 90.0), ('COVER', 105.0),
            ('SELL', 110.0)]
        assert all(i.size_fraction == 1.0 and i.quantity is None
                   for i in intents)
        with pytest.raises(ValueError, match='not open'):
            tracker.build_closing_intents(sid, 'stop_loss')  # idempotence

        # 'closing' serializes as C11 'open' until the legs are flat.
        report = tracker.to_report()[0]
        assert report['status'] == 'open'
        assert report['close_reason'] == 'profit_target'

        tracker.finalize(sid, '2023-05-20', pnl=87.5)
        report = tracker.to_report()[0]
        assert report['id'] == sid
        assert report['status'] == 'closed'
        assert report['closed'] == '2023-05-20'
        assert report['pnl'] == 87.5
        assert report['contracts'] == 3
        assert set(report) == {'id', 'type', 'underlying', 'opened',
                               'closed', 'credit', 'max_loss', 'contracts',
                               'status', 'pnl', 'close_reason', 'legs'}
        for leg in report['legs']:
            assert set(leg) == {'instrument', 'action', 'strike', 'expiry',
                                'right'}
            assert leg['expiry'] == EXPIRY
            assert leg['right'] in ('call', 'put')
        # Leg ownership is released at finalize.
        assert not tracker.leg_owner

    def test_has_open_on_and_remove(self):
        tracker = StructureTracker()
        sid = open_condor(tracker)
        assert tracker.has_open_on('XYZ')
        assert not tracker.has_open_on('ABC')
        tracker.remove(sid)
        assert not tracker.has_open_on('XYZ')
        assert tracker.to_report() == []
