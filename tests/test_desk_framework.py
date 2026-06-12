"""Tests for desks.base: Desk ABC contract, apply_risk wiring, notes.

Offline and deterministic — no engine, no data fetches; portfolios and
positions are hand-built so every risk number is checked exactly.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core.models import Asset, AssetType, Position
from desks.base import Desk, DeskIntent, TraderNote
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


class NullDesk(Desk):
    """Minimal concrete desk: generate_intents returns a scripted list."""

    def __init__(self, **kwargs):
        super().__init__(key='null', name='Null Desk',
                         description='test-only desk', accent='#123456',
                         **kwargs)
        self.scripted: list = []

    def generate_intents(self, all_data, date, portfolio):
        return list(self.scripted)


class TestDeskABCContract:
    def test_desk_is_abstract(self):
        with pytest.raises(TypeError):
            Desk(key='x', name='X', description='d', accent='#000000')

    def test_concrete_desk_defaults(self):
        desk = NullDesk()
        assert desk.capital_allocation == 1.0
        assert isinstance(desk.risk_manager, RiskManager)
        assert desk.notes == []
        assert desk.walk_forward_fits == []

    def test_get_status_shape(self):
        desk = NullDesk(capital_allocation=0.4)
        status = desk.get_status()
        assert status == {
            'key': 'null',
            'name': 'Null Desk',
            'capital_allocation': 0.4,
            'notes_count': 0,
            'last_note': None,
        }

        desk.set_clock(pd.Timestamp('2023-06-01'))
        desk.note('info', 'hello')
        status = desk.get_status()
        assert status['notes_count'] == 1
        assert status['last_note']['message'] == 'hello'

    def test_intent_validation(self):
        # Contract C4: BUY/SELL/SHORT/COVER are all valid actions.
        for action in ('BUY', 'SELL', 'SHORT', 'COVER'):
            intent = DeskIntent(asset=stock('A'), action=action,
                                size_fraction=0.1, reason='x')
            assert intent.action == action
        with pytest.raises(ValueError, match='action'):
            DeskIntent(asset=stock('A'), action='HEDGE', size_fraction=0.1,
                       reason='x')
        for bad_fraction in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match='size_fraction'):
                DeskIntent(asset=stock('A'), action='BUY',
                           size_fraction=bad_fraction, reason='x')

    def test_invalid_note_category_raises(self):
        desk = NullDesk()
        with pytest.raises(ValueError, match='category'):
            desk.note('gossip', 'not a real category')


class TestApplyRiskPositionSize:
    def test_oversized_buy_is_blocked_and_noted_with_numbers(self):
        desk = NullDesk()  # capital_allocation 1.0, max position 10%
        desk.set_clock(pd.Timestamp('2023-06-01'))
        portfolio = PortfolioManager(initial_capital=100000.0)

        oversized = DeskIntent(asset=stock('BIG'), action='BUY',
                               size_fraction=0.5, reason='test')
        approved = desk.apply_risk([oversized], portfolio, {},
                                   pd.Timestamp('2023-06-01'))

        assert approved == []
        risk_notes = [n for n in desk.notes if n.category == 'risk']
        assert len(risk_notes) == 1
        note = risk_notes[0]
        assert 'BIG' in note.message
        # trade value = 100000 * 1.0 * 0.5 = 50000 (50% > 10% limit)
        assert note.data['trade_value'] == pytest.approx(50000.0)
        assert note.data['portfolio_value'] == pytest.approx(100000.0)
        assert note.data['size_fraction'] == pytest.approx(0.5)
        assert note.data['max_position_size'] == pytest.approx(0.1)

    def test_buy_within_limit_passes(self):
        desk = NullDesk(capital_allocation=0.5)
        portfolio = PortfolioManager(initial_capital=100000.0)

        # trade value = 100000 * 0.5 * 0.1 = 5000 -> 5% of portfolio, OK.
        intent = DeskIntent(asset=stock('OK'), action='BUY',
                            size_fraction=0.1, reason='test')
        approved = desk.apply_risk([intent], portfolio, {},
                                   pd.Timestamp('2023-06-01'))
        assert approved == [intent]
        assert [n for n in desk.notes if n.category == 'risk'] == []

    def test_capital_allocation_scales_the_checked_trade_value(self):
        # size_fraction 0.5 of a desk running 10% of capital = 5% trade: OK.
        desk = NullDesk(capital_allocation=0.1)
        portfolio = PortfolioManager(initial_capital=100000.0)
        intent = DeskIntent(asset=stock('OK'), action='BUY',
                            size_fraction=0.5, reason='test')
        assert desk.apply_risk([intent], portfolio, {},
                               pd.Timestamp('2023-06-01')) == [intent]


class TestApplyRiskStopLoss:
    def test_stop_breach_generates_full_size_sell_with_correct_numbers(self):
        desk = NullDesk()
        desk.set_clock(pd.Timestamp('2023-06-02'))
        portfolio = PortfolioManager(initial_capital=100000.0)
        asset = stock('DOWN')
        # Default stop is 2% below entry: stop = 100 * 0.98 = 98.0.
        portfolio.add_position(Position(
            asset=asset, quantity=10, avg_entry_price=100.0,
            current_price=97.0, timestamp=pd.Timestamp('2023-05-01')))

        approved = desk.apply_risk([], portfolio, {},
                                   pd.Timestamp('2023-06-02'))

        assert len(approved) == 1
        sell = approved[0]
        assert sell.asset == asset
        assert sell.action == 'SELL'
        assert sell.size_fraction == 1.0

        risk_notes = [n for n in desk.notes if n.category == 'risk']
        assert len(risk_notes) == 1
        data = risk_notes[0].data
        assert data['entry_price'] == pytest.approx(100.0)
        assert data['current_price'] == pytest.approx(97.0)
        assert data['stop_price'] == pytest.approx(98.0)
        assert data['stop_loss_pct'] == pytest.approx(0.02)

    def test_position_above_stop_generates_no_sell(self):
        desk = NullDesk()
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.add_position(Position(
            asset=stock('UP'), quantity=10, avg_entry_price=100.0,
            current_price=99.0, timestamp=pd.Timestamp('2023-05-01')))
        assert desk.apply_risk([], portfolio, {},
                               pd.Timestamp('2023-06-02')) == []

    def test_no_duplicate_sell_when_desk_already_selling(self):
        desk = NullDesk()
        portfolio = PortfolioManager(initial_capital=100000.0)
        asset = stock('DOWN')
        portfolio.add_position(Position(
            asset=asset, quantity=10, avg_entry_price=100.0,
            current_price=97.0, timestamp=pd.Timestamp('2023-05-01')))

        desk_sell = DeskIntent(asset=asset, action='SELL', size_fraction=1.0,
                               reason='desk exit')
        approved = desk.apply_risk([desk_sell], portfolio, {},
                                   pd.Timestamp('2023-06-02'))
        sells = [i for i in approved if i.action == 'SELL']
        assert sells == [desk_sell]


class TestApplyRiskDailyLossCircuit:
    def _drawn_down_portfolio(self) -> PortfolioManager:
        """Portfolio that lost 6000 (6%+) since the previous snapshot."""
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.cash = 94000.0
        portfolio.portfolio_history.append({
            'timestamp': pd.Timestamp('2023-05-31'),
            'portfolio_value': 100000.0,
        })
        return portfolio

    def test_breach_blocks_all_buys_and_notes_once(self):
        desk = NullDesk()
        desk.set_clock(pd.Timestamp('2023-06-01'))
        portfolio = self._drawn_down_portfolio()

        buys = [DeskIntent(asset=stock(s), action='BUY', size_fraction=0.05,
                           reason='test') for s in ('AAA', 'BBB')]
        approved = desk.apply_risk(buys, portfolio, {},
                                   pd.Timestamp('2023-06-01'))

        assert approved == []
        circuit_notes = [n for n in desk.notes if n.category == 'risk']
        assert len(circuit_notes) == 1  # noted ONCE, not per blocked BUY
        assert 'circuit' in circuit_notes[0].message.lower()
        assert circuit_notes[0].data['daily_pnl'] == pytest.approx(-6000.0)

    def test_sells_still_pass_when_circuit_is_tripped(self):
        desk = NullDesk()
        portfolio = self._drawn_down_portfolio()
        asset = stock('HELD')
        portfolio.add_position(Position(
            asset=asset, quantity=10, avg_entry_price=100.0,
            current_price=100.0, timestamp=pd.Timestamp('2023-05-01')))
        # Keep the drawdown scenario: cash drop dominates position value.
        portfolio.cash = 93000.0

        sell = DeskIntent(asset=asset, action='SELL', size_fraction=1.0,
                          reason='exit')
        buy = DeskIntent(asset=stock('NEW'), action='BUY', size_fraction=0.05,
                         reason='entry')
        approved = desk.apply_risk([sell, buy], portfolio, {},
                                   pd.Timestamp('2023-06-01'))
        assert approved == [sell]

    def test_small_loss_does_not_trip_circuit(self):
        desk = NullDesk()
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.cash = 99000.0  # -1% day: under the 5% limit
        portfolio.portfolio_history.append({
            'timestamp': pd.Timestamp('2023-05-31'),
            'portfolio_value': 100000.0,
        })
        buy = DeskIntent(asset=stock('NEW'), action='BUY', size_fraction=0.05,
                         reason='entry')
        assert desk.apply_risk([buy], portfolio, {},
                               pd.Timestamp('2023-06-01')) == [buy]


class TestTraderNotes:
    def test_notes_carry_simulation_dates_not_wall_clock(self):
        desk = NullDesk()
        sim_date = pd.Timestamp('2019-03-15')  # far from any wall clock
        desk.set_clock(sim_date)
        note = desk.note('info', 'simulated')

        assert note.timestamp == sim_date
        assert note.to_dict()['timestamp'].startswith('2019-03-15')
        assert abs((datetime.now() - sim_date).days) > 365

    def test_clock_advances_with_simulation(self):
        desk = NullDesk()
        desk.set_clock(pd.Timestamp('2023-01-02'))
        first = desk.note('info', 'day one')
        desk.set_clock(pd.Timestamp('2023-01-03'))
        second = desk.note('info', 'day two')
        assert first.timestamp < second.timestamp

    def test_note_data_is_json_safe_with_numpy_inputs(self):
        desk = NullDesk()
        desk.set_clock(pd.Timestamp('2023-01-02'))
        note = desk.note(
            'signal', 'numpy soup',
            np_float=np.float64(1.5),
            np_int=np.int64(7),
            np_bool=np.bool_(True),
            np_array=np.array([1.0, 2.0]),
            timestamp=pd.Timestamp('2023-01-01'),
            nested={'inner': np.float32(0.25)},
        )

        payload = note.to_dict()
        json.dumps(payload)  # must not raise

        data = payload['data']
        assert data['np_float'] == 1.5 and isinstance(data['np_float'], float)
        assert data['np_int'] == 7 and isinstance(data['np_int'], int)
        assert data['np_bool'] is True
        assert data['np_array'] == [1.0, 2.0]
        assert data['timestamp'] == '2023-01-01T00:00:00'
        assert data['nested']['inner'] == pytest.approx(0.25)

    def test_to_dict_shape_matches_contract(self):
        note = TraderNote(timestamp=datetime(2023, 1, 2, 0, 0),
                          desk='Null Desk', category='allocation',
                          message='m', data={'k': 1})
        assert note.to_dict() == {
            'timestamp': '2023-01-02T00:00:00',
            'desk': 'Null Desk',
            'category': 'allocation',
            'message': 'm',
            'data': {'k': 1},
        }
