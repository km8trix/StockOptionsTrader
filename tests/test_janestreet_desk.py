"""Tests for desks.janestreet.JaneStreetDesk.

Unit tests drive generate_intents day by day with a STUB regime
controller (scripted states, instant fit) so every gate is controlled:
VRP IV-rank entries, regime gating + flatten, earnings timing, RV
z-boundaries, the short_vega risk-book limit, and sizing skips. The
end-to-end test runs the real BacktestEngine on synthetic data with a
synthetic earnings calendar and checks the C11/C12/C14 shapes, the
structures <-> closed_trades reconciliation, and 3-run determinism.
Offline, seeded, deterministic. PRICING REALITY: all option prices are
synthetic Black-Scholes values (desks/options_pricing).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from data.market_data import MarketDataHandler
from desks.base import DeskIntent
from desks.janestreet import JaneStreetDesk
from desks.options_pricing import SyntheticEarningsCalendar
from desks.regime import REGIME_LABELS
from desks.walk_forward import WalkForwardController, WalkForwardModel
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

C14_BOOKS = {'regime', 'vrp', 'earnings', 'relative_value'}


def regime(state, p=0.9):
    others = [label for label in REGIME_LABELS if label != state]
    rest = (1.0 - p) / 2
    return {'state': state,
            'probs': {state: p, others[0]: rest, others[1]: rest}}


class StubRegimeModel(WalkForwardModel):
    """Fixed default regime, optionally overridden per date."""

    def __init__(self, default=None, schedule=None):
        self.default = default
        self.schedule = {pd.Timestamp(d): r
                         for d, r in (schedule or {}).items()}

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        result = self.schedule.get(pd.Timestamp(date), self.default)
        return result if result is not None else {}


def wide_risk():
    return RiskManager(max_position_size=1.0, max_daily_loss=0.99,
                       position_stop_loss=0.90)


def make_desk(default_regime='trending', schedule=None, fitted=True,
              **kwargs):
    controller = WalkForwardController(
        StubRegimeModel(default=regime(default_regime) if default_regime
                        else None, schedule=schedule),
        min_train_days=1 if fitted else 10 ** 9)
    kwargs.setdefault('risk_manager', wide_risk())
    kwargs.setdefault('iv_rank_min_obs', 30)
    kwargs.setdefault('iv_rank_window', 60)
    # vega_limit deliberately NOT defaulted here: tests run the desk's
    # real default unless they override it (Phase 8 validation found a
    # units bug the old 1e6 test default masked).
    return JaneStreetDesk(regime_controller=controller, **kwargs)


def vol_frame(n=120, shift_at=None, start='2022-01-03', base=100.0,
              pre=(0.008, 0.004), post=0.02):
    """Alternating-sign returns: magnitudes DECLINE linearly pre-shift
    (today's IV ranks low) then jump to `post` (today's IV ranks ~100).
    Deterministic by construction — no RNG at all."""
    mags = np.linspace(pre[0], pre[1], n)
    if shift_at is not None:
        mags[shift_at:] = post
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    close = base * np.cumprod(1.0 + mags * signs)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({'open': close, 'high': close * 1.001,
                         'low': close * 0.999, 'close': close,
                         'volume': np.full(n, 1e6)}, index=index)


def drive(desk, frames, dates, portfolio):
    """Run generate_intents over `dates` with engine-style slicing."""
    intents_by_date = {}
    for date in dates:
        sliced = {symbol: frame[frame.index <= date]
                  for symbol, frame in frames.items()}
        sliced = {s: f for s, f in sliced.items() if not f.empty}
        desk.set_clock(date)
        intents_by_date[date] = desk.generate_intents(sliced, date,
                                                      portfolio)
    return intents_by_date


def fill_structure(portfolio, desk, structure_id,
                   prices=(1.5, 0.5, 1.5, 0.5), timestamp=None):
    """Manually fill all legs (SHORT legs negative quantity)."""
    structure = desk.tracker.get(structure_id)
    for leg, price in zip(structure['legs'], prices):
        quantity = (structure['contracts'] if leg['action'] == 'BUY'
                    else -structure['contracts'])
        portfolio.add_position(Position(
            asset=leg['asset'], quantity=quantity, avg_entry_price=price,
            current_price=price,
            timestamp=timestamp or pd.Timestamp('2022-01-03')))


def close_structure_positions(portfolio, desk, structure_id, exit_prices,
                              exit_time):
    structure = desk.tracker.get(structure_id)
    for leg, exit_price in zip(structure['legs'], exit_prices):
        position = portfolio.get_position(leg['asset'])
        portfolio.close_position(leg['asset'], exit_price,
                                 position.quantity, position.timestamp,
                                 exit_time)
        portfolio.remove_position(leg['asset'])


def option_intents(intents):
    return [i for i in intents
            if i.asset.asset_type in (AssetType.CALL, AssetType.PUT)]


SHIFT = 80


class TestVrpEntryGating:
    def test_enters_only_after_iv_rank_exceeds_60(self):
        frames = {'AAA': vol_frame(shift_at=SHIFT)}
        desk = make_desk()
        portfolio = PortfolioManager(100_000.0)
        by_date = drive(desk, frames, frames['AAA'].index, portfolio)

        entry_dates = [d for d, intents in by_date.items()
                       if option_intents(intents)]
        assert entry_dates  # the vol shift produced entries
        assert min(entry_dates) >= frames['AAA'].index[SHIFT]

        first = option_intents(by_date[entry_dates[0]])
        assert len(first) == 4  # condor: 2 SHORT + 2 BUY legs
        assert sorted(i.action for i in first) == ['BUY', 'BUY', 'SHORT',
                                                   'SHORT']
        contracts = {i.quantity for i in first}
        assert len(contracts) == 1 and contracts.pop() >= 1
        # Expiry = entry + vrp_dte_days (35, inside the 30-45 band).
        expected_expiry = (entry_dates[0] + pd.Timedelta(days=35)
                           ).strftime('%Y-%m-%d')
        assert {i.asset.expiration_date for i in first} == {expected_expiry}

        entry_notes = [n for n in desk.notes
                       if n.data.get('book') == 'vrp'
                       and 'iv_rank' in n.data]
        assert entry_notes and entry_notes[0].data['iv_rank'] > 60.0
        assert entry_notes[0].data['structure'] == 'iron_condor'
        assert 'structure_id' in entry_notes[0].data  # C14 lifecycle

    def test_no_second_entry_while_a_structure_is_open(self):
        frames = {'AAA': vol_frame(shift_at=SHIFT)}
        desk = make_desk()
        portfolio = PortfolioManager(100_000.0)
        dates = frames['AAA'].index
        by_date = drive(desk, frames, dates, portfolio)
        entry_dates = [d for d, intents in by_date.items()
                       if option_intents(intents)
                       and any(i.action == 'SHORT'
                               for i in option_intents(intents))]
        first = entry_dates[0]
        # The very next day the underlying is occupied (the entry is
        # not even filled yet) — no new structure may be opened.
        next_day = dates[dates.get_loc(first) + 1]
        assert not option_intents(by_date[next_day])

    def test_low_iv_rank_never_enters(self):
        # Declining vol, no shift: today's IV is always the LOWEST of
        # its window -> rank far below 60 -> the vrp book stays out.
        frames = {'AAA': vol_frame(shift_at=None)}
        desk = make_desk()
        by_date = drive(desk, frames, frames['AAA'].index,
                        PortfolioManager(100_000.0))
        assert not any(option_intents(i) for i in by_date.values())
        assert desk.tracker.to_report() == []

    def test_high_vol_regime_blocks_entries(self):
        frames = {'AAA': vol_frame(shift_at=SHIFT)}
        desk = make_desk(default_regime='high_vol')
        by_date = drive(desk, frames, frames['AAA'].index,
                        PortfolioManager(100_000.0))
        assert not any(option_intents(i) for i in by_date.values())

    def test_nothing_trades_before_the_first_regime_fit(self):
        # Renaissance precedent: an unfitted regime model gates EVERY
        # book (vrp, earnings, relative_value) dark.
        frames = {'AAA': vol_frame(shift_at=SHIFT)}
        desk = make_desk(fitted=False,
                         earnings_calendar=SyntheticEarningsCalendar(
                             {'AAA': ['2022-05-02']}))
        by_date = drive(desk, frames, frames['AAA'].index,
                        PortfolioManager(100_000.0))
        assert not any(intents for intents in by_date.values())
        assert desk.walk_forward_fits == []

    def test_zero_contract_sizing_skips_with_info_note(self):
        frames = {'AAA': vol_frame(shift_at=SHIFT)}
        desk = make_desk()
        # $2,000 portfolio -> vrp risk budget = 0.02 * 2000 * 0.5 = $20:
        # no condor's max loss fits -> skip, note, no structure.
        by_date = drive(desk, frames, frames['AAA'].index,
                        PortfolioManager(2_000.0))
        assert not any(option_intents(i) for i in by_date.values())
        skip_notes = [n for n in desk.notes if n.category == 'info'
                      and 'risk budget' in n.message]
        assert skip_notes
        assert skip_notes[0].data['book'] == 'vrp'
        assert desk.tracker.to_report() == []


class TestRegimeFlatten:
    def test_high_vol_closes_all_vrp_structures(self):
        frames = {'AAA': vol_frame(shift_at=SHIFT),
                  'BBB': vol_frame(shift_at=SHIFT)}
        dates = frames['AAA'].index
        # Find the entry day organically, then script high_vol 2 days on.
        probe = make_desk(profit_target_pct=0.0, stop_loss_mult=1e9,
                          time_exit_dte=-1)
        by_date = drive(probe, frames, dates, PortfolioManager(100_000.0))
        entry_date = min(d for d, intents in by_date.items()
                         if option_intents(intents))
        loc = dates.get_loc(entry_date)
        flatten_date = dates[loc + 2]

        desk = make_desk(profit_target_pct=0.0, stop_loss_mult=1e9,
                         time_exit_dte=-1,
                         schedule={flatten_date: regime('high_vol')})
        portfolio = PortfolioManager(100_000.0)
        drive(desk, frames, dates[:loc + 1], portfolio)
        open_ids = desk.tracker.open_ids()
        assert len(open_ids) == 2  # one condor per underlying
        for structure_id in open_ids:
            fill_structure(portfolio, desk, structure_id,
                           timestamp=entry_date)

        day2 = drive(desk, frames, dates[loc + 1:loc + 2], portfolio)
        assert not option_intents(day2[dates[loc + 1]])  # trending: holds

        flat = drive(desk, frames, [flatten_date], portfolio)
        closes = option_intents(flat[flatten_date])
        assert len(closes) == 8  # 4 legs x 2 structures, ALL vrp closed
        assert sorted(i.action for i in closes) == ['COVER'] * 4 + \
            ['SELL'] * 4
        for structure_id in open_ids:
            assert desk.tracker.get(structure_id)['close_reason'] == \
                'regime_flatten'
        flatten_notes = [n for n in desk.notes
                         if n.data.get('close_reason') == 'regime_flatten']
        assert len(flatten_notes) == 2
        for note in flatten_notes:
            assert note.data['book'] == 'vrp'
            assert note.data['structure_id'] in open_ids

    def test_closes_pass_even_when_vega_limit_is_breached(self):
        frames = {'AAA': vol_frame(shift_at=SHIFT)}
        dates = frames['AAA'].index
        probe = make_desk(profit_target_pct=0.0, stop_loss_mult=1e9,
                          time_exit_dte=-1)
        by_date = drive(probe, frames, dates, PortfolioManager(100_000.0))
        entry_date = min(d for d, intents in by_date.items()
                         if option_intents(intents))
        loc = dates.get_loc(entry_date)
        flatten_date = dates[loc + 2]

        desk = make_desk(profit_target_pct=0.0, stop_loss_mult=1e9,
                         time_exit_dte=-1,
                         schedule={flatten_date: regime('high_vol')})
        portfolio = PortfolioManager(100_000.0)
        drive(desk, frames, dates[:loc + 2], portfolio)
        (structure_id,) = desk.tracker.open_ids()
        fill_structure(portfolio, desk, structure_id, timestamp=entry_date)

        # The open book now carries far more |vega| than the cap allows:
        # closes must STILL go out (the limit blocks only new opens).
        desk.vega_limit = 1e-9
        flat = drive(desk, frames, [flatten_date], portfolio)
        assert len(option_intents(flat[flatten_date])) == 4


class TestVegaLimit:
    def test_would_breach_entry_is_blocked_by_the_risk_book(self):
        frames = {'AAA': vol_frame(shift_at=SHIFT)}
        desk = make_desk(vega_limit=1.0)  # $1 of vega per $100k: nothing fits
        by_date = drive(desk, frames, frames['AAA'].index,
                        PortfolioManager(100_000.0))
        assert not any(option_intents(i) for i in by_date.values())
        assert desk.tracker.to_report() == []
        blocked = [n for n in desk.notes
                   if 'Central risk book blocked' in n.message]
        assert blocked
        assert blocked[0].data['book'] == 'vrp'
        assert 'short_vega' in blocked[0].data['reason']

    def test_default_limit_admits_the_documented_condor(self):
        # Phase 8 validation regression: the default cap must be in the
        # SAME per-1.00-vol-point units as greeks_series (50,000 per
        # $100k == $500 per 1% vol point per $100k). A 2%-risk-budget
        # condor runs net vega around -1,800 in those units, and both
        # the limit and the vega scale linearly with capital — the old
        # per-1%-style default of 500 blocked EVERY entry at any size.
        assert JaneStreetDesk().vega_limit == 50_000.0
        frames = {'AAA': vol_frame(shift_at=SHIFT)}
        desk = make_desk()  # class-default vega_limit
        by_date = drive(desk, frames, frames['AAA'].index,
                        PortfolioManager(100_000.0))
        assert any(option_intents(i) for i in by_date.values())
        assert desk.tracker.to_report()
        assert not [n for n in desk.notes
                    if 'Central risk book blocked' in n.message]


class TestSharedRiskNotesCarryBooks:
    """C14 regression (Phase 8 validation): the SHARED Desk.apply_risk
    notes must carry data.book too. Registry-default runs emitted
    untagged 'Blocked SHORT ...' notes when the RV ETF leg hit the
    position cap; risk_note_data now tags every base-class note."""

    FRAMES = {'AAA': vol_frame()}
    DATE = FRAMES['AAA'].index[70]

    def test_stock_block_and_stop_are_tagged_relative_value(self):
        desk = make_desk(
            risk_manager=RiskManager(max_position_size=0.01))
        portfolio = PortfolioManager(100_000.0)
        # A stopped long: entry 100, current 90 breaches the 2% stop.
        portfolio.add_position(Position(
            asset=Asset(symbol='BBB', asset_type=AssetType.STOCK),
            quantity=100, avg_entry_price=100.0, current_price=90.0,
            timestamp=self.DATE))
        desk.set_clock(self.DATE)
        intent = DeskIntent(
            asset=Asset(symbol='AAA', asset_type=AssetType.STOCK),
            action='SHORT', size_fraction=0.10, reason='rv etf leg')
        approved = desk.apply_risk([intent], portfolio, self.FRAMES,
                                   self.DATE)
        assert intent not in approved  # blocked by the 1% cap
        block = next(n for n in desk.notes
                     if 'position-size limit' in n.message)
        assert block.data['book'] == 'relative_value'
        stop = next(n for n in desk.notes
                    if 'Stop-loss SELL' in n.message)
        assert stop.data['book'] == 'relative_value'

    def test_option_leg_block_is_tagged_with_its_structures_book(self):
        desk = make_desk(
            risk_manager=RiskManager(max_position_size=0.01))
        leg_asset = Asset(symbol='AAA', asset_type=AssetType.PUT,
                          strike_price=95.0, expiration_date='2022-05-20')
        structure_id = desk.tracker.open_structure(
            'iron_condor', 'AAA',
            [{'asset': leg_asset, 'action': 'SHORT'}],
            credit=100.0, max_loss=400.0, contracts=1, date=self.DATE)
        desk._structure_books[structure_id] = 'earnings'
        desk.set_clock(self.DATE)
        intent = DeskIntent(asset=leg_asset, action='SHORT',
                            size_fraction=0.10, reason='condor leg',
                            quantity=1)
        approved = desk.apply_risk([intent], PortfolioManager(100_000.0),
                                   self.FRAMES, self.DATE)
        assert approved == []
        block = next(n for n in desk.notes
                     if 'position-size limit' in n.message)
        assert block.data['book'] == 'earnings'

    def test_daily_loss_circuit_note_is_tagged_regime(self):
        desk = make_desk(risk_manager=RiskManager())  # 5% daily circuit
        portfolio = PortfolioManager(100_000.0)
        portfolio.portfolio_history.append({'portfolio_value': 200_000.0})
        desk.set_clock(self.DATE)
        intent = DeskIntent(
            asset=Asset(symbol='AAA', asset_type=AssetType.STOCK),
            action='BUY', size_fraction=0.05, reason='any opening intent')
        approved = desk.apply_risk([intent], portfolio, self.FRAMES,
                                   self.DATE)
        assert approved == []
        circuit = next(n for n in desk.notes
                       if 'circuit breaker' in n.message)
        assert circuit.data['book'] == 'regime'

    def test_default_sized_rv_etf_leg_clears_the_position_cap(
            self, monkeypatch):
        # Phase 8 validation regression: weight * 0.5 = 0.10 sat exactly
        # ON the default 10% cap and floating-point drift blocked the
        # ETF leg whenever the portfolio was above water (the RV book
        # was effectively dead at registry defaults). The leg is now
        # sized strictly INSIDE the cap.
        frames = {'AAA': vol_frame(), 'BBB': vol_frame(base=50.0),
                  'CCC': vol_frame(base=80.0)}
        dates = frames['AAA'].index
        desk = make_desk(risk_manager=RiskManager(),  # the real 10% cap
                         book_weights={'vrp': 0.0, 'earnings': 0.0,
                                       'relative_value': 0.2})
        portfolio = PortfolioManager(100_535.41)  # FP-hostile, above water
        monkeypatch.setattr(desk, '_rv_zscore',
                            lambda *args, **kwargs: 2.5)
        entry = drive(desk, frames, dates[70:71], portfolio)[dates[70]]
        assert {i.asset.symbol: i.action for i in entry} == {
            'AAA': 'SHORT', 'BBB': 'BUY', 'CCC': 'BUY'}
        approved = desk.apply_risk(entry, portfolio, frames, dates[70])
        assert len(approved) == 3  # the ETF leg passes the cap now
        assert not [n for n in desk.notes
                    if 'position-size limit' in n.message]
        # The clamp keeps the spread balanced: basket half == ETF half.
        by_symbol = {i.asset.symbol: i for i in approved}
        assert by_symbol['AAA'].size_fraction == pytest.approx(
            by_symbol['BBB'].size_fraction + by_symbol['CCC'].size_fraction)


class TestEarningsBook:
    EARNINGS = pd.Timestamp('2022-04-20')  # a Wednesday

    def frames(self):
        return {'AAA': vol_frame(shift_at=None)}

    def desk(self, **kwargs):
        calendar = SyntheticEarningsCalendar(
            {'AAA': [self.EARNINGS.date()]})
        return make_desk(
            earnings_calendar=calendar,
            book_weights={'vrp': 0.0, 'earnings': 0.5,
                          'relative_value': 0.0},
            **kwargs)

    def test_enters_exactly_two_trading_days_before_earnings(self):
        frames = self.frames()
        desk = self.desk()
        dates = frames['AAA'].index
        by_date = drive(desk, frames, dates, PortfolioManager(100_000.0))

        entry_dates = [d for d, intents in by_date.items()
                       if option_intents(intents)]
        assert len(entry_dates) == 1
        # Exactly 2 trading days before E: busday_count(date, E) == 2.
        assert int(np.busday_count(entry_dates[0].date(),
                                   self.EARNINGS.date())) == 2
        assert entry_dates[0] == self.EARNINGS - pd.offsets.BDay(2)

        # Inspect the tracker RIGHT AFTER entry (a never-filled entry is
        # reconciled away later in the full drive above): fresh desk,
        # driven through the entry date only.
        desk = self.desk()
        loc = dates.get_loc(entry_dates[0])
        drive(desk, frames, dates[:loc + 1], PortfolioManager(100_000.0))
        (structure_id,) = desk.tracker.open_ids()
        structure = desk.tracker.get(structure_id)
        # 21 DTE (inside the 14-30 band), spanning the event date.
        assert structure['expiry'] == (entry_dates[0]
                                       + pd.Timedelta(days=21)).date()
        note = next(n for n in desk.notes
                    if n.data.get('book') == 'earnings'
                    and 'earnings_date' in n.data)
        assert note.data['structure_id'] == structure_id

    def test_exits_on_the_first_trading_day_after_earnings(self):
        frames = self.frames()
        desk = self.desk()
        dates = frames['AAA'].index
        portfolio = PortfolioManager(100_000.0)
        entry_date = self.EARNINGS - pd.offsets.BDay(2)
        loc = dates.get_loc(entry_date)
        drive(desk, frames, dates[:loc + 1], portfolio)
        (structure_id,) = desk.tracker.open_ids()
        fill_structure(portfolio, desk, structure_id, timestamp=entry_date)

        # Through E itself: filled, regime fine, NO exit yet.
        e_loc = dates.get_loc(self.EARNINGS)
        held = drive(desk, frames, dates[loc + 1:e_loc + 1], portfolio)
        assert not any(option_intents(i) for i in held.values())

        # First trading day AFTER E: the crush capture closes it.
        day_after = dates[e_loc + 1]
        closes = option_intents(
            drive(desk, frames, [day_after], portfolio)[day_after])
        assert len(closes) == 4
        assert desk.tracker.get(structure_id)['close_reason'] == 'time_exit'
        crush = next(n for n in desk.notes if 'crush' in n.message.lower())
        assert crush.data['book'] == 'earnings'
        assert crush.data['structure_id'] == structure_id
        # The IV bump (1 + 0.4*exp(-d/3)) is in the entry mark and GONE
        # at exit: the captured crush is positive by construction.
        assert crush.data['entry_iv'] > crush.data['exit_iv']

    def test_without_a_calendar_the_book_idles_with_one_note(self):
        frames = self.frames()
        desk = make_desk()  # no calendar injected
        drive(desk, frames, frames['AAA'].index[:3],
              PortfolioManager(100_000.0))
        idle = [n for n in desk.notes if n.category == 'info'
                and 'idle' in n.message]
        assert len(idle) == 1
        assert idle[0].data['book'] == 'earnings'


class TestRelativeValueBook:
    def frames(self, n=120):
        return {'AAA': vol_frame(n=n), 'BBB': vol_frame(n=n, base=50.0),
                'CCC': vol_frame(n=n, base=80.0)}

    def rv_desk(self):
        return make_desk(book_weights={'vrp': 0.0, 'earnings': 0.0,
                                       'relative_value': 0.2})

    def test_z_boundaries_exact(self, monkeypatch):
        frames = self.frames()
        dates = frames['AAA'].index
        desk = self.rv_desk()
        portfolio = PortfolioManager(100_000.0)
        z_value = {'z': 2.0}
        monkeypatch.setattr(desk, '_rv_zscore',
                            lambda *args, **kwargs: z_value['z'])

        # |z| == 2.0 exactly: STRICT > required, no entry.
        day = drive(desk, frames, dates[70:71], portfolio)
        assert not day[dates[70]]

        z_value['z'] = 2.01  # beyond the boundary: enter
        entry = drive(desk, frames, dates[71:72], portfolio)[dates[71]]
        # ETF rich: SHORT the ETF (AAA, first alphabetically — the
        # documented default), BUY the equal-weight basket.
        by_symbol = {i.asset.symbol: i for i in entry}
        assert by_symbol['AAA'].action == 'SHORT'
        assert by_symbol['AAA'].size_fraction == pytest.approx(0.1)
        for symbol in ('BBB', 'CCC'):
            assert by_symbol[symbol].action == 'BUY'
            assert by_symbol[symbol].size_fraction == pytest.approx(0.05)

        # Fill the legs so the exit path has positions to close.
        for symbol, intent in by_symbol.items():
            quantity = 100 if intent.action == 'BUY' else -100
            portfolio.add_position(Position(
                asset=intent.asset, quantity=quantity,
                avg_entry_price=50.0, current_price=50.0,
                timestamp=dates[71]))

        z_value['z'] = 0.5  # |z| == 0.5 exactly: STRICT < required, hold
        hold = drive(desk, frames, dates[72:73], portfolio)
        assert not hold[dates[72]]

        z_value['z'] = 0.499  # inside: exit every leg
        exits = drive(desk, frames, dates[73:74], portfolio)[dates[73]]
        actions = {i.asset.symbol: i.action for i in exits}
        assert actions == {'AAA': 'COVER', 'BBB': 'SELL', 'CCC': 'SELL'}
        rv_notes = [n for n in desk.notes
                    if n.data.get('book') == 'relative_value']
        assert rv_notes  # C14 book tag on every RV note

    def test_organic_entry_on_a_real_spread_dislocation(self):
        n = 80
        frames = self.frames(n=n)
        # Hand the ETF a violent idiosyncratic jump on the last day:
        # spread = log(etf) - mean(log(basket)) dislocates -> z >> 2.
        jump = frames['AAA'].copy()
        jump.iloc[-1] = jump.iloc[-2] * 1.05
        frames['AAA'] = jump
        desk = self.rv_desk()
        dates = frames['AAA'].index
        by_date = drive(desk, frames, dates[-3:],
                        PortfolioManager(100_000.0))
        entry = by_date[dates[-1]]
        assert {i.asset.symbol: i.action for i in entry} == {
            'AAA': 'SHORT', 'BBB': 'BUY', 'CCC': 'BUY'}


# ----------------------------------------------------------------------
# End to end: real engine, synthetic universe + calendar
# ----------------------------------------------------------------------
class TestEndToEndWithEngine:
    N_DAYS = 160
    EARNINGS_IDX = 40

    def universe(self):
        return {'AAA': vol_frame(n=self.N_DAYS, shift_at=SHIFT),
                'BBB': vol_frame(n=self.N_DAYS, shift_at=SHIFT)}

    def build_report(self, monkeypatch):
        universe = self.universe()
        dates = universe['AAA'].index
        earnings = dates[self.EARNINGS_IDX].date()

        def fake_fetch(self, symbol, start_date, end_date):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        # No vega_limit override: the end-to-end run exercises the
        # registry default (the old explicit 50_000 masked a default
        # that blocked every entry).
        desk = make_desk(
            earnings_calendar=SyntheticEarningsCalendar(
                {'BBB': [earnings]}))
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        report = engine.run(sorted(universe), '2022-01-01', '2022-12-31',
                            benchmark_symbol=None)
        return report, desk, dates

    @pytest.fixture
    def run_once(self, monkeypatch):
        return self.build_report(monkeypatch)

    def test_c11_structures_shape_and_lifecycle(self, run_once):
        report, desk, dates = run_once
        structures = report['structures']
        assert structures
        json.dumps(structures)  # JSON-clean by contract
        books_seen = set()
        for s in structures:
            assert set(s) == {'id', 'type', 'underlying', 'opened',
                              'closed', 'credit', 'max_loss', 'contracts',
                              'status', 'pnl', 'close_reason', 'legs'}
            assert s['type'] in ('iron_condor', 'put_credit_spread',
                                 'call_credit_spread')
            assert s['status'] in ('open', 'closed', 'expired')
            assert s['contracts'] >= 1
            assert s['max_loss'] > 0
            assert len(s['legs']) == 4
            for leg in s['legs']:
                assert set(leg) == {'instrument', 'action', 'strike',
                                    'expiry', 'right'}
                assert leg['right'] in ('call', 'put')
                assert leg['action'] in ('SHORT', 'BUY')
            if s['status'] in ('closed', 'expired'):
                assert s['closed'] is not None
                assert s['pnl'] is not None
                assert s['close_reason'] in (
                    'profit_target', 'stop_loss', 'time_exit',
                    'regime_flatten', 'expiry')
            else:
                assert s['pnl'] is None
        # Both books traded: the earnings condor on BBB plus vrp entries.
        note_books = {n['data'].get('book') for n in report['trader_notes']
                      if 'structure_id' in n['data']}
        books_seen |= note_books
        assert {'vrp', 'earnings'} <= books_seen
        # At least one structure completed its lifecycle.
        assert any(s['status'] in ('closed', 'expired')
                   for s in structures)

    def test_structures_reconcile_with_closed_trades(self, run_once):
        report, _, _ = run_once
        trades_by_instrument = {}
        for trade in report['closed_trades']:
            trades_by_instrument.setdefault(trade['instrument'],
                                            []).append(trade)
        done = [s for s in report['structures']
                if s['status'] in ('closed', 'expired')]
        assert done
        for s in done:
            leg_pnl = 0.0
            for leg in s['legs']:
                matches = trades_by_instrument.get(leg['instrument'], [])
                assert len(matches) == 1, \
                    f"leg {leg['instrument']} has {len(matches)} trades"
                trade = matches[0]
                assert abs(trade['quantity']) == s['contracts']
                leg_pnl += trade['pnl']
            # Structure pnl == the sum of its legs' closed-trade P&L
            # (multiplier-aware; commissions are cash-only everywhere).
            assert s['pnl'] == pytest.approx(leg_pnl, abs=1e-9)

    def test_c12_greeks_series(self, run_once):
        report, _, dates = run_once
        series = report['greeks_series']
        json.dumps(series)
        assert len(series) == len(dates)  # one entry per simulated day
        for entry in series:
            assert set(entry) == {'date', 'delta', 'gamma', 'theta',
                                  'vega'}
        # Option-free warm-up days: all zeros.
        for entry in series[:20]:
            assert (entry['delta'], entry['gamma'], entry['theta'],
                    entry['vega']) == (0.0, 0.0, 0.0, 0.0)
        # Short premium on: net vega is NEGATIVE on at least one day,
        # and theta is positive there (short condors collect decay).
        short_days = [e for e in series if e['vega'] < 0]
        assert short_days
        assert any(e['theta'] > 0 for e in short_days)

    def test_c14_every_note_carries_a_book(self, run_once):
        report, _, _ = run_once
        notes = report['trader_notes']
        assert notes
        for note in notes:
            assert note['data'].get('book') in C14_BOOKS, note
            if 'structure' in note['data']:
                assert 'structure_id' in note['data']

    def test_route_equivalent_defaults_trade_and_stay_c14_clean(
            self, monkeypatch):
        # Phase 8 validation regression, both findings at once: with the
        # DEFAULT RiskManager (10% cap, 5% circuit, 2% stops — exactly
        # what POST /api/backtest/run wires) and the DEFAULT vega limit,
        # the desk must still open structures (the old 500 per-$100k
        # default blocked every entry) and EVERY note — including the
        # shared apply_risk notes the default run used to emit untagged
        # — must carry a C14 book.
        universe = self.universe()

        def fake_fetch(self, symbol, start_date, end_date):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        desk = make_desk(risk_manager=RiskManager())
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        report = engine.run(sorted(universe), '2022-01-01', '2022-12-31',
                            benchmark_symbol=None)
        assert report['structures']  # the default vega limit admits entries
        notes = report['trader_notes']
        assert notes
        for note in notes:
            assert note['data'].get('book') in C14_BOOKS, note

    def test_regime_fits_tagged_and_c13_instruments(self, run_once):
        report, _, _ = run_once
        assert report['walk_forward']
        assert {fit['model'] for fit in report['walk_forward']} == \
            {'regime'}
        assert report['desk'] == {'key': 'janestreet',
                                  'name': 'Jane Street Desk'}
        option_trades = [t for t in report['trades']
                         if 'call' in t['instrument']
                         or 'put' in t['instrument']]
        assert option_trades
        for trade in report['trades']:
            assert 'instrument' in trade

    def test_deterministic_across_three_runs(self, monkeypatch):
        def fingerprint():
            report, _, _ = self.build_report(monkeypatch)
            return json.dumps({
                'structures': report['structures'],
                'greeks_series': report['greeks_series'],
                'trader_notes': report['trader_notes'],
                'closed_trades': report['closed_trades'],
                'summary': report['summary'],
            }, sort_keys=True, default=str)

        first = fingerprint()
        assert fingerprint() == first
        assert fingerprint() == first


class TestExerciseStyle:
    """Phase 5: opt-in American (early-exercise) pricing on the desk's option
    legs. Default 'european' is byte-identical Black-Scholes; 'american' adds
    the early-exercise premium real single-name equity options carry. Price
    layer only (fills + structure marks); greeks stay Black-Scholes."""

    def test_default_is_european(self):
        assert JaneStreetDesk().exercise_style == 'european'

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="exercise_style must be"):
            JaneStreetDesk(exercise_style='bermudan')

    def test_european_leg_matches_black_scholes(self):
        from desks.options_pricing import black_scholes_price
        desk = JaneStreetDesk()  # default european
        got = desk._price_leg(100.0, 95.0, 0.25, 0.30, 'put')
        assert got == black_scholes_price(100.0, 95.0, 0.25, 0.30,
                                          desk.rate, 'put')

    def test_american_put_carries_early_exercise_premium(self):
        # Deep ITM put with positive carry: early exercise has real value,
        # so the American leg prices strictly above its European twin.
        eu = JaneStreetDesk(exercise_style='european')
        am = JaneStreetDesk(exercise_style='american')
        eu_px = eu._price_leg(80.0, 110.0, 0.5, 0.30, 'put')
        am_px = am._price_leg(80.0, 110.0, 0.5, 0.30, 'put')
        assert am_px > eu_px

    def test_american_call_no_dividend_equals_european(self):
        # No dividends are modeled, so an American call never exercises early
        # -> it equals the European (Black-Scholes) value.
        am = JaneStreetDesk(exercise_style='american')
        eu = JaneStreetDesk(exercise_style='european')
        am_c = am._price_leg(100.0, 95.0, 0.5, 0.30, 'call')
        eu_c = eu._price_leg(100.0, 95.0, 0.5, 0.30, 'call')
        assert am_c == pytest.approx(eu_c, rel=1e-3)
