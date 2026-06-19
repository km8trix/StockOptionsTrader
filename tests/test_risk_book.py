"""Tests for desks.risk_book.CentralRiskBook.

Exposure arithmetic on a constructed mixed long/short book, CUMULATIVE
batch evaluation (item-wise-fine batches cannot jointly breach the book),
closes-always-approved, boundary exactness (at the limit passes; epsilon
over blocks), the register_limit extension point, and the Phase 8 Greeks
stub raising when configured. Offline, deterministic, no engine.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from core.models import Asset, AssetType, Position
from desks.base import DeskIntent
from desks.factor_risk import FactorRiskModel
from desks.risk_book import CentralRiskBook
from portfolio.manager import PortfolioManager


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def intent(symbol: str, action: str, size_fraction: float = 0.1
           ) -> DeskIntent:
    return DeskIntent(asset=stock(symbol), action=action,
                      size_fraction=size_fraction, reason='test')


def add_position(portfolio: PortfolioManager, symbol: str, quantity: int,
                 price: float) -> None:
    """Add a position with the matching cash adjustment, so portfolio
    value stays consistent (shorts credit cash, longs debit it)."""
    portfolio.cash -= quantity * price
    portfolio.add_position(Position(
        asset=stock(symbol), quantity=quantity, avg_entry_price=price,
        current_price=price, timestamp=datetime(2023, 1, 2)))


def cash_portfolio(value: float = 100_000.0) -> PortfolioManager:
    return PortfolioManager(value)


class TestExposureArithmetic:
    def _mixed_book(self) -> PortfolioManager:
        portfolio = cash_portfolio()
        add_position(portfolio, 'AAA', 200, 100.0)   # long  20,000
        add_position(portfolio, 'BBB', -100, 50.0)   # short  5,000
        # cash = 100k - 20k + 5k = 85k; value = 85k + 20k - 5k = 100k.
        assert portfolio.get_portfolio_value() == pytest.approx(100_000.0)
        return portfolio

    def test_gross_net_short_per_symbol(self):
        exposures = CentralRiskBook().compute_exposures(self._mixed_book())
        assert exposures['per_symbol'] == {'AAA': 20_000.0, 'BBB': -5_000.0}
        assert exposures['gross'] == pytest.approx(25_000.0)
        assert exposures['net'] == pytest.approx(15_000.0)
        assert exposures['short'] == pytest.approx(5_000.0)

    def test_prices_override_position_marks(self):
        exposures = CentralRiskBook().compute_exposures(
            self._mixed_book(), prices={'AAA': 110.0, 'BBB': 40.0})
        assert exposures['per_symbol'] == {'AAA': 22_000.0, 'BBB': -4_000.0}
        assert exposures['gross'] == pytest.approx(26_000.0)
        assert exposures['net'] == pytest.approx(18_000.0)
        assert exposures['short'] == pytest.approx(4_000.0)


class TestCumulativeEvaluation:
    def test_two_intents_individually_fine_jointly_breaching_gross(self):
        book = CentralRiskBook(max_gross=0.5, max_symbol=0.5, max_net=1.0)
        first = intent('AAA', 'BUY', 0.3)    # 30,000 <= 50,000: fine alone
        second = intent('BBB', 'BUY', 0.3)   # alone fine; jointly 60,000
        approved, blocked = book.check([first, second], cash_portfolio())
        assert approved == [first]
        assert len(blocked) == 1
        blocked_intent, reason = blocked[0]
        assert blocked_intent is second
        assert 'gross' in reason

    def test_evaluation_respects_caller_order(self):
        # The book evaluates in the order GIVEN (callers pre-sort): the
        # first listed intent wins the remaining room, not the
        # alphabetically-first symbol.
        book = CentralRiskBook(max_gross=0.5, max_symbol=0.5, max_net=1.0)
        first = intent('ZZZ', 'BUY', 0.3)
        second = intent('AAA', 'BUY', 0.3)
        approved, blocked = book.check([first, second], cash_portfolio())
        assert approved == [first]
        assert blocked[0][0] is second

    def test_net_band_is_two_sided_and_cumulative(self):
        book = CentralRiskBook(max_gross=2.0, max_symbol=1.0, max_net=0.6)
        first = intent('AAA', 'SHORT', 0.4)   # net -40,000: inside band
        second = intent('BBB', 'SHORT', 0.3)  # would push net to -70,000
        approved, blocked = book.check([first, second], cash_portfolio())
        assert approved == [first]
        assert 'net' in blocked[0][1]

    def test_per_symbol_aggregates_existing_position_with_pending(self):
        portfolio = cash_portfolio()
        add_position(portfolio, 'AAA', 80, 100.0)  # existing 8,000
        book = CentralRiskBook(max_gross=1.0, max_symbol=0.10, max_net=1.0)
        # 8,000 existing + 3,000 pending = 11,000 > 10,000 (10% of 100k).
        approved, blocked = book.check([intent('AAA', 'BUY', 0.03)],
                                       portfolio)
        assert approved == []
        assert 'AAA' in blocked[0][1] and 'per-symbol' in blocked[0][1]


class TestClosingIntents:
    def test_closes_always_approved_even_over_limits(self):
        portfolio = cash_portfolio()
        add_position(portfolio, 'AAA', 500, 100.0)   # 50% of capital
        add_position(portfolio, 'BBB', -300, 100.0)  # 30% short
        book = CentralRiskBook(max_gross=0.01, max_symbol=0.01,
                               max_net=0.01)  # everything is over
        closes = [intent('AAA', 'SELL', 1.0), intent('BBB', 'COVER', 1.0)]
        approved, blocked = book.check(closes, portfolio)
        assert approved == closes
        assert blocked == []

    def test_close_relief_is_not_credited_within_the_batch(self):
        # Conservative by design: the SELL might not fill while the BUY
        # does, so the SELL's freed-up gross must NOT admit the BUY.
        portfolio = cash_portfolio()
        add_position(portfolio, 'AAA', 1000, 100.0)  # gross 100,000
        book = CentralRiskBook(max_gross=1.0, max_symbol=1.0, max_net=2.0)
        sell = intent('AAA', 'SELL', 1.0)
        buy = intent('BBB', 'BUY', 0.5)
        approved, blocked = book.check([sell, buy], portfolio)
        assert approved == [sell]
        assert blocked[0][0] is buy and 'gross' in blocked[0][1]


class TestBoundaryExactness:
    def _half_invested(self) -> PortfolioManager:
        portfolio = cash_portfolio()
        add_position(portfolio, 'AAA', 500, 100.0)  # 50,000 of 100,000
        assert portfolio.get_portfolio_value() == pytest.approx(100_000.0)
        return portfolio

    def test_gross_exactly_at_limit_passes(self):
        book = CentralRiskBook(max_gross=1.0, max_symbol=1.0, max_net=1.0)
        approved, blocked = book.check([intent('BBB', 'BUY', 0.5)],
                                       self._half_invested())
        # 50,000 + 50,000 == 100,000 == the limit: at-limit passes.
        assert len(approved) == 1 and blocked == []

    def test_gross_epsilon_over_limit_blocks(self):
        book = CentralRiskBook(max_gross=1.0, max_symbol=1.0, max_net=1.0)
        approved, blocked = book.check([intent('BBB', 'BUY', 0.500001)],
                                       self._half_invested())
        assert approved == [] and 'gross' in blocked[0][1]

    def test_symbol_exactly_at_limit_passes_epsilon_blocks(self):
        book = CentralRiskBook(max_gross=1.0, max_symbol=0.10, max_net=1.0)
        approved, blocked = book.check([intent('AAA', 'BUY', 0.10)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []
        approved, blocked = book.check([intent('AAA', 'BUY', 0.100001)],
                                       cash_portfolio())
        assert approved == [] and 'per-symbol' in blocked[0][1]

    def test_net_exactly_at_band_passes_epsilon_blocks(self):
        book = CentralRiskBook(max_gross=1.0, max_symbol=1.0, max_net=0.6)
        approved, blocked = book.check([intent('AAA', 'SHORT', 0.6)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []
        approved, blocked = book.check([intent('AAA', 'SHORT', 0.600001)],
                                       cash_portfolio())
        assert approved == [] and 'net' in blocked[0][1]


class TestRegisterLimit:
    def test_custom_limit_blocks_with_named_reason(self):
        book = CentralRiskBook(max_gross=2.0, max_symbol=1.0, max_net=2.0)
        seen_states = []

        def no_shorts(check_intent, state):
            seen_states.append(state)
            if check_intent.action == 'SHORT':
                return 'shorts disabled'
            return None

        book.register_limit('no_shorts', no_shorts)
        buy = intent('AAA', 'BUY', 0.1)
        short = intent('BBB', 'SHORT', 0.1)
        approved, blocked = book.check([buy, short], cash_portfolio())
        assert approved == [buy]
        assert blocked[0][0] is short
        assert "custom limit 'no_shorts': shorts disabled" in blocked[0][1]
        # The state handed to custom limits carries the documented view.
        assert {'desk_capital', 'gross', 'net', 'short', 'per_symbol',
                'intent_value', 'candidate_gross', 'candidate_net',
                'candidate_symbol', 'candidate_short'} <= set(seen_states[0])

    def test_custom_limit_sees_cumulative_state(self):
        book = CentralRiskBook(max_gross=2.0, max_symbol=1.0, max_net=2.0)
        grosses = []
        book.register_limit(
            'spy', lambda i, state: grosses.append(state['gross']))
        book.check([intent('AAA', 'BUY', 0.2), intent('BBB', 'BUY', 0.1)],
                   cash_portfolio())
        # Second intent must see the first one's committed 20,000.
        assert grosses == [pytest.approx(0.0), pytest.approx(20_000.0)]

    def test_non_callable_limit_rejected(self):
        with pytest.raises(ValueError, match='callable'):
            CentralRiskBook().register_limit('bad', 'not-a-function')


class TestGreeksStub:
    """Configured Greeks limits with NO greeks_aggregator injected must
    raise loudly (the feed needed to evaluate them is absent) rather than
    be silently skipped. The aggregator-present path is TestGreeksGate."""

    def test_short_vega_configured_raises(self):
        book = CentralRiskBook(short_vega_limit=1_000.0)
        with pytest.raises(NotImplementedError, match='Phase 8'):
            book.check([], cash_portfolio())

    def test_short_gamma_configured_raises(self):
        book = CentralRiskBook(short_gamma_limit=50.0)
        with pytest.raises(NotImplementedError, match='Phase 8'):
            book.check([intent('AAA', 'BUY', 0.1)], cash_portfolio())

    def test_unconfigured_book_does_not_raise(self):
        approved, blocked = CentralRiskBook().check([], cash_portfolio())
        assert approved == [] and blocked == []


class TestFailClosed:
    def test_non_positive_capital_blocks_opens_approves_closes(self):
        portfolio = PortfolioManager(0.0)
        book = CentralRiskBook()
        sell = intent('AAA', 'SELL', 1.0)
        buy = intent('BBB', 'BUY', 0.1)
        approved, blocked = book.check([sell, buy], portfolio)
        assert approved == [sell]
        assert 'failing closed' in blocked[0][1]


class TestConstructionValidation:
    def test_bad_capital_allocation_raises(self):
        with pytest.raises(ValueError, match='capital_allocation'):
            CentralRiskBook(capital_allocation=0.0)

    @pytest.mark.parametrize('kwargs', [{'max_gross': 0.0},
                                        {'max_symbol': -0.1},
                                        {'max_net': 0.0}])
    def test_non_positive_limits_raise(self, kwargs):
        with pytest.raises(ValueError, match='must be > 0'):
            CentralRiskBook(**kwargs)


class _FakeGreeks:
    """Minimal greeks_aggregator stand-in: returns a fixed aggregate so the
    gate logic is tested independently of the aggregator's own math (that
    math is covered in test_greeks_aggregator)."""

    def __init__(self, vega: float = 0.0, gamma: float = 0.0):
        self._totals = {'delta': 0.0, 'gamma': gamma, 'theta': 0.0,
                        'vega': vega}

    def aggregate(self, portfolio):
        return dict(self._totals)


class TestGreeksGate:
    """short_vega/short_gamma evaluate as a desk-wide gate once a
    greeks_aggregator is injected: when the HELD book's aggregate short
    Greeks exceed the cap, every opening intent is blocked all-or-nothing
    (closes always pass). Without an aggregator a configured limit still
    fails loudly (TestGreeksStub)."""

    def test_over_vega_budget_blocks_opens_keeps_closes(self):
        book = CentralRiskBook(short_vega_limit=1_000.0,
                               greeks_aggregator=_FakeGreeks(vega=-1_500.0))
        buy = intent('AAA', 'BUY', 0.1)
        short = intent('BBB', 'SHORT', 0.1)
        sell = intent('CCC', 'SELL', 1.0)
        approved, blocked = book.check([buy, short, sell], cash_portfolio())
        assert approved == [sell]                      # close still passes
        blocked_intents = [i for i, _ in blocked]
        assert buy in blocked_intents and short in blocked_intents
        assert all('Greeks over budget' in r for _, r in blocked)
        assert any('short vega' in r for _, r in blocked)

    def test_under_vega_budget_is_a_no_op(self):
        book = CentralRiskBook(short_vega_limit=2_000.0,
                               greeks_aggregator=_FakeGreeks(vega=-1_500.0))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []

    def test_exactly_at_vega_limit_passes(self):
        # Boundary: |short vega| exactly at the cap passes (strict >).
        book = CentralRiskBook(short_vega_limit=1_500.0,
                               greeks_aggregator=_FakeGreeks(vega=-1_500.0))
        approved, _ = book.check([intent('AAA', 'BUY', 0.1)],
                                 cash_portfolio())
        assert len(approved) == 1

    def test_long_vega_never_trips_short_gate(self):
        # Positive (long) aggregate vega is not short exposure -> no block.
        book = CentralRiskBook(short_vega_limit=10.0,
                               greeks_aggregator=_FakeGreeks(vega=5_000.0))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []

    def test_short_gamma_gate(self):
        book = CentralRiskBook(short_gamma_limit=2.0,
                               greeks_aggregator=_FakeGreeks(gamma=-5.0))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == []
        assert any('short gamma' in r for _, r in blocked)

    def test_empty_intents_with_aggregator_does_not_raise(self):
        book = CentralRiskBook(short_vega_limit=1.0,
                               greeks_aggregator=_FakeGreeks(vega=-9_999.0))
        assert book.check([], cash_portfolio()) == ([], [])

    def test_builtin_block_is_not_reclassified_by_the_gate(self):
        # An intent the gross cap already blocks never reaches `approved`,
        # so the Greeks gate leaves it with its original gross reason.
        book = CentralRiskBook(max_gross=0.05, short_vega_limit=1_000.0,
                               greeks_aggregator=_FakeGreeks(vega=-2_000.0))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == []
        assert len(blocked) == 1 and 'gross' in blocked[0][1]

    def test_real_aggregator_blocks_open(self):
        # End-to-end with the real PortfolioGreeksAggregator: a held short
        # call (-5 contracts, +8 per-share vega) -> short vega 4,000 > cap.
        from portfolio.greeks_aggregator import PortfolioGreeksAggregator
        opt = Asset(symbol='AAA', asset_type=AssetType.CALL,
                    strike_price=105.0, expiration_date='2022-03-18')
        portfolio = cash_portfolio()
        portfolio.add_position(Position(
            asset=opt, quantity=-5, avg_entry_price=1.0, current_price=1.0,
            timestamp=datetime(2022, 1, 3)))
        agg = PortfolioGreeksAggregator(
            lambda a: {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0,
                       'vega': 8.0})
        book = CentralRiskBook(short_vega_limit=1_000.0,
                               greeks_aggregator=agg)
        approved, blocked = book.check([intent('BBB', 'BUY', 0.1)],
                                       portfolio)
        assert approved == []
        assert any('short vega' in r for _, r in blocked)


# ======================================================================
# Phase E: the factor-neutrality band (a registerable custom limit)
# ======================================================================
#
# CentralRiskBook.factor_neutrality_limit(model, loadings_ref, band) builds a
# custom check_fn that blocks an OPENING intent whose CANDIDATE book net
# factor exposure |net_factor[f]| > band for ANY factor (band is a fraction
# of desk capital). It composes with the dollar limits and the Greeks overlay
# without changing them. These tests pin the band via a single-factor
# ('momentum') model and HAND loadings so every threshold is exact, then
# confirm cumulative enforcement, graceful degrade, composition, and that
# closing intents are never reached.


def _band_book(loadings, band, factors=('momentum',),
               max_gross=2.0, max_symbol=1.0, max_net=2.0):
    """A wide-dollar-limit book with the factor-neutrality limit registered
    over a fixed loadings dict, so only the band gate is under test."""
    model = FactorRiskModel(factors=list(factors))
    ref = {'loadings': loadings}
    book = CentralRiskBook(max_gross=max_gross, max_symbol=max_symbol,
                           max_net=max_net)
    book.register_limit(
        'factor_neutrality',
        CentralRiskBook.factor_neutrality_limit(model, ref, band))
    return book, ref


class TestFactorNeutralityBlocking:
    def test_open_past_band_is_blocked_with_clear_reason(self):
        # BUY 0.1 of a +1.0-momentum name -> net momentum 0.10 of capital;
        # band 0.05 -> blocked. The reason names the factor and the band.
        book, _ = _band_book({'AAA': {'momentum': 1.0}}, band=0.05)
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == []
        assert len(blocked) == 1
        reason = blocked[0][1]
        assert "custom limit 'factor_neutrality'" in reason
        assert 'momentum' in reason and 'factor-neutrality band' in reason

    def test_open_within_band_passes(self):
        # Same trade, a band wide enough to admit it (0.10 -> exactly 0.10).
        book, _ = _band_book({'AAA': {'momentum': 1.0}}, band=0.20)
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []

    def test_short_side_of_band_is_two_sided(self):
        # A SHORT pushes the net factor exposure NEGATIVE; the band is
        # symmetric so a large short is blocked just like a large long.
        book, _ = _band_book({'AAA': {'momentum': 1.0}}, band=0.05)
        approved, blocked = book.check([intent('AAA', 'SHORT', 0.1)],
                                       cash_portfolio())
        assert approved == []
        assert 'momentum' in blocked[0][1]

    def test_open_that_offsets_a_held_tilt_passes_a_tight_band(self):
        # The gate measures the CANDIDATE BOOK, not a single trade: a new
        # open that pulls an already-held factor tilt back toward neutral
        # passes even a tight band, because the candidate book's net factor
        # exposure is what is compared. Held +momentum long; a -momentum-
        # contribution SHORT of equal size nets the book to ~0.
        portfolio = cash_portfolio()
        add_position(portfolio, 'AAA', 100, 100.0)  # +10,000 held, +momentum
        book, _ = _band_book(
            {'AAA': {'momentum': 1.0}, 'BBB': {'momentum': 1.0}}, band=0.001)
        # SHORT BBB 0.1 -> -10,000 on a +1.0 name -> candidate net momentum
        # = (10000*1 + (-10000)*1)/100000 = 0.0 -> passes the tight band.
        approved, blocked = book.check([intent('BBB', 'SHORT', 0.1)],
                                       portfolio)
        assert len(approved) == 1 and blocked == []

    def test_first_directional_leg_of_a_neutral_pair_is_blocked(self):
        # The gate is INCREMENTAL: it cannot foresee a later offsetting leg.
        # Under a tight band the FIRST directional open is blocked before the
        # second (neutralizing) intent is ever considered — the conservative,
        # cumulative construction the dollar limits also use.
        book, _ = _band_book(
            {'AAA': {'momentum': 1.0}, 'BBB': {'momentum': -1.0}}, band=0.001)
        first = intent('AAA', 'BUY', 0.1)
        second = intent('BBB', 'SHORT', 0.1)
        approved, blocked = book.check([first, second], cash_portfolio())
        assert blocked[0][0] is first and 'momentum' in blocked[0][1]

    def test_any_factor_over_band_blocks(self):
        # The block fires if ANY single factor breaches, even when others
        # are within band. low_vol is huge here; momentum is tiny.
        model = FactorRiskModel(factors=['momentum', 'low_vol'])
        ref = {'loadings': {'AAA': {'momentum': 0.01, 'low_vol': 1.0}}}
        book = CentralRiskBook(max_gross=2.0, max_symbol=1.0, max_net=2.0)
        book.register_limit(
            'factor_neutrality',
            CentralRiskBook.factor_neutrality_limit(model, ref, 0.05))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == []
        assert 'low_vol' in blocked[0][1]


class TestFactorNeutralityMonotoneImprovement:
    """The gate is an accumulation backstop that allows corrective hedges: an
    open is blocked only when it would push a factor beyond
    max(band, |held exposure|). So a within-band factor may not cross the
    band, an already-drifted factor may not be made WORSE, but a corrective
    open that reduces it passes (the desk can hedge back toward neutral)."""

    def _check_fn(self, loadings, band=0.25):
        model = FactorRiskModel(factors=['momentum'])
        return CentralRiskBook.factor_neutrality_limit(
            model, {'loadings': loadings}, band)

    @staticmethod
    def _state(per_symbol, candidate_value, desk_capital=100000.0):
        return {'desk_capital': desk_capital, 'per_symbol': per_symbol,
                'candidate_symbol': candidate_value}

    def test_in_band_accumulation_is_blocked(self):
        # Empty book; an open that would push momentum to 0.80 (> band 0.25).
        fn = self._check_fn({'A': {'momentum': 2.0}})
        reason = fn(intent('A', 'BUY', 0.4), self._state({}, 40000.0))
        assert reason is not None and 'momentum' in reason

    def test_corrective_open_passes_even_while_over_band(self):
        # Held momentum 0.80 (drifted over the 0.25 band as loadings rotated);
        # a long on a -momentum name pulls the book to 0.60 — still over band
        # but strictly improving — so the gate ADMITS the corrective hedge.
        fn = self._check_fn({'A': {'momentum': 2.0}, 'B': {'momentum': -1.0}})
        reason = fn(intent('B', 'BUY', 0.2),
                    self._state({'A': 40000.0}, 20000.0))
        assert reason is None

    def test_worsening_open_is_blocked_when_already_over_band(self):
        # Held momentum 0.80; a long on a +momentum name would worsen it to
        # 1.10 -> blocked, and the reason names the held breach.
        fn = self._check_fn({'A': {'momentum': 2.0}, 'C': {'momentum': 1.5}})
        reason = fn(intent('C', 'BUY', 0.2),
                    self._state({'A': 40000.0}, 20000.0))
        assert reason is not None and 'momentum' in reason
        assert 'worsen' in reason


class TestFactorNeutralityBoundary:
    def test_exactly_at_band_passes_epsilon_over_blocks(self):
        # BUY sf of a +1.0 name -> net = sf (of capital). band 0.10:
        # sf=0.10 is exactly at the band (passes); sf=0.100001 blocks.
        book, _ = _band_book({'AAA': {'momentum': 1.0}}, band=0.10)
        approved, blocked = book.check([intent('AAA', 'BUY', 0.10)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []
        approved, blocked = book.check([intent('AAA', 'BUY', 0.100001)],
                                       cash_portfolio())
        assert approved == [] and 'momentum' in blocked[0][1]


class TestFactorNeutralityCumulative:
    def test_item_wise_fine_jointly_breaching_is_caught(self):
        # Two BUYs each +0.10 momentum: alone each is inside a 0.15 band, but
        # the SECOND (committed cumulatively) would push the book to 0.20 and
        # is blocked — the running per_symbol state makes the batch joint.
        book, _ = _band_book(
            {'AAA': {'momentum': 1.0}, 'BBB': {'momentum': 1.0}}, band=0.15)
        first = intent('AAA', 'BUY', 0.1)
        second = intent('BBB', 'BUY', 0.1)
        approved, blocked = book.check([first, second], cash_portfolio())
        assert approved == [first]
        assert blocked[0][0] is second
        assert 'momentum' in blocked[0][1]

    def test_running_per_symbol_includes_held_position(self):
        # An existing +momentum long already sits in the book; a new open in
        # a DIFFERENT +momentum name is measured ON TOP of it (candidate book
        # = held per_symbol + the candidate), so it can breach when a fresh
        # book would not.
        portfolio = cash_portfolio()
        add_position(portfolio, 'AAA', 100, 100.0)  # +10,000 held
        book, _ = _band_book(
            {'AAA': {'momentum': 1.0}, 'BBB': {'momentum': 1.0}}, band=0.15)
        # Held AAA already contributes 0.10; BBB BUY 0.1 adds 0.10 -> 0.20.
        approved, blocked = book.check([intent('BBB', 'BUY', 0.1)], portfolio)
        assert approved == []
        assert 'momentum' in blocked[0][1]


class TestFactorNeutralityGraceful:
    def test_no_loadings_passes(self):
        # The desk hasn't published loadings yet (thin/empty data): the gate
        # passes (no false block), like the Greeks overlay when its feed is
        # silent.
        book, _ = _band_book({}, band=0.001)
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []

    def test_band_none_disables_the_gate(self):
        # band None recovers the old non-factor-gated behavior (always pass).
        book, _ = _band_book({'AAA': {'momentum': 1.0}}, band=None)
        approved, blocked = book.check([intent('AAA', 'BUY', 0.5)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []

    def test_band_non_positive_disables_the_gate(self):
        book, _ = _band_book({'AAA': {'momentum': 1.0}}, band=0.0)
        approved, blocked = book.check([intent('AAA', 'BUY', 0.5)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []

    def test_none_model_passes(self):
        # A factory built with model=None never blocks.
        book = CentralRiskBook(max_gross=2.0, max_symbol=1.0, max_net=2.0)
        book.register_limit(
            'factor_neutrality',
            CentralRiskBook.factor_neutrality_limit(
                None, {'loadings': {'AAA': {'momentum': 9.0}}}, 0.001))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert len(approved) == 1 and blocked == []

    def test_live_reference_is_read_each_check_without_reregistration(self):
        # The factory reads loadings_ref LAZILY: mutating the same dict in
        # place changes which trades block, with no re-registration. This is
        # how the Citadel desk refreshes the day's loadings.
        book, ref = _band_book({}, band=0.05)
        # Empty loadings -> passes.
        approved, _ = book.check([intent('AAA', 'BUY', 0.1)],
                                 cash_portfolio())
        assert len(approved) == 1
        # Publish a directional loading into the SAME ref -> now it blocks.
        ref['loadings'] = {'AAA': {'momentum': 1.0}}
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == [] and 'momentum' in blocked[0][1]

    def test_list_shaped_reference_supported(self):
        # The factory also accepts any object exposing [0] (e.g. a 1-element
        # list) as the live loadings container.
        model = FactorRiskModel(factors=['momentum'])
        ref = [{'AAA': {'momentum': 1.0}}]
        book = CentralRiskBook(max_gross=2.0, max_symbol=1.0, max_net=2.0)
        book.register_limit(
            'factor_neutrality',
            CentralRiskBook.factor_neutrality_limit(model, ref, 0.05))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == [] and 'momentum' in blocked[0][1]


class TestFactorNeutralityClosesAlwaysPass:
    def test_close_is_never_reached_by_the_factor_gate(self):
        # Closing intents (SELL/COVER) are approved by check() BEFORE the
        # per-intent custom-limit loop, so the band can never strand a
        # position it cannot unwind — even with a near-zero band.
        portfolio = cash_portfolio()
        add_position(portfolio, 'AAA', 100, 100.0)   # held long to sell
        add_position(portfolio, 'BBB', -100, 100.0)  # held short to cover
        book, _ = _band_book(
            {'AAA': {'momentum': 9.0}, 'BBB': {'momentum': 9.0}},
            band=1e-9)
        closes = [intent('AAA', 'SELL', 1.0), intent('BBB', 'COVER', 1.0)]
        approved, blocked = book.check(closes, portfolio)
        assert approved == closes and blocked == []


class TestFactorNeutralityComposition:
    """The factor limit is just another custom limit: it composes with the
    built-in dollar limits (evaluated first) and the Greeks overlay
    (evaluated last on the held book), changing none of them."""

    def test_dollar_limit_blocks_before_factor_limit(self):
        # An intent the gross cap already blocks never reaches the custom
        # limit, so it keeps its gross reason (not a factor reason).
        model = FactorRiskModel(factors=['momentum'])
        ref = {'loadings': {'AAA': {'momentum': 9.0}}}
        book = CentralRiskBook(max_gross=0.05, max_symbol=1.0, max_net=2.0)
        book.register_limit(
            'factor_neutrality',
            CentralRiskBook.factor_neutrality_limit(model, ref, 0.001))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == []
        assert len(blocked) == 1 and 'gross' in blocked[0][1]

    def test_factor_limit_runs_after_passing_the_dollar_limits(self):
        # A trade WITHIN every dollar limit is then judged by the factor
        # band and blocked with the factor reason.
        book, _ = _band_book({'AAA': {'momentum': 1.0}}, band=0.05,
                             max_gross=2.0, max_symbol=1.0, max_net=2.0)
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == []
        assert "factor_neutrality" in blocked[0][1]

    def test_composes_with_greeks_overlay(self):
        # The Greeks overlay (held-book, all-or-nothing, runs AFTER custom
        # limits) and the factor band both apply: a held short-vega book
        # over budget blocks the open via the Greeks gate, and the factor
        # limit leaves the gross/net/symbol + Greeks paths unchanged.
        model = FactorRiskModel(factors=['momentum'])
        ref = {'loadings': {'AAA': {'momentum': 0.01}}}  # within the band
        book = CentralRiskBook(max_gross=2.0, max_symbol=1.0, max_net=2.0,
                               short_vega_limit=1_000.0,
                               greeks_aggregator=_FakeGreeks(vega=-2_000.0))
        book.register_limit(
            'factor_neutrality',
            CentralRiskBook.factor_neutrality_limit(model, ref, 0.05))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        # The factor band would pass (0.001 < 0.05); the Greeks overlay
        # blocks it instead, with the Greeks reason — the overlay is intact.
        assert approved == []
        assert any('Greeks over budget' in r for _, r in blocked)

    def test_factor_limit_blocks_while_greeks_under_budget(self):
        # The factor band blocks even when the Greeks gate is clear, with the
        # factor reason — the two mechanisms are independent.
        model = FactorRiskModel(factors=['momentum'])
        ref = {'loadings': {'AAA': {'momentum': 1.0}}}
        book = CentralRiskBook(max_gross=2.0, max_symbol=1.0, max_net=2.0,
                               short_vega_limit=10_000.0,
                               greeks_aggregator=_FakeGreeks(vega=-100.0))
        book.register_limit(
            'factor_neutrality',
            CentralRiskBook.factor_neutrality_limit(model, ref, 0.05))
        approved, blocked = book.check([intent('AAA', 'BUY', 0.1)],
                                       cash_portfolio())
        assert approved == []
        assert "factor_neutrality" in blocked[0][1]
