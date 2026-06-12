"""Tests for desks.market_making — the Avellaneda-Stoikov simulator.

The canonical quote formulas are pinned to hand math at chosen (q, t);
the P&L identity (cash + inventory * mid == spread_pnl + inventory_pnl)
is asserted EVERY STEP by mirroring the simulation against the same
seeded LOB; determinism and the inventory-control claim (A-S
inventory_std < naive symmetric quoting, mean over 20 seeds) close it
out. SIMULATION ONLY — this module never touches a desk or engine path.
"""

from __future__ import annotations

import math

import pytest

from desks.market_making import ASMarketMaker, SyntheticLOB, run_demo

GAMMA, K, A, SIGMA, T = 0.1, 1.5, 140.0, 2.0, 1.0


def maker():
    return ASMarketMaker(gamma=GAMMA, k=K, A=A, sigma=SIGMA, T=T)


class TestQuoteFormulas:
    def test_reservation_price_and_spread_vs_hand_math(self):
        # s=100, q=3, t=0.4:
        # r = 100 - 3 * 0.1 * 2^2 * (1 - 0.4) = 100 - 0.72 = 99.28
        # delta_total = 0.1*4*0.6 + (2/0.1)*ln(1 + 0.1/1.5)
        m = maker()
        r = m.reservation_price(100.0, 3, 0.4)
        assert r == pytest.approx(99.28, rel=1e-12)
        delta_total = m.total_spread(0.4)
        expected = 0.1 * 4.0 * 0.6 + 20.0 * math.log(1.0 + 0.1 / 1.5)
        assert delta_total == pytest.approx(expected, rel=1e-12)

        quotes = m.quotes(100.0, 3, 0.4)
        assert quotes['bid'] == pytest.approx(r - expected / 2, rel=1e-12)
        assert quotes['ask'] == pytest.approx(r + expected / 2, rel=1e-12)

    def test_negative_inventory_skews_quotes_up(self):
        m = maker()
        # Short inventory: r > s, the maker leans to buy back.
        assert m.reservation_price(100.0, -5, 0.0) > 100.0
        assert m.reservation_price(100.0, 5, 0.0) < 100.0

    def test_zero_inventory_quotes_symmetric_around_mid(self):
        m = maker()
        for t in (0.0, 0.33, 0.9):
            quotes = m.quotes(100.0, 0, t)
            assert quotes['ask'] - 100.0 == pytest.approx(
                100.0 - quotes['bid'], abs=1e-12)

    def test_spread_tightens_as_the_horizon_ends(self):
        m = maker()
        # The inventory-risk term gamma*sigma^2*(T-t) decays to leave
        # only the (2/gamma)*ln(1+gamma/k) liquidity term at t = T.
        assert m.total_spread(0.0) > m.total_spread(0.9)
        assert m.total_spread(T) == pytest.approx(
            (2.0 / GAMMA) * math.log(1.0 + GAMMA / K), rel=1e-12)


class TestPnlIdentity:
    @pytest.mark.parametrize('naive', [False, True])
    def test_identity_exact_every_step_via_mirror(self, naive):
        # Mirror the simulation against the SAME seeded LOB (identical
        # call order -> identical random stream) and assert
        # cash + q*mid == spread_pnl + inventory_pnl at EVERY step.
        m = maker()
        seed, n_steps = 11, 150
        lob = SyntheticLOB(s0=100.0, sigma=SIGMA, T=T, n_steps=n_steps,
                           A=A, k=K, seed=seed)
        cash = inventory = 0
        spread_pnl = inventory_pnl = 0.0
        for step in range(n_steps):
            t = step * lob.dt
            mid = lob.mid
            q = 0.0 if naive else inventory
            quotes = m.quotes(mid, q, t)
            if lob.fill_side(quotes['ask'] - mid):
                cash += quotes['ask']
                inventory -= 1
                spread_pnl += quotes['ask'] - mid
            if lob.fill_side(mid - quotes['bid']):
                cash -= quotes['bid']
                inventory += 1
                spread_pnl += mid - quotes['bid']
            new_mid = lob.step_mid()
            inventory_pnl += inventory * (new_mid - mid)
            # THE IDENTITY, every step, exact to float accumulation.
            assert cash + inventory * new_mid == pytest.approx(
                spread_pnl + inventory_pnl, abs=1e-9)

        # The mirror must agree with run()/run_naive() exactly.
        result = (m.run_naive(seed=seed, n_steps=n_steps) if naive
                  else m.run(seed=seed, n_steps=n_steps))
        assert result['final_pnl'] == pytest.approx(
            cash + inventory * lob.mid, abs=1e-9)
        assert result['final_inventory'] == inventory
        assert result['final_pnl'] == pytest.approx(
            result['spread_pnl'] + result['inventory_pnl'], abs=1e-9)

    @pytest.mark.parametrize('seed', [1, 7, 42])
    def test_decomposition_sums_to_final_pnl(self, seed):
        result = maker().run(seed=seed, n_steps=250)
        assert result['final_pnl'] == pytest.approx(
            result['spread_pnl'] + result['inventory_pnl'], abs=1e-9)


class TestDeterminismAndShape:
    EXPECTED_KEYS = {'final_pnl', 'spread_pnl', 'inventory_pnl',
                     'max_abs_inventory', 'inventory_std', 'n_fills',
                     'final_inventory'}

    def test_seeded_runs_are_identical(self):
        first = maker().run(seed=42, n_steps=200)
        second = maker().run(seed=42, n_steps=200)
        assert first == second
        assert set(first) == self.EXPECTED_KEYS

    def test_different_seeds_differ(self):
        assert maker().run(seed=1) != maker().run(seed=2)

    def test_run_demo_returns_both_arms_without_printing(self, capsys):
        demo = run_demo(seed=5, n_steps=100)
        assert set(demo) == {'avellaneda_stoikov', 'naive_symmetric',
                             'params'}
        assert set(demo['avellaneda_stoikov']) == self.EXPECTED_KEYS
        assert capsys.readouterr().out == ''


class TestInventoryControl:
    def test_as_controls_inventory_better_than_naive_over_20_seeds(self):
        # THE claim: reservation-price skew actively mean-reverts
        # inventory; a fixed symmetric quoter random-walks it. Compare
        # mean inventory_std over 20 seeds — statistically robust, not
        # a single lucky draw.
        m = maker()
        as_stds = [m.run(seed=s, n_steps=250)['inventory_std']
                   for s in range(20)]
        naive_stds = [m.run_naive(seed=s, n_steps=250)['inventory_std']
                      for s in range(20)]
        assert sum(as_stds) / 20 < sum(naive_stds) / 20
