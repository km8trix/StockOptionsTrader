#!/usr/bin/env python
"""Prototype: a hybrid 12-1 momentum + 5-day reversal cross-sectional L/S desk.

The signal-IC layer (scripts/signal_ic.py) found TWO price signals with real,
significant edge on this universe: 12-1 momentum (`mom_12_1`, t +2.66 @1d, cheap
~12% turnover) and 5-day reversal (`reversal_5`, t +2.86 @5d, but ~79% turnover).
The momentum-only desk (scripts/proto_momentum_desk.py) clears the IC gate but
fails the OOS gate (Sharpe 0.20) — the edge is real but too thin alone.

This combines BOTH good signals to raise edge DENSITY (the evidence-first
hypothesis): they are economically complementary — "buy the recent dip within a
12-month uptrend." The two signals use the EXACT definitions signal_ic.py
validated (same `.shift()` math), are standardized cross-sectionally per date,
and combined equal-weight (parameter-free — no in-sample weight fitting). The
desk reuses the shared CrossSectionalLongShortDesk machinery and the
desk_backtest harness, so the ONLY change vs the momentum desk is the signal.

Costs are modeled by the backtest engine (slippage_bps=5 + commission); the
verdict line is OOS-gated. Reversal is turnover-heavy, so watch the trade count
vs the momentum desk's 4236 — the whole question is whether the blend survives
costs. Standalone on purpose: not registered until it earns it.

RESULT (2026-06-29, 2015-2024 / 100 names) — TESTED AND REJECTED. Adding
reversal MONOTONICALLY destroys the desk (clean dose-response in w_rev):
    w_rev=0.0 (momentum-only):  4236 trades, +46.0%, Sharpe  0.20
    w_rev=0.5 (reversal-light): 17456 trades, -59.6%, Sharpe -1.20
    w_rev=1.0 (equal-weight):   25013 trades, -75.5%, Sharpe -2.15
reversal_5 has real cross-sectional edge (IC t +2.86 @5d) but is UN-TRADEABLE in
this daily-rebalanced desk: its ~79% turnover (per the IC study) explodes to
17-25k trades and slippage+commission obliterate the alpha. The win rate rises
(24% -> 45%, i.e. more small mean-reversion wins) yet the book loses in EVERY
regime — the cost-bleed signature, not a sign error (the reversal sign matches
signal_ic.py exactly; a flip would LOWER the win rate). Capturing reversal would
need a fundamentally lower-turnover treatment (match the desk cadence to the 5d
signal horizon, or an explicit turnover penalty / no-trade band), or it is
simply not viable after retail costs. Momentum-only remains the best desk and
still fails the OOS gate (Sharpe 0.20). The w_mom/w_rev knobs stay for research.

    python scripts/proto_hybrid_desk.py            # full window, 100 names
    python scripts/proto_hybrid_desk.py --limit 30
    python scripts/proto_hybrid_desk.py --w-rev 0.5  # tilt the reversal weight
    python scripts/proto_hybrid_desk.py --selftest # offline, asserts the signal
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling import

from backtesting.backtest_engine import BacktestEngine  # noqa: E402
from data.universe import LARGE_CAP_100  # noqa: E402
from desks.cross_sectional import CrossSectionalLongShortDesk  # noqa: E402
from desks.walk_forward import WalkForwardController, WalkForwardModel  # noqa: E402
from desk_backtest import (FULL_END, FULL_START, REGIMES, _slice_metrics,  # noqa: E402
                           _trades_in, build_result_dict, print_result,
                           save_result)


class HybridModel(WalkForwardModel):
    """Equal-weight blend of standardized 12-1 momentum and 5-day reversal.

    Both factors use the SAME formulas signal_ic.py validated:
      mom_12_1   = close.shift(skip) / close.shift(lookback) - 1   (long winners)
      reversal_5 = -(close / close.shift(rev_window) - 1)          (long recent losers)
    Each is z-scored across the cross-section (so the two live on a comparable
    scale), then summed `w_mom * z_mom + w_rev * z_rev`. Parameter-free: fit() is
    a no-op; the weights are fixed, not learned (equal-weight by default — no
    in-sample fitting). A name is scored only if BOTH factors are computable.
    """

    def __init__(self, lookback: int = 252, skip: int = 21, rev_window: int = 5,
                 w_mom: float = 1.0, w_rev: float = 1.0):
        self.lookback = lookback
        self.skip = skip
        self.rev_window = rev_window
        self.w_mom = w_mom
        self.w_rev = w_rev

    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        return None  # parameter-free

    @staticmethod
    def _zscore(d: Dict[str, float]) -> Dict[str, float]:
        """Cross-sectional z-score; all-equal -> zeros (degenerate, no tilt)."""
        vals = np.array(list(d.values()), dtype=float)
        mu = float(vals.mean())
        sd = float(vals.std())
        if sd <= 0:
            return {k: 0.0 for k in d}
        return {k: (v - mu) / sd for k, v in d.items()}

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        mom: Dict[str, float] = {}
        rev: Dict[str, float] = {}
        for symbol, frame in data.items():
            if frame is None or 'close' not in frame.columns:
                continue
            close = frame['close'].dropna()
            if len(close) <= self.lookback:  # need shift(lookback) at the last row
                continue
            # Byte-identical to signal_ic.py's signal definitions, taken at the
            # latest (simulation) date. The walk-forward controller has already
            # sliced `close` to index <= date, so iloc[-1] is the close AT date.
            m = (close.shift(self.skip) / close.shift(self.lookback) - 1.0).iloc[-1]
            r = (-(close / close.shift(self.rev_window) - 1.0)).iloc[-1]
            if pd.notna(m):
                mom[symbol] = float(m)
            if pd.notna(r):
                rev[symbol] = float(r)
        common = [s for s in mom if s in rev]
        if len(common) < 2:
            return {}
        z_mom = self._zscore({s: mom[s] for s in common})
        z_rev = self._zscore({s: rev[s] for s in common})
        return {s: self.w_mom * z_mom[s] + self.w_rev * z_rev[s] for s in common}


class HybridDesk(CrossSectionalLongShortDesk):
    """Cross-sectional L/S on the standardized mom_12_1 + reversal_5 blend.

    Mirrors MomentumDesk/AqrDesk wiring (top/bottom quintile, dollar-balanced,
    conviction sizing, turnover hysteresis). No crash overlay — this is a clean
    read on whether the blended SIGNAL has more edge density than momentum alone.
    """

    def __init__(self, capital_allocation: float = 1.0, risk_manager=None,
                 lookback: int = 252, skip: int = 21, rev_window: int = 5,
                 w_mom: float = 1.0, w_rev: float = 1.0,
                 quantile: float = 0.2, exit_quantile: Optional[float] = 0.3,
                 min_holding_days: int = 3):
        committee = [('mom12_1+rev5', WalkForwardController(
            HybridModel(lookback, skip, rev_window, w_mom, w_rev),
            min_train_days=lookback + 1))]
        super().__init__(
            key='hybrid', name='Hybrid Desk',
            description=('Cross-sectional long/short on a standardized blend of '
                         '12-1 momentum and 5-day reversal — the two signals '
                         'with validated cross-sectional edge — long the top '
                         'quintile, short the bottom, dollar-balanced.'),
            accent='#e3b341', note_label='HYB', reason_prefix='hyb',
            committee=committee, model_label='mom12_1+rev5',
            capital_allocation=capital_allocation, risk_manager=risk_manager,
            quantile=quantile, target_gross=1.0, max_name_size=0.10,
            min_scored=4, exit_quantile=exit_quantile,
            min_holding_days=min_holding_days, size_by_signal_strength=True)

    def _alpha_scores(self, all_data, date):
        return self._committee[0][1].predict(all_data, date)


def run(desk, symbols, start, end, capital: float = 100_000.0) -> dict:
    engine = BacktestEngine(desk=desk, initial_capital=capital)
    t0 = time.time()
    report = engine.run(symbols, start, end)
    summary = report['summary']
    closed = report['closed_trades']
    regimes = {name: {**(_slice_metrics(report['portfolio_history'], s, e) or {}),
                      **_trades_in(closed, s, e)}
               for name, s, e in REGIMES}
    return build_result_dict(
        'hybrid', None, symbols, summary, closed, regimes,
        report.get('oos_folds', {'available': False}), start, end, t0)


def _selftest():
    """A 12-month winner that just dipped scores positive on the blend; a
    12-month loser that just popped scores negative."""
    idx = pd.date_range('2018-01-01', periods=300, freq='B')
    # WINNER: long uptrend, small 5-day dip at the end (+mom, +reversal).
    win = [100.0 + i * 0.5 for i in range(295)] + [247, 246, 245, 244, 243]
    # LOSER: long downtrend, small 5-day pop at the end (-mom, -reversal).
    los = [250.0 - i * 0.5 for i in range(295)] + [106, 107, 108, 109, 110]
    up = pd.DataFrame({'close': win}, index=idx)
    down = pd.DataFrame({'close': los}, index=idx)
    m = HybridModel()
    s = m.predict({'UP': up, 'DOWN': down}, idx[-1])
    assert s['UP'] > 0 > s['DOWN'], s
    # Too-short history -> name dropped (need > lookback obs).
    short = pd.DataFrame({'close': [100.0] * 100}, index=idx[:100])
    assert m.predict({'X': short, 'UP': up}, idx[-1]).get('X') is None
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--w-mom', type=float, default=1.0, help='momentum weight')
    ap.add_argument('--w-rev', type=float, default=1.0, help='reversal weight')
    ap.add_argument('--symbols-file', default=None,
                    help='newline-separated tickers instead of LARGE_CAP_100')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.symbols_file:
        with open(args.symbols_file) as fh:
            symbols = [s.strip().upper() for s in fh if s.strip()]
    else:
        symbols = list(LARGE_CAP_100)
    if args.limit:
        symbols = symbols[:args.limit]
    result = run(HybridDesk(w_mom=args.w_mom, w_rev=args.w_rev),
                 symbols, args.start, args.end)
    print_result(result)
    save_result(result)


if __name__ == '__main__':
    main()
