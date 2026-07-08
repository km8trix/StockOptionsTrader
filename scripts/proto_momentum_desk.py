#!/usr/bin/env python
"""Prototype: a CLEAN 12-1 momentum cross-sectional long/short desk.

The signal-IC layer (scripts/signal_ic.py) showed 12-1 momentum is the one
price signal with real, low-turnover edge (t-stat ~2.7), while the desks dilute
it with anti-predictive factors (low-vol, short-momentum reversal). This builds
the minimal desk on the good signal alone: rank the universe by 12-1 momentum
(return from ~12 months ago to ~1 month ago, the standard skip), long the top
quintile, short the bottom, dollar-balanced — NO other factors.

It reuses the shared CrossSectionalLongShortDesk machinery (same as AQR/Two
Sigma) and the desk_backtest harness's metrics, then prints the OOS-gated
verdict. Standalone on purpose: not registered until it earns it.

    python scripts/proto_momentum_desk.py            # full window, 100 names
    python scripts/proto_momentum_desk.py --limit 30
    python scripts/proto_momentum_desk.py --selftest # offline, asserts the signal
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


class MomentumModel(WalkForwardModel):
    """Parameter-free 12-1 momentum scorer (no training needed).

    Score per symbol = close[-skip] / close[-lookback] - 1, the standard
    12-month-minus-1-month momentum (skip the most recent month to avoid the
    short-horizon reversal the IC study flagged as anti-predictive). Positive =
    uptrend = long signal. fit() is a no-op; the controller still 'fits' once so
    predict() turns on after `lookback` days of history exist.
    """

    def __init__(self, lookback: int = 252, skip: int = 21,
                 residual: bool = False):
        self.lookback = lookback
        self.skip = skip
        # Residual (idiosyncratic) momentum (Blitz-Huij-Martens): rank on the
        # part of momentum NOT explained by the stock's market beta. Strips the
        # systematic component that drives momentum crashes -> historically a
        # much higher Sharpe and lower crash risk than total-return momentum.
        self.residual = residual

    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        return None  # parameter-free

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        closes: Dict[str, pd.Series] = {}
        for symbol, frame in data.items():
            if frame is None or 'close' not in frame.columns:
                continue
            close = frame['close'].dropna()
            if len(close) >= self.lookback:
                closes[symbol] = close
        mom = {s: float(c.iloc[-self.skip]) / float(c.iloc[-self.lookback]) - 1.0
               for s, c in closes.items() if float(c.iloc[-self.lookback]) > 0}
        if not self.residual or not mom:
            return mom
        # Beta-adjust: resid_i = mom_i - beta_i * market_momentum, where beta_i
        # is the trailing-window regression of the stock's daily return on the
        # equal-weight market return (vectorized over the cross-section).
        rets = pd.DataFrame({s: c.pct_change() for s, c in closes.items()}
                            ).tail(self.lookback)
        market = rets.mean(axis=1)
        var_m = float(market.var())
        if var_m <= 0:
            return mom
        cross = rets.mul(market, axis=0).mean()
        betas = (cross - rets.mean() * float(market.mean())) / var_m
        mom_mkt = float(np.mean(list(mom.values())))
        return {s: mom[s] - float(betas.get(s, 1.0)) * mom_mkt for s in mom}


class MomentumDesk(CrossSectionalLongShortDesk):
    """Cross-sectional L/S on 12-1 momentum alone (mirrors AqrDesk wiring)."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager=None, lookback: int = 252, skip: int = 21,
                 quantile: float = 0.2, exit_quantile: Optional[float] = 0.3,
                 min_holding_days: int = 3,
                 vol_target: bool = False, target_vol: float = 0.15,
                 gross_scale_min: float = 0.2, vol_window: int = 21,
                 crash_filter: bool = True, crash_target_vol: float = 0.18,
                 crash_window: int = 126, crash_min_obs: int = 42,
                 crash_cap: float = 1.0, residual: bool = False):
        label = 'resid_mom_12_1' if residual else 'mom_12_1'
        committee = [(label, WalkForwardController(
            MomentumModel(lookback, skip, residual=residual),
            min_train_days=lookback))]
        super().__init__(
            key='momentum', name='Momentum Desk',
            description=('Cross-sectional long/short on 12-1 price momentum '
                         'alone — long the top quintile, short the bottom, '
                         'dollar-balanced, with a volatility-targeting crash '
                         'overlay.'),
            accent='#e3b341', note_label='MOM', reason_prefix='mom',
            committee=committee, model_label=label,
            capital_allocation=capital_allocation, risk_manager=risk_manager,
            quantile=quantile, target_gross=1.0, max_name_size=0.10,
            min_scored=4, exit_quantile=exit_quantile,
            min_holding_days=min_holding_days, size_by_signal_strength=True)
        # Vol-targeting crash overlay: scale gross DOWN when trailing market
        # vol exceeds target_vol (one-sided, capped at 1.0 — never lever up).
        # TESTED AND REJECTED (default off): on this universe it HURT
        # (+44.9% -> +16.0%, Sharpe 0.20 -> 0.00, max DD -18.5% -> -26.0%).
        # Trailing realized vol LAGS, so it de-grosses after the crash into the
        # rebound where momentum recovers (2020/2021 flipped negative). A proper
        # crash filter needs the strategy's OWN forecast vol (Barroso-Santa-
        # Clara) or a market-state signal, not lagging market vol — future work.
        self.vol_target = vol_target
        self.target_vol = target_vol
        self.gross_scale_min = gross_scale_min
        self.vol_window = vol_window
        # Barroso-Santa-Clara crash filter (default ON): scale gross by
        # crash_target_vol / forecast_vol, where forecast_vol is the trailing
        # crash_window-day annualized vol of the STRATEGY'S OWN daily returns
        # (read from the portfolio equity curve). Unlike market vol, the
        # strategy's own vol LEADS momentum crashes (the book whipsaws before
        # it craters), so this cuts exposure into 2018-Q4 without de-grossing
        # through the 2020/2021 rebounds where momentum kept working.
        self.crash_filter = crash_filter
        self.crash_target_vol = crash_target_vol
        self.crash_window = crash_window
        self.crash_min_obs = crash_min_obs
        self.crash_cap = crash_cap
        self._equity: list = []

    def _alpha_scores(self, all_data, date):
        return self._committee[0][1].predict(all_data, date)

    def _strategy_gross_scale(self, portfolio) -> float:
        """Gross scale from the strategy's OWN trailing realized vol."""
        self._equity.append(float(portfolio.get_portfolio_value()))
        if len(self._equity) < self.crash_min_obs + 1:
            return 1.0
        eq = np.array(self._equity[-(self.crash_window + 1):], dtype=float)
        rets = eq[1:] / eq[:-1] - 1.0
        sd = float(np.std(rets))
        if sd <= 0:
            return 1.0
        forecast_vol = sd * (252 ** 0.5)
        return min(self.crash_cap,
                   max(self.gross_scale_min, self.crash_target_vol / forecast_vol))

    def _market_vol(self, all_data, date) -> Optional[float]:
        """Annualized trailing vol of the equal-weight universe daily return."""
        cols = []
        for frame in all_data.values():
            if frame is None or 'close' not in frame.columns:
                continue
            c = frame['close'].dropna()
            c = c[c.index <= pd.Timestamp(date)]
            if len(c) >= self.vol_window + 1:
                cols.append(c.pct_change().iloc[-self.vol_window:])
        if not cols:
            return None
        mkt = pd.concat(cols, axis=1).mean(axis=1)
        sd = float(mkt.std())
        return sd * (252 ** 0.5) if sd > 0 else None

    def _gross_scale(self, all_data, date) -> float:
        mv = self._market_vol(all_data, date)
        if mv is None or mv <= 0:
            return 1.0
        return min(1.0, max(self.gross_scale_min, self.target_vol / mv))

    def generate_intents(self, all_data, date, portfolio):
        if self.crash_filter:
            scale = self._strategy_gross_scale(portfolio)
        elif self.vol_target:
            scale = self._gross_scale(all_data, date)
        else:
            scale = 1.0
        if scale >= 1.0:
            return super().generate_intents(all_data, date, portfolio)
        base = self.target_gross
        self.target_gross = base * scale
        try:
            return super().generate_intents(all_data, date, portfolio)
        finally:
            self.target_gross = base


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
        'momentum', None, symbols, summary, closed, regimes,
        report.get('oos_folds', {'available': False}), start, end, t0)


def _selftest():
    """A clean uptrend scores positive 12-1 momentum; a downtrend negative."""
    idx = pd.date_range('2018-01-01', periods=300, freq='B')
    up = pd.DataFrame({'close': [100 + i for i in range(300)]}, index=idx)
    down = pd.DataFrame({'close': [400 - i for i in range(300)]}, index=idx)
    m = MomentumModel()
    s = m.predict({'UP': up, 'DOWN': down}, idx[-1])
    assert s['UP'] > 0 > s['DOWN'], s
    # Too-short history -> no score.
    short = pd.DataFrame({'close': [100] * 100}, index=idx[:100])
    assert m.predict({'X': short}, idx[99]) == {}
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--residual', action='store_true',
                    help='rank on residual (beta-adjusted) momentum')
    ap.add_argument('--symbols-file', default=None,
                    help='newline-separated tickers (e.g. sp500.txt) instead of '
                         'the built-in LARGE_CAP_100 universe')
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
    result = run(MomentumDesk(residual=args.residual), symbols, args.start, args.end)
    print_result(result)
    save_result(result)


if __name__ == '__main__':
    main()
