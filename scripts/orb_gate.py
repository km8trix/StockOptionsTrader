#!/usr/bin/env python
"""ORB (5-minute Opening Range Breakout) gate — standalone minute-loop study.

docs/PLANNING.md ("The two starter strategies") specs ORB and warns the
published Zarattini-Aziz result is "regime- and cost-sensitive. Validate,
don't assume." This is that validation: a STANDALONE per-session simulator
(deliberately NOT BacktestEngine — ORB is flat by close, so each session
collapses to one daily return and the existing daily-series gate apparatus
applies unchanged). Data: Alpaca IEX 1-minute bars from the PitWarehouse
(scripts/ingest_alpaca_bars.py must have run first).

PRE-REGISTERED SPEC (Zarattini-Aziz-style, single config, fixed BEFORE any
run — this is the FIRST and ONLY registered ORB config; NO parameter sweeps
in this round, so n_trials=1 and DSR == PSR. AMENDED 2026-07-10 after
adversarial code review, still pre-registration: NO ORB backtest had run
yet, so the session-end detection rule and the full-OR requirement below
are spec amendments made before the first data run, not post-hoc tweaks):

  * Opening range : high/low of the 09:30-09:34 ET minute bars (five 1-min
                    bars; labels are bar START times; all times tz-naive ET).
                    No trade unless ALL FIVE bars are present
                    (OR_REQUIRED_BARS; amended 2026-07-10 — a data-quality
                    guard: a 1-2 bar OR on a thin IEX open understates the
                    range and oversizes the position). No trade on zero
                    range.
  * Session end   : DATA-DRIVEN early-close detection (amended 2026-07-10),
                    no memorized holiday list: a session with fewer than
                    HALF_DAY_MIN_PROBE_BARS (=30, a data-quality constant)
                    bars with ts in [13:15, 15:45) ends at 13:00 (early
                    close), else 16:00. Rationale: on regular days liquid
                    ETFs print >100 IEX bars in that window; NYSE
                    13:00-close half days (~9/yr) show only sparse
                    after-hours prints there. Without this rule the
                    [09:30,16:00) filter admits AH IEX prints on half days
                    -> phantom entries/stops/EOD exits. The cutoff bounds
                    the tradeable window (entries only through the bar
                    starting 6 min before the cutoff — 12:54, mirroring the
                    15:54 rule), the stop scan, and the EOD flat exit.
  * Entry         : scanning the 09:35 bar through the 15:54 bar (12:54 on
                    detected half days), the FIRST
                    bar that trades THROUGH the range — high > OR-high =>
                    long, low < OR-low => short. First breakout wins; ONE
                    trade per day max. If that first breaking bar breaches
                    BOTH sides, the intrabar path is ambiguous -> no trade.
                    FILL MODEL (pre-registered choice, documented): entry at
                    the breakout BAR'S CLOSE plus adverse slippage — the
                    first trade THROUGH the level, never the level itself.
  * Stop          : the opposite side of the opening range, checked on bars
                    AFTER the entry bar (session bars only — never
                    after-hours prints): filled at the bar's OPEN when it
                    gaps through the stop, else at the stop level; adverse
                    slippage either way.
  * Exit          : stop, else flat at the close of the session's LAST bar
                    strictly before the session cutoff — 16:00, or 13:00 on
                    detected early-close days (the 15:59 bar on full days) —
                    the close exit. No overnight positions, ever.
  * Sizing        : risk 1% of current equity on the OR range — shares =
                    0.01*equity / (ORhigh - ORlow) — with notional capped at
                    4x equity at the entry fill. Fractional shares (research
                    gate, not an OMS).
  * Costs         : $0.0035/share commission per side + $0.01/share
                    half-spread slippage per side, applied adversely on both
                    entry and exit (the Z-A-style retail-realistic setup).
  * Data caveat   : IEX-only prints (~2-3% of tape) understate consolidated
                    volume and can understate intraday range — breakouts
                    trigger later/less often than on SIP. Documented, not
                    corrected.
  * Daily series  : one return per session day, pnl / equity-at-open; 0.0 on
                    no-trade days (the strategy IS flat those days); equity
                    compounds per symbol from $100k.
  * GATE          : the EQUAL-WEIGHT portfolio across SPY/QQQ/IWM/TQQQ
                    (per-date mean of the symbols with data that day),
                    LONG+SHORT combined, fed to
                    analysis.research_stats.validate_strategy_oos with
                    psr_threshold=0.95 and calendar-year period labels
                    (>= 1 BH-significant OOS year required). The LONG-ONLY
                    cut and per-symbol cuts are REPORTED, not gated — both
                    cuts pre-registered as reported here.

    python scripts/orb_gate.py --start 2016-01-01 --end 2024-12-31
    python scripts/orb_gate.py --selftest       # offline, pinned PnL asserts
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.research_stats import validate_strategy_oos  # noqa: E402
from data.pit_warehouse import PitWarehouse  # noqa: E402

PILOT_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TQQQ']
FULL_START, FULL_END = '2016-01-01', '2024-12-31'

# --- pre-registered constants (see module docstring) -----------------------
OR_START = dtime(9, 30)
OR_END = dtime(9, 35)        # exclusive — OR bars are 09:30..09:34
OR_REQUIRED_BARS = 5         # data quality: ALL FIVE OR bars, else no trade
ENTRY_END = dtime(15, 55)    # exclusive — no fresh entries 15:55 onward
SESSION_END = dtime(16, 0)   # exclusive — regular session bars only
# Early-close (half-day) detection — data-driven, amended 2026-07-10:
HALF_DAY_PROBE_START = dtime(13, 15)   # probe window: did the afternoon
HALF_DAY_PROBE_END = dtime(15, 45)     # session actually trade? (exclusive)
HALF_DAY_MIN_PROBE_BARS = 30           # data quality: fewer bars than this
#                                        in the probe window => early close
HALF_DAY_SESSION_END = dtime(13, 0)    # exclusive — half-day session bars
HALF_DAY_ENTRY_END = dtime(12, 55)     # exclusive — last entry bar 12:54,
#                                        mirroring the 15:54 rule
COMMISSION = 0.0035          # $/share per side
SLIPPAGE = 0.01              # $/share per side, adverse (half-spread)
RISK_FRAC = 0.01             # fraction of equity risked on the OR range
MAX_LEVERAGE = 4.0           # notional cap at entry fill
INITIAL_CAPITAL = 100_000.0
PSR_THRESHOLD = 0.95


@dataclass
class Trade:
    """One completed ORB day-trade (fills are cost-adjusted, slippage baked
    in; ``pnl`` is net of both sides' commissions)."""
    date: object
    direction: int           # +1 long, -1 short
    entry_ts: pd.Timestamp
    entry_price: float       # fill (breakout bar close +/- slippage)
    exit_ts: pd.Timestamp
    exit_price: float        # fill (stop/open/close +/- slippage)
    shares: float
    pnl: float
    stopped: bool


def simulate_session(day: pd.DataFrame, equity: float, *,
                     long_only: bool = False) -> Optional[Trade]:
    """Run the pre-registered ORB rules over ONE session's minute bars.

    ``day``: frame indexed by tz-naive ET minute timestamps (one calendar
    day, ascending) with columns open/high/low/close. Returns the Trade or
    None (no-trade day). Pure function of (bars, equity) — no I/O.
    """
    times = day.index.time
    # Data-driven session end (amended 2026-07-10, pre-registered before any
    # run): a session with < HALF_DAY_MIN_PROBE_BARS bars in the probe window
    # is a 13:00 early close — its post-13:00 prints are after-hours, and
    # admitting them would fabricate phantom entries/stops/EOD exits.
    n_probe = int(((times >= HALF_DAY_PROBE_START)
                   & (times < HALF_DAY_PROBE_END)).sum())
    if n_probe < HALF_DAY_MIN_PROBE_BARS:
        session_end, entry_end = HALF_DAY_SESSION_END, HALF_DAY_ENTRY_END
    else:
        session_end, entry_end = SESSION_END, ENTRY_END
    rth = day[(times >= OR_START) & (times < session_end)]
    if rth.empty:
        return None
    rt = rth.index.time
    opens = rth['open'].to_numpy(dtype=float)
    highs = rth['high'].to_numpy(dtype=float)
    lows = rth['low'].to_numpy(dtype=float)
    closes = rth['close'].to_numpy(dtype=float)

    or_mask = rt < OR_END
    if int(or_mask.sum()) < OR_REQUIRED_BARS:
        return None                              # partial OR — no trade
    or_high = float(highs[or_mask].max())
    or_low = float(lows[or_mask].min())
    or_range = or_high - or_low
    if or_range <= 0:
        return None

    scan = (rt >= OR_END) & (rt < entry_end)
    up = scan & (highs > or_high)
    dn = scan & (lows < or_low)
    breaking = up | dn
    if not breaking.any():
        return None
    e = int(np.argmax(breaking))                 # first breaking bar
    if up[e] and dn[e]:
        return None                              # ambiguous intrabar path
    direction = 1 if up[e] else -1
    if long_only and direction < 0:
        return None

    entry_fill = closes[e] + SLIPPAGE * direction
    if entry_fill <= 0:
        return None
    stop = or_low if direction == 1 else or_high
    shares = min(RISK_FRAC * equity / or_range,
                 MAX_LEVERAGE * equity / entry_fill)

    # Stop scan on bars AFTER the entry bar: gap-through at the open wins,
    # else intrabar touch fills at the stop level. Adverse slippage.
    stopped = False
    if direction == 1:
        gap = opens[e + 1:] <= stop
        hit = lows[e + 1:] <= stop
    else:
        gap = opens[e + 1:] >= stop
        hit = highs[e + 1:] >= stop
    trig = gap | hit
    if trig.any():
        j = int(np.argmax(trig))
        x = e + 1 + j
        raw = opens[x] if gap[j] else stop
        exit_fill = raw - SLIPPAGE * direction
        exit_ts = rth.index[x]
        stopped = True
    else:
        exit_fill = closes[-1] - SLIPPAGE * direction   # EOD flat
        exit_ts = rth.index[-1]

    pnl = shares * (exit_fill - entry_fill) * direction \
        - 2 * COMMISSION * shares
    return Trade(date=rth.index[0].date(), direction=direction,
                 entry_ts=rth.index[e], entry_price=float(entry_fill),
                 exit_ts=exit_ts, exit_price=float(exit_fill),
                 shares=float(shares), pnl=float(pnl), stopped=stopped)


def run_symbol(bars: pd.DataFrame, *, initial_capital: float = INITIAL_CAPITAL,
               long_only: bool = False
               ) -> Tuple[list, List[float], List[Trade], float]:
    """Minute frame (possibly years) -> (dates, daily returns, trades, equity).

    One return per session day present in the data (0.0 on no-trade days);
    equity compounds through the sessions in date order.
    """
    dates, returns, trades = [], [], []
    equity = initial_capital
    for day_key, day_df in bars.groupby(bars.index.normalize(), sort=True):
        tr = simulate_session(day_df, equity, long_only=long_only)
        pnl = tr.pnl if tr is not None else 0.0
        dates.append(day_key.date())
        returns.append(pnl / equity)
        if tr is not None:
            trades.append(tr)
        equity += pnl
    return dates, returns, trades, equity


def equal_weight(per_symbol: Dict[str, Tuple[list, List[float]]]
                 ) -> Tuple[list, List[float]]:
    """Per-date mean of the symbols that HAVE data that date (pre-registered
    portfolio construction) -> (sorted dates, portfolio returns)."""
    by_date: Dict = {}
    for _sym, (dates, rets) in per_symbol.items():
        for d, r in zip(dates, rets):
            by_date.setdefault(d, []).append(r)
    days = sorted(by_date)
    return days, [float(np.mean(by_date[d])) for d in days]


def trade_stats(trades: List[Trade], n_days: int) -> Dict:
    """Reported cuts: trades/year (252-day years), win rate, avg win/loss $."""
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    years = n_days / 252.0 if n_days else 0.0
    return {
        'n_trades': len(trades),
        'trades_per_year': (len(trades) / years) if years else 0.0,
        'win_rate': (len(wins) / len(trades)) if trades else 0.0,
        'avg_win': float(np.mean(wins)) if wins else 0.0,
        'avg_loss': float(np.mean(losses)) if losses else 0.0,
    }


def run_gate(symbols, start, end, *, warehouse_dir=None) -> Dict:
    """Load bars, run both pre-registered cuts, gate the combined portfolio."""
    wh = PitWarehouse(warehouse_dir)
    out: Dict = {'symbols': {}, 'long_only': {}}
    per_symbol, per_symbol_lo = {}, {}
    for sym in symbols:
        bars = wh.ohlcv_intraday_range(sym, start, end)
        if bars.empty:
            raise SystemExit(
                f"no bars_1m data for {sym} in {wh.warehouse_dir} — run "
                f"scripts/ingest_alpaca_bars.py first")
        dates, rets, trades, equity = run_symbol(bars)
        per_symbol[sym] = (dates, rets)
        out['symbols'][sym] = {
            'n_days': len(dates), 'final_equity': equity,
            'total_return_pct': (equity / INITIAL_CAPITAL - 1.0) * 100.0,
            **trade_stats(trades, len(dates)),
        }
        dates_lo, rets_lo, trades_lo, equity_lo = run_symbol(
            bars, long_only=True)
        per_symbol_lo[sym] = (dates_lo, rets_lo)
        out['long_only'][sym] = {
            'final_equity': equity_lo,
            'total_return_pct': (equity_lo / INITIAL_CAPITAL - 1.0) * 100.0,
            **trade_stats(trades_lo, len(dates_lo)),
        }
    days, port = equal_weight(per_symbol)
    days_lo, port_lo = equal_weight(per_symbol_lo)
    out['portfolio'] = {'dates': days, 'returns': port}
    out['portfolio_long_only'] = {'dates': days_lo, 'returns': port_lo}
    out['gate'] = validate_strategy_oos(
        port, [str(d.year) for d in days], psr_threshold=PSR_THRESHOLD)
    # Long-only lens: reported, NOT the gate verdict.
    out['gate_long_only_reported'] = validate_strategy_oos(
        port_lo, [str(d.year) for d in days_lo], psr_threshold=PSR_THRESHOLD)
    return out


def print_report(res: Dict, symbols, start, end) -> None:
    print(f"\n{'=' * 72}\nORB 5-min gate — pre-registered Z-A config "
          f"({start} .. {end}, {', '.join(symbols)})\n{'=' * 72}")
    print("  [IEX-only minute bars: volume/range understated vs SIP — "
          "see docstring]")
    hdr = (f"  {'symbol':<7}{'days':>6}{'trades':>8}{'tr/yr':>8}"
           f"{'win%':>7}{'avgW$':>10}{'avgL$':>10}{'ret%':>9}")
    for title, key in (('LONG+SHORT (gated)', 'symbols'),
                       ('LONG-ONLY (reported)', 'long_only')):
        print(f"\n  {title}\n{hdr}")
        for sym in symbols:
            s = res[key][sym]
            n_days = res['symbols'][sym]['n_days']
            print(f"  {sym:<7}{n_days:>6}{s['n_trades']:>8}"
                  f"{s['trades_per_year']:>8.1f}{s['win_rate']:>7.1%}"
                  f"{s['avg_win']:>10.0f}{s['avg_loss']:>10.0f}"
                  f"{s['total_return_pct']:>9.2f}")
    for title, gkey in (('COMBINED long+short portfolio', 'gate'),
                        ('long-only portfolio (reported, not gated)',
                         'gate_long_only_reported')):
        gate = res[gkey]
        psr = gate['psr']
        psr_str = f"{psr:.4f}" if psr is not None else "n/a"
        rf = gate.get('risk_free_rate', 0.02)
        print(f"\n  GATE [{title}]: PSR(excess vs {rf:.0%} risk-free) >= "
              f"{PSR_THRESHOLD} AND >= 1 BH-significant OOS year")
        print(f"    PSR (vs risk-free)  : {psr_str}")
        print(f"    PSR pass (>=0.95)   : {gate['psr_pass']}")
        print(f"    OOS years tested    : {gate['n_periods_tested']}")
        print(f"    BH-significant years: {gate['bh']['n_significant_bh']}")
        print(f"    fold pass (>=1)     : {gate['fold_pass']}")
        print(f"    {'year':<8}{'p-value':>10}{'reject':>9}")
        for label, p, rej in zip(gate['fold_labels'], gate['fold_pvalues'],
                                 gate['bh']['rejected_bh']):
            ps = f"{p:.4f}" if p is not None else "n/a"
            print(f"    {str(label):<8}{ps:>10}{str(rej):>9}")
    print(f"\n  VERDICT (combined long+short): "
          f"{'PASS' if res['gate']['passed'] else 'FAIL'}\n")


# ---------------------------------------------------------------------------
# Selftest — synthetic minute bars, hand-computed PnL pinned (offline)
# ---------------------------------------------------------------------------
def make_session(date_str: str, bars) -> pd.DataFrame:
    """Synthetic session helper: bars = [(``HH:MM``, o, h, l, c), ...] ->
    minute frame indexed by tz-naive ET timestamps."""
    idx = pd.DatetimeIndex([f"{date_str} {hm}" for hm, *_ in bars])
    o, h, low, c = zip(*[b[1:] for b in bars])
    return pd.DataFrame({'open': o, 'high': h, 'low': low, 'close': c},
                        index=idx)


def _or_block():
    """09:30-09:34 bars pinning OR-high=101.0 / OR-low=100.0 (range 1.0)."""
    return [('09:30', 100.5, 101.0, 100.0, 100.5),
            ('09:31', 100.5, 100.9, 100.1, 100.5),
            ('09:32', 100.5, 100.9, 100.1, 100.5),
            ('09:33', 100.5, 100.9, 100.1, 100.5),
            ('09:34', 100.5, 100.9, 100.1, 100.5)]


def _regular_afternoon():
    """30 in-range 13:15-13:44 bars — marks a synthetic session as a REGULAR
    day under the data-driven session-end rule; never breaks the _or_block
    range or its stops."""
    return [(f'13:{m}', 100.5, 100.5, 100.5, 100.5) for m in range(15, 45)]


def _selftest() -> None:
    """Planted breakout / stop-out / no-trade / half-day / partial-OR days;
    PnL hand-computed from the pre-registered fills and costs (equity $100k,
    shares 1000):
      breakout+EOD : entry 101.20+0.01, exit 102.00-0.01
                     -> 1000*0.78 - 7.00 = +773.00
      stop-out     : entry 101.21, stop fill 100.00-0.01
                     -> 1000*(-1.22) - 7.00 = -1227.00
      no-trade     : never trades through either side -> None
      half-day EOD : entry 101.21, exit at the 12:59 close 101.80-0.01
                     -> 1000*0.58 - 7.00 = +573.00 (AH prints ignored)
      half-day AH  : no break before 13:00; the AH print through OR-high
                     must NOT enter -> None
      partial OR   : only 4 of 5 OR bars -> None
    """
    eq = 100_000.0
    day_a = make_session('2021-01-04', _or_block() + [
        ('09:35', 100.5, 100.8, 100.2, 100.6),
        ('09:36', 100.7, 101.5, 100.6, 101.2),   # breaks OR-high -> long
        ('09:37', 101.2, 101.6, 100.8, 101.4)]
        + _regular_afternoon() + [
        ('15:59', 101.8, 102.1, 101.7, 102.0)])  # EOD close 102.00
    tr = simulate_session(day_a, eq)
    assert tr is not None and tr.direction == 1 and not tr.stopped, tr
    assert abs(tr.shares - 1000.0) < 1e-9, tr.shares
    assert abs(tr.pnl - 773.0) < 1e-6, tr.pnl

    day_b = make_session('2021-01-05', _or_block() + [
        ('09:36', 100.7, 101.5, 100.6, 101.2),   # long entry, fill 101.21
        ('09:40', 100.5, 100.6, 99.9, 100.0)]    # low<=100 -> stop fill 99.99
        + _regular_afternoon() + [
        ('15:59', 100.2, 100.3, 100.1, 100.2)])
    tr = simulate_session(day_b, eq)
    assert tr is not None and tr.stopped, tr
    assert abs(tr.pnl - (-1227.0)) < 1e-6, tr.pnl

    day_c = make_session('2021-01-06', _or_block() + [
        ('09:36', 100.5, 101.0, 100.0, 100.5)]   # touches, never THROUGH
        + _regular_afternoon() + [
        ('15:59', 100.5, 100.9, 100.1, 100.5)])
    assert simulate_session(day_c, eq) is None

    # Half day (sparse probe window): EOD exit at the last pre-13:00 bar;
    # the 13:30 AH print through the stop must NOT fill anything.
    day_d = make_session('2021-11-26', _or_block() + [
        ('09:40', 100.7, 101.5, 100.6, 101.2),   # real breakout -> long
        ('12:59', 101.5, 101.9, 101.4, 101.8),   # last bar before 13:00
        ('13:30', 99.0, 99.1, 98.5, 98.6)])      # AH print through the stop
    tr = simulate_session(day_d, eq)
    assert tr is not None and not tr.stopped, tr
    assert tr.exit_ts == pd.Timestamp('2021-11-26 12:59'), tr.exit_ts
    assert abs(tr.pnl - 573.0) < 1e-6, tr.pnl

    # Half day, no break before 13:00: the AH print through OR-high is a
    # phantom breakout and must be suppressed.
    day_e = make_session('2021-11-27', _or_block() + [
        ('12:59', 100.5, 100.8, 100.3, 100.6),
        ('13:30', 101.2, 101.6, 101.1, 101.5)])  # through OR-high, but AH
    assert simulate_session(day_e, eq) is None

    day_f = make_session('2021-01-08', _or_block()[:4] + [
        ('09:36', 100.7, 101.5, 100.6, 101.2)]   # would break, but 4-bar OR
        + _regular_afternoon() + [
        ('15:59', 101.8, 102.1, 101.7, 102.0)])
    assert simulate_session(day_f, eq) is None
    print('selftest OK')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--symbols', nargs='+', default=PILOT_SYMBOLS)
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--dir', default=None,
                    help='warehouse dir (else PIT_WAREHOUSE_DIR resolution)')
    ap.add_argument('--selftest', action='store_true',
                    help='offline pinned-PnL asserts, then exit')
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    res = run_gate(args.symbols, args.start, args.end,
                   warehouse_dir=args.dir)
    print_report(res, args.symbols, args.start, args.end)


if __name__ == '__main__':
    main()
