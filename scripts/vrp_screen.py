#!/usr/bin/env python
"""PRE-REGISTERED VRP existence measurement — short ATM straddles, monthly.

REGISTERED 2026-07-10, BEFORE ANY RUN. Single fixed config, no sweeps, and
explicitly NOT a promotion gate. Context: the janestreet premium desk's
documented ruin came from SYNTHETIC IV (Black-Scholes on realized vol — no
variance risk premium by construction, ~26% win rate). Whether REAL-MARKET
VRP is harvestable at retail costs has never been measured in this repo;
this screen measures its EXISTENCE on real (free-tier) option closes.

DESIGN — every choice below was fixed before the first run:
  * Universe: SPY, QQQ, IWM. Monthly selection dates = the first trading
    session of each month, 2024-08 .. 2026-06 (~23 per name), from the SIP
    minute table's session closes.
  * Position: SELL one ATM straddle per month — the call and put selected
    by scripts/ingest_massive_options.py (expiry nearest 30 DTE in [21, 45],
    ties earlier; strike nearest the selection close among strikes listing
    both legs, ties lower). Entry at BOTH legs' daily-bar CLOSE on the
    selection date; a month missing either leg's close print that day is
    DROPPED (counted). Hold to expiry, no management.
  * Settlement: intrinsic |S_T - K| where S_T is the underlying's SIP
    session close ON the expiry date; if the expiry date is not a trading
    session in the data (weekend-dated expiry, holiday), the LAST PRIOR
    session close — documented fallback, flagged per month — but only
    within MAX_SETTLE_GAP_DAYS = 4 calendar days (a weekend plus one
    holiday). A month whose last available SIP session is further than
    that before its expiry is UNSETTLED and dropped (counted) — never
    marked with a stale close.
  * COSTS (pre-registered, applied at entry):
      - commission: $0.65/contract per leg per side, ENTRY LEGS ONLY (2
        legs -> $1.30 per straddle). Expiry settlement is assumed
        cash/exercise WITHOUT commission — an approximation, documented.
      - spread haircut per leg = max($0.05, 5% of that leg's close),
        subtracted from premium collected. This compensates the mid=close
        APPROXIMATION: the free tier has NO quotes/NBBO, so the day's last
        trade stands in for mid and real half-spreads are UNKNOWN.
  * P&L per straddle = premium - haircut - intrinsic - commission, x100
    (one contract per leg).
  * VRP measurement (the direct IV-vs-realized comparison): entry
    BS-implied vol per leg by inverting the repo's own
    ``desks.options_pricing.black_scholes_price`` (brentq), r = DTB3
    (data/riskfree.py) as of the selection date, t = calendar DTE / 365;
    entry IV = mean of the two legs' IVs (both ATM, so they proxy the same
    point on the surface). Realized vol to expiry = close-to-close log
    returns of SIP session closes from selection through settlement,
    ddof=1, annualized sqrt(252).
  * Statistics, per underlying and pooled: n months, mean/median premium
    as % of the underlying, win rate, mean P&L, Newey-West (HAC) t-stat of
    the monthly P&L series with maxlags=NW_LAG=2 (pre-registered: 21-45
    calendar-day holds can overlap the following monthly observation),
    worst month, mean entry IV / realized vol / IV-RV gap. Pooled = the
    EQUAL-WEIGHT monthly average of P&L across names.

HONESTY — read before quoting any number:
  * ~23 monthly observations per name is an EXISTENCE measurement, not a
    gate. Nothing here feeds gate_status/_PROMOTED_DESKS; there is NO
    promotion pathway from this screen.
  * SPY/QQQ/IWM straddles are the SAME index-vol bet three times: the
    pooled series is an average of cross-sectionally correlated names, NOT
    3x independent samples.
  * close != mid: entry premia are last-trade prints, not quote mids. The
    pre-registered haircut is a guess at the spread cost, not a measurement.
  * Single pre-registered config. No parameter sweeps were run and none
    will be attributed to this screen.
  * IV inversion uses the repo's European, no-dividend Black-Scholes on
    AMERICAN options over dividend-paying ETFs (SPY/IWM ~1.2%/yr, ex-div
    dates often inside the hold): the no-div bias mostly cancels in the
    two-leg mean, but the put's early-exercise premium only biases IV
    HIGH — the reported mean IV and IV-RV gap carry a small (tenths of a
    vol point) UPWARD bias. P&L numbers are unaffected. Months where only
    one leg inverts are excluded from the IV/RV stat (iv_partial count
    reported), never substituted.

    python scripts/vrp_screen.py                    # after the ingest
    python scripts/vrp_screen.py --underlyings SPY
    python scripts/vrp_screen.py --selftest         # offline pinned asserts

RESULT (2026-07-10, first and only run of the registered config; 23 months
x 3 underlyings, 2024-08..2026-06, real Massive EOD chains): **VRP EXISTS
BUT IS ECONOMICALLY UNHARVESTABLE AT RETAIL COSTS on this window.** Mean
entry IV exceeded subsequent RV by just +1.0 to +2.2 vol points (pooled
+1.3) — real, small. Short-straddle P&L after the registered costs: SPY
-$32.54/mo (win 39%), QQQ +$62.43 (61%), IWM +$77.07 (61%), pooled
+$35.66/mo, NW t +0.13 (p 0.896) — statistically indistinguishable from
zero, with a single month (2026-04) costing -$3.6k pooled (the classic
short-vol tail). The janestreet premium family's burial is hereby
re-confirmed WITH REAL DATA: the synthetic-IV ruin was by construction,
but even real premia at ~1-2 vol points cannot clear retail spread+
commission costs at monthly ATM tenor. No paid deep-history options data
is justified by this measurement. Nothing promoted.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.riskfree import load_dtb3, rate_asof  # noqa: E402
from desks.options_pricing import black_scholes_price  # noqa: E402
from scripts.ingest_massive_options import (  # noqa: E402
    BARS_END, BARS_START, UNDERLYINGS, sip_session_closes)

#: Pre-registered cost model (module docstring) — fixed 2026-07-10.
COMMISSION_PER_CONTRACT = 0.65   # $/contract per leg per side, ENTRY only
SPREAD_HAIRCUT_MIN = 0.05        # $/share per leg at entry
SPREAD_HAIRCUT_PCT = 0.05        # 5% of the leg's close
MULT = 100                       # shares per contract
NW_LAG = 2                       # pre-registered HAC maxlags (monthly series)
MIN_RV_RETURNS = 5               # fewer close-to-close returns -> RV = None
MAX_SETTLE_GAP_DAYS = 4          # settle fallback reach: weekend + 1 holiday


# ---------------------------------------------------------------------------
# Pure math (hermetically tested; hand-computed pins in --selftest)
# ---------------------------------------------------------------------------
def straddle_pnl(call_close: float, put_close: float, strike: float,
                 settle_price: float) -> dict:
    """SHORT one ATM straddle at the entry closes, held to settlement.

    Per-SHARE fields (premium_gross, haircut, commission, intrinsic, pnl)
    plus ``pnl_dollars`` = MULT x pnl (one contract per leg). Costs are the
    pre-registered model: entry-leg commissions only + per-leg spread
    haircut (module docstring)."""
    premium = call_close + put_close
    haircut = (max(SPREAD_HAIRCUT_MIN, SPREAD_HAIRCUT_PCT * call_close)
               + max(SPREAD_HAIRCUT_MIN, SPREAD_HAIRCUT_PCT * put_close))
    commission = 2.0 * COMMISSION_PER_CONTRACT / MULT
    intrinsic = abs(settle_price - strike)
    pnl = premium - haircut - intrinsic - commission
    return {'premium_gross': premium, 'haircut': haircut,
            'commission': commission, 'intrinsic': intrinsic,
            'pnl_per_share': pnl, 'pnl_dollars': MULT * pnl}


def implied_vol(price: float, spot: float, strike: float, t_years: float,
                rate: float, right: str):
    """BS-implied vol by inverting the repo's own ``black_scholes_price``
    (brentq on vol in [1e-4, 5.0]). Returns None when the price sits outside
    the bracket's no-arbitrage range (e.g. a print below intrinsic) or
    t_years <= 0 — no vol explains such a price."""
    from scipy.optimize import brentq
    if t_years <= 0 or price <= 0 or spot <= 0 or strike <= 0:
        return None
    lo, hi = 1e-4, 5.0
    p_lo = black_scholes_price(spot, strike, t_years, lo, rate, right)
    p_hi = black_scholes_price(spot, strike, t_years, hi, rate, right)
    if not (p_lo <= price <= p_hi):
        return None

    def gap(v: float) -> float:
        return black_scholes_price(spot, strike, t_years, v, rate, right) - price

    return float(brentq(gap, lo, hi, xtol=1e-8))


def realized_vol(closes: pd.Series, start, end):
    """Annualized close-to-close realized vol over the sessions in
    [start, end]: log returns of the SIP session closes, ddof=1, sqrt(252).
    None with fewer than MIN_RV_RETURNS returns."""
    window = closes.loc[pd.Timestamp(start):pd.Timestamp(end)]
    rets = np.log(window / window.shift(1)).dropna()
    if len(rets) < MIN_RV_RETURNS:
        return None
    return float(rets.std(ddof=1) * np.sqrt(252))


def nw_tstat(series, lag: int = NW_LAG):
    """Newey-West (HAC) t and p of the mean of ``series`` — the same
    statsmodels construction as scripts/factor_screen.py. (None, None) for
    n < 3 (no meaningful HAC fit)."""
    import statsmodels.api as sm
    arr = np.asarray(list(series), float)
    if len(arr) < 3:
        return None, None
    fit = sm.OLS(arr, np.ones(len(arr))).fit(cov_type='HAC',
                                             cov_kwds={'maxlags': lag})
    return float(fit.tvalues[0]), float(fit.pvalues[0])


# ---------------------------------------------------------------------------
# Screen core
# ---------------------------------------------------------------------------
def screen_underlying(wh, sym: str, *, closes: pd.Series = None,
                      dtb3: pd.Series = None) -> dict:
    """Compute the monthly short-straddle rows for one underlying from the
    warehouse's option_bars_eod + SIP session closes. Returns
    {'underlying', 'months': [row...], 'skipped': {reason: n}} where each
    row carries month/strike/expiry/premium/premium_pct/entry_iv/rv/pnl/
    settle fields (module docstring conventions)."""
    ob = wh.option_bars_eod(sym)
    skipped = {'no_entry_close': 0, 'no_underlying_close': 0, 'unsettled': 0}
    if ob.empty:
        return {'underlying': sym, 'months': [], 'skipped': skipped}
    if closes is None:
        closes = sip_session_closes(wh, sym, BARS_START, BARS_END)
    if dtb3 is None:
        dtb3 = load_dtb3()
    if closes.empty:
        return {'underlying': sym, 'months': [], 'skipped': skipped}
    months = []
    for sel, grp in ob.groupby('selection_date'):
        sel_ts = pd.Timestamp(sel)
        strike = float(grp['strike'].iloc[0])
        expiry = pd.Timestamp(grp['expiry'].iloc[0])
        c_rows = grp[(grp['type'] == 'call') & (grp['ts'] == sel_ts)]
        p_rows = grp[(grp['type'] == 'put') & (grp['ts'] == sel_ts)]
        if c_rows.empty or p_rows.empty:
            skipped['no_entry_close'] += 1
            continue
        if sel_ts not in closes.index:
            skipped['no_underlying_close'] += 1
            continue
        settle_window = closes.loc[:expiry]
        settle_date = settle_window.index[-1] if len(settle_window) else None
        if (settle_date is None
                or (expiry - settle_date).days > MAX_SETTLE_GAP_DAYS):
            skipped['unsettled'] += 1       # data ends before the expiry
            continue
        c_close = float(c_rows['close'].iloc[0])
        p_close = float(p_rows['close'].iloc[0])
        s_entry = float(closes.loc[sel_ts])
        settle_price = float(settle_window.iloc[-1])
        fallback = settle_date != expiry
        res = straddle_pnl(c_close, p_close, strike, settle_price)
        t_years = (expiry.date() - sel_ts.date()).days / 365.0
        rf = rate_asof(dtb3, sel_ts)
        ivs = [v for v in (
            implied_vol(c_close, s_entry, strike, t_years, rf, 'call'),
            implied_vol(p_close, s_entry, strike, t_years, rf, 'put'))
            if v is not None]
        # Registered definition: mean of BOTH legs' IVs. A month where one
        # leg fails inversion (stale/below-intrinsic print) is flagged and
        # EXCLUDED from the IV/RV diagnostic (P&L keeps it) — a one-leg IV
        # silently substituting for the straddle IV would drift the VRP
        # stat off its registered definition.
        entry_iv = float(np.mean(ivs)) if len(ivs) == 2 else None
        iv_partial = len(ivs) == 1
        rv = realized_vol(closes, sel_ts, settle_date)
        months.append({
            'month': sel_ts, 'strike': strike, 'expiry': expiry,
            'settle_date': settle_date, 'settle_fallback': fallback,
            'iv_partial': iv_partial,
            'spot_entry': s_entry, 'call_close': c_close,
            'put_close': p_close,
            'premium': res['premium_gross'],
            'premium_pct': res['premium_gross'] / s_entry,
            'entry_iv': entry_iv, 'rv': rv,
            'iv_minus_rv': (entry_iv - rv)
            if entry_iv is not None and rv is not None else None,
            'pnl': res['pnl_dollars'],
        })
    return {'underlying': sym, 'months': months, 'skipped': skipped}


def _stats(months) -> dict:
    """Aggregate one monthly row list into the pre-registered statistics."""
    if not months:
        return {'n': 0}
    pnls = [m['pnl'] for m in months]
    prem = [m['premium_pct'] for m in months]
    gaps = [m['iv_minus_rv'] for m in months if m['iv_minus_rv'] is not None]
    ivs = [m['entry_iv'] for m in months if m['entry_iv'] is not None]
    rvs = [m['rv'] for m in months if m['rv'] is not None]
    t, p = nw_tstat(pnls)
    worst = min(months, key=lambda m: m['pnl'])
    return {
        'n': len(months),
        'iv_partial_months': sum(1 for m in months if m.get('iv_partial')),
        'mean_premium_pct': float(np.mean(prem)),
        'median_premium_pct': float(np.median(prem)),
        'win_rate': float(np.mean([x > 0 for x in pnls])),
        'mean_pnl': float(np.mean(pnls)),
        'nw_t': t, 'nw_p': p,
        'worst_pnl': float(worst['pnl']),
        'worst_month': worst['month'],
        'mean_iv': float(np.mean(ivs)) if ivs else None,
        'mean_rv': float(np.mean(rvs)) if rvs else None,
        'mean_iv_minus_rv': float(np.mean(gaps)) if gaps else None,
        'n_fallback_settles': sum(1 for m in months if m['settle_fallback']),
    }


def pooled_monthly(results) -> list:
    """EQUAL-WEIGHT monthly average across names: for each calendar month
    (period of the selection date), the mean of the per-name rows' pnl /
    premium_pct / entry_iv / rv (None-safe). The names are cross-sectionally
    correlated — this is ONE blended series, not 3x the sample."""
    by_month: dict = {}
    for r in results:
        for m in r['months']:
            by_month.setdefault(m['month'].to_period('M'), []).append(m)
    out = []
    for period in sorted(by_month):
        rows = by_month[period]

        def _mean(key):
            vals = [x[key] for x in rows if x[key] is not None]
            return float(np.mean(vals)) if vals else None

        out.append({
            'month': rows[0]['month'], 'strike': None, 'expiry': None,
            'settle_fallback': any(x['settle_fallback'] for x in rows),
            'premium_pct': _mean('premium_pct'),
            'entry_iv': _mean('entry_iv'), 'rv': _mean('rv'),
            'iv_minus_rv': _mean('iv_minus_rv'),
            'pnl': _mean('pnl'),
        })
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(v, spec, na='   n/a'):
    return format(v, spec) if v is not None else na


def print_report(results) -> None:
    print("\nVRP existence screen — short ATM straddle, monthly "
          "(pre-registered 2026-07-10)")
    print("cost model: $0.65/contract/leg at entry + spread haircut "
          "max($0.05, 5% of leg close)\n")
    for r in results:
        sk = r['skipped']
        print(f"{r['underlying']}: {len(r['months'])} settled months "
              f"(skipped: {sk['no_entry_close']} no-entry-close, "
              f"{sk['no_underlying_close']} no-underlying-close, "
              f"{sk['unsettled']} unsettled)")
        if r['months']:
            print(f"  {'month':>10} {'K':>8} {'expiry':>10} {'prem%':>7} "
                  f"{'IV':>6} {'RV':>6} {'IV-RV':>7} {'pnl$':>9}")
            for m in r['months']:
                fb = '*' if m['settle_fallback'] else ' '
                print(f"  {m['month'].date()!s:>10} {m['strike']:>8.2f} "
                      f"{m['expiry'].date()!s:>10} "
                      f"{m['premium_pct'] * 100:>6.2f}% "
                      f"{_fmt(m['entry_iv'], '>6.3f')} "
                      f"{_fmt(m['rv'], '>6.3f')} "
                      f"{_fmt(m['iv_minus_rv'], '>+7.3f')} "
                      f"{m['pnl']:>+9.2f}{fb}")
        _print_stats(_stats(r['months']), label=r['underlying'])
        print()
    pooled = pooled_monthly(results)
    _print_stats(_stats(pooled), label='POOLED (equal-weight across names, '
                                       'per month — correlated, not 3x n)')
    print("""
HONESTY:
  * ~23 monthly observations per name — an EXISTENCE measurement, not a
    gate. NO promotion pathway from this screen.
  * close != mid: free-tier closes are last-trade prints, no NBBO exists
    here; the 5%/leg haircut is a pre-registered guess, not a measurement.
  * entry commissions only; settlement assumed cash/exercise commission-free.
  * SPY/QQQ/IWM are the same index-vol bet — pooled is a blend, not 3x n.
  * single pre-registered config (2026-07-10); no sweeps.
  * '*' settle rows: expiry was not a trading session in the data — settled
    at the last prior session close.""")


def _print_stats(s: dict, *, label: str) -> None:
    if s['n'] == 0:
        print(f"  {label}: no settled months")
        return
    t = _fmt(s['nw_t'], '+.2f')
    p = _fmt(s['nw_p'], '.3f')
    print(f"  {label}: n={s['n']}  "
          f"prem {s['mean_premium_pct'] * 100:.2f}% mean / "
          f"{s['median_premium_pct'] * 100:.2f}% median  "
          f"win {s['win_rate'] * 100:.0f}%  mean pnl ${s['mean_pnl']:+.2f}  "
          f"NW t {t} (p {p}, lag {NW_LAG})  "
          f"worst ${s['worst_pnl']:+.2f} ({s['worst_month'].date()})")
    print(f"  {'':>{len(label)}}  IV {_fmt(s['mean_iv'], '.3f')} vs RV "
          f"{_fmt(s['mean_rv'], '.3f')} -> mean IV-RV "
          f"{_fmt(s['mean_iv_minus_rv'], '+.3f')}  "
          f"[{s['n_fallback_settles']} fallback settle(s), "
          f"{s['iv_partial_months']} iv-partial month(s) excluded]")


# ---------------------------------------------------------------------------
# Selftest — synthetic contracts, hand-computed pins (offline)
# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Pinned asserts: an OTM-expiry win, an ITM blowout loss (both
    hand-computed), and a BS IV-inversion round-trip."""
    # (1) OTM expiry — the straddle expires worthless, we keep the premium.
    # c=3.00 p=2.80 K=100 S_T=100: premium 5.80; haircut .15+.14=.29;
    # commission 2*.65/100=.013; intrinsic 0 -> pnl/share 5.497 -> $549.70.
    win = straddle_pnl(3.00, 2.80, 100.0, 100.0)
    assert abs(win['haircut'] - 0.29) < 1e-12
    assert abs(win['commission'] - 0.013) < 1e-12
    assert win['intrinsic'] == 0.0
    assert abs(win['pnl_dollars'] - 549.70) < 1e-9

    # (2) ITM blowout — S_T=130 through a K=100 straddle sold for 5.00.
    # haircut .125*2=.25; intrinsic 30 -> pnl/share 5-.25-30-.013=-25.263
    # -> -$2526.30.
    loss = straddle_pnl(2.50, 2.50, 100.0, 130.0)
    assert abs(loss['haircut'] - 0.25) < 1e-12
    assert loss['intrinsic'] == 30.0
    assert abs(loss['pnl_dollars'] - (-2526.30)) < 1e-9

    # (3) IV inversion round-trip through the repo's own pricer.
    for right, vol in (('call', 0.234), ('put', 0.187)):
        px = black_scholes_price(100.0, 100.0, 30 / 365.0, vol, 0.045, right)
        iv = implied_vol(px, 100.0, 100.0, 30 / 365.0, 0.045, right)
        assert iv is not None and abs(iv - vol) < 1e-6, (right, vol, iv)
    # A print below intrinsic has no BS vol — must refuse, not extrapolate.
    assert implied_vol(0.50, 100.0, 90.0, 30 / 365.0, 0.045, 'call') is None
    print('selftest OK')


def main() -> None:  # pragma: no cover — warehouse CLI (pieces unit-tested)
    from data.pit_warehouse import PitWarehouse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--underlyings', nargs='+', default=UNDERLYINGS)
    ap.add_argument('--dir', default=None,
                    help='warehouse dir (else PIT_WAREHOUSE_DIR resolution)')
    ap.add_argument('--selftest', action='store_true',
                    help='offline pinned asserts, then exit')
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    wh = PitWarehouse(args.dir)
    dtb3 = load_dtb3()
    results = [screen_underlying(wh, sym.upper(), dtb3=dtb3)
               for sym in args.underlyings]
    if not any(r['months'] for r in results):
        raise SystemExit(
            "no settled straddles found — run scripts/ingest_massive_options.py "
            "(and the bars_1m_sip minute ingest) first")
    print_report(results)


if __name__ == '__main__':
    main()
