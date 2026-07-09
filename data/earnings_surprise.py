"""Standardized Unexpected Earnings (SUE) from PIT quarterly EPS rows.

Pure math over the frame shape produced by ``PitWarehouse.eps_quarterly``
(ticker, reportperiod, datekey, epsdil) — no I/O, no warehouse dependency, so
both the PEAD screen (scripts/pead_screen.py) and the PEAD desk (desks/pead.py)
share one definition and the tests are hermetic.

SUE (Bernard–Thomas style, seasonal random walk WITH drift):

    d_q  = epsdil_q - epsdil_{q-4 quarters}
    SUE  = (d_latest - mean(prior diffs)) / std(prior diffs)
           over the trailing 8 seasonal diffs (ddof=1, current EXCLUDED,
           >= 4 finite diffs required)

The drift term matters: without it a steadily-growing name posts a huge
"surprise" every quarter (constant d, tiny std) — the drift-adjusted form
centers steady growth at zero so SUE measures the UNEXPECTED component.

The seasonal match is by calendar (the latest prior reportperiod 330-410 days
earlier), not by row position — missing quarters therefore skip the diff
instead of silently misaligning it. The SUE of a filing is treated as KNOWN at
its datekey; input rows must already be deduped one-per-reportperiod on the
earliest datekey (eps_quarterly does this), which minimizes the residual
late-amendment ordering risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Seasonal-match window: the year-ago quarter must sit this many days back.
_SEASONAL_LO_DAYS, _SEASONAL_HI_DAYS = 330, 410


def sue_table(eps: pd.DataFrame, *, window: int = 8,
              min_diffs: int = 4) -> pd.DataFrame:
    """Per-filing SUE rows from quarterly EPS history.

    eps: columns ticker, reportperiod (datetime), datekey (datetime),
    epsdil (float); one row per (ticker, reportperiod). Returns the subset of
    filings with a computable SUE — columns ticker, reportperiod, datekey, sue
    — sorted by (ticker, reportperiod).
    """
    out = []
    if eps is None or len(eps) == 0:
        return pd.DataFrame(columns=['ticker', 'reportperiod', 'datekey', 'sue'])
    for ticker, g in eps.groupby('ticker', sort=False):
        g = g.sort_values('reportperiod')
        rp = g['reportperiod'].to_numpy(dtype='datetime64[ns]')
        e = g['epsdil'].to_numpy(dtype=float)
        n = len(g)
        # Seasonal diff vs the latest quarter 330-410 days earlier.
        lo = np.searchsorted(rp, rp - np.timedelta64(_SEASONAL_HI_DAYS, 'D'),
                             side='left')
        hi = np.searchsorted(rp, rp - np.timedelta64(_SEASONAL_LO_DAYS, 'D'),
                             side='right')
        diffs = np.full(n, np.nan)
        for i in range(n):
            if hi[i] > lo[i] and np.isfinite(e[i]):
                j = hi[i] - 1                      # latest match in the window
                if np.isfinite(e[j]):
                    diffs[i] = e[i] - e[j]
        sue = np.full(n, np.nan)
        for i in range(n):
            if not np.isfinite(diffs[i]):
                continue
            past = diffs[max(0, i - window):i]     # trailing, current EXCLUDED
            past = past[np.isfinite(past)]
            if len(past) >= min_diffs:
                sd = past.std(ddof=1)
                # Relative epsilon: a float-noise std on a near-constant diff
                # series (steady grower) must read as DEGENERATE, not divide
                # into an absurd SUE. Cent-level real stds survive easily.
                if np.isfinite(sd) and sd > 1e-9 * max(1.0, abs(past.mean())):
                    sue[i] = (diffs[i] - past.mean()) / sd
        # Out-of-order-filing guard: a SUE is only PIT-clean if every row it
        # consumed (seasonal match + trailing window) was filed on or before
        # its own datekey. Requiring the row's datekey to be the running max
        # in reportperiod order enforces exactly that (a delinquent old
        # quarter first-filed LATER would otherwise leak future EPS into an
        # earlier-dated SUE). Slightly conservative — it also drops the few
        # rows straddling the disorder — and exact for normal filers.
        dk = g['datekey'].to_numpy(dtype='datetime64[ns]')
        in_order = dk == np.maximum.accumulate(dk)
        keep = np.isfinite(sue) & in_order
        if keep.any():
            sub = g.loc[keep, ['ticker', 'reportperiod', 'datekey']].copy()
            sub['sue'] = sue[keep]
            out.append(sub)
    if not out:
        return pd.DataFrame(columns=['ticker', 'reportperiod', 'datekey', 'sue'])
    return (pd.concat(out, ignore_index=True)
            .sort_values(['ticker', 'reportperiod'])
            .reset_index(drop=True))


def latest_fresh_sue(sue_rows: pd.DataFrame, asof, *,
                     fresh_days: int = 63) -> pd.Series:
    """ticker -> SUE of each ticker's latest filing visible at ``asof``,
    kept only when that filing is FRESH (datekey within ``fresh_days``
    calendar days of asof). PIT: filings with datekey > asof are invisible.

    sue_rows: the sue_table output (any subset of tickers).
    """
    if sue_rows is None or len(sue_rows) == 0:
        return pd.Series(dtype=float)
    ts = pd.Timestamp(asof)
    vis = sue_rows[sue_rows['datekey'] <= ts]
    if len(vis) == 0:
        return pd.Series(dtype=float)
    latest = (vis.sort_values('datekey').groupby('ticker', sort=False).tail(1))
    fresh = latest[latest['datekey'] >= ts - pd.Timedelta(days=fresh_days)]
    return fresh.set_index('ticker')['sue'].astype(float)
