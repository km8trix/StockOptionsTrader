"""YoY net share issuance from PIT quarterly share-count rows.

Pure math over the frame shape produced by
``PitWarehouse.fundamentals_quarterly(fields=('sharesbas', 'sharefactor'))``
(ticker, reportperiod, datekey, sharesbas, sharefactor) — no I/O, no
warehouse dependency, so both the issuance screen
(scripts/issuance_screen.py) and the issuance desk (desks/issuance.py) share
one definition and the tests are hermetic. The repo seam convention: signal
math lives in data/ (``sue_table`` in data/earnings_surprise.py), screens and
desks import it.

The Pontiff-Woodgate / Daniel-Titman issuance anomaly: firms that ISSUE
shares (SEOs, stock comp, stock-financed M&A) subsequently UNDERPERFORM;
firms that RETIRE shares (buybacks) outperform. SIGN CONVENTION (explicit,
academic): HIGH issuance predicts LOW returns, so the LONG leg is the
LOW/NEGATIVE-issuance end (buyback names) and a NEGATIVE value (net buyback)
is the prime long-leg member, not a trap to drop.

Signal: adjusted shares outstanding = sharesbas * sharefactor from PIT SF1
ARQ (deduped to the earliest-datekey filing per quarter). SPLIT NEUTRALITY
comes from the VENDOR, not sharefactor: Sharadar retroactively restates
sharesbas to the current split basis (verified 2026-07-10 against the
warehouse — NVDA/TSLA/AAPL forward and CERO/UNCY reverse splits are invisible
in sharesbas), so both quarters of the ratio share one basis and future-split
factors cancel; the surviving factor is exactly the intervening split a PIT
investor would also know. sharefactor is a near-constant ADR/unit multiplier
(varies only for share-class/ADR cases like BRK.B, V, ONON); multiplying each
quarter by its OWN factor handles ADR-ratio changes.

YoY net issuance = adjshares_t / adjshares_{t-4q} - 1, with the year-ago
quarter matched per ticker BY CALENDAR (latest reportperiod 330-410 days
earlier — the sue_table convention) so missing quarters skip the row instead
of silently misaligning it; no cross-frame merge_asof. Guards: both fields
non-null at both quarters, relative-epsilon on zero/near-zero adjusted share
counts (sharefactor==0 garbage), >=4 prior quarters of filing history, and
the sue_table running-max-datekey rule so a delinquent old filing cannot leak
into an earlier-dated signal. The signal of a filing is KNOWN at its datekey.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Seasonal-match window (sue_table convention): the year-ago quarter must
#: sit this many calendar days back from the current reportperiod.
_SEASONAL_LO_DAYS, _SEASONAL_HI_DAYS = 330, 410

#: Relative epsilon for the share-count denominator/numerator: a >1e6-fold
#: change in SPLIT-ADJUSTED shares within a year is data garbage (e.g.
#: sharefactor==0 rows exist in SF1), not a corporate action.
_REL_EPS = 1e-6


def issuance_table(shares: pd.DataFrame, *, min_history: int = 4
                   ) -> pd.DataFrame:
    """Per-filing YoY net issuance rows from quarterly share-count history.

    shares: columns ticker, reportperiod (datetime), datekey (datetime),
    sharesbas, sharefactor — one row per (ticker, reportperiod), the
    fundamentals_quarterly(fields=('sharesbas','sharefactor')) shape. Returns
    the subset of filings with a computable issuance — columns ticker,
    reportperiod, datekey, issuance — sorted by (ticker, reportperiod).

    issuance = adjshares_t / adjshares_{t-4q} - 1 where adjshares =
    sharesbas * sharefactor (split-adjusted, so a 2:1 split is NOT issuance).
    NEGATIVE = net buyback = the academic long. Guards: both fields finite at
    both quarters; relative-epsilon against zero/near-zero adjusted counts;
    at least ``min_history`` PRIOR quarterly filings (thin gappy histories
    dropped); and the sue_table out-of-order-filing rule — a row's datekey
    must be the running max in reportperiod order, else the seasonal match
    consumed a share count not yet public at the row's own datekey.
    """
    cols = ['ticker', 'reportperiod', 'datekey', 'issuance']
    out = []
    if shares is None or len(shares) == 0:
        return pd.DataFrame(columns=cols)
    for ticker, g in shares.groupby('ticker', sort=False):
        g = g.sort_values('reportperiod')
        rp = g['reportperiod'].to_numpy(dtype='datetime64[ns]')
        base = pd.to_numeric(g['sharesbas'], errors='coerce').to_numpy(float)
        fact = pd.to_numeric(g['sharefactor'], errors='coerce').to_numpy(float)
        adj = base * fact                       # split-adjusted shares
        n = len(g)
        # Seasonal match vs the latest quarter 330-410 days earlier (calendar,
        # not row position — a missing quarter skips, never misaligns).
        lo = np.searchsorted(rp, rp - np.timedelta64(_SEASONAL_HI_DAYS, 'D'),
                             side='left')
        hi = np.searchsorted(rp, rp - np.timedelta64(_SEASONAL_LO_DAYS, 'D'),
                             side='right')
        iss = np.full(n, np.nan)
        for i in range(n):
            if hi[i] <= lo[i] or i < min_history:   # no match / thin history
                continue
            j = hi[i] - 1                           # latest match in window
            cur, prev = adj[i], adj[j]
            if not (np.isfinite(cur) and np.isfinite(prev)):
                continue                            # a field NULL at a quarter
            if min(cur, prev) <= _REL_EPS * max(cur, prev, 1.0):
                continue                            # zero/near-zero garbage
            iss[i] = cur / prev - 1.0
        # Out-of-order-filing guard (see sue_table): the issuance is only
        # PIT-clean if the year-ago row was filed on or before this row's own
        # datekey; running-max datekey in reportperiod order enforces that.
        dk = g['datekey'].to_numpy(dtype='datetime64[ns]')
        in_order = dk == np.maximum.accumulate(dk)
        keep = np.isfinite(iss) & in_order
        if keep.any():
            sub = g.loc[keep, ['ticker', 'reportperiod', 'datekey']].copy()
            sub['issuance'] = iss[keep]
            out.append(sub)
    if not out:
        return pd.DataFrame(columns=cols)
    return (pd.concat(out, ignore_index=True)
            .sort_values(['ticker', 'reportperiod'])
            .reset_index(drop=True))


def _share_rows(ticker, counts, start='2020-03-31', lag_days=40, factor=1.0):
    """Quarterly share rows: reportperiod every 3 months, datekey +lag_days.

    Fixture builder shared by the screen selftest and the hermetic tests
    (tests/test_issuance_screen.py, tests/test_issuance_desk.py) — one
    definition of the fundamentals_quarterly row shape, kept next to the
    math it feeds."""
    rows = []
    for q, c in enumerate(counts):
        rp = pd.Timestamp(start) + pd.DateOffset(months=3 * q)
        rows.append({'ticker': ticker, 'reportperiod': rp,
                     'datekey': rp + pd.Timedelta(days=lag_days),
                     'sharesbas': float(c), 'sharefactor': float(factor)})
    return rows
