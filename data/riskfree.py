"""Dated 3-month T-bill risk-free series (FRED DTB3) for cash-yield accrual.

The vendored CSV (data/vendored/dtb3.csv) is the FRED DTB3 series —
"3-Month Treasury Bill Secondary Market Rate, Discount Basis", daily,
percent — fetched 2026-07-10 from
https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3 (1954-01-04 ..
2026-07-08). It is committed so backtests are reproducible offline.

HONESTY: this module exists so idle-cash yield in a backtest is DATED — ZIRP
years (2015-2021, ~0.05-0.3%) accrue almost nothing while 2023-2024 accrue
~5%. A flat retroactive assumption ("cash earned 2% throughout") is
dishonest and deliberately impossible here: the only rate source is the
dated series. Note the asymmetry this repairs: the research gate
(analysis.research_stats.validate_strategy_oos) already SUBTRACTS a flat 2%
rf from every daily return on 100% of NAV; without accrual the engine's
idle cash earns zero on both legs of that comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Vendored FRED DTB3 CSV (columns: observation_date, DTB3 in percent).
DTB3_CSV = Path(__file__).resolve().parent / 'vendored' / 'dtb3.csv'


def load_dtb3(path: str | Path = DTB3_CSV) -> pd.Series:
    """Load a FRED DTB3 CSV into a Series: datetime index -> ANNUALIZED
    DECIMAL rate (percent / 100). Missing observations (FRED marks them
    '.' or leaves the field empty) are dropped, not filled — rate_asof
    forward-fills at lookup time."""
    frame = pd.read_csv(path, na_values='.',
                        parse_dates=['observation_date'])
    series = frame.set_index('observation_date')['DTB3'].dropna() / 100.0
    return series.sort_index()


def rate_asof(series: pd.Series, date) -> float:
    """Forward-fill lookup: the last observation on/before ``date`` (weekends
    and holidays carry the prior print). Dates before the first observation
    return 0.0 — no data means accrue nothing, never back-fill a later
    rate into the past."""
    pos = series.index.searchsorted(pd.Timestamp(date), side='right') - 1
    if pos < 0:
        return 0.0
    return float(series.iloc[pos])
