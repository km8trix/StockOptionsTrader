"""Computed quality ratios from raw PIT SF1 fields — the guarded num/den seam.

Pure math over ONE SF1 row (a dict-like of raw fundamental fields) — no I/O,
no warehouse dependency, so both the quality screen
(scripts/quality_screen.py) and the Value+Quality desk
(desks/value_quality.py) share one definition and the tests are hermetic.
The repo seam convention: signal math lives in data/ (``sue_table`` in
data/earnings_surprise.py, ``issuance_table`` in data/share_issuance.py);
screens and desks import it.

Why computed at all: the roe / roa RATIO columns are 100% NULL in the SF1
ARQ dimension (Sharadar only populates them in the trailing AR*T
dimensions) — but the RAW inputs (gp 3.4% null, netinc 3.5%, assets 0.05%)
are near-complete, so ratios like gp_assets (Novy-Marx 2013 gross
profitability) are rebuilt from raws per SF1 row: both inputs from the same
quarter by construction, denominator guarded below.
"""

from __future__ import annotations

_REL_EPS = 1e-9
#: Absolute denominator floor. The relative epsilon only rejects ratios
#: >= ~1e9, so real SF1 unit-inconsistency rows sail through (adversarial
#: review 2026-07-10: VIVS 2011-10-31 gp=1,677,000 / assets=3,628 ->
#: gp_assets=462; 235 ARQ rows with gp/assets > 2, 6,464 rows with
#: 0 < assets < $1M). Sub-$1M "total assets" for a filer with $M-scale
#: gross profit is a units error, not a company — and every such row lands
#: deterministically in the top-rank LONG leg. $1M is far below any real
#: small/mid-universe filer's balance sheet.
_MIN_ASSETS = 1e6


def computed_ratio(fund, num_field, den_field):
    """num/den from ONE SF1 row (same quarter by construction), or None.

    Guards: both inputs non-null, and the denominator positive with BOTH a
    relative-epsilon floor (den > eps * max(1, |num|), float-noise) and an
    absolute floor (_MIN_ASSETS, unit-inconsistency rows) — a near-zero
    assets base would otherwise explode the ratio into rank-dominating
    garbage in the long leg.
    """
    num, den = fund.get(num_field), fund.get(den_field)
    if num is None or den is None:
        return None
    num, den = float(num), float(den)            # DuckDB may hand Decimal
    if den <= max(_REL_EPS * max(1.0, abs(num)), _MIN_ASSETS):
        return None
    return num / den
