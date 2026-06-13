"""Portfolio Greeks aggregator — desk-agnostic share-equivalent Greeks.

Lifts the Jane-Street-only Greeks aggregation (contract C12) into a
reusable primitive so ANY desk's option book — and, in the fund seam, the
whole account across desks — can report aggregate delta/gamma/theta/vega
and feed them to risk limits (CentralRiskBook short_vega/short_gamma).

CONVENTION (C12 share-equivalent, byte-identical to
JaneStreetDesk.greeks_series): each option position contributes its
per-share Black-Scholes greek x its SIGNED contract count x the contract
multiplier (100 for options); the portfolio value is the sum over option
positions. delta is therefore the equivalent share exposure, gamma its
change per $1 of spot, theta $ decay per CALENDAR day, vega $ per 1.00 vol
point. A portfolio with no priceable option positions reports all zeros.

The aggregator knows NOTHING about Black-Scholes or where market inputs
come from: a GreeksProvider closure supplied by the caller maps an option
Asset to its per-share {'delta','gamma','theta','vega'} (or None when the
leg cannot be priced today — e.g. no bar/IV — in which case the leg
contributes nothing). That keeps the primitive reusable across desks with
different pricing engines, and lets the same instance be injected into a
CentralRiskBook to evaluate Greeks limits.

DETERMINISM: positions are visited in sorted(str(asset)) order and the
greeks are summed key-by-key in GREEK_KEYS order — the exact arithmetic
JaneStreetDesk._update_greeks performed inline, so the produced totals are
bit-for-bit identical to the pre-refactor greeks_series.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from core.models import Asset, AssetType
from portfolio.manager import PortfolioManager

#: Maps an option Asset to its per-SHARE greeks {'delta','gamma','theta',
#: 'vega'} (extra keys such as 'price' are ignored), or None when the leg
#: cannot be priced today and therefore contributes nothing.
GreeksProvider = Callable[[Asset], Optional[Dict[str, float]]]

#: The greeks aggregated, and the order the totals dict is built/summed in
#: (matching JaneStreetDesk's inline {'delta','gamma','theta','vega'}).
GREEK_KEYS = ('delta', 'gamma', 'theta', 'vega')

_OPTION_TYPES = (AssetType.CALL, AssetType.PUT)


class PortfolioGreeksAggregator:
    """Aggregate share-equivalent option Greeks over a portfolio (C12).

    The greeks_provider is supplied once at construction so a single
    instance is self-contained (it can be injected into a CentralRiskBook,
    which only knows the portfolio, not how to price options).
    """

    def __init__(self, greeks_provider: GreeksProvider):
        if not callable(greeks_provider):
            raise ValueError("greeks_provider must be callable")
        self._provider = greeks_provider

    @staticmethod
    def zero() -> Dict[str, float]:
        """A fresh all-zero totals dict (the option-free portfolio value)."""
        return {key: 0.0 for key in GREEK_KEYS}

    def aggregate(self, portfolio: PortfolioManager) -> Dict[str, float]:
        """Share-equivalent portfolio Greeks over the option positions.

        Visits option positions in sorted(str(asset)) order so the float
        summation is deterministic. Zero-quantity positions and legs the
        provider cannot price (returns None) are skipped; an option-free
        portfolio returns all zeros. SIGN: position.quantity carries the
        sign (short options are negative), so short premium yields negative
        vega/gamma — the same convention the short_vega limit reads.
        """
        totals = self.zero()
        for asset in sorted((a for a in portfolio.positions
                             if a.asset_type in _OPTION_TYPES), key=str):
            position = portfolio.positions[asset]
            if position.quantity == 0:
                continue
            per_share = self._provider(asset)
            if per_share is None:
                continue
            scale = position.quantity * asset.multiplier
            for key in totals:
                totals[key] += per_share[key] * scale
        return totals
