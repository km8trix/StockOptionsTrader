"""Cross-sectional 12-1 momentum desk — graduating the IC screen's one survivor.

The signal_ic.py screen (BH-corrected over the 12-hypothesis family, then
cost-netted) left 12-1 cross-sectional momentum as the single durable,
cost-survivable factor on LARGE_CAP_100. This desk graduates that factor end to
end: rank the universe by 12-1 momentum (the 12-month return skipping the most
recent month — the canonical AQR/UMD factor; the skip avoids the 1-month
reversal that contaminates raw 12-month momentum), long the top quantile and
short the bottom, on the shared CrossSectionalLongShortDesk book.

FIXED factor, not a fitted model: the committee is empty, so walk_forward_fits
stays [] and the validation runs at n_trials=1 (no deflation) — honest, because
the rule is pre-specified by decades of literature, not mined here. Exits are
the book's own cross-sectional reconcile, so the desk ships a WIDE default
RiskManager: the 2% default price stop would churn a slow monthly factor. Per
leg sizes to ~0.5*target_gross/k (a few % on a 100-name quintile), well inside
the position-size cap.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from desks.cross_sectional import CrossSectionalLongShortDesk
from portfolio.risk_manager import RiskManager

LOOKBACK = 252   # ~12 months of trading days
SKIP = 21        # skip the most recent ~1 month (1-month reversal contamination)


class CrossSectionalMomentumDesk(CrossSectionalLongShortDesk):
    """Long top-quantile / short bottom-quantile by 12-1 momentum."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 lookback: int = LOOKBACK, skip: int = SKIP,
                 quantile: float = 0.2):
        if risk_manager is None:
            # Slow monthly factor: the cross-sectional reconcile rebalances the
            # book; a tight 2% price stop would churn winners out. Per-leg size
            # is a few % (quintile of ~100 names), so the default 0.10 cap is
            # fine — only the stop needs widening.
            risk_manager = RiskManager(position_stop_loss=0.50)
        super().__init__(
            key='xs_momentum',
            name='Cross-Sectional Momentum',
            description=('12-1 cross-sectional momentum (UMD): long the top '
                         'quantile by 12-month return skipping the most recent '
                         'month, short the bottom.'),
            accent='#8957e5',
            note_label='XS-Mom',
            reason_prefix='xs-mom',
            committee=[],
            model_label='12-1 momentum (fixed factor)',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
            quantile=quantile,
        )
        self.lookback = lookback
        self.skip = skip

    def _alpha_scores(self, all_data: Dict[str, pd.DataFrame],
                      date) -> Optional[Dict[str, float]]:
        """12-1 momentum per symbol: close[~1mo ago] / close[~12mo ago] - 1.

        Uses only past closes (frames are sliced through `date` by the engine),
        matching signal_ic.py's `mom_12_1 = close.shift(21)/close.shift(252)-1`.
        """
        scores: Dict[str, float] = {}
        for symbol, df in all_data.items():
            if df is None or df.empty:
                continue
            c = df['close']
            if len(c) <= self.lookback:
                continue  # need 12 months + the skip of history
            past = c.iloc[-1 - self.skip]      # ~1 month ago
            base = c.iloc[-1 - self.lookback]  # ~12 months ago
            if base and base > 0:
                scores[symbol] = float(past / base - 1.0)
        return scores if len(scores) >= self.min_scored else None
