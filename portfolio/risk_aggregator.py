"""
Portfolio risk aggregator — the fund's ACCOUNT-LEVEL risk overlay.

The FundOrchestrator already runs one account-wide apply_risk (daily-loss
circuit, per-name position-size, stops on the unified book). That gate is
PER-NAME: it can pass every individual name (each <= the position-size cap)
while the BOOK as a whole is over-leveraged — fifteen names at 10% each is
150% gross with every name "in limit". PortfolioRiskAggregator closes that
gap by checking the COMBINED book after apply_risk:

  * GROSS LEVERAGE (hard-enforced, STRICT): total notional (multiplier-aware,
    abs — shorts count) plus the batch's opening intents must stay within
    max_gross_leverage x account capital. Evaluated cumulatively in a
    deterministic order; an opening intent that would breach is BLOCKED
    (closes always pass). This is the headline account rail.
  * CORRELATION (reuses RiskManager.check_correlation): trailing-window return
    series for the book's names (current + would-be opens) are checked for an
    over-correlated pair; on breach ALL still-pending opens are held (STRICT
    batch block). Ragged/short/absent series are recorded in `unevaluated`,
    never silently passed.
  * SECTOR CONCENTRATION (reuses RiskManager.check_portfolio_sector_
    concentration): a no-op until Asset carries sector tags — recorded in
    `unevaluated` so the skipped limit is explicit, not a false green.

MODE: STRICT only (block; no pro-rata SCALE) per the Phase 2 decision. Closes
(SELL/COVER) are ALWAYS approved — exits are never blocked by account risk.

ADDITIVE: instantiated only by a FundOrchestrator that is given one. Single-
desk and strategy backtests never construct it, so their paths are unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from core.models import AssetType
from desks.base import DeskIntent
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

_CLOSING_ACTIONS = ('SELL', 'COVER')

#: Default trailing window (trading days) for the correlation return series.
DEFAULT_CORRELATION_WINDOW = 21


class PortfolioRiskAggregator:
    """Account-level gross / correlation / sector overlay for the fund.

    Limits are fractions/multiples of ACCOUNT capital (portfolio value at
    check time). Defaults: 1.0x gross (no leverage), 0.95 correlation (on but
    lenient — equities co-move), sector off (no tags exist). check() returns
    (approved, blocked_with_reasons); blocked entries are (intent, reason).
    """

    def __init__(self, max_gross_leverage: float = 1.0,
                 max_correlation: float = 0.95,
                 max_sector_exposure: float = 1.0,
                 correlation_window: int = DEFAULT_CORRELATION_WINDOW):
        if max_gross_leverage <= 0:
            raise ValueError(
                f"max_gross_leverage must be > 0, got {max_gross_leverage}")
        if correlation_window < 3:
            raise ValueError(
                f"correlation_window must be >= 3, got {correlation_window}")
        self.max_gross_leverage = max_gross_leverage
        self.max_correlation = max_correlation
        self.max_sector_exposure = max_sector_exposure
        self.correlation_window = correlation_window
        # A dedicated RiskManager supplies the Phase 0 correlation + sector
        # primitives (and their violation/unevaluated discipline).
        self._risk = RiskManager(max_correlation=max_correlation,
                                 max_sector_exposure=max_sector_exposure)
        self.violations: List[str] = []
        self.unevaluated: List[str] = []

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    @staticmethod
    def _current_gross(portfolio: PortfolioManager) -> float:
        """Absolute notional of the open book, multiplier-aware (options x100,
        shorts counted positive)."""
        return sum(abs(pos.quantity * pos.current_price * pos.asset.multiplier)
                   for pos in portfolio.positions.values()
                   if pos.quantity != 0)

    def _returns(self, symbols: Sequence[str],
                 all_data: Dict[str, pd.DataFrame], date
                 ) -> Dict[str, Sequence[float]]:
        """Trailing-window simple returns per symbol from closes up to `date`
        (inclusive). Symbols without a usable series are simply omitted;
        RiskManager.check_correlation records the resulting gap."""
        out: Dict[str, Sequence[float]] = {}
        for symbol in symbols:
            frame = all_data.get(symbol)
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            closes = frame.loc[frame.index <= pd.Timestamp(date), 'close']
            closes = closes.dropna().iloc[-(self.correlation_window + 1):]
            rets = closes.pct_change().dropna()
            if len(rets) >= 3:
                out[symbol] = rets.to_numpy()
        return out

    @staticmethod
    def _book_symbols(portfolio: PortfolioManager,
                      opens: List[DeskIntent]) -> List[str]:
        symbols = {pos.asset.symbol for pos in portfolio.positions.values()
                   if pos.quantity != 0}
        symbols.update(intent.asset.symbol for intent in opens)
        return sorted(symbols)

    # ------------------------------------------------------------------
    # The check
    # ------------------------------------------------------------------
    def check(self, intents: List[DeskIntent], portfolio: PortfolioManager,
              all_data: Dict[str, pd.DataFrame], date
              ) -> Tuple[List[DeskIntent], List[Tuple[DeskIntent, str]]]:
        """Account-level overlay over a batch of (already per-desk-approved,
        netted) intents. Closes always pass; opening intents are gated on
        cumulative gross leverage (per-intent) then book-level correlation and
        sector (STRICT batch block on a breach). STRICT mode: block, no scale.
        """
        self.violations = []
        self.unevaluated = []
        self._risk.violations = []
        self._risk.unevaluated = []

        approved: List[DeskIntent] = []
        blocked: List[Tuple[DeskIntent, str]] = []
        opens: List[DeskIntent] = []
        for intent in intents:
            if intent.action in _CLOSING_ACTIONS:
                approved.append(intent)
            else:
                opens.append(intent)
        if not opens:
            return approved, blocked

        account_capital = portfolio.get_portfolio_value()
        if account_capital <= 0:
            reason = (f"account capital {account_capital:,.2f} is not "
                      f"positive; failing closed")
            self.violations.append(reason)
            blocked.extend((intent, reason) for intent in opens)
            return approved, blocked

        # 1. Gross leverage — cumulative, per-intent, deterministic order.
        gross = self._current_gross(portfolio)
        gross_limit = self.max_gross_leverage * account_capital
        pending: List[DeskIntent] = []
        for intent in sorted(opens, key=lambda i: (i.asset.symbol, i.action)):
            # Fractional intents use the account-fraction proxy.  Absolute
            # STOCK quantities are valued exactly from the latest causal close
            # so a huge quantity cannot pass behind a tiny size_fraction.  For
            # option structures the fraction represents defined max-loss risk,
            # rather than premium/underlying notional, and remains the more
            # conservative structure-level convention.
            notional = account_capital * intent.size_fraction
            if (intent.quantity is not None
                    and intent.asset.asset_type is AssetType.STOCK):
                frame = all_data.get(intent.asset.symbol)
                price = None
                if frame is not None and not frame.empty and 'close' in frame:
                    candidate = float(frame['close'].iloc[-1])
                    if math.isfinite(candidate) and candidate > 0:
                        price = candidate
                if price is None:
                    reason = (f"absolute quantity for {intent.asset.symbol} "
                              "cannot be valued from current data")
                    self.violations.append(reason)
                    blocked.append((intent, reason))
                    continue
                notional = abs(float(intent.quantity)) * price
            if gross + notional > gross_limit:
                reason = (f"account gross {gross + notional:,.2f} would exceed "
                          f"the {self.max_gross_leverage:.2f}x limit "
                          f"{gross_limit:,.2f}")
                self.violations.append(reason)
                blocked.append((intent, reason))
            else:
                gross += notional
                pending.append(intent)

        # 2/3. Book-level correlation + sector (STRICT batch block on breach).
        symbols = self._book_symbols(portfolio, pending)
        returns = self._returns(symbols, all_data, date)
        if self.max_correlation < 1.0 and len(returns) < 2:
            # A configured correlation cap with fewer than two usable return
            # series cannot be evaluated — record it (check_correlation's own
            # empty-input path returns True WITHOUT recording, so do it here)
            # rather than let the limit silently pass.
            self.unevaluated.append(
                'correlation: fewer than 2 usable return series')
            corr_ok = True
        else:
            corr_ok = self._risk.check_correlation(returns)
        sector_ok = self._evaluate_sector(portfolio, account_capital)
        self.violations.extend(self._risk.violations)
        self.unevaluated.extend(self._risk.unevaluated)

        if not (corr_ok and sector_ok):
            reason = ("account correlation/sector limit breached; opening "
                      "intents held this cycle")
            blocked.extend((intent, reason) for intent in pending)
            return approved, blocked

        approved.extend(pending)
        return approved, blocked

    def _evaluate_sector(self, portfolio: PortfolioManager,
                         account_capital: float) -> bool:
        """Sector concentration over the current book, or record the gap.

        Asset has no sector tag today, so this is a documented no-op recorded
        in `unevaluated` (never a silent pass). When tags exist it enforces
        via RiskManager.check_portfolio_sector_concentration.
        """
        has_sector = any(getattr(pos.asset, 'sector', None)
                         for pos in portfolio.positions.values())
        if not has_sector:
            if self.max_sector_exposure < 1.0:
                self._risk.unevaluated.append(
                    'sector concentration: positions carry no sector tags')
            return True
        return self._risk.check_portfolio_sector_concentration(
            portfolio.positions, account_capital)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def get_risk_report(self) -> Dict:
        """Last check()'s violations + unevaluated limits + the configured
        caps. `unevaluated` prevents a false green (a skipped correlation/
        sector cap must be distinguishable from a passed one)."""
        return {
            'violations': list(self.violations),
            'unevaluated': list(self.unevaluated),
            'max_gross_leverage': self.max_gross_leverage,
            'max_correlation': self.max_correlation,
            'max_sector_exposure': self.max_sector_exposure,
        }
