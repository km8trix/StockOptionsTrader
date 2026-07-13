"""
Central risk book — desk-agnostic exposure limits over a batch of intents.

Tracks, from the portfolio plus a pending-intents view, the desk's gross
exposure (sum of absolute per-symbol position values, including pending
fills' estimated value), net exposure, per-symbol exposure (aggregated
ACROSS pods/books — positions are netted per symbol first), and short
exposure. check() approves or blocks each OPENING intent against the
limits; CLOSING intents (SELL/COVER) are ALWAYS approved.

CUMULATIVE EVALUATION (the core guarantee): intents are evaluated in the
order given, and every approved opening intent's estimated value is
committed to the running exposure state before the next intent is
checked — so a batch that passes item-wise can never jointly breach the
book. Callers are responsible for a deterministic order; CitadelDesk
sorts closes first, then opens by pod weight descending, then symbol.

CONSERVATISM: closing intents do NOT credit exposure relief within the
same batch. A close might not fill (no bar) while a later open does, so
the book never counts unfilled exits as freed-up room. Approval is
therefore conservative by construction.

BOUNDARY SEMANTICS (exact): an intent that brings an exposure EXACTLY to
its limit passes; any amount over — even epsilon — blocks (strict ``>``
comparisons against the dollar limits).

Estimated intent value matches the engine's fill sizing exactly:
portfolio_value * capital_allocation * size_fraction, evaluated at check
time. ``prices`` (symbol -> latest close) is the preferred mark for
existing positions, falling back to each position's current_price.

MARGIN IS NOT MODELED: max_gross defaults to 1.0x desk capital (no
leverage) under the cash-account approximation used desk-wide.

--------------------------------------------------------------------------
GREEKS LIMITS (short_vega / short_gamma)
--------------------------------------------------------------------------
The Jane Street desk introduces options positions whose aggregate short
vega and short gamma can be capped desk-wide through this same book.
Evaluating ``short_vega_limit`` / ``short_gamma_limit`` requires a
per-position Greeks feed: inject a ``PortfolioGreeksAggregator`` as
``greeks_aggregator`` and check() evaluates them (see
_apply_greeks_overlay). Configuring either limit WITHOUT a feed still
raises NotImplementedError — a configured risk limit that cannot be
evaluated is a hard error, never a silent no-op.

TWO INDEPENDENT MECHANISMS, which compose:
  * COARSE desk-wide gate (this overlay): with a greeks_aggregator and a
    raw short_vega/short_gamma limit, the HELD book's aggregate short
    Greeks are measured each check(); once over budget, ALL opening
    intents in the batch are blocked all-or-nothing (closes always pass),
    so it can never leave a naked wing. Measured on the current book, so
    it backstops accumulation rather than judging a single trade.
  * PRECISE per-structure check (custom limit): JaneStreetDesk wires its
    vega cap through register_limit('short_vega', ...) — it owns the
    Greeks feed (its synthetic pricing engine) and judges whole multi-leg
    structures ATOMICALLY post-trade (the structure's net vega rides on
    its first short leg; blocking individual legs would leave naked
    wings). This stays the desk's primary control.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple

from desks.base import DeskIntent
from portfolio.manager import PortfolioManager

if TYPE_CHECKING:  # annotation only — avoids a runtime import edge
    from portfolio.greeks_aggregator import PortfolioGreeksAggregator

logger = logging.getLogger(__name__)

#: Intent actions that close exposure and are therefore always approved.
_CLOSING_ACTIONS = ('SELL', 'COVER')


class CentralRiskBook:
    """Cumulative gross/net/per-symbol/short exposure limits for a desk.

    Limits are fractions of DESK CAPITAL (portfolio value *
    capital_allocation, evaluated at check time):

        max_gross   — gross exposure cap (1.0 = no margin).
        max_symbol  — per-symbol exposure cap, aggregated across pods.
        max_net     — net exposure band: |net| <= max_net * capital.

    register_limit(name, check_fn) adds custom limits evaluated after the
    built-ins, in registration order. check_fn(intent, state) returns None
    to pass or a reason string to block; ``state`` is a dict with the
    PRE-COMMIT exposure view and the candidate values:
    {'desk_capital', 'gross', 'net', 'short', 'per_symbol', 'intent_value',
     'candidate_gross', 'candidate_net', 'candidate_symbol',
     'candidate_short'}.

    factor_neutrality_limit(model, loadings_ref, band) is a built-in FACTORY
    that returns such a check_fn enforcing a per-factor net-exposure BAND on
    the candidate book (the CitadelDesk registers it to keep its pods
    factor-neutral). It composes with the dollar limits and the Greeks
    overlay without changing any of them — just another custom limit reading
    the same running per_symbol state, so the whole batch stays inside the
    band cumulatively. See its docstring for the loadings-reference protocol
    and the graceful-degrade / disable semantics.
    """

    def __init__(self, capital_allocation: float = 1.0,
                 max_gross: float = 1.0,
                 max_symbol: float = 0.10,
                 max_net: float = 0.60,
                 short_vega_limit: Optional[float] = None,
                 short_gamma_limit: Optional[float] = None,
                 greeks_aggregator: Optional['PortfolioGreeksAggregator']
                 = None,
                 owner_key: Optional[str] = None):
        if not (0.0 < capital_allocation <= 1.0):
            raise ValueError(
                f"capital_allocation {capital_allocation} must be in (0, 1]")
        for name, value in (('max_gross', max_gross),
                            ('max_symbol', max_symbol),
                            ('max_net', max_net)):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        self.capital_allocation = capital_allocation
        self.max_gross = max_gross
        self.max_symbol = max_symbol
        self.max_net = max_net
        #: Fund-mode desk identity. When set, owner-tagged positions belonging
        #: only to other desks are invisible to this book. Untagged positions
        #: (the standalone path) remain visible for backward compatibility.
        self.owner_key = owner_key
        # Greeks limits (see module docstring): evaluated as a desk-wide
        # gate when a greeks_aggregator is injected; configured WITHOUT one
        # makes check() fail loudly (_require_greeks_feed), never silently.
        self.short_vega_limit = short_vega_limit
        self.short_gamma_limit = short_gamma_limit
        self.greeks_aggregator = greeks_aggregator
        #: name -> check_fn, evaluated in registration order (dicts
        #: preserve insertion order).
        self._custom_limits: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Extension point
    # ------------------------------------------------------------------
    def register_limit(self, name: str, check_fn: Callable) -> None:
        """Register a custom limit (see class docstring for the protocol)."""
        if not callable(check_fn):
            raise ValueError(f"check_fn for limit '{name}' must be callable")
        self._custom_limits[name] = check_fn
        logger.info("CentralRiskBook: registered custom limit '%s'", name)

    # ------------------------------------------------------------------
    # Factor-neutrality band (registerable custom limit)
    # ------------------------------------------------------------------
    @staticmethod
    def factor_neutrality_limit(model, loadings_ref: Dict,
                                band: float) -> Callable:
        """Build a custom check_fn enforcing a per-factor net-exposure BAND.

        The returned ``check_fn(intent, state)`` follows the custom-limit
        protocol (return None to pass, a reason to block). It measures the
        CANDIDATE book's net factor exposure per factor — the running held
        ``state['per_symbol']`` dollar exposures (prior approvals in the
        batch already committed) with the candidate symbol overridden to its
        post-trade ``state['candidate_symbol']`` value — using ``model`` (a
        :class:`desks.factor_risk.FactorRiskModel`) and the CURRENT loadings,
        and BLOCKS an open that would push any factor's |net exposure| beyond
        ``max(band, |held exposure on that factor|)``: a within-band factor
        may not cross the band, and a factor ALREADY over the band (held
        positions drift out as loadings rotate — the gate never force-unwinds,
        exactly like the Greeks overlay) may not be made WORSE, while a
        CORRECTIVE open that reduces or holds it PASSES (so the desk can hedge
        back toward neutral instead of being trapped directional). The band is
        a fraction of desk capital (``book_exposure`` normalizes by capital).
        Boundary semantics match the dollar limits: exactly at the ceiling
        passes, any amount over (strict ``>``) blocks. This is an ACCUMULATION
        backstop on NEW opens, NOT a realized-held-book cap — held exposure
        can drift as loadings rotate; the desk de-risks by closing (the
        Citadel transparency note surfaces the realized exposure each day).

        loadings_ref is a MUTABLE container (a ``dict`` holding the live
        loadings under a fixed key, or a 1-element list) the DESK refreshes
        each simulated day — registered ONCE, it always reads the day's
        loadings without re-registration. Two shapes are accepted:
          * ``{'loadings': {symbol: {factor: float}}}`` (preferred), or
          * any object exposing ``[0]`` that returns that loadings dict
            (e.g. a 1-element list ``[loadings]``).

        CUMULATIVE by construction (it reads the running per_symbol state, so
        the whole approved batch stays inside the band). GRACEFUL: a band of
        ``None`` / non-positive disables the gate (always pass — recovers the
        old non-factor-gated behavior); no loadings / no model likewise pass
        (no false block), exactly like the Greeks overlay degrading when its
        feed is silent. Closing intents never reach here (the book approves
        SELL/COVER before custom limits).
        """

        def _current_loadings() -> Dict:
            ref = loadings_ref
            if isinstance(ref, dict):
                return ref.get('loadings') or {}
            try:
                return ref[0] or {}
            except (IndexError, TypeError, KeyError):
                return {}

        def check_fn(intent: DeskIntent, state: Dict) -> Optional[str]:
            if band is None or band <= 0 or model is None:
                return None  # gate disabled -> old behavior recovered
            loadings = _current_loadings()
            if not loadings:
                return None  # graceful: no loadings -> pass (no false block)
            desk_capital = state['desk_capital']
            if desk_capital <= 0:
                return None  # base undefined -> dollar checks already gate
            # The gate is an ACCUMULATION backstop (like the Greeks overlay),
            # MONOTONE-IMPROVEMENT-aware: it measures the running HELD book
            # AND the candidate book. A within-band factor may not be pushed
            # past the band; a factor ALREADY over the band (held positions
            # drift out as loadings rotate — the gate never force-unwinds)
            # may not be made WORSE, but a corrective open that strictly
            # reduces or holds it PASSES (so the desk can hedge back toward
            # neutral instead of being trapped directional).
            held = model.book_exposure(
                state['per_symbol'], loadings, desk_capital)
            candidate = model.candidate_exposure(
                state['per_symbol'], intent.asset.symbol,
                state['candidate_symbol'], loadings, desk_capital)
            for factor, exposure in candidate.items():
                held_f = abs(held.get(factor, 0.0))
                ceiling = max(band, held_f)
                if abs(exposure) > ceiling:
                    extra = ""
                    if held_f > band:
                        extra = (f" (held {held.get(factor, 0.0):+.2%}; an "
                                 "open may not worsen an already-breaching "
                                 "factor)")
                    return (
                        f"net {factor} factor exposure would reach "
                        f"{exposure:+.2%} of desk capital, beyond the "
                        f"+/-{band:.2%} factor-neutrality band{extra}")
            return None

        return check_fn

    # ------------------------------------------------------------------
    # Exposure measurement
    # ------------------------------------------------------------------
    def compute_exposures(self, portfolio: PortfolioManager,
                          prices: Optional[Dict[str, float]] = None) -> Dict:
        """Current exposure snapshot from open positions (no pendings).

        When ``owner_key`` is configured, owner-tagged fund positions are
        included only when that desk is among their owners. Untagged positions
        remain visible, preserving standalone behavior.

        Positions are marked at prices[symbol] when available, else at
        their own current_price, and NETTED per symbol before gross/net/
        short are derived:

            per_symbol[s] = sum(quantity * mark) over positions in s
            gross = sum(|per_symbol|);  net = sum(per_symbol)
            short = sum(-per_symbol where per_symbol < 0)
        """
        prices = prices or {}
        per_symbol: Dict[str, float] = {}
        for asset, position in portfolio.positions.items():
            if position.quantity == 0:
                continue
            if (self.owner_key is not None
                    and position.owners is not None
                    and self.owner_key not in position.owners):
                continue
            mark = prices.get(asset.symbol, position.current_price)
            per_symbol[asset.symbol] = (per_symbol.get(asset.symbol, 0.0)
                                        + position.quantity * mark)
        gross = sum(abs(value) for value in per_symbol.values())
        net = sum(per_symbol.values())
        short = sum(-value for value in per_symbol.values() if value < 0)
        return {'gross': gross, 'net': net, 'short': short,
                'per_symbol': per_symbol}

    # ------------------------------------------------------------------
    # The check
    # ------------------------------------------------------------------
    def check(self, intents: List[DeskIntent], portfolio: PortfolioManager,
              prices: Optional[Dict[str, float]] = None
              ) -> Tuple[List[DeskIntent], List[Tuple[DeskIntent, str]]]:
        """Evaluate a batch of intents cumulatively, in the order given.

        Returns (approved, blocked_with_reasons) where blocked entries
        are (intent, reason) tuples. Closing intents (SELL/COVER) are
        always approved; opening intents commit their estimated value to
        the running exposure state when approved (see module docstring
        for ordering, conservatism and boundary semantics).
        """
        self._require_greeks_feed()

        approved: List[DeskIntent] = []
        blocked: List[Tuple[DeskIntent, str]] = []
        if not intents:
            return approved, blocked

        desk_capital = (portfolio.get_portfolio_value()
                        * self.capital_allocation)
        state = self.compute_exposures(portfolio, prices)
        gross, net, short = state['gross'], state['net'], state['short']
        per_symbol = state['per_symbol']

        gross_limit = self.max_gross * desk_capital
        symbol_limit = self.max_symbol * desk_capital
        net_limit = self.max_net * desk_capital

        for intent in intents:
            if intent.action in _CLOSING_ACTIONS:
                approved.append(intent)
                continue

            if desk_capital <= 0:
                # Fail closed: no meaningful limit base exists.
                blocked.append((intent,
                                f"desk capital {desk_capital:,.2f} is not "
                                f"positive; failing closed"))
                continue

            value = desk_capital * intent.size_fraction
            symbol = intent.asset.symbol
            delta = value if intent.action == 'BUY' else -value

            candidate_gross = gross + value
            candidate_net = net + delta
            candidate_symbol = per_symbol.get(symbol, 0.0) + delta
            candidate_short = short + (value if intent.action == 'SHORT'
                                       else 0.0)

            reason = None
            if candidate_gross > gross_limit:
                reason = (f"gross exposure {candidate_gross:,.2f} would "
                          f"exceed the {self.max_gross:.0%} limit "
                          f"{gross_limit:,.2f}")
            elif abs(candidate_net) > net_limit:
                reason = (f"net exposure {candidate_net:,.2f} would leave "
                          f"the +/-{self.max_net:.0%} band "
                          f"(limit {net_limit:,.2f})")
            elif abs(candidate_symbol) > symbol_limit:
                reason = (f"{symbol} exposure {abs(candidate_symbol):,.2f} "
                          f"would exceed the per-symbol {self.max_symbol:.0%} "
                          f"limit {symbol_limit:,.2f}")
            else:
                custom_state = {
                    'desk_capital': desk_capital,
                    'gross': gross, 'net': net, 'short': short,
                    'per_symbol': dict(per_symbol),
                    'intent_value': value,
                    'candidate_gross': candidate_gross,
                    'candidate_net': candidate_net,
                    'candidate_symbol': candidate_symbol,
                    'candidate_short': candidate_short,
                }
                for name, check_fn in self._custom_limits.items():
                    custom_reason = check_fn(intent, custom_state)
                    if custom_reason is not None:
                        reason = f"custom limit '{name}': {custom_reason}"
                        break

            if reason is not None:
                blocked.append((intent, reason))
                logger.debug("CentralRiskBook blocked %s %s: %s",
                             intent.action, symbol, reason)
                continue

            # Commit: the approved open is now part of the pending view.
            gross = candidate_gross
            net = candidate_net
            short = candidate_short
            per_symbol[symbol] = candidate_symbol
            approved.append(intent)

        self._apply_greeks_overlay(approved, blocked, portfolio)
        return approved, blocked

    # ------------------------------------------------------------------
    # Greeks limits (short_vega / short_gamma) — evaluated via an injected
    # PortfolioGreeksAggregator; fail loud without one.
    # ------------------------------------------------------------------
    def _require_greeks_feed(self) -> None:
        """A configured Greeks limit needs a feed to evaluate.

        short_vega_limit / short_gamma_limit cannot be evaluated without a
        per-position Greeks feed. When one is configured but no
        greeks_aggregator is injected the limit must fail LOUDLY rather
        than be silently skipped — a configured-but-unevaluable risk limit
        is a hard error. Inject a PortfolioGreeksAggregator to evaluate
        them (see _apply_greeks_overlay).
        """
        if (self.short_vega_limit is not None
                or self.short_gamma_limit is not None) \
                and self.greeks_aggregator is None:
            raise NotImplementedError(
                "short_vega/short_gamma limits are configured but no "
                "greeks_aggregator is injected to evaluate them (the "
                "Phase 8 options Greeks feed); inject a "
                "PortfolioGreeksAggregator or remove the limits")

    def _apply_greeks_overlay(self, approved: List[DeskIntent],
                              blocked: List[Tuple[DeskIntent, str]],
                              portfolio: PortfolioManager) -> None:
        """Desk-wide Greeks gate over the approved opening intents.

        With a short_vega/short_gamma limit configured AND a
        greeks_aggregator injected, the aggregator's CURRENT
        share-equivalent portfolio Greeks are measured (vega in $ per 1.00
        vol point, the greeks_series convention; short premium -> negative
        vega/gamma, so short exposure = max(0, -greek)). If the held book's
        short vega or short gamma already EXCEEDS its cap (strict >; exactly
        at the cap passes, matching the dollar-limit boundary semantics),
        every OPENING intent is moved from approved to blocked — the desk is
        over its Greeks budget and must not add risk. CLOSING intents
        (SELL/COVER) always stay approved (they relieve exposure).

        This is a COARSE desk-wide backstop measured on the HELD book, not a
        per-structure post-trade check: it blocks the batch's opens
        all-or-nothing, so it can never leave a naked option wing. A desk
        needing precise per-structure atomic post-trade vega judging
        registers a custom limit instead (JaneStreetDesk does this via
        register_limit('short_vega', ...)); the two mechanisms are
        independent and compose.
        """
        if self.short_vega_limit is None and self.short_gamma_limit is None:
            return
        if self.greeks_aggregator is None:  # _require_greeks_feed already
            return                          # raised if a limit was set
        greeks = self.greeks_aggregator.aggregate(portfolio)
        breaches: List[str] = []
        short_vega = max(0.0, -greeks['vega'])
        if self.short_vega_limit is not None \
                and short_vega > self.short_vega_limit:
            breaches.append(f"short vega {short_vega:,.2f} exceeds the "
                            f"limit {self.short_vega_limit:,.2f}")
        short_gamma = max(0.0, -greeks['gamma'])
        if self.short_gamma_limit is not None \
                and short_gamma > self.short_gamma_limit:
            breaches.append(f"short gamma {short_gamma:,.2f} exceeds the "
                            f"limit {self.short_gamma_limit:,.2f}")
        if not breaches:
            return
        reason = ("portfolio Greeks over budget (" + "; ".join(breaches)
                  + "); no new opening intents admitted")
        still_approved: List[DeskIntent] = []
        for intent in approved:
            if intent.action in _CLOSING_ACTIONS:
                still_approved.append(intent)
            else:
                blocked.append((intent, reason))
                logger.debug("CentralRiskBook Greeks gate blocked %s %s: %s",
                             intent.action, intent.asset.symbol, reason)
        approved[:] = still_approved
