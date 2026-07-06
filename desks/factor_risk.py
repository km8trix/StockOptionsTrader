"""
Factor risk model (Citadel desk) — measures a BOOK's net exposure to the
common price factors, so the central risk book can keep the pods
FACTOR-NEUTRAL.

WHERE desks/models/factor.py SCORES symbols cross-sectionally (a ranking
signal a pod trades ON), THIS module measures the AGGREGATE book's net
loading on each of those same factors (a risk the book trades AGAINST).
Two uses of one factor set: a pod tilts toward a factor to earn it; the
risk book CAPS new-open ACCUMULATION on any single factor so no pod (or the
pods jointly) can quietly ADD a levered factor bet. It is an accumulation
backstop measured at trade time (like the Greeks overlay), NOT a continuous
realized cap: an open is blocked when it would push a factor beyond
``max(band, current held exposure)`` — a within-band factor may not cross the
band, an already-drifted factor may not be made worse, while a CORRECTIVE
hedge that reduces it passes — but HELD positions can still drift past the
band as the cross-section rotates (the gate never force-unwinds). The Citadel
transparency note surfaces the realized exposure each day; the desk de-risks
by closing. Citadel's edge is idiosyncratic pod alpha, not a levered factor
bet — this is the seam that enforces that.

FACTOR SET (REUSED verbatim from desks/models/factor.py — NOT a new set):
momentum / reversal / low_vol / risk_adj_mom, each built so HIGHER raw
value = MORE attractive, then standardized CROSS-SECTIONALLY per date (a
z-score across symbols at that timestamp) with
:func:`desks.features.cross_sectional_rank`. The per-symbol LOADING is that
standardized z-score: a positive loading means the name sits on the high
(attractive) side of the factor's cross-section that day, a negative
loading the low side, ~0 the middle. The book's NET factor exposure is the
dollar-weighted sum of its holdings' loadings, expressed as a FRACTION of
desk capital (see :meth:`book_exposure`).

LEAKAGE GUARANTEE (the same one factor.py gives): every loading at date D
is a function of rows ``index <= D`` ONLY. ``loadings(all_data, date)``
first slices each frame to ``index <= date`` before building the raw factor
panel (which itself computes each factor from the prefix ``close[:t+1]``
only) and the per-date cross-section, so appending FUTURE rows and
re-slicing yields byte-identical loadings at D. Inside the Citadel order
flow the engine already hands the desk frames sliced to ``index <= date``,
so the slice here is a belt-and-braces idempotent guard; called directly on
an unsliced panel it is what makes the model leakage-free.

GRACEFUL DEGRADE (never crash, never false-block — mirrors the Greeks feed
degrading): thin/empty data, a degenerate cross-section (no spread to
standardize -> z-scores collapse to 0.0), or a symbol too short to form any
factor that day simply yields an EMPTY or zero-loading reading. A symbol
absent from the loadings dict is treated by callers as a 0/neutral loading
on every factor, so it contributes nothing to the net exposure and can
never trip the band. The factor gate therefore passes whenever the model
cannot speak, exactly like the dollar limits pass when an exposure is
undefined.

DETERMINISM: pure pandas/numpy reductions through the same cross-sectional
z-score factor.py uses — identical input frames yield byte-identical
loadings and exposures across runs. The model holds NO data between calls
(``loadings`` and ``book_exposure`` are pure functions of their args), so
the desk owns the loadings reference and refreshes it each simulated day.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from desks.models.factor import (
    FACTOR_COLUMNS, _factor_exposures, _raw_factor_panel)
from desks import features as feature_lib

logger = logging.getLogger(__name__)

#: Fast-path guard for :func:`_asof_factor_panel`: with more than this many
#: DISTINCT as-of dates in the universe, the per-prefix evaluation would run
#: O(symbols x dates) full-series passes, so one full vectorized panel build
#: (restricted to the as-of dates before standardization) is cheaper. Purely
#: a performance switch — both paths return byte-identical rows.
_MAX_FAST_ASOF_DATES = 8


def _asof_factor_panel(
        sliced: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Raw factor rows restricted to the symbols' AS-OF dates only.

    Returns, per symbol, the SAME rows ``_raw_factor_panel(sliced)`` would
    return RESTRICTED to the union of the symbols' last FORMABLE dates. A
    symbol's as-of date is its last raw-panel row — NOT necessarily the
    frame's last row, since a tail row where no factor is finite is dropped
    by the panel keep-mask; such a symbol falls back to the full builder to
    resolve its last formable row. Only these as-of rows are consumed by
    :meth:`FactorRiskModel.loadings`, and the per-date cross-sectional
    z-score depends on that date's row alone, so standardizing just these
    rows is byte-identical to standardizing the full history — while
    skipping the O(history) panel rebuild ``loadings()`` used to pay on
    every simulated day.

    BYTE-IDENTITY: every returned cell is produced either by
    :func:`_factor_exposures` on the exact prefix ``close[index <= d]``
    (bit-for-bit equal to the vectorized panel row — the golden equivalence
    ``tests/test_factor_model.py`` pins) or by :func:`_raw_factor_panel`
    itself (the non-formable-tail and many-dates fallbacks). The cross-
    section at each needed date keeps EVERY symbol with a formable row at
    that date, not just the symbol whose as-of date it is.
    """
    # symbol -> (as-of date, raw factor row at that date).
    asof: Dict[str, tuple] = {}
    # Symbols whose LAST row is not formable: their kept rows, from the
    # existing full builder (rare; single-symbol cost).
    fallback_frames: Dict[str, pd.DataFrame] = {}
    for symbol, past in sliced.items():
        # A non-finite return in the vol window (e.g. an inf from a zero
        # price) surfaces as an arithmetic RuntimeWarning inside the scalar
        # std before being squashed to NaN by the finite guard — exactly
        # the warning the vectorized builder silences too (see
        # _trailing_vol_series); it carries no information.
        with np.errstate(invalid='ignore', divide='ignore'):
            exposures = _factor_exposures(past['close'])
        if exposures is not None:
            asof[symbol] = (past.index[-1], exposures)
            continue
        sub = _raw_factor_panel({symbol: past}).get(symbol)
        if sub is not None and not sub.empty:
            fallback_frames[symbol] = sub
        # else: the symbol forms no factors at all -> omitted, as before.

    needed = {date for date, _ in asof.values()}
    needed.update(frame.index[-1] for frame in fallback_frames.values())
    if not needed:
        return {}

    if len(needed) > _MAX_FAST_ASOF_DATES:
        # Degenerate mixed-calendar universe: one full vectorized build is
        # cheaper than symbols x dates prefix evaluations. Restricting the
        # rows BEFORE standardization still skips the full-history
        # z-scoring.
        full = _raw_factor_panel(sliced)
        return {symbol: frame.loc[frame.index.isin(needed)]
                for symbol, frame in full.items()}

    needed_sorted = sorted(needed)
    panel: Dict[str, pd.DataFrame] = {}
    for symbol, past in sliced.items():
        if symbol in fallback_frames:
            frame = fallback_frames[symbol]
            panel[symbol] = frame.loc[frame.index.isin(needed)]
            continue
        if symbol not in asof:
            continue
        asof_date, asof_exposures = asof[symbol]
        dates = []
        rows = []
        for date in needed_sorted:
            if date == asof_date:
                dates.append(date)
                rows.append(asof_exposures)
            elif date < asof_date and date in past.index:
                # Another symbol's as-of date: this symbol is part of that
                # date's cross-section iff it has a FORMABLE row there.
                # (Same uninformative-warning silencing as above.)
                with np.errstate(invalid='ignore', divide='ignore'):
                    exposures = _factor_exposures(
                        past.loc[past.index <= date, 'close'])
                if exposures is not None:
                    dates.append(date)
                    rows.append(exposures)
        panel[symbol] = pd.DataFrame(
            [[row[c] for c in FACTOR_COLUMNS] for row in rows],
            index=pd.Index(dates), columns=list(FACTOR_COLUMNS))
    return panel


class FactorRiskModel:
    """Measures a book's net exposure to the common price factors.

    Stateless: every method is a pure function of its arguments; the model
    holds no data between calls. The factor set and the cross-sectional
    standardization are REUSED from :mod:`desks.models.factor` (see the
    module docstring), so the loadings here are the same z-scored factor
    exposures the AQR/factor machinery already trusts — there is exactly
    one factor definition in the codebase.

    Args:
        factors: the factor columns to measure (default the full
            :data:`desks.models.factor.FACTOR_COLUMNS`). Restricting it is
            supported for tests; production uses the full set.
    """

    def __init__(self, factors: Optional[Sequence[str]] = None):
        cols = tuple(factors) if factors is not None else tuple(FACTOR_COLUMNS)
        unknown = [c for c in cols if c not in FACTOR_COLUMNS]
        if unknown:
            raise ValueError(
                f"Unknown factor(s) {unknown}; must be a subset of "
                f"{tuple(FACTOR_COLUMNS)}")
        if not cols:
            raise ValueError("FactorRiskModel needs at least one factor")
        self.factors: Sequence[str] = cols

    # ------------------------------------------------------------------
    # Per-symbol loadings (cross-sectional, leakage-free)
    # ------------------------------------------------------------------
    def loadings(self, all_data: Dict[str, pd.DataFrame],
                 date) -> Dict[str, Dict[str, float]]:
        """Per-symbol cross-sectional factor loadings as of ``date``.

        REUSES :func:`desks.models.factor._raw_factor_panel` (the four
        backward-looking factors, each from the prefix ``close[:t+1]``) and
        :func:`desks.features.cross_sectional_rank` (the per-date z-score
        across symbols) — the identical pipeline factor.py standardizes its
        scores through. The loading returned for a symbol is the LATEST
        (as-of-date) standardized exposure per factor.

        Strictly backward-looking: each frame is sliced to ``index <= date``
        before any factor is formed, so appending future rows and
        re-slicing yields identical loadings (leakage-free; module
        docstring). A symbol without a usable factor row that day is simply
        OMITTED from the result — callers treat an absent symbol as a
        0/neutral loading on every factor.

        Returns ``{}`` (graceful) when no symbol has enough history, the
        data is empty, or the cross-section is degenerate — never raises.
        """
        if not all_data:
            return {}
        cutoff = pd.Timestamp(date)

        # Belt-and-braces no-lookahead slice: drop any row strictly after
        # `date` before building factors. Inside the Citadel flow the engine
        # already slices to index <= date, so this is idempotent there; it
        # is what guarantees leakage-freedom when called on an unsliced
        # panel (the leakage test exercises exactly this).
        sliced: Dict[str, pd.DataFrame] = {}
        for symbol, frame in all_data.items():
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            try:
                past = frame[frame.index <= cutoff]
            except TypeError:
                # Non-comparable index (defensive); skip the symbol rather
                # than crash the run.
                continue
            if past.empty:
                continue
            sliced[symbol] = past

        # Only each symbol's as-of row is consumed below, so build (or
        # restrict to) just the as-of date rows — byte-identical to the
        # former full-history build + standardization, because the per-date
        # z-score depends on that date's cross-section alone (see
        # _asof_factor_panel for the equivalence argument).
        raw_panel = _asof_factor_panel(sliced)
        if not raw_panel:
            return {}

        # Cross-sectional z-score per factor across symbols (no-lookahead,
        # row-wise across symbols) on the as-of rows only, then take each
        # symbol's LATEST row.
        z_by_factor: Dict[str, pd.DataFrame] = {}
        for factor in self.factors:
            z_by_factor[factor] = feature_lib.cross_sectional_rank(
                raw_panel, column=factor, method='zscore')

        result: Dict[str, Dict[str, float]] = {}
        for symbol, frame in raw_panel.items():
            if frame.empty:
                continue
            last_date = frame.index[-1]
            loading: Dict[str, float] = {}
            for factor in self.factors:
                zf = z_by_factor[factor]
                value = 0.0
                if symbol in zf.columns and last_date in zf.index:
                    cell = zf.at[last_date, symbol]
                    if pd.notna(cell):
                        value = float(cell)
                loading[factor] = value
            # The symbol is kept with whatever loadings it formed, including
            # all-0.0 from a degenerate no-spread cross-section: a 0 loading
            # is a valid neutral reading and contributes nothing to the net
            # exposure (so it can never trip the band). Symbols too short to
            # appear in raw_panel at all are already omitted upstream.
            result[symbol] = loading
        return result

    # ------------------------------------------------------------------
    # Book net factor exposure (dollar-weighted, as a fraction of capital)
    # ------------------------------------------------------------------
    def book_exposure(self, per_symbol_value: Dict[str, float],
                      loadings: Dict[str, Dict[str, float]],
                      desk_capital: float) -> Dict[str, float]:
        """Net factor exposure of a book, as a FRACTION of desk capital.

            net_factor[f] = sum_s value[s] * loading[s][f] / desk_capital

        ``per_symbol_value`` is the book's SIGNED dollar exposure per symbol
        (long positive, short negative — exactly the
        ``CentralRiskBook.compute_exposures`` ``per_symbol`` shape). A symbol
        with no entry in ``loadings`` (omitted because its factors were
        unformable that day) contributes 0 on every factor — neutral, never
        a false reading. A factor not present in a symbol's loading dict is
        likewise treated as 0 on that axis.

        Returns one fraction per configured factor (always all of
        ``self.factors``, defaulting to 0.0). GRACEFUL: ``desk_capital <= 0``
        yields all-zero exposures (no meaningful base to normalize against),
        so the band gate passes rather than dividing by zero.
        """
        net: Dict[str, float] = {factor: 0.0 for factor in self.factors}
        if not per_symbol_value or desk_capital <= 0:
            return net
        for symbol, value in per_symbol_value.items():
            if value == 0.0:
                continue
            symbol_loadings = loadings.get(symbol)
            if not symbol_loadings:
                continue  # neutral (0 loading on every factor)
            for factor in self.factors:
                load = symbol_loadings.get(factor, 0.0)
                if load:
                    net[factor] += value * load
        for factor in self.factors:
            net[factor] /= desk_capital
        return net

    # ------------------------------------------------------------------
    # Candidate book net factor exposure (incremental, for the risk book)
    # ------------------------------------------------------------------
    def candidate_exposure(self, per_symbol_value: Dict[str, float],
                           candidate_symbol: str,
                           candidate_value: float,
                           loadings: Dict[str, Dict[str, float]],
                           desk_capital: float) -> Dict[str, float]:
        """Net factor exposure of the HELD book PLUS a candidate position.

        Evaluates :meth:`book_exposure` on the held ``per_symbol_value`` with
        ``candidate_symbol`` overridden to ``candidate_value`` — i.e. the net
        factor exposure the book WOULD have if the candidate's signed dollar
        exposure were ``candidate_value``. The risk book passes the running
        per-symbol state (so prior approvals in the same batch are already
        committed) and the candidate's post-trade per-symbol dollar exposure,
        giving the CUMULATIVE candidate book exposure the band gate checks.

        Pure and non-mutating: the held dict is copied before the override.
        """
        candidate_book = dict(per_symbol_value)
        candidate_book[candidate_symbol] = candidate_value
        return self.book_exposure(candidate_book, loadings, desk_capital)


__all__ = ['FactorRiskModel']
