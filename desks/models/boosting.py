"""
Walk-forward ML models: LightGBM histogram booster and a stacking meta-model.

Both implement the EXACT public contract of
``ml_model.GradientBoostingModel`` (the golden baseline):

    * predict() returns a per-symbol CENTERED score ``P(up tomorrow) - 0.5``
      (positive = long-signal strength, per the WalkForwardModel protocol);
    * predict() returns ``{}`` when the model is unfitted, and skips
      individual symbols whose feature rows are insufficient/NaN;
    * fit() degrades gracefully — fewer than 2 samples or fewer than 2
      label classes leaves the model unfitted (predict returns ``{}``) with
      a WARNING, never an exception (mirrors GradientBoostingModel and the
      regime.py degrade pattern).

Label alignment is identical to the baseline:
``label[i] = sign(close[i+1]/close[i] - 1)`` and the final row of each frame
is excluded from training (no next-day close) but is still a valid
predict-time feature row.

Features come from the shared :mod:`desks.features` library (the extended
set: baseline ml_model columns plus ret_5/ret_10/vol_20/momentum_10/
zscore_20), so these models share one feature source of truth.

DETERMINISM is mandatory: every model below pins all RNG so identical input
frames yield byte-identical scores across runs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from desks import features as feature_lib
from desks.ml_model import GradientBoostingModel
from desks.walk_forward import WalkForwardModel

logger = logging.getLogger(__name__)

# LightGBM is an OPTIONAL dependency (macOS additionally needs
# ``brew install libomp``). The import is guarded so that importing this
# module — and therefore desks.features, desks.models, and the registry —
# never hard-fails when the wheel is missing. The chosen degrade policy
# (documented on LightGBMModel) mirrors regime.py's hmmlearn-style graceful
# pattern: the CONSTRUCTOR raises a clear ImportError (so a misconfigured
# install fails loudly and early at model construction, not silently at
# predict), while everything that merely IMPORTS this module keeps working.
try:
    import lightgbm as lgb
    _LIGHTGBM_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 - any import failure disables LightGBM
    lgb = None
    _LIGHTGBM_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'
    logger.warning(
        'lightgbm unavailable (%s) — LightGBMModel will raise on '
        'construction; install it (and `brew install libomp` on macOS) to '
        'enable it. The rest of desks.models is unaffected.',
        _LIGHTGBM_IMPORT_ERROR)

#: Feature column order shared by both models (the extended set).
FEATURE_COLUMNS: Sequence[str] = feature_lib.FEATURE_COLUMNS


def _feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Per-row feature frame from the shared library (extended set)."""
    return feature_lib.extended_feature_frame(data)


#: Pooled (X, y) training-matrix builder — the single source of truth shared
#: with the neural models (bodies were byte-for-byte identical), now
#: :func:`desks.features.build_training_set`. Kept under the local name so
#: the ``_build_training_set(train_data)`` call site is unchanged.
_build_training_set = feature_lib.build_training_set


class LightGBMModel(WalkForwardModel):
    """LightGBM histogram gradient booster predicting next-day direction.

    Same public contract as GradientBoostingModel (centered score, ``{}``
    when unfitted, graceful degrade). Uses the shared extended feature set.

    DEGRADE POLICY (documented choice): the CONSTRUCTOR raises a clear
    ``ImportError`` when the lightgbm wheel is missing. Rationale: a missing
    optional dependency is a deploy-time misconfiguration the operator
    should see immediately and loudly (the registry's ``build_model`` is the
    one place this is constructed for desk runs), not a silent
    always-unfitted model that quietly never gates anything. Once
    constructed, fit() still degrades gracefully on bad/insufficient data
    exactly like the baseline. Importing this module never raises — only
    constructing LightGBMModel without the wheel does.

    DETERMINISM: the booster is built with ``deterministic=True``,
    ``num_threads=1``, ``force_row_wise=True``, ``seed=42`` (plus
    ``bagging_seed``/``feature_fraction_seed``/``data_random_seed`` pinned
    to the same seed and ``verbose=-1``). With bagging/feature subsampling
    left at their defaults (off), these seeds are belt-and-suspenders; they
    guarantee identical input -> identical output even if a future tweak
    enables subsampling.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 3,
                 learning_rate: float = 0.1, random_state: int = 42):
        if lgb is None:
            raise ImportError(
                'LightGBMModel requires the lightgbm package '
                f'({_LIGHTGBM_IMPORT_ERROR}); install lightgbm==4.6.0 '
                "(and `brew install libomp` on macOS) to use this model.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._booster: Optional["lgb.LGBMClassifier"] = None
        self._fitted = False

    def _new_classifier(self) -> "lgb.LGBMClassifier":
        """A fully-seeded, single-threaded, deterministic LGBMClassifier."""
        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            # --- determinism pins ---
            deterministic=True,
            force_row_wise=True,
            num_threads=1,
            seed=self.random_state,
            bagging_seed=self.random_state,
            feature_fraction_seed=self.random_state,
            data_random_seed=self.random_state,
            verbose=-1,
        )

    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Train on pooled rows across symbols; degrade gracefully.

        With fewer than 2 samples or a single label class the booster
        cannot train — the model is left unfitted (predict returns ``{}``)
        with a WARNING until a later refit succeeds. Any LightGBM training
        failure is also caught and degrades to unfitted (parity with the
        regime.py robustness pattern), never crashing a run.
        """
        x_matrix, y = _build_training_set(train_data)
        if len(x_matrix) < 2 or len(np.unique(y)) < 2:
            self._fitted = False
            logger.warning(
                "LightGBMModel.fit skipped: %d samples, %d classes "
                "(need >= 2 of each)", len(x_matrix), len(np.unique(y)))
            return
        booster = self._new_classifier()
        try:
            booster.fit(x_matrix, y)
        except Exception as exc:  # noqa: BLE001 - never crash a run
            self._fitted = False
            logger.warning("LightGBMModel.fit failed (%s); staying unfitted",
                           exc)
            return
        self._booster = booster
        self._fitted = True
        logger.debug("LightGBMModel fitted on %d pooled samples",
                     len(x_matrix))

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Per-symbol centered score ``P(up tomorrow) - 0.5``.

        Symbols with insufficient/NaN feature rows are skipped; ``{}`` when
        the model never trained successfully.

        Feature vectors come from the O(1) enriched-column fast path when
        a frame qualifies (features.fast_last_extended_row) and the full
        O(history) _feature_frame rebuild otherwise; all rows are then
        scored with a SINGLE predict_proba call (the booster is built with
        deterministic=True / num_threads=1, and tree traversal is per-row
        independent, so batching is bit-identical to per-symbol calls —
        the same argument as ml_model.GradientBoostingModel.predict).
        """
        if not self._fitted or self._booster is None:
            return {}
        symbols = []
        rows = []
        for symbol, frame in data.items():
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            vector = feature_lib.fast_last_extended_row(frame)
            if vector is None:
                features = _feature_frame(frame)
                if features.empty:
                    continue
                vector = features.iloc[[-1]].to_numpy(dtype=float)[0]
            symbols.append(symbol)
            rows.append(vector)
        if not rows:
            return {}
        probs = self._booster.predict_proba(np.vstack(rows))[:, 1]
        return {symbol: float(prob_up) - 0.5
                for symbol, prob_up in zip(symbols, probs)}


class StackingMetaModel(WalkForwardModel):
    """Stacking ensemble: base learners combined by a logistic meta-learner.

    Default base learners are ``[GradientBoostingModel(), LightGBMModel()]``
    (LightGBM is dropped automatically if its wheel is missing). A
    scikit-learn ``LogisticRegression`` (fixed ``solver='lbfgs'``,
    ``random_state=42``) is the meta-learner that blends the base learners'
    out-of-fold probabilities.

    NO-LEAKAGE META-FEATURE GENERATION: the train window is split by TIME
    into two contiguous halves. Each base learner is trained on the EARLY
    half and produces probabilities for the LATE half — out-of-fold
    predictions only (a base learner never predicts rows it trained on). The
    meta-learner trains on those late-half base probabilities vs the
    late-half labels. The base learners are THEN refit on the FULL train
    window for use at predict time. No row ever serves as both a base
    learner's training row and its meta-feature row.

    Same public contract (centered score, ``{}`` unfitted, graceful
    degrade): if a base learner is unavailable or fails to fit it is dropped
    from the ensemble for that fit; if NONE remain usable the meta-model
    stays unfitted with a WARNING.

    DETERMINISM: the time split is deterministic, every base learner pins
    its own RNG (GradientBoosting random_state=42, LightGBM seed=42), and
    the LogisticRegression is seeded — identical input -> identical output.
    """

    #: Fraction of the (time-ordered) train window used to TRAIN the base
    #: learners for the out-of-fold meta-feature pass; the remainder is the
    #: out-of-fold block the meta-learner learns to blend on.
    _OOF_TRAIN_FRACTION = 0.5

    #: Minimum usable label rows needed in BOTH the oof-train and
    #: oof-holdout blocks (each must also carry both classes) before a
    #: stacked fit is attempted.
    _MIN_OOF_ROWS = 2

    def __init__(self,
                 base_models: Optional[Sequence[WalkForwardModel]] = None,
                 random_state: int = 42):
        self.random_state = random_state
        if base_models is None:
            base_models = self._default_base_models()
        self._base_models: List[WalkForwardModel] = list(base_models)
        self._meta = LogisticRegression(solver='lbfgs',
                                        random_state=random_state)
        # Base learners that successfully refit on the FULL window, in the
        # column order the meta-learner expects.
        self._active_bases: List[WalkForwardModel] = []
        self._fitted = False

    @staticmethod
    def _default_base_models() -> List[WalkForwardModel]:
        """``[GradientBoostingModel(), LightGBMModel()]``; LightGBM omitted
        (with a WARNING) when its wheel is missing so the meta-model still
        works as a GBM-only stack."""
        bases: List[WalkForwardModel] = [GradientBoostingModel()]
        if lgb is not None:
            bases.append(LightGBMModel())
        else:
            logger.warning(
                "StackingMetaModel: lightgbm unavailable — base learners "
                "reduced to [GradientBoostingModel]")
        return bases

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _time_split(train_data: Dict[str, pd.DataFrame],
                    fraction: float) -> Tuple[Dict[str, pd.DataFrame],
                                              Dict[str, pd.DataFrame]]:
        """Split each symbol's frame by TIME into (early, late) contiguous
        blocks at ``fraction`` of its rows. Per-symbol so every symbol
        contributes to both blocks; empties are dropped."""
        early: Dict[str, pd.DataFrame] = {}
        late: Dict[str, pd.DataFrame] = {}
        for symbol, frame in train_data.items():
            if frame is None or frame.empty:
                continue
            cut = int(len(frame) * fraction)
            if cut < 1 or cut >= len(frame):
                continue
            early[symbol] = frame.iloc[:cut]
            late[symbol] = frame.iloc[cut:]
        return early, late

    @staticmethod
    def _batch_scorer(base: WalkForwardModel):
        """Return ``(estimator, feature_frame_fn)`` for a base learner whose
        fitted estimator and matching per-row feature frame are both known,
        else ``None``.

        The estimator and the feature frame MUST be the pair the base trained
        with — the baseline ``GradientBoostingModel`` trains its
        ``_classifier`` on its OWN (5-column) ``_feature_frame``, whereas
        ``LightGBMModel`` trains its ``_booster`` on the module-level
        (10-column extended) ``_feature_frame``. Pairing them explicitly
        keeps the batched fast path byte-identical to the base's public
        ``predict``; mismatching them would feed the wrong feature width.

        Returns None for any other base learner, so the caller falls back to
        the per-row public-``predict`` path and arbitrary ``WalkForwardModel``
        base learners keep working unchanged.

        Crucially this also returns None when the base learner is UNFITTED
        (e.g. it degraded on single-class labels): the fast path would
        otherwise call ``predict_proba`` on an unfitted estimator and raise,
        whereas the base's public ``predict`` short-circuits to ``{}``. We
        mirror that ``_fitted`` guard so a degraded base yields no OOF rows
        (the caller then drops it) — identical to the per-row path.
        """
        # An unfitted base learner has no usable estimator: its public
        # predict() returns {} for every row, so the OOF result is empty.
        if not getattr(base, '_fitted', False):
            return None
        # LightGBMModel: _booster + module-level extended _feature_frame.
        booster = getattr(base, '_booster', None)
        if booster is not None and hasattr(booster, 'predict_proba'):
            return booster, _feature_frame
        # GradientBoostingModel (baseline): _classifier + its OWN bound
        # _feature_frame (the baseline 5-column set).
        classifier = getattr(base, '_classifier', None)
        own_ff = getattr(base, '_feature_frame', None)
        if classifier is not None and hasattr(classifier, 'predict_proba') \
                and callable(own_ff):
            return classifier, own_ff
        return None

    def _oof_meta_features(
            self, base: WalkForwardModel,
            early: Dict[str, pd.DataFrame],
            late: Dict[str, pd.DataFrame]) -> Optional[pd.Series]:
        """Out-of-fold P(up) from one base learner: train on `early`,
        predict each late row (per symbol). Returns a Series indexed by
        (symbol, timestamp) of centered-score + 0.5 = P(up), or None if the
        base learner could not fit the early block / produced nothing.

        NO-LEAKAGE: the base learner is trained ONLY on `early`, so every
        `late` row is out-of-fold (a base never scores a row it trained on),
        and `late` itself carries no rows beyond this fold.

        PERF (M1): every feature is backward-looking, so a feature row at
        timestamp t is identical whether computed from the prefix `<= t` or
        from the full `late` block (rows > t never enter a trailing window).
        We therefore build each symbol's feature frame ONCE over the full
        `late` block and score all its rows in a single ``predict_proba`` —
        O(rows) instead of the old O(rows²) growing-slice-per-row predict.
        This is NUMERICALLY EQUIVALENT to the per-row public path
        (``_feature_frame(frame.loc[:t]).iloc[-1] == _feature_frame(frame)
        .loc[t]``). For an unrecognised base learner (no fitted estimator
        exposed) we fall back to that per-row path verbatim.
        """
        try:
            base.fit(early)
        except Exception as exc:  # noqa: BLE001 - never crash a run
            logger.warning(
                "StackingMetaModel: base learner %s raised while fitting the "
                "out-of-fold early block (%s) — dropped for this fit",
                type(base).__name__, exc)
            return None

        probs: Dict[Tuple[str, pd.Timestamp], float] = {}
        scorer = self._batch_scorer(base)
        for symbol, frame in late.items():
            if 'close' not in frame.columns:
                continue
            if scorer is None:
                # Fallback: per-row public predict (correct, O(rows²)).
                for ts in frame.index:
                    scored = base.predict({symbol: frame.loc[:ts]}, ts)
                    if symbol in scored:
                        probs[(symbol, ts)] = scored[symbol] + 0.5
                continue
            # Fast path: one feature frame for the whole late block, one
            # batched predict_proba. Each surviving row's features equal the
            # per-row prefix's last row (backward-looking transforms), so this
            # is byte-identical to the per-row path but O(rows) not O(rows²).
            estimator, feature_frame_fn = scorer
            feats = feature_frame_fn(frame)
            if feats.empty:
                continue
            proba_up = estimator.predict_proba(
                feats.to_numpy(dtype=float))[:, 1]
            for ts, p in zip(feats.index, proba_up):
                probs[(symbol, ts)] = float(p)
        if not probs:
            return None
        return pd.Series(probs)

    # ------------------------------------------------------------------
    # WalkForwardModel protocol
    # ------------------------------------------------------------------
    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Fit the stack: out-of-fold meta-features then a full-window refit.

        Degrades gracefully (WARNING, stays unfitted) when the time split,
        any base learner, or the meta-learner cannot produce a usable,
        two-class training matrix. Previously-fitted state is replaced only
        on a successful fit.
        """
        early, late = self._time_split(train_data, self._OOF_TRAIN_FRACTION)
        if not early or not late:
            self._fitted = False
            logger.warning(
                "StackingMetaModel.fit skipped: train window too short to "
                "make a time split for out-of-fold meta-features")
            return

        # Labels for the late (out-of-fold) block, per (symbol, timestamp),
        # using the golden alignment; the last row per symbol drops out.
        late_labels: Dict[Tuple[str, pd.Timestamp], int] = {}
        for symbol, frame in late.items():
            if 'close' not in frame.columns:
                continue
            close = frame['close']
            next_return = close.shift(-1) / close - 1.0
            for ts, val in next_return.items():
                if pd.notna(val):
                    late_labels[(symbol, ts)] = int(val > 0)
        if not late_labels:
            self._fitted = False
            logger.warning(
                "StackingMetaModel.fit skipped: no labelled out-of-fold rows")
            return

        # Collect out-of-fold meta-features from each base learner that can
        # fit the early block. Drop any that degrade.
        meta_columns: List[pd.Series] = []
        usable_bases: List[WalkForwardModel] = []
        for base in self._base_models:
            oof = self._oof_meta_features(base, early, late)
            if oof is None or oof.empty:
                logger.warning(
                    "StackingMetaModel: base learner %s produced no "
                    "out-of-fold predictions — dropped for this fit",
                    type(base).__name__)
                continue
            meta_columns.append(oof)
            usable_bases.append(base)

        if not usable_bases:
            self._fitted = False
            logger.warning(
                "StackingMetaModel.fit skipped: no usable base learners")
            return

        # Build the meta matrix on the (symbol, timestamp) keys common to
        # every usable base learner AND the label map.
        label_series = pd.Series(late_labels)
        meta_df = pd.concat(meta_columns, axis=1)
        meta_df.columns = range(len(usable_bases))
        aligned = meta_df.join(label_series.rename('y'), how='inner').dropna()
        if len(aligned) < self._MIN_OOF_ROWS \
                or aligned['y'].nunique() < 2:
            self._fitted = False
            logger.warning(
                "StackingMetaModel.fit skipped: %d aligned out-of-fold rows, "
                "%d classes (need >= %d rows and 2 classes)",
                len(aligned), aligned['y'].nunique(), self._MIN_OOF_ROWS)
            return

        x_meta = aligned[list(range(len(usable_bases)))].to_numpy(dtype=float)
        y_meta = aligned['y'].to_numpy(dtype=int)
        try:
            self._meta.fit(x_meta, y_meta)
        except Exception as exc:  # noqa: BLE001 - never crash a run
            self._fitted = False
            logger.warning("StackingMetaModel meta-learner fit failed (%s); "
                           "staying unfitted", exc)
            return

        # Refit the usable base learners on the FULL train window for
        # predict time (out-of-fold blending learned above; final base
        # learners use all available history). The never-crash-a-run
        # invariant means one failing sklearn/LightGBM fit must not abort
        # the stack: catch any refit exception. The meta-learner was fitted
        # on EXACTLY one column per usable base, so the active base set must
        # keep that same width — if any base fails its full refit we cannot
        # safely build the meta row (a narrower set would mismatch the
        # meta-learner's input), so the whole stack degrades to unfitted.
        refit_bases: List[WalkForwardModel] = []
        for base in usable_bases:
            try:
                base.fit(train_data)
            except Exception as exc:  # noqa: BLE001 - never crash a run
                self._fitted = False
                logger.warning(
                    "StackingMetaModel.fit skipped: base learner %s raised "
                    "during the full refit (%s) — cannot build a meta row "
                    "matching the trained meta-learner; staying unfitted",
                    type(base).__name__, exc)
                return
            refit_bases.append(base)
        self._active_bases = refit_bases
        self._fitted = True
        logger.debug("StackingMetaModel fitted: %d base learners, %d "
                     "out-of-fold meta rows", len(refit_bases), len(aligned))

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Per-symbol centered score from the stacked ensemble.

        Each active base learner scores the symbols; the per-symbol base
        probabilities feed the meta-learner, whose P(up) is centered
        (``- 0.5``). A symbol is scored only when EVERY active base learner
        produced a probability for it (the meta matrix needs all columns);
        ``{}`` when the meta-model is unfitted.
        """
        if not self._fitted or not self._active_bases:
            return {}
        base_scores: List[Dict[str, float]] = [
            base.predict(data, date) for base in self._active_bases]
        # Symbols scored by every active base learner.
        common = set(base_scores[0])
        for scored in base_scores[1:]:
            common &= set(scored)
        if not common:
            return {}
        out: Dict[str, float] = {}
        for symbol in common:
            row = np.array(
                [[scored[symbol] + 0.5 for scored in base_scores]],
                dtype=float)
            prob_up = float(self._meta.predict_proba(row)[0, 1])
            out[symbol] = prob_up - 0.5
        return out
