"""
Walk-forward neural models: a feed-forward MLP and an LSTM sequence model.

Both implement the EXACT public contract of
``ml_model.GradientBoostingModel`` (the golden baseline):

    * predict() returns a per-symbol CENTERED score ``P(up tomorrow) - 0.5``
      (positive = long-signal strength, per the WalkForwardModel protocol);
    * predict() returns ``{}`` when the model is unfitted, and skips
      individual symbols whose feature rows are insufficient/NaN;
    * fit() degrades gracefully — fewer than 2 samples or fewer than 2
      label classes leaves the model unfitted (predict returns ``{}``) with
      a WARNING, never an exception (mirrors GradientBoostingModel,
      boosting.LightGBMModel, and the regime.py degrade pattern). Any torch
      training failure is caught and degrades to unfitted, never crashing a
      run.

Label alignment is identical to the baseline:
``label[i] = sign(close[i+1]/close[i] - 1)`` and the final row of each frame
is excluded from training (no next-day close) but is still a valid
predict-time feature row.

Features come from the shared :mod:`desks.features` library (the extended
set: baseline ml_model columns plus ret_5/ret_10/vol_20/momentum_10/
zscore_20), so these models share one feature source of truth.

LEAKAGE-CRITICAL STANDARDIZATION: both models standardize features using the
TRAIN-window mean/std computed in ``fit`` and stored on the model, then
re-apply those exact train statistics at predict time. The scaler is fitted
on the training rows ONLY — never on predict-time data — so no predict-time
information leaks into the fitted parameters.

torch is an OPTIONAL dependency (the CPU/MPS wheel is large, so it lives in
the ``ai`` extra alongside lightgbm, NOT in the base image). The import is
guarded so that importing this module — and therefore desks.models and the
registry — never hard-fails when torch is missing. The chosen degrade policy
mirrors boosting.LightGBMModel EXACTLY: the model CONSTRUCTOR raises a clear
ImportError (so a misconfigured install fails loudly and early at model
construction, via the registry's build_model), while everything that merely
IMPORTS this module keeps working.

DETERMINISM is mandatory: every fit/predict pins torch RNG/threading
(torch.manual_seed, deterministic algorithms with warn_only, a single thread,
CPU device) for the duration of its torch work, so identical input frames
yield byte-identical scores across runs. CPU is the only reproducible path —
MPS/CUDA are deliberately never used. The prior process-global torch state
(thread count + deterministic-algorithms flag) is SAVED and RESTORED around
each call, so these models never permanently mutate the torch runtime of a
long-lived process (e.g. a GUI server) that may run other torch work later.
(PYTHONHASHSEED is a launch-time setting — exported before the interpreter
starts, e.g. via start.sh — not something a model enforces at fit time; no
numeric result here depends on hash ordering regardless.)
"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from desks import features as feature_lib
from desks.walk_forward import WalkForwardModel

logger = logging.getLogger(__name__)

# torch is an OPTIONAL dependency (large CPU/MPS wheel). The import is
# guarded so importing this module — and therefore desks.features,
# desks.models, and the registry — never hard-fails when the wheel is
# missing. The degrade policy mirrors boosting.LightGBMModel: the
# CONSTRUCTOR raises a clear ImportError (a misconfigured install fails
# loudly and early at model construction, where build_model is the one place
# these are built for desk runs), while everything that merely IMPORTS this
# module keeps working.
try:
    import torch
    from torch import nn

    _TORCH_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 - any import failure disables torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'
    logger.warning(
        'torch unavailable (%s) — MLPModel/SequenceModel will raise on '
        'construction; install it (`pip install .[ai]`, torch==2.12.1) to '
        'enable them. The rest of desks.models is unaffected.',
        _TORCH_IMPORT_ERROR)

#: Feature column order shared by both models (the extended set), reused
#: from the shared library so all walk-forward models share one source of
#: truth for features.
FEATURE_COLUMNS: Sequence[str] = feature_lib.FEATURE_COLUMNS


def _feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Per-row feature frame from the shared library (extended set)."""
    return feature_lib.extended_feature_frame(data)


@contextlib.contextmanager
def _deterministic_torch(seed: int):
    """Pin torch to reproducible CPU settings for one fit/predict, then RESTORE
    the prior global torch state on exit.

    Yields the CPU device (the only reproducible one — MPS/CUDA kernels are
    nondeterministic, so the GPU is never touched). ``warn_only=True`` lets an
    unrelated torch op that lacks a deterministic CPU kernel degrade to a
    warning instead of raising; the prior thread count and deterministic-
    algorithms state are saved and restored so a model run never permanently
    single-threads the process or hard-pins the deterministic flag for other
    torch work that may follow.
    """
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        yield torch.device('cpu')
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


def _build_training_rows(
        train_data: Dict[str, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled per-row (X, y) across symbols with the golden label alignment.

    For each symbol, feature rows are aligned to
    ``label[i] = 1 if close[i+1]/close[i] - 1 > 0 else 0``; the final row of
    each frame has no next-day return and is dropped (the off-by-one the
    label demands — identical to GradientBoostingModel.build_training_set).
    """
    x_parts, y_parts = [], []
    for _symbol, data in train_data.items():
        if data is None or data.empty or 'close' not in data.columns:
            continue
        frame = _feature_frame(data)
        if frame.empty:
            continue
        close = data['close']
        next_return = close.shift(-1) / close - 1.0
        labels = (next_return > 0).astype(int)
        usable = frame.index.intersection(next_return.dropna().index)
        if usable.empty:
            continue
        x_parts.append(frame.loc[usable].to_numpy(dtype=float))
        y_parts.append(labels.loc[usable].to_numpy(dtype=int))

    if not x_parts:
        return (np.empty((0, len(FEATURE_COLUMNS))), np.empty((0,)))
    return np.vstack(x_parts), np.concatenate(y_parts)


class _StandardScaler:
    """Train-window mean/std standardizer (leakage-critical).

    Fitted on the training matrix ONLY in :meth:`fit`; the stored mean/std
    are re-applied verbatim at predict time. A zero-std (constant) column is
    given unit std so transform never divides by zero (the centered column is
    then all zeros, contributing no signal — the correct degenerate result).

    Standardized values are clipped to ``[-clip, +clip]`` (default 5 sigma)
    so a fat-tailed input — an earnings gap or regime break that lands many
    standard deviations out — cannot dominate the network's first layer.
    ``clip=None`` disables the clip. Deterministic; preserves the in-run
    byte-identical guarantee.
    """

    def __init__(self, clip: Optional[float] = 5.0) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.clip = clip

    def fit(self, x_matrix: np.ndarray) -> "_StandardScaler":
        self.mean_ = x_matrix.mean(axis=0)
        std = x_matrix.std(axis=0)
        # Avoid divide-by-zero on constant (and near-constant) columns; such a
        # column maps to 0 after centering. Threshold on a small epsilon (the
        # sklearn StandardScaler convention) rather than exact zero, so a tiny
        # but nonzero std can't blow standardized values up to ~1e19.
        std[std < 1e-8] = 1.0
        self.std_ = std
        return self

    def transform(self, x_matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('_StandardScaler.transform before fit')
        z = (x_matrix - self.mean_) / self.std_
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z


# ----------------------------------------------------------------------
# Feed-forward MLP
# ----------------------------------------------------------------------
class MLPModel(WalkForwardModel):
    """Feed-forward neural net predicting next-day direction.

    Same public contract as GradientBoostingModel (centered score, ``{}``
    when unfitted, graceful degrade). Uses the shared extended feature set,
    pooling per-row features across symbols exactly like the baseline.

    Architecture: per-row extended features -> stacked Linear+ReLU hidden
    layers (default ``[32, 16]``) -> 2-logit output. Trained with Adam and
    cross-entropy for a small fixed number of epochs (default 40) in full
    batch on CPU, so tests stay fast.

    LEAKAGE-CRITICAL: features are standardized with a scaler fitted on the
    TRAIN window only; the same train mean/std are re-applied at predict.

    DEGRADE POLICY (mirrors boosting.LightGBMModel EXACTLY): the CONSTRUCTOR
    raises a clear ``ImportError`` when torch is missing — a missing optional
    dependency is a deploy-time misconfiguration the operator should see
    immediately (build_model is the one place this is constructed for desk
    runs). Once constructed, fit() degrades gracefully on bad/insufficient
    data and catches any training failure; importing this module never
    raises, only constructing MLPModel without torch does.

    DETERMINISM: seed + single thread + deterministic algorithms + CPU are
    pinned before model init AND before training; weights are initialised
    under a fixed manual seed, training is full-batch (no unseeded shuffle),
    so identical input -> byte-identical scores.
    """

    def __init__(self, hidden_sizes: Sequence[int] = (32, 16),
                 epochs: int = 40, learning_rate: float = 0.01,
                 random_state: int = 42):
        if torch is None:
            raise ImportError(
                'MLPModel requires the torch package '
                f'({_TORCH_IMPORT_ERROR}); install torch==2.12.1 '
                '(`pip install .[ai]`) to use this model.')
        self.hidden_sizes = tuple(hidden_sizes)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._net: Optional["nn.Module"] = None
        self._scaler = _StandardScaler()
        self._fitted = False

    def _build_net(self, n_features: int) -> "nn.Module":
        """A small MLP: Linear+ReLU per hidden size, then a 2-logit head."""
        layers: List["nn.Module"] = []
        in_dim = n_features
        for width in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Train on pooled rows across symbols; degrade gracefully.

        With fewer than 2 samples or a single label class the net cannot
        train — the model is left unfitted (predict returns ``{}``) with a
        WARNING until a later refit succeeds. Any torch training failure is
        caught and degrades to unfitted, never crashing a run.
        """
        x_matrix, y = _build_training_rows(train_data)
        if len(x_matrix) < 2 or len(np.unique(y)) < 2:
            self._fitted = False
            logger.warning(
                "MLPModel.fit skipped: %d samples, %d classes "
                "(need >= 2 of each)", len(x_matrix), len(np.unique(y)))
            return
        try:
            with _deterministic_torch(self.random_state) as device:
                scaler = _StandardScaler().fit(x_matrix)
                x_std = scaler.transform(x_matrix)
                x_tensor = torch.tensor(x_std, dtype=torch.float32,
                                        device=device)
                y_tensor = torch.tensor(y, dtype=torch.long, device=device)

                # Seed immediately before init so weight init is fixed.
                torch.manual_seed(self.random_state)
                net = self._build_net(x_matrix.shape[1]).to(device)
                net.train()
                optimizer = torch.optim.Adam(net.parameters(),
                                             lr=self.learning_rate)
                loss_fn = nn.CrossEntropyLoss()
                for _epoch in range(self.epochs):
                    optimizer.zero_grad()
                    logits = net(x_tensor)
                    loss = loss_fn(logits, y_tensor)
                    loss.backward()
                    optimizer.step()
        except Exception as exc:  # noqa: BLE001 - never crash a run
            self._fitted = False
            logger.warning("MLPModel.fit failed (%s); staying unfitted", exc)
            return

        net.eval()
        self._net = net
        self._scaler = scaler
        self._fitted = True
        logger.debug("MLPModel fitted on %d pooled samples", len(x_matrix))

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Per-symbol centered score ``P(up tomorrow) - 0.5``.

        Each symbol's latest extended feature row is standardized with the
        stored train scaler and pushed through the net; softmax's up-class
        probability is centered (``- 0.5``). Symbols with insufficient/NaN
        feature rows are skipped; ``{}`` when never successfully fitted.
        """
        if not self._fitted or self._net is None:
            return {}
        scores: Dict[str, float] = {}
        self._net.eval()
        with _deterministic_torch(self.random_state) as device, \
                torch.no_grad():
            for symbol, frame in data.items():
                if frame is None or frame.empty or 'close' not in frame.columns:
                    continue
                features = _feature_frame(frame)
                if features.empty:
                    continue
                latest = features.iloc[[-1]].to_numpy(dtype=float)
                x_std = self._scaler.transform(latest)
                x_tensor = torch.tensor(x_std, dtype=torch.float32,
                                        device=device)
                logits = self._net(x_tensor)
                prob_up = float(torch.softmax(logits, dim=1)[0, 1].item())
                scores[symbol] = prob_up - 0.5
        return scores


# ----------------------------------------------------------------------
# LSTM sequence model
# ----------------------------------------------------------------------
class SequenceModel(WalkForwardModel):
    """LSTM over a lookback window of per-row features predicting direction.

    Same public contract as GradientBoostingModel (centered score, ``{}``
    when unfitted, graceful degrade). Uses the shared extended feature set.

    For each symbol the extended feature frame is cut into sliding windows of
    ``lookback`` consecutive rows (default 20). The label for a window is the
    direction of the day AFTER the window's last row — i.e. the golden
    ``label[i] = sign(close[i+1]/close[i] - 1)`` evaluated at the window-end
    row, so a window ending on the final frame row (which has no next-day
    close) contributes NO training sample (it is the predict-time window).

    Architecture: ``lookback x n_features`` sequence -> single-layer LSTM
    (default hidden 16) -> last hidden state -> Linear -> 2 logits. Trained
    with Adam + cross-entropy for a small fixed number of epochs in full
    batch on CPU, so tests stay fast.

    LEAKAGE-CRITICAL: features are standardized with a scaler fitted on the
    TRAIN windows only; the same train mean/std are re-applied at predict.

    DEGRADE POLICY (mirrors boosting.LightGBMModel EXACTLY): the CONSTRUCTOR
    raises a clear ``ImportError`` when torch is missing; once constructed
    fit() degrades gracefully (WARNING, stays unfitted) when there are fewer
    than 2 windows or a single label class, and catches any training
    failure. Importing this module never raises.

    DETERMINISM: seed + single thread + deterministic algorithms + CPU are
    pinned before model init AND before training; weights are initialised
    under a fixed manual seed, training is full-batch (no unseeded shuffle),
    so identical input -> byte-identical scores.
    """

    def __init__(self, lookback: int = 20, hidden_size: int = 16,
                 epochs: int = 30, learning_rate: float = 0.01,
                 random_state: int = 42):
        if torch is None:
            raise ImportError(
                'SequenceModel requires the torch package '
                f'({_TORCH_IMPORT_ERROR}); install torch==2.12.1 '
                '(`pip install .[ai]`) to use this model.')
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._net: Optional["nn.Module"] = None
        self._scaler = _StandardScaler()
        self._n_features = len(FEATURE_COLUMNS)
        self._fitted = False

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------
    def _symbol_windows(
            self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding ``lookback`` windows for one symbol with golden labels.

        Returns ``(X, y)`` where ``X`` has shape
        ``(n_windows, lookback, n_features)`` and ``y[k]`` is the next-day
        direction at window k's LAST row. A window whose last row is the
        frame's final row has no next-day close and is dropped (the off-by-one
        the label demands) — that window is the predict-time input instead.
        """
        empty = (np.empty((0, self.lookback, self._n_features)),
                 np.empty((0,), dtype=int))
        if data is None or data.empty or 'close' not in data.columns:
            return empty
        frame = _feature_frame(data)
        if len(frame) <= self.lookback:
            return empty
        close = data['close']
        # next_return aligned onto the (post-warm-up) feature index.
        next_return = (close.shift(-1) / close - 1.0).reindex(frame.index)
        feats = frame.to_numpy(dtype=float)
        labels = (next_return > 0).astype(int).to_numpy()
        has_label = next_return.notna().to_numpy()

        x_parts, y_parts = [], []
        # Window k spans rows [k, k+lookback); its label is the next-day
        # direction at the window-END row index (k + lookback - 1).
        for end in range(self.lookback - 1, len(frame)):
            if not has_label[end]:
                continue  # final row: no next-day close -> not a train window
            start = end - self.lookback + 1
            x_parts.append(feats[start:end + 1])
            y_parts.append(int(labels[end]))
        if not x_parts:
            return empty
        return np.stack(x_parts), np.asarray(y_parts, dtype=int)

    def _latest_window(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """The most recent complete ``lookback`` feature window for predict,
        shape ``(lookback, n_features)``; ``None`` if too short."""
        if data is None or data.empty or 'close' not in data.columns:
            return None
        frame = _feature_frame(data)
        if len(frame) < self.lookback:
            return None
        return frame.to_numpy(dtype=float)[-self.lookback:]

    def _build_net(self) -> "nn.Module":
        return _LSTMNet(self._n_features, self.hidden_size)

    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Train on sliding windows pooled across symbols; degrade gracefully.

        With fewer than 2 windows or a single label class the net cannot
        train — the model is left unfitted (predict returns ``{}``) with a
        WARNING. Any torch training failure is caught and degrades to
        unfitted, never crashing a run.
        """
        x_parts, y_parts = [], []
        for _symbol, data in train_data.items():
            x_sym, y_sym = self._symbol_windows(data)
            if len(x_sym):
                x_parts.append(x_sym)
                y_parts.append(y_sym)
        if x_parts:
            x_windows = np.concatenate(x_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
        else:
            x_windows = np.empty((0, self.lookback, self._n_features))
            y = np.empty((0,), dtype=int)

        if len(x_windows) < 2 or len(np.unique(y)) < 2:
            self._fitted = False
            logger.warning(
                "SequenceModel.fit skipped: %d windows, %d classes "
                "(need >= 2 of each)", len(x_windows), len(np.unique(y)))
            return
        try:
            with _deterministic_torch(self.random_state) as device:
                # Fit the scaler on flattened feature rows across all windows
                # (train rows only — leakage-critical), then standardize.
                flat = x_windows.reshape(-1, self._n_features)
                scaler = _StandardScaler().fit(flat)
                x_std = scaler.transform(flat).reshape(x_windows.shape)
                x_tensor = torch.tensor(x_std, dtype=torch.float32,
                                        device=device)
                y_tensor = torch.tensor(y, dtype=torch.long, device=device)

                torch.manual_seed(self.random_state)
                net = self._build_net().to(device)
                net.train()
                optimizer = torch.optim.Adam(net.parameters(),
                                             lr=self.learning_rate)
                loss_fn = nn.CrossEntropyLoss()
                for _epoch in range(self.epochs):
                    optimizer.zero_grad()
                    logits = net(x_tensor)
                    loss = loss_fn(logits, y_tensor)
                    loss.backward()
                    optimizer.step()
        except Exception as exc:  # noqa: BLE001 - never crash a run
            self._fitted = False
            logger.warning("SequenceModel.fit failed (%s); staying unfitted",
                           exc)
            return

        net.eval()
        self._net = net
        self._scaler = scaler
        self._fitted = True
        logger.debug("SequenceModel fitted on %d windows", len(x_windows))

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Per-symbol centered score ``P(up tomorrow) - 0.5``.

        Each symbol's latest complete ``lookback`` window is standardized
        with the stored train scaler and pushed through the LSTM; softmax's
        up-class probability is centered (``- 0.5``). Symbols too short to
        form a window are skipped; ``{}`` when never successfully fitted.
        """
        if not self._fitted or self._net is None:
            return {}
        scores: Dict[str, float] = {}
        self._net.eval()
        with _deterministic_torch(self.random_state) as device, \
                torch.no_grad():
            for symbol, frame in data.items():
                window = self._latest_window(frame)
                if window is None:
                    continue
                x_std = self._scaler.transform(
                    window.reshape(-1, self._n_features)
                ).reshape(1, self.lookback, self._n_features)
                x_tensor = torch.tensor(x_std, dtype=torch.float32,
                                        device=device)
                logits = self._net(x_tensor)
                prob_up = float(torch.softmax(logits, dim=1)[0, 1].item())
                scores[symbol] = prob_up - 0.5
        return scores


if torch is not None:

    class _LSTMNet(nn.Module):
        """Single-layer LSTM whose last hidden state feeds a 2-logit head."""

        def __init__(self, n_features: int, hidden_size: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features,
                                hidden_size=hidden_size,
                                num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden_size, 2)

        def forward(self, x):  # x: (batch, lookback, n_features)
            output, _ = self.lstm(x)
            last_hidden = output[:, -1, :]  # last timestep's hidden state
            return self.head(last_hidden)

else:  # pragma: no cover - torch-missing fallback so the name always exists
    _LSTMNet = None  # type: ignore[assignment]


__all__ = ['MLPModel', 'SequenceModel', 'FEATURE_COLUMNS']
