"""
Walk-forward fit/predict harness — leakage-proof by construction.

THE INVARIANT (enforced here, not by caller discipline):

    Every row any model ever receives — in fit() OR predict() — has
    index <= the current simulation date, and every prediction made at
    date D comes from a model whose train_end <= D.

WalkForwardController is the only object that hands data to a model. Both
entry points slice every frame to ``index <= date`` before the model sees
it (predict re-slices even if the caller already did — defense in depth, a
malicious or buggy caller passing future rows changes nothing), and fit
data is additionally capped to the trailing ``train_window_days`` rows per
symbol. Models therefore cannot leak the future even if they try to: they
hold no data beyond what fit/predict hand them.

Protocol contract for Phase 6+ implementers (HMM regime models, gradient
boosting cross-sectional models, cointegration pairs all plug in here):

    class MyModel(WalkForwardModel):
        def fit(self, train_data: dict[str, pd.DataFrame]) -> None:
            # train_data: symbol -> OHLCV(+indicator) frame, already
            # sliced/capped by the controller. Train on it; keep ONLY
            # fitted parameters — never retain the frames themselves.

        def predict(self, data: dict[str, pd.DataFrame], date) -> dict[str, float]:
            # data: symbol -> frame sliced to index <= date.
            # Return symbol -> score; positive = long signal strength.
            # Return {} (or omit symbols) when there is nothing to say.

    desk._controller = WalkForwardController(MyModel())
    # each simulated day:  controller.maybe_refit(all_data, date)
    #                      scores = controller.predict(all_data, date)
    #                      (scores is None until the first fit)

Refit cadence: the first fit happens once any symbol has at least
``min_train_days`` rows of history; thereafter the model is refit every
``refit_every_days`` distinct trading days. Each fit is recorded as a
WalkForwardFit (serialized into desk-mode backtest reports).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date as date_type
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardModel(ABC):
    """Abstract fit/predict model managed by WalkForwardController.

    Implementations must be stateless apart from fitted parameters: they
    hold NO market data outside what fit()/predict() receive, so the
    controller's slicing is the single source of temporal truth.

    One sanctioned exception: a predict-side DERIVED cache is permitted
    when every read is validated as an exact prefix/extension of the
    data the controller just handed in and every output is bit-identical
    to a stateless recompute on that handed slice (RegimeHMMModel's
    incremental forward recursion). The leakage invariant is preserved
    because the cache can never contribute information beyond the
    controller-sliced input that produced it.
    """

    @abstractmethod
    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Train on symbol -> frame data (pre-sliced by the controller)."""

    @abstractmethod
    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Score symbols at `date`: positive = long signal strength."""

    def checkpoint_spec(self) -> Dict[str, Any]:
        """Return the safe, JSON-native identity used for deterministic refit.

        Checkpoints never contain a serialized estimator.  A model that wants
        controller checkpoint support instead describes the constructor and
        runtime contract that make a refit deterministic.  Restore compares
        this value to the already-constructed model; it never imports or
        instantiates a type named by checkpoint input.
        """
        raise TypeError(
            f"{type(self).__name__} does not support safe checkpoint refits")


@dataclass
class WalkForwardFit:
    """One recorded model fit: when it happened and what it trained on."""
    fit_date: date_type
    train_start: date_type
    train_end: date_type
    n_samples: int

    def to_dict(self) -> Dict:
        """Serialize to the report shape (contract C3)."""
        return {
            'fit_date': self.fit_date.strftime('%Y-%m-%d'),
            'train_start': self.train_start.strftime('%Y-%m-%d'),
            'train_end': self.train_end.strftime('%Y-%m-%d'),
            'n_samples': self.n_samples,
        }


def _as_date(value) -> date_type:
    """Normalize a Timestamp/datetime/date to a plain datetime.date."""
    return pd.Timestamp(value).date()


_CHECKPOINT_SCHEMA_VERSION = 1
_SUPPORTED_DTYPES = frozenset({
    'bool',
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'float16', 'float32', 'float64',
})


def _canonical_json(value: Any) -> str:
    """Canonical JSON used for the checkpoint integrity digest."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint state must contain only JSON values") from exc


def _strict_date(value: Any, field: str) -> date_type:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string")
    try:
        parsed = date_type.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date string") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must be a canonical ISO date string")
    return parsed


def _frame_to_checkpoint(frame: pd.DataFrame) -> Dict[str, Any]:
    """Encode one training frame as primitive, standards-compliant JSON."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("checkpoint frames require a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("checkpoint frame indices must be sorted and unique")
    if any(not isinstance(column, str) or not column for column in frame.columns):
        raise ValueError("checkpoint frame columns must be non-empty strings")
    if not frame.columns.is_unique:
        raise ValueError("checkpoint frame columns must be unique")
    if frame.index.name is not None and not isinstance(frame.index.name, str):
        raise ValueError("checkpoint frame index names must be strings or null")

    dtypes = [str(dtype) for dtype in frame.dtypes]
    unsupported = sorted(set(dtypes).difference(_SUPPORTED_DTYPES))
    if unsupported:
        raise ValueError(
            "checkpoint frames only support fixed-width numeric/bool columns; "
            f"unsupported dtypes: {unsupported}")

    rows: List[List[Any]] = []
    for raw_row in frame.itertuples(index=False, name=None):
        row: List[Any] = []
        for value, dtype_name in zip(raw_row, dtypes):
            if pd.isna(value):
                if not dtype_name.startswith('float'):
                    raise ValueError(
                        "only floating checkpoint columns may contain nulls")
                row.append(None)
            elif dtype_name == 'bool':
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError("invalid boolean checkpoint value")
                row.append(bool(value))
            elif dtype_name.startswith(('int', 'uint')):
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError("invalid integer checkpoint value")
                row.append(int(value))
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(
                        "non-finite values are not allowed in checkpoints")
                row.append(number)
        rows.append(row)

    timezone = str(frame.index.tz) if frame.index.tz is not None else None
    if timezone is None:
        index = [timestamp.isoformat() for timestamp in frame.index]
    else:
        index = [timestamp.tz_convert('UTC').isoformat()
                 for timestamp in frame.index]
    return {
        'columns': list(frame.columns),
        'dtypes': dtypes,
        'index': index,
        'index_name': frame.index.name,
        'index_timezone': timezone,
        'rows': rows,
    }


def _frame_from_checkpoint(value: Any, symbol: str) -> pd.DataFrame:
    """Strict inverse of :func:`_frame_to_checkpoint`."""
    if not isinstance(value, Mapping):
        raise ValueError(f"training_data.{symbol} must be an object")
    required = {
        'columns', 'dtypes', 'index', 'index_name', 'index_timezone', 'rows'}
    if set(value) != required:
        raise ValueError(
            f"training_data.{symbol} fields must be {sorted(required)}")

    columns = value['columns']
    dtypes = value['dtypes']
    raw_index = value['index']
    rows = value['rows']
    index_name = value['index_name']
    timezone = value['index_timezone']
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(column, str) or not column
                   for column in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError(f"training_data.{symbol}.columns is invalid")
    if (not isinstance(dtypes, list) or len(dtypes) != len(columns)
            or any(dtype not in _SUPPORTED_DTYPES for dtype in dtypes)):
        raise ValueError(f"training_data.{symbol}.dtypes is invalid")
    if not isinstance(raw_index, list) or not isinstance(rows, list):
        raise ValueError(f"training_data.{symbol} index/rows must be arrays")
    if len(raw_index) != len(rows) or not raw_index:
        raise ValueError(
            f"training_data.{symbol} index/rows must have equal non-zero length")
    if any(not isinstance(timestamp, str) for timestamp in raw_index):
        raise ValueError(f"training_data.{symbol}.index is invalid")
    if index_name is not None and not isinstance(index_name, str):
        raise ValueError(f"training_data.{symbol}.index_name is invalid")
    if timezone is not None and (not isinstance(timezone, str) or not timezone):
        raise ValueError(f"training_data.{symbol}.index_timezone is invalid")

    try:
        if timezone is None:
            parsed_index = pd.DatetimeIndex(raw_index)
            if parsed_index.tz is not None:
                raise ValueError("naive checkpoint index contains an offset")
        else:
            parsed_index = pd.DatetimeIndex(
                pd.to_datetime(raw_index, utc=True)).tz_convert(timezone)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            f"training_data.{symbol}.index is invalid") from exc
    canonical_index = (
        [timestamp.isoformat() for timestamp in parsed_index]
        if timezone is None else
        [timestamp.tz_convert('UTC').isoformat() for timestamp in parsed_index]
    )
    if canonical_index != raw_index:
        raise ValueError(f"training_data.{symbol}.index is not canonical")
    parsed_index.name = index_name
    if (not parsed_index.is_monotonic_increasing
            or not parsed_index.is_unique):
        raise ValueError(
            f"training_data.{symbol}.index must be sorted and unique")

    decoded_columns: Dict[str, pd.Series] = {}
    for column_number, (column, dtype_name) in enumerate(zip(columns, dtypes)):
        decoded: List[Any] = []
        for row_number, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError(
                    f"training_data.{symbol}.rows[{row_number}] is invalid")
            item = row[column_number]
            if item is None:
                if not dtype_name.startswith('float'):
                    raise ValueError(
                        f"training_data.{symbol}.{column} has an invalid null")
                decoded.append(np.nan)
            elif dtype_name == 'bool':
                if type(item) is not bool:
                    raise ValueError(
                        f"training_data.{symbol}.{column} has an invalid bool")
                decoded.append(item)
            elif dtype_name.startswith(('int', 'uint')):
                if type(item) is not int:
                    raise ValueError(
                        f"training_data.{symbol}.{column} has an invalid int")
                info = np.iinfo(np.dtype(dtype_name))
                if item < info.min or item > info.max:
                    raise ValueError(
                        f"training_data.{symbol}.{column} integer overflows")
                decoded.append(item)
            else:
                if type(item) not in (int, float) or not math.isfinite(item):
                    raise ValueError(
                        f"training_data.{symbol}.{column} has an invalid float")
                decoded.append(item)
        decoded_columns[column] = pd.Series(
            decoded, index=parsed_index, dtype=dtype_name)
    return pd.DataFrame(decoded_columns, index=parsed_index)


class WalkForwardController:
    """Schedules model refits and guarantees temporal hygiene of all data.

    See the module docstring for the leakage invariant this class enforces
    by construction.
    """

    def __init__(self, model: WalkForwardModel,
                 train_window_days: int = 252,
                 refit_every_days: int = 21,
                 min_train_days: int = 120):
        self.model = model
        self.train_window_days = train_window_days
        self.refit_every_days = refit_every_days
        self.min_train_days = min_train_days
        self.fits: List[WalkForwardFit] = []
        self.last_fit_date: Optional[date_type] = None
        # Distinct trading days observed since the last fit (the engine
        # calls maybe_refit once per simulated day; duplicate calls on the
        # same date are not double-counted).
        self._days_since_fit = 0
        self._last_seen_date: Optional[date_type] = None
        self._checkpoint_supported = (
            type(model).checkpoint_spec is not WalkForwardModel.checkpoint_spec)
        # The controller, rather than the estimator, retains exactly the most
        # recent bounded fit input.  It is the only state needed to recreate a
        # deterministic estimator without deserializing executable objects.
        # Unsupported model types retain nothing, preserving their historical
        # memory profile.
        self._last_train_data: Optional[Dict[str, pd.DataFrame]] = None

    @property
    def is_fitted(self) -> bool:
        return bool(self.fits)

    # ------------------------------------------------------------------
    # Temporal slicing — the construction guarantee
    # ------------------------------------------------------------------
    def _slice_through(self, all_data: Dict[str, pd.DataFrame], date,
                       cap_window: bool) -> Dict[str, pd.DataFrame]:
        """Slice each frame to index <= date; optionally cap to the
        trailing train_window_days rows. Empty results are dropped."""
        sliced: Dict[str, pd.DataFrame] = {}
        for symbol, data in all_data.items():
            if data is None or data.empty:
                continue
            cutoff = pd.Timestamp(date)
            index_tz = getattr(data.index, "tz", None)
            # Live clocks are timezone-aware while daily OHLCV indices are
            # commonly timezone-naive.  Align the cutoff to the index before
            # comparison; timezone metadata must not make a same-session
            # history unusable (or tempt callers to strip the live clock).
            if index_tz is None and cutoff.tzinfo is not None:
                cutoff = cutoff.tz_localize(None)
            elif index_tz is not None and cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize(index_tz)
            elif index_tz is not None and cutoff.tzinfo is not None:
                cutoff = cutoff.tz_convert(index_tz)
            # Engine-driven callers pass frames already bounded through the
            # simulation date, making the mask an all-True full-frame copy;
            # share the frame instead (models are pure readers). The
            # monotonic guard is MANDATORY: index[-1] <= cutoff only implies
            # all-rows-bounded for a sorted index, and the mask path is the
            # leakage guarantee for anything unsorted.
            if data.index.is_monotonic_increasing and data.index[-1] <= cutoff:
                window = data
            else:
                window = data[data.index <= cutoff]
            if cap_window:
                window = window.tail(self.train_window_days)
            if not window.empty:
                sliced[symbol] = window
        return sliced

    # ------------------------------------------------------------------
    # Fit scheduling
    # ------------------------------------------------------------------
    def maybe_refit(self, all_data: Dict[str, pd.DataFrame], date) -> bool:
        """Refit the model if due; return True when a fit happened.

        Due when (a) never fitted and at least one symbol has
        min_train_days rows of history through `date`, or (b)
        refit_every_days distinct trading days have elapsed since the
        last fit. The data handed to model.fit is sliced to index <= date
        AND capped to the trailing train_window_days rows per symbol.
        """
        current = _as_date(date)
        if self._last_seen_date is not None and current != self._last_seen_date:
            self._days_since_fit += 1
        self._last_seen_date = current

        # Fitted and not yet due: return before slicing — on this path the
        # sliced history is discarded regardless of its contents, and this
        # runs every simulated day per controller.
        if self.is_fitted and self._days_since_fit < self.refit_every_days:
            return False

        history = self._slice_through(all_data, date, cap_window=False)
        if not history:
            return False

        if not self.is_fitted:
            # Depth is measured on the UNCAPPED history: the train-window
            # cap below bounds what the model sees, not whether enough
            # history exists to fit at all.
            depth = max(len(frame) for frame in history.values())
            if depth < self.min_train_days:
                return False

        train_data = {symbol: frame.tail(self.train_window_days)
                      for symbol, frame in history.items()}

        self.model.fit(train_data)
        if self._checkpoint_supported:
            self._last_train_data = {
                symbol: frame.copy(deep=True)
                for symbol, frame in train_data.items()
            }

        train_start = min(frame.index.min() for frame in train_data.values())
        train_end = max(frame.index.max() for frame in train_data.values())
        n_samples = sum(len(frame) for frame in train_data.values())
        fit = WalkForwardFit(fit_date=current,
                             train_start=_as_date(train_start),
                             train_end=_as_date(train_end),
                             n_samples=n_samples)
        self.fits.append(fit)
        self.last_fit_date = current
        self._days_since_fit = 0
        logger.info("Walk-forward fit on %s: train %s..%s (%d samples)",
                    fit.fit_date, fit.train_start, fit.train_end, n_samples)
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, all_data: Dict[str, pd.DataFrame],
                date) -> Optional[Dict[str, float]]:
        """Score symbols at `date`, or None if the model was never fitted.

        Defense in depth: frames are re-sliced to index <= date here even
        though callers are expected to pass already-bounded data — the
        model NEVER sees a row beyond the simulation date.
        """
        if not self.is_fitted:
            return None
        data = self._slice_through(all_data, date, cap_window=False)
        return self.model.predict(data, date)

    # ------------------------------------------------------------------
    # Safe restart checkpointing
    # ------------------------------------------------------------------
    def checkpoint_state(self) -> Dict[str, Any]:
        """Return a content-checked, JSON-native deterministic-refit state.

        Estimator bytes are deliberately absent.  A fitted checkpoint stores
        the exact final training window and restore calls ``model.fit`` on it.
        This is safe to persist with ``json.dump(..., allow_nan=False)``.
        """
        model_spec = self.model.checkpoint_spec()
        _canonical_json(model_spec)  # reject non-JSON model identities early

        if self.is_fitted:
            if self._last_train_data is None:
                raise RuntimeError("fitted controller has no refit checkpoint data")
            # A list preserves the original symbol insertion order.  Pooled
            # learners see rows in this order, so sorting or relying on JSON
            # object order would not reproduce the *exact* fit input.
            training_data = [
                {'frame': _frame_to_checkpoint(frame), 'symbol': symbol}
                for symbol, frame in self._last_train_data.items()
            ]
        else:
            if self._last_train_data is not None:
                raise RuntimeError("unfitted controller has unexpected training data")
            training_data = []

        payload: Dict[str, Any] = {
            'cadence': {
                'days_since_fit': self._days_since_fit,
                'last_fit_date': (self.last_fit_date.isoformat()
                                  if self.last_fit_date else None),
                'last_seen_date': (self._last_seen_date.isoformat()
                                   if self._last_seen_date else None),
            },
            'config': {
                'min_train_days': self.min_train_days,
                'refit_every_days': self.refit_every_days,
                'train_window_days': self.train_window_days,
            },
            'fits': [fit.to_dict() for fit in self.fits],
            'model': model_spec,
            'training_data': training_data,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        return {
            'payload': payload,
            'schema_version': _CHECKPOINT_SCHEMA_VERSION,
            'sha256': digest,
        }

    def restore_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore a checkpoint into this fresh, identically configured controller.

        Input is validated completely before cadence state is committed.  The
        model already attached to this controller is deterministically refit;
        checkpoint data can never select a Python class or execute a payload.
        """
        if (self.fits or self.last_fit_date is not None
                or self._last_seen_date is not None
                or self._days_since_fit != 0
                or self._last_train_data is not None):
            raise RuntimeError("checkpoint restore requires a fresh controller")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must be an object")
        if set(checkpoint) != {'payload', 'schema_version', 'sha256'}:
            raise ValueError("checkpoint fields are invalid")
        if (type(checkpoint['schema_version']) is not int
                or checkpoint['schema_version'] != _CHECKPOINT_SCHEMA_VERSION):
            raise ValueError("unsupported checkpoint schema_version")
        payload = checkpoint['payload']
        digest = checkpoint['sha256']
        if not isinstance(payload, Mapping) or not isinstance(digest, str):
            raise ValueError("checkpoint payload/sha256 is invalid")
        expected_digest = hashlib.sha256(
            _canonical_json(payload).encode()).hexdigest()
        if not hmac.compare_digest(digest, expected_digest):
            raise ValueError("checkpoint sha256 does not match its payload")

        required_payload = {'cadence', 'config', 'fits', 'model', 'training_data'}
        if set(payload) != required_payload:
            raise ValueError("checkpoint payload fields are invalid")
        config = payload['config']
        expected_config = {
            'min_train_days': self.min_train_days,
            'refit_every_days': self.refit_every_days,
            'train_window_days': self.train_window_days,
        }
        if (not isinstance(config, Mapping)
                or set(config) != set(expected_config)
                or any(type(config[field]) is not int or config[field] <= 0
                       for field in expected_config)
                or _canonical_json(config) != _canonical_json(expected_config)):
            raise ValueError("checkpoint controller config does not match runtime")
        runtime_model_spec = self.model.checkpoint_spec()
        if (_canonical_json(payload['model'])
                != _canonical_json(runtime_model_spec)):
            raise ValueError("checkpoint model spec does not match runtime")

        raw_fits = payload['fits']
        if not isinstance(raw_fits, list):
            raise ValueError("checkpoint fits must be an array")
        fits: List[WalkForwardFit] = []
        previous_fit_date: Optional[date_type] = None
        for number, raw_fit in enumerate(raw_fits):
            if not isinstance(raw_fit, Mapping) or set(raw_fit) != {
                    'fit_date', 'n_samples', 'train_end', 'train_start'}:
                raise ValueError(f"checkpoint fits[{number}] is invalid")
            fit_date = _strict_date(raw_fit['fit_date'], 'fit_date')
            train_start = _strict_date(raw_fit['train_start'], 'train_start')
            train_end = _strict_date(raw_fit['train_end'], 'train_end')
            n_samples = raw_fit['n_samples']
            if type(n_samples) is not int or n_samples <= 0:
                raise ValueError("checkpoint fit n_samples must be positive")
            if not train_start <= train_end <= fit_date:
                raise ValueError("checkpoint fit date bounds are invalid")
            if previous_fit_date is not None and fit_date <= previous_fit_date:
                raise ValueError("checkpoint fit dates must be strictly increasing")
            fits.append(WalkForwardFit(
                fit_date=fit_date, train_start=train_start,
                train_end=train_end, n_samples=n_samples))
            previous_fit_date = fit_date

        cadence = payload['cadence']
        if not isinstance(cadence, Mapping) or set(cadence) != {
                'days_since_fit', 'last_fit_date', 'last_seen_date'}:
            raise ValueError("checkpoint cadence is invalid")
        days_since_fit = cadence['days_since_fit']
        if type(days_since_fit) is not int or days_since_fit < 0:
            raise ValueError("checkpoint days_since_fit must be non-negative")
        last_fit_date = (None if cadence['last_fit_date'] is None else
                         _strict_date(cadence['last_fit_date'], 'last_fit_date'))
        last_seen_date = (None if cadence['last_seen_date'] is None else
                          _strict_date(cadence['last_seen_date'], 'last_seen_date'))
        if bool(fits) != (last_fit_date is not None):
            raise ValueError("checkpoint fits/last_fit_date are inconsistent")
        if fits and fits[-1].fit_date != last_fit_date:
            raise ValueError("checkpoint last_fit_date is not the latest fit")
        if (last_fit_date is not None and last_seen_date is not None
                and last_seen_date < last_fit_date):
            raise ValueError("checkpoint last_seen_date precedes last_fit_date")

        raw_training = payload['training_data']
        if not isinstance(raw_training, list):
            raise ValueError("checkpoint training_data must be an array")
        training_data: Dict[str, pd.DataFrame] = {}
        for number, entry in enumerate(raw_training):
            if (not isinstance(entry, Mapping)
                    or set(entry) != {'frame', 'symbol'}):
                raise ValueError(
                    f"checkpoint training_data[{number}] is invalid")
            symbol = entry['symbol']
            if (not isinstance(symbol, str) or not symbol
                    or symbol != symbol.strip()):
                raise ValueError("checkpoint training symbols are invalid")
            if symbol in training_data:
                raise ValueError("checkpoint training symbols must be unique")
            training_data[symbol] = _frame_from_checkpoint(
                entry['frame'], symbol)

        if bool(fits) != bool(training_data):
            raise ValueError("checkpoint fits/training_data are inconsistent")
        if fits:
            latest = fits[-1]
            if any(len(frame) > self.train_window_days
                   for frame in training_data.values()):
                raise ValueError("checkpoint training frame exceeds window")
            n_samples = sum(len(frame) for frame in training_data.values())
            train_start = min(_as_date(frame.index.min())
                              for frame in training_data.values())
            train_end = max(_as_date(frame.index.max())
                            for frame in training_data.values())
            if (n_samples != latest.n_samples
                    or train_start != latest.train_start
                    or train_end != latest.train_end):
                raise ValueError(
                    "checkpoint training_data does not match latest fit")

            # Safe reconstruction: deterministic refit on primitive tabular
            # data.  No checkpoint-provided function, class, or object loads.
            self.model.fit(training_data)

        self.fits = fits
        self.last_fit_date = last_fit_date
        self._last_seen_date = last_seen_date
        self._days_since_fit = days_since_fit
        self._last_train_data = ({
            symbol: frame.copy(deep=True)
            for symbol, frame in training_data.items()
        } if training_data else None)
