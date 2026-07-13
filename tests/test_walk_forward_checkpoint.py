"""Safe JSON checkpoint/restore tests for the Foundation GBM controller."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from desks.ml_model import GradientBoostingModel
from desks.foundation import FoundationDesk
from desks.walk_forward import WalkForwardController, WalkForwardModel


def _enriched_frame(start: str, periods: int, offset: float = 0.0,
                    timezone: str | None = None) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods, tz=timezone)
    step = np.arange(periods, dtype=float)
    close = 100.0 + offset + 0.04 * step + 2.5 * np.sin(step / 3.0)
    return pd.DataFrame({
        'open': close - 0.1,
        'high': close + 0.8,
        'low': close - 0.8,
        'close': close,
        'volume': (400_000 + (step.astype(np.int64) % 9) * 10_000),
        'rsi': 50.0 + 8.0 * np.sin(step / 4.0),
        'macd': np.sin(step / 5.0),
        'bb_upper': close + 5.0,
        'bb_lower': close - 5.0,
        'volume_sma': np.full(periods, 430_000.0),
    }, index=index)


def _controller(*, n_estimators: int = 9,
                train_window_days: int = 45) -> WalkForwardController:
    return WalkForwardController(
        GradientBoostingModel(
            n_estimators=n_estimators, max_depth=2, random_state=17),
        train_window_days=train_window_days,
        refit_every_days=7,
        min_train_days=35,
    )


def _resign(checkpoint: dict) -> None:
    canonical = json.dumps(
        checkpoint['payload'], sort_keys=True, separators=(',', ':'),
        allow_nan=False)
    checkpoint['sha256'] = hashlib.sha256(canonical.encode()).hexdigest()


def test_restart_every_cycle_matches_one_continuous_controller_exactly():
    """A paper worker recreated after every day has identical GBM state."""
    data = {
        # Deliberately non-lexicographic: pooled row order is fit input and the
        # checkpoint must preserve it rather than sorting JSON object keys.
        'ZZZ': _enriched_frame('2024-01-02', 90),
        'AAA': _enriched_frame('2024-01-02', 90, offset=25.0),
    }
    continuous = _controller()
    restarted = _controller()

    for decision_date in data['AAA'].index:
        continuous_refit = continuous.maybe_refit(data, decision_date)
        continuous_scores = continuous.predict(data, decision_date)

        restarted_refit = restarted.maybe_refit(data, decision_date)
        restarted_scores = restarted.predict(data, decision_date)

        assert restarted_refit == continuous_refit
        assert restarted_scores == continuous_scores
        assert [fit.to_dict() for fit in restarted.fits] == [
            fit.to_dict() for fit in continuous.fits]

        # Exercise the persistence boundary exactly as the paper runner does:
        # standards-compliant JSON, then a brand-new model/controller.
        serialized = json.dumps(restarted.checkpoint_state(), allow_nan=False)
        checkpoint = json.loads(serialized)
        restarted = _controller()
        restarted.restore_checkpoint(checkpoint)

    assert restarted.checkpoint_state() == continuous.checkpoint_state()


def test_checkpoint_is_json_native_and_round_trips_timezone_and_dtypes():
    frame = _enriched_frame('2024-03-01', 45, timezone='America/New_York')
    frame.iloc[3, frame.columns.get_loc('rsi')] = np.nan
    controller = _controller()
    date = frame.index[-1]
    assert controller.maybe_refit({'AAA': frame}, date)

    checkpoint = controller.checkpoint_state()
    encoded = json.dumps(checkpoint, allow_nan=False)
    assert 'pickle' not in encoded.lower()
    assert 'joblib' not in encoded.lower()

    restored = _controller()
    restored.restore_checkpoint(json.loads(encoded))
    original_train = controller._last_train_data['AAA']
    restored_train = restored._last_train_data['AAA']
    pd.testing.assert_frame_equal(restored_train, original_train,
                                  check_freq=False)
    assert restored.predict({'AAA': frame}, date) == controller.predict(
        {'AAA': frame}, date)


def test_foundation_desk_exposes_runner_checkpoint_boundary():
    frame = _enriched_frame('2024-01-02', 45)
    source = FoundationDesk(controller=_controller())
    source._controller.maybe_refit({'AAA': frame}, frame.index[-1])

    checkpoint = json.loads(json.dumps(source.model_checkpoint_state()))
    restored = FoundationDesk(controller=_controller())
    restored.restore_model_checkpoint(checkpoint)

    assert restored._controller.predict(
        {'AAA': frame}, frame.index[-1]) == source._controller.predict(
            {'AAA': frame}, frame.index[-1])


def test_checkpoint_detects_unresigned_payload_tampering():
    frame = _enriched_frame('2024-01-02', 45)
    controller = _controller()
    controller.maybe_refit({'AAA': frame}, frame.index[-1])
    checkpoint = controller.checkpoint_state()
    checkpoint['payload']['cadence']['days_since_fit'] = 999

    with pytest.raises(ValueError, match='sha256'):
        _controller().restore_checkpoint(checkpoint)


@pytest.mark.parametrize('mutation, message', [
    (lambda state: state['payload']['config'].__setitem__(
        'train_window_days', 44), 'config does not match'),
    (lambda state: state['payload']['model']['parameters'].__setitem__(
        'random_state', 999), 'model spec does not match'),
    (lambda state: state['payload']['training_data'][0]['frame']['rows'][0]
     .__setitem__(3, 'not-a-number'), 'invalid float'),
])
def test_resigned_but_invalid_state_is_rejected(mutation, message):
    frame = _enriched_frame('2024-01-02', 45)
    controller = _controller()
    controller.maybe_refit({'AAA': frame}, frame.index[-1])
    checkpoint = copy.deepcopy(controller.checkpoint_state())
    mutation(checkpoint)
    _resign(checkpoint)

    with pytest.raises(ValueError, match=message):
        _controller().restore_checkpoint(checkpoint)


def test_restore_requires_a_fresh_controller():
    frame = _enriched_frame('2024-01-02', 45)
    source = _controller()
    source.maybe_refit({'AAA': frame}, frame.index[-1])
    target = _controller()
    target.maybe_refit({'AAA': frame}, frame.index[-1])

    with pytest.raises(RuntimeError, match='fresh controller'):
        target.restore_checkpoint(source.checkpoint_state())


class _UnsupportedModel(WalkForwardModel):
    def fit(self, train_data):
        pass

    def predict(self, data, date):
        return {}


def test_models_must_explicitly_opt_in_to_safe_refit_checkpointing():
    controller = WalkForwardController(_UnsupportedModel())
    with pytest.raises(TypeError, match='does not support'):
        controller.checkpoint_state()


def test_unsupported_models_do_not_retain_training_frames():
    frame = _enriched_frame('2024-01-02', 45)
    controller = WalkForwardController(
        _UnsupportedModel(), min_train_days=1)
    assert controller.maybe_refit({'AAA': frame}, frame.index[-1])
    assert controller._last_train_data is None


def test_object_columns_are_rejected_instead_of_deserialized():
    frame = _enriched_frame('2024-01-02', 45)
    frame['unsafe'] = [{'callable': 'never'} for _ in range(len(frame))]
    controller = _controller()
    controller.maybe_refit({'AAA': frame}, frame.index[-1])

    with pytest.raises(ValueError, match='unsupported dtypes'):
        controller.checkpoint_state()
