"""THE INVARIANT SUITE for desks.walk_forward.

A SpyModel records a copy of every frame it is ever handed, in fit AND
predict, together with the simulation date at call time. The driver below
feeds the controller the FULL 300-day frame every day — deliberately
including future rows — so these tests prove the controller's slicing by
construction, not the caller's discipline:

    (a) every row any model receives has index <= the simulation date;
    (b) fit data is capped at train_window_days rows;
    (c) refits land exactly refit_every_days trading days apart;
    (d) predict returns None before the first fit;
    (e) every prediction at date D comes from a fit whose train_end <= D;
    (f) a malicious caller passing future rows to predict changes nothing.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import pytest

from desks.walk_forward import (WalkForwardController, WalkForwardFit,
                                WalkForwardModel)

TRAIN_WINDOW = 100
REFIT_EVERY = 21
MIN_TRAIN = 120
N_DAYS = 300


class SpyModel(WalkForwardModel):
    """Records every frame received; the driver stamps the current date."""

    def __init__(self):
        self.current_date = None  # set by the driver before each call
        self.fit_calls: List[Dict] = []
        self.predict_calls: List[Dict] = []

    def fit(self, train_data):
        self.fit_calls.append({
            'date': self.current_date,
            'frames': {s: df.copy() for s, df in train_data.items()},
        })

    def predict(self, data, date):
        self.predict_calls.append({
            'date': pd.Timestamp(date),
            'frames': {s: df.copy() for s, df in data.items()},
        })
        return {symbol: 0.0 for symbol in data}


@pytest.fixture
def driven_controller(make_ohlcv):
    """Drive a 300-day simulation, passing the FULL frame every day."""
    full = make_ohlcv(n_days=N_DAYS, seed=101)
    spy = SpyModel()
    controller = WalkForwardController(spy,
                                       train_window_days=TRAIN_WINDOW,
                                       refit_every_days=REFIT_EVERY,
                                       min_train_days=MIN_TRAIN)
    timeline = []  # (date, fitted_now, predictions, active_fit_at_call)
    for date in full.index:
        spy.current_date = date
        fitted_now = controller.maybe_refit({'SYM': full}, date)
        active_fit = controller.fits[-1] if controller.fits else None
        predictions = controller.predict({'SYM': full}, date)
        timeline.append((date, fitted_now, predictions, active_fit))
    return full, spy, controller, timeline


class TestNoFutureRowEverReachesTheModel:
    def test_a_every_fit_row_is_at_or_before_the_simulation_date(
            self, driven_controller):
        _, spy, _, _ = driven_controller
        assert spy.fit_calls  # the run actually fitted
        for call in spy.fit_calls:
            for frame in call['frames'].values():
                assert frame.index.max() <= call['date'], (
                    f"fit on {call['date']} saw row {frame.index.max()}")

    def test_a_every_predict_row_is_at_or_before_the_simulation_date(
            self, driven_controller):
        _, spy, _, _ = driven_controller
        assert spy.predict_calls
        for call in spy.predict_calls:
            for frame in call['frames'].values():
                assert frame.index.max() <= call['date'], (
                    f"predict on {call['date']} saw row {frame.index.max()}")

    def test_f_malicious_future_rows_in_predict_are_sliced_away(
            self, make_ohlcv):
        full = make_ohlcv(n_days=N_DAYS, seed=103)
        spy = SpyModel()
        controller = WalkForwardController(spy,
                                           train_window_days=TRAIN_WINDOW,
                                           refit_every_days=REFIT_EVERY,
                                           min_train_days=MIN_TRAIN)
        # Fit once, honestly, on the first MIN_TRAIN days.
        fit_date = full.index[MIN_TRAIN - 1]
        spy.current_date = fit_date
        assert controller.maybe_refit(
            {'SYM': full[full.index <= fit_date]}, fit_date)

        # MALICIOUS CALL: hand predict the full frame — 150 days of future.
        attack_date = full.index[150]
        scores = controller.predict({'SYM': full}, attack_date)

        assert scores == {'SYM': 0.0}
        seen = spy.predict_calls[-1]['frames']['SYM']
        assert seen.index.max() <= attack_date
        # And nothing before the date was lost: the slice is exact.
        assert len(seen) == 151

    def test_f_malicious_future_rows_in_maybe_refit_are_sliced_and_capped(
            self, make_ohlcv):
        full = make_ohlcv(n_days=N_DAYS, seed=105)
        spy = SpyModel()
        controller = WalkForwardController(spy,
                                           train_window_days=TRAIN_WINDOW,
                                           refit_every_days=REFIT_EVERY,
                                           min_train_days=MIN_TRAIN)
        fit_date = full.index[200]
        spy.current_date = fit_date
        assert controller.maybe_refit({'SYM': full}, fit_date)

        seen = spy.fit_calls[-1]['frames']['SYM']
        assert seen.index.max() <= fit_date
        assert len(seen) == TRAIN_WINDOW


class TestTrainWindowCap:
    def test_b_every_fit_is_capped_at_train_window_rows(
            self, driven_controller):
        _, spy, controller, _ = driven_controller
        for call in spy.fit_calls:
            for frame in call['frames'].values():
                assert len(frame) <= TRAIN_WINDOW
        # min_train (120) > window (100): the cap binds on every fit here.
        for fit in controller.fits:
            assert fit.n_samples == TRAIN_WINDOW

    def test_fit_records_actual_train_bounds(self, driven_controller):
        _, spy, controller, _ = driven_controller
        for fit, call in zip(controller.fits, spy.fit_calls):
            frame = call['frames']['SYM']
            assert fit.train_start == frame.index.min().date()
            assert fit.train_end == frame.index.max().date()
            assert fit.fit_date == call['date'].date()


class TestRefitCadence:
    def test_c_fit_dates_spaced_exactly_refit_every_trading_days(
            self, driven_controller):
        full, _, controller, _ = driven_controller
        # First fit on the day history depth reaches MIN_TRAIN, then every
        # REFIT_EVERY trading days.
        expected_positions = list(range(MIN_TRAIN - 1, N_DAYS, REFIT_EVERY))
        expected_dates = [full.index[i].date() for i in expected_positions]
        assert [fit.fit_date for fit in controller.fits] == expected_dates

    def test_last_fit_date_and_is_fitted_track_state(self, driven_controller):
        _, _, controller, _ = driven_controller
        assert controller.is_fitted
        assert controller.last_fit_date == controller.fits[-1].fit_date

    def test_fresh_controller_is_unfitted(self):
        controller = WalkForwardController(SpyModel())
        assert not controller.is_fitted
        assert controller.last_fit_date is None
        assert controller.fits == []


class TestPredictGating:
    def test_d_predict_before_first_fit_returns_none(self, driven_controller):
        full, _, _, timeline = driven_controller
        first_fit_position = MIN_TRAIN - 1
        for position, (date, _, predictions, _) in enumerate(timeline):
            if position < first_fit_position:
                assert predictions is None, f"premature prediction on {date}"
            else:
                assert predictions is not None

    def test_e_every_signal_date_is_at_or_after_active_fit_train_end(
            self, driven_controller):
        _, _, _, timeline = driven_controller
        checked = 0
        for date, _, predictions, active_fit in timeline:
            if predictions is None:
                continue
            assert active_fit is not None
            assert active_fit.train_end <= date.date(), (
                f"prediction on {date.date()} from model trained through "
                f"{active_fit.train_end}")
            checked += 1
        assert checked == N_DAYS - (MIN_TRAIN - 1)


class TestWalkForwardFitSerialization:
    def test_to_dict_matches_contract_shape(self, driven_controller):
        _, _, controller, _ = driven_controller
        for fit in controller.fits:
            payload = fit.to_dict()
            assert sorted(payload) == ['fit_date', 'n_samples',
                                       'train_end', 'train_start']
            for key in ('fit_date', 'train_start', 'train_end'):
                value = payload[key]
                assert isinstance(value, str) and len(value) == 10
                pd.Timestamp(value)  # parses as a date
            assert isinstance(payload['n_samples'], int)

    def test_hand_built_fit_serializes_exactly(self):
        fit = WalkForwardFit(fit_date=pd.Timestamp('2023-06-30').date(),
                             train_start=pd.Timestamp('2023-01-02').date(),
                             train_end=pd.Timestamp('2023-06-30').date(),
                             n_samples=124)
        assert fit.to_dict() == {
            'fit_date': '2023-06-30',
            'train_start': '2023-01-02',
            'train_end': '2023-06-30',
            'n_samples': 124,
        }
