"""Tests for the model registry and its wiring into the foundation desk
and the desk registry (Phase A model selection).

Covers:
  * desks.models.available_models() / build_model() — the single source of
    truth for selectable walk-forward models;
  * desks.foundation.FoundationDesk model_key precedence (default ==
    byte-identical to before; model_key selects; explicit controller wins);
  * desks.registry.create_desk(..., model_key=...) — foundation supports it,
    other desks reject it, the no-model_key path is unchanged.

Offline and deterministic — no network, no real market data.
"""

from __future__ import annotations

import pytest

from desks.foundation import FoundationDesk
from desks.ml_model import GradientBoostingModel
from desks.models import available_models, build_model
from desks.models.boosting import LightGBMModel, StackingMetaModel
from desks.registry import create_desk
from desks.walk_forward import WalkForwardController, WalkForwardModel

# The exact (ordered) registry contract.
EXPECTED_IDS = ['gbm', 'lightgbm', 'stacking']
EXPECTED_NAMES = ['Gradient Boosting', 'LightGBM', 'Stacking Ensemble']


class TestAvailableModels:
    def test_returns_exactly_three_models_in_order(self):
        models = available_models()
        assert [m['id'] for m in models] == EXPECTED_IDS
        assert [m['name'] for m in models] == EXPECTED_NAMES

    def test_each_entry_has_exactly_id_name_description(self):
        for entry in available_models():
            assert set(entry) == {'id', 'name', 'description'}
            assert isinstance(entry['id'], str) and entry['id']
            assert isinstance(entry['name'], str) and entry['name']
            assert isinstance(entry['description'], str) and entry['description']


class TestBuildModel:
    def test_gbm_builds_gradient_boosting(self):
        model = build_model('gbm')
        assert isinstance(model, GradientBoostingModel)
        assert isinstance(model, WalkForwardModel)

    def test_lightgbm_builds_lightgbm_model(self):
        model = build_model('lightgbm')
        assert isinstance(model, LightGBMModel)
        assert isinstance(model, WalkForwardModel)

    def test_stacking_builds_stacking_meta_model(self):
        model = build_model('stacking')
        assert isinstance(model, StackingMetaModel)
        assert isinstance(model, WalkForwardModel)

    def test_each_advertised_id_is_buildable(self):
        for entry in available_models():
            assert isinstance(build_model(entry['id']), WalkForwardModel)

    def test_build_returns_fresh_unshared_instances(self):
        assert build_model('gbm') is not build_model('gbm')

    def test_unknown_key_raises_value_error(self):
        with pytest.raises(ValueError, match='Unknown model'):
            build_model('bogus')


class TestFoundationModelSelection:
    def test_default_is_gradient_boosting_with_golden_windows(self):
        """model_key=None (default) -> WalkForwardController(
        GradientBoostingModel()) with the historical 252/21/120 windows —
        byte-identical to the pre-Phase-A default."""
        desk = FoundationDesk()
        controller = desk._controller
        assert isinstance(controller, WalkForwardController)
        assert isinstance(controller.model, GradientBoostingModel)
        assert controller.train_window_days == 252
        assert controller.refit_every_days == 21
        assert controller.min_train_days == 120

    def test_model_key_lightgbm_wraps_lightgbm_model(self):
        desk = FoundationDesk(model_key='lightgbm')
        assert isinstance(desk._controller, WalkForwardController)
        assert isinstance(desk._controller.model, LightGBMModel)

    def test_model_key_stacking_wraps_stacking_model(self):
        desk = FoundationDesk(model_key='stacking')
        assert isinstance(desk._controller.model, StackingMetaModel)

    def test_explicit_controller_wins_over_model_key(self):
        sentinel = WalkForwardController(GradientBoostingModel())
        desk = FoundationDesk(controller=sentinel, model_key='lightgbm')
        assert desk._controller is sentinel

    def test_unknown_model_key_raises_value_error(self):
        with pytest.raises(ValueError, match='Unknown model'):
            FoundationDesk(model_key='bogus')


class TestCreateDeskModelSelection:
    def test_foundation_with_model_key_builds_on_selected_model(self):
        desk = create_desk('foundation', model_key='lightgbm')
        assert isinstance(desk, FoundationDesk)
        assert isinstance(desk._controller.model, LightGBMModel)

    def test_foundation_without_model_key_is_unchanged(self):
        desk = create_desk('foundation')
        assert isinstance(desk, FoundationDesk)
        assert isinstance(desk._controller.model, GradientBoostingModel)
        assert desk.capital_allocation == 1.0

    def test_foundation_model_key_and_capital_allocation_together(self):
        desk = create_desk('foundation', capital_allocation=0.25,
                           model_key='stacking')
        assert desk.capital_allocation == 0.25
        assert isinstance(desk._controller.model, StackingMetaModel)

    def test_renaissance_with_model_key_rejected(self):
        with pytest.raises(ValueError,
                           match='does not support model selection'):
            create_desk('renaissance', model_key='lightgbm')

    @pytest.mark.parametrize('desk_key', ['citadel', 'janestreet'])
    def test_other_desks_reject_model_key(self, desk_key):
        with pytest.raises(ValueError,
                           match='does not support model selection'):
            create_desk(desk_key, model_key='gbm')

    def test_unknown_desk_with_model_key_still_reports_unknown_desk(self):
        # Unknown-desk check precedes the model-selection gate.
        with pytest.raises(ValueError, match='Unknown desk'):
            create_desk('warrenbuffett', model_key='gbm')
