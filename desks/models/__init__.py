"""
Model registry — the single source of truth for walk-forward model selection.

``build_model(key)`` maps a stable string id to a fresh
``WalkForwardModel`` instance; ``available_models()`` describes every id for
the GUI (the ``/api/models`` route reuses these verbatim). Keep these two in
lockstep: the ids ``available_models`` advertises are exactly the ids
``build_model`` accepts.

The default id ``'gbm'`` -> ``GradientBoostingModel`` is the CURRENT,
golden behavior: ``FoundationDesk(model_key='gbm')`` is byte-identical to
the historical default (and to ``model_key=None``).

Imports are lazy: LightGBM is optional, so the heavier model module is only
imported inside ``build_model``/``available_models``. Importing
``desks.models`` (and therefore the registry) never requires lightgbm.
"""

from __future__ import annotations

from typing import Dict, List

from desks.ml_model import GradientBoostingModel
from desks.walk_forward import WalkForwardModel

#: Stable id -> (display name, description). The ORDER here is the order
#: the GUI lists models in. 'gbm' is first because it is the default.
_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    'gbm': {
        'name': 'Gradient Boosting',
        'description': ('Scikit-learn gradient-boosting classifier on '
                        'next-day direction — the current default model.'),
    },
    'lightgbm': {
        'name': 'LightGBM',
        'description': ('LightGBM histogram gradient booster on next-day '
                        'direction; deterministic, faster on large feature '
                        'sets. Requires the lightgbm package.'),
    },
    'stacking': {
        'name': 'Stacking Ensemble',
        'description': ('Logistic meta-learner stacking Gradient Boosting '
                        'and LightGBM via leakage-free out-of-fold base '
                        'predictions.'),
    },
}


def available_models() -> List[Dict[str, str]]:
    """Describe every selectable model as ``{'id','name','description'}``.

    The list the GUI's model picker and the ``/api/models`` route render;
    order is the display order (``'gbm'`` first as the default).
    """
    return [
        {'id': key, 'name': spec['name'], 'description': spec['description']}
        for key, spec in _MODEL_SPECS.items()
    ]


def build_model(model_key: str) -> WalkForwardModel:
    """Instantiate a fresh ``WalkForwardModel`` for ``model_key``.

    Args:
        model_key: one of the ids from :func:`available_models`
            (``'gbm'``, ``'lightgbm'``, ``'stacking'``).

    Returns:
        A new, unfitted model instance.

    Raises:
        ValueError: ``Unknown model: <key>`` for any unrecognized id.
        ImportError: when ``'lightgbm'`` is requested but the lightgbm
            package is missing (surfaced by ``LightGBMModel``'s
            constructor — a loud, early signal of a misconfigured install).
    """
    if model_key == 'gbm':
        return GradientBoostingModel()
    if model_key == 'lightgbm':
        # Lazy import: keep lightgbm optional for everyone who never asks
        # for it (the registry, the default desk, the feature library).
        from desks.models.boosting import LightGBMModel
        return LightGBMModel()
    if model_key == 'stacking':
        from desks.models.boosting import StackingMetaModel
        return StackingMetaModel()
    raise ValueError(f"Unknown model: {model_key}")


__all__ = ['available_models', 'build_model']
