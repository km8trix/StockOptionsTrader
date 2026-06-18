"""
Trading-desk framework: firm-persona desks, walk-forward harness, registry.
"""

from desks.base import Desk, DeskIntent, TraderNote
from desks.citadel import CitadelDesk
from desks.foundation import FoundationDesk
from desks.janestreet import JaneStreetDesk
from desks.ml_model import GradientBoostingModel
from desks.models import available_models, build_model
from desks.pairs import PairsCointegrationModel
from desks.pods import MeanReversionPod, MomentumPod, Pod, StatArbPod
from desks.regime import RegimeHMMModel
from desks.registry import create_desk, list_desks
from desks.renaissance import RenaissanceDesk
from desks.risk_book import CentralRiskBook
from desks.walk_forward import (WalkForwardController, WalkForwardFit,
                                WalkForwardModel)

__all__ = [
    'CentralRiskBook',
    'CitadelDesk',
    'Desk',
    'DeskIntent',
    'TraderNote',
    'FoundationDesk',
    'GradientBoostingModel',
    'JaneStreetDesk',
    'MeanReversionPod',
    'MomentumPod',
    'PairsCointegrationModel',
    'Pod',
    'RegimeHMMModel',
    'RenaissanceDesk',
    'StatArbPod',
    'WalkForwardController',
    'WalkForwardFit',
    'WalkForwardModel',
    'available_models',
    'build_model',
    'create_desk',
    'list_desks',
]
