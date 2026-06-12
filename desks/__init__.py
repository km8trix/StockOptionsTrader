"""
Trading-desk framework: firm-persona desks, walk-forward harness, registry.
"""

from desks.base import Desk, DeskIntent, TraderNote
from desks.foundation import FoundationDesk
from desks.ml_model import GradientBoostingModel
from desks.registry import create_desk, list_desks
from desks.walk_forward import (WalkForwardController, WalkForwardFit,
                                WalkForwardModel)

__all__ = [
    'Desk',
    'DeskIntent',
    'TraderNote',
    'FoundationDesk',
    'GradientBoostingModel',
    'WalkForwardController',
    'WalkForwardFit',
    'WalkForwardModel',
    'create_desk',
    'list_desks',
]
