"""
Desk registry — the GUI's single source of truth for the Trading Floor.

list_desks() describes every desk (ready or planned, per the master plan
phases); create_desk() instantiates ready desks and raises informative
ValueErrors for unknown or not-yet-activated ones (contract C1).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from desks.base import Desk
from desks.citadel import CitadelDesk
from desks.foundation import FoundationDesk
from desks.janestreet import JaneStreetDesk
from desks.renaissance import RenaissanceDesk

logger = logging.getLogger(__name__)

#: Registry entries in display order. 'factory' is None for planned desks.
_DESK_SPECS: Dict[str, Dict] = {
    'foundation': {
        'name': 'Foundation Desk',
        'firm_inspiration': 'House',
        'description': ('House momentum desk with a walk-forward '
                        'gradient-boosting gate — proves the desk framework '
                        'and leakage-free harness end to end.'),
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#4493f8',
        'factory': FoundationDesk,
    },
    'renaissance': {
        'name': 'Renaissance Desk',
        'firm_inspiration': 'Renaissance Technologies',
        'description': ('HMM regime detection, regime-conditioned '
                        'short-horizon mean reversion, nonlinear stat-arb, '
                        'and cointegration pairs.'),
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#58a6ff',
        'factory': RenaissanceDesk,
    },
    'citadel': {
        'name': 'Citadel Desk',
        'firm_inspiration': 'Citadel',
        'description': ('Strategy pods with their own capital, a central '
                        'risk book, and performance-weighted capital '
                        'allocation with drawdown-based pod cuts.'),
        # Contract C10: ready as of Phase 7; accent stays '#bc8cff'.
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#bc8cff',
        'factory': CitadelDesk,
    },
    'janestreet': {
        'name': 'Jane Street Desk',
        'firm_inspiration': 'Jane Street',
        'description': ('Fair-value engine, market-making simulator '
                        '(simulation-only), and IV-rank-driven defined-risk '
                        'premium selling with an earnings IV-crush module.'),
        # Contract C15: ready as of Phase 8; accent stays '#d29922'.
        # All four desks are now live.
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#d29922',
        'factory': JaneStreetDesk,
    },
}


def list_desks() -> List[Dict]:
    """Describe all desks for the Trading Floor (contract C1 shape)."""
    return [
        {
            'key': key,
            'name': spec['name'],
            'firm_inspiration': spec['firm_inspiration'],
            'description': spec['description'],
            'status': spec['status'],
            'activates_in_phase': spec['activates_in_phase'],
            'accent': spec['accent'],
        }
        for key, spec in _DESK_SPECS.items()
    ]


def create_desk(key: str, capital_allocation: float = 1.0) -> Desk:
    """Instantiate a ready desk by key.

    Raises:
        ValueError: ``Unknown desk: <key>`` for unregistered keys, or
            ``Desk '<key>' activates in Phase N`` for planned desks.
    """
    spec = _DESK_SPECS.get(key)
    if spec is None:
        raise ValueError(f"Unknown desk: {key}")
    if spec['status'] != 'ready' or spec['factory'] is None:
        raise ValueError(
            f"Desk '{key}' activates in Phase {spec['activates_in_phase']}")
    logger.info("Creating desk %s (capital_allocation=%.2f)",
                key, capital_allocation)
    return spec['factory'](capital_allocation=capital_allocation)
