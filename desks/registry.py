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
from desks.orchestrator import FundOrchestrator
from desks.renaissance import RenaissanceDesk
from desks.twosigma import TwoSigmaDesk

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
    'twosigma': {
        'name': 'Two Sigma Desk',
        'firm_inspiration': 'Two Sigma',
        'description': ('Systematic cross-sectional long/short equity: a '
                        'walk-forward ML model zoo (single stacking ensemble '
                        'by default, optional multi-model committee) ranks the '
                        'universe — long the top quantile, short the bottom, '
                        'dollar-balanced.'),
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#3fb950',
        'factory': TwoSigmaDesk,
    },
}

#: Desk keys that accept a ``model_key`` (walk-forward model selection). Other
#: desks reject a non-None ``model_key`` with a clear ValueError.
_MODEL_SELECTABLE_DESKS = frozenset({'foundation', 'twosigma'})


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


def create_desk(key: str, capital_allocation: float = 1.0,
                model_key: Optional[str] = None) -> Desk:
    """Instantiate a ready desk by key.

    ``model_key`` selects the walk-forward model for desks that support
    model selection (``'foundation'`` and ``'twosigma'`` — see
    ``_MODEL_SELECTABLE_DESKS``). It is passed through to the desk factory;
    for any other desk a non-None ``model_key`` is rejected.
    ``model_key=None`` (the default) keeps every existing caller — including
    ``create_fund_orchestrator`` — byte-identical.

    Raises:
        ValueError: ``Unknown desk: <key>`` for unregistered keys;
            ``Desk '<key>' activates in Phase N`` for planned desks; or
            ``Desk '<key>' does not support model selection`` when
            ``model_key`` is given for a desk that does not accept one.
    """
    spec = _DESK_SPECS.get(key)
    if spec is None:
        raise ValueError(f"Unknown desk: {key}")
    if spec['status'] != 'ready' or spec['factory'] is None:
        raise ValueError(
            f"Desk '{key}' activates in Phase {spec['activates_in_phase']}")
    logger.info("Creating desk %s (capital_allocation=%.2f, model_key=%s)",
                key, capital_allocation, model_key)
    if model_key is not None:
        if key not in _MODEL_SELECTABLE_DESKS:
            raise ValueError(
                f"Desk '{key}' does not support model selection")
        return spec['factory'](capital_allocation=capital_allocation,
                               model_key=model_key)
    return spec['factory'](capital_allocation=capital_allocation)


def create_fund_orchestrator(allocations: Dict[str, float],
                             risk_aggregator=None) -> FundOrchestrator:
    """Instantiate the named ready desks at the given capital_allocations and
    wire them into a FundOrchestrator (convenience over create_desk + manual
    construction).

    ``allocations`` maps desk key -> capital_allocation; each desk is created
    via create_desk (so unknown/planned keys raise the same informative
    ValueErrors) and the FundOrchestrator validates the sum is <= 1.0. Pass an
    optional PortfolioRiskAggregator as ``risk_aggregator`` for the
    account-level overlay. Insertion order of ``allocations`` is preserved as
    the desk order (which the orchestrator's deterministic netting relies on).
    """
    if not allocations:
        raise ValueError(
            "create_fund_orchestrator requires at least one desk allocation")
    desks = [create_desk(key, allocation)
             for key, allocation in allocations.items()]
    return FundOrchestrator(desks, risk_aggregator=risk_aggregator)
