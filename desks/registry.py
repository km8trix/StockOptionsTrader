"""
Desk registry — the GUI's single source of truth for the Trading Floor.

list_desks() describes every desk (ready or planned, per the master plan
phases); create_desk() instantiates ready desks and raises informative
ValueErrors for unknown or not-yet-activated ones (contract C1).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from desks.aqr import AqrDesk
from desks.base import Desk
from desks.citadel import CitadelDesk
from desks.foundation import FoundationDesk
from desks.issuance import IssuanceDesk
from desks.janestreet import JaneStreetDesk
from desks.orchestrator import FundOrchestrator
from desks.pead import PEADDesk
from desks.renaissance import RenaissanceDesk
from desks.twosigma import TwoSigmaDesk
from desks.value_quality import ValueQualityDesk
from desks.vix_revert import VixReversionDesk

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
        # Backtest tuning R1 (2015-2024 multi-regime): the walk-forward gate
        # default (block score <= 0) was so conservative the desk barely
        # deployed and bled costs near flat (Sharpe -4.1). Lower the gate to
        # -0.05 so it trades more of a presumed-positive-edge model. Desk-class
        # default stays 0.0 (byte-identical).
        'config': {'gate_threshold': -0.05},
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
        # Earnings-entry gate (Phase 9): production Renaissance skips opening
        # new single-name MR / stat-arb positions within a few days of a
        # scheduled report (an idiosyncratic earnings jump swamps the signal).
        # create_desk lazily attaches an offline EarningsCache as the calendar
        # (see below); the desk-class default stays off and an un-ingested
        # cache reads empty, so this is byte-identical until the operator runs
        # `python -m data.earnings_cache`. Exits are never gated.
        # Backtest tuning R1: tried tightening the MR regime gate
        # (mr_prob_threshold 0.6 -> 0.75) to lift the low MR win rate, but it
        # was negligible (-57.1% -> -55.9%) — the loss is not driven by the
        # regime gate. Reverted; the param stays in the desk (default 0.6).
        'config': {'avoid_earnings_entries': True},
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
        # Robust pod Sharpe (Phase 5): production Citadel scores its pods by
        # median / MAD (outlier-resistant) instead of mean / std, so a single
        # blow-up day can't dominate a pod's capital weight. The desk-class
        # default stays mean/std (byte-identical); this is the config point.
        # Backtest tuning R1: tried allow_cut_recovery (re-admit cut pods) to
        # fix the post-2020 dormancy — but it made the desk WORSE (+8.8% ->
        # -22.3%): the pods have no edge, so re-deploying their capital just
        # churns and loses. The permanent cut was protective. Reverted; the
        # allow_cut_recovery param stays in the desk (default False, inert).
        'config': {'robust_pod_sharpe': True},
    },
    'janestreet': {
        'name': 'Jane Street Desk',
        'firm_inspiration': 'Jane Street',
        'description': ('Fair-value engine, market-making simulator '
                        '(simulation-only), and IV-rank-driven defined-risk '
                        'premium selling with an earnings IV-crush module. '
                        'Research/simulation-only: synthetic IV carries no '
                        'variance-risk-premium, so its premium-selling is not '
                        'a promotion candidate until a real options-data feed '
                        'is wired.'),
        # Contract C15: ready as of Phase 8; accent stays '#d29922'.
        # All four desks are now live.
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#d29922',
        'factory': JaneStreetDesk,
        # American pricing (Phase 5): every Jane Street option leg is a
        # single-name equity option (American in reality), so production
        # prices fills + structure marks with the CRR binomial tree, picking
        # up the early-exercise premium the put wings carry. Greeks stay
        # Black-Scholes; the desk-class default stays european (byte-
        # identical). This is the config point.
        # Backtest tuning R1: the relative-value book trades an UNBOUNDED-loss
        # stock pair sized at ~10% notional per leg — far hotter than the
        # defined-risk option books (~2% risk budget) — and spiraled the
        # account to -99.96% (total ruin) by 2018. Cap the RV leg fraction at
        # 0.02 so it can't blow up the desk. Desk-class default stays None
        # (byte-identical, uncapped).
        'config': {'exercise_style': 'american', 'rv_max_fraction': 0.02},
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
        # Turnover control (Phase 5): production desks run with a mild
        # hysteresis band (hold names while they stay in the top/bottom 30%,
        # vs the 20% entry quantile) and a 3-day minimum hold, damping churn.
        # The book mechanics + defaults stay no-op; this is the config point.
        # Backtest tuning R1: the ML book churned ~15 trades/day (38,750 total)
        # and lost in every regime — a daily-horizon signal swamped by turnover
        # costs. Widen the hold band (exit_quantile 0.4) and raise the minimum
        # hold to 21 days to cut churn ~7x and let any real edge survive costs.
        'turnover': {'exit_quantile': 0.4, 'min_holding_days': 21},
        # Signal-strength sizing (Phase 5): production splits each side's
        # budget by |alpha score| (conviction) instead of equal-weight,
        # clamped to the equal-weight cap so the book stays dollar-neutral
        # and gross-bounded. Desk-class default stays equal-weight.
        'config': {'size_by_signal_strength': True},
    },
    'aqr': {
        'name': 'AQR Desk',
        'firm_inspiration': 'AQR Capital Management',
        'description': ('Transparent classical-quant cross-sectional '
                        'long/short equity: price-based factors (momentum, '
                        'low-volatility, reversal, risk-adjusted momentum) '
                        'standardized cross-sectionally and combined by '
                        'ridge regression rank the universe — long the top '
                        'quantile, short the bottom, dollar-balanced.'),
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#f0883e',
        'factory': AqrDesk,
        'turnover': {'exit_quantile': 0.3, 'min_holding_days': 3},
        # Signal-strength sizing (Phase 5): same conviction tilt as Two Sigma
        # (per-side |score| budgeting, dollar-neutral, gross-bounded).
        # Backtest tuning R1: tried raising Ridge alpha 1.0 -> 5.0 to tame the
        # momentum-crash exposure, but it had ~no effect (-33.7% -> -33.5%) —
        # the crash isn't a regularization-strength problem. Reverted; the
        # factor_alpha param stays in the desk (default 1.0).
        'config': {'size_by_signal_strength': True},
    },
    'vixrevert': {
        'name': 'VIX Reversion Desk',
        'firm_inspiration': 'House',
        'description': ('Volatility mean reversion: buys SPY when VIX spikes '
                        'above 1.25x its 20-day average, exits when the spike '
                        'resolves or after a 21-day time stop; long-only, '
                        'reverse-martingale sized. Requires ^VIX as an '
                        'auxiliary (never traded) symbol in the run universe; '
                        'without it the desk stays flat. Gate 2015-2024 and '
                        '2005-2024: FAIL (positive expectancy, ~69% win rate, '
                        'but crisis-year losses leave it below the risk-free '
                        'hurdle) — see scripts/vixrevert_gate.py.'),
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#bf3989',
        'factory': VixReversionDesk,
    },
    'pead': {
        'name': 'PEAD Desk',
        'firm_inspiration': 'Academia (Bernard & Thomas)',
        'description': ('Post-earnings-announcement drift on true SUE from '
                        'PIT SF1 filings: long the strongest positive '
                        'standardized earnings surprises, short the strongest '
                        'negative, while the filing is fresh (63 days). '
                        'Requires the Sharadar PIT warehouse (sf1 ingested); '
                        'without it the desk stays flat. Run gates with '
                        'scripts/pead_desk_gate.py on the survivorship-free '
                        'warehouse feed.'),
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#9a6700',
        'factory': PEADDesk,
    },
    'value_quality': {
        'name': 'Value+Quality Desk',
        'firm_inspiration': 'AQR / Novy-Marx (academia)',
        'description': ('Cross-sectional value+quality composite: long the '
                        'cheapest (PIT price/book) most profitable (PIT net '
                        'margin) names, short the richest least profitable, '
                        'rank-blended monthly. Requires the Sharadar PIT '
                        'warehouse (sf1 + daily); without it the desk stays '
                        'flat. Built as the decorrelated second leg for the '
                        'PEAD combine (scripts/vq_fund_gate.py).'),
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#0969da',
        'factory': ValueQualityDesk,
    },
    'issuance': {
        'name': 'Net-Issuance Desk',
        'firm_inspiration': 'Academia (Pontiff & Woodgate / Daniel & Titman)',
        'description': ('Cross-sectional net share issuance: long the names '
                        'retiring shares (buybacks — lowest YoY '
                        'split-adjusted share growth from PIT SF1 filings), '
                        'short the heaviest issuers (suppressed by default: '
                        'long-only). Requires the Sharadar PIT warehouse '
                        '(sf1 + daily); without it the desk stays flat. '
                        'Built as the third-leg candidate for the PEAD+VQ '
                        'combine (scripts/vq_fund_gate.py --issuance).'),
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#8250df',
        'factory': IssuanceDesk,
    },
}

#: Desk keys that accept a ``model_key`` (walk-forward model selection). Other
#: desks reject a non-None ``model_key`` with a clear ValueError.
_MODEL_SELECTABLE_DESKS = frozenset({'foundation', 'twosigma'})

#: Backward-compatible Trading Floor research badges.  This set is never an
#: execution permission: paper/live construction is governed exclusively by
#: immutable artifacts in :mod:`analysis.promotion` and
#: :func:`create_deployed_desk`.  It remains empty until the UI is migrated to
#: display the richer three-tier decision without changing contract C1.
_PROMOTED_DESKS: frozenset = frozenset()


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
            'gate_status': 'promoted' if key in _PROMOTED_DESKS else 'research',
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
    # Production factory kwargs (empty for desks that don't declare them, so
    # those construct exactly as before): 'turnover' (cross-sectional churn
    # control) and 'config' (per-desk feature flags, e.g. citadel's robust
    # pod Sharpe) are merged into the factory call.
    # Validate model_key BEFORE constructing anything, so a rejected call
    # never instantiates the EarningsCache below.
    if model_key is not None and key not in _MODEL_SELECTABLE_DESKS:
        raise ValueError(f"Desk '{key}' does not support model selection")
    factory_kwargs = {**spec.get('turnover', {}), **spec.get('config', {})}
    # The earnings-entry gate is a dead flag without a calendar; attach an
    # OFFLINE EarningsCache (pure SQLite reads — no network in the backtest
    # loop). Imported and constructed lazily HERE, never at module import, so
    # listing/registering desks never opens the db. An un-ingested cache reads
    # empty => next_earnings None => byte-identical to no gate.
    if factory_kwargs.get('avoid_earnings_entries'):
        from data.earnings_cache import EarningsCache
        factory_kwargs.setdefault('earnings_calendar', EarningsCache())
    if model_key is not None:
        return spec['factory'](capital_allocation=capital_allocation,
                               model_key=model_key, **factory_kwargs)
    return spec['factory'](capital_allocation=capital_allocation,
                           **factory_kwargs)


def create_deployed_desk(
        key: str, *, artifact_hash: str, promotion_registry,
        required_level='live_eligible', runtime_code_sha: Optional[str] = None,
        runtime_parameters: Optional[Mapping[str, Any]] = None,
        capital_allocation: Optional[float] = None,
        model_key: Optional[str] = None) -> Desk:
    """Reconstruct Foundation solely from its exact approved artifact.

    ``create_desk`` intentionally remains available for research/backtests.
    Paper/live composition roots must use this boundary instead: the registry
    verifies an explicit immutable approval, then the complete frozen
    deployment configuration is parsed from ``artifact.parameters`` and built
    directly.  No caller-provided value participates in construction.

    ``runtime_code_sha`` and ``runtime_parameters`` are deprecated assertion-
    only compatibility arguments.  Code identity is always resolved
    internally from a clean git checkout; a supplied SHA must agree as an
    additional assertion.  ``capital_allocation`` and ``model_key`` likewise
    only assert equality when supplied; the approved values always win.
    """
    from analysis.promotion import (PromotionLevel, PromotionNotApproved,
                                    canonical_json)
    from desks.deployment_config import (
        FoundationDeploymentConfig, FoundationDeploymentIdentity)

    if key != 'foundation':
        raise PromotionNotApproved(
            f"deployed construction is not implemented for desk '{key}'")

    level = PromotionLevel(required_level)

    artifact = promotion_registry.require_approved(
        key, artifact_hash, level)

    try:
        config = FoundationDeploymentConfig.from_mapping(artifact.parameters)
    except (TypeError, ValueError) as exc:
        raise PromotionNotApproved(
            f"approved artifact has invalid Foundation parameters: {exc}") \
            from exc
    if config.strategy_version != artifact.strategy_version:
        raise PromotionNotApproved(
            'deployment strategy_version does not match the approved artifact')

    # Resolve code identity ourselves on every path.  A caller cannot prove
    # equivalence by merely echoing the SHA stored in the approved artifact.
    current_code_sha = _current_clean_code_sha(PromotionNotApproved)
    if artifact.code_sha != current_code_sha:
        raise PromotionNotApproved(
            'runtime code SHA does not match the approved artifact')
    if (runtime_code_sha is not None
            and str(runtime_code_sha) != current_code_sha):
        raise PromotionNotApproved(
            'caller code SHA assertion does not match the current runtime')

    if (runtime_parameters is not None
            and canonical_json(config.to_mapping())
            != canonical_json(runtime_parameters)):
        raise PromotionNotApproved(
            'runtime parameters do not match the approved artifact')
    if (capital_allocation is not None
            and float(capital_allocation) != config.capital_allocation):
        raise PromotionNotApproved(
            'runtime capital_allocation does not match the approved artifact')
    if model_key is not None and model_key != config.model_key:
        raise PromotionNotApproved(
            'runtime model_key does not match the approved artifact')

    desk = config.build()
    desk._bind_deployment_identity(FoundationDeploymentIdentity(
        artifact_hash=artifact.artifact_hash,
        strategy_id=artifact.strategy_id,
        strategy_version=artifact.strategy_version,
        required_level=level.value,
        code_sha=artifact.code_sha,
        config_hash=config.config_hash,
    ))
    return desk


def _current_clean_code_sha(error_type=RuntimeError) -> str:
    """Return the current clean checkout SHA or fail closed.

    Kept as a small seam so packaged deployments can replace the provenance
    resolver and tests can exercise the internal-identity path without shell
    mutation.  A dirty or unidentifiable checkout cannot claim equivalence to
    a committed artifact.
    """
    from utils.provenance import _git_dirty, _git_sha

    sha = _git_sha()
    dirty = _git_dirty()
    if not sha or dirty is not False:
        raise error_type('runtime code identity is dirty or unavailable')
    return sha


def create_fund_orchestrator(allocations: Dict[str, float],
                             risk_aggregator=None,
                             sizing_modulator=None) -> FundOrchestrator:
    """Instantiate the named ready desks at the given capital_allocations and
    wire them into a FundOrchestrator (convenience over create_desk + manual
    construction).

    ``allocations`` maps desk key -> capital_allocation; each desk is created
    via create_desk (so unknown/planned keys raise the same informative
    ValueErrors) and the FundOrchestrator validates the sum is <= 1.0. Pass an
    optional PortfolioRiskAggregator as ``risk_aggregator`` for the
    account-level overlay. Insertion order of ``allocations`` is preserved as
    the desk order (which the orchestrator's deterministic netting relies on).

    ``sizing_modulator`` (OPTIONAL, OFF BY DEFAULT) is the gated RL execution
    throttle (Phase F unit 2). When None (the default) the orchestrator step is
    byte-identical to before — nothing auto-enables it. Build one with
    :func:`create_rl_execution_throttle`; it is a STRICTLY SUBTRACTIVE size
    throttle, never a desk, never model-selectable, never live.
    """
    if not allocations:
        raise ValueError(
            "create_fund_orchestrator requires at least one desk allocation")
    desks = [create_desk(key, allocation)
             for key, allocation in allocations.items()]
    return FundOrchestrator(desks, risk_aggregator=risk_aggregator,
                            sizing_modulator=sizing_modulator)


def create_rl_execution_throttle(policy=None, scale_min: float = 0.5,
                                 scale_max: float = 1.0, enabled: bool = True,
                                 scaler=None):
    """Build a configured (orchestrator-detached) RL execution throttle.

    This is the ONLY registry entry point for the Phase F unit 2 throttle. It
    deliberately returns a detached :class:`RLExecutionThrottle` — the caller
    must explicitly pass it as ``sizing_modulator=`` to
    :func:`create_fund_orchestrator` (or directly to ``FundOrchestrator``) for
    it to take effect. It is NOT registered as a Desk, is NOT model-selectable,
    is NOT added to any default fund, and NOTHING auto-enables it. Clearing the
    research gate does not change that.

    ``policy`` is a FROZEN ``ThrottlePolicy`` (or None => identity pass-through);
    ``scale_max`` is HARD-CLAMPED to <= 1.0 inside the throttle (subtractive
    only). ``enabled=False`` makes the throttle a pure identity, and a still-
    untrained policy carries a NO-OP PRIOR (outputs ~scale_max), so even an
    attached-but-untrained policy leaves sizes essentially unchanged — throttling
    only ever shrinks once a policy has learned a signal. Only ``enabled=True``
    AND a trained ``ThrottlePolicy`` actually throttles.
    """
    # Imported lazily so the registry (and therefore the GUI) never hard-fails
    # to import when the optional torch wheel is missing — only constructing a
    # ThrottlePolicy requires torch, never importing this factory.
    from desks.rl_execution import RLExecutionThrottle
    return RLExecutionThrottle(
        policy=policy, scale_min=scale_min, scale_max=scale_max,
        enabled=enabled, scaler=scaler)
