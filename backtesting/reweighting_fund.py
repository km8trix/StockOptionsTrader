"""
Reweighting fund backtest — the shadow-solo-book coordinator.

This is the user-facing entry point for an "accurate dynamic reweighter": a
fund backtest whose desk capital is risk-parity rebalanced IN-RUN, walk-forward
honestly, from each desk's STANDALONE performance.

HOW IT WORKS (two passes, one composition):
  1. Run each desk SOLO over the same symbols/window at full NAV (a normal
     single-desk BacktestEngine). Its equity curve is an undistorted measure of
     that desk's own risk — no netting, no cross-desk coupling, no per-desk
     attribution surgery on the unified book (see desks/dynamic_reweighter.py
     for why that attribution path was rejected).
  2. Run the FUND (FundOrchestrator over the same desks) with a
     DynamicReweighter pre-loaded with those solo curves. Every
     ``rebalance_every`` trading days the reweighter slices each curve to the
     rebalance date, risk-parity-weights the desks, and writes the new
     capital_allocation for the next window.

The solo curves do NOT depend on the fund (a desk solo is a fixed function of
the market data), so computing them in advance and slicing them at each
boundary is exactly equivalent to co-simulating — and far simpler. Walk-forward
honesty holds because each solo curve is itself causal and only points dated
on/before a rebalance are ever read.

COST: this runs N + 1 backtests (one solo pass per desk plus the fund). That is
the deliberate price of the accurate, undistorted per-desk signal.

INJECTION: ``solo_curve_provider`` and ``orchestrator_factory`` default to real
single-desk backtests and the registry, but are overridable so the composition
can be driven deterministically in tests without market data.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtesting.backtest_engine import BacktestEngine
from desks.capital_allocator import CrossDeskCapitalAllocator
from desks.dynamic_reweighter import DynamicReweighter
from desks.orchestrator import FundOrchestrator
from desks.registry import create_desk, create_fund_orchestrator

logger = logging.getLogger(__name__)

#: A solo curve is (timestamp, equity) pairs; a provider builds one per desk.
Curve = List[Tuple[pd.Timestamp, float]]
SoloCurveProvider = Callable[[str, Sequence[str], str, str], Curve]
OrchestratorFactory = Callable[[Dict[str, float]], FundOrchestrator]


class ReweightingFundBacktest:
    """Runs a fund backtest with in-run risk-parity reweighting from each
    desk's standalone curve (the shadow-solo-book design)."""

    def __init__(self, allocations: Dict[str, float], *,
                 initial_capital: float = 100_000.0,
                 commission: float = 0.001, slippage_bps: float = 5.0,
                 spread_pct: float = 0.02,
                 per_contract_commission: float = 0.65,
                 rebalance_every: int = 21, warmup: int = 0,
                 target_gross: float = 1.0,
                 weighting: str = 'risk_parity',
                 risk_aggregator=None, seed: Optional[int] = None,
                 sizing_modulator=None,
                 solo_curve_provider: Optional[SoloCurveProvider] = None,
                 orchestrator_factory: Optional[OrchestratorFactory] = None,
                 market_data=None, cash_yield=None):
        if not allocations:
            raise ValueError(
                "ReweightingFundBacktest requires at least one desk allocation")
        # Preserve insertion order — the orchestrator's deterministic netting
        # relies on desk order, and so does any tie-break in allocation.
        self.allocations = dict(allocations)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_bps = slippage_bps
        self.spread_pct = spread_pct
        self.per_contract_commission = per_contract_commission
        self.rebalance_every = rebalance_every
        self.warmup = warmup
        self.target_gross = target_gross
        # 'risk_parity' (default, unchanged) or opt-in guarded 'performance'.
        self.weighting = weighting
        self.risk_aggregator = risk_aggregator
        self.seed = seed
        # OPTIONAL, OFF-BY-DEFAULT gated RL execution throttle (Phase F unit 2).
        # None (default) => the fund pass is byte-identical to before. Only the
        # FUND pass receives it — the N solo passes measure undistorted desk
        # risk and must never be throttled.
        self.sizing_modulator = sizing_modulator
        # OPTIONAL market-data feed shared by the solo and fund engines
        # (e.g. WarehouseMarketData for survivorship-free desks). None (the
        # default) keeps every engine on its own MarketDataHandler —
        # byte-identical to before.
        self.market_data = market_data
        # OPTIONAL idle-cash yield (BacktestEngine cash_yield): a dated
        # date->annual-rate lookup or Series. None (default) = byte-identical
        # (cash earns zero). Threaded into the fund engine AND the default
        # solo-curve engines: the reweighter must see the same economics in
        # the shadow books as in the fund, or the risk-parity weights would
        # be computed off curves missing the yield the fund actually earns.
        self.cash_yield = cash_yield
        self._solo_curve_provider = (solo_curve_provider
                                     or self._default_solo_curve)
        self._orchestrator_factory = (orchestrator_factory
                                      or self._default_orchestrator)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, symbols: Sequence[str], start_date: str, end_date: str,
            benchmark_symbol: Optional[str] = 'SPY',
            progress_callback=None) -> Dict:
        """Compute each desk's solo curve, then run the fund with a
        DynamicReweighter pre-loaded with those curves. Returns the fund
        engine report (which carries the additive 'reweight_log')."""
        curves: Dict[str, Curve] = {}
        for key in self.allocations:
            try:
                curves[key] = self._solo_curve_provider(
                    key, symbols, start_date, end_date)
            except Exception as exc:
                # Don't let one desk's solo pass discard the others' (each is
                # an expensive backtest). An empty curve makes this desk
                # degenerate -> the rebalance falls back to equal weight and
                # records it, rather than crashing the whole job.
                logger.error(
                    "Solo curve provider for desk '%s' raised (%s); using an "
                    "empty curve (equal-weight fallback at rebalance)",
                    key, exc)
                curves[key] = []

        orchestrator = self._orchestrator_factory(self.allocations)
        reweighter = DynamicReweighter(
            CrossDeskCapitalAllocator(self.target_gross),
            rebalance_every=self.rebalance_every, warmup=self.warmup,
            weighting=self.weighting)
        reweighter.set_curves(curves)

        engine = BacktestEngine(
            orchestrator=orchestrator, reweighter=reweighter,
            initial_capital=self.initial_capital, commission=self.commission,
            slippage_bps=self.slippage_bps, spread_pct=self.spread_pct,
            per_contract_commission=self.per_contract_commission,
            seed=self.seed, market_data=self.market_data,
            cash_yield=self.cash_yield)
        # progress_callback covers only the fund pass (the N solo passes run
        # first); callers wanting whole-job progress should wrap accordingly.
        return engine.run(symbols, start_date, end_date,
                          benchmark_symbol=benchmark_symbol,
                          progress_callback=progress_callback)

    # ------------------------------------------------------------------
    # Default providers (overridable for tests)
    # ------------------------------------------------------------------
    def _default_solo_curve(self, key: str, symbols: Sequence[str],
                            start_date: str, end_date: str) -> Curve:
        """Run a desk SOLO at full NAV and return its equity curve.

        The solo pass intentionally omits the fund's ACCOUNT-LEVEL
        risk_aggregator (gross-leverage / correlation / sector across desks) —
        that overlay is meaningless for a single desk, and the curve is meant
        to measure the desk's OWN standalone risk. The desk's own apply_risk
        (per-name stops, daily-loss circuit, position ceiling) still runs.
        """
        desk = create_desk(key)  # capital_allocation defaults to 1.0 (full NAV)
        engine = BacktestEngine(
            desk=desk, initial_capital=self.initial_capital,
            commission=self.commission, slippage_bps=self.slippage_bps,
            spread_pct=self.spread_pct,
            per_contract_commission=self.per_contract_commission,
            seed=self.seed, market_data=self.market_data,
            cash_yield=self.cash_yield)
        report = engine.run(list(symbols), start_date, end_date,
                            benchmark_symbol=None)
        if 'error' in report:
            logger.warning(
                "Solo backtest for desk '%s' returned an error (%s); using an "
                "empty curve, so it falls back to equal weight at rebalance",
                key, report['error'])
            return []
        return [(pd.Timestamp(snapshot['timestamp']),
                 float(snapshot['portfolio_value']))
                for snapshot in report.get('portfolio_history', [])]

    def _default_orchestrator(self,
                              allocations: Dict[str, float]) -> FundOrchestrator:
        return create_fund_orchestrator(
            allocations, risk_aggregator=self.risk_aggregator,
            sizing_modulator=self.sizing_modulator)
