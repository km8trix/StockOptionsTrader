"""Backtest benchmark, OOS analysis, and report assembly services.

This module is intentionally downstream-only: it consumes an immutable view of
completed engine state and never imports :mod:`backtesting.backtest_engine`.
Keeping that dependency direction lets the engine retain compatibility facade
methods without creating an import cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from analysis.research_stats import (
    benjamini_hochberg,
    bonferroni_alpha,
    fold_oos_pvalue,
    fold_oos_tstat,
)
from portfolio.manager import PortfolioManager


logger = logging.getLogger(__name__)

BenchmarkLoader = Callable[[str, Optional[str], Optional[str]], Optional[Dict]]
OosFoldsLoader = Callable[..., Dict]


@dataclass(frozen=True)
class BacktestReportState:
    """References needed to serialize one completed backtest.

    The container itself is frozen so report construction cannot replace engine
    collaborators.  Lists and mappings deliberately retain their original
    identity because historical report contracts expose ``trades_log`` and
    ``portfolio_history`` directly.
    """

    strategy: Any
    driver: Any
    desk: Any
    desk_mode: bool
    orchestrator: Any
    reweighter: Any
    portfolio: PortfolioManager
    trades_log: List[Dict]
    pending_intents: Mapping[Any, Dict]
    pending_structures: Mapping[str, Dict]
    mechanics_enabled: bool
    mechanics_log: List[Dict]


def build_benchmark(
    market_data: Any,
    initial_capital: float,
    benchmark_symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    *,
    log: logging.Logger = logger,
) -> Optional[Dict]:
    """Build an optional buy-and-hold benchmark without failing a report."""
    try:
        data = market_data.fetch_stock_data(
            benchmark_symbol, start_date, end_date
        )
        if data is None or data.empty or "close" not in data.columns:
            raise ValueError(f"no usable data for {benchmark_symbol}")

        closes = data["close"].dropna()
        if closes.empty:
            raise ValueError(f"all closes NaN for {benchmark_symbol}")

        base_close = float(closes.iloc[0])
        if base_close <= 0:
            raise ValueError(
                f"non-positive base close {base_close} for {benchmark_symbol}"
            )

        equity_curve = [
            {
                "date": ts.strftime("%Y-%m-%d"),
                "value": float(close) / base_close * initial_capital,
            }
            for ts, close in closes.items()
        ]
        return {"symbol": benchmark_symbol, "equity_curve": equity_curve}
    except Exception as exc:  # noqa: BLE001 - benchmark is deliberately optional
        log.warning(
            "Benchmark %s unavailable (%s..%s): %s",
            benchmark_symbol,
            start_date,
            end_date,
            exc,
        )
        return None


def compute_oos_folds(
    portfolio: PortfolioManager, desk: Any, alpha: float = 0.05
) -> Dict:
    """Compute corrected account-level OOS significance by refit window."""
    fits = list(desk.walk_forward_fits)
    # Read boundaries through the serialized C3 contract.  Tagged fit wrappers
    # intentionally do not expose a direct ``fit_date`` attribute.
    boundaries = sorted(
        {
            pd.Timestamp(fit.to_dict()["fit_date"]).date()
            for fit in fits
        }
    )
    dated = portfolio.get_daily_returns_with_dates()

    folds: List[Dict] = []
    pvalues: List[Optional[float]] = []
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else None
        fold_returns = [
            value
            for day, value in dated
            if day >= start and (end is None or day < end)
        ]
        tstat = fold_oos_tstat(fold_returns)
        pvalue = fold_oos_pvalue(fold_returns)
        pvalues.append(pvalue)
        folds.append(
            {
                "fit_date": start.strftime("%Y-%m-%d"),
                "oos_start": start.strftime("%Y-%m-%d"),
                "oos_end": end.strftime("%Y-%m-%d") if end is not None else None,
                "n_returns": len(fold_returns),
                "mean_return": (
                    float(np.mean(fold_returns)) if fold_returns else None
                ),
                "tstat": tstat,
                "pvalue": pvalue,
            }
        )

    bh = benjamini_hochberg(pvalues, alpha)
    testable_count = bh["m"]
    corrected_alpha = bonferroni_alpha(alpha, testable_count)
    significant_bonferroni = 0
    for fold, pvalue, rejected_bh in zip(
        folds, pvalues, bh["rejected_bh"]
    ):
        fold["significant_bh"] = rejected_bh
        rejected_bonferroni = (
            pvalue is not None
            and corrected_alpha is not None
            and pvalue <= corrected_alpha
        )
        fold["significant_bonferroni"] = rejected_bonferroni
        if rejected_bonferroni:
            significant_bonferroni += 1

    return {
        "available": True,
        "alpha": alpha,
        "test": "one-sided (mean OOS return > 0)",
        "n_folds": len(folds),
        "n_testable_folds": testable_count,
        "bonferroni_alpha": corrected_alpha,
        "bh_threshold": bh["bh_threshold"],
        "n_significant_bonferroni": significant_bonferroni,
        "n_significant_bh": bh["n_significant_bh"],
        "folds": folds,
        "caveat": (
            "Account-level OOS folds sliced at distinct walk-forward "
            "fit dates: blended account returns (T+1 fill lag, "
            "netting, stops), not isolated per-model P&L. The first "
            "return in each fold is realized under the PRIOR model "
            "state (T+1 fill lag), so a boundary return reflects the "
            "prior model, not the refit at that boundary. Overlapping "
            "refit windows over-count independent trials, so "
            "Bonferroni/BH is a heuristic upper bound on "
            "multiple-testing severity, NOT exact FWER/FDR control. "
            "Independent of the deflated-Sharpe n_trials lens; do not "
            "combine."
        ),
    }


def generate_report(
    state: BacktestReportState,
    *,
    benchmark_symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    benchmark_loader: BenchmarkLoader,
    oos_folds_loader: OosFoldsLoader,
    log: logging.Logger = logger,
) -> Dict:
    """Serialize completed engine state without owning simulation behavior."""
    n_trials = 1
    if state.desk_mode:
        n_trials = max(1, len(state.driver.walk_forward_fits))
    summary = state.portfolio.get_summary(n_trials=n_trials)

    benchmark = None
    if benchmark_symbol:
        benchmark = benchmark_loader(benchmark_symbol, start_date, end_date)

    report = {
        "strategy": (
            state.strategy.name
            if state.strategy is not None
            else state.driver.name
        ),
        "summary": summary,
        "benchmark": benchmark,
        "drawdown_series": state.portfolio.get_drawdown_series(),
        "trades": state.trades_log,
        "closed_trades": [
            {
                "symbol": trade.asset.symbol,
                "instrument": str(trade.asset),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "entry_time": trade.entry_time.strftime("%Y-%m-%d"),
                "exit_time": trade.exit_time.strftime("%Y-%m-%d"),
            }
            for trade in state.portfolio.closed_trades
        ],
        "portfolio_history": state.portfolio.portfolio_history,
        "pending_signals": [
            {
                "symbol": asset.symbol,
                "signal": intent["signal"],
                "signal_date": intent["signal_date"],
            }
            for asset, intent in state.pending_intents.items()
        ]
        + [
            {
                "symbol": pending["structure"].legs[0].asset.symbol,
                "signal": (
                    "STRUCTURE_OPEN"
                    if pending["structure"].opening
                    else "STRUCTURE_CLOSE"
                ),
                "signal_date": pending["signal_date"],
                "intent_id": pending["structure"].intent_id,
            }
            for pending in state.pending_structures.values()
        ],
    }
    if state.mechanics_enabled:
        report["account_mechanics"] = [
            {
                **entry,
                "date": (
                    entry["date"].strftime("%Y-%m-%d")
                    if hasattr(entry.get("date"), "strftime")
                    else str(entry.get("date"))
                ),
            }
            for entry in state.mechanics_log
        ]

    if state.desk_mode:
        driver = state.driver
        report["desk"] = {"key": driver.key, "name": driver.name}
        report["trader_notes"] = [note.to_dict() for note in driver.notes]
        report["walk_forward"] = [
            fit.to_dict() for fit in driver.walk_forward_fits
        ]
        if state.orchestrator is not None:
            report["oos_folds"] = {
                "available": False,
                "reason": (
                    "netted multi-desk fund book — per-fold OOS "
                    "significance is not attributable to a single model"
                ),
            }
        else:
            try:
                # The bound engine callback is intentional: callers that patch
                # ``engine._compute_oos_folds`` retain the historical contract.
                report["oos_folds"] = oos_folds_loader(state.desk, alpha=0.05)
            except Exception as exc:  # noqa: BLE001 - report degrades to N/A
                log.warning(
                    "OOS fold computation failed; reporting N/A: %s", exc
                )
                report["oos_folds"] = {
                    "available": False,
                    "reason": f"computation failed: {exc}",
                }

        regime_series = getattr(driver, "regime_series", None)
        if regime_series:
            report["regime_series"] = list(regime_series)
        pod_history = getattr(driver, "pod_history", None)
        if pod_history:
            report["pod_history"] = list(pod_history)
        structures = getattr(driver, "structures_report", None)
        if structures is not None:
            report["structures"] = list(structures)
        greeks_series = getattr(driver, "greeks_series", None)
        if greeks_series is not None:
            report["greeks_series"] = list(greeks_series)

        if state.orchestrator is not None:
            report["orchestrator"] = {
                "desks": [
                    {
                        "key": desk.key,
                        "name": desk.name,
                        "capital_allocation": desk.capital_allocation,
                        "notes_count": len(desk.notes),
                    }
                    for desk in state.orchestrator.desks
                ],
                "active_capital": state.orchestrator.active_capital,
                "conflicts_resolved": state.orchestrator.conflicts_resolved,
            }
            if state.reweighter is not None:
                report["reweight_log"] = [
                    {
                        "date": entry["date"].strftime("%Y-%m-%d"),
                        "day_number": entry["day_number"],
                        "weights": entry["weights"],
                        "fallback": entry["fallback"],
                        "degraded_desks": entry["degraded_desks"],
                        "degrade_reason": entry["degrade_reason"],
                    }
                    for entry in state.reweighter.rebalance_log
                ]

    return report


__all__ = [
    "BacktestReportState",
    "build_benchmark",
    "compute_oos_folds",
    "generate_report",
]
