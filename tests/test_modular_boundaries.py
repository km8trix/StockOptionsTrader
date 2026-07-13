"""Architecture budgets for the first behavior-preserving module split."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "backtesting" / "backtest_engine.py"
REPORTING = ROOT / "backtesting" / "reporting.py"
LIVE_BLUEPRINT = ROOT / "gui" / "routes" / "api_live.py"
LIVE_SUPPORT = ROOT / "gui" / "routes" / "live_order_support.py"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_reporting_dependency_points_away_from_engine():
    modules = imported_modules(REPORTING)
    assert "backtesting.backtest_engine" not in modules
    assert not any(module.startswith("desks.") for module in modules)


def test_live_order_support_is_framework_and_broker_independent():
    modules = imported_modules(LIVE_SUPPORT)
    assert "gui.routes.api_live" not in modules
    assert not any(module == "flask" or module.startswith("flask.")
                   for module in modules)
    assert not any(module.startswith("brokers.") for module in modules)


def test_monolith_line_budgets_pin_material_reduction():
    # Baselines before this extraction were 2,306 and 1,918 lines.  These
    # budgets preserve a >10% engine reduction and >6% live-route reduction.
    assert len(ENGINE.read_text(encoding="utf-8").splitlines()) <= 2_075
    assert len(LIVE_BLUEPRINT.read_text(encoding="utf-8").splitlines()) <= 1_800

