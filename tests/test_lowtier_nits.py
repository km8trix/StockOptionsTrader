"""Regression checks for the low-tier cleanup batch.

Each test fails against the pre-batch code and passes after the fix.
Offline, deterministic — no network, no real market data.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from core.models import Asset, AssetType, Position
from gui.app import create_app
from portfolio.manager import PortfolioManager
from utils.audit import AuditLog


@pytest.fixture()
def client():
    return create_app({'TESTING': True}).test_client()


# --- risk-settings validation (gui/routes/api_trading.py) ------------------
@pytest.mark.parametrize('payload', [
    {'max_position_size': 'not-a-number'},
    {'max_daily_loss': 2.0},         # > 1
    {'position_stop_loss': -0.1},    # <= 0
    {'max_sector_exposure': float('nan')},
])
def test_risk_settings_rejects_bad_input(client, payload):
    resp = client.post('/api/risk/settings', json=payload)
    assert resp.status_code == 400
    assert 'error' in resp.get_json()


def test_risk_settings_accepts_valid(client):
    resp = client.post('/api/risk/settings', json={'max_position_size': 0.1})
    assert resp.status_code == 200


# --- backtest export None-safe (gui/routes/api_backtest.py) -----------------
def test_export_report_null_metrics_no_500(client, monkeypatch):
    import gui.routes.api_backtest as m
    null_bt = {
        'strategy': 'x', 'timestamp': 't', 'start_date': 'a', 'end_date': 'b',
        'total_return': None, 'sharpe_ratio': None, 'max_drawdown': None,
        'win_rate': None, 'total_trades': 0, 'initial_capital': None,
    }
    monkeypatch.setattr(
        m, 'get_db', lambda: type('D', (), {'get_backtest': staticmethod(
            lambda _id: null_bt)})())
    resp = client.get('/api/export/report/1')
    assert resp.status_code == 200
    assert b'N/A' in resp.data


# --- realized-pnl cache is value-preserving (portfolio/manager.py) ----------
def test_realized_pnl_cache_equals_sum():
    pm = PortfolioManager(100_000.0)
    asset = Asset(symbol='AAA', asset_type=AssetType.STOCK)
    t0 = datetime(2024, 1, 1)
    for entry, exit_px, qty in [(10.0, 12.0, 100), (12.0, 11.0, 100),
                                (11.0, 15.0, 50)]:
        pm.add_position(Position(asset=asset, quantity=qty,
                                 avg_entry_price=entry, current_price=entry,
                                 timestamp=t0))
        pm.close_position(asset, exit_px, qty, t0, t0)
    assert pm.get_realized_pnl() == sum(t.pnl for t in pm.closed_trades)
    assert pm.get_realized_pnl() == pytest.approx((12 - 10) * 100
                                                  + (11 - 12) * 100
                                                  + (15 - 11) * 50)


# --- audit_log event_type index (utils/audit.py) ---------------------------
def test_audit_log_has_event_type_index(tmp_path):
    log = AuditLog(db_path=str(tmp_path / 'a.db'), env='sandbox')
    conn = log._connect()
    try:
        names = {r['name'] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
    finally:
        conn.close()
    assert 'idx_audit_log_event_type' in names
