"""Tests for the Flask application factory in gui/app.py.

Offline and deterministic: no network calls, no real market data. Market
data access points are monkeypatched; the app factory is exercised through
Flask's test client only.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gui.app import create_app
from data.market_data import MarketDataHandler

REPO_ROOT = Path(__file__).resolve().parents[1]

# SHARED INTERFACE CONTRACT: MarketDataHandler.get_last_fetch_info(symbol)
# returns a dict with exactly these keys (or None if never fetched).
PROVENANCE_CONTRACT_KEYS = {
    'provider', 'from_cache', 'fetched_at', 'failures', 'start_date', 'end_date',
}


def make_provenance_info(provider='cboe'):
    """A contract-shaped provenance dict for stubbing get_last_fetch_info."""
    return {
        'provider': provider,
        'from_cache': False,
        'fetched_at': '2026-06-11T00:00:00+00:00',
        'failures': [{'provider': 'fmp', 'error': 'missing API key'}],
        'start_date': '2025-06-11',
        'end_date': '2026-06-11',
    }


@pytest.fixture()
def app():
    """A fresh app per test, built through the factory with TESTING on."""
    return create_app({'TESTING': True})


@pytest.fixture()
def client(app):
    return app.test_client()


def test_health_returns_200_with_status_ok(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data == {'status': 'ok', 'service': 'stock-options-trader'}


def test_unknown_route_returns_404_json(client):
    response = client.get('/this-route-does-not-exist')
    assert response.status_code == 404
    assert response.get_json() == {'error': 'Not found'}


def test_500_handler_does_not_leak_traceback():
    app = create_app({'TESTING': True})

    @app.route('/boom')
    def boom():
        raise RuntimeError('synthetic explosion for testing')

    # Under TESTING, Flask propagates exceptions by default; force the
    # 500 error handler to run instead so we can inspect its response.
    app.config['PROPAGATE_EXCEPTIONS'] = False

    response = app.test_client().get('/boom')
    body = response.get_data(as_text=True)

    assert response.status_code == 500
    assert 'Internal server error' in body
    assert 'Traceback' not in body
    assert 'synthetic explosion for testing' not in body


def test_index_returns_200(client):
    response = client.get('/')
    assert response.status_code == 200


# ==================== LIVE BROKER UNAVAILABILITY (503) ====================


def test_live_trader_create_returns_503_when_broker_unavailable(client, monkeypatch):
    """Regression: a failed live-broker import must surface as a loud 503."""
    import gui.routes.api_trading as api_trading

    sentinel = 'ImportError: synthetic live-broker import failure'
    monkeypatch.setattr(api_trading, 'LiveEtradeBroker', None)
    monkeypatch.setattr(api_trading, 'LIVE_BROKER_IMPORT_ERROR', sentinel)

    response = client.post(
        '/api/trader/create', json={'trader_id': 't-live', 'mode': 'live'})

    assert response.status_code == 503
    assert response.get_json() == {
        'error': 'Live trading unavailable',
        'reason': sentinel,
    }


# ==================== LAZY DATABASE SINGLETON (gui/globals.py) ====================


class TestLazyDatabase:
    def test_importing_gui_app_creates_no_db_file_in_cwd(self, tmp_path):
        """Regression: gui.globals used to open TradingDatabase at import,
        leaking trading_data.db into whatever CWD imported it."""
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
        env.pop('TRADING_DB_PATH', None)
        code = (
            "import gui.app, gui.globals\n"
            "gui.app.create_app({'TESTING': True})\n"
        )
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert list(tmp_path.rglob('*.db')) == []

    def test_get_db_honors_trading_db_path_and_is_singleton(self, tmp_path, monkeypatch):
        import gui.globals as gui_globals

        db_file = tmp_path / 'nested' / 'custom.db'
        monkeypatch.setenv('TRADING_DB_PATH', str(db_file))
        monkeypatch.setattr(gui_globals, '_db', None)

        db = gui_globals.get_db()

        assert db is gui_globals.get_db()  # one shared instance per process
        assert db.db_path == str(db_file)
        assert db_file.exists()

    def test_module_level_db_attribute_resolves_lazily(self, tmp_path, monkeypatch):
        """Back-compat: ``from gui.globals import db`` still works."""
        import gui.globals as gui_globals

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'compat.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        assert gui_globals.db is gui_globals.get_db()


# ==================== DATA PROVENANCE (analysis + backtest) ====================


@pytest.fixture
def patch_fetch(monkeypatch):
    """Patch MarketDataHandler.fetch_stock_data class-wide to serve canned
    frames keyed by symbol (no network, deterministic)."""

    def _patch(frames_by_symbol):
        def fake_fetch(self, symbol, start_date, end_date):
            frame = frames_by_symbol.get(symbol)
            return frame.copy() if frame is not None else None

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

    return _patch


class TestAnalysisProvenance:
    def test_analyze_includes_data_source_matching_contract(
            self, client, monkeypatch, make_ohlcv, patch_fetch):
        patch_fetch({'TEST': make_ohlcv(n_days=120)})
        info = make_provenance_info()
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: info, raising=False)

        response = client.get('/api/analyze/TEST')

        assert response.status_code == 200
        body = response.get_json()
        # Existing keys are unchanged (additive only).
        assert body['symbol'] == 'TEST'
        assert body['current_price'] is not None
        assert 'data' in body
        # New provenance key matches the shared interface contract exactly.
        assert body['data_source'] == info
        assert set(body['data_source']) == PROVENANCE_CONTRACT_KEYS

    def test_analyze_data_source_is_null_for_never_fetched_symbol(
            self, client, monkeypatch, make_ohlcv, patch_fetch):
        """Contract: get_last_fetch_info returns None when never fetched."""
        patch_fetch({'TEST': make_ohlcv(n_days=120)})
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: None, raising=False)

        response = client.get('/api/analyze/TEST')

        assert response.status_code == 200
        assert response.get_json()['data_source'] is None

    def test_analyze_graceful_when_handler_lacks_get_last_fetch_info(
            self, client, monkeypatch, make_ohlcv, patch_fetch):
        """G3 guard: an older handler without the method must not 500."""
        patch_fetch({'TEST': make_ohlcv(n_days=120)})
        monkeypatch.delattr(
            MarketDataHandler, 'get_last_fetch_info', raising=False)

        response = client.get('/api/analyze/TEST')

        assert response.status_code == 200
        body = response.get_json()
        assert body['data_source'] is None
        assert body['current_price'] is not None


class TestBacktestProvenance:
    BACKTEST_PAYLOAD = {
        'strategy': 'momentum',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'position_size': 0.1,
    }

    def test_backtest_includes_data_sources_per_symbol(
            self, client, monkeypatch, make_ohlcv, patch_fetch):
        patch_fetch({
            'AAA': make_ohlcv(n_days=120, seed=1),
            'BBB': make_ohlcv(n_days=120, seed=2),
        })
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: make_provenance_info(provider=f'prov-{symbol}'),
            raising=False)

        response = client.post(
            '/api/backtest',
            json={**self.BACKTEST_PAYLOAD, 'symbols': 'AAA,BBB'})

        assert response.status_code == 200
        body = response.get_json()
        assert set(body['data_sources']) == {'AAA', 'BBB'}
        assert body['data_sources']['AAA']['provider'] == 'prov-AAA'
        assert body['data_sources']['BBB']['provider'] == 'prov-BBB'
        for info in body['data_sources'].values():
            assert set(info) == PROVENANCE_CONTRACT_KEYS

    def test_backtest_graceful_when_handler_lacks_get_last_fetch_info(
            self, client, monkeypatch, make_ohlcv, patch_fetch):
        """G3 guard: data_sources degrades to nulls, never a 500."""
        patch_fetch({'AAA': make_ohlcv(n_days=120, seed=1)})
        monkeypatch.delattr(
            MarketDataHandler, 'get_last_fetch_info', raising=False)

        response = client.post(
            '/api/backtest', json={**self.BACKTEST_PAYLOAD, 'symbols': 'AAA'})

        assert response.status_code == 200
        assert response.get_json()['data_sources'] == {'AAA': None}
