"""Tests for the Flask application factory in gui/app.py.

Offline and deterministic: no network calls, no real market data. Market
data access points are monkeypatched; the app factory is exercised through
Flask's test client only.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
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


@pytest.fixture(autouse=True)
def _isolate_live_singletons(monkeypatch, tmp_path):
    """Keep every test away from the developer's REAL trading_data.db.

    kill_switch_engaged() (the base-template banner source, hit on every
    page render) reads through to persisted SQLite state when
    TRADING_DB_PATH is set OR the default db file already exists — and
    pytest runs from the repo root, which may well hold a real
    trading_data.db. Point the env at a per-test temp file and reset
    api_live's process-wide singletons so no test reads, mutates, or
    leaks live-trading state (real or another test's).
    """
    import gui.routes.api_live as api_live

    monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'live-test.db'))
    monkeypatch.setattr(api_live, '_auth_manager', None)
    monkeypatch.setattr(api_live, '_kill_switch', None)
    monkeypatch.setattr(api_live, '_audit_log', None)
    monkeypatch.setattr(api_live, '_scheduler', None)
    monkeypatch.setattr(api_live, '_keepalive_scheduler', None)
    monkeypatch.setattr(api_live, '_client', None)
    monkeypatch.setattr(api_live, '_client_auth_manager', None)
    api_live._ORDER_REF_CACHE.clear()


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


def test_400_handler_returns_json_for_malformed_body():
    """werkzeug's BadRequest (strict request.json failing before the route
    body runs) must come back as JSON for API consumers, not the default
    HTML error page."""
    from flask import request

    app = create_app({'TESTING': True})

    @app.route('/echo-json', methods=['POST'])
    def echo_json():
        return {'echo': request.json}  # strict parse: raises BadRequest

    response = app.test_client().post(
        '/echo-json', data='{not json', content_type='application/json')

    assert response.status_code == 400
    body = response.get_json()
    assert body is not None
    assert 'error' in body
    assert '<!doctype html>' not in response.get_data(as_text=True).lower()


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


def test_live_trader_create_503_without_account_id(client, monkeypatch):
    """Gap 5: refuse a live broker when ETRADE_ACCOUNT_ID_KEY is unset rather
    than route orders to account 'None'."""
    import gui.routes.api_live as api_live
    monkeypatch.setattr(api_live, 'get_auth_manager', lambda: object())
    monkeypatch.setattr(api_live, 'get_kill_switch', lambda: object())
    monkeypatch.setattr(api_live, 'get_audit_log', lambda: object())
    monkeypatch.delenv('ETRADE_ACCOUNT_ID_KEY', raising=False)

    response = client.post(
        '/api/trader/create', json={'trader_id': 't-noacct', 'mode': 'live'})
    assert response.status_code == 503
    assert 'ETRADE_ACCOUNT_ID_KEY' in response.get_json()['reason']


def test_wire_daily_loss_gate_attaches_breaker(monkeypatch, tmp_path):
    """Gap 1: the GUI client gets a daily-loss gate when account id + kill
    switch are available."""
    import gui.routes.api_live as api_live
    from brokers.circuit_breaker import DailyLossGate
    from utils.audit import AuditLog
    from utils.kill_switch import KillSwitch
    db = str(tmp_path / 'g.db')
    audit = AuditLog(db, env='sandbox')
    monkeypatch.setattr(api_live, 'get_kill_switch',
                        lambda: KillSwitch(db, audit=audit))
    monkeypatch.setattr(api_live, 'get_audit_log', lambda: audit)
    monkeypatch.setenv('ETRADE_ACCOUNT_ID_KEY', 'ACCT')

    class FakeClient:
        circuit_breaker = None

        def get_balances(self, _a):
            return 100_000.0

    c = FakeClient()
    api_live._wire_daily_loss_gate(c)
    assert isinstance(c.circuit_breaker, DailyLossGate)


def test_wire_daily_loss_gate_noop_without_account(monkeypatch, tmp_path):
    """Gap 1: no gate (no crash) when the target account is not configured."""
    import gui.routes.api_live as api_live
    from utils.audit import AuditLog
    from utils.kill_switch import KillSwitch
    db = str(tmp_path / 'g.db')
    audit = AuditLog(db, env='sandbox')
    monkeypatch.setattr(api_live, 'get_kill_switch',
                        lambda: KillSwitch(db, audit=audit))
    monkeypatch.setattr(api_live, 'get_audit_log', lambda: audit)
    monkeypatch.delenv('ETRADE_ACCOUNT_ID_KEY', raising=False)

    class FakeClient:
        circuit_breaker = None

    c = FakeClient()
    api_live._wire_daily_loss_gate(c)
    assert c.circuit_breaker is None


def test_market_hours_block_only_blocks_production_when_closed(app, monkeypatch):
    """Gap 3: off-hours blocks PRODUCTION orders (override-able), never sandbox."""
    import gui.routes.api_live as api_live

    class FakeMH:
        def is_market_open(self, _dt):
            return False

    monkeypatch.setattr(api_live, 'MarketHours', FakeMH)

    class ProdMgr:
        env = 'production'

    class SbMgr:
        env = 'sandbox'

    with app.app_context():
        monkeypatch.setattr(api_live, 'get_auth_manager', lambda: ProdMgr())
        assert api_live._market_hours_block({}) is not None        # blocked
        assert api_live._market_hours_block(
            {'allow_after_hours': True}) is None                    # override
        monkeypatch.setattr(api_live, 'get_auth_manager', lambda: SbMgr())
        assert api_live._market_hours_block({}) is None            # sandbox ok


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


class TestChartData:
    """Production /charts page + /api/chart endpoint (Phase B chart shell)."""

    def test_chart_endpoint_returns_lightweight_charts_shape(
            self, client, make_ohlcv, patch_fetch):
        patch_fetch({'TEST': make_ohlcv(n_days=30)})
        r = client.get('/api/chart/TEST')
        assert r.status_code == 200
        j = r.get_json()
        assert j['symbol'] == 'TEST'
        assert len(j['candles']) > 0
        assert len(j['volume']) == len(j['candles'])  # index-aligned
        assert set(j['candles'][0]) == {'time', 'open', 'high', 'low', 'close'}

    def test_chart_endpoint_404_for_no_data(self, client, patch_fetch):
        patch_fetch({})  # fetch returns None for any symbol
        assert client.get('/api/chart/NOPE').status_code == 404

    def test_charts_page_is_production_and_ships_the_library(self, client):
        html = client.get('/charts').get_data(as_text=True)
        assert 'lightweight-charts.standalone' in html
        assert 'id="chartContainer"' in html
        assert 'ws-badge-production' in html  # Production workspace badge


class TestChartPositionOverlay:
    """GET /api/live/positions/<symbol> — the Charts entry-price overlay."""

    def test_409_when_not_connected(self, client):
        # No connected client wired -> _require_client returns 409, fail-closed.
        assert client.get('/api/live/positions/SPY').status_code == 409

    def _connect(self, monkeypatch, portfolio):
        """Wire a fake connected client returning `portfolio` from get_portfolio."""
        import gui.routes.api_live as api_live

        class FakeMgr:
            def status(self):
                return {'state': 'connected'}

        class FakeClient:
            def get_portfolio(self, _id_key):
                return portfolio

        monkeypatch.setattr(api_live, 'get_auth_manager', lambda: FakeMgr())
        monkeypatch.setattr(api_live, 'get_client', lambda: FakeClient())
        monkeypatch.setenv('ETRADE_ACCOUNT_ID_KEY', 'ACCT')

    def test_503_without_configured_account(self, client, monkeypatch):
        self._connect(monkeypatch, [])
        monkeypatch.delenv('ETRADE_ACCOUNT_ID_KEY', raising=False)
        assert client.get('/api/live/positions/SPY').status_code == 503

    def test_filters_to_symbol_and_projects_fields(self, client, monkeypatch):
        self._connect(monkeypatch, [
            {'Product': {'symbol': 'SPY'}, 'quantity': 100,
             'pricePaid': 123.45, 'positionType': 'LONG'},
            {'symbolDescription': 'AAPL', 'quantity': -10,
             'pricePaid': 200.0, 'positionType': 'SHORT'},
        ])
        j = client.get('/api/live/positions/spy').get_json()  # case-insensitive
        assert j['positions'] == [{
            'symbol': 'SPY', 'quantity': 100,
            'price_paid': 123.45, 'position_type': 'LONG',
        }]
        # No account-number fields leak into the chart payload.
        assert 'ACCT' not in client.get('/api/live/positions/SPY').get_data(as_text=True)

    def test_empty_when_flat_in_symbol(self, client, monkeypatch):
        self._connect(monkeypatch, [
            {'Product': {'symbol': 'AAPL'}, 'quantity': 5, 'pricePaid': 1.0},
        ])
        assert client.get('/api/live/positions/SPY').get_json() == {'positions': []}


class TestGlossary:
    """UX polish phase 1: glossary offcanvas + explain() info popovers."""

    def test_glossary_has_core_terms(self):
        from gui.glossary import GLOSSARY
        assert {'Sandbox', 'Production', 'Desk', 'Strategy'} <= set(GLOSSARY)
        assert all(GLOSSARY.values())  # no empty definitions

    def test_offcanvas_ships_on_every_page(self, client):
        import markupsafe
        from gui.glossary import GLOSSARY
        # Jinja autoescapes the copy (e.g. & -> &amp;), so compare escaped.
        sandbox_html = str(markupsafe.escape(GLOSSARY['Sandbox']))
        for path in ('/', '/charts', '/backtest'):
            html = client.get(path).get_data(as_text=True)
            assert 'id="glossaryPanel"' in html, path
            # The actual definition copy is rendered, not just the panel shell.
            assert sandbox_html in html, path

    def test_dashboard_renders_explain_popovers(self, client):
        html = client.get('/').get_data(as_text=True)
        assert 'data-bs-toggle="popover"' in html
        assert 'data-bs-title="Sandbox"' in html
        assert 'data-bs-title="Production"' in html


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


class TestViewsAndVendoredAssets:
    """Phase 3 UI: every page renders and references only vendored assets."""

    # /trading_floor is the Stage 1 alias for /floor; both must keep working.
    PAGES = ['/', '/analysis', '/backtest', '/paper_trade', '/floor',
             '/trading_floor', '/live']

    @pytest.mark.parametrize('path', PAGES)
    def test_page_renders_200(self, client, path):
        response = client.get(path)
        assert response.status_code == 200

    @pytest.mark.parametrize('path', PAGES)
    def test_page_references_no_cdn(self, client, path):
        """Docker (Phase 4) must work offline: no CDN URLs in any page."""
        html = client.get(path).get_data(as_text=True)
        for marker in ('cdn.jsdelivr.net', 'cdn.plot.ly', 'https://cdn'):
            assert marker not in html, f'{path} references {marker}'

    def test_base_template_is_dark_theme_with_vendored_assets(self, client):
        html = client.get('/').get_data(as_text=True)
        assert 'data-bs-theme="dark"' in html
        assert 'vendor/bootstrap/bootstrap.min.css' in html
        assert 'vendor/bootstrap/bootstrap.bundle.min.js' in html
        assert 'vendor/bootstrap-icons/bootstrap-icons.min.css' in html

    def test_analysis_page_loads_vendored_plotly(self, client):
        html = client.get('/analysis').get_data(as_text=True)
        assert 'vendor/plotly/plotly.min.js' in html

    def test_backtest_page_loads_vendored_plotly_and_page_script(self, client):
        html = client.get('/backtest').get_data(as_text=True)
        assert 'vendor/plotly/plotly.min.js' in html
        assert 'js/backtest.js' in html

    def test_base_template_on_disk_has_no_cdn_references(self):
        """Read the template source itself, not just rendered output."""
        template = (REPO_ROOT / 'gui' / 'templates' / 'base.html').read_text()
        assert 'cdn.' not in template

    def test_floor_page_renders_desk_grid_and_floor_js(self, client):
        """Phase 5: desk cards are JS-rendered from GET /api/floor/desks,
        so the page ships the grid container + script, not hardcoded cards."""
        html = client.get('/floor').get_data(as_text=True)
        assert 'id="deskGrid"' in html
        assert 'js/floor.js' in html
        # The cross-desk synergy footnote survives the JS-rendered rewrite.
        assert 'Cross-desk synergy' in html
        # Phase 8 copy refresh: the footnote flags the MM simulator as
        # simulation-only and drops the stale 'until Jane Street arrives'
        # phrasing now that all four desks are ready.
        assert 'simulation-only' in html
        assert 'never trades live' in html
        assert 'Until the Jane Street desk arrives' not in html
        assert 'once that desk activates' not in html

    @pytest.mark.parametrize('rel_path', [
        'gui/static/vendor/bootstrap/bootstrap.min.css',
        'gui/static/vendor/bootstrap/bootstrap.bundle.min.js',
        'gui/static/vendor/bootstrap-icons/bootstrap-icons.min.css',
        'gui/static/vendor/bootstrap-icons/fonts/bootstrap-icons.woff2',
        'gui/static/vendor/bootstrap-icons/fonts/bootstrap-icons.woff',
        'gui/static/vendor/plotly/plotly.min.js',
    ])
    def test_vendored_asset_exists_and_is_non_empty(self, rel_path):
        asset = REPO_ROOT / rel_path
        assert asset.is_file(), f'missing vendored asset: {rel_path}'
        assert asset.stat().st_size > 1024


class TestAnalyzeSignalsAndDateRange:
    """Additive /api/analyze params: start/end range and strategy overlay."""

    def test_analyze_with_strategy_returns_signal_list(
            self, client, monkeypatch, make_ohlcv, patch_fetch):
        patch_fetch({'TEST': make_ohlcv(n_days=120)})
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: None, raising=False)

        response = client.get('/api/analyze/TEST?strategy=momentum')

        assert response.status_code == 200
        body = response.get_json()
        assert body['strategy'] == 'momentum'
        assert isinstance(body['signals'], list)
        for mark in body['signals']:
            assert mark['signal'] in ('BUY', 'SELL')
            assert set(mark) == {'date', 'signal', 'price'}

    def test_analyze_without_strategy_has_no_signals_key(
            self, client, monkeypatch, make_ohlcv, patch_fetch):
        patch_fetch({'TEST': make_ohlcv(n_days=120)})
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: None, raising=False)

        body = client.get('/api/analyze/TEST').get_json()
        assert 'signals' not in body
        assert 'strategy' not in body

    def test_analyze_unknown_strategy_returns_400(self, client):
        response = client.get('/api/analyze/TEST?strategy=does-not-exist')
        assert response.status_code == 400
        assert 'Unknown strategy' in response.get_json()['error']

    def test_analyze_inverted_date_range_returns_400(self, client):
        response = client.get(
            '/api/analyze/TEST?start=2024-06-01&end=2024-01-01')
        assert response.status_code == 400
        assert response.get_json()['error'] == 'start must be before end'

    def test_analyze_passes_explicit_range_to_fetch(
            self, client, monkeypatch, make_ohlcv):
        seen = {}
        frame = make_ohlcv(n_days=120)

        def fake_fetch(self, symbol, start_date, end_date):
            seen['start'], seen['end'] = start_date, end_date
            return frame.copy()

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: None, raising=False)

        response = client.get(
            '/api/analyze/TEST?start=2023-01-02&end=2023-06-30')

        assert response.status_code == 200
        assert seen == {'start': '2023-01-02', 'end': '2023-06-30'}


# ==================== ASYNC BACKTESTS (JobManager) ====================


def _wait_for_job(client, job_id, timeout=10.0):
    """Poll the status endpoint until the job finishes (daemon thread)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f'/api/backtest/status/{job_id}')
        assert response.status_code == 200
        job = response.get_json()
        if job['status'] in ('done', 'error'):
            return job
        time.sleep(0.02)
    pytest.fail(f'job {job_id} did not finish within {timeout}s')


def _fake_engine_report():
    """Engine-shaped report with raw pandas Timestamps, as run() returns.

    Faithful to PortfolioManager.get_summary(): 'total_return' is a DOLLAR
    P&L (realized + unrealized), 'total_return_pct' is x100.
    """
    ts = pd.Timestamp('2023-03-01')
    return {
        'strategy': 'MomentumStrategy',
        'summary': {
            'initial_capital': 100000.0, 'current_value': 108000.0,
            'cash': 8000.0, 'total_return': 8000.0, 'total_return_pct': 8.0,
            'realized_pnl': 5000.0, 'unrealized_pnl': 3000.0,
            'positions_count': 1, 'closed_trades': 4, 'win_rate': 75.0,
            'max_drawdown': -6.5, 'sharpe_ratio': 1.4, 'sortino_ratio': 2.1,
            'calmar_ratio': 1.2,
        },
        'benchmark': {
            'symbol': 'SPY',
            'equity_curve': [
                {'date': '2023-01-03', 'value': 100000.0},
                {'date': '2023-12-29', 'value': 112000.0},
            ],
        },
        'drawdown_series': [
            {'date': '2023-01-03', 'drawdown_pct': 0.0},
            {'date': '2023-12-29', 'drawdown_pct': -6.5},
        ],
        'trades': [{
            'date': ts, 'signal_date': ts - pd.Timedelta(days=1),
            'symbol': 'AAA', 'action': 'BUY', 'quantity': 10,
            'price': 100.0, 'cost': 1001.0,
            # Contract C13 (Phase 8, additive): str(asset) — the bare
            # symbol for stocks, the full contract string for options.
            'instrument': 'AAA',
        }],
        'closed_trades': [],
        'portfolio_history': [
            {'timestamp': pd.Timestamp('2023-01-03'),
             'portfolio_value': 100000.0, 'cash': 100000.0,
             'positions_count': 0, 'unrealized_pnl': 0.0, 'realized_pnl': 0.0},
        ],
        'pending_signals': [
            {'symbol': 'AAA', 'signal': 'SELL',
             'signal_date': pd.Timestamp('2023-12-29')},
        ],
    }


class TestAsyncBacktest:
    """POST /api/backtest/run + GET /api/backtest/status/<job_id>."""

    PAYLOAD = {
        'symbols': 'AAA',
        'strategy': 'momentum',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'position_size': 0.1,
    }

    @pytest.fixture()
    def patch_engine_run(self, monkeypatch, tmp_path):
        """Stub BacktestEngine.run, provenance, and redirect the history DB."""
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        seen = {'progress_values': []}

        def fake_run(self, symbols, start_date, end_date, position_size,
                     progress_callback=None, benchmark_symbol='SPY'):
            seen['symbols'] = list(symbols)
            seen['benchmark_symbol'] = benchmark_symbol
            if progress_callback is not None:
                progress_callback(42.0)
                seen['progress_values'].append(42.0)
            return _fake_engine_report()

        monkeypatch.setattr(api_backtest.BacktestEngine, 'run', fake_run)
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: make_provenance_info(provider='cboe'),
            raising=False)
        return seen

    def test_run_returns_job_id_then_done_with_full_report(
            self, client, patch_engine_run):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)

        assert response.status_code == 202
        job_id = response.get_json()['job_id']
        assert job_id

        job = _wait_for_job(client, job_id)
        assert job['status'] == 'done'
        assert job['error'] is None
        assert job['progress'] == 100.0
        assert patch_engine_run['progress_values'] == [42.0]
        assert patch_engine_run['benchmark_symbol'] == 'SPY'

        result = job['result']
        # The engine summary passes through unconverted: total_return stays
        # a dollar P&L, total_return_pct stays x100.
        assert result['summary']['total_return'] == pytest.approx(8000.0)
        assert result['summary']['total_return_pct'] == pytest.approx(8.0)
        assert result['benchmark']['symbol'] == 'SPY'
        assert result['drawdown_series'][-1]['drawdown_pct'] == -6.5
        assert set(result['data_sources']) == {'AAA'}
        assert set(result['data_sources']['AAA']) == PROVENANCE_CONTRACT_KEYS
        # Dates are normalized to ISO strings for the frontend.
        assert result['trades'][0]['date'] == '2023-03-01'
        assert result['trades'][0]['signal_date'] == '2023-02-28'
        # Contract C13: the additive 'instrument' field survives the trade
        # date normalization in strategy mode too.
        assert result['trades'][0]['instrument'] == 'AAA'
        assert result['portfolio_history'][0]['timestamp'] == '2023-01-03'
        assert result['pending_signals'][0]['signal_date'] == '2023-12-29'

    def test_finished_run_is_saved_to_history(self, client, patch_engine_run):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        listing = client.get('/api/backtests')
        assert listing.status_code == 200
        rows = listing.get_json()['backtests']
        assert len(rows) == 1
        assert rows[0]['strategy'] == 'momentum'
        # All three ':.2%'-formatted DB columns store FRACTIONS: total_return
        # converted from the engine's dollar P&L ($8,000 on $100k = 0.08),
        # max_drawdown / win_rate converted from the engine's x100 percents
        # (-6.5 -> -0.065, 75.0 -> 0.75).
        assert rows[0]['total_return'] == pytest.approx(0.08)
        assert rows[0]['max_drawdown'] == pytest.approx(-0.065)
        assert rows[0]['win_rate'] == pytest.approx(0.75)

    def test_export_report_formats_saved_run_percents(
            self, client, patch_engine_run):
        """Regression: export_report's ':.2%' over a run saved via the async
        path must print human percents, not x100 garbage (e.g. '-650.00%')."""
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        backtest_id = client.get('/api/backtests').get_json()['backtests'][0]['id']
        report = client.get(f'/api/export/report/{backtest_id}')
        assert report.status_code == 200
        text = report.get_data(as_text=True)
        assert 'Total Return: 8.00%' in text
        assert 'Max Drawdown: -6.50%' in text
        assert 'Win Rate: 75.00%' in text
        assert 'Final Value: $108,000.00' in text

    def test_total_return_fraction_units(self):
        """Dollar->fraction conversion for the persisted total_return column."""
        from gui.routes.api_backtest import _total_return_fraction

        # Preferred path: total_return_pct (x100) -> fraction.
        assert _total_return_fraction(
            {'total_return': 8000.0, 'total_return_pct': 8.0}, 100000.0,
        ) == pytest.approx(0.08)
        # Fallback: dollar P&L over initial capital.
        assert _total_return_fraction(
            {'total_return': -2500.0}, 100000.0,
        ) == pytest.approx(-0.025)
        # No usable inputs -> None (NaN pct is nulled upstream).
        assert _total_return_fraction({'total_return': 8000.0}, 0) is None
        assert _total_return_fraction({}, 100000.0) is None

    def test_pct_to_fraction_units(self):
        """x100 percent -> fraction for the max_drawdown/win_rate columns."""
        from gui.routes.api_backtest import _pct_to_fraction

        assert _pct_to_fraction(-6.5) == pytest.approx(-0.065)
        assert _pct_to_fraction(75.0) == pytest.approx(0.75)
        assert _pct_to_fraction(0.0) == 0.0
        assert _pct_to_fraction(None) is None

    def test_engine_failure_surfaces_as_error_status(
            self, client, monkeypatch, tmp_path):
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        def exploding_run(self, *args, **kwargs):
            raise RuntimeError('synthetic backtest explosion')

        monkeypatch.setattr(api_backtest.BacktestEngine, 'run', exploding_run)

        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])

        assert job['status'] == 'error'
        assert job['error'] == 'synthetic backtest explosion'
        assert job['result'] is None
        # Only the message crosses the wire — never a traceback.
        assert 'Traceback' not in str(job)

    def test_run_records_provenance_with_seed(self, client, patch_engine_run):
        response = client.post(
            '/api/backtest/run', json={**self.PAYLOAD, 'seed': 99})
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        prov = job['result']['provenance']
        assert set(prov) == {'git_sha', 'git_dirty', 'python_version',
                             'dependency_versions', 'seed', 'captured_at'}
        assert prov['seed'] == 99  # the pinned seed is recorded

    def test_run_provenance_seed_none_when_unpinned(
            self, client, patch_engine_run):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['result']['provenance']['seed'] is None

    @pytest.mark.parametrize('bad_seed', ['abc', 3.14, True, False])
    def test_run_rejects_non_integer_seed(self, client, bad_seed):
        # bool is an int subclass, so True/False must be rejected explicitly.
        response = client.post(
            '/api/backtest/run', json={**self.PAYLOAD, 'seed': bad_seed})
        assert response.status_code == 400
        assert 'seed' in response.get_json()['error']

    def test_run_rejects_missing_symbols(self, client):
        response = client.post(
            '/api/backtest/run', json={**self.PAYLOAD, 'symbols': '  '})
        assert response.status_code == 400
        assert response.get_json()['error'] == 'No symbols provided'

    def test_run_accepts_symbols_as_json_array(self, client, patch_engine_run):
        """A JSON array of symbols is a natural client payload — no 500."""
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'symbols': ['aaa', ' bbb ']})

        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        assert patch_engine_run['symbols'] == ['AAA', 'BBB']

    @pytest.mark.parametrize('bad_symbols', [
        123, 1.5, {'AAA': 1}, ['AAA', 7], [None], [['AAA']],
    ])
    def test_run_rejects_non_string_symbols_with_400(
            self, client, bad_symbols):
        """Unparseable 'symbols' shapes are a 400 JSON, never a 500."""
        response = client.post(
            '/api/backtest/run', json={**self.PAYLOAD, 'symbols': bad_symbols})

        assert response.status_code == 400
        assert 'symbols' in response.get_json()['error']

    def test_run_empty_json_body_returns_400_json(self, client):
        """Empty body + JSON content type must yield a JSON 400 (the route
        parses with silent=True), not werkzeug's HTML BadRequest page."""
        response = client.post(
            '/api/backtest/run', data='', content_type='application/json')

        assert response.status_code == 400
        assert response.get_json() == {'error': 'No symbols provided'}

    def test_parse_symbols_units(self):
        """String and list payloads normalize; anything else returns None."""
        from gui.routes.api_backtest import _parse_symbols

        assert _parse_symbols('aapl, msft') == ['AAPL', 'MSFT']
        assert _parse_symbols(['aapl', ' msft ']) == ['AAPL', 'MSFT']
        assert _parse_symbols('  ') == []
        assert _parse_symbols([]) == []
        assert _parse_symbols(123) is None
        assert _parse_symbols(['AAPL', 7]) is None
        assert _parse_symbols({'0': 'AAPL'}) is None

    def test_run_rejects_unknown_strategy(self, client):
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'strategy': 'does-not-exist'})
        assert response.status_code == 400
        assert 'Unknown strategy' in response.get_json()['error']

    def test_run_rejects_inverted_date_range(self, client):
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD,
                  'start_date': '2024-01-01', 'end_date': '2023-01-01'})
        assert response.status_code == 400

    def test_status_unknown_job_returns_404(self, client):
        response = client.get('/api/backtest/status/no-such-job')
        assert response.status_code == 404
        assert response.get_json() == {'error': 'Unknown job id'}


def _fake_fund_report():
    """Fund-engine-shaped report (the additive 'orchestrator' + 'reweight_log'
    keys alongside the usual desk-mode shape), as ReweightingFundBacktest.run
    returns before _json_safe_report normalizes dates."""
    ts = pd.Timestamp('2023-03-01')
    return {
        'strategy': 'Fund Orchestrator',
        'desk': {'key': 'fund', 'name': 'Fund Orchestrator'},
        'summary': {
            'initial_capital': 100000.0, 'current_value': 105000.0,
            'cash': 5000.0, 'total_return': 5000.0, 'total_return_pct': 5.0,
            'realized_pnl': 3000.0, 'unrealized_pnl': 2000.0,
            'positions_count': 2, 'closed_trades': 6, 'win_rate': 66.7,
            'max_drawdown': -4.0, 'sharpe_ratio': 1.1, 'sortino_ratio': 1.6,
            'calmar_ratio': 0.9,
        },
        'benchmark': None,
        'drawdown_series': [{'date': '2023-01-03', 'drawdown_pct': 0.0}],
        'trades': [{
            'date': ts, 'signal_date': ts - pd.Timedelta(days=1),
            'symbol': 'AAA', 'action': 'BUY', 'quantity': 10,
            'price': 100.0, 'value': 1000.0, 'instrument': 'AAA',
        }],
        'closed_trades': [],
        'portfolio_history': [
            {'timestamp': pd.Timestamp('2023-01-03'),
             'portfolio_value': 100000.0, 'cash': 100000.0,
             'positions_count': 0, 'unrealized_pnl': 0.0, 'realized_pnl': 0.0},
        ],
        'pending_signals': [],
        'trader_notes': [],
        'walk_forward': [],
        'orchestrator': {
            'desks': [
                {'key': 'foundation', 'name': 'Foundation Desk',
                 'capital_allocation': 0.6, 'notes_count': 0},
                {'key': 'renaissance', 'name': 'Renaissance Desk',
                 'capital_allocation': 0.4, 'notes_count': 0},
            ],
            'active_capital': 1.0, 'conflicts_resolved': 0,
        },
        'reweight_log': [
            {'date': '2023-02-01', 'day_number': 21,
             'weights': {'foundation': 0.6, 'renaissance': 0.4},
             'fallback': False, 'degraded_desks': []},
        ],
    }


class TestFundBacktest:
    """POST /api/backtest/run fund mode (ReweightingFundBacktest)."""

    FUND_PAYLOAD = {
        'symbols': 'AAA,BBB',
        'fund': {'foundation': 0.6, 'renaissance': 0.4},
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
    }

    @pytest.fixture()
    def patch_fund_run(self, monkeypatch, tmp_path):
        """Stub ReweightingFundBacktest.run and redirect the history DB. The
        eager create_fund_orchestrator validation still runs against the REAL
        registry (foundation/renaissance are ready), so payload validation is
        exercised end to end; only the expensive .run is faked."""
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'fund-history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        seen = {'progress_values': []}

        def fake_run(self, symbols, start_date, end_date,
                     benchmark_symbol='SPY', progress_callback=None):
            seen['symbols'] = list(symbols)
            seen['benchmark_symbol'] = benchmark_symbol
            if progress_callback is not None:
                progress_callback(55.0)
                seen['progress_values'].append(55.0)
            return _fake_fund_report()

        monkeypatch.setattr(api_backtest.ReweightingFundBacktest, 'run',
                            fake_run)
        return seen

    def test_run_returns_job_then_done_with_reweight_log(
            self, client, patch_fund_run):
        response = client.post('/api/backtest/run', json=self.FUND_PAYLOAD)
        assert response.status_code == 202

        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        assert job['progress'] == 100.0
        assert patch_fund_run['symbols'] == ['AAA', 'BBB']
        assert patch_fund_run['benchmark_symbol'] == 'SPY'

        result = job['result']
        assert result['desk']['key'] == 'fund'
        assert result['orchestrator']['active_capital'] == 1.0
        entry = result['reweight_log'][0]
        assert entry['weights'] == {'foundation': 0.6, 'renaissance': 0.4}
        assert entry['fallback'] is False
        assert entry['degraded_desks'] == []
        # Trade dates still normalize to ISO strings on the fund path.
        assert result['trades'][0]['date'] == '2023-03-01'

    def test_fund_run_records_provenance_with_seed(
            self, client, patch_fund_run):
        response = client.post(
            '/api/backtest/run', json={**self.FUND_PAYLOAD, 'seed': 7})
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        assert job['result']['provenance']['seed'] == 7

    def test_finished_fund_run_saved_as_fund_strategy(
            self, client, patch_fund_run):
        response = client.post('/api/backtest/run', json=self.FUND_PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        rows = client.get('/api/backtests').get_json()['backtests']
        assert rows[0]['strategy'] == 'fund:foundation+renaissance'

    def test_fund_accepts_list_payload_equal_weight(
            self, client, patch_fund_run):
        response = client.post(
            '/api/backtest/run',
            json={**self.FUND_PAYLOAD, 'fund': ['foundation', 'renaissance']})
        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

    def test_fund_rejects_combo_with_strategy(self, client):
        response = client.post(
            '/api/backtest/run',
            json={**self.FUND_PAYLOAD, 'strategy': 'momentum'})
        assert response.status_code == 400
        assert 'only one' in response.get_json()['error']

    def test_fund_rejects_empty_allocations(self, client):
        response = client.post(
            '/api/backtest/run', json={**self.FUND_PAYLOAD, 'fund': {}})
        assert response.status_code == 400

    def test_fund_rejects_non_positive_weight(self, client):
        response = client.post(
            '/api/backtest/run',
            json={**self.FUND_PAYLOAD, 'fund': {'foundation': 0}})
        assert response.status_code == 400
        assert 'must be > 0' in response.get_json()['error']

    def test_fund_rejects_overallocation(self, client):
        response = client.post(
            '/api/backtest/run',
            json={**self.FUND_PAYLOAD, 'fund': {'foundation': 0.6,
                                                'citadel': 0.6}})
        assert response.status_code == 400
        assert 'must be <= 1.0' in response.get_json()['error']

    def test_fund_rejects_unknown_desk(self, client):
        response = client.post(
            '/api/backtest/run',
            json={**self.FUND_PAYLOAD, 'fund': {'warrenbuffett': 1.0}})
        assert response.status_code == 400
        assert 'Unknown desk' in response.get_json()['error']

    @pytest.mark.parametrize('bad', [
        {'rebalance_every': 0}, {'warmup': -1}, {'target_gross': 0},
        {'target_gross': 1.5},
    ])
    def test_fund_rejects_bad_reweight_params(self, client, bad):
        response = client.post(
            '/api/backtest/run', json={**self.FUND_PAYLOAD, **bad})
        assert response.status_code == 400

    def test_fund_503_when_framework_unavailable(self, client, monkeypatch):
        from gui.routes import api_backtest
        sentinel = 'ImportError: synthetic desks import failure'
        monkeypatch.setattr(api_backtest, 'create_desk', None)
        monkeypatch.setattr(api_backtest, 'ReweightingFundBacktest', None)
        monkeypatch.setattr(api_backtest, 'DESK_REGISTRY_IMPORT_ERROR',
                            sentinel)
        response = client.post('/api/backtest/run', json=self.FUND_PAYLOAD)
        assert response.status_code == 503
        assert response.get_json()['reason'] == sentinel

    def test_fund_rejects_desk_model_with_400(self, client, patch_fund_run):
        """L1: desk_model is a single-desk (foundation) selector with no
        meaning in a multi-desk fund run. A fund payload carrying desk_model
        is a clean 400 with the documented message, before any backtest
        runs (the patch_fund_run stub proves .run is never reached)."""
        response = client.post(
            '/api/backtest/run',
            json={**self.FUND_PAYLOAD, 'desk_model': 'lightgbm'})
        assert response.status_code == 400
        assert response.get_json()['error'] == (
            'desk_model is not valid in fund mode')
        # Rejected before submission: the fund engine was never invoked.
        assert 'symbols' not in patch_fund_run

    def test_fund_without_desk_model_is_not_rejected_on_that_ground(
            self, client, patch_fund_run):
        """L1 (counterpart): an otherwise-valid fund payload WITHOUT
        desk_model must NOT be rejected — it submits and runs (mocked)."""
        response = client.post('/api/backtest/run', json=self.FUND_PAYLOAD)
        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        # Not blocked on the desk_model ground: the run actually executed.
        assert patch_fund_run['symbols'] == ['AAA', 'BBB']

    def test_parse_allocations_units(self):
        from gui.routes.api_backtest import _parse_allocations

        assert _parse_allocations(['Foundation', 'Renaissance']) == (
            {'foundation': 0.5, 'renaissance': 0.5}, None)
        assert _parse_allocations({'foundation': 0.6, 'citadel': 0.4}) == (
            {'foundation': 0.6, 'citadel': 0.4}, None)
        assert _parse_allocations({})[0] is None
        assert _parse_allocations('nope')[0] is None
        assert _parse_allocations(['a', 'a'])[0] is None
        assert _parse_allocations({'foundation': 'x'})[0] is None
        assert _parse_allocations({'foundation': -1})[0] is None


# ==================== TRADING FLOOR DESKS (Phase 5) ====================

# SHARED INTERFACE CONTRACT C1: desks.registry.list_desks() entries carry
# exactly these keys.
DESK_CONTRACT_KEYS = {
    'key', 'name', 'firm_inspiration', 'description', 'status',
    'activates_in_phase', 'accent',
}


def make_desk_entry(**overrides):
    """A contract-C1-shaped desk dict for stubbing list_desks."""
    entry = {
        'key': 'foundation',
        'name': 'Foundation',
        'firm_inspiration': 'In-house',
        'description': 'Baseline desk wrapping the classic strategies.',
        'status': 'ready',
        'activates_in_phase': None,
        'accent': '#3fb950',
    }
    entry.update(overrides)
    return entry


class TestFloorDesksEndpoint:
    """GET /api/floor/desks proxies desks.registry.list_desks (contract C1).

    These tests stub the registry binding inside the route module, so they
    pin the ROUTE's behavior regardless of registry contents.
    """

    def test_returns_desks_from_registry(self, client, monkeypatch):
        from gui.routes import api_floor

        desks = [
            make_desk_entry(),
            make_desk_entry(key='citadel', name='Citadel',
                            firm_inspiration='Citadel',
                            status='planned', activates_in_phase=7,
                            accent='#bc8cff'),
        ]
        monkeypatch.setattr(api_floor, 'list_desks', lambda: desks)

        response = client.get('/api/floor/desks')

        assert response.status_code == 200
        body = response.get_json()
        assert body == {'desks': desks}
        for desk in body['desks']:
            assert set(desk) == DESK_CONTRACT_KEYS

    def test_returns_503_when_desk_framework_unavailable(
            self, client, monkeypatch):
        """Mirrors the live-broker pattern: a failed desks import is LOUD."""
        from gui.routes import api_floor

        sentinel = 'ImportError: synthetic desks import failure'
        monkeypatch.setattr(api_floor, 'list_desks', None)
        monkeypatch.setattr(api_floor, 'DESK_REGISTRY_IMPORT_ERROR', sentinel)

        response = client.get('/api/floor/desks')

        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Desk framework unavailable',
            'reason': sentinel,
        }


class TestFloorDesksRegistryContract:
    """End-to-end /api/floor/desks against the REAL desks.registry.

    Skips (instead of failing) while a parallel backend task that owns
    desks/ has not landed in this tree yet (Phase 5: the package itself;
    Phase 6/7/8: the renaissance/citadel/janestreet status flips —
    contracts C6, C10, C15).
    """

    @pytest.fixture(autouse=True)
    def _require_desks(self):
        pytest.importorskip(
            'desks.registry',
            reason='desks package not present (parallel Phase 5 backend task)')

    def test_every_desk_matches_contract_shape(self, client):
        response = client.get('/api/floor/desks')

        assert response.status_code == 200
        desks = response.get_json()['desks']
        assert desks, 'registry returned no desks'
        for desk in desks:
            assert set(desk) == DESK_CONTRACT_KEYS
            assert desk['status'] in ('ready', 'planned')
            assert re.fullmatch(r'#[0-9a-fA-F]{6}', desk['accent'])
            if desk['status'] == 'planned':
                assert isinstance(desk['activates_in_phase'], int)

    def test_all_four_desks_ready(self, client):
        """Phase 8 (contract C15): janestreet flips to ready with its vol-gold
        accent — every registry desk is now ready and no planned desk remains
        (the planned-badge plumbing stays for future desks)."""
        desks = client.get('/api/floor/desks').get_json()['desks']
        by_key = {d['key']: d for d in desks}

        assert by_key['foundation']['status'] == 'ready'
        assert by_key['renaissance']['status'] == 'ready'
        assert by_key['renaissance']['accent'] == '#58a6ff'
        assert by_key['citadel']['status'] == 'ready'
        assert by_key['citadel']['accent'] == '#bc8cff'
        if by_key['janestreet']['status'] != 'ready':
            pytest.skip('janestreet not ready yet '
                        '(parallel Phase 8 backend task, contract C15)')
        assert by_key['janestreet']['accent'] == '#d29922'
        assert [d['key'] for d in desks if d['status'] == 'planned'] == []

    def test_unknown_desk_key_returns_400_with_message(self, client):
        response = client.post('/api/backtest/run', json={
            'symbols': 'AAA', 'desk': 'no-such-desk',
            'start_date': '2023-01-01', 'end_date': '2023-12-31'})

        assert response.status_code == 400
        assert 'Unknown desk' in response.get_json()['error']
        assert 'no-such-desk' in response.get_json()['error']

    def test_planned_desk_returns_400_mentioning_its_phase(
            self, client, monkeypatch):
        """No REAL planned desk remains after the Phase 8 flip, so a fake
        planned desk is injected into the registry to keep pinning the
        create_desk ValueError -> 400 path end to end (monkeypatch.setitem
        restores the spec dict afterwards)."""
        import desks.registry as desks_registry

        monkeypatch.setitem(desks_registry._DESK_SPECS, 'futurefund', {
            'name': 'Future Fund Desk',
            'firm_inspiration': 'TBD',
            'description': 'Synthetic planned desk pinning the 400 path.',
            'status': 'planned',
            'activates_in_phase': 99,
            'accent': '#8b949e',
            'factory': None,
        })

        response = client.post('/api/backtest/run', json={
            'symbols': 'AAA', 'desk': 'futurefund',
            'start_date': '2023-01-01', 'end_date': '2023-12-31'})

        assert response.status_code == 400
        assert 'Phase 99' in response.get_json()['error']


def _fake_desk_report():
    """Desk-mode engine report: the strategy report + contract C3 additions."""
    report = _fake_engine_report()
    report['desk'] = {'key': 'foundation', 'name': 'Foundation'}
    report['trader_notes'] = [
        {'timestamp': '2023-03-01T00:00:00', 'desk': 'foundation',
         'category': 'signal', 'message': 'BUY AAA on momentum confirmation',
         'data': {'score': 0.82}},
        {'timestamp': '2023-06-01T00:00:00', 'desk': 'foundation',
         'category': 'risk', 'message': 'Position capped by risk limits',
         'data': {}},
    ]
    report['walk_forward'] = [
        {'fit_date': '2023-06-01', 'train_start': '2023-01-03',
         'train_end': '2023-05-31', 'n_samples': 103},
    ]
    return report


def _fake_renaissance_report():
    """Desk-mode report with the Phase 6 renaissance additions:
    regime_series (contract C5), book-tagged notes (contract C7), and
    model-tagged walk-forward fits."""
    report = _fake_engine_report()
    report['desk'] = {'key': 'renaissance', 'name': 'Renaissance'}
    report['trader_notes'] = [
        {'timestamp': '2023-03-01T00:00:00', 'desk': 'renaissance',
         'category': 'model', 'message': 'Regime flip: trending -> high_vol',
         'data': {'book': 'regime', 'confidence': 0.91}},
        {'timestamp': '2023-03-02T00:00:00', 'desk': 'renaissance',
         'category': 'signal', 'message': 'Reversion entry AAA at z=-2.1',
         'data': {'book': 'mean_reversion', 'zscore': -2.1}},
        {'timestamp': '2023-03-03T00:00:00', 'desk': 'renaissance',
         'category': 'signal', 'message': 'Pairs divergence AAA/BBB',
         'data': {'book': 'pairs', 'half_life': 12.5}},
        {'timestamp': '2023-03-06T00:00:00', 'desk': 'renaissance',
         'category': 'risk', 'message': 'Stat-arb gross exposure capped',
         'data': {'book': 'stat_arb'}},
        # A note without data.book must pass through unchanged too.
        {'timestamp': '2023-03-07T00:00:00', 'desk': 'renaissance',
         'category': 'info', 'message': 'Desk heartbeat', 'data': {}},
    ]
    report['walk_forward'] = [
        {'fit_date': '2023-06-01', 'train_start': '2023-01-03',
         'train_end': '2023-05-31', 'n_samples': 103, 'model': 'regime'},
        {'fit_date': '2023-06-01', 'train_start': '2023-01-03',
         'train_end': '2023-05-31', 'n_samples': 103, 'model': 'stat_arb'},
        {'fit_date': '2023-09-01', 'train_start': '2023-03-01',
         'train_end': '2023-08-31', 'n_samples': 127, 'model': 'pairs'},
    ]
    report['regime_series'] = [
        {'date': '2023-06-01', 'state': 'trending',
         'probs': {'mean_reverting': 0.1, 'trending': 0.8, 'high_vol': 0.1}},
        {'date': '2023-06-02', 'state': 'trending',
         'probs': {'mean_reverting': 0.15, 'trending': 0.7, 'high_vol': 0.15}},
        {'date': '2023-06-05', 'state': 'high_vol',
         'probs': {'mean_reverting': 0.05, 'trending': 0.25, 'high_vol': 0.7}},
    ]
    return report


class TestDeskModeBacktest:
    """POST /api/backtest/run with 'desk' instead of 'strategy' (contract C2/C3)."""

    PAYLOAD = {
        'symbols': 'AAA',
        'desk': 'foundation',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'position_size': 0.1,
    }

    @pytest.fixture()
    def patch_desk_stack(self, monkeypatch, tmp_path):
        """Stub create_desk + the whole engine; redirect the history DB."""
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        seen = {}
        desk_sentinel = object()

        def fake_create_desk(key, capital_allocation=1.0, model_key=None):
            seen['desk_key'] = key
            return desk_sentinel

        class FakeEngine:
            """Contract C2 stand-in: desk-mode construction, run() unchanged."""

            def __init__(self, strategy=None, desk=None,
                         initial_capital=100000, **kwargs):
                seen['ctor'] = {'strategy': strategy, 'desk': desk,
                                'initial_capital': initial_capital,
                                'kwargs': kwargs}
                # _fetch_info getattr-guards a missing provenance method.
                self.market_data = None

            def run(self, symbols, start_date, end_date, position_size,
                    progress_callback=None, benchmark_symbol='SPY'):
                seen['run_symbols'] = list(symbols)
                if progress_callback is not None:
                    progress_callback(50.0)
                return _fake_desk_report()

        monkeypatch.setattr(api_backtest, 'create_desk', fake_create_desk)
        monkeypatch.setattr(api_backtest, 'BacktestEngine', FakeEngine)
        return seen

    def test_desk_run_returns_report_with_desk_additions(
            self, client, patch_desk_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)

        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        # Contract C2: the engine was built in desk mode, not strategy mode.
        assert patch_desk_stack['desk_key'] == 'foundation'
        assert patch_desk_stack['ctor']['strategy'] is None
        assert patch_desk_stack['ctor']['desk'] is not None
        assert patch_desk_stack['run_symbols'] == ['AAA']

        result = job['result']
        # Contract C3 additive keys survive JSON-safing untouched.
        assert result['desk'] == {'key': 'foundation', 'name': 'Foundation'}
        assert [n['category'] for n in result['trader_notes']] == \
            ['signal', 'risk']
        assert result['trader_notes'][0]['data'] == {'score': 0.82}
        assert result['walk_forward'] == [
            {'fit_date': '2023-06-01', 'train_start': '2023-01-03',
             'train_end': '2023-05-31', 'n_samples': 103}]
        # ... while every existing report key keeps its legacy shape.
        assert result['summary']['total_return_pct'] == pytest.approx(8.0)
        assert result['trades'][0]['date'] == '2023-03-01'
        assert result['portfolio_history'][0]['timestamp'] == '2023-01-03'
        # Contract C5: a desk without a regime model must NOT gain
        # regime_series content — [] or absent are both acceptable.
        assert result.get('regime_series') in (None, [])
        # Contract C8: same for pod_history on a non-citadel desk.
        assert result.get('pod_history') in (None, [])
        # Contracts C11/C12: same for the janestreet-only structures and
        # greeks_series keys on a non-janestreet desk.
        assert result.get('structures') in (None, [])
        assert result.get('greeks_series') in (None, [])

    def test_realistic_fills_threads_to_engine(self, client, patch_desk_stack):
        # Step 6: default off; explicit true reaches the engine constructor.
        r1 = client.post('/api/backtest/run', json=self.PAYLOAD)
        _wait_for_job(client, r1.get_json()['job_id'])
        assert patch_desk_stack['ctor']['kwargs'].get(
            'enable_realistic_fills') is False

        r2 = client.post('/api/backtest/run',
                         json={**self.PAYLOAD, 'realistic_fills': True})
        _wait_for_job(client, r2.get_json()['job_id'])
        assert patch_desk_stack['ctor']['kwargs'].get(
            'enable_realistic_fills') is True

    def test_non_boolean_realistic_fills_rejected(self, client,
                                                  patch_desk_stack):
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'realistic_fills': 'yes'})
        assert response.status_code == 400
        assert 'realistic_fills' in response.get_json()['error']

    def test_desk_run_saved_with_desk_prefixed_strategy(
            self, client, patch_desk_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        rows = client.get('/api/backtests').get_json()['backtests']
        assert len(rows) == 1
        assert rows[0]['strategy'] == 'desk:foundation'
        assert rows[0]['name'] == 'desk:foundation AAA'
        assert rows[0]['total_return'] == pytest.approx(0.08)

        # The export/compare surfaces still render a desk-saved run.
        report = client.get(f"/api/export/report/{rows[0]['id']}")
        assert report.status_code == 200
        text = report.get_data(as_text=True)
        assert 'desk:foundation' in text
        assert 'Total Return: 8.00%' in text

    def test_desk_and_strategy_together_rejected(self, client):
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'strategy': 'momentum'})

        assert response.status_code == 400
        assert 'not both' in response.get_json()['error']

    @pytest.mark.parametrize('bad_desk', ['   ', '', 123, ['foundation']])
    def test_non_string_or_blank_desk_rejected(self, client, bad_desk):
        response = client.post(
            '/api/backtest/run', json={**self.PAYLOAD, 'desk': bad_desk})

        assert response.status_code == 400
        assert 'desk' in response.get_json()['error']

    def test_unknown_desk_returns_400_with_registry_message(
            self, client, monkeypatch):
        from gui.routes import api_backtest

        def raising_create_desk(key, capital_allocation=1.0, model_key=None):
            raise ValueError(f'Unknown desk: {key}')

        monkeypatch.setattr(api_backtest, 'create_desk', raising_create_desk)

        response = client.post(
            '/api/backtest/run', json={**self.PAYLOAD, 'desk': 'no-such-desk'})

        assert response.status_code == 400
        assert response.get_json()['error'] == 'Unknown desk: no-such-desk'

    def test_planned_desk_returns_400_mentioning_phase(
            self, client, monkeypatch):
        from gui.routes import api_backtest

        def raising_create_desk(key, capital_allocation=1.0, model_key=None):
            raise ValueError(f"Desk '{key}' activates in Phase 7")

        monkeypatch.setattr(api_backtest, 'create_desk', raising_create_desk)

        response = client.post(
            '/api/backtest/run', json={**self.PAYLOAD, 'desk': 'citadel'})

        assert response.status_code == 400
        assert 'Phase 7' in response.get_json()['error']

    def test_desk_run_returns_503_when_framework_unavailable(
            self, client, monkeypatch):
        from gui.routes import api_backtest

        sentinel = 'ImportError: synthetic desks import failure'
        monkeypatch.setattr(api_backtest, 'create_desk', None)
        monkeypatch.setattr(
            api_backtest, 'DESK_REGISTRY_IMPORT_ERROR', sentinel)

        response = client.post('/api/backtest/run', json=self.PAYLOAD)

        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Desk framework unavailable',
            'reason': sentinel,
        }

    def test_legacy_strategy_run_unaffected_by_desk_support(
            self, client, monkeypatch, tmp_path):
        """A strategy-only payload takes the legacy path bit-identically:
        same engine call, no desk keys in the report, plain strategy name
        in the saved-history row."""
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        def fake_run(self, symbols, start_date, end_date, position_size,
                     progress_callback=None, benchmark_symbol='SPY'):
            return _fake_engine_report()

        monkeypatch.setattr(api_backtest.BacktestEngine, 'run', fake_run)
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: None, raising=False)

        response = client.post('/api/backtest/run', json={
            'symbols': 'AAA', 'strategy': 'momentum',
            'start_date': '2023-01-01', 'end_date': '2023-12-31',
            'initial_capital': 100000, 'position_size': 0.1})

        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        result = job['result']
        assert 'desk' not in result
        assert 'trader_notes' not in result
        assert 'walk_forward' not in result
        assert 'regime_series' not in result
        assert result['summary']['total_return_pct'] == pytest.approx(8.0)

        rows = client.get('/api/backtests').get_json()['backtests']
        assert rows[0]['strategy'] == 'momentum'

    def test_backtest_page_ships_desk_mode_controls(self, client):
        """The Production backtest page (ws=production) carries the mode toggle
        + desk picker for backtest.js; the Sandbox page is strategy-only."""
        html = client.get('/backtest?ws=production').get_data(as_text=True)
        assert 'id="modeDesk"' in html
        assert 'id="btDesk"' in html
        assert 'id="traderNotesCard"' in html
        assert 'id="deskChipRow"' in html
        # Phase 6: book filter-chip row (contract C7), painted by backtest.js.
        assert 'id="noteBookFilters"' in html
        # Phase 7: pod allocation card + pod filter-chip row (contracts
        # C8/C9), painted by backtest.js only for citadel runs.
        assert 'id="podCard"' in html
        assert 'id="podCards"' in html
        assert 'id="podAllocChart"' in html
        assert 'id="notePodFilters"' in html
        # Sandbox workspace is strategy-only — the Desk radio is not rendered.
        assert 'id="modeDesk"' not in client.get('/backtest').get_data(as_text=True)
        # Phase 8: structures table + portfolio-Greeks card (contracts
        # C11/C12), painted by backtest.js only for janestreet runs.
        assert 'id="structuresCard"' in html
        assert 'id="structuresTable"' in html
        assert 'id="structuresBody"' in html
        assert 'id="greeksCard"' in html
        assert 'id="greeksChart"' in html

    def test_backtest_page_ships_synthetic_pricing_disclaimer(self, client):
        """Phase 8: the estimated-pricing info callout is static template
        markup (backtest.js only toggles its visibility for janestreet
        reports), so the disclaimer wording is pinned at template level."""
        html = client.get('/backtest').get_data(as_text=True)
        assert 'id="syntheticPricingNote"' in html
        # Shipped hidden — visible only once a janestreet report renders.
        assert 'note-line mb-2 d-none' in html
        assert 'Options are priced synthetically (Black-Scholes on' in html
        assert 'backtest approximation' in html
        assert 'E*TRADE quotes in Phase 9' in html


class TestRenaissanceDeskModeBacktest:
    """Phase 6 passthrough: a desk-mode run whose engine report carries
    regime_series (contract C5), book-tagged trader notes (contract C7),
    and model-tagged walk-forward fits must surface them all untouched in
    the async job result. The desk stack is fully stubbed, so this pins the
    ROUTE/serialization behavior independent of the real renaissance desk."""

    PAYLOAD = {
        'symbols': 'AAA',
        'desk': 'renaissance',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'position_size': 0.1,
    }

    @pytest.fixture()
    def patch_renaissance_stack(self, monkeypatch, tmp_path):
        """Stub create_desk + the whole engine; redirect the history DB."""
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        seen = {}

        def fake_create_desk(key, capital_allocation=1.0, model_key=None):
            seen['desk_key'] = key
            return object()

        class FakeEngine:
            def __init__(self, strategy=None, desk=None,
                         initial_capital=100000, **kwargs):
                seen['ctor'] = {'strategy': strategy, 'desk': desk}
                self.market_data = None

            def run(self, symbols, start_date, end_date, position_size,
                    progress_callback=None, benchmark_symbol='SPY'):
                return _fake_renaissance_report()

        monkeypatch.setattr(api_backtest, 'create_desk', fake_create_desk)
        monkeypatch.setattr(api_backtest, 'BacktestEngine', FakeEngine)
        return seen

    def test_regime_series_passes_through_untouched(
            self, client, patch_renaissance_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)

        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        assert patch_renaissance_stack['desk_key'] == 'renaissance'

        result = job['result']
        assert result['desk'] == {'key': 'renaissance', 'name': 'Renaissance'}
        # Contract C5: present, non-empty, byte-for-byte what the engine
        # emitted (dates already ISO strings; probs floats per state).
        assert result['regime_series'] == \
            _fake_renaissance_report()['regime_series']
        for entry in result['regime_series']:
            assert set(entry) == {'date', 'state', 'probs'}
            assert entry['state'] in ('mean_reverting', 'trending', 'high_vol')
            for prob in entry['probs'].values():
                assert isinstance(prob, float)

    def test_book_tagged_notes_and_model_tagged_fits_pass_through(
            self, client, patch_renaissance_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        result = job['result']
        # Contract C7: data.book survives on every book-specific note; the
        # untagged heartbeat note is unaffected.
        books = [n['data'].get('book') for n in result['trader_notes']]
        assert books == ['regime', 'mean_reversion', 'pairs', 'stat_arb', None]
        assert result['trader_notes'][1]['data']['zscore'] == \
            pytest.approx(-2.1)
        # Phase 6 fits carry 'model' for the color-coded refit markers.
        assert [wf['model'] for wf in result['walk_forward']] == \
            ['regime', 'stat_arb', 'pairs']
        assert result['walk_forward'][2]['fit_date'] == '2023-09-01'

    def test_renaissance_run_saved_with_desk_prefixed_strategy(
            self, client, patch_renaissance_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        rows = client.get('/api/backtests').get_json()['backtests']
        assert len(rows) == 1
        assert rows[0]['strategy'] == 'desk:renaissance'
        assert rows[0]['total_return'] == pytest.approx(0.08)


def _fake_citadel_report():
    """Desk-mode report with the Phase 7 citadel additions: pod_history
    (contract C8) and pod-tagged / reallocation / cut notes (contract C9)."""
    report = _fake_engine_report()
    report['desk'] = {'key': 'citadel', 'name': 'Citadel'}
    report['trader_notes'] = [
        {'timestamp': '2023-03-01T00:00:00', 'desk': 'citadel',
         'category': 'signal', 'message': 'Trend pod long AAA on breakout',
         'data': {'pod': 'trend', 'score': 0.7}},
        {'timestamp': '2023-06-01T00:00:00', 'desk': 'citadel',
         'category': 'allocation',
         'message': 'Vol-targeted performance-weighted reallocation',
         'data': {'allocations': {'trend': 0.45, 'mean_rev': 0.35,
                                  'vol_arb': 0.20},
                  'reason': 'performance-weighted reallocation'}},
        {'timestamp': '2023-09-01T00:00:00', 'desk': 'citadel',
         'category': 'risk', 'message': 'Pod vol_arb placed on probation',
         'data': {'pod': 'vol_arb', 'drawdown_pct': -8.4}},
        {'timestamp': '2023-10-02T00:00:00', 'desk': 'citadel',
         'category': 'risk', 'message': 'Pod vol_arb cut by central risk book',
         'data': {'pod': 'vol_arb', 'drawdown_pct': -12.6}},
        # A note without data.pod must pass through unchanged too.
        {'timestamp': '2023-12-29T00:00:00', 'desk': 'citadel',
         'category': 'info', 'message': 'Desk heartbeat', 'data': {}},
    ]
    report['walk_forward'] = [
        {'fit_date': '2023-06-01', 'train_start': '2023-01-03',
         'train_end': '2023-05-31', 'n_samples': 103},
    ]
    # Contract C8: one entry per simulated day (three representative days
    # here), weights as fractions, drawdown_pct x100 and <= 0.
    report['pod_history'] = [
        {'date': '2023-01-03',
         'pods': {'trend': {'weight': 0.34, 'nav': 34000.0,
                            'drawdown_pct': 0.0, 'status': 'active'},
                  'mean_rev': {'weight': 0.33, 'nav': 33000.0,
                               'drawdown_pct': 0.0, 'status': 'active'},
                  'vol_arb': {'weight': 0.33, 'nav': 33000.0,
                              'drawdown_pct': 0.0, 'status': 'active'}}},
        {'date': '2023-06-01',
         'pods': {'trend': {'weight': 0.45, 'nav': 40100.0,
                            'drawdown_pct': -1.2, 'status': 'active'},
                  'mean_rev': {'weight': 0.35, 'nav': 35200.0,
                               'drawdown_pct': -2.5, 'status': 'active'},
                  'vol_arb': {'weight': 0.20, 'nav': 30900.0,
                              'drawdown_pct': -8.4, 'status': 'probation'}}},
        {'date': '2023-10-02',
         'pods': {'trend': {'weight': 0.55, 'nav': 47800.0,
                            'drawdown_pct': -0.8, 'status': 'active'},
                  'mean_rev': {'weight': 0.45, 'nav': 39100.0,
                               'drawdown_pct': -3.1, 'status': 'active'},
                  'vol_arb': {'weight': 0.0, 'nav': 28700.0,
                              'drawdown_pct': -12.6, 'status': 'cut'}}},
    ]
    return report


class TestCitadelDeskModeBacktest:
    """Phase 7 passthrough: a desk-mode run whose engine report carries
    pod_history (contract C8) and pod-tagged / reallocation / cut notes
    (contract C9) must surface them all untouched in the async job result.
    The desk stack is fully stubbed, so this pins the ROUTE/serialization
    behavior independent of the real citadel desk landing."""

    PAYLOAD = {
        'symbols': 'AAA',
        'desk': 'citadel',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'position_size': 0.1,
    }

    @pytest.fixture()
    def patch_citadel_stack(self, monkeypatch, tmp_path):
        """Stub create_desk + the whole engine; redirect the history DB."""
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        seen = {}

        def fake_create_desk(key, capital_allocation=1.0, model_key=None):
            seen['desk_key'] = key
            return object()

        class FakeEngine:
            def __init__(self, strategy=None, desk=None,
                         initial_capital=100000, **kwargs):
                seen['ctor'] = {'strategy': strategy, 'desk': desk}
                self.market_data = None

            def run(self, symbols, start_date, end_date, position_size,
                    progress_callback=None, benchmark_symbol='SPY'):
                return _fake_citadel_report()

        monkeypatch.setattr(api_backtest, 'create_desk', fake_create_desk)
        monkeypatch.setattr(api_backtest, 'BacktestEngine', FakeEngine)
        return seen

    def test_pod_history_passes_through_untouched(
            self, client, patch_citadel_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)

        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        assert patch_citadel_stack['desk_key'] == 'citadel'
        assert patch_citadel_stack['ctor']['strategy'] is None

        result = job['result']
        assert result['desk'] == {'key': 'citadel', 'name': 'Citadel'}
        # Contract C8: present, non-empty, byte-for-byte what the engine
        # emitted (dates already ISO strings; pod stats JSON-safe floats).
        assert result['pod_history'] == \
            _fake_citadel_report()['pod_history']
        for entry in result['pod_history']:
            assert set(entry) == {'date', 'pods'}
            for pod in entry['pods'].values():
                assert set(pod) == {'weight', 'nav', 'drawdown_pct', 'status'}
                assert pod['drawdown_pct'] <= 0
                assert pod['status'] in ('active', 'probation', 'cut')
        # ... while every existing report key keeps its legacy shape.
        assert result['summary']['total_return_pct'] == pytest.approx(8.0)
        assert result['trades'][0]['date'] == '2023-03-01'

    def test_pod_tagged_notes_and_allocations_pass_through(
            self, client, patch_citadel_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        result = job['result']
        # Contract C9: data.pod survives on every pod-specific note; the
        # untagged reallocation/heartbeat notes are unaffected.
        pods = [n['data'].get('pod') for n in result['trader_notes']]
        assert pods == ['trend', None, 'vol_arb', 'vol_arb', None]
        # Reallocation note: data.allocations {pod: new_weight} + data.reason.
        realloc = result['trader_notes'][1]['data']
        assert realloc['allocations'] == {
            'trend': 0.45, 'mean_rev': 0.35, 'vol_arb': 0.20}
        assert realloc['reason'] == 'performance-weighted reallocation'
        # Cut/probation notes: data.pod + data.drawdown_pct.
        assert result['trader_notes'][2]['data']['drawdown_pct'] == \
            pytest.approx(-8.4)
        assert result['trader_notes'][3]['data']['drawdown_pct'] == \
            pytest.approx(-12.6)

    def test_citadel_run_saved_with_desk_prefixed_strategy(
            self, client, patch_citadel_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        rows = client.get('/api/backtests').get_json()['backtests']
        assert len(rows) == 1
        assert rows[0]['strategy'] == 'desk:citadel'
        assert rows[0]['name'] == 'desk:citadel AAA'
        assert rows[0]['total_return'] == pytest.approx(0.08)


def _fake_janestreet_report():
    """Desk-mode report with the Phase 8 janestreet additions: structures
    (contract C11), greeks_series (contract C12), option-contract
    'instrument' strings on trades/closed_trades (contract C13), and
    book/structure-tagged notes (contract C14). Option prices in the real
    desk are SYNTHETIC (Black-Scholes on historical volatility) — this stub
    only pins the route/serialization passthrough, not the pricing."""
    report = _fake_engine_report()
    report['desk'] = {'key': 'janestreet', 'name': 'Jane Street'}
    # Contract C13: options trades carry the full contract string while
    # 'symbol' stays the underlying.
    report['trades'].append({
        'date': pd.Timestamp('2023-06-02'),
        'signal_date': pd.Timestamp('2023-06-01'),
        'symbol': 'AAA', 'action': 'SELL', 'quantity': 2,
        'price': 1.45, 'proceeds': 290.0,
        'instrument': 'AAA 2023-07-21 95P',
    })
    report['closed_trades'] = [{
        'symbol': 'AAA', 'quantity': 2, 'pnl': 145.0,
        'instrument': 'AAA 2023-07-21 95P',
    }]
    # Contract C14: data.book in {regime, vrp, earnings, relative_value};
    # structure lifecycle notes add data.structure_id + data.structure.
    report['trader_notes'] = [
        {'timestamp': '2023-03-01T00:00:00', 'desk': 'janestreet',
         'category': 'model', 'message': 'Regime gate open: trending',
         'data': {'book': 'regime', 'state': 'trending'}},
        {'timestamp': '2023-06-02T00:00:00', 'desk': 'janestreet',
         'category': 'signal',
         'message': 'Opened iron condor on AAA (IV rank 0.82)',
         'data': {'book': 'vrp', 'structure_id': 'JS-IC-0001',
                  'structure': 'iron_condor', 'iv_rank': 0.82}},
        {'timestamp': '2023-06-30T00:00:00', 'desk': 'janestreet',
         'category': 'risk', 'message': 'Profit target hit on JS-IC-0001',
         'data': {'book': 'vrp', 'structure_id': 'JS-IC-0001',
                  'structure': 'iron_condor',
                  'close_reason': 'profit_target'}},
        {'timestamp': '2023-08-01T00:00:00', 'desk': 'janestreet',
         'category': 'signal', 'message': 'Earnings IV-crush setup on AAA',
         'data': {'book': 'earnings', 'days_to_earnings': 2}},
        {'timestamp': '2023-09-01T00:00:00', 'desk': 'janestreet',
         'category': 'info', 'message': 'ETF/constituent basis within band',
         'data': {'book': 'relative_value', 'basis_bps': 3.1}},
        # A note without data.book must pass through unchanged too.
        {'timestamp': '2023-12-29T00:00:00', 'desk': 'janestreet',
         'category': 'info', 'message': 'Desk heartbeat', 'data': {}},
    ]
    # Contract C11: one closed iron condor, one still-open put credit
    # spread; credit/max_loss are $ at entry, max loss capped at entry by
    # construction; pnl/close_reason null while open.
    report['structures'] = [
        {'id': 'JS-IC-0001', 'type': 'iron_condor', 'underlying': 'AAA',
         'opened': '2023-06-02', 'closed': '2023-06-30',
         'credit': 290.0, 'max_loss': 710.0, 'contracts': 2,
         'status': 'closed', 'pnl': 145.0, 'close_reason': 'profit_target',
         'legs': [
             {'instrument': 'AAA 2023-07-21 95P', 'action': 'SELL',
              'strike': 95.0, 'expiry': '2023-07-21', 'right': 'put'},
             {'instrument': 'AAA 2023-07-21 90P', 'action': 'BUY',
              'strike': 90.0, 'expiry': '2023-07-21', 'right': 'put'},
             {'instrument': 'AAA 2023-07-21 115C', 'action': 'SELL',
              'strike': 115.0, 'expiry': '2023-07-21', 'right': 'call'},
             {'instrument': 'AAA 2023-07-21 120C', 'action': 'BUY',
              'strike': 120.0, 'expiry': '2023-07-21', 'right': 'call'},
         ]},
        {'id': 'JS-PCS-0002', 'type': 'put_credit_spread',
         'underlying': 'AAA', 'opened': '2023-11-01', 'closed': None,
         'credit': 130.0, 'max_loss': 370.0, 'contracts': 1,
         'status': 'open', 'pnl': None, 'close_reason': None,
         'legs': [
             {'instrument': 'AAA 2023-12-15 100P', 'action': 'SELL',
              'strike': 100.0, 'expiry': '2023-12-15', 'right': 'put'},
             {'instrument': 'AAA 2023-12-15 95P', 'action': 'BUY',
              'strike': 95.0, 'expiry': '2023-12-15', 'right': 'put'},
         ]},
    ]
    # Contract C12: daily desk-level dollar Greeks; all 0.0 on option-free
    # days (short-premium book: theta positive, vega negative when on).
    report['greeks_series'] = [
        {'date': '2023-01-03', 'delta': 0.0, 'gamma': 0.0,
         'theta': 0.0, 'vega': 0.0},
        {'date': '2023-06-02', 'delta': -4.2, 'gamma': -0.31,
         'theta': 12.6, 'vega': -88.0},
        {'date': '2023-06-30', 'delta': -1.1, 'gamma': -0.05,
         'theta': 3.2, 'vega': -21.5},
    ]
    return report


class TestJaneStreetDeskModeBacktest:
    """Phase 8 passthrough: a desk-mode run whose engine report carries
    structures (contract C11), greeks_series (contract C12), 'instrument'
    fields (contract C13), and book/structure-tagged notes (contract C14)
    must surface them all untouched in the async job result. The desk stack
    is fully stubbed, so this pins the ROUTE/serialization behavior
    independent of the real janestreet desk landing."""

    PAYLOAD = {
        'symbols': 'AAA',
        'desk': 'janestreet',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'position_size': 0.1,
    }

    @pytest.fixture()
    def patch_janestreet_stack(self, monkeypatch, tmp_path):
        """Stub create_desk + the whole engine; redirect the history DB."""
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        seen = {}

        def fake_create_desk(key, capital_allocation=1.0, model_key=None):
            seen['desk_key'] = key
            return object()

        class FakeEngine:
            def __init__(self, strategy=None, desk=None,
                         initial_capital=100000, **kwargs):
                seen['ctor'] = {'strategy': strategy, 'desk': desk}
                self.market_data = None

            def run(self, symbols, start_date, end_date, position_size,
                    progress_callback=None, benchmark_symbol='SPY'):
                return _fake_janestreet_report()

        monkeypatch.setattr(api_backtest, 'create_desk', fake_create_desk)
        monkeypatch.setattr(api_backtest, 'BacktestEngine', FakeEngine)
        return seen

    def test_structures_pass_through_untouched(
            self, client, patch_janestreet_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)

        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        assert patch_janestreet_stack['desk_key'] == 'janestreet'
        assert patch_janestreet_stack['ctor']['strategy'] is None

        result = job['result']
        assert result['desk'] == {'key': 'janestreet', 'name': 'Jane Street'}
        # Contract C11: present, non-empty, byte-for-byte what the engine
        # emitted (dates already ISO strings; $ amounts JSON-safe floats).
        assert result['structures'] == \
            _fake_janestreet_report()['structures']
        for structure in result['structures']:
            assert set(structure) == {
                'id', 'type', 'underlying', 'opened', 'closed', 'credit',
                'max_loss', 'contracts', 'status', 'pnl', 'close_reason',
                'legs'}
            assert structure['type'] in (
                'iron_condor', 'put_credit_spread', 'call_credit_spread')
            assert structure['status'] in ('open', 'closed', 'expired')
            assert structure['max_loss'] > 0
            for leg in structure['legs']:
                assert set(leg) == {
                    'instrument', 'action', 'strike', 'expiry', 'right'}
                assert leg['right'] in ('call', 'put')
        # Open structures have null pnl/close_reason/closed.
        open_structure = result['structures'][1]
        assert open_structure['status'] == 'open'
        assert open_structure['pnl'] is None
        assert open_structure['close_reason'] is None
        assert open_structure['closed'] is None
        # ... while every existing report key keeps its legacy shape.
        assert result['summary']['total_return_pct'] == pytest.approx(8.0)
        assert result['trades'][0]['date'] == '2023-03-01'

    def test_greeks_series_passes_through_untouched(
            self, client, patch_janestreet_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        result = job['result']
        # Contract C12: present, non-empty, byte-for-byte what the engine
        # emitted; all four Greeks 0.0 on option-free days.
        assert result['greeks_series'] == \
            _fake_janestreet_report()['greeks_series']
        for entry in result['greeks_series']:
            assert set(entry) == {'date', 'delta', 'gamma', 'theta', 'vega'}
        option_free = result['greeks_series'][0]
        assert [option_free[k] for k in
                ('delta', 'gamma', 'theta', 'vega')] == [0.0, 0.0, 0.0, 0.0]

    def test_instrument_fields_pass_through_on_trades_and_closed_trades(
            self, client, patch_janestreet_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        result = job['result']
        # Contract C13: stocks carry the bare symbol, options the full
        # contract string — surviving the trade date normalization.
        assert result['trades'][0]['instrument'] == 'AAA'
        assert result['trades'][1]['instrument'] == 'AAA 2023-07-21 95P'
        assert result['trades'][1]['date'] == '2023-06-02'
        assert result['closed_trades'][0]['instrument'] == \
            'AAA 2023-07-21 95P'

    def test_book_and_structure_tagged_notes_pass_through(
            self, client, patch_janestreet_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        result = job['result']
        # Contract C14: data.book in {regime, vrp, earnings,
        # relative_value}; the untagged heartbeat note is unaffected.
        books = [n['data'].get('book') for n in result['trader_notes']]
        assert books == ['regime', 'vrp', 'vrp', 'earnings',
                         'relative_value', None]
        # Structure lifecycle notes carry structure_id + structure.
        opened = result['trader_notes'][1]['data']
        assert opened['structure_id'] == 'JS-IC-0001'
        assert opened['structure'] == 'iron_condor'
        closed = result['trader_notes'][2]['data']
        assert closed['structure_id'] == 'JS-IC-0001'
        assert closed['close_reason'] == 'profit_target'

    def test_janestreet_run_saved_with_desk_prefixed_strategy(
            self, client, patch_janestreet_stack):
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'

        rows = client.get('/api/backtests').get_json()['backtests']
        assert len(rows) == 1
        assert rows[0]['strategy'] == 'desk:janestreet'
        assert rows[0]['name'] == 'desk:janestreet AAA'
        assert rows[0]['total_return'] == pytest.approx(0.08)


# ==================== PAPER TRADING PENDING ORDERS ====================


class TestPaperPendingOrders:
    """GET /api/trader/<id>/orders + POST .../order/<order_id>/cancel."""

    def _create_trader(self, client, trader_id):
        response = client.post('/api/trader/create', json={
            'trader_id': trader_id, 'mode': 'paper', 'initial_capital': 10000})
        assert response.status_code == 200

    def test_limit_order_listed_then_cancelled(self, client):
        self._create_trader(client, 'orders-t1')

        placed = client.post('/api/trader/orders-t1/order', json={
            'symbol': 'AAPL', 'action': 'BUY', 'quantity': 5, 'price': 12.34})
        assert placed.status_code == 200
        order_id = placed.get_json()['order_id']

        listing = client.get('/api/trader/orders-t1/orders')
        assert listing.status_code == 200
        body = listing.get_json()
        assert body['count'] == 1
        order = body['orders'][0]
        assert order['order_id'] == order_id
        assert order['symbol'] == 'AAPL'
        assert order['side'] == 'BUY'
        assert order['quantity'] == 5
        assert order['limit_price'] == pytest.approx(12.34)
        assert order['timestamp']

        cancel = client.post(f'/api/trader/orders-t1/order/{order_id}/cancel')
        assert cancel.status_code == 200
        assert client.get('/api/trader/orders-t1/orders').get_json()['orders'] == []

        # Cancelling again is a 404, not a silent success.
        again = client.post(f'/api/trader/orders-t1/order/{order_id}/cancel')
        assert again.status_code == 404

    def test_market_order_has_null_limit_price(self, client):
        self._create_trader(client, 'orders-t2')
        client.post('/api/trader/orders-t2/order', json={
            'symbol': 'MSFT', 'action': 'SELL', 'quantity': 3, 'price': 0})

        order = client.get('/api/trader/orders-t2/orders').get_json()['orders'][0]
        assert order['side'] == 'SELL'
        assert order['limit_price'] is None

    def test_orders_for_unknown_trader_returns_404(self, client):
        assert client.get('/api/trader/nope-x/orders').status_code == 404
        assert client.post(
            '/api/trader/nope-x/order/ORD-000001/cancel').status_code == 404

    # Phase 1: robust order-input validation (no silent truncation/bad price).
    def test_order_rejects_fractional_quantity(self, client):
        self._create_trader(client, 'val-frac')
        r = client.post('/api/trader/val-frac/order', json={
            'symbol': 'AAPL', 'action': 'BUY', 'quantity': 1.9, 'price': 100.0})
        assert r.status_code == 400
        assert 'integer' in r.get_json()['error'].lower()

    def test_order_rejects_zero_quantity(self, client):
        self._create_trader(client, 'val-zero')
        r = client.post('/api/trader/val-zero/order', json={
            'symbol': 'AAPL', 'action': 'BUY', 'quantity': 0, 'price': 100.0})
        assert r.status_code == 400

    def test_order_rejects_negative_price(self, client):
        self._create_trader(client, 'val-neg')
        r = client.post('/api/trader/val-neg/order', json={
            'symbol': 'MSFT', 'action': 'SELL', 'quantity': 5, 'price': -50.0})
        assert r.status_code == 400
        assert 'non-negative' in r.get_json()['error'].lower()

    def test_order_rejects_empty_symbol(self, client):
        self._create_trader(client, 'val-sym')
        r = client.post('/api/trader/val-sym/order', json={
            'symbol': '   ', 'action': 'BUY', 'quantity': 5, 'price': 100.0})
        assert r.status_code == 400
        assert 'symbol' in r.get_json()['error'].lower()


# ==================== LIVE TRADING (Phase 9, contract C20) ====================
# The backend C16-C19 surfaces (brokers/etrade_auth, utils/kill_switch,
# utils/audit, brokers/reconcile) land in a parallel task, so every test
# below runs against MOCKED surfaces injected into gui.routes.api_live —
# pinning the ROUTE behavior with zero network and zero real OAuth state.


class FakeAuthManager:
    """Contract-C16-shaped auth manager covering the full state machine."""

    def __init__(self, state='disconnected', env='sandbox'):
        self.state = state
        self.env = env
        self.authorize_url = None
        self.verifier_codes = []
        self.renew_calls = 0
        self.renew_result = True

    def status(self):
        return {
            'state': self.state,
            'env': self.env,
            'authorize_url': (self.authorize_url
                              if self.state == 'pending_verifier' else None),
            'token_issued_at': ('2026-06-12T08:00:00-04:00'
                                if self.state == 'connected' else None),
            'renewed_at': ('2026-06-12T09:30:00-04:00'
                           if self.renew_calls else None),
        }

    def start_auth(self):
        self.state = 'pending_verifier'
        self.authorize_url = 'https://us.etrade.com/e/t/etws/authorize?key=K'
        return self.authorize_url

    def submit_verifier(self, code):
        self.verifier_codes.append(code)
        self.state = 'connected'
        return self.status()

    def renew(self):
        self.renew_calls += 1
        return self.renew_result

    def disconnect(self):
        self.state = 'disconnected'


class FakeKillSwitch:
    """Contract-C17-shaped kill switch recording every flip."""

    def __init__(self, engaged=False):
        self._engaged = engaged
        self.events = []

    def engaged(self):
        return self._engaged

    def engage(self, reason, actor):
        self._engaged = True
        self.events.append(('engage', reason, actor))

    def disengage(self, actor):
        self._engaged = False
        self.events.append(('disengage', actor))


class FakeAuditLog:
    """Contract-C18-shaped audit log over a canned in-memory list."""

    def __init__(self, n=0):
        self.rows = [
            {'seq': i + 1, 'ts': f'2026-06-12T10:{i % 60:02d}:00-04:00',
             'env': 'sandbox',
             'actor': 'gui' if i % 2 else 'system',
             'event_type': 'order_place' if i % 3 else 'kill_switch',
             'payload': {'i': i}}
            for i in range(n)
        ]
        self.verify_result = {'ok': True, 'first_bad_seq': None}

    def entries(self, limit, offset, event_type=None):
        rows = [r for r in self.rows
                if event_type is None or r['event_type'] == event_type]
        return rows[offset:offset + limit]

    def verify_chain(self):
        return self.verify_result


class FakeScheduler:
    """LiveScheduler-shaped fake: status() snapshot + recorded actions."""

    def __init__(self, running=False, paused_reason=None):
        self.running = running
        self.paused_reason = paused_reason
        self.interval_minutes = 15
        self.calls = []

    def status(self):
        return {
            'running': self.running,
            'last_run': ('2026-06-12T10:00:00-04:00'
                         if self.running else None),
            'last_result': ({'status': 'ok', 'reports': []}
                            if self.running else None),
            'next_run_estimate': ('2026-06-12T10:15:00-04:00'
                                  if self.running else None),
            'runs_today': 3 if self.running else 0,
            'consecutive_errors': 0,
            'paused_reason': self.paused_reason,
        }

    def start(self):
        self.calls.append('start')
        self.running = True
        self.paused_reason = None
        return True

    def stop(self):
        self.calls.append('stop')
        self.running = False
        return True


class FakeKeepAlive:
    """TokenKeepAliveScheduler-shaped fake: status() snapshot + recorded
    start/stop actions (no real thread)."""

    def __init__(self, running=False, paused_reason=None):
        self.running = running
        self.paused_reason = paused_reason
        self.interval_minutes = 15
        self.calls = []

    def status(self):
        return {
            'running': self.running,
            'last_renew': ('2026-06-12T10:00:00-04:00'
                           if self.running else None),
            'last_renew_ok': True if self.running else None,
            'last_state': 'connected' if self.running else None,
            'renews_today': 2 if self.running else 0,
            'consecutive_failures': 0,
            'next_run_estimate': ('2026-06-12T10:15:00-04:00'
                                  if self.running else None),
            'paused_reason': self.paused_reason,
            'interval_minutes': self.interval_minutes,
        }

    def start(self):
        self.calls.append('start')
        self.running = True
        self.paused_reason = None
        return True

    def stop(self):
        self.calls.append('stop')
        self.running = False
        return True


@pytest.fixture()
def patch_live(monkeypatch):
    """Inject fake C16/C17/C18 surfaces into api_live; reset module state."""
    import gui.routes.api_live as api_live

    fakes = {
        'auth': FakeAuthManager(),
        'kill': FakeKillSwitch(),
        'audit': FakeAuditLog(n=120),
        'keepalive': FakeKeepAlive(),
    }
    monkeypatch.setattr(api_live, 'get_auth_manager', lambda: fakes['auth'])
    monkeypatch.setattr(api_live, 'get_kill_switch', lambda: fakes['kill'])
    monkeypatch.setattr(api_live, 'get_audit_log', lambda: fakes['audit'])
    monkeypatch.setattr(api_live, '_last_reconciliation', None)
    # Install the keep-alive fake so the connect-time auto-start (and the
    # /keepalive routes) drive it instead of spawning a real daemon thread.
    monkeypatch.setattr(api_live, '_keepalive_scheduler', fakes['keepalive'])
    return fakes


@pytest.fixture()
def patch_scheduler(monkeypatch):
    """Inject a mocked LiveScheduler into api_live."""
    import gui.routes.api_live as api_live

    fake = FakeScheduler()
    monkeypatch.setattr(api_live, 'get_scheduler', lambda: fake)
    return fake


class TestLivePageAndChrome:
    """The /live page + base-template chrome (nav link, dot, banner)."""

    def test_live_page_ships_all_panels_and_script(self, client):
        html = client.get('/live').get_data(as_text=True)
        for marker in (
            'id="envBadge"', 'id="connectionBody"', 'id="authStateBadge"',
            'id="btnKillSwitch"', 'id="ksConfirmModal"', 'id="ksReason"',
            'id="workingOrdersBody"', 'id="workingOrdersTable"',
            'id="auditBody"', 'id="auditEventType"', 'id="auditVerify"',
            'id="auditPrev"', 'id="auditNext"',
            'id="reconBody"', 'id="btnReconcile"',
            'js/live.js',
        ):
            assert marker in html, f'/live missing {marker}'

    def test_nav_has_live_link_with_hidden_dot_on_every_page(self, client):
        for path in TestViewsAndVendoredAssets.PAGES:
            html = client.get(path).get_data(as_text=True)
            assert 'href="/live"' in html, f'{path} missing Live nav link'
            # The dot ships hidden; ONLY live.js (live page poll) paints it.
            assert 'id="navLiveDot"' in html
            assert 'nav-live-dot d-none' in html

    def test_kill_switch_banner_ships_hidden_by_default(self, client):
        """Static banner markup on every page, no polling: hidden until the
        server (context processor) or the live page's poll knows better."""
        for path in TestViewsAndVendoredAssets.PAGES:
            html = client.get(path).get_data(as_text=True)
            assert 'id="killSwitchBanner"' in html
            assert 'kill-banner d-none' in html, \
                f'{path} banner should be hidden while disengaged'

    def test_kill_switch_banner_visible_when_engaged(self, client, monkeypatch):
        """Server-rendered engaged state shows the red banner on EVERY page
        (kill_switch_engaged reads an existing singleton — no polling)."""
        import gui.routes.api_live as api_live

        monkeypatch.setattr(api_live, '_kill_switch', FakeKillSwitch(engaged=True))

        for path in ('/', '/backtest', '/live'):
            html = client.get(path).get_data(as_text=True)
            assert 'id="killSwitchBanner"' in html
            assert 'kill-banner d-none' not in html, \
                f'{path} banner must be visible while engaged'
            assert 'KILL SWITCH ENGAGED' in html

    def test_banner_survives_restart_via_default_db_file(
            self, client, monkeypatch, tmp_path):
        """An engaged switch persisted in the DEFAULT db file (no
        TRADING_DB_PATH) still banners every page after a process restart:
        kill_switch_engaged reads through when trading_data.db already
        exists on disk."""
        import gui.routes.api_live as api_live
        from utils.kill_switch import KillSwitch

        monkeypatch.delenv('TRADING_DB_PATH', raising=False)
        monkeypatch.chdir(tmp_path)
        # 'Previous run': engage and persist into the default-path db file.
        KillSwitch(str(tmp_path / 'trading_data.db')).engage(
            'reconcile mismatch drill', 'test')
        # 'Restart': this process has no singleton yet.
        monkeypatch.setattr(api_live, '_kill_switch', None)

        for path in ('/', '/backtest'):
            html = client.get(path).get_data(as_text=True)
            assert 'kill-banner d-none' not in html, \
                f'{path} banner must be visible after a restart'
            assert 'KILL SWITCH ENGAGED' in html

    def test_banner_check_never_creates_a_db_file(
            self, client, monkeypatch, tmp_path):
        """Without TRADING_DB_PATH and with no existing default db file,
        the per-render banner check must NOT create one (no filesystem
        side effects from arbitrary page renders)."""
        monkeypatch.delenv('TRADING_DB_PATH', raising=False)
        monkeypatch.chdir(tmp_path)

        html = client.get('/').get_data(as_text=True)

        assert 'kill-banner d-none' in html
        assert list(tmp_path.rglob('*.db')) == []

    def test_no_live_polling_on_other_pages(self, client):
        """live.js (the only /api/live poller) loads ONLY on /live."""
        for path in TestViewsAndVendoredAssets.PAGES:
            html = client.get(path).get_data(as_text=True)
            if path == '/live':
                assert 'js/live.js' in html
            else:
                assert 'js/live.js' not in html, f'{path} must not poll live'
                assert '/api/live/' not in html


class TestLiveStatusEndpoint:
    """GET /api/live/status (contract C20)."""

    def test_status_shape_disconnected(self, client, patch_live):
        response = client.get('/api/live/status')

        assert response.status_code == 200
        body = response.get_json()
        assert set(body) == {'auth', 'env', 'kill_switch', 'reconciliation'}
        assert body['auth']['state'] == 'disconnected'
        assert body['env'] == 'sandbox'
        assert body['kill_switch'] == {'engaged': False}
        assert body['reconciliation'] is None

    def test_status_production_env_and_engaged_switch(self, client, patch_live):
        patch_live['auth'].env = 'production'
        patch_live['kill']._engaged = True

        body = client.get('/api/live/status').get_json()
        assert body['env'] == 'production'
        assert body['kill_switch'] == {'engaged': True}

    def test_status_503_when_auth_surface_unavailable(self, client, monkeypatch):
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic etrade_auth import failure'
        monkeypatch.setattr(api_live, 'get_auth_manager', lambda: None)
        monkeypatch.setattr(api_live, 'ETRADE_AUTH_IMPORT_ERROR', sentinel)

        response = client.get('/api/live/status')
        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Live trading unavailable', 'reason': sentinel}

    def test_status_503_when_kill_switch_unavailable(self, client, monkeypatch):
        """An ambiguous kill-switch state must never read as fine."""
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic kill_switch import failure'
        monkeypatch.setattr(api_live, 'get_auth_manager',
                            lambda: FakeAuthManager())
        monkeypatch.setattr(api_live, 'get_kill_switch', lambda: None)
        monkeypatch.setattr(api_live, 'KILL_SWITCH_IMPORT_ERROR', sentinel)

        response = client.get('/api/live/status')
        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Kill switch unavailable', 'reason': sentinel}


class TestLiveAuthFlow:
    """POST /api/live/auth/* against the mocked C16 state machine."""

    def test_start_returns_authorize_url_and_pending_state(
            self, client, patch_live):
        response = client.post('/api/live/auth/start')

        assert response.status_code == 200
        body = response.get_json()
        assert body['authorize_url'].startswith('https://us.etrade.com/')
        assert body['auth']['state'] == 'pending_verifier'
        assert body['auth']['authorize_url'] == body['authorize_url']

    def test_verifier_completes_the_flow(self, client, patch_live):
        client.post('/api/live/auth/start')
        response = client.post('/api/live/auth/verifier',
                               json={'code': ' A1B2C '})

        assert response.status_code == 200
        body = response.get_json()
        assert body['auth']['state'] == 'connected'
        assert body['auth']['token_issued_at'] is not None
        # The code is trimmed before it reaches the manager.
        assert patch_live['auth'].verifier_codes == ['A1B2C']

    @pytest.mark.parametrize('payload', [None, {}, {'code': ''},
                                         {'code': '   '}])
    def test_verifier_requires_a_code(self, client, patch_live, payload):
        response = client.post('/api/live/auth/verifier', json=payload)
        assert response.status_code == 400
        assert 'code' in response.get_json()['error'].lower()
        assert patch_live['auth'].verifier_codes == []

    def test_renew_round_trip(self, client, patch_live):
        patch_live['auth'].state = 'connected'
        response = client.post('/api/live/auth/renew')

        assert response.status_code == 200
        body = response.get_json()
        assert body['renewed'] is True
        assert body['auth']['renewed_at'] is not None
        assert patch_live['auth'].renew_calls == 1

    def test_failed_renew_reports_false(self, client, patch_live):
        patch_live['auth'].renew_result = False
        assert client.post('/api/live/auth/renew').get_json()['renewed'] is False

    def test_disconnect_returns_disconnected_status(self, client, patch_live):
        patch_live['auth'].state = 'connected'
        response = client.post('/api/live/auth/disconnect')

        assert response.status_code == 200
        assert response.get_json()['auth']['state'] == 'disconnected'

    def test_start_returns_503_when_not_configured(self, client, patch_live):
        """EtradeNotConfigured (C16) surfaces as 503-with-reason."""
        import gui.routes.api_live as api_live

        def raising_start():
            raise api_live.EtradeNotConfigured('consumer keys absent')

        patch_live['auth'].start_auth = raising_start
        response = client.post('/api/live/auth/start')

        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'E*TRADE not configured', 'reason': 'consumer keys absent'}

    def test_start_surfaces_etrade_auth_error_reason(self, client, patch_live):
        """An EtradeAuthError (e.g. E*TRADE rejecting production keys at
        request_token) surfaces as 502-WITH-REASON, not a blind 500."""
        import gui.routes.api_live as api_live

        def raising_start():
            raise api_live.EtradeAuthError(
                'request_token failed with HTTP 401: '
                'oauth_problem=signature_invalid')

        patch_live['auth'].start_auth = raising_start
        response = client.post('/api/live/auth/start')

        assert response.status_code == 502
        body = response.get_json()
        assert body['error'] == 'Failed to start authorization'
        assert 'oauth_problem' in body['reason']

    def test_verifier_surfaces_etrade_auth_error_reason(
            self, client, patch_live):
        import gui.routes.api_live as api_live

        def raising_verifier(_code):
            raise api_live.EtradeAuthError('access_token failed with HTTP 401')

        patch_live['auth'].submit_verifier = raising_verifier
        response = client.post('/api/live/auth/verifier', json={'code': 'X'})

        assert response.status_code == 502
        body = response.get_json()
        assert body['error'] == 'Failed to submit verifier code'
        assert 'access_token failed' in body['reason']

    def test_renew_surfaces_etrade_auth_error_reason(self, client, patch_live):
        import gui.routes.api_live as api_live

        def raising_renew():
            raise api_live.EtradeAuthError('renew failed with HTTP 401')

        patch_live['auth'].renew = raising_renew
        response = client.post('/api/live/auth/renew')

        assert response.status_code == 502
        body = response.get_json()
        assert body['error'] == 'Failed to renew token'
        assert 'renew failed' in body['reason']

    @pytest.mark.parametrize('path,method', [
        ('/api/live/auth/start', 'post'),
        ('/api/live/auth/verifier', 'post'),
        ('/api/live/auth/renew', 'post'),
        ('/api/live/auth/disconnect', 'post'),
    ])
    def test_auth_routes_503_when_surface_unavailable(
            self, client, monkeypatch, path, method):
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic etrade_auth import failure'
        monkeypatch.setattr(api_live, 'get_auth_manager', lambda: None)
        monkeypatch.setattr(api_live, 'ETRADE_AUTH_IMPORT_ERROR', sentinel)

        response = getattr(client, method)(path, json={'code': 'X'})
        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Live trading unavailable', 'reason': sentinel}


class TestLiveKillSwitchEndpoint:
    """POST /api/live/killswitch round trip (contract C17/C20)."""

    def test_engage_then_disengage_round_trip(self, client, patch_live):
        engage = client.post('/api/live/killswitch', json={
            'engaged': True, 'reason': 'fat-finger drill'})
        assert engage.status_code == 200
        assert engage.get_json() == {'engaged': True}
        assert patch_live['kill'].events == [
            ('engage', 'fat-finger drill', 'gui')]
        assert client.get(
            '/api/live/status').get_json()['kill_switch']['engaged'] is True

        disengage = client.post('/api/live/killswitch', json={'engaged': False})
        assert disengage.status_code == 200
        assert disengage.get_json() == {'engaged': False}
        assert patch_live['kill'].events[-1] == ('disengage', 'gui')

    @pytest.mark.parametrize('payload', [
        {'engaged': True}, {'engaged': True, 'reason': '  '}])
    def test_engage_requires_a_reason(self, client, patch_live, payload):
        response = client.post('/api/live/killswitch', json=payload)
        assert response.status_code == 400
        assert 'reason' in response.get_json()['error'].lower()
        assert patch_live['kill'].events == []

    @pytest.mark.parametrize('engaged', [None, 'true', 1, 'yes'])
    def test_non_boolean_engaged_rejected(self, client, patch_live, engaged):
        response = client.post('/api/live/killswitch',
                               json={'engaged': engaged, 'reason': 'x'})
        assert response.status_code == 400
        assert 'boolean' in response.get_json()['error']

    def test_503_when_kill_switch_unavailable(self, client, monkeypatch):
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic kill_switch import failure'
        monkeypatch.setattr(api_live, 'get_kill_switch', lambda: None)
        monkeypatch.setattr(api_live, 'KILL_SWITCH_IMPORT_ERROR', sentinel)

        response = client.post('/api/live/killswitch',
                               json={'engaged': True, 'reason': 'x'})
        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Kill switch unavailable', 'reason': sentinel}


class TestLiveAuditEndpoint:
    """GET /api/live/audit paging / filtering / verify (contract C18/C20)."""

    def test_default_page_is_first_fifty(self, client, patch_live):
        body = client.get('/api/live/audit').get_json()

        assert body['limit'] == 50
        assert body['offset'] == 0
        assert body['event_type'] is None
        assert len(body['entries']) == 50
        assert body['entries'][0]['seq'] == 1
        assert 'verify' not in body
        assert set(body['entries'][0]) == {
            'seq', 'ts', 'env', 'actor', 'event_type', 'payload'}

    def test_paging_with_limit_and_offset(self, client, patch_live):
        body = client.get('/api/live/audit?limit=10&offset=100').get_json()

        assert [e['seq'] for e in body['entries']] == list(range(101, 111))
        assert body['limit'] == 10
        assert body['offset'] == 100

    def test_event_type_filter_passes_through(self, client, patch_live):
        body = client.get('/api/live/audit?event_type=kill_switch').get_json()

        assert body['event_type'] == 'kill_switch'
        assert body['entries']
        assert all(e['event_type'] == 'kill_switch' for e in body['entries'])

    def test_verify_param_runs_chain_verification(self, client, patch_live):
        body = client.get('/api/live/audit?verify=1').get_json()
        assert body['verify'] == {'ok': True, 'first_bad_seq': None}

        patch_live['audit'].verify_result = {'ok': False, 'first_bad_seq': 7}
        body = client.get('/api/live/audit?verify=true').get_json()
        assert body['verify'] == {'ok': False, 'first_bad_seq': 7}

    @pytest.mark.parametrize('query', [
        'limit=0', 'limit=501', 'limit=abc', 'offset=-1', 'offset=x'])
    def test_bad_paging_params_are_400(self, client, patch_live, query):
        response = client.get(f'/api/live/audit?{query}')
        assert response.status_code == 400
        assert 'error' in response.get_json()

    def test_503_when_audit_unavailable(self, client, monkeypatch):
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic audit import failure'
        monkeypatch.setattr(api_live, 'get_audit_log', lambda: None)
        monkeypatch.setattr(api_live, 'AUDIT_IMPORT_ERROR', sentinel)

        response = client.get('/api/live/audit')
        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Audit log unavailable', 'reason': sentinel}


class TestLiveParityEndpoint:
    """GET /api/live/parity (Phase 3 Step 5) — read-only execution parity."""

    class _ParityAudit:
        def __init__(self, rows):
            self._rows = rows

        def entries(self, limit=100, offset=0, event_type=None,
                    ascending=False):
            return [r for r in self._rows
                    if event_type is None or r['event_type'] == event_type]

    def _audit(self):
        return self._ParityAudit([
            {'seq': 1, 'ts': 't', 'env': 'sandbox', 'actor': 'live_session',
             'event_type': 'execution_report',
             'payload': {'symbol': 'AAA', 'action': 'BUY', 'status': 'filled',
                         'avg_fill': 100.10, 'shortfall_per_unit': 0.10,
                         'arrival_mid': 100.0, 'asset_type': 'stock',
                         'quantity': 100, 'commission': None}},
        ])

    def test_parity_report_returned(self, client, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'get_audit_log', lambda: self._audit())
        body = client.get('/api/live/parity').get_json()
        assert body['n_fills'] == 1
        fill = body['fills'][0]
        assert fill['slippage_drift'] == pytest.approx(0.05)
        assert fill['slippage_drift_total'] == pytest.approx(5.0)
        assert body['commission_available'] is False

    def test_bad_limit_is_400(self, client, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'get_audit_log', lambda: self._audit())
        response = client.get('/api/live/parity?limit=0')
        assert response.status_code == 400

    def test_503_when_parity_unavailable(self, client, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'parity_report', None)
        monkeypatch.setattr(api_live, 'PARITY_IMPORT_ERROR', 'ImportError: x')
        response = client.get('/api/live/parity')
        assert response.status_code == 503
        assert response.get_json()['reason'] == 'ImportError: x'

    def test_503_when_audit_unavailable(self, client, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'get_audit_log', lambda: None)
        monkeypatch.setattr(api_live, 'AUDIT_IMPORT_ERROR', 'ImportError: y')
        response = client.get('/api/live/parity')
        assert response.status_code == 503


class TestLiveReconcileEndpoint:
    """POST /api/live/reconcile (contract C19/C20)."""

    @staticmethod
    def _ok_result():
        return {'ok': True, 'mismatches': [],
                'checked_at': '2026-06-12T10:00:00-04:00'}

    @staticmethod
    def _mismatch_result():
        return {'ok': False,
                'mismatches': [
                    {'kind': 'position', 'symbol': 'AAPL',
                     'local': 10.0, 'broker': 12.0},
                    {'kind': 'cash', 'symbol': None,
                     'local': 5000.0, 'broker': 4910.5},
                ],
                'checked_at': '2026-06-12T10:00:00-04:00'}

    @pytest.fixture()
    def patch_reconcile(self, monkeypatch, patch_live):
        """Wire a fake reconcile fn + a fake live broker into api_live."""
        import gui.routes.api_live as api_live

        class FakeBroker:
            local_positions = {'AAPL': 10.0}
            local_cash = 5000.0

        seen = {'result': self._ok_result(), 'calls': []}

        def fake_reconcile(local_positions, local_cash, broker):
            seen['calls'].append((local_positions, local_cash, broker))
            return seen['result']

        broker = FakeBroker()
        monkeypatch.setattr(api_live, 'reconcile', fake_reconcile)
        monkeypatch.setattr(api_live, '_find_live_broker', lambda: broker)
        seen['broker'] = broker
        return {**patch_live, **seen, 'seen': seen}

    def test_ok_result_passes_through_and_is_remembered(
            self, client, patch_reconcile):
        response = client.post('/api/live/reconcile')

        assert response.status_code == 200
        body = response.get_json()
        assert body['ok'] is True
        assert body['mismatches'] == []
        assert body['kill_switch_engaged'] is False
        # The broker's local book reached the C19 call.
        local_positions, local_cash, broker = patch_reconcile['calls'][0]
        assert local_positions == {'AAPL': 10.0}
        assert local_cash == pytest.approx(5000.0)
        assert broker is patch_reconcile['broker']
        # Kill switch untouched; status remembers the result.
        assert patch_reconcile['kill'].events == []
        status = client.get('/api/live/status').get_json()
        assert status['reconciliation']['ok'] is True

    def test_mismatch_engages_kill_switch(self, client, patch_reconcile):
        patch_reconcile['seen']['result'] = self._mismatch_result()
        response = client.post('/api/live/reconcile')

        assert response.status_code == 200
        body = response.get_json()
        assert body['ok'] is False
        assert len(body['mismatches']) == 2
        assert body['mismatches'][0] == {
            'kind': 'position', 'symbol': 'AAPL', 'local': 10.0, 'broker': 12.0}
        assert body['kill_switch_engaged'] is True
        # C19: the caller (this route) engaged the switch, audit-attributed.
        kind, reason, actor = patch_reconcile['kill'].events[0]
        assert kind == 'engage'
        assert 'mismatch' in reason.lower()
        assert actor == 'reconcile'
        status = client.get('/api/live/status').get_json()
        assert status['kill_switch']['engaged'] is True
        assert status['reconciliation']['kill_switch_engaged'] is True

    def test_409_when_no_live_session(self, client, patch_live, monkeypatch):
        import gui.routes.api_live as api_live

        monkeypatch.setattr(api_live, 'reconcile',
                            lambda *a, **k: self._ok_result())
        monkeypatch.setattr(api_live, '_find_live_broker', lambda: None)

        response = client.post('/api/live/reconcile')
        assert response.status_code == 409
        assert response.get_json()['error'] == 'No live broker session'

    def test_503_when_reconcile_unavailable(self, client, monkeypatch):
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic reconcile import failure'
        monkeypatch.setattr(api_live, 'reconcile', None)
        monkeypatch.setattr(api_live, 'RECONCILE_IMPORT_ERROR', sentinel)

        response = client.post('/api/live/reconcile')
        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Reconciliation unavailable', 'reason': sentinel}


class TestLiveWorkingOrders:
    """GET /api/live/orders + cancel (patient-executor panel)."""

    def test_empty_without_a_live_session(self, client, monkeypatch):
        import gui.routes.api_live as api_live

        monkeypatch.setattr(api_live, '_find_live_broker', lambda: None)
        response = client.get('/api/live/orders')

        assert response.status_code == 200
        assert response.get_json() == {'orders': [], 'count': 0}

    def test_orders_pass_through_from_patient_executor(
            self, client, monkeypatch):
        import gui.routes.api_live as api_live

        order = {
            'order_id': 'PX-1', 'instrument': 'AAA 2026-07-17 95P',
            'side': 'SELL', 'quantity': 2, 'limit_price': 1.45,
            'steps': [{'limit': 1.5}, {'limit': 1.45}],
            'started_at': '2026-06-12T10:00:00-04:00',
            'remaining_seconds': 90,
        }

        class FakeBroker:
            def working_orders(self):
                return [order]

        monkeypatch.setattr(api_live, '_find_live_broker', lambda: FakeBroker())
        body = client.get('/api/live/orders').get_json()

        assert body['count'] == 1
        assert body['orders'] == [order]

    def test_cancel_round_trip_and_unknown_id(self, client, monkeypatch):
        import gui.routes.api_live as api_live

        cancelled = []

        class FakeBroker:
            def cancel_order(self, order_id):
                cancelled.append(order_id)
                return order_id == 'PX-1'

        monkeypatch.setattr(api_live, '_find_live_broker', lambda: FakeBroker())

        ok = client.post('/api/live/orders/PX-1/cancel')
        assert ok.status_code == 200
        assert ok.get_json() == {'message': 'Order cancelled',
                                 'order_id': 'PX-1'}

        missing = client.post('/api/live/orders/PX-9/cancel')
        assert missing.status_code == 404
        assert cancelled == ['PX-1', 'PX-9']

    def test_cancel_409_without_live_session(self, client, monkeypatch):
        import gui.routes.api_live as api_live

        monkeypatch.setattr(api_live, '_find_live_broker', lambda: None)
        response = client.post('/api/live/orders/PX-1/cancel')
        assert response.status_code == 409
        assert response.get_json()['error'] == 'No live broker session'


class TestLiveSchedulerEndpoint:
    """GET/POST /api/live/scheduler against a mocked LiveScheduler."""

    STATUS_KEYS = {'running', 'last_run', 'last_result',
                   'next_run_estimate', 'runs_today', 'consecutive_errors',
                   'paused_reason', 'interval_minutes'}

    def test_get_status_shape_while_stopped(self, client, patch_scheduler):
        response = client.get('/api/live/scheduler')

        assert response.status_code == 200
        body = response.get_json()
        assert set(body) == self.STATUS_KEYS
        assert body['running'] is False
        assert body['paused_reason'] is None
        assert body['interval_minutes'] == 15

    def test_get_passes_paused_state_through(self, client, patch_scheduler):
        patch_scheduler.paused_reason = 'kill_switch_engaged'

        body = client.get('/api/live/scheduler').get_json()
        assert body['running'] is False
        assert body['paused_reason'] == 'kill_switch_engaged'

    def test_start_calls_start_and_returns_running_status(
            self, client, patch_scheduler):
        response = client.post('/api/live/scheduler',
                               json={'action': 'start'})

        assert response.status_code == 200
        body = response.get_json()
        assert body['running'] is True
        assert set(body) == self.STATUS_KEYS
        assert patch_scheduler.calls == ['start']
        # No interval in the request: the configured one is untouched.
        assert patch_scheduler.interval_minutes == 15

    def test_start_applies_interval_before_starting(
            self, client, patch_scheduler):
        response = client.post('/api/live/scheduler', json={
            'action': 'start', 'interval_minutes': 30})

        assert response.status_code == 200
        assert response.get_json()['interval_minutes'] == 30
        assert patch_scheduler.interval_minutes == 30
        assert patch_scheduler.calls == ['start']

    def test_stop_calls_stop(self, client, patch_scheduler):
        patch_scheduler.running = True
        response = client.post('/api/live/scheduler', json={'action': 'stop'})

        assert response.status_code == 200
        assert response.get_json()['running'] is False
        assert patch_scheduler.calls == ['stop']

    @pytest.mark.parametrize('action', [None, '', 'pause', 'restart', 1])
    def test_unknown_action_rejected(self, client, patch_scheduler, action):
        response = client.post('/api/live/scheduler',
                               json={'action': action})
        assert response.status_code == 400
        assert 'action' in response.get_json()['error']
        assert patch_scheduler.calls == []

    @pytest.mark.parametrize('interval', [0, 241, -15, 2.5, '15', True])
    def test_bad_interval_rejected_without_side_effects(
            self, client, patch_scheduler, interval):
        response = client.post('/api/live/scheduler', json={
            'action': 'start', 'interval_minutes': interval})

        assert response.status_code == 400
        assert 'interval_minutes' in response.get_json()['error']
        assert patch_scheduler.calls == []
        assert patch_scheduler.interval_minutes == 15

    def test_interval_with_stop_rejected(self, client, patch_scheduler):
        response = client.post('/api/live/scheduler', json={
            'action': 'stop', 'interval_minutes': 30})
        assert response.status_code == 400
        assert patch_scheduler.calls == []

    def test_503_while_unconfigured_by_default(self, client):
        """No mocking at all: the GUI never builds a scheduler itself, so
        a fresh process answers 503-with-reason until the live wiring
        installs one via set_scheduler()."""
        for response in (client.get('/api/live/scheduler'),
                         client.post('/api/live/scheduler',
                                     json={'action': 'start'})):
            assert response.status_code == 503
            body = response.get_json()
            assert body['error'] == 'Scheduler unavailable'
            assert 'configured' in body['reason']

    def test_503_reason_is_import_error_when_import_failed(
            self, client, monkeypatch):
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic scheduler import failure'
        monkeypatch.setattr(api_live, 'get_scheduler', lambda: None)
        monkeypatch.setattr(api_live, 'SCHEDULER_IMPORT_ERROR', sentinel)

        response = client.get('/api/live/scheduler')
        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Scheduler unavailable', 'reason': sentinel}

    def test_set_scheduler_installs_the_singleton(self, client):
        """The real accessor pair the live wiring uses (not monkeypatched):
        set_scheduler -> get_scheduler -> routes go live."""
        import gui.routes.api_live as api_live

        fake = FakeScheduler(running=True)
        api_live.set_scheduler(fake)
        try:
            body = client.get('/api/live/scheduler').get_json()
            assert body['running'] is True
        finally:
            api_live.set_scheduler(None)

    def test_live_page_ships_scheduler_card(self, client):
        html = client.get('/live').get_data(as_text=True)
        for marker in (
            'id="schedulerCard"', 'id="schedState"', 'id="schedStateLabel"',
            'id="schedStateDetail"', 'id="schedInterval"',
            'id="schedLastRun"', 'id="schedNextRun"', 'id="schedRunsToday"',
            'id="schedErrors"', 'id="schedIntervalInput"',
            'id="btnSchedStart"', 'id="btnSchedStop"',
        ):
            assert marker in html, f'/live missing {marker}'

    def test_live_page_ships_keepalive_card_and_reconnect_banner(self, client):
        html = client.get('/live').get_data(as_text=True)
        for marker in (
            # Keep-alive card (Piece 2)
            'id="keepaliveCard"', 'id="kaState"', 'id="kaStateLabel"',
            'id="kaStateDetail"', 'id="kaExpiry"', 'id="kaInterval"',
            'id="kaLastRenew"', 'id="kaNextRun"', 'id="kaRenewsToday"',
            'id="kaFailures"', 'id="kaIntervalInput"',
            'id="btnKaStart"', 'id="btnKaStop"',
            # Morning reconnect prompt
            'id="reconnectBanner"', 'id="btnReconnectNow"',
            'id="btnEnableAlerts"',
        ):
            assert marker in html, f'/live missing {marker}'


class TestLiveKeepAliveEndpoint:
    """GET/POST /api/live/keepalive against a mocked keep-alive loop."""

    STATUS_KEYS = {'running', 'last_renew', 'last_renew_ok', 'last_state',
                   'renews_today', 'consecutive_failures',
                   'next_run_estimate', 'paused_reason', 'interval_minutes'}

    def test_get_status_shape(self, client, patch_live):
        response = client.get('/api/live/keepalive')
        assert response.status_code == 200
        body = response.get_json()
        assert set(body) == self.STATUS_KEYS
        assert body['running'] is False
        assert body['interval_minutes'] == 15

    def test_get_passes_paused_state_through(self, client, patch_live):
        patch_live['keepalive'].paused_reason = 'token_expired'
        body = client.get('/api/live/keepalive').get_json()
        assert body['paused_reason'] == 'token_expired'

    def test_start_calls_start_and_returns_running(self, client, patch_live):
        response = client.post('/api/live/keepalive', json={'action': 'start'})
        assert response.status_code == 200
        assert response.get_json()['running'] is True
        assert patch_live['keepalive'].calls == ['start']

    def test_start_applies_interval(self, client, patch_live):
        response = client.post('/api/live/keepalive', json={
            'action': 'start', 'interval_minutes': 30})
        assert response.status_code == 200
        assert response.get_json()['interval_minutes'] == 30
        assert patch_live['keepalive'].interval_minutes == 30

    def test_stop_calls_stop(self, client, patch_live):
        patch_live['keepalive'].running = True
        response = client.post('/api/live/keepalive', json={'action': 'stop'})
        assert response.status_code == 200
        assert response.get_json()['running'] is False
        assert patch_live['keepalive'].calls == ['stop']

    @pytest.mark.parametrize('action', [None, '', 'pause', 'restart', 1])
    def test_unknown_action_rejected(self, client, patch_live, action):
        response = client.post('/api/live/keepalive', json={'action': action})
        assert response.status_code == 400
        assert patch_live['keepalive'].calls == []

    @pytest.mark.parametrize('interval', [0, 91, 120, 241, -15, 2.5, '15', True])
    def test_bad_interval_rejected(self, client, patch_live, interval):
        """Keep-alive caps the interval at 90 min (under the 2h idle), so
        91/120/241 are rejected even though /scheduler allows up to 240."""
        response = client.post('/api/live/keepalive', json={
            'action': 'start', 'interval_minutes': interval})
        assert response.status_code == 400
        assert 'interval_minutes' in response.get_json()['error']
        assert patch_live['keepalive'].calls == []

    def test_interval_with_stop_rejected(self, client, patch_live):
        response = client.post('/api/live/keepalive', json={
            'action': 'stop', 'interval_minutes': 30})
        assert response.status_code == 400
        assert patch_live['keepalive'].calls == []

    def test_available_by_default_self_constructs(self, client):
        """Unlike /scheduler, keep-alive self-constructs over the auth
        manager (no desk/broker needed), so a fresh process answers 200
        (stopped), not 503."""
        response = client.get('/api/live/keepalive')
        assert response.status_code == 200
        assert response.get_json()['running'] is False

    def test_503_when_surface_unavailable(self, client, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'TokenKeepAliveScheduler', None)
        monkeypatch.setattr(api_live, 'KEEPALIVE_IMPORT_ERROR',
                            'ModuleNotFoundError: boom')
        for response in (client.get('/api/live/keepalive'),
                         client.post('/api/live/keepalive',
                                     json={'action': 'start'})):
            assert response.status_code == 503
            assert response.get_json()['error'] == 'Keep-alive unavailable'

    def test_set_keepalive_scheduler_installs_and_clears(self, client):
        """The real setter pair: set_keepalive_scheduler installs a fake the
        routes then read; clearing with None re-enables lazy construction."""
        import gui.routes.api_live as api_live

        fake = FakeKeepAlive(running=True)
        api_live.set_keepalive_scheduler(fake)
        try:
            body = client.get('/api/live/keepalive').get_json()
            assert body['running'] is True
            assert api_live.get_keepalive_scheduler() is fake
        finally:
            api_live.set_keepalive_scheduler(None)


class TestKeepAliveAutoStart:
    """Connecting starts keep-alive; disconnecting stops it (best-effort)."""

    def test_verifier_success_starts_keepalive(self, client, patch_live):
        patch_live['auth'].start_auth()  # -> pending_verifier
        response = client.post('/api/live/auth/verifier', json={'code': 'abc'})
        assert response.status_code == 200
        assert response.get_json()['auth']['state'] == 'connected'
        assert 'start' in patch_live['keepalive'].calls

    def test_disconnect_stops_keepalive(self, client, patch_live):
        response = client.post('/api/live/auth/disconnect')
        assert response.status_code == 200
        assert 'stop' in patch_live['keepalive'].calls

    def test_failed_connect_does_not_start_keepalive(
            self, client, patch_live, monkeypatch):
        def boom(_code):
            raise RuntimeError('nope')
        monkeypatch.setattr(patch_live['auth'], 'submit_verifier', boom)
        response = client.post('/api/live/auth/verifier', json={'code': 'abc'})
        assert response.status_code == 500
        assert patch_live['keepalive'].calls == []

    def test_best_effort_helpers_swallow_exceptions(self, monkeypatch):
        import gui.routes.api_live as api_live

        class Boom:
            def start(self):
                raise RuntimeError('x')

            def stop(self):
                raise RuntimeError('x')

        monkeypatch.setattr(api_live, 'get_keepalive_scheduler',
                            lambda: Boom())
        # Neither helper may propagate — connect/disconnect must never break.
        api_live._start_keepalive_best_effort()
        api_live._stop_keepalive_best_effort()


class FakeAuditEntries:
    """Minimal AuditLog stand-in returning canned keepalive lifecycle seqs."""

    def __init__(self, started_seq=None, stopped_seq=None):
        self._started = started_seq
        self._stopped = stopped_seq

    def entries(self, limit=1, offset=0, event_type=None, ascending=False):
        seq = None
        if event_type == 'keepalive_started':
            seq = self._started
        elif event_type == 'keepalive_stopped':
            seq = self._stopped
        return [{'seq': seq}] if seq is not None else []


class TestKeepAliveRestartRecovery:
    """resume_keepalive_if_desired (Piece 3) — derive the prior intent from
    the audit log and resume only a connected, operator-started loop."""

    def test_desired_running_started_after_stopped(self, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'get_audit_log',
                            lambda: FakeAuditEntries(started_seq=10,
                                                     stopped_seq=5))
        assert api_live._keepalive_desired_running() is True

    def test_not_desired_stopped_after_started(self, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'get_audit_log',
                            lambda: FakeAuditEntries(started_seq=5,
                                                     stopped_seq=10))
        assert api_live._keepalive_desired_running() is False

    def test_not_desired_never_started(self, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'get_audit_log',
                            lambda: FakeAuditEntries())
        assert api_live._keepalive_desired_running() is False

    def test_desired_started_with_no_stop(self, monkeypatch):
        import gui.routes.api_live as api_live
        monkeypatch.setattr(api_live, 'get_audit_log',
                            lambda: FakeAuditEntries(started_seq=7))
        assert api_live._keepalive_desired_running() is True

    def test_resume_starts_when_desired_and_connected(self, monkeypatch):
        import gui.routes.api_live as api_live
        open(api_live._live_db_path(), 'a').close()  # recovery reads an existing db
        fake_ka = FakeKeepAlive()
        api_live.set_keepalive_scheduler(fake_ka)
        try:
            monkeypatch.setattr(api_live, 'get_audit_log',
                                lambda: FakeAuditEntries(started_seq=10))
            monkeypatch.setattr(api_live, 'get_auth_manager',
                                lambda: FakeAuthManager(state='connected'))
            assert api_live.resume_keepalive_if_desired() is True
            assert 'start' in fake_ka.calls
        finally:
            api_live.set_keepalive_scheduler(None)

    def test_resume_noop_when_not_desired(self, monkeypatch):
        import gui.routes.api_live as api_live
        open(api_live._live_db_path(), 'a').close()  # recovery reads an existing db
        fake_ka = FakeKeepAlive()
        api_live.set_keepalive_scheduler(fake_ka)
        try:
            monkeypatch.setattr(api_live, 'get_audit_log',
                                lambda: FakeAuditEntries(started_seq=5,
                                                         stopped_seq=10))
            monkeypatch.setattr(api_live, 'get_auth_manager',
                                lambda: FakeAuthManager(state='connected'))
            assert api_live.resume_keepalive_if_desired() is False
            assert fake_ka.calls == []
        finally:
            api_live.set_keepalive_scheduler(None)

    def test_resume_noop_when_token_not_connected(self, monkeypatch):
        import gui.routes.api_live as api_live
        open(api_live._live_db_path(), 'a').close()  # recovery reads an existing db
        fake_ka = FakeKeepAlive()
        api_live.set_keepalive_scheduler(fake_ka)
        try:
            monkeypatch.setattr(api_live, 'get_audit_log',
                                lambda: FakeAuditEntries(started_seq=10))
            monkeypatch.setattr(api_live, 'get_auth_manager',
                                lambda: FakeAuthManager(state='expired'))
            assert api_live.resume_keepalive_if_desired() is False
            assert fake_ka.calls == []
        finally:
            api_live.set_keepalive_scheduler(None)

    def test_create_app_invokes_restart_recovery(self, monkeypatch):
        import gui.routes.api_live as api_live
        from gui.app import create_app
        called = []
        monkeypatch.setattr(api_live, 'resume_keepalive_if_desired',
                            lambda: called.append(True) or False)
        create_app({'TESTING': True})
        assert called == [True]

    def test_resume_creates_no_db_when_no_persistent_db(
            self, monkeypatch, tmp_path):
        """With no TRADING_DB_PATH and no existing default db file, recovery
        must be a no-op and must NOT create a db file (app construction stays
        side-effect free in a fresh cwd)."""
        import gui.routes.api_live as api_live
        monkeypatch.delenv('TRADING_DB_PATH', raising=False)
        monkeypatch.chdir(tmp_path)
        assert api_live.resume_keepalive_if_desired() is False
        assert list(tmp_path.rglob('*.db')) == []


class TestLiveTraderConstruction:
    """api_trading's live construction site now goes through the C16 auth
    manager (Phase 9) — the env-token constructor is gone."""

    def test_create_live_trader_uses_auth_manager(self, client, monkeypatch):
        """The broker gets the C16 auth manager PLUS the shared kill switch
        and audit log — EtradeClient only gates preview/place on a kill
        switch it was handed, so dropping it here would be a safety hole."""
        import gui.routes.api_trading as api_trading
        import gui.routes.api_live as api_live

        seen = {}

        class FakeBroker:
            def __init__(self, auth=None, account_id_key=None,
                         kill_switch=None, audit=None):
                seen['auth'] = auth
                seen['account_id_key'] = account_id_key
                seen['kill_switch'] = kill_switch
                seen['audit'] = audit

        sentinel_manager = object()
        sentinel_kill = FakeKillSwitch()
        sentinel_audit = FakeAuditLog()
        monkeypatch.setattr(api_trading, 'LiveEtradeBroker', FakeBroker)
        monkeypatch.setattr(api_live, 'get_auth_manager',
                            lambda: sentinel_manager)
        monkeypatch.setattr(api_live, 'get_kill_switch',
                            lambda: sentinel_kill)
        monkeypatch.setattr(api_live, 'get_audit_log',
                            lambda: sentinel_audit)
        monkeypatch.setenv('ETRADE_ACCOUNT_ID_KEY', 'ACCT-KEY-1')

        try:
            response = client.post('/api/trader/create', json={
                'trader_id': 't-live-ctor', 'mode': 'live'})
            assert response.status_code == 200
            assert response.get_json()['mode'] == 'live'
            assert seen['auth'] is sentinel_manager
            assert seen['account_id_key'] == 'ACCT-KEY-1'
            assert seen['kill_switch'] is sentinel_kill
            assert seen['audit'] is sentinel_audit
        finally:
            api_trading.active_traders.pop('t-live-ctor', None)

    def test_create_live_trader_503_when_auth_manager_unavailable(
            self, client, monkeypatch):
        import gui.routes.api_trading as api_trading
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic etrade_auth import failure'
        monkeypatch.setattr(api_trading, 'LiveEtradeBroker', object)
        monkeypatch.setattr(api_live, 'get_auth_manager', lambda: None)
        monkeypatch.setattr(api_live, 'ETRADE_AUTH_IMPORT_ERROR', sentinel)

        response = client.post('/api/trader/create', json={
            'trader_id': 't-live-noauth', 'mode': 'live'})

        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Live trading unavailable', 'reason': sentinel}
        assert 't-live-noauth' not in api_trading.active_traders

    def test_create_live_trader_503_when_kill_switch_unavailable(
            self, client, monkeypatch):
        """Fail CLOSED: a broker built with kill_switch=None would have no
        preview/place gate at all (EtradeClient only checks a switch it was
        handed), so an unavailable kill switch must refuse construction."""
        import gui.routes.api_trading as api_trading
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic kill_switch import failure'
        monkeypatch.setattr(api_trading, 'LiveEtradeBroker', object)
        monkeypatch.setattr(api_live, 'get_auth_manager',
                            lambda: FakeAuthManager())
        monkeypatch.setattr(api_live, 'get_kill_switch', lambda: None)
        monkeypatch.setattr(api_live, 'KILL_SWITCH_IMPORT_ERROR', sentinel)

        response = client.post('/api/trader/create', json={
            'trader_id': 't-live-noks', 'mode': 'live'})

        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Live trading unavailable', 'reason': sentinel}
        assert 't-live-noks' not in api_trading.active_traders

    def test_create_live_trader_503_when_kill_switch_construction_fails(
            self, client, monkeypatch):
        """Construction failure (import fine, KILL_SWITCH_IMPORT_ERROR is
        None) still 503s with a non-null reason."""
        import gui.routes.api_trading as api_trading
        import gui.routes.api_live as api_live

        monkeypatch.setattr(api_trading, 'LiveEtradeBroker', object)
        monkeypatch.setattr(api_live, 'get_auth_manager',
                            lambda: FakeAuthManager())
        monkeypatch.setattr(api_live, 'get_kill_switch', lambda: None)
        monkeypatch.setattr(api_live, 'KILL_SWITCH_IMPORT_ERROR', None)

        response = client.post('/api/trader/create', json={
            'trader_id': 't-live-ksctor', 'mode': 'live'})

        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Live trading unavailable',
            'reason': 'KillSwitch construction failed'}
        assert 't-live-ksctor' not in api_trading.active_traders

    def test_create_live_trader_503_when_audit_log_unavailable(
            self, client, monkeypatch):
        """Fail CLOSED on the audit log too: live orders must never trade
        unrecorded."""
        import gui.routes.api_trading as api_trading
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic audit import failure'
        monkeypatch.setattr(api_trading, 'LiveEtradeBroker', object)
        monkeypatch.setattr(api_live, 'get_auth_manager',
                            lambda: FakeAuthManager())
        monkeypatch.setattr(api_live, 'get_kill_switch',
                            lambda: FakeKillSwitch())
        monkeypatch.setattr(api_live, 'get_audit_log', lambda: None)
        monkeypatch.setattr(api_live, 'AUDIT_IMPORT_ERROR', sentinel)

        response = client.post('/api/trader/create', json={
            'trader_id': 't-live-noaudit', 'mode': 'live'})

        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Live trading unavailable', 'reason': sentinel}
        assert 't-live-noaudit' not in api_trading.active_traders


# ==================== LIVE TRADING: ACCOUNTS / QUOTES / ORDER TICKET ===========
# R1-R6 routes over a MOCKED EtradeClient injected into api_live.get_client.
# NO network, NO real OAuth. The fake records its calls so place_order's
# single-call / same-cached-request guarantees can be asserted.

#: A raw account number that must NEVER appear whole in any response.
RAW_ACCOUNT_NUMBER = '83056214'


class FakeTradeClient:
    """EtradeClient-shaped fake for the accounts/quotes/order-ticket routes."""

    def __init__(self):
        self.preview_calls = []
        self.place_calls = []
        self.cancel_calls = []
        self.cancel_result = True
        self.cancel_exc = None
        self.accounts = [{
            'accountId': RAW_ACCOUNT_NUMBER,
            'accountIdKey': 'KEY-ABC',
            'accountType': 'MARGIN',
            'accountDesc': 'Individual Brokerage',
            'accountStatus': 'ACTIVE',
        }]

    def list_accounts(self):
        return list(self.accounts)

    def get_balances(self, account_id_key):
        return {
            'accountId': RAW_ACCOUNT_NUMBER,
            'accountIdKey': account_id_key,
            'Computed': {
                'netAccountValue': 125000.50,
                'cashAvailableForInvestment': 40000.0,
                'RealTimeValues': {'accountId': RAW_ACCOUNT_NUMBER},
            },
        }

    def get_portfolio(self, account_id_key):
        return [{
            'symbol': 'SPY', 'quantity': 10, 'marketValue': 4500.0,
            'totalGain': 250.0, 'accountId': RAW_ACCOUNT_NUMBER,
        }]

    def get_quotes(self, symbols):
        return {s: {'bid': 1.0, 'ask': 1.1, 'last': 1.05} for s in symbols}

    def preview_order(self, account_id_key, order_request):
        self.preview_calls.append((account_id_key, order_request))
        return {
            'PreviewIds': [{'previewId': 555}],
            'Order': [{'estimatedTotalAmount': -101.25,
                       'estimatedCommission': 0.0}],
        }

    def place_order(self, account_id_key, order_request, preview_ids):
        self.place_calls.append((account_id_key, order_request, preview_ids))
        return {'order_id': 'ORD-LIVE-1',
                'response': {'OrderIds': [{'orderId': 'ORD-LIVE-1',
                                           'status': 'OPEN'}]}}

    def cancel_order(self, account_id_key, order_id):
        self.cancel_calls.append((account_id_key, order_id))
        if self.cancel_exc is not None:
            raise self.cancel_exc
        return self.cancel_result


@pytest.fixture()
def patch_trade(monkeypatch):
    """Inject a connected FakeAuthManager + FakeTradeClient into api_live."""
    import gui.routes.api_live as api_live

    fakes = {
        'auth': FakeAuthManager(state='connected'),
        'kill': FakeKillSwitch(),
        'audit': FakeAuditLog(),
        'client': FakeTradeClient(),
    }
    monkeypatch.setattr(api_live, 'get_auth_manager', lambda: fakes['auth'])
    monkeypatch.setattr(api_live, 'get_kill_switch', lambda: fakes['kill'])
    monkeypatch.setattr(api_live, 'get_audit_log', lambda: fakes['audit'])
    monkeypatch.setattr(api_live, 'get_client', lambda: fakes['client'])
    return fakes


def _no_raw_account_number(blob: str):
    assert RAW_ACCOUNT_NUMBER not in blob, \
        'raw account number leaked into the response'


class TestLiveAccountRoutes:
    """R1/R2/R3 GET routes: shape + masking + 409/401 guards."""

    def test_accounts_shape_and_masking(self, client, patch_trade):
        response = client.get('/api/live/accounts')
        assert response.status_code == 200
        body = response.get_json()
        _no_raw_account_number(response.get_data(as_text=True))
        assert len(body['accounts']) == 1
        acct = body['accounts'][0]
        assert acct['accountIdKey'] == 'KEY-ABC'
        assert acct['accountType'] == 'MARGIN'
        assert acct['accountDesc'] == 'Individual Brokerage'
        assert acct['accountStatus'] == 'ACTIVE'
        assert acct['accountId_masked'] == '••••6214'
        assert 'accountId' not in acct

    def test_balances_shape_and_deep_masking(self, client, patch_trade):
        response = client.get('/api/live/accounts/KEY-ABC/balances')
        assert response.status_code == 200
        text = response.get_data(as_text=True)
        _no_raw_account_number(text)
        balances = response.get_json()['balances']
        # Useful computed fields survive.
        assert balances['Computed']['netAccountValue'] == 125000.50
        assert balances['Computed']['cashAvailableForInvestment'] == 40000.0
        # Account number masked at EVERY depth.
        assert balances['accountId_masked'] == '••••6214'
        assert 'accountId' not in balances
        assert balances['Computed']['RealTimeValues']['accountId_masked'] \
            == '••••6214'

    def test_portfolio_shape_and_masking(self, client, patch_trade):
        response = client.get('/api/live/accounts/KEY-ABC/portfolio')
        assert response.status_code == 200
        _no_raw_account_number(response.get_data(as_text=True))
        positions = response.get_json()['positions']
        assert positions[0]['symbol'] == 'SPY'
        assert positions[0]['quantity'] == 10
        assert positions[0]['marketValue'] == 4500.0
        assert 'accountId' not in positions[0]
        assert positions[0]['accountId_masked'] == '••••6214'

    @pytest.mark.parametrize('path', [
        '/api/live/accounts',
        '/api/live/accounts/KEY-ABC/balances',
        '/api/live/accounts/KEY-ABC/portfolio',
        '/api/live/quotes?symbols=SPY',
    ])
    def test_not_connected_is_409_with_state(self, client, monkeypatch, path):
        import gui.routes.api_live as api_live

        monkeypatch.setattr(api_live, 'get_auth_manager',
                            lambda: FakeAuthManager(state='disconnected'))
        monkeypatch.setattr(api_live, 'get_client',
                            lambda: FakeTradeClient())
        response = client.get(path)
        assert response.status_code == 409
        body = response.get_json()
        assert body['error'] == 'not connected'
        assert body['state'] == 'disconnected'

    @pytest.mark.parametrize('path', [
        '/api/live/accounts',
        '/api/live/accounts/KEY-ABC/balances',
        '/api/live/accounts/KEY-ABC/portfolio',
        '/api/live/quotes?symbols=SPY',
    ])
    def test_auth_expired_is_401(self, client, patch_trade, path):
        import gui.routes.api_live as api_live

        def boom(*_a, **_k):
            raise api_live.EtradeAuthExpired('token expired')

        clt = patch_trade['client']
        clt.list_accounts = boom
        clt.get_balances = boom
        clt.get_portfolio = boom
        clt.get_quotes = boom

        response = client.get(path)
        assert response.status_code == 401
        assert response.get_json() == {'error': 'reauthorize'}


class TestLiveQuotes:
    """R4: /quotes shape, de-dupe, and the 25-symbol cap."""

    def test_quotes_round_trip(self, client, patch_trade):
        response = client.get('/api/live/quotes?symbols=SPY,AAPL')
        assert response.status_code == 200
        body = response.get_json()
        assert body['requested'] == ['SPY', 'AAPL']
        assert body['quotes']['SPY'] == {'bid': 1.0, 'ask': 1.1, 'last': 1.05}

    def test_quotes_requires_symbols(self, client, patch_trade):
        response = client.get('/api/live/quotes?symbols=')
        assert response.status_code == 400

    def test_quotes_cap_at_25(self, client, patch_trade):
        symbols = ','.join(f'SYM{i}' for i in range(26))
        response = client.get(f'/api/live/quotes?symbols={symbols}')
        assert response.status_code == 400
        assert 'max' in response.get_json()['error'].lower()


class TestLiveOrderTicket:
    """R5/R6: preview caches an order_ref; place is single-use and places
    exactly the cached request; kill switch / rejection / auth mapping."""

    EQUITY = {'account_id_key': 'KEY-ABC', 'kind': 'equity',
              'symbol': 'SPY', 'side': 'BUY', 'quantity': 1,
              'limit_price': 101.25}

    def test_preview_returns_order_ref_and_caches(self, client, patch_trade):
        import gui.routes.api_live as api_live

        response = client.post('/api/live/order/preview', json=self.EQUITY)
        assert response.status_code == 200
        body = response.get_json()
        assert body['preview']['previewIds'] == [555]
        ref = body['order_ref']
        assert ref and isinstance(ref, str)
        assert ref in api_live._ORDER_REF_CACHE
        # The client previewed exactly the built request.
        assert patch_trade['client'].preview_calls[0][0] == 'KEY-ABC'

    def test_preview_blocks_account_mismatch(self, client, patch_trade,
                                             monkeypatch):
        # Gap 1/4 review: when ETRADE_ACCOUNT_ID_KEY is set (the account the
        # daily-loss rail monitors), an order for a DIFFERENT account is refused
        # so the rail and the order can never diverge.
        monkeypatch.setenv('ETRADE_ACCOUNT_ID_KEY', 'OTHER-ACCT')
        response = client.post('/api/live/order/preview', json=self.EQUITY)
        assert response.status_code == 409
        assert 'account' in response.get_json()['error']

    def test_place_uses_cached_request_exactly_once(self, client, patch_trade):
        preview = client.post('/api/live/order/preview',
                              json=self.EQUITY).get_json()
        ref = preview['order_ref']
        built = patch_trade['client'].preview_calls[0][1]

        response = client.post('/api/live/order/place',
                               json={'order_ref': ref})
        assert response.status_code == 200
        assert response.get_json()['order'] == {
            'orderId': 'ORD-LIVE-1', 'status': 'OPEN'}

        calls = patch_trade['client'].place_calls
        assert len(calls) == 1
        # Placed with the SAME account + SAME cached request + the preview's
        # previewIds — not anything the client posted to /place.
        assert calls[0][0] == 'KEY-ABC'
        assert calls[0][1] is built
        assert calls[0][2] == [{'previewId': 555}]

    def test_order_ref_is_single_use(self, client, patch_trade):
        ref = client.post('/api/live/order/preview',
                          json=self.EQUITY).get_json()['order_ref']
        first = client.post('/api/live/order/place', json={'order_ref': ref})
        assert first.status_code == 200
        second = client.post('/api/live/order/place', json={'order_ref': ref})
        assert second.status_code == 404
        # Only the first place reached the client.
        assert len(patch_trade['client'].place_calls) == 1

    def test_place_unknown_ref_is_404(self, client, patch_trade):
        response = client.post('/api/live/order/place',
                               json={'order_ref': 'nope-nope'})
        assert response.status_code == 404
        assert patch_trade['client'].place_calls == []

    def test_place_cannot_post_arbitrary_payload(self, client, patch_trade):
        """No order_ref -> 400; an order_request body is ignored (place only
        ever works from a cached, previewed ref)."""
        response = client.post('/api/live/order/place',
                               json={'order_request': {'evil': True}})
        assert response.status_code == 400
        assert patch_trade['client'].place_calls == []

    def test_kill_switch_blocks_place_with_409(self, client, patch_trade):
        import gui.routes.api_live as api_live

        ref = client.post('/api/live/order/preview',
                          json=self.EQUITY).get_json()['order_ref']

        def engaged(*_a, **_k):
            raise api_live.KillSwitchEngaged(
                'Kill switch engaged — new orders are blocked')

        patch_trade['client'].place_order = engaged
        response = client.post('/api/live/order/place',
                               json={'order_ref': ref})
        assert response.status_code == 409
        assert response.get_json()['error'] == 'kill switch engaged'

    def test_circuit_breaker_labeled_in_409(self, client, patch_trade):
        import gui.routes.api_live as api_live

        ref = client.post('/api/live/order/preview',
                          json=self.EQUITY).get_json()['order_ref']

        def breaker(*_a, **_k):
            raise api_live.KillSwitchEngaged(
                'Daily-loss circuit breaker tripped — new orders are blocked')

        patch_trade['client'].place_order = breaker
        response = client.post('/api/live/order/place',
                               json={'order_ref': ref})
        assert response.status_code == 409
        assert response.get_json()['error'] == 'circuit breaker'

    def test_kill_switch_blocks_preview_with_409(self, client, patch_trade):
        import gui.routes.api_live as api_live

        def engaged(*_a, **_k):
            raise api_live.KillSwitchEngaged('Kill switch engaged')

        patch_trade['client'].preview_order = engaged
        response = client.post('/api/live/order/preview', json=self.EQUITY)
        assert response.status_code == 409
        assert response.get_json()['error'] == 'kill switch engaged'

    def test_order_rejected_is_422_with_reason(self, client, patch_trade):
        import gui.routes.api_live as api_live

        ref = client.post('/api/live/order/preview',
                          json=self.EQUITY).get_json()['order_ref']

        def reject(*_a, **_k):
            raise api_live.EtradeOrderRejected('Insufficient buying power')

        patch_trade['client'].place_order = reject
        response = client.post('/api/live/order/place',
                               json={'order_ref': ref})
        assert response.status_code == 422
        body = response.get_json()
        assert body['error'] == 'order rejected'
        assert body['reason'] == 'Insufficient buying power'

    def test_place_auth_expired_is_401(self, client, patch_trade):
        import gui.routes.api_live as api_live

        ref = client.post('/api/live/order/preview',
                          json=self.EQUITY).get_json()['order_ref']

        def expired(*_a, **_k):
            raise api_live.EtradeAuthExpired('expired')

        patch_trade['client'].place_order = expired
        response = client.post('/api/live/order/place',
                               json={'order_ref': ref})
        assert response.status_code == 401
        assert response.get_json() == {'error': 'reauthorize'}

    def test_unknown_kind_is_400(self, client, patch_trade):
        response = client.post('/api/live/order/preview', json={
            'account_id_key': 'KEY-ABC', 'kind': 'futures', 'symbol': 'ES'})
        assert response.status_code == 400
        assert patch_trade['client'].preview_calls == []

    @pytest.mark.parametrize('body', [
        {'account_id_key': 'KEY-ABC', 'kind': 'equity', 'side': 'BUY',
         'quantity': 1},                                    # no symbol
        {'account_id_key': 'KEY-ABC', 'kind': 'equity', 'symbol': 'SPY',
         'side': 'HOLD', 'quantity': 1},                    # bad side
        {'account_id_key': 'KEY-ABC', 'kind': 'equity', 'symbol': 'SPY',
         'side': 'BUY', 'quantity': 0},                     # non-positive qty
        {'kind': 'equity', 'symbol': 'SPY', 'side': 'BUY',
         'quantity': 1},                                    # no account
    ])
    def test_missing_required_fields_is_400(self, client, patch_trade, body):
        response = client.post('/api/live/order/preview', json=body)
        assert response.status_code == 400
        assert patch_trade['client'].preview_calls == []

    @pytest.mark.parametrize('bad', ['inf', '-inf', 'Infinity', 'nan', 'NaN'])
    def test_non_finite_quantity_is_400_not_500(self, client, patch_trade,
                                                bad):
        """A non-finite string quantity must be a clean 400 — never a route
        500. float('inf') parses, but int(float('inf')) raises OverflowError
        and int(float('nan')) raises ValueError; _to_int rejects both up
        front so the request never reaches the broker preview."""
        body = {'account_id_key': 'KEY-ABC', 'kind': 'equity',
                'symbol': 'SPY', 'side': 'BUY', 'quantity': bad}
        response = client.post('/api/live/order/preview', json=body)
        assert response.status_code == 400
        assert response.get_json()['error'] == \
            "'quantity' must be a positive integer"
        assert patch_trade['client'].preview_calls == []

    @pytest.mark.parametrize('bad', ['nan', 'NaN', 'inf', '-inf', 'Infinity'])
    def test_non_finite_option_strike_is_400_not_sent(self, client,
                                                      patch_trade, bad):
        """NaN/Inf must NOT slip past the strike positivity guard: every
        comparison with NaN is False, so without the finite check a
        strike='nan' would build a real OPTN order carrying a non-JSON
        NaN token and reach the broker preview. It must 400 and never be
        sent."""
        body = {'account_id_key': 'KEY-ABC', 'kind': 'option',
                'symbol': 'SPY', 'call_put': 'CALL', 'strike': bad,
                'expiry': '2026-07-17', 'action': 'BUY_OPEN', 'quantity': 1}
        response = client.post('/api/live/order/preview', json=body)
        assert response.status_code == 400
        assert response.get_json()['error'] == \
            "'strike' must be a positive number"
        assert patch_trade['client'].preview_calls == []

    @pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
    def test_non_finite_limit_price_dropped_to_market(self, client,
                                                      patch_trade, value):
        """A non-finite limit_price is treated as absent (-> MARKET order),
        never carried into the built order as a NaN/Inf limitPrice."""
        body = {'account_id_key': 'KEY-ABC', 'kind': 'equity',
                'symbol': 'SPY', 'side': 'BUY', 'quantity': 1,
                'limit_price': value}
        response = client.post('/api/live/order/preview', json=body)
        assert response.status_code == 200
        built = patch_trade['client'].preview_calls[0][1]
        order = built['Order'][0]
        assert order['priceType'] == 'MARKET'
        assert 'limitPrice' not in order

    def test_finite_limit_price_survives(self, client, patch_trade):
        """Sanity: a real limit price still produces a LIMIT order."""
        body = {'account_id_key': 'KEY-ABC', 'kind': 'equity',
                'symbol': 'SPY', 'side': 'BUY', 'quantity': 1,
                'limit_price': '99.50'}
        response = client.post('/api/live/order/preview', json=body)
        assert response.status_code == 200
        order = patch_trade['client'].preview_calls[0][1]['Order'][0]
        assert order['priceType'] == 'LIMIT'
        assert order['limitPrice'] == 99.50

    def test_non_finite_spread_net_price_is_400_not_sent(self, client,
                                                         patch_trade):
        """spread net_price='nan' must 400 (non-zero guard) and never be
        built into a SPREADS order carrying a NaN netPrice."""
        body = {
            'account_id_key': 'KEY-ABC', 'kind': 'spread', 'net_price': 'nan',
            'legs': [
                {'symbol': 'SPY', 'call_put': 'PUT', 'strike': 450,
                 'expiry': '2026-07-17', 'action': 'SELL_OPEN', 'quantity': 1},
                {'symbol': 'SPY', 'call_put': 'PUT', 'strike': 445,
                 'expiry': '2026-07-17', 'action': 'BUY_OPEN', 'quantity': 1},
            ],
        }
        response = client.post('/api/live/order/preview', json=body)
        assert response.status_code == 400
        assert patch_trade['client'].preview_calls == []

    def test_spread_vertical_round_trip(self, client, patch_trade):
        body = {
            'account_id_key': 'KEY-ABC', 'kind': 'spread', 'net_price': 1.25,
            'legs': [
                {'symbol': 'SPY', 'call_put': 'PUT', 'strike': 450,
                 'expiry': '2026-07-17', 'action': 'SELL_OPEN', 'quantity': 1},
                {'symbol': 'SPY', 'call_put': 'PUT', 'strike': 445,
                 'expiry': '2026-07-17', 'action': 'BUY_OPEN', 'quantity': 1},
            ],
        }
        response = client.post('/api/live/order/preview', json=body)
        assert response.status_code == 200
        assert response.get_json()['preview']['previewIds'] == [555]
        built = patch_trade['client'].preview_calls[0][1]
        assert built['orderType'] == 'SPREADS'

    def test_spread_rejects_single_leg(self, client, patch_trade):
        body = {
            'account_id_key': 'KEY-ABC', 'kind': 'spread', 'net_price': 1.0,
            'legs': [{'symbol': 'SPY', 'call_put': 'PUT', 'strike': 450,
                      'expiry': '2026-07-17', 'action': 'SELL_OPEN',
                      'quantity': 1}],
        }
        response = client.post('/api/live/order/preview', json=body)
        assert response.status_code == 400
        assert patch_trade['client'].preview_calls == []

    def test_order_routes_503_when_client_unavailable(self, client,
                                                      monkeypatch):
        import gui.routes.api_live as api_live

        sentinel = 'ImportError: synthetic etrade_client import failure'
        monkeypatch.setattr(api_live, 'EtradeClient', None)
        monkeypatch.setattr(api_live, 'CLIENT_IMPORT_ERROR', sentinel)

        response = client.get('/api/live/accounts')
        assert response.status_code == 503
        assert response.get_json()['reason'] == sentinel

    def test_live_page_ships_order_ticket_containers(self, client):
        html = client.get('/live').get_data(as_text=True)
        for marker in (
            'id="accountCard"', 'id="accountSelect"', 'id="accountBody"',
            'id="quoteCard"', 'id="quoteSymbols"', 'id="quoteBody"',
            'id="orderTicketCard"', 'id="orderTicketForm"', 'id="ticketKind"',
            'id="paneEquity"', 'id="paneOption"', 'id="paneSpread"',
            'id="btnPreview"', 'id="btnPlace"', 'id="placeConfirmModal"',
        ):
            assert marker in html, f'/live missing {marker}'


class TestLauncherEnvHygiene:
    """run_gui.py must never let Flask bulk-load the repo-root .env.

    Flask's app.run() calls cli.load_dotenv() whenever python-dotenv is
    installed, silently injecting the operator's real
    ETRADE_CONSUMER_KEY/SECRET into os.environ — even when the operator
    explicitly stripped all ETRADE_* vars. The auth manager then reports
    'disconnected' (configured) instead of 'unconfigured', and a single
    POST /api/live/auth/start makes a REAL HTTPS request to E*TRADE.
    Credentials enter the process only by explicit operator action, so
    the launcher must pass load_dotenv=False (Phase 9 QA regression).
    """

    def test_run_gui_disables_flask_dotenv_loading(self, monkeypatch):
        import runpy

        import flask

        captured = {}

        def fake_run(self, *args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs

        # Stub the server out entirely: nothing binds a port, and the
        # dotenv-loading branch inside the real run() never executes.
        monkeypatch.setattr(flask.Flask, 'run', fake_run)

        runpy.run_path(str(REPO_ROOT / 'run_gui.py'), run_name='__main__')

        assert 'kwargs' in captured, 'launcher never called app.run()'
        assert captured['kwargs'].get('load_dotenv') is False


class TestStartScript:
    """start.sh is the explicit-operator-action launcher: it sources .env in
    the operator's shell, then runs run_gui.py — WITHOUT weakening run_gui.py's
    no-silent-load control (TestLauncherEnvHygiene)."""

    def test_start_script_exists_and_is_executable(self):
        import os
        script = REPO_ROOT / 'start.sh'
        assert script.is_file(), 'start.sh is missing'
        assert os.access(str(script), os.X_OK), 'start.sh is not executable'

    def test_start_script_sources_env_then_runs_launcher(self):
        text = (REPO_ROOT / 'start.sh').read_text()
        # set -a / set +a bracket the source so .env vars are EXPORTED.
        assert 'set -a' in text and 'set +a' in text
        assert 'source .env' in text
        assert 'run_gui.py' in text

    def test_start_script_is_valid_bash(self):
        import subprocess
        result = subprocess.run(
            ['bash', '-n', str(REPO_ROOT / 'start.sh')],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_wrapper_does_not_touch_launcher_security_control(self):
        # The wrapper must NOT have changed run_gui.py's deliberate control.
        text = (REPO_ROOT / 'run_gui.py').read_text()
        assert 'load_dotenv=False' in text


class TestGetClientNoDeadlock:
    """Regression for the singleton-lock self-deadlock: get_client() holds
    _singleton_lock while calling get_kill_switch()/get_audit_log(), which
    re-acquire it. With a plain Lock this hangs forever on the first real
    call; the mocked route tests never caught it because they patch
    get_client() wholesale. This exercises the REAL accessor.
    """

    def test_real_get_client_builds_without_deadlock(self, monkeypatch,
                                                     tmp_path):
        import threading
        import gui.routes.api_live as api_live

        if api_live.EtradeClient is None:
            pytest.skip('EtradeClient unavailable')

        # Isolate from the developer's real DB and reset the singletons.
        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'dl.db'))
        monkeypatch.setenv('ETRADE_ENV', 'sandbox')
        monkeypatch.setenv('ETRADE_SANDBOX_CONSUMER_KEY', 'dl-key')
        monkeypatch.setenv('ETRADE_SANDBOX_CONSUMER_SECRET', 'dl-secret')
        for attr in ('_auth_manager', '_kill_switch', '_audit_log',
                     '_client', '_client_auth_manager'):
            monkeypatch.setattr(api_live, attr, None)

        # The lock must be reentrant for the nested accessors to work.
        assert type(api_live._singleton_lock).__name__ == 'RLock'

        result = {}

        def _build():
            result['client'] = api_live.get_client()  # offline construction

        worker = threading.Thread(target=_build)
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive(), (
            'get_client() deadlocked on the singleton lock')
        assert result.get('client') is not None


class TestCancelOrderRouting:
    """The cancel route has two sources: a GUI-placed order (account_id_key
    given) cancels via the shared client; a patient-executor working order
    (no account_id_key) falls back to the live broker. Regression for the
    gap where GUI-placed orders had no working cancel path."""

    def test_cancel_with_account_key_uses_client(self, client, patch_trade,
                                                 monkeypatch):
        import gui.routes.api_live as api_live
        # The broker path must NOT be taken when an account key is supplied.
        monkeypatch.setattr(api_live, '_find_live_broker',
                            lambda: (_ for _ in ()).throw(
                                AssertionError('broker path used')))
        resp = client.post('/api/live/orders/ORD-LIVE-1/cancel',
                           json={'account_id_key': 'KEY-ABC'})
        assert resp.status_code == 200
        assert patch_trade['client'].cancel_calls == [('KEY-ABC', 'ORD-LIVE-1')]

    def test_cancel_with_account_key_not_found(self, client, patch_trade):
        patch_trade['client'].cancel_result = False
        resp = client.post('/api/live/orders/ORD-X/cancel',
                           json={'account_id_key': 'KEY-ABC'})
        assert resp.status_code == 404

    def test_cancel_with_account_key_maps_client_error(self, client,
                                                       patch_trade):
        from brokers.etrade_client import EtradeApiError
        patch_trade['client'].cancel_exc = EtradeApiError('upstream boom')
        resp = client.post('/api/live/orders/ORD-LIVE-1/cancel',
                           json={'account_id_key': 'KEY-ABC'})
        # _client_error_response maps EtradeApiError -> 502 (not a 500 leak).
        assert resp.status_code == 502

    def test_cancel_without_account_key_falls_back_to_broker(self, client,
                                                            patch_trade,
                                                            monkeypatch):
        import gui.routes.api_live as api_live

        class _Broker:
            def __init__(self):
                self.calls = []

            def cancel_order(self, order_id):
                self.calls.append(order_id)
                return True

        broker = _Broker()
        monkeypatch.setattr(api_live, '_find_live_broker', lambda: broker)
        resp = client.post('/api/live/orders/WORK-1/cancel', json={})
        assert resp.status_code == 200
        assert broker.calls == ['WORK-1']
        # The client cancel path was NOT used.
        assert patch_trade['client'].cancel_calls == []


# ==================== WALK-FORWARD MODEL SELECTION (Phase A) ====================

# SHARED INTERFACE CONTRACT: GET /api/models entries carry exactly these keys
# (the desks.models.available_models() shape the desk picker renders).
MODEL_CONTRACT_KEYS = {'id', 'name', 'description'}


class TestModelsEndpoint:
    """GET /api/models proxies desks.models.available_models()."""

    def test_returns_models_from_registry(self, client):
        response = client.get('/api/models')

        assert response.status_code == 200
        body = response.get_json()
        assert set(body) == {'models'}
        models = body['models']
        # Exactly the six registry models, in order: the three Phase-A models,
        # the two Phase-B neural models (mlp, lstm), then the Phase-D
        # transparent factor model that powers the AQR desk.
        assert [m['id'] for m in models] == [
            'gbm', 'lightgbm', 'stacking', 'mlp', 'lstm', 'factor']
        assert [m['name'] for m in models] == [
            'Gradient Boosting', 'LightGBM', 'Stacking Ensemble',
            'Neural MLP', 'Neural LSTM', 'Factor (AQR)']
        for entry in models:
            assert set(entry) == MODEL_CONTRACT_KEYS

    def test_returns_503_when_framework_unavailable(self, client, monkeypatch):
        """Mirrors the desk/fund pattern: a failed desks import is a loud 503
        so the frontend can disable the picker (never a 500)."""
        from gui.routes import api_backtest

        sentinel = 'ImportError: synthetic desks import failure'
        monkeypatch.setattr(api_backtest, 'available_models', None)
        monkeypatch.setattr(api_backtest, 'DESK_REGISTRY_IMPORT_ERROR',
                            sentinel)

        response = client.get('/api/models')

        assert response.status_code == 503
        assert response.get_json() == {
            'error': 'Desk framework unavailable',
            'reason': sentinel,
        }


class TestDeskModelSelectionRoute:
    """POST /api/backtest/run desk_model validation (Phase A).

    The async run is faked exactly like TestDeskModeBacktest, but the
    create_desk stub here accepts the model_key keyword the route now
    passes, so the desk_model threads through to construction.
    """

    PAYLOAD = {
        'symbols': 'AAA',
        'desk': 'foundation',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'position_size': 0.1,
    }

    @pytest.fixture()
    def patch_desk_stack(self, monkeypatch, tmp_path):
        """Stub create_desk (model_key-aware) + the engine; redirect the DB."""
        import gui.globals as gui_globals
        from gui.routes import api_backtest

        monkeypatch.setenv('TRADING_DB_PATH', str(tmp_path / 'history.db'))
        monkeypatch.setattr(gui_globals, '_db', None)

        seen = {}
        desk_sentinel = object()

        def fake_create_desk(key, capital_allocation=1.0, model_key=None):
            seen['desk_key'] = key
            seen['model_key'] = model_key
            return desk_sentinel

        class FakeEngine:
            def __init__(self, strategy=None, desk=None,
                         initial_capital=100000, **kwargs):
                self.market_data = None

            def run(self, symbols, start_date, end_date, position_size,
                    progress_callback=None, benchmark_symbol='SPY'):
                if progress_callback is not None:
                    progress_callback(50.0)
                return _fake_desk_report()

        monkeypatch.setattr(api_backtest, 'create_desk', fake_create_desk)
        monkeypatch.setattr(api_backtest, 'BacktestEngine', FakeEngine)
        return seen

    def test_valid_model_id_threads_through_to_create_desk(
            self, client, patch_desk_stack):
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk_model': 'gbm'})

        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        # The model_key reached create_desk on the foundation desk.
        assert patch_desk_stack['desk_key'] == 'foundation'
        assert patch_desk_stack['model_key'] == 'gbm'

    def test_model_id_is_lowercased_before_validation(
            self, client, patch_desk_stack):
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk_model': 'LightGBM'})

        assert response.status_code == 202
        _wait_for_job(client, response.get_json()['job_id'])
        assert patch_desk_stack['model_key'] == 'lightgbm'

    def test_unknown_model_id_returns_400(self, client):
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk_model': 'bogus'})

        assert response.status_code == 400
        assert 'Unknown desk_model' in response.get_json()['error']

    @pytest.mark.parametrize('bad_model', ['', '   ', 123, ['gbm'], {'x': 1}])
    def test_blank_or_non_string_model_id_returns_400(self, client, bad_model):
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk_model': bad_model})

        assert response.status_code == 400
        assert 'desk_model' in response.get_json()['error']

    def test_valid_model_id_threads_through_on_twosigma(
            self, client, patch_desk_stack):
        """Phase C: desk='twosigma' is model-selectable, so a valid
        desk_model submits (202) and threads through to create_desk."""
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk': 'twosigma', 'desk_model': 'mlp'})

        assert response.status_code == 202
        job = _wait_for_job(client, response.get_json()['job_id'])
        assert job['status'] == 'done'
        assert patch_desk_stack['desk_key'] == 'twosigma'
        assert patch_desk_stack['model_key'] == 'mlp'

    def test_unknown_model_id_on_twosigma_returns_400(self, client):
        """A bogus model id is rejected even on the twosigma desk."""
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk': 'twosigma', 'desk_model': 'bogus'})

        assert response.status_code == 400
        assert 'Unknown desk_model' in response.get_json()['error']

    def test_desk_model_with_non_selectable_desk_returns_400(self, client):
        """desk_model is only valid for the foundation or twosigma desk."""
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk': 'renaissance', 'desk_model': 'gbm'})

        assert response.status_code == 400
        assert ('desk_model is only valid for the foundation or twosigma desk'
                in response.get_json()['error'])

    @pytest.mark.parametrize('desk', ['citadel', 'janestreet'])
    def test_desk_model_with_other_desks_returns_400(self, client, desk):
        """The remaining non-selectable desks also reject desk_model."""
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk': desk, 'desk_model': 'gbm'})

        assert response.status_code == 400
        assert 'twosigma desk' in response.get_json()['error']

    def test_desk_model_in_strategy_mode_returns_400(self, client):
        """desk_model with no desk (strategy mode) is rejected, not ignored."""
        payload = {k: v for k, v in self.PAYLOAD.items() if k != 'desk'}
        response = client.post(
            '/api/backtest/run',
            json={**payload, 'strategy': 'momentum', 'desk_model': 'gbm'})

        assert response.status_code == 400
        assert ('desk_model is only valid for the foundation or twosigma desk'
                in response.get_json()['error'])

    def test_foundation_without_desk_model_passes_none(
            self, client, patch_desk_stack):
        """Absent desk_model -> model_key=None (byte-identical default)."""
        response = client.post('/api/backtest/run', json=self.PAYLOAD)
        assert response.status_code == 202
        _wait_for_job(client, response.get_json()['job_id'])
        assert patch_desk_stack['model_key'] is None

    def test_desk_model_unavailable_framework_returns_400(
            self, client, monkeypatch):
        """When the desks package failed to import, a present desk_model is a
        clean 400 (available_models is None) rather than a silent ignore."""
        from gui.routes import api_backtest

        monkeypatch.setattr(api_backtest, 'available_models', None)
        response = client.post(
            '/api/backtest/run',
            json={**self.PAYLOAD, 'desk_model': 'gbm'})

        assert response.status_code == 400
        assert 'Desk framework unavailable' in response.get_json()['error']
