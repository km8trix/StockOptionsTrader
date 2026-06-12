"""Tests for the Flask application factory in gui/app.py.

Offline and deterministic: no network calls, no real market data. Market
data access points are monkeypatched; the app factory is exercised through
Flask's test client only.
"""
from __future__ import annotations

import os
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


class TestViewsAndVendoredAssets:
    """Phase 3 UI: every page renders and references only vendored assets."""

    # /trading_floor is the Stage 1 alias for /floor; both must keep working.
    PAGES = ['/', '/analysis', '/backtest', '/paper_trade', '/floor',
             '/trading_floor']

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

    def test_floor_page_shows_three_desks_with_phase_badges(self, client):
        html = client.get('/floor').get_data(as_text=True)
        for desk in ('Renaissance', 'Citadel', 'Jane Street'):
            assert desk in html
        for phase in ('Phase 6', 'Phase 7', 'Phase 8'):
            assert f'Activates in {phase}' in html

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

    def test_legacy_backtest_accepts_symbols_as_json_array(
            self, client, monkeypatch, make_ohlcv, patch_fetch):
        """A JSON array of symbols is a natural client payload — no 500."""
        patch_fetch({'AAA': make_ohlcv(n_days=120, seed=1)})
        monkeypatch.setattr(
            MarketDataHandler, 'get_last_fetch_info',
            lambda self, symbol: None, raising=False)

        response = client.post(
            '/api/backtest',
            json={**self.BACKTEST_PAYLOAD, 'symbols': ['aaa']})

        assert response.status_code == 200
        assert set(response.get_json()['data_sources']) == {'AAA'}

    def test_legacy_backtest_rejects_bad_symbols_type_with_400(self, client):
        response = client.post(
            '/api/backtest', json={**self.BACKTEST_PAYLOAD, 'symbols': 123})

        assert response.status_code == 400
        assert 'symbols' in response.get_json()['error']


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
