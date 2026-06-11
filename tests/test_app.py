"""Tests for the Flask application factory in gui/app.py.

Offline and deterministic: no network calls, no real market data. The app
factory is exercised through Flask's test client only.
"""
from __future__ import annotations

import pytest

from gui.app import create_app


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
