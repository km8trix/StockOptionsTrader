"""Registry-hazard tests for /api/trader/create.

Companion to the PaperTrader lock work (TestConcurrency in
test_paper_trader.py): under gunicorn --threads, create_trader used to
silently replace an existing trader mid-mutation. It now refuses with a
409 unless an explicit 'replace': true flag is sent. Lives here rather
than in test_app.py, which is frontend-owned.
"""

from __future__ import annotations

import pytest

import gui.routes.api_trading as api_trading
from gui.app import create_app


@pytest.fixture()
def client():
    app = create_app({'TESTING': True})
    return app.test_client()


@pytest.fixture(autouse=True)
def clean_registry():
    """active_traders is module-level state shared across tests; isolate it."""
    saved = dict(api_trading.active_traders)
    api_trading.active_traders.clear()
    yield
    api_trading.active_traders.clear()
    api_trading.active_traders.update(saved)


def _create(client, trader_id, **extra):
    payload = {'trader_id': trader_id, 'mode': 'paper',
               'initial_capital': 10_000, **extra}
    return client.post('/api/trader/create', json=payload)


def test_create_trader_succeeds_first_time(client):
    response = _create(client, 'reg-t1')

    assert response.status_code == 200
    assert response.get_json() == {
        'trader_id': 'reg-t1', 'mode': 'paper', 'status': 'created'}
    assert 'reg-t1' in api_trading.active_traders


def test_duplicate_trader_id_returns_409_and_keeps_original(client):
    assert _create(client, 'reg-t1').status_code == 200
    original = api_trading.active_traders['reg-t1']

    response = _create(client, 'reg-t1')

    assert response.status_code == 409
    body = response.get_json()
    assert 'already exists' in body['error']
    assert 'reg-t1' in body['error']
    # The existing trader must NOT have been silently replaced.
    assert api_trading.active_traders['reg-t1'] is original


def test_explicit_replace_flag_overwrites_existing_trader(client):
    assert _create(client, 'reg-t1').status_code == 200
    original = api_trading.active_traders['reg-t1']

    response = _create(client, 'reg-t1', replace=True)

    assert response.status_code == 200
    assert response.get_json()['status'] == 'created'
    assert api_trading.active_traders['reg-t1'] is not original
