"""Phase 4 ops: correlation ids on every response + the /metrics endpoint."""

import re

from gui.app import create_app
from utils import metrics


def _client():
    return create_app({'TESTING': True}).test_client()


def test_response_carries_generated_correlation_id():
    resp = _client().get('/health')
    cid = resp.headers.get('X-Request-ID')
    assert cid and re.fullmatch(r'[0-9a-f]{12}', cid)


def test_incoming_correlation_id_is_echoed():
    resp = _client().get('/health', headers={'X-Request-ID': 'trace-123'})
    assert resp.headers.get('X-Request-ID') == 'trace-123'


def test_metrics_endpoint_counts_requests():
    metrics.reset()
    client = _client()
    client.get('/health')  # one 2xx, counted by after_request before /metrics
    resp = client.get('/metrics')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'app_uptime_seconds' in body
    assert 'http_requests_total{status="2xx"}' in body
