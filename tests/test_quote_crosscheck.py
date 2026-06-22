"""Phase 4 ops: optional second live-quote source sanity cross-check.

Opt-in (default off => byte-identical), fail-open (a flaky reference never
blocks trading), and the primary E*TRADE price is always returned unchanged."""

from brokers.live_trader import LiveEtradeBroker
from utils import metrics


class _FakeClient:
    """Duck-typed client (get_quotes + preview_order) so the broker accepts it
    as a prebuilt client."""

    def __init__(self, last=100.0):
        self._last = last
        self.circuit_breaker = None

    def get_quotes(self, symbols):
        return {s: {'bid': self._last - 0.1, 'ask': self._last + 0.1,
                    'last': self._last} for s in symbols}

    def preview_order(self, *a, **k):  # marks this as a client
        return {}


def _broker(reference_price_fn=None, threshold=0.05, last=100.0):
    return LiveEtradeBroker(_FakeClient(last=last), account_id_key='ACC',
                            reference_price_fn=reference_price_fn,
                            quote_divergence_threshold=threshold)


def test_no_reference_means_no_crosscheck():
    metrics.reset()
    assert _broker(reference_price_fn=None, last=100.0).get_current_price('SPY') == 100.0
    assert 'quote_divergence_total' not in metrics.render()


def test_divergence_beyond_threshold_warns_and_counts():
    metrics.reset()
    b = _broker(reference_price_fn=lambda s: 100.0, threshold=0.05, last=110.0)  # 10%
    assert b.get_current_price('SPY') == 110.0  # primary unchanged
    assert 'quote_divergence_total{symbol="SPY"}' in metrics.render()


def test_within_threshold_does_not_count():
    metrics.reset()
    b = _broker(reference_price_fn=lambda s: 100.0, threshold=0.05, last=102.0)  # 2%
    assert b.get_current_price('SPY') == 102.0
    assert 'quote_divergence_total' not in metrics.render()


def test_reference_error_is_fail_open():
    metrics.reset()

    def boom(symbol):
        raise RuntimeError("reference down")

    assert _broker(reference_price_fn=boom, last=110.0).get_current_price('SPY') == 110.0
    assert 'quote_divergence_total' not in metrics.render()


def test_reference_none_is_skipped():
    metrics.reset()
    b = _broker(reference_price_fn=lambda s: None, last=110.0)
    assert b.get_current_price('SPY') == 110.0
    assert 'quote_divergence_total' not in metrics.render()
