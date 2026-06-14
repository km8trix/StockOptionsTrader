"""Tests for backtesting.parity_harness (live-vs-backtest execution parity).
Offline, deterministic — synthetic audit entries, no DB, no network."""

from __future__ import annotations

import pytest

from backtesting.parity_harness import compute_parity


def _report(seq, symbol, action, *, avg_fill, shortfall, status='filled',
            arrival_mid=None, asset_type=None, quantity=None, commission=None):
    payload = {'symbol': symbol, 'action': action, 'status': status,
               'avg_fill': avg_fill, 'shortfall_per_unit': shortfall}
    if arrival_mid is not None:
        payload['arrival_mid'] = arrival_mid
    if asset_type is not None:
        payload['asset_type'] = asset_type
    if quantity is not None:
        payload['quantity'] = quantity
    if commission is not None:
        payload['commission'] = commission
    return {'seq': seq, 'ts': '2026-06-14T15:00:00+00:00', 'env': 'sandbox',
            'actor': 'live_session', 'event_type': 'execution_report',
            'payload': payload}


def _start(seq, symbol, action, side, quantity):
    return {'seq': seq, 'ts': '2026-06-14T15:00:00+00:00', 'env': 'sandbox',
            'actor': 'live_session', 'event_type': 'execution_start',
            'payload': {'symbol': symbol, 'action': action, 'side': side,
                        'quantity': quantity}}


class TestComputeParitySingleFill:
    def test_enriched_stock_buy_drift(self):
        # arrival_mid 100, paid 100.10 -> realized 0.10/unit; model expects
        # 100*5/1e4 = 0.05; drift 0.05/unit over 100 shares = $5.00.
        entries = [_report(1, 'AAA', 'BUY', avg_fill=100.10, shortfall=0.10,
                           arrival_mid=100.0, asset_type='stock', quantity=100)]
        out = compute_parity(entries, slippage_bps=5.0)
        assert out['n_fills'] == 1 and out['n_skipped'] == 0
        f = out['fills'][0]
        assert f['arrival_mid'] == pytest.approx(100.0)
        assert f['arrival_mid_recovered'] is False
        assert f['realized_shortfall'] == pytest.approx(0.10)
        assert f['expected_shortfall'] == pytest.approx(0.05)
        assert f['slippage_drift'] == pytest.approx(0.05)
        assert f['slippage_drift_total'] == pytest.approx(5.0)
        # Modeled stock commission = qty * modeled_fill * commission,
        # modeled_fill = 100*(1+0.0005) = 100.05 -> 100*100.05*0.001 = 10.005.
        assert f['modeled_commission'] == pytest.approx(10.005)
        assert f['actual_commission'] is None
        assert f['commission_drift'] is None

    def test_enriched_option_buy_uses_haircut_floor(self):
        # call: expected = max(2.00*0.02, 0.05) = 0.05; realized 0.07 ->
        # drift 0.02/unit * 3 contracts * 100 multiplier = $6.00.
        entries = [_report(1, 'AAA', 'BUY', avg_fill=2.07, shortfall=0.07,
                           arrival_mid=2.00, asset_type='call', quantity=3)]
        out = compute_parity(entries, spread_pct=0.02, min_option_haircut=0.05,
                             per_contract_commission=0.65)
        f = out['fills'][0]
        assert f['asset_type'] == 'call'
        assert f['expected_shortfall'] == pytest.approx(0.05)
        assert f['slippage_drift'] == pytest.approx(0.02)
        assert f['slippage_drift_total'] == pytest.approx(6.0)  # x100 multiplier
        assert f['modeled_commission'] == pytest.approx(1.95)  # 3 * 0.65

    def test_legacy_row_recovers_arrival_mid_and_pairs_quantity(self):
        # No arrival_mid/asset_type/quantity on the report (pre-Step-5 row).
        # SELL: arrival_mid = avg_fill + shortfall = 99.90 + 0.10 = 100.00.
        entries = [_start(1, 'AAA', 'SELL', 'SELL', 50),
                   _report(2, 'AAA', 'SELL', avg_fill=99.90, shortfall=0.10)]
        out = compute_parity(entries, slippage_bps=5.0)
        f = out['fills'][0]
        assert f['arrival_mid'] == pytest.approx(100.0)
        assert f['arrival_mid_recovered'] is True
        assert f['asset_type'] == 'stock'  # default for legacy
        assert f['quantity'] == 50  # backfilled from execution_start
        assert f['expected_shortfall'] == pytest.approx(0.05)
        assert f['slippage_drift_total'] == pytest.approx(2.5)  # 0.05*50

    def test_filled_quantity_preferred_over_intended(self):
        # Partial fill: intended 100, only 60 filled -> totals scale by 60.
        entries = [_report(1, 'AAA', 'BUY', avg_fill=100.10, shortfall=0.10,
                           arrival_mid=100.0, asset_type='stock', quantity=100)]
        entries[0]['payload']['filled_quantity'] = 60
        out = compute_parity(entries, slippage_bps=5.0)
        f = out['fills'][0]
        assert f['quantity'] == 60  # filled, not the intended 100
        assert f['slippage_drift_total'] == pytest.approx(0.05 * 60)  # 3.0

    def test_non_finite_fill_is_skipped(self):
        entries = [_report(1, 'AAA', 'BUY', avg_fill=float('nan'),
                           shortfall=0.10, arrival_mid=100.0,
                           asset_type='stock', quantity=10),
                   _report(2, 'BBB', 'BUY', avg_fill=100.10,
                           shortfall=float('inf'), arrival_mid=100.0,
                           asset_type='stock', quantity=10)]
        out = compute_parity(entries)
        assert out['n_fills'] == 0 and out['n_skipped'] == 2

    def test_actual_commission_used_when_present(self):
        entries = [_report(1, 'AAA', 'BUY', avg_fill=100.10, shortfall=0.10,
                           arrival_mid=100.0, asset_type='stock', quantity=100,
                           commission=12.0)]
        out = compute_parity(entries, slippage_bps=5.0)
        f = out['fills'][0]
        assert f['actual_commission'] == pytest.approx(12.0)
        # commission_drift = actual - modeled = 12.0 - 10.005.
        assert f['commission_drift'] == pytest.approx(12.0 - 10.005)
        assert out['commission_available'] is True


class TestComputeParitySkips:
    def test_error_and_unfilled_reports_skipped(self):
        entries = [
            _report(1, 'AAA', 'BUY', avg_fill=None, shortfall=None,
                    status='error'),
            _report(2, 'BBB', 'BUY', avg_fill=None, shortfall=None,
                    status='killed'),
            _report(3, 'CCC', 'BUY', avg_fill=50.05, shortfall=0.05,
                    arrival_mid=50.0, asset_type='stock', quantity=10),
        ]
        out = compute_parity(entries, slippage_bps=5.0)
        assert out['n_fills'] == 1
        assert out['n_skipped'] == 2

    def test_unrecoverable_mid_skipped(self):
        # Legacy row, filled, but shortfall missing -> mid unrecoverable.
        entries = [_report(1, 'AAA', 'BUY', avg_fill=10.0, shortfall=None)]
        out = compute_parity(entries)
        assert out['n_fills'] == 0 and out['n_skipped'] == 1

    def test_non_execution_events_ignored(self):
        entries = [
            {'seq': 1, 'event_type': 'auth_ok', 'payload': {}},
            _report(2, 'AAA', 'BUY', avg_fill=100.10, shortfall=0.10,
                    arrival_mid=100.0, asset_type='stock', quantity=100),
        ]
        out = compute_parity(entries)
        assert out['n_fills'] == 1


class TestComputeParitySummary:
    def _mixed(self):
        return [
            _report(1, 'AAA', 'BUY', avg_fill=100.10, shortfall=0.10,
                    arrival_mid=100.0, asset_type='stock', quantity=100),
            _report(2, 'AAA', 'BUY', avg_fill=2.07, shortfall=0.07,
                    arrival_mid=2.00, asset_type='call', quantity=3),
            _start(3, 'AAA', 'SELL', 'SELL', 50),
            _report(4, 'AAA', 'SELL', avg_fill=99.90, shortfall=0.10),
        ]

    def test_totals_and_breakdown(self):
        out = compute_parity(self._mixed(), slippage_bps=5.0, spread_pct=0.02,
                             min_option_haircut=0.05)
        s = out['summary']
        assert out['n_fills'] == 3
        # 5.00 (stock buy) + 6.00 (option) + 2.50 (legacy sell) = 13.50.
        assert s['total_slippage_drift'] == pytest.approx(13.50)
        assert s['worst_slippage_drift_per_unit'] == pytest.approx(0.05)
        assert out['commission_available'] is False
        assert s['total_actual_commission'] is None
        assert s['total_commission_drift'] is None
        assert set(s['by_asset_type']) == {'stock', 'call'}
        assert s['by_asset_type']['stock']['n_fills'] == 2
        assert s['by_asset_type']['call']['n_fills'] == 1
        assert s['n_priced'] == 3  # all three fills carry a quantity

    def test_n_priced_excludes_quantity_none_fills(self):
        # A legacy report with no recoverable quantity is counted in n_fills
        # but not in n_priced (and not in the dollar totals).
        entries = [_report(1, 'AAA', 'BUY', avg_fill=100.10, shortfall=0.10)]
        out = compute_parity(entries, slippage_bps=5.0)
        assert out['n_fills'] == 1
        assert out['summary']['n_priced'] == 0
        assert out['fills'][0]['quantity'] is None
        assert out['fills'][0]['slippage_drift_total'] is None

    def test_empty_entries(self):
        out = compute_parity([])
        assert out['n_fills'] == 0 and out['fills'] == []
        assert out['summary']['total_slippage_drift'] == 0.0
        assert out['summary']['by_asset_type'] == {}
