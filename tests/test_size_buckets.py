"""Tests for point-in-time size bucketing (data/size_buckets)."""

import numpy as np

from data.size_buckets import pit_marketcaps, size_buckets


def test_terciles_split_evenly_by_cap():
    caps = {f'N{i}': float(i + 1) for i in range(9)}  # 1..9, distinct positive
    b = size_buckets(caps, n=3)
    assert b['N0'] == 0 and b['N1'] == 0 and b['N2'] == 0   # smallest third
    assert b['N3'] == 1 and b['N4'] == 1 and b['N5'] == 1
    assert b['N6'] == 2 and b['N7'] == 2 and b['N8'] == 2   # largest third


def test_ordering_is_by_cap_not_insertion():
    b = size_buckets({'BIG': 1e10, 'SMALL': 1e6, 'MID': 1e8}, n=3)
    assert (b['SMALL'], b['MID'], b['BIG']) == (0, 1, 2)


def test_missing_and_bad_caps_dropped():
    b = size_buckets({'A': 1.0, 'B': None, 'C': np.nan, 'D': -5.0, 'E': 2.0,
                      'F': 3.0}, n=3)
    assert set(b) == {'A', 'E', 'F'}          # B/C/D dropped
    assert (b['A'], b['E'], b['F']) == (0, 1, 2)


def test_too_few_names_returns_empty():
    assert size_buckets({'A': 1.0, 'B': 2.0}, n=3) == {}


class _StubProvider:
    def __init__(self, caps):
        self._caps = caps

    def daily_metric(self, ticker, date, field=None):
        mc = self._caps.get(ticker)
        return None if mc is None else mc


def test_pit_marketcaps_uses_daily_metric_and_omits_missing():
    prov = _StubProvider({'A': 1e9, 'B': 5e8, 'C': None})
    caps = pit_marketcaps(prov, ['A', 'B', 'C', 'D'], '2020-06-30')
    assert caps == {'A': 1e9, 'B': 5e8}       # C is None, D absent -> omitted
