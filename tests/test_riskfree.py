"""Tests for data.riskfree — the dated DTB3 loader and forward-fill lookup.

Hermetic: the parsing tests run on a small synthetic CSV written to tmp_path
(both FRED missing-marker styles: '.' and an empty field); the vendored-file
test reads the committed data/vendored/dtb3.csv (no network).
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.riskfree import DTB3_CSV, load_dtb3, rate_asof

SYNTHETIC = """observation_date,DTB3
2015-01-02,0.05
2015-01-05,.
2015-01-06,
2015-01-09,0.08
2023-06-01,5.25
"""


@pytest.fixture
def series(tmp_path):
    path = tmp_path / 'dtb3.csv'
    path.write_text(SYNTHETIC)
    return load_dtb3(path)


class TestLoadDtb3:
    def test_percent_to_annualized_decimal(self, series):
        # 0.05% -> 0.0005, 5.25% -> 0.0525: /100, never /1 or /10000.
        assert series[pd.Timestamp('2015-01-02')] == pytest.approx(0.0005)
        assert series[pd.Timestamp('2023-06-01')] == pytest.approx(0.0525)

    def test_missing_markers_dropped_both_styles(self, series):
        # '.' (FRED fredgraph style) and the empty field (FRED download
        # style, what the vendored file uses) both drop, leaving 3 rows.
        assert len(series) == 3
        assert pd.Timestamp('2015-01-05') not in series.index
        assert pd.Timestamp('2015-01-06') not in series.index

    def test_index_is_sorted_datetime(self, series):
        assert series.index.is_monotonic_increasing
        assert isinstance(series.index, pd.DatetimeIndex)


class TestRateAsof:
    def test_exact_observation(self, series):
        assert rate_asof(series, '2015-01-09') == pytest.approx(0.0008)

    def test_weekend_and_gap_forward_fill(self, series):
        # Sat 2015-01-03 and the missing 01-05/01-06 prints all carry the
        # last observation BEFORE them (0.05% from 01-02) — never the
        # 01-09 print from the future.
        for day in ('2015-01-03', '2015-01-04', '2015-01-05', '2015-01-07'):
            assert rate_asof(series, day) == pytest.approx(0.0005)

    def test_long_carry_until_next_print(self, series):
        # Years between prints still forward-fill (synthetic gap 2015->2023).
        assert rate_asof(series, '2020-06-15') == pytest.approx(0.0008)

    def test_before_first_observation_is_zero(self, series):
        # Pre-history: no data means accrue nothing, never back-fill.
        assert rate_asof(series, '2014-12-31') == 0.0
        assert rate_asof(series, '1990-01-01') == 0.0

    def test_after_last_observation_carries_last(self, series):
        assert rate_asof(series, '2030-01-01') == pytest.approx(0.0525)


class TestVendoredFile:
    def test_vendored_dtb3_loads_and_is_sane(self):
        # The committed FRED file: parses, decimal-annualized (Volcker-era
        # peak ~17% must read 0.17, not 17.0; 2015's briefly NEGATIVE
        # secondary-market prints (-0.05%) read -0.0005, not -0.05), and
        # covers the 2015-2024 gate window with dated regime variation.
        series = load_dtb3(DTB3_CSV)
        assert len(series) > 15_000
        assert series.index[0] == pd.Timestamp('1954-01-04')
        assert float(series.max()) < 0.20
        assert float(series.min()) > -0.01
        assert rate_asof(series, '2016-06-01') < 0.005   # ZIRP era
        assert rate_asof(series, '2023-09-01') > 0.045   # hiking era
