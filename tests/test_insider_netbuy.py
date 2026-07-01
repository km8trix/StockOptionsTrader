"""Hermetic tests for InsiderNetBuyDesk._alpha_scores.

An in-memory warehouse (synthetic daily marketcaps + sf2 filings) drives the
desk directly. Asserts: only the desk's PIT size band is scored, buyers score
long / sellers short, and scores are cached within a calendar month.
"""

import duckdb
import pandas as pd
import pytest

from data.pit_warehouse import PitWarehouse
from desks.insider_netbuy import InsiderNetBuyDesk


def _write(path, cols, rows):
    values = ",".join("(" + ",".join(r) + ")" for r in rows)
    names = ",".join(cols)
    duckdb.connect().execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS t({names})) "
        f"TO '{path}' (FORMAT PARQUET)")


@pytest.fixture
def warehouse(tmp_path):
    # 12 names, marketcap 1..12 -> terciles of 4; band 'small' == ranks 4-7 (N4..N7)
    daily_cols = ['ticker', 'date', 'marketcap', 'pe', 'pb', 'ps', 'ev',
                  'evebit', 'evebitda']
    _write(tmp_path / "daily.parquet", daily_cols, [
        (f"'N{i}'", "DATE '2020-06-30'", str(float(i + 1)), "0", "0", "0", "0",
         "0", "0") for i in range(12)])
    # insider filings only for the small band; N4/N5 buy, N6/N7 sell
    sf2_cols = ['ticker', 'filingdate', 'transactioncode', 'transactionshares',
                'transactionvalue']
    _write(tmp_path / "sf2.parquet", sf2_cols, [
        ("'N4'", "DATE '2020-06-15'", "'P'", "1000", "200000.0"),
        ("'N5'", "DATE '2020-06-15'", "'P'", "500", "100000.0"),
        ("'N6'", "DATE '2020-06-15'", "'S'", "-800", "150000.0"),
        ("'N7'", "DATE '2020-06-15'", "'S'", "-300", "50000.0"),
    ])
    return PitWarehouse(str(tmp_path))


def _all_data():
    df = pd.DataFrame({'close': [1.0]}, index=[pd.Timestamp('2020-06-30')])
    return {f'N{i}': df for i in range(12)}


def test_scores_only_the_band_with_correct_signs(warehouse):
    desk = InsiderNetBuyDesk('small', provider=warehouse)
    scores = desk._alpha_scores(_all_data(), pd.Timestamp('2020-06-30'))
    assert set(scores) == {'N4', 'N5', 'N6', 'N7'}   # small tercile only
    # score is SIGN only (magnitude doesn't predict): +1 buyers, -1 sellers
    assert scores['N4'] == 1.0 and scores['N5'] == 1.0
    assert scores['N6'] == -1.0 and scores['N7'] == -1.0


def test_wrong_band_is_empty(warehouse):
    # micro band (N0..N3) has no insider filings -> below min_scored -> None
    desk = InsiderNetBuyDesk('micro', provider=warehouse)
    assert desk._alpha_scores(_all_data(), pd.Timestamp('2020-06-30')) is None


def test_scores_cached_within_month(warehouse):
    desk = InsiderNetBuyDesk('small', provider=warehouse)
    first = desk._alpha_scores(_all_data(), pd.Timestamp('2020-06-30'))
    # a different day in the SAME month returns the identical cached object
    again = desk._alpha_scores(_all_data(), pd.Timestamp('2020-06-05'))
    assert again is first


def test_long_only_suppresses_shorts(warehouse):
    from portfolio.manager import PortfolioManager
    day = pd.Timestamp('2020-06-30')
    lo = InsiderNetBuyDesk('small', provider=warehouse, long_only=True)
    lo_intents = lo.generate_intents(_all_data(), day, PortfolioManager(100_000))
    assert 'SHORT' not in {i.action for i in lo_intents}
    assert any(i.action == 'BUY' for i in lo_intents)
    # the default L/S desk DOES open a short on the same book (control)
    ls = InsiderNetBuyDesk('small', provider=warehouse)
    ls_intents = ls.generate_intents(_all_data(), day, PortfolioManager(100_000))
    assert any(i.action == 'SHORT' for i in ls_intents)
