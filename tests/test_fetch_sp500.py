"""Test scripts/fetch_sp500.clean_symbols — the pure transform, no network."""

import importlib.util
from pathlib import Path

import pandas as pd

_path = Path(__file__).resolve().parent.parent / 'scripts' / 'fetch_sp500.py'
_spec = importlib.util.spec_from_file_location('fetch_sp500', _path)
fetch_sp500 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_sp500)


def test_clean_symbols_normalizes_dots_and_drops_blanks():
    df = pd.DataFrame({'Symbol': ['AAPL', 'BRK.B', 'BF.B', None, ' MMM ']})
    # dots -> hyphens (EDGAR shape), None dropped, whitespace stripped.
    assert fetch_sp500.clean_symbols(df) == ['AAPL', 'BRK-B', 'BF-B', 'MMM']
