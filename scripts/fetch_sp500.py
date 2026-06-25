#!/usr/bin/env python
"""Write the current S&P 500 tickers (one per line) for the earnings ingest.

Scrapes the Wikipedia constituents table (pandas — already a dep, no new
package, no API key) and normalizes share-class dots to hyphens so EDGAR's
CIK lookup matches (e.g. BRK.B -> BRK-B). NETWORK, operator-run:

    python scripts/fetch_sp500.py            # -> sp500.txt
    python scripts/fetch_sp500.py univ.txt   # -> univ.txt
    SEC_USER_AGENT='You you@email.com' \\
        python -m data.earnings_cache --symbols-file sp500.txt
"""

from __future__ import annotations

import sys

import pandas as pd

WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'


def clean_symbols(table: pd.DataFrame) -> list[str]:
    """Symbol column -> EDGAR-shaped tickers (dots to hyphens, blanks dropped)."""
    symbols = table['Symbol'].dropna().astype(str).str.strip()
    symbols = symbols.str.replace('.', '-', regex=False)
    return [s for s in symbols if s]


def main(out_path: str = 'sp500.txt') -> None:
    symbols = clean_symbols(pd.read_html(WIKI_URL)[0])
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(symbols) + '\n')
    print(f"Wrote {len(symbols)} S&P 500 tickers to {out_path}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'sp500.txt')
