"""Symbol universe: core/sector ETFs, a large-cap starting list, ETF
holdings lookup with a static fallback, and rate-limited batch fetching.

LARGE_CAP_100 and STATIC_HOLDINGS are hand-curated starting universes for
scanning and backtesting — they are NOT reproductions of any index or
fund and will drift from official constituents over time.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Sequence

logger = logging.getLogger(__name__)

#: Broad-market core ETFs.
CORE_ETFS = ['SPY', 'QQQ', 'IWM', 'DIA']

#: The 11 SPDR sector ETFs.
SECTOR_ETFS = [
    'XLB',   # Materials
    'XLC',   # Communication Services
    'XLE',   # Energy
    'XLF',   # Financials
    'XLI',   # Industrials
    'XLK',   # Technology
    'XLP',   # Consumer Staples
    'XLRE',  # Real Estate
    'XLU',   # Utilities
    'XLV',   # Health Care
    'XLY',   # Consumer Discretionary
]

#: Hand-curated, alphabetized list of ~100 well-known US mega/large-cap
#: tickers. A starting universe for scanners — not an index reproduction.
LARGE_CAP_100 = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADP', 'AMAT', 'AMD',
    'AMGN', 'AMT', 'AMZN', 'AVGO', 'AXP', 'BA', 'BAC', 'BKNG', 'BLK',
    'BMY', 'C', 'CAT', 'CB', 'CI', 'CMCSA', 'COP', 'COST', 'CRM', 'CSCO',
    'CVS', 'CVX', 'DE', 'DHR', 'DIS', 'DUK', 'ELV', 'EOG', 'ETN', 'FDX',
    'GE', 'GILD', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'ICE', 'INTC',
    'INTU', 'ISRG', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW',
    'LRCX', 'MA', 'MCD', 'MDT', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT',
    'NEE', 'NFLX', 'NKE', 'NOW', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG',
    'PLD', 'PM', 'QCOM', 'REGN', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPGI',
    'T', 'TJX', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'V',
    'VRTX', 'VZ', 'WFC', 'WM', 'WMT', 'XOM',
]

#: Hand-curated top ~10 holdings per ETF, used when OpenBB holdings lookup
#: fails. Approximate snapshots — not official constituent lists.
STATIC_HOLDINGS: Dict[str, List[str]] = {
    'SPY': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'AVGO',
            'TSLA', 'LLY', 'JPM'],
    'QQQ': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'AVGO', 'GOOGL',
            'TSLA', 'COST', 'NFLX'],
    'XLB': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'CTVA', 'DOW',
            'DD', 'MLM'],
    'XLC': ['META', 'GOOGL', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS',
            'EA', 'CHTR'],
    'XLE': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'WMB',
            'OKE', 'VLO'],
    'XLF': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'SPGI', 'MS', 'AXP',
            'C'],
    'XLI': ['GE', 'CAT', 'RTX', 'UNP', 'HON', 'ETN', 'UPS', 'BA', 'DE',
            'LMT'],
    'XLK': ['MSFT', 'AAPL', 'NVDA', 'AVGO', 'CRM', 'ORCL', 'ADBE',
            'AMD', 'CSCO', 'ACN'],
    'XLP': ['PG', 'COST', 'KO', 'PEP', 'WMT', 'PM', 'MDLZ', 'MO', 'CL',
            'TGT'],
    'XLRE': ['PLD', 'AMT', 'EQIX', 'WELL', 'SPG', 'PSA', 'O', 'DLR',
             'CCI', 'CBRE'],
    'XLU': ['NEE', 'SO', 'DUK', 'CEG', 'SRE', 'AEP', 'D', 'PCG', 'EXC',
            'XEL'],
    'XLV': ['LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'AMGN',
            'DHR', 'PFE'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'LOW', 'BKNG', 'TJX', 'SBUX',
            'NKE', 'CMG'],
}


def _get_openbb():
    """Import OpenBB only when it is needed; initialization can touch user-level files."""
    try:
        from openbb import obb
        return obb
    except Exception as e:
        logger.warning("OpenBB unavailable: %s", e)
        return None


def get_etf_holdings(etf_symbol: str) -> List[str]:
    """Constituent symbols for an ETF.

    Tries obb.etf.holdings (lazy import, every failure logged); on any
    failure falls back to STATIC_HOLDINGS. Unknown ETFs with no provider
    data return an empty list.
    """
    symbol = etf_symbol.upper()

    obb = _get_openbb()
    if obb is not None:
        try:
            result = obb.etf.holdings(symbol=symbol)
            if result is not None and hasattr(result, 'results'):
                symbols = []
                for item in result.results:
                    ticker = getattr(item, 'symbol', None)
                    if ticker:
                        symbols.append(str(ticker).upper())
                if symbols:
                    logger.info("Fetched %d holdings for %s from OpenBB",
                                len(symbols), symbol)
                    return symbols
            logger.warning("OpenBB returned no holdings for %s", symbol)
        except Exception as e:
            logger.warning("OpenBB etf.holdings failed for %s: %s", symbol, e)

    fallback = STATIC_HOLDINGS.get(symbol, [])
    if fallback:
        logger.info("Using static holdings fallback for %s (%d symbols)",
                    symbol, len(fallback))
    else:
        logger.warning("No holdings available for %s (no static fallback)",
                       symbol)
    return list(fallback)


def batch_fetch(handler, symbols: Sequence[str], start_date: str,
                end_date: str, delay_seconds: float = 0.2,
                sleep_fn: Callable[[float], None] = time.sleep) -> Dict[str, str]:
    """Fetch OHLCV for each symbol with per-symbol failure isolation.

    One symbol's failure never aborts the batch. Returns a status per
    symbol: 'ok' | 'empty' | 'error: <msg>'. Sleeps delay_seconds between
    PROVIDER fetches only — cache hits (per handler.get_last_fetch_info)
    skip the delay, as does the final symbol. sleep_fn is injectable so
    tests never actually sleep.

    Args:
        handler: a MarketDataHandler (anything with fetch_stock_data and,
            optionally, get_last_fetch_info).
    """
    results: Dict[str, str] = {}
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        try:
            df = handler.fetch_stock_data(symbol, start_date, end_date)
            if df is None or df.empty:
                results[symbol] = 'empty'
            else:
                results[symbol] = 'ok'
        except Exception as e:
            logger.warning("batch_fetch: %s failed: %s", symbol, e)
            results[symbol] = f'error: {e}'

        if (i + 1) % 10 == 0:
            logger.info("batch_fetch progress: %d/%d symbols", i + 1, total)

        if i + 1 >= total:
            break  # never sleep after the final symbol

        # Only rate-limit actual provider traffic: a cache hit costs nothing.
        from_cache = False
        get_info = getattr(handler, 'get_last_fetch_info', None)
        if callable(get_info):
            try:
                info = get_info(symbol)
                from_cache = bool(info and info.get('from_cache'))
            except Exception:
                logger.warning("batch_fetch: get_last_fetch_info failed for %s",
                               symbol, exc_info=True)
        if not from_cache:
            sleep_fn(delay_seconds)

    logger.info("batch_fetch complete: %d/%d ok",
                sum(1 for s in results.values() if s == 'ok'), total)
    return results
