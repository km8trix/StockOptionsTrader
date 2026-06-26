"""
Market Data Handler - Fetches and manages price data using OpenBB ODP
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Union
from data.cache import OHLCVCache

logger = logging.getLogger(__name__)


class MarketDataHandler:
    """Fetches and manages market data using OpenBB Open Data Platform (ODP)"""

    # Keyed equity-historical providers in preference order: (provider name,
    # OpenBB credential attribute, environment variable). Each is used ONLY
    # when its key is configured, and tried BEFORE the free yfinance fallback
    # so a paid key actually serves the data (higher rate limit, better
    # reliability) instead of yfinance.
    _KEYED_PROVIDERS = (
        ('fmp', 'fmp_api_key', 'FMP_API_KEY'),
        ('tiingo', 'tiingo_token', 'TIINGO_TOKEN'),
        ('intrinio', 'intrinio_api_key', 'INTRINIO_API_KEY'),
    )

    # Index tickers route to OpenBB's index.price.historical (yfinance only —
    # indices are not equities and Tiingo carries none). Keys are matched
    # case-insensitively with an optional leading '^'; values are the symbol
    # OpenBB's index endpoint expects. NOTE: an index is not directly
    # tradeable — backtesting one simulates trading the level (analysis use).
    _INDEX_SYMBOLS = {
        'SPX': 'SPX', 'GSPC': 'SPX',        # S&P 500
        'NDX': 'NDX',                        # Nasdaq 100
        'IXIC': 'IXIC', 'COMP': 'IXIC',      # Nasdaq Composite
        'DJI': 'DJI',                        # Dow Jones Industrial Average
        'RUT': 'RUT',                        # Russell 2000
        'VIX': 'VIX',                        # CBOE Volatility Index
        'NYA': 'NYA',                        # NYSE Composite
    }

    def __init__(self, cache: Union[OHLCVCache, bool, None] = None):
        """
        Args:
            cache: SQLite-backed OHLCV cache. None (default) constructs an
                OHLCVCache lazily on the first fetch; pass an OHLCVCache to
                share/configure one; pass False to disable persistent caching
                (pre-Phase-2 behavior).
        """
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.cache: Dict[str, pd.DataFrame] = {}
        # Provider order for equity.price.historical: any keyed provider whose
        # credential is configured comes first (reliable, higher rate limit),
        # then free keyless yfinance as the always-available fallback. Keyed
        # providers without a key are omitted (they would only fail with a
        # missing-credential error and add log noise). NOTE:
        # cboe/tmx/polygon/alpha_vantage/tradier are NOT valid equity-
        # historical providers in the installed OpenBB build.
        self.providers = self._resolve_providers()
        self._sqlite_cache: Optional[OHLCVCache] = (
            cache if isinstance(cache, OHLCVCache) else None
        )
        self._sqlite_cache_enabled = cache is not False
        # Per-symbol provenance of the most recent fetch (see
        # get_last_fetch_info for the contract).
        self._last_fetch_info: Dict[str, dict] = {}

    def _get_sqlite_cache(self) -> Optional[OHLCVCache]:
        """Return the persistent cache, constructing the default lazily."""
        if not self._sqlite_cache_enabled:
            return None
        if self._sqlite_cache is None:
            try:
                self._sqlite_cache = OHLCVCache()
            except Exception as e:
                logger.warning("OHLCV cache unavailable; caching disabled: %s", e)
                self._sqlite_cache_enabled = False
                return None
        return self._sqlite_cache

    def _resolve_providers(self) -> list:
        """Keyed providers (with a configured credential) first, then the free
        yfinance fallback. Recomputed from os.environ so a key added to .env
        takes effect on the next handler construction."""
        keyed = [name for (name, _attr, env) in self._KEYED_PROVIDERS
                 if os.environ.get(env)]
        return keyed + ['yfinance']

    def _index_symbol(self, symbol: str) -> Optional[str]:
        """Return the OpenBB index-endpoint symbol for a known index ticker
        (case-insensitive, optional leading '^'), or None for an equity/ETF.
        Used to route indices to obb.index.price.historical."""
        if not symbol:
            return None
        key = symbol.strip().lstrip('^').upper()
        return self._INDEX_SYMBOLS.get(key)

    def _apply_credentials(self, obb) -> None:
        """Push any configured provider keys from the environment into OpenBB's
        runtime credentials, so a key in .env is used without also editing
        OpenBB's user_settings.json. Best-effort: a failure to set one key
        never blocks the free yfinance path."""
        for name, attr, env in self._KEYED_PROVIDERS:
            value = os.environ.get(env)
            if not value:
                continue
            try:
                setattr(obb.user.credentials, attr, value)
            except Exception as e:  # noqa: BLE001 — credential plumbing is best-effort
                logger.warning("Could not apply %s credential to OpenBB: %s",
                               name, e)

    @staticmethod
    def _ensure_ssl_certs() -> None:
        """Point Python's TLS at the certifi CA bundle if no cert file is
        configured. python.org macOS builds ship without system CA certs, so
        OpenBB providers that use aiohttp (e.g. Tiingo) otherwise fail with
        'certificate verify failed'. Idempotent and non-destructive: respects
        an SSL_CERT_FILE the user/system already set."""
        if os.environ.get('SSL_CERT_FILE'):
            return
        try:
            import certifi
            bundle = certifi.where()
            os.environ.setdefault('SSL_CERT_FILE', bundle)
            os.environ.setdefault('REQUESTS_CA_BUNDLE', bundle)
        except Exception as e:  # noqa: BLE001 — cert plumbing is best-effort
            logger.warning("Could not configure SSL_CERT_FILE from certifi: %s", e)

    def _get_openbb(self):
        """Import OpenBB only when it is needed; initialization can touch user-level files."""
        try:
            self._ensure_ssl_certs()
            from openbb import obb
            self._apply_credentials(obb)
            return obb
        except Exception as e:
            logger.warning("OpenBB unavailable: %s", e)
            return None

    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical stock data with structural safeguards.

        Lookup order: in-memory dict -> persistent OHLCVCache -> OpenBB ODP
        providers. Provider successes are written back to the persistent
        cache; provenance is recorded for get_last_fetch_info either way.
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"
        failures: list = []
        info = {
            'provider': None,
            'from_cache': False,
            'fetched_at': datetime.now().isoformat(),
            'failures': failures,
            'start_date': start_date,
            'end_date': end_date,
        }
        try:
            if cache_key in self.cache:
                info['from_cache'] = True
                self._last_fetch_info[symbol] = info
                return self.cache[cache_key]

            sqlite_cache = self._get_sqlite_cache()
            if sqlite_cache is not None:
                cached = None
                try:
                    cached = sqlite_cache.get(symbol, start_date, end_date)
                except Exception as e:
                    logger.warning("OHLCV cache read failed for %s: %s", symbol, e)
                if cached is not None:
                    logger.info("Served %s %s..%s from OHLCV cache (%d rows)",
                                symbol, start_date, end_date, len(cached))
                    info['from_cache'] = True
                    self._last_fetch_info[symbol] = info
                    self.cache[cache_key] = cached
                    self.stock_data[symbol] = cached
                    return cached

            data = None
            used_provider = None
            obb = self._get_openbb()

            # Index symbols (SPX, ^GSPC, NDX, ...) are not equities and are
            # absent from Tiingo — route them to OpenBB's index endpoint via
            # yfinance. Equities/ETFs use the normal keyed-provider chain.
            index_symbol = self._index_symbol(symbol)
            if index_symbol is not None:
                attempts = [('yfinance', 'index', index_symbol)]
            else:
                attempts = [(p, 'equity', symbol) for p in self.providers]

            for provider, kind, fetch_symbol in (
                    attempts if obb is not None else []):
                try:
                    endpoint = (obb.index.price.historical if kind == 'index'
                                else obb.equity.price.historical)
                    result = endpoint(
                        symbol=fetch_symbol,
                        start_date=start_date,
                        end_date=end_date,
                        provider=provider,
                    )

                    if result is None or not hasattr(result, 'results'):
                        failures.append({'provider': provider,
                                         'error': 'no results returned'})
                        continue

                    # Convert OBB results to DataFrame
                    data_list = []
                    for item in result.results:
                        data_list.append({
                            'date': item.date,
                            'open': float(item.open) if item.open else None,
                            'high': float(item.high) if item.high else None,
                            'low': float(item.low) if item.low else None,
                            'close': float(item.close) if item.close else None,
                            'volume': float(item.volume) if item.volume else None,
                        })

                    if data_list:
                        data = pd.DataFrame(data_list)
                        used_provider = provider
                        logger.info(
                            "Fetched %s from OpenBB %s provider %s (%d rows)",
                            symbol, kind, provider, len(data_list)
                        )
                        break
                    failures.append({'provider': provider,
                                     'error': 'empty result set'})

                except Exception as e:
                    logger.warning(
                        "OpenBB provider %s failed for %s: %s", provider, symbol, e
                    )
                    failures.append({'provider': provider, 'error': str(e)})
                    continue

            # If all OpenBB providers failed, return no data.
            if data is None or data.empty:
                self._last_fetch_info[symbol] = info
                return self._empty_data(symbol)

            # Process the data. The index unit is canonicalized to 'us' so
            # provider-served and cache-served frames are indistinguishable
            # (providers may yield date objects, which infer as 's').
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            data.index = data.index.as_unit('us')

            # Standardize column names to lowercase
            data.columns = [str(col).lower() for col in data.columns]

            # Select only expected columns and handle missing ones
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in expected_columns if col in data.columns]

            if not available_columns:
                self._last_fetch_info[symbol] = info
                return pd.DataFrame()

            data = data[available_columns]

            info['provider'] = used_provider
            self._last_fetch_info[symbol] = info

            assert used_provider is not None  # a non-empty fetch has a provider
            if sqlite_cache is not None:
                try:
                    sqlite_cache.store(symbol, data, used_provider,
                                       start_date, end_date)
                except Exception as e:
                    logger.warning("OHLCV cache write failed for %s: %s", symbol, e)

            self.cache[cache_key] = data
            self.stock_data[symbol] = data
            return data

        except Exception as e:
            logger.warning("Error fetching data for %s: %s", symbol, e)
            self._last_fetch_info[symbol] = info
            return self._empty_data(symbol)

    def get_last_fetch_info(self, symbol: str) -> Optional[dict]:
        """Provenance for the most recent fetch_stock_data call for symbol.

        Returns a dict with exactly these keys (the shared interface
        contract consumed by the GUI routes):
            'provider': str | None — provider that served the data, None if
                served from cache (or if every provider failed),
            'from_cache': bool,
            'fetched_at': str (ISO-8601),
            'failures': list of {'provider': str, 'error': str},
            'start_date': str,
            'end_date': str.
        Returns None if the symbol was never fetched in this process.
        """
        return self._last_fetch_info.get(symbol)

    def _empty_data(self, symbol: str) -> pd.DataFrame:
        """Return a properly-SHAPED empty OHLCV frame when no data is available.

        A bare pd.DataFrame() has zero columns, so a failure-path frame makes
        downstream data['close'] / .iloc[-1] raise KeyError/IndexError — a
        hollow frame silently poisons indicator math. This empty frame still
        carries the OHLCV columns and a named DatetimeIndex, so `.empty` stays
        True (callers and the cache still detect "no data") while column
        access remains safe.
        """
        logger.warning("No OpenBB data available for %s", symbol)
        return pd.DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume'],
            index=pd.DatetimeIndex([], name='date'),
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        data['macd'] = data['ema_12'] - data['ema_26']
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['signal']
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
        
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift()),
                abs(data['low'] - data['close'].shift())
            )
        )
        if len(data) > 0:
            # Wilder convention: there is no prior close on the first bar, so
            # TR_1 = high_1 - low_1 (np.maximum would propagate the NaN from
            # close.shift() otherwise). ATR is then valid from bar 14.
            data.iloc[0, data.columns.get_loc('tr')] = (
                data['high'].iloc[0] - data['low'].iloc[0]
            )
        data['atr'] = data['tr'].rolling(window=14).mean()
        data['volume_sma'] = data['volume'].rolling(window=20).mean()

        return data
