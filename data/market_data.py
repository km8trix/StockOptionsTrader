"""
Market Data Handler - Fetches and manages price data using OpenBB ODP
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Union
from core.models import Asset, AssetType
from data.cache import OHLCVCache

logger = logging.getLogger(__name__)


class MarketDataHandler:
    """Fetches and manages market data using OpenBB Open Data Platform (ODP)"""

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
        # OpenBB providers for equity historical data. Some providers require API keys.
        self.providers = [
            'cboe',
            'tmx',
            'fmp',
            'intrinio',
            'polygon',
            'tiingo',
            'alpha_vantage',
            'tradier',
        ]
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

    def _get_openbb(self):
        """Import OpenBB only when it is needed; initialization can touch user-level files."""
        try:
            from openbb import obb
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

            # Try multiple ODP providers
            for provider in self.providers if obb is not None else []:
                try:
                    result = obb.equity.price.historical(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        provider=provider
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
                            "Fetched %s from OpenBB ODP provider %s (%d rows)",
                            symbol, provider, len(data_list)
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
        """Return an empty result when OpenBB cannot provide data."""
        logger.warning("No OpenBB data available for %s", symbol)
        return pd.DataFrame()
    
    def get_current_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get price for a specific date"""
        if symbol not in self.stock_data:
            return None
        
        data = self.stock_data[symbol]
        date_str = date.strftime('%Y-%m-%d')

        if date_str in data.index:
            raw_value = data.at[date_str, 'close']
            return float(raw_value)
        return None
    
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
    
    def estimate_option_price(self, stock_price: float, strike: float, 
                             time_to_expiry: float, volatility: float,
                             option_type: str = 'call', rate: float = 0.05) -> float:
        """Estimate option price using Black-Scholes"""
        from scipy.stats import norm
        
        d1 = (np.log(stock_price / strike) + (rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type.lower() == 'call':
            price = stock_price * norm.cdf(d1) - strike * np.exp(-rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * np.exp(-rate * time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
        
        return max(price, 0)
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate historical volatility"""
        returns = data['close'].pct_change()
        return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
