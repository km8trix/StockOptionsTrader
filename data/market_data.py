"""
Market Data Handler - Fetches and manages price data using OpenBB ODP
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Optional
from core.models import Asset, AssetType


class MarketDataHandler:
    """Fetches and manages market data using OpenBB Open Data Platform (ODP)"""
    
    def __init__(self):
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

    def _get_openbb(self):
        """Import OpenBB only when it is needed; initialization can touch user-level files."""
        try:
            from openbb import obb
            return obb
        except Exception as e:
            print(f"OpenBB unavailable: {e}")
            return None

    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical stock data from OpenBB ODP with structural safeguards
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"
        try:
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            data = None
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
                        print(f"Successfully fetched {symbol} from OpenBB ODP provider: {provider}")
                        break
                        
                except Exception as e:
                    continue
            
            # If all OpenBB providers failed, return no data.
            if data is None or data.empty:
                return self._empty_data(symbol)
            
            # Process the data
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            # Standardize column names to lowercase
            data.columns = [str(col).lower() for col in data.columns]
            
            # Select only expected columns and handle missing ones
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in expected_columns if col in data.columns]
            
            if not available_columns:
                return pd.DataFrame()
            
            data = data[available_columns]
            
            self.cache[cache_key] = data
            self.stock_data[symbol] = data
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return self._empty_data(symbol)
    
    def _empty_data(self, symbol: str) -> pd.DataFrame:
        """Return an empty result when OpenBB cannot provide data."""
        print(f"No OpenBB data available for {symbol}")
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
