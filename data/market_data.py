"""
Market Data Handler - Fetches and manages price data using OpenBB ODP
"""

import pandas as pd
import numpy as np
from openbb import obb
from datetime import date, datetime, timedelta
from typing import cast, Dict, Optional, Tuple, Any
from core.models import Asset, AssetType


class MarketDataHandler:
    """Fetches and manages market data using OpenBB Open Data Platform (ODP)"""
    
    def __init__(self):
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.cache: Dict[str, pd.DataFrame] = {}
        # OpenBB ODP providers available for equity historical data
        self.providers = ['fmp', 'intrinio', 'polygon', 'tiingo']

    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical stock data from OpenBB ODP with structural safeguards
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        try:
            cache_key = f"{symbol}_{start_date}_{end_date}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            data = None
            last_error = None
            
            # Try multiple ODP providers
            for provider in self.providers:
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
                    last_error = e
                    continue
            
            # If ODP failed, use fallback
            if data is None or data.empty:
                print(f"ODP fetch failed for {symbol}, attempting fallback")
                return self._fetch_with_fallback(symbol, start_date, end_date)
            
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
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return self._fetch_with_fallback(symbol, start_date, end_date)
    
    def _fetch_with_fallback(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fallback method using yfinance if ODP providers are unavailable
        Ensures backward compatibility when ODP/APIs are not accessible
        """
        try:
            import yfinance as yf
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return pd.DataFrame()
            
            # Flatten multi-index if needed
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            data.columns = [str(col).lower() for col in data.columns]
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in expected_columns if col in data.columns]
            data = data[available_columns]
            
            print(f"Fetched {symbol} using yfinance fallback")
            return data
        except Exception as e:
            print(f"Fallback fetch also failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get price for a specific date"""
        if symbol not in self.stock_data:
            return None
        
        data = self.stock_data[symbol]
        date_str = date.strftime('%Y-%m-%d')

        if date_str in data.index:
            raw_value = data.at[date_str, 'close']
            return cast(float, raw_value)
        return None
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Simple Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['signal']
        
        # RSI (Relative Strength Index)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
        
        # Average True Range
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift()),
                abs(data['low'] - data['close'].shift())
            )
        )
        data['atr'] = data['tr'].rolling(window=14).mean()
        
        # Volume indicators
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


