"""
Base Strategy Class - Framework for implementing trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from core.models import Asset, AssetType, OrderType, Order
from data.market_data import MarketDataHandler


class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.market_data = MarketDataHandler()
        self.signals: Dict[Asset, str] = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        """
        Generate trading signals: 'BUY', 'SELL', or 'HOLD'
        """
        pass
    
    def analyze(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Analyze data and generate signals"""
        data = self.market_data.fetch_stock_data(symbol, start_date, end_date)
        if data.empty:
            return data
        
        data = self.market_data.calculate_indicators(data)
        asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
        
        for idx in range(1, len(data)):
            signal = self.generate_signals(data.iloc[:idx+1], asset)
            self.signals[asset] = signal
        
        return data


class MomentumStrategy(Strategy):
    """
    Momentum-based strategy using technical indicators
    - BUY: When price crosses above SMA50 with strong RSI
    - SELL: When RSI > 70 or price crosses below SMA20
    """
    
    def __init__(self):
        super().__init__("Momentum Strategy")
        self.rsi_overbought = 70
        self.rsi_oversold = 30
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        if len(data) < 50:
            return 'HOLD'
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Check MACD crossover
        macd_bullish = (current['macd'] > current['signal']) and (prev['macd'] <= prev['signal'])
        macd_bearish = (current['macd'] < current['signal']) and (prev['macd'] >= prev['signal'])
        
        # Check RSI
        rsi_bullish = current['rsi'] < self.rsi_oversold
        rsi_bearish = current['rsi'] > self.rsi_overbought
        
        # Check price position relative to moving averages
        price_above_sma20 = current['close'] > current['sma_20']
        price_above_sma50 = current['close'] > current['sma_50']
        
        if macd_bullish and price_above_sma50:
            return 'BUY'
        elif macd_bearish or rsi_bearish:
            return 'SELL'
        
        return 'HOLD'


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands
    - BUY: When price touches lower Bollinger Band with oversold RSI
    - SELL: When price touches upper Bollinger Band with overbought RSI
    """
    
    def __init__(self):
        super().__init__("Mean Reversion Strategy")
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        if len(data) < 50:
            return 'HOLD'
        
        current = data.iloc[-1]
        
        # Check Bollinger Band positions
        at_lower_band = current['close'] <= current['bb_lower']
        at_upper_band = current['close'] >= current['bb_upper']
        
        # Check RSI
        rsi_oversold = current['rsi'] < 30
        rsi_overbought = current['rsi'] > 70
        
        if at_lower_band and rsi_oversold:
            return 'BUY'
        elif at_upper_band and rsi_overbought:
            return 'SELL'
        
        return 'HOLD'


class StatisticalArbitrageStrategy(Strategy):
    """
    Statistical arbitrage using Z-score and correlation
    - Identifies mean deviations and potential reversals
    """
    
    def __init__(self, z_score_threshold: float = 2.0):
        super().__init__("Statistical Arbitrage Strategy")
        self.z_score_threshold = z_score_threshold
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        if len(data) < 50:
            return 'HOLD'
        
        current = data.iloc[-1]
        sma = data['close'].rolling(window=20).mean()
        std = data['close'].rolling(window=20).std()
        
        z_score = (current['close'] - sma.iloc[-1]) / std.iloc[-1]
        
        if z_score < -self.z_score_threshold:
            return 'BUY'
        elif z_score > self.z_score_threshold:
            return 'SELL'
        
        return 'HOLD'
