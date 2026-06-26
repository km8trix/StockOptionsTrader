"""
Base Strategy Class - Framework for implementing trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
from core.models import Asset


class Strategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        """
        Generate trading signals: 'BUY', 'SELL', or 'HOLD'
        """
        pass


class MomentumStrategy(Strategy):
    """
    Momentum-based strategy using technical indicators
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
        
        macd_bullish = (current['macd'] > current['signal']) and (prev['macd'] <= prev['signal'])
        macd_bearish = (current['macd'] < current['signal']) and (prev['macd'] >= prev['signal'])
        
        rsi_bullish = current['rsi'] < self.rsi_oversold
        rsi_bearish = current['rsi'] > self.rsi_overbought
        
        price_above_sma20 = current['close'] > current['sma_20']
        price_above_sma50 = current['close'] > current['sma_50']
        
        if (macd_bullish or rsi_bullish) and price_above_sma20 and price_above_sma50:
            return 'BUY'
        elif macd_bearish or rsi_bearish:
            return 'SELL'
        
        return 'HOLD'


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands
    """
    
    def __init__(self):
        super().__init__("Mean Reversion Strategy")
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        if len(data) < 50:
            return 'HOLD'
        
        current = data.iloc[-1]
        
        at_lower_band = current['close'] <= current['bb_lower']
        at_upper_band = current['close'] >= current['bb_upper']
        
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
        
        if std.iloc[-1] == 0:
            return 'HOLD'
        
        z_score = (current['close'] - sma.iloc[-1]) / std.iloc[-1]
        
        if pd.isna(z_score):
            return 'HOLD'
        
        if z_score < -self.z_score_threshold:
            return 'BUY'
        elif z_score > self.z_score_threshold:
            return 'SELL'
        
        return 'HOLD'