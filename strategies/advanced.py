"""
Advanced Trading Strategies using Machine Learning and Advanced Techniques
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import Optional, Tuple
from strategies.base import Strategy
from core.models import Asset


class MachineLearningStrategy(Strategy):
    """
    Machine Learning-based strategy using Random Forest
    Trains on historical features and predicts BUY/SELL signals

    .. deprecated:: Phase 5
        This strategy trains once inside generate_signals on the expanding
        window it is handed, so early signals in a backtest come from a
        model that has already seen later prices (look-ahead
        contamination, flagged in Phase 1). Prefer
        desks.ml_model.GradientBoostingModel driven by
        desks.walk_forward.WalkForwardController, which enforces
        leakage-free walk-forward fitting by construction. This class
        remains importable and functional for backward compatibility.
    """
    
    def __init__(self, lookback: int = 50):
        super().__init__("Machine Learning Strategy")
        self.lookback = lookback
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _create_features(self, data: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Create features for ML model"""
        if len(data) < self.lookback + 1:
            return None
        
        features_list = []
        labels = []
        
        for i in range(self.lookback, len(data) - 1):
            # Create features from price data
            window = data.iloc[i-self.lookback:i]
            
            feature_row = [
                window['close'].pct_change().mean(),          # mean return
                window['close'].std(),                        # volatility
                (window['high'] - window['low']).mean(),      # avg range
                window['volume'].mean(),                      # avg volume
                window['close'].iloc[-1] / window['close'].mean(),  # price/avg
            ]
            features_list.append(feature_row)
            
            # Label: 1 if price goes up, 0 if down
            next_return = (data.iloc[i+1]['close'] - data.iloc[i]['close']) / data.iloc[i]['close']
            labels.append(1 if next_return > 0 else 0)
        
        return np.array(features_list), np.array(labels)
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        """Generate trading signals using ML model"""
        if len(data) < self.lookback + 10:
            return 'HOLD'
        
        # Train model on first use
        if not self.is_trained and len(data) > self.lookback + 50:
            result = self._create_features(data)
            if result is not None:
                X, y = result
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.is_trained = True
        
        if not self.is_trained:
            return 'HOLD'
        
        # Generate signal for current data
        result = self._create_features(data.iloc[:-1])
        if result is None:
            return 'HOLD'
        X, _ = result
        if len(X) == 0:
            return 'HOLD'
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled[-1:])
        
        if prediction[0] == 1:
            return 'BUY'
        else:
            return 'SELL'


class EnhancedMeanReversionStrategy(Strategy):
    """
    Enhanced mean reversion with multiple timeframe analysis
    Uses Bollinger Bands + RSI + Volatility
    """
    
    def __init__(self):
        super().__init__("Enhanced Mean Reversion")
        self.rsi_threshold_buy = 25
        self.rsi_threshold_sell = 75
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        """Generate signals using multiple indicators"""
        if len(data) < 50:
            return 'HOLD'
        
        current = data.iloc[-1]

        # Check multiple conditions
        at_lower_band = current['close'] <= current['bb_lower']
        at_upper_band = current['close'] >= current['bb_upper']
        
        rsi_oversold = current['rsi'] < self.rsi_threshold_buy
        rsi_overbought = current['rsi'] > self.rsi_threshold_sell
        
        # Volume confirmation
        volume_spike = current['volume'] > data['volume'].mean() * 1.5
        
        # Mean reversion signal
        price_vs_sma = current['close'] / current['sma_20']
        extreme_deviation = price_vs_sma < 0.95 or price_vs_sma > 1.05
        
        # BUY: Multiple conditions align for oversold, confirmed by volume
        if at_lower_band and rsi_oversold and extreme_deviation and volume_spike:
            return 'BUY'

        # SELL: Multiple conditions align for overbought, confirmed by volume
        if at_upper_band and rsi_overbought and extreme_deviation and volume_spike:
            return 'SELL'
        
        return 'HOLD'


class VolatilityBreakoutStrategy(Strategy):
    """
    Trades on volatility breakouts
    Uses ATR to identify volatility spikes
    """
    
    def __init__(self):
        super().__init__("Volatility Breakout")
        self.atr_multiplier = 2.0
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        """Generate signals on volatility breakouts"""
        if len(data) < 20:
            return 'HOLD'
        
        current = data.iloc[-1]
        
        # Calculate volatility levels
        avg_atr = data['atr'].rolling(window=20).mean().iloc[-1]
        current_atr = current['atr']
        
        # Check for volatility breakout
        volatility_spike = current_atr > avg_atr * self.atr_multiplier
        
        if not volatility_spike:
            return 'HOLD'
        
        # Determine direction based on momentum
        momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
        
        if momentum > 0.01:  # Positive momentum
            return 'BUY'
        elif momentum < -0.01:  # Negative momentum
            return 'SELL'
        
        return 'HOLD'


class CombinedStrategy(Strategy):
    """
    Combines multiple strategies for stronger signals
    Votes on BUY/SELL/HOLD decisions
    """
    
    def __init__(self):
        super().__init__("Combined Strategy")
        from strategies.base import MomentumStrategy, MeanReversionStrategy
        
        self.strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            EnhancedMeanReversionStrategy(),
            VolatilityBreakoutStrategy(),
        ]
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        """Generate signals by voting"""
        if len(data) < 50:
            return 'HOLD'
        
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for strategy in self.strategies:
            signal = strategy.generate_signals(data, asset)
            votes[signal] += 1
        
        # Return signal with most votes (tie = HOLD)
        if votes['BUY'] > votes['SELL'] and votes['BUY'] > votes['HOLD']:
            return 'BUY'
        elif votes['SELL'] > votes['BUY'] and votes['SELL'] > votes['HOLD']:
            return 'SELL'
        
        return 'HOLD'


class AdaptiveStrategy(Strategy):
    """
    Adapts between momentum and mean reversion based on market conditions
    Switches strategies based on volatility and trend strength
    """
    
    def __init__(self):
        super().__init__("Adaptive Strategy")
        from strategies.base import MomentumStrategy, MeanReversionStrategy
        
        self.momentum_strategy = MomentumStrategy()
        self.reversion_strategy = MeanReversionStrategy()
    
    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        """Adapt strategy based on market conditions"""
        if len(data) < 50:
            return 'HOLD'
        
        # Calculate market regime
        returns = data['close'].pct_change().tail(20)
        volatility = returns.std()
        trend_strength = abs(returns.mean())
        
        abs_returns = returns.abs()
        volatility_threshold = abs_returns.quantile(0.75)
        trend_threshold = abs_returns.quantile(0.50)
        
        # High volatility + weak trend = mean reversion
        # Low volatility + strong trend = momentum
        if volatility > volatility_threshold and trend_strength < trend_threshold:
            return self.reversion_strategy.generate_signals(data, asset)
        else:
            return self.momentum_strategy.generate_signals(data, asset)
