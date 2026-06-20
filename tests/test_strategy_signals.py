"""Focused tests for the strategy entry-signal wiring.

These lock in the behavior change that wired previously-computed-but-unused
indicator signals into the entry logic:
  * MomentumStrategy BUY now fires on (macd_bullish OR rsi_bullish) AND price
    above BOTH the 20- and 50-day SMAs (rsi_bullish and the sma20 gate were
    dead before).
  * EnhancedMeanReversionStrategy entries now require a volume_spike
    confirmation (the '# Volume confirmation' the code computed but never used).

The strategies consume pre-computed indicator columns, so each test builds a
minimal 50-row frame (the strategies' own min-length guard) with neutral
defaults and overrides only the final row for the scenario under test.
"""

import pandas as pd

from core.models import Asset, AssetType
from strategies.base import MomentumStrategy
from strategies.advanced import EnhancedMeanReversionStrategy

ASSET = Asset(symbol='TEST', asset_type=AssetType.STOCK)


def _frame(n: int = 50, **last) -> pd.DataFrame:
    """A 50-row OHLCV+indicator frame with neutral defaults; keyword args
    override the FINAL row (prev row keeps the neutral defaults)."""
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0,
        'volume': 1000.0,
        'sma_20': 100.0, 'sma_50': 100.0,
        'macd': 0.0, 'signal': 0.0, 'rsi': 50.0,
        'bb_lower': 95.0, 'bb_upper': 105.0,
    }, index=idx)
    for column, value in last.items():
        df.iloc[-1, df.columns.get_loc(column)] = value
    return df


class TestMomentumWiring:
    """MomentumStrategy: (macd_bullish OR rsi_bullish) AND above both SMAs."""

    def test_rsi_oversold_alone_now_triggers_buy(self):
        # No MACD cross, but RSI oversold and price above both SMAs.
        # Old logic (macd_bullish AND sma50 only) returned HOLD here.
        df = _frame(rsi=20.0, close=110.0, sma_20=100.0, sma_50=100.0)
        assert MomentumStrategy().generate_signals(df, ASSET) == 'BUY'

    def test_below_sma20_now_blocks_a_macd_buy(self):
        # MACD cross up and above the 50-SMA, but BELOW the 20-SMA.
        # Old logic returned BUY; the new sma20 gate must block it.
        df = _frame(macd=1.0, signal=0.0, rsi=50.0, close=105.0,
                    sma_20=110.0, sma_50=100.0)
        assert MomentumStrategy().generate_signals(df, ASSET) == 'HOLD'

    def test_macd_cross_above_both_smas_still_buys(self):
        df = _frame(macd=1.0, signal=0.0, rsi=50.0, close=110.0,
                    sma_20=100.0, sma_50=100.0)
        assert MomentumStrategy().generate_signals(df, ASSET) == 'BUY'


class TestEnhancedMeanReversionVolumeConfirmation:
    """EnhancedMeanReversionStrategy: entries require a volume spike."""

    def test_no_volume_spike_blocks_buy(self):
        # At lower band, oversold, extreme deviation — but flat volume.
        # Old logic returned BUY; the volume confirmation must block it.
        df = _frame(close=90.0, bb_lower=92.0, rsi=20.0, sma_20=100.0,
                    volume=1000.0)
        assert EnhancedMeanReversionStrategy().generate_signals(df, ASSET) == 'HOLD'

    def test_volume_spike_confirms_buy(self):
        # Same setup, but the final bar's volume spikes above 1.5x the mean.
        df = _frame(close=90.0, bb_lower=92.0, rsi=20.0, sma_20=100.0,
                    volume=5000.0)
        assert EnhancedMeanReversionStrategy().generate_signals(df, ASSET) == 'BUY'
