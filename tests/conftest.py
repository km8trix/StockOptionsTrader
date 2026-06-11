"""Shared fixtures: deterministic synthetic OHLCV data. No network, ever."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_ohlcv():
    """Factory producing seeded synthetic OHLCV DataFrames.

    The index is a business-day DatetimeIndex; columns are
    open/high/low/close/volume. Identical arguments always produce an
    identical frame (numpy Generator seeded per call).
    """

    def _make(n_days: int = 120, start: str = "2023-01-02", seed: int = 42,
              start_price: float = 100.0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        index = pd.bdate_range(start=start, periods=n_days)

        daily_returns = rng.normal(0.0005, 0.01, n_days)
        close = start_price * np.cumprod(1.0 + daily_returns)

        open_ = np.empty(n_days)
        open_[0] = start_price
        open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.002, n_days - 1))

        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, n_days)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, n_days)))
        volume = rng.integers(100_000, 1_000_000, n_days).astype(float)

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )

    return _make
