"""Survivorship-free price feed for the backtest engine.

``BacktestEngine`` fetches OHLCV through a ``MarketDataHandler`` (live OpenBB
providers), which only serves names that still trade today — so a small/mid
backtest silently drops delisted names and reintroduces survivorship bias.
``WarehouseMarketData`` swaps the fetch to the point-in-time Sharadar warehouse
(``PitWarehouse.ohlcv``), which keeps delisted names for their live span. It
subclasses ``MarketDataHandler`` so ``calculate_indicators`` and everything else
the engine calls are inherited unchanged; only the fetch is overridden.

Usage:  BacktestEngine(desk=..., market_data=WarehouseMarketData())
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from data.market_data import MarketDataHandler
from data.pit_warehouse import PitWarehouse

logger = logging.getLogger(__name__)


class WarehouseMarketData(MarketDataHandler):
    """MarketDataHandler backed by the PIT warehouse (adjusted, survivorship-free)."""

    def __init__(self, warehouse: Optional[PitWarehouse] = None):
        super().__init__()
        self._wh = warehouse or PitWarehouse()

    def fetch_stock_data(self, symbol: str, start_date: str,
                         end_date: str) -> pd.DataFrame:
        df = self._wh.ohlcv(symbol, start_date, end_date)
        self._last_fetch_info[symbol] = {
            'provider': 'pit_warehouse', 'from_cache': True, 'failures': [],
            'start_date': start_date, 'end_date': end_date,
            'fetched_at': datetime.now(timezone.utc).isoformat(),
        }
        if df is None or df.empty:
            return self._empty_data(symbol)
        self.cache[f"{symbol}_{start_date}_{end_date}"] = df
        self.stock_data[symbol] = df
        return df

    def delisting_date(self, symbol: str):
        """Final listed session for ``symbol``, or ``None`` when still live."""
        return self._wh.delisting_date(symbol)

    def delisting_payout(self, symbol: str, final_close: float) -> dict:
        """Return the non-qualifying final-close fallback and action metadata.

        SHARADAR/ACTIONS does not provide a verified per-share settlement term,
        so its reported value must never override the last observed close.  A
        future independently sourced settlement ledger can be checked before
        this fallback without weakening the current fail-closed contract.
        """
        action = self._wh.delisting_action(symbol)
        payout = {
            'price': float(final_close),
            'source': 'final_tradable_close',
            'quality_flags': ['delisting_terms_unavailable'],
            'action_date': None,
            'contraticker': None,
        }
        if action is not None:
            payout.update({
                'quality_flags': list(dict.fromkeys(
                    payout['quality_flags'] + action['quality_flags'])),
                'action_date': (action['date'].isoformat()
                                if action['date'] is not None else None),
                'action': action['action'],
                'contraticker': action.get('contraticker'),
                'reported_value': action.get('reported_value'),
                'value_semantics': action.get('value_semantics'),
            })
        return payout

    def data_snapshot(self, tables=None) -> dict:
        """Immutable content version for research/promotion artifacts."""
        return self._wh.snapshot_version(tables)
