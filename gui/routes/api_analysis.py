# gui/routes/api_analysis.py
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from gui.globals import market_data

logger = logging.getLogger(__name__)

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')

# Most bars returned to the chart in one payload.
MAX_CHART_ROWS = 500
# Most bars walked when overlaying per-bar strategy signals (bounds the cost
# of strategies that retrain models on every call).
MAX_SIGNAL_BARS = 100


def _parse_date(value):
    """'YYYY-MM-DD' -> datetime, or None if absent/invalid."""
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d')
    except ValueError:
        return None


def _compute_signals(strategy_name, data, symbol):
    """Walk the tail of an indicator-enriched frame and collect BUY/SELL
    marks for charting. Returns a list of {'date','signal','price'} dicts.

    Any failure degrades to None (logged) — signal overlay is auxiliary and
    must never fail the analyze call itself.
    """
    # Imported here (and lazily) to keep module import cheap; the mapping is
    # owned by the backtest blueprint and reused so the two stay in sync.
    from gui.routes.api_backtest import STRATEGIES
    from core.models import Asset, AssetType

    strategy_cls = STRATEGIES.get(strategy_name)
    if strategy_cls is None:
        return None
    try:
        strategy = strategy_cls()
        asset = Asset(symbol, AssetType.STOCK)
        signals = []
        start_idx = max(50, len(data) - MAX_SIGNAL_BARS)
        for i in range(start_idx, len(data)):
            window = data.iloc[: i + 1]
            signal = strategy.generate_signals(window, asset)
            if signal in ('BUY', 'SELL'):
                bar = data.iloc[i]
                close = bar['close']
                signals.append({
                    'date': pd.Timestamp(data.index[i]).strftime('%Y-%m-%d'),
                    'signal': signal,
                    'price': float(close) if not pd.isna(close) else None,
                })
        return signals
    except Exception:
        logger.warning(
            'Signal overlay failed for %s with strategy %s',
            symbol, strategy_name, exc_info=True)
        return None


def _fetch_info(handler, symbol):
    """Provenance for one symbol, or None.

    Guarded with getattr so this never breaks against an older
    MarketDataHandler that predates get_last_fetch_info; provenance is
    auxiliary metadata, so any failure degrades to None instead of a 500.
    """
    get_info = getattr(handler, 'get_last_fetch_info', None)
    if not callable(get_info):
        return None
    try:
        return get_info(symbol)
    except Exception:
        logger.warning('get_last_fetch_info failed for %s', symbol, exc_info=True)
        return None


@analysis_bp.route('/analyze/<symbol>')
def analyze_stock(symbol):
    """Indicator-enriched OHLCV for one symbol.

    Query params (all optional, JSON stays additive):
        days:     lookback in calendar days (default 252) — legacy param.
        start/end: explicit 'YYYY-MM-DD' range; when both are valid they
                  take precedence over ``days``.
        strategy: a key from the backtest STRATEGIES map; when present the
                  response gains 'signals' (BUY/SELL marks for charting)
                  and 'strategy'.
    """
    try:
        symbol = symbol.upper()
        days_back = request.args.get('days', 252, type=int)
        start_dt = _parse_date(request.args.get('start'))
        end_dt = _parse_date(request.args.get('end'))
        if start_dt and end_dt:
            if start_dt >= end_dt:
                return jsonify({'error': 'start must be before end'}), 400
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        strategy_name = (request.args.get('strategy') or '').strip().lower()
        if strategy_name:
            from gui.routes.api_backtest import STRATEGIES
            if strategy_name not in STRATEGIES:
                return jsonify({'error': f'Unknown strategy: {strategy_name}'}), 400

        data = market_data.fetch_stock_data(symbol, start_date, end_date)
        if data is None or data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404

        data = market_data.calculate_indicators(data)
        recent_data = data.tail(MAX_CHART_ROWS).copy()

        recent_data.index = pd.to_datetime(recent_data.index).strftime('%Y-%m-%d')
        recent_data = recent_data.replace({np.nan: None, pd.NA: None})

        def safe_float(val):
            return float(val) if not pd.isna(val) else None

        result = {
            'symbol': symbol,
            'data': recent_data.to_dict(orient='index'),
            'current_price': safe_float(data.iloc[-1]['close']),
            'current_rsi': safe_float(data.iloc[-1]['rsi']),
            'current_macd': safe_float(data.iloc[-1]['macd']),
            'current_sma_20': safe_float(data.iloc[-1]['sma_20']),
            'current_sma_50': safe_float(data.iloc[-1]['sma_50']),
            # Additive data provenance: which provider served this symbol.
            'data_source': _fetch_info(market_data, symbol),
        }
        if strategy_name:
            result['strategy'] = strategy_name
            result['signals'] = _compute_signals(strategy_name, data, symbol)
        return jsonify(result)
    except Exception:
        logger.error('Stock analysis failed', exc_info=True)
        return jsonify({'error': 'Analysis failed'}), 500
