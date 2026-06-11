"""
API routes for backtesting execution, history, and exports.
"""
from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import io
import logging

from gui.globals import db
from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy, MeanReversionStrategy, StatisticalArbitrageStrategy
from strategies.advanced import (
    MachineLearningStrategy, EnhancedMeanReversionStrategy,
    VolatilityBreakoutStrategy, CombinedStrategy, AdaptiveStrategy
)

logger = logging.getLogger(__name__)

backtest_bp = Blueprint('backtest', __name__, url_prefix='/api')

# Unified strategy mapping
STRATEGIES = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'stat_arb': StatisticalArbitrageStrategy,
    'enhanced_mean_reversion': EnhancedMeanReversionStrategy,
    'volatility_breakout': VolatilityBreakoutStrategy,
    'combined': CombinedStrategy,
    'adaptive': AdaptiveStrategy,
    'machine_learning': MachineLearningStrategy,
}

@backtest_bp.route('/backtest', methods=['POST'])
def run_backtest():
    """Run backtest with given parameters"""
    try:
        data = request.json or {}
        
        symbols = data.get('symbols', '').split(',')
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
            
        strategy_name = data.get('strategy', 'momentum').lower()
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-12-31')
        initial_capital = float(data.get('initial_capital', 100000))
        position_size = float(data.get('position_size', 0.1))
        
        if strategy_name not in STRATEGIES:
            return jsonify({'error': f'Unknown strategy: {strategy_name}'}), 400
            
        strategy_instance = STRATEGIES[strategy_name]()
        backtester = BacktestEngine(strategy_instance, initial_capital=initial_capital)
        
        results = backtester.run(symbols, start_date, end_date, position_size)
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 400
            
        return jsonify(results)
    except Exception:
        logger.error('Backtest failed', exc_info=True)
        return jsonify({'error': 'Backtest failed'}), 500

@backtest_bp.route('/strategies', methods=['GET'])
def list_strategies():
    """Get available strategies"""
    strategies_list = []
    for name, strategy_class in STRATEGIES.items():
        strategies_list.append({
            'id': name,
            'name': strategy_class.__name__,
            'description': strategy_class.__doc__ or 'No description available'
        })
    return jsonify({'strategies': strategies_list})

@backtest_bp.route('/backtests', methods=['GET'])
def list_backtests():
    """Get list of saved backtests from DB"""
    try:
        backtests = db.get_backtests(limit=50)
        return jsonify({'backtests': backtests, 'count': len(backtests)})
    except Exception:
        logger.error('Failed to list backtests', exc_info=True)
        return jsonify({'error': 'Failed to retrieve backtests'}), 500

@backtest_bp.route('/backtest/<int:backtest_id>', methods=['GET'])
def get_backtest_detail(backtest_id):
    """Get backtest details with trades"""
    try:
        backtest = db.get_backtest(backtest_id)
        if not backtest:
            return jsonify({'error': 'Backtest not found'}), 404
            
        trades = db.get_backtest_trades(backtest_id)
        return jsonify({'backtest': backtest, 'trades': trades, 'trade_count': len(trades)})
    except Exception:
        logger.error('Failed to fetch backtest detail', exc_info=True)
        return jsonify({'error': 'Failed to retrieve backtest'}), 500

@backtest_bp.route('/backtest/<int:backtest_id>', methods=['DELETE'])
def delete_backtest(backtest_id):
    """Delete backtest"""
    try:
        db.delete_backtest(backtest_id)
        return jsonify({'message': 'Backtest deleted'})
    except Exception:
        logger.error('Failed to delete backtest', exc_info=True)
        return jsonify({'error': 'Failed to delete backtest'}), 500

@backtest_bp.route('/export/backtest/<int:backtest_id>', methods=['GET'])
def export_backtest(backtest_id):
    """Export backtest trades as CSV"""
    try:
        backtest = db.get_backtest(backtest_id)
        if not backtest:
            return jsonify({'error': 'Backtest not found'}), 404
            
        trades = db.get_backtest_trades(backtest_id)
        df = pd.DataFrame(trades)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'backtest_{backtest_id}.csv'
        )
    except Exception:
        logger.error('Failed to export backtest CSV', exc_info=True)
        return jsonify({'error': 'Failed to export backtest'}), 500

@backtest_bp.route('/export/report/<int:backtest_id>', methods=['GET'])
def export_report(backtest_id):
    """Export backtest as text report"""
    try:
        backtest = db.get_backtest(backtest_id)
        if not backtest:
            return jsonify({'error': 'Backtest not found'}), 404
            
        report = f"""
BACKTEST REPORT
===============
Strategy: {backtest.get('strategy', 'Unknown')}
Date: {backtest.get('timestamp', 'Unknown')}
Period: {backtest.get('start_date', '')} to {backtest.get('end_date', '')}

PERFORMANCE METRICS
-------------------
Total Return: {backtest.get('total_return', 0):.2%}
Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.2f}
Max Drawdown: {backtest.get('max_drawdown', 0):.2%}
Win Rate: {backtest.get('win_rate', 0):.2%}
Total Trades: {backtest.get('total_trades', 0)}

Initial Capital: ${backtest.get('initial_capital', 0):,.2f}
Final Value: ${backtest.get('initial_capital', 0) * (1 + backtest.get('total_return', 0)):,.2f}
"""
        return report, 200, {'Content-Type': 'text/plain'}
    except Exception:
        logger.error('Failed to export backtest report', exc_info=True)
        return jsonify({'error': 'Failed to export report'}), 500