"""
API routes for backtesting execution, history, and exports.

Two execution paths:
    POST /api/backtest      — legacy synchronous run (kept for compatibility).
    POST /api/backtest/run  — async run on the process-wide JobManager; poll
                              GET /api/backtest/status/<job_id> for progress
                              and the final report.
"""
from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import io
import logging

from gui.globals import get_db
from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy, MeanReversionStrategy, StatisticalArbitrageStrategy
from strategies.advanced import (
    MachineLearningStrategy, EnhancedMeanReversionStrategy,
    VolatilityBreakoutStrategy, CombinedStrategy, AdaptiveStrategy
)
from utils.jobs import get_job_manager

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


def _parse_symbols(raw):
    """Normalize the 'symbols' payload field to an upper-cased list.

    Accepts either a comma-separated string ('aapl, msft') or a JSON array
    of strings (['aapl', 'msft']) — both natural client payloads. Returns
    None for any other shape (number, dict, list with non-string entries)
    so callers can reject the request with a 400 instead of letting an
    AttributeError surface as a 500.
    """
    if isinstance(raw, str):
        raw = raw.split(',')
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        return None
    return [s.strip().upper() for s in raw if s.strip()]


def _date_str(value):
    """'YYYY-MM-DD' for datetimes/Timestamps; str fallback for anything else."""
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d')
    return str(value)[:10]


def _nan_to_none(value):
    """NaN floats serialize as literal ``NaN`` (invalid JSON) — null them."""
    if isinstance(value, float) and value != value:
        return None
    return value


def _json_safe_report(report):
    """Normalize report date fields to 'YYYY-MM-DD' strings.

    The engine report carries pandas Timestamps in trades, portfolio_history,
    and pending_signals. The async job result is stored and re-serialized
    outside a request, so dates are normalized once here — keeping the job
    payload deterministic for the frontend (ISO date strings everywhere).
    Summary metrics that degenerate to NaN (e.g. Sharpe with zero variance)
    become null so the browser's JSON.parse never chokes.
    """
    safe = dict(report)
    safe['summary'] = {
        key: _nan_to_none(value)
        for key, value in report.get('summary', {}).items()
    }
    safe['trades'] = [
        {**t, 'date': _date_str(t.get('date')),
         'signal_date': _date_str(t.get('signal_date'))}
        for t in report.get('trades', [])
    ]
    safe['portfolio_history'] = [
        {**h, 'timestamp': _date_str(h.get('timestamp'))}
        for h in report.get('portfolio_history', [])
    ]
    safe['pending_signals'] = [
        {**p, 'signal_date': _date_str(p.get('signal_date'))}
        for p in report.get('pending_signals', [])
    ]
    return safe


def _total_return_fraction(summary, initial_capital):
    """Total return as a FRACTION of initial capital (0.08 == +8%).

    The engine summary's 'total_return' is a DOLLAR P&L
    (PortfolioManager.get_total_return = realized + unrealized), while
    'total_return_pct' is x100. The DB total_return column — consumed by
    the dashboard/history UI via fmtPct and by export_report via ':.2%' —
    stores the fraction, so convert here. Returns None when neither form
    is available.
    """
    pct = summary.get('total_return_pct')
    if pct is not None:
        return float(pct) / 100.0
    dollars = summary.get('total_return')
    if dollars is not None and initial_capital and initial_capital > 0:
        return float(dollars) / float(initial_capital)
    return None


def _pct_to_fraction(value):
    """x100 percent -> fraction (-6.5 -> -0.065); None passes through.

    The engine summary's 'max_drawdown' and 'win_rate' are x100 percents
    (PortfolioManager.get_max_drawdown / get_win_rate), but — like
    total_return — their DB columns store FRACTIONS: that is the unit
    export_report's ':.2%' formatting and the pre-existing writers
    (e.g. examples_advanced.py) assume.
    """
    if value is None:
        return None
    return float(value) / 100.0


def _save_report(name, symbols, start_date, end_date, initial_capital,
                 strategy_name, position_size, report):
    """Persist a finished async run so the saved-history panel can list it.

    Strictly best-effort: persistence failure must never fail the job, so
    the user still gets their results (with a WARNING server-side).
    """
    summary = report.get('summary', {})
    try:
        get_db().save_backtest(
            name=name,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_capital=initial_capital,
            strategy=strategy_name,
            parameters={'position_size': position_size},
            results={
                'total_return': _total_return_fraction(summary,
                                                       initial_capital),
                'sharpe_ratio': summary.get('sharpe_ratio'),
                'max_drawdown': _pct_to_fraction(summary.get('max_drawdown')),
                'win_rate': _pct_to_fraction(summary.get('win_rate')),
                'total_trades': summary.get('closed_trades'),
                'summary': summary,
                'portfolio_history': report.get('portfolio_history', []),
                'drawdown_series': report.get('drawdown_series', []),
                'benchmark': report.get('benchmark'),
            },
        )
    except Exception:
        logger.warning('Could not save backtest %r to history', name,
                       exc_info=True)


def _run_backtest_job(symbols, strategy_name, start_date, end_date,
                      initial_capital, position_size, name, progress=None):
    """JobManager job body: run the engine and return a JSON-safe report.

    The JobManager injects ``progress`` (callable taking float 0-100); it is
    wired straight into BacktestEngine.run's progress_callback. Raising here
    marks the job 'error' with the exception message.
    """
    strategy_instance = STRATEGIES[strategy_name]()
    backtester = BacktestEngine(strategy_instance,
                                initial_capital=initial_capital)
    results = backtester.run(symbols, start_date, end_date, position_size,
                             progress_callback=progress,
                             benchmark_symbol='SPY')

    if 'error' in results:
        raise ValueError(results['error'])

    results['data_sources'] = {
        symbol: _fetch_info(backtester.market_data, symbol)
        for symbol in symbols
    }
    report = _json_safe_report(results)
    _save_report(name, symbols, start_date, end_date, initial_capital,
                 strategy_name, position_size, report)
    return report


@backtest_bp.route('/backtest/run', methods=['POST'])
def run_backtest_async():
    """Submit a backtest to the JobManager; returns {'job_id'} immediately."""
    # silent=True: a missing/malformed JSON body degrades to {} and gets a
    # specific 400 below, instead of werkzeug's HTML BadRequest page.
    data = request.get_json(silent=True) or {}

    symbols = _parse_symbols(data.get('symbols', ''))
    if symbols is None:
        return jsonify({'error': 'symbols must be a comma-separated string '
                                 'or a list of strings'}), 400
    if not symbols:
        return jsonify({'error': 'No symbols provided'}), 400

    strategy_name = data.get('strategy', 'momentum').lower()
    if strategy_name not in STRATEGIES:
        return jsonify({'error': f'Unknown strategy: {strategy_name}'}), 400

    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2023-12-31')
    if start_date >= end_date:
        return jsonify({'error': 'start_date must be before end_date'}), 400

    try:
        initial_capital = float(data.get('initial_capital', 100000))
        position_size = float(data.get('position_size', 0.1))
    except (TypeError, ValueError):
        return jsonify({'error': 'initial_capital and position_size must be numeric'}), 400
    if initial_capital <= 0:
        return jsonify({'error': 'initial_capital must be positive'}), 400
    if not 0 < position_size <= 1:
        return jsonify({'error': 'position_size must be in (0, 1]'}), 400

    name = (data.get('name') or '').strip() or \
        f"{strategy_name} {','.join(symbols)}"

    job_id = get_job_manager().submit(
        _run_backtest_job, symbols, strategy_name, start_date, end_date,
        initial_capital, position_size, name)
    logger.info('Backtest job %s submitted (%s on %s)', job_id,
                strategy_name, symbols)
    return jsonify({'job_id': job_id}), 202


@backtest_bp.route('/backtest/status/<job_id>', methods=['GET'])
def backtest_job_status(job_id):
    """Proxy JobManager.get: full job record incl. result once 'done'."""
    job = get_job_manager().get(job_id)
    if job is None:
        return jsonify({'error': 'Unknown job id'}), 404
    return jsonify(job)


@backtest_bp.route('/backtest', methods=['POST'])
def run_backtest():
    """Run backtest with given parameters"""
    try:
        data = request.get_json(silent=True) or {}

        symbols = _parse_symbols(data.get('symbols', ''))
        if symbols is None:
            return jsonify({'error': 'symbols must be a comma-separated '
                                     'string or a list of strings'}), 400
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

        # Additive data provenance: which provider served each symbol.
        results['data_sources'] = {
            symbol: _fetch_info(backtester.market_data, symbol)
            for symbol in symbols
        }

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
        backtests = get_db().get_backtests(limit=50)
        return jsonify({'backtests': backtests, 'count': len(backtests)})
    except Exception:
        logger.error('Failed to list backtests', exc_info=True)
        return jsonify({'error': 'Failed to retrieve backtests'}), 500

@backtest_bp.route('/backtest/<int:backtest_id>', methods=['GET'])
def get_backtest_detail(backtest_id):
    """Get backtest details with trades"""
    try:
        db = get_db()
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
        get_db().delete_backtest(backtest_id)
        return jsonify({'message': 'Backtest deleted'})
    except Exception:
        logger.error('Failed to delete backtest', exc_info=True)
        return jsonify({'error': 'Failed to delete backtest'}), 500

@backtest_bp.route('/export/backtest/<int:backtest_id>', methods=['GET'])
def export_backtest(backtest_id):
    """Export backtest trades as CSV"""
    try:
        db = get_db()
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
        backtest = get_db().get_backtest(backtest_id)
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