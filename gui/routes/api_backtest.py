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
from utils.provenance import capture_run_provenance

logger = logging.getLogger(__name__)

# The desks package is built in the parallel Phase 5 backend task. Legacy
# strategy backtests must keep working without it, so the import is guarded
# (mirroring the LiveEtradeBroker pattern in api_trading); desk-mode requests
# surface a loud 503 instead of an ImportError 500.
DESK_REGISTRY_IMPORT_ERROR: str | None
try:
    from desks.registry import create_desk, create_fund_orchestrator
    from desks.models import available_models
    from backtesting.reweighting_fund import ReweightingFundBacktest
    DESK_REGISTRY_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - any import failure disables desk/fund mode
    create_desk = None
    create_fund_orchestrator = None
    available_models = None
    ReweightingFundBacktest = None
    DESK_REGISTRY_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import desks.registry / backtesting.reweighting_fund — '
        'desk- and fund-mode backtests are unavailable: %s',
        DESK_REGISTRY_IMPORT_ERROR,
    )

backtest_bp = Blueprint('backtest', __name__, url_prefix='/api')

# Desk keys that accept the optional walk-forward ``desk_model`` selector.
# Mirrors desks.registry._MODEL_SELECTABLE_DESKS (kept local so a desks-package
# import failure still disables desk mode cleanly without breaking this list).
MODEL_SELECTABLE_DESKS = ('foundation', 'twosigma')

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


def _parse_allocations(fund):
    """Normalize a 'fund' payload to an ordered {desk_key: weight} dict.

    Accepts either a list of desk keys (['foundation', 'renaissance'] -> equal
    weight summing to 1.0) or a {key: weight} object (weights as numbers, used
    as-is). Insertion order is preserved (the orchestrator's deterministic
    netting depends on desk order). Returns (allocations, None) on success or
    (None, error_message) so the caller can answer 400 instead of letting a
    bad shape surface as a 500. Desk-key validity and the sum<=1.0 rule are
    enforced downstream by create_fund_orchestrator.
    """
    if isinstance(fund, list):
        keys = [k.strip().lower() for k in fund
                if isinstance(k, str) and k.strip()]
        if not keys:
            return None, 'fund must list at least one desk key'
        if len(set(keys)) != len(keys):
            return None, 'fund desk keys must be unique'
        weight = 1.0 / len(keys)
        return {key: weight for key in keys}, None
    if isinstance(fund, dict):
        if not fund:
            return None, 'fund must include at least one desk'
        allocations = {}
        for key, weight in fund.items():
            if not isinstance(key, str) or not key.strip():
                return None, 'fund desk keys must be non-empty strings'
            try:
                value = float(weight)
            except (TypeError, ValueError):
                return None, f"fund weight for '{key}' must be numeric"
            if not value > 0:
                return None, f"fund weight for '{key}' must be > 0"
            allocations[key.strip().lower()] = value
        return allocations, None
    return None, 'fund must be a list of desk keys or a {key: weight} object'


def _parse_seed(data):
    """Optional RNG seed for reproducibility. Returns (seed, None) with seed an
    int or None, or (None, error) when the value is present but not a plain
    integer. bool is rejected (it is an int subclass but a nonsense seed)."""
    seed = data.get('seed')
    if seed is None:
        return None, None
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None, 'seed must be an integer'
    return seed, None


def _parse_realistic_fills(data):
    """Opt-in realistic execution flag (Phase 3 Step 6). Returns (bool, None),
    or (False, error) when present but not a real boolean. Default False keeps
    the vanilla cost model (and byte-identical reports). Applies to strategy and
    desk modes; fund mode does not expose it yet."""
    value = data.get('realistic_fills', False)
    if not isinstance(value, bool):
        return False, 'realistic_fills must be a boolean'
    return value, None


def _parse_desk_model(data):
    """Opt-in walk-forward model selection (Phase A). Returns (model_key, None)
    with model_key a valid id or None, or (None, error) when present but
    invalid. Default (absent) -> None, which keeps desk runs byte-identical to
    the historical default model. Only meaningful in desk mode; the route
    additionally restricts it to the model-selectable desks
    (MODEL_SELECTABLE_DESKS: 'foundation', 'twosigma'). Valid ids are those
    from desks.models.available_models(); when the desk framework failed to
    import (available_models is None) any non-absent value is rejected so the
    request fails clearly rather than silently ignoring the field."""
    value = data.get('desk_model')
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, 'desk_model must be a non-empty model id'
    if available_models is None:
        return None, 'Desk framework unavailable'
    model_key = value.strip().lower()
    valid_ids = {m['id'] for m in available_models()}
    if model_key not in valid_ids:
        return None, (f"Unknown desk_model: {model_key} "
                      f"(valid: {', '.join(sorted(valid_ids))})")
    return model_key, None


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
                # Run-level reproducibility provenance (git/deps/seed). Additive
                # to the JSON results blob — no DB schema change.
                'provenance': report.get('provenance'),
            },
        )
    except Exception:
        logger.warning('Could not save backtest %r to history', name,
                       exc_info=True)


def _execute_engine_job(backtester, symbols, start_date, end_date,
                        initial_capital, position_size, name, strategy_label,
                        progress, seed=None):
    """Shared job tail: run an engine, attach provenance, persist, return.

    ``strategy_label`` is what the saved-history row stores in its strategy
    column — the strategy key for legacy runs, ``desk:<key>`` for desk runs.
    ``seed`` is the RNG seed the run was pinned to (recorded in provenance;
    None when the run was not pinned).
    """
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
    # Run-level reproducibility provenance, attached AFTER json-safing (it is
    # already JSON-safe) so the saved results blob and the returned report both
    # carry it. Additive — never present on pre-Phase-3 saved runs.
    report['provenance'] = capture_run_provenance(seed)
    _save_report(name, symbols, start_date, end_date, initial_capital,
                 strategy_label, position_size, report)
    return report


def _run_backtest_job(symbols, strategy_name, start_date, end_date,
                      initial_capital, position_size, name, seed=None,
                      enable_realistic_fills=False, progress=None):
    """JobManager job body: run the engine and return a JSON-safe report.

    The JobManager injects ``progress`` (callable taking float 0-100); it is
    wired straight into BacktestEngine.run's progress_callback. ``seed`` (if
    given) pins the engine's RNG for reproducibility. ``enable_realistic_fills``
    (Step 6, default False) turns on ADV market impact + cap-and-requeue partial
    fills. Raising here marks the job 'error' with the exception message.
    """
    strategy_instance = STRATEGIES[strategy_name]()
    backtester = BacktestEngine(
        strategy_instance, initial_capital=initial_capital, seed=seed,
        enable_realistic_fills=enable_realistic_fills)
    return _execute_engine_job(backtester, symbols, start_date, end_date,
                               initial_capital, position_size, name,
                               strategy_name, progress, seed=seed)


def _run_desk_backtest_job(symbols, desk, desk_key, start_date, end_date,
                           initial_capital, position_size, name, seed=None,
                           enable_realistic_fills=False, progress=None):
    """JobManager job body (desk mode): engine built per contract C2.

    The report comes back with the additive desk keys (contract C3:
    ``desk`` / ``trader_notes`` / ``walk_forward``) which pass through
    :func:`_json_safe_report` untouched — they are JSON-safe by contract.
    The saved-history row stores the strategy as ``desk:<key>`` so desk runs
    stay distinguishable in the history/compare/export UIs (additive: the
    column still holds a plain string). ``seed`` (if given) pins the engine RNG.
    """
    backtester = BacktestEngine(
        desk=desk, initial_capital=initial_capital, seed=seed,
        enable_realistic_fills=enable_realistic_fills)
    return _execute_engine_job(backtester, symbols, start_date, end_date,
                               initial_capital, position_size, name,
                               f'desk:{desk_key}', progress, seed=seed)


def _run_fund_backtest_job(symbols, allocations, rebalance_every, warmup,
                           target_gross, start_date, end_date,
                           initial_capital, name, seed=None, progress=None):
    """JobManager job body (fund mode): a multi-desk fund backtest with in-run
    risk-parity reweighting (ReweightingFundBacktest — shadow solo books).

    Runs each desk solo for its standalone curve, then the fund with a
    DynamicReweighter. The report carries the additive ``orchestrator`` roster
    and ``reweight_log`` (rebalance dates + weights + fallback flags), both
    JSON-safe by construction, alongside the usual desk keys (the driver is the
    'fund' orchestrator). The saved-history row stores the strategy as
    ``fund:<k1>+<k2>`` so fund runs stay distinguishable in history/compare.
    Reweighting has no single position_size, so none is persisted.
    """
    backtester = ReweightingFundBacktest(
        allocations, initial_capital=initial_capital,
        rebalance_every=rebalance_every, warmup=warmup,
        target_gross=target_gross, seed=seed)
    results = backtester.run(symbols, start_date, end_date,
                             benchmark_symbol='SPY', progress_callback=progress)
    if 'error' in results:
        raise ValueError(results['error'])
    report = _json_safe_report(results)
    report['provenance'] = capture_run_provenance(seed)
    _save_report(name, symbols, start_date, end_date, initial_capital,
                 f"fund:{'+'.join(allocations)}", None, report)
    return report


def _submit_fund_backtest(data, symbols):
    """Validate a fund-mode payload and submit the job (helper for the route).

    Returns a Flask (json, status) tuple. Fund mode is mutually exclusive with
    'strategy'/'desk'; desk-key validity and the sum<=1.0 rule are checked
    eagerly via create_fund_orchestrator so the user gets a 400 up front rather
    than a failed background job.
    """
    if 'strategy' in data or 'desk' in data:
        return jsonify({'error': "Provide only one of 'fund', 'desk', or "
                                 "'strategy'"}), 400
    if create_desk is None or ReweightingFundBacktest is None:
        return jsonify({'error': 'Desk framework unavailable',
                        'reason': DESK_REGISTRY_IMPORT_ERROR}), 503

    allocations, err = _parse_allocations(data.get('fund'))
    if err:
        return jsonify({'error': err}), 400

    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2023-12-31')
    if start_date >= end_date:
        return jsonify({'error': 'start_date must be before end_date'}), 400

    try:
        initial_capital = float(data.get('initial_capital', 100000))
    except (TypeError, ValueError):
        return jsonify({'error': 'initial_capital must be numeric'}), 400
    if initial_capital <= 0:
        return jsonify({'error': 'initial_capital must be positive'}), 400

    try:
        rebalance_every = int(data.get('rebalance_every', 21))
        warmup = int(data.get('warmup', 0))
        target_gross = float(data.get('target_gross', 1.0))
    except (TypeError, ValueError):
        return jsonify({'error': 'rebalance_every and warmup must be '
                                 'integers; target_gross must be numeric'}), 400
    if rebalance_every < 1:
        return jsonify({'error': 'rebalance_every must be >= 1'}), 400
    if warmup < 0:
        return jsonify({'error': 'warmup must be >= 0'}), 400
    if not 0 < target_gross <= 1.0:
        return jsonify({'error': 'target_gross must be in (0, 1]'}), 400

    seed, seed_err = _parse_seed(data)
    if seed_err:
        return jsonify({'error': seed_err}), 400

    # Eager validation: unknown desk keys and over-allocation (sum > 1.0)
    # surface as a clean 400 here (contract C1 messages) instead of a failed
    # job. Constructs throwaway desks — cheap; the heavy work is .run().
    try:
        create_fund_orchestrator(allocations)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    name = (data.get('name') or '').strip() or \
        f"fund:{'+'.join(allocations)} {','.join(symbols)}"
    job_id = get_job_manager().submit(
        _run_fund_backtest_job, symbols, allocations, rebalance_every,
        warmup, target_gross, start_date, end_date, initial_capital, name,
        seed)
    logger.info('Fund backtest job %s submitted (fund:%s on %s)', job_id,
                '+'.join(allocations), symbols)
    return jsonify({'job_id': job_id}), 202


@backtest_bp.route('/backtest/run', methods=['POST'])
def run_backtest_async():
    """Submit a backtest to the JobManager; returns {'job_id'} immediately.

    Accepts EITHER 'strategy' (legacy, default 'momentum') or 'desk' (a
    desks.registry key, Phase 5). Desk mode builds the engine through
    desks.registry.create_desk + BacktestEngine(desk=...) — contract C2.
    """
    # silent=True: a missing/malformed JSON body degrades to {} and gets a
    # specific 400 below, instead of werkzeug's HTML BadRequest page.
    data = request.get_json(silent=True) or {}

    symbols = _parse_symbols(data.get('symbols', ''))
    if symbols is None:
        return jsonify({'error': 'symbols must be a comma-separated string '
                                 'or a list of strings'}), 400
    if not symbols:
        return jsonify({'error': 'No symbols provided'}), 400

    # Fund mode (multi-desk + in-run risk-parity reweighting) is fully
    # self-contained; the desk/strategy paths below stay unchanged.
    if data.get('fund') is not None:
        # desk_model is a single-desk (foundation) selector; it has no
        # meaning for a multi-desk fund run. Reject it explicitly with a
        # 400, consistent with the strategy / non-foundation-desk modes
        # below, instead of silently ignoring it.
        if data.get('desk_model') is not None:
            return jsonify(
                {'error': "desk_model is not valid in fund mode"}), 400
        return _submit_fund_backtest(data, symbols)

    # Optional walk-forward model selection (Phase A); only valid in desk
    # mode on the foundation desk, enforced below. Parsed up front so a bad
    # id is a clean 400 regardless of mode.
    desk_model, desk_model_err = _parse_desk_model(data)
    if desk_model_err:
        return jsonify({'error': desk_model_err}), 400

    desk_key = data.get('desk')
    desk = None
    strategy_name = None
    if desk_key is not None:
        if 'strategy' in data:
            return jsonify({'error': "Provide either 'strategy' or 'desk', "
                                     'not both'}), 400
        if not isinstance(desk_key, str) or not desk_key.strip():
            return jsonify({'error': 'desk must be a non-empty desk key'}), 400
        if create_desk is None:
            return jsonify({'error': 'Desk framework unavailable',
                            'reason': DESK_REGISTRY_IMPORT_ERROR}), 503
        desk_key = desk_key.strip().lower()
        # desk_model is only meaningful for the model-selectable desks; reject
        # it for any other desk with a clear 400 before construction.
        if desk_model is not None and desk_key not in MODEL_SELECTABLE_DESKS:
            return jsonify({'error': "desk_model is only valid for the "
                                     "foundation or twosigma desk"}), 400
        try:
            desk = create_desk(desk_key, model_key=desk_model)
        except ValueError as exc:
            # Contract C1 messages pass straight through: 'Unknown desk: ...',
            # "Desk '<key>' activates in Phase N", or the model-selection
            # rejection from the registry.
            return jsonify({'error': str(exc)}), 400
    else:
        if desk_model is not None:
            return jsonify({'error': "desk_model is only valid for the "
                                     "foundation or twosigma desk"}), 400
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

    seed, seed_err = _parse_seed(data)
    if seed_err:
        return jsonify({'error': seed_err}), 400

    realistic, realistic_err = _parse_realistic_fills(data)
    if realistic_err:
        return jsonify({'error': realistic_err}), 400

    if desk is not None:
        name = (data.get('name') or '').strip() or \
            f"desk:{desk_key} {','.join(symbols)}"
        job_id = get_job_manager().submit(
            _run_desk_backtest_job, symbols, desk, desk_key, start_date,
            end_date, initial_capital, position_size, name, seed, realistic)
        logger.info('Desk backtest job %s submitted (desk:%s on %s)', job_id,
                    desk_key, symbols)
        return jsonify({'job_id': job_id}), 202

    name = (data.get('name') or '').strip() or \
        f"{strategy_name} {','.join(symbols)}"

    job_id = get_job_manager().submit(
        _run_backtest_job, symbols, strategy_name, start_date, end_date,
        initial_capital, position_size, name, seed, realistic)
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

        seed, seed_err = _parse_seed(data)
        if seed_err:
            return jsonify({'error': seed_err}), 400

        realistic, realistic_err = _parse_realistic_fills(data)
        if realistic_err:
            return jsonify({'error': realistic_err}), 400

        strategy_instance = STRATEGIES[strategy_name]()
        backtester = BacktestEngine(
            strategy_instance, initial_capital=initial_capital, seed=seed,
            enable_realistic_fills=realistic)

        results = backtester.run(symbols, start_date, end_date, position_size)

        if 'error' in results:
            return jsonify({'error': results['error']}), 400

        # Additive data provenance: which provider served each symbol.
        results['data_sources'] = {
            symbol: _fetch_info(backtester.market_data, symbol)
            for symbol in symbols
        }
        # Run-level reproducibility provenance, parity with the async path.
        results['provenance'] = capture_run_provenance(seed)

        # Normalize dates + null any NaN summary metric, parity with the async
        # path — without this a NaN Sharpe/Sortino/Calmar would make jsonify
        # raise and the run would 500 with no diagnostic.
        return jsonify(_json_safe_report(results))
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

@backtest_bp.route('/models', methods=['GET'])
def list_models():
    """Available walk-forward models for desk-mode model selection.

    Returns ``{'models': [{'id','name','description'}, ...]}`` straight from
    desks.models.available_models() — the single source of truth the desk
    factory also uses. Surfaces the same 503 as desk mode when the desks
    package failed to import (so the frontend can disable the picker), rather
    than a 500.
    """
    if available_models is None:
        return jsonify({'error': 'Desk framework unavailable',
                        'reason': DESK_REGISTRY_IMPORT_ERROR}), 503
    return jsonify({'models': available_models()})

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