# gui/routes/views.py
from flask import Blueprint, render_template, request

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    return render_template('index.html')

@views_bp.route('/backtest')
def backtest_page():
    # ws=production reveals Desk/Fund modes (Production workspace); the default
    # (Sandbox) shows strategy backtests only.
    return render_template('backtest.html', ws=request.args.get('ws'))

@views_bp.route('/paper_trade')
def paper_trade_page():
    return render_template('paper_trade.html')

@views_bp.route('/analysis')
def analysis_page():
    return render_template('analysis.html')

@views_bp.route('/floor')
def floor_page():
    return render_template('floor.html')

@views_bp.route('/charts')
def charts_page():
    # Production live-market chart (Lightweight Charts). Always a Production
    # page, so base.html resolves the PRODUCTION section without ?ws.
    return render_template('charts.html')

@views_bp.route('/live')
def live_page():
    return render_template('live.html')

@views_bp.route('/trading_floor')
def trading_floor_page():
    # Back-compat alias for the Stage 1 nav link; same page as /floor.
    return render_template('floor.html')
