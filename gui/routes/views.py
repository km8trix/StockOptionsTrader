# gui/routes/views.py
from flask import Blueprint, render_template

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    return render_template('index.html')

@views_bp.route('/backtest')
def backtest_page():
    return render_template('backtest.html')

@views_bp.route('/paper_trade')
def paper_trade_page():
    return render_template('paper_trade.html')

@views_bp.route('/analysis')
def analysis_page():
    return render_template('analysis.html')