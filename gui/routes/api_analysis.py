# gui/routes/api_analysis.py
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback

from gui.globals import market_data

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')

@analysis_bp.route('/analyze/<symbol>')
def analyze_stock(symbol):
    try:
        symbol = symbol.upper()
        days_back = request.args.get('days', 252, type=int)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        data = market_data.fetch_stock_data(symbol, start_date, end_date)
        if data is None or data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
            
        data = market_data.calculate_indicators(data)
        recent_data = data.tail(50).copy()
        
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
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@analysis_bp.route('/chart/indicators', methods=['POST'])
def get_indicators_chart():
    try:
        data = request.json or {}
        symbol = data.get('symbol', 'AAPL')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        df = market_data.fetch_stock_data(symbol, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return jsonify({'error': 'No data available'}), 400
            
        df = market_data.calculate_indicators(df)
        
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Price', 'RSI', 'MACD'), vertical_spacing=0.05)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='white')), row=1, col=1)
        
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash='dash', line_color='red', row='2', col='1')
            fig.add_hline(y=30, line_dash='dash', line_color='green', row='2', col='1')
            
        if 'macd' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')), row=3, col=1)
            if 'macd_signal' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')), row=3, col=1)
                
        fig.update_layout(height=800, template='plotly_dark', title_text=f'Technical Indicators - {symbol}')
        return jsonify({'chart': fig.to_json()})
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# You can move the rest of your charting/pricing routes (like /chart/price, /price_option) here!