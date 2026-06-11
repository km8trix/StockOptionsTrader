"""
API routes for Paper Trading, Live Trading, Alerts, and Risk Management.
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify
import logging
import os

from gui.globals import paper_traders, alert_manager, risk_manager
from brokers.paper_trader import PaperTrader
from core.models import Asset, AssetType, OrderType

logger = logging.getLogger(__name__)

# Any failure to import the live broker must be LOUD: this module previously
# failed silently on Python 3.9 (``float | None`` syntax raised at import,
# which a bare ModuleNotFoundError catch never saw). Catch everything, keep
# the message, and log it at ERROR.
LIVE_BROKER_IMPORT_ERROR: str | None
try:
    from brokers.live_trader import LiveEtradeBroker
    LIVE_BROKER_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - import failure of any kind disables live trading
    LiveEtradeBroker = None
    LIVE_BROKER_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import LiveEtradeBroker — live trading is unavailable: %s',
        LIVE_BROKER_IMPORT_ERROR,
    )

trading_bp = Blueprint('trading', __name__, url_prefix='/api')

# ==================== TRADING SESSIONS ====================
active_traders = {}

@trading_bp.route('/trader/create', methods=['POST'])
def create_trader():
    """Create a new paper or live trading session"""
    try:
        data = request.json or {}
        trader_id = data.get('trader_id', 'default')
        mode = data.get('mode', 'paper')
        
        if mode == 'live':
            if LiveEtradeBroker is None:
                return jsonify({
                    'error': 'Live trading unavailable',
                    'reason': LIVE_BROKER_IMPORT_ERROR,
                }), 503

            active_traders[trader_id] = LiveEtradeBroker(
                consumer_key=os.getenv("ETRADE_CONSUMER_KEY"),
                consumer_secret=os.getenv("ETRADE_CONSUMER_SECRET"),
                access_token=os.getenv("ETRADE_ACCESS_TOKEN"),
                access_secret=os.getenv("ETRADE_ACCESS_SECRET"),
                account_id_key=os.getenv("ETRADE_ACCOUNT_ID_KEY")
            )
        else:
            initial_capital = float(data.get('initial_capital', 50000))
            active_traders[trader_id] = PaperTrader(initial_capital)
            
        return jsonify({'trader_id': trader_id, 'mode': mode, 'status': 'created'})
    except Exception:
        logger.error('Failed to create trading session', exc_info=True)
        return jsonify({'error': 'Failed to create trading session'}), 500

@trading_bp.route('/trader/<trader_id>/order', methods=['POST'])
def place_order(trader_id):
    """Place an order using the active trader instance"""
    try:
        if trader_id not in active_traders:
            return jsonify({'error': f'Trader {trader_id} not found'}), 404
            
        data = request.json or {}
        symbol = data.get('symbol', '').upper()
        action = data.get('action', 'BUY').upper()
        quantity = int(data.get('quantity', 0))
        price = float(data.get('price', 0))
        
        if not symbol or quantity <= 0:
            return jsonify({'error': 'Invalid parameters'}), 400
            
        trader = active_traders[trader_id]
        asset = Asset(symbol, AssetType.STOCK)
        order_type = OrderType.BUY if action == 'BUY' else OrderType.SELL
        
        order_id = trader.place_order(asset, order_type, quantity, limit_price=price if price > 0 else None)
        
        return jsonify({'order_id': order_id, 'status': 'placed'})
    except Exception:
        logger.error('Failed to place order', exc_info=True)
        return jsonify({'error': 'Failed to place order'}), 500

@trading_bp.route('/trader/<trader_id>/status', methods=['GET'])
def get_trader_status(trader_id):
    """Get status of active trader"""
    try:
        if trader_id not in active_traders:
            return jsonify({'error': f'Trader {trader_id} not found'}), 404
            
        trader = active_traders[trader_id]
        status = trader.get_portfolio_status()
        return jsonify(status)
    except Exception:
        logger.error('Failed to fetch trader status', exc_info=True)
        return jsonify({'error': 'Failed to retrieve trader status'}), 500

# ==================== ALERTS ====================

@trading_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Get active alerts"""
    try:
        alerts = alert_manager.get_alerts(unread_only=False)
        unread_count = len([a for a in alerts if not a.get('read', False)])
        
        return jsonify({
            'alerts': alerts,
            'unread_count': unread_count,
            'total_count': len(alerts)
        })
    except Exception:
        logger.error('Failed to fetch alerts', exc_info=True)
        return jsonify({'error': 'Failed to retrieve alerts'}), 500

@trading_bp.route('/alert/<int:alert_id>/read', methods=['POST'])
def mark_alert_read(alert_id):
    """Mark alert as read"""
    try:
        alert_manager.mark_read(alert_id)
        return jsonify({'message': 'Alert marked as read'})
    except Exception:
        logger.error('Failed to mark alert as read', exc_info=True)
        return jsonify({'error': 'Failed to mark alert as read'}), 500

@trading_bp.route('/price-target', methods=['POST'])
def add_price_target():
    """Add price target alert"""
    try:
        data = request.json or {}
        symbol = data.get('symbol')
        target_price = data.get('target_price')
        
        if not symbol or not target_price:
            return jsonify({'error': 'Missing symbol or target_price'}), 400
            
        alert = alert_manager.price_alert(
            symbol, 0, target_price,
            details={'type': 'price_target'}
        )
        return jsonify(alert)
    except Exception:
        logger.error('Failed to add price target alert', exc_info=True)
        return jsonify({'error': 'Failed to add price target'}), 500


# ==================== RISK MANAGEMENT ====================

@trading_bp.route('/risk/report', methods=['GET'])
def get_risk_report():
    """Get risk management report"""
    try:
        report = risk_manager.get_report()
        return jsonify(report)
    except Exception:
        logger.error('Failed to generate risk report', exc_info=True)
        return jsonify({'error': 'Failed to retrieve risk report'}), 500

@trading_bp.route('/risk/settings', methods=['GET', 'POST'])
def manage_risk_settings():
    """Get or update risk settings"""
    try:
        if request.method == 'GET':
            return jsonify({
                'max_position_size': risk_manager.max_position_size,
                'max_sector_exposure': risk_manager.max_sector_exposure,
                'max_daily_loss': risk_manager.max_daily_loss,
                'position_stop_loss': risk_manager.position_stop_loss,
            })
            
        data = request.json or {}
        if 'max_position_size' in data:
            risk_manager.max_position_size = data['max_position_size']
        if 'max_sector_exposure' in data:
            risk_manager.max_sector_exposure = data['max_sector_exposure']
        if 'max_daily_loss' in data:
            risk_manager.max_daily_loss = data['max_daily_loss']
        if 'position_stop_loss' in data:
            risk_manager.position_stop_loss = data['position_stop_loss']
            
        return jsonify({'message': 'Settings updated'})
    except Exception:
        logger.error('Failed to manage risk settings', exc_info=True)
        return jsonify({'error': 'Failed to update risk settings'}), 500
