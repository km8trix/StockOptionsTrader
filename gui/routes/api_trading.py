"""
API routes for Paper Trading, Alerts, and Risk Management.
"""
from __future__ import annotations

import logging
import math
import threading

from flask import Blueprint, jsonify, request

from brokers.paper_trader import PaperTrader
from core.models import Asset, AssetType, OrderType
from gui.globals import alert_manager, risk_manager

logger = logging.getLogger(__name__)

trading_bp = Blueprint('trading', __name__, url_prefix='/api')

# ==================== TRADING SESSIONS ====================
active_traders: dict[str, PaperTrader] = {}
# Guards registration only. Under gunicorn --threads, two concurrent
# create_trader calls could otherwise silently replace a trader mid-mutation.
# Reads stay lock-free: dict lookup is atomic, and entries are only ever
# added/replaced under this lock.
_active_traders_lock = threading.Lock()

@trading_bp.route('/trader/create', methods=['POST'])
def create_trader():
    """Create a paper-trading session.

    Live order entry deliberately has one route only: the guarded
    ``/api/live/order/preview`` -> ``/api/live/order/place`` workflow.  The
    former generic trader route used ``LiveEtradeBroker.place_order()``, which
    previewed and placed in one request and therefore bypassed the GUI's
    explicit preview confirmation and market-hours check.
    """
    try:
        data = request.get_json(silent=True) or {}
        trader_id = data.get('trader_id', 'default')
        mode = str(data.get('mode', 'paper') or '').strip().lower()
        replace = bool(data.get('replace', False))

        if mode == 'live':
            return jsonify({
                'error': 'Legacy live trading sessions are disabled',
                'detail': 'Use /api/live/order/preview and then '
                          '/api/live/order/place with the returned order_ref.',
            }), 400
        if mode != 'paper':
            return jsonify({'error': "'mode' must be 'paper'"}), 400

        initial_capital = float(data.get('initial_capital', 50000))
        trader = PaperTrader(initial_capital)

        with _active_traders_lock:
            if trader_id in active_traders and not replace:
                return jsonify({
                    'error': f'Trader {trader_id} already exists',
                    'detail': "Pass 'replace': true to overwrite the "
                              'existing trading session.',
                }), 409
            active_traders[trader_id] = trader

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

        trader = active_traders[trader_id]
        # Defense in depth for long-lived processes/tests that may already
        # have inserted another broker in the module-level registry.  This
        # generic route must never become an alternate live-order surface.
        if not isinstance(trader, PaperTrader):
            return jsonify({
                'error': 'This endpoint accepts paper-trading sessions only',
                'detail': 'Use /api/live/order/preview for live orders.',
            }), 409

        data = request.get_json(silent=True) or {}
        symbol = str(data.get('symbol') or '').strip().upper()
        action = str(data.get('action') or 'BUY').strip().upper()
        if action not in ('BUY', 'SELL'):
            return jsonify({'error': "'action' must be BUY or SELL"}), 400

        # Robust parsing (mirrors api_live's _to_int/_to_float): reject a
        # fractional quantity instead of silently truncating int(1.9)->1, and
        # reject a negative or non-finite (NaN/Inf) price instead of routing it.
        try:
            qty_raw = float(data.get('quantity', 0))
        except (TypeError, ValueError):
            return jsonify({'error': "'quantity' must be a positive integer"}), 400
        if not math.isfinite(qty_raw) or qty_raw != int(qty_raw) or qty_raw <= 0:
            return jsonify({'error': "'quantity' must be a positive integer"}), 400
        quantity = int(qty_raw)

        try:
            price = float(data.get('price', 0))
        except (TypeError, ValueError):
            return jsonify({'error': "'price' must be a non-negative number"}), 400
        if not math.isfinite(price) or price < 0:
            return jsonify({'error': "'price' must be a finite, non-negative number"}), 400

        if not symbol:
            return jsonify({'error': "'symbol' is required"}), 400
            
        asset = Asset(symbol, AssetType.STOCK)
        order_type = OrderType.BUY if action == 'BUY' else OrderType.SELL
        
        order_id = trader.place_order(asset, order_type, quantity, limit_price=price if price > 0 else None)
        
        return jsonify({'order_id': order_id, 'status': 'placed'})
    except Exception:
        logger.error('Failed to place order', exc_info=True)
        return jsonify({'error': 'Failed to place order'}), 500

@trading_bp.route('/trader/<trader_id>/orders', methods=['GET'])
def list_pending_orders(trader_id):
    """List the trader's pending (unfilled) orders with their ids.

    The portfolio status endpoint only exposes a pending-order COUNT; the
    paper-trading UI needs the actual orders to offer per-order cancel.
    Brokers without a pending_orders queue degrade to an empty list.
    """
    try:
        if trader_id not in active_traders:
            return jsonify({'error': f'Trader {trader_id} not found'}), 404

        trader = active_traders[trader_id]
        pending = getattr(trader, 'pending_orders', None) or []
        orders = [
            {
                'order_id': order.order_id,
                'symbol': order.asset.symbol,
                'side': order.order_type.value.upper(),
                'quantity': order.quantity,
                'limit_price': order.price,  # None == market order
                'timestamp': order.timestamp.isoformat()
                if hasattr(order.timestamp, 'isoformat') else str(order.timestamp),
            }
            for order in pending
        ]
        return jsonify({'orders': orders, 'count': len(orders)})
    except Exception:
        logger.error('Failed to list pending orders', exc_info=True)
        return jsonify({'error': 'Failed to retrieve pending orders'}), 500


@trading_bp.route('/trader/<trader_id>/order/<order_id>/cancel', methods=['POST'])
def cancel_pending_order(trader_id, order_id):
    """Cancel one pending order by id."""
    try:
        if trader_id not in active_traders:
            return jsonify({'error': f'Trader {trader_id} not found'}), 404

        trader = active_traders[trader_id]
        if trader.cancel_order(order_id):
            return jsonify({'message': 'Order cancelled', 'order_id': order_id})
        return jsonify({'error': f'Order {order_id} not found'}), 404
    except Exception:
        logger.error('Failed to cancel order %s', order_id, exc_info=True)
        return jsonify({'error': 'Failed to cancel order'}), 500


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
        data = request.get_json(silent=True) or {}
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
            
        data = request.get_json(silent=True) or {}
        # Each risk limit is a fraction of the portfolio: numeric, finite,
        # in (0, 1]. Reject bad input rather than silently setting a limit
        # that would disable or invert downstream risk gates.
        for key in ('max_position_size', 'max_sector_exposure',
                    'max_daily_loss', 'position_stop_loss'):
            if key not in data:
                continue
            try:
                value = float(data[key])
            except (TypeError, ValueError):
                return jsonify({'error': f"'{key}' must be a number"}), 400
            if not math.isfinite(value) or not 0 < value <= 1:
                return jsonify(
                    {'error': f"'{key}' must be in range (0, 1]"}), 400
            setattr(risk_manager, key, value)
            
        return jsonify({'message': 'Settings updated'})
    except Exception:
        logger.error('Failed to manage risk settings', exc_info=True)
        return jsonify({'error': 'Failed to update risk settings'}), 500
