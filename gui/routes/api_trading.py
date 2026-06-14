"""
API routes for Paper Trading, Live Trading, Alerts, and Risk Management.
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify
import logging
import math
import os
import threading

from gui.globals import alert_manager, risk_manager
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
# Guards registration only. Under gunicorn --threads, two concurrent
# create_trader calls could otherwise silently replace a trader mid-mutation.
# Reads stay lock-free: dict lookup is atomic, and entries are only ever
# added/replaced under this lock.
_active_traders_lock = threading.Lock()

@trading_bp.route('/trader/create', methods=['POST'])
def create_trader():
    """Create a new paper or live trading session"""
    try:
        data = request.get_json(silent=True) or {}
        trader_id = data.get('trader_id', 'default')
        mode = data.get('mode', 'paper')
        replace = bool(data.get('replace', False))

        if mode == 'live':
            if LiveEtradeBroker is None:
                return jsonify({
                    'error': 'Live trading unavailable',
                    'reason': LIVE_BROKER_IMPORT_ERROR,
                }), 503

            # Phase 9: the broker authenticates through the shared C16
            # auth manager (gui/routes/api_live owns the singleton; the
            # manager reads ETRADE_* config itself) instead of raw env
            # tokens, and the shared kill switch + audit log MUST ride
            # along: EtradeClient only gates preview/place on a kill
            # switch it was given. Imported lazily — api_live and this
            # module only reference each other inside functions (no
            # import cycle).
            from gui.routes.api_live import (
                AUDIT_IMPORT_ERROR, KILL_SWITCH_IMPORT_ERROR,
                auth_unavailable_reason, get_audit_log, get_auth_manager,
                get_kill_switch)
            auth_manager = get_auth_manager()
            if auth_manager is None:
                return jsonify({
                    'error': 'Live trading unavailable',
                    'reason': auth_unavailable_reason(),
                }), 503
            # Fail CLOSED on the safety surfaces: a live broker built with
            # kill_switch=None has NO preview/place gate (EtradeClient only
            # checks a switch it was handed), and audit=None would trade
            # unrecorded. Match /api/live/status, which already 503s loudly
            # when the kill switch is unavailable.
            kill_switch = get_kill_switch()
            if kill_switch is None:
                return jsonify({
                    'error': 'Live trading unavailable',
                    'reason': (KILL_SWITCH_IMPORT_ERROR
                               or 'KillSwitch construction failed'),
                }), 503
            audit_log = get_audit_log()
            if audit_log is None:
                return jsonify({
                    'error': 'Live trading unavailable',
                    'reason': (AUDIT_IMPORT_ERROR
                               or 'AuditLog construction failed'),
                }), 503
            # Fail CLOSED on a missing target account: a live broker built with
            # account_id_key=None would route every order to account 'None'
            # (a typed E*TRADE rejection), not a real account — refuse it
            # explicitly rather than rely on that downstream rejection.
            account_id_key = os.getenv("ETRADE_ACCOUNT_ID_KEY")
            if not account_id_key:
                return jsonify({
                    'error': 'Live trading unavailable',
                    'reason': 'ETRADE_ACCOUNT_ID_KEY is not set — refusing to '
                              'build a live broker without a target account',
                }), 503
            trader = LiveEtradeBroker(
                auth=auth_manager,
                account_id_key=account_id_key,
                kill_switch=kill_switch,
                audit=audit_log,
            )
        else:
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
            
        data = request.get_json(silent=True) or {}
        symbol = str(data.get('symbol') or '').strip().upper()
        action = data.get('action', 'BUY').upper()

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
            
        trader = active_traders[trader_id]
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
