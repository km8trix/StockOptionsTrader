"""
API routes for the Live Trading page (Phase 9, contract C20).

Frontend-owned routes over the backend Phase 9 surfaces:

* C16  brokers/etrade_auth.EtradeAuthManager  (OAuth 1.0a lifecycle)
* C17  utils/kill_switch.KillSwitch           (persistent, audit-logged)
* C18  utils/audit.AuditLog                   (append-only sha256 chain)
* C19  brokers/reconcile.reconcile            (local book vs broker)

Every backend import is guarded with the 503-with-reason pattern
(precedent: api_trading.LIVE_BROKER_IMPORT_ERROR): a missing or broken
surface returns ``503 {'error': ..., 'reason': <import error>}`` instead of
crashing the app, so this blueprint registers cleanly while the parallel
backend task lands.

NO route here performs network I/O itself — the auth manager and broker own
their (injected) transports, and nothing in this module ever constructs an
HTTP session.

Response shapes consumed by gui/static/js/live.js:

  GET  /api/live/status      -> {'auth': <C16 status>, 'env': str|None,
                                 'kill_switch': {'engaged': bool},
                                 'reconciliation': <last result>|None}
  POST /api/live/auth/start  -> {'authorize_url': str, 'auth': <C16 status>}
  POST /api/live/auth/verifier {code}      -> {'auth': <C16 status>}
  POST /api/live/auth/renew  (additive)    -> {'renewed': bool, 'auth': ...}
  POST /api/live/auth/disconnect (additive)-> {'auth': <C16 status>}
  POST /api/live/killswitch {engaged, reason} -> {'engaged': bool}
  GET  /api/live/audit?limit=&offset=&event_type=&verify=
        -> {'entries': [...], 'limit', 'offset', 'event_type'[, 'verify']}
  POST /api/live/reconcile   -> <C19 result> + {'kill_switch_engaged': bool}
  GET  /api/live/orders (additive)         -> {'orders': [...], 'count': int}
  POST /api/live/orders/<id>/cancel (additive)
  GET  /api/live/scheduler   -> <LiveScheduler.status()> + interval_minutes
  POST /api/live/scheduler {action: 'start'|'stop', interval_minutes?}
        -> same shape as GET (503 until a scheduler is configured)
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request
import logging
import os
import threading

logger = logging.getLogger(__name__)

# ==================== GUARDED BACKEND IMPORTS (503 pattern) ====================

ETRADE_AUTH_IMPORT_ERROR: str | None
try:
    from brokers.etrade_auth import EtradeAuthManager, EtradeNotConfigured
    ETRADE_AUTH_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - any import failure disables the surface
    EtradeAuthManager = None
    ETRADE_AUTH_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import EtradeAuthManager — live auth is unavailable: %s',
        ETRADE_AUTH_IMPORT_ERROR,
    )

    class EtradeNotConfigured(Exception):
        """Placeholder so except-clauses stay valid while the backend
        surface is absent (never raised through this fallback)."""

KILL_SWITCH_IMPORT_ERROR: str | None
try:
    from utils.kill_switch import KillSwitch
    KILL_SWITCH_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    KillSwitch = None
    KILL_SWITCH_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import KillSwitch — kill switch is unavailable: %s',
        KILL_SWITCH_IMPORT_ERROR,
    )

AUDIT_IMPORT_ERROR: str | None
try:
    from utils.audit import AuditLog
    AUDIT_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    AuditLog = None
    AUDIT_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import AuditLog — audit log is unavailable: %s',
        AUDIT_IMPORT_ERROR,
    )

RECONCILE_IMPORT_ERROR: str | None
try:
    from brokers.reconcile import reconcile
    RECONCILE_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    reconcile = None
    RECONCILE_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import reconcile — reconciliation is unavailable: %s',
        RECONCILE_IMPORT_ERROR,
    )

SCHEDULER_IMPORT_ERROR: str | None
try:
    from utils.scheduler import LiveScheduler
    SCHEDULER_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    LiveScheduler = None
    SCHEDULER_IMPORT_ERROR = f'{type(e).__name__}: {e}'
    logger.error(
        'Failed to import LiveScheduler — the scheduler is unavailable: %s',
        SCHEDULER_IMPORT_ERROR,
    )

live_bp = Blueprint('live', __name__, url_prefix='/api/live')

# ==================== SHARED SINGLETONS ====================
# One auth manager / kill switch / audit log per GUI process. The kill
# switch and audit log share the trading SQLite file (TRADING_DB_PATH,
# same resolution as gui.globals.get_db) so state survives restarts.

_singleton_lock = threading.Lock()
_auth_manager = None
_kill_switch = None
_audit_log = None
# Construction-time failure (e.g. EtradeNotConfigured for a production env
# without the explicit ack gate) — preserved as the 503 reason.
_auth_construct_error: str | None = None

# Last reconciliation result for GET /api/live/status (per-process; the
# durable record is the audit log, which the backend live session writes).
_last_reconciliation: dict | None = None


def _live_db_path() -> str:
    """SQLite path for kill-switch/audit state (mirrors gui.globals)."""
    return os.environ.get('TRADING_DB_PATH') or 'trading_data.db'


def get_auth_manager():
    """Shared EtradeAuthManager, or None when the surface is unavailable.

    The manager reads its own ETRADE_* configuration from the environment
    (C16: status() reports 'unconfigured' when consumer keys are absent) —
    this module never reads or logs key material itself.
    """
    global _auth_manager, _auth_construct_error
    if EtradeAuthManager is None:
        return None
    if _auth_manager is None:
        with _singleton_lock:
            if _auth_manager is None:
                try:
                    _auth_manager = EtradeAuthManager(
                        db_path=_live_db_path())
                except Exception as e:
                    _auth_construct_error = f'{type(e).__name__}: {e}'
                    logger.error('EtradeAuthManager construction failed',
                                 exc_info=True)
                    return None
    return _auth_manager


def get_kill_switch():
    """Shared KillSwitch, or None when the surface is unavailable."""
    global _kill_switch
    if KillSwitch is None:
        return None
    if _kill_switch is None:
        with _singleton_lock:
            if _kill_switch is None:
                try:
                    _kill_switch = KillSwitch(_live_db_path())
                except Exception:
                    logger.error('KillSwitch construction failed',
                                 exc_info=True)
                    return None
    return _kill_switch


def get_audit_log():
    """Shared AuditLog, or None when the surface is unavailable."""
    global _audit_log
    if AuditLog is None:
        return None
    if _audit_log is None:
        with _singleton_lock:
            if _audit_log is None:
                try:
                    _audit_log = AuditLog(_live_db_path())
                except Exception:
                    logger.error('AuditLog construction failed', exc_info=True)
                    return None
    return _audit_log


def auth_unavailable_reason() -> str:
    """Reason string for 503s when the auth surface is unusable."""
    return (ETRADE_AUTH_IMPORT_ERROR or _auth_construct_error
            or 'EtradeAuthManager construction failed')


# The process-wide LiveScheduler. Deliberately NOT auto-constructed: a
# scheduler needs a LiveTradingSession (desk + broker + portfolio +
# data_fn), which only the operator's live wiring can provide. Whoever
# builds that session installs its scheduler here via set_scheduler();
# until then the /scheduler routes answer 503 'unconfigured' — automation
# that materializes by itself is exactly what Phase 9 ruled out.
_scheduler = None


def set_scheduler(scheduler) -> None:
    """Install (or clear, with None) the shared LiveScheduler."""
    global _scheduler
    with _singleton_lock:
        _scheduler = scheduler


def get_scheduler():
    """The installed LiveScheduler, or None while unconfigured."""
    return _scheduler


def scheduler_unavailable_reason() -> str:
    """Reason string for 503s when the scheduler surface is unusable."""
    return (SCHEDULER_IMPORT_ERROR
            or 'No live scheduler is configured — create a live trading '
               'session (which installs its scheduler) first.')


def kill_switch_engaged() -> bool:
    """Best-effort engaged state for the base-template banner.

    Called by the app-wide context processor on every page render, so it
    must never raise and must stay cheap. It reads through the shared
    singleton when one already exists; otherwise it constructs one when a
    persistent DB is in play — TRADING_DB_PATH explicitly configured (the
    Docker/production case) OR the default db file already existing on
    disk (a dev server that has traded before). That keeps the 'red banner
    on EVERY page' guarantee across process restarts in both cases, while
    a page render never CREATES a db file: with no configured path and no
    existing default file there is nothing persisted to read, and the Live
    page's own status poll (the only poller) lights the banner up.
    """
    try:
        if _kill_switch is not None:
            return bool(_kill_switch.engaged())
        if KillSwitch is None:
            return False
        if (not os.environ.get('TRADING_DB_PATH')
                and not os.path.exists(_live_db_path())):
            return False
        ks = get_kill_switch()
        return bool(ks.engaged()) if ks is not None else False
    except Exception:
        logger.error('Kill-switch banner state check failed', exc_info=True)
        return False


def _find_live_broker():
    """The active LiveEtradeBroker session, if one exists.

    Looks through api_trading.active_traders (imported lazily — both
    modules reference each other only inside functions, so there is no
    import cycle). Returns None when live trading is unavailable or no
    live session has been created.
    """
    from gui.routes import api_trading

    if api_trading.LiveEtradeBroker is None:
        return None
    for trader in api_trading.active_traders.values():
        if isinstance(trader, api_trading.LiveEtradeBroker):
            return trader
    return None


def _local_state(broker) -> tuple[dict, float]:
    """Local (system-side) positions/cash for reconciliation (C19).

    The live session owns the canonical local book; the broker wrapper
    exposes it as ``local_positions`` / ``local_cash``. A session that
    tracks nothing locally reconciles as an empty book — conservative in
    the fail-safe direction (any broker-side position becomes a mismatch
    that engages the kill switch rather than passing silently).
    """
    positions = getattr(broker, 'local_positions', None) or {}
    cash = getattr(broker, 'local_cash', None)
    return dict(positions), float(cash if cash is not None else 0.0)


def _unavailable(what: str, reason: str | None):
    return jsonify({'error': f'{what} unavailable', 'reason': reason}), 503


def _int_arg(name: str, default: int, lo: int, hi: int):
    """Parse a bounded int query param; None signals a 400 to the caller."""
    raw = request.args.get(name)
    if raw is None or raw == '':
        return default
    try:
        value = int(raw)
    except ValueError:
        return None
    if not lo <= value <= hi:
        return None
    return value


# ==================== STATUS ====================

@live_bp.route('/status', methods=['GET'])
def live_status():
    """Auth status + env + kill switch + last reconciliation (C20).

    The live page is the ONLY page that polls this endpoint; an ambiguous
    kill-switch state must never read as fine, so an unavailable kill
    switch is a loud 503 here rather than a null field.
    """
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    ks = get_kill_switch()
    if ks is None:
        return _unavailable('Kill switch', KILL_SWITCH_IMPORT_ERROR)
    try:
        auth = manager.status()
        return jsonify({
            'auth': auth,
            'env': auth.get('env'),
            'kill_switch': {'engaged': bool(ks.engaged())},
            'reconciliation': _last_reconciliation,
        })
    except Exception:
        logger.error('Failed to retrieve live status', exc_info=True)
        return jsonify({'error': 'Failed to retrieve live status'}), 500


# ==================== AUTH LIFECYCLE (C16) ====================

@live_bp.route('/auth/start', methods=['POST'])
def auth_start():
    """Begin the OAuth flow. ONLY ever called from an explicit user click —
    the live page never auto-starts auth on load."""
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    try:
        authorize_url = manager.start_auth()
    except EtradeNotConfigured as e:
        return jsonify({'error': 'E*TRADE not configured',
                        'reason': str(e)}), 503
    except Exception:
        logger.error('Failed to start E*TRADE authorization', exc_info=True)
        return jsonify({'error': 'Failed to start authorization'}), 500
    return jsonify({'authorize_url': authorize_url, 'auth': manager.status()})


@live_bp.route('/auth/verifier', methods=['POST'])
def auth_verifier():
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    data = request.get_json(silent=True) or {}
    code = str(data.get('code') or '').strip()
    if not code:
        return jsonify({'error': 'Verifier code is required'}), 400
    try:
        status = manager.submit_verifier(code)
    except EtradeNotConfigured as e:
        return jsonify({'error': 'E*TRADE not configured',
                        'reason': str(e)}), 503
    except Exception:
        logger.error('Failed to submit verifier code', exc_info=True)
        return jsonify({'error': 'Failed to submit verifier code'}), 500
    return jsonify({'auth': status})


@live_bp.route('/auth/renew', methods=['POST'])
def auth_renew():
    """Additive beyond C20's minimum: the connected-state Renew button."""
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    try:
        renewed = bool(manager.renew())
        return jsonify({'renewed': renewed, 'auth': manager.status()})
    except EtradeNotConfigured as e:
        return jsonify({'error': 'E*TRADE not configured',
                        'reason': str(e)}), 503
    except Exception:
        logger.error('Failed to renew E*TRADE token', exc_info=True)
        return jsonify({'error': 'Failed to renew token'}), 500


@live_bp.route('/auth/disconnect', methods=['POST'])
def auth_disconnect():
    """Additive beyond C20's minimum: the connected-state Disconnect button."""
    manager = get_auth_manager()
    if manager is None:
        return _unavailable('Live trading', auth_unavailable_reason())
    try:
        manager.disconnect()
        return jsonify({'auth': manager.status()})
    except Exception:
        logger.error('Failed to disconnect E*TRADE session', exc_info=True)
        return jsonify({'error': 'Failed to disconnect'}), 500


# ==================== KILL SWITCH (C17) ====================

@live_bp.route('/killswitch', methods=['POST'])
def set_kill_switch():
    """Engage/disengage the kill switch. Engaging REQUIRES a reason; every
    flip is audit-logged by the KillSwitch itself (C17)."""
    ks = get_kill_switch()
    if ks is None:
        return _unavailable('Kill switch', KILL_SWITCH_IMPORT_ERROR)
    data = request.get_json(silent=True) or {}
    engaged = data.get('engaged')
    if not isinstance(engaged, bool):
        return jsonify({'error': "'engaged' must be a boolean"}), 400
    reason = str(data.get('reason') or '').strip()
    try:
        if engaged:
            if not reason:
                return jsonify(
                    {'error': 'A reason is required to engage the '
                              'kill switch'}), 400
            ks.engage(reason, 'gui')
        else:
            ks.disengage('gui')
        return jsonify({'engaged': bool(ks.engaged())})
    except Exception:
        logger.error('Failed to flip kill switch', exc_info=True)
        return jsonify({'error': 'Failed to update kill switch'}), 500


# ==================== AUDIT LOG (C18) ====================

@live_bp.route('/audit', methods=['GET'])
def audit_entries():
    """Paged audit entries; '?verify=1' is the additive query param that
    also runs the hash-chain verification (C20 allows this addition)."""
    log = get_audit_log()
    if log is None:
        return _unavailable('Audit log', AUDIT_IMPORT_ERROR)

    limit = _int_arg('limit', default=50, lo=1, hi=500)
    if limit is None:
        return jsonify(
            {'error': "'limit' must be an integer between 1 and 500"}), 400
    offset = _int_arg('offset', default=0, lo=0, hi=10_000_000)
    if offset is None:
        return jsonify({'error': "'offset' must be a non-negative "
                                 'integer'}), 400
    event_type = (request.args.get('event_type') or '').strip() or None

    try:
        payload = {
            'entries': log.entries(limit=limit, offset=offset,
                                   event_type=event_type),
            'limit': limit,
            'offset': offset,
            'event_type': event_type,
        }
        if (request.args.get('verify') or '').lower() in ('1', 'true'):
            payload['verify'] = log.verify_chain()
        return jsonify(payload)
    except Exception:
        logger.error('Failed to read audit log', exc_info=True)
        return jsonify({'error': 'Failed to read audit log'}), 500


# ==================== RECONCILIATION (C19) ====================

@live_bp.route('/reconcile', methods=['POST'])
def run_reconcile():
    """Run reconciliation against the active live broker session.

    On mismatches this route engages the kill switch itself (C19: the
    caller engages on not-ok) and says so in the response, so the UI can
    show the red mismatch table together with the 'kill switch engaged'
    note in one round trip.
    """
    global _last_reconciliation
    if reconcile is None:
        return _unavailable('Reconciliation', RECONCILE_IMPORT_ERROR)
    ks = get_kill_switch()
    if ks is None:
        return _unavailable('Kill switch', KILL_SWITCH_IMPORT_ERROR)

    broker = _find_live_broker()
    if broker is None:
        return jsonify({
            'error': 'No live broker session',
            'detail': 'Create a live trading session before reconciling.',
        }), 409

    try:
        local_positions, local_cash = _local_state(broker)
        result = dict(reconcile(local_positions, local_cash, broker))
        if not result.get('ok'):
            mismatches = result.get('mismatches') or []
            ks.engage(
                f'Reconciliation mismatch ({len(mismatches)} mismatch(es))',
                'reconcile',
            )
            result['kill_switch_engaged'] = True
            logger.error('Reconciliation found %d mismatch(es) — kill switch '
                         'engaged', len(mismatches))
        else:
            result['kill_switch_engaged'] = False
        _last_reconciliation = result
        return jsonify(result)
    except Exception:
        logger.error('Reconciliation failed', exc_info=True)
        return jsonify({'error': 'Reconciliation failed'}), 500


# ==================== WORKING ORDERS (patient executor) ====================

@live_bp.route('/orders', methods=['GET'])
def working_orders():
    """Patient-executor working orders (additive beyond C20's minimum).

    Shaped to the ExecutionReport/steps contract: each order carries
    instrument, side, quantity, limit_price, steps (list of repricing
    steps so far), started_at and remaining_seconds/expires_at. With no
    live session — or a broker without a patient executor — this degrades
    to an empty list and the page shows its empty state.
    """
    broker = _find_live_broker()
    orders: list = []
    if broker is not None:
        getter = getattr(broker, 'working_orders', None)
        if callable(getter):
            try:
                orders = list(getter())
            except Exception:
                logger.error('Failed to list working orders', exc_info=True)
                return jsonify({'error': 'Failed to list working orders'}), 500
    return jsonify({'orders': orders, 'count': len(orders)})


@live_bp.route('/orders/<order_id>/cancel', methods=['POST'])
def cancel_working_order(order_id):
    """Cancel one working order on the live broker (cancel stays allowed
    even while the kill switch is engaged — C17 blocks placement only)."""
    broker = _find_live_broker()
    if broker is None:
        return jsonify({
            'error': 'No live broker session',
            'detail': 'There is no live session whose orders could be '
                      'cancelled.',
        }), 409
    try:
        if broker.cancel_order(order_id):
            return jsonify({'message': 'Order cancelled',
                            'order_id': order_id})
        return jsonify({'error': f'Order {order_id} not found'}), 404
    except Exception:
        logger.error('Failed to cancel working order %s', order_id,
                     exc_info=True)
        return jsonify({'error': 'Failed to cancel order'}), 500


# ==================== MARKET-HOURS SCHEDULER ====================

#: Bounds for the GUI-settable evaluation interval (minutes).
SCHEDULER_INTERVAL_MIN = 1
SCHEDULER_INTERVAL_MAX = 240


def _scheduler_payload(scheduler) -> dict:
    """LiveScheduler.status() plus the configured interval (one shape for
    GET and POST so live.js renders both identically)."""
    payload = dict(scheduler.status())
    payload['interval_minutes'] = getattr(scheduler, 'interval_minutes', None)
    return payload


@live_bp.route('/scheduler', methods=['GET'])
def scheduler_status():
    """Scheduler state for the Live page card.

    503 until a scheduler is configured (set_scheduler) — the GUI never
    constructs a trading loop on its own.
    """
    scheduler = get_scheduler()
    if scheduler is None:
        return _unavailable('Scheduler', scheduler_unavailable_reason())
    try:
        return jsonify(_scheduler_payload(scheduler))
    except Exception:
        logger.error('Failed to read scheduler status', exc_info=True)
        return jsonify({'error': 'Failed to read scheduler status'}), 500


@live_bp.route('/scheduler', methods=['POST'])
def scheduler_action():
    """Start/stop the scheduler ({action: 'start'|'stop'}).

    'start' optionally takes interval_minutes (an int, 1-240) applied
    before the loop starts. The scheduler itself audits every start/stop/
    pause; pause states (kill switch, circuit breaker, error storm) can
    ONLY be cleared by an explicit 'start' — there is no 'resume'.
    """
    scheduler = get_scheduler()
    if scheduler is None:
        return _unavailable('Scheduler', scheduler_unavailable_reason())
    data = request.get_json(silent=True) or {}
    action = data.get('action')
    if action not in ('start', 'stop'):
        return jsonify(
            {'error': "'action' must be 'start' or 'stop'"}), 400
    interval = data.get('interval_minutes')
    if interval is not None:
        if action != 'start':
            return jsonify({'error': "'interval_minutes' only applies to "
                                     "action 'start'"}), 400
        if (not isinstance(interval, int) or isinstance(interval, bool)
                or not SCHEDULER_INTERVAL_MIN <= interval
                <= SCHEDULER_INTERVAL_MAX):
            return jsonify(
                {'error': f"'interval_minutes' must be an integer between "
                          f'{SCHEDULER_INTERVAL_MIN} and '
                          f'{SCHEDULER_INTERVAL_MAX}'}), 400
    try:
        if action == 'start':
            if interval is not None:
                scheduler.interval_minutes = interval
            scheduler.start()
        else:
            scheduler.stop()
        return jsonify(_scheduler_payload(scheduler))
    except Exception:
        logger.error('Scheduler %s failed', action, exc_info=True)
        return jsonify({'error': f'Scheduler {action} failed'}), 500
