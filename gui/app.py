# gui/app.py
"""Flask application factory for the Stock Options Trading System GUI.

Exposes :func:`create_app` (used by run_gui.py, tests, and — in Phase 4 —
gunicorn/Docker) plus :func:`get_server_config` for environment-driven
host/port/debug settings. There is intentionally no module-level ``app``
side effect.
"""
from __future__ import annotations

import atexit
import logging
import os

from flask import Flask, Response, g, jsonify, request
from flask_cors import CORS

from utils import metrics
from utils.logging_config import (
    get_correlation_id, new_correlation_id, set_correlation_id)

# utils/logging_config is being added in a parallel Phase 1 task; the app
# must still start if it has not landed yet.
try:
    from utils.logging_config import setup_logging
except ImportError:  # pragma: no cover - depends on parallel work landing
    setup_logging = None

logger = logging.getLogger(__name__)

# Registered once per process (create_app runs many times in tests).
_shutdown_hook_registered = False


def _env_truthy(value: str | None) -> bool:
    """Return True only for explicitly truthy env values ('1' or 'true')."""
    return (value or '').strip().lower() in ('1', 'true')


def get_server_config() -> tuple[str, int, bool]:
    """Read Flask server settings from the environment.

    Returns:
        (host, port, debug). Defaults: 127.0.0.1:5001 with debug off.
        Docker overrides FLASK_HOST via env in Phase 4 — do not default
        to 0.0.0.0 here.
    """
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5001'))
    debug = _env_truthy(os.environ.get('FLASK_DEBUG'))
    return host, port, debug


def create_app(config: dict | None = None) -> Flask:
    """Application factory.

    Args:
        config: Optional mapping applied to ``app.config`` last, so tests
            can override anything (e.g. ``{'TESTING': True}``).
    """
    if setup_logging is not None:
        setup_logging()

    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-insecure-change-me')
    # Restrict CORS to localhost (and any explicitly configured origins). A
    # bare CORS(app) allows ALL origins, which on a money-moving order API is a
    # CSRF-equivalent exposure. Phase 4 Docker can widen this via CORS_ORIGINS.
    cors_origins = [o.strip() for o in os.environ.get(
        'CORS_ORIGINS',
        'http://localhost:5001,http://127.0.0.1:5001').split(',') if o.strip()]
    CORS(app, resources={r"/*": {"origins": cors_origins}})

    @app.context_processor
    def inject_app_version():
        """Expose the installed package version to templates (status bar)."""
        try:
            from importlib.metadata import version
            app_version = version('stock-options-trader')
        except Exception:  # pragma: no cover - metadata missing in some envs
            app_version = '0.1.0'
        return {'app_version': app_version}

    @app.context_processor
    def inject_glossary():
        """Expose the glossary to every template (offcanvas + explain() macro)."""
        from gui.glossary import GLOSSARY
        return {'GLOSSARY': GLOSSARY}

    # Import blueprints inside the factory so importing gui.app stays cheap
    # and side-effect free until an app is actually built.
    from gui.routes.views import views_bp
    from gui.routes.api_analysis import analysis_bp
    from gui.routes.api_backtest import backtest_bp
    from gui.routes.api_trading import trading_bp
    from gui.routes.api_floor import floor_bp
    from gui.routes.api_live import live_bp

    app.register_blueprint(views_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(backtest_bp)
    app.register_blueprint(trading_bp)
    app.register_blueprint(floor_bp)
    app.register_blueprint(live_bp)

    # Restart recovery: if the operator had the token keep-alive loop running
    # (last explicit intent was 'start') and the E*TRADE token is still
    # connected, resume the renew-only loop so a process restart does not
    # silently let the session idle out. Best-effort and short-circuiting —
    # a fresh process with no prior 'start' intent does nothing. It can only
    # ever resume a renew-only loop for a live token; it never trades.
    try:
        from gui.routes.api_live import resume_keepalive_if_desired
        resume_keepalive_if_desired()
    except Exception:  # pragma: no cover - never block app startup
        logger.warning('Keep-alive restart-recovery hook failed',
                       exc_info=True)

    # Graceful shutdown: stop background threads (keep-alive, scheduler) once
    # per process so `docker stop` / Ctrl-C tears them down cleanly. atexit
    # fires on gunicorn worker exit and dev-server KeyboardInterrupt alike.
    global _shutdown_hook_registered
    if not _shutdown_hook_registered:
        from gui.routes.api_live import shutdown_background_workers
        atexit.register(shutdown_background_workers)
        _shutdown_hook_registered = True

    @app.context_processor
    def inject_kill_switch_banner():
        """Kill-switch banner state for base.html (Phase 9).

        Server-rendered so the red banner can show on EVERY page without
        adding JS status polling anywhere but the Live page (the only
        poller). kill_switch_engaged() is best-effort and never raises:
        it reads the existing singleton, or the persisted SQLite state
        when TRADING_DB_PATH is set or the default db file already exists
        (so an engaged switch survives restarts), and degrades to False
        while the backend kill-switch surface has not landed.
        """
        from gui.routes.api_live import kill_switch_engaged
        return {'kill_switch_engaged': kill_switch_engaged()}

    # --- Observability: correlation ids + request metrics (Phase 4) ---
    @app.before_request
    def _bind_correlation_id():
        # Honor an upstream/proxy-supplied id; otherwise mint a fresh one.
        incoming = request.headers.get('X-Request-ID')
        if incoming:
            set_correlation_id(incoming)
            g.correlation_id = incoming
        else:
            g.correlation_id = new_correlation_id()

    @app.after_request
    def _record_request(response):
        response.headers['X-Request-ID'] = get_correlation_id()
        metrics.inc('http_requests_total',
                    {'status': f'{response.status_code // 100}xx'})
        return response

    @app.route('/health')
    def health():
        """Liveness probe (Docker healthcheck target in Phase 4)."""
        return jsonify({'status': 'ok', 'service': 'stock-options-trader'}), 200

    @app.route('/metrics')
    def metrics_endpoint():
        """Prometheus text exposition of in-process metrics (Phase 4 ops)."""
        return Response(metrics.render(),
                        mimetype='text/plain; version=0.0.4')

    @app.errorhandler(400)
    def bad_request(error):
        # Keep API errors JSON even when werkzeug raises BadRequest before a
        # route runs (e.g. request.json on an empty/malformed body).
        description = getattr(error, 'description', None)
        return jsonify({'error': description or 'Bad request'}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        # Full traceback goes to the server log ONLY — never to the client.
        app.logger.error('Internal server error: %s', error, exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    # Apply test/override config last so it wins over everything above.
    if config:
        app.config.update(config)

    return app


if __name__ == '__main__':
    host, port, debug = get_server_config()

    print("\n" + "=" * 70)
    print("Stock Options Trading System - Web GUI")
    print("=" * 70)
    print("\n🌐 Starting Modular Flask server...\n")
    print(f"📱 Access the GUI at: http://{host}:{port}")
    print("\n" + "=" * 70 + "\n")

    create_app().run(host=host, port=port, debug=debug)
