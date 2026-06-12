# gui/app.py
"""Flask application factory for the Stock Options Trading System GUI.

Exposes :func:`create_app` (used by run_gui.py, tests, and — in Phase 4 —
gunicorn/Docker) plus :func:`get_server_config` for environment-driven
host/port/debug settings. There is intentionally no module-level ``app``
side effect.
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS

# utils/logging_config is being added in a parallel Phase 1 task; the app
# must still start if it has not landed yet.
try:
    from utils.logging_config import setup_logging
except ImportError:  # pragma: no cover - depends on parallel work landing
    setup_logging = None

logger = logging.getLogger(__name__)


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
    CORS(app)

    @app.context_processor
    def inject_app_version():
        """Expose the installed package version to templates (status bar)."""
        try:
            from importlib.metadata import version
            app_version = version('stock-options-trader')
        except Exception:  # pragma: no cover - metadata missing in some envs
            app_version = '0.1.0'
        return {'app_version': app_version}

    # Import blueprints inside the factory so importing gui.app stays cheap
    # and side-effect free until an app is actually built.
    from gui.routes.views import views_bp
    from gui.routes.api_analysis import analysis_bp
    from gui.routes.api_backtest import backtest_bp
    from gui.routes.api_trading import trading_bp
    from gui.routes.api_floor import floor_bp

    app.register_blueprint(views_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(backtest_bp)
    app.register_blueprint(trading_bp)
    app.register_blueprint(floor_bp)

    @app.route('/health')
    def health():
        """Liveness probe (Docker healthcheck target in Phase 4)."""
        return jsonify({'status': 'ok', 'service': 'stock-options-trader'}), 200

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
