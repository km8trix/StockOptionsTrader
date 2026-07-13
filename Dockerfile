# StockOptionsTrader — production image (Phase 4)
#
# python:3.13-slim matches the development venv (3.13.x) exactly; the
# requirements.txt pins (pandas 3.0.3, numpy 2.4.6, ...) target 3.13 and
# will not install on older interpreters.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first: requirements.txt changes rarely relative to the
# application code, so this layer (several GB with openbb) stays cached.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (runtime modules only; see .dockerignore).
COPY core/ core/
COPY data/ data/
COPY brokers/ brokers/
COPY portfolio/ portfolio/
COPY backtesting/ backtesting/
COPY strategies/ strategies/
COPY desks/ desks/
COPY analysis/ analysis/
COPY execution/ execution/
COPY utils/ utils/
COPY gui/ gui/
COPY run_gui.py .

# Non-root runtime user. /data is created and chowned BEFORE switching to
# the app user so that a named volume mounted at /data inherits writable
# ownership from the image directory.
RUN useradd -r app \
    && mkdir -p /data \
    && chown -R app:app /data /app
USER app

# Runtime defaults — all overridable at run time (compose env_file / -e).
ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5001 \
    TRADING_DB_PATH=/data/trading_data.db \
    LOG_LEVEL=INFO

EXPOSE 5001

# slim has no curl; probe /health with the stdlib instead of installing one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=4).status == 200 else 1)"]

# IMPORTANT: --workers MUST stay 1.
#
# The JobManager registry (utils/jobs.py), the in-memory OHLCV cache, and
# paper-trader instances are all per-process state. With 2+ gunicorn
# workers, a backtest job started by one worker is invisible to the worker
# that serves the next status poll (jobs appear to vanish), and caches /
# paper portfolios silently split across processes. Scale concurrency with
# --threads only.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5001", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "gui.app:create_app()"]
