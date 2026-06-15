# StockOptionsTrader

Personal quant trading platform: rigorous no-lookahead backtesting, walk-forward
validation, enforced risk management, and a paper-to-live pipeline through
E*TRADE (sole brokerage). The long-term architecture is a "trading floor" of
firm-style desks (Renaissance / Citadel / Jane Street personas) — see
[PLAN.md](PLAN.md) for the approved 9-phase roadmap and current status.

Legacy documentation lives in [`readme/`](readme/).

## Local development

Requires Python 3.13 (the dependency pins target it).

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q     # offline, deterministic suite
.venv/bin/python run_gui.py       # dev server on http://127.0.0.1:5001
```

## Docker

```bash
# .env must exist in the repo root (E*TRADE credentials; see the table
# below for the variable names). It is passed to the container at runtime
# only — never baked into the image.
docker compose up --build -d
```

The app listens on <http://localhost:5001>; container health is probed via
`GET /health`. The first build downloads several GB of Python dependencies
(openbb pulls a large tree) — subsequent builds reuse the cached layer.

### Single-worker constraint (important)

The container runs gunicorn with **`--workers 1`** by design. The background
JobManager registry, the in-memory OHLCV cache, and paper-trader instances
are all per-process state: with two or more workers, job-status polling
silently breaks (a job started in one worker is invisible to another) and
caches/paper portfolios split across processes. Scale concurrency with
gunicorn `--threads` only — do not raise the worker count. Within the single
worker, paper-trader operations (order placement, fills, status polls) are
thread-safe via a per-trader lock, so concurrent threads cannot double-fill
an order.

### Data persistence

SQLite state is written to `/data/trading_data.db` inside the container,
backed by the named volume `trading-data`. Data survives container rebuilds;
remove it with `docker volume rm stockoptionstrader_trading-data`.

## Environment variables

Names only — values belong in `.env` (git- and docker-ignored), never in
code, images, or compose files.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASK_HOST` | `127.0.0.1` (image: `0.0.0.0`) | Bind address |
| `FLASK_PORT` | `5001` | Bind port |
| `FLASK_DEBUG` | unset (off) | Dev-server debug mode; never enable in production |
| `SECRET_KEY` | insecure dev value | Flask session signing key; set a real one in production |
| `TRADING_DB_PATH` | `trading_data.db` (image: `/data/trading_data.db`) | SQLite file used by the trade DB, OHLCV cache, and options/IV store |
| `LOG_LEVEL` | `INFO` | Root logging level |
| `ETRADE_ENV` | `sandbox` | `sandbox` or `production` — selects the host and which consumer-key pair is read |
| `ETRADE_ALLOW_NETWORK` | unset | Must be `1` to allow **any** real E*TRADE network call (sandbox included) |
| `ETRADE_PRODUCTION_ACK` | unset | Must equal `I_UNDERSTAND_LIVE_TRADING` to construct a **production** auth manager |
| `ETRADE_SANDBOX_CONSUMER_KEY` / `_SECRET` | — | Sandbox OAuth consumer key/secret (used when `ETRADE_ENV=sandbox`) |
| `ETRADE_PROD_CONSUMER_KEY` / `_SECRET` | — | Production OAuth consumer key/secret (used when `ETRADE_ENV=production`) |
| `ETRADE_CONSUMER_KEY` / `_SECRET` | — | Generic fallback consumer key/secret (used only if the env-prefixed pair is unset) |
| `ETRADE_ACCOUNT_ID_KEY` | — | E*TRADE account id key (anchors the daily-loss gate + account-match guard) |

> **Live trading:** launch with **`./start.sh`** (it sources `.env`; the app
> never auto-loads it). The OAuth **access token/secret are obtained via the
> connect flow and stored in SQLite — they are not read from `.env`.** Full
> walkthrough (sandbox → production, the gates, daily re-auth, troubleshooting)
> is in **[docs/ETRADE_SETUP.md](docs/ETRADE_SETUP.md)**.

Cache behaviour (OHLCV coverage/staleness, options chains, IV history) is
keyed off `TRADING_DB_PATH`; there are no separate cache env knobs today.
