# StockOptionsTrader — Master Plan (approved 2026-06-11)

Goal: evolve this prototype into a professional-grade personal quant platform that emulates the
*process* of Renaissance Technologies, Citadel, and Jane Street at retail scale: rigorous
no-lookahead backtesting, walk-forward validation, enforced risk management, firm-style trading
desks, and a hardened paper→live pipeline through E*TRADE (sole brokerage).

## Architecture target

```
Trading Floor (desks = firm personas, strategy + risk + trader's-notes rationale)
├── RenaissanceDesk: HMM regime detection, short-horizon mean reversion, nonlinear stat-arb, pairs
├── CitadelDesk:     pod architecture, central risk book, performance-weighted capital allocation
└── JaneStreetDesk:  fair-value engine
      ├── simulation mode: Avellaneda-Stoikov market-making on synthetic LOB (never live)
      └── live mode (taker): VRP harvesting (core), earnings IV-crush (event),
                             ETF/options relative value (opportunistic)
            ↓ all desks route orders through
ExecutionBroker interface → PaperTrader | LiveEtradeBroker (E*TRADE, sandbox-first)
```

Account facts: equity > $25k (no PDT constraint), options Level 3 planned (spreads allowed).
Cross-desk synergy: Renaissance HMM regime model gates Jane Street premium selling; Citadel
central risk book enforces portfolio-wide vega/gamma limits.

## Phases

1. **Foundations (IN PROGRESS)** — fix RiskManager `entry_price`→`avg_entry_price` crash; correct
   Sharpe (daily portfolio returns) + add Sortino/Calmar; eliminate same-bar execution bias
   (signal on close T, fill at T+1 open) + slippage/commission model; `ExecutionBroker` ABC;
   loud broker-import failures; pyproject.toml + pytest suite + structured logging; app-factory
   Flask refactor (no debug/traceback leak); repo hygiene (done: .pyc/.DS_Store untracked).
   Env: venv is Python 3.13.3; system python3 is 3.9 — always use `.venv/bin/python`.
2. **Data layer** — SQLite OHLCV cache; visible provider failures; options chains via OpenBB;
   universe support (100+ symbols, ETF + constituents); IV history table (for scanner).
3. **Professional UI** — dark trading-terminal theme; dashboard (portfolio, saved backtests,
   alerts); equity-curve/drawdown/candlestick charts; background-job backtests with progress;
   trading-floor desks view; self-hosted assets.
4. **Docker** — python:3.11+ slim image, gunicorn, compose with SQLite volume + .env passthrough,
   /health healthcheck, .dockerignore.
5. **Desk framework + walk-forward harness** — `Desk` abstraction (capital, risk limits, strategy
   stack, trader's notes, attribution); fit(train)/predict(test) walk-forward protocol enforced by
   construction; generic risk checks wired into every desk's order flow.
6. **Renaissance desk** — Gaussian HMM (hmmlearn) regime model; regime-conditioned short-horizon
   (1–5d) mean reversion; cross-sectional nonlinear stat-arb (gradient boosting, long/short
   deciles); cointegration pairs (statsmodels).
7. **Citadel desk** — strategy pods with own capital; central risk book (incl. short-vega/gamma
   limits); volatility-targeted performance-weighted reallocation; drawdown-based pod cuts.
8. **Jane Street desk** — fair-value engine; MM simulator (simulation-only); live taker book:
   IV-rank-driven defined-risk VRP selling gated by Renaissance regime model; earnings IV-crush
   module; ETF/options relative value; IV-rank/IV-vs-realized scanner.
9. **E*TRADE live integration** — full OAuth 1.0a lifecycle (daily expiry, renewal, GUI re-auth);
   sandbox→production config gate; preview→place order flow; multi-leg options orders (L3);
   E*TRADE quotes; reconciliation; patient execution engine (limit-at-mid working, edge-decay
   cancel) as default for all desks; kill switch; daily-loss circuit breaker; append-only audit log.

## Standing constraints

- Tests are offline and deterministic (synthetic data; monkeypatch data fetches; no network).
- Public signatures stay backward compatible (GUI routes depend on them).
- Short-vol strategies: defined-risk structures only; max loss capped at entry by construction.
- Market-making desk never trades live; live = taker-mode dislocation/premium strategies only.
- New deps planned: hmmlearn, statsmodels (Phase 6); pytest (installed).
