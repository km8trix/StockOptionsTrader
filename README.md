# StockOptionsTrader

Personal quant trading platform: rigorous no-lookahead backtesting, walk-forward
validation, enforced risk management, and a paper-to-live pipeline through
E*TRADE (sole brokerage). The architecture is a "trading floor" of firm-style
desks — Citadel, Jane Street, Renaissance, Two Sigma, and AQR — coordinated by a
fund orchestrator and an autonomy layer. See the [Architecture](#architecture)
section below for how they fit together, and [PLAN.md](PLAN.md) for the approved
roadmap and current status.

It runs **low-frequency, daily-cadence systematic strategies** — desks rebalance on
daily-to-monthly schedules and hold options for weeks. It is **not** high-frequency
trading or live market-making, neither of which is reachable on retail infrastructure
(the Avellaneda-Stoikov market maker is simulation-only and never trades). The firm
names are **process templates** — how those shops construct portfolios and allocate
risk — not performance claims; the platform does not promise returns.

## Architecture

StockOptionsTrader is a multi-desk quant-trading **simulation** platform that mimics the *process* of Citadel, Jane Street, Renaissance, Two Sigma, and AQR — combining AI trading (ML + neural nets) with quant trading (statistical and factor models). It models how these firms construct portfolios and allocate risk; it does **not** promise or imply returns. Every risky change is validated through a research-integrity gate (Deflated/Probabilistic Sharpe + multiple-testing correction) and defaults to byte-identical current behavior. The platform is **backtest/simulation-first** and never routes a live order without explicit, deliberate wiring — there is no live order path in the autonomy layer at all.

### System overview

```
                         market data (per-symbol OHLCV frames, index <= sim date)
                                              |
            +----------------+----------------+----------------+----------------+
            v                v                v                v                v
       Citadel          Jane Street       Renaissance       Two Sigma          AQR
     (pods, factor-     (options MM,      (HMM regime,      (ML committee     (transparent
      neutral book)      VRP/earnings)     3 books)          long/short)       factor L/S)
            |                |                |                |                |
            +----------------+--------+-------+----------------+----------------+
                                      v
                          FundOrchestrator.step()
                          1. fan out -> desk.generate_intents (desk-internal risk fires here)
                          2. _net_intents (algebraically net opposing stock intents per asset)
                          3. [optional] RL throttle  <-- subtractive-only, OFF by default
                          4. ONE account-wide gate: CentralRiskBook.check via _AccountDesk
                                      |
                                      v
                       shared PortfolioManager / account / fund equity
                                      |
                                      v
       meta-allocator (CrossDeskCapitalAllocator + DynamicReweighter.on_day)
       feeds desk.capital_allocation weights back to the desks  ---------+ (loop back up)
```

Each desk runs its own internal risk controls inside `generate_intents`. The fund nets opposing intents, then applies **one** unified account-level risk gate (single daily-loss circuit, single position-size ceiling, single Greeks gate) — per-desk gates do **not** re-run on the shared book, mirroring how a prime broker consolidates sub-portfolios and applies fund-level risk once.

### Desks at a glance

| Desk | Firm process mirrored | Trades | Core engine | Key opt-in knob (default) |
|------|----------------------|--------|-------------|---------------------------|
| Citadel | Multi-pod capital allocation | Equity long/short (pods) | Factor-neutral central risk book | `max_factor_exposure` (0.25) |
| Jane Street | Variance-risk-premium harvest | Short iron condors + RV pairs | IV-rank + short-vega gate | `vol_skew_slope` (None) |
| Renaissance | Regime-aware multi-book quant | Stat-arb, mean-rev, pairs | 3-state HMM regime engine | `stat_arb_size_by_conviction` (True) |
| Two Sigma | Systematic ML stat-arb | Cross-sectional equity L/S | Walk-forward ML committee | `models` committee (None) |
| AQR | Classical price-factor research | Cross-sectional equity L/S | Ridge-weighted factor model | `alpha` ridge (1.0) |

#### Citadel — factor-neutral multi-pod desk

Mirrors a multi-pod firm's capital-allocation process. Three pods (momentum, mean reversion, stat arb) start on a 1/3-1/3-1/3 split and compete for capital, reallocating every 21 trading days on 63-day trailing Sharpe-to-vol with bounds `[0.10, 0.50]` per pod. Pods are placed on probation (weight halved) at −5% drawdown and permanently cut at −8%; every opening intent passes a central risk book that holds the aggregate book factor-neutral across four cross-sectional factors.

- `CitadelDesk._maybe_reallocate` — 21-day performance reweight via `pod_score_inputs` + `clamp_renormalize` (water-filling, bounds `[0.10,0.50]`).
- `CitadelDesk._apply_drawdown_policy` — probation at ≤−5%, cut at ≤−8%, recovery at >−2.5%.
- `CitadelDesk._update_pod_navs` — daily realized+unrealized P&L attribution against allocated capital, conserving the weight invariant.
- `CentralRiskBook.factor_neutrality_limit` — blocks an open whose candidate net factor exposure exceeds `max(band, held)`, allowing corrective hedges but not directional accumulation.

**Opt-in knobs:** `max_factor_exposure=0.25` (set `None` to recover old non-gated behavior), `realloc_every_days=21` (0 disables), `sharpe_window_days=63`, `weight_min=0.10`/`weight_max=0.50`.

#### Jane Street — options market making

Mirrors the variance-risk-premium harvest. A premium-selling, defined-risk desk driven by IV-rank (percentile of daily synthetic IV in the calm regime) and the earnings calendar. It sells short iron condors (~35 DTE) when IV-rank > 60, plus a separate earnings book (~21 DTE, entered 2 trading days pre-announcement) and stock RV pairs. Capital split: VRP 0.5 / Earnings 0.3 / RV 0.2. All structures are gated atomically by a portfolio-level short-vega limit; high-vol regime force-flattens VRP.

- `_iv_rank` — percentile of today's IV within a trailing 252-day window; entry needs ≥126 obs.
- `_short_vega_check` (registered on `CentralRiskBook`) — blocks a structure when total short vega would exceed `vega_limit * desk_capital/100k`, evaluated before any leg fills.
- `_build_condor` / `black_scholes_greeks` — constructs legs and per-contract Greeks.
- `_update_regime` — no trades until the HMM is fitted; high_vol forces VRP flat.

**Opt-in knobs:** `vol_skew_slope=None` (flat IV; positive applies an equity moneyness tilt), `exclude_earnings_from_iv_rank=False`, `iv_rank_earnings_window_days=5`.

#### Renaissance — HMM regime-gated multi-book

Mirrors a regime-aware capital-allocation process. A 3-state Gaussian HMM (on market return, 20-day log-vol, volume ratio) gates three books — mean-reversion (active only when P(mean-reverting) > 0.6), gradient-boosting stat-arb, and Engle-Granger cointegration pairs — budgeted 0.4 / 0.4 / 0.2. All books are dollar-neutral via per-side conviction sizing; models refit on walk-forward schedules (regime/stat-arb every 21 days, pairs every 63).

- `RegimeHMMModel.fit/predict` — seeded EM with Viterbi labeling (deterministic, reproducible).
- `RenaissanceDesk._mean_reversion_intents` — regime-gated z-score entries (|z|>1.5 on 3-day returns vs 60-day window), per-side conviction sizing.
- `RenaissanceDesk._stat_arb_intents` — rank-stability hysteresis (exit band 2× entry width, min 5-day hold) to limit turnover.
- `RenaissanceDesk._pairs_intents` — cointegration scan, rolling z-center, enter |z|>2.0 / exit |z|<0.5.

**Opt-in knobs:** `stat_arb_size_by_conviction=True`, `stat_arb_exit_band_mult=2.0` (1.0 = old immediate exit), `stat_arb_min_hold_days=5`, `mr_size_by_conviction=True`, pairs `z_mean_window=60`, `mr_prob_threshold=0.6`.

#### Two Sigma — systematic ML long/short

Mirrors a single-purpose systematic stat-arb book. Ranks a cross-sectional equity universe daily using a walk-forward ML committee (default: one `stacking` meta-model; opt-in committee of zoo models), longs the top quantile and shorts the bottom, dollar-balanced. Shares the `CrossSectionalLongShortDesk` base (daily reconcile, closing-state guard, orphan sweep) verbatim with AQR.

- `TwoSigmaDesk._alpha_scores` — committee-averaged centered scores (P(up)−0.5); `None` until all members fit once.
- `CrossSectionalLongShortDesk.generate_intents` — refit-if-due → score → rank → graceful degrade → reconcile once → emit opens.
- `CrossSectionalLongShortDesk._reconcile_book` — closes unwanted legs with a closing-state guard; orphan-sweep scoped to desk-traded symbols.
- `WalkForwardController.maybe_refit` — first fit at 120 days, refit every 21, train window capped to 252.

**Opt-in knobs:** `model_key='stacking'` (or `gbm`/`lightgbm`/`mlp`/`lstm`/`factor`), `models=[...]` committee, `quantile=0.2`, `target_gross=1.0`, `max_name_size=0.10`, `min_scored=4`.

#### AQR — transparent price-factor long/short

Mirrors a classical-quant research shop: hand-engineered, economically-signed price factors (momentum 12-1, short-term reversal, low-vol, risk-adjusted momentum), standardized cross-sectionally per date, combined by ridge regression to rank a daily long/short book. Fully transparent — every factor and refit weight is published; equal-weight fallback is documented for thin data. Same execution discipline as Two Sigma.

- `FactorModel.fit` — pools standardized exposures against next-day returns; Cholesky ridge solver; degrades to equal-weight on any failure.
- `FactorModel.predict` — latest standardized exposures dotted with fitted weights; absent factors treated as 0.
- `_raw_factor_panel` — O(n) vectorized panel, byte-identical to the scalar loop.
- `AqrDesk._refit_models` — emits live factor weights + degradation flag in trader notes for transparency.

**Opt-in knobs:** `alpha=1.0` (ridge), `quantile=0.2`, `target_gross=1.0`, `max_name_size=0.10`, `min_scored=4`.

### Evidence gate & strategy promotion

No strategy is eligible for deployment until it has one immutable, reproducible
evidence artifact. `analysis.promotion` computes/validates PSR, DSR, OOS fold-BH,
cost-adjusted return, turnover, and multi-regime requirements under one versioned
policy. The artifact also pins the data snapshot, universe, parameters, random
seed, dependency versions, and code SHA; canonical JSON makes identical inputs
produce the same SHA-256 artifact ID.

Research evidence and operational approval are separate. `ArtifactStore` writes
the content-addressed artifact without overwriting, then `PromotionRegistry`
creates an immutable paper or live approval that references that exact hash.
Failed or missing evidence cannot be approved, and live approval cannot skip
paper approval for the same artifact. Paper/live composition must use
`desks.registry.create_deployed_desk`, which re-verifies the approval plus the
runtime code SHA and parameters before constructing the desk.

The Trading Floor's legacy `gate_status` remains a research-display badge; it is
not an execution permission. `_PROMOTED_DESKS` is retained only for that backward-
compatible UI contract. The immutable promotion registry is the authoritative
paper/live decision path. The complete operator workflow for the first narrow
Foundation strategy—including durable paper execution, reconciliation evidence,
dual live approvals, activation preflight, limits, and rollback—is in
[`docs/FOUNDATION_DEPLOYMENT.md`](docs/FOUNDATION_DEPLOYMENT.md).
Its timing contract is close D to trade D+1: signals stop at the prior completed
NYSE session, the order uses a fresh quote from the current execution session,
and a current partial bar never enters the indicators. Paper cycles must be
prospective and market-hours-only, with sourced quotes no more than five minutes
old; the provider's observation timestamp is required and the request clock is
never substituted. Model state is safely checkpointed across the
one-step-per-process CLI, sealed in the paper artifact, and restored exactly for
a current-or-prior-session live handoff. Default qualification requires at least
20 cycles across 15 sessions, a flat completed round trip, and 41 strict
reconciliation checks. Passing summary counts and failures are independently
recomputed from the sealed facts. Controlled live orders additionally require a
timestamped E*TRADE `REALTIME` quote no more than 60 seconds old.

**Current state (2026-07):** `_PROMOTED_DESKS` is empty. The current Foundation
artifact is `research_only` (negative return, Sharpe about -3.10, PSR/DSR near
zero, and no BH-significant OOS folds), so the new paper/live lane correctly
blocks it. The strongest price
signal — 12-1 momentum (`mom_12_1`) — passes the IC gate (t +2.66 @1d) but the
desk built on it fails the OOS gate (Sharpe ≈0.20): the edge is real but too
thin to be risk-efficient after costs and momentum crashes. A `mom_12_1` +
`reversal_5` hybrid was also tested (`scripts/proto_hybrid_desk.py`) and rejected
— `reversal_5` has real edge but ~79% turnover makes it un-tradeable at daily
cadence (costs win). The six firm-style desks below are research/backtesting
tools, not validated production strategies.

### Autonomy layer (Phase F)

Two cooperating layers sit around the fund — both conservative and gated.

**Meta-allocator (capital weighting).** `CrossDeskCapitalAllocator` + `DynamicReweighter.on_day` rebalance desk capital from each desk's *shadow-solo* equity curve (each desk backtested standalone, giving an undistorted risk signal). The default is **risk-parity** (inverse-volatility, byte-identical, equal-weight fallback on degeneracy). An opt-in `performance` mode tilts by Sharpe behind five overfitting guards (degeneracy gate, feasibility check, non-negative clip, shrinkage blend, capped-simplex projection). Rebalancing is walk-forward-honest — solo curves are sliced to `ts <= as_of`, so weights depend only on causally observable history.

**RL execution throttle (gated, off).** `RLExecutionThrottle` is a one-step contextual bandit (not deep multi-step RL) that can only **shrink** plain-stock opening sizes — never grow them. It is **OFF by default** (`sizing_modulator=None` → `orchestrator.step` returns the same list object, byte-identical), **never live** (zero E*TRADE/order-routing imports; it only modulates intent sizes inside the backtest seam between netting and the account-wide risk gate), and validated only through research — deflated-Sharpe + OOS fold significance, with `enables_agent` always `False`. The subtractive invariant is enforced three ways: `scale_max` hard-clamped ≤ 1.0, a per-intent guard that raises if a new size exceeds the original, and an `_assert_subtractive` post-condition on every emitted batch.

### Cross-cutting engineering disciplines

- **Leakage-proof seam.** `WalkForwardController._slice_through` slices every frame to `index <= simulation date` before any model fit or predict, and re-slices in `predict` even if the caller pre-sliced. Fit data is additionally capped to the trailing `train_window_days` (252). Models hold no data beyond fitted parameters, so future leakage is impossible by construction.
- **Research-integrity gate.** Risky changes are validated by Deflated/Probabilistic Sharpe plus OOS-fold multiple-testing correction (Bonferroni/BH) before they are trusted — the autonomy layer in particular is research-gated and never auto-enabled.
- **Determinism.** Seeded RNG, CPU-only, and `_deterministic_torch` give reproducible runs and **byte-identical goldens**; the HMM and ML models are seeded specifically to block local-optima drift.
- **Opt-in / additive pattern.** Every sharpening feature defaults to the prior behavior (e.g. `max_factor_exposure=None`, `vol_skew_slope=None`, `sizing_modulator=None`, `weighting='risk_parity'`). Turning a knob off recovers byte-identical legacy output, so new risk is always opt-in.

### Status

All six phases of the AI + Quant desk program (A–F) are merged into `main`.

## Local development

Requires Python 3.13 (the dependency pins target it).

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q     # offline, deterministic suite
APP_AUTH_USERNAME=operator APP_AUTH_PASSWORD='replace-with-a-local-secret' \
  .venv/bin/python run_gui.py      # dev server on http://127.0.0.1:5001
```

The launcher intentionally does not auto-load `.env`; supply the two auth
variables through your shell or secret manager when running locally.

## Docker

```bash
# Authentication is mandatory outside tests. Put unique, high-entropy values
# in .env alongside the E*TRADE variables; do not commit that file.
# APP_AUTH_USERNAME=operator
# APP_AUTH_PASSWORD=<generate-and-store-in-a-password-manager>

# .env must exist in the repo root (E*TRADE credentials; see the table
# below for the variable names). It is passed to the container at runtime
# only — never baked into the image.
docker compose up --build -d
```

The app is published only on host loopback at <http://localhost:5001> and
requires HTTP Basic authentication. The unauthenticated `GET /health` endpoint
exposes liveness only and is used by the container health probe. For remote
access, use an SSH tunnel or an authenticated TLS reverse proxy; do not widen
the Compose port binding directly. The first build downloads several GB of
Python dependencies (openbb pulls a large tree) — subsequent builds reuse the
cached layer.

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
| `APP_AUTH_USERNAME` | — (required outside tests) | HTTP Basic username protecting the GUI, APIs, static assets, and metrics |
| `APP_AUTH_PASSWORD` | — (required outside tests) | HTTP Basic password; use a unique, high-entropy value stored only in `.env` |
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
