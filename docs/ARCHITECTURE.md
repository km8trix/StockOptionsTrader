# Architecture

StockOptionsTrader is a multi-desk quant-trading **simulation** platform that mimics the *process* of Citadel, Jane Street, Renaissance, Two Sigma, and AQR — combining AI trading (ML + neural nets) with quant trading (statistical and factor models). It models how these firms construct portfolios and allocate risk; it does **not** promise or imply returns. Every risky change is validated through a research-integrity gate (Deflated/Probabilistic Sharpe + multiple-testing correction) and defaults to byte-identical current behavior. The platform is **backtest/simulation-first** and never routes a live order without explicit, deliberate wiring — there is no live order path in the autonomy layer at all.

## System overview

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

## Desks at a glance

| Desk | Firm process mirrored | Trades | Core engine | Key opt-in knob (default) |
|------|----------------------|--------|-------------|---------------------------|
| Citadel | Multi-pod capital allocation | Equity long/short (pods) | Factor-neutral central risk book | `max_factor_exposure` (0.25) |
| Jane Street | Variance-risk-premium harvest | Short iron condors + RV pairs | IV-rank + short-vega gate | `vol_skew_slope` (None) |
| Renaissance | Regime-aware multi-book quant | Stat-arb, mean-rev, pairs | 3-state HMM regime engine | `stat_arb_size_by_conviction` (True) |
| Two Sigma | Systematic ML stat-arb | Cross-sectional equity L/S | Walk-forward ML committee | `models` committee (None) |
| AQR | Classical price-factor research | Cross-sectional equity L/S | Ridge-weighted factor model | `alpha` ridge (1.0) |

### Citadel — factor-neutral multi-pod desk

Mirrors a multi-pod firm's capital-allocation process. Three pods (momentum, mean reversion, stat arb) start on a 1/3-1/3-1/3 split and compete for capital, reallocating every 21 trading days on 63-day trailing Sharpe-to-vol with bounds `[0.10, 0.50]` per pod. Pods are placed on probation (weight halved) at −5% drawdown and permanently cut at −8%; every opening intent passes a central risk book that holds the aggregate book factor-neutral across four cross-sectional factors.

- `CitadelDesk._maybe_reallocate` — 21-day performance reweight via `pod_score_inputs` + `clamp_renormalize` (water-filling, bounds `[0.10,0.50]`).
- `CitadelDesk._apply_drawdown_policy` — probation at ≤−5%, cut at ≤−8%, recovery at >−2.5%.
- `CitadelDesk._update_pod_navs` — daily realized+unrealized P&L attribution against allocated capital, conserving the weight invariant.
- `CentralRiskBook.factor_neutrality_limit` — blocks an open whose candidate net factor exposure exceeds `max(band, held)`, allowing corrective hedges but not directional accumulation.

**Opt-in knobs:** `max_factor_exposure=0.25` (set `None` to recover old non-gated behavior), `realloc_every_days=21` (0 disables), `sharpe_window_days=63`, `weight_min=0.10`/`weight_max=0.50`.

### Jane Street — options market making

Mirrors the variance-risk-premium harvest. A premium-selling, defined-risk desk driven by IV-rank (percentile of daily synthetic IV in the calm regime) and the earnings calendar. It sells short iron condors (~35 DTE) when IV-rank > 60, plus a separate earnings book (~21 DTE, entered 2 trading days pre-announcement) and stock RV pairs. Capital split: VRP 0.5 / Earnings 0.3 / RV 0.2. All structures are gated atomically by a portfolio-level short-vega limit; high-vol regime force-flattens VRP.

- `_iv_rank` — percentile of today's IV within a trailing 252-day window; entry needs ≥126 obs.
- `_short_vega_check` (registered on `CentralRiskBook`) — blocks a structure when total short vega would exceed `vega_limit * desk_capital/100k`, evaluated before any leg fills.
- `_build_condor` / `black_scholes_greeks` — constructs legs and per-contract Greeks.
- `_update_regime` — no trades until the HMM is fitted; high_vol forces VRP flat.

**Opt-in knobs:** `vol_skew_slope=None` (flat IV; positive applies an equity moneyness tilt), `exclude_earnings_from_iv_rank=False`, `iv_rank_earnings_window_days=5`.

### Renaissance — HMM regime-gated multi-book

Mirrors a regime-aware capital-allocation process. A 3-state Gaussian HMM (on market return, 20-day log-vol, volume ratio) gates three books — mean-reversion (active only when P(mean-reverting) > 0.6), gradient-boosting stat-arb, and Engle-Granger cointegration pairs — budgeted 0.4 / 0.4 / 0.2. All books are dollar-neutral via per-side conviction sizing; models refit on walk-forward schedules (regime/stat-arb every 21 days, pairs every 63).

- `RegimeHMMModel.fit/predict` — seeded EM with Viterbi labeling (deterministic, reproducible).
- `RenaissanceDesk._mean_reversion_intents` — regime-gated z-score entries (|z|>1.5 on 3-day returns vs 60-day window), per-side conviction sizing.
- `RenaissanceDesk._stat_arb_intents` — rank-stability hysteresis (exit band 2× entry width, min 5-day hold) to limit turnover.
- `RenaissanceDesk._pairs_intents` — cointegration scan, rolling z-center, enter |z|>2.0 / exit |z|<0.5.

**Opt-in knobs:** `stat_arb_size_by_conviction=True`, `stat_arb_exit_band_mult=2.0` (1.0 = old immediate exit), `stat_arb_min_hold_days=5`, `mr_size_by_conviction=True`, pairs `z_mean_window=60`, `mr_prob_threshold=0.6`.

### Two Sigma — systematic ML long/short

Mirrors a single-purpose systematic stat-arb book. Ranks a cross-sectional equity universe daily using a walk-forward ML committee (default: one `stacking` meta-model; opt-in committee of zoo models), longs the top quantile and shorts the bottom, dollar-balanced. Shares the `CrossSectionalLongShortDesk` base (daily reconcile, closing-state guard, orphan sweep) verbatim with AQR.

- `TwoSigmaDesk._alpha_scores` — committee-averaged centered scores (P(up)−0.5); `None` until all members fit once.
- `CrossSectionalLongShortDesk.generate_intents` — refit-if-due → score → rank → graceful degrade → reconcile once → emit opens.
- `CrossSectionalLongShortDesk._reconcile_book` — closes unwanted legs with a closing-state guard; orphan-sweep scoped to desk-traded symbols.
- `WalkForwardController.maybe_refit` — first fit at 120 days, refit every 21, train window capped to 252.

**Opt-in knobs:** `model_key='stacking'` (or `gbm`/`lightgbm`/`mlp`/`lstm`/`factor`), `models=[...]` committee, `quantile=0.2`, `target_gross=1.0`, `max_name_size=0.10`, `min_scored=4`.

### AQR — transparent price-factor long/short

Mirrors a classical-quant research shop: hand-engineered, economically-signed price factors (momentum 12-1, short-term reversal, low-vol, risk-adjusted momentum), standardized cross-sectionally per date, combined by ridge regression to rank a daily long/short book. Fully transparent — every factor and refit weight is published; equal-weight fallback is documented for thin data. Same execution discipline as Two Sigma.

- `FactorModel.fit` — pools standardized exposures against next-day returns; Cholesky ridge solver; degrades to equal-weight on any failure.
- `FactorModel.predict` — latest standardized exposures dotted with fitted weights; absent factors treated as 0.
- `_raw_factor_panel` — O(n) vectorized panel, byte-identical to the scalar loop.
- `AqrDesk._refit_models` — emits live factor weights + degradation flag in trader notes for transparency.

**Opt-in knobs:** `alpha=1.0` (ridge), `quantile=0.2`, `target_gross=1.0`, `max_name_size=0.10`, `min_scored=4`.

## Autonomy layer (Phase F)

Two cooperating layers sit around the fund — both conservative and gated.

**Meta-allocator (capital weighting).** `CrossDeskCapitalAllocator` + `DynamicReweighter.on_day` rebalance desk capital from each desk's *shadow-solo* equity curve (each desk backtested standalone, giving an undistorted risk signal). The default is **risk-parity** (inverse-volatility, byte-identical, equal-weight fallback on degeneracy). An opt-in `performance` mode tilts by Sharpe behind five overfitting guards (degeneracy gate, feasibility check, non-negative clip, shrinkage blend, capped-simplex projection). Rebalancing is walk-forward-honest — solo curves are sliced to `ts <= as_of`, so weights depend only on causally observable history.

**RL execution throttle (gated, off).** `RLExecutionThrottle` is a one-step contextual bandit (not deep multi-step RL) that can only **shrink** plain-stock opening sizes — never grow them. It is **OFF by default** (`sizing_modulator=None` → `orchestrator.step` returns the same list object, byte-identical), **never live** (zero E*TRADE/order-routing imports; it only modulates intent sizes inside the backtest seam between netting and the account-wide risk gate), and validated only through research — deflated-Sharpe + OOS fold significance, with `enables_agent` always `False`. The subtractive invariant is enforced three ways: `scale_max` hard-clamped ≤ 1.0, a per-intent guard that raises if a new size exceeds the original, and an `_assert_subtractive` post-condition on every emitted batch.

## Cross-cutting engineering disciplines

- **Leakage-proof seam.** `WalkForwardController._slice_through` slices every frame to `index <= simulation date` before any model fit or predict, and re-slices in `predict` even if the caller pre-sliced. Fit data is additionally capped to the trailing `train_window_days` (252). Models hold no data beyond fitted parameters, so future leakage is impossible by construction.
- **Research-integrity gate.** Risky changes are validated by Deflated/Probabilistic Sharpe plus OOS-fold multiple-testing correction (Bonferroni/BH) before they are trusted — the autonomy layer in particular is research-gated and never auto-enabled.
- **Determinism.** Seeded RNG, CPU-only, and `_deterministic_torch` give reproducible runs and **byte-identical goldens**; the HMM and ML models are seeded specifically to block local-optima drift.
- **Opt-in / additive pattern.** Every sharpening feature defaults to the prior behavior (e.g. `max_factor_exposure=None`, `vol_skew_slope=None`, `sizing_modulator=None`, `weighting='risk_parity'`). Turning a knob off recovers byte-identical legacy output, so new risk is always opt-in.

## Status

All six program phases (A–F) are merged into `main`.
