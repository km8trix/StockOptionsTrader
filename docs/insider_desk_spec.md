# Spec — Insider Net-Buy Desk (size-bucketed cross-sectional long/short)

> Graduate the validated insider-buying signal into a proper `Desk`, score it through
> the honest OOS gate, and combine the size sleeves with the existing capital allocator.
> Every reuse claim below is grounded against the code (file:line). Engineering plan,
> not investment advice.

## STATUS: tested — DOCUMENTED NEGATIVE (2026-07-01)

The desk was built and gated. **It fails in every tradeable form**, so it ships as
a documented-negative research desk (not promoted, not live), and the reusable
infrastructure below (warehouse feed, PIT size buckets, engine `market_data`
injection, `long_only` mode, the desk + gate) is the lasting value.

| Form | 600-name small-cap gate (2015–2024) | Verdict |
|---|---|---|
| Long/short (both legs) | −34% return, Sharpe −0.51, PSR 0.055, 0/10 yrs | FAIL |
| Long-only (short leg dropped) | −18% return, Sharpe −0.21, PSR 0.25, 0/10 yrs | FAIL |

The screen's +2% cross-sectional spread is **paper alpha that does not survive
implementation**: the short leg bleeds (shorting illiquid small-caps), and
long-only eats small-cap beta (the signal is relative, not directional). This is
the graduation gate working as intended — it caught a signal that looked
deployable and would have lost money live. The build plan below is retained as
the (correct) record of how it was done.

## 0. Why this desk

The full-universe insider screen (`scripts/insider_screen.py`, PR #68/#69) found the
first signal in the program to clear the honest bar: a 63d insider **buy-vs-sell**
long/short, BH-significant and cost-positive **within each size bucket** and
2020-independent — Small-cap strongest (t=4.33 ex-2020, net +1.39% @30bp/leg), Mid-cap
tradeable (t=2.38, net +0.56%), Micro (t=2.50). Two load-bearing facts from that work:

- **Signal = SIGN, not magnitude.** Direction (net buyers vs net sellers) predicts;
  dollar size (`net_value`) does not (pooled rank-IC ≈ −0.02). Don't size by dollars.
- **Condition within size buckets.** Pooling micro+small+mid into one cross-section
  *dilutes* (pooled ex-2020 → t=1.73, insignificant); within-bucket it holds.

## 1. Grounding ledger — reuse vs net-new

| Capability | Verdict | Evidence (file:line) |
|---|---|---|
| Cross-sectional L/S book (rank → top/bottom quantile → dollar-balanced BUY/SHORT, reconcile, orphan-sweep, turnover) | ✅ reuse wholesale — subclass, don't touch | `desks/cross_sectional.py:70` (base), `generate_intents:240-404`, sizing `:323-327` |
| Concrete fixed-factor template (committee=[], n_trials=1, wide stop) | ✅ clone shape | `desks/cross_sectional_momentum.py:1-82`; wide `RiskManager(position_stop_loss=0.50)` `:40-45` |
| The one hook to implement | ✅ | `CrossSectionalLongShortDesk._alpha_scores(all_data, date)->Optional[Dict[sym,float]]` `desks/cross_sectional.py:227` |
| Insider signal source (PIT, trailing 90d, filingdate≤asof) | ✅ reuse verbatim | `data/pit_warehouse.py:175-207` / `data/pit_provider.py:238-284` `insider_net_buys(ticker,asof,lookback_days)` |
| Point-in-time size (for bucketing) | ✅ exists | `data/pit_warehouse.py:235-252` `daily_metric(ticker,date,'marketcap')` |
| No-lookahead contract (T+1 open fills, `date`-sliced `all_data`, `set_clock`) | ✅ confirmed | `backtesting/backtest_engine.py:342-381`, fills `:775-992` |
| Data-source injection into a desk (RenaissanceDesk/EarningsCache pattern) | ✅ pattern to copy | `desks/registry.py:201-247`, injection `:240-242` |
| OOS validation gate | ✅ reuse the strict one | `analysis/research_stats.py:251-334` `validate_strategy_oos` (PSR-vs-rf≥0.95 **AND** ≥1 BH-significant calendar-year fold); driver pattern `scripts/trend_follower_gate.py:56-63` |
| Combine desks into one book + decorrelation weighting | ✅ **already built — do NOT build** | `desks/capital_allocator.py:55-690` `CrossDeskCapitalAllocator` (`risk_parity_cov_weights:196`, `max_diversification_weights:328`); book `desks/orchestrator.py:111-359`; in-run reweight `desks/dynamic_reweighter.py`; entry point `backtesting/reweighting_fund.py:55-180` |
| Size bucket from PIT marketcap (marketcap→bucket) | ✅ **built** | `data/size_buckets.py` `size_buckets()` (cross-sectional quantile, unit-agnostic) + `pit_marketcaps()` over `daily_metric` |
| **`InsiderNetBuyDesk` (subclass + provider handle)** | ❌ net-new (~60-90 lines) | insider data is **not** in `all_data` (OHLCV-only) — desk must query an injected provider |
| Survivorship-free prices for the desk's book | ✅ **built** (was the biggest gap) | `data/warehouse_feed.py` `WarehouseMarketData(MarketDataHandler)` + `PitWarehouse.ohlcv` (adjusted); engine now takes `market_data=` — see §4 |

## 2. The one design decision that matters — size-neutral, point-in-time

The validated edge is **within-bucket**. Two ways to express that in a desk:

- **Chosen: one desk per PIT size band, combined by the allocator.** `InsiderNetBuyDesk`
  takes a `band` param; its `_alpha_scores` scores **only** names whose *point-in-time*
  marketcap (`daily_metric(sym, date, 'marketcap')`) falls in that band, signs them
  ±1, and returns them — so the base's cross-sectional rank is automatically
  within-band. Instantiate 3× (micro/small/mid) and combine with
  `CrossDeskCapitalAllocator` (`risk_parity_cov` or `max_diversification`). This reuses
  the whole book + allocator stack and *is* the plan's "combine decorrelated survivors"
  (`desks/capital_allocator.py`).

- Rejected: one desk that encodes bucket-relative ranking into a single global score —
  messier, and the base's global quantile split fights it.

**Lookahead trap (must not get this wrong):** `scalemarketcap` on the tickers table is
the **current** band as of data download, *not* point-in-time (`data/pit_warehouse.py:118,132-134`;
`scripts/insider_screen.py:50-54`). Using it to decide which bucket a name trades in
leaks the future. **Use `daily_metric(..., 'marketcap')` (PIT) to form buckets;**
`scalemarketcap` is fine only to coarsely scope the candidate universe.

## 3. Build order (each step gates the next)

| # | Step | Reuse | Net-new | Done when |
|---|------|-------|---------|-----------|
| 1 | **MVP: single Small-cap desk** (strongest sleeve, t≈4.3) | subclass `CrossSectionalLongShortDesk`; clone `cross_sectional_momentum.py`; `insider_net_buys` sign | `desks/insider_netbuy.py`: `_alpha_scores` = ±1 sign for names with PIT marketcap in the Small band (`daily_metric`) & non-zero activity; inject a `PitWarehouse`; wide `RiskManager` | runs under `BacktestEngine(desk=...)`; a no-lookahead check passes (`daily_metric`/`insider_net_buys` both keyed on `date`); book is non-empty on a small/mid universe |
| 2 | **Honest gate** | `validate_strategy_oos`; `trend_follower_gate.py` driver | `scripts/insider_desk_gate.py` (mirror trend-follower): run desk → daily returns + calendar-year labels → `validate_strategy_oos(psr_threshold=0.95)` | single PASS/FAIL verdict; deterministic with `seed=`; the desk P&L (T+1, slippage, caps) is the real bar, expected weaker than the event-study spread |
| 3 | **Three size sleeves** | same class | `band` param → 3 registry entries (`insider_micro/small/mid`) in `desks/registry.py` `_DESK_SPECS` + provider injection branch (copy `:240-242`) | all 3 register, run, and gate independently |
| 4 | **Combine** | `ReweightingFundBacktest`, `CrossDeskCapitalAllocator`, `FundOrchestrator` | wire the 3 sleeves into `reweighting_fund.py` with `weighting='risk_parity_cov'` | one insider book; report inter-sleeve correlation + reweight_log; gate the combined book |
| 5 | **Realism** | cost model already in engine | per-band spreads/capacity/turnover; decide E*TRADE vs **IBKR** (tight small/mid execution) | net-of-realistic-cost verdict per band; capacity note |

## 4. Survivorship-free prices — RESOLVED (option A built)

The desk engine builds `all_data` from `MarketDataHandler.fetch_stock_data` (live
OHLCV), which drops delisted names — silently reintroducing the survivorship bias the
warehouse removes. **Fixed:** `data/warehouse_feed.py::WarehouseMarketData` subclasses
`MarketDataHandler` and serves adjusted OHLCV from the warehouse (`PitWarehouse.ohlcv`,
split/div-adjusted via the `closeadj/close` factor); `BacktestEngine` now accepts
`market_data=`. Run the desk gate with
`BacktestEngine(desk=..., market_data=WarehouseMarketData())` so the book is
survivorship-free. **Requires** the warehouse `sep` table ingested (done). The desk's
Step-2 gate number is only honest with this feed.

## 5. Open decisions (recommended defaults in **bold**)

1. Score value — **±1 sign** (validated) vs signed `net_shares`. Magnitude dilutes; use sign.
2. Bucketing — **PIT `daily_metric` marketcap terciles/bands** vs `scalemarketcap` (lookahead — no).
3. Committee — **empty (`committee=[]`, n_trials=1, no deflation)**; it's a pre-specified rule, validate via `validate_strategy_oos`, not the lenient `_profitable` harness bar (`desk_backtest.py:120`).
4. Universe — **provider-derived small/mid** (`universe_asof`, survivorship-free) vs `LARGE_CAP_100` (the harness default — wrong universe for this edge).
5. Rebalance/holding — **monthly rebalance** (matches the screen); tune `min_holding_days`/`exit_quantile` for turnover.

## 6. Risks / gotchas (grounded)

- **Sparsity:** insider activity is sparse; per-band books can be thin or one-sided on a
  given date → the long-short starves. Watch `min_scored` (default 4) and per-band
  both-leg availability (`desks/cross_sectional.py:266-275`).
- **Don't override `generate_intents`/reconcile** — the base docstring warns re-implementing
  the book reintroduces the churn/orphan bug it fixed (`desks/cross_sectional.py:6-8`).
- **Sign convention** was double-negated once (commit `3faacdb`/PR #67) — score on
  `net_shares`/`net_value` as returned; higher = more buying = stronger long.
- **`committee=[]` → n_trials=1 → deflated_sharpe==psr** (no deflation) — honest for a
  fixed rule, but claim no OOS-fold deflation credit from it (`backtest_engine.py:1136-1139`).
- **The allocator down-weights but does not prune** correlated desks; with only 3 size
  sleeves that's fine, no pruning layer needed (`desks/capital_allocator.py` net-new gap).
- **`_PROMOTED_DESKS` stays empty** — a passing gate makes this a *research* candidate, not
  a live desk. "Promoted" is a research badge, never production-live (`desks/registry.py:177-195`).
