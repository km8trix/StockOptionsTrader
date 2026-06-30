# Build Plan — First Vertical Slice: a validated daily trend signal, end-to-end on paper

> Derived from [`docs/PLANNING.md`](docs/PLANNING.md). Every reuse claim below was grounded
> against the actual code (file:line cited). Engineering plan, not investment advice.

## Status

| | |
|---|---|
| **Goal of the slice** | Take ONE specified strategy (daily trend follower) all the way through research → validation gate → E*TRADE **sandbox/paper**, reusing the existing engine. Prove the loop before adding a second strategy or any options. |
| **Locked decisions (§13)** | D1 **refactor in place** · D2 **MTP-style daily trend follower** first · D3 **underlying-first** · D4 **keep E*TRADE** (sandbox as paper) · Decimal **deferred to Phase R** · D7 **diagnostics-only learning later** |
| **Sequencing change from the doc** | `docs/PLANNING.md §12` builds the Appendix-A contracts first. Grounding shows that re-derives plumbing the desk engine already has → reordered: **slice reuses the desk engine; the pure-contract architecture is Phase R, after a signal validates.** Rationale in §2. |

---

## 1. Grounding ledger — what's reused vs. net-new

The "~80% already exists" claim, verified:

| Capability | Verdict | Evidence (file:line) |
|---|---|---|
| No same-bar lookahead (fills at **T+1 open**, options priced off closes strictly before fill) | ✅ confirmed | `backtest_engine.py` docstring:4, `_fill_pending_intents:497-521` |
| Walk-forward leakage guard (`index<=date` by construction, 31-test invariant suite) | ✅ confirmed | `desks/walk_forward.py:96`; `tests/test_walk_forward.py` |
| Deflated Sharpe + PSR + Bonferroni/BH multiple-testing | ✅ confirmed | `analysis/research_stats.py:89,124,196,210`; `tests/test_research_stats.py` (31 pass) |
| Cost/slippage (adverse bps, commission, √-impact + ADV cap, options half-spread) | ✅ confirmed | `backtest_engine.py:810,861,866,566` |
| Risk gate (stops, daily-loss, position-size) | ✅ confirmed | `desks/base.py:213-386 apply_risk` |
| Reconcile (ledger vs broker positions, qty/cash tolerance) | ⚠️ partial — **logs only**, caller must engage kill-switch | `brokers/reconcile.py:12-15,72-88` |
| Kill-switch (2% daily-loss circuit → global `KillSwitch.engage`) | ✅ confirmed | `brokers/circuit_breaker.py:105-145`; `utils/kill_switch.py` |
| `strategies/` purity | ✅ already pure (tangle lives in `desks/`, not here) | grep `strategies/` for execution/brokers/portfolio/risk → none |
| ATR(14, Wilder) indicator | ✅ exists | `data/market_data.py calculate_indicators` |
| **SMA-200** indicator | ❌ only sma_20/sma_50 | one-line add to `calculate_indicators` (~:285) |
| **Idempotent `TargetPosition`** (signed desired end-state) | ❌ engine is **action-based** (BUY/SELL/SHORT/COVER) | `desks/base.py:104-143 DeskIntent` |
| **Single pass/fail gate** | ❌ two contradictory bars: `DSR>0` vs `DSR>=0.95` | `scripts/desk_backtest.py:120` vs `desks/rl_execution.py:701` |
| **PBO / CSCV** | ❌ absent repo-wide | grep `pbo\|cscv\|combinatorial` → 0 hits |
| **E*TRADE paper adapter** | ⚠️ `paper_trader.py` is a local simulator; the real sandbox path is `etrade_client.py` (proven by `scripts/sandbox_*.py`) | `brokers/paper_trader.py:163-193`; `brokers/etrade_client.py` |
| `order_status()` on the broker ABC | ❌ ad-hoc in impls, required by `PatientExecutor` | `brokers/base.py:21-78` vs `execution/patient_executor.py:69-70` |

---

## 2. The load-bearing reframe (read this before the steps)

`docs/PLANNING.md` describes the **end-state**: one engine, pure `Strategy` plugins emitting idempotent
`TargetPosition`, `Decimal` money, a single data-in/intent-out contract. That is the right north star and
**Phase R below commits to it.** But two grounded facts say *don't build it first*:

1. **The gate is desk-coupled.** Honest deflation (the whole point of "was the signal ever real")
   only fires when the strategy is a registered `Desk` exposing `walk_forward_fits`
   ([backtest_engine.py:1130-1139](backtesting/backtest_engine.py)). A bare pure `Strategy` gets
   `n_trials=1`, `deflated_sharpe == psr` — no multiple-testing penalty. Validating a *pure* strategy
   first means rebuilding the gate for the bare-strategy path.
2. **The engine is action-based, not target-based.** Switching to idempotent `TargetPosition` means a
   new diffing portfolio layer + touching every consumer — a real change to working, no-lookahead code.

So: **the slice ships the trend follower as a `Desk`** (reusing the engine, the gate, the risk layer,
the cost model — all verified no-lookahead) and **Phase R extracts the pure `Strategy`/`TargetPosition`/
`Decimal` architecture afterward, on a signal that already works.** This is what "refactor in place, don't
rebuild" (D1) actually means once grounded. If you'd rather pay the architecture cost up front, say so —
it's a real option, just a slower path to the first honest answer.

---

## 3. The slice — build order (each step gates the next)

| # | Step | Reuse | Net-new | Done when (verify) |
|---|------|-------|---------|--------------------|
| 1 | **SMA-200 indicator** | ATR(14) already there | one line in `calculate_indicators` (~:285): `data['sma_200'] = data['close'].rolling(200).mean()` | unit test: `sma_200` present; NaN for first 199 bars; matches a hand-rolled mean |
| 2 | **`TrendFollowerDesk(Desk)`** in `desks/trend_follower.py` | `Desk.apply_risk` (stops/daily-loss/size), `TraderNote` logging, FoundationDesk plumbing to fork | `generate_intents`: long when `close>SMA200`, flat when `close<SMA200`; ATR-based stop (~3·ATR); pyramid by raising size on new highs | unit test on a synthetic bar series: regime-off (close<SMA200 → flat) and ATR-stop fire on the expected bars; runs under `BacktestEngine(desk=...)` unchanged |
| 3 | **Walk-forward wiring** | `WalkForwardController` ([walk_forward.py:96](desks/walk_forward.py)) | expose `walk_forward_fits` (even for a parameter-light rule) so the OOS-fold + deflation layer populates | `report['oos_folds']` present; `n_trials == #refits`; `deflated_sharpe < psr` |
| 4 | **One centralized gate** in `analysis/research_stats.py` | `deflated_sharpe_ratio`, `benjamini_hochberg`, `fold_oos_pvalue` | thin `validate_strategy_oos(returns, n_trials, dsr_threshold) -> {passed: bool, ...}`; pick **one** DSR bar (resolve `>0` vs `>=0.95`) | deterministic with `seed=`; run TrendFollower on SPY+QQQ → single pass/fail; re-run byte-identical |
| 5 | **Paper/live track** on E*TRADE **sandbox** | `ExecutionBroker` ABC (`place_order`→submit, `get_portfolio_status`→positions, `cancel_order`), `circuit_breaker`, `PatientExecutor` | add `order_status()` to the ABC (de-ad-hoc it); point `etrade_client` at the sandbox endpoint as the paper adapter; make `reconcile` **fail-closed** (engage kill-switch on drift) | full loop on sandbox; `reconcile()` zero-drift across a restart; circuit-breaker halts submission; every decision logged |

**Reproducibility (mandatory):** all gated runs construct `BacktestEngine(..., seed=...)` — byte-identical
is conditional on the seed ([backtest_engine.py:198-204](backtesting/backtest_engine.py)); the pure
`research_stats` functions are deterministic regardless.

---

## 4. The validation gate — honest scope

- **Bar (recommended):** `deflated_sharpe >= 0.95` **AND** ≥1 Benjamini-Hochberg-significant OOS fold
  (the stricter, honest bar from `validate_throttle_oos`, generalized). **Not** `desk_backtest._profitable`'s
  `DSR>0`, which is trivially passable.
- **PBO/CSCV is deferred** (net-new, absent today). Deflated Sharpe + BH-significant OOS folds is a
  legitimate gate for **one specified strategy**; PBO earns its keep when you're *mining many* candidates.
  Revisit in Phase R or when a discovered (mined) strategy enters.
- **Caveat to keep in the doc, not paper over:** `report['oos_folds']` uses blended account returns over
  overlapping refit windows with a T+1 boundary return — the engine's own comment
  ([backtest_engine.py:1108-1118](backtesting/backtest_engine.py)) calls it a *heuristic upper bound on
  multiple-testing severity, not exact FWER/FDR*. Treat it as a strong filter, not a proof.

---

## 5. Phase R — the `docs/PLANNING.md` architecture refactor (committed, sequenced after the slice)

Do this **once the trend signal has cleared the gate**, so you're refactoring something that works:

- Introduce `core/contracts.py` (Appendix A `Protocol`/`ABC`s) with **`Decimal`** money; convert at the
  boundary to the legacy `float` core (`core/models.py`) — no big-bang retrofit.
- Extract a **pure `Strategy`** seam: move `apply_risk` and the `PortfolioManager`/`RiskManager` params
  *out* of the signal class (the `desks/base.py` tangle), so signal/portfolio/risk stop being fused.
- Idempotent **`TargetPosition`** + a `PortfolioConstructor` that diffs targets→orders (pyramiding lives here).
- **`tests/test_import_purity.py`** (stdlib `ast`, no new dep): every module under `strategies/` imports
  only `{core.contracts, stdlib, numpy, pandas}` — reconcile the allowlist with reality (`strategies/`
  imports `core.models`; `strategies/advanced.py` imports `sklearn`).
- Harden **both** dormant wall-clock fallbacks: `desks/base.py:181` and `desks/orchestrator.py:190`
  (`datetime.now()`/`Timestamp.now()` when `_clock` is unset) — raise/assert instead.

---

## 6. Gated later phases (unchanged from §7/§14 of the doc)

- **L1 — Options expression + Greek-level risk** (`max_net_delta`/`max_vega`/`max_contracts_per_expiry`,
  Appendix A.5). Only after a signal is validated on the underlying. Revisit IBKR vs E*TRADE here —
  IBKR's options API is the doc's long-term rec, but not needed until now.
- **L2 — Diagnostics / signal-decay only** (§14 bucket 1): P&L attribution, realized-vs-assumed slippage,
  rolling IC. Writes to dashboards + training dataset; **never** to live order flow.
- **L3 — Periodic gated retraining** (§14 bucket 3): retrain candidate on expanding walk-forward window →
  same gate → version → promote in a `ModelRegistry`. Live engine reads a **versioned, immutable
  `ModelArtifact`, read-only, never hot-patched**; logs `model_version` per decision; one-line rollback.
  Gated on resolved-outcome count, not the calendar. **Never** online/RL (bucket 4).

---

## 7. Open decisions surfaced by grounding (recommended defaults in **bold**)

1. **Gate threshold** — **`DSR>=0.95` + ≥1 BH-significant OOS fold**, or the weaker `DSR>0`? Picking the
   weak bar makes the gauntlet theater.
2. **PBO scope** — **defer** (deflated Sharpe + OOS folds suffices for one specified strategy), or implement
   Bailey CSCV now (~half-day net-new)?
3. **TrendFollower as `Desk` now** (recommended; reuses the gate) **vs.** pure `Strategy` now (forces the
   Phase-R gate refactor up front). The slice assumes Desk.

(The four §13 decisions are already locked — see Status. These three are new and have safe defaults; the
slice is buildable as written without blocking on them.)

---

## 8. First action

**Step 1 (SMA-200, one line + a unit test).** Smallest possible change, zero behavior risk to existing desks,
and it's the prerequisite for the `TrendFollowerDesk` signal.
