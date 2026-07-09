# New desks: VIX Mean Reversion + PEAD-SUE (+ reverse-martingale sizing)

**Date:** 2026-07-09 · **Status:** approved-by-delegation (user requested 2-3 high-quality
desks from a candidate list, selection delegated "at your discretion")

## Strategy selection (data-feasibility driven)

| Candidate | Verdict | Reason |
|---|---|---|
| VIX mean reversion | **BUILD** | Daily bars suffice; `^VIX` flows through `MarketDataHandler`'s index route; trades SPY (deepest, cheapest instrument). |
| Post-earnings surprise drift (small-cap) | **BUILD** | True SUE now computable: SF1 ARQ quarterly `epsdil` with PIT `datekey` is ingested (676k rows); survivorship-free small/mid universe via `PitWarehouse`. This was the roadmap's designated "next hunting ground". |
| Reverse-martingale money management | **BUILD (overlay)** | Not a desk — a sizing rule. Implemented as a small reusable sizer used by the VIX desk. Sizing reshapes the return distribution; it cannot create edge, and is documented as such. |
| Opening Range Breakout | **DEFER** | Needs intraday history; E*TRADE retail REST has none, yfinance intraday is ~60 days — an honest multi-year backtest is impossible today. Already deferred by locked decision D2 (2026-06-29). Unlock: intraday feed (IBKR/Polygon). |
| VWAP mean reversion | **DEFER** | Same intraday-data constraint as ORB. |
| Option premium harvesting | **DEFER** | The janestreet desk's documented ruin: synthetic IV has no volatility-risk premium by construction (~26% win rate selling premium). On record (2026-06-28/29): drop options/VRP until a real options feed is wired. |

## Desk 1 — VIX Mean Reversion (`vixrevert`)

Economic hypothesis: volatility spikes mean-revert; equity drawdowns that drive the
spike partially retrace as vol normalizes (the vol-risk-premium / panic-reversion
effect). Long-only SPY expression — no VIX ETPs (contango bleed, blowup risk), no
shorting.

Mechanics (all parameters pre-specified before any backtest; `n_trials=1`, no sweep):

- **Signal on** the day VIX close **crosses above** `1.25 × sma_20(VIX)` (yesterday
  below, today at/above). Episode-level entries only — no re-entry while the signal
  stays on, no averaging down.
- **Entry:** BUY SPY at next open (engine T+1), base `size_fraction = 0.20` of desk
  capital, scaled by the reverse-martingale sizer (below).
- **Exit:** VIX close ≤ `sma_20(VIX)` (spike resolved) **or** 21 trading days elapsed
  (time stop), whichever first.
- **Stop:** shared `apply_risk` percentage stop, `position_stop_loss = 0.10` — buying
  panics can catch falling knives (Mar 2020); a stop-out ends the episode (fresh
  crossing required to re-enter).
- **RiskManager:** `max_position_size = 0.60` (headroom above the sizer's 0.40 cap —
  the strict `>` check needs slack, per the trend-follower lesson),
  `position_stop_loss = 0.10`.
- `^VIX` is an **auxiliary, never-traded** symbol: the desk reads `all_data['^VIX']`
  and must skip it when emitting intents (the engine treats every key as tradeable).
  If `^VIX` is absent from the universe the desk degrades to a flat book with one
  'info' note.
- `walk_forward_fits = []` (fixed rule) → gate runs at n_trials=1, honestly.

Gate: `scripts/vixrevert_gate.py` (mirrors `trend_follower_gate.py`) —
`BacktestEngine(desk=..., seed=42).run(['SPY','^VIX'], 2015-01-01, 2024-12-31)` →
`validate_strategy_oos` (PSR-vs-2%-rf ≥ 0.95 AND ≥ 1 BH-significant OOS year).
2005-2024 reported as a robustness window (more vol episodes: 2008, 2010, 2011).

## Desk 2 — PEAD-SUE small/mid (`pead`)

Economic hypothesis: post-earnings-announcement drift — prices underreact to
earnings surprises, strongest in small/low-coverage names. Prior program finding:
the dates-only PEAD proxy on large caps was directionally right but not significant;
"needs true SUE + small/mid" — both now available.

**SUE definition (pre-specified):** for the latest quarterly filing with
`datekey ≤ t`: seasonal diff `d_q = epsdil_q − epsdil_{q−4}`;
`SUE = d_latest / std(trailing d over 8 quarters, ddof=1)`, requiring ≥ 4 valid
seasonal diffs; multiple filings per `reportperiod` deduped keeping the earliest
`datekey` (first-known, PIT-cleanest). **Freshness:** signal only if the filing's
`datekey` is within 63 calendar days of `t` (drift decays; stale filings are noise).
PIT timestamp is `datekey` (SEC filing date, median 41 days after quarter end) — this
is conservative vs. press-release dating and lookahead-clean by construction.

**Step 1 — honest screen first** (`scripts/pead_screen.py`): SUE as a continuous
cross-sectional factor at BMS rebalance dates on the survivorship-free small/mid
universe, reusing `factor_screen.factor_study` verbatim
(`cheap_is_long=False`, `drop_nonpositive=False`, `winsor_returns=0.01`,
horizons [21, 63], 30bp/leg cost, BH across the family, Newey-West per-date spread —
the exact apparatus that killed/validated insider, value, quality). Per-size-tercile
breakdown included (the pooling-dilution lesson).

**Step 2 — desk** (built regardless; promoted only on evidence):
`PEADDesk(CrossSectionalLongShortDesk)` per the insider template — `_alpha_scores` =
fresh-SUE (monthly cached, PIT provider injected, sign-preserving score), `committee=[]`,
wide `position_stop_loss=0.50`, `long_only` supported (the insider lesson: the short
leg bleeds in small caps). Gate: `scripts/pead_desk_gate.py` with
`WarehouseMarketData` feed on a seeded 600-name subset, `validate_strategy_oos`.

If the screen or gate fails, the desk ships as a **documented negative** research desk
(the repo's established practice) — `_PROMOTED_DESKS` changes only on a PASS.

## Reverse-martingale sizer (`desks/sizing.py`)

`ReverseMartingaleSizer(base_size, step=0.5, max_mult=2.0)`:
`size = base_size × min(1 + step × win_streak, max_mult)`. Streak increments on a
closed winning round-trip, resets to 0 on a loss. Pure, deterministic, desk-internal
(the desk classifies a round-trip at exit-signal time from `avg_entry_price` vs the
exit-day close). Used by `vixrevert`; documented as risk-shaping, not alpha.

## Registration & contracts

Both desks enter `desks/registry._DESK_SPECS` as `status='ready'`,
`gate_status` derives to `'research'` (promotion untouched unless a gate passes).
PEAD's `PitWarehouse` provider is constructed lazily in the desk `__init__` (insider
pattern) so listing/registering never touches parquet; missing warehouse degrades all
reads to None → flat book. Registry/app contract tests (`test_desk_registry.py`,
`test_app.py`) updated. GUI Floor picks both up automatically via `list_desks()`;
Production workspace semantics untouched (desks stay Sandbox-only).

## Testing

- Unit: sizer streak math; VIX crossing/exit/stop logic on synthetic frames
  (including the ^VIX-never-traded invariant); SUE math (seasonal diff, dedupe,
  freshness window, PIT gating) on a hermetic in-memory warehouse fixture; PEAD
  `_alpha_scores` monthly cache.
- Gates: both gate scripts have offline `--selftest`; live runs reported honestly in
  the PR (PASS or FAIL — no promotion without a pass).
