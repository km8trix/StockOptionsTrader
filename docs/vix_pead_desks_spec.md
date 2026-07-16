# New desks: VIX Mean Reversion + PEAD-SUE (+ reverse-martingale sizing)

**Date:** 2026-07-09 · **Status:** approved-by-delegation (user requested 2-3 high-quality
desks from a candidate list, selection delegated "at your discretion")

> **Evidence correction (2026-07-13):** PEAD/SUE screen results recorded in this
> document used cost-net displayed means but gross-spread t/p values and BH
> decisions, and the historical candidate universe was filtered with today's
> `scalemarketcap` label. Treat all screen survivor counts and downstream gate
> results below as non-qualifying historical provenance, not evidence of a net
> tradeable edge. The repaired screen uses the full point-in-time eligible
> universe, dated market cap only for contemporaneous diagnostics, and deducts
> costs from every formation-date spread before one-sided net inference. A full
> qualifying rerun is required before promotion research resumes.

## Strategy selection (data-feasibility driven)

| Candidate | Verdict | Reason |
|---|---|---|
| VIX mean reversion | **BUILD** | Daily bars suffice; `^VIX` flows through `MarketDataHandler`'s index route; trades SPY (deepest, cheapest instrument). |
| Post-earnings surprise drift (originally small-cap focused) | **BUILD** | SUE is computable from SF1 ARQ quarterly `epsdil` with PIT `datekey` (676k rows). Qualifying research uses the full PIT-eligible `PitWarehouse` universe and dated market cap only for contemporaneous diagnostics. |
| Reverse-martingale money management | **BUILD (overlay)** | Not a desk — a sizing rule. Implemented as a small reusable sizer used by the VIX desk. Sizing reshapes the return distribution; it cannot create edge, and is documented as such. |
| Opening Range Breakout | **DEFER** | Needs intraday history; E*TRADE retail REST has none, yfinance intraday is ~60 days — an honest multi-year backtest is impossible today. Already deferred by locked decision D2 (2026-06-29). Unlock: intraday feed (IBKR/Polygon). |
| VWAP mean reversion | **DEFER** | Same intraday-data constraint as ORB. |
| Option premium harvesting | **DEFER** | The janestreet desk's documented ruin: synthetic IV has no volatility-risk premium by construction (~26% win rate selling premium). On record (2026-06-28/29): drop options/VRP until a real options feed is wired. |

## Desk 1 — VIX Mean Reversion (`vixrevert`)

Economic hypothesis: volatility spikes mean-revert; equity drawdowns that drive the
spike partially retrace as vol normalizes (the vol-risk-premium / panic-reversion
effect). Long-only SPY expression — no VIX ETPs (contango bleed, blowup risk), no
shorting.

Mechanics as recorded by the historical study (no parameter sweep was reported):

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
- `walk_forward_fits = []` (fixed model rule). The historical gate used
  `n_trials=1`; a qualifying rerun derives trial breadth from the research ledger
  rather than inferring it from the final rule.

Gate: `scripts/vixrevert_gate.py` (mirrors `trend_follower_gate.py`) —
`BacktestEngine(desk=..., seed=42).run(['SPY','^VIX'], 2015-01-01, 2024-12-31)` →
`validate_strategy_oos` (PSR-vs-2%-rf ≥ 0.95 AND ≥ 1 BH-significant OOS year).
2005-2024 reported as a robustness window (more vol episodes: 2008, 2010, 2011).

## Desk 2 — PEAD-SUE (`pead`; originally researched as small/mid)

Economic hypothesis: post-earnings-announcement drift — prices underreact to
earnings surprises, strongest in small/low-coverage names. Prior program finding:
the dates-only PEAD proxy on large caps was directionally right but not significant;
"needs true SUE + small/mid" — both now available.

**SUE definition (pre-specified):** for the latest quarterly filing with
`datekey ≤ t`: seasonal diff `d_q = epsdil_q − epsdil_{q−4}` (calendar-matched,
330-410 days back); `SUE = (d_latest − mean(trailing d)) / std(trailing d)` over
the prior 8 quarters (current excluded, ddof=1, ≥ 4 valid diffs) — the
drift-adjusted Bernard–Thomas form, so steady growth scores ~0, not as a
perpetual "surprise"; multiple filings per `reportperiod` deduped keeping the earliest
`datekey` (first-known, PIT-cleanest). **Freshness:** signal only if the filing's
`datekey` is within 63 calendar days of `t` (drift decays; stale filings are noise).
PIT timestamp is `datekey` (SEC filing date, median 41 days after quarter end) — this
is conservative vs. press-release dating and lookahead-clean by construction.

**Step 1 — qualifying screen first** (`scripts/pead_screen.py`): SUE as a continuous
cross-sectional factor at BMS rebalance dates on the full PIT-eligible universe,
reusing `factor_screen.factor_study` verbatim
(`cheap_is_long=False`, `drop_nonpositive=False`, `winsor_returns=0.01`,
horizons [21, 63], 30bp/leg cost, BH across the family, Newey-West per-date spread —
with inference after costs). Per-size-tercile breakdowns use dated market cap and
are diagnostics, not a current-label universe filter.

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

## Legacy results (2026-07-09; non-qualifying provenance)

These numbers preserve the state of the research after its contemporaneous
adversarial review (including frozen-universe IPO exclusion, strict `datekey < t`
screen entries, and the out-of-order filing guard). They are not final evidence:
the gross-inference and current-size-label defects described above require a new
qualifying run.

**Historical PEAD-SUE screen** (current-size-filtered universe, 4,481 names,
188,624 events, 30bp/leg): the legacy analysis labeled micro-tercile 63d a BH
survivor — gross
+2.71%, net +2.11%, t=3.48, p=0.0005 at winsor 0.01; **robust to aggressive
5% winsorization** (net +1.50%, t=3.34 under the same invalid inference). Pooled
63d was right-signed (legacy t=2.26); the other historical size slices were weak
and all 21d horizons were cost-negative. These are descriptive numbers only.

**Historical PEAD desk gates** (micro band, 600 seeded names, survivorship-free feed,
2015-2024, `validate_strategy_oos`):
- L/S: +53.3%, Sharpe 0.21, PSR 0.76, 0 BH years, maxDD −36.6% → **FAIL**
  (the short leg still halves the long book's Sharpe — the insider lesson).
- Long-only: **+115.0%, Sharpe 0.36, PSR 0.891**, 0 BH years, maxDD −30.6%
  → **FAIL, but the strongest historical realized desk result in the program** (vs
  trend 0.07, momentum 0.20, insider −0.21). The 30bp/leg cost assumption is
  optimistic on micro-caps, reinforcing the FAIL.

**VIX-MR desk gate** (SPY + ^VIX): 2015-2024 −0.1%, Sharpe −0.70, PSR 0.01 →
**FAIL**. Robustness 2005-2024: 74 round-trips, ~69% win rate, but avg loss >
avg win and every crisis year (2008/10/11/18/20) is net negative → +5.0%/20y
→ **FAIL**. Positive expectancy, no risk-adjusted edge.

**Adversarial review (16-agent, 4 lenses + per-finding refuters): 12 confirmed
findings, all fixed or dispositioned** — highlights: PEAD desk froze its EPS
universe to day-one symbols (fixed: re-pull on new symbols; this RAISED both
gate results materially, +5%→+53% L/S); screen admitted same-day filings and
out-of-order delinquent filings (both fixed); shared `apply_risk` now blocks
opening intents on '^' index levels desk-wide; VIX desk got fund-mode
ownership tracking, stalled-exit retry, and fill-day win/loss classification.

**Verdict: both desks ship as research desks; `_PROMOTED_DESKS` stays empty.**
PEAD long-only micro was the best historical result (PSR 0.891 vs the 0.95 bar),
but is not a qualifying promotion candidate. Its investigated unlocks were
(a) lower execution
cost (IBKR), (b) press-release dating instead of the ~41-day filing lag
(needs an earnings-announcement-date feed for small caps), (c) combining with
the value+quality composite via the capital allocator.

## Legacy PEAD unlock results (2026-07-09, follow-up branch feat/pead-unlocks)

All three spec unlocks were pursued; nothing cleared the 0.95 PSR bar, but two
moved the needle materially. Baseline: long-only micro, filing-dated, PSR 0.891.

**(a) IBKR-level costs — NO EFFECT.** Gate at 2bp/5bp/10bp per-side
commission: PSR 0.890 / 0.862 / 0.891 (differences are fill-path noise, not
monotone). Commission is not the binding constraint — and real micro-cap
slippage exceeds the modeled 5bp, so live IBKR would be worse, not better.

**(b) Press-release dating — THE BIG LEVER.** SHARADAR/EVENTS ingested
(2.53M rows); event code 22 = 8-K Item 2.02 (earnings press release);
`--dating announce` re-times each SUE from the 10-Q/10-K `datekey` to the
announcement (median lead: days for large caps, weeks for small caps).
Documented approximation: the SF1 EPS is assumed equal to the 8-K figure.
- This is not an announcement-value replication: qualifying PEAD research needs
  the actual surprise value and timestamp observed at announcement time.
- Legacy screen: **4/8 BH survivors** (was 1/8) — micro 63d net **+2.99%, t=5.09**;
  micro 21d turns cost-positive (net +0.52%, t=3.84); pooled 63d clears
  (net +0.88%, t=3.09).
- Desk gate: **+147.7%, Sharpe 0.42, PSR 0.926** (was 0.891) → still FAIL
  (0 BH-significant years).

**(c) Combine with value+quality — REAL DIVERSIFICATION, STILL SHORT, plus an
architectural finding.** New `ValueQualityDesk` (pb×netmargin rank composite,
PIT, monthly; solo long-only all-bands: +68.5%, Sharpe 0.35, PSR 0.867,
maxDD −22.9% → FAIL; ex-micro slice weaker, +40.3%, as the validation battery
predicted). Paper combine of the two solo legs (PEAD micro LO announce + VQ
ex-micro LO): daily correlation **+0.29**, 50/50 blend **ann Sharpe 0.55,
PSR 0.911** — the program's best portfolio-level number — **FAIL** (0 BH
years; inverse-vol blend 0.885). This is a historical comparison, not qualifying
evidence. The ENGINE-level fund
(`scripts/vq_fund_gate.py --mode fund`, ReweightingFundBacktest) cannot
honestly measure the combine yet: two cross-sectional desks on the shared
book fight — each desk's orphan sweep covers every symbol it EVER traded, and
band membership migrates monthly, so books that are disjoint today overlap
through time (24k trades / Sharpe 0.09 overlapping; still 15k / 0.25
"disjoint"). Fix = desk-scoped position ownership in the fund engine —
follow-up work, out of scope here.

**Post-unlocks verdict: still no promotion.** Best historical books:
announce-dated PEAD long-only micro (PSR 0.926) and the PEAD+VQ paper blend
(PSR 0.911, Sharpe 0.55). Remaining honest levers: desk-scoped fund ownership
(realize the blend in-engine), a third decorrelated leg, or accepting the
strategy class is near — but below — the bar on this decade.

## Fund ownership fix (2026-07-09, follow-up: desk-scoped positions)

The engine-level blocker above is fixed. Fund mode now records which desk(s)'
netted opening intent created each position (`core.models.Position.owners`,
stamped at fill time from `DeskIntent.desk_keys`, which the orchestrator's
netting sets — winning side only on opposing residuals), and the
cross-sectional book machinery (reconcile held-check, orphan sweep, turnover
retention) treats a position owned by another desk as invisible: a desk can
no longer close a position another desk opened later in a symbol it once
traded. `_symbol_is_free` still checks the whole portfolio (the unified book
holds ONE position per symbol; cross-desk exclusion there is what keeps every
position single-owner). Untagged positions — always the case outside fund
mode — behave exactly as before, so single-desk mode is byte-identical by
construction (pinned by `tests/test_fund_position_ownership.py`; full suite
2216 green). Two companions: (1) the gate's fund factory now gives the
account-wide risk overlay the same wide 0.50 stop the solo legs validated
with — the orchestrator's default 2% account stop would churn the monthly
books daily; (2) a FUND-ONLY close retry (`CLOSE_RETRY_DAYS`) re-emits an
in-flight close the engine dropped (bar-less pending expiry / netting
conflict), since ownership scoping means no other desk can ever close the
position for us — pre-fix that leg leaked forever. Ownership is ENFORCED by
the cross-sectional family only; the frozen legacy desks (Renaissance,
Citadel, Foundation, Jane Street, trend follower) still sweep unscoped —
documented in `desks/orchestrator.py`, follow-up filed.

Historical result — `vq_fund_gate.py --mode fund --limit 600 --dating announce`:
**1,382 trades (was 24k), +96.9%, Sharpe 0.41, maxDD −21.8%, PSR 0.9113,
0 BH-significant years → FAIL.** The engine fund now reproduces the paper
blend's PSR (0.911) almost exactly: the shared book no longer destroys the
combine, and what remains is the strategy class sitting below the bar, not an
engine artifact. The Sharpe gap vs the paper 0.55 is real engine economics
the paper blend ignores (shared cash earning zero under ~half deployment,
risk-parity reweighting vs static 50/50, cross-desk same-symbol exclusion) —
not churn: 1,382 fund trades ≈ the sum of the two ~800-trade solo books minus
overlap exclusions.

**Verdict unchanged: no promotion.** The historical engine-level combine is PSR
0.911 vs the 0.95 bar — the fund seam is no longer the constraint.

## Testing

- Unit: sizer streak math; VIX crossing/exit/stop logic on synthetic frames
  (including the ^VIX-never-traded invariant); SUE math (seasonal diff, dedupe,
  freshness window, PIT gating) on a hermetic in-memory warehouse fixture; PEAD
  `_alpha_scores` monthly cache.
- Gates: both gate scripts have offline `--selftest`; live runs reported honestly in
  the PR (PASS or FAIL — no promotion without a pass).
