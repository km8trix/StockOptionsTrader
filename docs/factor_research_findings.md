# Factor research findings — consolidated

Historical conclusion of the exploratory cross-sectional signal hunt run with a
current `scalemarketcap` small/mid filter (Sharadar PIT warehouse, 2015–2024,
monthly rebalance,
Fama-MacBeth per-date long-short spread, Newey-West/HAC t, Benjamini-Hochberg
across each family, legacy 30 bp/leg cost-netting, forward returns winsorized
per date).

> **Statistical correction (2026-07-13):** The recorded screen table below used
> cost-net displayed means but gross-spread t/p values and BH decisions. Those
> significance claims are invalid as evidence of a net edge and are retained
> only as research provenance. Current `factor_study` deducts cost from each
> formation-date spread before HAC/BH, uses one-sided H1 net-mean>0 p-values,
> and reports raw economic returns separately from winsorized robust inference.
> The legacy fixed-cost code also deducted two 30 bp charges while its CLIs
> described 30 bp as one-way. A long-short cohort enters and exits both legs,
> so the corrected fixed model deducts four charges (120 bp total). Legacy net
> means therefore require recomputation, not just new p-values.
> The historical universe also used today's `scalemarketcap` label. Qualifying
> screens now use the full PIT-eligible universe and dated market cap only for
> contemporaneous diagnostics. Every family must be rerun before its survivor
> count is cited.

## One line

The apparatus is the durable asset. The legacy results suggested, at best,
**marginal performance in the selected slice**; they do not establish an edge.

## Corrected issuance falsification

The first credibility rerun does provide a decision, but it is negative. Net
share issuance was rerun over 2015-01 through 2024-09 using the first observed
SEP session of each month, the dated eligible universe and market-cap terciles,
T+1 entry, raw return economics, HAC on the per-date cost-net series, and 30bp
per one-way trade (120bp for both entries and exits).

Across pooled and three dated size terciles at 21/63-day horizons, **0/8 tests
survived BH**. The middle tercile returned net -0.458% at 21 days (t=-0.97,
p=0.8344) and +0.699% at 63 days (t=+0.55, p=0.2904). The nominal smallest
p-value, 0.0491 for the smallest tercile at 63 days, did not survive the family
correction and is not the intended tradeable slice. The candidate is stopped at
development; dated arrays and limitations are preserved in
`research/issuance_midcap_ls_v1/development_falsification.json`.

## Legacy signals screened (winsorized gross inference; not current evidence)

| family | screen (63d net, t) | tradeable-slice verdict |
|---|---|---|
| insider net-buying | +1.0% (t≈2.4) | **dead as a desk** — L/S −34%, long-only −18% (short-leg bleed + small-cap beta) |
| value (pb) | +1.6% (t≈2.2) | marginal (ex-micro t≈1.7); OOS-only (nothing 2015-19) |
| quality (netmargin) | +4.0% (t≈2.3) | marginal (ex-micro t≈1.5); decays OOS 2020-24 |
| value+quality composite | +2.55% (t≈1.9) | **best candidate, still marginal**; legs only 0.56 correlated |

Price-action families (trend, 12-1 momentum, 5d reversal, PEAD-by-dates) were
rejected earlier — see the research log. roe/roa are unavailable (100% null in
SF1's ARQ dimension); grossmargin is weak.

## Methodology correction (winsorization)

The screens originally did **not** winsorize forward returns. On this
survivorship-free micro universe raw forward returns reach **+2330x** (micro-cap
/ split artifacts), and a single such name dominates a leg — inflating spreads and
widening SEs. Two concrete consequences, both now fixed (`factor_study`'s
`winsor_returns`, default on in the screens):

- The first quality run reported a **spurious negative** spread; winsorization
  made the legacy netmargin estimate positive, but it remains non-qualifying.
- **Value (#71) was ~40% inflated:** pb 63d net **+2.52% → +1.60%** (t 3.01→2.17).
  It was labeled BH-significant only under the invalid gross-return inference;
  the raw headline also overstated the estimate. Any prior screen number
  computed before this fix is similarly non-qualifying.

## What the legacy implementation test showed

Insider is the cautionary tale the gate exists to expose: a historical **+2%
event-study estimate** became **−34%** as a realized long-short desk (T+1
fills, slippage on illiquid small-caps, short-leg squeeze/borrow, integer sizing,
top/bottom-quintile selection vs the screen's all-vs-all breadth). The legacy
value and quality estimates were weaker than the legacy insider estimate; that
comparison is a prioritization clue, not qualifying evidence.

## Historical next-unlock hypothesis

This was the interpretation before the evidence corrections above; it is retained
for provenance, not as the current research recommendation.

1. **Broker / execution cost (highest EV).** The marginal signals are marginal
   largely because retail (E*TRADE ~30 bp+) cost eats the spread. IBKR (~5-10 bp)
   plus tight small/mid execution is the lever that could move value/quality/
   composite from "marginal screen" to "tradeable." This was already true for 5d
   reversal (legacy BH claim, with modeled turnover cost eliminating the
   displayed net return). **Decision
   needed: IBKR vs E*TRADE for the research-to-live path.**
2. **Less-crowded universe / better features** (true SUE, more orthogonal factors)
   only matter after the cost lever — a marginal edge at retail cost stays
   marginal regardless of how many you stack (the composite's 0.56-correlated
   legs showed diversification is limited here).

## Status

The legacy screen log is retained as provenance. New work follows
[`RESEARCH_CREDIBILITY.md`](RESEARCH_CREDIBILITY.md); no survivor count in this
document is qualifying until it is reproduced under that protocol.
