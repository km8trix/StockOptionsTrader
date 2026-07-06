# Factor research findings — consolidated

Honest conclusion of the cross-sectional signal hunt run on the survivorship-free
small/mid-cap universe (Sharadar PIT warehouse, 2015–2024, monthly rebalance,
Fama-MacBeth per-date long-short spread, Newey-West/HAC t, Benjamini-Hochberg
across each family, 30 bp/leg cost-netting, forward returns winsorized per date).

## One line

The apparatus works and is the durable asset. Every signal family screened is,
at best, **marginal once restricted to the tradeable slice** (small/mid ex-micro,
retail cost). The binding constraint is **data/universe/cost, not more signals.**

## Signals screened (all winsorized)

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

- The first quality run reported a **spurious negative** spread; winsorized, the
  netmargin premium is real and positive.
- **Value (#71) was ~40% inflated:** pb 63d net **+2.52% → +1.60%** (t 3.01→2.17).
  Still BH-significant, but the raw headline overstated the edge. Any prior
  screen number computed before this fix is similarly inflated.

## Why the wall is real

Insider is the cautionary tale the gate exists to expose: a clean **+2%
event-study spread** that became **−34%** as a realized long-short desk (T+1
fills, slippage on illiquid small-caps, short-leg squeeze/borrow, integer sizing,
top/bottom-quintile selection vs the screen's all-vs-all breadth). Value and
quality are *weaker* screens than insider was on the tradeable slice, so a desk
gate would most likely reproduce the same paper-vs-implementable gap.

## Recommended next unlock — a data/cost change, not another signal

1. **Broker / execution cost (highest EV).** The marginal signals are marginal
   largely because retail (E*TRADE ~30 bp+) cost eats the spread. IBKR (~5-10 bp)
   plus tight small/mid execution is the lever that could move value/quality/
   composite from "marginal screen" to "tradeable." This was already true for 5d
   reversal (BH-significant, killed only by E*TRADE turnover cost). **Decision
   needed: IBKR vs E*TRADE for the research-to-live path.**
2. **Less-crowded universe / better features** (true SUE, more orthogonal factors)
   only matter after the cost lever — a marginal edge at retail cost stays
   marginal regardless of how many you stack (the composite's 0.56-correlated
   legs showed diversification is limited here).

## Status

Paused at the broker/cost decision. Screens + validation + winsorization fix are
committed; the composite is a scratchpad experiment (not yet a durable artifact —
build `scripts/composite_screen.py` if the composite is pursued as a desk).
