# Research credibility protocol

This document is the authority for deciding whether StockOptionsTrader has
evidence of a tradeable edge.  A profitable exploratory screen, a passing unit
test, and a safe deployment path are different facts.  None substitutes for the
others.

## Current status

No strategy is currently supported as having a tradeable edge.  Existing
results are hypothesis-generating because at least one of the following applies:

- the historical universe was conditioned on a current vendor size label;
- displayed net spread significance was calculated from gross cohort returns;
- winsorized returns were treated too much like executable economics;
- the strategy/configuration was selected on the reported sample;
- the complete program-wide trial history was not sealed;
- execution, borrow, or capacity assumptions were not calibrated to observable
  historical or prospective evidence.

The promotion and Foundation deployment systems remain useful operational
controls.  They do not turn exploratory returns into research evidence.

The first corrected development falsification is complete:
`issuance-midcap-ls-v1` produced zero BH survivors across its eight screen
cells and is stopped. The reconstructed program inventory therefore has a
conservative floor of at least 316 selection-bearing development cells; it
remains incomplete and unattested, so 316 is a lower bound, not an exact trial
count.

## Implemented controls

The codebase now enforces the first credibility layer:

- `analysis.research_integrity` freezes canonical protocols, registers every
  attempt in a hash-chained append-only ledger, derives trial counts, and
  permits exactly one opening and one permanent decision per holdout.
- `scripts.research_integrity` exposes that lifecycle without any manual
  trial-count option and can verify the chain against an externally anchored
  head hash.
- the authoritative Foundation v4 runner opens the sealed holdout before
  selecting the universe or executing the strategy, uses the captured
  program-wide trial count for DSR, persists the complete raw report, and
  records the result back into the ledger. Legacy v1-v3 evidence is not
  approvable.
- `analysis.independent_replication` content-addresses the full outputs of two
  distinct code implementations, binds a nonempty expected checkpoint-key
  manifest, and reconciles signal, eligibility, rank, target, order, position,
  cash, fees, and P&L. Missing, unexpected, duplicate, non-finite, or
  out-of-tolerance observations are preserved as failing evidence.
- qualifying screens retain dated universe membership rather than treating the
  union of all names seen during the window as eligible on every date; size
  diagnostics use dated market cap, and HAC/BH inference consumes net returns.
- missing ADV fails closed for qualifying Foundation fills, participation is
  capped at one percent, and the complete research report is stored by hash.

These controls make future evidence auditable. They do not retroactively
validate any existing strategy result.

## Evidence classes

Every result must be labelled as exactly one of:

1. **Development** — data may be inspected and hypotheses may change.  Results
   cannot support a deployment claim.
2. **Historical replication** — a frozen protocol evaluated once on data not
   used to select or tune that protocol.  An earlier period from the same data
   vendor is temporal replication, not independent data replication.
3. **Independent replication** — the frozen signal is reconstructed from a
   second source and by a separately written implementation.
4. **Prospective signal evidence** — outputs accumulated after the protocol was
   frozen, with no rule changes.
5. **Execution calibration** — paper or tiny live orders used to measure fill
   probability, opportunity cost, shortfall, fees, borrow, and parity.  This is
   not statistical proof of alpha.

All dates and datasets inspected before the first sealed protocol on or after
2026-07-13 are development evidence unless an access audit proves otherwise.

## Research constitution

Before opening validation data, a protocol must declare and hash:

- strategy family, version, mechanism, and falsifiable hypothesis;
- development and holdout dataset identities and exact date boundaries;
- point-in-time universe, security types, and dated liquidity eligibility;
- signal inputs, publication delays, formation time, holding period, and exit;
- portfolio construction, factor/sector constraints, and sizing;
- broker/account model, intended capital, margin and shorting assumptions;
- commission, spread, impact, financing, borrow, dividend, and cash-yield rules;
- permitted robustness specifications and the complete correction family;
- primary observation unit, dependence correction, test direction, and alpha;
- minimum economically worthwhile effect and cost-stress requirements;
- concentration, drawdown, regime, and capacity falsifiers;
- the number of independent cohorts required by a power calculation;
- a single holdout decision rule and permanent failure semantics.

Protocol changes create a new protocol and may use only unopened evidence.

## Trial accounting and holdouts

Every attempted signal, universe, factor slice, horizon, cost assumption,
weighting rule, and portfolio variant is a trial.  The ledger is append-only and
content-addressed.  Statistical code derives the trial count from that ledger;
operators do not type a favorable count into a gate.

A holdout can be opened only after its protocol is frozen.  Opening is recorded
before results are accepted.  Exactly one result and decision may be attached to
the opening.  Failure is permanent for that protocol hash; parameter changes do
not convert the same data back into out-of-sample evidence.

Fixed historical holdouts bind an already sealed `data_artifact_hash` when they
open. Prospective holdouts must instead use an exact declaration containing
`mode: prospective`, start/end dates, SHA-256 identities for the source
manifest, query, and acquisition plan, plus `append_only: true` and
`sealed: true`. Their registered trial uses
`prospective-commitment:<commitment_hash>` as its data version. The ledger CLI
returns that commitment hash when it freezes the protocol, opens the holdout
with `--data-commitment-hash` strictly before the first observation, and
rejects any realized-data hash at opening. The final immutable snapshot is
supplied only with
`decide-holdout --realized-data-artifact-hash ...`; pass/fail cannot be decided
before the declared end. After any holdout for a protocol opens, no new trial
may be registered under that protocol. Every ledger mutation also carries the
ledger's trusted clock timestamp; caller-supplied content timestamps cannot
backdate an opening or future-date a decision. A fixed snapshot digest or
prospective commitment is consumable only once across the whole program, not
once per protocol or holdout label.

Prospective commitment is currently a ledger-level research control only. The
authoritative Foundation runner and promotion verifier still require a fixed,
fully realized snapshot at protocol freeze/open time. A prospective ledger
decision is therefore not Foundation promotion evidence until the runner,
artifact, terminal receipt, and promotion verifier explicitly bind the frozen
commitment to a verified acquisition manifest and its realized snapshot.

For Foundation, opaque plan mappings are insufficient: the protocol and trial
must exactly bind code, snapshot, seed, candidate, strategy parameters, window,
PIT universe rule, execution economics, regimes, aggregate primary test, and
promotion-policy hash. The ledger lives inside the artifact registry. A
post-decision receipt links the artifact to its verified protocol, trial,
opening, outcome, decision, and chain checkpoints. The receipt/head must still
be anchored outside the mutable host for deletion resistance.

Benjamini–Hochberg or another declared family correction applies across the
hypotheses/configurations being selected.  Calendar years and regimes are
stability diagnostics, not a search for one unusually significant year.

## Data qualification

Qualifying research must:

- select listings and membership using information available on each date;
- form size and liquidity buckets from dated market data, never current labels;
- include delisted securities and dated corporate-action economics;
- fail closed when price, volume, ADV, spread, locate, or borrow inputs required
  by the strategy are unavailable;
- publish coverage, missingness, restatement, identifier, and anomaly reports;
- manually reconcile a deterministic random sample to raw filings/source data;
- preserve content hashes for every input table before and after a run.

Documented cash or zero-recovery delisting terms are corporate-action
settlements, not market fills. Qualifying research rejects a held delisting
whose economic terms are unavailable; it never substitutes an uncapped final
close for a capacity-constrained exit. Ordinary entries and exits, including
SELL/COVER, respect the declared ADV cap, and sub-share capacity defers rather
than applying a one-share floor.

`SHARADAR/ACTIONS.value` is event metadata and is not accepted as a per-share
cash term or proof of zero recovery. Acquisition rows can report large deal
values rather than consideration per share. Those rows may classify a terminal
event, but qualifying settlement economics require a separately sourced,
content-addressed record of the per-share cash/stock consideration (or explicit
zero recovery). Until that evidence exists, the final-close diagnostic remains
flagged `delisting_terms_unavailable` and strict research fails closed.

PEAD horizons are positions on the global observed SEP session calendar, not
row offsets within a ticker's surviving price history. A missing exact target
bar remains unresolved and never slides to a later date. Entry-time ranks and
long/short names are frozen independently for each horizon; if a selected name
is unresolved, that whole date-by-horizon cohort is withheld from inference
rather than reranked around the missing security.

Observed session dates do not establish their closing time. Announcement
visibility and preannouncement-price eligibility must use a separately
content-addressed authoritative exchange calendar whose exact official source
bytes, retrieval timestamp, HTTP metadata, and deterministic extraction are
archived. Published NYSE 13:00 ET early closes replace the regular 16:00 ET
cutoff. The primary and reference implementations must independently decode,
hash, and parse those bytes, prove every named calendar row and the full source
union, and verify NYSE core hours. Missing, mutated, invented, or out-of-range
close evidence fails closed. The complete evidence identity is part of the
combined data snapshot and must be unchanged across both runs.

Content hashes prove internal consistency, not external origin by themselves.
Candidate-grade acquisition receipts also require a plausible chronology:
HTTP `Date` must be present and close to the trusted retrieval clock,
`Last-Modified` cannot follow that response time, and retrieval must precede
receipt creation. A signature or timestamp anchored outside the mutable host is
still required before the capture time itself is treated as tamper-evident.

Per-share signal inputs must share a split basis. Zacks historical EPS and
consensus values are split-restated, so PEAD scales them by split-normalized
`SEP.close`. The contemporaneous `SEP.closeunadj` is retained separately as
execution evidence. Calling `SEP.close` “unadjusted,” or substituting
`closeunadj` without reconstructing the EPS share basis, is not qualifying.

PEAD economic returns use the same split-normalized basis: exact `SEP.close`
at entry and exit plus cash distributions whose ex-dates fall after entry and
on or before exit, with no reinvestment. Every distribution amount must also
reconcile to the contemporaneous `close`/`closeadj` adjustment transition;
that check detects many amount or split-basis errors but does not prove vendor
semantics. The archived ACTIONS metadata currently defines only a generic
numeric `value`, so mechanically reconstructed dividend returns remain
nonqualifying until authoritative source bytes establish ex-date, cash per
share, currency, and split-basis meaning. Terminal economics always come from
a separate settlement ledger; `ACTIONS.value`, final close, and implicit zero
recovery are forbidden fallbacks.

The historical `insider_desk_gate.py`, `pead_desk_gate.py`, and
`vq_fund_gate.py` drivers use the union of dated members as the engine's static
retrieval list. They are explicitly non-qualifying and fail closed in their
returned `passed` field until dated eligibility is enforced at each signal
date; their statistical result is retained only as `diagnostic_passed`.

For fixed-cost factor screens, `--cost-bps` means a one-way cost for one trade
on one leg. A forward long-short cohort therefore pays four charges (enter and
exit both long and short). Dated spread-level costs remain total strategy drag
and override that approximation.

For PEAD, a later filed EPS or revenue value may not be moved to an earlier
announcement timestamp as qualifying evidence.  The actual timestamped release
value and expectation must be reconstructed.  For issuance, split-restated
vendor shares must be independently checked against dated filings and actions.

## Statistical and economic evidence

The primary test is that the aggregate untouched-holdout **net** return exceeds
the predeclared economic hurdle.  For overlapping monthly cohorts, inference
uses HAC or a stationary/block bootstrap whose dependence length covers the
holding horizon.

Every report must preserve and distinguish:

- unmodified executable cohort and portfolio returns;
- gross and net return series;
- robust/winsorized diagnostics (never executable P&L);
- one-sided primary p-values and two-sided diagnostics;
- confidence intervals for IC, alpha, Sharpe, and break-even cost;
- PSR and DSR with the ledger-derived trial breadth;
- factor, sector, beta, liquidity, name, period, and tail attribution;
- base, 2x, and 3x calibrated cost cases;
- capacity curves across the declared capital range;
- the fraction of P&L contributed by every name, cohort, and regime.

Missing raw evidence prevents promotion even if summary metrics pass. For
Foundation, promotion reloads the complete content-addressed report and
recomputes PSR, DSR, net compounded return, turnover, cost estimate, and every
regime return; any difference from the sealed summary is a hard failure.

## Independent implementation

The vectorized signal study and event-driven portfolio implementation must be
written independently from the protocol.  They are reconciled for signal value,
rank, eligibility, target, order, position, cash, fee, and daily P&L.  Unresolved
differences fail replication.

An independent reviewer must be able to rebuild the environment, load inputs by
hash, rerun both implementations, and reproduce the evidence artifact.

## Candidate program

The initial correction family is deliberately small:

1. `issuance-midcap-ls-v1` — **stopped at development on 2026-07-13**. The
   corrected eight-test screen used observed sessions, dated relative size,
   T+1 entry, raw net HAC inference, and 120bp fixed round-trip cost; zero tests
   survived BH. The middle tercile was net -0.458% at 21 days (p=0.8344) and
   +0.699% at 63 days (p=0.2904). No protocol, paper desk, or parameter repair
   may be built from that result.
2. `pead-vq-locked-replication-v1` — **independent-source development sample
   executed; replication remains blocked**. The frozen primary signal is Zacks actual EPS minus the
   last strictly prior-date Zacks consensus, scaled by the preannouncement
   split-normalized close while preserving `closeunadj` as execution evidence.
   It forms on the first observed monthly session, enters T+1,
   and declares the pooled/dated-size-tercile by 21/63-session eight-cell BH
   family. `ZACKS/ES`, `SS`, `EEH`, `SEH`, `MT`, and `EA` are bound in
   `research/pead_vq_locked_replication_v1/`. The accepted local credential
   exposes only a 29-identity 2018 sample. The content-addressed run reconstructed
   91 EPS events and 165 observations. Both implementations also reconciled all
   330 exact-horizon mechanical cash returns: 94 unique dividend rows were
   applied 230 times without reinvestment, three issuer-external acquisition
   rows were audited and ignored economically, and no held terminal event was
   encountered. A content-addressed primary modeled ledger independently
   rebuilds the pooled tails and reconciles 16 normalized date-by-horizon
   cohort books, 88 selected constituent paths, fixed 120-bp round-trip fees,
   and 264 target/entry/exit projection rows through cash and P&L. Two disjoint
   daily implementations now extend that boundary across 330 exhaustive
   formation keys and 3,784 entry-through-exit constituent marks. Their
   PEAD-specific 4,114-key receipt has zero generic discrepancies and prevents
   an arbitrary generic one-key comparison from clearing the PEAD blocker. A
   second component audit reconciles 3,784 per-name states, 688 cohort states,
   and 61 distribution-action applications with zero discrepancies. Receipt
   acceptance is replay-gated and binds the exact inputs, key manifest, and
   calculation/verification code. Candidate distributions accrue to a
   separately labeled signed receivable-or-payable balance and never settled
   cash because payment dates are absent. This remains synthetic
   pooled cohort accounting: split-normalized quantities are not broker
   shares, cohorts do not share constrained capital, and the daily closes are
   not observed fills. Distribution semantics remain source-blocked. Only
   eight pooled formation dates met
   the ten-name floor versus twelve required for inference; no test statistic or
   edge conclusion was produced. This is development plumbing evidence, not a
   historical holdout or prospective signal record. No PEAD desk may promote
   until complete independent coverage, issuer/SEC reconciliation, two
   reconciled implementations, a sealed prospective commitment, borrow, cost,
   capacity, and independent-review evidence all exist.
3. `mom12-1-pit-control-v1` — apparatus control.  It uses the canonical monthly
   rule and a PIT liquid universe.  Crash filters, residual momentum, and ML are
   separate hypotheses and are not introduced after results are seen.

No additional strategy family joins this round.  A candidate stops permanently
if an independent replication reverses sign, the net evidence misses its frozen
threshold, or the executable/borrowable portfolio materially differs from the
researched book.

## Paper and live roles

Paper qualification continues to require exact artifact construction, timing
parity, durable ledgers, reconciliation, and safe checkpoints.  Credibility adds
attempt-level arrival mid, spread, ADV, fill fraction, opportunity cost, fees,
borrow rejection/fee, and implementation shortfall.

The PEAD modeled cohort and independently reconciled daily ledgers are neither
paper execution nor execution-cost calibration. Their fixed fee,
split-normalized accounting quantities, candidate signed distribution balances,
and close-price transitions cannot be used as observed quotes, routeable
shares, orders, fills, spendable distribution cash, borrow availability,
margin treatment, financing, or capacity evidence.

The paper duration is determined by the predeclared precision target and number
of independent order opportunities, not an arbitrary calendar deadline.  Paper
calibrates execution; prospective signal returns accumulate separately.

Live begins only as a bounded experiment with fixed review dates and stopping
rules.  Any strategy-rule change restarts validation.  Capital cannot increase
while realized costs, rejects, exposures, capacity, or reconciliation differ
materially from the sealed evidence.

## Required live evidence package

Live eligibility requires all of the following, referenced by immutable hashes:

- frozen research protocol;
- complete trial ledger;
- audited PIT data manifest;
- two reconciled implementations;
- untouched historical result with uncertainty intervals;
- independent-source replication;
- prospective signal record;
- complete raw net returns, trades, orders, and portfolio history;
- calibrated cost, borrow, financing, liquidity, and capacity evidence;
- factor, concentration, regime, and tail attribution;
- passing execution-parity and reconciliation evidence;
- independent reproduction report.

If no candidate produces this package, the correct conclusion is that the
project does not yet have a demonstrated tradeable edge.
