# Issuance mid-cap long/short research record

This directory starts the evidence record for `issuance-midcap-ls-v1`. The first
artifact is a conservative reconstruction of the research program that preceded
this candidate:

- [`historical_trial_inventory.json`](historical_trial_inventory.json) records a
  program-wide floor of **at least 316 selection-bearing development cells**.
- [`development_falsification.json`](development_falsification.json) preserves
  the dated inference arrays from the corrected issuance screen.
- `complete` is `false`.
- `qualifying_evidence` is `false`.
- `attestation` is `null`.

Those fields are intentional. The inventory is not a preregistration, not an
independent replication, and not evidence that issuance has an edge. It prevents
the next protocol from pretending the candidate was discovered after only one
trial.

## Corrected issuance result

The 2026-07-13 development rerun removed the legacy result's main optimistic
assumptions: formations use observed SEP sessions, dated market cap, T+1 entry,
raw cost-net HAC inference, and a 120bp four-trade round trip. None of the eight
pooled/size-by-horizon tests survived Benjamini-Hochberg. The middle tercile was
net -0.458% at 21 days (one-sided p=0.8344) and +0.699% at 63 days (p=0.2904).

This stops `issuance-midcap-ls-v1` at development. The report is not qualifying
evidence: the period was already inspected, borrow and independent share-count
data are absent, fixed costs are not calibrated, corporate actions are missing,
and raw event/order evidence is incomplete.

## What the 316 floor means

The unit is stricter than a top-level command invocation. A different signal,
horizon, universe slice, data window/feed, portfolio expression, parameter set,
cost model, inference target, or weighting rule can be selected after looking at
results, so it contributes a trial cell. Exact deterministic reruns are
deduplicated when identity can be established; materially changed or ambiguous
records are not silently collapsed.

The grouped counts reconcile to 316, but this is only a lower-bound research
breadth. It can increase after a row-level audit. It must not be represented as an
exact `n_trials`, and the grouped file is not ready for automatic import into the
hash-chained research ledger.

## Evidence limitations

Much of the historical evidence lives in ignored, mutable JSON/SQLite artifacts,
commit messages, script docstrings, and amended research notes. Many runs lack a
bound data snapshot, clean code hash, dependency lock, seed, full configuration,
cost specification, or raw return series. Historical factor screens also inferred
significance from gross spreads while displaying net means and selected current
size labels; those results are development provenance only.

All periods and datasets already inspected during this work are development data.
They cannot become a fresh holdout by changing their label.

## Required review before a qualifying trial

1. Expand every group into one canonical row per selection-bearing cell.
2. Attach all evidence runs and prove exact duplicates before collapsing them.
3. Have an independent reviewer resolve the medium/low-confidence groups, hash the
   reviewed inventory, and supply the currently absent attestation.
4. Import the reviewed rows under a clearly named legacy-development-backfill
   protocol. Do not backdate or describe it as preregistered.
5. Freeze a separate issuance protocol and genuinely uninspected holdout. Bind its
   code, data snapshot, full configuration, costs, inference, seed, and trial to
   the ledger before opening that holdout.

Until those steps are complete, the credible interpretation is: corrected
development evidence does not support the issuance candidate, and the program
has a reconstructed trial breadth of `>=316`, not a demonstrated tradeable edge.
