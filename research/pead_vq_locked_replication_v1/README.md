# PEAD locked replication research package

This directory fixes the data and analysis contract for
`pead-vq-locked-replication-v1`. It does not claim that PEAD has an edge and it
does not authorize a `PEADDesk` promotion.

The primary signal is the Zacks EPS forecast error divided by the last positive
split-normalized `SEP.close` known before the announcement. Zacks historical
per-share values are split-restated (the 2018 Apple sample, for example, is on
the later 4:1 split basis), so contemporaneous `SEP.closeunadj` would mix share
bases. The exact same-session `closeunadj` is preserved separately as execution
evidence. Actual EPS and exact release
timing come from `ZACKS/ES`; consensus is separately reconstructed from the
last matching `ZACKS/EEH` observation strictly before the report date. Same-day
estimate vintages are excluded because `obs_date` has no intraday timestamp.
The pre-acquisition cross-check tolerance is USD 0.01 per share between the
reconstructed `EEH` mean and the `ES` pre-release mean; larger differences are
retained and excluded pending resolution.
`ZACKS/SS` and `ZACKS/SEH` preserve the corresponding sales evidence but cannot
alter the frozen EPS selection rule.

The analysis forms monthly on the first observed SEP session, admits only
announcements no more than 63 calendar days old and public before formation
close, enters at the next observed session close, and reports 21- and 63-session
long-short diagnostics. Every exit is the exact global observed session at the
declared horizon; a missing ticker bar never slides the exit or changes another
horizon's entry-time selection. The exact eight-test family is pooled plus three dated-size
terciles at both horizons, with cost-net HAC inference and Benjamini-Hochberg
correction.

Close timestamps come from `nyse_session_close_calendar.json` and the
content-addressed acquisition receipt under `nyse_session_close_sources/`.
That receipt binds the exact archived HTML bytes, retrieval and HTTP metadata,
visible-text digest, and deterministically extracted dates for every frozen
official ICE/NYSE URL. Observed SEP dates use 13:00 ET on every proved early
close and 16:00 ET otherwise. Both implementations separately decode and hash
the raw bytes, parse the official statements, require the calendar/source union
and named-source attribution to match exactly, prove NYSE core hours, and fail
closed on any missing, mutated, invented, or out-of-range timestamp.

## Current evidence boundary

The repository's `NASDAQ_DATA_LINK_API_KEY` is accepted by the required Zacks
tables, but the verified history is sample-limited. AAPL `EEH` and `SEH`
observations cover only 2018, while `ES` and `SS` expose four 2018 fiscal-quarter
announcements. That slice may test ingestion, joins, timestamp handling, and
reconciliation. It is development evidence only and cannot be relabeled as an
independent holdout or prospective evidence.

No qualifying source snapshot has been created, so the locked candidate
specification intentionally contains no qualifying snapshot filename or digest.
The local, non-redistributable development capture has source hash
`d242c3343a2b2c3a877e71b4a2c78217164db176d69cbd7c9d4866cb65b93901`.
The earlier v2 diagnostic produced 91 strictly reconstructed EPS events and 165 portfolio observations,
but only eight pooled formation dates cleared the frozen ten-name floor versus
the twelve required for inference. No test statistic or edge claim was produced.
The corrected [`development_sample_report_v6.json`](development_sample_report_v6.json)
is the immutable calculation source bound by the daily protocol. The derived
[`development_sample_report_v7.json`](development_sample_report_v7.json)
preserves the same research core and records successful replay-gated acceptance
of the v3 daily receipt. These reports bind the content-addressed ACTIONS v2
raw-to-Parquet equivalence proof,
archived official session-close source evidence, exact-session reference
implementation inputs, and explicit cash-return policy artifacts. Its
independently written signal/economic comparison is preserved as
[`independent_reference_comparison_v5.json`](independent_reference_comparison_v5.json).
`development_sample_status.json` records the aggregate result and all evidence
identities without embedding licensed rows. `source_manifest.json` records the
exact six-table schema and the expected content-addressed wrapper;
`acquisition_plan.json` separates the 2018 development lane, complete historical
source reconstruction, and a future append-only prospective lane.

`sample_filters.json` freezes the exact 29-ticker provider sample used for the
parser and join trial. It is a bounded entitlement audit, not the candidate's
investable universe. The provider does not permit `act_rpt_date` filtering on
`ZACKS/ES`; a licensed reconstruction must therefore bind the historical query
by period end (or use a complete bulk export) and apply the frozen announcement
window only after preserving the raw response.

The local v6 report binds 669,891 validated ACTIONS rows, a preserved raw bulk
ZIP, and exact CSV-to-Parquet typed row-multiset equality. All 330
name-by-horizon records in the development slice now have both an exact
global-session `closeadj` diagnostic and a mechanically reconstructed
no-reinvestment return using `SEP.close` plus explicit cash distributions.
The reconstruction applies 94 unique dividend rows 230 times across overlapping
holding paths, audits three instances of one issuer-external MRK acquisition,
and encounters no held split, spinoff, merger, or terminal event. Security
lifecycle evidence is complete and there are no horizon exclusions. An
independently written reference path
reconstructed 91 accepted and 25 excluded EPS events, 69 accepted and 47
excluded sales events, all 165 portfolio observations, all 330 economic-return
components, and the aggregate action counts with zero discrepancies. The
mechanical return layer is still nonqualifying: the archived ACTIONS metadata
proves the exact bytes and a generic `value` type, but does not authoritatively
define `dividend.value` as split-normalized cash per share or `date` as the
ex-date. `cash_distribution_semantics.json` therefore records that interpretation
as unproven, while `terminal_settlement_ledger.json` is an empty fail-closed
lookup and explicitly forbids treating ACTIONS value as terminal proceeds.
[`modeled_execution_ledger_v1.json`](modeled_execution_ledger_v1.json) now
rebuilds the pooled selections and carries 16 independent normalized
formation-by-horizon cohort books through target, modeled entry, and modeled
exit. It accounts for 88 selected constituent paths, dividend accruals, fixed
30-bp one-way charges on each entry and exit (120 bp per long-short cohort),
cash, fees, and P&L with exact decimal equations. This is synthetic research
accounting, not paper execution: split-normalized quantities are not broker
shares, dividend payment dates are not present, overlapping cohorts do not
share capital, and no quote, fill, margin, financing, borrow, impact, or
capacity evidence is claimed. The reference comparison remains deliberately
ineligible on its own for generic `ReplicationEvidence` because signal and
terminal-return agreement do not prove a money path.

The next bounded layer is now complete. `daily_money_path_protocol_v2.json`
freezes the daily accounting contract, exact development input identities, and
the exact 4,114-key manifest hash. `daily_input_snapshot_v1.json`
content-addresses the 273 global SEP
sessions, 3,549 exact close/closeadj rows, 60 action rows, 13 currencies, 330
formation observations, and 88 selected paths consumed by both implementations.
`primary_daily_ledger_v2.json` extends the modeled cohort ledger; the separately
written `independent_daily_ledger_v2.json` starts from the independent signal
output and processes prices, orders, fees, and actions one session at a time.
They reconcile all 330 exhaustive formation checkpoints and 3,784 selected
daily constituent checkpoints with zero generic discrepancies under
`daily_money_path_reconciliation_v3.json`. A second PEAD-specific component
audit reconciles all 3,784 per-name states, 688 cohort states, and 61 exact
distribution-action applications with zero discrepancies; its largest decimal
difference is 0.000000000000000002.

The daily receipt is a deliberately narrow success: it proves that two
identified implementations agree on this pooled development-sample modeled
money path. Candidate ACTIONS rows create separately labeled signed
receivable-or-payable balances on the unproved candidate ex-date and never
settled cash; payment dates remain
unknown. The paths use split-normalized accounting quantities, independent
normalized cohort capital, and close-price fills. They are not broker orders,
paper fills, a shared capital portfolio, the full eight-cell candidate family,
or evidence of a tradeable edge. A PEAD report can no longer clear its
independent-implementation blocker with an arbitrary generic or self-consistent
receipt artifact: acceptance now requires a sealed token minted only after a
full replay from all frozen daily-reconciliation inputs and both daily ledgers.
The entire v1 daily chain remains archived but is superseded because v2 makes exit-date
pre-liquidation distribution accrual ordering explicit and closes the source,
key, component-attribution, and acceptance-code binding gaps. Receipt v3 then
adds the independently generated same-date distribution validation code to the
frozen shared-code boundary.

## Qualification boundary

Full historical access is required to reconcile coverage and independently
reimplement the signal. Historical reconstruction can falsify the legacy PEAD
result, but the period has already been inspected through other data and cannot
stand in for prospective evidence.

Before any edge or promotion claim, the project still needs a ledger-frozen
prospective commitment, issuer/SEC reconciliation of actuals and timestamps,
two independently written implementations that reconcile exactly, complete
corporate-action and delisting economics, dated borrow and locate evidence,
calibrated execution and financing costs, capacity curves, and an independent
reviewer attestation. Until then `pead_desk_promotion_allowed` remains `false`.

Official references:

- [Nasdaq Data Link product and API organization](https://docs.data.nasdaq.com/docs/data-organization)
- [Nasdaq Data Link Tables API and bulk export](https://docs.data.nasdaq.com/docs/in-depth-usage-1)
- [NYSE holidays and trading hours](https://www.nyse.com/trade/hours-calendars)
- [SEC EDGAR data APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [Form 8-K, including Item 2.02](https://www.sec.gov/info/edgar/forms/form8-k.pdf)
