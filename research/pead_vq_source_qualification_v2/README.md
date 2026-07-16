# PEAD source-qualification package v2

This directory defines the evidence architecture for
`pead-vq-source-qualification-v2`. It inherits the signal and analysis rules of
`../pead_vq_locked_replication_v1/` as read-only lineage. Nothing here amends,
relabels, or qualifies the v1 development sample.

The exact event-universe, consensus, announcement, and upstream source-
reconciliation contracts are implemented, but no qualifying evidence artifact
exists yet. The null artifact references in `qualification_status.json` are
intentional. They must be replaced only by content-addressed artifacts created
from preserved source material; placeholder files or invented digests are not
acceptable.

## Source separation

Qualification requires three non-substitutable evidence lanes:

1. **Sharadar market/accounting lane.** Sharadar may supply dated security
   identity and lifecycle data, universe membership, split-normalized and
   contemporaneous prices, market-cap/size inputs, trading-session observations,
   and corporate-action/terminal-return accounting after the relevant field
   semantics are independently proved. Sharadar SF1 values, filing dates, or
   derived fields may not supply the PEAD actual EPS, analyst consensus, consensus
   vintage time, or earnings-announcement time.
2. **Provider-neutral consensus lane.** A licensed source must preserve the
   historical consensus value, analyst count, estimate basis, currency, period,
   and a source timestamp proving that the selected vintage existed strictly
   before the announcement. Zacks, I/B/E/S, or another provider may implement
   this contract, but provider-specific fields stay in the raw receipt and are
   mapped into `pead_consensus_evidence.v1`. A vendor's reported actual or release
   time is a cross-check only and cannot satisfy the independent announcement
   lane.
3. **Independent SEC/issuer announcement lane.** Preserved issuer releases and/or
   SEC-filed exhibits must establish the reported actual, metric basis, fiscal
   period, currency, issuer identity, and announcement availability time under
   `pead_announcement_evidence.v1`. EDGAR acceptance time is retained separately
   from issuer publication or distribution time: acceptance is provenance for a
   filing, not an automatic claim of the exact first-public earnings-release
   timestamp. If exact public availability cannot be proved, the event fails
   closed or receives a conservative visibility assignment explicitly allowed by
   the frozen protocol.

No lane may backfill another lane's missing required value.

## Exhaustive event manifest

Before the event manifest can be frozen, the project needs a candidate-grade
Sharadar SF1/SEP/TICKERS source snapshot and a validated dated CIK-permaticker-
ticker identity snapshot. Existing warehouse Parquet files contain useful
Sharadar data, but their generic ingest did not preserve the raw ZIPs, provider
snapshot metadata, or CSV-to-Parquet equivalence receipts required at this
qualification boundary. This is an acquisition-provenance gap, not a need to
replace Sharadar as the market source.

The future `pead_event_universe.v1` artifact is the exhaustive event manifest.
It must enumerate every event in the frozen issuer/security universe and date
window exactly once, including delisted securities. Each row carries a stable
issuer/security and fiscal-period identity plus the source-census commitment
that put it in scope. It is frozen before consensus or announcement evidence is
selected, so those later evidence references and their accepted/excluded
dispositions belong to `pead_source_reconciliation.v1`, not to this manifest.

Completeness is a first-class proof. The manifest must bind the frozen universe
query/census receipts, report expected and enumerated counts, and reconcile those
counts without survivor filtering. An empty manifest, an accepted-events-only
file, or a file without a source census is not exhaustive and cannot qualify.

## Two-stage reconciliation

`pead_source_reconciliation.v1` is the upstream announcement/consensus boundary.
It revalidates both input artifacts, rebuilds its complete event disposition,
and may mark a normalized event-source row reconciled only after:

- issuer/security and fiscal-period identities agree across all lanes;
- the consensus vintage is strictly earlier than proved announcement
  availability and has sufficient timestamp precision;
- actual and consensus use compatible metric, per-share basis, currency, and
  split treatment;
- the announcement actual and timing have independent SEC/issuer evidence; and
- every discrepancy is either resolved by a frozen rule or excludes the event.

The upstream receipt also says explicitly whether it has any reconciled rows and
whether all frozen events reconciled. It never calls those rows source-qualified:
the current provider-neutral consensus contract binds raw hashes but does not
yet replay normalized values from archived provider response bytes, and the
candidate/source/code digests still require exact-byte verification. Both are
mandatory at the final boundary.

That receipt is deliberately not consumable by research. A future
`pead_market_accounting_evidence.v1` artifact must independently bind
candidate-grade raw SEP/TICKERS acquisition receipts, a dated CIK/security
identity snapshot, verified NYSE close evidence, official SEP field semantics,
and one exact preannouncement denominator disposition per frozen event. A final
`pead_signal_input_reconciliation.v1` must replay both receipts, prove that the
market close is strictly before the independent announcement time, and reconcile
the common split/share basis. Only that final receipt is a source-qualified
research-input boundary. It still cannot establish an edge or authorize paper or
live execution.

The current announcement v1 implementation replays SEC complete-submission and
EX-99 bytes to establish an actual, but it intentionally rejects every non-null
first-public claim because no authoritative first-public adapter has been
implemented. EDGAR acceptance and a later HTTP observation remain distinct
facts. Thus the implemented source reconciler currently produces only explicit
excluded-event receipts; this is the intended fail-closed result, not missing
plumbing.

Implemented entry points are:

- `data/pead_event_universe.py`;
- `data/pead_consensus_evidence.py`;
- `data/pead_announcement_evidence.py`;
- `analysis/pead_source_reconciliation.py`; and
- `scripts/pead_source_reconciliation.py`.

## Fail-closed progression

The required order is:

1. reacquire and seal candidate-grade Sharadar SF1/SEP/TICKERS source and dated
   identity snapshots;
2. derive, freeze, and seal the exhaustive event universe;
3. acquire immutable provider-neutral consensus evidence;
4. acquire independent SEC/issuer actual and timing evidence;
5. reconcile every manifest event across announcement and consensus lanes,
   including exclusions;
6. replay the bound Sharadar snapshot into exact denominator evidence for every
   manifest event;
7. create the final signal-input reconciliation receipt;
8. reproduce the inherited signal and money path independently; and
9. accumulate separately committed prospective evidence before considering
   controlled paper or live deployment.

Any source-rule, event-universe, timestamp, signal, or statistical-family change
requires a new package version. Current blockers and deliberately null artifact
bindings are recorded in `qualification_status.json`; the exact contracts are in
`source_architecture.json`.
