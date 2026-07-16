# PEAD source-qualification package v3

This package records the evidence boundary for
`pead-vq-source-qualification-v3`. The protocol in
`candidate_specification.json` remains frozen and read-only; none of the work
described here changes its universe, signal, portfolio, statistical family, or
stop rules.

The architecture is now implemented through the final ten-partition signal
index. A coherent licensed Sharadar TICKERS/SF1/SEP snapshot has also been
sealed locally, its TICKERS identities replayed, and the official
SHARADAR/INDICATORS definitions for SEP `close` and `closeunadj` preserved.
The exact local artifacts are recorded in `qualification_status.json`.

That is meaningful progress, but it is not evidence of an edge. The current
Nasdaq Data Link Zacks entitlement exposes only a 2018 development slice, not
the frozen 2014–2024 point-in-time consensus history. A complete independent
SEC actual corpus and licensed historical release-distribution evidence are
also absent. Local operator hash admission is not an independent trust
registry. Consequently, event-source qualification, research consumption,
historical replication, edge claims, paper execution, and live deployment all
remain false.

## Implemented authority boundaries

- `data/sharadar_source_evidence.py` seals raw Sharadar TICKERS, SF1, and SEP
  exports, converts them to typed Parquet, proves bidirectional row-multiset
  equivalence, builds the three-table source root, and rederives dated identity
  from exact TICKERS bytes.
- `data/sharadar_semantics_evidence.py` preserves the exact licensed
  `SHARADAR/INDICATORS` response and requires the provider's closed definitions:
  SEP `close` is adjusted for stock splits and stock dividends but not cash
  dividends or spinoffs; `closeunadj` is the unadjusted official exchange close.
- `data/pead_event_universe.py` assigns every source-census row exactly one
  disposition. Frozen v1 semantics are unchanged. The active v2 contract keeps
  identity gaps as affected-row exclusions and cannot qualify an empty or
  incomplete census.
- `data/pead_sharadar_event_universe_replay.py` reopens SF1/TICKERS evidence,
  uses the acquired `ARQ` dimension as the frozen quarterly scope, retains all
  SF1 revisions and provider fiscal-period labels, refuses ambiguous share
  classes, rebuilds ten annual v2 children plus their root index, and publishes
  the replay/index atomically under admitted specification and code hashes.
- `data/pead_consensus_replay.py` rebuilds normalized point-in-time consensus
  from exact provider bytes under a closed adapter, reviewed source manifest,
  reviewed metric profile, dated identity, child universe, and five external
  trust sets.
- `data/pead_announcement_evidence.py` replays exact SEC 8-K Item 2.02 filing
  metadata and exhibit bytes into independent actual EPS evidence. It never
  treats EDGAR acceptance or later retrieval as exact global first-public time.
- `data/pead_announcement_availability.py` proves only a conservative
  `known_public_by` upper bound. Historical reconstruction requires a licensed
  distribution manifest and exact externally admitted provider record;
  prospective evidence requires a contemporaneous positive SEC observation
  bound by an external append-only checkpoint. Evidence classes cannot be
  relabeled.
- `analysis/pead_source_reconciliation_v2.py` authoritatively replays one
  annual child's consensus, SEC actual, and availability evidence; requires
  exact metric/share-basis agreement; applies the strict prior-date rule; and
  dispositions every event. Its output is not itself research-consumable.
- `data/pead_market_accounting_evidence.py` replays source reconciliation,
  Sharadar source/identity/event roots, official NYSE session evidence, and the
  externally admitted SEP semantic profile. It selects only the unique latest
  positive SEP row on an observed session strictly before the activation's
  Eastern date, preserves `closeunadj`, and allows no same-day or older-row
  fallback.
- `analysis/pead_signal_input_reconciliation.py` is the per-child research-input
  boundary. It authoritatively replays source and market receipts, checks dated
  identity and split-restated share basis, computes the frozen signal using
  exact rational inputs plus Decimal-34 output, and accepts or excludes every
  child event. Edge, paper, and live permissions remain hard false.
- `analysis/pead_signal_input_index.py` binds exactly ten authoritative annual
  final receipts to the 2015–2024 replay/index roots, proves ordered exhaustive
  event coverage and cross-year uniqueness, and aggregates only replayed counts
  and blocker identities. It replaces informal file concatenation.

All qualifying artifact publishers are create-only, canonical, strict about
duplicate keys/non-finite numbers, collision refusing, and authoritative by
default. Structural-only publication, where supported for development, must be
requested explicitly and is not qualifying evidence.

## Annual partition contract

The fiscal-period-end target is 2015-01-01 through 2024-09-30. It is divided
into ten calendar-year `pead_event_universe.v2` children. Every SF1 ARQ source
row in each intersection is retained and dispositioned. Event identity is the
10-digit CIK, `reportperiod`, and type `Q`; the provider's year-qualified
`fiscalperiod` label is diagnostic and may reflect a non-calendar fiscal year.

Children share the same source, identity, specification, construction-code,
and freeze roots. Missing identity excludes the affected row. Multiple
permatickers for an issuer-period are excluded rather than guessed. Event IDs
must be unique within and across children. The root
`pead_event_universe_index.v1` is structural and never substitutes for
authoritative source replay or external trust.

Consensus, announcement, availability, source-reconciliation, market, and
final signal receipts are produced per child. The final
`pead_signal_input_index.v1` requires all ten child receipts and authoritative
verification contexts before historical research can consume accepted rows.

## Evidence still required

The remaining blockers are external evidence and independent admission, not
missing computational paths:

1. License or otherwise obtain complete point-in-time consensus history from
   at least 2014 through 2024, including analyst counts and provider
   availability fields, then independently admit the raw bytes, source
   manifests, and metric profiles. The preserved 2018 Zacks slice is
   development-only and cannot be promoted.
2. Acquire a complete independent SEC 8-K Item 2.02/exhibit actual-EPS corpus
   for the frozen event census, with exact metric/share-basis extraction and an
   explicit disposition for every event.
3. Acquire licensed historical distribution records that support conservative
   `known_public_by` bounds, including independent manifest and exact-record
   trust roots. Retrospective SEC retrieval alone is insufficient.
4. Register the locally sealed Sharadar, identity, event, SEP-semantics, and
   NYSE calendar roots in a controlled external trust registry after
   independent review. Hashes copied from the producing artifacts do not supply
   that independence.
5. If prospective evidence is desired, precommit the universe and observer,
   deploy the SEC/licensed-release capture path, and checkpoint observations in
   an external append-only registry before the first signal.

After those inputs exist, the implemented pipeline can produce ten source,
market, and signal receipts plus their final index. Only then may the frozen
eight-cell historical replication run. An edge claim additionally requires an
independent implementation reconciliation and cost-net survival of the frozen
multiple-testing family. Paper and live deployment remain later, separately
approved stages with execution, borrow, capacity, kill-switch, and ledger
evidence.
