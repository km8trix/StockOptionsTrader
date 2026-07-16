# Foundation strategy deployment lane

This is the only supported path from strategy research to autonomous live
execution. It deliberately handles one strategy (`foundation-target-v1`) and
one desk. Adding more strategies before this path produces real evidence would
multiply operational surface without proving the deployment machinery.

## Current truth

There is no promoted Foundation artifact today. The saved exploratory result in
`analysis/backtests/foundation.json` lost about 2.38%, has Sharpe about -3.10,
PSR effectively zero, DSR effectively zero, and no BH-significant OOS folds.
It is `research_only`. The pipeline does not reinterpret, grandfather, or
manually override that result.

The passing fixtures in `tests/test_foundation_deployment_pipeline.py` prove the
state transitions and safety controls. They are synthetic contract tests, not
trading evidence and not entries in an operator's promotion registry.

## What was implemented

| Stage | Required evidence/control | Implementation |
|---|---|---|
| Research | Exact frozen protocol/trial binding, PIT fixed-start universe, dated market-cap ranking, hashed source files before/after, clean Git SHA, exact dependencies/config/seed/window/costs/regimes/policy, ledger-derived trial count, strict ADV fills, aggregate OOS PSR/DSR/economic gates, and diagnostic calendar/regime results | `analysis.foundation_research` |
| Paper approval | Immutable human approval of one exact research hash | `analysis.promotion.PromotionRegistry` |
| Paper execution | Exact artifact-built target-native desk; signals use only the prior completed NYSE session (D close), while pricing and orders use a provider-timestamped quote observed within five minutes in the current session (D+1); durable independent broker ledger and LocalBook; safe model checkpoints; idempotent target orders; explicit costs; pre/post one-cent reconciliation each cycle | `deployment.rehearsal`, `brokers.paper_trader` |
| Paper qualification | Independently recomputed, content-addressed cycle/quote/order/reconciliation evidence tied to the research/config/code hash; at least 20 cycles across 15 real NYSE sessions, a completed flat buy/sell round trip and 41 reconciliation checks; zero derived errors, unknown/open orders, or reconciliation failures; valid audit chain and a fitted, restorable final model checkpoint | `PaperValidationArtifact` |
| Live approval | A second named actor, distinct from the paper approver, explicitly binds the exact research, passing paper, and independent-replication hashes | `PromotionRegistry.promote(... live_eligible ...)` |
| Live staging | Expiring account/environment/universe manifest, absolute order/day limits, generic gross/per-name limits, distinct risk and operations approvers | `deployment.state` |
| Activation | Production account match, connected auth, clean exact build, verified audit, engaged kill switch, initialized/flat book, no working orders/reservations, strict reconciliation, exact desk/config, and exact restoration of a current-or-prior-session paper checkpoint | `deployment.live.FoundationLiveController.prepare` |
| Running/rollback | Manifest authorization immediately below strategy code, persistent daily limits, audited state, engage-kill-first pause/revoke and final reconciliation | `FoundationExecutionGuard`, `FoundationLiveController` |

Production `LiveExecutionContext.configure_session` now rejects an arbitrary desk
unless it receives the opaque capability returned by this preflight. Research
desk factories remain available for research only.

## Invariants

- Evidence, approval, and activation are separate facts.
- An approval is for a content hash, never a mutable desk name.
- Artifact parameters are the sole source used to construct a deployed desk.
- `foundation-target-v1` uses 10% desk allocation, GBM, target-native execution,
  and the full versioned signal configuration in its artifact.
- The timing contract is close D to trade D+1: every signal frame ends at the
  prior completed NYSE session, and only the fresh current-session quote is used
  for execution. A current partial bar is never a signal input.
- Qualifying paper cycles are prospective, strictly ordered, within five
  minutes of wall clock, and run only during NYSE regular hours. Historical
  replay cannot count toward the soak.
- The exact research universe is used in paper and live. Removing names would
  change position sizing and model input, so a manifest cannot deploy a subset.
- Research artifacts never promote themselves.
- Paper promotion recomputes the complete raw report statistics and replays the
  on-disk integrity chain through its terminal result receipt.
- Live approval cannot skip paper approval or paper qualification.
- A different paper run cannot reuse an existing live approval.
- The complete fitted paper checkpoint is inside the paper artifact. Live
  preparation restores and re-exports the exact same safe JSON state; a hash,
  model-spec, cadence, universe, or payload mismatch blocks activation.
- A paper-to-live handoff is valid only when its last paper cycle is from the
  current or immediately preceding NYSE session. An exact but old model is not
  treated as current evidence.
- Production automation cannot be configured through the raw desk/session path.
- V1 starts in a dedicated flat account. Existing holdings, unknown orders, or
  active reservations block activation.
- The kill switch stays engaged through preflight and is released only by the
  controller for the bounded first-cycle handshake or an approved running loop.
- Exits are never blocked by opening notional limits.
- Pause and revoke engage the kill switch before stopping automation.
- Controlled live orders require an E*TRADE `REALTIME` quote carrying a
  provider timestamp no more than 60 seconds old. Delayed, untimestamped,
  future, or stale quotes are rejected before order authorization and again by
  the patient-execution quote adapter.
- The executable-side quote used for durable authorization becomes a one-shot
  notional envelope for that exact intent. Every patient child/replacement and
  cumulative fill must stay inside it; quote drift cancels and halts instead of
  spending beyond the order/day reservation. Exit envelopes remain uncapped so
  a capacity limit cannot trap an existing position.

## Operator workflow

Use the project interpreter for every command:

```bash
PY=.venv/bin/python
PIPELINE="scripts/foundation_pipeline.py"
REGISTRY="var/foundation-promotion"
export TRADING_DB_PATH="$(pwd)/var/foundation-control.db"
CONTROL_DB="$TRADING_DB_PATH"
```

`CONTROL_DB` and the production `LiveExecutionContext` database must resolve
to the same path. The deployment state, audit chain, kill switch, LocalBook,
and risk reservations intentionally share that transaction boundary.

### 1. Produce research evidence

First ingest the required PIT warehouse tables (`tickers`, `sep`, `daily`, and
`actions`). Commit all code changes: qualifying research refuses an unknown or
dirty Git checkout. Before touching the holdout, freeze the complete protocol
and register every attempted strategy/model/parameter variant—including failed
and abandoned historical attempts—in the append-only integrity ledger. The
ledger CLI accepts strict JSON matching `ResearchProtocol.create` and
`TrialRegistration.create`; it exposes no trial-count override. Protocol,
trial, and outcome JSON must include explicit `frozen_at`, `registered_at`, and
`recorded_at` timestamps respectively. Holdout commands likewise require
`--opened-at` / `--decided-at`, keeping exact retries content-identical.

```bash
INTEGRITY="$REGISTRY/research-integrity"
PROGRAM="stock-options-trader"

$PY -m scripts.research_integrity \
  --ledger-root "$INTEGRITY" --program-id "$PROGRAM" \
  freeze-protocol --json research-protocol.json

$PY -m scripts.research_integrity \
  --ledger-root "$INTEGRITY" --program-id "$PROGRAM" \
  register-trial --json foundation-trial.json
```

Keep the returned `protocol_hash`, `trial_hash`, and ledger `head_hash` in the
review record. Anchor the head outside this mutable checkout (for example an
object-locked record or signed release) so deleting a valid ledger tail is
detectable.

```bash
$PY $PIPELINE research \
  --registry-root "$REGISTRY" \
  --warehouse-dir pit_warehouse \
  --integrity-root "$INTEGRITY" \
  --program-id "$PROGRAM" \
  --protocol-hash PROTOCOL_HASH \
  --trial-hash TRIAL_HASH \
  --holdout-id foundation-oos-v1 \
  --reviewer independent-reviewer@example.com \
  --start HOLDOUT_START \
  --end HOLDOUT_END \
  --max-symbols 100
```

The command opens the named holdout before universe selection or strategy
execution, derives DSR trial breadth from the count captured by that opening,
persists the complete raw report and immutable summary, and writes one
permanent pass/fail decision back to the ledger plus an immutable terminal
receipt under the same registry. Promotion recomputes PSR, DSR, net return,
turnover, costs, and regime results from that raw report and re-verifies the
protocol/trial/opening/outcome/decision chain. It creates no deployment
approval. If its printed decision is `research_only`, that protocol/holdout has
failed permanently. Improve the next protocol using development data; do not
reuse the holdout. `--start` and `--end` are intentionally mandatory: all
locally inspected 2015–2024 data is development evidence and must not be used
as the example or default holdout window.

If `--start`/`--universe-as-of` is not an exchange session (for example,
2015-01-01), membership is still frozen at that requested date but ranking uses
the most recent DAILY market-cap observation on or before it. The artifact
records both dates and exact coverage counts. Every eligible name must have a
finite positive market cap; missing coverage fails instead of falling back to
alphabetical ticker order. `--universe-as-of` cannot be later than `--start`.

The Foundation protocol is not an opaque note. Its data plan exactly binds the
snapshot and PIT-universe method; its evaluation plan declares aggregate net
OOS as primary, calendar years as diagnostics, realized execution costs, and
independent replication; and its decision rules bind the promotion-policy hash
and terminal one-shot failure. The registered trial must exactly equal
`foundation_trial_inputs(...)`, including the full strategy configuration,
window, universe rule, engine economics, regimes, and policy hash. Code, data,
candidate, and seed identities must also match the run.

Inspect an artifact:

```bash
$PY $PIPELINE inspect --registry-root "$REGISTRY" --artifact RESEARCH_HASH
```

### 2. Approve the exact artifact for paper

```bash
$PY $PIPELINE approve-paper \
  --registry-root "$REGISTRY" \
  --artifact RESEARCH_HASH \
  --actor research-reviewer@example.com
```

### 3. Run forward paper sessions

Start a durable run once:

```bash
$PY $PIPELINE paper-start \
  --registry-root "$REGISTRY" \
  --db "$CONTROL_DB" \
  --run-id foundation-forward-001 \
  --artifact RESEARCH_HASH \
  --initial-capital 100000
```

Run one cycle during each observed NYSE execution session. `paper-step` takes
its timestamp from the process clock and accepts no historical `--as-of`
override. For every researched symbol, it fetches signal history only through
the prior completed NYSE session and separately calls OpenBB's quote endpoint.
The quote must carry the market provider's own timestamp; the request time is
never substituted as evidence. In practice, configure FMP or Intrinio quote
credentials because an untimestamped yfinance snapshot is deliberately
rejected. No current-session partial daily bar enters the indicators. Missing,
untimestamped, delayed, stale, future-dated, unordered, or misaligned input
fails closed.

```bash
$PY $PIPELINE paper-step \
  --registry-root "$REGISTRY" \
  --db "$CONTROL_DB" \
  --run-id foundation-forward-001
```

In this example, signals are computed through Friday, 2026-07-10, and the
2026-07-13 quote is used to price and execute Monday's paper order.

Each CLI invocation reconstructs the desk from the exact artifact and restores
the last safe JSON-native model/cadence checkpoint. The checkpoint and completed
cycle are committed together; no executable estimator serialization is loaded.
A crash leaves a durable `STARTED` cycle. Re-running that same input resumes it
with the deterministic intent ID, while finalization refuses incomplete cycles.

Status and finalization:

```bash
$PY $PIPELINE paper-status \
  --registry-root "$REGISTRY" --db "$CONTROL_DB" \
  --run-id foundation-forward-001

$PY $PIPELINE paper-finalize \
  --registry-root "$REGISTRY" --db "$CONTROL_DB" \
  --run-id foundation-forward-001
```

Finalization is immutable. A failed result remains failed; start a new run
after correcting the cause. Do not delete its evidence.

The default qualification policy requires at least 20 successful cycles across
15 distinct NYSE execution sessions, at least one completed buy/sell round trip
ending flat in both broker and LocalBook, and at least 41 strict reconciliation
checks. These are minimums, not a replay shortcut. Result/report failures are
derived from the sealed facts rather than trusted from a summary counter, and
the run must have zero errors, unknown/open orders, or reconciliation failures.

### 4. Bind live approval to paper and replication evidence

Before approval, an independent implementation must be run against the exact
frozen protocol and warehouse snapshot. Reconcile its complete checkpoints
with `analysis.independent_replication`, then persist the passing evidence with
`ReplicationEvidenceStore(REGISTRY)`. Record the returned content hash as
`REPLICATION_HASH`. A missing artifact, any unresolved discrepancy, or a
protocol/data hash mismatch fails live approval. The replication contract must
freeze a nonempty expected checkpoint-key manifest; agreeing empty or partial
outputs fail even when both implementations omit the same checkpoints.

```bash
$PY $PIPELINE approve-live \
  --registry-root "$REGISTRY" \
  --artifact RESEARCH_HASH \
  --paper-artifact PAPER_HASH \
  --replication-artifact REPLICATION_HASH \
  --actor live-risk-owner@example.com
```

### 5. Stage narrow live authority

Use a dedicated production account that is flat. Start with a short expiry and
small absolute limits:

```bash
$PY $PIPELINE manifest-stage \
  --registry-root "$REGISTRY" \
  --db "$CONTROL_DB" \
  --artifact RESEARCH_HASH \
  --paper-artifact PAPER_HASH \
  --account "$ETRADE_ACCOUNT_ID_KEY" \
  --expires-at "$DEPLOYMENT_EXPIRES_AT" \
  --actor deployer@example.com \
  --max-order-notional 1000 \
  --max-daily-notional 2500 \
  --max-daily-orders 3 \
  --interval-minutes 15
```

Two different people approve it in order:

```bash
$PY $PIPELINE manifest-approve-risk \
  --db "$CONTROL_DB" --manifest MANIFEST_HASH \
  --actor risk-owner@example.com

$PY $PIPELINE manifest-approve-ops \
  --db "$CONTROL_DB" --manifest MANIFEST_HASH \
  --actor operations-owner@example.com
```

The second command rejects the risk approver's identity. Check state with
`manifest-status`.

### 6. Prepare and start in the application composition root

Set `DEPLOYMENT_EXPIRES_AT` to a near-term timezone-aware timestamp appropriate
for the observation window; do not reuse an old timestamp as standing authority.

The persistent application process must already have its canonical production
`LiveExecutionContext`,
connected E*TRADE auth, initialized LocalBook, persistent audit/kill switch, and
the live indicator data function. This is the application composition API, not
a short-lived CLI command: the process that calls `start()` owns the scheduler
thread for the deployment's lifetime. The activation code is intentionally
small:

```python
from analysis.promotion import PromotionRegistry
from deployment.live import FoundationLiveController
from deployment.state import DeploymentStore

controller = FoundationLiveController(
    store=DeploymentStore(control_db, live_context.audit),
    promotion_registry=PromotionRegistry(registry_root),
    context=live_context,
)

# The switch must already be engaged. prepare() starts no worker.
verified = controller.prepare(
    manifest_hash,
    data_fn=live_foundation_data,
    actor="production-operator@example.com",
)

# Rechecks auth/audit/orders/reservations/reconciliation, then starts.
controller.start(verified, actor="production-operator@example.com")
```

`prepare()` restores the exact final model/cadence state sealed in `PAPER_HASH`.
`start()` is accepted only during NYSE regular hours. Do not call
`scheduler.start()` directly; the GUI endpoint also refuses direct actions for
a controlled scheduler. The controller records `ARMED`,
disengages the kill switch for one immediate bounded cycle, and waits for its
result. It then holds the scheduler, re-engages the kill switch, proves there
are no working risks, and reconciles. Only an `ok` cycle and clean post-cycle
state become `RUNNING`; the controller then disengages the switch and releases
the scheduler. A halt, exception, timeout, drift, or unsafe report kills, stops,
and pauses the manifest.

### 7. Pause or revoke

```python
controller.pause(
    verified,
    actor="production-operator@example.com",
    reason="planned observation stop",
)

# Permanent for this manifest:
controller.revoke(
    verified,
    actor="risk-owner@example.com",
    reason="evidence or operational authority withdrawn",
)
```

Both paths engage the kill switch first, stop/join the scheduler, attempt a
strict reconciliation, and persist the reason. Resuming requires a new explicit
manifest and approvals; `PAUSED` has no resume transition and V1 does not
silently restart after a process failure.

## State sequence

```text
research_only
    └─ passing immutable research artifact
       └─ PAPER_ELIGIBLE + named paper approval
          └─ durable paper cycles + reconciliation evidence
             └─ passing PAPER_HASH + independently reconciled REPLICATION_HASH
                └─ LIVE_ELIGIBLE approval bound to all three hashes
                   └─ STAGED
                      └─ RISK_APPROVED (actor A)
                         └─ OPS_APPROVED (actor B)
                            └─ ACTIVATED (preflight passed; kill engaged)
                               └─ ARMED
                                  └─ RUNNING
                                     ├─ PAUSED
                                     └─ REVOKED
```

Any missing or mismatched hash, weaker runtime risk policy, stale data, dirty
build, broken audit chain, disconnected auth, drift, open order, reservation,
non-flat V1 account, stale paper checkpoint, delayed/untimestamped live quote,
expired authority, or breached absolute limit fails closed.
