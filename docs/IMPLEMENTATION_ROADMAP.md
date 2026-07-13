# Implementation Roadmap

This roadmap is ordered by operational risk. Each phase must pass its focused
tests, the complete offline suite, Ruff, and the configured mypy scope before
the next phase begins. New strategy research is deliberately deferred until the
execution and evidence paths are trustworthy.

## 1. Canonical live execution context

**Goal:** replace the fragmented live singletons with one composition root that
owns the configured account's client, broker, persistent local book, executor,
reconciliation, working-order view, optional live session, and scheduler.

**Implementation:**

- Add `LiveExecutionContext` with an immutable database/environment/account
  identity and a thread-safe lifecycle.
- Route preview validation, placement registration, cancellation, reconciliation,
  order status, scheduler access, and shutdown through that context.
- Scope the local book to the configured account and persist placed orders so
  cumulative fills can be applied idempotently after refresh or restart.
- Treat the broker as the authoritative source for working-order state.
- Allow explicit session/scheduler configuration, but never select a desk or
  start automation automatically.
- Support patient execution for equities only until a real option/package quote
  adapter exists.

**Acceptance:** concurrent first access produces one context; account/auth changes
cannot reuse stale components; reconciliation uses one atomic persistent snapshot;
open orders survive context reconstruction; shutdown is idempotent; no scheduler
starts implicitly.

## 2. Transactional kill switch and audit

**Goal:** a switch transition and its audit record either both commit or neither
does.

**Implementation:** move canonical audit-row creation into the same SQLite
transaction as the state transition, preserve hash-chain ordering under
concurrency, and make no-op flips idempotent.

**Acceptance:** injected failures roll back both state and audit; concurrent flips
produce one transition and one valid chain row; disengagement can never enable
trading without a durable audit record.

## 3. Pending-order risk reservations

**Goal:** approved but unfilled orders consume risk and cash capacity.

**Implementation:** introduce a reservation ledger keyed by intent/order identity;
reserve name, gross, cash, sector, and option-risk capacity before submission;
decrement it on partial fills and release it on cancellation, rejection, expiry,
or reconciliation. Revalidate against current prices immediately before every
fill/replacement.

**Acceptance:** overlapping pending orders cannot jointly exceed limits; partial
fills conserve reserved capacity; restart recovery reconstructs reservations from
durable nonterminal orders.

## 4. Idempotent target-position contract

**Goal:** strategies describe desired signed end-state rather than imperative
BUY/SELL/SHORT/COVER actions.

**Implementation:** add `TargetPosition`, `PortfolioSnapshot`, and a portfolio
constructor that diffs targets against filled positions plus reservations. Adapt
legacy strategies/desks through compatibility shims and migrate one simple desk
end to end before broad conversion.

**Acceptance:** evaluating the same target twice emits no duplicate order; partial
fills emit only the remaining delta; restart/reconciliation produces the same
orders as uninterrupted execution.

## 5. Atomic multi-leg structures

**Goal:** an option structure remains one risk and execution unit.

**Implementation:** add a structure intent containing all legs, package quantity,
net price, max loss, and Greeks. Risk-check and reserve the package once; backtest
and live execution fill/cancel it atomically; persist package-level status and
leg-level positions.

**Acceptance:** no partial naked structure can be created by local logic; rejected
packages leave no phantom risk; package fill quantities and cash reconcile across
restart.

## 6. Realistic portfolio mechanics

**Goal:** remove the most material cash-account approximations.

**Implementation:** add configurable Reg-T/portfolio-margin rules, short-borrow
availability and fees, margin interest, option exercise/assignment, expiration
settlement, and an early-exercise model for American options.

**Acceptance:** opening orders fail when buying power or borrow is unavailable;
daily financing accrues causally; assignment/exercise produces correct stock,
cash, and audit entries; live and backtest accounting share the same contracts.

## 7. Historical-data integrity

**Goal:** make cached and point-in-time data internally consistent.

**Implementation:** stop granting full cache coverage to truncated fetches; align
correlation returns by session date; adjust split volume consistently with price;
model delisting payouts and corporate actions; version source snapshots and data
quality flags.

**Acceptance:** incomplete history refetches; correlation never pairs different
dates by ordinal position; split participation is invariant; delisting/action
fixtures reconcile to hand-computed returns.

## 8. Authoritative promotion pipeline

**Goal:** one versioned decision determines whether a strategy is research-only,
paper-eligible, or live-eligible.

**Implementation:** consolidate DSR/PSR, OOS-fold, multiple-testing, cost, turnover,
and regime requirements into one validator. Persist immutable artifacts containing
data version, universe, parameters, seed, dependency versions, code SHA, results,
and decision. Promotion updates a registry only by referencing a passing artifact.

**Acceptance:** identical inputs reproduce the artifact hash and decision; failed
or missing evidence cannot be promoted; the runtime loads only an explicitly
approved immutable version.

## 9. Built-container runtime smoke test

**Goal:** CI proves the image starts, authenticates, and serves its main surfaces.

**Implementation:** build the image, run it with isolated test credentials and a
temporary volume, wait for `/health`, assert unauthenticated requests fail, assert
authenticated page/API requests succeed, then inspect logs and shut it down.

**Acceptance:** missing runtime packages, broken gunicorn imports, authentication
regressions, unhealthy startup, and writable-volume failures break CI.

## 10. Behavior-preserving modularization

**Goal:** reduce the change risk in the largest modules.

**Implementation:** extract live auth/context/order/reconciliation blueprints,
backtest data/fill/accounting/report services, and desk-specific model/book/risk
components behind typed contracts. Move code in small slices with characterization
tests and golden outputs.

**Acceptance:** public routes and report schemas remain compatible; deterministic
goldens do not move except through explicitly documented migrations; module sizes
and dependency cycles materially decrease.

