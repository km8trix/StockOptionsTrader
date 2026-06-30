# StockOptionsTrader — Architecture & Planning Brief

> **Purpose of this file.** This is a context/handoff document distilled from a design
> conversation. Drop it into the StockOptionsTrader repo (e.g. as `docs/PLANNING.md`,
> or fold the relevant parts into `CLAUDE.md`) and use it to seed a planning session.
> It captures the architectural decisions, their rationale, the two starter strategies,
> and the open decisions that still need to be made. It is an engineering brief, not
> investment advice — nothing here is a recommendation to trade or a claim that any
> strategy is profitable.

---

## 1. Project context & goal

- **Goal:** build a quantitative, systematic, automated trading application.
- **Scale:** single operator / personal project (not an institutional or multi-team build).
- **Timescale:** second-to-day. Explicitly *not* true HFT — no colocation, FPGA,
  kernel-bypass, or microsecond concerns. This means Python is adequate and iteration
  speed should be optimized over raw latency.
- **Instruments:** options-centric (the existing codebase is `StockOptionsTrader`), with
  directional signals potentially expressed on underlyings first (see §7).
- **Existing asset:** a personal codebase, `StockOptionsTrader`. An open decision is
  whether to refactor it or rebuild — see §10. The rebuild instinct usually comes from
  signal logic and order routing being tangled together; if that's the case here, it
  argues for a clean re-architecture around the boundaries below.

---

## 2. The load-bearing decision: event-driven core with backtest–live parity

This is the single most important and hardest-to-change decision, so make it first.

- **Vectorized** engines (operate on whole price series at once) are fast and great for
  *research/screening*, but do not translate to live trading and quietly permit lookahead bias.
- **Event-driven** engines (process one market event at a time, as it would arrive live)
  are slower but allow the **same strategy code to run in both backtest and live**.

**Decision:** architect around an **event-driven core**. Use vectorized tooling only as a
research accelerator that feeds candidates into the event-driven validator. The property
this buys is **backtest–live parity**: strategy code is written once and cannot tell whether
it is running on historical data or this morning's live feed.

The classic failure that forces a rebuild: signal logic tangled into vectorized research
code, with a separate bolted-on live path. The two paths drift, the backtest lies, and you
find out with real capital.

---

## 3. Layered architecture

One agnostic engine. Strategies are plugins. The layers, with the invariant each must hold:

### Data layer
- Connects to historical store + live feed and exposes them through **one identical interface**
  (e.g. a single `get_bar()` / `on_event()` contract). The strategy must not be able to tell
  historical from live.
- **Invariant:** point-in-time correctness. Asking "what was knowable at 10:31am" returns only
  what was knowable then — no peeking ahead.
- Over-engineer this early; most hidden bugs accumulate here.

### Strategy core (the plugin slot)
- Pure: no I/O, no broker calls, no wall-clock reads. Everything it needs (time, prices,
  current positions) is **injected**.
- Consumes data, emits an **intent** (target position / score / entry-exit), and nothing else.
- This purity is what makes parity real rather than aspirational.

### Research track (offline)
- **Backtester:** event-driven, shares code with the live track.
- **Validation harness:** the overfitting gauntlet — walk-forward, out-of-sample,
  multiple-testing correction (deflated Sharpe, probability of backtest overfitting),
  realistic cost/slippage modeling, regime/sub-period robustness. Treat as a **gate**, not a formality.
- Vectorized screening can sit in front of this for speed, but survivors are re-run through
  the event-driven backtester before being trusted.

### Live track
- **Portfolio construction:** turns strategy intents + current holdings into desired position
  *deltas* (sizing, pyramiding/scaling lives here).
- **Risk layer (gatekeeper):** a single chokepoint every order passes through, regardless of
  which strategy generated it. Owns max position, max drawdown, exposure limits, kill-switch,
  per-trade limits. Can veto or scale down any order.
- **Execution / OMS:** works orders against the broker, handles order types, partial fills,
  rejections, and feeds fills back into state.

### State, persistence & monitoring (spans everything)
- **Invariant:** reconciliation. What the system *thinks* it holds vs. what the broker *says*
  it holds. These drift (missed fills, restarts, partials); silent drift causes wrong-sized or
  duplicate orders. Reconcile on every startup and periodically while running; alert loudly on mismatch.
- Logging, dashboards, alerting, audit trail.

---

## 4. Cross-cutting principles

- **Backtest–live parity** — same strategy core in both tracks.
- **Point-in-time correctness** — no lookahead anywhere in the data path.
- **Clock abstraction** — the strategy reads time from an injected clock: simulated in
  backtest, wall-clock live. Getting this wrong reintroduces lookahead everywhere.
- **Reconciliation** — treat broker state as the source of truth and reconcile against it.
- **Reproducibility** — config-driven, versioned runs; a backtest should be re-runnable to the same result.

---

## 5. Strategy model: one agnostic engine, strategies as plugins

- "Strategy-agnostic" is a **property of the infrastructure**, not a kind of trading. The data
  layer, backtester, risk gatekeeper, and execution don't know which strategy is plugged in.
- **Do not build two pipelines** (one "generic quant", one "specific strategies"). That
  duplicates all plumbing, doubles the reconciliation surface, and breaks parity for the
  duplicated strategies. Build **one engine**; ORB, MTP, and any mined signal are all plugins
  behind the same data-in / intent-out contract.
- **The real seam is research vs. production**, sharing the strategy core. That split already
  exists in the architecture and is the correct one.
- **Secondary distinction — sourcing (where a strategy comes from):**
  - *Discovered:* output of the alpha-research search (mine a feature space, measure information
    coefficients, correct for multiple testing). You don't know the rule in advance.
  - *Specified:* the rules are written down already (ORB, MTP). You implement and then stress-test them.
  - These are different *research front-ends*, but both emit the **same artifact** (a strategy
    module) into the **same engine** through the **same validation gauntlet**. Let the workflows
    diverge in research tooling, never in the trading pipeline.
- **Operational separation is fine and expected:** run intraday ORB and daily MTP as separate
  *runners/processes* (different timescales and failure modes; an intraday crash shouldn't take
  down the daily system) — but they are instances of the *same framework* hitting the *same*
  data layer and risk discipline. Multiple **execution adapters** (e.g. IBKR adapter, crypto
  adapter) behind one OMS interface, not duplicate stacks.

---

## 6. The two starter strategies (both "specified")

### ORB — Opening Range Breakout
- **Mechanics:** define the opening range (high/low of the first N minutes after the open;
  5/15/30 common), go long on a break above the range high / short below the low, stop at the
  opposite end or an ATR multiple, flatten by the close. Pure intraday momentum.
- **Stresses:** intraday bars (1-min or finer), tight point-in-time correctness on the opening
  window, the intraday clock abstraction, time-sensitive (but not HFT) execution, end-of-day
  flatten logic, and the PDT rule in the risk layer (relevant under the $25k threshold).
- **Note:** published research exists (e.g. Zarattini–Aziz 5-min ORB on leveraged ETFs) but is
  regime- and cost-sensitive. Validate, don't assume.

### MTP — Market Timing Pro (Travis Woo)
- **What it actually is:** a long-term, daily-timeframe **trend-following** strategy in the
  Turtle tradition. Reported mechanics: no take-profit targets; wide ATR-based stops (~3–4 ATR);
  entries on new all-time highs or "dip buys" above the 200-day MA; incremental **pyramiding**
  during trends; the 200-day SMA as a regime filter (no longs below it); a bear-market
  "auto-off switch"; a position-sizing plan. Marketed for stock indices, BTC, and gold.
- **Stresses:** daily bars (simple cadence), the regime filter (signal-core or portfolio
  overlay), and especially the **portfolio layer** (pyramiding = sizing-over-time across many
  bars) and **risk layer** (ATR stops, position sizing).
- **IMPORTANT CAVEAT — reimplement, don't integrate:** MTP the *product* is a closed TradingView
  indicator that is **signal-only** (it emits alerts for manual confirmation; it does not execute).
  It cannot be cleanly dropped into a Python event-driven engine — it's a black box on someone
  else's platform, and bridging its webhook alerts would break backtest–live parity. The
  *strategy logic*, however (Turtle-style trend following on the 200-SMA with ATR stops and
  pyramiding), is generic, public-domain mechanics. **Plan to reimplement the rules directly in
  the strategy core** so it backtests in your own engine with full parity. Do not try to wire in
  the product.

---

## 7. Options-specific considerations (StockOptionsTrader)

- ORB and MTP, as described, are **directional signals on underlyings** (indices, ETFs, BTC,
  gold), not options strategies per se.
- You *can* express the directional view through options (ORB breakout via long calls/puts; a
  trend position via leveraged/defined-risk structures), but the moment you do, the **portfolio
  and risk layers inherit Greeks-and-expiry complexity** — you are managing delta/gamma/theta/vega
  and expiration, not just dollar exposure. This is a materially harder modeling problem than spot.
- **Recommended sequence:** validate the signals on the underlying first, then layer options
  expression on afterward. This keeps the first validation clean and isolates "is the signal real"
  from "is the options expression sound."
- The risk gatekeeper for an options book needs Greek-level limits (net delta, vega exposure,
  per-expiry concentration), not only position-size limits.

---

## 8. Strategy count & sequencing

- The engine supports many strategies; that is never the binding constraint. Research capacity,
  capital/per-strategy capacity, operational bandwidth, and signal decay are.
- **For a solo build: start with ONE strategy**, taken all the way through the lifecycle
  (research → validation → paper → small live → reconciliation running reliably) **before adding a
  second.**
- Add the second only when it is (a) genuinely **decorrelated** from the first and (b) the first
  is **boringly stable** in production.
- The characteristic solo failure mode is assembling a paper "portfolio of strategies" before a
  single one has survived live capital — multiplying half-validated, half-monitored things that
  can fail quietly.

---

## 9. Correlation: ORB vs. MTP

- **Both are momentum strategies** (ORB = intraday momentum; MTP = low-frequency trend following).
  They share a latent style factor and are **not** structurally orthogonal.
- **Average correlation is likely low** (rough guess ~0.1–0.3, *must be measured*) because of
  frequency separation, opposite overnight-exposure profiles (ORB flat overnight; MTP almost
  entirely overnight/multi-day), differing holding periods, and possibly different instruments.
- **But tail correlation is the risk:** both win together in strong sustained trends and **lose
  together in choppy/whipsaw markets** (ORB eats false breakouts while MTP gets stopped out of
  reversing entries). The diversification thins out exactly in the regime where you needed it —
  the classic "momentum has correlated bad months" pattern.
- **Combination math:** you will get *some* but **less** than the idealized uncorrelated payoff
  (two uncorrelated Sharpe-0.5 strategies → ~0.7; these will land below that), and the blended
  drawdown will be deeper than the average correlation suggests.
- **Measure the right thing** (validation-track task): not the single static Pearson over the whole
  sample (it flatters and hides tail dependence). Compute **rolling correlation**, **regime-/down-
  month-conditional correlation**, and ideally **correlation of drawdowns**. Align both daily return
  series over the same out-of-sample window and inspect co-movement in the worst decile of months.
  If drawdown correlation is high, MTP is less a diversifier than a leveraged second momentum bet —
  size accordingly.

---

## 10. Build vs. buy (the refactor-vs-rebuild decision in disguise)

The deciding question: **is the engine itself the part you want to own, or is it undifferentiated
plumbing standing between you and signal research?**

- **Adopt a mature event-driven framework** — `nautilus_trader` is purpose-built for
  backtest–live parity (parity is designed-in rather than enforced by discipline) and is the
  strongest candidate to evaluate first. **QuantConnect LEAN** is the other serious open engine
  (more platform gravity).
- **Build your own** — most work and most overfitting risk in the plumbing, but the best teacher
  and total control. Justifiable mainly if the learning *is* a goal. If building: build the
  **event-driven core first**, and resist starting from vectorized research scripts (that is the
  path that produced the tangle being rebuilt away from).
- This maps directly onto whether to refactor `StockOptionsTrader` or rebuild: if signal research
  is the real interest, adopt an engine and spend effort on alpha; if owning the whole machine is
  the point, rebuild around the event-driven core.

---

## 11. Recommended tech stack (starting point)

- **Language:** Python (timescale makes it adequate; optimize iteration speed).
- **Data:** vendor API — Polygon, Alpaca, or Databento (cleaner history); or Interactive Brokers
  for data + execution from one source. Store as **Parquet** partitioned by date, or a time-series
  DB (QuestDB / TimescaleDB) if you outgrow flat files.
- **Broker:** **Interactive Brokers via `ib_async`** is the standard for serious retail algo
  trading and is strong for **options** (mature API, broad instrument coverage). Alpaca is simpler
  for equities-only.
- **Engine:** decision in §10 (`nautilus_trader` / LEAN / custom).

---

## 12. Recommended build sequence

Each step needs the previous one to be testable:

1. **Data layer + storage**, with the unified historical/live interface and point-in-time correctness.
2. **Event-driven backtester** against that data, with the clock abstraction. Verify the loop with a
   trivial buy-and-hold strategy end to end.
3. **Strategy core API**, then port/write one simple, well-understood signal. Prove the same code runs
   in the backtester.
4. **Validation harness** (the overfitting controls) — so every subsequent strategy faces the gauntlet.
5. **Portfolio + risk layers.** Even a one-strategy system benefits from a real risk gatekeeper before
   any live order flows. (For options: Greek-level limits here.)
6. **Execution against the broker's PAPER endpoint.** Run the full live track with zero real capital
   until reconciliation, fills, and state are boringly reliable.
7. **Small live allocation**, treating the first weeks of live trading as continued validation.

The recurring risks the whole design exists to neutralize: **parity** (does live match backtest),
**overfitting** (was the signal ever real), and **reconciliation** (does state match reality).

---

## 13. Open decisions for the planning session

Things the Claude Code plan should resolve:

1. **Build vs. buy the engine** — evaluate `nautilus_trader` vs. LEAN vs. custom against the goal
   (own the machine vs. focus on alpha). If keeping/refactoring `StockOptionsTrader`, assess how
   tangled signal/execution currently are.
2. **First strategy** — ORB (intraday) or an MTP-style trend follower (daily)? Daily is lighter on
   infrastructure (no intraday clock pressure) and may be the simpler first end-to-end loop.
3. **Underlying-first vs. options-first** — recommended: validate the chosen signal on the underlying
   before adding options expression.
4. **Data vendor + storage** choice (and whether the broker doubles as the data source).
5. **Repo structure** — how to lay out data / strategy / research / live / risk / execution as cleanly
   separable modules with the plugin contract at the center.
6. **Reuse audit of `StockOptionsTrader`** — what (if anything) survives the re-architecture.
7. **Learning-loop scope & cadence** — start diagnostics-only (recommended)? If/when to enable gated
   periodic retraining, and at what cadence *per strategy* (ORB resolves daily; an MTP-style trend
   follower only every several weeks). Explicitly decide **not** to build online/RL policy adaptation.
   See §14.

---

## 14. Nightly learning component (gated periodic retraining)

A learning loop that adapts the system from realized outcomes is legitimate and desirable —
but the naive version ("each night, refit on the day's positions, improve tomorrow") is an
**overfitting footgun**. One day is a near-meaningless sample in a low-signal, non-stationary,
adversarial system; a fast nightly refit chases noise and overfits to a regime that's already
ending. This component is a **research-track** addition. The live track only ever reads a
**versioned, immutable model artifact** — it is never hot-patched by last night's fit.

### "Improve future decisions" is four different things (safe → dangerous)

1. **Diagnostics & attribution** — per-trade P&L attribution, realized vs. assumed slippage,
   fill quality, signal-decay tracking. Pure analysis, can't blow anything up. **Do this first.**
2. **Execution learning** — better slippage/fill models, order slicing. Lower risk (dense data,
   clear ground truth, fast feedback).
3. **Periodic gated model retraining** — refit the alpha model on an expanding window through the
   full validation gauntlet. Legitimate, but **periodic and gated**, never nightly-on-one-day.
   (This is the bucket the original request maps to.)
4. **Online / RL policy adaptation from daily P&L** — most seductive, by far most dangerous (noisy
   reward, non-stationary environment, can't generate samples). **Do not build this.**

### Horizon-matched labeling rule (critical, strategy-specific)

At tonight's close, most of today's positions are **unresolved**, so they **cannot be labeled yet**.
The label horizon must match the holding horizon:
- **ORB** is flat by the close → today's trades resolve same-day → labelable nightly.
- **MTP-style trend follower** holds weeks–months → today's entries are unresolved for a long time.
  Labeling them by tonight's mark trains on unrealized **noise**. It can only learn on a multi-week cadence.
- Retrain frequency should track the count of **new, resolved, independent** outcomes — **not the calendar.**

### Two-speed architecture with a deploy gate

- **Nightly (always, safe):** run diagnostics/monitoring and append **point-in-time-correct** rows
  (only for newly *resolved* outcomes) to the training dataset. Writes only to dashboards and the
  dataset — **cannot change what trades tomorrow.**
- **Periodic + gated:** retrain a *candidate* on an expanding, walk-forward window → pass it through
  the **same out-of-sample, cost-aware validation gate** any new strategy faces → version → stage →
  promote. The live engine loads the promoted **versioned artifact** read-only and logs which version
  made each decision (clean attribution + one-line rollback).
- **Point-in-time discipline extends into this loop.** Training rows must capture only what was
  knowable at decision time; snapshotting end-of-day values the strategy didn't have reintroduces
  lookahead into the training set (the model then validates beautifully and fails live).

### Recommended first build

Build the **diagnostics / signal-decay layer only**, nothing downstream. It can't hurt you, and it
answers the question the whole retraining apparatus exists to serve: *is the signal genuinely decaying,
or was this just a noisy week?* Defer the retraining loop until a single strategy is stable in live
paper trading and enough **resolved** outcomes have accumulated to validate against. Skip the RL idea.

Interface stubs for this component are in **Appendix B**.

---

## 15. Suggested first prompt for the Claude Code session

> "Read `docs/PLANNING.md`. Help me produce a concrete build plan for StockOptionsTrader based on it.
> Start by proposing a repo/module layout that enforces the layer boundaries (data, strategy core,
> research/backtest, portfolio, risk, execution, state) with strategies as plugins behind a single
> data-in/intent-out contract. Then recommend build-vs-buy for the engine given that I want
> [own the machine / focus on alpha — pick one], and lay out the first vertical slice: data layer +
> event-driven backtester + one strategy + validation, runnable end to end on paper. Flag the
> options-specific work (Greek-level risk limits) and the nightly learning component (§14) as later,
> gated phases — build the diagnostics layer before any retraining, and have the live engine read a
> versioned model from a registry rather than ever being hot-patched. Before writing code, list the
> decisions from section 13 you need me to make."

---

## Appendix A — Core interface stubs

Concrete Python contracts for the plugin architecture. These are stubs (signatures +
docstrings encoding the invariants), not implementations — the point is that the layer
boundaries and the "everything is a plugin" model are real code from line one. Target
Python 3.11+. Style: `@dataclass` for value types, `Protocol`/`ABC` for interfaces,
`Decimal` for all money/quantities.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence, Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable
```

### A.1 Value types

```python
class Side(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class OptionRight(Enum):
    CALL = "call"
    PUT = "put"

@dataclass(frozen=True)
class Instrument:
    """Identifies a tradeable. Linear instruments (equity/etf/crypto/future) leave
    the option-only fields None; an option populates them."""
    symbol: str
    asset_class: str                 # "equity" | "etf" | "crypto" | "option" | "future"
    underlying: str | None = None    # option-only
    expiry: datetime | None = None   # option-only
    strike: Decimal | None = None    # option-only
    right: OptionRight | None = None # option-only
    multiplier: int = 1              # options typically 100

@dataclass(frozen=True)
class Bar:
    """One OHLCV bar — the unit the strategy core consumes. Identical shape whether
    produced from the historical store or the live feed. That sameness IS parity.
    `ts` is the bar CLOSE time: the moment this bar becomes 'known'."""
    instrument: Instrument
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

@dataclass(frozen=True)
class Greeks:
    delta: Decimal = Decimal(0)
    gamma: Decimal = Decimal(0)
    theta: Decimal = Decimal(0)
    vega: Decimal = Decimal(0)

@dataclass(frozen=True)
class Position:
    """Current holding. `greeks` is populated for options, None for linear instruments."""
    instrument: Instrument
    quantity: Decimal                # signed; negative = short
    avg_price: Decimal
    greeks: Greeks | None = None

@dataclass(frozen=True)
class Order:
    """A concrete instruction the execution layer can submit. Produced by portfolio
    construction, approved by risk, sent by an adapter."""
    instrument: Instrument
    side: Side
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None

@dataclass(frozen=True)
class Fill:
    order: Order
    filled_qty: Decimal
    fill_price: Decimal
    ts: datetime

@dataclass(frozen=True)
class TargetPosition:
    """What a Strategy EMITS — a desired end-state, not an order. Idempotent:
    re-emitting the same target is a no-op. The portfolio layer diffs targets against
    current holdings to produce Orders. This is what keeps strategies pure and the
    system reconciliation-friendly."""
    instrument: Instrument
    target_quantity: Decimal         # signed desired holding; 0 = flat
    stop: Decimal | None = None      # optional protective stop the risk layer may use
    reason: str = ""                 # for logging/audit
```

### A.2 Injected services (the strategy depends only on these)

```python
@runtime_checkable
class Clock(Protocol):
    """Injected time source. The simulated clock advances with the event stream in
    backtest; the live clock returns wall-time. The strategy NEVER calls
    datetime.now() itself — direct wall-clock reads are how lookahead leaks in."""
    def now(self) -> datetime: ...

@runtime_checkable
class MarketDataView(Protocol):
    """Read-only, point-in-time-correct market access handed to the strategy.
    Implementations MUST return only what was knowable at clock.now() — never future
    data in backtest."""
    def last_price(self, instrument: Instrument) -> Decimal | None: ...
    def history(self, instrument: Instrument, lookback: int) -> Sequence[Bar]: ...

@runtime_checkable
class DataFeed(Protocol):
    """Produces the event stream that drives the engine. A historical feed replays the
    store; a live feed yields from the broker/vendor socket. Same interface -> same
    engine loop -> parity."""
    def stream(self) -> Iterator[Bar]: ...

class ExecutionAdapter(ABC):
    """One interface, many venues (paper, IBKR, crypto exchange). The OMS talks only to
    this. Adding a venue is a new adapter, not a new pipeline."""

    @abstractmethod
    def submit(self, order: Order) -> str:
        """Submit an order; return a broker order id."""

    @abstractmethod
    def positions(self) -> Sequence[Position]:
        """The broker's view of holdings — the source of truth for reconciliation."""

    @abstractmethod
    def on_fill(self, callback: Callable[[Fill], None]) -> None:
        """Register a fill callback."""
```

### A.3 The Strategy base class — THE plugin slot

```python
@dataclass
class StrategyContext:
    """Everything a strategy is ALLOWED to touch. Note what's absent: no
    ExecutionAdapter, no broker handle, no datetime.now(). The strategy reads time and
    market data through here and returns intent. That is the entire contract."""
    clock: Clock
    market: MarketDataView
    positions: Sequence[Position]    # current holdings, read-only

class Strategy(ABC):
    """Every strategy — a discovered factor, ORB, an MTP-style trend follower —
    subclasses this and is treated identically by the engine. Pure: consumes data +
    context, emits TargetPositions, performs NO I/O."""

    def on_start(self, ctx: StrategyContext) -> None:
        """Optional warmup hook (e.g. preload indicators). Default: no-op."""

    @abstractmethod
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> Sequence[TargetPosition]:
        """Called once per bar by BOTH the backtester and the live engine with
        identical semantics. Return desired target positions (empty = no change)."""
```

### A.4 Portfolio construction

```python
class PortfolioConstructor(ABC):
    @abstractmethod
    def to_orders(
        self,
        targets: Sequence[TargetPosition],
        current: Sequence[Position],
    ) -> Sequence[Order]:
        """Diff desired targets against current holdings -> concrete orders. Pyramiding
        (the MTP-style scaling-in) lives here: a target_quantity that rises across bars
        becomes a sequence of add orders."""
```

### A.5 Risk gatekeeper — the single chokepoint

```python
@dataclass(frozen=True)
class RiskLimits:
    max_position_value: Decimal
    max_gross_exposure: Decimal
    max_drawdown: Decimal
    # Options-specific (leave None for a linear-only book):
    max_net_delta: Decimal | None = None
    max_vega: Decimal | None = None
    max_contracts_per_expiry: int | None = None

class RiskModel(ABC):
    """The SINGLE chokepoint. Every order from every strategy passes through here
    before execution. It can scale down or veto, and it owns the kill-switch."""

    @abstractmethod
    def evaluate(
        self,
        proposed: Sequence[Order],
        positions: Sequence[Position],
        limits: RiskLimits,
    ) -> Sequence[Order]:
        """Return the approved (possibly scaled, possibly empty) order set. For an
        options book this is where net-delta / vega / per-expiry checks live — not just
        dollar position size."""

    @abstractmethod
    def halted(self) -> bool:
        """Kill-switch state. While True, the engine submits nothing."""
```

### A.6 The engine loop — parity made concrete

```python
class TradingEngine:
    """The same loop drives backtest and live. Only the injected DataFeed, Clock,
    MarketDataView, and ExecutionAdapter differ. If this loop runs a strategy in
    backtest, the identical loop runs it live — that is the parity guarantee."""

    def __init__(
        self,
        strategy: Strategy,
        feed: DataFeed,
        clock: Clock,
        market: MarketDataView,
        portfolio: PortfolioConstructor,
        risk: RiskModel,
        execution: ExecutionAdapter,
        limits: RiskLimits,
    ) -> None:
        self.strategy = strategy
        self.feed = feed
        self.clock = clock
        self.market = market
        self.portfolio = portfolio
        self.risk = risk
        self.execution = execution
        self.limits = limits

    def run(self) -> None:
        ctx = StrategyContext(self.clock, self.market, self.execution.positions())
        self.strategy.on_start(ctx)
        for bar in self.feed.stream():
            ctx = StrategyContext(self.clock, self.market, self.execution.positions())
            targets  = self.strategy.on_bar(bar, ctx)                       # 1. intent
            orders   = self.portfolio.to_orders(targets, ctx.positions)     # 2. delta
            approved = self.risk.evaluate(orders, ctx.positions, self.limits)  # 3. gate
            if not self.risk.halted():
                for order in approved:
                    self.execution.submit(order)                           # 4. execute
            # 5. reconcile periodically: compare an internal ledger against
            #    self.execution.positions() and alert on drift (omitted here).
```

Wiring backtest vs. live — note the strategy object is byte-for-byte identical:

```python
# BACKTEST: historical feed, simulated clock, simulated fills
engine = TradingEngine(
    strategy=OpeningRangeBreakout(range_minutes=15),
    feed=HistoricalFeed(store, start, end),
    clock=SimulatedClock(),
    market=HistoricalMarketData(store),
    portfolio=TargetDiffConstructor(),
    risk=StandardRiskModel(),
    execution=SimulatedExecution(slippage_model),
    limits=limits,
)

# LIVE: same engine, swapped feed/clock/adapter — strategy code UNCHANGED
engine = TradingEngine(
    strategy=OpeningRangeBreakout(range_minutes=15),
    feed=IBKRLiveFeed(ib),
    clock=WallClock(),
    market=IBKRMarketData(ib),
    portfolio=TargetDiffConstructor(),
    risk=StandardRiskModel(),
    execution=IBKRExecution(ib),     # point at the PAPER endpoint first
    limits=limits,
)
```

### A.7 ORB and MTP as plugins (sketch)

```python
class OpeningRangeBreakout(Strategy):
    """Intraday momentum. Build the opening range over the first N minutes, target
    +unit on a break above the high / -unit below the low, flat by the close."""

    def __init__(self, range_minutes: int = 15, unit: Decimal = Decimal(1)) -> None:
        self.range_minutes = range_minutes
        self.unit = unit
        self._range_high: Decimal | None = None
        self._range_low: Decimal | None = None

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> Sequence[TargetPosition]:
        # pseudocode:
        #   reset _range_high/_range_low at each new session open
        #   within the opening-range window: update range, target 0
        #   past window and close > _range_high: target +unit (stop at _range_low)
        #   past window and close < _range_low:  target -unit (stop at _range_high)
        #   near session close: target 0 (flatten)  <-- EOD flatten is mandatory
        return []


class TrendFollower(Strategy):
    """Daily trend following, MTP-style — REIMPLEMENTED from public Turtle-style
    mechanics, NOT the closed product. Long above the 200-SMA, dip-buys, ATR stops,
    pyramids by raising target_quantity, auto-off below the 200-SMA."""

    def __init__(self, sma_window: int = 200, atr_mult: Decimal = Decimal("3.5")) -> None:
        self.sma_window = sma_window
        self.atr_mult = atr_mult

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> Sequence[TargetPosition]:
        # pseudocode:
        #   bars = ctx.market.history(bar.instrument, self.sma_window)
        #   compute SMA(200) and ATR from bars
        #   if close < SMA200: target 0            <-- regime auto-off
        #   elif new high or qualifying dip-buy: raise target_quantity  <-- pyramid
        #   stop = close - atr_mult * ATR
        return []
```

### A.8 Conventions encoded above

- **`Decimal` for every price, quantity, and money value — never `float`.** Float rounding
  is unacceptable in a system that sizes orders and tracks P&L.
- **Strategies emit targets, not orders.** Idempotent and reconciliation-friendly; turning
  targets into orders is the portfolio layer's job, not the strategy's.
- **Strategies are pure.** `StrategyContext` deliberately omits the execution adapter and the
  wall-clock. If a strategy needs something not in the context, that is a design smell to
  resolve in review, not by widening the context.
- **One risk chokepoint.** Every order routes through `RiskModel.evaluate`, regardless of which
  strategy produced it. Options books extend `RiskLimits` with Greek-level caps.
- **Parity by construction.** `TradingEngine.run()` is venue-agnostic; backtest vs. live is a
  wiring choice, not a forked code path.

---

## Appendix B — Nightly learning interface stubs

Contracts for the §14 learning component. Same conventions as Appendix A (`Decimal`, `ABC`/`Protocol`,
purity). The load-bearing rule: **the live engine reads a versioned model; it never trains and is never
hot-patched.** The learning loop publishes candidates and promotes survivors through the registry.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
```

### B.1 Resolved outcomes (the only thing that becomes a label)

**Label decision:** the supervised target is **return on capital at risk** —
`label = realized_pnl / capital_at_risk` — where `capital_at_risk` is the **frozen initial
margin (buying-power requirement) at entry**. This denominator is scale-free, comparable across
instruments, equals true max loss for bounded structures, and is an *observed* broker number for
unbounded ones (rather than an invented scenario model). It measures decision quality per unit of
capital committed and keeps sizing in the portfolio layer, not the label. Distributional
risk-adjustment (Sharpe, drawdown, tails) stays **downstream** in the validation gate — never bake
it into the per-trade label.

```python
class RiskClass(Enum):
    """Outcome distributions differ sharply by structure; train (or at least validate) per class.
    Only NAKED_UNBOUNDED lacks a finite max loss — a short put (NAKED_BOUNDED) is capped at
    strike-to-zero. Margin-at-entry is a well-defined denominator for all four."""
    LONG_PREMIUM = "long_premium"        # long calls/puts; margin = premium = max loss
    DEFINED_RISK = "defined_risk"        # debit/credit spreads; margin = max loss
    NAKED_BOUNDED = "naked_bounded"      # short put / cash-secured; bounded but margin << max loss
    NAKED_UNBOUNDED = "naked_unbounded"  # short call / strangle; no finite max loss

class MarginModel(Enum):
    """Reg-T and portfolio margin give very different requirements for the same naked short.
    Store which one computed capital_at_risk so labels are never silently mixed across regimes."""
    REG_T = "reg_t"
    PORTFOLIO = "portfolio"

@dataclass(frozen=True)
class ResolvedOutcome:
    """A decision whose result is now KNOWN. Only resolved outcomes are training labels.
    A position still open at analysis time is NOT resolved and must not be labeled —
    labeling an unrealized mark trains on noise. The resolve horizon must match the
    strategy's holding horizon (ORB: same day; trend follower: weeks to months).

    label = float(realized_pnl / capital_at_risk). capital_at_risk is the INITIAL margin at
    entry, frozen — never ending or peak margin (maintenance margin rises as a naked short moves
    against you; a denominator that moves with the outcome leaks the result into the label)."""
    decision_id: str
    strategy_id: str
    model_version: str         # which model made the decision — closes the attribution loop
    opened_at: datetime
    resolved_at: datetime
    realized_pnl: Decimal
    capital_at_risk: Decimal   # FROZEN initial margin at entry; the label denominator
    risk_class: RiskClass      # bucket for separate models / validation; tail behavior differs
    margin_model: MarginModel  # which margin regime computed capital_at_risk
    label: float               # = float(realized_pnl / capital_at_risk), return on capital at risk
```

### B.2 The versioned model artifact

```python
@dataclass(frozen=True)
class ModelArtifact:
    """An immutable, versioned model the live engine can load. The metadata is what makes
    attribution and rollback possible — never deploy an unversioned blob."""
    version: str                          # e.g. "orb-classifier-v7"
    model_hash: str                       # hash of the serialized weights/params
    created_at: datetime
    strategy_id: str                      # which strategy this serves
    train_window: tuple[date, date]       # inclusive, point-in-time training range
    feature_spec: str                     # versioned hash/description of features used
    validation_stats: dict[str, float]    # OOS Sharpe, deflated Sharpe, PBO, cost-adjusted, ...
    artifact_uri: str                     # where the serialized model lives
```

### B.3 The model registry (boundary between learning loop and live track)

```python
class ModelRegistry(ABC):
    """The boundary between the learning loop and the live track. The learning loop PUBLISHES
    candidates and PROMOTES survivors; the live engine only ever READS the current production
    version. The live engine never trains and never hot-patches."""

    @abstractmethod
    def publish(self, artifact: ModelArtifact) -> None:
        """Store a newly trained candidate (not yet live)."""

    @abstractmethod
    def promote(self, version: str) -> None:
        """Mark a version as the production model for its strategy. Only call AFTER the
        validation gate passes. This is the single deploy action in the whole system."""

    @abstractmethod
    def current(self, strategy_id: str) -> ModelArtifact:
        """The production model the live engine loads. Immutable."""

    @abstractmethod
    def rollback(self, strategy_id: str) -> None:
        """Promote the previous production version — one-line recovery from a bad deploy."""
```

### B.4 The nightly analysis job (always-on, safe)

```python
@dataclass(frozen=True)
class NightlyReport:
    as_of: date
    attribution: dict[str, Decimal]       # per-strategy / per-signal P&L attribution
    realized_slippage: Decimal            # realized vs. assumed
    signal_decay: dict[str, float]        # rolling IC / decay metric per signal
    newly_resolved: Sequence[ResolvedOutcome]

class NightlyAnalysis(ABC):
    """Runs every night after the close. SAFE by construction: it only READS the day's fills
    and positions and WRITES to dashboards and the training dataset. It cannot change what
    trades tomorrow — there is no path from here to the live order flow."""

    @abstractmethod
    def run(self, as_of: date) -> NightlyReport:
        """Produce attribution + monitoring, and append point-in-time-correct rows (only for
        newly RESOLVED outcomes) to the training store."""
```

### B.5 The periodic retrain job (gated, not nightly)

```python
class RetrainJob(ABC):
    """Runs PERIODICALLY, not nightly — only when enough new resolved outcomes have accumulated.
    Produces a candidate ModelArtifact; it does NOT deploy. Deployment happens only if the
    candidate clears the validation gate (the Appendix A.5-style out-of-sample, cost-aware checks)
    and is then promoted in the ModelRegistry."""

    @abstractmethod
    def should_run(self, registry: ModelRegistry, strategy_id: str) -> bool:
        """True only when enough NEW, resolved, independent outcomes exist to justify retraining.
        Gate on outcome count, NOT the calendar."""

    @abstractmethod
    def train_candidate(
        self, strategy_id: str, train_window: tuple[date, date]
    ) -> ModelArtifact:
        """Fit on an expanding, point-in-time-correct window (walk-forward). Return an
        unpromoted candidate with validation_stats populated. Caller runs the gate, then
        registry.promote(...) only on pass."""
```

### B.6 How the live side consumes it

```python
# A model-backed strategy loads its pinned model at startup from the registry — it does NOT
# train. It records model_version on every decision so outcomes can be attributed later.
class ModelBackedStrategy(Strategy):          # Strategy is from Appendix A.3
    def __init__(self, strategy_id: str, registry: ModelRegistry) -> None:
        self.strategy_id = strategy_id
        self._registry = registry
        self._model: ModelArtifact | None = None

    def on_start(self, ctx: StrategyContext) -> None:
        self._model = self._registry.current(self.strategy_id)   # read-only, versioned

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> Sequence[TargetPosition]:
        # use self._model to score; tag emitted intents with self._model.version
        # for downstream attribution into ResolvedOutcome.model_version
        return []
```

### B.7 Conventions encoded above

- **Only resolved outcomes become labels**, and the resolve horizon matches the strategy's holding
  horizon. Never label open positions by their unrealized mark.
- **Label = return on capital at risk** (`realized_pnl / capital_at_risk`), with `capital_at_risk`
  the **frozen initial margin at entry**. Scale-free, comparable, equals max loss where bounded, and
  observed (not modeled) where unbounded. Keep sizing in the portfolio layer, not the label.
- **Use initial margin, never ending/peak margin.** A denominator that moves with the position
  (maintenance margin on a losing naked short) leaks the outcome into the label.
- **Fix one `margin_model` and store it.** Reg-T vs. portfolio margin produce very different
  requirements for the same naked short; tag each outcome so labels aren't mixed across regimes.
- **Winsorize asymmetrically.** Clip the long-premium *upside* (a 2000% lotto call otherwise dominates
  the loss function); do **not** clip the naked-short *downside* — the catastrophic left tail is the
  entire risk of a naked short, and a model blind to it is the failure mode. Let the gate's tail/
  drawdown metrics and the risk gatekeeper's caps handle the catastrophe; keep the label honest.
- **Separate models per `risk_class`.** Naked shorts are fat-tailed and negatively skewed in a way
  long premium isn't; pooling teaches a model they're the same animal. The per-`strategy_id` registry
  already supports this — split a strategy that mixes structures.
- **Distributional risk-adjustment stays downstream.** The label is per-trade capital efficiency;
  Sharpe / deflated Sharpe / drawdown live in the validation gate, and hard exposure caps in the risk
  gatekeeper. Don't make the label itself risk-adjusted.
- **The live track reads a versioned artifact and never trains.** All learning is offline; the only
  thing that crosses into live is a promoted `ModelArtifact`.
- **Retraining is gated on resolved-outcome count, not the calendar**, and every candidate clears the
  same validation gate as a brand-new strategy before promotion.
- **Every decision logs its `model_version`** — that single field powers both attribution
  ("did v7 beat v6?") and one-line rollback.
- **Point-in-time correctness extends into training data** — training rows use only decision-time
  features; end-of-day leakage reintroduces lookahead.
