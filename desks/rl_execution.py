"""
Gated reinforcement-learning EXECUTION THROTTLE (Phase F unit 2).

HONEST FRAMING — read this first.
================================
This module is a ONE-STEP CONTEXTUAL-BANDIT / policy-gradient EXECUTION
THROTTLE, NOT multi-step deep RL. There is no environment rollout, no value
function over a trajectory, no temporal-difference bootstrapping. It maps a
single leakage-free context vector to a single scalar throttle and stops. Full
multi-step deep RL was DELIBERATELY left out of scope: on this little data it is
the most overfitting-prone construction in the whole program, and a one-step
bandit with a hard no-op prior is the conservative thing that can possibly work.

The agent is GATED OFF, EXPERIMENTAL, OPT-IN, and OFF BY DEFAULT. It is NEVER a
full trader, NEVER live, and can NEVER increase risk. Clearing the research gate
(:func:`validate_throttle_oos`) does NOT auto-enable it — it stays flag-OFF and
is never wired to any live/order-routing path. There is no E*TRADE / live import
anywhere in this module; the agent only ever modulates desk-intent SIZES inside
the backtest sizing seam, between netting and the account-wide risk gate.

THE SAFETY CONTRACT (hard invariants — enforced here and tested by QA).
======================================================================
1. SUBTRACTIVE-ONLY. The throttle multiplier ``s`` lives in
   ``[scale_min, scale_max]`` with ``scale_max`` HARD-CLAMPED to ``<= 1.0`` in
   the constructor. It can only ever SHRINK a desk-intended size or leave it
   unchanged. It can NEVER scale above 1.0, so the throttled fund gross is
   ALWAYS <= the baseline fund gross, position by position. This is the headline
   property and it is asserted as a post-condition on every ``apply`` call.
2. DIRECTION IS UNTOUCHED. Only ``DeskIntent.size_fraction`` is multiplied. The
   throttle never reads or writes ``DeskIntent.action``; it never flips,
   creates, or re-targets an intent.
3. CLOSING INTENTS ARE NEVER THROTTLED. ``SELL``/``COVER`` pass through
   UNCHANGED — throttling a close would leave residual risk on the book.
4. OPTION INTENTS PASS THROUGH UNCHANGED. Any intent whose asset is an option
   (``AssetType.CALL``/``PUT``) or that carries an absolute ``quantity``
   (Jane Street defined-risk structures) is NOT throttled — scaling one leg
   would unbalance a defined-risk wing. Only plain-stock OPENING intents
   (``BUY``/``SHORT``) are ever throttled.
5. INVARIANT PRESERVATION. ``DeskIntent.__post_init__`` requires
   ``0 < size_fraction <= 1``. Because the input is ``<= 1`` and ``s <= 1`` the
   product is already ``<= 1``; we clamp to ``(eps, 1.0]`` for safety. If a
   scaled size rounds to ``<= eps`` we DROP the intent (and record it) rather
   than emit an invalid ``DeskIntent``. New intents are built with
   :func:`dataclasses.replace`.
6. OFF BY DEFAULT / BYTE-IDENTICAL. The throttle is only ever invoked when a
   caller explicitly passes ``sizing_modulator=`` to the orchestrator. With no
   modulator the orchestrator step is byte-identical to today.
7. NEVER LIVE. No order-routing / E*TRADE / live path is imported or referenced.
8. DETERMINISTIC. Seeded, CPU-only, wrapped in :func:`_deterministic_torch`.
   Same seed + same inputs => byte-identical throttles within a run. No
   cross-platform golden files for the NN — within-run reproducibility only.
9. LEAKAGE-FREE. Every feature is computed only from data with index
   ``<= date``. The policy is trained on a TRAIN window only and then FROZEN; it
   is never refit during OOS evaluation.

torch is an OPTIONAL dependency (the large CPU/MPS wheel lives in the ``ai``
extra). The import is guarded EXACTLY like :mod:`desks.models.neural`: importing
this module never hard-fails when torch is missing, but CONSTRUCTING a
:class:`ThrottlePolicy` raises a clear ``ImportError`` so a misconfigured install
fails loudly at the one place a policy is built.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, cast

import numpy as np
import pandas as pd

from core.models import AssetType
from desks.base import DeskIntent
from portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)

# torch is an OPTIONAL dependency (large CPU/MPS wheel). The import is guarded
# so importing this module — and therefore the registry — never hard-fails when
# the wheel is missing. The degrade policy mirrors desks.models.neural EXACTLY:
# the ThrottlePolicy CONSTRUCTOR raises a clear ImportError (a misconfigured
# install fails loudly and early where a policy is built), while everything that
# merely IMPORTS this module keeps working.
try:
    import torch
    from torch import nn

    _TORCH_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 - any import failure disables torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'
    logger.warning(
        'torch unavailable (%s) — ThrottlePolicy will raise on construction; '
        'install it (`pip install .[ai]`, torch==2.12.1) to enable it. The '
        'rest of desks.rl_execution (identity throttle, validation) is '
        'unaffected.', _TORCH_IMPORT_ERROR)

#: Ordered names of the leakage-free context features (one source of truth so
#: the scaler, the policy input dim, and any introspection agree).
THROTTLE_FEATURE_NAMES: Sequence[str] = (
    'size_fraction',      # the intent's account-absolute size_fraction
    'fund_drawdown',      # trailing fund drawdown vs trailing peak (<= date)
    'fund_vol_21d',       # trailing ~21d fund realized vol (<= date)
    'symbol_vol_21d',     # the intent symbol's trailing realized vol (<= date)
    'is_opening',         # 1.0 for BUY/SHORT opens (always 1 here), else 0.0
)

#: Number of context features (the policy input dimension).
N_THROTTLE_FEATURES: int = len(THROTTLE_FEATURE_NAMES)

#: Opening / closing actions (mirrors desks.base; closes are never throttled).
_OPENING_ACTIONS = ('BUY', 'SHORT')
_CLOSING_ACTIONS = ('SELL', 'COVER')
_OPTION_TYPES = (AssetType.CALL, AssetType.PUT)

#: Default smallest emitted size_fraction; a scaled size <= eps is DROPPED
#: rather than emitted as an invalid (<= 0 after rounding) DeskIntent.
_DEFAULT_EPS = 1e-6

#: Trailing window (trading days) for the realized-vol features.
_VOL_WINDOW = 21


@contextlib.contextmanager
def _deterministic_torch(seed: int):
    """Pin torch to reproducible CPU settings for one fit/predict, then RESTORE
    the prior global torch state on exit.

    Replicated from :mod:`desks.models.neural` so the throttle never permanently
    single-threads the process or hard-pins the deterministic flag for other
    torch work that may follow. ``warn_only=True`` lets an unrelated torch op
    that lacks a deterministic CPU kernel degrade to a warning instead of
    raising. Yields the CPU device (the only reproducible one).
    """
    if torch is None:  # pragma: no cover - callers gate on a constructed policy
        raise ImportError(
            'torch is required for deterministic throttle ops '
            f'({_TORCH_IMPORT_ERROR}); install torch==2.12.1 (`pip install '
            '.[ai]`). This helper is private to ThrottlePolicy, which already '
            'raises on construction when torch is missing.')
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        yield torch.device('cpu')
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


# ----------------------------------------------------------------------
# Feature scaling (train-fit mean/std — leakage-critical)
# ----------------------------------------------------------------------
class FeatureScaler:
    """Train-window mean/std standardizer for throttle features.

    Fitted on the TRAIN matrix ONLY in :meth:`fit`; the stored mean/std are
    re-applied verbatim at decision time so no decision-time information leaks
    into the fitted parameters. A zero-std (constant) column is given unit std
    so transform never divides by zero (mirrors the neural ``_StandardScaler``).
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x_matrix: np.ndarray) -> "FeatureScaler":
        x_matrix = np.asarray(x_matrix, dtype=float)
        self.mean_ = x_matrix.mean(axis=0)
        std = x_matrix.std(axis=0)
        std[std < 1e-8] = 1.0
        self.std_ = std
        return self

    def transform(self, x_vector: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('FeatureScaler.transform before fit')
        return (np.asarray(x_vector, dtype=float) - self.mean_) / self.std_


# ----------------------------------------------------------------------
# Leakage-free feature extraction
# ----------------------------------------------------------------------
def _trailing_equity(portfolio: PortfolioManager,
                     date) -> np.ndarray:
    """Fund equity snapshots with timestamp <= ``date`` (leakage-free).

    Reads ``portfolio.portfolio_history`` (list of dicts each carrying a
    ``timestamp`` and ``portfolio_value``) and keeps only snapshots on or before
    the decision date. NO snapshot dated after ``date`` is ever read.
    """
    cutoff = pd.Timestamp(date)
    history = getattr(portfolio, 'portfolio_history', None) or []
    values: List[float] = []
    for snapshot in history:
        ts = snapshot.get('timestamp')
        if ts is None or pd.Timestamp(ts) <= cutoff:
            # A snapshot with no timestamp is treated as already-past (the
            # engine stamps every real snapshot; this only guards ad-hoc use).
            values.append(float(snapshot.get('portfolio_value', 0.0)))
    return np.asarray(values, dtype=float)


def _fund_drawdown(equity: np.ndarray) -> float:
    """Trailing drawdown of the latest equity vs its trailing peak, in [0, 1].

    0.0 when at/above the running peak or when there is too little history.
    """
    if equity.size == 0:
        return 0.0
    peak = float(np.max(equity))
    if peak <= 0.0:
        return 0.0
    last = float(equity[-1])
    return max(0.0, min(1.0, (peak - last) / peak))


def _realized_vol(values: np.ndarray, window: int = _VOL_WINDOW) -> float:
    """Trailing realized vol from a price/equity series' simple returns.

    Uses only the trailing ``window`` returns of the supplied (already
    leakage-trimmed) series. 0.0 when there are fewer than two points.
    """
    if values is None or values.size < 2:
        return 0.0
    rets = values[1:] / values[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        return 0.0
    tail = rets[-window:]
    if tail.size < 2:
        return float(abs(tail[0])) if tail.size == 1 else 0.0
    return float(np.std(tail, ddof=1))


def _symbol_trailing_vol(all_data: Dict[str, pd.DataFrame], symbol: str,
                         date, window: int = _VOL_WINDOW) -> float:
    """The intent symbol's trailing realized vol from closes with index <= date.

    Slices ``all_data[symbol]`` to rows on or before the decision date (the
    engine already guarantees this, but we re-slice defensively so a leak is
    impossible by construction) and reuses :func:`_realized_vol`.
    """
    frame = all_data.get(symbol) if all_data else None
    if frame is None or frame.empty or 'close' not in frame.columns:
        return 0.0
    cutoff = pd.Timestamp(date)
    try:
        sliced = frame.loc[frame.index <= cutoff]
    except TypeError:
        # Non-datetime index: fall back to the whole (engine-trimmed) frame.
        sliced = frame
    closes = pd.to_numeric(sliced['close'], errors='coerce').dropna()
    if closes.empty:
        return 0.0
    return _realized_vol(closes.to_numpy(dtype=float), window=window)


def extract_throttle_features(intent: DeskIntent,
                              portfolio: PortfolioManager,
                              all_data: Dict[str, pd.DataFrame],
                              date,
                              scalers: Optional[FeatureScaler] = None
                              ) -> np.ndarray:
    """Fixed-length LEAKAGE-FREE context vector for a single intent.

    Features (order = :data:`THROTTLE_FEATURE_NAMES`):
      * ``size_fraction``  — the intent's account-absolute size_fraction;
      * ``fund_drawdown``  — trailing fund drawdown vs trailing peak, computed
        from equity snapshots with timestamp ``<= date``;
      * ``fund_vol_21d``   — trailing ~21d fund realized vol (``<= date``);
      * ``symbol_vol_21d`` — the intent symbol's trailing realized vol from
        closes with index ``<= date``;
      * ``is_opening``     — 1.0 for opening (BUY/SHORT) intents, else 0.0.

    NO row with index > ``date`` is ever read (leakage-free invariant #9). If a
    train-fit :class:`FeatureScaler` is supplied, the raw vector is standardized
    with it before return.
    """
    equity = _trailing_equity(portfolio, date)
    fund_dd = _fund_drawdown(equity)
    fund_vol = _realized_vol(equity)
    symbol_vol = _symbol_trailing_vol(all_data, intent.asset.symbol, date)
    is_opening = 1.0 if intent.action in _OPENING_ACTIONS else 0.0

    raw = np.asarray([
        float(intent.size_fraction),
        float(fund_dd),
        float(fund_vol),
        float(symbol_vol),
        float(is_opening),
    ], dtype=float)
    if scalers is not None:
        return scalers.transform(raw)
    return raw


# ----------------------------------------------------------------------
# The tiny throttle policy (torch MLP, import-guarded)
# ----------------------------------------------------------------------
class ThrottlePolicy:
    """A tiny one-hidden-layer MLP mapping context -> throttle in
    ``[scale_min, scale_max]`` with ``scale_max <= 1.0``.

    The head is a single logit passed through a sigmoid and then affinely mapped
    onto ``[scale_min, scale_max]``. The output-bias is initialised large and
    POSITIVE so that an UNTRAINED policy (or any policy with no learned signal)
    outputs ``~scale_max`` — the NO-OP PRIOR. With no learned signal the throttle
    leaves sizes essentially unchanged; learning can only ever pull the
    multiplier DOWN toward ``scale_min`` (subtractive-only, contract #1).

    DEGRADE POLICY (mirrors :class:`desks.models.neural.MLPModel`): the
    CONSTRUCTOR raises a clear ``ImportError`` when torch is missing. All torch
    ops run inside :func:`_deterministic_torch` (contract #8).

    EPHEMERAL BY DESIGN: a policy is trained fresh per backtest window and is
    never serialized — there is deliberately no cross-platform golden / state
    export (contract #8: assert within-run reproducibility, never ship a frozen
    NN artifact whose bytes drift across torch builds or platforms).
    """

    def __init__(self, scale_min: float = 0.5, scale_max: float = 1.0,
                 hidden_size: int = 8, n_features: int = N_THROTTLE_FEATURES,
                 random_state: int = 42, noop_bias: float = 6.0):
        if torch is None:
            raise ImportError(
                'ThrottlePolicy requires the torch package '
                f'({_TORCH_IMPORT_ERROR}); install torch==2.12.1 '
                '(`pip install .[ai]`) to use it.')
        # HARD-CLAMP scale_max <= 1.0 (subtractive-only, contract #1) and
        # validate the interval (contract #1: raise on degenerate bounds).
        scale_max = min(float(scale_max), 1.0)
        if scale_min <= 0.0 or scale_min > scale_max:
            raise ValueError(
                f"scale_min={scale_min} must be in (0, scale_max={scale_max}]")
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.hidden_size = int(hidden_size)
        self.n_features = int(n_features)
        self.random_state = int(random_state)
        self.noop_bias = float(noop_bias)
        with _deterministic_torch(self.random_state):
            torch.manual_seed(self.random_state)
            self._net = self._build_net()
            # No-op prior INSIDE the deterministic context (contract #8) so the
            # weight init AND its overwrite are pinned together — a big positive
            # output bias drives sigmoid -> ~1 -> throttle ~= scale_max for an
            # untrained net. (fill_/mul_ are deterministic, but keeping every
            # weight-touching op under the same pinned state makes the
            # same-seed => byte-identical guarantee airtight, not incidental.)
            self._init_noop_prior()

    def _build_net(self) -> "nn.Sequential":
        """Context -> hidden(ReLU) -> single logit."""
        return nn.Sequential(
            nn.Linear(self.n_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def _init_noop_prior(self) -> None:
        """Bias the output logit large+positive so an untrained policy outputs
        ~scale_max (the no-op prior, contract: no learnable signal => no-op)."""
        with torch.no_grad():
            head = cast("nn.Linear", self._net[-1])
            head.bias.fill_(self.noop_bias)
            # Tiny output weights so the prior dominates before any training.
            head.weight.mul_(0.01)

    def _logit_to_throttle(self, logit: "torch.Tensor") -> float:
        """Sigmoid(logit) in (0,1) affinely mapped onto [scale_min, scale_max]."""
        unit = float(torch.sigmoid(logit).item())
        return self.scale_min + unit * (self.scale_max - self.scale_min)

    def predict_throttle(self, features: np.ndarray) -> float:
        """Throttle ``s`` in ``[scale_min, scale_max]`` (<= 1.0) for one context.

        Deterministic and CPU-only (contract #8). The returned value is clamped
        to the interval as a belt-and-suspenders guard on top of the sigmoid map
        (the sigmoid already bounds it, but a degenerate input must never
        produce an out-of-range multiplier).
        """
        with _deterministic_torch(self.random_state), torch.no_grad():
            self._net.eval()
            x = torch.tensor(np.asarray(features, dtype=float).reshape(1, -1),
                             dtype=torch.float32)
            logit = self._net(x).reshape(())
            throttle = self._logit_to_throttle(logit)
        # Subtractive-only clamp: s in [scale_min, scale_max <= 1.0].
        return float(min(self.scale_max, max(self.scale_min, throttle)))


# ----------------------------------------------------------------------
# Training: one-step REINFORCE with a baseline (deterministic, CPU)
# ----------------------------------------------------------------------
@dataclass
class ThrottleDecision:
    """One train-window decision: its leakage-free context and the realized
    forward reward (forward P&L proxy) measured on TRAIN-window prices only."""
    features: np.ndarray
    forward_reward: float


class ThrottlePolicyTrainer:
    """Deterministic, seeded, CPU one-step policy-gradient trainer.

    :meth:`fit` consumes ``train_decisions`` built ONLY from the train window as
    ``(features, realized_forward_reward)`` pairs and returns a FROZEN
    :class:`ThrottlePolicy`. The objective is the one-step bandit return

        ``E[ s * forward_pnl - lambda * risk_penalty(s, context) ]``

    maximised by REINFORCE with a moving-average baseline (variance reduction).
    Heavy L2, a short epoch budget with early stopping, and the policy's no-op
    prior together mean that when there is NO learnable signal the trainer leaves
    the policy near ``scale_max`` (an honest no-op rather than a fitted illusion).

    LEAKAGE RULE: every ``forward_reward`` MUST be computed from TRAIN-window
    prices only (the caller's responsibility when building the decisions); the
    returned policy is FROZEN and is never refit during OOS evaluation
    (contract #9).
    """

    def __init__(self, scale_min: float = 0.5, scale_max: float = 1.0,
                 hidden_size: int = 8, epochs: int = 60,
                 learning_rate: float = 0.01, l2: float = 1e-2,
                 risk_lambda: float = 0.1, random_state: int = 42,
                 early_stop_patience: int = 8):
        self.scale_min = float(scale_min)
        self.scale_max = min(float(scale_max), 1.0)  # hard-clamp (contract #1)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.risk_lambda = float(risk_lambda)
        self.random_state = int(random_state)
        self.early_stop_patience = int(early_stop_patience)

    def _scaled_throttle(self, unit: "torch.Tensor") -> "torch.Tensor":
        """sigmoid-unit -> throttle in [scale_min, scale_max] (differentiable)."""
        return self.scale_min + unit * (self.scale_max - self.scale_min)

    def fit(self, train_decisions: Sequence[ThrottleDecision]) -> ThrottlePolicy:
        """Train on TRAIN-window decisions; return a FROZEN ThrottlePolicy.

        Degrades gracefully: with no decisions (or any training failure) the
        UNTRAINED, no-op-prior policy is returned unchanged — the safe default
        is "do not throttle". torch is required to construct the policy at all.
        """
        policy = ThrottlePolicy(
            scale_min=self.scale_min, scale_max=self.scale_max,
            hidden_size=self.hidden_size, random_state=self.random_state)
        if not train_decisions:
            logger.warning(
                "ThrottlePolicyTrainer.fit: no train decisions; returning the "
                "no-op-prior policy (no throttling learned)")
            return policy

        x = np.asarray([d.features for d in train_decisions], dtype=float)
        rewards = np.asarray([d.forward_reward for d in train_decisions],
                             dtype=float)
        if x.ndim != 2 or x.shape[0] < 2:
            logger.warning(
                "ThrottlePolicyTrainer.fit: <2 usable decisions; returning the "
                "no-op-prior policy")
            return policy

        try:
            with _deterministic_torch(self.random_state):
                net = policy._net
                net.train()
                optimizer = torch.optim.Adam(
                    net.parameters(), lr=self.learning_rate,
                    weight_decay=self.l2)  # heavy L2 (overfitting guard)
                x_t = torch.tensor(x, dtype=torch.float32)
                r_t = torch.tensor(rewards, dtype=torch.float32)
                baseline = 0.0  # moving-average REINFORCE baseline
                best_obj = -float('inf')
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
                stale = 0
                for _epoch in range(self.epochs):
                    optimizer.zero_grad()
                    logits = net(x_t).reshape(-1)
                    unit = torch.sigmoid(logits)
                    # Mean throttle as the deterministic "action" (one-step,
                    # no sampling distribution to log-prob over — this is the
                    # smooth policy-gradient surrogate of E[s*pnl - penalty]).
                    s = self._scaled_throttle(unit)
                    # Risk penalty grows with size used (1 - s small => more
                    # size => more risk); lambda trades reward against it.
                    risk_penalty = self.risk_lambda * (s * s)
                    obj_per = s * r_t - risk_penalty
                    advantage = obj_per - baseline
                    # Maximise advantage-weighted objective (minimise -obj).
                    loss = -(advantage * obj_per).mean()
                    loss.backward()
                    optimizer.step()
                    epoch_obj = float(obj_per.mean().item())
                    baseline = 0.9 * baseline + 0.1 * epoch_obj
                    if epoch_obj > best_obj + 1e-9:
                        best_obj = epoch_obj
                        best_state = {k: v.detach().clone()
                                      for k, v in net.state_dict().items()}
                        stale = 0
                    else:
                        stale += 1
                        if stale >= self.early_stop_patience:
                            break  # early stop (overfitting guard)
                net.load_state_dict(best_state)
                net.eval()
        except Exception as exc:  # noqa: BLE001 - never crash a run
            logger.warning(
                "ThrottlePolicyTrainer.fit failed (%s); returning the "
                "no-op-prior policy", exc)
            return ThrottlePolicy(
                scale_min=self.scale_min, scale_max=self.scale_max,
                hidden_size=self.hidden_size, random_state=self.random_state)

        logger.debug("ThrottlePolicyTrainer fitted on %d decisions", x.shape[0])
        return policy


# ----------------------------------------------------------------------
# The sizing-modulator object passed to FundOrchestrator(sizing_modulator=...)
# ----------------------------------------------------------------------
@dataclass
class ThrottleLogEntry:
    """One append-only audit record of an apply() throttling decision."""
    date: object
    symbol: str
    action: str
    original_size: float
    throttle: float
    new_size: float
    dropped: bool


class RLExecutionThrottle:
    """The gated, subtractive sizing modulator handed to the orchestrator.

    Construct with a FROZEN :class:`ThrottlePolicy` (or ``None`` => identity),
    ``scale_min`` (default 0.5), ``scale_max`` (default 1.0, HARD-CLAMPED to
    ``<= 1.0``), an ``enabled`` flag (default ``True``; ``enabled=False`` =>
    pure pass-through identity), and an ``eps`` floor.

    :meth:`apply` is the only method the orchestrator calls. For each intent it
    passes through UNCHANGED when ANY of these holds (the safety contract):
      * the throttle is disabled, or the policy is ``None`` (identity);
      * the action is a CLOSE (``SELL``/``COVER``) — contract #3;
      * the asset is an option, or the intent carries an absolute ``quantity``
        (defined-risk leg) — contract #4.
    Otherwise (a plain-stock OPENING intent) it computes leakage-free features,
    asks the policy for ``s`` clamped to ``[scale_min, scale_max]``, multiplies
    the size, clamps to ``(eps, 1.0]``, DROPS the intent if ``<= eps``, else
    rebuilds it with :func:`dataclasses.replace`. A POST-CONDITION asserts every
    emitted intent shrank (size ``<=`` original, in ``(0, 1]``) and that total
    emitted gross ``<=`` input gross; a violation FAILS LOUD (contract #1).
    """

    def __init__(self, policy: Optional[ThrottlePolicy] = None,
                 scale_min: float = 0.5, scale_max: float = 1.0,
                 enabled: bool = True, eps: float = _DEFAULT_EPS,
                 scaler: Optional[FeatureScaler] = None):
        # HARD-CLAMP scale_max <= 1.0 (subtractive-only, contract #1) and
        # validate the interval (raise on degenerate bounds, contract #1).
        scale_max = min(float(scale_max), 1.0)
        if scale_min <= 0.0 or scale_min > scale_max:
            raise ValueError(
                f"scale_min={scale_min} must be in (0, scale_max={scale_max}]")
        if eps <= 0.0:
            raise ValueError(f"eps={eps} must be > 0")
        self.policy = policy
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.enabled = bool(enabled)
        self.eps = float(eps)
        self.scaler = scaler
        #: Append-only audit trail of every throttling decision.
        self.decision_log: List[ThrottleLogEntry] = []

    # ------------------------------------------------------------------
    def _passthrough(self, intent: DeskIntent) -> bool:
        """True when this intent must pass through UNCHANGED (contracts 3,4)."""
        if intent.action in _CLOSING_ACTIONS:
            return True  # contract #3: never throttle a close
        if intent.asset.asset_type in _OPTION_TYPES:
            return True  # contract #4: option leg passes through
        if intent.quantity is not None:
            return True  # contract #4: absolute defined-risk quantity
        return False

    def apply(self, intents: List[DeskIntent], portfolio: PortfolioManager,
              all_data: Dict[str, pd.DataFrame], date) -> List[DeskIntent]:
        """Return a strictly-subtractive throttling of ``intents``.

        When disabled or policy-less this is the identity (contract #6: the
        off path is byte-identical — the same list object is returned). The
        post-condition guarantees the emitted set is never over-leveraged
        relative to the input (contract #1).
        """
        # Identity fast-paths: disabled, or no policy => pure pass-through. The
        # SAME list is returned so the off path is byte-identical (contract #6).
        if not self.enabled or self.policy is None:
            return intents

        emitted: List[DeskIntent] = []
        for intent in intents:
            # Contracts #3 & #4: closes, options, and absolute-quantity legs
            # pass through UNTOUCHED (direction and quantity preserved).
            if self._passthrough(intent):
                emitted.append(intent)
                continue

            # Plain-stock OPENING intent: throttle its size only (contract #2:
            # action is never read/written here beyond the opening guard).
            features = extract_throttle_features(
                intent, portfolio, all_data, date, scalers=self.scaler)
            s = self.policy.predict_throttle(features)
            # Belt-and-suspenders clamp to the subtractive interval (contract #1:
            # s in [scale_min, scale_max <= 1.0], so s can only shrink size).
            s = min(self.scale_max, max(self.scale_min, float(s)))
            new_size = intent.size_fraction * s
            # Clamp to (eps, 1.0]; product is already <= 1 (input<=1, s<=1).
            new_size = min(1.0, new_size)
            # Per-INTENT subtractive guard (contract #1), enforced at the source
            # so the guarantee holds per position regardless of upstream netting
            # — does not rely on the by-asset aggregation in _assert_subtractive.
            if new_size > intent.size_fraction + 1e-9:
                raise AssertionError(
                    f"RLExecutionThrottle increased size for "
                    f"{intent.asset.symbol}: {new_size:.6f} > original "
                    f"{intent.size_fraction:.6f} (subtractive-only violated)")
            dropped = new_size <= self.eps
            self.decision_log.append(ThrottleLogEntry(
                date=date, symbol=intent.asset.symbol, action=intent.action,
                original_size=float(intent.size_fraction), throttle=float(s),
                new_size=float(new_size), dropped=bool(dropped)))
            if dropped:
                # Contract #5: a size that rounds to <= eps would be an invalid
                # DeskIntent — DROP it (recorded above) rather than emit it.
                continue
            # Contract #5: rebuild via dataclasses.replace (direction, asset,
            # reason, quantity all preserved; only size_fraction changes).
            emitted.append(dataclasses.replace(intent, size_fraction=new_size))

        self._assert_subtractive(intents, emitted)
        return emitted

    # ------------------------------------------------------------------
    def _assert_subtractive(self, original: List[DeskIntent],
                            emitted: List[DeskIntent]) -> None:
        """POST-CONDITION (contract #1): fail loud on any over-leverage.

        Verifies (a) every emitted intent has ``0 < size_fraction <= 1`` and
        ``<=`` the matching original's size, and (b) total emitted gross ``<=``
        total input gross. A violation raises rather than letting an
        over-leveraged set escape into the risk gate.
        """
        # Per-name original size: for the throttled (plain-stock opening) names
        # there is exactly one intent per asset in the netted batch, so a
        # by-asset map is an exact original lookup.
        original_size: Dict[object, float] = {}
        for intent in original:
            original_size[intent.asset] = (
                original_size.get(intent.asset, 0.0) + intent.size_fraction)
        emitted_size: Dict[object, float] = {}
        for intent in emitted:
            if not (0.0 < intent.size_fraction <= 1.0):
                raise AssertionError(
                    f"RLExecutionThrottle emitted invalid size_fraction "
                    f"{intent.size_fraction} for {intent.asset.symbol}")
            emitted_size[intent.asset] = (
                emitted_size.get(intent.asset, 0.0) + intent.size_fraction)
            if emitted_size[intent.asset] > original_size.get(
                    intent.asset, 0.0) + 1e-9:
                raise AssertionError(
                    f"RLExecutionThrottle increased size for "
                    f"{intent.asset.symbol}: emitted "
                    f"{emitted_size[intent.asset]:.6f} > original "
                    f"{original_size.get(intent.asset, 0.0):.6f} "
                    f"(subtractive-only violated)")
        total_in = sum(i.size_fraction for i in original)
        total_out = sum(i.size_fraction for i in emitted)
        if total_out > total_in + 1e-9:
            raise AssertionError(
                f"RLExecutionThrottle increased total gross: emitted "
                f"{total_out:.6f} > input {total_in:.6f} "
                f"(subtractive-only violated)")


# ----------------------------------------------------------------------
# Research validation (OOS gate — does NOT enable the agent)
# ----------------------------------------------------------------------
def validate_throttle_oos(baseline_returns: Sequence[float],
                          throttled_returns: Sequence[float],
                          n_trials: int = 1,
                          n_folds: Optional[int] = None,
                          alpha: float = 0.05,
                          dsr_threshold: float = 0.95) -> Dict:
    """Research verdict on the INCREMENTAL (throttled - baseline) return series.

    IMPORTANT — passing this gate does NOT enable the agent. The throttle stays
    flag-OFF and is never wired to any live path; this function produces RESEARCH
    VALIDATION OUTPUT ONLY. A "passed" verdict is necessary evidence, never an
    activation switch.

    The two series are aligned position-by-position and reduced to the per-period
    INCREMENTAL series (``throttled - baseline``). On that incremental series:
      * :func:`deflated_sharpe_ratio` against ``n_trials`` (multiple-testing
        deflation);
      * :func:`probabilistic_sharpe_ratio` against 0.0;
      * if ``n_folds`` is given (>= 2), the INCREMENTAL series is split into
        ``n_folds`` contiguous folds HERE — the caller never supplies the fold
        data, so a baseline/throttled mix can never be passed by mistake — and
        each fold's incremental returns are tested via :func:`fold_oos_pvalue`
        (H0: mean incremental return <= 0) under a :func:`benjamini_hochberg`
        multiple-testing correction.

    ``passed`` is ``True`` iff ``deflated_sharpe_ratio >= dsr_threshold`` AND (if
    ``n_folds`` was given) at least one BH-significant OOS fold; with no folds the
    BH clause is vacuously satisfied so the DSR threshold governs alone.
    """
    # Imported lazily so this module imports even if scipy/analysis is heavy.
    from analysis.research_stats import (
        benjamini_hochberg, deflated_sharpe_ratio, fold_oos_pvalue,
        probabilistic_sharpe_ratio)

    base = np.asarray(list(baseline_returns), dtype=float)
    thr = np.asarray(list(throttled_returns), dtype=float)
    n = int(min(base.size, thr.size))
    incremental = thr[:n] - base[:n]

    dsr = deflated_sharpe_ratio(incremental.tolist(), n_trials)
    psr = probabilistic_sharpe_ratio(incremental.tolist(), 0.0)

    # Folds are sliced from the INCREMENTAL series internally (never caller-
    # supplied): each fold tests the throttle's own edge (throttled - baseline),
    # so there is no way to accidentally analyze the baseline series. Each fold
    # needs >= 2 points for a t-stat; below that the request is recorded as
    # made-but-unusable rather than silently honored.
    fold_pvalues: List[Optional[float]] = []
    bh: Dict = {}
    folds_used = 0
    if n_folds is not None and int(n_folds) >= 2 and n >= 2 * int(n_folds):
        folds = np.array_split(incremental, int(n_folds))
        for fold in folds:
            fold_pvalues.append(fold_oos_pvalue(fold.tolist()))
        bh = benjamini_hochberg(fold_pvalues, alpha=alpha)
        folds_used = len(folds)

    bh_clause = True if folds_used == 0 else (bh.get('n_significant_bh', 0) > 0)
    passed = bool(dsr is not None and dsr >= dsr_threshold and bh_clause)

    return {
        'deflated_sharpe_ratio': dsr,
        'probabilistic_sharpe_ratio': psr,
        'fold_pvalues': fold_pvalues,
        'benjamini_hochberg': bh,
        'n_folds_used': folds_used,
        'n_incremental': n,
        'dsr_threshold': dsr_threshold,
        'passed': passed,
        # Explicit, machine-readable restatement of the docstring contract.
        'enables_agent': False,
    }


__all__ = [
    'THROTTLE_FEATURE_NAMES',
    'N_THROTTLE_FEATURES',
    'FeatureScaler',
    'extract_throttle_features',
    'ThrottlePolicy',
    'ThrottleDecision',
    'ThrottlePolicyTrainer',
    'ThrottleLogEntry',
    'RLExecutionThrottle',
    'validate_throttle_oos',
]
