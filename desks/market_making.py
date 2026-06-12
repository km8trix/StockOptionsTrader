"""
Avellaneda-Stoikov market-making simulator — SIMULATION ONLY.

==========================================================================
THIS MODULE NEVER TRADES. It is not reachable from any live or backtest
desk path: no Desk imports it, no engine fill route touches it, and it
emits no DeskIntent. It exists as the CALIBRATION BENCH for the Phase 8
fair-value engine — a controlled environment for studying how quoting
around a model price with inventory-aware skew controls risk, before any
of that thinking goes near real order flow (Phase 9+, E*TRADE only).
==========================================================================

THE MODEL (Avellaneda & Stoikov, "High-frequency trading in a
limit-order book", Quantitative Finance 8(3), 2008 — the canonical
closed-form quotes for an exponential-utility market maker on an
arithmetic Brownian mid with exponential fill intensities):

    reservation price   r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
    total spread        delta_total(t) = gamma * sigma^2 * (T - t)
                                         + (2 / gamma) * ln(1 + gamma / k)
    quotes              bid = r - delta_total / 2
                        ask = r + delta_total / 2

where s is the mid, q the signed inventory, gamma the risk aversion,
sigma the mid volatility per unit time, T the horizon and k the decay of
the fill-intensity lambda(delta) = A * exp(-k * delta). Implemented
EXACTLY as written (tests pin the formulas to hand math).

SYNTHETIC LOB: the mid follows a seeded ARITHMETIC random walk
(s_{t+dt} = s_t + sigma * sqrt(dt) * N(0,1)); market orders arrive
Poisson per side with intensity lambda(delta) of the quoted distance, so
the per-step fill probability is 1 - exp(-lambda * dt), at most one fill
per side per step (the standard discretization of the A-S simulation).

P&L IDENTITY (held exactly, asserted every step in tests):

    pnl == cash + inventory * mid
        == spread_pnl + inventory_pnl

with spread_pnl accruing (ask - mid) per sell and (mid - bid) per buy at
the moment of the fill, and inventory_pnl accruing q * (mid' - mid) on
every mid move. The decomposition is exact by construction, not an
approximation.

run_naive() quotes the SAME total spread but symmetrically around the
MID (no reservation-price skew): the difference in inventory behavior
between the two quoters isolates the A-S inventory-control term.
"""

from __future__ import annotations

import logging
import math
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class SyntheticLOB:
    """Seeded synthetic order-flow environment (module docstring).

    Mid: arithmetic random walk with per-unit-time vol sigma over
    n_steps of dt = T / n_steps. Market orders hit each side as a
    Poisson process with intensity A * exp(-k * delta) of the quote's
    distance delta from the mid.
    """

    def __init__(self, s0: float = 100.0, sigma: float = 2.0,
                 T: float = 1.0, n_steps: int = 200,
                 A: float = 140.0, k: float = 1.5, seed: int = 42):
        self.s0 = s0
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.A = A
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.mid = s0

    def step_mid(self) -> float:
        """Advance the mid one step; returns the new mid."""
        self.mid += (self.sigma * math.sqrt(self.dt)
                     * float(self.rng.standard_normal()))
        return self.mid

    def fill_side(self, delta: float) -> bool:
        """One side's fill draw this step: P = 1 - exp(-lambda * dt)
        with lambda = A * exp(-k * delta). Negative delta (quoting
        through the mid) only intensifies arrivals, as it should."""
        intensity = self.A * math.exp(-self.k * delta)
        p_fill = 1.0 - math.exp(-intensity * self.dt)
        return bool(self.rng.random() < p_fill)


class ASMarketMaker:
    """Avellaneda-Stoikov quoter (module docstring; formulas exact)."""

    def __init__(self, gamma: float = 0.1, k: float = 1.5, A: float = 140.0,
                 sigma: float = 2.0, T: float = 1.0):
        if gamma <= 0 or k <= 0:
            raise ValueError(f"gamma ({gamma}) and k ({k}) must be > 0")
        self.gamma = gamma
        self.k = k
        self.A = A
        self.sigma = sigma
        self.T = T

    # ------------------------------------------------------------------
    # The canonical formulas
    # ------------------------------------------------------------------
    def reservation_price(self, s: float, q: float, t: float) -> float:
        """r = s - q * gamma * sigma^2 * (T - t)."""
        return s - q * self.gamma * self.sigma ** 2 * (self.T - t)

    def total_spread(self, t: float) -> float:
        """delta_total = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)."""
        return (self.gamma * self.sigma ** 2 * (self.T - t)
                + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k))

    def quotes(self, s: float, q: float, t: float) -> Dict[str, float]:
        """{'bid', 'ask'} = reservation price -/+ half the total spread.

        With q == 0 the reservation price equals the mid, so the quotes
        are exactly symmetric around it."""
        r = self.reservation_price(s, q, t)
        half = self.total_spread(t) / 2.0
        return {'bid': r - half, 'ask': r + half}

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def run(self, seed: int = 42, n_steps: int = 200,
            s0: float = 100.0) -> Dict[str, float]:
        """Simulate one session quoting A-S bid/ask each step.

        Returns {'final_pnl','spread_pnl','inventory_pnl',
        'max_abs_inventory','inventory_std','n_fills','final_inventory'}.
        Internal identity (exact): cash + inventory * final_mid ==
        final_pnl == spread_pnl + inventory_pnl.
        """
        return self._simulate(seed, n_steps, s0, naive=False)

    def run_naive(self, seed: int = 42, n_steps: int = 200,
                  s0: float = 100.0) -> Dict[str, float]:
        """Same session, but a FIXED symmetric quoter: the same total
        spread centered on the MID, ignoring inventory — the control arm
        for the inventory-management claim."""
        return self._simulate(seed, n_steps, s0, naive=True)

    def _simulate(self, seed: int, n_steps: int, s0: float,
                  naive: bool) -> Dict[str, float]:
        lob = SyntheticLOB(s0=s0, sigma=self.sigma, T=self.T,
                           n_steps=n_steps, A=self.A, k=self.k, seed=seed)
        cash = 0.0
        inventory = 0
        spread_pnl = 0.0
        inventory_pnl = 0.0
        n_fills = 0
        inventory_path = []

        for step in range(n_steps):
            t = step * lob.dt
            mid = lob.mid
            quote = (self.quotes(mid, 0.0, t) if naive
                     else self.quotes(mid, inventory, t))
            bid, ask = quote['bid'], quote['ask']

            # Fill draws (ask side first, then bid — fixed order keeps
            # the seeded stream deterministic). At most 1 lot per side.
            if lob.fill_side(ask - mid):
                cash += ask
                inventory -= 1
                spread_pnl += ask - mid
                n_fills += 1
            if lob.fill_side(mid - bid):
                cash -= bid
                inventory += 1
                spread_pnl += mid - bid
                n_fills += 1

            new_mid = lob.step_mid()
            inventory_pnl += inventory * (new_mid - mid)
            inventory_path.append(inventory)

        final_mid = lob.mid
        final_pnl = cash + inventory * final_mid
        result = {
            'final_pnl': float(final_pnl),
            'spread_pnl': float(spread_pnl),
            'inventory_pnl': float(inventory_pnl),
            'max_abs_inventory': float(max(abs(q) for q in inventory_path)
                                       if inventory_path else 0.0),
            'inventory_std': float(np.std(inventory_path)
                                   if inventory_path else 0.0),
            'n_fills': int(n_fills),
            'final_inventory': int(inventory),
        }
        logger.debug("A-S %s run (seed %d): %s",
                     'naive' if naive else 'skewed', seed, result)
        return result


def run_demo(seed: int = 42, n_steps: int = 200) -> Dict[str, Dict]:
    """Side-by-side A-S vs naive symmetric quoting on the same seed.

    Returns {'avellaneda_stoikov': {...}, 'naive_symmetric': {...},
    'params': {...}} (run() result shapes). No prints — callers decide
    how to present it. SIMULATION ONLY (module banner).
    """
    maker = ASMarketMaker()
    return {
        'avellaneda_stoikov': maker.run(seed=seed, n_steps=n_steps),
        'naive_symmetric': maker.run_naive(seed=seed, n_steps=n_steps),
        'params': {'gamma': maker.gamma, 'k': maker.k, 'A': maker.A,
                   'sigma': maker.sigma, 'T': maker.T,
                   'seed': seed, 'n_steps': n_steps},
    }
