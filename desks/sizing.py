"""Position-sizing overlays shared by desks.

Sizing reshapes the return DISTRIBUTION of a strategy; it cannot create edge
where none exists. Overlays here are deterministic, desk-internal state
machines — no I/O, no randomness — so backtest/live parity holds by
construction.
"""

from __future__ import annotations


class ReverseMartingaleSizer:
    """Anti-martingale sizing: press winners, reset on a loss.

    size() = base_size * min(1 + step * win_streak, max_mult)

    The desk records each closed round-trip via record(won); consecutive wins
    scale the NEXT entry up linearly, a single loss resets to base. Capped at
    max_mult so the position-size risk check (which must be configured with
    headroom above base_size * max_mult) is never the silent binding limit.
    """

    def __init__(self, base_size: float, *, step: float = 0.5,
                 max_mult: float = 2.0):
        if base_size <= 0:
            raise ValueError(f"base_size {base_size} must be positive")
        if step < 0:
            raise ValueError(f"step {step} must be >= 0")
        if max_mult < 1.0:
            raise ValueError(f"max_mult {max_mult} must be >= 1.0")
        self.base_size = base_size
        self.step = step
        self.max_mult = max_mult
        self.win_streak = 0

    def size(self) -> float:
        """Entry size for the next trade, given the current win streak."""
        return self.base_size * min(1.0 + self.step * self.win_streak,
                                    self.max_mult)

    def record(self, won: bool) -> None:
        """Record a closed round-trip outcome."""
        self.win_streak = self.win_streak + 1 if won else 0

    def reset(self) -> None:
        self.win_streak = 0
