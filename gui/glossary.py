"""Single source of truth for the UI glossary.

One dict, term -> plain-language definition. The base template renders it as a
glossary offcanvas (available on every page) and the ``explain()`` Jinja macro
looks definitions up here for inline info popovers, so the copy never drifts
between the two. Wording mirrors the desks-vs-strategies copy on the Dashboard
chooser and the Sandbox/Production nav so the model the app teaches stays
consistent.
"""
from __future__ import annotations

# Insertion order is the display order in the offcanvas.
GLOSSARY: dict[str, str] = {
    "Sandbox": (
        "Research & practice workspace — analyze symbols, backtest strategies "
        "and the firm-style desks, and paper trade. Nothing here ever places a "
        "live order."
    ),
    "Production": (
        "Live-capable workspace — connect E*TRADE for live execution and watch "
        "the production market charts. Desks are backtest-only and live in the "
        "Sandbox, not here."
    ),
    "Strategy": (
        "A single rule set for entering and exiting trades. Strategies live in "
        "the Sandbox and are what you backtest and paper trade."
    ),
    "Desk": (
        "A firm-style trading unit that runs strategies under one risk book. "
        "Desks are backtesting tools in the Sandbox — they are not wired to "
        "live execution."
    ),
    "Fund": (
        "Several desks run together under one shared risk book, with capital "
        "allocated across them and exposure netted account-wide."
    ),
    "Backtest": (
        "Replays a strategy or desk over historical data to estimate how it "
        "would have performed. Simulation only — it never places a live order."
    ),
    "Paper Trading": (
        "Live-like trading with simulated money and fills — a safe dress "
        "rehearsal before risking real capital."
    ),
    "Live": (
        "Real-money execution through your connected E*TRADE account, gated "
        "behind explicit acknowledgements and market-hours checks."
    ),
    "Kill switch": (
        "An operator stop that blocks all new live order placement. Engaging "
        "it never auto-closes existing positions."
    ),
    "Greeks": (
        "Option risk sensitivities (delta, gamma, theta, vega) — how an "
        "option's value moves with price, time, and volatility."
    ),
}


if __name__ == "__main__":  # ponytail: one runnable check — copy stays sane
    assert GLOSSARY, "glossary must not be empty"
    assert all(isinstance(k, str) and k.strip() for k in GLOSSARY)
    assert all(isinstance(v, str) and len(v) > 20 for v in GLOSSARY.values())
    assert "Sandbox" in GLOSSARY and "Production" in GLOSSARY
    print(f"glossary OK — {len(GLOSSARY)} terms")
