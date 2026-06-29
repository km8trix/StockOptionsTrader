# Plan: Evidence-First Desk Overhaul

**Source:** free-form request (2026-06-28) — "none of the desks are profitable, overhaul them; review the architecture."
**Decisions (user):** Goal = **Both** (profitable where evidence supports, rigorous everywhere else) · Approach = **Evidence-first rebuild** · Options data = **Drop options/VRP for now** · Promotion semantics = **gate-status badge in Sandbox** (do NOT overload the GUI's "Production = live E*TRADE execution" meaning; desks never trade live)
**Complexity:** Small–Medium (the desk, the gates, and the layering already exist)

## Summary
The architecture is sound and the alpha work is largely done. The overhaul makes the **gate the law**: a desk reaches the Production track only after passing both the IC gate (`scripts/signal_ic.py`) and the OOS gate (`scripts/desk_backtest.py`). Build the promotion rail, run the existing momentum desk through the gate, promote it only if it clears, and quarantine the six edgeless desks to an explicit Sandbox/research track.

## The reframe (why this is not a rebuild)
- Structure is correct; it is *over*-built, not broken. Re-layering creates zero alpha.
- The IC study already found the edge (`mom_12_1`) and the anti-signals (low-vol, 1-mo mom, reversal_5).
- The momentum desk on the good signal alone is **already coded** (`scripts/proto_momentum_desk.py`) and **already positive** (+44.9% total return) but **Sharpe ~0.20 — below the gate's 0.5**.
- The real gap is rigor-plumbing: "production" is currently a cosmetic nav badge. That is the fix.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Desk wiring | `scripts/proto_momentum_desk.py:93` | `MomentumDesk(CrossSectionalLongShortDesk)` — reuse base, single-signal committee |
| Registry entry | `desks/registry.py:142` (`aqr`) | Spec dict; `create_desk` merges `config`/`turnover` kwargs |
| OOS gate | `scripts/desk_backtest.py` | `--desk X` → 4 metrics + Deflated Sharpe + per-regime; saves JSON |
| IC gate | `scripts/signal_ic.py` | Rank-IC vs forward returns, leakage-free, `--selftest` |
| Registry test | `tests/test_desk_registry.py` | Asserts every ready desk constructs + `list_desks()` shape (`EXPECTED_KEYS` is exact-match) |
| Floor API contract | `gui/routes/api_floor.py` (C1) | `/api/floor/desks` proxies `list_desks()`; docstring lists exact keys |
| Workspace badge | `gui/static/js/backtest.js:287` | `?ws=production` flips badge — extend to filter desks |

## Gate definition (the promotion criterion)
A signal earns a Production desk slot only if **both** hold:
1. **IC gate** — passes `signal_ic.py` (stable, significant rank-IC; turnover not cost-prohibitive).
2. **OOS gate** — the resulting desk passes `desk_backtest.py`: total return > 0 **AND** Sharpe > 0.5 **AND** Deflated Sharpe > 0 **AND** no catastrophic regime.

## Tasks

### Phase 1 — Build the promotion rail (gate-status, not a workspace move)
- **Semantics correction:** the GUI's `Production` workspace already means *live E*TRADE execution* (`base.html:17-20`), with desks deliberately excluded. So "promoted" must NOT route a desk into Production — it means *passed both evidence gates*. All desks stay in the Sandbox Desks page; promotion is a status badge.
- Add `_PROMOTED_DESKS: frozenset = frozenset()` to `registry.py` (mirrors the existing `_MODEL_SELECTABLE_DESKS` pattern; empty until a desk earns it). Derive `gate_status: 'promoted' | 'research'` in `list_desks()` from membership.
- Touches: `desks/registry.py` (frozenset + `gate_status` in list_desks), `tests/test_desk_registry.py` (`EXPECTED_KEYS` += `gate_status`; invariant test vs `_PROMOTED_DESKS`), `gui/routes/api_floor.py` (C1 docstring), `gui/static/js/floor.js` (render gate badge).
- Promotion (Phase 3) = add the desk key to `_PROMOTED_DESKS`. One word.
- Verify: registry test asserts `gate_status` ∈ {research, promoted} and matches `_PROMOTED_DESKS`; Floor shows a Research/Gate-passed badge per desk. (Filter deferred until ≥1 promoted desk exists — YAGNI.)

### Phase 2 — Run the momentum candidate through the gate (honestly)
- Re-confirm `mom_12_1` IC on the current universe (`signal_ic.py`); run `proto_momentum_desk.py` through the OOS gate; test `--residual` and the crash filter to lift Sharpe over 0.5.
- Decision: clears → Phase 3. Does not clear → stays a Sandbox candidate; report the honest finding. Both are valid "Both" outcomes. No overfitting past the gate.

### Phase 3 — Promote (only if it earns it)
- Move `MomentumModel`/`MomentumDesk` from `scripts/` into `desks/momentum.py`; register with `track: 'production'`; keep the script runnable via re-export.
- Verify: registry test covers `momentum`; `desk_backtest.py --desk momentum` reproduces the gate-passing run.

### Phase 4 — Quarantine the edgeless desks (subtractive)
- Six current desks → `track: 'sandbox'`. janestreet labeled explicitly options-simulation-only. No deletion — still runnable for research.
- Verify: Production shows only promoted desk(s); all desks still construct.

### Phase 5 — Document the repeatable loop
- Short section: new signal → `signal_ic.py` (IC gate) → single-signal desk on the base → `desk_backtest.py` (OOS gate) → flip `track`. Next candidates: residual momentum; reversal_5 only in a cost-aware low-freq form. Everything else failed.

### Phase 6 — (Optional) prune the over-build
- RL throttle, LSTM/MLP zoo, HMM are unused once production = momentum. Tag sandbox-research-only; stop maintaining as production. No reflexive deletion.

## Validation
```bash
.venv/bin/python scripts/signal_ic.py --selftest
.venv/bin/python scripts/signal_ic.py
.venv/bin/python scripts/proto_momentum_desk.py --selftest
.venv/bin/python scripts/proto_momentum_desk.py
.venv/bin/python scripts/proto_momentum_desk.py --residual
.venv/bin/python -m pytest tests/test_desk_registry.py -q
```

## Phase 2 — RESULT (2026-06-29): no promotion (verdict code-verified)
IC gate ✅ for `mom_12_1` (t +2.66 @1d, 11.7% turnover, monotone) — edge is real but **1d-concentrated, decaying to insignificance by 21d** (t 1.24).
OOS gate ❌ for the desk built on it:
- Plain (crash-filter on): +46.04% return, **Sharpe 0.20**, Deflated 0.03, MaxDD −17.9%.
- Residual (beta-stripped): +24.26%, **Sharpe 0.07**, Deflated 0.01, MaxDD −23.3% (tamed 2018Q4 but gave back more in 2021).
- Both far below the 0.5 Sharpe gate. `_PROMOTED_DESKS` stays empty.

**Adversarial verification (workflow `wf_f9debbfc-605`, 4 lenses + Opus synthesis):** a Haiku lens claimed a short-book-dropout bug (conviction sizing → net-long tilt). **Refuted by direct code read** — `_conviction_sizes` is per-side and scale-free (`cross_sectional.py:325-327,345-348,462-467`); the `Dropping SHORT intent` log is ~0.2% of trades and symmetric with longs. **No artifact. Sharpe 0.20 is a faithful "edge too thin to be risk-efficient" measurement.** Three lenses + synthesis converge: gate is well-calibrated, signal density (mean IC ≈0.033 @1d) is simply too low.

**Best evidence-motivated next step (if pursued):** `min_holding_days` 3→1 + `exit_quantile` 0.3→0.2 to trade the 1d alpha peak — but honest P(clear 0.5) ≈ 15–20%, leaning low; it's a *diagnostic confirmation* run (in-sample param-mining risk), not a promotion candidate. **Real direction: raise edge DENSITY, not turnover** — hybrid `mom_12_1` + `reversal_5` (with explicit cost modeling), signal-level vol scaling, or a forward-vol crash overlay. Phase 3 = research, not a desk promotion.

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Momentum Sharpe ~0.2; gate needs >0.5 — may not clear even as the best signal | HIGH | Residual momentum + crash filter are the levers; an honest "no production desk yet" *is* the finding |
| LARGE_CAP_100 survivorship bias flatters +44.9% | MED | Deflated Sharpe corrects multiple-testing; note bias |
| `track` changes `list_desks()` shape (GUI contract C1) | MED | Additive, default `sandbox`; update `EXPECTED_KEYS`, C1 docstring, floor JS |
| Moving desk out of `scripts/` breaks its `desk_backtest` import | LOW | Re-export or update import; keep script runnable |

## Acceptance
- [ ] Production/Sandbox functional, not cosmetic — desks filter by `track`
- [ ] Momentum desk run through both gates; pass/fail recorded honestly
- [ ] Production track contains only gate-passing desk(s); six others quarantined
- [ ] Promotion loop documented; tests green
