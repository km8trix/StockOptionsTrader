# Plan: Allocator Optimizer Zoo — max-diversification & mean-variance (Vibe-Trading idea #2)

**Source idea**: Vibe-Trading `agent/backtest/optimizers/` ships a family of covariance-aware portfolio optimizers (risk_parity, mean_variance, max_diversification, equal_volatility).
**Adaptation**: extend SOT's cross-desk allocator with `max_diversification` and `mean_variance` weighting modes, mirroring exactly how `risk_parity_cov` was added in #49.
**Complexity**: Small–Medium

## Summary
SOT's allocator now offers three weighting modes — `risk_parity` (inverse-vol),
`performance`, and `risk_parity_cov` (full-covariance, #49). Add two more
covariance-aware optimizers from Vibe-Trading: **max-diversification** (maximize
the diversification ratio) and **mean-variance** (long-only max-Sharpe). Both are
numpy-only and slot in as new opt-in modes; the inverse-vol default stays
byte-identical.

**Explicitly excluded:** Vibe's `equal_volatility` optimizer is `weight_i ∝ 1/vol_i`
— identical to SOT's existing `risk_parity` (inverse-vol) mode. Adding it would be
a duplicate, so it is intentionally NOT included.

**Scope choice:** extend the existing mode-string + method-dispatch pattern (what
`risk_parity_cov` did), NOT a new `BaseOptimizer` ABC / `build_optimizer` registry.
A formal registry (mirroring `build_model`) is a reasonable future refactor only if
the mode count keeps growing — out of scope here per simplicity.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| New covariance optimizer + degrade | [`desks/capital_allocator.py:143`](../../desks/capital_allocator.py) `risk_parity_cov_weights` (#49) | the exact precedent: covariance method, overlap-aligned returns, degrade to inverse-vol→equal-weight |
| Status variant | [`desks/capital_allocator.py:185`](../../desks/capital_allocator.py) `risk_parity_cov_weights_with_status` | if a `_with_status` variant exists for cov, mirror it for the new optimizers |
| Base interface / fallback | [`desks/capital_allocator.py:110`](../../desks/capital_allocator.py) `risk_parity_weights`, `:81` `degenerate_desks`, `:244` `_bounded_renormalize`, `:300` `_equal_weight` | `Dict[str,Sequence[float]] -> Dict[str,float]`, scaled to `target_gross`, conservative degrade |
| Mode wiring | [`desks/dynamic_reweighter.py:60`](../../desks/dynamic_reweighter.py) `_WEIGHTING_MODES = ('risk_parity','performance','risk_parity_cov')` | add modes + dispatch; validated in `__init__` |
| No heavy deps | [`desks/capital_allocator.py:21`](../../desks/capital_allocator.py) "NO scipy/sklearn" | numpy-only; for mean-variance use `np.linalg` and guard singular Σ |
| Tests | [`tests/test_capital_allocator_cov.py`](../../tests/test_capital_allocator_cov.py), [`tests/test_dynamic_reweighter_cov.py`](../../tests/test_dynamic_reweighter_cov.py) (created in #49) | mirror these for the two new modes |

## Files to Change
| File | Action | Why |
|---|---|---|
| `desks/capital_allocator.py` | UPDATE | add `max_diversification_weights` and `mean_variance_weights` (+ `_with_status` variants if the cov one has them) |
| `desks/dynamic_reweighter.py` | UPDATE | add `'max_diversification'`, `'mean_variance'` to `_WEIGHTING_MODES` + dispatch |
| `tests/test_capital_allocator_optimizers.py` | CREATE | unit-test both optimizers + degrade, mirroring the `_cov` tests |
| `tests/test_dynamic_reweighter_optimizers.py` | CREATE | mode-wiring + default byte-identical |

## Tasks
### Task 1: `max_diversification_weights`
- **Action**: align per-desk returns on the overlapping window (reuse the #49 helper); compute σ (per-desk vol) and Σ (covariance). Maximize the diversification ratio `(wᵀσ)/sqrt(wᵀΣw)`, long-only, weights→`target_gross` via `_bounded_renormalize`. A simple numpy iterative scheme (or the standard `Σ⁻¹σ` closed form, clipped non-negative + renormalized) is fine. **Degrade** to inverse-vol→equal-weight on short overlap / degenerate vol / singular Σ.
- **Mirror**: `risk_parity_cov_weights` structure + degrade.
- **Validate**: unit test — a redundant (highly-correlated) desk receives LESS weight than under inverse-vol; weights sum to `target_gross`.

### Task 2: `mean_variance_weights`
- **Action**: long-only max-Sharpe — `w ∝ Σ⁻¹ μ` where `μ` = per-desk mean return over the aligned window; clip negatives to 0, renormalize to `target_gross`. Guard singular/non-finite Σ (`np.linalg.LinAlgError`) → degrade to inverse-vol→equal-weight.
- **Mirror**: same degrade contract.
- **Validate**: unit test — a desk with higher risk-adjusted mean gets more weight; singular Σ degrades cleanly; sums to `target_gross`.

### Task 3: Wire modes + parity
- **Action**: add both modes to `_WEIGHTING_MODES` + dispatch; default `weighting='risk_parity'` unchanged. If a test pins `_WEIGHTING_MODES` exactly, update it.
- **Validate**: default run byte-identical; new modes selectable.

## Validation
```bash
.venv/bin/python -m pytest tests/test_capital_allocator_optimizers.py \
    tests/test_dynamic_reweighter_optimizers.py tests/test_capital_allocator.py \
    tests/test_capital_allocator_cov.py tests/test_dynamic_reweighter.py -q
.venv/bin/python -m pytest -n auto -q        # default path byte-identical
.venv/bin/ruff check desks/capital_allocator.py desks/dynamic_reweighter.py
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| mean-variance needs Σ⁻¹ (singular/ill-conditioned covariance) | Med | ε-regularize or catch `LinAlgError` → degrade to inverse-vol (mirror #49) |
| Return-series alignment across desks (different lengths) | High | reuse the overlap-alignment from `risk_parity_cov_weights` (#49) |
| `_WEIGHTING_MODES` pinned in a test | Med | update that assertion |
| Perturbing existing allocation | High | new methods + opt-in modes; default inverse-vol untouched → byte-identical |
| Accidentally re-adding `equal_volatility` (= inverse-vol) | Low | explicitly excluded; documented above |

## Acceptance
- [ ] `max_diversification_weights` + `mean_variance_weights` implemented, numpy-only, degrade like `risk_parity_cov`
- [ ] Both wired as opt-in modes; `equal_volatility` NOT added (duplicate of inverse-vol)
- [ ] Default `weighting='risk_parity'` byte-identical; full suite green; ruff clean
