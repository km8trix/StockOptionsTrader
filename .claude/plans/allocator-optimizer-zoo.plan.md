# Plan: Allocator Optimizer Zoo (Vibe-Trading idea #2)

**Source idea**: Vibe-Trading `agent/backtest/optimizers/` ships a small zoo of
portfolio optimizers (`mean_variance`, `max_diversification`, `equal_volatility`, …).
**Adaptation**: add the two NON-redundant optimizers to the cross-desk allocator
as opt-in weighting modes, numpy-only, mirroring the `risk_parity_cov` precedent
(#49). `equal_volatility` is EXCLUDED — Vibe's `weight_i ∝ 1/vol_i` is identical
to SOT's existing inverse-vol `risk_parity`, so adding it would be a duplicate.
**Complexity**: Small

## Summary
The allocator has `risk_parity` (inverse-vol, default), `performance`, and
`risk_parity_cov` (full-covariance, #49). Add two MORE covariance-aware modes:

- **`max_diversification`** — maximize the diversification ratio
  `DR = (wᵀσ)/sqrt(wᵀΣw)` via the closed form `w ∝ Σ⁻¹σ` (Choueifaty & Coignard's
  Most-Diversified Portfolio); long-only (clip <0), scaled to `target_gross`. A
  redundant (highly correlated) desk earns less weight than inverse-vol gives it.
- **`mean_variance`** — long-only max-Sharpe tangency `w ∝ Σ⁻¹μ` with μ the
  per-desk mean over the aligned window; clip negatives, scale to `target_gross`.
  Tilts toward desks with a better risk-adjusted mean.

Both reuse the #49 overlap alignment + degrade discipline and stay purely
additive: the inverse-vol default is byte-identical.

## Patterns Mirrored (#49)
| Category | Source | Pattern |
|---|---|---|
| Allocator method + status variant | `desks/capital_allocator.py` `risk_parity_cov_weights[_with_status]` | covariance method, overlap-aligned returns, degrade-to-inverse-vol→equal-weight, `(weights, reason)` for honest audit |
| Alignment | `risk_parity_cov_weights_with_status` inline | factored into shared `_aligned_matrix` helper used by both new modes |
| Degeneracy / scaling / fallback | `degenerate_desks`, `_equal_weight`, `risk_parity_weights` | same conservative gate + inverse-vol fallback |
| Mode wiring | `desks/dynamic_reweighter.py` `_WEIGHTING_MODES` + `on_day` dispatch | mode string → allocator `*_with_status` method, validated in `__init__` |
| Tests | `tests/test_*_cov.py` | mirrored into `tests/test_*_optimizers.py` |

## Files Changed
| File | Action |
|---|---|
| `desks/capital_allocator.py` | UPDATE — add `_aligned_matrix`, `max_diversification_weights[_with_status]`, `mean_variance_weights[_with_status]` |
| `desks/dynamic_reweighter.py` | UPDATE — add both modes to `_WEIGHTING_MODES` + dispatch |
| `tests/test_capital_allocator_optimizers.py` | CREATE |
| `tests/test_dynamic_reweighter_optimizers.py` | CREATE |
| `tests/test_dynamic_reweighter_cov.py` | UPDATE — bump the pinned `_WEIGHTING_MODES` assertion |

## Degrade (both modes, mirroring `risk_parity_cov`)
Fall back to `risk_parity_weights` (inverse-vol) — which itself falls back to
`_equal_weight` — on: too-short overlap, any degenerate-vol desk (same
`degenerate_desks` gate), singular/non-finite covariance (`np.linalg.LinAlgError`
or non-finite guard), or a long-only solution that clips to nothing
(mean-variance: no desk with positive risk-adjusted mean). All wrapped in
try/except so any numerical failure degrades rather than raises. The
`_with_status` variants name the numerical cause so the reweighter's audit log
records an honest fallback.

## Validation
```bash
.venv/bin/python -m pytest tests/test_capital_allocator_optimizers.py \
    tests/test_dynamic_reweighter_optimizers.py tests/test_capital_allocator.py \
    tests/test_capital_allocator_cov.py tests/test_dynamic_reweighter.py -q
.venv/bin/python -m pytest -n auto -q        # default path stays byte-identical
.venv/bin/ruff check desks/capital_allocator.py desks/dynamic_reweighter.py
```

## Acceptance
- [x] `max_diversification` down-weights a redundant/correlated desk vs inverse-vol; raises the diversification ratio
- [x] `mean_variance` favors the higher risk-adjusted-mean desk; long-only
- [x] Both sum to `target_gross`; numpy-only (no scipy/sklearn)
- [x] Degrade to inverse-vol → equal-weight on short overlap / degenerate / singular cov
- [x] Default `weighting='risk_parity'` byte-identical; full suite green (1998); ruff clean
