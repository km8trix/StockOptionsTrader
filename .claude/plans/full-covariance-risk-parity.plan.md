# Plan: Full-Covariance Risk Parity (Vibe-Trading idea #1)

**Source idea**: Vibe-Trading `agent/backtest/optimizers/risk_parity.py` does *true* risk parity over the full covariance matrix (Newton iteration equalizing marginal risk contributions), not just inverse-vol.
**Adaptation**: add a covariance-aware weighting mode to the cross-desk allocator alongside the existing inverse-vol one, numpy-only, opt-in.
**Complexity**: Small–Medium

## Summary
SOT's "risk parity" is **inverse-vol only** (`weight_i ∝ 1/vol_i`) — a diagonal-
covariance approximation that ignores correlation between desks, so two
correlated desks carry more combined risk than intended. Add a
`risk_parity_cov` weighting mode that equalizes *actual* marginal risk
contributions using the full covariance (5-iteration Newton, pure numpy, no
scipy), exposed as a new opt-in mode. The existing inverse-vol default stays
byte-identical.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Allocator method | [`desks/capital_allocator.py:107`](../../desks/capital_allocator.py) `risk_parity_weights` | `Dict[str, Sequence[float]] -> Dict[str, float]`, scaled to `target_gross`, equal-weight fallback on degenerate vol |
| Second mode precedent | [`desks/capital_allocator.py:140`](../../desks/capital_allocator.py) `performance_weights` | how a parallel weighting method was added alongside inverse-vol |
| Normalization / fallback | [`desks/capital_allocator.py:244`](../../desks/capital_allocator.py) `_bounded_renormalize`, `:300` `_equal_weight`, `:81` `degenerate_desks` | reuse for scaling + the conservative degrade |
| Mode wiring | [`desks/dynamic_reweighter.py:60`](../../desks/dynamic_reweighter.py) `_WEIGHTING_MODES = ('risk_parity','performance')` | a mode string maps to an allocator method; validated in `__init__` |
| No heavy deps | [`desks/capital_allocator.py:21`](../../desks/capital_allocator.py) "NO scipy/sklearn dependency" | Newton iteration in plain numpy |
| Tests | [`tests/test_capital_allocator_performance.py`](../../tests/test_capital_allocator_performance.py), [`tests/test_dynamic_reweighter_performance.py`](../../tests/test_dynamic_reweighter_performance.py) | the `performance` mode shipped its own test file — mirror for `_cov` |

## Files to Change
| File | Action | Why |
|---|---|---|
| `desks/capital_allocator.py` | UPDATE | add `risk_parity_cov_weights(returns_by_desk)` |
| `desks/dynamic_reweighter.py` | UPDATE | add `'risk_parity_cov'` to `_WEIGHTING_MODES` + dispatch |
| `tests/test_capital_allocator_cov.py` | CREATE | unit-test the covariance weighting + degrade |
| `tests/test_dynamic_reweighter_cov.py` | CREATE | mode-wiring + default byte-identical |

## Tasks
### Task 1: `risk_parity_cov_weights`
- **Action**: align the per-desk return series on their overlapping window (equal length, dropna); compute covariance `cov` (numpy). Seed `w = inverse-vol` (reuse `risk_parity_weights`), then iterate ~5×:
  `port_vol = sqrt(wᵀ·cov·w); mrc = (cov·w)/port_vol; rc = w*mrc; w *= (port_vol/n)/(rc+ε); w = clip(w,0,·); w/=w.sum()`. Scale to `target_gross` via `_bounded_renormalize`. **Degrade** (mirror the existing fallback) to `risk_parity_weights` (inverse-vol) — and that in turn to `_equal_weight` — when overlap is too short, any vol is degenerate, or the covariance is singular/non-finite.
- **Mirror**: `risk_parity_weights` interface + `degenerate_desks`/`_bounded_renormalize`/`_equal_weight`.
- **Validate**: unit test — two positively-correlated desks + one uncorrelated; assert the correlated pair's *combined* weight is LESS than inverse-vol gives them, weights sum to `target_gross`, and marginal risk contributions are ~equal.

### Task 2: Wire the mode
- **Action**: add `'risk_parity_cov'` to `_WEIGHTING_MODES` and dispatch to the new method (mirror the `performance` branch). Default `weighting='risk_parity'` unchanged.
- **Validate**: `weighting='risk_parity_cov'` selects the new path; default run byte-identical to current.

### Task 3: Tests + parity
- **Action**: new test files mirroring the `_performance` ones; degrade paths; a pinned check that the default mode is unchanged.
- **Validate**: full suite green.

## Validation
```bash
.venv/bin/python -m pytest tests/test_capital_allocator_cov.py tests/test_dynamic_reweighter_cov.py \
    tests/test_capital_allocator.py tests/test_dynamic_reweighter.py -q
.venv/bin/python -m pytest -n auto -q        # default path stays byte-identical
.venv/bin/ruff check desks/capital_allocator.py desks/dynamic_reweighter.py
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Covariance needs aligned, equal-length series (desks have different-length curves) | High | align on overlapping window + dropna; too-short overlap → degrade to inverse-vol |
| Singular / ill-conditioned covariance → Newton blows up | Med | ε regularize, cap iterations, non-finite/NaN guard → degrade to inverse-vol |
| A test pins `_WEIGHTING_MODES` exactly | Med | update that assertion (mirrors the FEATURE_COLUMNS count bump from Kronos #1) |
| Perturbing existing allocation | High | new method + opt-in mode; default `'risk_parity'` (inverse-vol) untouched → byte-identical |

## Acceptance
- [ ] `risk_parity_cov_weights` equalizes marginal risk contributions over the full covariance (numpy-only, no scipy)
- [ ] Correlated desks down-weighted vs inverse-vol; weights sum to `target_gross`
- [ ] Degrades to inverse-vol → equal-weight on short overlap / degenerate / singular covariance
- [ ] Default `weighting='risk_parity'` byte-identical; full suite green; ruff clean
