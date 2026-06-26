# Plan: Uncertainty-Scaled Sizing via Committee Disagreement (Kronos idea #3)

**Source idea**: Kronos samples `sample_count` paths and averages them — but its own backtest discards the *spread*. The spread across paths is forecast uncertainty.
**Adaptation**: the cross-sectional desk already runs a **committee** of models. Use the std of committee members' per-symbol scores as a free uncertainty proxy and shrink position size when the committee disagrees.
**Complexity**: Medium (highest conceptual value of the four)

## Summary
Position size today scales with `|score|` (point conviction) but not with *confidence*.
Add an opt-in sizing multiplier that shrinks a name's size when the committee's
members disagree about it (high cross-model score dispersion), concentrating risk
where the ensemble agrees. Single-member committees and the default-off flag keep
existing backtests byte-identical.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Committee | [`desks/cross_sectional.py:161`](../../desks/cross_sectional.py) `self._committee: List[Tuple[str, WalkForwardController]]` | each member produces a per-symbol score |
| Flag-gated sizing | [`desks/cross_sectional.py:315`](../../desks/cross_sectional.py) `if self.size_by_signal_strength:` | new sizing behavior gated behind a default-off flag → byte-identical when off |
| Conviction sizing | [`desks/cross_sectional.py:399`](../../desks/cross_sectional.py) `_conviction_sizes` | budget split ∝ `|score|`, clamped `[floor, flat]`; the multiplier hooks in here |
| Predict contract | [`desks/ml_model.py:159`](../../desks/ml_model.py) | each model returns `{symbol: P(up)-0.5}`; dispersion = std across members |
| Tests | desk backtest tests (deterministic, golden) | assert byte-identical when flag off; documented behavior when on |

## Files to Change
| File | Action | Why |
|---|---|---|
| `desks/cross_sectional.py` | UPDATE | compute per-symbol committee dispersion; `shrink_by_disagreement` flag; apply multiplier in `_conviction_sizes` |
| `desks/walk_forward.py` | READ/maybe UPDATE | expose per-member scores if the committee currently pre-averages them |
| `tests/test_cross_sectional*.py` | UPDATE | flag-off byte-identical; flag-on shrinks a high-dispersion name |

## Tasks
### Task 1: Surface per-member scores
- **Action**: at the committee aggregation point, in addition to the mean `scores` dict, collect each member's score per symbol → compute `dispersion[symbol] = std(member_scores)` (NaN/0 when <2 members reported).
- **Mirror**: however the committee currently merges controller `predict()` outputs into `scores`.
- **Validate**: unit test with two stub controllers returning known scores asserts the dispersion value.

### Task 2: Confidence multiplier (opt-in)
- **Action**: add `shrink_by_disagreement: bool = False` and `disagreement_lambda: float`. When on, multiply each name's conviction size by `1 / (1 + λ · normalized_dispersion)`, renormalizing per side so gross is preserved and the book stays dollar-neutral. When off, `_conviction_sizes` is unchanged.
- **Mirror**: the `size_by_signal_strength` gating and the per-side renormalization already in `_conviction_sizes`.
- **Validate**: flag-off run byte-identical to current golden; flag-on run sizes a high-dispersion name smaller than an equal-`|score|` low-dispersion name.

## Validation
```bash
.venv/bin/python -m pytest tests/ -k "cross_sectional or committee or sizing" -q
.venv/bin/python -m pytest -n auto -q        # default-off path stays golden
.venv/bin/ruff check desks/cross_sectional.py desks/walk_forward.py
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Single-member committees give zero dispersion (no signal) | Med | document: needs ≥2 members; degrade to current behavior (multiplier=1) |
| Dispersion scale/normalization is arbitrary | Med | normalize per-rebalance (e.g. by cross-section median dispersion); expose `λ` |
| Touches risk/sizing — easy to perturb existing backtests | High | hard default-off flag → byte-identical; renormalize to preserve gross/neutrality |
| Committee may pre-average scores (members not retained) | Med | Task 1 first confirms/adds per-member retention before sizing work |

## Acceptance
- [ ] Per-symbol committee dispersion computed (≥2 members)
- [ ] `shrink_by_disagreement` default-off → existing backtests byte-identical
- [ ] Flag-on: higher disagreement → smaller size, gross & dollar-neutrality preserved
- [ ] Suite green; ruff clean
