# Plan: Outlier Clipping on Standardized Features (Kronos idea #2)

**Source idea**: Kronos instance-normalizes then `np.clip(x, -clip, +clip)` (clip≈5) so fat-tailed financial inputs can't blow up the model.
**Adaptation**: clip standardized features in the neural models' train-window scaler. Tree models are scale-invariant and untouched.
**Complexity**: Tiny (clip) / Medium (full per-window RevIN — out of scope)

## Summary
Add a symmetric clip to `_StandardScaler.transform` so earnings-gap / regime-break
outliers in standardized features are capped at ±k (default 5σ) before they reach
the MLP/LSTM. Hardens the torch models against fat tails with one line; the golden
GBM, LightGBM, and feature library are unaffected.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Standardization | [`desks/models/neural.py:160`](../../desks/models/neural.py) `_StandardScaler` | train-window mean/std fit in `fit`, re-applied in `transform`; zero-std→1.0 guard |
| Clip locus | [`desks/models/neural.py:184`](../../desks/models/neural.py) `transform` | `(x - mean_) / std_` — clip the result, mirroring Kronos `np.clip(x,-clip,clip)` |
| Determinism | [`desks/models/neural.py:216`](../../desks/models/neural.py) | clip is deterministic; "identical input → byte-identical scores" preserved |
| Tests | existing neural model tests (torch-optional skip pattern) | skip when torch absent; deterministic asserts otherwise |

## Files to Change
| File | Action | Why |
|---|---|---|
| `desks/models/neural.py` | UPDATE | `_StandardScaler(clip=...)` + `np.clip` in `transform` |
| `tests/test_*neural*.py` | UPDATE | assert clipping caps a synthetic outlier row; determinism |
| neural model golden fixtures | UPDATE | regenerate if clip is enabled by default (see Risks) |

## Tasks
### Task 1: Add clip to the scaler
- **Action**: `def __init__(self, clip: float | None = 5.0)`; in `transform`, `z = (x - self.mean_) / self.std_; return np.clip(z, -self.clip, self.clip) if self.clip is not None else z`. Tree models (`gbm/lightgbm/stacking/factor`) never use this scaler, so they are unaffected.
- **Mirror**: the existing `transform` body and the epsilon/zero-std guard in `fit`.
- **Validate**: unit test feeding a row with a 50σ feature asserts the output is `±clip`.

### Task 2: Golden decision
- **Action (recommended)**: default `clip=5.0` and regenerate MLP/LSTM goldens (only rows with |z|>5 change — a tiny, reviewable diff). **Alternative**: default `clip=None` (byte-identical) and enable per-model.
- **Validate**: full suite green.

## Validation
```bash
.venv/bin/python -m pytest tests/ -k "neural or mlp or lstm" -q
.venv/bin/python -m pytest -n auto -q
.venv/bin/ruff check desks/models/neural.py
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Enabling clip changes MLP/LSTM golden outputs | Med | regenerate the two fixtures (small diff), or default `clip=None` |
| Clip too tight removes real signal | Low | 5σ on standardized features is far in the tail; make it a constructor param |
| Scope creep into full RevIN (per-window price norm + denorm) | Low | explicitly out of scope; the feature-level clip captures most of the robustness benefit |

## Acceptance
- [ ] `_StandardScaler` clips standardized features to ±clip
- [ ] Tree models / feature lib unchanged; GBM golden untouched
- [ ] MLP/LSTM goldens regenerated-and-reviewed OR clip defaulted off
- [ ] Determinism preserved; suite green; ruff clean
