# Plan: Calendar / Seasonality Features (Kronos idea #1)

**Source idea**: Kronos `calc_time_stamps` derives `minute/hour/weekday/day/month` from the bar timestamp and feeds them as model inputs.
**Adaptation**: daily equities → encode **day-of-week**, **turn-of-month**, and **month-of-year** seasonality (cyclically) as shared features, picked up by every extended-set zoo model.
**Complexity**: Small

## Summary
Add calendar-derived features to the shared feature library so the walk-forward
model zoo (`mlp/lstm/lightgbm/stacking/factor`) can learn documented daily-equity
seasonality (Monday/Friday effects, turn-of-month, January effect). Features come
purely from the existing `DatetimeIndex` — no new data, no new dependency.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Feature definition | [`desks/features.py:132`](../../desks/features.py) `extended_feature_frame` | extra columns derived from the frame, joined to base, single combined `dropna()` |
| Column registry | [`desks/features.py:54`](../../desks/features.py) `EXTRA_FEATURE_COLUMNS` | new columns appended here drive every extended-set model |
| Golden parity | [`desks/features.py:6`](../../desks/features.py) "byte-identical baseline" | `base_feature_frame` MUST NOT change; additions go in extras only |
| No look-ahead | [`desks/features.py:19`](../../desks/features.py) | every row uses index `<= i`; calendar fields are functions of the row's own timestamp (trivially leak-free) |
| Tests | [`tests/test_market_data.py`](../../tests/test_market_data.py) `make_ohlcv` fixture, deterministic asserts | fixture-built OHLCV frame + exact-value assertions |

## Files to Change
| File | Action | Why |
|---|---|---|
| `desks/features.py` | UPDATE | add `SEASONAL_FEATURE_COLUMNS` + computation in `extended_feature_frame` |
| `tests/test_features.py` | CREATE/UPDATE | assert new columns, value correctness, no-NaN, determinism, no-lookahead |
| golden fixtures for extended-set models | UPDATE | regenerate the pinned outputs (see Risks) |

## Tasks
### Task 1: Add seasonal columns
- **Action**: define `SEASONAL_FEATURE_COLUMNS = ('dow_sin','dow_cos','month_sin','month_cos','turn_of_month')`. In `extended_feature_frame`, compute from `data.index` (a `DatetimeIndex`): `dow = index.dayofweek` → `sin/cos(2π·dow/5)`; `month` → `sin/cos(2π·month/12)`; `turn_of_month = ((index.day <= 3) | index.is_month_end).astype(float)`. Append to `FEATURE_COLUMNS` after the extras; include in the combined `dropna()`/reindex.
- **Mirror**: the extras block (`extras['ret_5'] = ...`) and the `combined = base.join(...).reindex(...).dropna()` flow.
- **Validate**: `pytest tests/test_features.py -k seasonal -q`.

### Task 2: Preserve or regenerate golden parity (decision-gated — see Risks)
- **Action (recommended)**: enable by default and regenerate the extended-set models' golden fixtures in the same PR, with the diff reviewed. **Alternative (conservative)**: add `include_seasonal: bool = False` param to `extended_feature_frame` (default off → byte-identical) and flip it on only for chosen models.
- **Validate**: full suite green after fixture regeneration.

## Validation
```bash
.venv/bin/python -m pytest tests/test_features.py -q
.venv/bin/python -m pytest -n auto -q        # confirm (regenerated) goldens pass
.venv/bin/ruff check desks/features.py
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Adding to `EXTRA`/`FEATURE_COLUMNS` changes inputs for ALL extended-set models → breaks pinned golden/regression tests | High | Deliberately regenerate those fixtures (review the diff), OR gate behind `include_seasonal=False` default |
| Weekly cycle scaling (`/5` vs `/7`) | Low | trading week is 5 days; document the choice; unit-test the encoding |
| Seasonality is weak/overfit alpha | Med | cyclic encoding adds only 5 low-cardinality columns; tree models can ignore them |

## Acceptance
- [ ] Seasonal columns computed from the index, leak-free, deterministic
- [ ] `base_feature_frame` byte-identical (golden GBM unaffected)
- [ ] Extended-model goldens regenerated-and-reviewed OR feature gated off by default
- [ ] Full suite green; ruff clean
