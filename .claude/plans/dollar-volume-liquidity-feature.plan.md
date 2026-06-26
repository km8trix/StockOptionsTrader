# Plan: Dollar-Volume ("Amount") Liquidity Feature (Kronos idea #4)

**Source idea**: Kronos treats `amount` (dollar volume) as a channel, synthesizing it as `volume × mean(OHLC)` when missing.
**Adaptation**: add a dollar-volume liquidity feature to the shared feature set (a turnover signal distinct from share-volume `volume_ratio`), and optionally a liquidity floor for the universe.
**Complexity**: Small

## Summary
The models see `volume_ratio` (share volume vs its average) but nothing in *dollar*
terms, so a $2 stock and a $2,000 stock with equal share volume look identical in
liquidity. Add a 20-day dollar-volume ratio feature (`close*volume` vs its average)
to the extended set; optionally gate the tradable universe on a minimum average
dollar volume.

## Patterns to Mirror
| Category | Source | Pattern |
|---|---|---|
| Feature definition | [`desks/features.py:132`](../../desks/features.py) `extended_feature_frame` | extras derived from the frame, combined `dropna()` |
| Existing volume feature | [`desks/features.py:126`](../../desks/features.py) `volume_ratio = volume / volume_sma` | mirror as a *dollar* ratio; prefer enriched cols, else recompute |
| Column registry | [`desks/features.py:54`](../../desks/features.py) `EXTRA_FEATURE_COLUMNS` | append the new column |
| Universe (optional) | [`data/universe.py`](../../data/universe.py) `LARGE_CAP_100` / holdings | a liquidity floor would filter symbols here |
| Tests | [`tests/test_market_data.py`](../../tests/test_market_data.py) `make_ohlcv`; `tests/test_universe.py` | OHLCV fixture + exact-value asserts; universe filter tests |

## Files to Change
| File | Action | Why |
|---|---|---|
| `desks/features.py` | UPDATE | add `dollar_vol_ratio` extra (20-day) |
| `data/universe.py` | UPDATE (optional) | optional `min_adv_dollars` liquidity floor |
| `tests/test_features.py` (+ `tests/test_universe.py`) | UPDATE | feature value/NaN/determinism; filter behavior |
| extended-model golden fixtures | UPDATE | regenerate (see Risks) |

## Tasks
### Task 1: Dollar-volume feature
- **Action**: in `extended_feature_frame`, `dollar_vol = data['close'] * data['volume']`; `dollar_vol_ratio = dollar_vol / dollar_vol.rolling(20).mean()`. Append `'dollar_vol_ratio'` to the extras and `FEATURE_COLUMNS`; include in combined `dropna()`. Backward-looking only (leak-free).
- **Mirror**: the `volume_ratio` computation and the extras→join→dropna flow.
- **Validate**: `pytest tests/test_features.py -k dollar -q`.

### Task 2 (optional): Universe liquidity floor
- **Action**: helper that drops symbols whose trailing average dollar volume < threshold, applied where the scan universe is built.
- **Validate**: `pytest tests/test_universe.py -q`.

### Task 3: Golden parity
- **Action (recommended)**: enable the feature and regenerate extended-model goldens (reviewed). **Alternative**: gate via the same `include_*` flag as the seasonality plan.
- **Validate**: full suite green.

## Validation
```bash
.venv/bin/python -m pytest tests/test_features.py tests/test_universe.py -q
.venv/bin/python -m pytest -n auto -q
.venv/bin/ruff check desks/features.py data/universe.py
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| New `EXTRA` column changes extended-model inputs → breaks goldens | High | regenerate-and-review, or gate off by default (shared decision with the seasonality plan) |
| `volume` missing/zero in some frames | Med | reuse the base frame's volume handling; `replace inf→NaN` then `dropna` already covers it |
| Redundant with `volume_ratio` for single-symbol models | Low | dollar terms add cross-name liquidity comparability the share ratio lacks; tree models can ignore it |

## Acceptance
- [ ] `dollar_vol_ratio` computed, leak-free, deterministic
- [ ] `base_feature_frame` byte-identical (GBM golden untouched)
- [ ] Extended-model goldens regenerated-and-reviewed OR feature gated off
- [ ] (Optional) universe liquidity floor tested; suite green; ruff clean
