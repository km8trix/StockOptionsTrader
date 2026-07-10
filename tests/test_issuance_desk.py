"""Tests for the IssuanceDesk: sign convention (buybacks long), PIT slicing,
staleness, exclude_micro, the stable key contract, and one-pull-per-run
caching. All hermetic — in-memory fake provider, no network, no warehouse
(the test_pead.py desk-test pattern)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.share_issuance import _share_rows
from desks.issuance import IssuanceDesk
from portfolio.manager import PortfolioManager

_NAMES = ['BBK', 'FL2', 'FLT', 'ISS']


def _share_universe():
    """4 names, 9 quarters (reportperiods 2020-03-31..2022-03-31, datekey
    +40d so the last filing lands 2022-05-10):

    BBK  big buyback in the final quarter only (100 -> 50: issuance -0.50
         at q8, 0.0 before) — the planted long;
    ISS  issuer (+20%/yr from q5 on: issuance +0.20) — the planted short;
    FLT / FL2  flat share counts (issuance 0.0)."""
    rows = (_share_rows('BBK', [100.0] * 8 + [50.0])
            + _share_rows('ISS', [100.0] * 5 + [120.0] * 4)
            + _share_rows('FLT', [100.0] * 9)
            + _share_rows('FL2', [200.0] * 9))
    return pd.DataFrame(rows)


class _FakeProvider:
    """Minimal PitWarehouse stand-in: canned share rows + optional caps."""

    def __init__(self, shares, caps=None):
        self._shares = shares
        self._caps = caps or {}
        self.pull_calls = 0

    def fundamentals_quarterly(self, tickers=None, *, fields=()):
        assert tuple(fields) == ('sharesbas', 'sharefactor')
        self.pull_calls += 1
        df = self._shares
        if tickers is not None:
            df = df[df['ticker'].isin(tickers)]
        return df.reset_index(drop=True)

    def daily_marketcaps(self, tickers, date):
        return {t: self._caps[t] for t in tickers if t in self._caps}


def _frames(names, start, periods=15):
    idx = pd.bdate_range(start, periods=periods)
    return {n: pd.DataFrame({'close': np.full(periods, 50.0)}, index=idx)
            for n in names}, idx


# ---------------------------------------------------------------------------
# Sign convention — LOW issuance (buyback) = highest score = long
# ---------------------------------------------------------------------------
class TestSignConvention:
    def test_buyback_scores_above_issuer(self):
        desk = IssuanceDesk(provider=_FakeProvider(_share_universe()),
                            exclude_micro=False)
        frames, idx = _frames(_NAMES, '2022-06-01')
        scores = desk._alpha_scores(frames, idx[-1])   # after q8 datekeys
        assert scores is not None
        # score = MINUS issuance: the -0.50 buyback reads +0.50 (top of the
        # book), the +0.20 issuer reads -0.20 (bottom).
        assert scores['BBK'] == pytest.approx(0.50)
        assert scores['ISS'] == pytest.approx(-0.20)
        assert scores['BBK'] == max(scores.values())
        assert scores['ISS'] == min(scores.values())

    def test_planted_buyback_longed_over_issuer(self):
        # Through the actual book machinery: highest score = long, so the
        # buyback is the ONE name opened (quantile 0.25 of 4) and the issuer
        # is never longed; long_only (the default) suppresses the short leg.
        desk = IssuanceDesk(provider=_FakeProvider(_share_universe()),
                            exclude_micro=False, quantile=0.25)
        frames, idx = _frames(_NAMES, '2022-06-01')
        date = idx[-1]
        desk.set_clock(date)
        intents = desk.generate_intents(frames, date,
                                        PortfolioManager(100_000.0))
        assert [i.asset.symbol for i in intents if i.action == 'BUY'] == \
            ['BBK']
        assert all(i.asset.symbol != 'ISS' for i in intents)
        assert not any(i.action == 'SHORT' for i in intents)


# ---------------------------------------------------------------------------
# PIT boundary + staleness
# ---------------------------------------------------------------------------
class TestPitAndStaleness:
    def test_filing_after_the_rebalance_is_invisible(self):
        # BBK's buyback files 2022-05-10; a May-1 score must not see it even
        # though the desk's one-shot pull contains it (datekey <= t slice —
        # the PEAD PIT convention, pinned here).
        desk = IssuanceDesk(provider=_FakeProvider(_share_universe()),
                            exclude_micro=False)
        frames, _ = _frames(_NAMES, '2022-04-01')
        early = desk._alpha_scores(frames, pd.Timestamp('2022-05-01'))
        assert early is not None
        assert early['BBK'] == pytest.approx(0.0)      # q7 (0.0), not q8
        # ...and the SAME desk sees it once the filing lands (new month).
        frames2, idx2 = _frames(_NAMES, '2022-06-01')
        late = desk._alpha_scores(frames2, idx2[-1])
        assert late is not None and late['BBK'] == pytest.approx(0.50)

    def test_stale_names_drop_out(self):
        # Last filing 2022-05-10 + 400 stale days ~ 2023-06-14: a June-1
        # rebalance still scores, a July one finds every name stale (all
        # four stopped filing) and the desk goes flat (None).
        prov = _FakeProvider(_share_universe())
        fresh = IssuanceDesk(provider=prov, exclude_micro=False)
        frames, _ = _frames(_NAMES, '2023-05-15')
        assert fresh._alpha_scores(frames, pd.Timestamp('2023-06-01')) \
            is not None
        stale = IssuanceDesk(provider=prov, exclude_micro=False)
        assert stale._alpha_scores(frames, pd.Timestamp('2023-07-03')) is None


# ---------------------------------------------------------------------------
# exclude_micro
# ---------------------------------------------------------------------------
def test_exclude_micro_drops_the_bottom_tercile():
    # PIT caps ascending: ISS < FLT < FL2 < BBK -> terciles of 4 put the two
    # smallest (ISS, FLT) in bucket 0 = micro -> dropped, computable or not.
    caps = {'ISS': 1e8, 'FLT': 2e8, 'FL2': 3e9, 'BBK': 5e9}
    desk = IssuanceDesk(provider=_FakeProvider(_share_universe(), caps=caps))
    desk.min_scored = 2                              # tiny universe
    frames, idx = _frames(_NAMES, '2022-06-01')
    scores = desk._alpha_scores(frames, idx[-1])
    assert scores is not None
    assert set(scores) == {'BBK', 'FL2'}
    assert 'ISS' not in scores and 'FLT' not in scores


# ---------------------------------------------------------------------------
# Key contract + fund-slot defaults
# ---------------------------------------------------------------------------
def test_key_is_stable_regardless_of_params():
    # The fund allocation dict addresses the desk by .key
    # (tests/test_vq_fund_gate.py key contract): 'issuance', never derived
    # from params.
    prov = _FakeProvider(_share_universe())
    assert IssuanceDesk(provider=prov).key == 'issuance'
    assert IssuanceDesk(provider=prov, exclude_micro=False, long_only=False,
                        quantile=0.1, stale_days=100).key == 'issuance'


def test_fund_slot_defaults():
    # The defaults ARE the fund slot: long-only, ex-micro, the screen's
    # 400d staleness, the wide 0.50 stop (monthly signal), fixed factor.
    desk = IssuanceDesk(provider=_FakeProvider(_share_universe()))
    assert desk._long_only is True
    assert desk._exclude_micro is True
    assert desk._stale_days == 400
    assert desk.quantile == 0.2
    assert desk.risk_manager.position_stop_loss == 0.50
    assert desk.walk_forward_fits == []              # committee=[] (n_trials=1)


# ---------------------------------------------------------------------------
# One pull per run + monthly cache
# ---------------------------------------------------------------------------
class TestCaching:
    def test_one_pull_per_run_and_monthly_cache(self):
        prov = _FakeProvider(_share_universe())
        desk = IssuanceDesk(provider=prov, exclude_micro=False)
        frames, idx = _frames(_NAMES, '2022-06-01')
        scores = desk._alpha_scores(frames, idx[-1])
        assert scores is not None and prov.pull_calls == 1
        # Same month => cached object, provider untouched.
        again = desk._alpha_scores(frames, idx[-1] + pd.Timedelta(days=1))
        assert again is scores and prov.pull_calls == 1
        # New month, same symbols => recomputed from the ONE cumulative pull.
        frames2, idx2 = _frames(_NAMES, '2022-07-01')
        s2 = desk._alpha_scores(frames2, idx2[-1])
        assert s2 is not None and prov.pull_calls == 1

    def test_new_symbols_trigger_repull(self):
        # A name entering all_data mid-run (IPO) must be scored, not frozen
        # out by the first pull (the PEAD adversarial-review finding).
        shares = pd.concat(
            [_share_universe(),
             pd.DataFrame(_share_rows('NEW', [100.0] * 5 + [80.0] * 4))],
            ignore_index=True)
        prov = _FakeProvider(shares)
        desk = IssuanceDesk(provider=prov, exclude_micro=False)
        frames1, idx1 = _frames(_NAMES, '2022-06-01')
        assert desk._alpha_scores(frames1, idx1[-1]) is not None
        assert prov.pull_calls == 1
        frames2, idx2 = _frames(_NAMES + ['NEW'], '2022-07-01')
        scores2 = desk._alpha_scores(frames2, idx2[-1])
        assert prov.pull_calls == 2
        assert scores2 is not None
        assert scores2['NEW'] == pytest.approx(0.20)   # -(-0.20) buyback
