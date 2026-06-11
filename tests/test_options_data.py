"""OptionsDataHandler: normalization, filters, same-day SQLite snapshot
cache, and IV history (ATM selection + fallback paths).

Offline: OptionsDataHandler._get_openbb is monkeypatched to a synthetic
chain; the clock is injected so DTE math is deterministic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.options_data import CHAIN_COLUMNS, OptionsDataHandler

FIXED_NOW = datetime(2026, 6, 11, 10, 30, 0)
TODAY = FIXED_NOW.date()
UNDERLYING = 100.0

EXP_10 = (TODAY + timedelta(days=10)).isoformat()
EXP_30 = (TODAY + timedelta(days=30)).isoformat()
EXP_60 = (TODAY + timedelta(days=60)).isoformat()


def _contract(expiration, strike, option_type, bid=1.0, ask=1.2, last=1.1,
              iv=0.25, volume=10.0, open_interest=100.0,
              underlying_price=UNDERLYING):
    return SimpleNamespace(
        expiration=expiration, strike=strike, option_type=option_type,
        bid=bid, ask=ask, last=last, volume=volume,
        open_interest=open_interest, implied_volatility=iv,
        underlying_price=underlying_price,
    )


def synthetic_chain():
    """Hand-computable chain: both option types, 3 expirations, zero-bid rows,
    and strikes outside the default moneyness band (70 and 130)."""
    contracts = []
    # 30-DTE expiration: full strike ladder, known IVs for the ATM test.
    for strike, civ, piv in [(95.0, 0.25, 0.27), (100.0, 0.22, 0.24),
                             (105.0, 0.20, 0.26)]:
        contracts.append(_contract(EXP_30, strike, 'call', iv=civ))
        contracts.append(_contract(EXP_30, strike, 'put', iv=piv))
    # Outside the 0.8..1.2 moneyness band: must be filtered.
    contracts.append(_contract(EXP_30, 70.0, 'call'))
    contracts.append(_contract(EXP_30, 130.0, 'put'))
    # 10-DTE expiration: zero-bid row exercising the mid -> last fallback.
    contracts.append(_contract(EXP_10, 100.0, 'call', bid=0.0, ask=0.0,
                               last=1.5))
    contracts.append(_contract(EXP_10, 100.0, 'put', bid=2.0, ask=2.2,
                               last=2.05))
    # 60-DTE expiration.
    contracts.append(_contract(EXP_60, 100.0, 'call', bid=3.0, ask=3.4))
    contracts.append(_contract(EXP_60, 100.0, 'put', bid=3.1, ask=3.5))
    return contracts


class FakeOBB:
    def __init__(self, contracts):
        self.contracts = contracts
        self.calls = []
        self.derivatives = SimpleNamespace(
            options=SimpleNamespace(chains=self._chains)
        )

    def _chains(self, symbol, provider):
        self.calls.append(provider)
        return SimpleNamespace(results=self.contracts)


@pytest.fixture
def fake_obb():
    return FakeOBB(synthetic_chain())


@pytest.fixture
def handler(tmp_path, monkeypatch, fake_obb):
    h = OptionsDataHandler(db_path=str(tmp_path / "options.db"),
                           now_fn=lambda: FIXED_NOW)
    monkeypatch.setattr(OptionsDataHandler, "_get_openbb",
                        lambda self: fake_obb)
    return h


class TestNormalization:
    def test_columns_and_dtypes(self, handler):
        chain = handler.fetch_chain("SPY")

        assert list(chain.columns) == CHAIN_COLUMNS
        assert chain["dte"].dtype.kind == "i"
        assert set(chain["option_type"].unique()) <= {"call", "put"}

    def test_dte_computed_vs_today(self, handler):
        chain = handler.fetch_chain("SPY")

        assert set(chain["dte"].unique()) == {10, 30, 60}

    def test_mid_from_bid_ask_when_both_positive(self, handler):
        chain = handler.fetch_chain("SPY")
        row = chain[(chain["dte"] == 30) & (chain["strike"] == 95.0)
                    & (chain["option_type"] == "call")].iloc[0]

        assert row["mid"] == pytest.approx((1.0 + 1.2) / 2)

    def test_mid_falls_back_to_last_on_zero_bid(self, handler):
        chain = handler.fetch_chain("SPY")
        row = chain[(chain["dte"] == 10)
                    & (chain["option_type"] == "call")].iloc[0]

        assert row["bid"] == 0.0
        assert row["mid"] == pytest.approx(1.5)  # == last

    def test_moneyness_filter_default(self, handler):
        chain = handler.fetch_chain("SPY")

        # 70/100 = 0.7 and 130/100 = 1.3 fall outside [0.8, 1.2].
        assert 70.0 not in chain["strike"].values
        assert 130.0 not in chain["strike"].values
        assert (chain["strike"] / UNDERLYING).between(0.8, 1.2).all()

    def test_dte_filter(self, handler):
        chain = handler.fetch_chain("SPY", min_dte=20, max_dte=45)

        assert set(chain["dte"].unique()) == {30}

    def test_missing_fields_become_nan_not_keyerror(self, tmp_path,
                                                    monkeypatch):
        # Contracts lacking bid/ask/iv entirely: defensive NaN, no KeyError.
        sparse = [SimpleNamespace(expiration=EXP_30, strike=100.0,
                                  option_type='call')]
        fake = FakeOBB(sparse)
        h = OptionsDataHandler(db_path=str(tmp_path / "sparse.db"),
                               now_fn=lambda: FIXED_NOW)
        monkeypatch.setattr(OptionsDataHandler, "_get_openbb",
                            lambda self: fake)

        chain = h.fetch_chain("XYZ")

        assert len(chain) == 1
        assert np.isnan(chain["bid"].iloc[0])
        assert np.isnan(chain["implied_volatility"].iloc[0])
        assert np.isnan(chain["mid"].iloc[0])  # no bid/ask, no last


class TestSameDaySnapshotCache:
    def test_second_call_served_from_sqlite(self, handler, fake_obb):
        first = handler.fetch_chain("SPY")
        assert fake_obb.calls == ["cboe"]

        second = handler.fetch_chain("SPY")

        assert fake_obb.calls == ["cboe"]  # provider NOT touched again
        pd.testing.assert_frame_equal(first, second)

    def test_snapshot_serves_different_filters(self, handler, fake_obb):
        handler.fetch_chain("SPY")  # populates the snapshot

        narrow = handler.fetch_chain("SPY", min_dte=20, max_dte=45)

        assert fake_obb.calls == ["cboe"]
        assert set(narrow["dte"].unique()) == {30}

    def test_provider_failover_logged_and_used(self, tmp_path, monkeypatch,
                                               caplog):
        contracts = synthetic_chain()

        class FlakyOBB(FakeOBB):
            def _chains(self, symbol, provider):
                self.calls.append(provider)
                if provider == "cboe":
                    raise RuntimeError("cboe down")
                return SimpleNamespace(results=contracts)

        fake = FlakyOBB(contracts)
        h = OptionsDataHandler(db_path=str(tmp_path / "flaky.db"),
                               now_fn=lambda: FIXED_NOW)
        monkeypatch.setattr(OptionsDataHandler, "_get_openbb",
                            lambda self: fake)

        with caplog.at_level(logging.WARNING, logger="data.options_data"):
            chain = h.fetch_chain("SPY")

        assert fake.calls == ["cboe", "tradier"]
        assert not chain.empty
        assert any("cboe" in r.message and "failed" in r.message
                   for r in caplog.records)


class TestIVHistory:
    def test_atm_iv_selection_hand_computed(self, handler):
        """Window [20,45] -> the 30-DTE expiry; underlying 100 -> strike 100;
        ATM IV = (0.22 + 0.24) / 2 = 0.23."""
        chain = handler.fetch_chain("SPY")

        recorded = handler.record_iv_snapshot("SPY", chain, hv_20=0.18)

        assert recorded == pytest.approx(0.23)
        history = handler.get_iv_history("SPY")
        assert len(history) == 1
        assert history.index[0] == pd.Timestamp(TODAY)
        assert history["atm_iv"].iloc[0] == pytest.approx(0.23)
        assert history["hv_20"].iloc[0] == pytest.approx(0.18)
        assert history["underlying_price"].iloc[0] == pytest.approx(UNDERLYING)

    def test_out_of_window_fallback_logs_and_records(self, handler, caplog):
        # Only DTE 5 and 50 available: no expiry in [20, 45]; the fallback
        # must pick the nearest DTE >= 7 (50), warn, and still record.
        chain = pd.DataFrame([
            {'expiration': TODAY + timedelta(days=5), 'strike': 100.0,
             'option_type': 'call', 'bid': 1.0, 'ask': 1.2, 'last': 1.1,
             'mid': 1.1, 'volume': 1.0, 'open_interest': 1.0,
             'implied_volatility': 0.50, 'dte': 5,
             'underlying_price': UNDERLYING},
            {'expiration': TODAY + timedelta(days=5), 'strike': 100.0,
             'option_type': 'put', 'bid': 1.0, 'ask': 1.2, 'last': 1.1,
             'mid': 1.1, 'volume': 1.0, 'open_interest': 1.0,
             'implied_volatility': 0.52, 'dte': 5,
             'underlying_price': UNDERLYING},
            {'expiration': TODAY + timedelta(days=50), 'strike': 100.0,
             'option_type': 'call', 'bid': 2.0, 'ask': 2.2, 'last': 2.1,
             'mid': 2.1, 'volume': 1.0, 'open_interest': 1.0,
             'implied_volatility': 0.30, 'dte': 50,
             'underlying_price': UNDERLYING},
            {'expiration': TODAY + timedelta(days=50), 'strike': 100.0,
             'option_type': 'put', 'bid': 2.0, 'ask': 2.2, 'last': 2.1,
             'mid': 2.1, 'volume': 1.0, 'open_interest': 1.0,
             'implied_volatility': 0.32, 'dte': 50,
             'underlying_price': UNDERLYING},
        ], columns=CHAIN_COLUMNS)

        with caplog.at_level(logging.WARNING, logger="data.options_data"):
            recorded = handler.record_iv_snapshot("SPY", chain, hv_20=0.20)

        assert recorded == pytest.approx((0.30 + 0.32) / 2)
        assert any("falling back" in r.message for r in caplog.records)
        history = handler.get_iv_history("SPY")
        assert history["atm_iv"].iloc[0] == pytest.approx(0.31)

    def test_percent_scale_hv20_warns_and_is_not_stored(self, handler,
                                                        caplog):
        # hv_20 contract: annualized DECIMAL (0.18), same units as atm_iv.
        # 18.0 is near-certain percent/decimal confusion: warn, store NULL,
        # but still record the ATM IV itself.
        chain = handler.fetch_chain("SPY")

        with caplog.at_level(logging.WARNING, logger="data.options_data"):
            recorded = handler.record_iv_snapshot("SPY", chain, hv_20=18.0)

        assert recorded == pytest.approx(0.23)  # atm_iv recording unaffected
        assert any("hv_20" in r.message and "sanity" in r.message
                   for r in caplog.records)
        history = handler.get_iv_history("SPY")
        assert len(history) == 1
        assert history["atm_iv"].iloc[0] == pytest.approx(0.23)
        assert history["hv_20"].isna().all()  # NULL, not 18.0

    def test_decimal_hv20_at_bound_stored_without_warning(self, handler,
                                                          caplog):
        # 3.0 (300% annualized) is extreme but allowed: bound is exclusive.
        chain = handler.fetch_chain("SPY")

        with caplog.at_level(logging.WARNING, logger="data.options_data"):
            handler.record_iv_snapshot("SPY", chain, hv_20=3.0)

        assert not any("sanity" in r.message for r in caplog.records)
        history = handler.get_iv_history("SPY")
        assert history["hv_20"].iloc[0] == pytest.approx(3.0)

    def test_nan_iv_skips_and_logs(self, handler, caplog):
        chain = pd.DataFrame([
            {'expiration': TODAY + timedelta(days=30), 'strike': 100.0,
             'option_type': 'call', 'bid': 1.0, 'ask': 1.2, 'last': 1.1,
             'mid': 1.1, 'volume': 1.0, 'open_interest': 1.0,
             'implied_volatility': float('nan'), 'dte': 30,
             'underlying_price': UNDERLYING},
            {'expiration': TODAY + timedelta(days=30), 'strike': 100.0,
             'option_type': 'put', 'bid': 1.0, 'ask': 1.2, 'last': 1.1,
             'mid': 1.1, 'volume': 1.0, 'open_interest': 1.0,
             'implied_volatility': 0.25, 'dte': 30,
             'underlying_price': UNDERLYING},
        ], columns=CHAIN_COLUMNS)

        with caplog.at_level(logging.WARNING, logger="data.options_data"):
            recorded = handler.record_iv_snapshot("SPY", chain, hv_20=0.20)

        assert recorded is None
        assert handler.get_iv_history("SPY").empty
        assert any("NaN IV" in r.message for r in caplog.records)

    def test_get_iv_history_date_bounds(self, handler):
        conn_dates = ["2026-06-09", "2026-06-10", "2026-06-11"]
        import sqlite3
        conn = sqlite3.connect(handler.db_path)
        for i, d in enumerate(conn_dates):
            conn.execute(
                "INSERT OR REPLACE INTO iv_history VALUES (?, ?, ?, ?, ?)",
                ("SPY", d, 0.2 + i / 100.0, 0.18, 100.0))
        conn.commit()
        conn.close()

        bounded = handler.get_iv_history("SPY", start="2026-06-10",
                                         end="2026-06-10")

        assert len(bounded) == 1
        assert bounded["atm_iv"].iloc[0] == pytest.approx(0.21)
