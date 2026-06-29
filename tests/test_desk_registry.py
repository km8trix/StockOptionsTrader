"""Tests for desks.registry (contract C1): shapes, errors, construction."""

from __future__ import annotations

import re

import pytest

from desks.aqr import AqrDesk
from desks.base import Desk
from desks.citadel import CitadelDesk
from desks.foundation import FoundationDesk
from desks.janestreet import JaneStreetDesk
from desks.orchestrator import FundOrchestrator
from desks.registry import (_MODEL_SELECTABLE_DESKS, _PROMOTED_DESKS,
                            create_desk, create_fund_orchestrator, list_desks)
from desks.renaissance import RenaissanceDesk
from desks.twosigma import TwoSigmaDesk

EXPECTED_KEYS = {'key', 'name', 'firm_inspiration', 'description', 'status',
                 'activates_in_phase', 'accent', 'gate_status'}


class TestListDesks:
    def test_every_entry_has_the_contract_shape(self):
        desks = list_desks()
        assert desks  # non-empty
        for entry in desks:
            assert set(entry) == EXPECTED_KEYS
            assert isinstance(entry['key'], str) and entry['key']
            assert isinstance(entry['name'], str) and entry['name']
            assert isinstance(entry['firm_inspiration'], str)
            assert isinstance(entry['description'], str) and entry['description']
            assert entry['status'] in ('ready', 'planned')
            assert re.fullmatch(r'#[0-9a-f]{6}', entry['accent'])
            assert entry['gate_status'] == (
                'promoted' if entry['key'] in _PROMOTED_DESKS else 'research')
            if entry['status'] == 'planned':
                assert isinstance(entry['activates_in_phase'], int)
            else:
                assert entry['activates_in_phase'] is None

    def test_all_desks_present_with_plan_metadata(self):
        by_key = {entry['key']: entry for entry in list_desks()}
        assert set(by_key) == {'foundation', 'renaissance', 'citadel',
                               'janestreet', 'twosigma', 'aqr'}

        assert by_key['foundation']['status'] == 'ready'
        assert by_key['foundation']['accent'] == '#4493f8'
        assert by_key['foundation']['firm_inspiration'] == 'House'

        # Contract C6: renaissance is ready as of Phase 6; accent stays.
        assert by_key['renaissance']['status'] == 'ready'
        assert by_key['renaissance']['activates_in_phase'] is None
        assert by_key['renaissance']['accent'] == '#58a6ff'

        # Contract C10: citadel is ready as of Phase 7; accent stays.
        assert by_key['citadel']['status'] == 'ready'
        assert by_key['citadel']['activates_in_phase'] is None
        assert by_key['citadel']['accent'] == '#bc8cff'

        # Contract C15: janestreet is ready as of Phase 8; accent stays.
        assert by_key['janestreet']['status'] == 'ready'
        assert by_key['janestreet']['activates_in_phase'] is None
        assert by_key['janestreet']['accent'] == '#d29922'

        # Phase C: twosigma is ready; systematic cross-sectional L/S desk.
        assert by_key['twosigma']['status'] == 'ready'
        assert by_key['twosigma']['activates_in_phase'] is None
        assert by_key['twosigma']['accent'] == '#3fb950'
        assert by_key['twosigma']['firm_inspiration'] == 'Two Sigma'

        # Phase D: aqr is ready; the transparent classical-quant factor desk
        # (reuses the shared cross-sectional book, distinct accent).
        assert by_key['aqr']['status'] == 'ready'
        assert by_key['aqr']['activates_in_phase'] is None
        assert by_key['aqr']['accent'] == '#f0883e'
        assert by_key['aqr']['firm_inspiration'] == 'AQR Capital Management'


class TestCreateDesk:
    def test_creates_foundation_desk_with_default_allocation(self):
        desk = create_desk('foundation')
        assert isinstance(desk, FoundationDesk)
        assert isinstance(desk, Desk)
        assert desk.key == 'foundation'
        assert desk.capital_allocation == 1.0

    def test_capital_allocation_is_passed_through(self):
        desk = create_desk('foundation', capital_allocation=0.25)
        assert desk.capital_allocation == 0.25

    def test_creates_renaissance_desk(self):
        # Contract C6: create_desk('renaissance') returns the desk.
        desk = create_desk('renaissance')
        assert isinstance(desk, RenaissanceDesk)
        assert isinstance(desk, Desk)
        assert desk.key == 'renaissance'
        assert desk.accent == '#58a6ff'
        assert desk.capital_allocation == 1.0

    def test_renaissance_capital_allocation_is_passed_through(self):
        desk = create_desk('renaissance', capital_allocation=0.3)
        assert desk.capital_allocation == 0.3

    def test_creates_citadel_desk(self):
        # Contract C10: create_desk('citadel') returns the desk.
        desk = create_desk('citadel')
        assert isinstance(desk, CitadelDesk)
        assert isinstance(desk, Desk)
        assert desk.key == 'citadel'
        assert desk.accent == '#bc8cff'
        assert desk.capital_allocation == 1.0

    def test_citadel_capital_allocation_is_passed_through(self):
        desk = create_desk('citadel', capital_allocation=0.4)
        assert desk.capital_allocation == 0.4

    def test_unknown_key_raises_value_error(self):
        with pytest.raises(ValueError, match='Unknown desk: warrenbuffett'):
            create_desk('warrenbuffett')

    def test_creates_janestreet_desk(self):
        # Contract C15: create_desk('janestreet') returns the desk.
        desk = create_desk('janestreet')
        assert isinstance(desk, JaneStreetDesk)
        assert isinstance(desk, Desk)
        assert desk.key == 'janestreet'
        assert desk.accent == '#d29922'
        assert desk.capital_allocation == 1.0

    def test_janestreet_capital_allocation_is_passed_through(self):
        desk = create_desk('janestreet', capital_allocation=0.2)
        assert desk.capital_allocation == 0.2

    def test_creates_twosigma_desk(self):
        # Phase C: create_desk('twosigma') returns the desk.
        desk = create_desk('twosigma')
        assert isinstance(desk, TwoSigmaDesk)
        assert isinstance(desk, Desk)
        assert desk.key == 'twosigma'
        assert desk.accent == '#3fb950'
        assert desk.capital_allocation == 1.0

    def test_twosigma_capital_allocation_is_passed_through(self):
        desk = create_desk('twosigma', capital_allocation=0.35)
        assert desk.capital_allocation == 0.35

    def test_twosigma_model_key_threads_through(self):
        # twosigma is model-selectable: model_key reaches the factory and
        # selects the single controller (status reflects the chosen id).
        desk = create_desk('twosigma', model_key='lightgbm')
        assert isinstance(desk, TwoSigmaDesk)
        assert desk.get_status()['models'] == ['lightgbm']

    def test_foundation_model_key_threads_through(self):
        # Companion check: the OTHER model-selectable desk still accepts a
        # model_key (guards the _MODEL_SELECTABLE_DESKS set both ways).
        desk = create_desk('foundation', model_key='gbm')
        assert isinstance(desk, FoundationDesk)

    def test_model_key_on_non_selectable_desk_raises(self):
        # renaissance/citadel/janestreet do NOT support model selection.
        for key in ('renaissance', 'citadel', 'janestreet'):
            with pytest.raises(ValueError,
                               match='does not support model selection'):
                create_desk(key, model_key='gbm')

    def test_no_planned_desks_remain(self):
        # Phase 8 flipped the last planned desk; every entry is ready.
        assert all(entry['status'] == 'ready' for entry in list_desks())


class TestGateStatus:
    """``gate_status`` reflects ``_PROMOTED_DESKS`` membership — the single
    source of truth for which desks have cleared BOTH evidence gates (IC +
    OOS backtest). Empty until a desk earns it; promotion is one entry in that
    frozenset. 'promoted' is a research-quality status only and never means the
    desk trades live."""

    def test_promoted_keys_are_real_desks(self):
        # Guards against a typo'd or stale promotion key in _PROMOTED_DESKS.
        keys = {entry['key'] for entry in list_desks()}
        assert _PROMOTED_DESKS <= keys

    def test_gate_status_matches_membership(self):
        for entry in list_desks():
            expected = ('promoted' if entry['key'] in _PROMOTED_DESKS
                        else 'research')
            assert entry['gate_status'] == expected


class TestModelSelectableDeskConsistency:
    """The backtest route keeps a hand-maintained copy of the model-selectable
    desk set (``gui.routes.api_backtest.MODEL_SELECTABLE_DESKS``) so the
    desks package stays optional for the import. This pins the two copies as
    set-equal so the route's copy can't silently drift from the registry's
    source of truth (``desks.registry._MODEL_SELECTABLE_DESKS``)."""

    def test_route_copy_matches_registry_source_of_truth(self):
        from gui.routes.api_backtest import MODEL_SELECTABLE_DESKS
        assert set(MODEL_SELECTABLE_DESKS) == set(_MODEL_SELECTABLE_DESKS)

    def test_every_model_selectable_desk_actually_accepts_model_key(self):
        # Both directions: each id in the set must really thread model_key
        # through create_desk (a guard against an id being added to the set
        # for a desk whose factory rejects model selection).
        for key in _MODEL_SELECTABLE_DESKS:
            desk = create_desk(key, model_key='gbm')
            assert desk.key == key


class TestCreateFundOrchestrator:
    def test_builds_orchestrator_from_allocations(self):
        orch = create_fund_orchestrator(
            {'foundation': 0.5, 'renaissance': 0.5})
        assert isinstance(orch, FundOrchestrator)
        assert [d.key for d in orch.desks] == ['foundation', 'renaissance']
        assert all(d.capital_allocation == 0.5 for d in orch.desks)
        assert orch.active_capital == pytest.approx(1.0)

    def test_preserves_allocation_order(self):
        orch = create_fund_orchestrator(
            {'renaissance': 0.3, 'foundation': 0.2, 'citadel': 0.1})
        assert [d.key for d in orch.desks] == [
            'renaissance', 'foundation', 'citadel']

    def test_overallocation_raises(self):
        with pytest.raises(ValueError, match='must be <= 1.0'):
            create_fund_orchestrator({'foundation': 0.6, 'citadel': 0.6})

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='at least one'):
            create_fund_orchestrator({})

    def test_unknown_desk_raises(self):
        with pytest.raises(ValueError, match='Unknown desk'):
            create_fund_orchestrator({'foundation': 0.5, 'nope': 0.5})

    def test_risk_aggregator_is_wired(self):
        from portfolio.risk_aggregator import PortfolioRiskAggregator
        agg = PortfolioRiskAggregator()
        orch = create_fund_orchestrator({'foundation': 1.0},
                                        risk_aggregator=agg)
        assert orch.risk_aggregator is agg


class TestTurnoverEnabled:
    """Phase 5: the production cross-sectional desks ship turnover control on
    (mechanism + defaults stay no-op; the registry is the config point)."""

    # Backtest tuning R1: twosigma's turnover was tightened (exit 0.4, min hold
    # 21d) to damp the cost bleed from ~15 trades/day; aqr keeps the Phase-5
    # values. The registry stays the single config point.
    @pytest.mark.parametrize('key,exit_q,min_hold',
                             [('twosigma', 0.4, 21), ('aqr', 0.3, 3)])
    def test_cross_sectional_desks_enable_turnover(self, key, exit_q, min_hold):
        desk = create_desk(key)
        assert desk._exit_quantile == exit_q   # hysteresis band wider than entry
        assert desk.min_holding_days == min_hold

    def test_turnover_enabled_through_model_selection_path(self):
        # twosigma is model-selectable; turnover must apply on that branch too.
        desk = create_desk('twosigma', model_key='gbm')
        assert desk._exit_quantile == 0.4
        assert desk.min_holding_days == 21

    def test_non_turnover_desk_keeps_defaults(self):
        # A desk without a turnover spec constructs exactly as before.
        desk = create_desk('foundation')
        assert getattr(desk, 'min_holding_days', 0) == 0


class TestRobustPodSharpeEnabled:
    """Phase 5: production Citadel scores its pods by median / MAD (the
    registry is the config point; the desk-class default stays mean/std)."""

    def test_citadel_enables_robust_pod_sharpe(self):
        desk = create_desk('citadel')
        assert desk.robust_pod_sharpe is True
        assert desk.get_status()['robust_pod_sharpe'] is True

    def test_default_citadel_desk_keeps_mean_std(self):
        # Constructing the desk class directly is unchanged (byte-identical).
        assert CitadelDesk().robust_pod_sharpe is False
        assert CitadelDesk().get_status()['robust_pod_sharpe'] is False

    def test_config_spread_preserves_capital_allocation(self):
        # The new 'config' kwargs merge must not disturb capital_allocation.
        desk = create_desk('citadel', capital_allocation=0.25)
        assert desk.capital_allocation == 0.25
        assert desk.robust_pod_sharpe is True


class TestSignalStrengthEnabled:
    """Phase 5: the production cross-sectional desks ship signal-strength
    (conviction) sizing on (mechanism + desk-class defaults stay equal-weight;
    the registry is the config point)."""

    @pytest.mark.parametrize('key', ['twosigma', 'aqr'])
    def test_cross_sectional_desks_enable_signal_strength(self, key):
        assert create_desk(key).size_by_signal_strength is True

    def test_enabled_through_the_model_selection_path(self):
        # twosigma is model-selectable; the config must apply on that branch
        # too (create_desk merges turnover + config on BOTH paths).
        desk = create_desk('twosigma', model_key='gbm')
        assert desk.size_by_signal_strength is True
        assert desk._exit_quantile == 0.4  # turnover still applied alongside (R1)

    def test_desk_class_defaults_stay_equal_weight(self):
        assert TwoSigmaDesk().size_by_signal_strength is False
        assert AqrDesk().size_by_signal_strength is False


class TestAmericanPricingEnabled:
    """Phase 5: production Jane Street prices its option legs American (the
    registry is the config point; the desk-class default stays european)."""

    def test_janestreet_enables_american(self):
        assert create_desk('janestreet').exercise_style == 'american'

    def test_desk_class_default_stays_european(self):
        assert JaneStreetDesk().exercise_style == 'european'

    def test_capital_allocation_still_threads(self):
        desk = create_desk('janestreet', capital_allocation=0.2)
        assert desk.capital_allocation == 0.2
        assert desk.exercise_style == 'american'


class TestEarningsGateEnabled:
    """Phase 9: production Renaissance gates new single-name entries near
    earnings, with an OFFLINE EarningsCache the registry attaches lazily. The
    desk-class default stays off, and an un-ingested cache is a no-op."""

    def test_renaissance_enables_gate_with_offline_calendar(self, monkeypatch, tmp_path):
        from data.earnings_cache import EarningsCache
        # Isolate the cache db so the read can't pick up a real earnings_data.db.
        monkeypatch.setenv('EARNINGS_DB_PATH', str(tmp_path / 'earnings_data.db'))
        desk = create_desk('renaissance')
        assert desk.avoid_earnings_entries is True
        assert isinstance(desk.earnings_calendar, EarningsCache)
        # Un-ingested cache => no-op (no file even created on read).
        assert desk.earnings_calendar.next_earnings('AAPL', '2026-01-01') is None
        assert desk._earnings_imminent('AAPL', '2026-01-01') is False

    def test_desk_class_default_stays_off(self):
        assert RenaissanceDesk().avoid_earnings_entries is False
        assert RenaissanceDesk().earnings_calendar is None

    def test_capital_allocation_still_threads(self, monkeypatch, tmp_path):
        monkeypatch.setenv('EARNINGS_DB_PATH', str(tmp_path / 'earnings_data.db'))
        desk = create_desk('renaissance', capital_allocation=0.3)
        assert desk.capital_allocation == 0.3
        assert desk.avoid_earnings_entries is True

    def test_construction_creates_no_db_file(self, monkeypatch, tmp_path):
        # Building the desk (and its cache) must not touch disk until ingest.
        db = tmp_path / 'earnings_data.db'
        monkeypatch.setenv('EARNINGS_DB_PATH', str(db))
        create_desk('renaissance')
        assert not db.exists()
