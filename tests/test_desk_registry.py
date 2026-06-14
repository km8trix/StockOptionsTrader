"""Tests for desks.registry (contract C1): shapes, errors, construction."""

from __future__ import annotations

import re

import pytest

from desks.base import Desk
from desks.citadel import CitadelDesk
from desks.foundation import FoundationDesk
from desks.janestreet import JaneStreetDesk
from desks.orchestrator import FundOrchestrator
from desks.registry import (create_desk, create_fund_orchestrator,
                            list_desks)
from desks.renaissance import RenaissanceDesk

EXPECTED_KEYS = {'key', 'name', 'firm_inspiration', 'description', 'status',
                 'activates_in_phase', 'accent'}


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
            if entry['status'] == 'planned':
                assert isinstance(entry['activates_in_phase'], int)
            else:
                assert entry['activates_in_phase'] is None

    def test_all_four_desks_present_with_plan_metadata(self):
        by_key = {entry['key']: entry for entry in list_desks()}
        assert set(by_key) == {'foundation', 'renaissance', 'citadel',
                               'janestreet'}

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
        # All four desks are now live.
        assert by_key['janestreet']['status'] == 'ready'
        assert by_key['janestreet']['activates_in_phase'] is None
        assert by_key['janestreet']['accent'] == '#d29922'


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

    def test_no_planned_desks_remain(self):
        # Phase 8 flipped the last planned desk; every entry is ready.
        assert all(entry['status'] == 'ready' for entry in list_desks())


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
