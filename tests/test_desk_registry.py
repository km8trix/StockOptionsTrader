"""Tests for desks.registry (contract C1): shapes, errors, construction."""

from __future__ import annotations

import re

import pytest

from desks.base import Desk
from desks.foundation import FoundationDesk
from desks.registry import create_desk, list_desks

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

        assert by_key['renaissance']['status'] == 'planned'
        assert by_key['renaissance']['activates_in_phase'] == 6
        assert by_key['renaissance']['accent'] == '#58a6ff'

        assert by_key['citadel']['activates_in_phase'] == 7
        assert by_key['citadel']['accent'] == '#bc8cff'

        assert by_key['janestreet']['activates_in_phase'] == 8
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

    def test_unknown_key_raises_value_error(self):
        with pytest.raises(ValueError, match='Unknown desk: warrenbuffett'):
            create_desk('warrenbuffett')

    @pytest.mark.parametrize('key,phase', [('renaissance', 6),
                                           ('citadel', 7),
                                           ('janestreet', 8)])
    def test_planned_desks_raise_with_their_phase(self, key, phase):
        with pytest.raises(ValueError,
                           match=f"Desk '{key}' activates in Phase {phase}"):
            create_desk(key)
