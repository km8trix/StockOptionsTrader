"""Audit log (C18) + kill switch (C17): append-only hash chain, tamper
detection, persistence across instances, audited flips."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from utils.audit import AuditLog
from utils.kill_switch import KillSwitch, KillSwitchEngaged  # noqa: F401


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "audit.db")


class TestAuditLog:
    def test_append_returns_sequential_seq(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        assert audit.append("tester", "event_a", {"x": 1}) == 1
        assert audit.append("tester", "event_b", {"x": 2}) == 2
        assert audit.append("tester", "event_a", {"x": 3}) == 3

    def test_entries_shape_and_filtering(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        audit.append("alice", "order_placed", {"order_id": 7})
        audit.append("bob", "auth_renewed", {})
        entries = audit.entries(limit=10, ascending=True)
        assert [e["seq"] for e in entries] == [1, 2]
        first = entries[0]
        assert set(first) == {"seq", "ts", "env", "actor", "event_type",
                              "payload"}
        assert first["actor"] == "alice"
        assert first["env"] == "sandbox"
        assert first["payload"] == {"order_id": 7}
        only_auth = audit.entries(event_type="auth_renewed")
        assert len(only_auth) == 1 and only_auth[0]["actor"] == "bob"

    def test_entries_default_newest_first_with_offset(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        for i in range(5):
            audit.append("t", "e", {"i": i})
        page = audit.entries(limit=2, offset=1)
        assert [e["seq"] for e in page] == [4, 3]

    def test_no_public_mutation_api(self, db_path):
        """APPEND-ONLY: the class exposes no update/delete surface."""
        audit = AuditLog(db_path)
        public = [name for name in dir(audit) if not name.startswith("_")]
        for forbidden in ("update", "delete", "remove", "truncate", "clear"):
            assert not any(forbidden in name.lower() for name in public), \
                f"audit log must not expose a {forbidden} API"

    def test_verify_chain_clean(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        for i in range(10):
            audit.append("t", "e", {"i": i})
        assert audit.verify_chain() == {"ok": True, "first_bad_seq": None}

    def test_verify_chain_empty_log_ok(self, db_path):
        audit = AuditLog(db_path)
        assert audit.verify_chain() == {"ok": True, "first_bad_seq": None}

    def test_tampering_payload_pinpoints_first_bad_seq(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        for i in range(6):
            audit.append("t", "e", {"i": i})
        # Tamper behind the API's back: rewrite row 4's payload raw.
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE audit_log SET payload = '{\"i\":999}' WHERE seq = 4")
        conn.commit()
        conn.close()
        assert audit.verify_chain() == {"ok": False, "first_bad_seq": 4}

    def test_deleting_a_row_breaks_the_chain(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        for i in range(4):
            audit.append("t", "e", {"i": i})
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM audit_log WHERE seq = 2")
        conn.commit()
        conn.close()
        result = audit.verify_chain()
        assert result["ok"] is False
        assert result["first_bad_seq"] == 3  # row 3 no longer chains to 1


class TestKillSwitch:
    def test_starts_disengaged(self, db_path):
        switch = KillSwitch(db_path)
        assert switch.engaged() is False

    def test_engage_disengage_roundtrip(self, db_path):
        switch = KillSwitch(db_path)
        switch.engage("manual halt", actor="spencer")
        assert switch.engaged() is True
        state = switch.state()
        assert state["engaged"] is True
        assert state["reason"] == "manual halt"
        assert state["actor"] == "spencer"
        switch.disengage(actor="spencer")
        assert switch.engaged() is False

    def test_persists_across_instances(self, db_path):
        KillSwitch(db_path).engage("drawdown", actor="circuit_breaker")
        # A brand-new instance on the same database sees the engagement —
        # a kill switch that resets on restart is not a kill switch.
        assert KillSwitch(db_path).engaged() is True

    def test_flips_are_audited(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        switch.engage("bad fill", actor="ops")
        switch.disengage(actor="ops")
        events = [e["event_type"] for e in audit.entries(ascending=True)]
        assert events == ["kill_switch_engaged", "kill_switch_disengaged"]
        engaged_row = audit.entries(event_type="kill_switch_engaged")[0]
        assert engaged_row["payload"] == {"reason": "bad fill"}
        assert engaged_row["actor"] == "ops"

    def test_redundant_flips_are_noops(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        switch.engage("first", actor="ops")
        switch.engage("second", actor="ops")  # no-op: already engaged
        switch.disengage(actor="ops")
        switch.disengage(actor="ops")  # no-op: already clear
        assert len(audit.entries(limit=50)) == 2
        assert switch.state()["reason"] is None  # cleared on disengage

    def test_concurrent_engages_audit_exactly_once(self, db_path):
        """The flip is a single atomic update-where-changed, not a
        read-then-write across two connections: simultaneous engage()
        calls resolve to exactly ONE winner and one audit row."""
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        errors: list = []

        def worker(i):
            try:
                barrier.wait()
                switch.engage(f"race-{i}", actor=f"worker-{i}")
            except Exception as e:  # noqa: BLE001 - surfaced via assert
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert switch.engaged() is True
        engaged_rows = audit.entries(event_type="kill_switch_engaged",
                                     limit=50)
        assert len(engaged_rows) == 1  # one winner, one audit row
