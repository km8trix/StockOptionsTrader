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

    def test_audit_failure_rolls_back_state_and_partial_audit(self, db_path):
        """A failure after the audit insert still rolls back both writes."""
        class FailingAudit(AuditLog):
            def _append_in_transaction(self, conn, actor, event_type,
                                       payload, timestamp=None):
                super()._append_in_transaction(
                    conn, actor, event_type, payload, timestamp=timestamp)
                raise RuntimeError("injected audit failure")

        audit = FailingAudit(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)

        with pytest.raises(RuntimeError, match="injected audit failure"):
            switch.engage("must roll back", actor="ops")

        assert switch.engaged() is False
        assert audit.entries(limit=10) == []
        assert audit.verify_chain() == {"ok": True, "first_bad_seq": None}

    def test_disengage_audit_failure_leaves_switch_engaged(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        switch.engage("existing halt", actor="ops")

        original_append = audit._append_in_transaction

        def fail_after_insert(conn, actor, event_type, payload,
                              timestamp=None):
            original_append(
                conn, actor, event_type, payload, timestamp=timestamp)
            raise RuntimeError("injected audit failure")

        audit._append_in_transaction = fail_after_insert
        with pytest.raises(RuntimeError, match="injected audit failure"):
            switch.disengage(actor="ops")

        assert switch.engaged() is True
        assert [entry["event_type"] for entry in audit.entries()] == [
            "kill_switch_engaged"]
        assert audit.verify_chain() == {"ok": True, "first_bad_seq": None}

    def test_rejects_audit_log_on_different_database(self, db_path, tmp_path):
        other_audit = AuditLog(str(tmp_path / "other.db"))
        with pytest.raises(ValueError, match="same database"):
            KillSwitch(db_path, audit=other_audit)

    @pytest.mark.parametrize("unsupported_path", [
        ":memory:",
        "file:shared?mode=memory&cache=shared",
    ])
    def test_rejects_non_filesystem_database(self, unsupported_path):
        with pytest.raises(ValueError, match="filesystem-backed"):
            KillSwitch(unsupported_path)

    def test_flip_state_and_audit_share_one_timestamp(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)

        switch.engage("timestamp invariant", actor="ops")

        event = audit.entries(event_type="kill_switch_engaged")[0]
        assert switch.state()["flipped_at"] == event["ts"]

    def test_noop_flip_does_not_call_audit_clock(self, db_path):
        clock_calls = 0

        def clock():
            nonlocal clock_calls
            clock_calls += 1
            from datetime import datetime, timezone
            return datetime.now(timezone.utc)

        audit = AuditLog(db_path, env="sandbox", clock=clock)
        switch = KillSwitch(db_path, audit=audit)
        switch.disengage(actor="ops")
        assert clock_calls == 0

        switch.engage("first", actor="ops")
        assert clock_calls == 1
        switch.engage("no-op", actor="ops")
        assert clock_calls == 1

    def test_database_trigger_audit_failure_rolls_back_state(self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TRIGGER reject_audit_insert
            BEFORE INSERT ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'injected audit trigger failure');
            END
        """)
        conn.commit()
        conn.close()

        with pytest.raises(sqlite3.IntegrityError,
                           match="injected audit trigger failure"):
            switch.engage("must roll back", actor="ops")

        assert switch.engaged() is False
        assert audit.entries(limit=10) == []

    def test_actor_and_event_are_hashed_as_stored_text(self, db_path):
        audit = AuditLog(db_path, env=789)
        audit.append(123, 456, {})
        event = audit.entries()[0]
        assert event["env"] == "789"
        assert event["actor"] == "123"
        assert event["event_type"] == "456"
        assert audit.verify_chain() == {"ok": True, "first_bad_seq": None}

    def test_mixed_concurrent_flips_keep_state_and_chain_consistent(
            self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        n_threads = 10
        n_flips = 20
        barrier = threading.Barrier(n_threads)
        errors: list = []

        def worker(index):
            try:
                barrier.wait()
                for iteration in range(n_flips):
                    if (index + iteration) % 2:
                        switch.engage(
                            f"race-{index}-{iteration}",
                            actor=f"worker-{index}")
                    else:
                        switch.disengage(actor=f"worker-{index}")
            except Exception as exc:  # noqa: BLE001 - surfaced via assert
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert audit.verify_chain() == {"ok": True, "first_bad_seq": None}
        events = [entry["event_type"]
                  for entry in audit.entries(limit=1000, ascending=True)]
        assert events
        assert all(previous != current
                   for previous, current in zip(events, events[1:]))
        expected_engaged = events[-1] == "kill_switch_engaged"
        assert switch.engaged() is expected_engaged

    def test_concurrent_external_appends_and_flips_keep_chain_valid(
            self, db_path):
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        errors: list = []

        def worker(index):
            try:
                barrier.wait()
                for iteration in range(12):
                    if index % 2:
                        audit.append(
                            f"auditor-{index}", "heartbeat",
                            {"iteration": iteration})
                    elif (index + iteration) % 4:
                        switch.engage(
                            f"race-{index}-{iteration}",
                            actor=f"worker-{index}")
                    else:
                        switch.disengage(actor=f"worker-{index}")
            except Exception as exc:  # noqa: BLE001 - surfaced via assert
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(audit.entries(event_type="heartbeat", limit=1000)) == 48
        assert audit.verify_chain() == {"ok": True, "first_bad_seq": None}
