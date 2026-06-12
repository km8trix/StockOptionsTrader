"""EtradeAuthManager (contract C16): 3-legged flow on a fake transport,
renewal, midnight-ET hard expiry (DST + standard time + the 23:59->00:01
boundary), oauth_problem mapping, production ack gate, unconfigured, and
the ETRADE_ALLOW_NETWORK opt-in gate on the default (real) factory."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from brokers.etrade_auth import (ACCESS_TOKEN_URL, ALLOW_NETWORK_ENV_VAR,
                                 RENEW_TOKEN_URL, REQUEST_TOKEN_URL,
                                 EtradeAuthExpired, EtradeAuthManager,
                                 EtradeNotConfigured, EtradeNotConnected,
                                 default_session_factory)
from utils.audit import AuditLog

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class FakeResponse:
    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class FakeTransport:
    """Routes GET urls to canned responses; records every request."""

    def __init__(self, routes, log):
        self.routes = routes
        self.log = log

    def get(self, url, params=None, json=None):
        self.log.append(("GET", url))
        for prefix, response in self.routes.items():
            if url.startswith(prefix):
                return response() if callable(response) else response
        raise AssertionError(f"unrouted url {url}")


class FakeFactory:
    """Injected session factory: records construction args (so tests can
    assert the request token + verifier are threaded correctly)."""

    def __init__(self):
        self.routes = {}
        self.request_log = []
        self.constructions = []

    def __call__(self, consumer_key, consumer_secret,
                 resource_owner_key=None, resource_owner_secret=None,
                 verifier=None):
        self.constructions.append({
            "consumer_key": consumer_key,
            "resource_owner_key": resource_owner_key,
            "resource_owner_secret": resource_owner_secret,
            "verifier": verifier,
        })
        return FakeTransport(self.routes, self.request_log)


@pytest.fixture
def sandbox_env(monkeypatch):
    monkeypatch.setenv("ETRADE_CONSUMER_KEY", "ck-test")
    monkeypatch.setenv("ETRADE_CONSUMER_SECRET", "cs-test")
    monkeypatch.setenv("ETRADE_ENV", "sandbox")
    monkeypatch.delenv("ETRADE_PRODUCTION_ACK", raising=False)
    # The network opt-in gate stays OFF in the suite: every test goes
    # through injected fake factories, which never consult the gate.
    monkeypatch.delenv(ALLOW_NETWORK_ENV_VAR, raising=False)


class Clock:
    """Mutable frozen clock."""

    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_manager(tmp_path, clock=None):
    factory = FakeFactory()
    factory.routes[REQUEST_TOKEN_URL] = FakeResponse(
        200, "oauth_token=req-tok&oauth_token_secret=req-sec")
    factory.routes[ACCESS_TOKEN_URL] = FakeResponse(
        200, "oauth_token=acc-tok&oauth_token_secret=acc-sec")
    factory.routes[RENEW_TOKEN_URL] = FakeResponse(
        200, "Access Token has been renewed")
    db_path = str(tmp_path / "auth.db")
    manager = EtradeAuthManager(db_path=db_path, session_factory=factory,
                                clock=clock)
    return manager, factory


def connect(manager):
    manager.start_auth()
    return manager.submit_verifier("VERIF")


class TestThreeLeggedFlow:
    def test_happy_path(self, sandbox_env, tmp_path):
        clock = Clock(datetime(2026, 6, 12, 14, 0, tzinfo=UTC))
        manager, factory = make_manager(tmp_path, clock)
        assert manager.status()["state"] == "disconnected"

        authorize_url = manager.start_auth()
        assert authorize_url == ("https://us.etrade.com/e/t/etws/authorize"
                                 "?key=ck-test&token=req-tok")
        status = manager.status()
        assert status["state"] == "pending_verifier"
        assert status["authorize_url"] == authorize_url

        status = manager.submit_verifier("VERIF")
        assert status["state"] == "connected"
        assert status["env"] == "sandbox"
        assert status["token_issued_at"] == clock.now.isoformat()
        assert status["renewed_at"] is None
        # The verifier leg was built ON the request token with the code.
        leg3 = factory.constructions[-1]
        assert leg3["resource_owner_key"] == "req-tok"
        assert leg3["resource_owner_secret"] == "req-sec"
        assert leg3["verifier"] == "VERIF"

    def test_auth_events_audited_without_secrets(self, sandbox_env,
                                                 tmp_path):
        manager, _ = make_manager(tmp_path)
        connect(manager)
        manager.renew()
        manager.disconnect()
        audit = AuditLog(manager.db_path)
        rows = audit.entries(ascending=True)
        assert [r["event_type"] for r in rows] == [
            "auth_started", "auth_connected", "auth_renewed",
            "auth_disconnected"]
        # SECRETS NEVER IN PAYLOADS.
        for row in rows:
            blob = str(row["payload"])
            for secret in ("acc-tok", "acc-sec", "req-sec", "cs-test"):
                assert secret not in blob

    def test_submit_verifier_without_start_raises(self, sandbox_env,
                                                  tmp_path):
        manager, _ = make_manager(tmp_path)
        with pytest.raises(Exception, match="start_auth"):
            manager.submit_verifier("X")

    def test_tokens_persist_across_instances(self, sandbox_env, tmp_path):
        manager, factory = make_manager(tmp_path)
        connect(manager)
        clone = EtradeAuthManager(db_path=manager.db_path,
                                  session_factory=factory)
        assert clone.status()["state"] == "connected"

    def test_disconnect(self, sandbox_env, tmp_path):
        manager, _ = make_manager(tmp_path)
        connect(manager)
        manager.disconnect()
        assert manager.status()["state"] == "disconnected"
        with pytest.raises(EtradeNotConnected):
            manager.get_session()


class TestRenewal:
    def test_renew_success_sets_renewed_at(self, sandbox_env, tmp_path):
        clock = Clock(datetime(2026, 6, 12, 14, 0, tzinfo=UTC))
        manager, factory = make_manager(tmp_path, clock)
        connect(manager)
        clock.now = datetime(2026, 6, 12, 16, 0, tzinfo=UTC)
        assert manager.renew() is True
        status = manager.status()
        assert status["state"] == "connected"
        assert status["renewed_at"] == clock.now.isoformat()
        assert ("GET", RENEW_TOKEN_URL) in factory.request_log

    def test_renew_with_no_token_returns_false(self, sandbox_env, tmp_path):
        manager, _ = make_manager(tmp_path)
        assert manager.renew() is False

    def test_renew_401_oauth_problem_expires_token(self, sandbox_env,
                                                   tmp_path):
        manager, factory = make_manager(tmp_path)
        connect(manager)
        factory.routes[RENEW_TOKEN_URL] = FakeResponse(
            401, "oauth_problem=token_rejected")
        assert manager.renew() is False
        assert manager.status()["state"] == "expired"
        # The expired token maps to the typed exception on use.
        with pytest.raises(EtradeAuthExpired):
            manager.get_session()


class TestMidnightEasternExpiry:
    """Tokens die at midnight US/Eastern — DST and standard time alike."""

    def _connected_at(self, tmp_path, issued_et: datetime):
        clock = Clock(issued_et.astimezone(UTC))
        manager, _ = make_manager(tmp_path, clock)
        connect(manager)
        return manager, clock

    def test_july_dst_boundary_2359_to_0001(self, sandbox_env, tmp_path):
        # EDT (UTC-4): issued 23:59 ET July 10 — alive...
        manager, clock = self._connected_at(
            tmp_path, datetime(2026, 7, 10, 23, 59, tzinfo=ET))
        assert manager.status()["state"] == "connected"
        # ...dead two minutes later at 00:01 ET July 11.
        clock.now = datetime(2026, 7, 11, 0, 1, tzinfo=ET).astimezone(UTC)
        assert manager.status()["state"] == "expired"
        with pytest.raises(EtradeAuthExpired):
            manager.get_session()

    def test_january_standard_time_boundary(self, sandbox_env, tmp_path):
        # EST (UTC-5): same midnight rule in standard time.
        manager, clock = self._connected_at(
            tmp_path, datetime(2026, 1, 9, 23, 59, tzinfo=ET))
        assert manager.status()["state"] == "connected"
        clock.now = datetime(2026, 1, 10, 0, 1, tzinfo=ET).astimezone(UTC)
        assert manager.status()["state"] == "expired"

    def test_utc_date_roll_does_not_expire_et_day(self, sandbox_env,
                                                  tmp_path):
        # 19:00 ET Jan 9 == 00:00 UTC Jan 10: UTC's date rolled, ET's did
        # NOT — the token must survive (the rule is EASTERN midnight).
        manager, clock = self._connected_at(
            tmp_path, datetime(2026, 1, 9, 10, 0, tzinfo=ET))
        clock.now = datetime(2026, 1, 9, 20, 0, tzinfo=ET).astimezone(UTC)
        assert clock.now.date() != datetime(
            2026, 1, 9, tzinfo=ET).date()  # sanity: UTC rolled
        assert manager.status()["state"] == "connected"

    def test_renew_on_expired_token_returns_false_and_audits(
            self, sandbox_env, tmp_path):
        manager, clock = self._connected_at(
            tmp_path, datetime(2026, 7, 10, 12, 0, tzinfo=ET))
        clock.now = datetime(2026, 7, 11, 9, 30, tzinfo=ET).astimezone(UTC)
        assert manager.renew() is False
        audit = AuditLog(manager.db_path)
        expired = audit.entries(event_type="auth_expired")
        assert len(expired) == 1
        assert expired[0]["payload"]["reason"] == "midnight_et_hard_expiry"


class TestConfigurationGates:
    def test_unconfigured_state_and_typed_raise(self, monkeypatch,
                                                tmp_path):
        monkeypatch.delenv("ETRADE_CONSUMER_KEY", raising=False)
        monkeypatch.delenv("ETRADE_CONSUMER_SECRET", raising=False)
        monkeypatch.setenv("ETRADE_ENV", "sandbox")
        manager = EtradeAuthManager(db_path=str(tmp_path / "a.db"),
                                    session_factory=FakeFactory())
        assert manager.status()["state"] == "unconfigured"
        with pytest.raises(EtradeNotConfigured):
            manager.start_auth()
        with pytest.raises(EtradeNotConfigured):
            manager.get_session()

    def test_production_requires_ack(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ETRADE_CONSUMER_KEY", "ck")
        monkeypatch.setenv("ETRADE_CONSUMER_SECRET", "cs")
        monkeypatch.setenv("ETRADE_ENV", "production")
        monkeypatch.delenv("ETRADE_PRODUCTION_ACK", raising=False)
        with pytest.raises(EtradeNotConfigured,
                           match="ETRADE_PRODUCTION_ACK"):
            EtradeAuthManager(db_path=str(tmp_path / "a.db"),
                              session_factory=FakeFactory())

    def test_production_with_ack_constructs(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ETRADE_CONSUMER_KEY", "ck")
        monkeypatch.setenv("ETRADE_CONSUMER_SECRET", "cs")
        monkeypatch.setenv("ETRADE_ENV", "production")
        monkeypatch.setenv("ETRADE_PRODUCTION_ACK",
                           "I_UNDERSTAND_LIVE_TRADING")
        manager = EtradeAuthManager(db_path=str(tmp_path / "a.db"),
                                    session_factory=FakeFactory())
        assert manager.env == "production"
        assert manager.api_base_url == "https://api.etrade.com"

    def test_sandbox_api_base(self, sandbox_env, tmp_path):
        manager, _ = make_manager(tmp_path)
        assert manager.api_base_url == "https://apisb.etrade.com"


class TestNetworkOptInGate:
    """Defense-in-depth: consumer keys in the environment alone must
    NEVER produce a real transport. The default factory refuses to
    construct without ETRADE_ALLOW_NETWORK=1 — so a stray POST to an
    auth route gets a typed error, not a signed request to
    api.etrade.com."""

    def test_default_factory_refuses_without_gate(self, monkeypatch):
        monkeypatch.delenv(ALLOW_NETWORK_ENV_VAR, raising=False)
        with pytest.raises(EtradeNotConfigured,
                           match="ETRADE_ALLOW_NETWORK"):
            default_session_factory("ck", "cs")

    def test_default_factory_refuses_on_wrong_value(self, monkeypatch):
        for wrong in ("0", "true", "yes", ""):
            monkeypatch.setenv(ALLOW_NETWORK_ENV_VAR, wrong)
            with pytest.raises(EtradeNotConfigured,
                               match="ETRADE_ALLOW_NETWORK"):
                default_session_factory("ck", "cs")

    def test_start_auth_with_default_factory_blocked_and_unaudited(
            self, sandbox_env, tmp_path):
        """The QA reproduction: keys configured, default factory (the
        GUI singleton's construction), POST-equivalent start_auth() —
        must raise typed BEFORE any transport exists. No auth_started
        row may appear: the flow never began."""
        manager = EtradeAuthManager(db_path=str(tmp_path / "gate.db"))
        with pytest.raises(EtradeNotConfigured,
                           match="ETRADE_ALLOW_NETWORK"):
            manager.start_auth()
        audit = AuditLog(manager.db_path)
        assert audit.entries(event_type="auth_started") == []
        assert manager.status()["state"] == "disconnected"  # not pending

    def test_gate_set_builds_real_session_without_network(
            self, sandbox_env, monkeypatch):
        """With the gate deliberately set, the factory constructs an
        OAuth1Session. Constructing one performs no I/O — the suite's
        socket hard-block proves it by not firing."""
        monkeypatch.setenv(ALLOW_NETWORK_ENV_VAR, "1")
        from requests_oauthlib import OAuth1Session
        session = default_session_factory("ck", "cs",
                                          resource_owner_key="tok",
                                          resource_owner_secret="sec")
        assert isinstance(session, OAuth1Session)

    def test_injected_fake_factory_never_consults_gate(self, sandbox_env,
                                                       tmp_path):
        """Tests and offline rehearsal inject fakes; the gate guards the
        REAL factory only. Full 3-legged flow works with the gate off."""
        manager, _ = make_manager(tmp_path)
        connect(manager)
        assert manager.status()["state"] == "connected"
