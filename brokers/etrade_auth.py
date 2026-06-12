"""
E*TRADE OAuth 1.0a lifecycle manager (contract C16).

THE PHASE-9 RULE: this module never opens a connection on its own. Every
HTTP interaction goes through an injected ``session_factory``; the default
factory builds a requests_oauthlib.OAuth1Session, but tests inject fakes
and the suite runs with sockets hard-blocked. The first real sandbox
contact is a human decision made AFTER this phase ships — and that is
ENFORCED IN CODE, not by convention: the default factory refuses to
construct a real transport unless the explicit opt-in gate
ETRADE_ALLOW_NETWORK=1 is set (mirroring the ETRADE_PRODUCTION_ACK
pattern), so consumer keys in the environment alone can NEVER produce a
network request — not from the GUI, not from a stray curl to an auth
route, not from a monitoring probe.

Flow (E*TRADE's documented 3-legged dance — note the OAuth endpoints live
on api.etrade.com for BOTH environments; only the resource API base
differs, apisb.etrade.com vs api.etrade.com):

    GET  /oauth/request_token   (oauth_callback="oob")
    user visits https://us.etrade.com/e/t/etws/authorize?key=..&token=..
    GET  /oauth/access_token    (oauth_verifier=<code the user pastes>)
    GET  /oauth/renew_access_token   to clear the 2h idle timeout

TOKEN LIFETIME (both rules enforced here):
  * idle timeout: tokens go inactive after ~2h idle; renew() clears it.
    The live session calls renew() opportunistically.
  * MIDNIGHT-ET HARD EXPIRY: tokens die at midnight US/Eastern every day
    regardless of renewal. Detected by comparing the token_issued_at
    instant's ET calendar date against now's ET calendar date via
    zoneinfo("America/New_York") — DST-correct by construction. The clock
    is injectable so tests freeze July (EDT) and January (EST) dates.

Tokens persist in SQLite (table etrade_tokens in TRADING_DB_PATH). The
secrets are stored locally only — they are NEVER logged and NEVER placed
in audit payloads; audit rows carry event metadata (env, reason) only.

Config (read from os.environ only — never from the .env file directly):
    ETRADE_SANDBOX_CONSUMER_KEY / ETRADE_SANDBOX_CONSUMER_SECRET
        consumer credentials used when ETRADE_ENV=sandbox (preferred)
    ETRADE_PROD_CONSUMER_KEY / ETRADE_PROD_CONSUMER_SECRET
        consumer credentials used when ETRADE_ENV=production (preferred)
    ETRADE_CONSUMER_KEY / ETRADE_CONSUMER_SECRET
        legacy generic fallback for either env when the env-scoped
        names are absent
    ETRADE_ENV                                     'sandbox' (default) |
                                                   'production'
    ETRADE_PRODUCTION_ACK                          must equal
        'I_UNDERSTAND_LIVE_TRADING' or production construction refuses.
    ETRADE_ALLOW_NETWORK                           must equal '1' or the
        DEFAULT session factory refuses to build a real transport
        (injected fakes are unaffected). The deliberate human act that
        authorizes first sandbox contact is setting this gate.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

from utils.audit import AuditLog

logger = logging.getLogger(__name__)

#: OAuth endpoints — SAME host for sandbox and production (documented).
OAUTH_BASE_URL = "https://api.etrade.com"
REQUEST_TOKEN_URL = f"{OAUTH_BASE_URL}/oauth/request_token"
ACCESS_TOKEN_URL = f"{OAUTH_BASE_URL}/oauth/access_token"
RENEW_TOKEN_URL = f"{OAUTH_BASE_URL}/oauth/renew_access_token"
AUTHORIZE_URL_TEMPLATE = ("https://us.etrade.com/e/t/etws/authorize"
                          "?key={key}&token={token}")

#: Resource API bases, selected by env (used by EtradeClient).
API_BASE_URLS = {
    "sandbox": "https://apisb.etrade.com",
    "production": "https://api.etrade.com",
}

PRODUCTION_ACK_VALUE = "I_UNDERSTAND_LIVE_TRADING"

#: Defense-in-depth network gate (mirrors the ETRADE_PRODUCTION_ACK
#: pattern): the DEFAULT (real) session factory refuses to construct
#: unless ETRADE_ALLOW_NETWORK=1 is set in the environment. Keys present
#: in .env therefore never suffice to reach api.etrade.com — first
#: contact requires a deliberate human act in BOTH the launcher and the
#: environment.
ALLOW_NETWORK_ENV_VAR = "ETRADE_ALLOW_NETWORK"
ALLOW_NETWORK_VALUE = "1"

ET_ZONE = ZoneInfo("America/New_York")


class EtradeAuthError(RuntimeError):
    """Base class for E*TRADE auth failures."""


class EtradeNotConfigured(EtradeAuthError):
    """Consumer keys absent, or the production ack gate not satisfied."""


class EtradeNotConnected(EtradeAuthError):
    """No active access token (disconnected / authorization incomplete)."""


class EtradeAuthExpired(EtradeAuthError):
    """Access token expired (midnight-ET hard expiry or oauth_problem)."""


def default_session_factory(consumer_key: str, consumer_secret: str,
                            resource_owner_key: Optional[str] = None,
                            resource_owner_secret: Optional[str] = None,
                            verifier: Optional[str] = None):
    """Build a signing requests_oauthlib.OAuth1Session.

    Imported lazily so an environment without requests-oauthlib can still
    import this module (the GUI's guarded-import precedent). This is the
    ONLY place a real transport is created, and the explicit human
    go-ahead is ENFORCED here: without ETRADE_ALLOW_NETWORK=1 in the
    environment this factory raises EtradeNotConfigured instead of
    constructing anything — so an unauthenticated POST to an auth route
    (or any other code path) with consumer keys configured surfaces as a
    typed 503-able error, never as a signed request to api.etrade.com.
    Injected test/fake factories never hit this gate.
    """
    if os.environ.get(ALLOW_NETWORK_ENV_VAR) != ALLOW_NETWORK_VALUE:
        raise EtradeNotConfigured(
            "Refusing to build a real E*TRADE transport: the explicit "
            f"opt-in gate {ALLOW_NETWORK_ENV_VAR}={ALLOW_NETWORK_VALUE} is "
            "not set. Consumer keys alone do NOT authorize network "
            "contact in this phase — set the gate deliberately when the "
            "first sandbox contact is authorized. (Offline use injects a "
            "fake session_factory.)")
    from requests_oauthlib import OAuth1Session  # local: optional dep
    return OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
        callback_uri="oob",
    )


def _parse_token_body(text: str) -> Dict[str, str]:
    """Parse an urlencoded oauth token response body."""
    parsed = parse_qs(text, keep_blank_values=True)
    return {key: values[0] for key, values in parsed.items()}


class EtradeAuthManager:
    """OAuth 1.0a state machine: unconfigured -> disconnected ->
    pending_verifier -> connected -> expired (daily) -> ... (contract C16).

    Args:
        db_path: SQLite path for token persistence (defaults to env
            TRADING_DB_PATH, then ``trading_data.db``).
        session_factory: callable with default_session_factory's signature
            returning a session exposing ``.get(url)`` -> response with
            ``.status_code`` / ``.text`` / ``.json()``. Tests inject fakes.
        clock: callable returning an AWARE datetime; default UTC now.
        audit: AuditLog; defaults to one on the same database.
    """

    def __init__(self, db_path: Optional[str] = None,
                 session_factory: Optional[Callable] = None,
                 clock: Optional[Callable[[], datetime]] = None,
                 audit: Optional[AuditLog] = None):
        self.env = os.environ.get("ETRADE_ENV", "sandbox").strip().lower()
        if self.env not in API_BASE_URLS:
            raise EtradeNotConfigured(
                f"ETRADE_ENV must be 'sandbox' or 'production', "
                f"got {self.env!r}")
        # Sandbox and production are DIFFERENT consumer-key pairs at
        # E*TRADE. Prefer env-scoped names so both pairs can coexist in
        # .env without swapping; fall back to the legacy generic names.
        _prefix = ("ETRADE_SANDBOX_" if self.env == "sandbox"
                   else "ETRADE_PROD_")
        self.consumer_key = (os.environ.get(_prefix + "CONSUMER_KEY")
                             or os.environ.get("ETRADE_CONSUMER_KEY"))
        self.consumer_secret = (os.environ.get(_prefix + "CONSUMER_SECRET")
                                or os.environ.get("ETRADE_CONSUMER_SECRET"))
        if self.env == "production" and (
                os.environ.get("ETRADE_PRODUCTION_ACK")
                != PRODUCTION_ACK_VALUE):
            raise EtradeNotConfigured(
                "ETRADE_ENV=production requires the explicit gate "
                f"ETRADE_PRODUCTION_ACK={PRODUCTION_ACK_VALUE}. Refusing to "
                "construct a production auth manager without it — this is "
                "real money.")
        self.db_path = (db_path or os.environ.get("TRADING_DB_PATH")
                        or "trading_data.db")
        self._session_factory = session_factory or default_session_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.audit = audit if audit is not None else AuditLog(
            self.db_path, env=self.env, clock=self._clock)
        #: pending request token: {'oauth_token', 'oauth_token_secret'}
        self._pending: Optional[Dict[str, str]] = None
        self._session = None  # cached signing session for the active token
        self._init_table()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS etrade_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    env TEXT NOT NULL,
                    access_token TEXT NOT NULL,
                    access_secret TEXT NOT NULL,
                    token_issued_at TEXT NOT NULL,
                    renewed_at TEXT,
                    status TEXT NOT NULL DEFAULT 'active'
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _current_row(self) -> Optional[sqlite3.Row]:
        """Latest token row for this env still in 'active'/'expired'."""
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT * FROM etrade_tokens WHERE env = ? AND "
                "status IN ('active', 'expired') "
                "ORDER BY id DESC LIMIT 1", (self.env,)).fetchone()
        finally:
            conn.close()

    def _update_row(self, row_id: int, **fields) -> None:
        sets = ", ".join(f"{name} = ?" for name in fields)
        conn = self._connect()
        try:
            conn.execute(
                f"UPDATE etrade_tokens SET {sets} WHERE id = ?",
                (*fields.values(), row_id))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Expiry rules
    # ------------------------------------------------------------------
    def _midnight_expired(self, token_issued_at: str) -> bool:
        """True when the ET calendar date has rolled since issuance.

        Both instants are converted to America/New_York before comparing
        DATES, so the boundary is midnight EASTERN regardless of DST
        (23:59 ET -> 00:01 ET flips it; a UTC date change at 8pm ET does
        not).
        """
        issued = datetime.fromisoformat(token_issued_at)
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=timezone.utc)
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return issued.astimezone(ET_ZONE).date() != now.astimezone(
            ET_ZONE).date()

    def _expire_row(self, row: sqlite3.Row, reason: str) -> None:
        self._update_row(row["id"], status="expired")
        self._session = None
        logger.warning("E*TRADE token expired (%s); re-auth required", reason)
        self.audit.append("auth_manager", "auth_expired",
                          {"env": self.env, "reason": reason})

    def mark_expired(self, reason: str) -> None:
        """Flip the active token to expired (client calls this on a 401
        oauth_problem response). Idempotent; audited once."""
        row = self._current_row()
        if row is not None and row["status"] == "active":
            self._expire_row(row, reason)

    # ------------------------------------------------------------------
    # Contract C16 surface
    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.consumer_key and self.consumer_secret)

    @property
    def api_base_url(self) -> str:
        return API_BASE_URLS[self.env]

    def _require_configured(self) -> None:
        if not self.configured:
            raise EtradeNotConfigured(
                "ETRADE_CONSUMER_KEY / ETRADE_CONSUMER_SECRET are not set; "
                "configure them in the environment (.env) first.")

    def status(self) -> Dict:
        """State snapshot per contract C16 (never raises — the GUI needs
        to render 'unconfigured' rather than catch exceptions)."""
        authorize_url = None
        token_issued_at = None
        renewed_at = None
        if not self.configured:
            state = "unconfigured"
        elif self._pending is not None:
            state = "pending_verifier"
            authorize_url = AUTHORIZE_URL_TEMPLATE.format(
                key=self.consumer_key, token=self._pending["oauth_token"])
        else:
            row = self._current_row()
            if row is None:
                state = "disconnected"
            else:
                token_issued_at = row["token_issued_at"]
                renewed_at = row["renewed_at"]
                if row["status"] == "expired":
                    state = "expired"
                elif self._midnight_expired(row["token_issued_at"]):
                    # Lazily persist + audit the daily hard expiry the
                    # first time anyone looks after midnight ET.
                    self._expire_row(row, "midnight_et_hard_expiry")
                    state = "expired"
                else:
                    state = "connected"
        return {
            "state": state,
            "env": self.env,
            "authorize_url": authorize_url,
            "token_issued_at": token_issued_at,
            "renewed_at": renewed_at,
        }

    def start_auth(self) -> str:
        """Leg 1: fetch a request token; return the authorize URL."""
        self._require_configured()
        session = self._session_factory(self.consumer_key,
                                        self.consumer_secret)
        response = session.get(REQUEST_TOKEN_URL)
        if response.status_code != 200:
            # Surface E*TRADE's oauth_problem reason for diagnosability.
            # The body carries no secrets, but redact the consumer key
            # defensively in case the gateway ever echoes request params.
            body = (response.text or "")[:300].replace(
                self.consumer_key, "<consumer_key>")
            raise EtradeAuthError(
                f"request_token failed with HTTP {response.status_code}: "
                f"{body}")
        tokens = _parse_token_body(response.text)
        if "oauth_token" not in tokens or "oauth_token_secret" not in tokens:
            raise EtradeAuthError(
                "request_token response missing oauth_token fields")
        self._pending = {
            "oauth_token": tokens["oauth_token"],
            "oauth_token_secret": tokens["oauth_token_secret"],
        }
        self.audit.append("auth_manager", "auth_started", {"env": self.env})
        logger.info("E*TRADE auth started (%s); awaiting verifier", self.env)
        return AUTHORIZE_URL_TEMPLATE.format(
            key=self.consumer_key, token=self._pending["oauth_token"])

    def submit_verifier(self, code: str) -> Dict:
        """Leg 3: exchange the request token + verifier for an access
        token, persist it, and return status()."""
        self._require_configured()
        if self._pending is None:
            raise EtradeAuthError(
                "No authorization in progress — call start_auth() first")
        session = self._session_factory(
            self.consumer_key, self.consumer_secret,
            resource_owner_key=self._pending["oauth_token"],
            resource_owner_secret=self._pending["oauth_token_secret"],
            verifier=code)
        response = session.get(ACCESS_TOKEN_URL)
        if response.status_code != 200:
            raise EtradeAuthError(
                f"access_token failed with HTTP {response.status_code}")
        tokens = _parse_token_body(response.text)
        if "oauth_token" not in tokens or "oauth_token_secret" not in tokens:
            raise EtradeAuthError(
                "access_token response missing oauth_token fields")
        issued_at = self._clock().isoformat()
        conn = self._connect()
        try:
            # One active row: everything older is superseded.
            conn.execute(
                "UPDATE etrade_tokens SET status = 'disconnected' "
                "WHERE env = ? AND status IN ('active', 'expired')",
                (self.env,))
            conn.execute(
                "INSERT INTO etrade_tokens "
                "(env, access_token, access_secret, token_issued_at, "
                " renewed_at, status) VALUES (?, ?, ?, ?, NULL, 'active')",
                (self.env, tokens["oauth_token"],
                 tokens["oauth_token_secret"], issued_at))
            conn.commit()
        finally:
            conn.close()
        self._pending = None
        self._session = None
        self.audit.append("auth_manager", "auth_connected",
                          {"env": self.env})
        logger.info("E*TRADE connected (%s); token issued %s",
                    self.env, issued_at)
        return self.status()

    def renew(self) -> bool:
        """Clear the 2h idle timeout. True on success; False when there is
        nothing renewable (no token / midnight-expired / oauth_problem —
        the latter two flip state to 'expired' and require re-auth)."""
        if not self.configured:
            return False
        row = self._current_row()
        if row is None or row["status"] != "active":
            return False
        if self._midnight_expired(row["token_issued_at"]):
            self._expire_row(row, "midnight_et_hard_expiry")
            return False
        session = self._build_session(row)
        response = session.get(RENEW_TOKEN_URL)
        if response.status_code == 200:
            renewed_at = self._clock().isoformat()
            self._update_row(row["id"], renewed_at=renewed_at)
            self.audit.append("auth_manager", "auth_renewed",
                              {"env": self.env})
            logger.info("E*TRADE token renewed at %s", renewed_at)
            return True
        if response.status_code == 401 and "oauth_problem" in response.text:
            self._expire_row(row, "oauth_problem on renew")
            return False
        logger.warning("E*TRADE renew failed: HTTP %s", response.status_code)
        self.audit.append("auth_manager", "auth_renew_failed",
                          {"env": self.env,
                           "status_code": response.status_code})
        return False

    def disconnect(self) -> None:
        """Drop the local token (no remote call — revocation is a
        deliberate manual step the user performs at E*TRADE)."""
        row = self._current_row()
        if row is not None:
            self._update_row(row["id"], status="disconnected")
        self._pending = None
        self._session = None
        self.audit.append("auth_manager", "auth_disconnected",
                          {"env": self.env})
        logger.info("E*TRADE disconnected (%s)", self.env)

    # ------------------------------------------------------------------
    # Client surface
    # ------------------------------------------------------------------
    def _build_session(self, row: sqlite3.Row):
        if self._session is None:
            self._session = self._session_factory(
                self.consumer_key, self.consumer_secret,
                resource_owner_key=row["access_token"],
                resource_owner_secret=row["access_secret"])
        return self._session

    def get_session(self):
        """Signing session for the ACTIVE token; the EtradeClient calls
        this per request so renewals/re-auth are picked up.

        Raises:
            EtradeNotConfigured: consumer keys absent.
            EtradeNotConnected: no token (run the auth flow).
            EtradeAuthExpired: token hit midnight ET or an oauth_problem.
        """
        self._require_configured()
        row = self._current_row()
        if row is None:
            raise EtradeNotConnected(
                "No E*TRADE access token — complete the authorization flow")
        if row["status"] == "expired":
            raise EtradeAuthExpired(
                "E*TRADE access token expired; re-authorization required")
        if self._midnight_expired(row["token_issued_at"]):
            self._expire_row(row, "midnight_et_hard_expiry")
            raise EtradeAuthExpired(
                "E*TRADE access token hit the midnight-ET hard expiry; "
                "re-authorization required")
        return self._build_session(row)
