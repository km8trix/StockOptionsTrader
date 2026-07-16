# E\*TRADE setup

How to connect the Live Trading page to E\*TRADE — sandbox first, then
production. Every credential value lives in `.env` (git- and docker-ignored)
and **never** in code, images, chat, or logs.

## TL;DR

```bash
# 1. Put the right vars in .env (see Sandbox / Production below).
# 2. Launch with the wrapper that loads .env into the process:
./start.sh
# 3. Live page → Connect → authorize on E*TRADE → paste the verifier code.
```

---

## How `.env` is loaded — use `start.sh`

The app **deliberately does not auto-load `.env`** — `run_gui.py` passes
`load_dotenv=False` so credentials only ever enter the process by explicit
operator action (enforced by `TestLauncherEnvHygiene`). So launching
`python run_gui.py` directly will report **"E\*TRADE not configured"** because
it sees none of your `.env` values.

Use the wrapper, which parses `.env` as data with interpolation disabled and
then launches:

```bash
./start.sh
```

Equivalent by hand:

```bash
.venv/bin/python -m scripts.launch_with_env \
  --env-file "$PWD/.env" --script "$PWD/run_gui.py"
```

`.env` is never evaluated as shell code. Values containing `$` therefore stay
literal instead of being expanded as variable references. Standard dotenv
quoting is supported; quote values containing spaces or `#`.

> The auth manager is built once at startup and is a process-wide singleton.
> After editing `.env`, **restart** (`Ctrl-C`, then `./start.sh`) — a running
> server will not pick up changes live.

---

## The gates (why a connect can be refused)

Three independent, fail-closed gates. Each is a deliberate opt-in, so the app
never auto-arms real-money access just by booting.

| Gate | Value | Needed for | Enforced at |
| --- | --- | --- | --- |
| `ETRADE_ALLOW_NETWORK` | `1` | **Any** real E\*TRADE network call (sandbox included) | `default_session_factory` |
| `ETRADE_ENV` | `sandbox` (default) / `production` | Selects host + which key pair is read | `EtradeAuthManager.__init__` |
| `ETRADE_PRODUCTION_ACK` | `I_UNDERSTAND_LIVE_TRADING` | Constructing a **production** auth manager | `EtradeAuthManager.__init__` |

Note: OAuth (`request_token` / `authorize` / `access_token`) always uses
`https://api.etrade.com` for **both** environments — only the consumer key
distinguishes sandbox from production. The account/quote/order API calls use
`apisb.etrade.com` (sandbox) vs `api.etrade.com` (production).

---

## Sandbox (start here — zero money risk)

```dotenv
ETRADE_ENV=sandbox            # or leave unset; sandbox is the default
ETRADE_SANDBOX_CONSUMER_KEY=<sandbox key>
ETRADE_SANDBOX_CONSUMER_SECRET=<sandbox secret>
ETRADE_ALLOW_NETWORK=1        # required even for sandbox
```

Then `./start.sh` → **Connect** → authorize on E\*TRADE → paste the verifier.

Sandbox returns E\*TRADE's **canned test accounts** (generic descriptions and
masked numbers you won't recognize) and canned quotes — that is expected, and
it proves the full OAuth → accounts → ticket flow works end to end.

## Production (real money)

```dotenv
ETRADE_ENV=production
ETRADE_PROD_CONSUMER_KEY=<production key>
ETRADE_PROD_CONSUMER_SECRET=<production secret>
ETRADE_PRODUCTION_ACK=I_UNDERSTAND_LIVE_TRADING
ETRADE_ALLOW_NETWORK=1
ETRADE_ACCOUNT_ID_KEY=<your production account id key>
```

In production the manager reads the `ETRADE_PROD_*` pair (it falls back to the
generic `ETRADE_CONSUMER_KEY/SECRET` only if the prefixed ones are unset — so
set the prefixed pair to avoid accidentally signing with a sandbox key).

> **Production API access is a separate approval from sandbox.** Sandbox keys
> work instantly; production keys must be **approved/activated** by E\*TRADE
> (accept the production agreement; freshly issued or rotated keys can take
> time to go live). Until then, `request_token` returns **HTTP 401** — see
> troubleshooting.

The env badge (top-right of the Live page) reads **SANDBOX** (amber) or
**PRODUCTION** (red) so you always know which you're on.

---

## What the app does NOT read from `.env`

`ETRADE_ACCESS_TOKEN` / `ETRADE_ACCESS_SECRET` are **not** read from the
environment. The OAuth access token + secret are obtained by completing the
connect flow and are persisted in SQLite (the `etrade_tokens` table, scoped by
env). Setting them in `.env` has no effect — connect through the GUI instead.

---

## Daily lifecycle (token expiry + keep-alive)

- **Midnight-ET hard expiry — no programmatic bypass.** E\*TRADE access tokens
  die at midnight US/Eastern every day; a human must re-authorize once per ET
  day (Connect → authorize → verifier). The Live page shows a **reconnect
  banner** and an expiry **countdown** when re-auth is due.
- **Within-day keep-alive.** A renew-only loop clears E\*TRADE's ~2h idle
  timeout during market hours. It starts automatically when you connect,
  pauses at the daily expiry, and **never trades**. Manage it from the Token
  keep-alive card (or `GET/POST /api/live/keepalive`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Connection card says **"E\*TRADE is not configured"** | `.env` not loaded, or the key/secret for the active `ETRADE_ENV` are absent | Launch with `./start.sh`; ensure the `ETRADE_(SANDBOX\|PROD)_CONSUMER_KEY/SECRET` matching `ETRADE_ENV` are set |
| Click Connect → **"E\*TRADE not configured — …`ETRADE_ALLOW_NETWORK=1` is not set"** | The network gate isn't open | Add `ETRADE_ALLOW_NETWORK=1`, restart |
| Click Connect → **"Failed to start authorization"** with `request_token failed with HTTP 401` (HTML page) | E\*TRADE is rejecting the **production** consumer key — production access not approved/active, or key/secret not a valid active pair | Confirm in the E\*TRADE Developer portal that **production access is approved/active**; verify the prod key+secret match and the rotation is complete (contact E\*TRADE API support) |
| Click Connect → 401 with `oauth_problem=signature_invalid` | Key/secret mismatch, or quoting/whitespace in `.env` | Verify the pair belongs together; remove any quotes/trailing spaces (values must be bare) |
| Connected, but **account numbers are unfamiliar** | You're on **sandbox** — those are canned test accounts | Switch to the production block to see real accounts |
| Was connected, now **"token expired"** the next day | The daily midnight-ET expiry (expected) | Reconnect each morning (the banner prompts you) |

Production-keys 401 is the most common first-production hurdle and is almost
always an E\*TRADE-side approval/activation matter, not a config bug — the
local checklist above (env, key/secret format, gates, no quotes) is quick to
verify, and once it's clean the issue is on E\*TRADE's side.

---

## Safety rails (production)

When `ETRADE_ENV=production`, real orders hit your real account. These rails
are always on: the **kill switch** (blocks all order placement), a **blocking
fat-finger guard** (rejects limits far from market), the **−2% daily-loss
circuit breaker**, **market-hours blocking**, and the token **keep-alive**
loop. Rehearse the full ticket flow in **sandbox** first; flip to production
only when you intend to trade for real.

## Rotating the consumer secret

Rotating a production consumer key/secret is **not self-service** — contact
E\*TRADE API/developer support to revoke and reissue. If a secret is ever
exposed (pasted into chat, a log, or a commit), rotate it **before** any
real-money use, and never put a secret anywhere but `.env`.
