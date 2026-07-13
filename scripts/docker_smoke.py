#!/usr/bin/env python3
"""Start the built image and exercise its real runtime/security surfaces.

This script is intentionally stdlib-only.  CI supplies Docker; the application
dependencies exist solely inside the image being tested.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


FATAL_LOG_MARKERS = (
    "worker failed to boot",
    "modulenotfounderror",
    "traceback (most recent call last)",
    "permissionerror",
    "unable to open database file",
)


class SmokeFailure(RuntimeError):
    pass


def docker(*args: str, check: bool = True, timeout: float = 60.0,
           capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], check=check, timeout=timeout, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def request(base_url: str, path: str,
            credentials: tuple[str, str] | None = None) -> tuple[int, bytes, Any]:
    headers = {"Accept": "application/json"}
    if credentials is not None:
        raw = f"{credentials[0]}:{credentials[1]}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
    req = Request(base_url + path, headers=headers)
    try:
        with urlopen(req, timeout=10.0) as response:
            return response.status, response.read(), response.headers
    except HTTPError as exc:
        return exc.code, exc.read(), exc.headers
    except URLError as exc:
        raise SmokeFailure(f"request failed for {path}: {exc}") from exc


def require_status(base_url: str, path: str, expected: int,
                   credentials: tuple[str, str] | None = None) -> bytes:
    status, body, _headers = request(base_url, path, credentials)
    if status != expected:
        preview = body.decode("utf-8", errors="replace")[:500]
        raise SmokeFailure(
            f"{path} returned {status}, expected {expected}: {preview}")
    return body


def wait_healthy(container: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "unknown"
    while time.monotonic() < deadline:
        state = docker(
            "inspect", "--format",
            "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
            container,
        ).stdout.strip()
        last = state
        parts = state.split()
        if parts and parts[0] == "exited":
            raise SmokeFailure("container exited before becoming healthy")
        if len(parts) > 1 and parts[1] == "healthy":
            return
        if len(parts) > 1 and parts[1] == "unhealthy":
            raise SmokeFailure("container health check reported unhealthy")
        time.sleep(1.0)
    raise SmokeFailure(f"container did not become healthy in {timeout:.0f}s ({last})")


def mapped_port(container: str) -> int:
    output = docker("port", container, "5001/tcp").stdout.strip().splitlines()
    if not output:
        raise SmokeFailure("Docker did not publish container port 5001")
    try:
        return int(output[0].rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise SmokeFailure(f"cannot parse Docker port mapping: {output[0]!r}") from exc


def verify_http(base_url: str, credentials: tuple[str, str]) -> None:
    health = json.loads(require_status(base_url, "/health", 200))
    if health != {"service": "stock-options-trader", "status": "ok"}:
        raise SmokeFailure(f"unexpected /health payload: {health!r}")

    protected = (
        "/", "/live", "/api/strategies", "/api/backtests",
        "/api/floor/desks", "/api/live/status",
    )
    for path in protected:
        require_status(base_url, path, 401)
    require_status(base_url, "/", 401, (credentials[0], "wrong-password"))
    for path in protected:
        body = require_status(base_url, path, 200, credentials)
        if path.startswith("/api/"):
            try:
                json.loads(body)
            except json.JSONDecodeError as exc:
                raise SmokeFailure(f"{path} did not return JSON") from exc

    live = json.loads(require_status(
        base_url, "/api/live/status", 200, credentials))
    if live.get("env") != "sandbox" or live.get("auth", {}).get("state") != "disconnected":
        raise SmokeFailure(f"live runtime is not sandbox/disconnected: {live!r}")


def verify_writable_database(container: str) -> None:
    code = """
from pathlib import Path
import sqlite3

sentinel = Path('/data/.ci-write-probe')
sentinel.write_text('ok', encoding='utf-8')
assert sentinel.read_text(encoding='utf-8') == 'ok'
sentinel.unlink()

conn = sqlite3.connect('/data/trading_data.db')
try:
    assert conn.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    required = {'backtests', 'etrade_tokens', 'kill_switch_state'}
    assert required <= tables, (required, tables)
finally:
    conn.close()
"""
    result = docker("exec", container, "python", "-c", code,
                    check=False, timeout=30.0)
    if result.returncode != 0:
        raise SmokeFailure(
            "non-root /data or SQLite verification failed:\n" + result.stdout)


def run(image: str, health_timeout: float) -> None:
    suffix = uuid.uuid4().hex[:12]
    container = f"stockoptionstrader-smoke-{suffix}"
    volume = f"stockoptionstrader-smoke-data-{suffix}"
    username = "ci-" + secrets.token_hex(8)
    password = secrets.token_urlsafe(24)
    credentials = (username, password)
    container_created = False
    volume_created = False
    failure: BaseException | None = None
    logs = ""

    try:
        docker("volume", "create", volume)
        volume_created = True
        result = docker(
            "run", "--detach", "--name", container,
            "--publish", "127.0.0.1::5001",
            "--mount", f"type=volume,source={volume},target=/data",
            "--health-interval", "1s", "--health-timeout", "3s",
            "--health-start-period", "2s", "--health-retries", "30",
            "--env", f"APP_AUTH_USERNAME={username}",
            "--env", f"APP_AUTH_PASSWORD={password}",
            "--env", f"SECRET_KEY={secrets.token_urlsafe(32)}",
            "--env", "ETRADE_ENV=sandbox",
            "--env", "ETRADE_ALLOW_NETWORK=0",
            "--env", "ETRADE_SANDBOX_CONSUMER_KEY=ci-dummy-key",
            "--env", "ETRADE_SANDBOX_CONSUMER_SECRET=ci-dummy-secret",
            "--env", "ETRADE_ACCOUNT_ID_KEY=ci-account",
            image,
        )
        if not result.stdout.strip():
            raise SmokeFailure("docker run returned no container id")
        container_created = True
        wait_healthy(container, health_timeout)
        base_url = f"http://127.0.0.1:{mapped_port(container)}"
        verify_http(base_url, credentials)
        verify_writable_database(container)
        logs = docker("logs", container, check=False).stdout
        lower_logs = logs.lower()
        fatal = [marker for marker in FATAL_LOG_MARKERS if marker in lower_logs]
        if fatal:
            raise SmokeFailure(f"fatal runtime log marker(s): {', '.join(fatal)}")
        print(f"Container smoke test passed for {image} at {base_url}")
    except BaseException as exc:
        failure = exc
        if container_created:
            try:
                logs = docker("logs", container, check=False).stdout
            except BaseException as log_exc:  # cleanup/reporting must continue
                logs = f"<unable to read container logs: {log_exc}>"
        print("\n--- container logs ---", file=sys.stderr)
        print(logs or "<no logs>", file=sys.stderr)
        print("--- end container logs ---\n", file=sys.stderr)
    finally:
        cleanup_errors: list[str] = []
        # The names are unique and known before `docker run`; attempt both
        # removals even if the run command or an interrupt happened between
        # Docker creating the resource and our local bookkeeping assignment.
        try:
            docker("rm", "--force", container, check=False, timeout=30.0)
        except BaseException as exc:
            cleanup_errors.append(f"container cleanup failed: {exc}")
        if volume_created:
            try:
                docker("volume", "rm", "--force", volume,
                       check=False, timeout=30.0)
            except BaseException as exc:
                cleanup_errors.append(f"volume cleanup failed: {exc}")
        if cleanup_errors:
            message = "; ".join(cleanup_errors)
            if failure is None:
                failure = SmokeFailure(message)
            else:
                print(message, file=sys.stderr)
    if failure is not None:
        raise failure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="stockoptionstrader:ci")
    parser.add_argument("--health-timeout", type=float, default=90.0)
    args = parser.parse_args()
    try:
        run(args.image, args.health_timeout)
    except (SmokeFailure, subprocess.SubprocessError, OSError) as exc:
        print(f"docker smoke failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
