"""Offline sanity checks for the Phase 4 Docker deployment artifacts.

Stdlib-only (no PyYAML in the pinned deps), no network, no docker daemon:
plain-text assertions on the Dockerfile, .dockerignore, and
docker-compose.yml guarding the invariants that matter:

* gunicorn runs with exactly 1 worker (JobManager / in-memory caches are
  per-process — see the comment block above the Dockerfile CMD);
* secrets and local databases never reach the image;
* the compose file passes .env through at runtime only and contains no
  inline secret values.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
COMPOSE = REPO_ROOT / "docker-compose.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCKER_SMOKE = REPO_ROOT / "scripts" / "docker_smoke.py"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def dockerignore_text() -> str:
    return DOCKERIGNORE.read_text()


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE.read_text()


@pytest.fixture(scope="module")
def ci_text() -> str:
    return CI_WORKFLOW.read_text()


@pytest.fixture(scope="module")
def smoke_text() -> str:
    return DOCKER_SMOKE.read_text()


class TestDockerfile:
    def test_exists(self):
        assert DOCKERFILE.is_file()

    def test_base_image_matches_venv_python(self, dockerfile_text):
        assert "FROM python:3.13-slim" in dockerfile_text

    def test_single_worker_enforced_in_cmd(self, dockerfile_text):
        # The value immediately following "--workers" in the CMD must be "1".
        assert '"--workers"' in dockerfile_text
        after_flag = dockerfile_text.split('"--workers",', 1)[1]
        first_value = after_flag.replace("\\", " ").split('"')[1]
        assert first_value == "1"

    def test_single_worker_constraint_documented(self, dockerfile_text):
        assert "JobManager" in dockerfile_text
        assert "--workers MUST stay 1" in dockerfile_text

    def test_runs_as_non_root(self, dockerfile_text):
        assert "USER app" in dockerfile_text
        # /data must be chowned before USER so the named volume is writable.
        assert dockerfile_text.index("chown") < dockerfile_text.index("USER app")

    def test_healthcheck_uses_stdlib(self, dockerfile_text):
        assert "HEALTHCHECK" in dockerfile_text
        assert "urllib.request" in dockerfile_text
        assert "curl" not in dockerfile_text.lower().replace("no curl", "")

    def test_no_secret_or_db_copy(self, dockerfile_text):
        assert ".env" not in dockerfile_text
        copy_lines = [
            line for line in dockerfile_text.splitlines() if line.startswith("COPY")
        ]
        assert copy_lines, "Dockerfile should contain COPY instructions"
        assert not any(".db" in line for line in copy_lines)

    @pytest.mark.parametrize("runtime_package", ["desks", "analysis", "execution"])
    def test_copies_runtime_import_packages(self, dockerfile_text, runtime_package):
        assert f"COPY {runtime_package}/ {runtime_package}/" in dockerfile_text


class TestDockerignore:
    @pytest.mark.parametrize(
        "entry",
        [".env", ".env.*", "*.db", ".git", ".venv", "__pycache__/", "*.pyc", "tests/"],
    )
    def test_excludes(self, dockerignore_text, entry):
        assert entry in dockerignore_text.splitlines()


class TestComposeFile:
    def test_exists(self):
        assert COMPOSE.is_file()

    def test_env_file_passthrough_not_inline_secrets(self, compose_text):
        assert "env_file" in compose_text
        # No plausible secret material inline.
        for marker in ("ETRADE_CONSUMER", "ACCESS_TOKEN", "SECRET_KEY"):
            assert marker not in compose_text

    def test_named_volume_for_data(self, compose_text):
        assert "trading-data:/data" in compose_text
        assert "trading-data:" in compose_text.split("volumes:")[-1]

    def test_restart_policy(self, compose_text):
        assert "restart: unless-stopped" in compose_text

    def test_host_port_is_published_on_loopback_only(self, compose_text):
        assert '"127.0.0.1:5001:5001"' in compose_text
        assert '\n      - "5001:5001"' not in compose_text


class TestBuiltContainerSmoke:
    def test_stdlib_script_exists_and_is_python_entrypoint(self, smoke_text):
        assert DOCKER_SMOKE.is_file()
        assert smoke_text.startswith("#!/usr/bin/env python3")
        assert 'if __name__ == "__main__"' in smoke_text

    def test_ci_runs_smoke_after_build(self, ci_text):
        build = "docker build -t stockoptionstrader:ci ."
        smoke = "python3 scripts/docker_smoke.py --image stockoptionstrader:ci"
        assert build in ci_text
        assert smoke in ci_text
        assert ci_text.index(build) < ci_text.index(smoke)

    def test_uses_isolated_named_data_volume_and_ephemeral_loopback_port(
            self, smoke_text):
        assert 'docker("volume", "create", volume)' in smoke_text
        assert 'type=volume,source={volume},target=/data' in smoke_text
        assert '"127.0.0.1::5001"' in smoke_text

    def test_exercises_public_health_and_basic_auth(self, smoke_text):
        assert 'require_status(base_url, "/health", 200)' in smoke_text
        assert '"Authorization"' in smoke_text
        assert 'require_status(base_url, path, 401)' in smoke_text
        assert 'require_status(base_url, path, 200, credentials)' in smoke_text

    def test_exercises_representative_runtime_surfaces(self, smoke_text):
        for path in ("/", "/live", "/api/strategies", "/api/backtests",
                     "/api/floor/desks", "/api/live/status"):
            assert f'"{path}"' in smoke_text

    def test_network_is_disabled_and_live_state_is_checked(self, smoke_text):
        assert '"ETRADE_ALLOW_NETWORK=0"' in smoke_text
        assert 'live.get("env") != "sandbox"' in smoke_text
        assert 'live.get("auth", {}).get("state") != "disconnected"' in smoke_text

    def test_nonroot_volume_and_sqlite_are_verified(self, smoke_text):
        assert "Path('/data/.ci-write-probe')" in smoke_text
        assert "PRAGMA quick_check" in smoke_text
        for table in ("backtests", "etrade_tokens", "kill_switch_state"):
            assert table in smoke_text

    def test_cleanup_removes_container_before_volume(self, smoke_text):
        container_cleanup = 'docker("rm", "--force", container'
        volume_cleanup = 'docker("volume", "rm", "--force", volume'
        assert container_cleanup in smoke_text
        assert volume_cleanup in smoke_text
        assert smoke_text.index(container_cleanup) < smoke_text.index(volume_cleanup)

    def test_fatal_runtime_log_signatures_fail_smoke(self, smoke_text):
        for marker in ("worker failed to boot", "modulenotfounderror",
                       "traceback (most recent call last)", "permissionerror",
                       "unable to open database file"):
            assert marker in smoke_text
