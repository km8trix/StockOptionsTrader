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


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def dockerignore_text() -> str:
    return DOCKERIGNORE.read_text()


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE.read_text()


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
