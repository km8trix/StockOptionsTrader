"""
Run-level reproducibility provenance for backtests.

Per-symbol DATA provenance (which provider served each symbol, from cache or
live) already rides in a report's ``data_sources`` via
MarketDataHandler.get_last_fetch_info. What was missing — and what makes a saved
run actually reproducible — is RUN-LEVEL provenance: the exact code the run
executed (git commit + working-tree dirty flag), the interpreter and the
versions of the dependencies whose numerics drive results, and the RNG seed the
run was pinned to (if any).

capture_run_provenance() returns that as a JSON-safe dict to fold into the saved
``results`` blob. It is strictly best-effort: every field degrades to a safe
default rather than raising, because provenance capture must never break a
backtest. A missing seed (None) is itself meaningful — it records that the run
was NOT pinned and therefore is not bit-reproducible.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

#: Dependencies whose versions materially affect backtest numerics.
_TRACKED_DEPENDENCIES = ('numpy', 'pandas', 'scipy', 'scikit-learn',
                         'statsmodels', 'openbb', 'flask')

#: Hard cap on each git subprocess so a slow/hung git never stalls a job.
_GIT_TIMEOUT_S = 5


def _git(args: List[str]) -> Optional[str]:
    """Run ``git <args>`` and return trimmed stdout, or None on ANY failure
    (git missing, not a repo, non-zero exit, timeout)."""
    try:
        completed = subprocess.run(
            ['git', *args], capture_output=True, text=True,
            timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_sha() -> Optional[str]:
    """Full HEAD commit SHA, or None when git/the repo is unavailable.

    None (not '') signals 'unknown' — mirroring _git_dirty — so a saved run
    whose code state could not be determined is never confused with a real
    value. (A fresh repo with no commits also yields None, since
    ``git rev-parse HEAD`` errors there.)"""
    return _git(['rev-parse', 'HEAD'])


def _git_dirty() -> Optional[bool]:
    """True if the working tree has uncommitted changes, False if clean,
    None when git is unavailable (so 'unknown' is never confused with 'clean')."""
    status = _git(['status', '--porcelain'])
    if status is None:
        return None
    return status != ''


def _dependency_versions() -> Dict[str, Optional[str]]:
    """Installed version of each tracked dependency; None if not installed.
    Never raises — a lookup failure degrades that entry to None."""
    versions: Dict[str, Optional[str]] = {}
    for name in _TRACKED_DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
        except Exception:  # noqa: BLE001 - provenance must not break a run
            logger.debug('version lookup failed for %s', name, exc_info=True)
            versions[name] = None
    return versions


def capture_run_provenance(seed: Optional[int] = None) -> Dict:
    """Capture run-level reproducibility provenance as a JSON-safe dict.

    Keys:
        git_sha (str|None):       HEAD commit, None if git unavailable.
        git_dirty (bool|None):    uncommitted changes? None if git unavailable.
        python_version (str):     e.g. '3.13.1'.
        dependency_versions (dict): {name: version|None} for tracked deps.
        seed (int|None):          the RNG seed the run was pinned to (None =
                                  unpinned, i.e. not bit-reproducible).
        captured_at (str):        ISO-8601 UTC timestamp.

    Best-effort: never raises. On total failure it still returns the fixed key
    set (with safe defaults) so the saved-run schema stays stable.
    """
    try:
        return {
            'git_sha': _git_sha(),
            'git_dirty': _git_dirty(),
            'python_version': sys.version.split()[0],
            'dependency_versions': _dependency_versions(),
            'seed': seed,
            'captured_at': datetime.now(timezone.utc).isoformat(),
        }
    except Exception:  # noqa: BLE001 - provenance must never break a run
        logger.warning('run provenance capture failed', exc_info=True)
        return {
            'git_sha': None,
            'git_dirty': None,
            'python_version': '',
            'dependency_versions': {},
            'seed': seed,
            'captured_at': '',
        }
