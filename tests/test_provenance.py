"""Tests for utils.provenance (run-level reproducibility) + BacktestEngine
seed pinning. Offline, no network."""

from __future__ import annotations

import json

import numpy as np

from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy
from utils import provenance
from utils.provenance import capture_run_provenance

EXPECTED_KEYS = {'git_sha', 'git_dirty', 'python_version',
                 'dependency_versions', 'seed', 'captured_at'}


class TestCaptureProvenance:
    def test_shape_and_seed_echo(self):
        prov = capture_run_provenance(seed=7)
        assert set(prov) == EXPECTED_KEYS
        assert prov['seed'] == 7
        assert isinstance(prov['python_version'], str) and prov['python_version']
        assert isinstance(prov['dependency_versions'], dict)
        assert 'numpy' in prov['dependency_versions']
        assert prov['git_sha'] is None or isinstance(prov['git_sha'], str)
        assert prov['git_dirty'] in (True, False, None)
        assert isinstance(prov['captured_at'], str) and prov['captured_at']

    def test_seed_none_when_unpinned(self):
        assert capture_run_provenance()['seed'] is None

    def test_git_failure_degrades_gracefully(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError('git not installed')
        monkeypatch.setattr(provenance.subprocess, 'run', boom)
        prov = capture_run_provenance(seed=1)
        assert prov['git_sha'] is None     # unknown -> None, not a crash
        assert prov['git_dirty'] is None   # unknown != clean
        assert prov['python_version']      # the rest still captured

    def test_git_nonzero_exit_degrades(self, monkeypatch):
        class _Result:
            returncode = 128
            stdout = ''
            stderr = 'fatal: not a git repository'
        monkeypatch.setattr(provenance.subprocess, 'run',
                            lambda *a, **k: _Result())
        prov = capture_run_provenance()
        assert prov['git_sha'] is None
        assert prov['git_dirty'] is None

    def test_never_raises_returns_stable_keys_on_total_failure(self, monkeypatch):
        # If an internal call blows up, the top-level guard still returns the
        # fixed key set (with the seed echoed) so the saved schema is stable.
        def boom():
            raise RuntimeError('synthetic provenance explosion')
        monkeypatch.setattr(provenance, '_git_sha', boom)
        prov = capture_run_provenance(seed=5)
        assert set(prov) == EXPECTED_KEYS
        assert prov['seed'] == 5

    def test_dependency_lookup_missing_package_is_none(self, monkeypatch):
        real_version = provenance.metadata.version

        def fake_version(name):
            if name == 'openbb':
                raise provenance.metadata.PackageNotFoundError(name)
            return real_version(name)
        monkeypatch.setattr(provenance.metadata, 'version', fake_version)
        deps = capture_run_provenance()['dependency_versions']
        assert deps['openbb'] is None
        assert deps['numpy'] is not None

    def test_json_serializable(self):
        json.dumps(capture_run_provenance(seed=3))  # must not raise


class TestEngineSeed:
    def test_seed_is_stored(self):
        assert BacktestEngine(MomentumStrategy(), seed=42).rng_seed == 42

    def test_no_seed_is_none(self):
        assert BacktestEngine(MomentumStrategy()).rng_seed is None

    def _draw_at_impl_start(self, seed):
        """Seed is applied at the START of run() (not __init__), so capture the
        RNG draw at the moment _run_impl begins — stubbed so no data/network is
        needed."""
        captured = {}
        engine = BacktestEngine(MomentumStrategy(), seed=seed)
        engine._run_impl = lambda *a, **k: captured.setdefault(
            'draw', np.random.random(5).tolist()) or {}
        engine.run(['AAA'], '2022-01-01', '2022-12-31')
        return captured['draw']

    def test_run_pins_global_numpy_rng(self):
        # Same seed -> identical RNG stream at the start of the run.
        assert self._draw_at_impl_start(123) == self._draw_at_impl_start(123)

    def test_different_seeds_diverge(self):
        assert self._draw_at_impl_start(1) != self._draw_at_impl_start(2)

    def test_unseeded_run_takes_no_lock_and_does_not_seed(self):
        # An unseeded run must skip the seeded-run lock entirely.
        from backtesting import backtest_engine as be
        engine = BacktestEngine(MomentumStrategy())  # seed=None
        engine._run_impl = lambda *a, **k: {'ok': True}
        assert engine.run(['AAA'], '2022-01-01', '2022-12-31') == {'ok': True}
        assert not be._SEEDED_RUN_LOCK.locked()  # released / never taken
