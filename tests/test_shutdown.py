"""Phase 4 ops: graceful shutdown stops background threads, is safe when none
are running, and never raises (reads the singletons; constructs nothing)."""

from gui.routes import api_live


class _FakeScheduler:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_shutdown_stops_installed_workers():
    ka, live = _FakeScheduler(), _FakeScheduler()
    api_live.set_keepalive_scheduler(ka)
    api_live.set_scheduler(live)
    try:
        api_live.shutdown_background_workers()
        assert ka.stopped
        assert live.stopped
    finally:
        api_live.set_keepalive_scheduler(None)
        api_live.set_scheduler(None)


def test_shutdown_is_safe_with_nothing_installed():
    api_live.set_keepalive_scheduler(None)
    api_live.set_scheduler(None)
    api_live.shutdown_background_workers()  # must not raise or construct one
    assert api_live.get_scheduler() is None


def test_shutdown_swallows_stop_errors():
    class _Boom:
        def stop(self):
            raise RuntimeError("boom")

    api_live.set_keepalive_scheduler(_Boom())
    api_live.set_scheduler(None)
    try:
        api_live.shutdown_background_workers()  # must not propagate
    finally:
        api_live.set_keepalive_scheduler(None)
