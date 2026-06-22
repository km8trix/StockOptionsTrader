"""Phase 4 ops: alerts persist to SQLite when a db_provider is supplied,
and fall back to in-memory otherwise."""

from utils.alerts import AlertManager, AlertPriority, AlertType
from utils.database import TradingDatabase


def test_db_backed_alert_survives_a_fresh_manager(tmp_path):
    db = TradingDatabase(db_path=str(tmp_path / "a.db"))
    created = AlertManager(db_provider=lambda: db).create_alert(
        AlertType.SIGNAL_ALERT, 'SPY', 'buy signal',
        AlertPriority.NORMAL, {'confidence': 0.9})
    assert created['id'] is not None

    # A new manager over the same db sees it — survives a "restart".
    alerts = AlertManager(db_provider=lambda: db).get_alerts()
    assert len(alerts) == 1
    a = alerts[0]
    assert a['symbol'] == 'SPY'
    assert a['type'] == 'signal_alert'
    assert a['details'] == {'confidence': 0.9}
    assert a['read'] is False


def test_db_backed_mark_read(tmp_path):
    db = TradingDatabase(db_path=str(tmp_path / "a.db"))
    m = AlertManager(db_provider=lambda: db)
    created = m.create_alert(AlertType.PRICE_ALERT, 'AAPL', 'hit')
    m.mark_read(created['id'])
    assert m.get_alerts(unread_only=True) == []
    assert m.get_alerts()[0]['read'] is True


def test_db_backed_mark_all_and_clear(tmp_path):
    db = TradingDatabase(db_path=str(tmp_path / "a.db"))
    m = AlertManager(db_provider=lambda: db)
    m.create_alert(AlertType.ERROR, 'sys', 'x')
    m.create_alert(AlertType.ERROR, 'sys', 'y')
    m.mark_all_read()
    assert m.get_alerts(unread_only=True) == []
    m.clear_alerts()
    assert m.get_alerts() == []


def test_in_memory_fallback_without_db():
    m = AlertManager()  # no db_provider
    m.create_alert(AlertType.SIGNAL_ALERT, 'SPY', 'msg')
    alerts = m.get_alerts()
    assert len(alerts) == 1
    assert alerts[0]['read'] is False
