"""
Alerts and Notifications System
Generates real-time alerts for trading events
"""

import json
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    PRICE_ALERT = "price_alert"
    SIGNAL_ALERT = "signal_alert"
    RISK_ALERT = "risk_alert"
    PERFORMANCE_ALERT = "performance_alert"
    ERROR = "error"


class AlertManager:
    """
    Manages trading alerts and notifications
    Supports in-app, email, and webhook notifications
    """
    
    def __init__(self, db_provider: Optional[Callable[[], object]] = None):
        # In-memory store, used when no db_provider is supplied (tests, scripts).
        self.alerts: List[Dict] = []
        # Lazy DB accessor: called per-op so we never create a SQLite file at
        # import (mirrors gui.globals.get_db). When set, alerts persist.
        self._db_provider = db_provider

    def _db(self):
        return self._db_provider() if self._db_provider is not None else None

    @staticmethod
    def _row_to_alert(row: Dict) -> Dict:
        """Map a persisted alerts row to the in-app/GUI alert shape."""
        details = row.get('details')
        if isinstance(details, str):
            details = json.loads(details) if details else {}
        return {
            'id': row.get('id'),
            'timestamp': row.get('timestamp'),
            'type': row.get('alert_type'),
            'symbol': row.get('symbol'),
            'message': row.get('message'),
            'priority': row.get('priority'),
            'details': details or {},
            'read': bool(row.get('is_read')),
        }
    
    def create_alert(self, alert_type: AlertType, symbol: str, message: str,
                    priority: AlertPriority = AlertPriority.NORMAL,
                    details: Dict = None) -> Dict:
        """Create an alert"""
        db = self._db()
        alert = {
            'id': None,
            'timestamp': datetime.now().isoformat(),
            'type': alert_type.value,
            'symbol': symbol,
            'message': message,
            'priority': priority.value,
            'details': details or {},
            'read': False,
        }
        if db is not None:
            alert['id'] = db.create_alert(alert_type.value, symbol, message,
                                          priority.value, details or {})
        else:
            alert['id'] = len(self.alerts)
            self.alerts.append(alert)

        return alert
    
    def price_alert(self, symbol: str, current_price: float, target_price: float,
                   direction: str = 'above', details: Dict = None):
        """Price target alert"""
        message = f"{symbol} hit ${current_price:.2f} ({direction} ${target_price:.2f})"
        priority = AlertPriority.HIGH if abs(current_price - target_price) < 0.5 else AlertPriority.NORMAL
        
        return self.create_alert(
            AlertType.PRICE_ALERT,
            symbol,
            message,
            priority,
            details or {'current_price': current_price, 'target_price': target_price}
        )
    
    def get_alerts(self, unread_only: bool = False) -> List[Dict]:
        """Get alerts"""
        db = self._db()
        if db is not None:
            return [self._row_to_alert(r)
                    for r in db.get_alerts(unread_only=unread_only)]
        if unread_only:
            return [a for a in self.alerts if not a['read']]
        return self.alerts

    def mark_read(self, alert_id: int):
        """Mark alert as read"""
        db = self._db()
        if db is not None:
            db.mark_alert_read(alert_id)
            return
        if 0 <= alert_id < len(self.alerts):
            self.alerts[alert_id]['read'] = True

    def mark_all_read(self):
        """Mark all alerts as read"""
        db = self._db()
        if db is not None:
            db.mark_all_alerts_read()
            return
        for alert in self.alerts:
            alert['read'] = True

    def clear_alerts(self):
        """Clear all alerts"""
        db = self._db()
        if db is not None:
            db.clear_alerts()
            return
        self.alerts = []
