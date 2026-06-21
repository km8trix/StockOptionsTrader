"""
Alerts and Notifications System
Generates real-time alerts for trading events
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List
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
    
    def __init__(self, enable_email: bool = False):
        self.alerts: List[Dict] = []
        self.enable_email = enable_email
        self.email_config = {}
    
    def configure_email(self, smtp_server: str, sender_email: str, sender_password: str,
                       recipient_email: str):
        """Configure email notifications"""
        self.email_config = {
            'smtp_server': smtp_server,
            'sender_email': sender_email,
            'sender_password': sender_password,
            'recipient_email': recipient_email,
        }
    
    def create_alert(self, alert_type: AlertType, symbol: str, message: str,
                    priority: AlertPriority = AlertPriority.NORMAL,
                    details: Dict = None) -> Dict:
        """Create an alert"""
        alert = {
            'id': len(self.alerts),
            'timestamp': datetime.now().isoformat(),
            'type': alert_type.value,
            'symbol': symbol,
            'message': message,
            'priority': priority.value,
            'details': details or {},
            'read': False,
        }
        
        self.alerts.append(alert)
        
        # Send notifications
        if priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]:
            self._send_notifications(alert)
        
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
    
    def signal_alert(self, symbol: str, signal: str, confidence: float = 1.0):
        """Trading signal alert"""
        message = f"Strong {signal} signal for {symbol} (confidence: {confidence:.1%})"
        priority = AlertPriority.HIGH if confidence > 0.8 else AlertPriority.NORMAL
        
        return self.create_alert(
            AlertType.SIGNAL_ALERT,
            symbol,
            message,
            priority,
            {'signal': signal, 'confidence': confidence}
        )
    
    def risk_alert(self, message: str, details: Dict = None):
        """Risk management alert"""
        return self.create_alert(
            AlertType.RISK_ALERT,
            'portfolio',
            message,
            AlertPriority.CRITICAL,
            details
        )
    
    def performance_alert(self, metric: str, value: float, threshold: float, details: Dict = None):
        """Performance metric alert"""
        message = f"{metric} reached {value:.2f} (threshold: {threshold:.2f})"
        
        return self.create_alert(
            AlertType.PERFORMANCE_ALERT,
            'portfolio',
            message,
            AlertPriority.NORMAL,
            details or {'metric': metric, 'value': value, 'threshold': threshold}
        )
    
    def error_alert(self, error_message: str, details: Dict = None):
        """Error alert"""
        return self.create_alert(
            AlertType.ERROR,
            'system',
            f"Error: {error_message}",
            AlertPriority.HIGH,
            details
        )
    
    def _send_notifications(self, alert: Dict):
        """Send notifications via email/webhook"""
        if self.enable_email and self.email_config:
            self._send_email(alert)
    
    def _send_email(self, alert: Dict):
        """Send email notification"""
        try:
            config = self.email_config
            
            subject = f"[{alert['priority'].upper()}] Trading Alert: {alert['symbol']}"
            body = f"""
Trading Alert
=============
Type: {alert['type']}
Symbol: {alert['symbol']}
Message: {alert['message']}
Priority: {alert['priority']}
Time: {alert['timestamp']}

Details:
{str(alert['details'])}
            """
            
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = config['recipient_email']
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(config['smtp_server'], 587) as server:
                server.starttls()
                server.login(config['sender_email'], config['sender_password'])
                server.send_message(msg)
        
        except Exception as e:
            logger.error("Failed to send email: %s", e)
    
    def get_alerts(self, unread_only: bool = False) -> List[Dict]:
        """Get alerts"""
        if unread_only:
            return [a for a in self.alerts if not a['read']]
        return self.alerts
    
    def mark_read(self, alert_id: int):
        """Mark alert as read"""
        if 0 <= alert_id < len(self.alerts):
            self.alerts[alert_id]['read'] = True
    
    def mark_all_read(self):
        """Mark all alerts as read"""
        for alert in self.alerts:
            alert['read'] = True
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
