"""
Database persistence for backtests, trades, and alerts
Uses SQLite for simplicity and portability
"""

import sqlite3
import json
from typing import Dict, List, Optional
import os


class TradingDatabase:
    """
    SQLite database for persisting trading data
    """
    
    DB_PATH = "trading_data.db"
    
    def __init__(self, db_path: str = None):
        """db_path defaults to env TRADING_DB_PATH, falling back to DB_PATH."""
        self.db_path = db_path or os.environ.get('TRADING_DB_PATH') or self.DB_PATH
        # Ensure parent directory exists for non-memory databases
        if db_path and db_path != ':memory:':
            os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Backtests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                start_date TEXT,
                end_date TEXT,
                symbols TEXT,
                initial_capital REAL,
                strategy TEXT,
                parameters TEXT,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                total_trades INTEGER,
                results JSON,
                UNIQUE(name, timestamp)
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                side TEXT,
                price REAL,
                quantity REAL,
                commission REAL,
                pnl REAL,
                pnl_percent REAL,
                FOREIGN KEY(backtest_id) REFERENCES backtests(id)
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT,
                symbol TEXT,
                message TEXT,
                priority TEXT,
                is_read INTEGER DEFAULT 0,
                details JSON
            )
        ''')
        
        # Paper trader sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trader_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                initial_capital REAL,
                strategy TEXT,
                symbols TEXT,
                state JSON,
                performance JSON
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_backtest(self, name: str, start_date: str, end_date: str, 
                     symbols: List[str], initial_capital: float, 
                     strategy: str, parameters: Dict, results: Dict) -> int:
        """Save backtest results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtests 
            (name, start_date, end_date, symbols, initial_capital, strategy, parameters, 
             total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name,
            start_date,
            end_date,
            json.dumps(symbols),
            initial_capital,
            strategy,
            json.dumps(parameters),
            results.get('total_return'),
            results.get('sharpe_ratio'),
            results.get('max_drawdown'),
            results.get('win_rate'),
            results.get('total_trades'),
            json.dumps(results),
        ))
        
        backtest_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return backtest_id
    
    def save_trade(self, backtest_id: int, symbol: str, side: str, 
                   price: float, quantity: float, commission: float, pnl: float):
        """Save individual trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        pnl_percent = (pnl / (price * quantity)) * 100 if price * quantity > 0 else 0
        
        cursor.execute('''
            INSERT INTO trades (backtest_id, symbol, side, price, quantity, commission, pnl, pnl_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (backtest_id, symbol, side, price, quantity, commission, pnl, pnl_percent))
        
        conn.commit()
        conn.close()
    
    def create_alert(self, alert_type: str, symbol: str, message: str, 
                    priority: str = 'normal', details: Dict = None):
        """Create an alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (alert_type, symbol, message, priority, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (alert_type, symbol, message, priority, json.dumps(details or {})))
        
        conn.commit()
        conn.close()
    
    def get_alerts(self, unread_only: bool = False) -> List[Dict]:
        """Get alerts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM alerts'
        if unread_only:
            query += ' WHERE is_read = 0'
        query += ' ORDER BY timestamp DESC LIMIT 100'
        
        cursor.execute(query)
        alerts = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return alerts
    
    def mark_alert_read(self, alert_id: int):
        """Mark alert as read"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE alerts SET is_read = 1 WHERE id = ?', (alert_id,))
        
        conn.commit()
        conn.close()
    
    def get_backtest(self, backtest_id: int) -> Optional[Dict]:
        """Get backtest by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM backtests WHERE id = ?', (backtest_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            row = dict(result)
            row['results'] = json.loads(row['results'])
            return row
        
        return None
    
    def get_backtests(self, limit: int = 50) -> List[Dict]:
        """Get recent backtests"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM backtests ORDER BY timestamp DESC LIMIT ?', (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return results
    
    def get_backtest_trades(self, backtest_id: int) -> List[Dict]:
        """Get trades for a backtest"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM trades WHERE backtest_id = ? ORDER BY timestamp', (backtest_id,))
        trades = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return trades
    
    def delete_backtest(self, backtest_id: int):
        """Delete backtest and related trades"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM trades WHERE backtest_id = ?', (backtest_id,))
        cursor.execute('DELETE FROM backtests WHERE id = ?', (backtest_id,))
        
        conn.commit()
        conn.close()
