# gui/globals.py
from data.market_data import MarketDataHandler
from utils.database import TradingDatabase
from utils.alerts import AlertManager
from portfolio.risk_manager import RiskManager

# Initialize shared global components
market_data = MarketDataHandler()
db = TradingDatabase()
alert_manager = AlertManager()
risk_manager = RiskManager()
paper_traders = {}