# Integration Guide: Advanced Features

This guide shows how to integrate the advanced features into your trading system.

## Quick Start: 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key new packages:
- `scikit-learn` - Machine learning strategies
- `plotly` - Interactive charts
- SQLite3 - (built-in) Database persistence

### 2. Import Advanced Modules

```python
# Strategies
from strategies.advanced import (
    MachineLearningStrategy,
    EnhancedMeanReversionStrategy,
    VolatilityBreakoutStrategy,
    CombinedStrategy,
    AdaptiveStrategy,
)

# Risk Management
from portfolio.risk_manager import RiskManager

# Database
from utils.database import TradingDatabase

# Alerts
from utils.alerts import AlertManager, RealTimeMonitor
```

### 3. Run Examples

```bash
python examples_advanced.py
```

## Feature Integration Examples

### Example 1: ML-Based Backtest with Risk Management

```python
from data.market_data import MarketDataHandler
from strategies.advanced import MachineLearningStrategy
from backtesting.backtest_engine import BacktestEngine
from portfolio.risk_manager import RiskManager
from utils.database import TradingDatabase
from datetime import datetime, timedelta

# Setup
market_data = MarketDataHandler()
db = TradingDatabase('trading.db')

# Fetch data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
data = market_data.fetch_stock_data('AAPL', start_date, end_date)
data = market_data.calculate_indicators(data)

# Create strategy with risk management
strategy = MachineLearningStrategy(lookback=50)
risk_manager = RiskManager(
    max_position_size=0.1,
    max_daily_loss=0.05,
    position_stop_loss=0.02
)

# Run backtest
engine = BacktestEngine(100000, strategy, risk_manager)
results = engine.run(data, 'AAPL')

# Save to database
backtest_id = db.save_backtest(
    name='ML Strategy AAPL 2023',
    start_date=start_date,
    end_date=end_date,
    symbols=['AAPL'],
    initial_capital=100000,
    strategy='machine_learning',
    parameters={'lookback': 50},
    results=results
)

print(f"Backtest {backtest_id} saved!")
print(f"Return: {results['total_return']:.2%}")
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

### Example 2: Multi-Strategy Ensemble with Alerts

```python
from strategies.advanced import (
    MachineLearningStrategy,
    CombinedStrategy,
    AdaptiveStrategy,
)
from utils.alerts import AlertManager

# Create strategy ensemble
combined = CombinedStrategy()

# Alert manager
alerts = AlertManager()

# Get signal
signal = combined.generate_signals(data, asset)

# Generate alerts based on signal
if signal == 'BUY':
    alerts.signal_alert('AAPL', 'BUY', confidence=0.85)
    print("Alert: Strong BUY signal generated!")

elif signal == 'SELL':
    alerts.signal_alert('AAPL', 'SELL', confidence=0.80)
    print("Alert: Strong SELL signal generated!")

# View all alerts
all_alerts = alerts.get_alerts()
print(f"Total alerts: {len(all_alerts)}")
```

### Example 3: Real-Time Price Monitoring with Alerts

```python
from utils.alerts import AlertManager, RealTimeMonitor

# Create alert system
alert_manager = AlertManager(enable_email=False)

# Create monitor
monitor = RealTimeMonitor(alert_manager)

# Add price targets to monitor
monitor.add_price_target('AAPL', 160.0, direction='above')
monitor.add_price_target('AAPL', 140.0, direction='below')

# In live trading loop:
def check_price_alerts(symbol, current_price):
    monitor.check_price_targets(symbol, current_price)
    
    # Check for new alerts
    alerts = alert_manager.get_alerts(unread_only=True)
    for alert in alerts:
        print(f"NEW ALERT: {alert['message']}")
        alert_manager.mark_read(alert['id'])

# Usage:
# check_price_alerts('AAPL', 159.50)
```

### Example 4: Database-Backed Backtest History

```python
from utils.database import TradingDatabase

db = TradingDatabase('trading.db')

# List all backtests
backtests = db.get_backtests(limit=50)
print(f"Total backtests: {len(backtests)}")

for bt in backtests:
    print(f"- {bt['name']}: {bt['total_return']:.2%} return")

# Get specific backtest details
backtest_id = 1
backtest = db.get_backtest(backtest_id)
trades = db.get_backtest_trades(backtest_id)

print(f"Backtest: {backtest['name']}")
print(f"Trades: {len(trades)}")

for trade in trades[:5]:
    print(f"  {trade['symbol']} {trade['side']} @ ${trade['price']}")

# Delete old backtest
db.delete_backtest(5)
```

### Example 5: Risk-Adjusted Portfolio

```python
from portfolio.risk_manager import RiskManager
from portfolio.manager import PortfolioManager

# Initialize with risk constraints
risk_manager = RiskManager(
    max_position_size=0.1,          # 10% per position
    max_sector_exposure=0.3,        # 30% per sector
    max_daily_loss=0.05,            # Stop trading if -5% today
    position_stop_loss=0.02         # 2% stop per position
)

portfolio = PortfolioManager(100000, risk_manager=risk_manager)

# Before entering trade:
portfolio_value = portfolio.total_value
max_position_size = risk_manager.get_max_trade_size(portfolio_value)
position_qty = max_position_size / current_price

print(f"Max position: ${max_position_size:.2f}")
print(f"Max shares: {position_qty:.0f}")

# Check daily loss limit
daily_pnl = portfolio.daily_pnl()
if not risk_manager.check_daily_loss_limit(daily_pnl, portfolio_value):
    print("⚠️  Daily loss limit reached - STOP TRADING")

# Check all constraints
violations = risk_manager.check_all_constraints(
    portfolio.positions,
    portfolio_value,
    daily_pnl
)

if not violations:
    print("✓ All risk constraints satisfied")
else:
    for violation in risk_manager.violations:
        print(f"⚠️  {violation}")
```

### Example 6: Enhanced GUI with Charts

Update `gui/app.py` to use the new endpoints:

```javascript
// Frontend - Get interactive chart
async function loadChart() {
    const response = await fetch('/api/chart/price', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            symbol: 'AAPL',
            period: '1y'
        })
    });
    
    const data = await response.json();
    Plotly.newPlot('chart', JSON.parse(data.chart).data, 
                   JSON.parse(data.chart).layout);
}

// Get risk report
async function loadRiskReport() {
    const response = await fetch('/api/risk/report');
    const report = await response.json();
    
    console.log('Risk Report:', report);
    console.log('Trading allowed:', report.trading_allowed);
    console.log('Violations:', report.violations);
}

// Get alerts
async function loadAlerts() {
    const response = await fetch('/api/alerts');
    const data = await response.json();
    
    console.log(`Alerts: ${data.total_count} (${data.unread_count} unread)`);
    data.alerts.forEach(alert => {
        console.log(`- [${alert.priority}] ${alert.message}`);
    });
}
```

## Architecture Integration

### System Flow with Advanced Features

```
Data Layer (Market Data)
    ↓
Strategies Layer (5 advanced strategies)
    ├→ MachineLearningStrategy (trained model)
    ├→ EnhancedMeanReversionStrategy (multi-indicator)
    ├→ VolatilityBreakoutStrategy (ATR-based)
    ├→ CombinedStrategy (voting ensemble)
    └→ AdaptiveStrategy (market regime)
    ↓
Portfolio Layer (with Risk Management)
    ├→ RiskManager (constraints & limits)
    ├→ Position Sizing (risk-adjusted)
    └→ Stop-Loss Calculation
    ↓
Backtesting Engine
    ├→ Simulation
    └→ Performance Metrics
    ↓
Database Layer (Persistence)
    ├→ Save Backtests
    ├→ Save Trades
    ├→ Store Alerts
    └→ Query History
    ↓
Alert System (Notifications)
    ├→ Price Alerts
    ├→ Signal Alerts
    ├→ Risk Alerts
    └→ Email Notifications
    ↓
GUI & API (Flask)
    ├→ Interactive Charts (Plotly)
    ├→ Risk Dashboard
    ├→ Alert Dashboard
    ├→ Backtest History
    └→ Export Reports
```

## Common Workflows

### Workflow 1: Run Complete Backtest Pipeline

```python
# 1. Fetch data
data = market_data.fetch_stock_data('AAPL', start_date, end_date)
data = market_data.calculate_indicators(data)

# 2. Create strategy
strategy = MachineLearningStrategy()

# 3. Apply risk management
risk_manager = RiskManager()

# 4. Run backtest
engine = BacktestEngine(100000, strategy, risk_manager)
results = engine.run(data, 'AAPL')

# 5. Save results
db.save_backtest(..., results=results)

# 6. Generate alerts
if results['total_return'] > 0.20:
    alerts.performance_alert('High Return', results['total_return'], 0.15)

# 7. Export report
trades = db.get_backtest_trades(backtest_id)
df = pd.DataFrame(trades)
df.to_csv('backtest_report.csv')
```

### Workflow 2: Live Trading with Real-Time Alerts

```python
# 1. Initialize
strategy = AdaptiveStrategy()
risk_manager = RiskManager()
alerts = AlertManager(enable_email=True)
monitor = RealTimeMonitor(alerts)

# 2. Set up monitoring
monitor.add_price_target('AAPL', 160, direction='above')

# 3. Main trading loop
while trading:
    # Get current data
    data = get_live_data('AAPL')
    
    # Generate signal
    signal = strategy.generate_signals(data, asset)
    
    # Check alerts
    current_price = data.iloc[-1]['close']
    monitor.check_price_targets('AAPL', current_price)
    
    # Check risk constraints
    daily_pnl = portfolio.daily_pnl()
    if not risk_manager.check_daily_loss_limit(daily_pnl, portfolio_value):
        print("STOP LOSS HIT - Exiting all positions")
        break
    
    # Execute trade if signal
    if signal == 'BUY':
        position_size = risk_manager.get_max_trade_size(portfolio_value)
        execute_trade('BUY', position_size / current_price)
        alerts.signal_alert('AAPL', 'BUY')
    
    # Check for unread alerts
    unread_alerts = alerts.get_alerts(unread_only=True)
    for alert in unread_alerts:
        send_notification(alert['message'])
        alerts.mark_read(alert['id'])
```

## Performance Optimization Tips

1. **ML Strategy**: Use 50-100 lookback periods for best accuracy
2. **Database Queries**: Limit to 50 backtests per query
3. **Alerts**: Use database persistence for historical analysis
4. **Charts**: Cache Plotly figures for faster rendering
5. **Risk Manager**: Pre-calculate thresholds once, reuse throughout

## Troubleshooting

### Database Not Initializing
```python
# Check if database exists
import os
if os.path.exists('trading.db'):
    print("Database exists")

# Recreate schema
db = TradingDatabase('trading.db')
db.init_database()
```

### ML Strategy Not Training
```python
# Check if model is trained
if strategy.is_trained:
    print("Model is trained")
else:
    print("Need more data:", len(data), "rows (need >", 
          strategy.lookback + 50, ")")
```

### Alerts Not Sending (Email)
```python
# Test email configuration
try:
    alerts.configure_email(
        smtp_server='smtp.gmail.com',
        sender_email='your-email@gmail.com',
        sender_password='app-password',  # Use app password, not regular password
        recipient_email='alert@example.com'
    )
except Exception as e:
    print(f"Email config error: {e}")
```

## Next Steps

1. **Deploy to Production**: Use Gunicorn + Nginx
2. **Add Authentication**: Implement user login
3. **Live Broker Integration**: Connect to Alpaca, TD Ameritrade
4. **Mobile App**: Build React Native app
5. **Advanced Monitoring**: Add Prometheus metrics
6. **Scheduled Backtests**: Use APScheduler for automation

