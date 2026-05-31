# Advanced Trading System Features

## 🎯 New Capabilities

This enhanced version adds enterprise-grade features to your trading system:

### 1. **Machine Learning Strategies**

#### MachineLearningStrategy
- Uses Random Forest classifier trained on historical price patterns
- Features: Returns, volatility, price-to-average ratio, volume
- Adaptive learning from live data
- Confidence-based position sizing

```python
from strategies.advanced import MachineLearningStrategy

ml_strategy = MachineLearningStrategy(lookback=50)
signal = ml_strategy.generate_signals(data, asset)
```

#### EnhancedMeanReversionStrategy
- Multi-indicator approach (Bollinger Bands + RSI + Volatility)
- Confirmation requirements for stronger signals
- Volume-based trade entry validation

#### VolatilityBreakoutStrategy
- Identifies volatility spikes using ATR
- Directional bias from momentum indicators
- Ideal for high-volatility periods

#### CombinedStrategy
- Voting-based approach combining 4 strategies
- Strongest signals from consensus
- Reduces false positives through ensemble voting

#### AdaptiveStrategy
- Switches between momentum and mean reversion
- Market regime detection (trending vs ranging)
- Dynamic strategy selection

### 2. **Risk Management Module**

```python
from portfolio.risk_manager import RiskManager

risk_manager = RiskManager(
    max_position_size=0.1,        # 10% per position
    max_sector_exposure=0.3,      # 30% per sector
    max_daily_loss=0.05,          # 5% daily limit
    position_stop_loss=0.02       # 2% per position
)

# Check constraints
risk_manager.check_position_size(portfolio_value, position_size)
risk_manager.check_sector_exposure(positions, sector, size, portfolio_value)
risk_manager.check_daily_loss_limit(daily_pnl, portfolio_value)

# Get stop-loss prices
stop_price = risk_manager.calculate_position_stop_loss(entry_price)

# Generate risk report
report = risk_manager.get_report()
```

**Features:**
- Position size limits (max capital per trade)
- Sector concentration limits
- Daily loss cutoffs (circuit breaker)
- Leverage limits
- Automatic stop-loss calculation
- Comprehensive violation reporting

### 3. **Database Persistence**

```python
from utils.database import TradingDatabase

db = TradingDatabase('trading_data.db')

# Save backtest results
backtest_id = db.save_backtest(
    name='ML Strategy Test',
    start_date='2023-01-01',
    end_date='2023-12-31',
    symbols=['AAPL', 'MSFT'],
    initial_capital=100000,
    strategy='machine_learning',
    parameters={'lookback': 50},
    results=backtest_results
)

# Save individual trades
db.save_trade(backtest_id, 'AAPL', 'BUY', 150.0, 100, 10.0, 500.0)

# Retrieve data
backtests = db.get_backtests(limit=50)
trades = db.get_backtest_trades(backtest_id)
backtest = db.get_backtest(backtest_id)

# Delete backtest
db.delete_backtest(backtest_id)
```

**Tables:**
- `backtests` - Backtest metadata and performance metrics
- `trades` - Individual trade details
- `alerts` - Trading alerts and notifications
- `paper_trader_sessions` - Paper trading session state

### 4. **Alerts & Notifications System**

```python
from utils.alerts import AlertManager, AlertType, AlertPriority, RealTimeMonitor

alert_manager = AlertManager(enable_email=True)

# Configure email
alert_manager.configure_email(
    smtp_server='smtp.gmail.com',
    sender_email='your-email@gmail.com',
    sender_password='app-password',
    recipient_email='alert@example.com'
)

# Price alerts
alert_manager.price_alert('AAPL', 150.50, 150.00, 'above')

# Signal alerts
alert_manager.signal_alert('TSLA', 'BUY', confidence=0.92)

# Risk alerts
alert_manager.risk_alert('Portfolio drawdown exceeded limit', 
    {'drawdown': 0.08, 'limit': 0.05})

# Performance alerts
alert_manager.performance_alert('Sharpe Ratio', 1.8, 1.5)

# Real-time monitoring
monitor = RealTimeMonitor(alert_manager)
monitor.add_price_target('AAPL', 155.0, direction='above')
monitor.check_price_targets('AAPL', 155.05)

# Get alerts
alerts = alert_manager.get_alerts(unread_only=True)
alert_manager.mark_read(alert_id)
alert_manager.mark_all_read()
```

**Alert Types:**
- Price alerts - Monitor target prices
- Signal alerts - Trading signal notifications
- Risk alerts - Risk threshold breaches (HIGH PRIORITY)
- Performance alerts - Metric achievements
- Error alerts - System errors (HIGH PRIORITY)

**Priorities:**
- LOW - Informational
- NORMAL - Standard alerts
- HIGH - Requires attention
- CRITICAL - Immediate action needed

### 5. **Enhanced Web GUI**

#### Interactive Charts (Plotly)

- **Price Charts**: Candlestick with SMAs and Bollinger Bands
- **Performance Charts**: Equity curve visualization
- **Technical Indicators**: RSI, MACD, Volume analysis
- **Multi-timeframe**: 1d, 1w, 1mo, 3mo, 1y, 5y

```
GET /api/chart/price?symbol=AAPL&period=1y
GET /api/chart/performance?backtest_id=5
GET /api/chart/indicators?symbol=MSFT
```

#### Backtest Management

- Save backtest results to database
- View backtest history with performance metrics
- Compare multiple backtests
- Delete old backtests

```
GET /api/backtests                     # List all backtests
GET /api/backtest/5                    # Get details
DELETE /api/backtest/5                 # Delete
```

#### Alert Dashboard

- View all alerts with priority levels
- Real-time alert count
- Mark alerts as read
- Filter by type/priority
- Add price target monitors

```
GET /api/alerts
POST /api/alert/5/read
POST /api/price-target
```

#### Risk Management Dashboard

- View current risk settings
- Update position limits, sector exposure, daily loss limits
- Risk violations report
- Risk-adjusted position sizing

```
GET /api/risk/settings
POST /api/risk/settings
GET /api/risk/report
```

#### Export Functionality

- Export backtest results as CSV
- Generate PDF reports
- Download trade history
- Email reports

```
GET /api/export/backtest/5
GET /api/export/report/5
```

#### Strategy Management

- View available strategies
- Strategy descriptions
- Parameter configuration
- Performance comparison

```
GET /api/strategies
```

## 📊 Example: Complete Workflow

```python
from data.market_data import MarketData
from strategies.advanced import MachineLearningStrategy, CombinedStrategy
from portfolio.risk_manager import RiskManager
from backtesting.backtest_engine import BacktestEngine
from utils.database import TradingDatabase
from utils.alerts import AlertManager

# Initialize
market_data = MarketData()
risk_manager = RiskManager(max_daily_loss=0.05)
db = TradingDatabase()
alerts = AlertManager()

# Fetch data
data = market_data.fetch_data('AAPL', period='1y')

# Run backtest with ML strategy
strategy = MachineLearningStrategy()
engine = BacktestEngine(
    initial_capital=100000,
    strategy=strategy,
    risk_manager=risk_manager
)
results = engine.run(data, 'AAPL')

# Save results
backtest_id = db.save_backtest(
    name='ML Strategy v1',
    start_date='2023-01-01',
    end_date='2023-12-31',
    symbols=['AAPL'],
    initial_capital=100000,
    strategy='machine_learning',
    parameters={'lookback': 50},
    results=results
)

# Generate alerts
if results['sharpe_ratio'] > 1.5:
    alerts.performance_alert('Sharpe Ratio', results['sharpe_ratio'], 1.5)

print(f"Backtest saved: ID {backtest_id}")
print(f"Return: {results['total_return']:.2%}")
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

## 🚀 Advanced Usage

### Multi-Strategy Backtesting

```python
strategies = {
    'momentum': MomentumStrategy(),
    'ml': MachineLearningStrategy(),
    'combined': CombinedStrategy(),
    'adaptive': AdaptiveStrategy()
}

results_comparison = {}
for name, strategy in strategies.items():
    engine = BacktestEngine(100000, strategy, risk_manager)
    results = engine.run(data, 'AAPL')
    results_comparison[name] = results
    
    db.save_backtest(
        name=f'Backtest {name}',
        # ... other params
        results=results
    )
```

### Real-Time Monitoring

```python
from utils.alerts import RealTimeMonitor

monitor = RealTimeMonitor(alerts)

# Monitor multiple symbols
for symbol in ['AAPL', 'MSFT', 'TSLA']:
    monitor.add_price_target(symbol, target_price=target, direction='above')

# In live trading loop:
for symbol in symbols:
    current_price = market_data.get_latest_price(symbol)
    monitor.check_price_targets(symbol, current_price)
```

### Risk-Adjusted Position Sizing

```python
from portfolio.manager import PortfolioManager

portfolio = PortfolioManager(initial_capital=100000, risk_manager=risk_manager)

# Position size limited by risk manager
max_size = risk_manager.get_max_trade_size(portfolio.total_value)

# Position sized to risk tolerance
position_qty = max_size / current_price
```

## 📈 Performance Tips

1. **ML Strategy**: Use 50-100 period lookback for optimal training
2. **Volatility Strategy**: Works best in choppy markets
3. **Combined Strategy**: Best for consistent returns, reduces whipsaws
4. **Adaptive Strategy**: Best for varying market conditions

## 🔧 Configuration

```python
# Risk Manager Configuration
risk_manager = RiskManager(
    max_position_size=0.1,           # 10% per position
    max_sector_exposure=0.3,         # 30% per sector
    max_correlation=0.8,             # 0.8 max correlation
    max_leverage=1.0,                # 1x leverage
    max_daily_loss=0.05,             # 5% daily stop
    position_stop_loss=0.02          # 2% stop-loss
)

# Alert Manager with Email
alerts = AlertManager(enable_email=True)
alerts.configure_email(
    smtp_server='smtp.gmail.com',
    sender_email='your-email@gmail.com',
    sender_password='your-app-password',
    recipient_email='alert@example.com'
)
```

## 📝 Database Schema

```sql
-- Backtests
CREATE TABLE backtests (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    timestamp DATETIME,
    start_date TEXT,
    end_date TEXT,
    symbols TEXT,           -- JSON
    initial_capital REAL,
    strategy TEXT,
    parameters TEXT,        -- JSON
    total_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    total_trades INTEGER,
    results TEXT            -- JSON
);

-- Trades
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    backtest_id INTEGER,
    timestamp DATETIME,
    symbol TEXT,
    side TEXT,
    price REAL,
    quantity REAL,
    commission REAL,
    pnl REAL,
    pnl_percent REAL
);

-- Alerts
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    alert_type TEXT,
    symbol TEXT,
    message TEXT,
    priority TEXT,
    is_read INTEGER,
    details TEXT            -- JSON
);
```

## 🔐 Security Notes

- Store email credentials in environment variables
- Use app-specific passwords for Gmail
- Never commit credentials to version control
- Database file should be in .gitignore

## 📚 Next Steps

1. Deploy with gunicorn + nginx
2. Add user authentication
3. Integrate live broker APIs
4. Add machine learning model persistence
5. Build mobile app with React Native

