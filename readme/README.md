# Stock Options Trading System

A comprehensive quantitative trading platform for multi-asset trading with support for stocks and options.

## Features

### Core Components

- **Multi-Asset Support**: Trade stocks and options contracts
- **Multiple Strategies**: 
  - Momentum Strategy (MACD, RSI, Moving Averages)
  - Mean Reversion Strategy (Bollinger Bands)
  - Statistical Arbitrage (Z-score based)
  
- **Backtesting Engine**: Full historical backtesting with:
  - Commission simulation
  - Position tracking
  - Performance metrics
  
- **Paper Trading**: Simulated trading with real market data
- **Portfolio Management**: Complete P&L tracking and metrics
- **Risk Analytics**: Sharpe ratio, max drawdown, win rate

### Technical Indicators

- Simple Moving Averages (SMA)
- Exponential Moving Average (EMA)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- Average True Range (ATR)
- Volume Analysis

### Performance Metrics

- Total Return & Return %
- Realized & Unrealized P&L
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Trade Count
- Portfolio History

## Project Structure

```
StockOptionsTrader/
├── core/                 # Core data models
│   └── models.py        # Asset, Order, Position, Trade classes
├── data/                # Market data handling
│   └── market_data.py   # Data fetching and indicators
├── strategies/          # Trading strategies
│   └── base.py         # Strategy implementations
├── portfolio/           # Portfolio management
│   └── manager.py      # Portfolio tracking
├── backtesting/         # Backtesting framework
│   └── backtest_engine.py
├── brokers/            # Broker integrations
│   └── paper_trader.py # Simulated trading
├── utils/              # Utility functions
├── main.py            # Main entry point
└── requirements.txt   # Dependencies
```

## Installation

```bash
# Clone or navigate to repository
cd StockOptionsTrader

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Single Strategy Backtest

```python
from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy

strategy = MomentumStrategy()
backtester = BacktestEngine(strategy, initial_capital=100000)

results = backtester.run(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)

print(results['summary'])
```

### 2. Compare Multiple Strategies

```python
from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy, MeanReversionStrategy

strategies = [
    ('Momentum', MomentumStrategy()),
    ('Mean Reversion', MeanReversionStrategy()),
]

for name, strategy in strategies:
    backtester = BacktestEngine(strategy)
    results = backtester.run(['AAPL'], '2023-01-01', '2024-01-01')
    print(f"{name}: {results['summary']['total_return_pct']:.2f}%")
```

### 3. Paper Trading

```python
from brokers.paper_trader import PaperTrader
from core.models import Asset, AssetType, OrderType

trader = PaperTrader(initial_capital=100000)

# Place a buy order
asset = Asset('AAPL', AssetType.STOCK)
order_id = trader.place_order(asset, OrderType.BUY, 100, 150.00)

# Check portfolio status
status = trader.get_portfolio_status()
print(status)
```

### 4. Run Main Script

```bash
python main.py
```

This will run example backtests and strategy comparisons.

## Strategy Details

### Momentum Strategy
- **Signal**: BUY when MACD crosses above signal line + price > SMA50
- **Signal**: SELL when MACD crosses below signal line or RSI > 70
- **Use Case**: Trending markets

### Mean Reversion Strategy
- **Signal**: BUY when price touches lower Bollinger Band + RSI < 30
- **Signal**: SELL when price touches upper Bollinger Band + RSI > 70
- **Use Case**: Range-bound markets

### Statistical Arbitrage
- **Signal**: BUY when Z-score < -2.0 (oversold)
- **Signal**: SELL when Z-score > 2.0 (overbought)
- **Use Case**: Mean reversion opportunities

## Key Classes

### Asset
```python
asset = Asset(
    symbol='AAPL',
    asset_type=AssetType.STOCK,
    strike_price=None,  # For options
    expiration_date=None  # For options
)
```

### Position
```python
position = Position(
    asset=asset,
    quantity=100,
    avg_entry_price=150.50,
    current_price=155.20,
    timestamp=datetime.now()
)

print(position.pnl())       # Unrealized P&L in dollars
print(position.pnl_pct())   # Unrealized P&L %
```

### Trade
```python
trade = Trade(
    asset=asset,
    entry_price=150.50,
    exit_price=155.20,
    quantity=100,
    entry_time=datetime(2024, 1, 1),
    exit_time=datetime(2024, 1, 10)
)

print(trade.pnl)      # Realized P&L
print(trade.pnl_pct)  # Realized P&L %
```

## Configuration

### Backtest Parameters
- `initial_capital`: Starting cash (default: 100000)
- `commission`: Trading commission % (default: 0.001 = 0.1%)
- `position_size`: % of portfolio per trade (default: 0.1 = 10%)

### Strategy Parameters
- `z_score_threshold`: Statistical arbitrage threshold (default: 2.0)
- `rsi_overbought`: RSI overbought level (default: 70)
- `rsi_oversold`: RSI oversold level (default: 30)

## Data Sources

- **Market Data**: OpenBB Open Data Platform providers
- **Options Data**: (Ready for integration)

## Performance Optimization

- Vectorized indicator calculations
- Caching of downloaded data
- Efficient position tracking
- Minimal object allocation in loops

## Future Enhancements

- [ ] Options greeks calculation
- [ ] Live trading with real brokers
- [ ] Machine learning strategy optimization
- [ ] Portfolio hedging strategies
- [ ] Risk management framework
- [ ] Real-time alerts and notifications
- [ ] Performance attribution analysis

## Risk Disclaimer

This system is for educational purposes. Past performance does not guarantee future results. 
Always use proper risk management and never trade with capital you can't afford to lose.

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests with improvements.

## Contact

For issues, questions, or suggestions, please create an issue.
