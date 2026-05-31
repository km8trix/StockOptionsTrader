# Quick Start Guide

## Installation

```bash
cd StockOptionsTrader
pip install -r requirements.txt
```

## Verify Installation

```bash
python test_system.py
```

Expected output:
```
✓ All tests passed! System is ready to use.
```

## 5-Minute Tutorial

### 1. Run Example Backtests

```bash
python main.py
```

This runs:
- Momentum strategy backtest on AAPL, MSFT, GOOGL
- Strategy comparison (Momentum, Mean Reversion, Stat Arb)
- Detailed results and metrics

### 2. Run All Examples

```bash
python examples.py
```

Shows:
- Simple backtest
- Strategy comparison
- Technical analysis
- Option pricing
- Paper trading simulation

### 3. Write Your Own Strategy

```python
from strategies.base import Strategy
from datetime import datetime
import pandas as pd

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__("My Custom Strategy")
    
    def generate_signals(self, data: pd.DataFrame, asset) -> str:
        if len(data) < 20:
            return 'HOLD'
        
        # Your trading logic here
        close = data.iloc[-1]['close']
        sma20 = data['close'].rolling(20).mean().iloc[-1]
        
        if close > sma20 * 1.02:  # Price 2% above SMA
            return 'BUY'
        elif close < sma20 * 0.98:  # Price 2% below SMA
            return 'SELL'
        
        return 'HOLD'
```

### 4. Backtest Your Strategy

```python
from backtesting.backtest_engine import BacktestEngine

strategy = MyStrategy()
backtester = BacktestEngine(strategy, initial_capital=100000)

results = backtester.run(
    symbols=['AAPL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

print(f"Total Return: {results['summary']['total_return_pct']:.2f}%")
print(f"Win Rate: {results['summary']['win_rate']:.2f}%")
print(f"Sharpe Ratio: {results['summary']['sharpe_ratio']:.2f}")
```

### 5. Paper Trade Live Data

```python
from brokers.paper_trader import PaperTrader
from core.models import Asset, AssetType, OrderType

trader = PaperTrader(initial_capital=50000)

# Buy Apple
apple = Asset('AAPL', AssetType.STOCK)
trader.place_order(apple, OrderType.BUY, 100, 150.00)

# Check status
status = trader.get_portfolio_status()
print(f"Portfolio Value: ${status['portfolio_value']:.2f}")
print(f"P&L: ${status['unrealized_pnl']:.2f}")
```

## Common Tasks

### Task 1: Backtest Multiple Symbols

```python
from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy

strategy = MomentumStrategy()
backtester = BacktestEngine(strategy, initial_capital=100000)

symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
results = backtester.run(symbols, '2022-01-01', '2024-01-01')
```

### Task 2: Analyze a Stock

```python
from data.market_data import MarketDataHandler

market_data = MarketDataHandler()
data = market_data.fetch_stock_data('TSLA', '2023-01-01', '2023-12-31')

# Calculate indicators
data = market_data.calculate_indicators(data)

# View recent values
print(data[['close', 'rsi', 'macd', 'sma_20', 'sma_50']].tail(10))
```

### Task 3: Price an Option

```python
from data.market_data import MarketDataHandler

market_data = MarketDataHandler()

call_price = market_data.estimate_option_price(
    stock_price=150.0,
    strike=150.0,
    time_to_expiry=30/365,  # 30 days
    volatility=0.25,  # 25%
    option_type='call'
)

print(f"Call Option Price: ${call_price:.2f}")
```

### Task 4: Compare Strategies

```python
from backtesting.backtest_engine import BacktestEngine
from strategies.base import *

strategies = [
    ('Momentum', MomentumStrategy()),
    ('Mean Reversion', MeanReversionStrategy()),
    ('Stat Arb', StatisticalArbitrageStrategy()),
]

results = {}
for name, strategy in strategies:
    backtester = BacktestEngine(strategy)
    results[name] = backtester.run(['AAPL'], '2023-01-01', '2023-12-31')

# Print results
for name, result in results.items():
    print(f"{name}: {result['summary']['total_return_pct']:.2f}%")
```

### Task 5: Optimize Position Sizing

```python
from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy

strategy = MomentumStrategy()
position_sizes = [0.01, 0.05, 0.10, 0.20, 0.50]

for pos_size in position_sizes:
    backtester = BacktestEngine(strategy)
    results = backtester.run(
        ['AAPL'], '2023-01-01', '2023-12-31',
        position_size=pos_size
    )
    print(f"Position Size {pos_size:.1%}: Return {results['summary']['total_return_pct']:.2f}%")
```

## Project Structure at a Glance

```
StockOptionsTrader/
├── main.py                 # Main entry point
├── examples.py             # Example usage
├── test_system.py          # System validation
├── requirements.txt        # Python dependencies
├── README.md              # Full documentation
├── ARCHITECTURE.md        # System design
├── QUICKSTART.md          # This file
│
├── core/
│   └── models.py          # Data models
├── data/
│   └── market_data.py     # Market data & indicators
├── strategies/
│   └── base.py            # Strategy implementations
├── portfolio/
│   └── manager.py         # Portfolio management
├── backtesting/
│   └── backtest_engine.py # Backtesting engine
├── brokers/
│   └── paper_trader.py    # Paper trading
└── utils/                 # Utilities (expandable)
```

## Key Concepts

### Asset
Represents a security (stock or option)
```python
stock = Asset('AAPL', AssetType.STOCK)
call = Asset('AAPL', AssetType.CALL, 
             strike_price=150.0, 
             expiration_date='2024-12-20')
```

### Position
Current holding in an asset
```python
position.quantity        # 100 shares
position.avg_entry_price # $150.25
position.current_price   # $155.50
position.pnl()          # $525.00 (unrealized)
position.pnl_pct()      # 3.48%
```

### Trade
Closed position with realized P&L
```python
trade.entry_price       # $150.25
trade.exit_price        # $155.50
trade.quantity          # 100
trade.pnl               # $525.00
trade.pnl_pct           # 3.48%
```

### Signal
Strategy output: 'BUY', 'SELL', or 'HOLD'

### Portfolio Metrics
- **Total Return %**: Percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return (>1.0 is good)
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: % of profitable trades

## Troubleshooting

### Q: "No module named 'core'"
**A:** Make sure you're running from the project root:
```bash
cd StockOptionsTrader
python examples.py
```

### Q: "Error fetching data for AAPL"
**A:** Check internet connection. Yahoo Finance might be rate-limiting. Try:
```python
import time
time.sleep(1)  # Wait between requests
```

### Q: "Portfolio value is decreasing without trading"
**A:** This shouldn't happen in backtesting. Check:
- Commission is realistic (0.001 = 0.1%)
- Position sizing not too large
- Strategy signals are reasonable

### Q: Can I trade options?
**A:** Yes! Create option assets:
```python
from core.models import Asset, AssetType

put = Asset('SPY', AssetType.PUT, 
            strike_price=400.0,
            expiration_date='2024-12-20')
```

## Performance Tips

1. **Cache Data**: System caches downloaded data automatically
2. **Vectorize**: Use NumPy/pandas for calculations
3. **Reduce Slippage**: Set realistic commission rates
4. **Backtest Smaller Universes**: Start with 5-10 symbols

## Next Steps

1. **Read ARCHITECTURE.md** for system design details
2. **Read README.md** for comprehensive documentation
3. **Study examples.py** for implementation patterns
4. **Create custom strategies** for your trading ideas
5. **Backtest thoroughly** before deploying

## Additional Resources

- **Technical Indicators**: Learn about MACD, RSI, Bollinger Bands
- **Options Pricing**: Black-Scholes model explained
- **Portfolio Theory**: Modern Portfolio Theory, Sharpe Ratio
- **Trading Strategies**: Common patterns and techniques

## Support

For issues, questions, or contributions:
1. Check existing documentation
2. Review example code
3. Create an issue with details

Good luck with your trading system!
