# Trading System Architecture

## System Overview

This is a quantitative multi-asset trading system designed for stocks and options trading. It provides:

- **Backtesting**: Test strategies on historical data
- **Paper Trading**: Simulate trades with real market data
- **Live Trading**: Integration point for real brokers
- **Analytics**: Comprehensive performance metrics

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Strategies   │    │ Data Sources │    │  Brokers     │  │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤  │
│  │ • Momentum   │    │ • Yahoo Fin  │    │ • Paper      │  │
│  │ • Mean Rev   │    │ • (Live API) │    │ • (Live)     │  │
│  │ • Stat Arb   │    │ • (Options)  │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │         │
│         └────────────────────┼────────────────────┘         │
│                              │                              │
│                    ┌─────────────────────┐                 │
│                    │  Backtester Engine  │                 │
│                    │  Paper Trader       │                 │
│                    └──────────┬──────────┘                  │
│                               │                             │
│                    ┌──────────────────────┐                │
│                    │ Portfolio Manager    │                │
│                    │ - Positions          │                │
│                    │ - Cash               │                │
│                    │ - P&L Tracking       │                │
│                    └──────────┬───────────┘                │
│                               │                             │
│                    ┌──────────────────────┐                │
│                    │  Analytics & Reports │                │
│                    │ - Metrics            │                │
│                    │ - Performance        │                │
│                    │ - Risk Analysis      │                │
│                    └──────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### Core (`core/models.py`)

Defines fundamental trading objects:

- **Asset**: Represents a tradeable security (stock or option)
- **Order**: Represents a trading order
- **Position**: Current holding in an asset
- **Trade**: Completed trade with P&L
- **Enums**: AssetType, OrderType, OrderStatus

### Data (`data/market_data.py`)

Handles market data:

- Fetches historical prices from Yahoo Finance
- Caches data to avoid redundant downloads
- Calculates technical indicators
- Estimates option prices using Black-Scholes
- Computes volatility metrics

**Technical Indicators Calculated:**
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- Average True Range (ATR)
- Volume indicators

### Strategies (`strategies/base.py`)

Trading strategy implementations:

1. **Momentum Strategy**
   - Uses MACD crossovers and RSI
   - Buys on bullish MACD + price above SMA50
   - Sells on bearish MACD or RSI > 70

2. **Mean Reversion Strategy**
   - Uses Bollinger Bands and RSI
   - Buys at lower band with RSI < 30
   - Sells at upper band with RSI > 70

3. **Statistical Arbitrage**
   - Uses Z-score of price deviation
   - Buys when price is 2σ below mean
   - Sells when price is 2σ above mean

All strategies inherit from abstract `Strategy` class and implement `generate_signals()`.

### Portfolio (`portfolio/manager.py`)

Manages portfolio state:

- **Positions**: Tracks all open positions
- **Cash**: Maintains cash balance
- **Trades**: Records closed trades
- **Metrics**: Calculates P&L, Sharpe ratio, drawdown, win rate

**Key Methods:**
- `add_position()`: Open new position
- `close_position()`: Close position and record trade
- `get_portfolio_value()`: Total portfolio worth
- `get_portfolio_pnl()`: Unrealized P&L
- `get_summary()`: Complete portfolio snapshot

### Backtesting (`backtesting/backtest_engine.py`)

Simulates trading on historical data:

- Iterates through dates
- Generates signals for each symbol
- Executes trades based on signals
- Tracks performance metrics
- Generates detailed reports

**Backtesting Features:**
- Commission simulation
- Accurate price-based execution
- Position averaging
- Multi-asset support
- Complete trade logging

### Paper Trading (`brokers/paper_trader.py`)

Simulates live trading:

- Places orders at specified prices
- Fetches current market prices
- Processes orders when prices match
- Tracks portfolio in real-time
- Ready for live broker integration

### Main Entry Point (`main.py`)

Orchestrates system:

- Runs single strategy backtests
- Compares multiple strategies
- Displays results and metrics
- Generates comparison tables

## Data Flow

### Backtesting Flow

```
1. User initiates backtest with:
   - Strategy
   - Symbols list
   - Date range
   - Position size

2. BacktestEngine:
   - Fetches historical data for all symbols
   - Aligns dates across assets
   - Iterates through each date

3. For each date:
   - Strategy generates signal for each symbol
   - BacktestEngine executes trades based on signal
   - Portfolio is updated
   - Snapshot is recorded

4. Post-backtest:
   - Calculate performance metrics
   - Close remaining positions
   - Generate report with trades and statistics

5. Report includes:
   - Total return
   - Sharpe ratio
   - Max drawdown
   - Win rate
   - Trade history
```

### Paper Trading Flow

```
1. Initialize PaperTrader with capital

2. Place orders:
   - BUY/SELL orders with limit prices
   - Orders queued as PENDING

3. Each timestep:
   - Fetch current market prices
   - Check if any orders can execute
   - Execute if price threshold met
   - Update portfolio positions

4. Monitor:
   - Current portfolio value
   - Unrealized P&L
   - Position details
   - Order status
```

## Performance Metrics

### Risk-Adjusted Returns

- **Sharpe Ratio**: Risk-adjusted return metric
  - Formula: (Mean Return - Risk-Free Rate) / Std Dev
  - Annualized (×√252)

- **Maximum Drawdown**: Largest peak-to-trough decline
  - Shows worst-case scenario
  - Important for risk management

### Trade Statistics

- **Total Return %**: Portfolio gain/loss percentage
- **Win Rate**: % of trades that were profitable
- **Realized P&L**: Profit from closed trades
- **Unrealized P&L**: Profit from open positions

### Position Metrics

- **P&L**: Absolute profit/loss in dollars
- **P&L %**: Profit/loss percentage
- **Entry/Exit Prices**: Trade execution prices

## Extensibility Points

### Adding New Strategies

```python
from strategies.base import Strategy

class MyStrategy(Strategy):
    def generate_signals(self, data, asset):
        # Implement your logic
        return 'BUY' or 'SELL' or 'HOLD'
```

### Adding New Data Sources

```python
def fetch_stock_data(self, symbol, start_date, end_date):
    # Configure an OpenBB provider or add your own API adapter
    # Return DataFrame with OHLCV data
```

### Adding Live Broker Support

```python
class LiveBroker:
    def place_order(self, asset, order_type, quantity, price):
        # Use broker API to place actual orders
        pass
    
    def get_positions(self):
        # Fetch actual positions from broker
        pass
```

## Configuration Parameters

### Backtest Engine
- `initial_capital`: Starting cash
- `commission`: Trading costs (%)
- `position_size`: Position sizing (% of portfolio)

### Strategy Parameters
- `rsi_overbought`: RSI threshold for overbought (default: 70)
- `rsi_oversold`: RSI threshold for oversold (default: 30)
- `z_score_threshold`: Z-score threshold for stat arb (default: 2.0)

### Technical Indicators
- `sma_windows`: [20, 50] day periods
- `ema_windows`: [12, 26] day periods
- `rsi_window`: 14 days
- `bollinger_window`: 20 days
- `atr_window`: 14 days

## Performance Considerations

1. **Data Caching**: Market data is cached to avoid redundant downloads
2. **Vectorization**: Indicator calculations use NumPy/pandas for speed
3. **Position Tracking**: Efficient dictionary-based position management
4. **Minimal Allocations**: Reuse objects to reduce GC pressure

## Future Enhancements

1. **Options Trading**
   - Full Greeks calculations (delta, gamma, vega, theta)
   - Spread strategies (straddles, strangles, spreads)
   - Exercise/assignment handling

2. **Advanced Strategies**
   - Machine learning predictions
   - Ensemble methods
   - Reinforcement learning optimization

3. **Risk Management**
   - Stop-loss orders
   - Take-profit targets
   - Portfolio hedging
   - VaR calculations

4. **Live Trading**
   - Real broker integration (IB, Alpaca, TD Ameritrade)
   - Real-time feeds
   - Execution optimization

5. **Analytics**
   - Performance attribution
   - Factor analysis
   - Scenario analysis
   - Monte Carlo simulations

## Class Diagrams

### Core Models

```
Asset
├── symbol: str
├── asset_type: AssetType
├── strike_price: Optional[float]
└── expiration_date: Optional[str]

Position
├── asset: Asset
├── quantity: int
├── avg_entry_price: float
├── current_price: float
├── pnl(): float
└── pnl_pct(): float

Trade
├── asset: Asset
├── entry_price: float
├── exit_price: float
├── quantity: int
├── pnl: float (property)
└── pnl_pct: float (property)

Order
├── asset: Asset
├── order_type: OrderType
├── quantity: int
├── price: float
└── status: OrderStatus
```

### Key Classes

```
Strategy (abstract)
├── name: str
├── generate_signals(data, asset): str
└── analyze(symbol, dates): DataFrame

MomentumStrategy
└── generate_signals()

PortfolioManager
├── positions: Dict[Asset, Position]
├── cash: float
├── closed_trades: List[Trade]
├── get_portfolio_value(): float
├── get_portfolio_pnl(): float
└── get_summary(): Dict

BacktestEngine
├── strategy: Strategy
├── portfolio: PortfolioManager
├── run(symbols, dates): Dict
└── _execute_signal()

PaperTrader
├── portfolio: PortfolioManager
├── pending_orders: List[Order]
├── place_order(): str
└── process_orders()
```

## Testing Strategy

### Unit Tests
- Model creation and calculations
- Indicator calculations
- P&L computations

### Integration Tests
- Backtest execution
- Strategy signal generation
- Order processing

### System Tests
- Multi-symbol backtests
- Strategy comparisons
- Performance metrics validation
