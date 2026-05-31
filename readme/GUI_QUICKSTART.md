# 🌐 Trading System GUI - Quick Start

## Installation (2 minutes)

```bash
cd StockOptionsTrader
pip install -r requirements.txt
```

## Start GUI (1 minute)

```bash
python run_gui.py
```

You should see:
```
Stock Options Trading System - Web GUI
========================================================================

🌐 Starting Flask server...

📱 Open your browser and go to: http://localhost:5000
📱 Press CTRL+C to stop the server
```

## Open in Browser

Go to: **http://localhost:5000**

## Quick Tasks

### 1. Backtest a Strategy (3 min)

1. Click **Backtest** in navigation
2. Enter symbols: `AAPL, MSFT`
3. Select strategy: `Momentum`
4. Click **Run Backtest**
5. View results:
   - Total Return: XXX%
   - Sharpe Ratio: X.XX
   - Win Rate: XX%

### 2. Paper Trade (3 min)

1. Click **Paper Trade**
2. Create trader:
   - Trader ID: `trader1`
   - Capital: `$50,000`
   - Click **Create Trader**
3. Place order:
   - Symbol: `AAPL`
   - Action: `BUY`
   - Quantity: `100`
   - Price: `$150`
   - Click **Place Order**
4. Watch portfolio update every 5 seconds

### 3. Analyze Stock (2 min)

1. Click **Analysis**
2. Analyze stock:
   - Symbol: `AAPL`
   - Click **Analyze**
   - View indicators
3. Price option:
   - Stock Price: `$150`
   - Strike: `$150`
   - Days: `30`
   - Volatility: `25%`
   - Click **Price Option**

## Pages Overview

| Page | Purpose | Time |
|------|---------|------|
| **Home** | Overview & features | 1 min |
| **Backtest** | Test strategies | 5 min |
| **Paper Trade** | Simulate trading | 5 min |
| **Analysis** | Analyze stocks & options | 5 min |

## What Each Page Does

### 🏠 Home
- Overview of system
- Features list
- Quick links to other pages

### 📊 Backtest
- Test trading strategies on historical data
- Select from 3 strategies
- Customize parameters
- View performance metrics

### 💰 Paper Trade
- Create trading accounts
- Place simulated orders
- Monitor portfolio
- Track positions and P&L

### 📈 Analysis
- Analyze stock technical indicators
- Calculate option prices
- View current metrics

## Example Backtest

**Setup:**
- Symbols: AAPL, MSFT, GOOGL
- Strategy: Momentum
- Date: 2023-01-01 to 2024-01-01
- Capital: $100,000
- Position: 10%

**View Results:**
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Closed Trades

## Example Paper Trade

**Setup:**
1. Create trader (trader1, $50,000)
2. Buy 100 AAPL @ $150
3. Buy 50 MSFT @ $350

**Monitor:**
- Portfolio Value
- Cash Available
- Unrealized P&L
- Open Positions

## Example Analysis

**Stock Analysis:**
- Enter: AAPL
- View: RSI, MACD, SMA20, SMA50
- See: Current Price

**Option Pricing:**
- Stock Price: $150
- Strike: $150
- Days: 30
- Volatility: 25%
- Get: Call Price, Put Price

## Features

✅ **3 Strategies** - Momentum, Mean Reversion, Stat Arb  
✅ **Backtesting** - Historical data testing  
✅ **Paper Trading** - Simulated live trading  
✅ **Technical Analysis** - Indicators & options pricing  
✅ **Real-time Updates** - 5-second refresh  
✅ **Responsive Design** - Works on desktop & mobile  
✅ **No Database** - Everything in memory  

## Troubleshooting

**Can't start server?**
```bash
# Port 5000 already in use? Edit run_gui.py:
# Change app.run(..., port=5001)
```

**No data showing?**
- Check internet connection
- Verify symbol exists (AAPL, MSFT, etc.)
- Check date range
- Look at browser console (F12)

**Slow performance?**
- Use fewer symbols
- Use shorter date ranges
- Close other browser tabs

## Browser Tips

- **Chrome/Edge**: Works best
- **Firefox**: Also works great
- **Mobile**: Responsive, works on phones
- **Dev Tools**: Press F12 for console

## Keyboard Shortcuts

- F12: Open developer console
- Ctrl+Shift+R: Hard refresh (clear cache)
- Ctrl+L: Focus address bar

## API Calls (for developers)

```bash
# Backtest
curl -X POST http://localhost:5000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbols":"AAPL","strategy":"momentum",...}'

# Get strategies
curl http://localhost:5000/api/strategies

# Create trader
curl -X POST http://localhost:5000/api/paper_trader/create \
  -d '{"trader_id":"t1","initial_capital":50000}'

# Analyze stock
curl http://localhost:5000/api/analyze/AAPL

# Price option
curl -X POST http://localhost:5000/api/price_option \
  -d '{"stock_price":150,"strike":150,...}'
```

## Next Steps

1. **Explore Pages**: Visit each page
2. **Run Backtest**: Try different strategies
3. **Paper Trade**: Practice trading
4. **Analyze**: Study technical indicators
5. **Read Docs**: Check GUI_README.md for details

## Stopping Server

Press `CTRL+C` in terminal

## Restarting Server

```bash
python run_gui.py
```

Then refresh browser: F5 or Ctrl+R

## Tips

- Default date is 1 year back
- Commission is 0.1%
- Position size is % of portfolio
- RSI > 70 = overbought, < 30 = oversold
- Sharpe > 1.0 is good

## Support

**Questions?** Check:
1. GUI_README.md (detailed docs)
2. Browser console (F12)
3. Server logs (terminal)

**Issues?** Make sure:
1. Dependencies installed
2. Internet connected
3. Port 5000 free
4. Valid symbols used

## What's Next?

- Customize strategies
- Add new analysis tools
- Deploy to production
- Integrate live broker
- Add authentication

Happy trading! 📈
