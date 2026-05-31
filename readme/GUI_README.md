# Trading System Web GUI

A modern, responsive web interface for the Stock Options Trading System built with Flask and Bootstrap.

## Features

### 🎯 Backtesting Module
- Test strategies on historical data
- Support for multiple symbols
- Select from 3 built-in strategies
- Customize date range, capital, position size
- View detailed results with metrics

### 💰 Paper Trading Module
- Simulate trading with virtual money
- Place buy/sell orders
- Track portfolio in real-time
- Monitor open positions
- Auto-update portfolio status

### 📊 Technical Analysis Module
- Analyze any stock symbol
- View technical indicators
- Price options using Black-Scholes

## Installation

### 1. Install Dependencies

```bash
cd StockOptionsTrader
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python run_gui.py
```

### 3. Open in Browser

```
http://localhost:5000
```

## Usage

### Backtesting

1. Go to **Backtest** page
2. Enter symbols (comma-separated): AAPL, MSFT
3. Select strategy: Momentum, Mean Reversion, or Stat Arb
4. Set date range
5. Configure capital and position size
6. Click **Run Backtest**
7. View results

### Paper Trading

1. Go to **Paper Trade** page
2. Create trader with initial capital
3. Place buy/sell orders
4. Monitor portfolio status
5. Track positions and P&L

### Analysis

1. Go to **Analysis** page
2. Enter stock symbol
3. View indicators or price options
4. Try Black-Scholes calculator

## Project Structure

```
gui/
├── app.py                    # Flask application
├── templates/
│   ├── base.html            # Base template
│   ├── index.html           # Home page
│   ├── backtest.html        # Backtesting page
│   ├── paper_trade.html     # Paper trading page
│   └── analysis.html        # Analysis page
└── static/
    ├── css/style.css        # Styles
    └── js/utils.js          # Utilities
```

## API Endpoints

```
GET  /                          Home page
GET  /backtest                  Backtest page
POST /api/backtest              Run backtest
GET  /paper_trade               Paper trading page
POST /api/paper_trader/create   Create trader
POST /api/paper_trader/<id>/order  Place order
GET  /api/paper_trader/<id>/status Get status
GET  /analysis                  Analysis page
GET  /api/analyze/<symbol>      Analyze stock
POST /api/price_option          Price option
```

## Running

### Development

```bash
python run_gui.py
```

Opens at http://localhost:5000

### Production

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 gui.app:app
```

## Troubleshooting

**Port Already in Use:**
```bash
# Linux/Mac: Change port in run_gui.py or app.py
# Windows: Use different port
```

**Import Errors:**
```bash
# Make sure in project root
cd StockOptionsTrader
python run_gui.py
```

**Data Not Loading:**
- Check internet connection
- Verify symbol exists (e.g., AAPL)
- Check date range validity
- Look at browser console (F12)

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Future Enhancements

- User authentication
- Save/load backtests
- Real-time charts
- Email notifications
- Mobile app

## Support

For issues, check:
1. Browser console (F12)
2. Server logs
3. Dependencies installed
4. Internet connection
