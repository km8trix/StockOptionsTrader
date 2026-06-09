# OpenBB ODP Migration Guide

## Overview
This project has been refactored to use **OpenBB Open Data Platform (ODP)** for market data retrieval.

## Why OpenBB ODP?

**OpenBB ODP provides several advantages:**
- **Multiple data providers**: Access to FMP, Intrinio, Polygon, Tiingo, and more
- **Enterprise-grade data**: Higher quality, more reliable data sources
- **Unified API**: Consistent interface across different data providers
- **Real-time support**: Live market data capabilities
- **Alternative data**: Access to non-traditional financial data sources
- **Active development**: Regular updates and improvements

OpenBB itself is open source and free to install from the OpenBB-finance repository or PyPI. Individual data providers exposed through OpenBB may still require API keys or have provider-specific limits.

## Key Changes

### 1. Dependencies Update
- **Added**: `openbb==4.5.0`

Update requirements:
```bash
pip install -r requirements.txt
```

### 2. MarketDataHandler Refactoring

#### New Features
- **Multi-provider support**: Automatically tries multiple ODP providers (FMP, Intrinio, Polygon, Tiingo)
- **Graceful failure**: Returns an empty DataFrame if OpenBB providers are unavailable
- **Better error handling**: Improved logging and error messages
- **Provider discovery**: Automatically selects working providers

#### Method Signature Changes
The public API remains **unchanged** for backward compatibility:
```python
mdh = MarketDataHandler()
data = mdh.fetch_stock_data('AAPL', '2024-01-01', '2024-01-31')
```

### 3. Available ODP Providers

For equity historical price data, the following providers are available:

| Provider | Best For | API Key Required |
|----------|----------|------------------|
| CBOE | Market data where supported | No/varies |
| TMX | Canadian market data where supported | No/varies |
| FMP | Financial data, detailed metrics | Yes |
| Intrinio | High-quality data, alternative data | Yes |
| Polygon | Crypto and equity data, real-time | Yes |
| Tiingo | End-of-day and intraday data | Yes |
| Alpha Vantage | Equity and technical data | Yes |
| Tradier | Brokerage and market data | Yes |

## Configuration

### Using Different Providers

The `MarketDataHandler` automatically tries configured OpenBB providers and returns an empty DataFrame if none can provide data.

### Setting API Keys

OpenBB ODP uses environment variables for API keys:

```bash
# Create a .env file or set environment variables
export FMP_API_KEY="your-fmp-key"
export INTRINIO_API_KEY="your-intrinio-key"
export POLYGON_API_KEY="your-polygon-key"
export TIINGO_API_KEY="your-tiingo-key"
```

Or create a `.env` file:
```
FMP_API_KEY=your-fmp-key
INTRINIO_API_KEY=your-intrinio-key
POLYGON_API_KEY=your-polygon-key
TIINGO_API_KEY=your-tiingo-key
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Usage Examples

### Basic Usage (No changes required)
```python
from data.market_data import MarketDataHandler

mdh = MarketDataHandler()

# Fetch historical data
data = mdh.fetch_stock_data('AAPL', '2024-01-01', '2024-06-01')

# Calculate indicators
indicators = mdh.calculate_indicators(data.copy())

# Calculate volatility
vol = mdh.calculate_volatility(data)

# Estimate option price
call_price = mdh.estimate_option_price(
    stock_price=150,
    strike=155,
    time_to_expiry=0.25,
    volatility=0.2
)
```

### With Backtesting
```python
from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy

engine = BacktestEngine(MomentumStrategy(), initial_capital=100000)
results = engine.run(['AAPL', 'MSFT'], '2024-01-01', '2024-06-01')
```

## Provider Failure Behavior

The system is designed to fail gracefully when data is unavailable:

1. **If OpenBB providers fail**: Returns an empty DataFrame (gracefully handled)
2. **Cached data**: Previously fetched OpenBB data is cached in memory

## Advantages of OpenBB ODP

### 1. Better Data Quality
- Multiple data sources ensure accuracy
- Real-time data when available
- Historical data back to available dates

### 2. Scalability
- Handle multiple assets efficiently
- Built for enterprise use
- Better rate limiting

### 3. Alternative Data
OpenBB ODP provides access to:
- Economic indicators
- News sentiment
- Options chains
- Cryptocurrency data
- Fixed income data
- And much more!

### 4. Future-Proof
The OpenBB platform is actively developed with regular updates and new features.

## Migration Checklist

- ✅ Updated `requirements.txt`
- ✅ Refactored `MarketDataHandler`
- ✅ Added multi-provider support
- ✅ Maintained backward compatibility
- ✅ Removed direct yfinance fallback
- ✅ Updated tests
- ✅ All system tests pass

## Troubleshooting

### Issue: "No module named 'openbb'"
```bash
pip install openbb==4.5.0
```

### Issue: Provider validation errors
Ensure you're using a provider supported by your installed OpenBB version. Newer OpenBB releases support more providers for `obb.equity.price.historical` than older releases.

### Issue: API rate limits
If you exceed rate limits, the code will automatically try the next provider. Consider:
- Getting API keys from multiple providers
- Implementing request throttling
- Using cached data when available

### Issue: Empty data returned
Possible causes:
1. Invalid symbol
2. API rate limit exceeded
3. No internet connection (uses fallback)
4. Invalid date range

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Documentation

For detailed OpenBB ODP API documentation, visit:
- https://docs.openbb.co/odp/python/reference
- https://docs.openbb.co/odp/python/reference/equity/price/historical

## Support

For issues or questions:
1. Check the OpenBB documentation: https://docs.openbb.co
2. Review the migration guide in this file
3. Check the code comments in `data/market_data.py`

## Version History

### v1.0 (Current)
- Initial OpenBB ODP integration
- Multi-provider support
- Yfinance fallback
- Backward compatible API
