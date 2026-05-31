"""
Example usage of the trading system
"""

from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy, MeanReversionStrategy, StatisticalArbitrageStrategy
from brokers.paper_trader import PaperTrader
from core.models import Asset, AssetType, OrderType
from data.market_data import MarketDataHandler
from portfolio.manager import PortfolioManager
from datetime import datetime


def example_1_simple_backtest():
    """Example 1: Run a simple backtest"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Momentum Strategy Backtest")
    print("="*70)
    
    strategy = MomentumStrategy()
    backtester = BacktestEngine(strategy, initial_capital=100000)
    
    results = backtester.run(
        symbols=['AAPL', 'MSFT'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        position_size=0.05
    )
    
    print("\nPortfolio Summary:")
    for key, value in results['summary'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def example_2_strategy_comparison():
    """Example 2: Compare multiple strategies"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Strategy Comparison")
    print("="*70)
    
    strategies = {
        'Momentum': MomentumStrategy(),
        'Mean Reversion': MeanReversionStrategy(),
        'Stat Arb': StatisticalArbitrageStrategy(),
    }
    
    symbols = ['GOOGL', 'AMZN']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    results_data = {}
    
    for strat_name, strategy in strategies.items():
        print(f"\nTesting {strat_name}...", end=" ", flush=True)
        backtester = BacktestEngine(strategy, initial_capital=100000)
        results = backtester.run(symbols, start_date, end_date)
        
        if 'error' not in results:
            results_data[strat_name] = results['summary']
            print("✓")
        else:
            print("✗")
    
    # Print comparison
    print("\n" + "-"*70)
    print(f"{'Strategy':<20} {'Return %':>15} {'Sharpe':>12} {'Win Rate %':>12}")
    print("-"*70)
    
    for strat_name, summary in results_data.items():
        return_pct = summary.get('total_return_pct', 0)
        sharpe = summary.get('sharpe_ratio', 0)
        win_rate = summary.get('win_rate', 0)
        
        print(f"{strat_name:<20} {return_pct:>14.2f}% {sharpe:>12.2f} {win_rate:>11.2f}%")


def example_3_market_analysis():
    """Example 3: Technical analysis of a stock"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Market Data & Technical Indicators")
    print("="*70)
    
    market_data = MarketDataHandler()
    
    print("\nFetching data for TSLA...")
    data = market_data.fetch_stock_data('TSLA', '2023-06-01', '2023-08-31')
    
    print(f"Data points: {len(data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Calculate indicators
    data = market_data.calculate_indicators(data)
    
    # Show recent values
    print("\nRecent Technical Indicators:")
    print("-"*70)
    print(f"{'Date':<12} {'Close':>10} {'RSI':>8} {'MACD':>10} {'SMA20':>10} {'SMA50':>10}")
    print("-"*70)
    
    for idx in range(-5, 0):
        row = data.iloc[idx]
        print(f"{data.index[idx].strftime('%Y-%m-%d'):<12} "
              f"{row['close']:>10.2f} {row['rsi']:>8.2f} "
              f"{row['macd']:>10.4f} {row['sma_20']:>10.2f} {row['sma_50']:>10.2f}")


def example_4_option_pricing():
    """Example 4: Estimate option prices"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Option Pricing (Black-Scholes)")
    print("="*70)
    
    market_data = MarketDataHandler()
    
    stock_price = 150.0
    strikes = [140, 145, 150, 155, 160]
    time_to_expiry = 30 / 365  # 30 days
    volatility = 0.25  # 25% implied volatility
    
    print(f"\nStock Price: ${stock_price:.2f}")
    print(f"Time to Expiry: {time_to_expiry*365:.0f} days")
    print(f"Volatility: {volatility*100:.1f}%")
    print("\n" + "-"*70)
    print(f"{'Strike':<10} {'Call Price':>12} {'Put Price':>12}")
    print("-"*70)
    
    for strike in strikes:
        call_price = market_data.estimate_option_price(
            stock_price, strike, time_to_expiry, volatility, 'call'
        )
        put_price = market_data.estimate_option_price(
            stock_price, strike, time_to_expiry, volatility, 'put'
        )
        
        print(f"${strike:<9.2f} ${call_price:>11.2f} ${put_price:>11.2f}")


def example_5_paper_trading():
    """Example 5: Simulate paper trading"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Paper Trading Simulation")
    print("="*70)
    
    trader = PaperTrader(initial_capital=50000)
    
    # Place some orders
    print("\nPlacing orders...")
    
    apple = Asset('AAPL', AssetType.STOCK)
    order1 = trader.place_order(apple, OrderType.BUY, 50, 180.00)
    print(f"  Order {order1}: BUY 50 AAPL @ $180.00")
    
    microsoft = Asset('MSFT', AssetType.STOCK)
    order2 = trader.place_order(microsoft, OrderType.BUY, 30, 350.00)
    print(f"  Order {order2}: BUY 30 MSFT @ $350.00")
    
    # Process orders and check status
    print("\nProcessing orders...")
    status = trader.get_portfolio_status()
    
    print(f"\nPortfolio Status:")
    print(f"  Cash: ${status['cash']:.2f}")
    print(f"  Portfolio Value: ${status['portfolio_value']:.2f}")
    print(f"  Unrealized P&L: ${status['unrealized_pnl']:.2f}")
    print(f"  Positions: {len(status['positions'])}")
    print(f"  Pending Orders: {status['pending_orders']}")
    
    if status['positions']:
        print(f"\n  Positions:")
        for pos in status['positions']:
            print(f"    {pos['symbol']}: {pos['quantity']} @ ${pos['entry_price']:.2f} "
                  f"(Current: ${pos['current_price']:.2f}, P&L: ${pos['pnl']:.2f})")


def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("STOCK OPTIONS TRADING SYSTEM - EXAMPLES")
    print("="*70)
    
    try:
        example_1_simple_backtest()
    except Exception as e:
        print(f"Error in Example 1: {e}")
    
    try:
        example_2_strategy_comparison()
    except Exception as e:
        print(f"Error in Example 2: {e}")
    
    try:
        example_3_market_analysis()
    except Exception as e:
        print(f"Error in Example 3: {e}")
    
    try:
        example_4_option_pricing()
    except Exception as e:
        print(f"Error in Example 4: {e}")
    
    try:
        example_5_paper_trading()
    except Exception as e:
        print(f"Error in Example 5: {e}")
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
