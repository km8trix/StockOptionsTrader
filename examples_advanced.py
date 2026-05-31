"""
Advanced Examples: Machine Learning, Risk Management, Database, and Alerts
"""

from data.market_data import MarketDataHandler
from strategies.advanced import (
    MachineLearningStrategy, EnhancedMeanReversionStrategy,
    VolatilityBreakoutStrategy, CombinedStrategy, AdaptiveStrategy
)
from strategies.base import Asset, MomentumStrategy
from backtesting.backtest_engine import BacktestEngine
from portfolio.risk_manager import RiskManager
from utils.database import TradingDatabase
from utils.alerts import AlertManager, AlertType, AlertPriority
from datetime import datetime, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def example_1_machine_learning_strategy():
    """Example 1: Machine Learning Strategy"""
    print("=" * 60)
    print("EXAMPLE 1: Machine Learning Strategy")
    print("=" * 60)
    
    market_data = MarketDataHandler()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = market_data.fetch_stock_data('AAPL', start_date, end_date)
    data = market_data.calculate_indicators(data)
    
    if data is not None and len(data) > 100:
        strategy = MachineLearningStrategy(lookback=50)
        asset = Asset('AAPL', 'STOCK')
        
        # Generate signals
        print(f"Data points: {len(data)}")
        for i in range(-10, 0):
            signal = strategy.generate_signals(data.iloc[:i], asset)
            print(f"Signal at {data.index[i].strftime('%Y-%m-%d')}: {signal}")
        
        print(f"Model trained: {strategy.is_trained}")
        print()


def example_2_risk_management():
    """Example 2: Risk Management Constraints"""
    print("=" * 60)
    print("EXAMPLE 2: Risk Management")
    print("=" * 60)
    
    risk_manager = RiskManager(
        max_position_size=0.1,        # 10% per position
        max_daily_loss=0.05,          # 5% daily limit
        position_stop_loss=0.02       # 2% stop-loss
    )
    
    portfolio_value = 100000
    position_size = 8000  # 8%
    
    # Check position size
    is_ok = risk_manager.check_position_size(portfolio_value, position_size)
    print(f"Position size ${position_size} ({position_size/portfolio_value:.1%}) OK: {is_ok}")
    
    # Calculate stop-loss
    entry_price = 150.0
    stop_price = risk_manager.calculate_position_stop_loss(entry_price)
    print(f"Entry: ${entry_price:.2f} → Stop: ${stop_price:.2f}")
    
    # Max trade size
    max_size = risk_manager.get_max_trade_size(portfolio_value)
    print(f"Max trade size: ${max_size:,.2f}")
    
    # Daily loss limit
    daily_pnl = -3000  # Lost $3000
    ok = risk_manager.check_daily_loss_limit(daily_pnl, portfolio_value)
    print(f"Daily P&L ${daily_pnl:,} OK: {ok}")
    
    # Risk report
    report = risk_manager.get_report()
    print(f"Trading allowed: {report['trading_allowed']}")
    print()


def example_3_database_persistence():
    """Example 3: Save and Retrieve Backtests"""
    print("=" * 60)
    print("EXAMPLE 3: Database Persistence")
    print("=" * 60)
    
    db = TradingDatabase('trading_examples.db')  # Use file-based DB
    
    # Backtest results
    backtest_results = {
        'total_return': 0.25,
        'sharpe_ratio': 1.8,
        'max_drawdown': 0.12,
        'win_rate': 0.58,
        'total_trades': 42,
        'equity_curve': [100000, 102000, 105000, 108000, 110000]
    }
    
    # Save backtest
    backtest_id = db.save_backtest(
        name='ML Strategy v1',
        start_date='2023-01-01',
        end_date='2023-12-31',
        symbols=['AAPL'],
        initial_capital=100000,
        strategy='machine_learning',
        parameters={'lookback': 50},
        results=backtest_results
    )
    
    print(f"✓ Saved backtest: ID {backtest_id}")
    
    # Save trades
    db.save_trade(backtest_id, 'AAPL', 'BUY', 150.0, 100, 10.0, 500.0)
    db.save_trade(backtest_id, 'AAPL', 'SELL', 155.0, 100, 10.0, 490.0)
    
    print(f"✓ Saved 2 trades")
    
    # Retrieve
    backtest = db.get_backtest(backtest_id)
    trades = db.get_backtest_trades(backtest_id)
    
    print(f"✓ Retrieved backtest: {backtest['name']}")
    print(f"  Return: {backtest['total_return']:.2%}")
    print(f"  Sharpe: {backtest['sharpe_ratio']:.2f}")
    print(f"  Trades retrieved: {len(trades)}")
    print()


def example_4_alerts_and_monitoring():
    """Example 4: Alerts and Monitoring"""
    print("=" * 60)
    print("EXAMPLE 4: Alerts and Monitoring")
    print("=" * 60)
    
    alert_manager = AlertManager()
    
    # Create various alerts
    alert1 = alert_manager.price_alert('AAPL', 150.50, 150.00, 'above')
    print(f"✓ Price alert: {alert1['message']}")
    
    alert2 = alert_manager.signal_alert('TSLA', 'BUY', confidence=0.92)
    print(f"✓ Signal alert: {alert2['message']}")
    
    alert3 = alert_manager.risk_alert(
        'Daily loss limit exceeded',
        {'loss': 6000, 'limit': 5000}
    )
    print(f"✓ Risk alert: {alert3['message']} (Priority: {alert3['priority']})")
    
    # Get alerts
    all_alerts = alert_manager.get_alerts()
    print(f"✓ Total alerts: {len(all_alerts)}")
    
    # Mark as read
    alert_manager.mark_read(0)
    unread = len([a for a in alert_manager.get_alerts() if not a['read']])
    print(f"✓ Unread: {unread}")
    print()


def example_5_combined_strategy():
    """Example 5: Combined Strategy (Voting)"""
    print("=" * 60)
    print("EXAMPLE 5: Combined Strategy (Voting Ensemble)")
    print("=" * 60)
    
    market_data = MarketDataHandler()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    data = market_data.fetch_stock_data('MSFT', start_date, end_date)
    data = market_data.calculate_indicators(data)
    
    if data is not None and len(data) > 100:
        strategy = CombinedStrategy()
        asset = Asset('MSFT', 'STOCK')
        
        signal = strategy.generate_signals(data, asset)
        print(f"✓ Combined strategy signal: {signal}")
        
        # Individual votes
        print("\n Individual strategy votes:")
        for i, s in enumerate(strategy.strategies):
            individual_signal = s.generate_signals(data, asset)
            print(f"  {i+1}. {s.name}: {individual_signal}")
        
        print()


def example_6_enhanced_mean_reversion():
    """Example 6: Enhanced Mean Reversion Strategy"""
    print("=" * 60)
    print("EXAMPLE 6: Enhanced Mean Reversion Strategy")
    print("=" * 60)
    
    market_data = MarketDataHandler()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    data = market_data.fetch_stock_data('TSLA', start_date, end_date)
    data = market_data.calculate_indicators(data)
    
    if data is not None and len(data) > 50:
        strategy = EnhancedMeanReversionStrategy()
        asset = Asset('TSLA', 'STOCK')
        
        signal = strategy.generate_signals(data, asset)
        
        current = data.iloc[-1]
        print(f"Current Price: ${current['close']:.2f}")
        print(f"RSI: {current['rsi']:.1f}")
        print(f"BB Upper: ${current['bb_upper']:.2f}")
        print(f"BB Lower: ${current['bb_lower']:.2f}")
        print(f"SMA 20: ${current['sma_20']:.2f}")
        print(f"Signal: {signal}")
        print()


def example_7_adaptive_strategy():
    """Example 7: Adaptive Strategy (Market Regime Detection)"""
    print("=" * 60)
    print("EXAMPLE 7: Adaptive Strategy")
    print("=" * 60)
    
    market_data = MarketDataHandler()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = market_data.fetch_stock_data('QQQ', start_date, end_date)
    data = market_data.calculate_indicators(data)
    
    if data is not None and len(data) > 100:
        strategy = AdaptiveStrategy()
        asset = Asset('QQQ', 'STOCK')
        
        # Recent volatility
        returns = data['close'].pct_change().tail(20)
        volatility = returns.std()
        trend = abs(returns.mean())
        
        print(f"Recent volatility: {volatility:.4f}")
        print(f"Trend strength: {abs(trend):.4f}")
        
        # Adaptive signal
        signal = strategy.generate_signals(data, asset)
        print(f"Adaptive signal: {signal}")
        
        if volatility > returns.std().quantile(0.75):
            print("Market regime: HIGH VOLATILITY → Using Mean Reversion")
        else:
            print("Market regime: LOW VOLATILITY → Using Momentum")
        
        print()


def example_8_multi_strategy_comparison():
    """Example 8: Compare Multiple Strategies"""
    print("=" * 60)
    print("EXAMPLE 8: Multi-Strategy Comparison")
    print("=" * 60)
    
    market_data = MarketDataHandler()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = market_data.fetch_stock_data('SPY', start_date, end_date)
    data = market_data.calculate_indicators(data)
    
    if data is not None and len(data) > 100:
        strategies = {
            'Momentum': MomentumStrategy(),
            'Enhanced Mean Reversion': EnhancedMeanReversionStrategy(),
            'Volatility Breakout': VolatilityBreakoutStrategy(),
            'Combined': CombinedStrategy(),
            'Adaptive': AdaptiveStrategy(),
        }
        
        asset = Asset('SPY', 'STOCK')
        
        print("Strategy Comparison (SPY):")
        print("-" * 40)
        
        for name, strategy in strategies.items():
            signal = strategy.generate_signals(data, asset)
            print(f"{name:25} → {signal:6}")
        
        print()


if __name__ == '__main__':
    try:
        example_1_machine_learning_strategy()
        example_2_risk_management()
        example_3_database_persistence()
        example_4_alerts_and_monitoring()
        example_5_combined_strategy()
        example_6_enhanced_mean_reversion()
        example_7_adaptive_strategy()
        example_8_multi_strategy_comparison()
        
        print("=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
