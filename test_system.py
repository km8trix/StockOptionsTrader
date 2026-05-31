"""
Quick system validation
"""

import sys
from datetime import datetime

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from core.models import Asset, AssetType, Order, OrderType, Position, Trade
        print("  ✓ Core models")
        
        from data.market_data import MarketDataHandler
        print("  ✓ Market data handler")
        
        from strategies.base import MomentumStrategy, MeanReversionStrategy, StatisticalArbitrageStrategy
        print("  ✓ Strategies")
        
        from portfolio.manager import PortfolioManager
        print("  ✓ Portfolio manager")
        
        from backtesting.backtest_engine import BacktestEngine
        print("  ✓ Backtest engine")
        
        from brokers.paper_trader import PaperTrader
        print("  ✓ Paper trader")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_models():
    """Test core models"""
    print("\nTesting core models...")
    try:
        from core.models import Asset, AssetType, Position
        from datetime import datetime
        
        # Create asset
        asset = Asset('AAPL', AssetType.STOCK)
        print(f"  ✓ Created asset: {asset}")
        
        # Create position
        position = Position(
            asset=asset,
            quantity=100,
            avg_entry_price=150.0,
            current_price=155.0,
            timestamp=datetime.now()
        )
        print(f"  ✓ Created position with P&L: ${position.pnl():.2f} ({position.pnl_pct():.2f}%)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_portfolio():
    """Test portfolio manager"""
    print("\nTesting portfolio manager...")
    try:
        from portfolio.manager import PortfolioManager
        
        portfolio = PortfolioManager(100000)
        print(f"  ✓ Created portfolio with ${portfolio.initial_capital:.2f}")
        print(f"  ✓ Initial cash: ${portfolio.cash:.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_strategies():
    """Test strategy implementations"""
    print("\nTesting strategies...")
    try:
        from strategies.base import MomentumStrategy, MeanReversionStrategy
        
        strategies = [
            ('Momentum', MomentumStrategy()),
            ('Mean Reversion', MeanReversionStrategy()),
        ]
        
        for name, strategy in strategies:
            print(f"  ✓ Created {name} strategy: {strategy.name}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("STOCK OPTIONS TRADING SYSTEM - SYSTEM VALIDATION")
    print("="*70)
    
    tests = [
        test_imports,
        test_models,
        test_portfolio,
        test_strategies,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*70)
    if all(results):
        print("✓ All tests passed! System is ready to use.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*70 + "\n")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
