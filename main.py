"""
Stock Options Trading System - Quantitative Multi-Asset Trading Platform
"""

from typing import List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting.backtest_engine import BacktestEngine
from strategies.base import MomentumStrategy, MeanReversionStrategy, StatisticalArbitrageStrategy
from portfolio.manager import PortfolioManager
from data.market_data import MarketDataHandler
import json


def print_header():
    """Print system header"""
    print("\n" + "="*70)
    print("   QUANTITATIVE MULTI-ASSET TRADING SYSTEM")
    print("   Stocks & Options Trading Platform")
    print("="*70 + "\n")


def run_single_backtest(strategy_name: str, symbols: List[str], 
                        start_date: str, end_date: str):
    """Run a single backtest with given strategy"""
    
    strategies = {
        'momentum': MomentumStrategy(),
        'mean_reversion': MeanReversionStrategy(),
        'statistical_arbitrage': StatisticalArbitrageStrategy(),
    }
    
    if strategy_name not in strategies:
        print(f"Unknown strategy: {strategy_name}")
        return
    
    print(f"\n{'='*70}")
    print(f"Running {strategy_name.upper()} Strategy Backtest")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    strategy = strategies[strategy_name]
    backtester = BacktestEngine(strategy, initial_capital=100000)
    
    results = backtester.run(symbols, start_date, end_date)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    # Print results
    print("PORTFOLIO SUMMARY:")
    print("-" * 70)
    summary = results['summary']
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:.<50} {value:>15.2f}")
        else:
            print(f"{key:.<50} {value:>15}")
    
    print("\n" + "-" * 70)
    print("TRADE LOG:")
    print("-" * 70)
    trades = results['trades']
    if trades:
        for trade in trades[-10:]:  # Show last 10 trades
            print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['action']:>4} | "
                  f"{trade['symbol']:>6} | {trade['quantity']:>5} @ ${trade['price']:>8.2f}")
    else:
        print("No trades executed")
    
    print("\n" + "-" * 70)
    print("CLOSED TRADES:")
    print("-" * 70)
    closed_trades = results['closed_trades']
    if closed_trades:
        for trade in closed_trades[-5:]:  # Show last 5 closed trades
            print(f"{trade['symbol']:>6} | Entry: ${trade['entry_price']:>8.2f} | "
                  f"Exit: ${trade['exit_price']:>8.2f} | "
                  f"P&L: ${trade['pnl']:>10.2f} ({trade['pnl_pct']:>6.2f}%)")
    else:
        print("No closed trades")
    
    return results


def compare_strategies(symbols: List[str], start_date: str, end_date: str):
    """Compare different strategies"""
    
    strategies = {
        'Momentum': MomentumStrategy(),
        'Mean Reversion': MeanReversionStrategy(),
        'Statistical Arbitrage': StatisticalArbitrageStrategy(),
    }
    
    print(f"\n{'='*70}")
    print(f"STRATEGY COMPARISON BACKTEST")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    results_summary = {}
    
    for strat_name, strategy in strategies.items():
        print(f"Running {strat_name} Strategy...", end=" ", flush=True)
        backtester = BacktestEngine(strategy, initial_capital=100000)
        results = backtester.run(symbols, start_date, end_date)
        
        if 'error' not in results:
            results_summary[strat_name] = results['summary']
            print("✓")
        else:
            print("✗")
    
    # Print comparison table
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    print(f"{'Strategy':<20} {'Return %':>12} {'Sharpe':>10} {'Win Rate':>12} {'Max DD':>12}")
    print("-" * 70)
    
    for strat_name, summary in results_summary.items():
        return_pct = summary.get('total_return_pct', 0)
        sharpe = summary.get('sharpe_ratio', 0)
        win_rate = summary.get('win_rate', 0)
        max_dd = summary.get('max_drawdown', 0)
        
        print(f"{strat_name:<20} {return_pct:>11.2f}% {sharpe:>10.2f} {win_rate:>11.2f}% {max_dd:>11.2f}%")
    
    print("="*70)


def main():
    """Main entry point"""
    from typing import List
    
    print_header()
    
    # Example: Backtest a single strategy
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    # Run single strategy backtest
    run_single_backtest('momentum', symbols, start_date, end_date)
    
    # Compare strategies
    compare_strategies(symbols, start_date, end_date)
    
    print("\n" + "="*70)
    print("Backtesting completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
