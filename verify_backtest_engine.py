"""
verify_backtest_engine.py - Verification script for quantitative backtester engine
"""

import numpy as np
import pandas as pd
import backtest_engine

def run_verification():
    print("Initializing Quantitative Backtester automated tests...")
    
    # 1. Create synthetic history (100 days of daily data)
    n = 100
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    trading_dates = [d for d in dates if d.weekday() < 5]
    n_days = len(trading_dates)
    
    t_vals = np.linspace(0, 5, n_days)
    prices = 100.0 + 8.0 * np.sin(t_vals) + np.random.randn(n_days) * 1.5
    
    df = pd.DataFrame({
        'Open': prices * 0.995,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(5000, 15000, n_days),
        'Donchian_Upper': prices * 1.025,
        'Donchian_Lower': prices * 0.975,
        'Donchian_Middle': prices,
        'RVOL': np.random.uniform(0.7, 1.7, n_days)
    }, index=trading_dates)
    
    metrics = pd.DataFrame({
        'Log_Ret': np.log(df['Close'] / df['Close'].shift(1)).fillna(0.0),
        'Vol_20d': np.random.uniform(0.12, 0.22, n_days),
        'EWMA': np.random.uniform(0.12, 0.22, n_days),
        'SMA_20d': np.random.uniform(0.12, 0.22, n_days)
    }, index=trading_dates)
    
    cot_df = pd.DataFrame({
        'Speculator_Net': np.random.randint(-2000, 2000, n_days),
        'Commercial_Net': np.random.randint(-2000, 2000, n_days),
        'COT_Index': np.random.uniform(15.0, 85.0, n_days)
    }, index=trading_dates)
    
    daily_cvd = pd.Series(np.cumsum(np.random.randn(n_days) * 500), index=trading_dates)
    daily_div_signals = pd.Series(np.random.choice([-1, 0, 1], n_days, p=[0.1, 0.8, 0.1]), index=trading_dates)
    
    pivot_df = pd.DataFrame([
        {'price': 94.0, 'type': 'Trough'},
        {'price': 105.0, 'type': 'Peak'}
    ], index=[trading_dates[5], trading_dates[20]])
    
    # 2. Test Volatility Band breaches
    print("Testing volatility band coverage validation...")
    res_vol = backtest_engine.validate_volatility_bands(
        close_prices=df['Close'],
        high_prices=df['High'],
        low_prices=df['Low'],
        vol_series=metrics['EWMA'],
        multiplier=1.5
    )
    assert 0.0 <= res_vol['hit_rate'] <= 1.0, "Volatility hit rate out of bounds"
    assert res_vol['total_days'] == n_days - 1, "Incorrect total day evaluation in volatility test"
    print(f"[OK] Volatility coverage test complete. Hit rate: {res_vol['hit_rate']:.1%}, Breaches: {res_vol['total_breaches']}")
    
    # 3. Test CVD Divergence Analysis
    print("Testing CVD divergence statistical analysis...")
    res_cvd = backtest_engine.analyze_cvd_divergences(
        prices=df['Close'],
        cvd_signals=daily_div_signals,
        horizons=[1, 3, 5]
    )
    assert 'bullish' in res_cvd and 'bearish' in res_cvd, "Divergence keys missing"
    assert '3d' in res_cvd['bullish'], "Forward horizon keys missing"
    print("[OK] CVD divergence statistical forward returns computed.")
    
    # 4. Test Walk-forward EOD AI Model validation
    print("Testing walk-forward EOD predictions (simulating training gaps)...")
    res_wf = backtest_engine.run_walk_forward_predictions(
        data=df,
        metrics=metrics,
        cot_df=cot_df,
        daily_cvd=daily_cvd,
        daily_div_signals=daily_div_signals,
        pivot_df=pivot_df,
        retrain_freq=25,
        initial_train_days=35,
        epochs=3
    )
    assert len(res_wf) == n_days - 35, f"Expected {n_days - 35} outputs, got {len(res_wf)}"
    assert 'pred_trend' in res_wf.columns, "Predictions trend column missing"
    print("[OK] Walk-forward daily predictions completed.")
    
    # 5. Test quantitative trading strategy simulator
    print("Testing quantitative strategy simulator...")
    res_strat = backtest_engine.run_strategy_backtest(
        data=df,
        metrics=metrics,
        cot_df=cot_df,
        daily_cvd=daily_cvd,
        daily_div_signals=daily_div_signals,
        pivot_df=pivot_df,
        run_ai_predictions=True,
        stop_loss_pct=3.0,
        take_profit_pct=6.0,
        initial_train_days=35,
        retrain_freq=25
    )
    
    assert len(res_strat['equity_curve']) == n_days, "Equity curve length mismatch"
    assert res_strat['sharpe_ratio'] is not None, "Sharpe ratio is None"
    assert -1.0 <= res_strat['max_drawdown'] <= 0.0, "Max drawdown out of bounds"
    print(f"[OK] Strategy simulator test complete. Strategy Return: {res_strat['cumulative_return']:.2%}, Drawdown: {res_strat['max_drawdown']:.2%}")
    print("All backtest engine verification tests passed successfully!")

if __name__ == "__main__":
    run_verification()
