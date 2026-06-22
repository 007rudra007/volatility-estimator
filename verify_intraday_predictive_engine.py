"""
verify_intraday_predictive_engine.py - Verification Script for 15-Minute next-day price path predictive engine
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from unittest.mock import patch
import predictive_engine
import intraday_predictive_engine

def get_mock_intraday_data(ticker, period="60d", interval="15m", progress=False):
    """Generates synthetic 15-minute price series for testing."""
    n_days = 30
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="D")
    trading_dates = [d for d in dates if d.weekday() < 5]
    
    times = [
        "09:15", "09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00",
        "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00",
        "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00", "15:15"
    ]
    
    idx = []
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []
    
    base_price = 100.0
    for d in trading_dates:
        for t in times:
            dt = pd.Timestamp(f"{d.strftime('%Y-%m-%d')} {t}:00")
            idx.append(dt)
            ret = np.random.randn() * 0.005
            close_p = base_price * (1.0 + ret)
            closes.append(close_p)
            opens.append(base_price)
            highs.append(max(close_p, base_price) * 1.002)
            lows.append(min(close_p, base_price) * 0.998)
            volumes.append(np.random.randint(500, 3000))
            base_price = close_p
            
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=idx)
    return df

@patch('yfinance.download', side_effect=get_mock_intraday_data)
def run_verification(mock_download):
    print("Initializing Intraday Predictive Engine automated tests...")
    
    # Setup daily synthetic data matching the time horizon of our mock intraday data
    n_days = 35
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="D")
    trading_dates = [d for d in dates if d.weekday() < 5]
    n = len(trading_dates)
    
    t_vals = np.linspace(0, 10, n)
    prices = 100.0 + 10.0 * np.sin(t_vals) + np.random.randn(n) * 1.0
    
    daily_df = pd.DataFrame({
        'Open': prices * 0.995,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(10000, 50000, n),
        'Donchian_Upper': prices * 1.03,
        'Donchian_Lower': prices * 0.97,
        'Donchian_Middle': prices,
        'RVOL': np.random.uniform(0.8, 1.8, n)
    }, index=trading_dates)
    
    daily_metrics = pd.DataFrame({
        'Log_Ret': np.log(daily_df['Close'] / daily_df['Close'].shift(1)).fillna(0),
        'Vol_20d': np.random.uniform(0.15, 0.25, n),
        'EWMA': np.random.uniform(0.15, 0.25, n),
        'SMA_20d': np.random.uniform(0.15, 0.25, n)
    }, index=trading_dates)
    
    daily_cot = pd.DataFrame({
        'Speculator_Net': np.random.randint(-5000, 5000, n),
        'Commercial_Net': np.random.randint(-5000, 5000, n),
        'COT_Index': np.random.uniform(20, 80, n)
    }, index=trading_dates)
    
    daily_cvd = pd.Series(np.cumsum(np.random.randn(n) * 2000), index=trading_dates)
    daily_div_signals = pd.Series(np.random.choice([-1, 0, 1], n, p=[0.05, 0.9, 0.05]), index=trading_dates)
    
    daily_pivots = pd.DataFrame([
        {'price': 92.0, 'type': 'Trough'},
        {'price': 108.0, 'type': 'Peak'},
        {'price': 96.0, 'type': 'Trough'}
    ], index=[trading_dates[2], trading_dates[10], trading_dates[18]])
    
    ticker = "TEST_INTRA"
    
    # Initialize Engine
    print("Instantiating IntradayPredictiveEngine...")
    engine = intraday_predictive_engine.IntradayPredictiveEngine(
        ticker=ticker,
        daily_data=daily_df,
        daily_metrics=daily_metrics,
        daily_cot=daily_cot,
        daily_cvd=daily_cvd,
        daily_div_signals=daily_div_signals,
        daily_pivots=daily_pivots,
        lookback_days=30
    )
    
    # 1. Fetch Clean Intraday
    print("Testing intraday cleanup pivot creation...")
    clean_intra = engine.fetch_and_clean_intraday()
    assert not clean_intra.empty, "Standardized pivot df is empty"
    print(f"[OK] Cleaned intraday pivot dataframe shape: {clean_intra.shape}")
    assert len(engine.standard_times) == 25, f"Expected 25 standard time slots, got {len(engine.standard_times)}"
    print(f"[OK] Standard times successfully identified: {engine.standard_times[0]} to {engine.standard_times[-1]}")
    
    # 2. Train Models
    print("Training Random Forest and PyTorch Multi-Task models...")
    summary = engine.train(epochs=10)
    assert summary['samples'] > 0, "No training samples generated"
    assert summary['val_mse'] >= 0, "MSE invalid"
    print(f"[OK] Training complete. Samples matched: {summary['samples']}, MSE: {summary['val_mse']:.6f}")
    
    # 3. Next-day Inference
    print("Testing next-day 15m price path inference...")
    predictions = engine.predict_next_day()
    assert len(predictions) == 25, f"Expected 25 prediction outputs, got {len(predictions)}"
    first_pred = predictions[0]
    assert 'time' in first_pred
    assert 'pred_price' in first_pred
    assert 'low_bound' in first_pred
    assert 'high_bound' in first_pred
    assert first_pred['low_bound'] <= first_pred['pred_price'] <= first_pred['high_bound'], "Expected price not in range"
    print(f"[OK] Inference pricing is structurally sound. 9:15 predicted close: INR {first_pred['pred_price']:.2f}")
    
    # 4. Check ledger logging
    print("Verifying SQLite ledger writes...")
    conn = sqlite3.connect(intraday_predictive_engine.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM intraday_prediction_ledger WHERE ticker = ?
    """, (ticker,))
    count = cursor.fetchone()[0]
    assert count == 25, f"Expected 25 ledger entries, got {count}"
    print("[OK] Intraday prediction entries logged in DB.")
    
    # 5. Concept Drift & Retrain trigger
    print("Verifying feedback loops (concept drift and dynamic retraining)...")
    
    # Inject actual values with a high deviation to simulate market regime change
    pred_date = engine.daily_data.index[-1]
    t_date = predictive_engine.get_next_trading_day(pred_date)
    target_date_str = str(t_date.date())
    
    for item in predictions:
        # actuals are 6% lower than predictions to trigger drift
        actual_val = item['pred_price'] * 0.94
        cursor.execute("""
            UPDATE intraday_prediction_ledger
            SET actual_price = ?
            WHERE ticker = ? AND target_date = ? AND candle_time = ?
        """, (actual_val, ticker, target_date_str, item['time']))
    conn.commit()
    conn.close()
    
    # Run drift check manually
    is_drifted, rmse = engine.check_concept_drift()
    assert is_drifted, "Regime drift not detected despite 6% price divergence"
    print(f"[OK] Concept drift detected successfully. Realized RMSE: {rmse:.2%}")
    
    # Run dynamic adaptation
    print("Testing dynamic model adaptation (online weights fine-tuning)...")
    initial_trees = engine.rf_reg.n_estimators
    engine.retrain_online()
    assert engine.rf_reg.n_estimators == initial_trees + 5, "Warm-start RF tree injection failed"
    print(f"[OK] Warm-start RF tree count increased to {engine.rf_reg.n_estimators}")
    print("Dynamic adaptation completed successfully.")
    
    # Clean up test ledger records
    conn = sqlite3.connect(intraday_predictive_engine.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM intraday_prediction_ledger WHERE ticker = ?", (ticker,))
    conn.commit()
    conn.close()
    
    print("All intraday predictive engine verification tests passed successfully!")

if __name__ == "__main__":
    run_verification()
