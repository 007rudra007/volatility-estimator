"""
verify_wave_theory.py - Automated Tests for Wave Theory & Fibonacci Metrics
"""

import numpy as np
import pandas as pd
import wave_theory


def test_zigzag_and_elliott():
    print("Running ZigZag and Elliott Wave tests...")
    
    # 1. Create a synthetic price history representing a perfect bullish wave structure
    # Pivots:
    # 0. Trough at Day 0: Price = 100
    # 1. Peak at Day 20: Price = 150 (Wave 1)
    # 2. Trough at Day 40: Price = 120 (Wave 2, retraced 60% of Wave 1, above 100)
    # 3. Peak at Day 60: Price = 220 (Wave 3, longest, above 150)
    # 4. Trough at Day 80: Price = 180 (Wave 4, above Wave 1 peak at 150 - no overlap)
    # 5. Peak at Day 100: Price = 250 (Wave 5, above 220)
    # 6. Trough at Day 120: Price = 210 (Wave A)
    # 7. Peak at Day 140: Price = 230 (Wave B)
    # 8. Trough at Day 160: Price = 190 (Wave C)
    
    dates = pd.date_range(start="2026-01-01", periods=180, freq="D")
    prices = []
    
    # Pre-populate segments to build a smooth curve between pivots
    pivots_list = [100, 150, 120, 220, 180, 250, 210, 230, 190, 200]
    segment_len = 20
    
    for segment_idx in range(len(pivots_list) - 1):
        p_start = pivots_list[segment_idx]
        p_end = pivots_list[segment_idx + 1]
        for step in range(segment_len):
            alpha = step / segment_len
            # Linear interpolation with some micro-noise
            price = p_start + alpha * (p_end - p_start) + 0.5 * np.sin(step)
            prices.append(price)
            
    # Add a final buffer
    prices.append(200.0)
    
    dates = pd.date_range(start="2026-01-01", periods=len(prices), freq="D")
    df = pd.DataFrame({
        'Close': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Volume': [10000] * len(prices)
    }, index=dates)
    
    print(f"Created synthetic dataset with {len(df)} bars.")
    
    # 2. Test ZigZag Pivot Detection (with a 10% swing threshold)
    pivot_df = wave_theory.calculate_zigzag(df, deviation_pct=10.0)
    print(f"ZigZag found {len(pivot_df)} pivots.")
    assert len(pivot_df) >= 8, f"Expected at least 8 pivots, got {len(pivot_df)}"
    
    # Ensure types alternate
    types = pivot_df['type'].tolist()
    for i in range(len(types) - 1):
        assert types[i] != types[i+1], "Pivots do not alternate!"
    print("[OK] ZigZag pivots alternate correctly.")
    
    # 3. Test Elliott Wave classification
    labeled_df = wave_theory.label_elliott_waves(pivot_df)
    labels = labeled_df['Wave_Label'].tolist()
    print("Wave Labels found:", [l for l in labels if l != ''])
    print("Rule Status:", labeled_df['Rule_Status'].iloc[-1])
    
    # Check that standard labels exist
    active_labels = [l for l in labels if l != '']
    assert '(1) Start' in active_labels, "Wave 1 Start label missing"
    assert '(1)' in active_labels, "Wave 1 label missing"
    assert '(2)' in active_labels, "Wave 2 label missing"
    assert '(3)' in active_labels, "Wave 3 label missing"
    assert '(4)' in active_labels, "Wave 4 label missing"
    assert '(5)' in active_labels, "Wave 5 label missing"
    
    # Verify rules status says satisfied
    assert "Rules Satisfied" in labeled_df['Rule_Status'].iloc[-1], f"Expected satisfied rules, got: {labeled_df['Rule_Status'].iloc[-1]}"
    print("[OK] Elliott Wave classification and rule-checking logic works perfectly.")
    
    # 4. Test active swing Fibonacci retracements
    current_p = float(df['Close'].iloc[-1])
    fibs, swing_dates = wave_theory.calculate_active_swing_fibs(pivot_df, current_p)
    assert fibs is not None, "Fibonacci levels could not be calculated"
    assert '38.2%' in fibs, "Fibonacci level 38.2% missing"
    assert '61.8%' in fibs, "Fibonacci level 61.8% missing"
    print("[OK] Fibonacci price retracements logic works perfectly.")
    
    # 5. Test Fibonacci volatility retracements
    # Create fake volatility series
    vol_series = pd.Series(np.linspace(0.15, 0.45, 100), index=dates[:100])
    vol_fibs = wave_theory.calculate_fib_vol_retracement(vol_series)
    assert vol_fibs is not None
    assert vol_fibs['peak'] == 0.45
    assert vol_fibs['trough'] == 0.15
    assert abs(vol_fibs['levels']['50.0%'] - 0.30) < 1e-6
    print("[OK] Fibonacci volatility retracements calculations work perfectly.")
    
    # 6. Test Fibonacci Volatility Bands
    vol_series_all = pd.Series([0.20] * len(prices), index=df.index)
    bands = wave_theory.calculate_fib_vol_bands(df['Close'], vol_series_all, window=20)
    assert 'Middle_Band' in bands.columns
    assert 'Upper_Band_1.618' in bands.columns
    assert 'Lower_Band_4.236' in bands.columns
    print("[OK] Fibonacci Volatility Bands calculations work perfectly.")
    
    print("All tests passed successfully!")


if __name__ == "__main__":
    test_zigzag_and_elliott()
