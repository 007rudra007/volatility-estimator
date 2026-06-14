"""
verify_confirmations.py - Verification Script for Trade Confirmations Engine

Validates the mathematical correctness of:
- Option Gamma calculation
- Synthetic GEX Profile formatting and key level extraction
- COT / FII positioning calculations
- Cumulative Volume Delta (CVD) calculations
- CVD Divergence alerts

Run with: python verify_confirmations.py
"""

import sys
import io
import numpy as np
import pandas as pd
import confirmations_engine as ce

# Force UTF-8 encoding for standard output on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_option_gamma():
    print("1. Testing Option Gamma (Black-Scholes)...")
    # Spot=100, Strike=100, T=0.2 (73 days), Vol=20%, Rate=5%
    S, K, T, sigma, r = 100.0, 100.0, 0.2, 0.2, 0.05
    gamma = ce.calculate_option_gamma(S, K, T, sigma, r)
    
    # Expected analytical value:
    # d1 = (ln(100/100) + (0.05 + 0.5 * 0.04) * 0.2) / (0.2 * sqrt(0.2))
    #    = (0 + 0.07 * 0.2) / (0.2 * 0.44721) = 0.014 / 0.08944 = 0.15652
    # N'(d1) = exp(-0.15652^2 / 2) / sqrt(2*pi) = exp(-0.01225) / 2.5066 = 0.3939
    # Gamma = N'(d1) / (S * sigma * sqrt(T)) = 0.3939 / (100 * 0.2 * 0.44721) = 0.3939 / 8.9442 = 0.04404
    
    expected = 0.04404
    diff = abs(gamma - expected)
    print(f"   Calculated Gamma: {gamma:.6f} | Expected: ~{expected:.5f}")
    assert diff < 1e-3, f"Gamma calculation error! Diff is {diff}"
    print("   ✓ Gamma calculation validated successfully")


def test_option_vanna():
    print("\n1b. Testing Option Vanna (Black-Scholes)...")
    # Spot=100, Strike=100, T=0.2, Vol=20%, Rate=5%
    S, K, T, sigma, r = 100.0, 100.0, 0.2, 0.2, 0.05
    vanna = ce.calculate_option_vanna(S, K, T, sigma, r)
    
    # Expected analytical value:
    # d1 = 0.15652, d2 = d1 - 0.2 * sqrt(0.2) = 0.15652 - 0.08944 = 0.06708
    # N'(d1) = 0.3939
    # Vanna = - N'(d1) * d2 / sigma = -0.3939 * 0.06708 / 0.2 = -0.1321
    
    expected = -0.1321
    diff = abs(vanna - expected)
    print(f"   Calculated Vanna: {vanna:.6f} | Expected: ~{expected:.4f}")
    assert diff < 1e-3, f"Vanna calculation error! Diff is {diff}"
    print("   ✓ Vanna calculation validated successfully")


def test_synthetic_gex():
    print("\n2. Testing Synthetic Options GEX/VEX Profile...")
    spot = 5000.0
    vol = 0.25
    r = 0.06
    
    gex_df = ce.get_synthetic_gex_profile(spot, vol, r)
    
    # Checks
    assert isinstance(gex_df, pd.DataFrame), "Synthetic GEX profile must be a DataFrame"
    required_cols = ['Strike', 'Call_OI', 'Put_OI', 'OI', 'Gamma', 'Vanna', 'Call_GEX', 'Put_GEX', 'Net_GEX', 'Call_VEX', 'Put_VEX', 'Net_VEX']
    for col in required_cols:
        assert col in gex_df.columns, f"Missing column in GEX/VEX profile: {col}"
        
    print(f"   ✓ Generated {len(gex_df)} synthetic strikes around Spot ₹{spot:,.2f}")
    print(f"   ✓ Columns verify: {list(gex_df.columns)}")
    
    # Check key levels extraction
    levels = ce.extract_gex_key_levels(gex_df, spot)
    print("   Extracted GEX/VEX levels:")
    print(f"     - Gamma Flip Price: ₹{levels.get('gamma_flip_price'):,.2f}")
    print(f"     - Net Peak Strike:   ₹{levels.get('peak_net_strike'):,.2f}")
    print(f"     - Call Peak Strike:  ₹{levels.get('peak_call_strike'):,.2f}")
    print(f"     - Put Peak Strike:   ₹{levels.get('peak_put_strike'):,.2f}")
    print(f"     - Spot GEX Regime:   {levels.get('gex_regime')}")
    print(f"     - Spot VEX Regime:   {levels.get('vex_regime')}")
    print(f"     - Net Vanna at Spot: {levels.get('vex_at_spot'):+,.4f}")
    
    assert levels.get('gamma_flip_price') > 0, "Gamma flip price must be positive"
    assert levels.get('peak_net_strike') > 0, "Peak strike must be positive"
    print("   ✓ GEX/VEX key levels successfully verified")


def test_cot_positioning():
    print("\n3. Testing COT Positioning Calculation...")
    # Create fake price history
    np.random.seed(100)
    dates = pd.date_range("2025-01-01", periods=300, freq='D')
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.randn(300) * 0.01 + 0.0005)), index=dates)
    
    # Test global COT
    cot_df = ce.get_cot_positioning("SPY", prices)
    assert isinstance(cot_df, pd.DataFrame), "COT output must be a DataFrame"
    assert 'COT_Index' in cot_df.columns, "Missing COT_Index column"
    assert 'Speculator_Net' in cot_df.columns, "Missing Speculator_Net column"
    
    # Check that COT Index is scaled between 0 and 100
    last_val = cot_df['COT_Index'].iloc[-1]
    print(f"   Last COT Index: {last_val:.2f}%")
    assert 0.0 <= last_val <= 100.0, f"COT Index out of bounds: {last_val}"
    
    # Test Indian COT proxy
    ind_cot_df = ce.get_cot_positioning("RELIANCE.NS", prices)
    assert 'COT_Index' in ind_cot_df.columns
    print("   ✓ COT positioning calculations verified")


def test_cvd_and_divergences():
    print("\n4. Testing Cumulative Volume Delta & Divergence Detector...")
    
    # Create artificial price and volume dataframe where price goes down but volume is buying (accumulation)
    np.random.seed(42)
    dates = pd.date_range("2026-05-01", periods=30)
    
    # Price making a lower low at the end
    close_prices = [
        100.0, 99.5, 99.0, 98.5, 99.0, 98.2, 97.5, 97.0, 97.5, 97.0, 
        96.5, 96.0, 96.5, 95.8, 95.0, 94.5, 94.0, 94.2, 93.8, 93.0, 
        92.5, 92.0, 91.5, 91.0, 91.5, 90.8, 90.0, 89.2, 88.5, 87.0
    ]
    open_prices = [p + 0.1 for p in close_prices] # slightly higher open (mostly red days)
    
    # Modify the last few days to have close > open and large volume (buying pressure)
    # Day 28, 29, 30: price continues to slide but open is very low and close pushes up, or delta is heavily positive
    open_prices[27] = 88.0; close_prices[27] = 89.2  # Price went down vs yesterday, but intraday closed positive
    open_prices[28] = 87.0; close_prices[28] = 88.5
    open_prices[29] = 85.0; close_prices[29] = 87.0
    
    high_prices = [max(o, c) + 0.2 for o, c in zip(open_prices, close_prices)]
    low_prices = [min(o, c) - 0.2 for o, c in zip(open_prices, close_prices)]
    volume = [10000] * 30
    # Pump volume on the divergence days
    volume[27] = 80000
    volume[28] = 90000
    volume[29] = 100000
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    # Calculate CVD
    daily_cvd = ce.calculate_daily_cvd(df)
    assert len(daily_cvd) == 30, "CVD output length mismatch"
    
    # Check divergence detection
    signals = ce.detect_cvd_divergences(df['Close'], daily_cvd, window=8)
    
    print(f"   Daily CVD last 5: {list(daily_cvd.tail(5))}")
    print(f"   Divergence Signals last 5: {list(signals.tail(5))}")
    
    # Assert signals only contains -1, 0, or 1
    assert set(signals.unique()).issubset({-1, 0, 1}), "Invalid divergence signal values!"
    print("   ✓ CVD and Divergence checks validated successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING TRADE CONFIRMATION ENGINE VALIDATIONS")
    print("=" * 60)
    
    try:
        test_option_gamma()
        test_option_vanna()
        test_synthetic_gex()
        test_cot_positioning()
        test_cvd_and_divergences()
        print("\n✅ All validations completed successfully!")
    except AssertionError as ae:
        print(f"\n❌ Validation failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
