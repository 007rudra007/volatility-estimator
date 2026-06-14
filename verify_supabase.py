"""
verify_supabase.py - Verify Supabase Table Structure & Client Operations
=====================================================================
Pushes mock quantitative calculations to all 6 Supabase tables to verify correctness
of the schema structure and python interface layer.

Run with:
  python verify_supabase.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add current path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import supabase_db as sdb

def test_connection():
    print("1. Checking connection parameters...")
    if not sdb.SUPABASE_ENABLED:
        print("[-] Supabase connection is NOT enabled.")
        print("    Please ensure .env has correct SUPABASE_URL and SUPABASE_KEY defined.")
        return False
    print("[+] Supabase connection is enabled.")
    return True

def test_volatility_analysis():
    print("\n2. Testing volatility_analysis insert...")
    dates = pd.date_range(datetime.today() - timedelta(days=5), periods=5)
    df = pd.DataFrame({
        'Close': [150.2, 151.3, 149.8, 150.5, 152.0],
        'Vol_20d': [0.22, 0.23, 0.21, 0.22, 0.24],
        'Vol_52w_Avg': [0.20, 0.20, 0.20, 0.20, 0.20],
        'Percentile': [70.0, 72.0, 68.0, 70.0, 75.0],
        'Regime': ['Elevated', 'Elevated', 'Neutral', 'Elevated', 'Expansion'],
        'EWMA': [0.215, 0.222, 0.212, 0.218, 0.231],
        'SMA_20d': [0.21, 0.22, 0.21, 0.21, 0.22],
        'Donchian_Upper': [155.0, 155.0, 155.0, 155.0, 155.0],
        'Donchian_Lower': [145.0, 145.0, 145.0, 145.0, 145.0],
        'Volume_SMA': [500000.0, 510000.0, 520000.0, 515000.0, 530000.0],
        'RVOL': [1.1, 1.2, 0.9, 1.0, 1.4]
    }, index=dates)
    
    success = sdb.save_volatility_analysis("MOCK_STOCK", df)
    if success:
        print("[+] Mock volatility analysis inserted successfully.")
    else:
        print("[-] Failed to insert mock volatility analysis.")
    return success

def test_gex_key_levels():
    print("\n3. Testing gex_key_levels insert...")
    levels = {
        'total_net_gex': 1205000.50,
        'total_net_vex': -4500.20,
        'peak_call_strike': 155.0,
        'peak_put_strike': 145.0,
        'peak_net_strike': 150.0,
        'gamma_flip_price': 148.5,
        'gex_regime': 'Positive Gamma (Dampens Volatility)',
        'vex_regime': 'Positive Vanna (Vol compression triggers dealer buying)',
        'gex_at_spot': 25000.0,
        'vex_at_spot': -50.0
    }
    success = sdb.save_gex_key_levels("MOCK_STOCK", levels, spot_price=150.5)
    if success:
        print("[+] Mock GEX key levels inserted successfully.")
    else:
        print("[-] Failed to insert mock GEX key levels.")
    return success

def test_gex_profiles():
    print("\n4. Testing gex_profiles insert...")
    df = pd.DataFrame({
        'Strike': [145.0, 150.0, 155.0],
        'Call_OI': [100.0, 500.0, 1000.0],
        'Put_OI': [1200.0, 600.0, 150.0],
        'OI': [1300.0, 1100.0, 1150.0],
        'Gamma': [0.012, 0.044, 0.015],
        'Vanna': [-0.02, 0.0, 0.03],
        'Call_GEX': [12000.0, 220000.0, 150000.0],
        'Put_GEX': [-144000.0, -264000.0, -22500.0],
        'Net_GEX': [-132000.0, -44000.0, 127500.0],
        'Call_VEX': [-2000.0, 0.0, 3000.0],
        'Put_VEX': [24000.0, 0.0, -4500.0],
        'Net_VEX': [22000.0, 0.0, -1500.0]
    })
    
    success = sdb.save_gex_profiles("MOCK_STOCK", df)
    if success:
        print("[+] Mock GEX strike profiles inserted successfully.")
    else:
        print("[-] Failed to insert mock GEX strike profiles.")
    return success

def test_positioning_data():
    print("\n5. Testing positioning_data insert...")
    dates = pd.date_range(datetime.today() - timedelta(days=5), periods=5)
    df = pd.DataFrame({
        'Speculator_Net': [12000, 14500, 11000, 13200, 15000],
        'Commercial_Net': [-13000, -15000, -12500, -14000, -16200],
        'COT_Index': [75.0, 80.0, 65.0, 72.0, 85.0],
        'Speculator_52w_Min': [5000, 5000, 5000, 5000, 5000],
        'Speculator_52w_Max': [20000, 20000, 20000, 20000, 20000]
    }, index=dates)
    
    success = sdb.save_positioning_data("MOCK_STOCK", df)
    if success:
        print("[+] Mock positioning data inserted successfully.")
    else:
        print("[-] Failed to insert mock positioning data.")
    return success

def test_cvd_data():
    print("\n6. Testing cvd_data insert...")
    dates = pd.date_range(datetime.today() - timedelta(days=5), periods=5)
    df = pd.DataFrame({
        'Close': [150.2, 151.3, 149.8, 150.5, 152.0],
        'Volume': [100000, 120000, 95000, 105000, 130000],
        'Delta': [20000, 45000, -30000, 15000, 60000],
        'CVD': [100000, 145000, 115000, 130000, 190000]
    }, index=dates)
    
    signals = pd.Series([0, 0, 1, 0, -1], index=dates)
    
    success = sdb.save_cvd_data("MOCK_STOCK", df, signals)
    if success:
        print("[+] Mock CVD and divergence signals inserted successfully.")
    else:
        print("[-] Failed to insert mock CVD data.")
    return success

def test_read_macro_events():
    print("\n7. Testing macro_events read...")
    df = sdb.load_macro_events()
    print(f"[+] Retrieved {len(df)} events from macro_events table.")
    if len(df) > 0:
        print("   Sample events:")
        print(df.head(3))
    return True

def main():
    print("=" * 60)
    print("SUPABASE INTEGRATION SCHEMA VERIFICATION")
    print("=" * 60)
    
    if not test_connection():
        sys.exit(1)
        
    all_ok = True
    all_ok &= test_volatility_analysis()
    all_ok &= test_gex_key_levels()
    all_ok &= test_gex_profiles()
    all_ok &= test_positioning_data()
    all_ok &= test_cvd_data()
    all_ok &= test_read_macro_events()
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All Supabase table schema checks completed successfully!")
    else:
        print("⚠️ Some table tests failed. Check schema definition and connection permissions.")
        sys.exit(1)

if __name__ == "__main__":
    main()
