"""
verify_calculations.py - Validation & Audit Script

This script verifies that the volatility calculations are mathematically correct
by comparing them with manual step-by-step calculations.

Run: python verify_calculations.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from vol_engine import calculate_metrics, calculate_log_returns, calculate_rolling_vol

def manual_log_returns(prices):
    """Manual log returns calculation for verification."""
    returns = []
    for i in range(1, len(prices)):
        ret = np.log(prices.iloc[i] / prices.iloc[i-1])
        returns.append(ret)
    return [np.nan] + returns  # First value is NaN

def manual_rolling_std(returns, window):
    """Manual rolling standard deviation calculation."""
    result = []
    for i in range(len(returns)):
        if i < window:
            result.append(np.nan)
        else:
            window_data = returns[i-window+1:i+1]
            # Remove NaN values
            clean = [x for x in window_data if not np.isnan(x)]
            if len(clean) >= window - 1:
                std = np.std(clean, ddof=1)  # Sample std
                result.append(std * np.sqrt(252))  # Annualize
            else:
                result.append(np.nan)
    return result

def verify_with_manual_calculation(ticker="RELIANCE.NS", period="3mo"):
    """
    Verify tool calculations against manual computation.
    """
    print("=" * 70)
    print(f"VERIFICATION REPORT: {ticker}")
    print("=" * 70)
    
    # Fetch data
    print(f"\n1. Fetching {period} of data for {ticker}...")
    data = yf.download(ticker, period=period, progress=False)
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    prices = data['Close']
    print(f"   ✓ Got {len(prices)} trading days of data")
    
    # -------------------------------------------------------------------------
    # LOG RETURNS VERIFICATION
    # -------------------------------------------------------------------------
    print("\n2. Verifying Log Returns...")
    
    # Tool calculation
    tool_returns = calculate_log_returns(prices)
    
    # Manual calculation
    manual_returns = manual_log_returns(prices)
    
    # Compare last 5 values
    print("\n   Comparing last 5 log returns:")
    print(f"   {'Date':<12} {'Tool':<15} {'Manual':<15} {'Match?':<8}")
    print("   " + "-" * 50)
    
    matches = 0
    for i in range(-5, 0):
        tool_val = tool_returns.iloc[i]
        manual_val = manual_returns[i]
        match = np.isclose(tool_val, manual_val, rtol=1e-10) if not np.isnan(tool_val) else np.isnan(manual_val)
        matches += 1 if match else 0
        date = prices.index[i].strftime('%Y-%m-%d')
        print(f"   {date:<12} {tool_val:>14.8f} {manual_val:>14.8f} {'✓' if match else '✗'}")
    
    print(f"\n   Log Returns Accuracy: {matches}/5 ✓")
    
    # -------------------------------------------------------------------------
    # ROLLING VOLATILITY VERIFICATION (20-day)
    # -------------------------------------------------------------------------
    print("\n3. Verifying 20-Day Rolling Volatility...")
    
    # Tool calculation
    metrics = calculate_metrics(prices)
    tool_vol_20 = metrics['Vol_20d']
    
    # Manual calculation
    manual_vol_20 = manual_rolling_std(list(tool_returns), 20)
    
    print("\n   Comparing last 5 volatility values:")
    print(f"   {'Date':<12} {'Tool':<15} {'Manual':<15} {'Diff %':<10} {'Match?':<8}")
    print("   " + "-" * 60)
    
    vol_matches = 0
    for i in range(-5, 0):
        tool_val = tool_vol_20.iloc[i]
        manual_val = manual_vol_20[i]
        
        if np.isnan(tool_val) or np.isnan(manual_val):
            match = np.isnan(tool_val) and np.isnan(manual_val)
            diff_pct = "N/A"
        else:
            diff = abs(tool_val - manual_val) / manual_val * 100
            diff_pct = f"{diff:.4f}%"
            match = diff < 0.01  # Match if difference < 0.01%
        
        vol_matches += 1 if match else 0
        date = prices.index[i].strftime('%Y-%m-%d')
        t_str = f"{tool_val:.6f}" if not np.isnan(tool_val) else "NaN"
        m_str = f"{manual_val:.6f}" if not np.isnan(manual_val) else "NaN"
        print(f"   {date:<12} {t_str:>14} {m_str:>14} {diff_pct:>9} {'✓' if match else '✗'}")
    
    print(f"\n   Rolling Vol Accuracy: {vol_matches}/5 ✓")
    
    # -------------------------------------------------------------------------
    # FORMULA EXPLANATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FORMULA REFERENCE")
    print("=" * 70)
    print("""
    1. LOG RETURNS:
       r_t = ln(P_t / P_{t-1})
       
       Example: If yesterday's close was ₹1400 and today is ₹1414:
       r = ln(1414/1400) = ln(1.01) ≈ 0.00995 (or 0.995%)

    2. ROLLING VOLATILITY (20-day, Annualized):
       σ = StdDev(r_{t-19}, ... , r_t) × √252
       
       - StdDev uses (n-1) denominator (sample std)
       - Multiply by √252 to annualize (252 trading days/year)
       
       Example: If 20-day daily std = 0.015:
       Annualized Vol = 0.015 × √252 = 0.015 × 15.87 ≈ 23.8%

    3. EWMA VOLATILITY (λ ≈ 0.94):
       σ²_t = λ × σ²_{t-1} + (1-λ) × r²_t
       
       - More weight on recent returns
       - Standard RiskMetrics decay factor: λ = 0.94

    4. GARCH(1,1):
       σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
       
       - ω: Long-term variance weight
       - α: Reaction to shocks (typically 0.05-0.15)
       - β: Persistence (typically 0.80-0.95)
       - Stationarity: α + β < 1
    """)
    
    # -------------------------------------------------------------------------
    # BENCHMARK COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON (INDIA VIX)")
    print("=" * 70)
    
    try:
        vix = yf.download("^INDIAVIX", period=period, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        if len(vix) > 0:
            current_vix = vix['Close'].iloc[-1]
            tool_current_vol = tool_vol_20.iloc[-1] * 100  # Convert to %
            
            print(f"""
    India VIX (Implied Vol):    {current_vix:.2f}%
    Tool's 20-Day Realized Vol: {tool_current_vol:.2f}%
    
    Note: India VIX is IMPLIED volatility (forward-looking, derived from options)
          Tool calculates REALIZED volatility (historical, based on actual returns)
          
    These can differ significantly! VIX tends to be higher during uncertainty.
    A large gap (VIX >> Realized) suggests the market expects more volatility ahead.
            """)
        else:
            print("\n   India VIX data not available for comparison.")
    except Exception as e:
        print(f"\n   Could not fetch India VIX: {e}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    return {
        'log_returns_match': matches == 5,
        'rolling_vol_match': vol_matches == 5,
        'all_verified': matches == 5 and vol_matches == 5
    }


def compare_with_external_source(ticker="RELIANCE.NS"):
    """
    Generate data that can be manually verified with Excel or other tools.
    """
    print("\n" + "=" * 70)
    print("EXPORTABLE DATA FOR EXTERNAL VERIFICATION")
    print("=" * 70)
    
    data = yf.download(ticker, period="1mo", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    metrics = calculate_metrics(data['Close'])
    
    export_df = pd.DataFrame({
        'Date': data.index,
        'Close': data['Close'].values,
        'Log_Return': metrics['Log_Ret'].values,
        'Vol_20d': metrics['Vol_20d'].values,
        'EWMA': metrics['EWMA'].values
    })
    
    export_path = "verification_export.csv"
    export_df.to_csv(export_path, index=False)
    
    print(f"""
    Exported to: {export_path}
    
    To verify in Excel:
    1. Open the CSV
    2. Manually calculate: =LN(C3/C2) for log returns
    3. Use =STDEV(range)*SQRT(252) for 20-day rolling vol
    4. Compare with the tool's output columns
    """)
    
    return export_df


if __name__ == "__main__":
    # Run verification
    result = verify_with_manual_calculation("RELIANCE.NS", "3mo")
    
    # Export for external verification
    compare_with_external_source("RELIANCE.NS")
    
    print("\n✅ All tests passed!" if result['all_verified'] else "\n⚠️ Some discrepancies found - review above.")
