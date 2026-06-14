"""
wave_theory.py - Wave Theory & Fibonacci Metrics Calculations Core

This module provides functions for:
- ZigZag pivot detection (Peak/Trough identification)
- Elliott Wave fitting and rule-checking
- Fibonacci Price Retracements
- Fibonacci Volatility Retracements
- Fibonacci Volatility Bands (FVB)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def get_dynamic_deviation(df: pd.DataFrame, metrics: pd.DataFrame) -> float:
    """
    Computes a dynamic ZigZag deviation percentage based on the asset's
    recent realized volatility.
    
    Rule: 2.0 * daily realized volatility
    """
    vol_col = next((c for c in ['Vol_20d', 'Vol_20', 'EWMA'] if c in metrics.columns), None)
    if vol_col:
        latest_vol = metrics[vol_col].dropna().iloc[-1]
        # Convert annualized volatility to daily
        daily_vol = latest_vol / np.sqrt(252)
        # Scale to percentage and double for a 2 standard deviations swing
        dynamic_dev = float(daily_vol * 100 * 2.0)
        # Bound between 1.0% and 8.0% to keep pivots reasonable
        return min(max(dynamic_dev, 1.0), 8.0)
    return 2.0


def calculate_zigzag(df: pd.DataFrame, deviation_pct: float = 2.0) -> pd.DataFrame:
    """
    Computes ZigZag pivots for high/low price series.
    A pivot is confirmed when the price moves by deviation_pct in the opposite direction.
    
    Args:
        df: DataFrame with High, Low, Close columns
        deviation_pct: Percentage threshold for swing reversal
        
    Returns:
        DataFrame with index matching subset of df.index, containing:
        - price: Pivot price (High for Peaks, Low for Troughs)
        - type: 'Peak' or 'Trough'
    """
    highs = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
    lows = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
    closes = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    
    high_vals = highs.values
    low_vals = lows.values
    close_vals = closes.values
    times = df.index
    
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=['price', 'type'])
        
    pivots = []
    
    # State tracking
    curr_type = None  # 'Peak' or 'Trough'
    ext_price = close_vals[0]
    ext_idx = 0
    
    for i in range(1, n):
        h = float(high_vals[i])
        l = float(low_vals[i])
        c = float(close_vals[i])
        
        if curr_type is None:
            # Reversal check from first price
            if h > ext_price * (1 + deviation_pct / 100):
                pivots.append({'timestamp': times[ext_idx], 'price': ext_price, 'type': 'Trough'})
                curr_type = 'Trough'
                ext_price = h
                ext_idx = i
            elif l < ext_price * (1 - deviation_pct / 100):
                pivots.append({'timestamp': times[ext_idx], 'price': ext_price, 'type': 'Peak'})
                curr_type = 'Peak'
                ext_price = l
                ext_idx = i
        elif curr_type == 'Trough':
            # Looking for a Peak; update if we hit a higher high
            if h > ext_price:
                ext_price = h
                ext_idx = i
            # Reversal to Trough confirmed
            elif l < ext_price * (1 - deviation_pct / 100):
                pivots.append({'timestamp': times[ext_idx], 'price': ext_price, 'type': 'Peak'})
                curr_type = 'Peak'
                ext_price = l
                ext_idx = i
        elif curr_type == 'Peak':
            # Looking for a Trough; update if we hit a lower low
            if l < ext_price:
                ext_price = l
                ext_idx = i
            # Reversal to Peak confirmed
            elif h > ext_price * (1 + deviation_pct / 100):
                pivots.append({'timestamp': times[ext_idx], 'price': ext_price, 'type': 'Trough'})
                curr_type = 'Trough'
                ext_price = h
                ext_idx = i
                
    # Add final point to close the last wave
    pivots.append({'timestamp': times[ext_idx], 'price': ext_price, 'type': 'Peak' if curr_type == 'Trough' else 'Trough'})
    
    # Clean consecutive duplicate types (keep the most extreme)
    cleaned = []
    for p in pivots:
        if not cleaned:
            cleaned.append(p)
            continue
        last = cleaned[-1]
        if last['type'] == p['type']:
            if last['type'] == 'Peak' and p['price'] > last['price']:
                cleaned[-1] = p
            elif last['type'] == 'Trough' and p['price'] < last['price']:
                cleaned[-1] = p
        else:
            cleaned.append(p)
            
    if cleaned:
        res = pd.DataFrame(cleaned).set_index('timestamp')
        return res
    return pd.DataFrame(columns=['price', 'type'])


def label_elliott_waves(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits and labels Elliott Wave count (1-2-3-4-5-A-B-C) on the pivots.
    Checks institutional trading guidelines (Wave 2 retracement, Wave 3 shortest,
    Wave 4 overlap, Wave 5 truncation).
    
    Returns:
        DataFrame with same structure as pivot_df plus:
        - Wave_Label: Str label (e.g. '(1)', '(2)', '(A)')
        - Rule_Status: General status label for the fitted wave structure
    """
    pivot_df = pivot_df.copy()
    pivot_df['Wave_Label'] = ''
    pivot_df['Rule_Status'] = 'No Impulse Found'
    
    n = len(pivot_df)
    if n < 5:
        return pivot_df
        
    pivots_list = []
    for ts, row in pivot_df.iterrows():
        pivots_list.append({
            'timestamp': ts,
            'price': float(row['price']),
            'type': row['type']
        })
        
    best_start = None
    best_score = -9999
    best_status = "No Pattern Match"
    best_trend = "Bullish"
    best_labels = {}
    
    # Scan backwards to test sequences of 6 pivots [i, i+1, i+2, i+3, i+4, i+5]
    for i in range(max(0, n - 10), n - 5):
        sub = pivots_list[i:i+6]
        p0, p1, p2, p3, p4, p5 = [s['price'] for s in sub]
        t0, t1, t2, t3, t4, t5 = [s['type'] for s in sub]
        
        is_bullish = (t0 == 'Trough' and t1 == 'Peak')
        is_bearish = (t0 == 'Peak' and t1 == 'Trough')
        
        violations = []
        rules_ok = True
        
        if is_bullish:
            # Rule 1: Wave 2 cannot retrace below start of Wave 1
            if p2 <= p0:
                rules_ok = False
                violations.append("Wave 2 retraced > 100% of Wave 1")
            
            # Lengths for Wave 3 shortest check
            w1_len = p1 - p0
            w3_len = p3 - p2
            w5_len = p5 - p4
            
            if w3_len <= 0:
                rules_ok = False
                violations.append("Wave 3 has negative length")
            elif w3_len < w1_len and w3_len < w5_len:
                rules_ok = False
                violations.append("Wave 3 is the shortest")
                
            # Rule 2: Wave 4 cannot overlap price territory of Wave 1
            if p4 <= p1:
                violations.append("Wave 4 overlaps Wave 1 (Diagonal)")
                
            # Rule 3: Wave 5 must exceed Wave 3
            if p5 <= p3:
                violations.append("Truncated Wave 5")
                
            trend = "Bullish"
            
        elif is_bearish:
            # Rule 1: Wave 2 cannot retrace above start of Wave 1
            if p2 >= p0:
                rules_ok = False
                violations.append("Wave 2 retraced > 100% of Wave 1")
            
            # Lengths
            w1_len = p0 - p1
            w3_len = p2 - p3
            w5_len = p4 - p5
            
            if w3_len <= 0:
                rules_ok = False
                violations.append("Wave 3 has negative length")
            elif w3_len < w1_len and w3_len < w5_len:
                rules_ok = False
                violations.append("Wave 3 is the shortest")
                
            # Rule 2: Wave 4 cannot overlap price territory of Wave 1
            if p4 >= p1:
                violations.append("Wave 4 overlaps Wave 1 (Diagonal)")
                
            # Rule 3: Wave 5 must exceed Wave 3
            if p5 >= p3:
                violations.append("Truncated Wave 5")
                
            trend = "Bearish"
        else:
            continue
            
        # Scoring logic
        # Perfect wave: 15
        # Minor violations: 10 - len(violations)
        score = 10 - len(violations)
        if rules_ok:
            score += 5
        # Prefer the most recent pattern
        score += (i / n) * 2.0
        
        if score > best_score:
            best_start = i
            best_score = score
            best_trend = trend
            best_status = "Rules Satisfied" if not violations else "Partially Satisfied (" + ", ".join(violations) + ")"
            
            # Map labels
            best_labels = {
                sub[0]['timestamp']: '(1) Start',
                sub[1]['timestamp']: '(1)',
                sub[2]['timestamp']: '(2)',
                sub[3]['timestamp']: '(3)',
                sub[4]['timestamp']: '(4)',
                sub[5]['timestamp']: '(5)'
            }
            
            # Look forward for corrective A-B-C wave structure (requires 3 more pivots)
            if i + 8 < n:
                sub_abc = pivots_list[i+6:i+9]
                best_labels[sub_abc[0]['timestamp']] = '(A)'
                best_labels[sub_abc[1]['timestamp']] = '(B)'
                best_labels[sub_abc[2]['timestamp']] = '(C)'
                
    if best_start is not None:
        for ts, lbl in best_labels.items():
            pivot_df.loc[ts, 'Wave_Label'] = lbl
        pivot_df['Rule_Status'] = f"{best_trend} Impulse: {best_status}"
    else:
        pivot_df['Rule_Status'] = "No standard 5-wave impulse detected."
        
    return pivot_df


def calculate_active_swing_fibs(pivot_df: pd.DataFrame, current_price: float) -> Tuple[Optional[dict], Optional[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Finds the most recent major swing (high/low) in the pivots and
    calculates its Fibonacci Price Retracement levels.
    """
    n = len(pivot_df)
    if n < 2:
        return None, None
        
    # Get the last two pivots
    p1_ts = pivot_df.index[-2]
    p1_price = float(pivot_df.iloc[-2]['price'])
    p1_type = pivot_df.iloc[-2]['type']
    
    p2_ts = pivot_df.index[-1]
    p2_price = float(pivot_df.iloc[-1]['price'])
    
    # Determine the trend of the active swing
    if p1_type == 'Trough':  # Price went from Trough to Peak (Upward swing)
        swing_high = p2_price
        swing_low = p1_price
        trend = 'bullish'
    else:  # Price went from Peak to Trough (Downward swing)
        swing_high = p1_price
        swing_low = p2_price
        trend = 'bearish'
        
    diff = swing_high - swing_low
    if diff == 0:
        return None, None
        
    # Standard Fibonacci retracement calculation
    # Bullish: 0% is high, 100% is low. Retracements pull back down from high.
    # Bearish: 0% is low, 100% is high. Retracements bounce back up from low.
    if trend == 'bullish':
        levels = {
            '0.0% (High)': swing_high,
            '23.6%': swing_high - 0.236 * diff,
            '38.2%': swing_high - 0.382 * diff,
            '50.0%': swing_high - 0.500 * diff,
            '61.8%': swing_high - 0.618 * diff,
            '78.6%': swing_high - 0.786 * diff,
            '100.0% (Low)': swing_low
        }
    else:
        levels = {
            '0.0% (Low)': swing_low,
            '23.6%': swing_low + 0.236 * diff,
            '38.2%': swing_low + 0.382 * diff,
            '50.0%': swing_low + 0.500 * diff,
            '61.8%': swing_low + 0.618 * diff,
            '78.6%': swing_low + 0.786 * diff,
            '100.0% (High)': swing_high
        }
        
    return levels, (p1_ts, p2_ts)


def calculate_fib_vol_retracement(vol_series: pd.Series) -> Optional[dict]:
    """
    Finds the high and low volatility levels in the series and
    calculates their Fibonacci Retracement levels.
    """
    clean_vol = vol_series.dropna()
    if len(clean_vol) < 20:
        return None
        
    peak_vol = float(clean_vol.max())
    trough_vol = float(clean_vol.min())
    diff = peak_vol - trough_vol
    
    if diff == 0:
        return None
        
    # Volatility levels from bottom to top
    levels = {
        '0.0% (Trough)': trough_vol,
        '23.6%': trough_vol + 0.236 * diff,
        '38.2%': trough_vol + 0.382 * diff,
        '50.0%': trough_vol + 0.500 * diff,
        '61.8%': trough_vol + 0.618 * diff,
        '78.6%': trough_vol + 0.786 * diff,
        '100.0% (Peak)': peak_vol
    }
    
    current_vol = float(clean_vol.iloc[-1])
    
    # Locate where current vol sits
    nearest_lvl = '0.0% (Trough)'
    min_dist = 999.0
    for name, val in levels.items():
        dist = abs(current_vol - val)
        if dist < min_dist:
            min_dist = dist
            nearest_lvl = name
            
    return {
        'peak': peak_vol,
        'trough': trough_vol,
        'levels': levels,
        'current_vol': current_vol,
        'nearest_level': nearest_lvl
    }


def calculate_fib_vol_bands(prices: pd.Series, vol_series: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Calculates Fibonacci Volatility Bands around the asset price.
    - Middle Band = SMA(Close, window)
    - Upper/Lower Bands = Middle Band * (1 +/- ratio * daily_volatility)
    
    Daily Volatility = Annualized Volatility / sqrt(252)
    """
    middle_band = prices.rolling(window=window).mean()
    
    # Convert annualized volatility to daily multiplier
    daily_vol = vol_series / np.sqrt(252)
    
    bands = pd.DataFrame(index=prices.index)
    bands['Middle_Band'] = middle_band
    
    # Standard Fibonacci multiples
    fib_ratios = [1.618, 2.618, 4.236]
    
    for r in fib_ratios:
        bands[f'Upper_Band_{r}'] = middle_band * (1 + r * daily_vol)
        bands[f'Lower_Band_{r}'] = middle_band * (1 - r * daily_vol)
        
    return bands
