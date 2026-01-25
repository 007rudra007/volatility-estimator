"""
event_analyzer.py - Macro Event Impact Analysis

This module provides functions for analyzing volatility behavior
around key macro events (RBI MPC, CPI Releases, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


def load_events(filepath: str = None) -> pd.DataFrame:
    """
    Load macro events from CSV file.
    
    Args:
        filepath: Path to macro_events_india.csv
                 If None, uses default path in data/ folder
                 
    Returns:
        DataFrame with Date as DatetimeIndex
    """
    if filepath is None:
        # Default path relative to this file
        filepath = Path(__file__).parent / "data" / "macro_events_india.csv"
    
    events = pd.read_csv(filepath, parse_dates=['Date'])
    events = events.sort_values('Date').reset_index(drop=True)
    
    return events


def filter_events_in_range(
    events: pd.DataFrame, 
    start_date: datetime, 
    end_date: datetime,
    event_type: str = None
) -> pd.DataFrame:
    """
    Filter events within a date range.
    
    Args:
        events: DataFrame from load_events()
        start_date: Start of date range
        end_date: End of date range
        event_type: Optional filter by event type (e.g., 'RBI MPC')
        
    Returns:
        Filtered DataFrame
    """
    mask = (events['Date'] >= pd.Timestamp(start_date)) & \
           (events['Date'] <= pd.Timestamp(end_date))
    
    filtered = events[mask].copy()
    
    if event_type:
        filtered = filtered[filtered['Event_Type'] == event_type]
    
    return filtered.reset_index(drop=True)


def analyze_event_impact(
    df: pd.DataFrame, 
    event_date: datetime, 
    window: int = 5
) -> Optional[Dict]:
    """
    Analyze volatility impact around a specific event.
    
    Compares Pre-Event (T-window to T-1) vs Post-Event (T to T+window-1)
    
    Args:
        df: DataFrame with 'Log_Ret' column and DatetimeIndex
        event_date: The event date to analyze
        window: Number of days before/after to analyze
        
    Returns:
        Dict with pre/post volatility metrics, or None if event not in data
    """
    event_date = pd.Timestamp(event_date)
    
    # Find nearest trading day if exact date not in index
    if event_date not in df.index:
        # Find closest date within 3 days
        idx = df.index.get_indexer([event_date], method='nearest')[0]
        if idx == -1:
            return None
        nearest_date = df.index[idx]
        if abs((nearest_date - event_date).days) > 3:
            return None
        event_date = nearest_date
    
    try:
        loc = df.index.get_loc(event_date)
    except KeyError:
        return None
    
    # Ensure we have enough data
    if loc < window or loc + window >= len(df):
        return None
    
    # Slice Pre and Post periods
    pre_slice = df.iloc[loc - window : loc]['Log_Ret']
    post_slice = df.iloc[loc : loc + window]['Log_Ret']
    
    # Calculate realized volatility for short windows
    pre_vol = pre_slice.std() * np.sqrt(252)
    post_vol = post_slice.std() * np.sqrt(252)
    
    # Calculate absolute return on event day
    event_return = df.iloc[loc]['Log_Ret'] if 'Log_Ret' in df.columns else 0
    
    # Determine impact direction
    vol_change = post_vol - pre_vol
    vol_change_pct = (post_vol / pre_vol - 1) * 100 if pre_vol > 0 else 0
    
    if vol_change_pct > 10:
        impact = "Volatility Expansion"
    elif vol_change_pct < -10:
        impact = "Volatility Contraction"
    else:
        impact = "Neutral"
    
    return {
        'event_date': event_date,
        'pre_vol': pre_vol,
        'post_vol': post_vol,
        'vol_change': vol_change,
        'vol_change_pct': vol_change_pct,
        'event_return': event_return,
        'impact': impact
    }


def get_event_summary(
    df: pd.DataFrame, 
    events: pd.DataFrame, 
    window: int = 5
) -> pd.DataFrame:
    """
    Generate summary table for all events within the data range.
    
    Args:
        df: DataFrame with 'Log_Ret' and volatility columns
        events: DataFrame from load_events() or filter_events_in_range()
        window: Days to analyze before/after each event
        
    Returns:
        DataFrame with event impact analysis
    """
    results = []
    
    for _, event in events.iterrows():
        impact = analyze_event_impact(df, event['Date'], window)
        
        if impact:
            results.append({
                'Date': event['Date'].strftime('%Y-%m-%d'),
                'Event': event['Event_Type'],
                'Outcome': event['Outcome'],
                'Pre-Event Vol': f"{impact['pre_vol']:.1%}",
                'Post-Event Vol': f"{impact['post_vol']:.1%}",
                'Change %': f"{impact['vol_change_pct']:+.1f}%",
                'Impact': impact['impact'],
                'Event Return': f"{impact['event_return']*100:+.2f}%"
            })
    
    return pd.DataFrame(results)


def get_event_type_stats(
    df: pd.DataFrame, 
    events: pd.DataFrame, 
    event_type: str,
    window: int = 5
) -> Dict:
    """
    Get aggregate statistics for a specific event type.
    
    E.g., "On average, does volatility expand after RBI MPC meetings?"
    
    Args:
        df: DataFrame with returns data
        events: DataFrame from load_events()
        event_type: Type of event to analyze (e.g., 'RBI MPC')
        window: Analysis window in days
        
    Returns:
        Dict with aggregate statistics
    """
    type_events = events[events['Event_Type'] == event_type]
    
    impacts = []
    for _, event in type_events.iterrows():
        impact = analyze_event_impact(df, event['Date'], window)
        if impact:
            impacts.append(impact)
    
    if not impacts:
        return {'event_type': event_type, 'count': 0}
    
    vol_changes = [i['vol_change_pct'] for i in impacts]
    
    expansion_count = sum(1 for v in vol_changes if v > 10)
    contraction_count = sum(1 for v in vol_changes if v < -10)
    neutral_count = len(vol_changes) - expansion_count - contraction_count
    
    return {
        'event_type': event_type,
        'count': len(impacts),
        'avg_vol_change_pct': np.mean(vol_changes),
        'median_vol_change_pct': np.median(vol_changes),
        'expansions': expansion_count,
        'contractions': contraction_count,
        'neutral': neutral_count,
        'expansion_rate': expansion_count / len(impacts) * 100
    }


def get_upcoming_events(events: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
    """
    Get upcoming events in the next N days.
    
    Args:
        events: DataFrame from load_events()
        days_ahead: Number of days to look ahead
        
    Returns:
        DataFrame with upcoming events
    """
    today = pd.Timestamp.today().normalize()
    end_date = today + timedelta(days=days_ahead)
    
    upcoming = events[
        (events['Date'] >= today) & 
        (events['Date'] <= end_date)
    ].copy()
    
    upcoming['Days_Until'] = (upcoming['Date'] - today).dt.days
    
    return upcoming.sort_values('Date').reset_index(drop=True)
