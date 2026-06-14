"""
supabase_db.py - Supabase Client Interface Layer
================================================
Handles DB connection, type sanitization (casts numpy types & handles NaNs), and schema writes/reads.
"""

import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SUPABASE_ENABLED = False
client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        SUPABASE_ENABLED = True
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("Supabase environment variables (SUPABASE_URL/SUPABASE_KEY) not set. Operating in Local Mode.")


def sanitize_value(val):
    """Convert numpy types to python natives and NaN/NaT to None for json serialization."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        return None if np.isnan(val) or np.isinf(val) else float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (str, datetime)):
        return val
    if pd.isna(val):
        return None
    return val


def sanitize_dict(d: dict) -> dict:
    """Recursively clean dict values."""
    return {k: sanitize_value(v) for k, v in d.items()}


def get_current_timestamp_str() -> str:
    """Return ISO formatted current timestamp."""
    return datetime.utcnow().isoformat() + "Z"


# ==============================================================================
# Database Write Functions
# ==============================================================================

def save_volatility_analysis(ticker: str, metrics_df: pd.DataFrame) -> bool:
    """
    Saves computed volatility metrics from a pandas DataFrame to volatility_analysis table.
    """
    if not SUPABASE_ENABLED or client is None:
        return False

    try:
        records = []
        # We only save records with valid close price
        df = metrics_df.dropna(subset=['Close']) if 'Close' in metrics_df.columns else metrics_df
        
        for idx, row in df.iterrows():
            timestamp_val = idx
            if isinstance(timestamp_val, str):
                timestamp_str = timestamp_val
            elif hasattr(timestamp_val, 'isoformat'):
                timestamp_str = timestamp_val.isoformat()
                if not timestamp_str.endswith('Z') and '+' not in timestamp_str:
                    timestamp_str += 'Z'
            else:
                timestamp_str = str(timestamp_val)

            rec = {
                'ticker': ticker,
                'timestamp': timestamp_str,
                'close_price': row.get('Close'),
                'vol_20d': row.get('Vol_20d') or row.get('Vol_20'),
                'vol_52w_avg': row.get('Vol_52w_Avg'),  # might be computed or null
                'percentile': row.get('Percentile'),
                'regime': row.get('Regime'),
                'ewma_vol': row.get('EWMA'),
                'sma_vol': row.get('SMA_20d') or row.get('SMA_Vol'),
                'donchian_upper': row.get('Donchian_Upper'),
                'donchian_lower': row.get('Donchian_Lower'),
                'volume_sma': row.get('Volume_SMA'),
                'rvol': row.get('RVOL')
            }
            records.append(sanitize_dict(rec))

        if not records:
            return True

        # Perform upsert or insert in batches to prevent payload size limits
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            client.table("volatility_analysis").insert(batch).execute()
        
        logger.info(f"Successfully saved {len(records)} volatility analysis rows to Supabase for {ticker}.")
        return True
    except Exception as e:
        logger.error(f"Error saving volatility analysis to Supabase: {e}")
        return False


def save_gex_key_levels(ticker: str, levels: dict, spot_price: float) -> bool:
    """
    Saves summarized options GEX key levels.
    """
    if not SUPABASE_ENABLED or client is None:
        return False

    try:
        rec = {
            'ticker': ticker,
            'timestamp': get_current_timestamp_str(),
            'spot_price': spot_price,
            'total_net_gex': levels.get('total_net_gex'),
            'total_net_vex': levels.get('total_net_vex'),
            'peak_call_strike': levels.get('peak_call_strike'),
            'peak_put_strike': levels.get('peak_put_strike'),
            'peak_net_strike': levels.get('peak_net_strike'),
            'gamma_flip_price': levels.get('gamma_flip_price'),
            'gex_regime': levels.get('gex_regime'),
            'vex_regime': levels.get('vex_regime'),
            'gex_at_spot': levels.get('gex_at_spot'),
            'vex_at_spot': levels.get('vex_at_spot')
        }
        sanitized = sanitize_dict(rec)
        client.table("gex_key_levels").insert(sanitized).execute()
        logger.info(f"Successfully saved GEX key levels to Supabase for {ticker}.")
        return True
    except Exception as e:
        logger.error(f"Error saving GEX key levels to Supabase: {e}")
        return False


def save_gex_profiles(ticker: str, gex_df: pd.DataFrame) -> bool:
    """
    Saves strike-by-strike synthetic or real GEX profile to gex_profiles table.
    """
    if not SUPABASE_ENABLED or client is None:
        return False

    try:
        records = []
        timestamp = get_current_timestamp_str()
        
        for _, row in gex_df.iterrows():
            rec = {
                'ticker': ticker,
                'timestamp': timestamp,
                'strike': row.get('Strike'),
                'call_oi': row.get('Call_OI'),
                'put_oi': row.get('Put_OI'),
                'oi': row.get('OI'),
                'gamma': row.get('Gamma'),
                'vanna': row.get('Vanna'),
                'call_gex': row.get('Call_GEX'),
                'put_gex': row.get('Put_GEX'),
                'net_gex': row.get('Net_GEX'),
                'call_vex': row.get('Call_VEX'),
                'put_vex': row.get('Put_VEX'),
                'net_vex': row.get('Net_VEX')
            }
            records.append(sanitize_dict(rec))

        if not records:
            return True

        # Write in batches
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            client.table("gex_profiles").insert(batch).execute()

        logger.info(f"Successfully saved {len(records)} GEX profile rows to Supabase for {ticker}.")
        return True
    except Exception as e:
        logger.error(f"Error saving GEX profiles to Supabase: {e}")
        return False


def save_positioning_data(ticker: str, cot_df: pd.DataFrame) -> bool:
    """
    Saves speculative/commercial net positioning data.
    """
    if not SUPABASE_ENABLED or client is None:
        return False

    try:
        records = []
        for idx, row in cot_df.iterrows():
            timestamp_val = idx
            if isinstance(timestamp_val, str):
                timestamp_str = timestamp_val
            elif hasattr(timestamp_val, 'isoformat'):
                timestamp_str = timestamp_val.isoformat()
                if not timestamp_str.endswith('Z') and '+' not in timestamp_str:
                    timestamp_str += 'Z'
            else:
                timestamp_str = str(timestamp_val)

            rec = {
                'ticker': ticker,
                'timestamp': timestamp_str,
                'speculator_net': row.get('Speculator_Net'),
                'commercial_net': row.get('Commercial_Net'),
                'cot_index': row.get('COT_Index'),
                'speculator_52w_min': row.get('Speculator_52w_Min'),
                'speculator_52w_max': row.get('Speculator_52w_Max')
            }
            records.append(sanitize_dict(rec))

        if not records:
            return True

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            client.table("positioning_data").insert(batch).execute()

        logger.info(f"Successfully saved {len(records)} positioning rows to Supabase for {ticker}.")
        return True
    except Exception as e:
        logger.error(f"Error saving positioning data to Supabase: {e}")
        return False


def save_cvd_data(ticker: str, cvd_df: pd.DataFrame, signals_series: pd.Series = None) -> bool:
    """
    Saves cumulative volume delta series and divergence signals.
    """
    if not SUPABASE_ENABLED or client is None:
        return False

    try:
        records = []
        for idx, row in cvd_df.iterrows():
            timestamp_val = idx
            if isinstance(timestamp_val, str):
                timestamp_str = timestamp_val
            elif hasattr(timestamp_val, 'isoformat'):
                timestamp_str = timestamp_val.isoformat()
                if not timestamp_str.endswith('Z') and '+' not in timestamp_str:
                    timestamp_str += 'Z'
            else:
                timestamp_str = str(timestamp_val)

            sig = 0
            if signals_series is not None and timestamp_val in signals_series.index:
                sig = int(signals_series.loc[timestamp_val])

            rec = {
                'ticker': ticker,
                'timestamp': timestamp_str,
                'close_price': row.get('Close'),
                'volume': row.get('Volume'),
                'delta': row.get('Delta'),
                'cvd': row.get('CVD'),
                'divergence_signal': sig
            }
            records.append(sanitize_dict(rec))

        if not records:
            return True

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            client.table("cvd_data").insert(batch).execute()

        logger.info(f"Successfully saved {len(records)} CVD rows to Supabase for {ticker}.")
        return True
    except Exception as e:
        logger.error(f"Error saving CVD data to Supabase: {e}")
        return False


def save_macro_events(events: list) -> bool:
    """
    Seeds macro events into the macro_events table.
    events: list of dicts with keys: date (str 'YYYY-MM-DD'), event_type (str), outcome (str)
    """
    if not SUPABASE_ENABLED or client is None:
        return False

    try:
        records = []
        for event in events:
            rec = {
                'date': event.get('date') or event.get('Date'),
                'event_type': event.get('event_type') or event.get('Event_Type'),
                'outcome': event.get('outcome') or event.get('Outcome')
            }
            records.append(sanitize_dict(rec))

        if not records:
            return True

        # Use upsert to avoid duplicate key violations on unique constraint (date, event_type)
        # Note: supabase-py client supports upsert. Specify on_conflict if necessary.
        client.table("macro_events").upsert(records, on_conflict="date,event_type").execute()
        logger.info(f"Successfully seeded/updated {len(records)} macro events in Supabase.")
        return True
    except Exception as e:
        logger.error(f"Error seeding macro events: {e}")
        return False


# ==============================================================================
# Database Read Functions
# ==============================================================================

def load_macro_events() -> pd.DataFrame:
    """
    Loads macro events from the Supabase macro_events table.
    Returns: DataFrame with columns ['Date', 'Event_Type', 'Outcome'], sorted by Date.
    """
    if not SUPABASE_ENABLED or client is None:
        logger.warning("Supabase disabled. Cannot load macro events from DB.")
        return pd.DataFrame()

    try:
        res = client.table("macro_events").select("date, event_type, outcome").order("date").execute()
        data = res.data
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        # Rename columns to match the local CSV expectation (Date, Event_Type, Outcome)
        df.rename(columns={
            'date': 'Date',
            'event_type': 'Event_Type',
            'outcome': 'Outcome'
        }, inplace=True)
        
        # Parse Dates
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    except Exception as e:
        logger.error(f"Error fetching macro events from Supabase: {e}")
        return pd.DataFrame()
