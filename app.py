"""
app.py - Indian Market Dynamic Volatility Analyzer Dashboard

A Streamlit application for analyzing realized volatility in Indian
equity markets with macro event impact analysis.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

# Import local modules
from vol_engine import (
    calculate_metrics, 
    get_regime_comparison, 
    classify_regime,
    fit_garch
)
from event_analyzer import (
    load_events, 
    filter_events_in_range, 
    get_event_summary,
    get_event_type_stats,
    get_upcoming_events
)

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="🇮🇳 Indian Volatility Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Quant-Grade Institutional Color Palette
# ==============================================================================
QUANT_COLORS = {
    # Canvas & Structure
    'bg': '#F8F9FA',           # Light grey background
    'panel': '#FFFFFF',         # White panels
    'grid': '#E5E7EB',          # Light grid
    'text': '#111827',          # Dark text
    'text_muted': '#6B7280',    # Muted text
    
    # Price & Volatility (Core Signals)
    'price': '#111827',         # Price line (dark)
    'vol_20': '#2563EB',        # Vol 20d (blue)
    'vol_60': '#475569',        # Vol 60d (slate)
    'vol_120': '#94A3B8',       # Vol 120d (light slate)
    'ewma': '#B45309',          # EWMA (muted brown)
    'sma': '#7C3AED',           # SMA Vol (purple)
    
    # Event types
    'rbi': '#EF4444',           # Red - RBI
    'cpi': '#0891B2',           # Cyan - CPI
    'budget': '#D97706',        # Amber - Budget
}

# Minimal CSS - Force light theme everywhere
st.markdown("""
<style>
    /* Force light background everywhere */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: #F8F9FA !important;
    }
    
    /* All text must be dark */
    .stMarkdown, .stMarkdown *, p, span, label, div {
        color: #111827 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #111827 !important;
    }
    
    /* Buttons - ensure white text */
    .stButton > button,
    .stButton > button > div,
    .stButton > button > div > p,
    .stButton > button span,
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"] * {
        background-color: #1E40AF !important;
        color: #FFFFFF !important;
    }
    
    .stButton > button:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
    }
    
    /* DataFrame / Tables - force light theme */
    .stDataFrame, [data-testid="stDataFrame"],
    .stDataFrame > div, [data-testid="stDataFrame"] > div,
    .stDataFrame iframe, [data-testid="stDataFrame"] iframe {
        background-color: #FFFFFF !important;
    }
    
    /* Plotly chart container */
    .stPlotlyChart, [data-testid="stPlotlyChart"] {
        background-color: #FFFFFF !important;
    }
    
    /* Force all nested elements to have readable text */
    .element-container, .stElementContainer {
        color: #111827 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }
    
    /* Captions */
    .stCaption, [data-testid="stCaption"] {
        color: #6B7280 !important;
    }
    
    /* Multiselect pills (Rolling Windows 20, 60 tags) - white text */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] span,
    [data-testid="stMultiSelect"] [data-baseweb="tag"],
    .stMultiSelect span[data-baseweb="tag"] span {
        color: #FFFFFF !important;
    }
    
    /* Print Layout Styling */
    @media print {
        /* Hide sidebar and navigation completely */
        section[data-testid="stSidebar"],
        [data-testid="stSidebar"],
        header,
        footer,
        .stDeployButton,
        [data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Force main page to take full printing width */
        .main, .stApp, [data-testid="stAppViewContainer"], .block-container {
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
            background-color: #FFFFFF !important;
        }
        
        /* Hide print button, other utility buttons, and links */
        .stButton, .stDownloadButton, hr, .stDivider, button, .print-btn-container {
            display: none !important;
        }
        
        /* Avoid breaking middle of charts when printing */
        .stPlotlyChart, .element-container, [data-testid="column"] {
            page-break-inside: avoid !important;
        }
        
        /* Premium metric card styling on paper */
        [data-testid="stMetricValue"] {
            font-size: 20px !important;
        }
        
        /* Ensure dynamic charts colors are exact on PDF print */
        body {
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }

        /* Force Streamlit horizontal columns to stack vertically to ensure absolute vertical readability in PDF */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 16px !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            margin-bottom: 12px !important;
        }

        /* Tables full width & no paging cuts */
        table, [data-testid="stTable"] table {
            width: 100% !important;
            table-layout: auto !important;
            page-break-inside: avoid !important;
        }

        /* Set clean standard margins for printing */
        @page {
            size: auto;
            margin: 15mm;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Sidebar - User Inputs
# ==============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/a/a9/Flag_of_India.svg", width=60)
st.sidebar.title("🇮🇳 Volatility Analyzer")
st.sidebar.markdown("---")

# Exchange / Market selection
exchange = st.sidebar.radio(
    "📈 Select Exchange",
    ["NSE", "BSE", "Global"],
    horizontal=True,
    help="Global: enter any Yahoo Finance ticker (e.g. AAPL, BTC-USD, ^GSPC)"
)

# Popular tickers by exchange
POPULAR_NSE = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "TATASTEEL.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS",
    "^NSEI", "^NSEBANK"  # Nifty 50 and Bank Nifty indices
]

POPULAR_BSE = [
    "RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO", "ICICIBANK.BO",
    "TATASTEEL.BO", "SBIN.BO", "BHARTIARTL.BO", "ITC.BO", "HINDUNILVR.BO",
    "^BSESN"  # Sensex
]

POPULAR_GLOBAL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "^GSPC", "^DJI", "^VIX",
    "BTC-USD", "GC=F", "CL=F"  # Bitcoin, Gold, Crude Oil
]

suffix = ".NS" if exchange == "NSE" else (".BO" if exchange == "BSE" else "")

# --- Custom ticker input (always visible at top) ---
st.sidebar.markdown("### 🔍 Custom Ticker")
custom_raw = st.sidebar.text_input(
    "Enter any ticker symbol",
    value="",
    placeholder=f"e.g. {'RELIANCE' if exchange != 'Global' else 'AAPL'}",
    help="Type any Yahoo Finance symbol. Leave blank to use the popular list below."
)

# Auto-append suffix for NSE/BSE if user forgot it
auto_suffix = st.sidebar.checkbox(
    f"Auto-add {suffix} suffix" if suffix else "Auto-add suffix",
    value=True,
    disabled=(exchange == "Global"),
    help=f"Automatically appends '{suffix}' if not already present. Disable for indices (^NSEI)."
)

if custom_raw.strip():
    raw = custom_raw.strip().upper()
    if exchange != "Global" and auto_suffix and suffix and not raw.endswith(suffix) and not raw.startswith("^"):
        ticker = raw + suffix
        st.sidebar.caption(f"ℹ️ Using symbol: `{ticker}`")
    else:
        ticker = raw
        st.sidebar.caption(f"ℹ️ Using symbol: `{ticker}`")
else:
    # Fall back to popular tickers dropdown
    st.sidebar.markdown("---")
    st.sidebar.markdown("**— or pick from popular —**")
    POPULAR_TICKERS = POPULAR_NSE if exchange == "NSE" else (POPULAR_BSE if exchange == "BSE" else POPULAR_GLOBAL)
    ticker_option = st.sidebar.selectbox(
        "📌 Popular Stocks",
        POPULAR_TICKERS,
        index=3  # Default to HDFCBANK / NVDA
    )
    ticker = ticker_option

st.sidebar.markdown("---")

# Date range selection
st.sidebar.markdown("### 📅 Date Range")
col1, col2 = st.sidebar.columns(2)

default_end = datetime.today()
default_start = default_end - timedelta(days=2*365)  # 2 years

with col1:
    start_date = st.date_input("From", value=default_start)
with col2:
    end_date = st.date_input("To", value=default_end)

# Analysis options
st.sidebar.markdown("### ⚙️ Analysis Options")
show_garch = st.sidebar.checkbox("Enable GARCH(1,1) Forecast", value=False)
show_egarch = st.sidebar.checkbox(
    "Enable EGARCH(1,1) Forecast",
    value=False,
    help="Exponential GARCH: adds γ (gamma) to capture the leverage effect (bearish shocks spike volatility harder)"
)

from lstm_vol_engine import is_available as lstm_available
_lstm_ready = lstm_available()
show_lstm = st.sidebar.checkbox(
    "Enable GARCH→LSTM Forecast",
    value=False,
    disabled=not _lstm_ready,
    help="Pipes GARCH features into a stacked LSTM for multi-horizon volatility forecasting"
)
show_advanced = st.sidebar.checkbox(
    "Enable Advanced Time Series & Simulations",
    value=True,
    help="Enables Monte Carlo simulations, ARIMA, Hurst memory, and automated Support/Resistance models"
)

# New SMA Volatility & Donchian / Volume Options
st.sidebar.markdown("### 📊 SMA & Donchian Options")
sma_window = st.sidebar.slider(
    "SMA Volatility Window (days)",
    5, 60, 20,
    help="Lookback period for Simple Moving Average Volatility (zero-mean assumption)"
)
show_donchian = st.sidebar.checkbox(
    "Show Donchian Channels",
    value=True,
    help="Overlays rolling price channels on the price subplot"
)
donchian_window = st.sidebar.slider(
    "Donchian & Volume Window (days)",
    5, 60, 20,
    help="Lookback period for Donchian Channels and RVOL indicator"
)

event_window = st.sidebar.slider("Event Analysis Window (days)", 3, 10, 5)

# Volatility windows
vol_windows = st.sidebar.multiselect(
    "Rolling Windows",
    [10, 20, 30, 60, 90, 120],
    default=[20, 60]
)

# ==============================================================================
# Data Fetching
# ==============================================================================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        # Take only the first level (Price, not Ticker)
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLC data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if len(data) == 0:
            return None
        # Flatten MultiIndex columns (yfinance returns MultiIndex for single ticker)
        data = flatten_columns(data)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_event_calendar():
    """Load macro events calendar."""
    try:
        return load_events()
    except:
        st.warning("⚠️ Event calendar not found. Event analysis disabled.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_benchmark_returns(start: str, end: str) -> dict:
    """Fetch Nifty 50 (^NSEI) and Sensex (^BSESN) returns for correlation analysis."""
    benchmarks = {
        'Nifty 50': '^NSEI',
        'Sensex': '^BSESN'
    }
    rets = {}
    for name, symbol in benchmarks.items():
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if df is not None and not df.empty:
                df = flatten_columns(df)
                if 'Close' in df.columns:
                    closes = df['Close'].dropna()
                    # Strip timezone from index for consistent alignment
                    closes.index = closes.index.tz_localize(None)
                    ret = np.log(closes / closes.shift(1)).dropna()
                    rets[name] = ret
        except Exception as e:
            pass
    return rets

@st.cache_data(ttl=300)  # Cache for 5 minutes (near real-time)
def fetch_intraday_hl(ticker: str) -> dict:
    """Fetch today's 1-day high/low/open/close from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d", interval="1d")
        if hist.empty:
            return None
        hist = flatten_columns(hist) if isinstance(hist.columns, pd.MultiIndex) else hist
        today = hist.iloc[-1]
        prev  = hist.iloc[-2] if len(hist) > 1 else None
        day_range = float(today['High']) - float(today['Low'])
        close     = float(today['Close'])
        position  = ((close - float(today['Low'])) / day_range * 100) if day_range > 0 else 50.0
        chg       = ((close - float(prev['Close'])) / float(prev['Close']) * 100) if prev is not None else None
        return {
            'high':     float(today['High']),
            'low':      float(today['Low']),
            'open':     float(today['Open']),
            'close':    close,
            'range':    day_range,
            'position': position,   # 0–100 % from low
            'chg_pct':  chg,
            'date':     str(hist.index[-1].date()),
        }
    except Exception as e:
        return None

def calculate_price_targets(data: pd.DataFrame, metrics: pd.DataFrame, current_price: float = None) -> dict:
    """
    Derive statistical price targets from realized vol, ATR, and Bollinger Bands.
    All levels are based purely on price/vol data — no fundamental inputs.
    """
    close   = data['Close'].dropna()
    high    = data['High'].dropna()
    low     = data['Low'].dropna()
    last    = current_price if current_price is not None else float(close.iloc[-1])

    # --- 1. Vol-based lognormal bands (±1σ, ±2σ daily move) ---
    vol_col = next((c for c in ['Vol_20d', 'Vol_20', 'EWMA'] if c in metrics.columns), None)
    ann_vol = float(metrics[vol_col].dropna().iloc[-1]) if vol_col else None
    daily_sigma = ann_vol / (252 ** 0.5) if ann_vol else None

    sigma_bands = {}
    if daily_sigma:
        sigma_bands = {
            'bull_2s': last * np.exp( 2 * daily_sigma),
            'bull_1s': last * np.exp( 1 * daily_sigma),
            'bear_1s': last * np.exp(-1 * daily_sigma),
            'bear_2s': last * np.exp(-2 * daily_sigma),
        }

    # --- 2. ATR-based support / resistance (14-day) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])
    atr_bands = {
        'atr_res': last + 1.5 * atr14,
        'atr_sup': last - 1.5 * atr14,
        'atr14':   atr14,
    }

    # --- 3. Bollinger Bands (20-day, ±2σ price) ---
    sma20   = float(close.rolling(20).mean().iloc[-1])
    std20   = float(close.rolling(20).std().iloc[-1])
    bb_bands = {
        'bb_upper': sma20 + 2 * std20,
        'bb_mid':   sma20,
        'bb_lower': sma20 - 2 * std20,
    }

    # --- 4. 52-week high / low ---
    w52 = close.iloc[-252:] if len(close) >= 252 else close
    week52 = {
        'high52': float(w52.max()),
        'low52':  float(w52.min()),
    }

    return {
        'last':       last,
        'daily_sigma': daily_sigma,
        'ann_vol':    ann_vol,
        'atr14':      atr14,
        **sigma_bands,
        **atr_bands,
        **bb_bands,
        **week52,
    }

# ==============================================================================
# Main App
# ==============================================================================

# Simple header and Print button using columns
hc1, hc2 = st.columns([4, 1])
with hc1:
    st.title("Volatility Risk Dashboard")
    st.caption("Dynamic Realized Volatility Estimator • NSE/BSE • India")
with hc2:
    st.markdown("""
        <div class="print-btn-container" style="text-align: right; padding-top: 24px;">
            <button onclick="window.print()" style="
                background-color: #1E40AF;
                color: white;
                border: none;
                padding: 10px 18px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
                cursor: pointer;
                font-family: 'IBM Plex Sans', sans-serif;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: background-color 0.2s;
            " onmouseover="this.style.backgroundColor='#1E3A8A'" onmouseout="this.style.backgroundColor='#1E40AF'">
                🖨️ Print PDF Report
            </button>
        </div>
    """, unsafe_allow_html=True)

# Ticker info
st.markdown(f"**Symbol:** `{ticker}` | **Period:** {start_date} → {end_date}")
st.divider()

# Add analyze button
analyze_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

if analyze_clicked or 'data_loaded' in st.session_state:
    with st.spinner("Fetching market data..."):
        data = fetch_data(ticker, str(start_date), str(end_date))
    
    if data is None or len(data) == 0:
        st.error(f"❌ No data found for {ticker}. Please check the symbol and try again.")
        st.info("💡 Tip: Use .NS suffix for NSE stocks (e.g., RELIANCE.NS)")
    else:
        st.session_state['data_loaded'] = True
        
        # Calculate volatility metrics with dynamic SMA window
        metrics = calculate_metrics(data['Close'], sma_window=sma_window)
        
        # Recalculate with custom windows if different from default
        if vol_windows and vol_windows != [20, 60]:
            from vol_engine import calculate_log_returns, calculate_rolling_vol, calculate_ewma_vol
            log_ret = calculate_log_returns(data['Close'])
            custom_vol = calculate_rolling_vol(log_ret, vol_windows)
            for col in custom_vol.columns:
                if col not in metrics.columns:
                    metrics[col] = custom_vol[col]
        
        # Calculate Donchian and Volume channels
        from vol_engine import calculate_donchian_channels, calculate_volume_metrics
        
        high_s = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
        low_s = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
        vol_s = data['Volume'].iloc[:, 0] if isinstance(data['Volume'], pd.DataFrame) else data['Volume']
        
        donchian = calculate_donchian_channels(high_s, low_s, window=donchian_window)
        vol_metrics = calculate_volume_metrics(vol_s, window=donchian_window)
        
        # Merge metrics with original data
        full_data = data.copy()
        for col in metrics.columns:
            full_data[col] = metrics[col].values
            
        full_data['Donchian_Upper'] = donchian['Donchian_Upper'].values
        full_data['Donchian_Lower'] = donchian['Donchian_Lower'].values
        full_data['Donchian_Middle'] = donchian['Donchian_Middle'].values
        full_data['Volume_SMA'] = vol_metrics['Volume_SMA'].values
        full_data['RVOL'] = vol_metrics['RVOL'].values
        
        # Load events
        events_df = load_event_calendar()
        if len(events_df) > 0:
            events_in_range = filter_events_in_range(events_df, start_date, end_date)
        else:
            events_in_range = pd.DataFrame()
        
        # ------------------------------------------------------------------
        # Key Metrics Row
        # ------------------------------------------------------------------
        st.markdown("### Current Risk Regime")
        
        # Get regime stats for 20-day vol
        vol_col = 'Vol_20d' if 'Vol_20d' in metrics.columns else 'Vol_20'
        if vol_col not in metrics.columns:
            vol_col = [c for c in metrics.columns if 'Vol_20' in c][0] if any('Vol_20' in c for c in metrics.columns) else 'EWMA'
        
        regime_stats = get_regime_comparison(metrics[vol_col])
        current_regime = classify_regime(regime_stats['current'], regime_stats)
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Current Volatility",
                f"{regime_stats['current']:.1%}" if regime_stats['current'] else "N/A"
            )
        with col2:
            st.metric(
                "52-Week Average",
                f"{regime_stats['avg_52w']:.1%}" if regime_stats['avg_52w'] else "N/A"
            )
        with col3:
            st.metric(
                "52-Week Range",
                f"{regime_stats['min_52w']:.1%} - {regime_stats['max_52w']:.1%}" if regime_stats['min_52w'] else "N/A"
            )
        with col4:
            st.metric(
                "Percentile",
                f"{regime_stats['percentile']:.0f}th" if regime_stats['percentile'] else "N/A"
            )
        with col5:
            st.metric(
                "Regime",
                current_regime
            )
        
        # ------------------------------------------------------------------
        # 1-Day High / Low Snapshot
        # ------------------------------------------------------------------
        st.markdown("### 📅 1-Day High / Low Snapshot")

        hl = fetch_intraday_hl(ticker)
        if hl:
            hl_c1, hl_c2, hl_c3, hl_c4, hl_c5 = st.columns(5)
            chg_str  = f"{hl['chg_pct']:+.2f}%" if hl['chg_pct'] is not None else "N/A"
            chg_delta = f"{hl['chg_pct']:.2f}" if hl['chg_pct'] is not None else None

            with hl_c1:
                st.metric("Day High",  f"₹{hl['high']:,.2f}")
            with hl_c2:
                st.metric("Day Low",   f"₹{hl['low']:,.2f}")
            with hl_c3:
                st.metric("Last Close", f"₹{hl['close']:,.2f}",
                          delta=chg_str if hl['chg_pct'] is not None else None)
            with hl_c4:
                st.metric("Day Range", f"₹{hl['range']:,.2f}")
            with hl_c5:
                st.metric("Position in Range", f"{hl['position']:.1f}%",
                          help="0% = at day low, 100% = at day high")

            # Visual gauge — horizontal bar showing where close sits in the day range
            pos = hl['position'] / 100  # 0.0 – 1.0
            gauge_color = (
                "#059669" if pos >= 0.65 else   # upper third → green
                "#DC2626" if pos <= 0.35 else   # lower third → red
                "#D97706"                        # middle       → amber
            )
            st.markdown(
                f"""
                <div style="margin:6px 0 2px 0; font-size:12px; color:#6B7280;">
                    Day Range Gauge &nbsp;|&nbsp;
                    <span style="color:#6B7280;">Low ₹{hl['low']:,.2f}</span>
                    &nbsp;―&nbsp;
                    <span style="color:{gauge_color}; font-weight:600;">Close ₹{hl['close']:,.2f}</span>
                    &nbsp;―&nbsp;
                    <span style="color:#6B7280;">High ₹{hl['high']:,.2f}</span>
                    &nbsp;&nbsp;<span style='color:#9CA3AF; font-size:11px;'>({hl['date']})</span>
                </div>
                <div style="background:#E5E7EB; border-radius:6px; height:14px; position:relative; overflow:hidden;">
                    <div style="
                        width:{hl['position']:.1f}%;
                        background:{gauge_color};
                        height:100%;
                        border-radius:6px;
                        transition:width 0.4s ease;
                    "></div>
                    <div style="
                        position:absolute;
                        left:calc({hl['position']:.1f}% - 1px);
                        top:0; bottom:0;
                        width:3px;
                        background:#111827;
                        border-radius:2px;
                    "></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(f"🕐 Refreshes every 5 min  |  Data: Yahoo Finance  |  {hl['date']}")
        else:
            st.info("⚠️ Could not fetch intraday data for this ticker. Historical analysis continues below.")

        st.divider()

        # ------------------------------------------------------------------
        # GARCH(1,1) Forecast  (shown above charts for forward-looking context)
        # ------------------------------------------------------------------
        if show_garch:
            st.markdown("### GARCH(1,1) Volatility Forecast")

            with st.spinner("Fitting GARCH model..."):
                garch_result = fit_garch(metrics['Log_Ret'])
                st.session_state['_garch_result_cache'] = garch_result

            if garch_result:
                gcol1, gcol2, gcol3, gcol4 = st.columns(4)

                with gcol1:
                    st.metric("α (Shock Reaction)",  f"{garch_result['alpha']:.4f}"      if garch_result['alpha']       else "N/A")
                with gcol2:
                    st.metric("β (Persistence)",      f"{garch_result['beta']:.4f}"       if garch_result['beta']        else "N/A")
                with gcol3:
                    st.metric("Persistence (α+β)",    f"{garch_result['persistence']:.4f}" if garch_result['persistence'] else "N/A")
                with gcol4:
                    st.metric("1-Day Forecast",        f"{garch_result['forecast_1d']:.1%}" if garch_result['forecast_1d'] else "N/A")

                if garch_result['persistence'] and garch_result['persistence'] > 0.95:
                    st.warning("⚠️ High persistence — volatility shocks are slow to decay.")
            else:
                st.warning("GARCH model could not be fitted. Try a longer data period.")

            st.divider()

        # ------------------------------------------------------------------
        # EGARCH(1,1) Forecast — Leverage Effect
        # ------------------------------------------------------------------
        if show_egarch:
            st.markdown("### 📐 EGARCH(1,1) — Leverage Effect")
            st.caption(
                "EGARCH solves GARCH's symmetry blind spot by adding γ (gamma) to capture "
                "shock direction — bearish sell-offs spike volatility harder than bullish rallies of equal size."
            )

            with st.spinner("Fitting EGARCH model..."):
                from vol_engine import fit_egarch
                egarch_result = fit_egarch(metrics['Log_Ret'])
                st.session_state['_egarch_result_cache'] = egarch_result

            if egarch_result:
                ec1, ec2, ec3, ec4, ec5 = st.columns(5)
                with ec1:
                    st.metric("α (Shock Size)", f"{egarch_result['alpha']:.4f}" if egarch_result['alpha'] is not None else "N/A")
                with ec2:
                    st.metric("γ (Leverage)", f"{egarch_result['gamma']:.4f}" if egarch_result['gamma'] is not None else "N/A",
                              delta=egarch_result.get('leverage_dir', ''),
                              delta_color="inverse" if (egarch_result.get('gamma') or 0) < 0 else "normal")
                with ec3:
                    st.metric("β (Persistence)", f"{egarch_result['beta']:.4f}" if egarch_result['beta'] is not None else "N/A")
                with ec4:
                    lev_pct = egarch_result.get('leverage_pct')
                    st.metric("Leverage Premium", f"{lev_pct:+.1f}%" if lev_pct is not None else "N/A")
                with ec5:
                    st.metric("1-Day Forecast", f"{egarch_result['forecast_1d']:.1%}" if egarch_result['forecast_1d'] else "N/A")
            else:
                st.warning("EGARCH model could not be fitted. Try a longer data period.")

            st.divider()

        # ------------------------------------------------------------------
        # Stacked LSTM Forecast — Neural Volatility
        # ------------------------------------------------------------------
        lstm_ok = False
        if show_lstm:
            st.markdown("### 🧠 Neural Volatility Forecast — GARCH → LSTM")
            st.caption("Pipes GARCH baseline conditional volatility features into a stacked LSTM to learn non-linear regime transitions.")

            _garch_for_lstm = st.session_state.get('_garch_result_cache')
            if _garch_for_lstm is None:
                with st.spinner("Fitting GARCH for neural feature extraction..."):
                    _garch_for_lstm = fit_garch(metrics['Log_Ret'])
                    st.session_state['_garch_result_cache'] = _garch_for_lstm

            if _garch_for_lstm is None:
                st.error("GARCH fitting failed — neural forecast unavailable. Try a longer data period.")
            else:
                lstm_progress_bar = st.progress(0, text="Initialising LSTM...")
                def _lstm_progress(epoch, total, loss):
                    pct = int(epoch / total * 100)
                    lstm_progress_bar.progress(pct, text=f"Training LSTM — epoch {epoch}/{total} | loss {loss:.5f}")

                with st.spinner("Training GARCH→LSTM neural forecaster..."):
                    try:
                        from lstm_vol_engine import GARCHLSTMForecaster
                        forecaster = GARCHLSTMForecaster(returns=metrics['Log_Ret'], garch_params=_garch_for_lstm, epochs=80, seq_len=30)
                        lstm_result = forecaster.run(progress_callback=_lstm_progress)
                        lstm_progress_bar.progress(100, text="✅ Training complete")
                        lstm_ok = True
                    except Exception as _lstm_err:
                        st.error(f"LSTM training failed: {_lstm_err}")
                        lstm_ok = False

                if lstm_ok:
                    rp = lstm_result['regime_probs']
                    regime_lbl = lstm_result['regime']
                    regime_color = {'Compression': '#059669', 'Neutral': '#D97706', 'Expansion': '#DC2626'}.get(regime_lbl, '#6B7280')

                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid {regime_color}; padding: 10px 16px; background: #F9FAFB; border-radius: 0 8px 8px 0; margin-bottom: 12px;">
                            <span style="font-size:16px; font-weight:700; color:{regime_color};">{regime_lbl.upper()} REGIME</span>
                            &nbsp;&nbsp;
                            <span style="font-size:12px; color:#6B7280;">
                                Compression {rp['Compression']*100:.1f}% &middot;
                                Neutral {rp['Neutral']*100:.1f}% &middot;
                                Expansion {rp['Expansion']*100:.1f}%
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    fc = lstm_result['forecasts']
                    lc1, lc2, lc3, lc4, lc5 = st.columns(5)
                    with lc1:
                        st.metric("GARCH σ (current)", f"{lstm_result['current_sigma']*100:.2f}%")
                    with lc2:
                        st.metric("LSTM 1-Day", f"{fc[1]*100:.2f}%", delta=f"{lstm_result['lstm_vs_garch']:+.1f}% vs GARCH")
                    with lc3:
                        st.metric("LSTM 1-Week", f"{fc[5]*100:.2f}%")
                    with lc4:
                        st.metric("LSTM 2-Week", f"{fc[10]*100:.2f}%")
                    with lc5:
                        st.metric("LSTM 1-Month", f"{fc[21]*100:.2f}%")

                    # In-sample chart
                    st.markdown("**GARCH σ_t vs LSTM-predicted σ_t (in-sample)**")
                    in_sigma = lstm_result['in_sample_sigma']
                    garch_sigma = forecaster._features['sigma_t']

                    fig_lstm = go.Figure()
                    fig_lstm.add_trace(go.Scatter(x=garch_sigma.index, y=garch_sigma * 100, name="GARCH σ_t", line=dict(color='#1E40AF', width=1.2)))
                    fig_lstm.add_trace(go.Scatter(x=in_sigma.index, y=in_sigma * 100, name="LSTM σ̂_t", line=dict(color='#DC2626', width=1.2, dash='dot')))
                    fig_lstm.update_layout(
                        height=320,
                        paper_bgcolor=QUANT_COLORS['bg'],
                        plot_bgcolor=QUANT_COLORS['panel'],
                        font=dict(color=QUANT_COLORS['text'], family='IBM Plex Sans'),
                        margin=dict(l=50, r=30, t=40, b=40),
                        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1)
                    )
                    st.plotly_chart(fig_lstm, use_container_width=True)

            st.divider()

        # ------------------------------------------------------------------
        # 3-Model Synthesis Narrative Verdict
        # ------------------------------------------------------------------
        _any_forecast = show_garch or show_egarch or show_lstm
        if _any_forecast:
            st.markdown("### 🧾 Forecast Conclusion — Model Synthesis")
            st.caption("Side-by-side comparison of active forecasting models.")

            _gc = st.session_state.get('_garch_result_cache')
            _ec = st.session_state.get('_egarch_result_cache')
            _lc = lstm_result if show_lstm and 'lstm_result' in locals() and lstm_ok else None

            rows = []
            if show_garch and _gc:
                g_alpha = f"{_gc['alpha']:.4f}" if _gc['alpha'] is not None else "N/A"
                g_beta = f"{_gc['beta']:.4f}" if _gc['beta'] is not None else "N/A"
                rows.append({
                    'Model': 'GARCH(1,1)',
                    'Edge': 'Symmetric shock response, variance mean-reversion',
                    'α / γ / β': f"α={g_alpha}  β={g_beta}",
                    'Persistence': f"{_gc['persistence']:.4f}" if _gc['persistence'] is not None else 'N/A',
                    '1D Forecast': f"{_gc['forecast_1d']:.2%}" if _gc['forecast_1d'] is not None else 'N/A'
                })
            if show_egarch and _ec:
                e_alpha = f"{_ec['alpha']:.4f}" if _ec['alpha'] is not None else "N/A"
                e_gamma = f"{_ec['gamma']:.4f}" if _ec['gamma'] is not None else "N/A"
                e_beta = f"{_ec['beta']:.4f}" if _ec['beta'] is not None else "N/A"
                rows.append({
                    'Model': 'EGARCH(1,1)',
                    'Edge': f"Leverage effect — {_ec.get('leverage_dir', 'N/A')}",
                    'α / γ / β': f"α={e_alpha}  γ={e_gamma}  β={e_beta}",
                    'Persistence': f"{_ec['persistence']:.4f}" if _ec['persistence'] is not None else 'N/A',
                    '1D Forecast': f"{_ec['forecast_1d']:.2%}" if _ec['forecast_1d'] is not None else 'N/A'
                })
            if show_lstm and _lc:
                fc = _lc['forecasts']
                rows.append({
                    'Model': 'GARCH → LSTM',
                    'Edge': f"Non-linear regimes. Regime: {_lc['regime']}",
                    'α / γ / β': 'Learned by Neural Weights',
                    'Persistence': 'Implicit (LSTM Memory)',
                    '1D Forecast': f"{fc[1]:.2%}"
                })
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

                # Narrative verdict
                forecasts_1d = {}
                if show_garch and _gc and _gc['forecast_1d']: forecasts_1d['GARCH'] = _gc['forecast_1d']
                if show_egarch and _ec and _ec['forecast_1d']: forecasts_1d['EGARCH'] = _ec['forecast_1d']
                if show_lstm and _lc: forecasts_1d['LSTM'] = _lc['forecasts'][1]

                if forecasts_1d:
                    avg_fc = np.mean(list(forecasts_1d.values()))
                    st.markdown(
                        f"""
                        <div style="border-left:4px solid #1E40AF; padding:12px 18px; background:#F9FAFB; border-radius:0 8px 8px 0; margin-top:12px;">
                            <div style="font-size:13px; font-weight:700; color:#1E40AF; margin-bottom:6px;">VERDICT — MODEL SYNTHESIS</div>
                            <div style="font-size:12px; color:#374151; line-height:1.7;">
                                Consensus 1-day annualised volatility forecast: <strong>{avg_fc*100:.2f}%</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.divider()

        # ------------------------------------------------------------------
        # Statistical Price Targets
        # ------------------------------------------------------------------
        st.markdown("### 🎯 Statistical Price Targets")
        st.caption("Derived from realized volatility, ATR-14, and Bollinger Bands — no fundamental inputs.")

        current_p = hl['close'] if 'hl' in locals() and hl else None
        pt = calculate_price_targets(full_data, metrics, current_price=current_p)
        last = pt['last']

        # ---- Metric cards row 1: Vol-based bands ----
        if pt.get('daily_sigma'):
            st.markdown("**📊 Volatility Bands** &nbsp; *(±1σ / ±2σ daily lognormal move from last close)*", unsafe_allow_html=True)
            vc1, vc2, vc3, vc4, vc5 = st.columns(5)
            with vc1:
                st.metric("🔴 Bear 2σ",  f"₹{pt['bear_2s']:,.2f}", f"{(pt['bear_2s']/last - 1)*100:+.2f}%")
            with vc2:
                st.metric("🟠 Bear 1σ",  f"₹{pt['bear_1s']:,.2f}", f"{(pt['bear_1s']/last - 1)*100:+.2f}%")
            with vc3:
                st.metric("⚪ Last Close", f"₹{last:,.2f}")
            with vc4:
                st.metric("🟠 Bull 1σ",  f"₹{pt['bull_1s']:,.2f}", f"{(pt['bull_1s']/last - 1)*100:+.2f}%")
            with vc5:
                st.metric("🟢 Bull 2σ",  f"₹{pt['bull_2s']:,.2f}", f"{(pt['bull_2s']/last - 1)*100:+.2f}%")

        # ---- Metric cards row 2: ATR & Bollinger ----
        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        with ac1:
            st.metric("ATR Support",    f"₹{pt['atr_sup']:,.2f}", f"{(pt['atr_sup']/last - 1)*100:+.2f}%")
        with ac2:
            st.metric("BB Lower",       f"₹{pt['bb_lower']:,.2f}", f"{(pt['bb_lower']/last - 1)*100:+.2f}%")
        with ac3:
            st.metric("BB Mid (SMA20)", f"₹{pt['bb_mid']:,.2f}",   f"{(pt['bb_mid']/last - 1)*100:+.2f}%")
        with ac4:
            st.metric("BB Upper",       f"₹{pt['bb_upper']:,.2f}", f"{(pt['bb_upper']/last - 1)*100:+.2f}%")
        with ac5:
            st.metric("ATR Resistance", f"₹{pt['atr_res']:,.2f}",  f"{(pt['atr_res']/last - 1)*100:+.2f}%")

        # ---- Horizontal target chart ----
        levels = []
        if pt.get('bear_2s'):
            levels += [
                ("Bear 2σ",        pt['bear_2s'],    "#DC2626"),
                ("Bear 1σ",        pt['bear_1s'],    "#F87171"),
            ]
        levels += [
            ("ATR Support",     pt['atr_sup'],    "#B45309"),
            ("BB Lower",        pt['bb_lower'],   "#6B7280"),
            ("BB Mid (SMA20)",  pt['bb_mid'],     "#475569"),
            ("Last Close",      last,             "#111827"),
            ("BB Upper",        pt['bb_upper'],   "#0891B2"),
            ("ATR Resistance",  pt['atr_res'],    "#92400E"),
        ]
        if pt.get('bull_1s'):
            levels += [
                ("Bull 1σ",        pt['bull_1s'],    "#34D399"),
                ("Bull 2σ",        pt['bull_2s'],    "#059669"),
            ]
        levels += [
            ("52W High",        pt['high52'],     "#4F46E5"),
            ("52W Low",         pt['low52'],      "#7C3AED"),
        ]

        # Sort by price for a clean waterfall look
        levels_sorted = sorted(levels, key=lambda x: x[1])
        names  = [l[0] for l in levels_sorted]
        prices = [l[1] for l in levels_sorted]
        colors = [l[2] for l in levels_sorted]
        pct    = [(p / last - 1) * 100 for p in prices]

        fig_pt = go.Figure()
        fig_pt.add_trace(go.Bar(
            y=names,
            x=pct,
            orientation='h',
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"₹{p:,.2f}  ({d:+.2f}%)" for p, d in zip(prices, pct)],
            textposition='outside',
            textfont=dict(size=11, color='#111827'),
            hovertemplate='%{y}: ₹%{text}<extra></extra>',
        ))
        # Zero line = last close
        fig_pt.add_vline(x=0, line=dict(color='#111827', width=1.5, dash='dot'))
        fig_pt.update_layout(
            height=340,
            paper_bgcolor=QUANT_COLORS['bg'],
            plot_bgcolor=QUANT_COLORS['panel'],
            font=dict(color=QUANT_COLORS['text'], family='IBM Plex Sans', size=11),
            xaxis=dict(
                title="% from Last Close",
                tickformat='.1f',
                ticksuffix='%',
                gridcolor=QUANT_COLORS['grid'],
                zeroline=False,
            ),
            yaxis=dict(gridcolor='rgba(0,0,0,0)'),
            margin=dict(l=120, r=140, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_pt, use_container_width=True)
        st.caption(
            f"📌 ATR-14: ₹{pt['atr14']:,.2f}  ·  "
            f"Daily σ: {pt['daily_sigma']*100:.2f}%  ·  "
            f"Ann. Vol: {pt['ann_vol']*100:.1f}%" if pt.get('daily_sigma') else ""
        )

        st.divider()

        # ------------------------------------------------------------------
        # Main Volatility Chart - Quant Style
        # ------------------------------------------------------------------
        st.markdown("### PRICE & VOLATILITY")
        
        # Create dual-axis chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.55, 0.45],
            subplot_titles=(f'{ticker} Close', 'Realized Volatility')
        )
        
        # Price line - Near black, thin
        fig.add_trace(
            go.Scatter(
                x=full_data.index,
                y=full_data['Close'],
                name="Close",
                line=dict(color=QUANT_COLORS['price'], width=1.2),
                hovertemplate='₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Donchian Channels (shaded background on price chart)
        if show_donchian and 'Donchian_Upper' in full_data.columns and 'Donchian_Lower' in full_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=full_data.index,
                    y=full_data['Donchian_Upper'],
                    name="Donchian Upper",
                    line=dict(color='rgba(107, 114, 128, 0.25)', width=0.8),
                    showlegend=True,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=full_data.index,
                    y=full_data['Donchian_Lower'],
                    name="Donchian Lower",
                    fill='tonexty',
                    fillcolor='rgba(107, 114, 128, 0.05)',
                    line=dict(color='rgba(107, 114, 128, 0.25)', width=0.8),
                    showlegend=True,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
        
        # Volatility lines - Quant palette (blue-grey spectrum + brown for EWMA)
        quant_vol_colors = {
            'Vol_20d': QUANT_COLORS['vol_20'],   # Deep blue
            'Vol_60d': QUANT_COLORS['vol_60'],   # Slate
            'Vol_120d': QUANT_COLORS['vol_120'], # Lighter slate
            'EWMA': QUANT_COLORS['ewma'],         # Muted brown
        }
        vol_columns = [c for c in metrics.columns if 'Vol_' in c or c == 'EWMA' or 'SMA_' in c]
        
        for col in vol_columns:
            if 'SMA_' in col:
                color = QUANT_COLORS['sma']
                dash_style = 'dashdot'
            else:
                color = quant_vol_colors.get(col, QUANT_COLORS['text_muted'])
                dash_style = 'dot' if col == 'EWMA' else 'solid'
                
            fig.add_trace(
                go.Scatter(
                    x=metrics.index,
                    y=metrics[col],
                    name=col.replace('_', ' ').replace('Vol ', ''),
                    line=dict(color=color, width=1.2, dash=dash_style),
                    hovertemplate='%{y:.1%}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add event markers if available - Thin dashed lines
        if len(events_in_range) > 0:
            event_colors = {
                'RBI MPC': QUANT_COLORS['rbi'],
                'India CPI': QUANT_COLORS['cpi'],
                'Union Budget': QUANT_COLORS['budget'],
                'Interim Budget': QUANT_COLORS['budget'],
            }
            for _, event in events_in_range.iterrows():
                event_date = event['Date']
                evt_type = event['Event_Type']
                evt_color = event_colors.get(evt_type, QUANT_COLORS['text_muted'])
                if event_date in full_data.index:
                    fig.add_vline(
                        x=event_date,
                        line=dict(color=evt_color, width=0.8, dash='dash'),
                        row='all'
                    )
        
        # Update layout - Institutional quant style
        fig.update_layout(
            height=550,
            paper_bgcolor=QUANT_COLORS['bg'],
            plot_bgcolor=QUANT_COLORS['panel'],
            font=dict(color=QUANT_COLORS['text'], family='IBM Plex Sans'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            hovermode='x unified',
            margin=dict(l=50, r=20, t=40, b=40)
        )
        
        # Grid styling
        fig.update_xaxes(gridcolor=QUANT_COLORS['grid'], gridwidth=0.5, zeroline=False)
        fig.update_yaxes(gridcolor=QUANT_COLORS['grid'], gridwidth=0.5, zeroline=False)
        fig.update_yaxes(title_text="Price (₹)", title_font=dict(size=10), row=1, col=1)
        fig.update_yaxes(title_text="Vol (Ann.)", title_font=dict(size=10), tickformat='.0%', row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        
        # ------------------------------------------------------------------
        # Event Impact Analysis
        # ------------------------------------------------------------------
        st.markdown("### Macro Event Impact Analysis")
        
        if len(events_in_range) == 0:
            st.info("No macro events found in the selected date range.")
        else:
            # Display event summary
            event_summary = get_event_summary(full_data, events_in_range, window=event_window)
            
            if len(event_summary) > 0:
                st.table(event_summary)
                
                # Event type statistics
                st.markdown("#### Event Type Summary")
                
                event_types = events_in_range['Event_Type'].unique()
                stat_cols = st.columns(len(event_types))
                
                for i, evt_type in enumerate(event_types):
                    stats = get_event_type_stats(full_data, events_in_range, evt_type, window=event_window)
                    
                    with stat_cols[i]:
                        st.markdown(f"**{evt_type}**")
                        st.markdown(f"- Events: {stats['count']}")
                        if stats['count'] > 0:
                            st.markdown(f"- Avg Vol Change: {stats['avg_vol_change_pct']:+.1f}%")
                            st.markdown(f"- Expansions: {stats['expansions']} ({stats['expansion_rate']:.0f}%)")
            else:
                st.info("Not enough data points around events for analysis.")
        
        # ------------------------------------------------------------------
        # Volume & Liquidity Analysis
        # ------------------------------------------------------------------
        if 'Volume_SMA' in full_data.columns and 'RVOL' in full_data.columns:
            st.markdown("### 📊 Volume & Liquidity Analysis")
            st.caption("Identifies institutional participation and volume-led volatility breakouts.")

            curr_vol = float(full_data['Volume'].iloc[-1])
            vol_sma = float(full_data['Volume_SMA'].iloc[-1])
            rvol = float(full_data['RVOL'].iloc[-1])

            # Regime calculation
            if rvol >= 1.5:
                vol_regime = "Volume Expansion (High Activity)"
                vol_color = "#059669"  # Green
            elif rvol < 0.5:
                vol_regime = "Volume Contraction (Compression)"
                vol_color = "#D97706"  # Amber
            else:
                vol_regime = "Normal Volume Activity"
                vol_color = "#6B7280"  # Grey

            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.metric("Current Volume", f"{curr_vol:,.0f}")
            with vc2:
                st.metric("Volume SMA (20d)", f"{vol_sma:,.0f}")
            with vc3:
                st.metric("Relative Volume (RVOL)", f"{rvol:.2f}x",
                          help="Current Volume divided by 20-day Volume SMA. > 1.5x indicates strong volume breakout.")

            st.markdown(
                f"""
                <div style="border-left: 4px solid {vol_color}; padding: 10px 16px; background: #F9FAFB; border-radius: 0 8px 8px 0; margin-bottom: 12px;">
                    <span style="font-size:13px; font-weight:700; color:{vol_color};">LIQUIDITY REGIME — {vol_regime.upper()}</span>
                    <br>
                    <span style="font-size:12px; color:#6B7280;">
                        RVOL measures buying/selling pressure intensity. Expansion suggests high stress or trend validation, while contraction suggests compression preceding a potential breakout.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.divider()

        # ------------------------------------------------------------------
        # Benchmark Correlation Analysis
        # ------------------------------------------------------------------
        tk_ret = metrics['Log_Ret'].copy()
        tk_ret.index = tk_ret.index.tz_localize(None)
        tk_ret.name = ticker

        with st.spinner("Fetching benchmark indices for correlation analysis..."):
            bench_rets = fetch_benchmark_returns(str(start_date), str(end_date))
        
        if bench_rets and 'Nifty 50' in bench_rets and 'Sensex' in bench_rets:
            nifty_ret = bench_rets['Nifty 50']
            sensex_ret = bench_rets['Sensex']
            
            # Align the series
            corr_df = pd.concat([tk_ret, nifty_ret, sensex_ret], axis=1).dropna()
            corr_df.columns = [ticker, 'Nifty 50', 'Sensex']
            
            if len(corr_df) >= 10:
                corr_matrix = corr_df.corr()
                
                st.markdown("### 🔗 Benchmark Correlation Analysis")
                st.caption("Pearson correlation matrix calculated on daily log returns over the selected period.")
                
                # Metric cards
                c_nifty = corr_matrix.loc[ticker, 'Nifty 50']
                c_sensex = corr_matrix.loc[ticker, 'Sensex']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Correlation vs Nifty 50",
                        value=f"{c_nifty:+.3f}",
                        help="Linear correlation of ticker daily returns with Nifty 50 index returns."
                    )
                with col2:
                    st.metric(
                        label="Correlation vs Sensex",
                        value=f"{c_sensex:+.3f}",
                        help="Linear correlation of ticker daily returns with Sensex index returns."
                    )
                with col3:
                    # Provide text summary of systematic risk
                    beta_desc = "Neutral/Moderate"
                    if abs(c_nifty) >= 0.7:
                        beta_desc = "High Systematic Co-movement"
                    elif abs(c_nifty) < 0.3:
                        beta_desc = "High Idiosyncratic (Diversification)"
                    st.metric(
                        label="Systematic Risk Profile",
                        value=beta_desc,
                        help="Indicates whether the asset behaves systematically with the market or moves independently (idiosyncratically)."
                    )
                
                # Render heatmap using Plotly
                z = corr_matrix.values
                x_labels = list(corr_matrix.columns)
                y_labels = list(corr_matrix.index)
                
                # Custom text annotations
                z_text = [[f"{val:+.3f}" for val in row] for row in z]
                
                custom_colorscale = [
                    [0.0, '#EF4444'],  # Bright Red
                    [0.5, '#F8F9FA'],  # Light grey / White background matching app bg
                    [1.0, '#2563EB']   # Royal Blue
                ]
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=z,
                    x=x_labels,
                    y=y_labels,
                    text=z_text,
                    texttemplate="%{text}",
                    textfont=dict(size=12, family='IBM Plex Sans', color='#111827'),
                    colorscale=custom_colorscale,
                    zmin=-1.0,
                    zmax=1.0,
                    colorbar=dict(
                        title="Correlation",
                        titleside="right",
                        tickmode="array",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
                    ),
                    showscale=True
                ))
                
                fig_corr.update_layout(
                    height=320,
                    paper_bgcolor=QUANT_COLORS['bg'],
                    plot_bgcolor=QUANT_COLORS['panel'],
                    font=dict(color=QUANT_COLORS['text'], family='IBM Plex Sans'),
                    margin=dict(l=80, r=40, t=20, b=40),
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Explanation/Alert Box
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid #1E40AF; padding: 10px 16px; background: #F9FAFB; border-radius: 0 8px 8px 0; margin-bottom: 12px;">
                        <span style="font-size:13px; font-weight:700; color:#1E40AF;">RISK INTERPRETATION</span>
                        <br>
                        <span style="font-size:12px; color:#6B7280;">
                            A correlation closer to <strong>+1.0</strong> implies the stock is heavily driven by systematic market factors (Nifty/Sensex direction). 
                            A correlation near <strong>0.0</strong> suggests the stock's volatility and movements are idiosyncratic, making it a strong candidate for portfolio diversification. 
                            Negative correlation (rare) offers potential hedging properties.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.divider()
            else:
                st.warning("⚠️ Insufficient overlapping data points between the selected ticker and benchmarks to compute correlation.")
                st.divider()
        else:
            st.warning("⚠️ Benchmark index data could not be fetched. Correlation analysis is unavailable.")
            st.divider()

        # ------------------------------------------------------------------
        # Advanced Time Series & Simulation Analysis
        # ------------------------------------------------------------------
        if show_advanced:
            st.markdown("### 📊 Advanced Time Series & Simulation Analysis")
            st.caption("Deep risk metrics, path projections, memory coefficients, and barrier hit times.")
            st.divider()

            # 1. Monte Carlo Simulation Section
            st.markdown("#### 🎲 Monte Carlo Simulation")
            with st.spinner("Running Monte Carlo simulations..."):
                try:
                    _mc_lr  = metrics['Log_Ret'].dropna()
                    _mc_mu  = float(_mc_lr.mean())
                    _mc_sig = float(_mc_lr.std())
                    _mc_nu  = _mc_mu - 0.5 * _mc_sig**2
                    _mc_s0  = float(data['Close'].iloc[-1])
                    _mc_T   = 20
                    np.random.seed(42)
                    _mc_r   = np.random.randn(1000, _mc_T)
                    _mc_p   = _mc_s0 * np.exp(np.cumsum(_mc_nu + _mc_sig * _mc_r, axis=1))
                    _mc_ter = _mc_p[:, -1]
                    _var5   = float(np.percentile(_mc_ter, 5))
                    _var1   = float(np.percentile(_mc_ter, 1))
                    _cvar5  = float(_mc_ter[_mc_ter <= _var5].mean()) if (_mc_ter <= _var5).any() else _var5
                    _p_up   = float((_mc_ter > _mc_s0).mean())
                    _pct5   = float(np.percentile(_mc_ter, 5))
                    _pct95  = float(np.percentile(_mc_ter, 95))
                    _mean20 = float(_mc_ter.mean())

                    # Display Metric Grid
                    mcc1, mcc2, mcc3, mcc4, mcc5 = st.columns(5)
                    with mcc1:
                        st.metric("P(Up 20d)", f"{_p_up:.1%}")
                    with mcc2:
                        st.metric("VaR 5% (20d)", f"₹{_var5:,.2f}", f"{(_var5/_mc_s0-1)*100:+.1f}%")
                    with mcc3:
                        st.metric("CVaR 5% (Tail)", f"₹{_cvar5:,.2f}", f"{(_cvar5/_mc_s0-1)*100:+.1f}%")
                    with mcc4:
                        st.metric("5th Percentile", f"₹{_pct5:,.2f}")
                    with mcc5:
                        st.metric("95th Percentile", f"₹{_pct95:,.2f}")

                    # Render simulation chart (50 paths sample)
                    fig_mc = go.Figure()
                    x_days = list(range(21))
                    for i in range(50):
                        y_path = [_mc_s0] + list(_mc_p[i])
                        fig_mc.add_trace(go.Scatter(
                            x=x_days,
                            y=y_path,
                            mode='lines',
                            line=dict(width=0.6, color='rgba(99, 102, 241, 0.15)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    # Highlight mean pathway in bright blue
                    y_mean = [_mc_s0] + list(np.mean(_mc_p, axis=0))
                    fig_mc.add_trace(go.Scatter(
                        x=x_days,
                        y=y_mean,
                        mode='lines',
                        name="Mean Pathway",
                        line=dict(color='#2563EB', width=1.5),
                    ))
                    # Draw horizontal lines for 5th/95th boundaries
                    fig_mc.add_hline(y=_pct95, line=dict(color='#059669', width=1.2, dash='dash'), annotation_text=f"95th Pct (₹{_pct95:,.2f})", annotation_position="top right")
                    fig_mc.add_hline(y=_pct5, line=dict(color='#DC2626', width=1.2, dash='dash'), annotation_text=f"5th Pct (₹{_pct5:,.2f})", annotation_position="bottom right")

                    fig_mc.update_layout(
                        height=360,
                        paper_bgcolor=QUANT_COLORS['bg'],
                        plot_bgcolor=QUANT_COLORS['panel'],
                        font=dict(color=QUANT_COLORS['text'], family='IBM Plex Sans'),
                        xaxis=dict(title="Projection Days", gridcolor=QUANT_COLORS['grid']),
                        yaxis=dict(title="Simulated Price (₹)", gridcolor=QUANT_COLORS['grid']),
                        margin=dict(l=50, r=100, t=30, b=40),
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)

                except Exception as _mce:
                    st.error(f"Monte Carlo simulation failed: {_mce}")

            st.divider()

            # 2. Time Series Analysis Section
            st.markdown("#### 📈 Time Series Analysis (ARIMA & Hurst)")
            try:
                # Hurst Exponent R/S calculation
                def _hurst_rs_b(ts, min_lag=8, max_lag=200):
                    ts = np.asarray(ts, dtype=float)
                    lags = np.unique(np.logspace(
                        np.log10(min_lag),
                        np.log10(min(max_lag, len(ts) // 2)),
                        num=20, dtype=int))
                    rs_vals = []
                    for lag in lags:
                        sub = ts[:lag]
                        dev = np.cumsum(sub - np.mean(sub))
                        R, S = dev.max() - dev.min(), np.std(sub, ddof=1)
                        if S > 0: rs_vals.append(R / S)
                    if len(rs_vals) < 4: return 0.5
                    H, _ = np.polyfit(np.log(lags[:len(rs_vals)]), np.log(rs_vals), 1)
                    return float(np.clip(H, 0.01, 0.99))
                
                _H = _hurst_rs_b(metrics['Log_Ret'].dropna().values)
                h_regime = ('Trending (Persistent)' if _H > 0.6 else
                            'Mean-Reverting' if _H < 0.45 else 'Random Walk (Geometric)')
                h_color = '#059669' if _H > 0.6 else ('#D97706' if _H < 0.45 else '#475569')

                # ARIMA model
                from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                _ar = _ARIMA(metrics['Log_Ret'].dropna().values, order=(1, 0, 1)).fit()
                ar_c  = float(_ar.params[0])
                ar_phi = float(_ar.params[1])
                ar_theta = float(_ar.params[2])
                ar_fc = float(_ar.forecast(steps=1)[0])

                # UI metrics display
                tsc1, tsc2, tsc3 = st.columns(3)
                with tsc1:
                    st.metric("Hurst Exponent (H)", f"{_H:.4f}", help="H > 0.6 trending, H < 0.45 mean-reverting, ~0.5 random walk")
                with tsc2:
                    st.metric("Fractal Dimension (D)", f"{2-_H:.4f}")
                with tsc3:
                    st.metric("Memory Regime", h_regime)

                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {h_color}; padding: 10px 16px; background: #F9FAFB; border-radius: 0 8px 8px 0; margin-bottom: 16px;">
                        <span style="font-size:12px; color:#6B7280;">Hurst coefficient measures market memory. Fractal dimension (2-H) defines geometric complexity.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("**ARIMA(1,0,1) Mean Equation Parameters**")
                ar1, ar2, ar3, ar4 = st.columns(4)
                with ar1:
                    st.metric("Constant (c)", f"{ar_c*100:+.4f}%")
                with ar2:
                    st.metric("AR Coefficient (φ₁)", f"{ar_phi:+.4f}")
                with ar3:
                    st.metric("MA Coefficient (θ₁)", f"{ar_theta:+.4f}")
                with ar4:
                    st.metric("1-Day Return Forecast", f"{ar_fc*100:+.3f}%")

                # Visualise Log Returns vs ARIMA Fitted Values
                st.markdown("**ARIMA(1,0,1) In-Sample Fitted Returns vs Actual Returns**")
                fig_arima = go.Figure()
                
                # Log returns index and values
                lr_series = metrics['Log_Ret'].dropna()
                
                fig_arima.add_trace(go.Scatter(
                    x=lr_series.index,
                    y=lr_series * 100,
                    name="Actual Log Returns",
                    line=dict(color='rgba(156, 163, 175, 0.5)', width=1.0)
                ))
                
                # Fitted values
                fitted_y = _ar.fittedvalues
                fig_arima.add_trace(go.Scatter(
                    x=lr_series.index,
                    y=fitted_y * 100,
                    name="ARIMA Fitted",
                    line=dict(color='#1E40AF', width=1.2, dash='dash')
                ))
                
                fig_arima.update_layout(
                    height=260,
                    paper_bgcolor=QUANT_COLORS['bg'],
                    plot_bgcolor=QUANT_COLORS['panel'],
                    font=dict(color=QUANT_COLORS['text'], family='IBM Plex Sans'),
                    margin=dict(l=50, r=20, t=30, b=40),
                    legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
                    xaxis=dict(gridcolor=QUANT_COLORS['grid']),
                    yaxis=dict(title="Daily Return (%)", gridcolor=QUANT_COLORS['grid'])
                )
                st.plotly_chart(fig_arima, use_container_width=True)

            except Exception as _tse:
                st.error(f"Time Series analysis failed: {_tse}")

            st.divider()

            # 3. Barrier & Thresholds Section
            st.markdown("#### ⏱ Barrier & Threshold Analysis")
            try:
                # First Passage Time (FPT) to ±1σ / ±2σ
                _lr_s  = metrics['Log_Ret'].dropna()
                _mu_d  = float(_lr_s.mean())
                _sig_d = float(_lr_s.std())
                _nu    = _mu_d - 0.5 * _sig_d**2
                
                def _fpt_label(b):
                    if abs(_nu) < 1e-8 or (_nu * b) <= 0: return 'infinite hit window'
                    d = b / _nu
                    if d < 1:  return f'{d*24:.1f} hours'
                    if d < 5:  return f'{d:.1f} days'
                    if d < 22: return f'{d/5:.1f} weeks'
                    return     f'{d/21:.1f} months'
                
                fpt_up1 = _fpt_label(+_sig_d)
                fpt_up2 = _fpt_label(+2*_sig_d)
                fpt_dn1 = _fpt_label(-_sig_d)
                fpt_dn2 = _fpt_label(-2*_sig_d)

                # Auto Support & Resistance
                _cl_s  = data['Close'].dropna()
                _lc_px = float(_cl_s.iloc[-1])
                _w     = 10
                _px    = _cl_s.values
                _highs = sorted(set(
                    round(float(_px[k]), 2)
                    for k in range(_w, len(_px) - _w)
                    if _px[k] == max(_px[k-_w:k+_w+1]) and _px[k] > _lc_px
                ))
                _lows  = sorted(set(
                    round(float(_px[k]), 2)
                    for k in range(_w, len(_px) - _w)
                    if _px[k] == min(_px[k-_w:k+_w+1]) and _px[k] < _lc_px
                ), reverse=True)
                _resist  = _highs[0] if _highs else round(_lc_px * 1.03, 2)
                _support = _lows[0]  if _lows  else round(_lc_px * 0.97, 2)

                # Logistic breach probability model
                _thr_w = 10
                _dist  = np.log(_resist / _cl_s)
                _tgt_s, _feats = [], []
                for k in range(len(_cl_s)):
                    fut = _cl_s.values[k+1:k+1+_thr_w]
                    if len(fut) < _thr_w: continue
                    _tgt_s.append(1 if fut.max() >= _resist else 0)
                    _feats.append([float(_dist.iloc[k])])
                    
                _breach_prob_str = "insufficient historical samples"
                _breach_ok = False
                if len(_feats) >= 30 and len(set(_tgt_s)) == 2:
                    from sklearn.linear_model import LogisticRegression as _LRb
                    from sklearn.preprocessing import StandardScaler as _SSb
                    _sc = _SSb(); _Xb = _sc.fit_transform(np.array(_feats))
                    _lrb = _LRb(max_iter=300, class_weight='balanced', solver='lbfgs')
                    _lrb.fit(_Xb, np.array(_tgt_s))
                    _curr_Xb = _sc.transform([[float(np.log(_resist / _lc_px))]])
                    _pb = float(_lrb.predict_proba(_curr_Xb)[0][1])
                    _breach_prob_str = f"{_pb:.1%}"
                    _breach_ok = True
                
                st.markdown("**Barrier Hit Times (First Passage Time)**")
                fptc1, fptc2 = st.columns(2)
                with fptc1:
                    st.markdown(f"- **Upper +1σ Vol Barrier:** hit in ~`{fpt_up1}`")
                    st.markdown(f"- **Upper +2σ Vol Barrier:** hit in ~`{fpt_up2}`")
                with fptc2:
                    st.markdown(f"- **Lower -1σ Vol Barrier:** hit in ~`{fpt_dn1}`")
                    st.markdown(f"- **Lower -2σ Vol Barrier:** hit in ~`{fpt_dn2}`")

                st.divider()
                st.markdown("**Support, Resistance & Breach Probabilities**")
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    st.metric("Auto Support Level", f"₹{_support:,.2f}", f"{(_support/_lc_px-1)*100:+.2f}%", delta_color="normal")
                with bc2:
                    st.metric("Auto Resistance Level", f"₹{_resist:,.2f}", f"{(_resist/_lc_px-1)*100:+.2f}%", delta_color="inverse")
                with bc3:
                    st.metric("P(Breach 10d)", _breach_prob_str, help="Predicted probability of price breaching nearest resistance level within 10 trading days")

                # Display visual warning alert box for high probability
                if _breach_ok and _pb >= 0.60:
                    st.warning(f"⚠️ High Breach Alert: Logistic model predicts a {_breach_prob_str} probability of price breaking through ₹{_resist:,.2f} in next 10 days.")
                elif _breach_ok and _pb <= 0.30:
                    st.success(f"✓ Compression Alert: Logistic model predicts low breach probability ({_breach_prob_str}) of support/resistance.")

            except Exception as _bare:
                st.error(f"Barrier & Threshold analysis failed: {_bare}")

            st.divider()

        # ------------------------------------------------------------------
        # Data Export
        # ------------------------------------------------------------------
        st.markdown("### Export Data")
        
        export_df = full_data[['Open', 'High', 'Low', 'Close', 'Volume'] + list(metrics.columns)].copy()
        
        csv = export_df.to_csv()
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{ticker.replace('.', '_')}_volatility_analysis.csv",
            mime="text/csv"
        )

# ==============================================================================
# Footer
# ==============================================================================
st.divider()
st.caption("Dynamic Volatility Estimator v1.0 | Data: Yahoo Finance | Not investment advice")

