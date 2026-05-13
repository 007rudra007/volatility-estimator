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
    'price': '#111827',         # Near black for price
    'vol_20': '#1E40AF',        # Blue - 20d vol
    'vol_60': '#475569',        # Slate - 60d vol
    'vol_120': '#6B7280',       # Grey - 120d vol
    'ewma': '#92400E',          # Brown - EWMA
    
    # Regime Classification
    'compress': '#059669',      # Green - compression
    'neutral': '#6B7280',       # Grey - neutral
    'expand': '#DC2626',        # Red - expansion
    
    # Macro Event Overlays
    'rbi': '#4F46E5',           # Indigo - RBI
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

def calculate_price_targets(data: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    """
    Derive statistical price targets from realized vol, ATR, and Bollinger Bands.
    All levels are based purely on price/vol data — no fundamental inputs.
    """
    close   = data['Close'].dropna()
    high    = data['High'].dropna()
    low     = data['Low'].dropna()
    last    = float(close.iloc[-1])

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

# Simple header using Streamlit native
st.title("Volatility Risk Dashboard")
st.caption("Dynamic Realized Volatility Estimator • NSE/BSE • India")

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
        
        # Calculate volatility metrics
        metrics = calculate_metrics(data['Close'])
        
        # Recalculate with custom windows if different from default
        if vol_windows and vol_windows != [20, 60]:
            from vol_engine import calculate_log_returns, calculate_rolling_vol, calculate_ewma_vol
            log_ret = calculate_log_returns(data['Close'])
            custom_vol = calculate_rolling_vol(log_ret, vol_windows)
            for col in custom_vol.columns:
                if col not in metrics.columns:
                    metrics[col] = custom_vol[col]
        
        # Merge metrics with original data
        full_data = data.copy()
        for col in metrics.columns:
            full_data[col] = metrics[col].values
        
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
        # Statistical Price Targets
        # ------------------------------------------------------------------
        st.markdown("### 🎯 Statistical Price Targets")
        st.caption("Derived from realized volatility, ATR-14, and Bollinger Bands — no fundamental inputs.")

        pt = calculate_price_targets(full_data, metrics)
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
        
        # Volatility lines - Quant palette (blue-grey spectrum + brown for EWMA)
        quant_vol_colors = {
            'Vol_20d': QUANT_COLORS['vol_20'],   # Deep blue
            'Vol_60d': QUANT_COLORS['vol_60'],   # Slate
            'Vol_120d': QUANT_COLORS['vol_120'], # Lighter slate
            'EWMA': QUANT_COLORS['ewma'],         # Muted brown
        }
        vol_columns = [c for c in metrics.columns if 'Vol_' in c or c == 'EWMA']
        
        for col in vol_columns:
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

