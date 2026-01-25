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

# Exchange selection
exchange = st.sidebar.radio(
    "📈 Select Exchange",
    ["NSE", "BSE"],
    horizontal=True
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

POPULAR_TICKERS = POPULAR_NSE if exchange == "NSE" else POPULAR_BSE
suffix = ".NS" if exchange == "NSE" else ".BO"

# Ticker selection
ticker_option = st.sidebar.selectbox(
    "📌 Select a Stock",
    ["Custom"] + POPULAR_TICKERS,
    index=4  # Default to HDFCBANK
)

if ticker_option == "Custom":
    ticker = st.sidebar.text_input(
        f"Enter {exchange} Symbol",
        f"RELIANCE{suffix}",
        help=f"Use {suffix} suffix for {exchange} stocks (e.g., RELIANCE{suffix})"
    )
else:
    ticker = ticker_option

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
        # GARCH Analysis (Optional)
        # ------------------------------------------------------------------
        if show_garch:
            st.markdown("### GARCH(1,1) Volatility Forecast")
            
            with st.spinner("Fitting GARCH model..."):
                garch_result = fit_garch(metrics['Log_Ret'])
            
            if garch_result:
                gcol1, gcol2, gcol3, gcol4 = st.columns(4)
                
                with gcol1:
                    st.metric("α (Shock Reaction)", f"{garch_result['alpha']:.4f}" if garch_result['alpha'] else "N/A")
                with gcol2:
                    st.metric("β (Persistence)", f"{garch_result['beta']:.4f}" if garch_result['beta'] else "N/A")
                with gcol3:
                    st.metric("Persistence (α+β)", f"{garch_result['persistence']:.4f}" if garch_result['persistence'] else "N/A")
                with gcol4:
                    st.metric("1-Day Forecast", f"{garch_result['forecast_1d']:.1%}" if garch_result['forecast_1d'] else "N/A")
                
                if garch_result['persistence'] and garch_result['persistence'] > 0.95:
                    st.warning("⚠️ High persistence indicates volatility shocks last a long time!")
            else:
                st.warning("GARCH model could not be fitted. Try a longer data period.")
        
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

