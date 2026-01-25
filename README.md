# Dynamic Volatility Estimator 📊

A **quant-grade realized volatility dashboard** for Indian equity markets (NSE/BSE). Built for risk analysts, quant researchers, and traders who need institutional-quality volatility analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### 📈 Volatility Analysis
- **Rolling Volatility**: Configurable windows (10, 20, 30, 60, 90, 120 days)
- **EWMA Volatility**: Exponentially weighted with RiskMetrics decay (λ ≈ 0.94)
- **GARCH(1,1)**: Advanced volatility forecasting with persistence metrics
- **Log Returns**: Automatically calculated from close prices

### 🎯 Regime Classification
- **Z-Score Based**: Statistical thresholds for regime detection
  - `EXPANSION`: z ≥ 1.5 (high stress)
  - `Elevated`: z ≥ 0.5
  - `Neutral`: -0.5 ≤ z < 0.5
  - `Low`: z < -0.5
  - `Compression`: z < -1.0

### 📅 Macro Event Impact
- Pre-loaded calendar: RBI MPC, India CPI, Union Budget
- Pre vs Post-event volatility comparison
- Automatic expansion/contraction classification

### 🏦 Exchange Support
- **NSE**: RELIANCE.NS, TCS.NS, HDFCBANK.NS, etc.
- **BSE**: RELIANCE.BO, TCS.BO, HDFCBANK.BO, etc.
- **Indices**: ^NSEI (Nifty 50), ^BSESN (Sensex)

## Installation

```bash
# Clone the repository
git clone https://github.com/007rudra007/volatility-estimator.git
cd volatility-estimator

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.36
plotly>=5.18.0
arch>=6.2.0
```

## Project Structure

```
volatility_estimator/
├── app.py                 # Streamlit dashboard (main entry)
├── vol_engine.py          # Volatility calculation engine
├── event_analyzer.py      # Macro event impact analysis
├── verify_calculations.py # Mathematical verification script
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
└── data/
    └── macro_events_india.csv  # Indian macro event calendar
```

## Usage

### Quick Start
1. Run `streamlit run app.py`
2. Select exchange (NSE/BSE) and stock
3. Set date range
4. Click "Run Analysis"

### Volatility Formulas

**Log Returns:**
```
r_t = ln(P_t / P_{t-1})
```

**Rolling Volatility (Annualized):**
```
σ = StdDev(r_{t-n+1}, ..., r_t) × √252
```

**EWMA Volatility:**
```
σ²_t = λ × σ²_{t-1} + (1-λ) × r²_t
```
Where λ = 0.94 (RiskMetrics standard)

**GARCH(1,1):**
```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

### Verification
Run the verification script to audit calculations:
```bash
python verify_calculations.py
```

## Design Philosophy

This dashboard follows **institutional quant design principles**:

1. **Color is information, not aesthetics** - Blue-grey spectrum, no decorative colors
2. **Grayscale-safe** - All visualizations survive black-and-white printing
3. **Semantic color logic** - Red only appears for statistically justified expansion
4. **Thin lines** (1.2px) - Professional, clean appearance
5. **IBM Plex fonts** - Clear, technical typography

## Color Palette

| Element | Hex | Purpose |
|---------|-----|---------|
| Background | `#F8F9FA` | Research-paper neutral |
| Text | `#111827` | Near-black (readable) |
| Vol 20d | `#1E40AF` | Deep blue - short-term |
| Vol 60d | `#475569` | Slate - regime baseline |
| EWMA | `#92400E` | Brown - shock sensitivity |
| RBI Events | `#4F46E5` | Indigo |
| CPI Events | `#0891B2` | Cyan |

## Data Sources

- **Price Data**: Yahoo Finance (via `yfinance`)
- **Event Calendar**: Manually curated Indian macro events

## Disclaimer

⚠️ **Not Investment Advice**

This tool is for educational and research purposes only. Past volatility does not predict future volatility. Always consult a qualified financial advisor before making investment decisions.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Author

Built by Rudra Trivedi

---

*Dynamic Volatility Estimator v1.0*
