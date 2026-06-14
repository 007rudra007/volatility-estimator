"""
cli_run.py - Headless Quantitative Volatility and Time Series Analysis Pipeline
=============================================================================
Runs GARCH, EGARCH, LSTM, ARIMA, Hurst, FPT, Thresholds, and Monte Carlo
for the watchlist and saves the results to a CSV.
"""

import sys
import io
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Force UTF-8 encoding for standard output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ignore warnings
warnings.filterwarnings("ignore")

# Setup import path
_VOL_DIR = os.path.dirname(os.path.abspath(__file__))
if _VOL_DIR not in sys.path:
    sys.path.insert(0, _VOL_DIR)

from vol_engine import (
    calculate_log_returns,
    calculate_rolling_vol,
    calculate_ewma_vol,
    calculate_metrics,
    fit_garch,
    fit_egarch,
    get_regime_comparison,
    classify_regime,
    get_zscore,
    calculate_donchian_channels,
    calculate_volume_metrics,
)

try:
    from lstm_vol_engine import GARCHLSTMForecaster, is_available as lstm_available
    HAS_LSTM = lstm_available()
except ImportError:
    HAS_LSTM = False

# Watchlist tickers
TICKERS = [
    "HFCL.NS",
    "IDFCFIRSTB.NS",
    "SUZLON.NS",
    "SJVN.NS",
    "IRFC.NS",
    "NMDC.NS",
    "INOXWIND.NS",
    "PNB.NS",
    "IREDA.NS"
]

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_price_targets(data: pd.DataFrame, metrics: pd.DataFrame, current_price: float = None) -> dict:
    """Derive statistical price targets from realized vol, ATR, and Bollinger Bands."""
    close = data['Close'].dropna()
    high = data['High'].dropna()
    low = data['Low'].dropna()
    last = current_price if current_price is not None else float(close.iloc[-1])

    # 1. Vol-based bands (±1σ, ±2σ daily move)
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

    # 2. ATR-based support / resistance (14-day)
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

    # 3. Bollinger Bands (20-day, ±2σ price)
    sma20   = float(close.rolling(20).mean().iloc[-1])
    std20   = float(close.rolling(20).std().iloc[-1])
    bb_bands = {
        'bb_upper': sma20 + 2 * std20,
        'bb_mid':   sma20,
        'bb_lower': sma20 - 2 * std20,
    }

    # 4. 52-week high / low
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

def _fetch_ticker_history(sym: str, start: str, end: str) -> pd.DataFrame | None:
    """Robust single-ticker fetcher: tries yf.download."""
    try:
        raw = yf.download(sym, start=start, end=end, progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            # Normalise columns
            raw.columns = raw.columns.get_level_values(0)
        # Standardise capitalisation
        raw.columns = [str(c).strip().title() for c in raw.columns]
        return raw
    except Exception as e:
        print(f"Error downloading {sym}: {e}")
        return None

def run_pipeline(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Runs full batch analysis for a list of tickers."""
    rows = []
    total = len(tickers)
    
    print("=" * 80)
    print(f"RUNNING QUANT VOLATILITY PIPELINE ON {total} TICKERS")
    print(f"Period: {start} -> {end}")
    print("=" * 80)

    # 1. Fetch Nifty & Sensex for correlation
    print("Fetching indices (^NSEI, ^BSESN) for correlation calculations...")
    _nifty_ret = None
    _sensex_ret = None
    try:
        nifty = yf.download("^NSEI", start=start, end=end, progress=False)
        nifty = flatten_columns(nifty)
        closes = nifty['Close'].dropna()
        _nifty_ret = np.log(closes / closes.shift(1)).dropna()
        _nifty_ret.name = "^NSEI"
        _nifty_ret.index = _nifty_ret.index.tz_localize(None)
    except Exception as e:
        print(f"Index fetch failed (^NSEI): {e}")

    try:
        sensex = yf.download("^BSESN", start=start, end=end, progress=False)
        sensex = flatten_columns(sensex)
        closes = sensex['Close'].dropna()
        _sensex_ret = np.log(closes / closes.shift(1)).dropna()
        _sensex_ret.name = "^BSESN"
        _sensex_ret.index = _sensex_ret.index.tz_localize(None)
    except Exception as e:
        print(f"Index fetch failed (^BSESN): {e}")

    # 2. Iterate through tickers
    for idx, sym in enumerate(tickers):
        print(f"\n[{idx+1}/{total}] Processing ticker: {sym}")
        time.sleep(0.2)  # Avoid rate limit

        try:
            raw = _fetch_ticker_history(sym, start, end)
            if raw is None or len(raw) < 30:
                print(f"   x No historical data for {sym}!")
                rows.append({
                    'Ticker': sym,
                    'Status': 'No data'
                })
                continue

            close_col = 'Close'
            if close_col not in raw.columns:
                print(f"   x Close column missing in data for {sym}")
                rows.append({'Ticker': sym, 'Status': 'Close missing'})
                continue

            met = calculate_metrics(raw[close_col], sma_window=20)
            rs = get_regime_comparison(met['Vol_20d'])
            reg = classify_regime(rs['current'], rs)
            
            vol_pr_str = f"{rs['current']:.1%}" if rs['current'] is not None else 'N/A'
            print(f"   Price: INR {float(raw[close_col].iloc[-1]):.2f} | 20d Realized Vol: {vol_pr_str} | Regime: {reg}")

            # Donchian & Volume calculations
            high_s = raw['High'].iloc[:, 0] if isinstance(raw['High'], pd.DataFrame) else raw['High']
            low_s = raw['Low'].iloc[:, 0] if isinstance(raw['Low'], pd.DataFrame) else raw['Low']
            vol_s = raw['Volume'].iloc[:, 0] if isinstance(raw['Volume'], pd.DataFrame) else raw['Volume']
            
            donchian = calculate_donchian_channels(high_s, low_s, window=20)
            volume_met = calculate_volume_metrics(vol_s, window=20)

            row = {
                'Ticker':         sym,
                'Last Close':     round(float(raw[close_col].iloc[-1]), 2),
                'Data Points':    len(raw),
                'Vol 20d (Ann.)': f"{rs['current']:.1%}" if rs['current'] else 'N/A',
                'Vol 52W Avg':    f"{rs['avg_52w']:.1%}"  if rs['avg_52w'] else 'N/A',
                'Percentile':     f"{rs['percentile']:.0f}th" if rs['percentile'] else 'N/A',
                'Regime':         reg,
                'EWMA Vol':       f"{float(met['EWMA'].dropna().iloc[-1]):.1%}",
                'SMA Vol':        f"{float(met['SMA_20d'].dropna().iloc[-1]):.1%}",
                'Donchian Upper': round(float(donchian['Donchian_Upper'].iloc[-1]), 2) if not np.isnan(donchian['Donchian_Upper'].iloc[-1]) else 'N/A',
                'Donchian Lower': round(float(donchian['Donchian_Lower'].iloc[-1]), 2) if not np.isnan(donchian['Donchian_Lower'].iloc[-1]) else 'N/A',
                'Volume SMA (20d)': round(float(volume_met['Volume_SMA'].iloc[-1]), 0) if not np.isnan(volume_met['Volume_SMA'].iloc[-1]) else 'N/A',
                'RVOL':           round(float(volume_met['RVOL'].iloc[-1]), 2) if not np.isnan(volume_met['RVOL'].iloc[-1]) else 'N/A',
                'Status':         'OK',
            }

            # ---- Price Targets (ATR, Bollinger Bands, Vol-based σ-bands) ----
            high_col = 'High'
            low_col = 'Low'
            if high_col in raw.columns and low_col in raw.columns:
                pt = calculate_price_targets(
                    raw.rename(columns={close_col: 'Close', high_col: 'High', low_col: 'Low'}),
                    met
                )
                last = pt['last']
                row['ATR-14']      = f"INR {pt['atr14']:,.2f}"
                row['ATR Sup']     = f"INR {pt['atr_sup']:,.2f} ({(pt['atr_sup']/last-1)*100:+.1f}%)"
                row['ATR Res']     = f"INR {pt['atr_res']:,.2f} ({(pt['atr_res']/last-1)*100:+.1f}%)"
                row['BB Upper']    = f"INR {pt['bb_upper']:,.2f} ({(pt['bb_upper']/last-1)*100:+.1f}%)"
                row['BB Mid']      = f"INR {pt['bb_mid']:,.2f}"
                row['BB Lower']    = f"INR {pt['bb_lower']:,.2f} ({(pt['bb_lower']/last-1)*100:+.1f}%)"
                row['52W High']    = f"INR {pt['high52']:,.2f} ({(pt['high52']/last-1)*100:+.1f}%)"
                row['52W Low']     = f"INR {pt['low52']:,.2f}  ({(pt['low52']/last-1)*100:+.1f}%)"
                if pt.get('bull_1s'):
                    row['Bull 1σ'] = f"INR {pt['bull_1s']:,.2f} ({(pt['bull_1s']/last-1)*100:+.1f}%)"
                    row['Bear 1σ'] = f"INR {pt['bear_1s']:,.2f} ({(pt['bear_1s']/last-1)*100:+.1f}%)"

            # ---- Correlation vs Nifty & Sensex ----
            try:
                _tk_ret = np.log(raw[close_col] / raw[close_col].shift(1)).dropna()
                _tk_ret.index = _tk_ret.index.tz_localize(None)
                
                # Nifty
                if _nifty_ret is not None:
                    _aligned = pd.concat([_tk_ret, _nifty_ret], axis=1).dropna()
                    if len(_aligned) >= 20:
                        _rho = float(_aligned.iloc[:, 0].corr(_aligned.iloc[:, 1]))
                        row['ρ Nifty'] = f"{_rho:+.3f}"
                    else:
                        row['ρ Nifty'] = 'N/A'
                else:
                    row['ρ Nifty'] = 'N/A'

                # Sensex
                if _sensex_ret is not None:
                    _aligned = pd.concat([_tk_ret, _sensex_ret], axis=1).dropna()
                    if len(_aligned) >= 20:
                        _rho = float(_aligned.iloc[:, 0].corr(_aligned.iloc[:, 1]))
                        row['ρ Sensex'] = f"{_rho:+.3f}"
                    else:
                        row['ρ Sensex'] = 'N/A'
                else:
                    row['ρ Sensex'] = 'N/A'
            except Exception as e:
                print(f"      ! Correlation failed: {e}")
                row['ρ Nifty'] = row['ρ Sensex'] = 'err'

            # ---- GARCH(1,1) ----
            print("   Fitting GARCH(1,1)...")
            gr = fit_garch(met['Log_Ret'])
            if gr:
                row['GARCH α']       = f"{gr['alpha']:.4f}" if gr['alpha'] is not None else 'N/A'
                row['GARCH β']       = f"{gr['beta']:.4f}"  if gr['beta'] is not None else 'N/A'
                row['GARCH 1D Fcst'] = f"{gr['forecast_1d']:.2%}" if gr['forecast_1d'] is not None else 'N/A'
                
                ga_pr = f"{gr['alpha']:.4f}" if gr['alpha'] is not None else 'N/A'
                gb_pr = f"{gr['beta']:.4f}" if gr['beta'] is not None else 'N/A'
                gf_pr = f"{gr['forecast_1d']:.2%}" if gr['forecast_1d'] is not None else 'N/A'
                print(f"      GARCH Alpha (shock): {ga_pr} | Beta (persist): {gb_pr} | 1D Forecast: {gf_pr}")
            else:
                row['GARCH α'] = row['GARCH β'] = row['GARCH 1D Fcst'] = 'failed'

            # ---- EGARCH(1,1) ----
            print("   Fitting EGARCH(1,1)...")
            er = fit_egarch(met['Log_Ret'])
            if er:
                row['EGARCH γ (Lev)'] = f"{er['gamma']:.4f}" if er['gamma'] is not None else 'N/A'
                row['EGARCH Lev Dir'] = er.get('leverage_dir') or 'N/A'
                lp = er.get('leverage_pct')
                row['EGARCH Lev%']    = f"{lp:+.1f}%" if lp is not None else 'N/A'
                row['EGARCH 1D Fcst'] = f"{er['forecast_1d']:.2%}" if er.get('forecast_1d') is not None else 'N/A'
                
                eg_pr = f"{er['gamma']:.4f}" if er['gamma'] is not None else 'N/A'
                ed_pr = er.get('leverage_dir') or 'N/A'
                ef_pr = f"{er['forecast_1d']:.2%}" if er.get('forecast_1d') is not None else 'N/A'
                print(f"      EGARCH Gamma (leverage): {eg_pr} | Direction: {ed_pr} | 1D Forecast: {ef_pr}")
            else:
                row['EGARCH γ (Lev)'] = row['EGARCH Lev Dir'] = row['EGARCH Lev%'] = row['EGARCH 1D Fcst'] = 'failed'

            # ---- LSTM neural network forecast (PyTorch) ----
            if HAS_LSTM and gr:
                print("   Training GARCH->LSTM neural network forecaster...")
                try:
                    # GARCHLSTMForecaster fits sequences and runs multi-horizon
                    fc_obj = GARCHLSTMForecaster(returns=met['Log_Ret'], garch_params=gr, epochs=50, seq_len=20)
                    lr = fc_obj.run()
                    fc = lr['forecasts']
                    row['LSTM 1D Fcst'] = f"{fc[1]:.2%}"
                    row['LSTM 1W Fcst'] = f"{fc[5]:.2%}"
                    row['LSTM 1M Fcst'] = f"{fc[21]:.2%}"
                    row['LSTM Regime']  = lr['regime']
                    print(f"      LSTM 1D Forecast: {fc[1]:.2%} | LSTM Regime: {lr['regime']}")
                except Exception as _le:
                    print(f"      ! LSTM training failed: {_le}")
                    row['LSTM 1D Fcst'] = row['LSTM 1W Fcst'] = row['LSTM 1M Fcst'] = row['LSTM Regime'] = 'failed'
            else:
                row['LSTM 1D Fcst'] = row['LSTM 1W Fcst'] = row['LSTM 1M Fcst'] = row['LSTM Regime'] = 'N/A'

            # ---- Hurst Exponent ----
            try:
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
                _H = _hurst_rs_b(met['Log_Ret'].dropna().values)
                row['Hurst H']   = f"{_H:.4f}"
                row['Fractal D'] = f"{2-_H:.4f}"
                row['H Regime']  = ('Trending' if _H > 0.6 else
                                    'RandWalk' if _H > 0.45 else 'MeanRev')
            except Exception as e:
                print(f"      ! Hurst failed: {e}")
                row['Hurst H'] = 'err'

            # ---- ARIMA(1,0,1) model ----
            try:
                from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                _ar = _ARIMA(met['Log_Ret'].dropna().values, order=(1, 0, 1)).fit()
                row['ARIMA c']  = f"{float(_ar.params[0])*100:+.5f}%"
                row['ARIMA φ₁'] = f"{float(_ar.params[1]):+.4f}"
                row['ARIMA θ₁'] = f"{float(_ar.params[2]):+.4f}"
                row['ARIMA 1D'] = f"{float(_ar.forecast(steps=1)[0])*100:+.3f}%"
            except Exception as e:
                print(f"      ! ARIMA failed: {e}")
                row['ARIMA c'] = row['ARIMA φ₁'] = row['ARIMA θ₁'] = row['ARIMA 1D'] = 'err'

            # ---- First Passage Time (FPT) to ±1σ / ±2σ ----
            try:
                _lr_s  = met['Log_Ret'].dropna()
                _mu_d  = float(_lr_s.mean())
                _sig_d = float(_lr_s.std())
                _nu    = _mu_d - 0.5 * _sig_d**2
                def _fpt_label(b):
                    if abs(_nu) < 1e-8 or (_nu * b) <= 0: return 'inf'
                    d = b / _nu
                    if d < 1:  return f'{d*24:.1f}h'
                    if d < 5:  return f'{d:.1f}d'
                    if d < 22: return f'{d/5:.1f}w'
                    return     f'{d/21:.1f}mo'
                row['FPT +1σ'] = _fpt_label(+_sig_d)
                row['FPT +2σ'] = _fpt_label(+2*_sig_d)
                row['FPT -1σ'] = _fpt_label(-_sig_d)
                row['FPT -2σ'] = _fpt_label(-2*_sig_d)
                row['Itô ν/d'] = f"{_nu*100:+.5f}%"
            except Exception as e:
                print(f"      ! FPT failed: {e}")
                row['FPT +1σ'] = 'err'

            # ---- Auto Support & Resistance + P(breach 10d) ----
            try:
                _cl_s  = raw[close_col].dropna()
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
                row['Auto Resist']  = f"INR {_resist:,.2f} ({(_resist/_lc_px-1)*100:+.1f}%)"
                row['Auto Support'] = f"INR {_support:,.2f} ({(_support/_lc_px-1)*100:+.1f}%)"
                
                # Logistic breach probability model
                _thr_w = 10
                _dist  = np.log(_resist / _cl_s)
                _tgt_s, _feats = [], []
                for k in range(len(_cl_s)):
                    fut = _cl_s.values[k+1:k+1+_thr_w]
                    if len(fut) < _thr_w: continue
                    _tgt_s.append(1 if fut.max() >= _resist else 0)
                    _feats.append([float(_dist.iloc[k])])
                if len(_feats) >= 30 and len(set(_tgt_s)) == 2:
                    from sklearn.linear_model import LogisticRegression as _LRb
                    from sklearn.preprocessing import StandardScaler as _SSb
                    _sc = _SSb(); _Xb = _sc.fit_transform(np.array(_feats))
                    _lrb = _LRb(max_iter=300, class_weight='balanced', solver='lbfgs')
                    _lrb.fit(_Xb, np.array(_tgt_s))
                    _curr_Xb = _sc.transform([[float(np.log(_resist / _lc_px))]])
                    _pb = float(_lrb.predict_proba(_curr_Xb)[0][1])
                    row['P(breach 10d)'] = f"{_pb:.1%}"
                else:
                    row['P(breach 10d)'] = 'insuf'
            except Exception as e:
                print(f"      ! Auto Support/Resist failed: {e}")
                row['Auto Resist'] = row['P(breach 10d)'] = 'err'

            # ---- Monte Carlo Geometric Brownian Motion Simulation ----
            try:
                _mc_lr  = met['Log_Ret'].dropna()
                _mc_mu  = float(_mc_lr.mean())
                _mc_sig = float(_mc_lr.std())
                _mc_nu  = _mc_mu - 0.5 * _mc_sig**2
                _mc_s0  = float(raw[close_col].iloc[-1])
                _mc_T   = 20
                np.random.seed(42)
                _mc_r   = np.random.randn(1000, _mc_T)
                _mc_p   = _mc_s0 * np.exp(np.cumsum(_mc_nu + _mc_sig * _mc_r, axis=1))
                _mc_ter = _mc_p[:, -1]
                _var5   = float(np.percentile(_mc_ter, 5))
                _var1   = float(np.percentile(_mc_ter, 1))
                _cvar5  = float(_mc_ter[_mc_ter <= _var5].mean()) if (_mc_ter <= _var5).any() else _var5
                row['MC P(Up 20d)'] = f"{float((_mc_ter > _mc_s0).mean()):.1%}"
                row['MC VaR 5%']    = f"INR {_var5:,.2f} ({(_var5/_mc_s0-1)*100:+.1f}%)"
                row['MC CVaR 5%']   = f"INR {_cvar5:,.2f} ({(_cvar5/_mc_s0-1)*100:+.1f}%)"
                row['MC 5th Pct']   = f"INR {float(np.percentile(_mc_ter,5)):,.2f}"
                row['MC 95th Pct']  = f"INR {float(np.percentile(_mc_ter,95)):,.2f}"
                row['MC Mean 20d']  = f"INR {float(_mc_ter.mean()):,.2f} ({(float(_mc_ter.mean())/_mc_s0-1)*100:+.1f}%)"
            except Exception as e:
                print(f"      ! Monte Carlo failed: {e}")
                row['MC P(Up 20d)'] = 'err'

            rows.append(row)
            
            # Sync to Supabase
            try:
                from supabase_db import SUPABASE_ENABLED, save_volatility_analysis
                if SUPABASE_ENABLED:
                    print("   Syncing volatility analysis to Supabase...")
                    full_data = raw.copy()
                    for col in met.columns:
                        full_data[col] = met[col].values
                    full_data['Donchian_Upper'] = donchian['Donchian_Upper'].values
                    full_data['Donchian_Lower'] = donchian['Donchian_Lower'].values
                    full_data['Donchian_Middle'] = donchian['Donchian_Middle'].values
                    full_data['Volume_SMA'] = volume_met['Volume_SMA'].values
                    full_data['RVOL'] = volume_met['RVOL'].values
                    full_data['Percentile'] = rs.get('percentile')
                    full_data['Regime'] = reg
                    
                    save_volatility_analysis(sym, full_data)
            except Exception as se:
                print(f"      ! Supabase sync failed: {se}")

            print("   -> Completed successfully")

        except Exception as e:
            print(f"   x Outer exception for {sym}: {e}")
            rows.append({'Ticker': sym, 'Status': f'Error: {e}'})

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("ALL MODELS COMPLETED")
    print("=" * 80)
    return df

if __name__ == "__main__":
    # Define time period (past 1 year for statistical models stability)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    output_df = run_pipeline(TICKERS, start_str, end_str)
    
    # Save output to CSV
    csv_filename = "full_portfolio_volatility_analysis.csv"
    output_df.to_csv(csv_filename, index=False)
    print(f"\n[OK] Successfully exported all results to: {os.path.abspath(csv_filename)}")
