"""
predictive_engine.py - Machine Learning & Deep Learning Predictive Synthesis Engine

Features:
- Daily synthesis of all dashboard indicators (Price, Vol, COT, CVD, Waves, Fibs)
- CVD transformed to rolling 20-day Z-Score to maintain stationarity
- Volatility-Normalized Targets: targets divided by GARCH daily standard deviation
- Overlap-Purged Time-Series Split: 5-day purged gap to prevent target overlap leakage
- Scikit-Learn Multi-Output Random Forest baseline
- PyTorch Multi-Task ResNet MLP trained using Quantile (Pinball) Loss (10th/90th percentiles)
- Consensus engine averaging and scaling back predicted Z-scores to price bounds
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import warnings
import sqlite3
import os

warnings.filterwarnings("ignore")

# PyTorch import
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# ==============================================================================
# SQLite ML prediction ledger & online retraining helpers
# ==============================================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_ledger.db')

def init_ledger_db():
    """Initializes SQLite database and prediction_ledger table if they do not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            prediction_date TEXT,
            target_date TEXT,
            close_price REAL,
            daily_vol REAL,
            pred_nd_high REAL,
            pred_nd_low REAL,
            pred_w_high REAL,
            pred_w_low REAL,
            pred_trend INTEGER,
            actual_nd_high REAL,
            actual_nd_low REAL,
            actual_w_high REAL,
            actual_w_low REAL,
            actual_trend INTEGER,
            UNIQUE(ticker, target_date)
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(ticker: str, prediction_date: str, target_date: str, close_price: float, daily_vol: float, pred_dict: dict):
    """Logs a single consensus prediction to the SQLite ledger."""
    init_ledger_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO prediction_ledger (
            ticker, prediction_date, target_date, close_price, daily_vol,
            pred_nd_high, pred_nd_low, pred_w_high, pred_w_low, pred_trend
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker, prediction_date, target_date, float(close_price), float(daily_vol),
        float(pred_dict['nd_high']), float(pred_dict['nd_low']),
        float(pred_dict['weekly_high']), float(pred_dict['weekly_low']),
        int(0 if pred_dict['trend'] == 'Bearish Breakdown' else 2 if pred_dict['trend'] == 'Bullish Breakout' else 1)
    ))
    conn.commit()
    conn.close()

def update_ledger_actuals(ticker: str, data: pd.DataFrame):
    """
    Scans the ledger for missing actuals and updates them using new historical EOD data.
    """
    init_ledger_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, target_date, close_price FROM prediction_ledger
        WHERE ticker = ? AND actual_nd_high IS NULL
    """, (ticker,))
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return
        
    close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
    high = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
    low = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
    
    for row_id, target_date_str, close_price in rows:
        try:
            target_date = pd.Timestamp(target_date_str)
            matching_dates = high.index[high.index >= target_date]
            if len(matching_dates) > 0:
                actual_trading_date = matching_dates[0]
                if (actual_trading_date - target_date).days <= 4:
                    actual_nd_high = float(high.loc[actual_trading_date])
                    actual_nd_low = float(low.loc[actual_trading_date])
                    
                    target_idx = close.index.get_loc(actual_trading_date)
                    if target_idx + 4 < len(close):
                        w_highs = high.iloc[target_idx : target_idx + 5]
                        w_lows = low.iloc[target_idx : target_idx + 5]
                        actual_w_high = float(w_highs.max())
                        actual_w_low = float(w_lows.min())
                        
                        w_return = (float(close.iloc[target_idx + 4]) / close_price) - 1.0
                        if w_return >= 0.015:
                            actual_trend = 2
                        elif w_return <= -0.015:
                            actual_trend = 0
                        else:
                            actual_trend = 1
                            
                        cursor.execute("""
                            UPDATE prediction_ledger
                            SET actual_nd_high = ?, actual_nd_low = ?,
                                actual_w_high = ?, actual_w_low = ?,
                                actual_trend = ?
                            WHERE id = ?
                        """, (actual_nd_high, actual_nd_low, actual_w_high, actual_w_low, actual_trend, row_id))
        except Exception:
            pass
            
    conn.commit()
    conn.close()

def check_concept_drift(ticker: str, threshold: float = 0.015) -> Tuple[bool, float, Optional[List[dict]]]:
    """
    Computes Mean Squared Relative Error of the last 5 completed predictions.
    Returns (is_drifted, current_rmse, completed_rows_list).
    """
    init_ledger_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM prediction_ledger
        WHERE ticker = ? AND actual_nd_high IS NOT NULL
        ORDER BY target_date DESC
        LIMIT 5
    """, (ticker,))
    rows = cursor.fetchall()
    conn.close()
    
    if len(rows) < 5:
        return False, 0.0, None
        
    errors = []
    for r in rows:
        act_h = r['actual_nd_high']
        act_l = r['actual_nd_low']
        pred_h = r['pred_nd_high']
        pred_l = r['pred_nd_low']
        err_h = ((act_h - pred_h) / act_h) ** 2
        err_l = ((act_l - pred_l) / act_l) ** 2
        errors.extend([err_h, err_l])
        
    mse = float(np.mean(errors))
    rmse = np.sqrt(mse)
    return bool(rmse > threshold), rmse, [dict(r) for r in rows]

def get_next_trading_day(date_val):
    """Helper to return the next calendar day, skipping weekends."""
    next_day = date_val + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    return next_day

def retrain_online_pytorch(
    model: nn.Module,
    ticker: str,
    scaler: StandardScaler,
    aligned_features_df: pd.DataFrame,
    aligned_reg_targets_df: pd.DataFrame,
    aligned_class_targets_df: pd.DataFrame,
    hmm: "RegimeSwitchHMM",
    epochs: int = 8
):
    """
    PyTorch online learning:
    - Freezes model shared layers.
    - Unfreezes classification and regression heads.
    - Trains on the last 10 days of drifted data.
    """
    if not TORCH_AVAILABLE or model is None:
        return
    if len(aligned_features_df) < 10:
        return
        
    X_drift = aligned_features_df.iloc[-10:].values
    y_reg_drift = aligned_reg_targets_df.iloc[-10:].values
    y_class_drift = aligned_class_targets_df.iloc[-10:].values.astype(np.int64)
    
    # Extend features with HMM state probabilities
    hmm_feats = X_drift[:, [2, 3]]
    hmm_probs = hmm.predict_state_probs(hmm_feats)
    X_drift_extended = np.hstack([X_drift, hmm_probs])
    
    X_drift_scaled = scaler.transform(X_drift_extended).astype(np.float32)
    X_tensor = torch.tensor(X_drift_scaled)
    y_reg_tensor = torch.tensor(y_reg_drift, dtype=torch.float32)
    y_class_tensor = torch.tensor(y_class_drift, dtype=torch.long)
    
    for param in model.shared.parameters():
        param.requires_grad = False
    for param in model.reg_head.parameters():
        param.requires_grad = True
    for param in model.class_head.parameters():
        param.requires_grad = True
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return
        
    optimizer = torch.optim.Adam(trainable_params, lr=5e-4)
    class_loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_reg, pred_class = model(X_tensor)
        
        l_nd_high = pinball_loss(pred_reg[:, 0], y_reg_tensor[:, 0], 0.90)
        l_nd_low = pinball_loss(pred_reg[:, 1], y_reg_tensor[:, 1], 0.10)
        l_w_high = pinball_loss(pred_reg[:, 2], y_reg_tensor[:, 2], 0.90)
        l_w_low = pinball_loss(pred_reg[:, 3], y_reg_tensor[:, 3], 0.10)
        
        l_reg = l_nd_high + l_nd_low + l_w_high + l_w_low
        l_class = class_loss_fn(pred_class, y_class_tensor)
        
        loss = l_reg + 0.2 * l_class
        loss.backward()
        optimizer.step()
        
    # Re-enable requires_grad for subsequent full retrainings
    for param in model.shared.parameters():
        param.requires_grad = True

def retrain_online_rf(
    rf_reg: RandomForestRegressor,
    rf_clf: RandomForestClassifier,
    scaler: StandardScaler,
    aligned_features_df: pd.DataFrame,
    aligned_reg_targets_df: pd.DataFrame,
    aligned_class_targets_df: pd.DataFrame,
    hmm: "RegimeSwitchHMM"
):
    """
    Random Forest pseudo-online learning:
    - Sets warm_start=True.
    - Increases estimators size by 10.
    - Retrains exclusively on the last 30 days of data.
    """
    if len(aligned_features_df) < 30:
        return
        
    X_drift = aligned_features_df.iloc[-30:].values
    y_reg_drift = aligned_reg_targets_df.iloc[-30:].values
    y_class_drift = aligned_class_targets_df.iloc[-30:].values.astype(np.int64)
    
    # Extend features with HMM state probabilities
    hmm_feats = X_drift[:, [2, 3]]
    hmm_probs = hmm.predict_state_probs(hmm_feats)
    X_drift_extended = np.hstack([X_drift, hmm_probs])
    
    X_drift_scaled = scaler.transform(X_drift_extended)
    
    rf_reg.warm_start = True
    rf_clf.warm_start = True
    
    rf_reg.n_estimators += 10
    rf_clf.n_estimators += 10
    
    rf_reg.fit(X_drift_scaled, y_reg_drift)
    rf_clf.fit(X_drift_scaled, y_class_drift)

# Streamlit caching support
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def cache_data_decorator(*args, **kwargs):
    if HAS_STREAMLIT and st.runtime.exists():
        return st.cache_data(*args, **kwargs)
    return lambda f: f

def cache_resource_decorator(*args, **kwargs):
    if HAS_STREAMLIT and st.runtime.exists():
        return st.cache_resource(*args, **kwargs)
    return lambda f: f


def compute_rolling_cvar(returns_series: pd.Series, window: int = 20, alpha: float = 0.05) -> pd.Series:
    """Computes rolling Conditional Value at Risk (CVaR) at the specified alpha level."""
    def cvar_helper(w):
        var = np.percentile(w, alpha * 100)
        below_var = w[w <= var]
        return float(below_var.mean()) if len(below_var) > 0 else float(var)
    return returns_series.rolling(window).apply(cvar_helper).fillna(0.0)


def compute_rolling_hurst(returns_series: pd.Series, window: int = 63) -> pd.Series:
    """Computes rolling Hurst exponent using R/S analysis proxy."""
    ret_vals = returns_series.fillna(0.0).values
    n = len(ret_vals)
    hurst_vals = np.zeros(n)
    hurst_vals[:window] = 0.5
    for t in range(window, n):
        w = ret_vals[t - window : t]
        try:
            lags = [8, 16, 32, 48]
            tau = [float(np.std(w[lag:] - w[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst_vals[t] = float(poly[0] * 2.0)
        except:
            hurst_vals[t] = 0.5
    return pd.Series(np.clip(hurst_vals, 0.0, 1.0), index=returns_series.index)


class RegimeSwitchHMM:
    """
    Gaussian Mixture Model-based Markov regime switching model.
    Partitions market state into unobserved components (regimes) and estimates
    transition probabilities between them.
    """
    def __init__(self, n_states: int = 3):
        from sklearn.mixture import GaussianMixture
        self.n_states = n_states
        self.gmm = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
        self.transitions = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> np.ndarray:
        # Fit GMM to assign states
        states = self.gmm.fit_predict(X)
        
        # Compute transition matrix A
        n = len(states)
        A = np.zeros((self.n_states, self.n_states))
        for t in range(1, n):
            A[states[t-1], states[t]] += 1
            
        # Normalize rows to transition probabilities
        row_sums = A.sum(axis=1, keepdims=True)
        self.transitions = np.where(row_sums > 0, A / row_sums, 1.0 / self.n_states)
        self.is_fitted = True
        return states
        
    def predict_state_probs(self, X: np.ndarray) -> np.ndarray:
        # X shape: (N, n_features)
        if not self.is_fitted:
            return np.ones((len(X), self.n_states)) / self.n_states
        # Get emission probabilities from GMM
        gmm_probs = self.gmm.predict_proba(X) # shape: (N, n_states)
        
        # Predict next-state probabilities using transition matrix: P(S_{t+1}) = P(S_t) * A
        pred_probs = np.zeros_like(gmm_probs)
        pred_probs[0] = gmm_probs[0]
        for t in range(1, len(X)):
            pred_probs[t] = np.dot(pred_probs[t-1], self.transitions) * gmm_probs[t]
            row_sum = pred_probs[t].sum()
            if row_sum > 0:
                pred_probs[t] /= row_sum
            else:
                pred_probs[t] = gmm_probs[t]
        return pred_probs


# ==============================================================================
# 1. Feature Synthesis
# ==============================================================================

@cache_data_decorator(ttl=900)
def extract_synthesis_features(
    data: pd.DataFrame, 
    metrics: pd.DataFrame, 
    cot_df: pd.DataFrame, 
    daily_cvd: pd.Series, 
    daily_div_signals: pd.Series, 
    pivot_df: pd.DataFrame,
    lstm_series: pd.Series = None,
    event_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Synthesizes all dashboard indicators into a unified daily feature matrix X_t.
    Ensures all scale properties are standardized and stationary.
    """
    idx = data.index
    close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
    high = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
    low = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
    volume = data['Volume'].iloc[:, 0] if isinstance(data['Volume'], pd.DataFrame) else data['Volume']
    
    features = pd.DataFrame(index=idx)
    
    # A. Price & Momentum
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    features['close_to_ma20'] = (close / ma20.replace(0, np.nan) - 1.0).fillna(0.0)
    features['close_to_ma50'] = (close / ma50.replace(0, np.nan) - 1.0).fillna(0.0)
    log_ret = np.log(close / close.shift(1).replace(0, np.nan)).fillna(0.0)
    features['log_returns'] = log_ret
    
    # B. Volatilities
    for col in metrics.columns:
        if col != 'Log_Ret':
            vals = metrics[col].iloc[:, 0] if isinstance(metrics[col], pd.DataFrame) else metrics[col]
            features[f'vol_{col.lower()}'] = vals
            
    # C. Donchian Width
    if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns:
        upper = data['Donchian_Upper'].iloc[:, 0] if isinstance(data['Donchian_Upper'], pd.DataFrame) else data['Donchian_Upper']
        lower = data['Donchian_Lower'].iloc[:, 0] if isinstance(data['Donchian_Lower'], pd.DataFrame) else data['Donchian_Lower']
        middle = data['Donchian_Middle'].iloc[:, 0] if isinstance(data['Donchian_Middle'], pd.DataFrame) else data['Donchian_Middle']
        features['donchian_width'] = ((upper - lower) / middle.replace(0, np.nan)).fillna(0.0)
        
    # D. Volume metrics (RVOL)
    if 'RVOL' in data.columns:
        rvol_s = data['RVOL'].iloc[:, 0] if isinstance(data['RVOL'], pd.DataFrame) else data['RVOL']
        features['rvol'] = rvol_s.fillna(1.0)
    else:
        features['rvol'] = 1.0
        
    # E. COT positioning index
    if cot_df is not None and not cot_df.empty:
        cot_idx = cot_df['COT_Index'].iloc[:, 0] if isinstance(cot_df['COT_Index'], pd.DataFrame) else cot_df['COT_Index']
        spec_net = cot_df['Speculator_Net'].iloc[:, 0] if isinstance(cot_df['Speculator_Net'], pd.DataFrame) else cot_df['Speculator_Net']
        features['cot_index'] = (cot_idx.fillna(50.0) / 100.0)  # Standardized to [0, 1]
        features['speculator_net_pct'] = (spec_net / spec_net.abs().rolling(252).mean().replace(0, np.nan)).fillna(0.0)
    else:
        features['cot_index'] = 0.5
        features['speculator_net_pct'] = 0.0
        
    # F. Cumulative Volume Delta (CVD) - Stationary Rolling Z-Score
    if daily_cvd is not None:
        cvd_val = daily_cvd.iloc[:, 0] if isinstance(daily_cvd, pd.DataFrame) else daily_cvd
        cvd_roll = cvd_val.rolling(window=20)
        features['cvd_zscore'] = ((cvd_val - cvd_roll.mean()) / cvd_roll.std().replace(0, np.nan)).fillna(0.0)
        features['cvd_5d_slope'] = (cvd_val.diff(5).fillna(0.0) / cvd_roll.std().replace(0, np.nan).fillna(1.0))
        
        # One-Hot encoding of daily CVD divergence signals: Bullish (1), Bearish (-1), Neutral (0)
        if daily_div_signals is not None:
            div_sig = daily_div_signals.iloc[:, 0] if isinstance(daily_div_signals, pd.DataFrame) else daily_div_signals
            features['cvd_div_bull'] = (div_sig == 1.0).astype(float)
            features['cvd_div_bear'] = (div_sig == -1.0).astype(float)
            features['cvd_div_neut'] = (div_sig == 0.0).astype(float)
        else:
            features['cvd_div_bull'] = 0.0
            features['cvd_div_bear'] = 0.0
            features['cvd_div_neut'] = 1.0
    else:
        features['cvd_zscore'] = 0.0
        features['cvd_5d_slope'] = 0.0
        features['cvd_div_bull'] = 0.0
        features['cvd_div_bear'] = 0.0
        features['cvd_div_neut'] = 1.0
        
    # G. Wave & Fibonacci
    if pivot_df is not None and not pivot_df.empty:
        days_since_pivot = []
        last_pivot_type = []
        fib_0_dist = []
        fib_618_dist = []
        
        for t in idx:
            past_pivots = pivot_df[pivot_df.index <= t]
            if len(past_pivots) >= 2:
                last_p = past_pivots.iloc[-1]
                prev_p = past_pivots.iloc[-2]
                
                days_since_pivot.append(float((t - last_p.name).days))
                last_pivot_type.append(1.0 if last_p['type'] == 'Peak' else -1.0)
                
                p1_val = float(prev_p['price'])
                p2_val = float(last_p['price'])
                if last_p['type'] == 'Trough':
                    sh, sl = p1_val, p2_val
                    trend = 'bearish'
                else:
                    sh, sl = p2_val, p1_val
                    trend = 'bullish'
                    
                diff = sh - sl
                if diff > 0:
                    fib_618 = sh - 0.618 * diff if trend == 'bearish' else sl + 0.618 * diff
                    fib_0 = sl if trend == 'bearish' else sh
                    
                    price_t = float(close.loc[t])
                    fib_0_dist.append((price_t / fib_0) - 1.0)
                    fib_618_dist.append((price_t / fib_618) - 1.0)
                else:
                    fib_0_dist.append(0.0)
                    fib_618_dist.append(0.0)
            else:
                days_since_pivot.append(0.0)
                last_pivot_type.append(0.0)
                fib_0_dist.append(0.0)
                fib_618_dist.append(0.0)
                
        features['days_since_pivot'] = days_since_pivot
        features['last_pivot_type'] = last_pivot_type
        features['fib_0_dist'] = fib_0_dist
        features['fib_618_dist'] = fib_618_dist
    else:
        features['days_since_pivot'] = 0.0
        features['last_pivot_type'] = 0.0
        features['fib_0_dist'] = 0.0
        features['fib_618_dist'] = 0.0

    # H. VOLATILITY & RISK PROFILE (Tensor 1)
    ewma_vol = metrics['EWMA'].iloc[:, 0] if isinstance(metrics['EWMA'], pd.DataFrame) else metrics['EWMA']
    ewma_vol = ewma_vol.ffill().bfill().fillna(0.20)
    
    # Simple estimate of daily vol based on EWMA
    daily_vol = (ewma_vol / np.sqrt(252)).ffill().bfill().fillna(0.02)
    daily_vol_ann = daily_vol * np.sqrt(252)

    if lstm_series is not None:
        lstm_s = lstm_series.reindex(idx).ffill().bfill().fillna(daily_vol_ann)
        features['lstm_garch_delta'] = ((lstm_s - daily_vol_ann) / daily_vol_ann.replace(0, 1e-4)).fillna(0.0)
    else:
        # Fallback to rolling std vs EWMA delta
        vol_20d = metrics['Vol_20d'].iloc[:, 0] if isinstance(metrics['Vol_20d'], pd.DataFrame) else metrics['Vol_20d']
        features['lstm_garch_delta'] = ((vol_20d - ewma_vol) / ewma_vol.replace(0, 1e-4)).fillna(0.0)

    features['vol_percentile'] = ewma_vol.rolling(252, min_periods=1).rank(pct=True).fillna(0.5)
    
    # CVaR Tail Risk ratio helper
    features['cvar_ratio'] = compute_rolling_cvar(log_ret, window=20, alpha=0.05)

    # I. STRUCTURAL MEMORY & TIME SERIES (Tensor 2)
    features['hurst'] = compute_rolling_hurst(log_ret, window=63)
    features['arima_drift'] = log_ret.rolling(20).corr(log_ret.shift(1)).fillna(0.0)
    features['mc_bias'] = (log_ret.rolling(20).mean() / log_ret.rolling(20).std().replace(0, np.nan)).fillna(0.0)

    # J. OPTIONS GRAVITY MATRIX (Tensor 3)
    # Generate historical synthetic GEX and Gamma Flip levels
    net_gex_pins = []
    gamma_flips = []
    import confirmations_engine as ce
    close_clean = close.ffill().bfill().fillna(100.0)
    ewma_vol_clean = ewma_vol.ffill().bfill().fillna(0.20)
    for t_idx in idx:
        spot = float(close_clean.loc[t_idx])
        vol = float(ewma_vol_clean.loc[t_idx])
        # Generate synthetic GEX profile
        gex_profile = ce.get_synthetic_gex_profile(spot, vol)
        levels = ce.extract_gex_key_levels(gex_profile, spot)
        net_gex_pins.append(levels.get('peak_net_strike', spot))
        gamma_flips.append(levels.get('flip_price', spot))
        
    gex_pin_s = pd.Series(net_gex_pins, index=idx)
    gamma_flip_s = pd.Series(gamma_flips, index=idx)
    
    features['dist_to_gex_pin'] = (gex_pin_s - close) / close
    features['dist_to_gamma_flip'] = (gamma_flip_s - close) / close
    
    # Standard normal CDF for option breach probability
    from scipy.stats import norm
    dist_sd = features['dist_to_gamma_flip'].abs() / (daily_vol * np.sqrt(10))
    features['breach_prob'] = pd.Series(2.0 * (1.0 - norm.cdf(dist_sd)), index=idx).fillna(0.0)

    # K. MACRO & EVENT PROXIMITY (Tensor 5)
    event_dates = []
    event_types_map = {}
    if event_df is not None and not event_df.empty:
        for _, row in event_df.iterrows():
            dt = pd.Timestamp(row['Date'])
            event_dates.append(dt)
            event_types_map[dt] = row['Event_Type']
            
    vol_impact_dict = {
        'RBI MPC': -0.234,      # post-meeting vol crush
        'CPI Release': 0.125,    # inflation day expansion
        'Union Budget': 0.284,   # high expansion
        'US Fed Meeting': 0.085  # standard expansion
    }
    
    days_to_event_series = []
    event_impact_series = []
    for t_idx in idx:
        future_evs = [ev for ev in event_dates if ev > t_idx]
        if future_evs:
            next_ev = future_evs[0]
            days_to_event_series.append(float((next_ev - t_idx).days))
            ev_type = event_types_map.get(next_ev, 'Other')
            event_impact_series.append(vol_impact_dict.get(ev_type, 0.0))
        else:
            days_to_event_series.append(30.0)
            event_impact_series.append(0.0)
            
    features['days_to_event'] = days_to_event_series
    features['event_vol_impact'] = event_impact_series

    # L. MICROSTRUCTURE ORDER BOOK IMPLANCE (OBI)
    # EOD proxy representing the buyers/sellers imbalance ratio from Cumulative Volume Delta (CVD)
    features['obi'] = (daily_cvd.diff().fillna(0.0) / volume.replace(0, 1.0)).clip(-1.0, 1.0).fillna(0.0)

    return features.ffill().bfill().fillna(0.0)


# ==============================================================================
# 2. Volatility-Normalized Target Compiler
# ==============================================================================

def build_synthesis_targets(data: pd.DataFrame, daily_vol: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compiles price high/low targets and weekly trend classifications.
    All regression targets are Z-Scored (divided by GARCH daily volatility)
    to standardize learning across regimes and assets.
    """
    close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
    high = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
    low = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
    
    close_vals = close.values
    high_vals = high.values
    low_vals = low.values
    vol_vals = daily_vol.values
    n = len(data)
    
    reg_targets = []
    class_targets = []
    
    for t in range(n):
        if t >= n - 5:
            # Not enough future data to define weekly behavior targets
            reg_targets.append([np.nan, np.nan, np.nan, np.nan])
            class_targets.append(np.nan)
            continue
            
        # Daily volatility threshold (prevent division by zero)
        sigma = max(vol_vals[t], 0.0001)
        
        # 1. Next-day Intraday High / Low (Z-Scored)
        nd_high_z = ((high_vals[t+1] / close_vals[t]) - 1.0) / sigma
        nd_low_z = ((low_vals[t+1] / close_vals[t]) - 1.0) / sigma
        
        # 2. Weekly (5-day) Max High / Min Low (Z-Scored)
        w_highs = high_vals[t+1 : t+6]
        w_lows = low_vals[t+1 : t+6]
        w_max_high_z = ((np.max(w_highs) / close_vals[t]) - 1.0) / sigma
        w_min_low_z = ((np.min(w_lows) / close_vals[t]) - 1.0) / sigma
        
        # 3. Weekly Trend direction (Un-normalized % to maintain standard category checks)
        w_return = (close_vals[t+5] / close_vals[t]) - 1.0
        if w_return >= 0.015:
            trend = 2  # Bullish
        elif w_return <= -0.015:
            trend = 0  # Bearish
        else:
            trend = 1  # Neutral
            
        reg_targets.append([nd_high_z, nd_low_z, w_max_high_z, w_min_low_z])
        class_targets.append(trend)
        
    df_reg = pd.DataFrame(
        reg_targets, 
        columns=['nd_high', 'nd_low', 'w_high', 'w_low'], 
        index=data.index
    )
    df_class = pd.DataFrame(
        class_targets, 
        columns=['weekly_trend'], 
        index=data.index
    )
    
    return df_reg, df_class


# ==============================================================================
# 3. PyTorch Model Definition & Quantile Loss
# ==============================================================================

if TORCH_AVAILABLE:
    def pinball_loss(pred, target, tau):
        """
        Quantile Loss function (Pinball Loss) for boundary estimations.
        Highs are trained at tau = 0.90, Lows at tau = 0.10.
        """
        diff = target - pred
        return torch.max(tau * diff, (tau - 1.0) * diff).mean()

    class SynthesisMultiTaskNet(nn.Module):
        """
        Deep Residual Multi-Task network outputting continuous levels (regression)
        and weekly return category probabilities (classification).
        """
        def __init__(self, input_size: int, hidden_size: int = 64):
            super().__init__()
            # Core shared feature learning blocks
            self.shared = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.2)
            )
            
            # Regression Head: outputs 4 values [next-day high, next-day low, weekly max high, weekly min low]
            self.reg_head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 4)
            )
            
            # Classification Head: outputs log-probabilities for 3 classes [Bearish, Neutral, Bullish]
            self.class_head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 3)
            )
            
        def forward(self, x):
            feats = self.shared(x)
            reg_out = self.reg_head(feats)
            class_out = self.class_head(feats)
            return reg_out, class_out
else:
    class SynthesisMultiTaskNet:
        pass


# ==============================================================================
# 4. Pipeline Engine
# ==============================================================================

class SynthesisPredictiveEngine:
    """
    Prepares data, trains Random Forest models & PyTorch Multi-Task Net,
    and runs consensus pricing and trend inference.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        metrics: pd.DataFrame,
        cot_df: pd.DataFrame,
        daily_cvd: pd.Series,
        daily_div_signals: pd.Series,
        pivot_df: pd.DataFrame,
        lstm_series: pd.Series = None,
        event_df: pd.DataFrame = None,
        ticker: str = "UNKNOWN"
    ):
        self.data = data
        self.metrics = metrics
        self.cot_df = cot_df
        self.daily_cvd = daily_cvd
        self.daily_div_signals = daily_div_signals
        self.pivot_df = pivot_df
        self.lstm_series = lstm_series
        self.event_df = event_df
        self.ticker = ticker
        
        self.scaler = StandardScaler()
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)
        self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True)
        
        self.nn_model = None
        self.is_trained = False
        
        # Aligned historical data placeholders
        self.aligned_features_df = None
        self.aligned_reg_targets_df = None
        self.aligned_class_targets_df = None
        
        # Meta stats
        self.features_count = 0
        self.train_samples = 0
        self.val_accuracy = 0.0
        self.val_mse = 0.0
        self.train_loss_history = []
        
        # Establish active daily GARCH volatility series
        garch_fitted = False
        if 'Log_Ret' in self.metrics.columns or 'Close' in self.data.columns:
            try:
                from arch import arch_model
                returns = self.metrics['Log_Ret'] if 'Log_Ret' in self.metrics.columns else np.log(self.data['Close'] / self.data['Close'].shift(1)).dropna()
                clean_returns = returns.dropna() * 100  # Scale returns to percentage units for stability
                model = arch_model(clean_returns, vol='Garch', p=1, q=1)
                result = model.fit(disp='off')
                # Convert conditional volatility back to daily decimal units (from % units)
                garch_vol = pd.Series(result.conditional_volatility / 100, index=clean_returns.index)
                self.daily_vol = garch_vol.reindex(self.data.index).ffill().bfill().fillna(0.02)
                garch_fitted = True
            except Exception as e:
                pass
        
        if not garch_fitted:
            vol_series = self.metrics['Vol_20d'] if 'Vol_20d' in self.metrics.columns else (
                self.metrics['Vol_20'] if 'Vol_20' in self.metrics.columns else self.metrics['EWMA']
            )
            self.daily_vol = (vol_series / np.sqrt(252)).ffill().bfill().fillna(0.02)
        
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs the extraction pipeline, aligns, and drop NaNs."""
        # 1. Feature matrix
        X_df = extract_synthesis_features(
            self.data, self.metrics, self.cot_df, 
            self.daily_cvd, self.daily_div_signals, self.pivot_df,
            lstm_series=self.lstm_series, event_df=self.event_df
        )
        self.features_count = X_df.shape[1]
        
        # 2. Target matrices (with GARCH daily vol division)
        y_reg_df, y_class_df = build_synthesis_targets(self.data, self.daily_vol)
        
        # Align: drop rows at the end where target is NaN (last 5 days)
        # also drop initial rows where rolling features are NaN
        merged = pd.concat([X_df, y_reg_df, y_class_df], axis=1).dropna()
        
        # Keep features & targets
        self.aligned_index = merged.index
        
        feat_cols = X_df.columns
        self.aligned_features_df = merged[feat_cols]
        self.aligned_reg_targets_df = merged[['nd_high', 'nd_low', 'w_high', 'w_low']]
        self.aligned_class_targets_df = merged['weekly_trend']
        
        X = merged[feat_cols].values
        y_reg = merged[['nd_high', 'nd_low', 'w_high', 'w_low']].values
        y_class = merged['weekly_trend'].values.astype(np.int64)
        
        return X, y_reg, y_class
        
    def train(self, epochs: int = 50, progress_callback=None) -> Dict:
        """Trains Hidden Markov Model, Random Forest models & PyTorch Multi-Task Net."""
        X, y_reg, y_class = self._prepare_data()
        n_samples = len(X)
        self.train_samples = n_samples
        
        if n_samples < 20:
            raise ValueError(f"Insufficient historical samples to train AI models (need >20, got {n_samples}). Try increasing the date range.")
            
        # Fit RegimeSwitchHMM on returns (index 2) and vol (index 3)
        hmm_feats = X[:, [2, 3]]
        self.hmm = RegimeSwitchHMM(n_states=3)
        self.hmm.fit(hmm_feats)
        hmm_probs = self.hmm.predict_state_probs(hmm_feats)
        
        # Extend features X with HMM state probabilities
        X_extended = np.hstack([X, hmm_probs])
        self.features_count = X_extended.shape[1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_extended).astype(np.float32)
        
        # Chronological Train / Validation Split (80/20)
        # Enforce a 5-day purging gap between sets to prevent target leakage
        split = int(n_samples * 0.8)
        X_train = X_scaled[:split - 5]
        y_reg_train = y_reg[:split - 5]
        y_class_train = y_class[:split - 5]
        
        X_val = X_scaled[split:]
        y_reg_val = y_reg[split:]
        y_class_val = y_class[split:]
        
        # --- A. Train Random Forest (ML Baseline) ---
        self.rf_reg.fit(X_train, y_reg_train)
        self.rf_clf.fit(X_train, y_class_train)
        
        # Evaluate RF Baseline
        rf_reg_pred = self.rf_reg.predict(X_val)
        rf_clf_pred = self.rf_clf.predict(X_val)
        self.val_mse = float(mean_squared_error(y_reg_val, rf_reg_pred))
        self.val_accuracy = float(accuracy_score(y_class_val, rf_clf_pred))
        
        # --- B. Train PyTorch Multi-Task Net ---
        self.q_nd = 0.0
        self.q_w = 0.0
        
        if TORCH_AVAILABLE:
            train_ds = TensorDataset(
                torch.tensor(X_train),
                torch.tensor(y_reg_train, dtype=torch.float32),
                torch.tensor(y_class_train, dtype=torch.long)
            )
            loader = DataLoader(train_ds, batch_size=16, shuffle=False)
            
            model = SynthesisMultiTaskNet(input_size=self.features_count, hidden_size=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            
            # Loss Functions: Quantile pinball for regression bounds + CrossEntropy for classification
            class_loss_fn = nn.CrossEntropyLoss()
            
            self.train_loss_history = []
            
            for epoch in range(epochs):
                model.train()
                batch_losses = []
                for Xb, yrb, ycb in loader:
                    optimizer.zero_grad()
                    pred_reg, pred_class = model(Xb)
                    
                    # Quantile loss on regressor bounds
                    l_nd_high = pinball_loss(pred_reg[:, 0], yrb[:, 0], 0.90)  # High bounds targeting 90th pct
                    l_nd_low = pinball_loss(pred_reg[:, 1], yrb[:, 1], 0.10)   # Low bounds targeting 10th pct
                    l_w_high = pinball_loss(pred_reg[:, 2], yrb[:, 2], 0.90)
                    l_w_low = pinball_loss(pred_reg[:, 3], yrb[:, 3], 0.10)
                    
                    l_reg = l_nd_high + l_nd_low + l_w_high + l_w_low
                    l_class = class_loss_fn(pred_class, ycb)
                    
                    loss = l_reg + 0.2 * l_class
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                    
                epoch_loss = float(np.mean(batch_losses))
                self.train_loss_history.append(epoch_loss)
                
                if progress_callback:
                    progress_callback(epoch + 1, epochs, epoch_loss)
                    
            self.nn_model = model
            model.eval()
            
            # Neural Net metrics check & Conformal Calibration
            with torch.no_grad():
                X_val_t = torch.tensor(X_val)
                nn_reg, nn_cls = model(X_val_t)
                nn_reg = nn_reg.cpu().numpy()
                nn_cls_lbl = torch.argmax(nn_cls, dim=1).cpu().numpy()
                
                nn_mse = float(mean_squared_error(y_reg_val, nn_reg))
                nn_acc = float(accuracy_score(y_class_val, nn_cls_lbl))
                
                self.val_mse = min(self.val_mse, nn_mse)
                self.val_accuracy = max(self.val_accuracy, nn_acc)
                
                # Conformalized Quantile Regression calibration (95% coverage)
                # Calculates how much validation target values exceeded the predicted 10th/90th quantile bounds
                err_nd = np.maximum(nn_reg[:, 1] - y_reg_val[:, 1], y_reg_val[:, 0] - nn_reg[:, 0])
                err_w = np.maximum(nn_reg[:, 3] - y_reg_val[:, 3], y_reg_val[:, 2] - nn_reg[:, 2])
                
                alpha = 0.05
                self.q_nd = float(np.percentile(err_nd, (1.0 - alpha) * 100))
                self.q_w = float(np.percentile(err_w, (1.0 - alpha) * 100))
                
        self.is_trained = True
        return {
            'samples': self.train_samples,
            'features': self.features_count,
            'val_mse': self.val_mse,
            'val_accuracy': self.val_accuracy
        }
        
    def predict_latest(self) -> Dict:
        """
        Runs predictions on the very latest data row to project
        intraday range and weekly trend. Scales the predicted Z-Scores back 
        to price levels using the latest GARCH daily volatility.
        Calibrates outputs using Conformal Prediction (CQR) and predicts HMM regime state.
        Monitors concept drift and performs online updates of models when necessary.
        """
        if not self.is_trained:
            raise RuntimeError("Engine must be trained before predicting.")
            
        # --- ML Exclusive Online Retrain Loop (Concept Drift) ---
        # 1. Update ledger actuals using latest historical data
        try:
            update_ledger_actuals(self.ticker, self.data)
        except Exception:
            pass
            
        # 2. Check for concept drift (rolling 5-day error threshold of 1.5%)
        is_drifted = False
        rmse = 0.0
        try:
            is_drifted, rmse, recent_rows = check_concept_drift(self.ticker, threshold=0.015)
        except Exception:
            pass
            
        # 3. If drift is triggered, run model-specific online updates
        drift_event_fired = False
        if is_drifted:
            drift_event_fired = True
            try:
                if TORCH_AVAILABLE and self.nn_model is not None:
                    retrain_online_pytorch(
                        model=self.nn_model,
                        ticker=self.ticker,
                        scaler=self.scaler,
                        aligned_features_df=self.aligned_features_df,
                        aligned_reg_targets_df=self.aligned_reg_targets_df,
                        aligned_class_targets_df=self.aligned_class_targets_df,
                        hmm=self.hmm,
                        epochs=8
                    )
                retrain_online_rf(
                    rf_reg=self.rf_reg,
                    rf_clf=self.rf_clf,
                    scaler=self.scaler,
                    aligned_features_df=self.aligned_features_df,
                    aligned_reg_targets_df=self.aligned_reg_targets_df,
                    aligned_class_targets_df=self.aligned_class_targets_df,
                    hmm=self.hmm
                )
            except Exception:
                pass
        
        # 1. Compile today's feature row
        X_all = extract_synthesis_features(
            self.data, self.metrics, self.cot_df, 
            self.daily_cvd, self.daily_div_signals, self.pivot_df,
            lstm_series=self.lstm_series, event_df=self.event_df
        )
        latest_row = X_all.iloc[[-1]].values
        
        # Predict HMM probabilities for latest row
        latest_hmm_feat = latest_row[:, [2, 3]]
        latest_hmm_probs = self.hmm.predict_state_probs(latest_hmm_feat)[0]
        
        # Extend latest row with HMM state probabilities and scale
        latest_row_extended = np.hstack([latest_row, latest_hmm_probs.reshape(1, -1)])
        latest_row_scaled = self.scaler.transform(latest_row_extended).astype(np.float32)
        
        # Classify HMM regime based on volatility sorting
        vol_means = self.hmm.gmm.means_[:, 1]
        sorted_states = np.argsort(vol_means)
        state_map = {
            sorted_states[0]: 'Low Volatility Rotational Grind',
            sorted_states[1]: 'Neutral Range',
            sorted_states[2]: 'High Volatility Distribution'
        }
        active_state_idx = int(np.argmax(latest_hmm_probs))
        active_regime = state_map.get(active_state_idx, 'Neutral Range')
        
        close_latest = float(self.data['Close'].dropna().iloc[-1])
        daily_vol_latest = float(max(self.daily_vol.iloc[-1], 0.0001))
        
        # A. Random Forest Predictions
        rf_reg_pred = self.rf_reg.predict(latest_row_scaled)[0]
        rf_clf_probs = self.rf_clf.predict_proba(latest_row_scaled)[0]
        rf_clf_pred = int(np.argmax(rf_clf_probs))
        
        # B. PyTorch Predictions
        if TORCH_AVAILABLE and self.nn_model is not None:
            self.nn_model.eval()
            with torch.no_grad():
                row_t = torch.tensor(latest_row_scaled)
                nn_reg, nn_cls = self.nn_model(row_t)
                nn_reg_pred = nn_reg.cpu().numpy()[0]
                nn_cls_probs = torch.softmax(nn_cls, dim=1).cpu().numpy()[0]
                nn_clf_pred = int(np.argmax(nn_cls_probs))
        else:
            nn_reg_pred = rf_reg_pred
            nn_cls_probs = rf_clf_probs
            nn_clf_pred = rf_clf_pred
            
        # C. Consensus Synthesis (Average)
        cons_reg_pred = 0.5 * (rf_reg_pred + nn_reg_pred)
        cons_cls_probs = 0.5 * (rf_clf_probs + nn_cls_probs)
        cons_clf_pred = int(np.argmax(cons_cls_probs))
        
        trend_map = {0: 'Bearish Breakdown', 1: 'Neutral Range', 2: 'Bullish Breakout'}
        
        def _compile_prediction(reg, trend_idx, probs):
            # Scale the Z-Score regressor outputs back to absolute prices
            # target_price = close * (1.0 + z_score * vol_daily)
            return {
                'nd_high': close_latest * (1.0 + reg[0] * daily_vol_latest),
                'nd_low': close_latest * (1.0 + reg[1] * daily_vol_latest),
                'weekly_high': close_latest * (1.0 + reg[2] * daily_vol_latest),
                'weekly_low': close_latest * (1.0 + reg[3] * daily_vol_latest),
                'trend': trend_map[trend_idx],
                'prob_bear': float(probs[0]),
                'prob_neut': float(probs[1]),
                'prob_bull': float(probs[2])
            }
            
        res = {
            'close_price': close_latest,
            'ml_model': _compile_prediction(rf_reg_pred, rf_clf_pred, rf_clf_probs),
            'nn_model': _compile_prediction(nn_reg_pred, nn_clf_pred, nn_cls_probs),
            'consensus': _compile_prediction(cons_reg_pred, cons_clf_pred, cons_cls_probs),
            'conformal': {
                'nd_high': close_latest * (1.0 + (nn_reg_pred[0] + self.q_nd) * daily_vol_latest),
                'nd_low': close_latest * (1.0 + (nn_reg_pred[1] - self.q_nd) * daily_vol_latest),
                'weekly_high': close_latest * (1.0 + (nn_reg_pred[2] + self.q_w) * daily_vol_latest),
                'weekly_low': close_latest * (1.0 + (nn_reg_pred[3] - self.q_w) * daily_vol_latest),
            },
            'hmm_regime': active_regime,
            'hmm_probs': {
                'Rotational Grind': float(latest_hmm_probs[sorted_states[0]]),
                'Neutral Range': float(latest_hmm_probs[sorted_states[1]]),
                'Volatility Distribution': float(latest_hmm_probs[sorted_states[2]])
            },
            'val_accuracy': self.val_accuracy,
            'val_mse': self.val_mse,
            'concept_drift_event': drift_event_fired,
            'concept_drift_rmse': rmse
        }
        
        # 4. Log the generated consensus prediction to SQLite ledger for next trading day
        try:
            pred_date = self.data.index[-1]
            t_date = get_next_trading_day(pred_date)
            log_prediction(
                ticker=self.ticker,
                prediction_date=str(pred_date.date()),
                target_date=str(t_date.date()),
                close_price=close_latest,
                daily_vol=daily_vol_latest,
                pred_dict=res['consensus']
            )
        except Exception:
            pass
            
        return res
        
@cache_resource_decorator(ttl=900)
def get_trained_predictive_engine(
    ticker: str,
    start_date: str,
    end_date: str,
    _data: pd.DataFrame,
    _metrics: pd.DataFrame,
    _cot_df: pd.DataFrame,
    _daily_cvd: pd.Series,
    _daily_div_signals: pd.Series,
    _pivot_df: pd.DataFrame,
    predict_epochs: int,
    _lstm_series: pd.Series = None,
    _event_df: pd.DataFrame = None
) -> SynthesisPredictiveEngine:
    """
    Initializes and trains the synthesis predictive engine resource, caching it in global RAM.
    Avoids hashing large pandas structures by prefixing them with an underscore.
    """
    engine = SynthesisPredictiveEngine(
        data=_data,
        metrics=_metrics,
        cot_df=_cot_df,
        daily_cvd=_daily_cvd,
        daily_div_signals=_daily_div_signals,
        pivot_df=_pivot_df,
        lstm_series=_lstm_series,
        event_df=_event_df,
        ticker=ticker
    )
    engine.train(epochs=predict_epochs)
    return engine
