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

# Streamlit caching support
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def cache_data_decorator(*args, **kwargs):
    if HAS_STREAMLIT:
        return st.cache_data(*args, **kwargs)
    return lambda f: f

def cache_resource_decorator(*args, **kwargs):
    if HAS_STREAMLIT:
        return st.cache_resource(*args, **kwargs)
    return lambda f: f


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
    pivot_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Synthesizes all dashboard indicators into a unified daily feature matrix.
    Enforces stationarity on cumulative volume delta (CVD).
    """
    idx = data.index
    close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
    high = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
    low = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
    
    features = pd.DataFrame(index=idx)
    
    # A. Price & Momentum
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    features['close_to_ma20'] = (close / ma20.replace(0, np.nan) - 1.0).fillna(0.0)
    features['close_to_ma50'] = (close / ma50.replace(0, np.nan) - 1.0).fillna(0.0)
    features['log_returns'] = np.log(close / close.shift(1).replace(0, np.nan)).fillna(0.0)
    
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
        
    # D. Volume metrics
    if 'RVOL' in data.columns:
        rvol_s = data['RVOL'].iloc[:, 0] if isinstance(data['RVOL'], pd.DataFrame) else data['RVOL']
        features['rvol'] = rvol_s.fillna(1.0)
    else:
        features['rvol'] = 1.0
        
    # E. COT derivative positioning
    if cot_df is not None and not cot_df.empty:
        cot_idx = cot_df['COT_Index'].iloc[:, 0] if isinstance(cot_df['COT_Index'], pd.DataFrame) else cot_df['COT_Index']
        spec_net = cot_df['Speculator_Net'].iloc[:, 0] if isinstance(cot_df['Speculator_Net'], pd.DataFrame) else cot_df['Speculator_Net']
        features['cot_index'] = cot_idx.fillna(50.0)
        features['speculator_net_pct'] = (spec_net / spec_net.abs().rolling(252).mean().replace(0, np.nan)).fillna(0.0)
    else:
        features['cot_index'] = 50.0
        features['speculator_net_pct'] = 0.0
        
    # F. Cumulative Volume Delta (CVD) - Stationary Rolling Z-Score
    if daily_cvd is not None:
        cvd_val = daily_cvd.iloc[:, 0] if isinstance(daily_cvd, pd.DataFrame) else daily_cvd
        cvd_roll = cvd_val.rolling(window=20)
        features['cvd_zscore'] = ((cvd_val - cvd_roll.mean()) / cvd_roll.std().replace(0, np.nan)).fillna(0.0)
        features['cvd_5d_slope'] = (cvd_val.diff(5).fillna(0.0) / cvd_roll.std().replace(0, np.nan).fillna(1.0))
    else:
        features['cvd_zscore'] = 0.0
        features['cvd_5d_slope'] = 0.0
        
    if daily_div_signals is not None:
        div_sig = daily_div_signals.iloc[:, 0] if isinstance(daily_div_signals, pd.DataFrame) else daily_div_signals
        features['cvd_div_sig'] = div_sig.fillna(0.0)
    else:
        features['cvd_div_sig'] = 0.0
        
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
        pivot_df: pd.DataFrame
    ):
        self.data = data
        self.metrics = metrics
        self.cot_df = cot_df
        self.daily_cvd = daily_cvd
        self.daily_div_signals = daily_div_signals
        self.pivot_df = pivot_df
        
        self.scaler = StandardScaler()
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.nn_model = None
        self.is_trained = False
        
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
            self.daily_cvd, self.daily_div_signals, self.pivot_df
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
        X = merged[feat_cols].values
        y_reg = merged[['nd_high', 'nd_low', 'w_high', 'w_low']].values
        y_class = merged['weekly_trend'].values.astype(np.int64)
        
        return X, y_reg, y_class
        
    def train(self, epochs: int = 50, progress_callback=None) -> Dict:
        """Trains both standard ML models and Deep PyTorch Network."""
        X, y_reg, y_class = self._prepare_data()
        n_samples = len(X)
        self.train_samples = n_samples
        
        if n_samples < 20:
            raise ValueError(f"Insufficient historical samples to train AI models (need >20, got {n_samples}). Try increasing the date range.")
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        
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
            
            # Neural Net metrics check
            with torch.no_grad():
                X_val_t = torch.tensor(X_val)
                nn_reg, nn_cls = model(X_val_t)
                nn_reg = nn_reg.cpu().numpy()
                nn_cls_lbl = torch.argmax(nn_cls, dim=1).cpu().numpy()
                
                nn_mse = float(mean_squared_error(y_reg_val, nn_reg))
                nn_acc = float(accuracy_score(y_class_val, nn_cls_lbl))
                
                self.val_mse = min(self.val_mse, nn_mse)
                self.val_accuracy = max(self.val_accuracy, nn_acc)
                
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
        """
        if not self.is_trained:
            raise RuntimeError("Engine must be trained before predicting.")
            
        # 1. Compile today's feature row
        X_all = extract_synthesis_features(
            self.data, self.metrics, self.cot_df, 
            self.daily_cvd, self.daily_div_signals, self.pivot_df
        )
        latest_row = X_all.iloc[[-1]].values
        latest_row_scaled = self.scaler.transform(latest_row).astype(np.float32)
        
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
            
        return {
            'close_price': close_latest,
            'ml_model': _compile_prediction(rf_reg_pred, rf_clf_pred, rf_clf_probs),
            'nn_model': _compile_prediction(nn_reg_pred, nn_clf_pred, nn_cls_probs),
            'consensus': _compile_prediction(cons_reg_pred, cons_clf_pred, cons_cls_probs),
            'val_accuracy': self.val_accuracy,
            'val_mse': self.val_mse
        }


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
    predict_epochs: int
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
        pivot_df=_pivot_df
    )
    engine.train(epochs=predict_epochs)
    return engine

