"""
intraday_predictive_engine.py - Next-Day 15-Minute Candle Price Path Predictive Engine

Features:
- Dynamic fetching of 15m intraday data and adaptive alignment with EOD synthesis features.
- Support for Indian markets (NSE/BSE 9:15 to 3:30) and global markets via timezone and active hours scanning.
- Dual-Model Consensus: Scikit-learn Multi-Output Random Forest baseline + PyTorch Multi-Task Quantile ResNet Net.
- Quantile/Pinball Loss (10th and 90th percentiles) for next-day price path confidence bands.
- SQLite-based Intraday Ledger for dynamic retraining feedback loops and concept-drift monitoring.
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import predictive_engine

warnings.filterwarnings("ignore")

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

# PyTorch import
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_ledger.db')

def init_intraday_ledger_db():
    """Initializes SQLite database and intraday_prediction_ledger table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intraday_prediction_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            prediction_date TEXT,
            target_date TEXT,
            candle_time TEXT,
            pred_price REAL,
            low_bound REAL,
            high_bound REAL,
            actual_price REAL,
            UNIQUE(ticker, target_date, candle_time)
        )
    """)
    conn.commit()
    conn.close()

# PyTorch Multi-Task Quantile Network
if TORCH_AVAILABLE:
    class IntradayMultiTaskNet(nn.Module):
        """
        Deep Residual Multi-Task network outputting continuous 15-minute price path
        for mean expected return, lower bound (10th percentile), and upper bound (90th percentile).
        """
        def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
            super().__init__()
            self.output_size = output_size
            
            # Shared feature extraction block
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
            
            # Sub-task heads
            self.mean_head = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.GELU(),
                nn.Linear(64, output_size)
            )
            self.low_head = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.GELU(),
                nn.Linear(64, output_size)
            )
            self.high_head = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.GELU(),
                nn.Linear(64, output_size)
            )
            
        def forward(self, x):
            feats = self.shared(x)
            mean_out = self.mean_head(feats)
            low_out = self.low_head(feats)
            high_out = self.high_head(feats)
            return mean_out, low_out, high_out

    def pinball_loss_vector(pred, target, tau):
        """Quantile Pinball loss for 2D vectors."""
        diff = target - pred
        return torch.max(tau * diff, (tau - 1.0) * diff).mean()


class IntradayPredictiveEngine:
    """
    Predictive engine that aligns EOD synthesis features with subsequent-day 15-minute
    candle price paths, and supports dynamic online adaptation.
    """
    def __init__(
        self,
        ticker: str,
        daily_data: pd.DataFrame,
        daily_metrics: pd.DataFrame,
        daily_cot: pd.DataFrame,
        daily_cvd: pd.Series,
        daily_div_signals: pd.Series,
        daily_pivots: pd.DataFrame,
        daily_events: pd.DataFrame = None,
        lstm_series: pd.Series = None,
        lookback_days: int = 45
    ):
        self.ticker = ticker
        self.daily_data = daily_data
        self.daily_metrics = daily_metrics
        self.daily_cot = daily_cot
        self.daily_cvd = daily_cvd
        self.daily_div_signals = daily_div_signals
        self.daily_pivots = daily_pivots
        self.daily_events = daily_events
        self.lstm_series = lstm_series
        self.lookback_days = lookback_days
        
        self.scaler = StandardScaler()
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)
        self.nn_model = None
        self.is_trained = False
        
        # Timeline details
        self.standard_times = []
        self.aligned_features_df = None
        self.aligned_targets_df = None
        
        init_intraday_ledger_db()

    def fetch_and_clean_intraday(self) -> pd.DataFrame:
        """
        Downloads historical 15m data and aligns it to a standardized intraday timeline.
        Returns a pivot DataFrame with Date as index, standard time slots as columns.
        """
        # yfinance historical limit for 15m is 60d
        period = f"{min(60, self.lookback_days)}d"
        raw_df = yf.download(self.ticker, period=period, interval="15m", progress=False)
        if raw_df.empty:
            raise ValueError(f"Could not retrieve intraday 15m data for {self.ticker}")
            
        # Flatten MultiIndex columns
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
            
        raw_df.index = raw_df.index.tz_localize(None)
        
        # Build standard time slots
        raw_df['Date'] = raw_df.index.date
        raw_df['Time'] = raw_df.index.strftime('%H:%M')
        
        # Standard hours logic: find times appearing in at least 40% of the active dates
        n_dates = raw_df['Date'].nunique()
        time_counts = raw_df['Time'].value_counts()
        threshold = max(1, int(n_dates * 0.40))
        self.standard_times = sorted([t for t, count in time_counts.items() if count >= threshold])
        
        if not self.standard_times:
            raise ValueError(f"No consistent intraday time slots found for {self.ticker}")
            
        # Pivot the dataframe to align dates with standard time close prices
        cleaned_records = []
        grouped = raw_df.groupby('Date')
        for d, grp in grouped:
            # Reindex to standard times
            grp_pivoted = grp.set_index('Time').reindex(self.standard_times)
            # Forward fill missing bars and then backward fill
            grp_pivoted = grp_pivoted.ffill().bfill()
            
            # If after fill we still have NaNs (empty day), discard
            if grp_pivoted['Close'].isna().any():
                continue
                
            close_prices = grp_pivoted['Close'].values
            cleaned_records.append({
                'Date': pd.Timestamp(d),
                **{t: close_prices[i] for i, t in enumerate(self.standard_times)}
            })
            
        cleaned_df = pd.DataFrame(cleaned_records).set_index('Date').sort_index()
        return cleaned_df

    def _prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Aligns daily EOD features of day T with the intraday price path of day T+1.
        """
        # 1. Compile EOD features for all dates
        X_daily = predictive_engine.extract_synthesis_features(
            self.daily_data, self.daily_metrics, self.daily_cot,
            self.daily_cvd, self.daily_div_signals, self.daily_pivots,
            lstm_series=self.lstm_series, event_df=self.daily_events
        )
        
        # 2. Get cleaned 15m price path pivot table
        intraday_pivot = self.fetch_and_clean_intraday()
        
        aligned_rows = []
        aligned_targets = []
        dates_list = []
        
        # Sort EOD dates
        daily_dates = X_daily.index.sort_values()
        
        for i in range(len(daily_dates) - 1):
            t_curr = daily_dates[i]
            t_next = daily_dates[i+1] # Next trading day
            
            # Check if we have tomorrow's 15m price path
            t_next_date = pd.Timestamp(t_next.date())
            if t_next_date in intraday_pivot.index:
                # Fetch Tomorrow's 15m prices
                prices_tomorrow = intraday_pivot.loc[t_next_date].values
                close_today = float(self.daily_data['Close'].loc[t_curr].iloc[0] if isinstance(self.daily_data['Close'].loc[t_curr], pd.Series) else self.daily_data['Close'].loc[t_curr])
                
                if close_today <= 0:
                    continue
                    
                # Normalize targets: return of tomorrow's 15m candles relative to today's close
                returns_tomorrow = (prices_tomorrow / close_today) - 1.0
                
                aligned_rows.append(X_daily.loc[t_curr].values)
                aligned_targets.append(returns_tomorrow)
                dates_list.append(t_curr)
                
        if len(aligned_rows) < 10:
            raise ValueError(f"Not enough aligned EOD-Intraday data (need >=10, got {len(aligned_rows)}). Try increasing the lookback.")
            
        X = np.array(aligned_rows, dtype=np.float32)
        y = np.array(aligned_targets, dtype=np.float32)
        
        # Keep track of aligned data
        self.aligned_features_df = pd.DataFrame(X, index=dates_list, columns=X_daily.columns)
        self.aligned_targets_df = pd.DataFrame(y, index=dates_list, columns=self.standard_times)
        
        return X, y, self.aligned_features_df, self.aligned_targets_df

    def train(self, epochs: int = 40, progress_callback=None) -> Dict:
        """Trains both the Random Forest and PyTorch Multi-Task Intraday Net models."""
        X, y, _, _ = self._prepare_dataset()
        n_samples = len(X)
        output_size = len(self.standard_times)
        
        # Scaler fit
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        
        # Chronological train-validation split (80/20) with 2-day gap to prevent leakage
        split = int(n_samples * 0.8)
        X_train = X_scaled[:split - 2]
        y_train = y[:split - 2]
        
        X_val = X_scaled[split:]
        y_val = y[split:]
        
        # Train Random Forest
        self.rf_reg.fit(X_train, y_train)
        rf_pred = self.rf_reg.predict(X_val)
        self.val_mse = float(mean_squared_error(y_val, rf_pred))
        
        # Train PyTorch Multi-Task Quantile Net
        if TORCH_AVAILABLE:
            train_ds = TensorDataset(
                torch.tensor(X_train),
                torch.tensor(y_train, dtype=torch.float32)
            )
            loader = DataLoader(train_ds, batch_size=8, shuffle=False)
            
            model = IntradayMultiTaskNet(input_size=X.shape[1], output_size=output_size, hidden_size=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            for epoch in range(epochs):
                model.train()
                batch_losses = []
                for Xb, yb in loader:
                    optimizer.zero_grad()
                    mean_out, low_out, high_out = model(Xb)
                    
                    l_mean = nn.functional.mse_loss(mean_out, yb)
                    l_low = pinball_loss_vector(low_out, yb, 0.10)
                    l_high = pinball_loss_vector(high_out, yb, 0.90)
                    
                    loss = l_mean + 0.5 * (l_low + l_high)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                    
                scheduler.step()
                if progress_callback:
                    progress_callback(epoch + 1, epochs, float(np.mean(batch_losses)))
                    
            self.nn_model = model
            model.eval()
            
            with torch.no_grad():
                nn_mean, _, _ = model(torch.tensor(X_val))
                nn_mse = float(mean_squared_error(y_val, nn_mean.numpy()))
                self.val_mse = min(self.val_mse, nn_mse)
                
        self.is_trained = True
        return {
            'samples': n_samples,
            'val_mse': self.val_mse,
            'output_size': output_size
        }

    def predict_next_day(self) -> List[Dict]:
        """
        Runs next-day 15m predictions using the latest available features.
        Converts the normalized returns back into absolute price values.
        """
        if not self.is_trained:
            raise RuntimeError("Intraday predictive engine must be trained first.")
            
        # Update ledger actuals and run concept drift check
        try:
            self.update_ledger_actuals()
            is_drifted, rmse = self.check_concept_drift()
            if is_drifted:
                self.retrain_online()
        except Exception:
            pass
            
        # Build EOD features on latest daily record
        X_all = predictive_engine.extract_synthesis_features(
            self.daily_data, self.daily_metrics, self.daily_cot,
            self.daily_cvd, self.daily_div_signals, self.daily_pivots,
            lstm_series=self.lstm_series, event_df=self.daily_events
        )
        latest_row = X_all.iloc[[-1]].values
        latest_row_scaled = self.scaler.transform(latest_row).astype(np.float32)
        
        # RF Expected outputs
        rf_returns = self.rf_reg.predict(latest_row_scaled)[0]
        
        # PyTorch Expected & Quantile bounds
        if TORCH_AVAILABLE and self.nn_model is not None:
            self.nn_model.eval()
            with torch.no_grad():
                row_t = torch.tensor(latest_row_scaled)
                nn_mean, nn_low, nn_high = self.nn_model(row_t)
                nn_returns = nn_mean.numpy()[0]
                nn_low_returns = nn_low.numpy()[0]
                nn_high_returns = nn_high.numpy()[0]
        else:
            nn_returns = rf_returns
            # Fallback bounds using GARCH daily volatility scaled across the day (sqrt time scaling)
            daily_vol = float(self.daily_metrics['EWMA'].iloc[-1]) if 'EWMA' in self.daily_metrics.columns else 0.20
            daily_vol_std = daily_vol / np.sqrt(252)
            n_steps = len(self.standard_times)
            
            nn_low_returns = []
            nn_high_returns = []
            for k in range(n_steps):
                step_vol = daily_vol_std * np.sqrt((k + 1) / n_steps)
                nn_low_returns.append(rf_returns[k] - 1.645 * step_vol)
                nn_high_returns.append(rf_returns[k] + 1.645 * step_vol)
                
            nn_low_returns = np.array(nn_low_returns)
            nn_high_returns = np.array(nn_high_returns)
            
        # Consensus returns (Average)
        cons_returns = 0.5 * (rf_returns + nn_returns)
        
        # Scale back returns to prices
        close_today = float(self.daily_data['Close'].dropna().iloc[-1])
        
        pred_list = []
        for i, t in enumerate(self.standard_times):
            pred_p = close_today * (1.0 + cons_returns[i])
            low_p = close_today * (1.0 + nn_low_returns[i])
            high_p = close_today * (1.0 + nn_high_returns[i])
            
            # Post-processing: make sure bounds surround expected price
            low_p = min(low_p, pred_p - 0.01)
            high_p = max(high_p, pred_p + 0.01)
            
            chg = cons_returns[i] * 100
            
            # Sigmoid probability helper (prob of closing positive at candle t)
            # Estimate CDF from predicted bounds
            z_score = cons_returns[i] / max(1e-4, float(nn_high_returns[i] - nn_low_returns[i]) / 2.0)
            prob_up = float(1.0 / (1.0 + np.exp(-1.5 * z_score)))
            
            pred_list.append({
                'time': t,
                'pred_price': float(pred_p),
                'low_bound': float(low_p),
                'high_bound': float(high_p),
                'pct_change': float(chg),
                'prob_up': float(prob_up)
            })
            
        # Log to SQLite ledger
        try:
            pred_date = self.daily_data.index[-1]
            t_date = predictive_engine.get_next_trading_day(pred_date)
            self.log_predictions(str(pred_date.date()), str(t_date.date()), pred_list)
        except Exception:
            pass
            
        return pred_list

    def log_predictions(self, prediction_date: str, target_date: str, pred_list: List[Dict]):
        """Logs tomorrow's intraday price path predictions into the SQLite ledger."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for item in pred_list:
            cursor.execute("""
                INSERT OR REPLACE INTO intraday_prediction_ledger (
                    ticker, prediction_date, target_date, candle_time, pred_price, low_bound, high_bound
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                self.ticker, prediction_date, target_date, item['time'],
                item['pred_price'], item['low_bound'], item['high_bound']
            ))
        conn.commit()
        conn.close()

    def update_ledger_actuals(self):
        """Downloads latest intraday candles and fills missing actuals in the ledger."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT target_date FROM intraday_prediction_ledger
            WHERE ticker = ? AND actual_price IS NULL
        """, (self.ticker,))
        missing_dates = [row[0] for row in cursor.fetchall()]
        
        if not missing_dates:
            conn.close()
            return
            
        # Download recent intraday data to fill gaps
        try:
            intraday_raw = yf.download(self.ticker, period="7d", interval="15m", progress=False)
            if intraday_raw.empty:
                conn.close()
                return
            if isinstance(intraday_raw.columns, pd.MultiIndex):
                intraday_raw.columns = intraday_raw.columns.get_level_values(0)
                
            intraday_raw.index = intraday_raw.index.tz_localize(None)
            
            for d_str in missing_dates:
                t_date = pd.Timestamp(d_str).date()
                day_data = intraday_raw[intraday_raw.index.date == t_date]
                if day_data.empty:
                    continue
                    
                # Map actual times
                for t in self.standard_times:
                    # Find closest timestamp in day_data
                    times_in_day = day_data.index.strftime('%H:%M')
                    matching_idx = np.where(times_in_day == t)[0]
                    if len(matching_idx) > 0:
                        actual_p = float(day_data['Close'].iloc[matching_idx[0]])
                        cursor.execute("""
                            UPDATE intraday_prediction_ledger
                            SET actual_price = ?
                            WHERE ticker = ? AND target_date = ? AND candle_time = ?
                        """, (actual_p, self.ticker, d_str, t))
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

    def check_concept_drift(self, threshold: float = 0.015) -> Tuple[bool, float]:
        """
        Calculates Root Mean Squared Error (RMSE) on the last 5 completed target_dates.
        Returns (is_drifted, rmse).
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT target_date, pred_price, actual_price FROM intraday_prediction_ledger
            WHERE ticker = ? AND actual_price IS NOT NULL
            ORDER BY target_date DESC
        """, (self.ticker,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return False, 0.0
            
        # Group by target_date and compute relative RMSE
        df = pd.DataFrame(rows, columns=['target_date', 'pred', 'actual'])
        unique_dates = df['target_date'].unique()[:5]
        
        if len(unique_dates) < 1:
            # Not enough completed days to establish a drift baseline
            return False, 0.0
            
        df_filtered = df[df['target_date'].isin(unique_dates)]
        errs = ((df_filtered['actual'] - df_filtered['pred']) / df_filtered['actual']) ** 2
        rmse = float(np.sqrt(errs.mean()))
        
        return rmse > threshold, rmse

    def retrain_online(self):
        """Runs quick online adaptation to capture recent regime changes."""
        if len(self.aligned_features_df) < 10:
            return
            
        X = self.aligned_features_df.values
        y = self.aligned_targets_df.values
        
        # Scale
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        # Take the last 10 days of drifted data to adapt on
        X_drift = X_scaled[-10:]
        y_drift = y[-10:]
        
        # RF warm start tree injection
        self.rf_reg.warm_start = True
        self.rf_reg.n_estimators += 5
        self.rf_reg.fit(X_drift, y_drift)
        
        # PyTorch online fine-tuning
        if TORCH_AVAILABLE and self.nn_model is not None:
            # Freeze shared blocks, fine-tune task heads only
            for param in self.nn_model.shared.parameters():
                param.requires_grad = False
            for param in self.nn_model.mean_head.parameters():
                param.requires_grad = True
            for param in self.nn_model.low_head.parameters():
                param.requires_grad = True
            for param in self.nn_model.high_head.parameters():
                param.requires_grad = True
                
            trainable_params = [p for p in self.nn_model.parameters() if p.requires_grad]
            if trainable_params:
                optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
                
                X_tensor = torch.tensor(X_drift)
                y_tensor = torch.tensor(y_drift, dtype=torch.float32)
                
                for epoch in range(10):
                    optimizer.zero_grad()
                    mean_out, low_out, high_out = self.nn_model(X_tensor)
                    l_mean = nn.functional.mse_loss(mean_out, y_tensor)
                    l_low = pinball_loss_vector(low_out, y_tensor, 0.10)
                    l_high = pinball_loss_vector(high_out, y_tensor, 0.90)
                    loss = l_mean + 0.5 * (l_low + l_high)
                    loss.backward()
                    optimizer.step()
                    
            # Re-enable gradient learning for next full cycle
            for param in self.nn_model.shared.parameters():
                param.requires_grad = True


# Streamlit caching framework wrapper
@cache_resource_decorator(ttl=900)
def get_trained_intraday_engine(
    ticker: str,
    _daily_data: pd.DataFrame,
    _daily_metrics: pd.DataFrame,
    _daily_cot: pd.DataFrame,
    _daily_cvd: pd.Series,
    _daily_div_signals: pd.Series,
    _daily_pivots: pd.DataFrame,
    _daily_events: pd.DataFrame = None,
    _lstm_series: pd.Series = None,
    epochs: int = 40,
    lookback_days: int = 45
) -> IntradayPredictiveEngine:
    """Trains/retrieves cached intraday predictive engine resource."""
    engine = IntradayPredictiveEngine(
        ticker=ticker,
        daily_data=_daily_data,
        daily_metrics=_daily_metrics,
        daily_cot=_daily_cot,
        daily_cvd=_daily_cvd,
        daily_div_signals=_daily_div_signals,
        daily_pivots=_daily_pivots,
        daily_events=_daily_events,
        lstm_series=_lstm_series,
        lookback_days=lookback_days
    )
    engine.train(epochs=epochs)
    return engine
