"""
lstm_vol_engine.py - GARCH → LSTM Neural Volatility Forecaster

Architecture:
  Classical econometrics (GARCH) extracts rigorous baseline features:
    - ω  (omega)  : long-run variance weight
    - α  (alpha)  : reaction to recent shocks
    - β  (beta)   : volatility persistence
    - σ_t series  : full conditional-variance path

  Those features feed an LSTM that learns:
    - Non-linear regime transitions that GARCH cannot capture
    - Multi-step ahead volatility forecasts (1 / 5 / 10 / 21 days)
    - Probabilistic regime classification (Compression / Neutral / Expansion)

Usage (standalone):
    from lstm_vol_engine import GARCHLSTMForecaster
    fc = GARCHLSTMForecaster(returns_series, garch_params)
    fc.train()
    result = fc.forecast()
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional torch import — graceful fallback so the rest of the app still runs
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ===========================================================================
# 1.  GARCH Feature Extractor
# ===========================================================================

def extract_garch_features(returns: pd.Series, garch_params: Dict) -> pd.DataFrame:
    """
    Build a feature matrix from GARCH-derived quantities.

    Columns produced
    ----------------
    sigma_t          : conditional daily std (annualised, from α/β recursion)
    log_sigma_t      : log of sigma_t  (stabilises LSTM gradients)
    alpha_x_eps2     : α × ε²_{t-1}   (shock contribution)
    beta_x_sigma2    : β × σ²_{t-1}   (persistence contribution)
    abs_return       : |r_t|           (raw shock magnitude)
    sq_return        : r²_t            (squared return)
    ewma_vol         : RiskMetrics EWMA with λ=0.94
    rolling_vol_5    : 5-day rolling realised vol (annualised)
    vol_of_vol       : 5-day rolling std of sigma_t  (vol clustering signal)

    Args
    ----
    returns     : log return series (float, not %-scaled)
    garch_params: output dict from vol_engine.fit_garch()

    Returns
    -------
    pd.DataFrame  (same index as `returns`, NaN rows at head dropped)
    """
    alpha = garch_params.get("alpha") or 0.10
    beta  = garch_params.get("beta")  or 0.85
    omega = garch_params.get("omega") or 1e-6

    r = returns.dropna().values * 100        # scale to %-returns for GARCH recursion
    n = len(r)

    # Recursive GARCH(1,1) conditional-variance path
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

    sigma_t_pct = np.sqrt(sigma2)            # daily std in %-units
    sigma_t_ann = sigma_t_pct / 100 * np.sqrt(252)   # annualised

    # Lagged squared returns
    r_sq    = r ** 2
    abs_r   = np.abs(r)

    # EWMA (λ=0.94)
    ewma_var = np.zeros(n)
    ewma_var[0] = sigma2[0]
    lam = 0.94
    for t in range(1, n):
        ewma_var[t] = lam * ewma_var[t - 1] + (1 - lam) * r[t - 1] ** 2
    ewma_vol_ann = np.sqrt(ewma_var) / 100 * np.sqrt(252)

    # 5-day rolling vol
    roll5 = pd.Series(r / 100).rolling(5).std() * np.sqrt(252)
    roll5 = roll5.bfill().values

    # Vol-of-vol: 5-day rolling std of annualised sigma_t
    vol_of_vol = pd.Series(sigma_t_ann).rolling(5).std().fillna(0).values

    idx = returns.dropna().index

    df = pd.DataFrame({
        "sigma_t":       sigma_t_ann,
        "log_sigma_t":   np.log(sigma_t_ann + 1e-8),
        "alpha_x_eps2":  alpha * r_sq / 1e4,          # keep scale ~ 1e-4
        "beta_x_sigma2": beta  * sigma2 / 1e4,
        "abs_return":    abs_r / 100,
        "sq_return":     r_sq  / 1e4,
        "ewma_vol":      ewma_vol_ann,
        "rolling_vol_5": roll5,
        "vol_of_vol":    vol_of_vol,
    }, index=idx)

    return df.dropna()


# ===========================================================================
# 2.  Sequence Builder
# ===========================================================================

def build_sequences(
    features: pd.DataFrame,
    target_col: str = "sigma_t",
    seq_len: int = 30,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct sliding-window input/output arrays for LSTM training.

    Args
    ----
    features   : Feature DataFrame from extract_garch_features()
    target_col : Column to predict
    seq_len    : Look-back window (timesteps per sample)
    horizon    : How many steps ahead to forecast

    Returns
    -------
    X : (n_samples, seq_len, n_features)
    y : (n_samples,)  — target at t + horizon
    """
    X_list, y_list = [], []
    values = features.values
    target_idx = features.columns.get_loc(target_col)
    n = len(values)

    for i in range(seq_len, n - horizon + 1):
        X_list.append(values[i - seq_len: i])
        y_list.append(values[i + horizon - 1][target_idx])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ===========================================================================
# 3.  LSTM Model Definition
# ===========================================================================

def _build_lstm_model(
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> "nn.Module":
    """Return a stacked LSTM → Linear regressor."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Run: pip install torch")

    class VolLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    return VolLSTM()


# ===========================================================================
# 4.  Min-Max Scaler (pure numpy — no sklearn dependency)
# ===========================================================================

class _MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "_MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = self.max_ - self.min_
        denom[denom == 0] = 1.0
        return (X - self.min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform_col(self, y: np.ndarray, col_idx: int) -> np.ndarray:
        """Inverse-transform a 1-D array that corresponds to a single feature column."""
        denom = self.max_[col_idx] - self.min_[col_idx] or 1.0
        return y * denom + self.min_[col_idx]


# ===========================================================================
# 5.  Main Forecaster Class
# ===========================================================================

class GARCHLSTMForecaster:
    """
    End-to-end GARCH → LSTM hybrid volatility forecaster.

    Parameters
    ----------
    returns      : pd.Series of log returns
    garch_params : dict from vol_engine.fit_garch()
    seq_len      : LSTM look-back window (default 30 trading days)
    hidden_size  : LSTM hidden units
    num_layers   : stacked LSTM depth
    epochs       : training epochs
    batch_size   : mini-batch size
    lr           : Adam learning rate
    device       : 'cpu' or 'cuda'
    """

    HORIZONS = [1, 5, 10, 21]   # 1d, 1w, 2w, 1m forecasts

    def __init__(
        self,
        returns: pd.Series,
        garch_params: Dict,
        seq_len: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Run: pip install torch")

        self.returns      = returns
        self.garch_params = garch_params
        self.seq_len      = seq_len
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.device       = torch.device(device)

        self._features: Optional[pd.DataFrame] = None
        self._scaler: Optional[_MinMaxScaler] = None
        self._model: Optional["nn.Module"] = None
        self._train_losses: List[float] = []
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # A.  Feature extraction
    # ------------------------------------------------------------------

    def _build_features(self) -> pd.DataFrame:
        feat = extract_garch_features(self.returns, self.garch_params)
        self._features = feat
        return feat

    # ------------------------------------------------------------------
    # B.  Train
    # ------------------------------------------------------------------

    def train(self, progress_callback=None) -> Dict:
        """
        Train the LSTM on 1-step ahead sigma_t prediction.

        Args
        ----
        progress_callback : optional callable(epoch, total, loss) — for Streamlit progress bars

        Returns
        -------
        dict with 'train_loss', 'val_loss', 'epochs_run'
        """
        feat = self._build_features()

        # Scale features — cast to float32 to match LSTM weight dtype
        scaler = _MinMaxScaler()
        feat_scaled = scaler.fit_transform(feat.values).astype(np.float32)
        self._scaler = scaler

        # Build sequences
        X, y = build_sequences(
            pd.DataFrame(feat_scaled, columns=feat.columns, index=feat.index),
            target_col="sigma_t",
            seq_len=self.seq_len,
            horizon=1,
        )

        # Train / val split (80/20 time-ordered)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_ds = TensorDataset(
            torch.tensor(X_train, device=self.device),
            torch.tensor(y_train, device=self.device),
        )
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        # Model
        model = _build_lstm_model(
            input_size=X.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)
        self._model = model

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn   = nn.HuberLoss(delta=0.5)   # robust to vol spikes

        best_val = float("inf")
        best_state = None
        patience_counter = 0
        patience = 15

        train_losses, val_losses = [], []
        model.train()

        for epoch in range(self.epochs):
            batch_losses = []
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            scheduler.step()
            train_loss = float(np.mean(batch_losses))
            train_losses.append(train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, device=self.device)
                yv = torch.tensor(y_val, device=self.device)
                val_pred = model(Xv)
                val_loss = loss_fn(val_pred, yv).item()
            val_losses.append(val_loss)
            model.train()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break   # early stopping

            if progress_callback:
                progress_callback(epoch + 1, self.epochs, train_loss)

        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)

        self._train_losses = train_losses
        self._is_trained = True

        return {
            "train_loss": train_losses[-1],
            "val_loss":   best_val,
            "epochs_run": len(train_losses),
        }

    # ------------------------------------------------------------------
    # C.  Multi-horizon forecast
    # ------------------------------------------------------------------

    def forecast(self) -> Dict:
        """
        Produce multi-horizon vol forecasts + regime probabilities.

        Returns
        -------
        dict with keys:
          'forecasts'       : {1: float, 5: float, 10: float, 21: float}  (annualised vol)
          'regime'          : str  — 'Compression' / 'Neutral' / 'Expansion'
          'regime_probs'    : dict {regime: probability}
          'current_sigma'   : float  — latest GARCH conditional vol (annualised)
          'lstm_vs_garch'   : float  — % diff: LSTM 1d forecast vs GARCH 1d
          'in_sample_sigma' : pd.Series — full historical LSTM-predicted sigma path
          'train_losses'    : list[float]
        """
        if not self._is_trained:
            raise RuntimeError("Call .train() before .forecast()")

        feat = self._features
        feat_scaled = self._scaler.transform(feat.values).astype(np.float32)
        sigma_col_idx = feat.columns.get_loc("sigma_t")

        model = self._model
        model.eval()

        # --- In-sample predictions (for chart) ---
        X_all, _ = build_sequences(
            pd.DataFrame(feat_scaled, columns=feat.columns, index=feat.index),
            target_col="sigma_t",
            seq_len=self.seq_len,
            horizon=1,
        )
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_all, device=self.device)).cpu().numpy()

        preds_ann = self._scaler.inverse_transform_col(preds_scaled, sigma_col_idx)
        pred_index = feat.index[self.seq_len:]
        in_sample_sigma = pd.Series(preds_ann, index=pred_index, name="LSTM_sigma")

        # --- Multi-horizon: iterative 1-step rollout ---
        last_window = feat_scaled[-self.seq_len:].copy()   # (seq_len, n_feat)
        forecasts = {}

        for h in self.HORIZONS:
            window = last_window.copy()
            for _ in range(h):
                x_t = torch.tensor(window[np.newaxis], device=self.device)
                with torch.no_grad():
                    y_hat_scaled = model(x_t).item()

                # Build a synthetic next row: replace sigma_t, shift window
                next_row = window[-1].copy()
                next_row[sigma_col_idx] = y_hat_scaled
                window = np.vstack([window[1:], next_row])

            vol_ann = self._scaler.inverse_transform_col(
                np.array([y_hat_scaled], dtype=np.float32), sigma_col_idx
            )[0]
            forecasts[h] = float(max(vol_ann, 0.0))

        # --- Current GARCH sigma ---
        current_sigma = float(feat["sigma_t"].iloc[-1])

        # --- Regime classification via probabilistic thresholds ---
        hist_sigma = feat["sigma_t"].dropna()
        p25, p75 = float(hist_sigma.quantile(0.25)), float(hist_sigma.quantile(0.75))
        f1 = forecasts[1]

        # Sigmoid-based soft probabilities relative to percentile bands
        def _sigmoid(x, mu, scale=20.0):
            return float(1 / (1 + np.exp(-scale * (x - mu))))

        p_expand   = _sigmoid(f1, p75)
        p_compress = _sigmoid(p25, f1)
        p_neutral  = max(0.0, 1.0 - p_expand - p_compress)
        total      = p_expand + p_compress + p_neutral or 1.0
        regime_probs = {
            "Expansion":   round(p_expand  / total, 4),
            "Neutral":     round(p_neutral / total, 4),
            "Compression": round(p_compress / total, 4),
        }
        regime = max(regime_probs, key=regime_probs.get)

        # LSTM vs GARCH delta
        garch_1d = self.garch_params.get("forecast_1d") or current_sigma
        lstm_vs_garch = (forecasts[1] / garch_1d - 1) * 100 if garch_1d else 0.0

        return {
            "forecasts":       forecasts,
            "regime":          regime,
            "regime_probs":    regime_probs,
            "current_sigma":   current_sigma,
            "lstm_vs_garch":   lstm_vs_garch,
            "in_sample_sigma": in_sample_sigma,
            "train_losses":    self._train_losses,
        }

    # ------------------------------------------------------------------
    # D.  Quick run helper (train + forecast in one call)
    # ------------------------------------------------------------------

    def run(self, progress_callback=None) -> Dict:
        """Convenience: train then forecast. Returns forecast dict + train metadata."""
        train_meta = self.train(progress_callback=progress_callback)
        result     = self.forecast()
        result["train_meta"] = train_meta
        return result


# ===========================================================================
# 6.  Public availability check
# ===========================================================================

def is_available() -> bool:
    """Returns True if PyTorch is installed and LSTM training is possible."""
    return TORCH_AVAILABLE
