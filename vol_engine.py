"""
vol_engine.py - Volatility Calculation Engine

This module provides functions for computing various volatility metrics:
- Historical (Rolling) Volatility
- EWMA Volatility
- GARCH(1,1) Forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Log returns: ln(Pₜ / Pₜ₋₁)
    
    Args:
        prices: Series of asset prices
        
    Returns:
        Series of log returns
    """
    return np.log(prices / prices.shift(1))


def calculate_rolling_vol(
    returns: pd.Series, 
    windows: list = [20, 60, 120]
) -> pd.DataFrame:
    """
    Calculate rolling (historical) volatility for multiple windows.
    
    Formula: σ = std(returns) × √252 (annualized)
    
    Args:
        returns: Series of log returns
        windows: List of rolling window sizes in days
        
    Returns:
        DataFrame with volatility for each window
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0] if returns.shape[1] == 1 else returns.squeeze()
    
    result = {}
    for window in windows:
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        result[f'Vol_{window}d'] = vol
    
    return pd.DataFrame(result, index=returns.index)


def calculate_ewma_vol(
    returns: pd.Series, 
    span: int = 20,
    lambda_decay: float = None
) -> pd.Series:
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.
    
    When span=20, this roughly corresponds to λ=0.94 (RiskMetrics standard)
    λ = 1 - (2 / (span + 1))
    
    Args:
        returns: Series of log returns
        span: EWMA span parameter
        lambda_decay: Optional explicit decay factor (overrides span)
        
    Returns:
        Series of EWMA volatility (annualized)
    """
    if lambda_decay is not None:
        # Convert lambda to span: span = (2 / (1 - λ)) - 1
        span = int((2 / (1 - lambda_decay)) - 1)
    
    ewma_vol = returns.ewm(span=span).std() * np.sqrt(252)
    return ewma_vol


def calculate_sma_vol(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:
    r"""
    Calculate SMA (Simple Moving Average) volatility (zero-mean assumption).
    
    Formula: σ²_t = (1 / N) * \sum_{i=0}^{N-1} r²_{t-i}
    Annualized: σ_t = sqrt(σ²_t) * sqrt(252)
    
    Args:
        returns: Series of log returns
        window: SMA rolling window size in days
        
    Returns:
        Series of SMA volatility (annualized)
    """
    # Ensure returns is a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0] if returns.shape[1] == 1 else returns.squeeze()
    
    squared_returns = returns ** 2
    sma_variance = squared_returns.rolling(window=window).mean()
    sma_vol = np.sqrt(sma_variance) * np.sqrt(252)
    return sma_vol


def calculate_metrics(prices: pd.Series, sma_window: int = 20) -> pd.DataFrame:
    """
    Calculate all volatility metrics from price series.
    
    This is the main function combining all calculations.
    
    Args:
        prices: Series of asset closing prices (or DataFrame with single column)
        sma_window: Rolling window size for SMA volatility (default 20)
        
    Returns:
        DataFrame with Log Returns and all volatility metrics
    """
    # Handle DataFrame input (squeeze to Series if single column)
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            prices = prices.squeeze()
    
    # Ensure we have a Series
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    
    log_ret = calculate_log_returns(prices)
    rolling_vol = calculate_rolling_vol(log_ret, windows=[20, 60, 120])
    ewma_vol = calculate_ewma_vol(log_ret, span=20)
    sma_vol = calculate_sma_vol(log_ret, window=sma_window)
    
    result = pd.DataFrame({
        'Log_Ret': log_ret,
        'EWMA': ewma_vol,
        f'SMA_{sma_window}d': sma_vol
    }, index=prices.index)
    result = pd.concat([result, rolling_vol], axis=1)
    
    return result


def fit_garch(
    returns: pd.Series, 
    p: int = 1, 
    q: int = 1
) -> Optional[Dict]:
    """
    Fit GARCH(p,q) model to returns and extract parameters.
    
    GARCH(1,1) model:
    σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
    
    Parameters:
    - ω (omega): Long-term average variance weight
    - α (alpha): Reaction to recent market shock
    - β (beta): Persistence of volatility
    
    Args:
        returns: Series of log returns (dropna applied)
        p: GARCH lag order
        q: ARCH lag order
        
    Returns:
        Dict with fitted parameters or None if fitting fails
    """
    try:
        from arch import arch_model
        
        # Clean returns
        clean_returns = returns.dropna() * 100  # Scale for numerical stability
        
        # Fit GARCH model
        model = arch_model(clean_returns, vol='Garch', p=p, q=q)
        result = model.fit(disp='off')
        
        # Extract parameters
        params = {
            'omega': result.params.get('omega', None),
            'alpha': result.params.get('alpha[1]', None),
            'beta': result.params.get('beta[1]', None),
            'persistence': None,
            'long_term_vol': None,
            'forecast_1d': None,
            'aic': result.aic,
            'bic': result.bic
        }
        
        # Calculate persistence (α + β)
        if params['alpha'] and params['beta']:
            params['persistence'] = params['alpha'] + params['beta']
            
            # Long-term volatility = sqrt(ω / (1 - α - β))
            if params['persistence'] < 1:
                ltv = np.sqrt(params['omega'] / (1 - params['persistence']))
                params['long_term_vol'] = ltv / 100 * np.sqrt(252)  # Annualize
        
        # 1-day ahead forecast
        forecast = result.forecast(horizon=1)
        if forecast.variance is not None and len(forecast.variance) > 0:
            params['forecast_1d'] = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100 * np.sqrt(252)
        
        return params
        
    except ImportError:
        print("Warning: 'arch' package not installed. GARCH fitting unavailable.")
        return None
    except Exception as e:
        print(f"GARCH fitting failed: {e}")
        return None


def fit_egarch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1
) -> Optional[Dict]:
    """
    Fit EGARCH(p,q) model to returns and extract parameters including the leverage effect.

    EGARCH(1,1) model (Nelson 1991):
        log(σ²ₜ) = ω + α·[|εₜ₋₁|/σₜ₋₁ - E|z|] + γ·(εₜ₋₁/σₜ₋₁) + β·log(σ²ₜ₋₁)

    Key difference vs GARCH:
    - α (alpha): magnitude of shock (symmetric, always positive)
    - γ (gamma): LEVERAGE EFFECT — direction of shock
        γ < 0  → bearish shocks amplify volatility MORE than bullish shocks of equal size
        γ = 0  → symmetric (reduces to GARCH-like behaviour)
        γ > 0  → bullish shocks dominate (rare)
    - β (beta): log-variance persistence
    - Because the model is in log space, σ² stays positive without constraints.

    Args:
        returns: Series of log returns (dropna applied)
        p: EGARCH lag order (ARCH component)
        q: GARCH lag order (persistence component)

    Returns:
        Dict with fitted parameters or None if fitting fails.
        Extra keys vs fit_garch():
          'gamma'            : leverage parameter γ
          'leverage_dir'     : 'Bearish amplification' / 'Bullish amplification' / 'Symmetric'
          'leverage_pct'     : how much MORE a −1σ shock raises vol vs a +1σ shock (%)
    """
    try:
        from arch import arch_model

        clean_returns = returns.dropna() * 100   # scale for numerical stability

        model = arch_model(clean_returns, vol='EGARCH', p=p, q=q)
        result = model.fit(disp='off')

        alpha = result.params.get('alpha[1]', None)
        gamma = result.params.get('gamma[1]', None)
        beta  = result.params.get('beta[1]',  None)
        omega = result.params.get('omega',     None)

        params = {
            'omega':        omega,
            'alpha':        alpha,
            'gamma':        gamma,
            'beta':         beta,
            'persistence':  abs(beta) if beta is not None else None,
            'long_term_vol': None,
            'forecast_1d':  None,
            'aic': result.aic,
            'bic': result.bic,
            # Leverage interpretation
            'leverage_dir': None,
            'leverage_pct': None,
        }

        # Leverage direction & magnitude
        if gamma is not None:
            if gamma < -0.01:
                params['leverage_dir'] = 'Bearish amplification'
            elif gamma > 0.01:
                params['leverage_dir'] = 'Bullish amplification'
            else:
                params['leverage_dir'] = 'Symmetric'

            # % vol premium for a negative shock vs positive shock of same size
            # log(σ²) changes by (α + γ) for z<0 vs (α - γ) for z>0  (using |z|=1)
            neg_shock_effect = abs(alpha or 0) + gamma
            pos_shock_effect = abs(alpha or 0) - gamma
            if pos_shock_effect != 0:
                params['leverage_pct'] = (neg_shock_effect / pos_shock_effect - 1) * 100

        # 1-day ahead forecast
        try:
            forecast = result.forecast(horizon=1)
            if forecast.variance is not None and len(forecast.variance) > 0:
                params['forecast_1d'] = (
                    np.sqrt(forecast.variance.iloc[-1].values[0]) / 100 * np.sqrt(252)
                )
        except Exception:
            pass

        return params

    except ImportError:
        print("Warning: 'arch' package not installed. EGARCH fitting unavailable.")
        return None
    except Exception as e:
        print(f"EGARCH fitting failed: {e}")
        return None


def get_regime_comparison(vol_series: pd.Series) -> Dict:
    """
    Compare current volatility to historical regime stats.
    
    Args:
        vol_series: Series of volatility values
        
    Returns:
        Dict with regime statistics
    """
    clean_vol = vol_series.dropna()
    
    if len(clean_vol) < 252:
        lookback = len(clean_vol)
    else:
        lookback = 252  # 52-week lookback
    
    recent_vol = clean_vol.tail(lookback)
    
    return {
        'current': clean_vol.iloc[-1] if len(clean_vol) > 0 else None,
        'avg_52w': recent_vol.mean(),
        'min_52w': recent_vol.min(),
        'max_52w': recent_vol.max(),
        'percentile': (recent_vol < clean_vol.iloc[-1]).mean() * 100 if len(clean_vol) > 0 else None,
        'std_52w': recent_vol.std()
    }


def classify_regime(current_vol: float, stats: Dict) -> str:
    """
    Classify current volatility regime using z-score thresholds.
    
    Quant-grade classification:
    - Compression: z < -1.0 (vol significantly below mean)
    - Low: -1.0 <= z < -0.5
    - Neutral: -0.5 <= z < 0.5
    - Elevated: 0.5 <= z < 1.0
    - Expansion: z >= 1.0 (vol significantly above mean)
    
    Args:
        current_vol: Current volatility value
        stats: Dict from get_regime_comparison
        
    Returns:
        Regime classification string (no emoji - institutional style)
    """
    if current_vol is None or stats['avg_52w'] is None or stats['std_52w'] is None:
        return "--"
    
    if stats['std_52w'] == 0:
        return "Neutral"
    
    # Calculate z-score
    z_score = (current_vol - stats['avg_52w']) / stats['std_52w']
    
    if z_score >= 1.5:
        return "EXPANSION"  # High stress
    elif z_score >= 0.5:
        return "Elevated"
    elif z_score >= -0.5:
        return "Neutral"
    elif z_score >= -1.0:
        return "Low"
    else:
        return "Compression"


def get_zscore(current_vol: float, stats: Dict) -> float:
    """
    Calculate z-score for current volatility.
    
    Args:
        current_vol: Current volatility value
        stats: Dict from get_regime_comparison
        
    Returns:
        Z-score value
    """
    if current_vol is None or stats['avg_52w'] is None or stats['std_52w'] is None:
        return 0.0
    if stats['std_52w'] == 0:
        return 0.0
    return (current_vol - stats['avg_52w']) / stats['std_52w']


def calculate_donchian_channels(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Calculate Donchian Channels (Upper, Lower, Middle bands).
    
    Formula:
    Upper Band = rolling max of High
    Lower Band = rolling min of Low
    Middle Band = (Upper + Lower) / 2
    
    Args:
        high: Series of asset high prices
        low: Series of asset low prices
        window: Donchian look-back period (default 20)
        
    Returns:
        DataFrame with Donchian_Upper, Donchian_Lower, Donchian_Middle columns
    """
    upper = high.rolling(window=window).max()
    lower = low.rolling(window=window).min()
    middle = (upper + lower) / 2
    
    return pd.DataFrame({
        'Donchian_Upper': upper,
        'Donchian_Lower': lower,
        'Donchian_Middle': middle
    }, index=high.index)


def calculate_volume_metrics(volume: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Calculate trading Volume SMA and Relative Volume (RVOL).
    
    Formula:
    Volume_SMA = rolling mean of Volume
    RVOL = Volume / Volume_SMA
    
    Args:
        volume: Series of asset volumes
        window: SMA window size (default 20)
        
    Returns:
        DataFrame with Volume_SMA and RVOL columns
    """
    volume_sma = volume.rolling(window=window).mean()
    # Avoid zero division
    safe_sma = volume_sma.replace(0, np.nan)
    rvol = volume / safe_sma
    
    return pd.DataFrame({
        'Volume_SMA': volume_sma,
        'RVOL': rvol
    }, index=volume.index)
