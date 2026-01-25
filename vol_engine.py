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


def calculate_metrics(prices: pd.Series) -> pd.DataFrame:
    """
    Calculate all volatility metrics from price series.
    
    This is the main function combining all calculations.
    
    Args:
        prices: Series of asset closing prices (or DataFrame with single column)
        
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
    
    result = pd.DataFrame({
        'Log_Ret': log_ret,
        'EWMA': ewma_vol
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
