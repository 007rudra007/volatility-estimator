"""
backtest_engine.py - Quantitative Backtesting and Evaluation Engine

Includes:
- Vectorized Volatility Band Breach & Hit Rate calculations.
- CVD Divergence Forward Return Analysis (evaluating statistical edge).
- Walk-forward AI Predictor validation framework.
- Strategy Simulator modeling trades with Stop Loss (SL) and Take Profit (TP),
  computing Sharpe, Sortino, Drawdown, and Equity Curves.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import predictive_engine

def validate_volatility_bands(
    close_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    vol_series: pd.Series,
    multiplier: float = 1.0
) -> Dict:
    """
    Computes daily volatility bands and checks the percentage of days where price
    remains within the bands (hit rate) and details of breaches.
    
    Bands: P_{t-1} * exp(± multiplier * daily_vol)
    where daily_vol = annual_vol / sqrt(252)
    """
    n = len(close_prices)
    if n < 2:
        return {'hit_rate': 1.0, 'breaches': [], 'total_breaches': 0}
        
    daily_vol = vol_series.shift(1) / np.sqrt(252)
    prev_close = close_prices.shift(1)
    
    upper_band = prev_close * np.exp(multiplier * daily_vol)
    lower_band = prev_close * np.exp(-multiplier * daily_vol)
    
    # Identify breaches
    upper_breach = high_prices > upper_band
    lower_breach = low_prices < lower_band
    
    breaches = []
    for t in close_prices.index[1:]:
        if upper_breach.loc[t]:
            breaches.append({
                'Date': t.strftime('%Y-%m-%d'),
                'Type': 'Upper Breach (Bullish Expansion)',
                'Close': float(close_prices.loc[t]),
                'High': float(high_prices.loc[t]),
                'Limit': float(upper_band.loc[t]),
                'Exceedance %': float(((high_prices.loc[t] / upper_band.loc[t]) - 1.0) * 100)
            })
        elif lower_breach.loc[t]:
            breaches.append({
                'Date': t.strftime('%Y-%m-%d'),
                'Type': 'Lower Breach (Bearish Breakdown)',
                'Close': float(close_prices.loc[t]),
                'Low': float(low_prices.loc[t]),
                'Limit': float(lower_band.loc[t]),
                'Exceedance %': float(((lower_band.loc[t] / low_prices.loc[t]) - 1.0) * 100)
            })
            
    total_days = n - 1
    total_breaches = len(breaches)
    hit_rate = float((total_days - total_breaches) / total_days) if total_days > 0 else 1.0
    
    return {
        'hit_rate': hit_rate,
        'breaches': breaches,
        'total_breaches': total_breaches,
        'total_days': total_days,
        'upper_band': upper_band.fillna(close_prices),
        'lower_band': lower_band.fillna(close_prices)
    }


def analyze_cvd_divergences(
    prices: pd.Series,
    cvd_signals: pd.Series,
    horizons: List[int] = [1, 3, 5, 10]
) -> Dict:
    """
    Computes statistical forward returns and win rates following
    CVD Bullish (+1) and Bearish (-1) Divergence signals.
    """
    bull_results = {h: [] for h in horizons}
    bear_results = {h: [] for h in horizons}
    
    n = len(prices)
    idx_positions = {prices.index[i]: i for i in range(n)}
    
    # Walk through signals
    for t, sig in cvd_signals.items():
        if sig == 0 or pd.isna(sig):
            continue
            
        curr_pos = idx_positions.get(t)
        if curr_pos is None:
            continue
            
        p_curr = float(prices.iloc[curr_pos])
        if p_curr <= 0:
            continue
            
        for h in horizons:
            if curr_pos + h < n:
                p_future = float(prices.iloc[curr_pos + h])
                fwd_ret = (p_future / p_curr) - 1.0
                if sig == 1:
                    bull_results[h].append(fwd_ret)
                elif sig == -1:
                    bear_results[h].append(fwd_ret)
                    
    # Compile stats
    def compile_stats(results_dict):
        stats = {}
        for h in horizons:
            rets = results_dict[h]
            if rets:
                mean_ret = float(np.mean(rets))
                win_rate = float(np.mean([1 if r > 0 else 0 for r in rets])) if results_dict == bull_results else float(np.mean([1 if r < 0 else 0 for r in rets]))
                stats[f'{h}d'] = {
                    'count': len(rets),
                    'mean_return': mean_ret,
                    'win_rate': win_rate
                }
            else:
                stats[f'{h}d'] = {'count': 0, 'mean_return': 0.0, 'win_rate': 0.0}
        return stats
        
    return {
        'bullish': compile_stats(bull_results),
        'bearish': compile_stats(bear_results)
    }


def run_walk_forward_predictions(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    cot_df: pd.DataFrame,
    daily_cvd: pd.Series,
    daily_div_signals: pd.Series,
    pivot_df: pd.DataFrame,
    retrain_freq: int = 15,
    initial_train_days: int = 60,
    epochs: int = 10
) -> pd.DataFrame:
    """
    Simulates walk-forward out-of-sample consensus predictions.
    Retrains the SynthesisPredictiveEngine every `retrain_freq` days
    to avoid lookahead bias, running inference on active daily slots in between.
    """
    n = len(data)
    if n < initial_train_days + 5:
        # Fallback to empty predictions if history is too short
        return pd.DataFrame(index=data.index, columns=['pred_trend', 'pred_high', 'pred_low'])
        
    pred_trends = []
    pred_highs = []
    pred_lows = []
    dates_list = []
    
    active_engine = None
    
    # Walk forward step-by-step
    for t_idx in range(initial_train_days, n):
        t_date = data.index[t_idx]
        
        # Retrain boundary trigger
        is_retrain_day = ((t_idx - initial_train_days) % retrain_freq == 0) or (active_engine is None)
        
        if is_retrain_day:
            # Slice historical data up to day t_idx
            hist_data = data.iloc[:t_idx]
            hist_metrics = metrics.iloc[:t_idx]
            hist_cot = cot_df.iloc[:t_idx] if cot_df is not None else None
            hist_cvd = daily_cvd.iloc[:t_idx] if daily_cvd is not None else None
            hist_div = daily_div_signals.iloc[:t_idx] if daily_div_signals is not None else None
            
            # Filter pivots in range
            hist_pivots = pivot_df[pivot_df.index < t_date] if pivot_df is not None and not pivot_df.empty else pd.DataFrame()
            
            try:
                active_engine = predictive_engine.SynthesisPredictiveEngine(
                    data=hist_data,
                    metrics=hist_metrics,
                    cot_df=hist_cot,
                    daily_cvd=hist_cvd,
                    daily_div_signals=hist_div,
                    pivot_df=hist_pivots,
                    ticker="BACKTEST_TICKER"
                )
                active_engine.train(epochs=epochs)
            except Exception:
                pass
                
        if active_engine is not None and active_engine.is_trained:
            # Predict daily step using current data slice
            try:
                # To predict day t_idx + 1 range, the engine requires the metrics/features of day t_idx
                slice_engine = predictive_engine.SynthesisPredictiveEngine(
                    data=data.iloc[:t_idx+1],
                    metrics=metrics.iloc[:t_idx+1],
                    cot_df=cot_df.iloc[:t_idx+1] if cot_df is not None else None,
                    daily_cvd=daily_cvd.iloc[:t_idx+1] if daily_cvd is not None else None,
                    daily_div_signals=daily_div_signals.iloc[:t_idx+1] if daily_div_signals is not None else None,
                    pivot_df=pivot_df[pivot_df.index <= t_date] if pivot_df is not None and not pivot_df.empty else pd.DataFrame(),
                    ticker="BACKTEST_TICKER"
                )
                slice_engine.is_trained = True
                slice_engine.scaler = active_engine.scaler
                slice_engine.rf_reg = active_engine.rf_reg
                slice_engine.rf_clf = active_engine.rf_clf
                slice_engine.nn_model = active_engine.nn_model
                slice_engine.hmm = active_engine.hmm
                slice_engine.q_nd = active_engine.q_nd
                slice_engine.q_w = active_engine.q_w
                slice_engine.daily_vol = active_engine.daily_vol.reindex(data.index[:t_idx+1]).ffill().bfill()
                
                preds = slice_engine.predict_latest()
                consensus = preds['consensus']
                
                pred_trends.append(consensus['trend'])
                pred_highs.append(consensus['nd_high'])
                pred_lows.append(consensus['nd_low'])
                dates_list.append(t_date)
            except Exception:
                pred_trends.append('Neutral Range')
                pred_highs.append(np.nan)
                pred_lows.append(np.nan)
                dates_list.append(t_date)
        else:
            pred_trends.append('Neutral Range')
            pred_highs.append(np.nan)
            pred_lows.append(np.nan)
            dates_list.append(t_date)
            
    df_out = pd.DataFrame({
        'pred_trend': pred_trends,
        'pred_high': pred_highs,
        'pred_low': pred_lows
    }, index=dates_list)
    return df_out


def run_strategy_backtest(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    cot_df: pd.DataFrame,
    daily_cvd: pd.Series,
    daily_div_signals: pd.Series,
    pivot_df: pd.DataFrame,
    run_ai_predictions: bool = True,
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 6.0,
    initial_train_days: int = 60,
    retrain_freq: int = 20
) -> Dict:
    """
    Simulates trades based on technical confirmations (CVD, COT) and AI predictions.
    Calculates portfolio metrics, Sharpe/Sortino ratios, and drawdowns.
    """
    close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
    n = len(close)
    
    # 1. Run Walk-Forward AI Predictions if enabled
    ai_predictions = None
    if run_ai_predictions:
        try:
            ai_predictions = run_walk_forward_predictions(
                data=data,
                metrics=metrics,
                cot_df=cot_df,
                daily_cvd=daily_cvd,
                daily_div_signals=daily_div_signals,
                pivot_df=pivot_df,
                retrain_freq=retrain_freq,
                initial_train_days=initial_train_days,
                epochs=5
            )
        except Exception:
            pass
            
    # 2. Backtesting iteration starting from initial days
    start_idx = initial_train_days
    
    equity = 1.0
    equity_curve = [1.0] * start_idx
    cash = 1.0
    position = 0 # 0 = Cash, 1 = Long, -1 = Short
    entry_price = 0.0
    entry_date = None
    
    trades = []
    
    for i in range(start_idx, n):
        t_date = data.index[i]
        p_close = float(close.iloc[i])
        
        # Pull Indicators
        cvd_sig = int(daily_div_signals.loc[t_date]) if t_date in daily_div_signals.index else 0
        cot_idx = float(cot_df['COT_Index'].loc[t_date]) if (cot_df is not None and t_date in cot_df.index) else 50.0
        
        # Pull AI signal
        ai_trend = 'Neutral Range'
        if ai_predictions is not None and t_date in ai_predictions.index:
            ai_trend = ai_predictions.loc[t_date, 'pred_trend']
            
        # Manage active trade
        if position == 1:
            # Long exit checks (Stop Loss or Take Profit)
            pct_return = (p_close / entry_price) - 1.0
            if pct_return <= -stop_loss_pct / 100.0:
                # Stop Loss Triggered
                equity = cash * (1.0 - stop_loss_pct / 100.0)
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': t_date.strftime('%Y-%m-%d'),
                    'Type': 'Long',
                    'Entry Price': entry_price,
                    'Exit Price': entry_price * (1.0 - stop_loss_pct / 100.0),
                    'Return %': -stop_loss_pct,
                    'Exit Reason': 'Stop Loss'
                })
                position = 0
            elif pct_return >= take_profit_pct / 100.0:
                # Take Profit Triggered
                equity = cash * (1.0 + take_profit_pct / 100.0)
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': t_date.strftime('%Y-%m-%d'),
                    'Type': 'Long',
                    'Entry Price': entry_price,
                    'Exit Price': entry_price * (1.0 + take_profit_pct / 100.0),
                    'Return %': take_profit_pct,
                    'Exit Reason': 'Take Profit'
                })
                position = 0
            elif cvd_sig == -1 or cot_idx > 80.0 or ai_trend == 'Bearish Breakdown':
                # Technical Exit
                equity = cash * (1.0 + pct_return)
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': t_date.strftime('%Y-%m-%d'),
                    'Type': 'Long',
                    'Entry Price': entry_price,
                    'Exit Price': p_close,
                    'Return %': pct_return * 100.0,
                    'Exit Reason': 'Technical Exit'
                })
                position = 0
            else:
                # Unchanged position
                equity = cash * (1.0 + pct_return)
                
        elif position == -1:
            # Short exit checks
            pct_return = 1.0 - (p_close / entry_price)
            if pct_return <= -stop_loss_pct / 100.0:
                # Stop Loss Triggered
                equity = cash * (1.0 - stop_loss_pct / 100.0)
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': t_date.strftime('%Y-%m-%d'),
                    'Type': 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': entry_price * (1.0 + stop_loss_pct / 100.0),
                    'Return %': -stop_loss_pct,
                    'Exit Reason': 'Stop Loss'
                })
                position = 0
            elif pct_return >= take_profit_pct / 100.0:
                # Take Profit Triggered
                equity = cash * (1.0 + take_profit_pct / 100.0)
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': t_date.strftime('%Y-%m-%d'),
                    'Type': 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': entry_price * (1.0 - take_profit_pct / 100.0),
                    'Return %': take_profit_pct,
                    'Exit Reason': 'Take Profit'
                })
                position = 0
            elif cvd_sig == 1 or cot_idx < 20.0 or ai_trend == 'Bullish Breakout':
                # Technical Exit
                equity = cash * (1.0 + pct_return)
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': t_date.strftime('%Y-%m-%d'),
                    'Type': 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': p_close,
                    'Return %': pct_return * 100.0,
                    'Exit Reason': 'Technical Exit'
                })
                position = 0
            else:
                equity = cash * (1.0 + pct_return)
                
        else:
            # Look for entry signals
            is_buy = (cvd_sig == 1) or (cot_idx < 25.0)
            is_sell = (cvd_sig == -1) or (cot_idx > 75.0)
            
            # Align with AI predictions if active
            if run_ai_predictions and ai_predictions is not None:
                if is_buy and ai_trend == 'Bullish Breakout':
                    position = 1
                    entry_price = p_close
                    entry_date = t_date
                    cash = equity
                elif is_sell and ai_trend == 'Bearish Breakdown':
                    position = -1
                    entry_price = p_close
                    entry_date = t_date
                    cash = equity
            else:
                # Technical indicators only
                if is_buy:
                    position = 1
                    entry_price = p_close
                    entry_date = t_date
                    cash = equity
                elif is_sell:
                    position = -1
                    entry_price = p_close
                    entry_date = t_date
                    cash = equity
                    
        equity_curve.append(equity)
        
    # Calculate performance metrics
    eq_series = pd.Series(equity_curve, index=data.index)
    daily_rets = eq_series.pct_change().fillna(0.0)
    
    cumulative_return = float(eq_series.iloc[-1] - 1.0)
    
    # Benchmark return (Buy & Hold)
    benchmark_return = float(close.iloc[-1] / close.iloc[start_idx] - 1.0)
    
    # Sharpe Ratio (annualized, 5% risk free rate)
    rf_daily = 0.05 / 252.0
    excess_rets = daily_rets - rf_daily
    sharpe = float(np.mean(excess_rets) / max(1e-6, np.std(excess_rets)) * np.sqrt(252)) if np.std(excess_rets) > 0 else 0.0
    
    # Sortino Ratio (downside deviation)
    downside_rets = excess_rets[excess_rets < 0]
    sortino = float(np.mean(excess_rets) / max(1e-6, np.std(downside_rets)) * np.sqrt(252)) if len(downside_rets) > 0 else 0.0
    
    # Drawdown profile
    rolling_max = eq_series.cummax()
    drawdowns = (eq_series - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())
    
    # Win Rate
    win_rate = 0.0
    if trades:
        win_rate = float(np.mean([1 if t['Return %'] > 0 else 0 for t in trades]))
        
    return {
        'cumulative_return': cumulative_return,
        'benchmark_return': benchmark_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades': trades,
        'equity_curve': eq_series,
        'drawdowns': drawdowns
    }
