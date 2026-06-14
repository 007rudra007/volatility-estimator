"""
verify_predictive_engine.py - Verification Script for ML/Deep Learning Synthesis Engine
"""

import numpy as np
import pandas as pd
import predictive_engine


def run_verification():
    print("Initializing Predictive Engine automated tests...")
    
    # 1. Create a synthetic history containing price, volume, and volatility metrics
    # Needs at least 100 samples to support train/validation split
    n = 150
    dates = pd.date_range(start="2026-01-01", periods=n, freq="D")
    
    # Pre-populate dummy price series representing a noisy cyclical pattern
    t_vals = np.linspace(0, 10, n)
    prices = 100.0 + 15.0 * np.sin(t_vals) + np.random.randn(n) * 2.0
    
    df = pd.DataFrame({
        'Open': prices * 0.995,
        'High': prices * 1.015,
        'Low': prices * 0.985,
        'Close': prices,
        'Volume': np.random.randint(5000, 20000, n),
        'Donchian_Upper': prices * 1.03,
        'Donchian_Lower': prices * 0.97,
        'Donchian_Middle': prices,
        'RVOL': np.random.uniform(0.5, 2.0, n)
    }, index=dates)
    
    # Fake Volatility Metrics
    metrics = pd.DataFrame({
        'Log_Ret': np.log(df['Close'] / df['Close'].shift(1)).fillna(0),
        'Vol_20d': np.random.uniform(0.15, 0.35, n),
        'EWMA': np.random.uniform(0.15, 0.35, n),
        'SMA_20d': np.random.uniform(0.15, 0.35, n)
    }, index=dates)
    
    # Fake COT positioning
    cot_df = pd.DataFrame({
        'Speculator_Net': np.random.randint(-10000, 10000, n),
        'Commercial_Net': np.random.randint(-10000, 10000, n),
        'COT_Index': np.random.uniform(10.0, 90.0, n)
    }, index=dates)
    
    # Fake CVD
    daily_cvd = pd.Series(np.cumsum(np.random.randn(n) * 1000), index=dates)
    daily_div_signals = pd.Series(np.random.choice([-1, 0, 1], n, p=[0.1, 0.8, 0.1]), index=dates)
    
    # Fake Pivots
    pivot_df = pd.DataFrame([
        {'price': 90.0, 'type': 'Trough'},
        {'price': 112.0, 'type': 'Peak'},
        {'price': 98.0, 'type': 'Trough'},
        {'price': 120.0, 'type': 'Peak'}
    ], index=[dates[10], dates[40], dates[70], dates[100]])
    
    # 2. Extract Features Check
    print("Extracting synthesized features...")
    X_df = predictive_engine.extract_synthesis_features(
        df, metrics, cot_df, daily_cvd, daily_div_signals, pivot_df
    )
    assert X_df.shape[0] == n, f"Expected {n} rows, got {X_df.shape[0]}"
    print(f"[OK] Feature matrix compiled. Shape: {X_df.shape}")
    
    # 3. Target Compiler Check
    print("Compiling targets...")
    daily_vol_mock = pd.Series([0.02] * len(df), index=df.index)
    y_reg, y_class = predictive_engine.build_synthesis_targets(df, daily_vol_mock)
    assert y_reg.shape[0] == n, "Targets rows size mismatch"
    assert y_class.shape[0] == n, "Classification target size mismatch"
    print("[OK] Targets matrices compiled.")
    
    # 4. Instantiate Engine
    print("Initializing SynthesisPredictiveEngine...")
    engine = predictive_engine.SynthesisPredictiveEngine(
        df, metrics, cot_df, daily_cvd, daily_div_signals, pivot_df
    )
    
    # 5. Train Models Check (10 epochs for quick validation)
    print("Training models (Random Forest & PyTorch multi-task NN)...")
    summary = engine.train(epochs=10)
    print(f"[OK] Models trained successfully. Samples: {summary['samples']}, Features: {summary['features']}")
    
    # Check that validation metrics exist
    assert summary['val_mse'] >= 0, "MSE invalid"
    assert 0.0 <= summary['val_accuracy'] <= 1.0, "Accuracy out of bounds"
    print(f"[OK] Validation metrics - Accuracy: {summary['val_accuracy']:.1%}, MSE: {summary['val_mse']:.6f}")
    
    # 6. Inference Check
    print("Running latest predictions inference...")
    pred = engine.predict_latest()
    
    # Verify predictions dictionaries
    for model_key in ['ml_model', 'nn_model', 'consensus']:
        assert model_key in pred, f"Prediction key {model_key} missing"
        model_pred = pred[model_key]
        assert model_pred['nd_high'] > 0, "Next-day high must be positive"
        assert model_pred['nd_low'] > 0, "Next-day low must be positive"
        assert model_pred['weekly_high'] > 0, "Weekly high must be positive"
        assert model_pred['weekly_low'] > 0, "Weekly low must be positive"
        assert model_pred['trend'] in ['Bearish Breakdown', 'Neutral Range', 'Bullish Breakout'], "Invalid trend category"
        
    print("[OK] Inference outputs structured correctly.")
    print("Consensus Predictions:")
    print(f"  - Current Price: INR {pred['close_price']:.2f}")
    print(f"  - Next-Day Range: INR {pred['consensus']['nd_low']:.2f} to INR {pred['consensus']['nd_high']:.2f}")
    print(f"  - Weekly Range: INR {pred['consensus']['weekly_low']:.2f} to INR {pred['consensus']['weekly_high']:.2f}")
    print(f"  - Weekly Trend: {pred['consensus']['trend']}")
    
    # 7. ML-Exclusive Feedback Loop Verification
    print("Testing ML-Exclusive Feedback Loop (Concept Drift)...")
    
    # Ensure ledger database file gets created
    import os
    import sqlite3
    db_path = os.path.join(os.path.dirname(os.path.abspath(predictive_engine.__file__)), 'ml_ledger.db')
    
    # Remove existing test ledger database to start fresh
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    # Init DB
    predictive_engine.init_ledger_db()
    assert os.path.exists(db_path), "ml_ledger.db was not created"
    print("[OK] Ledger database successfully initialized.")
    
    # Log 5 dummy predictions to trigger drift (we will make actuals different from predictions to exceed the 1.5% threshold)
    ticker_test = "TEST_TICKER"
    pred_dict_dummy = {
        'nd_high': 100.0,
        'nd_low': 98.0,
        'weekly_high': 105.0,
        'weekly_low': 95.0,
        'trend': 'Neutral Range'
    }
    
    for i in range(5):
        pred_date = dates[10 + i]
        target_date = dates[11 + i]
        predictive_engine.log_prediction(
            ticker=ticker_test,
            prediction_date=str(pred_date.date()),
            target_date=str(target_date.date()),
            close_price=99.0,
            daily_vol=0.02,
            pred_dict=pred_dict_dummy
        )
        
    # Check that predictions are logged
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prediction_ledger WHERE ticker = ?", (ticker_test,))
    count = cursor.fetchone()[0]
    assert count == 5, f"Expected 5 predictions logged, got {count}"
    print("[OK] Predictions successfully written to ledger.")
    
    # Inject actuals with a high error (e.g. 5% error) to trigger drift
    for i in range(5):
        target_date = dates[11 + i]
        # Actuals are 105 (for high) and 93 (for low), creating ~5% error relative to predicted 100 and 98
        cursor.execute("""
            UPDATE prediction_ledger
            SET actual_nd_high = ?, actual_nd_low = ?,
                actual_w_high = ?, actual_w_low = ?,
                actual_trend = ?
            WHERE ticker = ? AND target_date = ?
        """, (105.0, 93.0, 110.0, 90.0, 1, ticker_test, str(target_date.date())))
    conn.commit()
    conn.close()
    
    # Check for drift
    is_drifted, rmse, rows = predictive_engine.check_concept_drift(ticker_test, threshold=0.015)
    assert is_drifted, "Drift was not triggered on 5% prediction error"
    assert rmse > 0.015, f"RMSE should exceed 1.5% threshold, got {rmse:.2%}"
    print(f"[OK] Concept Drift calculation verified. Detected RMSE: {rmse:.2%}")
    
    # Run PyTorch online retrain check
    if predictive_engine.TORCH_AVAILABLE:
        import torch
        print("Running online PyTorch heads-only retraining test...")
        initial_params = [p.clone() for p in engine.nn_model.shared.parameters()]
        
        predictive_engine.retrain_online_pytorch(
            model=engine.nn_model,
            ticker=ticker_test,
            scaler=engine.scaler,
            aligned_features_df=engine.aligned_features_df,
            aligned_reg_targets_df=engine.aligned_reg_targets_df,
            aligned_class_targets_df=engine.aligned_class_targets_df,
            hmm=engine.hmm,
            epochs=5
        )
        
        # Verify that self.shared parameters are unchanged (frozen)
        for p_init, p_curr in zip(initial_params, engine.nn_model.shared.parameters()):
            assert torch.equal(p_init, p_curr), "Shared parameters should be frozen"
            
        print("[OK] PyTorch online learning (layer freezing & heads retraining) passed.")
        
    # Run Random Forest online retrain check
    print("Running online Random Forest tree injection test...")
    initial_trees_reg = engine.rf_reg.n_estimators
    initial_trees_clf = engine.rf_clf.n_estimators
    
    predictive_engine.retrain_online_rf(
        rf_reg=engine.rf_reg,
        rf_clf=engine.rf_clf,
        scaler=engine.scaler,
        aligned_features_df=engine.aligned_features_df,
        aligned_reg_targets_df=engine.aligned_reg_targets_df,
        aligned_class_targets_df=engine.aligned_class_targets_df,
        hmm=engine.hmm
    )
    
    assert engine.rf_reg.n_estimators == initial_trees_reg + 10, "Random Forest regressor trees not injected"
    assert engine.rf_clf.n_estimators == initial_trees_clf + 10, "Random Forest classifier trees not injected"
    print(f"[OK] Random Forest warm-start tree injection passed (Reg Trees: {engine.rf_reg.n_estimators}, Clf Trees: {engine.rf_clf.n_estimators}).")
    
    # Cleanup test ledger database
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    print("All predictive engine tests passed successfully!")


if __name__ == "__main__":
    run_verification()
