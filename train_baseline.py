import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, precision_score

FEATURE_DIR = "nifty100_features"
RESULTS_DIR = "research_results"
# A logical out-of-sample test split for recent history
TEST_SPLIT_DATE = "2024-01-01"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    
    if not parquet_files:
        print(f"No feature files found in '{FEATURE_DIR}'.")
        return

    print("Loading and aggregating feature store...")
    df_list = []
    for file in parquet_files:
        temp_df = pd.read_parquet(file)
        if isinstance(temp_df.columns, pd.MultiIndex):
            temp_df.columns = temp_df.columns.droplevel(1)
        df_list.append(temp_df)
        
    df = pd.concat(df_list)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Define our primary target and feature set
    target = "target_20_before_m8_20d"
    features = [
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 
        'dist_sma_20', 'dist_sma_60', 'dist_ema_20', 'dist_ema_60', 
        'dist_52w_high', 'dist_20d_high', 'range_expansion', 
        'atr_14', 'realized_vol_20d', 'volatility_expansion', 
        'rel_volume_20d', 'turnover_acceleration', 'dist_obv_20', 
        'rsi_14', 'roc_20', 'excess_ret_1d', 'excess_ret_20d'
    ]
    
    # Drop rows with NaNs in our specific features or target 
    # (e.g., the most recent 20 days which haven't completed their forward window)
    clean_df = df.dropna(subset=features + [target]).copy()

    # Walk-Forward Chronological Split
    train_mask = clean_df.index < TEST_SPLIT_DATE
    test_mask = clean_df.index >= TEST_SPLIT_DATE
    
    X_train, y_train = clean_df.loc[train_mask, features], clean_df.loc[train_mask, target]
    X_test, y_test = clean_df.loc[test_mask, features], clean_df.loc[test_mask, target]

    print(f"Training Baseline Model on {len(X_train)} historical samples...")
    print(f"Testing on {len(X_test)} unseen out-of-sample records...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression (balanced to handle rare 20% moves)
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict probabilities on the unseen test set
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    # Calculate Evaluation Metrics
    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred_class, zero_division=0)
    
    report = (
        f"--- INSTITUTIONAL V1 BASELINE METRICS ---\n"
        f"Target: {target}\n"
        f"Out-of-Sample Period: {TEST_SPLIT_DATE} to Present\n"
        f"Base Rate (Actual Event Frequency): {y_test.mean():.4f}\n"
        f"ROC-AUC Score: {auc:.4f}\n"
        f"Precision (Accuracy of 'Buy' Signals): {precision:.4f}\n\n"
        f"Classification Report:\n{classification_report(y_test, y_pred_class, zero_division=0)}\n"
    )
    
    print(report)
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
