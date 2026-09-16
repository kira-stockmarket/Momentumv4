import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, precision_score

FEATURE_DIR = "nifty100_features"
RESULTS_DIR = "research_results"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    
    if not parquet_files:
        print(f"No feature files found in '{FEATURE_DIR}'.")
        return

    print("Loading Feature Store...")
    df_list = []
    for file in parquet_files:
        temp_df = pd.read_parquet(file)
        if isinstance(temp_df.columns, pd.MultiIndex):
            temp_df.columns = temp_df.columns.droplevel(1)
        # Keep track of the ticker for signal generation later
        temp_df['Ticker'] = os.path.basename(file).replace("_features.parquet", "")
        df_list.append(temp_df)
        
    df = pd.concat(df_list)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    target = "target_20_before_m8_20d"
    features = [
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 
        'dist_sma_20', 'dist_sma_60', 'dist_ema_20', 'dist_ema_60', 
        'dist_52w_high', 'dist_20d_high', 'range_expansion', 
        'atr_14', 'realized_vol_20d', 'volatility_expansion', 
        'rel_volume_20d', 'turnover_acceleration', 'dist_obv_20', 
        'rsi_14', 'roc_20', 'excess_ret_1d', 'excess_ret_20d'
    ]

    clean_df = df.dropna(subset=features + [target]).copy()

    print("Initiating Expanding Walk-Forward Engine...")
    
    # Define Walk-Forward boundaries
    start_year = 2018 # Initial training window: 2010 to 2017
    end_year = clean_df.index.year.max()
    
    oos_predictions = []
    oos_actuals = []

    for test_year in range(start_year, end_year + 1):
        print(f"Training up to {test_year - 1}, Testing out-of-sample on {test_year}...")
        
        # 1. Train on everything strictly BEFORE the test year
        train_mask = clean_df.index.year < test_year
        # 2. Test strictly on the unseen test year
        test_mask = clean_df.index.year == test_year
        
        X_train = clean_df.loc[train_mask, features]
        y_train = clean_df.loc[train_mask, target]
        
        X_test = clean_df.loc[test_mask, features]
        y_test = clean_df.loc[test_mask, target]
        
        if X_test.empty:
            continue

        # Calculate scale_pos_weight to handle the rare 3% base rate
        positive_cases = y_train.sum()
        negative_cases = len(y_train) - positive_cases
        scale_weight = negative_cases / positive_cases if positive_cases > 0 else 1

        # Institutional XGBoost Parameters (focused on preventing overfitting)
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,              # Shallow trees prevent memorizing noise
            learning_rate=0.05,
            subsample=0.8,            # Randomly sample 80% of rows per tree
            colsample_bytree=0.8,     # Randomly sample 80% of features per tree
            scale_pos_weight=scale_weight,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predict on unseen year
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        oos_predictions.extend(y_pred_prob)
        oos_actuals.extend(y_test.values)

    # Calculate final Walk-Forward Metrics across all test years combined
    oos_actuals = np.array(oos_actuals)
    oos_predictions = np.array(oos_predictions)
    
    # We use a 0.70 probability threshold for a "High Conviction Buy" instead of 0.50
    high_conviction_preds = (oos_predictions >= 0.70).astype(int)

    auc = roc_auc_score(oos_actuals, oos_predictions)
    precision = precision_score(oos_actuals, high_conviction_preds, zero_division=0)
    
    report = (
        f"--- INSTITUTIONAL V1 XGBOOST WALK-FORWARD METRICS ---\n"
        f"Target: {target}\n"
        f"Validation Method: Expanding Window Walk-Forward ({start_year}-{end_year})\n"
        f"Total Out-of-Sample Predictions: {len(oos_actuals)}\n"
        f"Base Rate (Actual Event Frequency): {oos_actuals.mean():.4f}\n"
        f"ROC-AUC Score: {auc:.4f}\n"
        f"Precision (Accuracy of High-Conviction 'Buy' Signals >= 0.70): {precision:.4f}\n\n"
        f"Classification Report (High Conviction):\n{classification_report(oos_actuals, high_conviction_preds, zero_division=0)}\n"
    )
    
    print(report)
    
    with open(os.path.join(RESULTS_DIR, "xgboost_wf_metrics.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
