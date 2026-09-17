import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_score

FEATURE_DIR = "nifty100_features"
RESULTS_DIR = "research_results"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    if not parquet_files: return

    df_list = []
    for file in parquet_files:
        temp_df = pd.read_parquet(file)
        if isinstance(temp_df.columns, pd.MultiIndex):
            temp_df.columns = temp_df.columns.droplevel(1)
        df_list.append(temp_df)
        
    df = pd.concat(df_list)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    target = "target_20_before_m8_45d"
    features = [
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 
        'dist_sma_20', 'dist_sma_60', 'dist_ema_20', 'dist_ema_60', 
        'dist_52w_high', 'dist_20d_high', 'range_expansion', 
        'atr_14', 'realized_vol_20d', 'volatility_expansion', 
        'rel_volume_20d', 'turnover_acceleration', 'dist_obv_20', 
        'rsi_14', 'roc_20', 'excess_ret_1d', 'excess_ret_20d'
    ]

    clean_df = df.dropna(subset=features + [target]).copy()

    start_year = 2018 
    end_year = clean_df.index.year.max()
    oos_predictions, oos_actuals = [], []

    for test_year in range(start_year, end_year + 1):
        test_start_date = pd.to_datetime(f"{test_year}-01-01")
        # 45 trading days roughly equals 65 calendar days
        embargo_cutoff = test_start_date - pd.Timedelta(days=65) 
        
        train_mask = clean_df.index <= embargo_cutoff
        test_mask = clean_df.index.year == test_year
        
        X_train, y_train = clean_df.loc[train_mask, features], clean_df.loc[train_mask, target]
        X_test, y_test = clean_df.loc[test_mask, features], clean_df.loc[test_mask, target]
        
        if X_test.empty: continue

        scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        clf_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_weight, random_state=42, n_jobs=-1, verbose=-1)
        clf_cat = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)
        
        ensemble = VotingClassifier(estimators=[('xgb', clf_xgb), ('lgb', clf_lgb), ('cat', clf_cat)], voting='soft')
        ensemble.fit(X_train, y_train)
        
        y_pred_prob = ensemble.predict_proba(X_test)[:, 1]
        oos_predictions.extend(y_pred_prob)
        oos_actuals.extend(y_test.values)

    oos_actuals = np.array(oos_actuals)
    oos_predictions = np.array(oos_predictions)
    high_conviction_preds = (oos_predictions >= 0.75).astype(int)

    auc = roc_auc_score(oos_actuals, oos_predictions)
    precision = precision_score(oos_actuals, high_conviction_preds, zero_division=0)
    
    report = (
        f"--- INSTITUTIONAL V3 HEAVY ENSEMBLE METRICS (45-DAY) ---\n"
        f"Target: {target}\n"
        f"Validation: Purged Walk-Forward (65-Day Embargo)\n"
        f"Base Rate: {oos_actuals.mean():.4f}\n"
        f"ROC-AUC Score: {auc:.4f}\n"
        f"Precision (Buy Signals >= 0.75): {precision:.4f}\n\n"
        f"Classification Report:\n{classification_report(oos_actuals, high_conviction_preds, zero_division=0)}\n"
    )
    
    print(report)
    with open(os.path.join(RESULTS_DIR, "ensemble_wf_metrics.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
