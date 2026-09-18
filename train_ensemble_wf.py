import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURE_DIR = "nifty100_features"
RESULTS_DIR = "research_results"
TARGET = "target_20_before_m8_45d"

FEATURES = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 
    'dist_sma_20', 'dist_sma_60', 'dist_ema_20', 'dist_ema_60', 
    'dist_52w_high', 'dist_20d_high', 'range_expansion', 
    'atr_14', 'realized_vol_20d', 'volatility_expansion', 
    'rel_volume_20d', 'turnover_acceleration', 'dist_obv_20', 
    'rsi_14', 'roc_20', 'excess_ret_1d', 'excess_ret_20d',
    'excess_sector_ret_20d', 'sector_regime_dist'
]

def load_clean_panel():
    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    dfs = []
    for f in files:
        ticker = os.path.basename(f).replace("_features.parquet", "")
        tdf = pd.read_parquet(f)
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = tdf.columns.droplevel(1)
        tdf['Ticker'] = ticker
        dfs.append(tdf)
        
    panel = pd.concat(dfs).sort_index()
    panel.index = pd.to_datetime(panel.index)

    # Fill any missing optional feature values with 0
    for col in FEATURES:
        if col not in panel.columns:
            panel[col] = 0.0
        else:
            panel[col] = panel[col].fillna(0.0)

    # Only keep historical rows where resolved targets exist
    clean = panel.dropna(subset=[TARGET, 'Open', 'High', 'Low', 'Close']).copy()
    return clean

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_clean_panel()

    start_year = 2018
    end_year = df.index.year.max()
    metrics = []

    print(f"Starting Walk-Forward Training ({start_year} to {end_year}) with Sector Regime Features...")

    for test_year in range(start_year, end_year + 1):
        test_start = pd.to_datetime(f"{test_year}-01-01")
        embargo_cutoff = test_start - pd.Timedelta(days=65)

        train_mask = df.index <= embargo_cutoff
        test_mask = df.index.year == test_year

        X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET]
        X_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, TARGET]

        if len(X_train) == 0 or len(X_test) == 0 or y_train.sum() == 0 or y_test.sum() == 0:
            continue

        scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()

        clf_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            scale_pos_weight=scale_pos, random_state=42, n_jobs=-1
        )
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1
        )
        clf_cat = CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.03,
            auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1
        )

        ensemble = VotingClassifier(
            estimators=[('xgb', clf_xgb), ('lgb', clf_lgb), ('cat', clf_cat)],
            voting='soft'
        )
        ensemble.fit(X_train, y_train)

        probs = ensemble.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        high_conv_mask = probs >= 0.75
        prec_75 = (y_test[high_conv_mask] == 1).mean() if high_conv_mask.sum() > 0 else 0.0

        metrics.append({
            "Year": test_year,
            "Train_Samples": len(X_train),
            "Test_Samples": len(X_test),
            "AUC": round(auc, 4),
            "Brier_Score": round(brier, 4),
            "Signals_Above_0.75": int(high_conv_mask.sum()),
            "Precision_Above_0.75": f"{prec_75 * 100:.2f}%"
        })
        print(f"[{test_year}] AUC: {auc:.4f} | Brier: {brier:.4f} | Conviction >= 0.75: {high_conv_mask.sum()} (Prec: {prec_75*100:.1f}%)")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "walk_forward_metrics.csv"), index=False)
    print("\nEnsemble Walk-Forward Validation Complete.")

if __name__ == "__main__":
    main()
