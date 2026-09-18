import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score
import itertools

FEATURE_DIR = "nifty100_features"

FEATURES = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 
    'dist_sma_20', 'dist_sma_60', 'dist_ema_20', 'dist_ema_60', 
    'dist_52w_high', 'dist_20d_high', 'range_expansion', 
    'atr_14', 'realized_vol_20d', 'volatility_expansion', 
    'rel_volume_20d', 'turnover_acceleration', 'dist_obv_20', 
    'rsi_14', 'roc_20', 'excess_ret_1d', 'excess_ret_20d',
    'excess_sector_ret_20d', 'sector_regime_dist'
]

def load_base_data():
    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    dfs = []
    for f in files:
        tdf = pd.read_parquet(f)
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = tdf.columns.droplevel(1)
        dfs.append(tdf)
    
    df = pd.concat(dfs).sort_index()
    df.index = pd.to_datetime(df.index)

    for col in FEATURES:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0.0)
        
    return df.dropna(subset=['Open', 'High', 'Low', 'Close'])

def apply_dynamic_labels(df, gain, stop, window):
    """Fast numpy label generator for specific target profiles."""
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    labels = np.zeros(n)
    
    for i in range(n - window - 1):
        entry = opens[i + 1]
        if pd.isna(entry) or entry <= 0: continue
        
        tgt_price = entry * (1.0 + gain)
        stp_price = entry * (1.0 + stop)
        
        w_highs = highs[i + 1 : i + 1 + window]
        w_lows = lows[i + 1 : i + 1 + window]
        
        for h, l in zip(w_highs, w_lows):
            if l <= stp_price:
                break # Stopped out
            if h >= tgt_price:
                labels[i] = 1 # Hit target
                break
                
    return labels

def run_discovery():
    print("Loading base features for Target Discovery...")
    df = load_base_data()
    
    if df.empty:
        print(f"No data found in {FEATURE_DIR}. Please run build_features.py first.")
        return

    # THE DISCOVERY GRID
    target_gains = [0.08, 0.12, 0.15, 0.20, 0.25, 0.30]
    time_windows = [15, 20, 30, 45, 60, 90]
    
    results = []
    combinations = list(itertools.product(target_gains, time_windows))
    
    print(f"Evaluating {len(combinations)} different target profiles...\n")
    
    for gain, window in combinations:
        # Dynamically scale stop loss to maintain a 2.5 Reward/Risk ratio
        stop = -(gain / 2.5) 
        
        # 1. Apply Labels
        df['dynamic_target'] = apply_dynamic_labels(df, gain, stop, window)
        
        # 2. Train/Test Split (Train 2018-2023, Test 2024-Present)
        # We leave a gap of 'window' days so no future data bleeds into training
        train_end = pd.to_datetime("2023-12-31") - pd.Timedelta(days=window)
        test_start = pd.to_datetime("2024-01-01")
        
        train_mask = df.index <= train_end
        test_mask = df.index >= test_start
        
        X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, 'dynamic_target']
        X_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, 'dynamic_target']
        
        if y_train.sum() == 0 or y_test.sum() == 0:
            continue
            
        scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
        
        # 3. Train Fast ML (LightGBM)
        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, 
                                 scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_train)
        
        # 4. Evaluate Precision on Out-of-Sample Test Set
        probs = clf.predict_proba(X_test)[:, 1]
        
        # Check precision at a 0.70 ML conviction threshold
        high_conv_mask = probs >= 0.70
        signals_generated = high_conv_mask.sum()
        
        precision = 0.0
        if signals_generated > 0:
            precision = (y_test[high_conv_mask] == 1).mean()
            
        # Win Rate of the overall target in the raw dataset
        base_hit_rate = y_train.mean()
            
        results.append({
            "Target": f"+{int(gain*100)}%",
            "Stop": f"{int(stop*100)}%",
            "Days": window,
            "Base_Hit_Rate": f"{base_hit_rate*100:.1f}%",
            "ML_Precision": f"{precision*100:.1f}%",
            "OOS_Signals": signals_generated
        })
        print(f"Evaluated +{int(gain*100)}% in {window} days... ML Precision: {precision*100:.1f}% (Signals: {signals_generated})")

    # Generate Leaderboard
    leaderboard = pd.DataFrame(results)
    # Sort by ML Precision (descending), then by number of signals
    leaderboard = leaderboard.sort_values(by=["ML_Precision", "OOS_Signals"], ascending=[False, False])
    
    print("\n==========================================================")
    print("             MACHINE LEARNING TARGET LEADERBOARD          ")
    print("==========================================================")
    print(leaderboard.to_string(index=False))
    print("==========================================================")

if __name__ == "__main__":
    run_discovery()
