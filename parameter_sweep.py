import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import itertools

FEATURE_DIR = "nifty100_features"
TARGET = "target_20_before_m8_45d"
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 4
ROUNDTRIP_FRICTION = 0.0035
RISK_FREE_RATE = 0.06

FEATURES = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 
    'dist_sma_20', 'dist_sma_60', 'dist_ema_20', 'dist_ema_60', 
    'dist_52w_high', 'dist_20d_high', 'range_expansion', 
    'atr_14', 'realized_vol_20d', 'volatility_expansion', 
    'rel_volume_20d', 'turnover_acceleration', 'dist_obv_20', 
    'rsi_14', 'roc_20', 'excess_ret_1d', 'excess_ret_20d',
    'excess_sector_ret_20d', 'sector_regime_dist'
]

def generate_signals():
    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    dfs = []
    for f in files:
        ticker = os.path.basename(f).replace("_features.parquet", "")
        tdf = pd.read_parquet(f)
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = tdf.columns.droplevel(1)
        tdf["Ticker"] = ticker
        dfs.append(tdf)
    df = pd.concat(dfs).sort_index()
    df.index = pd.to_datetime(df.index)

    for col in FEATURES:
        if col not in df.columns: df[col] = 0.0
        else: df[col] = df[col].fillna(0.0)

    clean = df.dropna(subset=[TARGET, 'Open', 'High', 'Low', 'Close']).copy()
    start_year = 2018
    end_year = df.index.year.max()
    signals = []

    print("Training OOS Engine Once (This takes a moment)...")
    for test_year in range(start_year, end_year + 1):
        test_start = pd.to_datetime(f"{test_year}-01-01")
        embargo = test_start - pd.Timedelta(days=65)

        train_mask = clean.index <= embargo
        test_mask = df.index.year == test_year

        X_train, y_train = clean.loc[train_mask, FEATURES], clean.loc[train_mask, TARGET]
        test_data = df.loc[test_mask]

        if test_data.empty or len(X_train) == 0: continue
        scale_pos = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0

        clf_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_pos, random_state=42, n_jobs=-1)
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1)
        clf_cat = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)

        ensemble = VotingClassifier(estimators=[('xgb', clf_xgb), ('lgb', clf_lgb), ('cat', clf_cat)], voting='soft')
        ensemble.fit(X_train, y_train)

        probs = ensemble.predict_proba(test_data[FEATURES])[:, 1]
        subset = test_data[["Ticker", "Open", "High", "Low", "Close"]].copy()
        subset["Signal_Prob"] = probs
        signals.append(subset)

    return pd.concat(signals).sort_index()

def run_sweep():
    sig_df = generate_signals()
    unique_dates = sig_df.index.unique().sort_values()
    daily_rf = (1.0 + RISK_FREE_RATE) / 252

    # Parameter grid designed to find the new calibration
    prob_thresholds = [0.75, 0.80, 0.82, 0.85]
    stop_losses = [-0.10, -0.12, -0.15]
    hold_days = [40, 50, 60]
    trail_activations = [0.25, 0.35]
    
    combinations = list(itertools.product(prob_thresholds, stop_losses, hold_days, trail_activations))
    results = []

    print(f"Executing Fast-Sweep across {len(combinations)} parameter sets...")

    for (prob, stop, hold, trail) in combinations:
        nav, cash = INITIAL_CAPITAL, INITIAL_CAPITAL
        open_positions = []
        portfolio_history = []

        for i in range(len(unique_dates) - 1):
            current_date, next_date = unique_dates[i], unique_dates[i + 1]
            cash *= (1.0 + daily_rf)

            todays_data = sig_df.loc[current_date]
            if isinstance(todays_data, pd.Series): todays_data = todays_data.to_frame().T
            stock_map = todays_data.set_index("Ticker").to_dict(orient="index")

            surviving_positions = []
            for pos in open_positions:
                ticker = pos["Ticker"]
                pos["Days_Held"] += 1

                if ticker not in stock_map:
                    surviving_positions.append(pos)
                    continue

                row = stock_map[ticker]
                pos["Highest_High"] = max(pos.get("Highest_High", pos["Entry_Price"]), row["High"])
                max_gain = (pos["Highest_High"] - pos["Entry_Price"]) / pos["Entry_Price"]

                low_ret = (row["Low"] - pos["Entry_Price"]) / pos["Entry_Price"]
                open_ret = (row["Open"] - pos["Entry_Price"]) / pos["Entry_Price"]

                current_stop = stop
                if max_gain >= trail: current_stop = max(stop, max_gain - 0.10)

                exit_trade = False
                if low_ret <= current_stop:
                    exit_trade = True
                    raw_return = min(current_stop, open_ret)
                elif pos["Days_Held"] >= hold:
                    exit_trade = True
                    raw_return = (row["Close"] - pos["Entry_Price"]) / pos["Entry_Price"]

                if exit_trade:
                    cash += pos["Allocated"] * (1.0 + (raw_return - ROUNDTRIP_FRICTION))
                else:
                    pos["Current_Value"] = pos["Allocated"] * (1.0 + (row["Close"] - pos["Entry_Price"]) / pos["Entry_Price"])
                    surviving_positions.append(pos)

            open_positions = surviving_positions
            nav = cash + sum(p["Current_Value"] for p in open_positions)
            portfolio_history.append(nav)

            eligible = todays_data[todays_data["Signal_Prob"] >= prob].sort_values(by="Signal_Prob", ascending=False)
            open_tickers = {p["Ticker"] for p in open_positions}
            slots = MAX_POSITIONS - len(open_positions)

            if slots > 0 and not eligible.empty:
                cands = eligible[~eligible["Ticker"].isin(open_tickers)].head(slots)
                next_day_data = sig_df.loc[next_date]
                if isinstance(next_day_data, pd.Series): next_day_data = next_day_data.to_frame().T
                next_open_map = next_day_data.set_index("Ticker")["Open"].to_dict()
                alloc = nav / MAX_POSITIONS

                for _, cand in cands.iterrows():
                    tk = cand["Ticker"]
                    if tk in next_open_map and cash >= alloc:
                        fill = next_open_map[tk]
                        if fill > 0:
                            cash -= alloc
                            open_positions.append({
                                "Ticker": tk, "Entry_Price": fill, "Allocated": alloc, 
                                "Current_Value": alloc, "Days_Held": 0, "Highest_High": fill
                            })

        if not portfolio_history: continue
        
        perf = pd.Series(portfolio_history)
        total_days = (unique_dates[-1] - unique_dates[0]).days
        cagr = ((perf.iloc[-1] / perf.iloc[0]) ** (365.25 / total_days)) - 1.0
        max_dd = ((perf - perf.cummax()) / perf.cummax()).min()
        
        daily_returns = perf.pct_change().dropna()
        sharpe = ((daily_returns.mean() - (RISK_FREE_RATE / 252)) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        results.append({
            "Prob": prob, "Stop": stop, "Hold": hold, "Trail": trail, 
            "CAGR": cagr, "Sharpe": sharpe, "Max_DD": max_dd
        })

    res_df = pd.DataFrame(results).sort_values(by="Sharpe", ascending=False).head(15)
    print("\n--- TOP 15 REGIME-ADJUSTED PARAMETERS ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_sweep()
