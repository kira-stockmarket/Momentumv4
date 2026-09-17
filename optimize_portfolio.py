import os
import glob
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

FEATURE_DIR = "nifty100_features"
BENCHMARK_FILE = "benchmark_data/NSEI.parquet" 
RESULTS_DIR = "research_results"

# --- Fixed Execution Parameters ---
INITIAL_CAPITAL = 1_000_000.0  
ROUNDTRIP_FRICTION = 0.0035    
RISK_FREE_RATE = 0.06          
TRAILING_DISTANCE = 0.10       

# --- 10,584 Combinations Grid ---
PARAM_GRID = {
    "MAX_POSITIONS": [2, 3, 4, 5, 6, 8, 10],                 # 50% down to 10% allocation
    "HARD_STOP_LOSS": [-0.05, -0.06, -0.08, -0.10, -0.12, -0.15], # Tight to extremely loose stops
    "MAX_HOLD_DAYS": [15, 20, 30, 40, 45, 50, 60],           # Fast swing to multi-month holds
    "TARGET_ACTIVATION": [0.12, 0.15, 0.20, 0.25, 0.30, 0.35],# When to lock in profits / trail
    "PROB_THRESHOLD": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]   # ML conviction limits
}

# Global variable for multiprocessing memory efficiency
global_sig_df = None

def load_panel_data():
    parquet_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    df_list = []
    for f in parquet_files:
        ticker = os.path.basename(f).replace("_features.parquet", "")
        tdf = pd.read_parquet(f)
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = tdf.columns.droplevel(1)
        tdf["Ticker"] = ticker
        df_list.append(tdf)
    panel = pd.concat(df_list)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()

def generate_signals_once(clean_df, features, target):
    start_year = 2018
    end_year = clean_df.index.year.max()
    signals = []

    print("Training ML Ensemble once to generate base probabilities...")
    for test_year in range(start_year, end_year + 1):
        test_start_date = pd.to_datetime(f"{test_year}-01-01")
        embargo_cutoff = test_start_date - pd.Timedelta(days=65) 
        
        train_mask = clean_df.index <= embargo_cutoff
        test_mask = clean_df.index.year == test_year

        X_train, y_train = clean_df.loc[train_mask, features], clean_df.loc[train_mask, target]
        test_data = clean_df.loc[test_mask]
        if test_data.empty: continue

        scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        clf_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_weight, random_state=42, n_jobs=-1, verbose=-1)
        clf_cat = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)
        
        ensemble = VotingClassifier(estimators=[('xgb', clf_xgb), ('lgb', clf_lgb), ('cat', clf_cat)], voting='soft')
        ensemble.fit(X_train, y_train)

        probs = ensemble.predict_proba(test_data[features])[:, 1]
        test_subset = test_data[["Ticker", "Open", "High", "Low", "Close"]].copy()
        test_subset["Signal_Prob"] = probs
        signals.append(test_subset)

    return pd.concat(signals).sort_index()

def init_worker(sig_df_shared):
    global global_sig_df
    global_sig_df = sig_df_shared

def simulate_wrapper(params):
    max_pos = params["MAX_POSITIONS"]
    stop_loss = params["HARD_STOP_LOSS"]
    max_hold = params["MAX_HOLD_DAYS"]
    trail_act = params["TARGET_ACTIVATION"]
    prob_thresh = params["PROB_THRESHOLD"]
    
    unique_dates = global_sig_df.index.unique().sort_values()
    nav, cash = INITIAL_CAPITAL, INITIAL_CAPITAL
    open_positions, portfolio_history = [], []
    daily_rf = (1.0 + RISK_FREE_RATE) ** (1 / 252) - 1.0

    for i in range(len(unique_dates) - 1):
        current_date, next_date = unique_dates[i], unique_dates[i + 1]
        cash *= (1.0 + daily_rf) 

        todays_data = global_sig_df.loc[current_date]
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

            current_stop_loss = stop_loss
            if max_gain >= trail_act:
                current_stop_loss = max(stop_loss, max_gain - TRAILING_DISTANCE)

            exit_trade, raw_return = False, 0.0

            if low_ret <= current_stop_loss:
                exit_trade = True
                raw_return = min(current_stop_loss, open_ret) 
            elif pos["Days_Held"] >= max_hold:
                exit_trade = True
                raw_return = (row["Close"] - pos["Entry_Price"]) / pos["Entry_Price"]

            if exit_trade:
                net_return = raw_return - ROUNDTRIP_FRICTION
                cash += pos["Allocated_Capital"] * (1.0 + net_return)
            else:
                pos["Current_Value"] = pos["Allocated_Capital"] * (1.0 + (row["Close"] - pos["Entry_Price"]) / pos["Entry_Price"])
                surviving_positions.append(pos)

        open_positions = surviving_positions
        nav = cash + sum(p["Current_Value"] for p in open_positions)
        portfolio_history.append({"Date": current_date, "NAV": nav})

        eligible_signals = todays_data[todays_data["Signal_Prob"] >= prob_thresh].sort_values(by="Signal_Prob", ascending=False)
        open_tickers = {p["Ticker"] for p in open_positions}
        available_slots = max_pos - len(open_positions)

        if available_slots > 0 and not eligible_signals.empty:
            candidates = eligible_signals[~eligible_signals["Ticker"].isin(open_tickers)].head(available_slots)
            next_day_data = global_sig_df.loc[next_date]
            if isinstance(next_day_data, pd.Series): next_day_data = next_day_data.to_frame().T
            next_open_map = next_day_data.set_index("Ticker")["Open"].to_dict()
            allocation = nav * (1.0 / max_pos)

            for _, cand in candidates.iterrows():
                tk = cand["Ticker"]
                if tk in next_open_map and cash >= allocation:
                    fill_price = next_open_map[tk]
                    if fill_price > 0:
                        cash -= allocation
                        open_positions.append({
                            "Ticker": tk, "Entry_Date": next_date, "Entry_Price": fill_price,
                            "Allocated_Capital": allocation, "Current_Value": allocation, "Days_Held": 0,
                            "Highest_High": fill_price
                        })

    perf_df = pd.DataFrame(portfolio_history).set_index("Date")
    total_days = (perf_df.index[-1] - perf_df.index[0]).days
    
    if total_days == 0 or perf_df["NAV"].iloc[0] == 0: return params, 0, 0, 0
    
    cagr = ((perf_df["NAV"].iloc[-1] / perf_df["NAV"].iloc[0]) ** (365.25 / total_days)) - 1.0
    daily_returns = perf_df["NAV"].pct_change().dropna()
    sharpe = ((daily_returns.mean() - (RISK_FREE_RATE / 252)) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    max_dd = ((perf_df["NAV"] - perf_df["NAV"].cummax()) / perf_df["NAV"].cummax()).min()

    return params, cagr, sharpe, max_dd

def run_optimization():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_panel_data()

    target = "target_20_before_m8_45d"
    features = [
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 
        'dist_sma_20', 'dist_sma_60', 'dist_ema_20', 'dist_ema_60', 
        'dist_52w_high', 'dist_20d_high', 'range_expansion', 
        'atr_14', 'realized_vol_20d', 'volatility_expansion', 
        'rel_volume_20d', 'turnover_acceleration', 'dist_obv_20', 
        'rsi_14', 'roc_20', 'excess_ret_1d', 'excess_ret_20d'
    ]

    clean_df = df.dropna(subset=features + [target, "Open", "High", "Low", "Close"]).copy()
    sig_df = generate_signals_once(clean_df, features, target)

    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_combs = len(combinations)
    
    print(f"\nStarting Multiprocessing Grid Search over {total_combs} combinations...")
    
    cores = max(1, multiprocessing.cpu_count() - 1)
    results = []
    
    with multiprocessing.Pool(processes=cores, initializer=init_worker, initargs=(sig_df,)) as pool:
        for idx, (params, cagr, sharpe, max_dd) in enumerate(pool.imap_unordered(simulate_wrapper, combinations)):
            results.append({
                "Prob Thresh": f"{params['PROB_THRESHOLD']:.2f}",
                "Max Pos": params["MAX_POSITIONS"],
                "Stop Loss": f"{params['HARD_STOP_LOSS']*100:.0f}%",
                "Hold Days": params["MAX_HOLD_DAYS"],
                "Target/Trail": f"{params['TARGET_ACTIVATION']*100:.0f}%",
                "CAGR": cagr,
                "Sharpe": sharpe,
                "Max DD": max_dd
            })
            
            if (idx + 1) % 500 == 0:
                print(f"[{idx+1}/{total_combs}] combinations evaluated...")

    results_df = pd.DataFrame(results).sort_values(by="CAGR", ascending=False)
    
    results_df["CAGR"] = (results_df["CAGR"] * 100).map("{:.2f}%".format)
    results_df["Sharpe"] = results_df["Sharpe"].map("{:.2f}".format)
    results_df["Max DD"] = (results_df["Max DD"] * 100).map("{:.2f}%".format)

    print("\n==================================================================================")
    print("                      TOP 15 EXECUTION SETUPS (BY CAGR)                           ")
    print("==================================================================================")
    print(results_df.head(15).to_string(index=False))
    print("==================================================================================")

    results_df.to_csv(os.path.join(RESULTS_DIR, "parameter_sweep_10k_results.csv"), index=False)

if __name__ == "__main__":
    run_optimization()
