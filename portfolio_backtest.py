import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

FEATURE_DIR = "nifty100_features"
BENCHMARK_FILE = "benchmark_data/Market.parquet"
RESULTS_DIR = "research_results"
TARGET = "target_20_before_m8_45d"

# --- INSTITUTIONAL PARAMETERS ---
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 4              # 25% allocation per trade
PROB_THRESHOLD = 0.75          # High conviction filter
ROUNDTRIP_FRICTION = 0.0035    # 35 bps friction
RISK_FREE_RATE = 0.06          # 6% annual cash interest

HARD_STOP_LOSS = -0.15         # -15% structural stop
MAX_HOLD_DAYS = 60             # 60 trading days runway
TRAILING_ACTIVATION = 0.35     # Activate trailing stop after +35%
TRAILING_DISTANCE = 0.10       # Trail 10% below peak

FEATURES = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 
    'dist_sma_20', 'dist_sma_60', 'dist_ema_20', 'dist_ema_60', 
    'dist_52w_high', 'dist_20d_high', 'range_expansion', 
    'atr_14', 'realized_vol_20d', 'volatility_expansion', 
    'rel_volume_20d', 'turnover_acceleration', 'dist_obv_20', 
    'rsi_14', 'roc_20', 'excess_ret_1d', 'excess_ret_20d',
    'excess_sector_ret_20d', 'sector_regime_dist'
]

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
    if not df_list:
        raise ValueError(f"No parquet files found in {FEATURE_DIR}.")
    panel = pd.concat(df_list).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel

def generate_out_of_sample_signals(df):
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    clean_train = df.dropna(subset=[TARGET, 'Open', 'High', 'Low', 'Close']).copy()
    start_year = 2018
    end_year = df.index.year.max()
    signals = []

    print(f"Generating OOS Ensemble Signals ({start_year}-{end_year})...")
    for test_year in range(start_year, end_year + 1):
        test_start = pd.to_datetime(f"{test_year}-01-01")
        embargo_cutoff = test_start - pd.Timedelta(days=65)

        train_mask = clean_train.index <= embargo_cutoff
        test_mask = df.index.year == test_year

        X_train = clean_train.loc[train_mask, FEATURES]
        y_train = clean_train.loc[train_mask, TARGET]
        test_data = df.loc[test_mask]

        if test_data.empty or len(X_train) == 0:
            continue

        scale_pos = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0

        clf_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_pos, random_state=42, n_jobs=-1)
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1)
        clf_cat = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)

        ensemble = VotingClassifier(estimators=[('xgb', clf_xgb), ('lgb', clf_lgb), ('cat', clf_cat)], voting='soft')
        ensemble.fit(X_train, y_train)

        probs = ensemble.predict_proba(test_data[FEATURES])[:, 1]
        test_subset = test_data[["Ticker", "Open", "High", "Low", "Close"]].copy()
        test_subset["Signal_Prob"] = probs
        signals.append(test_subset)

    return pd.concat(signals).sort_index()

def run_simulation():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_panel_data()
    sig_df = generate_out_of_sample_signals(df)

    unique_dates = sig_df.index.unique().sort_values()
    nav, cash = INITIAL_CAPITAL, INITIAL_CAPITAL
    open_positions, portfolio_history, trade_ledger = [], [], []
    daily_rf = (1.0 + RISK_FREE_RATE) ** (1 / 252) - 1.0

    print("Simulating execution ledger with Regime Filters...")
    for i in range(len(unique_dates) - 1):
        current_date, next_date = unique_dates[i], unique_dates[i + 1]
        cash *= (1.0 + daily_rf)

        todays_data = sig_df.loc[current_date]
        if isinstance(todays_data, pd.Series):
            todays_data = todays_data.to_frame().T
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

            current_stop_loss = HARD_STOP_LOSS
            if max_gain >= TRAILING_ACTIVATION:
                current_stop_loss = max(HARD_STOP_LOSS, max_gain - TRAILING_DISTANCE)

            exit_trade, raw_return, exit_reason = False, 0.0, ""

            if low_ret <= current_stop_loss:
                exit_trade = True
                raw_return = min(current_stop_loss, open_ret)
                exit_reason = "TRAILING_STOP_PROFIT" if current_stop_loss > 0 else "HARD_STOP"
            elif pos["Days_Held"] >= MAX_HOLD_DAYS:
                exit_trade = True
                raw_return = (row["Close"] - pos["Entry_Price"]) / pos["Entry_Price"]
                exit_reason = "TIME_EXPIRY"

            if exit_trade:
                net_return = raw_return - ROUNDTRIP_FRICTION
                cash += pos["Allocated_Capital"] * (1.0 + net_return)
                trade_ledger.append({
                    "Ticker": ticker, "Entry_Date": pos["Entry_Date"], "Exit_Date": current_date,
                    "Days_Held": pos["Days_Held"], "Net_Return": net_return, "Exit_Reason": exit_reason
                })
            else:
                pos["Current_Value"] = pos["Allocated_Capital"] * (1.0 + (row["Close"] - pos["Entry_Price"]) / pos["Entry_Price"])
                surviving_positions.append(pos)

        open_positions = surviving_positions
        nav = cash + sum(p["Current_Value"] for p in open_positions)
        portfolio_history.append({"Date": current_date, "NAV": nav, "Cash": cash, "Positions_Count": len(open_positions)})

        eligible_signals = todays_data[todays_data["Signal_Prob"] >= PROB_THRESHOLD].sort_values(by="Signal_Prob", ascending=False)
        open_tickers = {p["Ticker"] for p in open_positions}
        available_slots = MAX_POSITIONS - len(open_positions)

        if available_slots > 0 and not eligible_signals.empty:
            candidates = eligible_signals[~eligible_signals["Ticker"].isin(open_tickers)].head(available_slots)
            next_day_data = sig_df.loc[next_date]
            if isinstance(next_day_data, pd.Series):
                next_day_data = next_day_data.to_frame().T
            next_open_map = next_day_data.set_index("Ticker")["Open"].to_dict()
            allocation = nav * (1.0 / MAX_POSITIONS)

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
    trades_df = pd.DataFrame(trade_ledger)

    daily_returns = perf_df["NAV"].pct_change().dropna()
    total_days = (perf_df.index[-1] - perf_df.index[0]).days
    cagr = ((perf_df["NAV"].iloc[-1] / perf_df["NAV"].iloc[0]) ** (365.25 / total_days)) - 1.0

    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ((daily_returns.mean() - (RISK_FREE_RATE / 252)) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    max_dd = ((perf_df["NAV"] - perf_df["NAV"].cummax()) / perf_df["NAV"].cummax()).min()

    nifty_cagr = np.nan
    if os.path.exists(BENCHMARK_FILE):
        bench = pd.read_parquet(BENCHMARK_FILE)
        bench.index = pd.to_datetime(bench.index)
        col_to_use = "Market_Close" if "Market_Close" in bench.columns else "Close"
        bench_aligned = bench.loc[perf_df.index[0]:perf_df.index[-1], col_to_use]
        if not bench_aligned.empty:
            nifty_cagr = ((bench_aligned.iloc[-1] / bench_aligned.iloc[0]) ** (365.25 / total_days)) - 1.0

    win_rate = (trades_df["Net_Return"] > 0).mean() if len(trades_df) > 0 else 0.0

    report = (
        f"==========================================================\n"
        f"        INSTITUTIONAL MOMENTUM RIDING SIMULATION          \n"
        f"==========================================================\n"
        f"Period:                     {perf_df.index[0].date()} to {perf_df.index[-1].date()}\n"
        f"Initial Capital:            INR {INITIAL_CAPITAL:,.2f}\n"
        f"Final Net Asset Value:      INR {perf_df['NAV'].iloc[-1]:,.2f}\n"
        f"Max Positions & Sizing:     {MAX_POSITIONS} ({100/MAX_POSITIONS:.0f}% NAV per trade)\n"
        f"Friction Deducted:          {ROUNDTRIP_FRICTION * 10000:.0f} bps round-trip\n"
        f"----------------------------------------------------------\n"
        f"Strategy CAGR:              {cagr * 100:.2f}%\n"
        f"Nifty Benchmark CAGR:       {nifty_cagr * 100:.2f}%\n"
        f"Annualized Volatility:      {ann_vol * 100:.2f}%\n"
        f"Sharpe Ratio (Rf=6.0%):     {sharpe:.2f}\n"
        f"Max Drawdown:               {max_dd * 100:.2f}%\n"
        f"----------------------------------------------------------\n"
        f"Total Trades Completed:     {len(trades_df)}\n"
        f"Win Rate:                   {win_rate * 100:.2f}%\n"
        f"Average Trade Return (Net): {trades_df['Net_Return'].mean() * 100:.2f}%\n"
        f"Max Profit in a Single Trade: {trades_df['Net_Return'].max() * 100:.2f}%\n"
        f"Riding Trailing Stop Exits: {(trades_df['Exit_Reason'] == 'TRAILING_STOP_PROFIT').mean() * 100:.2f}%\n"
        f"Time Expiry Exits:          {(trades_df['Exit_Reason'] == 'TIME_EXPIRY').mean() * 100:.2f}%\n"
        f"Hard Stop Hit Rate:         {(trades_df['Exit_Reason'] == 'HARD_STOP').mean() * 100:.2f}%\n"
        f"==========================================================\n"
    )
    print(report)
    with open(os.path.join(RESULTS_DIR, "portfolio_backtest_report.txt"), "w") as f:
        f.write(report)
    perf_df.to_parquet(os.path.join(RESULTS_DIR, "equity_curve.parquet"), engine="pyarrow")
    trades_df.to_parquet(os.path.join(RESULTS_DIR, "trade_ledger.parquet"), engine="pyarrow")

if __name__ == "__main__":
    run_simulation()
