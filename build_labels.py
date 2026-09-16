import os
import glob
import pandas as pd
import numpy as np

INPUT_DIR = "nifty100_data"
OUTPUT_DIR = "nifty100_labels"

# Multi-horizon testing as mandated by Roadmap Section 1 & 3
HORIZONS = [10, 20, 60] 
PRIMARY_H = 20
TARGET_PCT = 0.20
STOP_PCT = -0.08

def clean_and_validate_data(df: pd.DataFrame, ticker: str):
    """Institutional Data QA check according to Section 4 of the roadmap."""
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        print(f"QA FAIL [{ticker}]: Missing required OHLCV columns.")
        return None
    
    if df.index.duplicated().any():
        print(f"QA WARNING [{ticker}]: Duplicate timestamps detected. Keeping last.")
        df = df[~df.index.duplicated(keep="last")]
        
    # Drop rows with negative or zero prices instead of failing the whole stock
    bad_price_mask = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if bad_price_mask.any():
        bad_count = bad_price_mask.sum()
        print(f"QA WARNING [{ticker}]: {bad_count} rows with non-positive prices detected and dropped.")
        df = df[~bad_price_mask]
        
    if df.empty:
        print(f"QA FAIL [{ticker}]: No valid data remaining after cleaning.")
        return None
        
    return df

def generate_institutional_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    n = len(df)
    
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    
    # -------------------------------------------------------------
    # 1. Multi-Horizon Simple Targets (10D, 20D, 60D)
    # -------------------------------------------------------------
    for h in HORIZONS:
        hit_col = f"hit_20pct_{h}d"
        ret_col = f"fwd_ret_{h}d"
        max_col = f"fwd_max_{h}d"
        
        df[hit_col] = np.nan
        df[ret_col] = np.nan
        df[max_col] = np.nan
        
        for i in range(n - h):
            ref = closes[i]
            if ref <= 0:
                continue
            window_highs = highs[i + 1 : i + 1 + h]
            max_ret = (np.max(window_highs) - ref) / ref
            close_ret = (closes[i + h] - ref) / ref
            
            df.iat[i, df.columns.get_loc(max_col)] = max_ret
            df.iat[i, df.columns.get_loc(ret_col)] = close_ret
            df.iat[i, df.columns.get_loc(hit_col)] = 1 if max_ret >= TARGET_PCT else 0

    # -------------------------------------------------------------
    # 2. Primary Candidate: +20% before -8% (20-Day Horizon)
    # -------------------------------------------------------------
    df[f"target_{int(TARGET_PCT*100)}_before_m{abs(int(STOP_PCT*100))}_{PRIMARY_H}d"] = np.nan
    col_primary = df.columns.get_loc(f"target_{int(TARGET_PCT*100)}_before_m{abs(int(STOP_PCT*100))}_{PRIMARY_H}d")

    # -------------------------------------------------------------
    # 3. Triple-Barrier Method (+1 = Target, -1 = Stop, 0 = Expired)
    # -------------------------------------------------------------
    df[f"triple_barrier_{PRIMARY_H}d"] = np.nan
    df[f"triple_barrier_ret_{PRIMARY_H}d"] = np.nan
    col_tb = df.columns.get_loc(f"triple_barrier_{PRIMARY_H}d")
    col_tb_ret = df.columns.get_loc(f"triple_barrier_ret_{PRIMARY_H}d")

    # -------------------------------------------------------------
    # 4. Realistic Execution Label: Entry at Open(t+1)
    # -------------------------------------------------------------
    df[f"tradable_hit_20pct_{PRIMARY_H}d"] = np.nan
    col_tradable = df.columns.get_loc(f"tradable_hit_20pct_{PRIMARY_H}d")

    for i in range(n - PRIMARY_H - 1):
        ref_close = closes[i]
        exec_open = opens[i + 1]  # Realistic fill price next morning
        if ref_close <= 0 or exec_open <= 0:
            continue
            
        fwd_highs = highs[i + 1 : i + 1 + PRIMARY_H]
        fwd_lows = lows[i + 1 : i + 1 + PRIMARY_H]
        
        # Primary candidate calculation (Close reference)
        hit_target, hit_stop = False, False
        barrier_hit = 0
        realized_ret = 0.0

        for d in range(PRIMARY_H):
            high_ret = (fwd_highs[d] - ref_close) / ref_close
            low_ret = (fwd_lows[d] - ref_close) / ref_close
            
            # Stop loss takes priority on dual-breach days
            if low_ret <= STOP_PCT:
                hit_stop = True
                barrier_hit = -1
                realized_ret = STOP_PCT
                break
            elif high_ret >= TARGET_PCT:
                hit_target = True
                barrier_hit = 1
                realized_ret = TARGET_PCT
                break
                
        # If neither barrier was touched during the 20 days:
        if barrier_hit == 0:
            realized_ret = (closes[i + PRIMARY_H] - ref_close) / ref_close

        df.iat[i, col_primary] = 1 if (hit_target and not hit_stop) else 0
        df.iat[i, col_tb] = barrier_hit
        df.iat[i, col_tb_ret] = realized_ret

        # Execution feasibility: Measured from t+1 Open instead of t Close
        fwd_exec_highs = highs[i + 1 : i + 1 + PRIMARY_H]
        tradable_max_ret = (np.max(fwd_exec_highs) - exec_open) / exec_open
        df.iat[i, col_tradable] = 1 if tradable_max_ret >= TARGET_PCT else 0

    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in '{INPUT_DIR}'.")
        return
        
    print(f"Running Institutional Label Engine across {len(parquet_files)} stocks...")
    success_count = 0
    
    for file_path in parquet_files:
        ticker = os.path.basename(file_path).replace(".parquet", "")
        try:
            df = pd.read_parquet(file_path)
            
            # Use the new cleaning function
            cleaned_df = clean_and_validate_data(df, ticker)
            if cleaned_df is None:
                continue
                
            # Pass the cleaned data to the label engine
            labeled_df = generate_institutional_labels(cleaned_df)
            
            out_file = os.path.join(OUTPUT_DIR, f"{ticker}_labeled.parquet")
            labeled_df.to_parquet(out_file, engine="pyarrow")
            success_count += 1
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    print(f"Completed: {success_count}/{len(parquet_files)} stocks generated with full label suite.")

if __name__ == "__main__":
    main()
