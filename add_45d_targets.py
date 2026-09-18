import os
import glob
import pandas as pd
import numpy as np

FEATURE_DIR = "nifty100_features"
TARGET_COL = "target_20_before_m8_45d"
TARGET_GAIN = 0.20
STOP_LOSS = -0.08
MAX_WINDOW = 45

def process_file(file_path):
    df = pd.read_parquet(file_path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df = df.sort_index()
    n = len(df)
    target_values = np.full(n, np.nan)

    if n > MAX_WINDOW + 1:
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values

        # Only label rows where the full 45-day forward horizon has resolved
        for i in range(n - (MAX_WINDOW + 1)):
            entry_price = opens[i + 1]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            target_price = entry_price * (1.0 + TARGET_GAIN)
            stop_price = entry_price * (1.0 + STOP_LOSS)

            window_highs = highs[i + 1 : i + 1 + MAX_WINDOW]
            window_lows = lows[i + 1 : i + 1 + MAX_WINDOW]

            hit = 0
            for h, l in zip(window_highs, window_lows):
                if l <= stop_price:
                    hit = 0
                    break
                if h >= target_price:
                    hit = 1
                    break
            target_values[i] = hit

    df[TARGET_COL] = target_values
    df.to_parquet(file_path, engine="pyarrow")

def main():
    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    print(f"Stamping 45-day targets across {len(files)} feature files...")
    for f in files:
        try:
            process_file(f)
        except Exception as e:
            print(f"Failed to process {f}: {e}")
    print("Target generation complete.")

if __name__ == "__main__":
    main()
