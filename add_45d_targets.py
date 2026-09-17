import os
import glob
import pandas as pd
import numpy as np

FEATURE_DIR = "nifty100_features"

def main():
    parquet_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.parquet")))
    print(f"Adding 45-day targets to {len(parquet_files)} files...")
    
    for file in parquet_files:
        df = pd.read_parquet(file)
        
        target_col = np.full(len(df), np.nan)
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        # Calculate exactly 45 trading days forward
        for i in range(len(df) - 46):
            entry_price = opens[i+1] # Execute at next morning's Open
            if pd.isna(entry_price) or entry_price <= 0:
                continue
                
            target_price = entry_price * 1.20
            stop_price = entry_price * 0.92
            
            window_highs = highs[i+1 : i+46]
            window_lows = lows[i+1 : i+46]
            
            hit_target = 0
            for h, l in zip(window_highs, window_lows):
                if l <= stop_price:
                    break  # Stopped out first
                if h >= target_price:
                    hit_target = 1
                    break  # Hit target first
                    
            target_col[i] = hit_target
            
        df['target_20_before_m8_45d'] = target_col
        df.to_parquet(file, engine="pyarrow")
        
    print("45-day targets added successfully.")

if __name__ == "__main__":
    main()
