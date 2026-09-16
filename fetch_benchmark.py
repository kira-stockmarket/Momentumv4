import yfinance as yf
import pandas as pd
import os

def main():
    ticker = "^NSEI"
    output_dir = "benchmark_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading benchmark data for {ticker}...")
    try:
        df = yf.download(ticker, period="max", progress=False)
        
        if not df.empty:
            # Flatten MultiIndex headers if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            df.columns = df.columns.astype(str)
            
            file_path = os.path.join(output_dir, "NSEI.parquet")
            df.to_parquet(file_path, engine="pyarrow")
            print("Successfully saved NSEI.parquet")
        else:
            print("No data found for ^NSEI")
            
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

if __name__ == "__main__":
    main()
