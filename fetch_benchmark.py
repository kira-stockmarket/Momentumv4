import yfinance as yf
import pandas as pd
import os

def main():
    # Dictionary mapping sector names to Yahoo Finance tickers for NSE Indices
    indices = {
        "Market": "^NSEI",       # Nifty 50
        "Bank": "^NSEBANK",      # Nifty Bank
        "IT": "^CNXIT",          # Nifty IT
        "Auto": "^CNXAUTO",      # Nifty Auto
        "FMCG": "^CNXFMCG",      # Nifty FMCG
        "Metal": "^CNXMETAL",    # Nifty Metal
        "Pharma": "^CNXPHARMA"   # Nifty Pharma
    }
    
    output_dir = "benchmark_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {len(indices)} Market and Sector indices...")
    
    for name, ticker in indices.items():
        try:
            # Download max available historical data
            df = yf.download(ticker, period="max", progress=False)
            
            if not df.empty:
                # Handle MultiIndex columns if yfinance returns them
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                    
                # Convert column names to strings
                df.columns = df.columns.astype(str)
                
                # We only need the Close price for relative strength and regime filters
                df = df[['Close']].rename(columns={'Close': f'{name}_Close'})
                
                # Save to parquet
                file_path = os.path.join(output_dir, f"{name}.parquet")
                df.to_parquet(file_path, engine="pyarrow")
                
                print(f"[{name}] Downloaded and saved successfully. (Rows: {len(df)})")
            else:
                print(f"[{name}] Warning: No data returned from Yahoo Finance.")
                
        except Exception as e:
            print(f"[{name}] Failed: {e}")

if __name__ == "__main__":
    main()
