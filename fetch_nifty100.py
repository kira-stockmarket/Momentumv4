import yfinance as yf
import pandas as pd
import os

def main():
    tickers = [
        "ABB.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
        "ADANIPOWER.NS", "ATGL.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
        "DMART.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
        "BAJAJHLDNG.NS", "BANKBARODA.NS", "BEL.NS", "BHEL.NS", "BPCL.NS",
        "BHARTIARTL.NS", "BOSCHLTD.NS", "BRITANNIA.NS", "CANBK.NS", "CHOLAFIN.NS",
        "CIPLA.NS", "COALINDIA.NS", "COLPAL.NS", "DLF.NS", "DABUR.NS",
        "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GAIL.NS", "GODREJCP.NS",
        "GODREJPROP.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS",
        "HDFCLIFE.NS", "HAVELLS.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HAL.NS",
        "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "ITC.NS",
        "IOC.NS", "IRCTC.NS", "IRFC.NS", "INDUSINDBK.NS", "NAUKRI.NS",
        "INFY.NS", "INDIGO.NS", "JIOFIN.NS", "JSWSTEEL.NS", "JINDALSTEL.NS",
        "KOTAKBANK.NS", "LT.NS", "LTIM.NS", "M&M.NS", "MARICO.NS",
        "MARUTI.NS", "MUTHOOTFIN.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
        "PIDILITIND.NS", "PFC.NS", "POWERGRID.NS", "PNB.NS", "RECLTD.NS",
        "RELIANCE.NS", "SBICARD.NS", "SBILIFE.NS", "SRF.NS", "MOTHERSON.NS",
        "SHREECEM.NS", "SIEMENS.NS", "SBIN.NS", "SUNPHARMA.NS", "TVSMOTOR.NS",
        "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TMPV.NS", "TMCV.NS", "TATAPOWER.NS", "TATASTEEL.NS",
        "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "ULTRACEMCO.NS",
        "VBL.NS", "VEDL.NS", "WIPRO.NS", "ZOMATO.NS", "ZYDUSLIFE.NS"
    ]
    
    output_dir = 'nifty100_data'
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            df = yf.download(ticker, period='max', progress=False)
            
            if not df.empty:
                # Flatten MultiIndex headers (removes the repeating ticker symbol)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Cast all column headers to strings for Parquet compatibility
                df.columns = df.columns.astype(str)
                
                clean_name = ticker.replace('.NS', '')
                file_path = os.path.join(output_dir, f"{clean_name}.parquet")
                
                # Save data to parquet using pyarrow
                df.to_parquet(file_path, engine='pyarrow')
                print(f"Successfully saved {clean_name}.parquet")
            else:
                print(f"No data found for {ticker}")
                
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

if __name__ == "__main__":
    main()
