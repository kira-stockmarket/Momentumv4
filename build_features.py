import os
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

# Suppress pandas fragmentation warnings
warnings.filterwarnings('ignore')

FEATURE_DIR = "nifty100_features"
BENCHMARK_DIR = "benchmark_data"

# Comprehensive Nifty 100 Sector Mapping (Updated for ETERNAL and Tata Motors Demerger)
SECTOR_MAP = {
    # IT
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT", "TECHM.NS": "IT", 
    "LTM.NS": "IT", "COFORGE.NS": "IT", "PERSISTENT.NS": "IT",
    
    # Banks
    "HDFCBANK.NS": "Bank", "ICICIBANK.NS": "Bank", "SBIN.NS": "Bank", "KOTAKBANK.NS": "Bank", 
    "AXISBANK.NS": "Bank", "INDUSINDBK.NS": "Bank", "BANKBARODA.NS": "Bank", "PNB.NS": "Bank", 
    "IDFCFIRSTB.NS": "Bank", "YESBANK.NS": "Bank", "CANBK.NS": "Bank",
    
    # Auto
    "TMCV.NS": "Auto", "TMPV.NS": "Auto", "M&M.NS": "Auto", "MARUTI.NS": "Auto", "BAJAJ-AUTO.NS": "Auto", 
    "HEROMOTOCO.NS": "Auto", "EICHERMOT.NS": "Auto", "TVSMOTOR.NS": "Auto", "BOSCHLTD.NS": "Auto", 
    "TIINDIA.NS": "Auto", "MOTHERSON.NS": "Auto",
    
    # FMCG
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG", 
    "TATACONSUM.NS": "FMCG", "GODREJCP.NS": "FMCG", "DABUR.NS": "FMCG", "MARICO.NS": "FMCG", 
    "VBL.NS": "FMCG", "COLPAL.NS": "FMCG", "UBL.NS": "FMCG",
    
    # Metals & Mining
    "TATASTEEL.NS": "Metal", "HINDALCO.NS": "Metal", "JSWSTEEL.NS": "Metal", "COALINDIA.NS": "Metal", 
    "VEDL.NS": "Metal", "NMDC.NS": "Metal", "JINDALSTEL.NS": "Metal",
    
    # Pharma & Healthcare
    "SUNPHARMA.NS": "Pharma", "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma", "DIVISLAB.NS": "Pharma", 
    "LUPIN.NS": "Pharma", "APOLLOHOSP.NS": "Pharma", "MAXHEALTH.NS": "Pharma", "TORNTPHARM.NS": "Pharma", 
    "ZYDUSLIFE.NS": "Pharma", "MANKIND.NS": "Pharma", "AUROPHARMA.NS": "Pharma",
    
    # Financial Services (Mapped to Market Benchmark)
    "BAJFINANCE.NS": "Market", "BAJAJFINSV.NS": "Market", "CHOLAFIN.NS": "Market", "SHRIRAMFIN.NS": "Market",
    "MUTHOOTFIN.NS": "Market", "SBICARD.NS": "Market", "HDFCAMC.NS": "Market", "HDFCLIFE.NS": "Market",
    "SBILIFE.NS": "Market", "ICICIGI.NS": "Market", "ICICIPRULI.NS": "Market", "LICI.NS": "Market", 
    "PFC.NS": "Market", "RECLTD.NS": "Market", "IRFC.NS": "Market", "IREDA.NS": "Market", "JIOFIN.NS": "Market",
    
    # Energy, Oil & Gas (Mapped to Market Benchmark)
    "RELIANCE.NS": "Market", "ONGC.NS": "Market", "NTPC.NS": "Market", "POWERGRID.NS": "Market", 
    "BPCL.NS": "Market", "IOC.NS": "Market", "TATAPOWER.NS": "Market", "ADANIGREEN.NS": "Market", 
    "ADANIPOWER.NS": "Market", "ADANIENSOL.NS": "Market", "GAIL.NS": "Market",
    
    # Infrastructure, Cement & Others (Mapped to Market Benchmark)
    "LT.NS": "Market", "ULTRACEMCO.NS": "Market", "GRASIM.NS": "Market", "AMBUJACEM.NS": "Market", 
    "SHREECEM.NS": "Market", "ASIANPAINT.NS": "Market", "BERGEPAINT.NS": "Market", "PIDILITIND.NS": "Market",
    "TITAN.NS": "Market", "BHARTIARTL.NS": "Market", "ADANIENT.NS": "Market", "ADANIPORTS.NS": "Market",
    "HAL.NS": "Market", "BEL.NS": "Market", "SIEMENS.NS": "Market", "ABB.NS": "Market", "CGPOWER.NS": "Market",
    "TRENT.NS": "Market", "DMART.NS": "Market", "INDIGO.NS": "Market", "ETERNAL.NS": "Market", 
    "DLF.NS": "Market", "MACROTECH.NS": "Market", "GODREJPROP.NS": "Market", "SRF.NS": "Market", 
    "HAVELLS.NS": "Market", "POLYCAB.NS": "Market", "CUMMINSIND.NS": "Market"
}

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calc_obv_dist(df, period=20):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_sma = obv.rolling(period).mean()
    obv_std = obv.rolling(period).std() + 1e-9
    return (obv - obv_sma) / obv_std

def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    tickers = list(SECTOR_MAP.keys())
    
    print(f"Building features for {len(tickers)} stocks with Sector Regime Filters...")

    for ticker in tickers:
        try:
            # 1. Download Stock Data
            df = yf.download(ticker, period="max", progress=False)
            if df.empty or len(df) < 252:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            clean_ticker = ticker.replace(".NS", "")
            
            # 2. Core Returns
            df['ret_1d'] = df['Close'].pct_change(1)
            df['ret_5d'] = df['Close'].pct_change(5)
            df['ret_10d'] = df['Close'].pct_change(10)
            df['ret_20d'] = df['Close'].pct_change(20)
            df['ret_60d'] = df['Close'].pct_change(60)
            
            # 3. Moving Average Distances
            df['dist_sma_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
            df['dist_sma_60'] = (df['Close'] - df['Close'].rolling(60).mean()) / df['Close'].rolling(60).mean()
            df['dist_ema_20'] = (df['Close'] - df['Close'].ewm(span=20).mean()) / df['Close'].ewm(span=20).mean()
            df['dist_ema_60'] = (df['Close'] - df['Close'].ewm(span=60).mean()) / df['Close'].ewm(span=60).mean()
            
            # 4. High/Low Distances
            df['dist_52w_high'] = (df['Close'] - df['Close'].rolling(252).max()) / df['Close'].rolling(252).max()
            df['dist_20d_high'] = (df['Close'] - df['Close'].rolling(20).max()) / df['Close'].rolling(20).max()
            
            # 5. Volatility & Ranges
            df['range_expansion'] = (df['High'] - df['Low']) / (df['High'] - df['Low']).rolling(20).mean()
            df['atr_14'] = calc_atr(df, 14)
            df['realized_vol_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
            df['volatility_expansion'] = df['realized_vol_20d'] / df['realized_vol_20d'].rolling(60).mean()
            
            # 6. Volume & Flow
            df['rel_volume_20d'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['turnover_acceleration'] = (df['Volume'] * df['Close']) / (df['Volume'] * df['Close']).rolling(20).mean()
            df['dist_obv_20'] = calc_obv_dist(df, 20)
            
            # 7. Momentum Oscillators
            df['rsi_14'] = calc_rsi(df['Close'], 14)
            df['roc_20'] = df['ret_20d']
            
            # =================================================================
            # 8. MACRO OVERLAY: Market & Sector Relative Strength + Regimes
            # =================================================================
            
            # A. Market Excess Return
            market_path = os.path.join(BENCHMARK_DIR, "Market.parquet")
            if os.path.exists(market_path):
                market_df = pd.read_parquet(market_path)
                market_df['market_ret_1d'] = market_df['Market_Close'].pct_change(1)
                market_df['market_ret_20d'] = market_df['Market_Close'].pct_change(20)
                
                df = df.join(market_df[['market_ret_1d', 'market_ret_20d']], how='left')
                df['excess_ret_1d'] = df['ret_1d'] - df['market_ret_1d']
                df['excess_ret_20d'] = df['ret_20d'] - df['market_ret_20d']
            else:
                df['excess_ret_1d'] = 0.0
                df['excess_ret_20d'] = 0.0
                
            # B. Sector Excess Return & Regime Filter
            assigned_sector = SECTOR_MAP.get(ticker, None)
            if assigned_sector and assigned_sector != "Market":
                sector_path = os.path.join(BENCHMARK_DIR, f"{assigned_sector}.parquet")
                if os.path.exists(sector_path):
                    sdf = pd.read_parquet(sector_path)
                    
                    sdf['sector_ret_20d'] = sdf[f'{assigned_sector}_Close'].pct_change(20)
                    sdf['sector_sma_200'] = sdf[f'{assigned_sector}_Close'].rolling(window=200).mean()
                    sdf['sector_regime_dist'] = (sdf[f'{assigned_sector}_Close'] - sdf['sector_sma_200']) / sdf['sector_sma_200']
                    
                    df = df.join(sdf[['sector_ret_20d', 'sector_regime_dist']], how='left')
                    df['excess_sector_ret_20d'] = df['ret_20d'] - df['sector_ret_20d']
                else:
                    df['excess_sector_ret_20d'] = 0.0
                    df['sector_regime_dist'] = 0.0
            else:
                # If mapped to "Market", use the market as the sector benchmark
                if os.path.exists(market_path):
                    market_df['market_sma_200'] = market_df['Market_Close'].rolling(window=200).mean()
                    market_df['sector_regime_dist'] = (market_df['Market_Close'] - market_df['market_sma_200']) / market_df['market_sma_200']
                    df = df.join(market_df[['sector_regime_dist']], how='left')
                    df['excess_sector_ret_20d'] = df['excess_ret_20d']
                else:
                    df['excess_sector_ret_20d'] = 0.0
                    df['sector_regime_dist'] = 0.0
                
            df = df.drop(columns=['market_ret_1d', 'market_ret_20d', 'sector_ret_20d'], errors='ignore')
            
            # Drop rows where base features aren't calculated yet (burn-in period)
            df = df.dropna(subset=['dist_sma_60', 'realized_vol_20d', 'sector_regime_dist'])
            
            # Save to Parquet
            file_path = os.path.join(FEATURE_DIR, f"{clean_ticker}_features.parquet")
            df.to_parquet(file_path, engine="pyarrow")
            
        except Exception as e:
            print(f"[{ticker}] Failed Feature Engineering: {e}")

    print("Feature generation complete.")

if __name__ == "__main__":
    main()
