import os
import glob
import pandas as pd
import numpy as np

INPUT_DIR = "nifty100_labels"
OUTPUT_DIR = "nifty100_features"

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates Phase 3 features strictly using point-in-time data."""
    df = df.copy().sort_index()
    
    # --- 1. Trend Family ---
    # Multi-horizon returns
    for days in [1, 5, 10, 20, 60, 120]:
        df[f'ret_{days}d'] = df['Close'].pct_change(periods=days)
        
    # Simple Moving Average (SMA) distance
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_60'] = df['Close'].rolling(window=60).mean()
    df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
    df['dist_sma_60'] = (df['Close'] - df['sma_60']) / df['sma_60']

    # Exponential Moving Average (EMA) distance
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema_60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['dist_ema_20'] = (df['Close'] - df['ema_20']) / df['ema_20']
    df['dist_ema_60'] = (df['Close'] - df['ema_60']) / df['ema_60']

    # --- 2. Breakout Family ---
    # 52W (252 days) high distance
    df['high_52w'] = df['High'].rolling(window=252).max()
    df['dist_52w_high'] = (df['Close'] - df['high_52w']) / df['high_52w']
    
    # 20D Local Channel Breakout (Price relative to recent high)
    df['high_20d'] = df['High'].rolling(window=20).max()
    df['dist_20d_high'] = (df['Close'] - df['high_20d']) / df['high_20d']

    # Range expansion (Today's range vs 20-day average range)
    df['daily_range'] = df['High'] - df['Low']
    df['avg_range_20d'] = df['daily_range'].rolling(window=20).mean()
    df['range_expansion'] = df['daily_range'] / df['avg_range_20d']

    # --- 3. Volatility Family ---
    df['atr_14'] = calc_atr(df['High'], df['Low'], df['Close'], period=14)
    # Realized volatility (20D rolling standard deviation of daily returns)
    df['realized_vol_20d'] = df['ret_1d'].rolling(window=20).std() * np.sqrt(252)
    # Volatility Expansion (Short-term ATR vs Long-term ATR)
    df['atr_60'] = df['atr_14'].rolling(window=60).mean()
    df['volatility_expansion'] = df['atr_14'] / df['atr_60']

    # --- 4. Volume & Liquidity Family ---
    # Relative volume
    df['sma_vol_20'] = df['Volume'].rolling(window=20).mean()
    df['rel_volume_20d'] = df['Volume'] / df['sma_vol_20']
    
    # Turnover (Execution Liquidity)
    df['turnover'] = df['Close'] * df['Volume']
    df['avg_turnover_20d'] = df['turnover'].rolling(window=20).mean()
    # Turnover Acceleration
    df['turnover_acceleration'] = df['turnover'] / df['avg_turnover_20d']

    # On-Balance Volume (OBV) proxy
    price_change_sign = np.sign(df['Close'].diff())
    df['obv'] = (price_change_sign * df['Volume']).fillna(0).cumsum()
    # Normalize OBV so it's comparable across stocks/time
    df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
    df['dist_obv_20'] = (df['obv'] - df['obv_sma_20']) / df['obv_sma_20'].replace(0, np.nan)

    # --- 5. Momentum Family ---
    df['rsi_14'] = calc_rsi(df['Close'], period=14)
    df['roc_20'] = df['Close'].pct_change(periods=20)
    
    # Clean up intermediate raw columns so the feature store remains clean
    cols_to_drop = [
        'sma_20', 'sma_60', 'ema_20', 'ema_60', 'high_52w', 'high_20d', 
        'daily_range', 'avg_range_20d', 'atr_60', 'sma_vol_20', 'turnover', 
        'obv', 'obv_sma_20'
    ]
    df = df.drop(columns=cols_to_drop)
    
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    
    if not parquet_files:
        print(f"No labeled files found in '{INPUT_DIR}'. Run label generation first.")
        return
        
    print(f"Starting Feature Engine across {len(parquet_files)} stocks...")
    success_count = 0
    
    for file_path in parquet_files:
        ticker = os.path.basename(file_path).replace("_labeled.parquet", "")
        try:
            df = pd.read_parquet(file_path)
            
            # Generate features
            feature_df = generate_features(df)
            
            # Save the final dataset combining OHLCV, Labels, and Features
            out_file = os.path.join(OUTPUT_DIR, f"{ticker}_features.parquet")
            feature_df.to_parquet(out_file, engine="pyarrow")
            success_count += 1
            
        except Exception as e:
            print(f"Error processing features for {ticker}: {e}")
            
    print(f"Feature generation completed: {success_count}/{len(parquet_files)} stocks successfully processed.")

if __name__ == "__main__":
    main()
