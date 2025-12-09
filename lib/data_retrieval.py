# SCRIPTNAME: data_retrieval.py

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- PATH CONFIG (env-driven) ---
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)

def BASE_DATA_PATH() -> str:
    return _env("BASE_DATA_PATH", "/dev/shm/data")

def BASE_CACHE_PATH() -> str:
    return _env("BASE_CACHE_PATH", "/dev/shm/cache")

def TMP_DIR() -> str:
    return _env("TMP_DIR", "/dev/shm/tmp")

def IMAGES_SUBDIR() -> str:
    val = _env("IMAGES_SUBDIR", "/dev/shm/images")
    return val or "images"

def PNGS_SUBDIR() -> str:
    val = _env("PNGS_SUBDIR", "/dev/shm/PNGS")
    return val or "PNGS"
# --- END PATH CONFIG ---

def fix_yfinance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures columns are single-level (no multiindex).
    Drops 'Adj Close' if it exists and coerces OHLCV to numeric.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if 'Adj Close' in df.columns:
        df.drop(columns=['Adj Close'], inplace=True, errors='ignore')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def create_dated_directory(ticker: str) -> str:
    """
    Creates BASE_DATA_PATH/YYYY-MM-DD/TICKER and returns its path.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    base = BASE_DATA_PATH()
    directory = os.path.join(base, today, ticker)
    os.makedirs(directory, exist_ok=True)
    return directory

def get_local_cache_path(ticker: str, period: str = "1y") -> str:
    """
    Returns a path for caching the raw CSV: BASE_CACHE_PATH/TICKER_<period>.csv
    """
    cache_base = BASE_CACHE_PATH()
    os.makedirs(cache_base, exist_ok=True)
    filename = f"{ticker}_{period}.csv"
    return os.path.join(cache_base, filename)

def load_or_download_ticker(ticker: str, period: str = "1y", start=None, end=None) -> pd.DataFrame:
    """
    If a local CSV exists, load it. Otherwise, download via yfinance, save, and return.
    Cache path is determined by get_local_cache_path().
    """
    if start and end:
        start_str = start.replace('-', '') if isinstance(start, str) else start.strftime('%Y%m%d')
        end_str = end.replace('-', '') if isinstance(end, str) else end.strftime('%Y%m%d')
        cache_key = f"{start_str}_{end_str}"
        cache_path = get_local_cache_path(ticker, cache_key)
    else:
        cache_path = get_local_cache_path(ticker, period)

    df = None
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=True, index_col=0)
        except Exception as e:
            print(f"Error reading cache file {cache_path}: {e}. Will attempt to download.")

    if df is None:
        if start and end:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, actions=False)
        else:
            df = yf.download(ticker, period=period, auto_adjust=False, actions=False)

        if not df.empty:
            df = fix_yfinance_dataframe(df)
            try:
                df.to_csv(cache_path)
            except Exception as e:
                print(f"Error saving cache file {cache_path}: {e}")
        else:
            print(f"Failed to download data for {ticker}.")
            return pd.DataFrame()

    df = fix_yfinance_dataframe(df)
    return df

def create_output_directory(ticker: str) -> str:
    """
    Returns a daily subdir for outputs, e.g. BASE_DATA_PATH/YYYY-MM-DD/TICKER.
    """
    return create_dated_directory(ticker)

def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Retrieves historical stock data using local caching logic.
    """
    return load_or_download_ticker(ticker, period=period)

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds various moving average columns to the DataFrame.
    """
    if 'Close' not in df.columns:
        print("Warning: 'Close' column not found for adding moving averages.")
        return df
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_300'] = df['Close'].rolling(window=300).mean()
    return df

def get_ratio_dataframe(ticker1: str, ticker2: str, date_range_input: str) -> pd.DataFrame:
    """
    Calculate ratio data between two tickers, preserving OHLC structure.
    """
    if ',' in date_range_input:
        start_date, end_date = [d.strip() for d in date_range_input.split(',')]
        df1 = load_or_download_ticker(ticker1, start=start_date, end=end_date)
        df2 = load_or_download_ticker(ticker2, start=start_date, end=end_date)
    else:
        period_str = str(date_range_input).strip()
        df1 = load_or_download_ticker(ticker1, period=period_str)
        df2 = load_or_download_ticker(ticker2, period=period_str)

    if df1.empty or df2.empty:
        print(f"Warning: Could not load data for one or both tickers for ratio: {ticker1}, {ticker2}")
        return pd.DataFrame()

    aligned_df1, aligned_df2 = df1.align(df2, join='inner')

    if aligned_df1.empty or aligned_df2.empty:
        print(f"Warning: No overlapping data for ratio calculation between {ticker1} and {ticker2}")
        return pd.DataFrame()

    ratio_df = pd.DataFrame(index=aligned_df1.index)
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in aligned_df1.columns or col not in aligned_df2.columns:
            print(f"Warning: Column '{col}' missing in one of the dataframes for ratio calculation.")
            ratio_df[col] = np.nan
        else:
            ratio_df[col] = np.where(aligned_df2[col] == 0, np.nan, aligned_df1[col] / aligned_df2[col])

    ratio_df['Relative_Close'] = ratio_df['Close']

    if 'Volume' in aligned_df1.columns and 'Volume' in aligned_df2.columns:
        ratio_df['Volume'] = (
            aligned_df1['Volume'].replace(0, np.nan) /
            aligned_df2['Volume'].replace(0, np.nan)
        )
    else:
        ratio_df['Volume'] = np.nan

    return ratio_df

