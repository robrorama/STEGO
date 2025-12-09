#!/usr/bin/env python3
import sys
import os

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

from pathlib import Path
import pandas as pd
import yfinance as yf

# Import the cache path helper from your data_retrieval module
try:
    from data_retrieval import get_local_cache_path  # uses BASE_CACHE_PATH from data_retrieval.py
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

def main():
    # Keep the same behavior as before, default to AAPL if no arg is provided
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    # Reuse your data_retrieval cache path logic; use a stable key for earnings data
    # This ensures the path defaults to /dev/shm/cache/...
    cache_path = Path(get_local_cache_path(ticker, period="earnings_dates"))

    # 1) If cached file exists, read it from disk
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Warning: Could not read cache {cache_path}: {e}. Re-downloading.")
            df = None
    else:
        df = None

    # 2) Otherwise download from yfinance, persist to disk, then read back from disk
    if df is None:
        try:
            df_dl = yf.Ticker(ticker).earnings_dates
        except Exception as e:
            sys.exit(f"Error fetching earnings dates: {e}")

        if df_dl is None or df_dl.empty:
            sys.exit(f"No earnings_dates returned for {ticker}")
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_dl.to_csv(cache_path)
        
        # Strict reload from disk to satisfy protocol
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)

    # 3) Rest of the logic (unchanged intent): print the table
    print(df)

if __name__ == "__main__":
    main()
