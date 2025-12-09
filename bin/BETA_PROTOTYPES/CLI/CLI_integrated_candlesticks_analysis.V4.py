#!/usr/bin/env python3
# SCRIPTNAME: ok.CLI_integrated_candlesticks_analysis.V4.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Integrated Candlestick Patterns Analysis

This single script combines:
  1) Last 90 Days Up/Down Trends & simple Bullish/Bearish detection
  2) Max-data Candlestick Patterns with Doji detection (body < threshold * range)

Data Retrieval:
  - Uses data_retrieval.py to download/cache OHLCV.
  - ALWAYS writes a CSV to data/YYYY-MM-DD/TICKER/TICKER.csv and then RELOADS
    that CSV before manipulating/analyzing.

Usage examples:
  # Download AAPL (1y), save+reload, run both analyses
  ./integrated_candlesticks.py --ticker AAPL --period 1y --mode both

  # Use explicit date range with downloads via data_retrieval (saved+reloaded)
  ./integrated_candlesticks.py --ticker NVDA --start 2024-01-01 --end 2025-10-01 --mode max

  # Analyze an existing CSV already on disk (no downloads)
  ./integrated_candlesticks.py --csv data/2025-10-01/AAPL/AAPL.csv --mode last90
"""

import argparse
import sys
import os

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# uses your local module
try:
    from data_retrieval import (
        get_stock_data,
        load_or_download_ticker,
        create_output_directory,
        fix_yfinance_dataframe,
    )
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _ensure_numeric_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce OHLCV columns to numeric if present."""
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV with a 'Date' column."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"])
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    if "Date" not in df.columns:
        print("CSV must contain a 'Date' column.", file=sys.stderr)
        sys.exit(1)
    df = _ensure_numeric_ohlc(df)
    return df

def retrieve_to_csv_and_reload(ticker: str, period: str = None, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Retrieve via data_retrieval, write to data/YYYY-MM-DD/TICKER/TICKER.csv,
    then RELOAD that CSV and return the DataFrame.
    """
    # get/calc dataframe using data_retrieval API
    if start and end:
        df = load_or_download_ticker(ticker, start=start, end=end)
    else:
        if period is None:
            period = "1y"
        df = get_stock_data(ticker, period=period)

    if df is None or df.empty:
        print(f"No data found for {ticker}")
        sys.exit(1)

    # sanitize columns (handles MultiIndex etc.)
    df = fix_yfinance_dataframe(df)

    # ensure Date column for saving
    if "Date" in df.columns:
        df_to_save = df.copy()
    else:
        df_to_save = df.reset_index().rename(columns={"index": "Date"})
        # Sometimes yfinance uses DatetimeIndex.name == 'Date'
        if "Date" not in df_to_save.columns and df.index.name:
            df_to_save = df.reset_index().rename(columns={df.index.name: "Date"})

    # ensure dtypes
    df_to_save = _ensure_numeric_ohlc(df_to_save)

    # write to dated output directory (CONSTRAINT: /dev/shm via data_retrieval)
    out_dir = create_output_directory(ticker)
    out_csv = os.path.join(out_dir, f"{ticker}.csv")
    df_to_save.to_csv(out_csv, index=False)

    # REQUIRED: reload from disk before manipulation
    return load_csv(out_csv)

# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def last90_analysis(df: pd.DataFrame) -> None:
    """
    Compute last-90-days Uptrend/Downtrend and simple Bullish/Bearish candle label.
    Mirrors behavior integrated earlier, but on a clean, reloaded DataFrame.
    """
    required = ("Open", "Close", "Date")
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    today = datetime.today()
    ninety_days_ago = today - timedelta(days=90)
    recent = df[df["Date"] >= ninety_days_ago].copy()
    if recent.empty:
        print("No data found for last 90 days.")
        return

    recent["Trend"] = np.where(recent["Close"] > recent["Open"], "Uptrend", "Downtrend")
    recent["CandlePattern"] = np.where(recent["Close"] > recent["Open"], "Bullish", "Bearish")

    print("\nLast 90 Days Analysis (last 10 rows):")
    print(recent[["Date", "Trend", "CandlePattern"]].tail(10).to_string(index=False))

def max_data_analysis(df: pd.DataFrame, doji_threshold: float = 0.10) -> None:
    """
    Compute 'Doji' when |Close-Open| < doji_threshold * (High-Low);
    else Bullish if Close>Open, otherwise Bearish.
    """
    required = ("Open", "Close", "High", "Low", "Date")
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    def pattern(row):
        day_range = row["High"] - row["Low"]
        if pd.isna(day_range) or day_range == 0:
            return "Undefined"
        body = abs(row["Close"] - row["Open"])
        if body < doji_threshold * day_range:
            return "Doji"
        return "Bullish" if row["Close"] > row["Open"] else "Bearish"

    df = df.copy()
    df["CandlePattern"] = df.apply(pattern, axis=1)

    print("\nMax Data Analysis (last 10 rows):")
    print(df[["Date", "CandlePattern"]].tail(10).to_string(index=False))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Integrated Candlestick Analyses (download via data_retrieval.py, save+reload, then analyze).")
    mode_choices = ("last90", "max", "both")
    p.add_argument("--mode", choices=mode_choices, default="both",
                   help="Which analysis to run (default: both)")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", help="Ticker to download via data_retrieval and analyze")
    src.add_argument("--csv", help="Path to an existing CSV with Date,Open,High,Low,Close[,Volume]")

    p.add_argument("--period", default="1y",
                   help="If --ticker is used without --start/--end, download period (default: 1y)")
    p.add_argument("--start", help="Start date YYYY-MM-DD (requires --end)")
    p.add_argument("--end", help="End date YYYY-MM-DD (requires --start)")

    p.add_argument("--doji-threshold", type=float, default=0.10,
                   help="Doji body/range threshold for max analysis (default: 0.10)")

    return p.parse_args()

def main():
    args = parse_args()

    if args.csv:
        df = load_csv(args.csv)
    else:
        # validate date args if provided
        if (args.start and not args.end) or (args.end and not args.start):
            print("If specifying a date range, both --start and --end are required.", file=sys.stderr)
            sys.exit(1)
        df = retrieve_to_csv_and_reload(
            ticker=args.ticker,
            period=None if (args.start and args.end) else args.period,
            start=args.start,
            end=args.end,
        )

    if args.mode in ("last90", "both"):
        last90_analysis(df)
    if args.mode in ("max", "both"):
        max_data_analysis(df, doji_threshold=args.doji_threshold)

if __name__ == "__main__":
    main()
