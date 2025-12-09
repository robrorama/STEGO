#!/usr/bin/env python3
# SCRIPTNAME: ok.cliAnalysis.Candlesticks.V3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Unified Candlestick Patterns Analysis Script (Using data_retrieval.py)

Modes:
  - last90: analyze trends/patterns over last 90 days
  - max:    analyze patterns (incl. Doji) over the full fetched period
  - both:   run both

Usage:
  ./unified_candle_analysis.py --ticker <TICKER> [--period <PERIOD>] [--mode <MODE>]
Defaults: --ticker AAPL --period 1y --mode both
"""

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import load_or_download_ticker
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

def run_last90_analysis(data: pd.DataFrame) -> bool:
    print("\n--- Running Last 90 Days Analysis ---")
    for col in ['Open', 'Close']:
        if col not in data.columns:
            print(f"Error: Data must contain '{col}'.")
            return False

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            else:
                data.index = pd.to_datetime(data.index)
        except Exception as e:
            print(f"Error making DatetimeIndex: {e}")
            return False

    today = datetime.today().date()
    ninety_days_ago = today - timedelta(days=90)
    recent = data[data.index.date >= ninety_days_ago].copy()
    if recent.empty:
        print("No data found for the last 90 days.")
        return True

    recent['Trend'] = np.where(recent['Close'] > recent['Open'], 'Uptrend', 'Downtrend')
    recent['CandlePattern'] = np.where(recent['Close'] > recent['Open'], 'Bullish', 'Bearish')

    print("Last 10 days (Date, Trend, Candle Pattern):")
    print(recent[['Trend','CandlePattern']].tail(10).reset_index().to_string(index=False))
    print("--- Last 90 Days Analysis Complete ---")
    return True

def run_max_data_analysis(data: pd.DataFrame) -> bool:
    print("\n--- Running Max Period Pattern Analysis ---")
    for col in ['Open', 'Close', 'High', 'Low']:
        if col not in data.columns:
            print(f"Error: Data must contain '{col}'.")
            return False

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            else:
                data.index = pd.to_datetime(data.index)
        except Exception as e:
            print(f"Error making DatetimeIndex: {e}")
            return False

    def get_pattern(row):
        day_range = row['High'] - row['Low']
        if day_range < 1e-9:
            return "Near Zero Range" if abs(row['Close'] - row['Open']) < 1e-9 else ("Bullish" if row['Close'] > row['Open'] else "Bearish")
        body = abs(row['Close'] - row['Open'])
        return "Doji" if body < 0.1 * day_range else ("Bullish" if row['Close'] > row['Open'] else "Bearish")

    out = data.copy()
    out['CandlePattern'] = out.apply(get_pattern, axis=1)

    print(f"Last 10 entries ({out.index.min().date()} → {out.index.max().date()}) (Date, Candle Pattern):")
    print(out[['CandlePattern']].tail(10).reset_index().to_string(index=False))
    print("--- Max Period Pattern Analysis Complete ---")
    return True

def main():
    p = argparse.ArgumentParser(
        description="Unified Candlestick Patterns Analysis (uses data_retrieval.py)"
    )
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--period", default="1y",
                   help="1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max")
    p.add_argument("--mode", choices=["last90","max","both"], default="both")
    args = p.parse_args()

    print(f"--- Starting Unified Analysis ---")
    print(f"Ticker: {args.ticker}, Period: {args.period}, Mode: {args.mode}")

    # CONSTRAINT: Use data_retrieval logic (checks cache, downloads if needed, returns DF)
    df = load_or_download_ticker(args.ticker, period=args.period)
    
    if df is None or df.empty:
        print(f"Failed to get data for {args.ticker} (Period: {args.period}).")
        sys.exit(1)

    print(f"Data loaded: rows={len(df)}, range={df.index.min()} → {df.index.max()}")

    ok = False
    if args.mode in ("last90","both"):
        ok |= run_last90_analysis(df.copy())
    if args.mode in ("max","both"):
        ok |= run_max_data_analysis(df.copy())
    if not ok:
        print("No analysis completed.")
    print("\n--- Unified Analysis Finished ---")

if __name__ == "__main__":
    main()
