#!/usr/bin/env python3
# SCRIPTNAME: VolumeByPrice.V1.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import (
        load_or_download_ticker,
        create_output_directory,
        fix_yfinance_dataframe,
    )
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

def retrieve_to_csv_and_reload(ticker: str, period=None, start_date=None, end_date=None) -> pd.DataFrame:
    if start_date and end_date:
        df = load_or_download_ticker(ticker, start=start_date, end=end_date)
    else:
        df = load_or_download_ticker(ticker, period=period or '1y')
    
    if df is None or df.empty:
        return pd.DataFrame()

    df = fix_yfinance_dataframe(df)
    if 'Date' in df.columns:
        to_save = df.copy()
    else:
        idx_name = df.index.name or 'Date'
        to_save = df.reset_index().rename(columns={idx_name: 'Date'})
    
    # CONSTRAINT: Save to /dev/shm via data_retrieval logic
    out_dir = create_output_directory(ticker)
    out_csv = os.path.join(out_dir, f"{ticker}.csv")
    to_save.to_csv(out_csv, index=False)
    
    re = pd.read_csv(out_csv, parse_dates=['Date'])
    re.sort_values('Date', inplace=True)
    re.set_index('Date', inplace=True)
    return re

def _prepare_ohlcv(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    df['Volume'] = df['Volume'].fillna(0)
    return df.sort_index()

def plot_stock_with_vbp_plotly(ticker, period="1y", start_date=None, end_date=None, num_bins=20, show_plot=True):
    data = retrieve_to_csv_and_reload(ticker, period=period, start_date=start_date, end_date=end_date)
    data = _prepare_ohlcv(data)
    if data is None or data.empty:
        print(f"No usable OHLCV found for {ticker}.")
        return

    idx_min, idx_max = data.index.min(), data.index.max()
    start_lbl = idx_min.strftime('%Y-%m-%d')
    end_lbl   = idx_max.strftime('%Y-%m-%d')

    min_price = float(np.nanmin(data['Low'].values))
    max_price = float(np.nanmax(data['High'].values))
    if not np.isfinite(min_price) or not np.isfinite(max_price) or min_price >= max_price or num_bins < 1:
        print("Insufficient price range to compute VBP.")
        return

    price_bins = np.linspace(min_price, max_price, num_bins + 1)

    vbp = np.zeros(num_bins, dtype=float)
    for i in range(num_bins):
        lo, hi = price_bins[i], price_bins[i + 1]
        mask = (data['Low'] <= hi) & (data['High'] >= lo)
        vol_sum = data.loc[mask, 'Volume'].sum()
        vbp[i] = 0.0 if np.isnan(vol_sum) else float(vol_sum)

    vmax = vbp.max()
    vbp_norm = (vbp / vmax) if vmax > 0 else np.zeros_like(vbp)

    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], row_heights=[0.7, 0.3],
                        vertical_spacing=0.03, horizontal_spacing=0.02)

    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlestick'), row=1, col=1)

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightgray', showlegend=False), row=2, col=1)

    fig.add_trace(go.Bar(y=price_bins[:-1], x=vbp_norm, orientation='h', name='VBP', opacity=0.4,
                         width=(price_bins[1] - price_bins[0]), showlegend=False), row=1, col=2)

    fig.update_layout(title=f'{ticker} — Candles + Volume by Price ({start_lbl} to {end_lbl})',
                      xaxis_title='Date', yaxis_title='Price', xaxis2_title='VBP (normalized)',
                      yaxis2_visible=False, xaxis_rangeslider_visible=False)
    
    # CONSTRAINT: Output to /dev/shm via data_retrieval logic
    out_dir = create_output_directory(ticker)
    file_path = os.path.join(out_dir, f"{ticker}_VolumeByPrice.html")
    fig.write_html(file_path)
    print(f"Saved interactive chart to: {file_path}")

    if show_plot:
        fig.show()

def main():
    parser = argparse.ArgumentParser(description="Volume by Price (VBP) Analyzer")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--period", default="1y", help="Data period (e.g. 1y, 6mo, max). Default: 1y")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--bins", type=int, default=24, help="Number of price bins for VBP histogram. Default: 24")
    parser.add_argument("--no-show", action="store_true", help="Do not open browser tab")
    
    args = parser.parse_args()
    
    plot_stock_with_vbp_plotly(
        args.ticker.upper(),
        period=args.period,
        start_date=args.start,
        end_date=args.end,
        num_bins=args.bins,
        show_plot=not args.no_show
    )

if __name__ == '__main__':
    main()
