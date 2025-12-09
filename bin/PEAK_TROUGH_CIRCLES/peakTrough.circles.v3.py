#!/usr/bin/env python3
# SCRIPTNAME: peakTrough.circles.v3.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Visualizes Peaks and Troughs with automated Trendlines and Multiscale Linear Regressions.
#   - Uses data_retrieval.py
#   - Uses Plotly

import sys
import os

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# --- Logic ---
def find_peaks_troughs(df: pd.DataFrame, order: int = 5):
    highs, lows = df['High'].to_numpy(), df['Low'].to_numpy()
    peaks, troughs = [], []
    n = len(df)
    for i in range(order, n - order):
        if highs[i] == np.max(highs[i - order:i + order + 1]):
            peaks.append((i, highs[i]))
        if lows[i] == np.min(lows[i - order:i + order + 1]):
            troughs.append((i, lows[i]))
    return peaks, troughs

def _build_lr_line(points, window_bars: int, n_bars: int, df_index):
    if not points or window_bars <= 0: return [], []
    # Select points in window
    start_idx = max(0, n_bars - 1 - window_bars + 1)
    subset = [(i, y) for (i, y) in points if i >= start_idx]
    
    if len(subset) < 2: return [], []
    
    x = np.array([i for (i, _) in subset], dtype=float)
    y = np.array([y for (_, y) in subset], dtype=float)
    m, b = np.polyfit(x, y, 1)
    
    x_start, x_end = int(x.min()), n_bars - 1
    return [df_index[x_start], df_index[x_end]], [m * x_start + b, m * x_end + b]

# --- Visualization ---
def create_figure(df, ticker, peaks, troughs, args):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

    # Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                 low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)

    # Peaks/Troughs
    if peaks:
        fig.add_trace(go.Scatter(x=[df.index[i] for i, _ in peaks], y=[p for _, p in peaks],
                                 mode='markers', marker=dict(symbol='circle', size=15, color='rgba(0,255,0,0.3)', line=dict(color='green', width=2)),
                                 name='Peaks'), row=1, col=1)
    if troughs:
        fig.add_trace(go.Scatter(x=[df.index[i] for i, _ in troughs], y=[p for _, p in troughs],
                                 mode='markers', marker=dict(symbol='circle', size=15, color='rgba(255,0,0,0.3)', line=dict(color='red', width=2)),
                                 name='Troughs'), row=1, col=1)

    # Multiscale LR Lines
    n = len(df)
    for label, days, color in [("Short", args.short_days, "green"), ("Med", args.medium_days, "blue"), ("Long", args.long_days, "purple")]:
        # Peak Lines
        xs, ys = _build_lr_line(peaks, days, n, df.index)
        if xs: fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=color, width=2, dash='dot'), name=f'{label} Peak LR'), row=1, col=1)
        # Trough Lines
        xs, ys = _build_lr_line(troughs, days, n, df.index)
        if xs: fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=color, width=2), name=f'{label} Trough LR'), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='grey'), row=2, col=1)

    fig.update_layout(title=f"{ticker} Peaks & Troughs Analysis", xaxis_rangeslider_visible=False, height=900)
    return fig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("--period", default="1y")
    p.add_argument("--dates", help="YYYY-MM-DD,YYYY-MM-DD")
    p.add_argument("--order", type=int, default=5, help="Pivot detection window")
    p.add_argument("--short-days", type=int, default=60)
    p.add_argument("--medium-days", type=int, default=120)
    p.add_argument("--long-days", type=int, default=252)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    # Load Data
    if args.dates:
        start, end = args.dates.split(",")
        df = dr.load_or_download_ticker(args.ticker, start=start.strip(), end=end.strip())
    else:
        df = dr.load_or_download_ticker(args.ticker, period=args.period)
    
    if df is None or df.empty: sys.exit("No Data.")
    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)

    peaks, troughs = find_peaks_troughs(df, args.order)
    fig = create_figure(df, args.ticker, peaks, troughs, args)

    # CONSTRAINT: Output to /dev/shm via dr.create_output_directory
    outdir = dr.create_output_directory(args.ticker)
    path = os.path.join(outdir, f"{args.ticker}_peaks_troughs.html")
    try:
        fig.write_html(path)
        print(f"Saved chart to {path}")
    except Exception as e:
        print(f"Error saving chart: {e}")
    
    if not args.no_show: fig.show()

if __name__ == "__main__":
    main()
