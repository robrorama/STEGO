#!/usr/bin/env python3
# SCRIPTNAME: unified.darvas.v3.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys
import os

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Use ONLY data_retrieval.py for downloads and output directories
# This ensures compliance with the Stego Financial Framework
try:
    from data_retrieval import (
        get_stock_data,          # fetches & caches OHLCV
        add_moving_averages,     # adds baseline SMAs (9,10,20,50,100,200,300)
        create_output_directory  # daily/ticker output folder
    )
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ----------------------------------------------------------------------
# Pivot-high detection "apparent the day after":
# A bar i is a pivot high if High[i] > High[i+1]. We also require High[i] > High[i-1]
# to avoid flat/plateau tops. This matches "top is apparent the day after once it has fallen".
# ----------------------------------------------------------------------
def _pivot_high_indices_day_after(H: np.ndarray) -> list:
    n = len(H)
    pivots = []
    for i in range(1, n - 1):
        if H[i] > H[i - 1] and H[i] > H[i + 1]:
            pivots.append(i)
    return pivots

# ----------------------------------------------------------------------
# Darvas boxes per the spec:
#  - start at the pivot high date (top day)
#  - bottom = min low AFTER the top and BEFORE breakout
#  - breakout day = first day High > top
#  - end = breakout day (inclusive)
#  - if no breakout, skip (prevents boxes extending to current day)
# ----------------------------------------------------------------------
def find_darvas_boxes(df: pd.DataFrame):
    if df.empty:
        return []

    H = df['High'].to_numpy(dtype=float)
    L = df['Low'].to_numpy(dtype=float)
    idx = df.index
    n = len(df)

    boxes = []
    pivots = _pivot_high_indices_day_after(H)

    for i in pivots:
        top = H[i]
        t_top = i

        # search for first breakout ABOVE the originally drawn top
        t_breakout = None
        for j in range(i + 1, n):
            if H[j] > top:
                t_breakout = j
                break

        # must have at least one bar between top and breakout to define a bottom
        if t_breakout is None or t_breakout <= i + 1:
            continue

        # bottom is the minimum low strictly after top and strictly before breakout
        window_lows = L[i + 1 : t_breakout]
        if window_lows.size == 0:
            continue
        rel_min = int(np.nanargmin(window_lows))
        t_bottom = (i + 1) + rel_min
        bottom = L[t_bottom]

        # sanity: ensure bottom < top
        if not np.isfinite(bottom) or bottom >= top:
            continue

        boxes.append({
            "top": float(top),
            "bottom": float(bottom),
            "start": idx[t_top],        # start on the day of the top
            "end": idx[t_breakout],     # end on the breakout day (inclusive)
            "pivot": idx[t_top],
            "bottom_date": idx[t_bottom]
        })

    return boxes

# ----------------------------------------------------------------------
# Moving-average helpers
# data_retrieval.add_moving_averages adds baseline SMAs (9,10,20,50,100,200,300).
# We then add any extra windows required by your two original scripts.
# ----------------------------------------------------------------------
def add_missing_smas(df: pd.DataFrame, extra_windows):
    df = add_moving_averages(df)  # from data_retrieval.py
    if 'Close' not in df.columns:
        return df
    for w in sorted(set(extra_windows)):
        col = f"SMA_{w}"
        if col not in df.columns:
            df[col] = df['Close'].rolling(window=w).mean()
    return df

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def build_plot(df: pd.DataFrame, ticker: str, windows, darvas_boxes, title_suffix: str):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name=f'{ticker} Price'
    ))

    # SMAs
    for w in windows:
        col = f"SMA_{w}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'{w}-Day SMA'))

    # Boxes: draw rectangles bounded to [start, end] so they never extend to "today"
    for bx in darvas_boxes:
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=bx['start'], y0=bx['bottom'],
            x1=bx['end'],   y1=bx['top'],
            line=dict(width=1),
            fillcolor="LightSkyBlue",
            opacity=0.20
        )

    fig.update_layout(
        title=f'{ticker} — Darvas Boxes & SMAs — {title_suffix}',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=700,
        width=1200,
        legend_title_text='Overlays'
    )
    fig.update_yaxes(type="log")
    return fig

def write_and_open(fig, out_dir, filename, open_browser=True):
    path = os.path.join(out_dir, filename)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(path)}", new=2)  # open in separate tab
    return path

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified Darvas Box & Multi-Timeframe SMA Visualizer"
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--period", default="max", help="Data lookback period (e.g., 1y, 2y, max). Default: max")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open charts in the browser")
    
    args = parser.parse_args()
    ticker = args.ticker.upper()

    # 1) Data via data_retrieval.py ONLY (Constraint: checks cache first)
    df = get_stock_data(ticker, period=args.period)

    if df is None or df.empty:
        print(f"Error: No data found for {ticker}.")
        sys.exit(1)

    # 2) SMA sets from both originals
    #    V2: with.many.moving.averages.V2.py ; V1: working.darvas.boxes.V1.py
    windows_v2 = [4, 20, 50, 75, 100, 125, 150, 175, 200, 225, 250]   # V2 set
    windows_v1 = [9, 15, 20, 50, 100, 200]                            # V1 set
    windows_all = sorted(set(windows_v2 + windows_v1))
    df = add_missing_smas(df, windows_all)

    # 3) Darvas boxes per your spec
    darvas_boxes = find_darvas_boxes(df)

    # 4) Output directory via data_retrieval.py (Constraint: /dev/shm)
    out_dir = create_output_directory(ticker)

    # 5) Two figure variants in separate tabs
    fig_v2 = build_plot(df, ticker, windows_v2, darvas_boxes, "Many MAs (V2 set)")
    fig_v1 = build_plot(df, ticker, windows_v1, darvas_boxes, "Compact MAs (V1 set)")

    p2 = write_and_open(fig_v2, out_dir, f"{ticker}_darvas_many_MAs_V2.html", not args.no_browser)
    p1 = write_and_open(fig_v1, out_dir, f"{ticker}_darvas_compact_MAs_V1.html", not args.no_browser)

    print(f"Analysis complete for {ticker}")
    print(f"  [Saved] {p2}")
    print(f"  [Saved] {p1}")
    if not args.no_browser:
        print("  [Info] Opened plots in browser tabs.")

if __name__ == "__main__":
    main()
