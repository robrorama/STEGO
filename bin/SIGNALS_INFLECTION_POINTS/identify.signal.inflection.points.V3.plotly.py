#!/usr/bin/env python3
# SCRIPTNAME: identify.signal.inflection.points.V3.plotly.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Identifies signal inflection points using Bollinger Bands and Linear Regression Channels.
#   - Visualizes data exclusively via Plotly.
#   - Loads data exclusively via data_retrieval.py.

import sys
import os

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

def analyze_and_plot(
    data: pd.DataFrame,
    ticker: str,
    moving_avg_periods: list,
    bollinger_period: int,
    std_dev: float,
    regression_period: int,
    title_suffix: str,
    output_dir: str
):
    if "Close" not in data.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    # --- Calculations ---
    # 1. Bollinger Bands
    roll_close = data["Close"].rolling(window=bollinger_period)
    data["BB_Middle"] = roll_close.mean()
    bb_std = roll_close.std()
    data["BB_Upper"] = data["BB_Middle"] + (bb_std * std_dev)
    data["BB_Lower"] = data["BB_Middle"] - (bb_std * std_dev)

    # 2. Simple Moving Averages
    for period in moving_avg_periods:
        data[f"SMA_{period}"] = data["Close"].rolling(window=period).mean()

    # 3. Linear Regression Channel (Moving Window)
    def _lr_mid(x: np.ndarray) -> float:
        idx = np.arange(len(x))
        m, b = np.polyfit(idx, x, 1)
        return m * (len(x) // 2) + b

    data["LR_Mid"] = data["Close"].rolling(window=regression_period).apply(
        lambda x: _lr_mid(np.asarray(x)), raw=False
    )
    lr_std = data["Close"].rolling(window=regression_period).std()
    data["LR_Upper"] = data["LR_Mid"] + lr_std
    data["LR_Lower"] = data["LR_Mid"] - lr_std

    # 4. Signal Detection
    data["Pierced_BB_Upper"] = data["Close"] > data["BB_Upper"]
    data["Pierced_BB_Lower"] = data["Close"] < data["BB_Lower"]
    
    uidx = data.index[data["Pierced_BB_Upper"]]
    lidx = data.index[data["Pierced_BB_Lower"]]

    # --- Plotly Visualization ---
    fig = go.Figure()

    # Bollinger Bands (Filled area)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Upper"],
        line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
        name="BB Upper"
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Lower"],
        line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
        fill='tonexty',  # Fill to BB Upper
        fillcolor='rgba(173, 216, 230, 0.1)',
        name="BB Lower"
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Middle"],
        line=dict(color='rgba(173, 216, 230, 0.8)', dash='dot', width=1),
        name="BB Middle"
    ))

    # Linear Regression Channel
    fig.add_trace(go.Scatter(
        x=data.index, y=data["LR_Upper"],
        line=dict(color='rgba(255, 165, 0, 0.6)', width=1, dash='dot'),
        name="LR Upper"
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["LR_Lower"],
        line=dict(color='rgba(255, 165, 0, 0.6)', width=1, dash='dot'),
        name="LR Lower"
    ))

    # Main Price
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Close"],
        line=dict(color='black', width=1.5),
        name="Close Price"
    ))

    # Moving Averages
    colors = ['purple', 'blue', 'brown']
    for i, period in enumerate(moving_avg_periods):
        col_name = f"SMA_{period}"
        if col_name in data.columns:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name],
                line=dict(color=color, width=1),
                name=f"SMA {period}"
            ))

    # Signal Markers
    if not uidx.empty:
        fig.add_trace(go.Scatter(
            x=uidx, y=data.loc[uidx, "Close"],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name="Pierced +2σ (Sell?)"
        ))
    if not lidx.empty:
        fig.add_trace(go.Scatter(
            x=lidx, y=data.loc[lidx, "Close"],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name="Pierced -2σ (Buy?)"
        ))

    fig.update_layout(
        title=f"{ticker}: Inflection Points (BB + LinReg) - {title_suffix}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        template="plotly_white",
        hovermode="x unified"
    )

    # Output handling: Show and Save to /dev/shm
    fig.show()
    
    # CONSTRAINT: Ensure output is written to the correct /dev/shm path
    out_path = os.path.join(output_dir, f"{ticker}_inflection_points.html")
    try:
        fig.write_html(out_path)
        print(f"Interactive chart saved to: {out_path}")
    except Exception as e:
        print(f"Warning: Could not save HTML to {out_path}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Identify signal points (Plotly Only).")
    ap.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    ap.add_argument("--period", default="6mo", help="Period (e.g., 1y, 6mo). Ignored if --start/--end provided.")
    ap.add_argument("--start", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", help="End date YYYY-MM-DD")
    ap.add_argument("--ma", nargs="+", type=int, default=[20, 50, 200], help="SMA periods.")
    ap.add_argument("--bollinger-period", type=int, default=20)
    ap.add_argument("--std-dev", type=float, default=2.0)
    ap.add_argument("--regression-period", type=int, default=21)
    args = ap.parse_args()

    # Data Retrieval
    print(f"Loading data for {args.ticker}...")
    if args.start and args.end:
        df = dr.load_or_download_ticker(args.ticker, start=args.start, end=args.end)
        suffix = f"{args.start} to {args.end}"
    else:
        df = dr.load_or_download_ticker(args.ticker, period=args.period)
        suffix = args.period

    if df is None or df.empty:
        sys.exit(f"Error: No data found for {args.ticker}")

    # CONSTRAINT: Create standard output directory in /dev/shm
    output_dir = dr.create_output_directory(args.ticker)

    analyze_and_plot(
        df,
        args.ticker,
        args.ma,
        args.bollinger_period,
        args.std_dev,
        args.regression_period,
        suffix,
        output_dir
    )

if __name__ == "__main__":
    main()
