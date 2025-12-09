#!/usr/bin/env python3
# File: master.third-hyperbolic.py

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import load_or_download_ticker, create_output_directory, get_local_cache_path
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

def download_stock_data(ticker, period="1y"):
    """
    Ensures the shared cache is populated, then reads the cached CSV back from disk.
    """
    _ = load_or_download_ticker(ticker, period=period)
    cache_path = get_local_cache_path(ticker, period)
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file not found at {cache_path}")
        sys.exit(1)
        
    df = pd.read_csv(cache_path, parse_dates=True, index_col=0)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    print(f"Data for {ticker} retrieved by cache read: {cache_path}")
    return df

def prepare_data(df):
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df.dropna(inplace=True)
    mean_ret = df['Daily_Return'].mean()
    std_ret  = df['Daily_Return'].std(ddof=0)
    df['Scaled_Return'] = (df['Daily_Return'] - mean_ret) / std_ret
    df['Tanh_Return']   = np.tanh(df['Scaled_Return'])
    df['Momentum_14']   = df['Close'].diff(14)
    df.dropna(inplace=True)
    alpha = 0.01
    df['Momentum_Tanh'] = np.tanh(alpha * df['Momentum_14'])
    df['Momentum_Sinh'] = np.sinh(alpha * df['Momentum_14'])
    df['Momentum_Cosh'] = np.cosh(alpha * df['Momentum_14'])
    return df

def plot_data(df, ticker):
    significance_threshold = 0.8
    significant_days_mask = (df['Tanh_Return'] > significance_threshold) | (df['Tanh_Return'] < -significance_threshold)
    significant_days = df[significant_days_mask]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.5, 0.5],
                        subplot_titles=[f"{ticker.upper()} – Daily Returns vs. Tanh(Scaled Returns)",
                                        "Momentum(14) with Hyperbolic Transforms"])

    fig.add_trace(go.Scatter(x=df.index, y=df['Daily_Return'], mode='lines', name='Daily % Return',
                             line=dict(color='skyblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Tanh_Return'], mode='lines', name='Tanh(Scaled Return)',
                             line=dict(color='magenta')), row=1, col=1)
    fig.add_trace(go.Scatter(x=significant_days.index, y=significant_days['Daily_Return'], mode='markers',
                             name='Significant Days', marker=dict(color='red', size=8, symbol='circle-open')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Momentum_14'], mode='lines', name='Momentum(14)',
                             line=dict(color='lightgreen')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Momentum_Tanh'], mode='lines', name='Momentum Tanh',
                             line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Momentum_Sinh'], mode='lines', name='Momentum Sinh',
                             line=dict(color='cyan')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Momentum_Cosh'], mode='lines', name='Momentum Cosh',
                             line=dict(color='violet')), row=2, col=1)

    fig.update_layout(template='plotly_dark', title=f"{ticker.upper()} – Hyperbolic-Based Visualization",
                      width=1200, height=800)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Daily % Return / Tanh(Return)", row=1, col=1)
    fig.update_yaxes(title_text="Momentum(14) & Hyperbolic Transforms", row=2, col=1)

    # CONSTRAINT: Output to /dev/shm via data_retrieval logic
    output_directory = create_output_directory(ticker)
    image_file = os.path.join(output_directory, f"{ticker}_hyperbolic_analysis.png")
    html_file  = os.path.join(output_directory, f"{ticker}_hyperbolic_analysis.html")
    
    try:
        fig.write_image(image_file)
        print(f"Static image saved to: {image_file}")
    except Exception as e:
        print(f"Warning: Could not save static image (kaleido missing?): {e}")
        
    fig.write_html(html_file)
    print(f"Interactive HTML saved to: {html_file}")
    fig.show()
    return image_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python master.third-hyperbolic.py <TICKER> [PERIOD]")
        print("Example: python master.third-hyperbolic.py AAPL 1y")
        sys.exit(1)

    ticker = sys.argv[1]
    period = sys.argv[2] if len(sys.argv) > 2 else "1y"
    
    df = download_stock_data(ticker, period=period)  # cache-backed read
    df = prepare_data(df)
    plot_data(df, ticker)

if __name__ == '__main__':
    main()
