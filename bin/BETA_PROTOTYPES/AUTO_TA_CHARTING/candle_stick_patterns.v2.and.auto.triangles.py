#!/usr/bin/env python3
# SCRIPTNAME: ok.candle_stick_patterns.v2.and.auto.triangles.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import webbrowser
from datetime import timedelta

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

import talib  # TA-Lib must be installed

# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import (
        get_stock_data,
        fix_yfinance_dataframe,
        create_output_directory,
    )
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# Patterns used in both originals
SELECTED_PATTERNS = [
    'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLENGULFING', 'CDLPIERCING',
    'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR',
    'CDLSHOOTINGSTAR', 'CDLHARAMI'
]

def add_selected_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    # TA-Lib expects ndarray-like series
    o = df['Open'].values
    h = df['High'].values
    l = df['Low'].values
    c = df['Close'].values
    for pat in SELECTED_PATTERNS:
        fn = getattr(talib, pat)
        df[pat] = fn(o, h, l, c)
    return df

def _save_png(fig: go.Figure, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.write_image(path, width=1920, height=1080, scale=4)
        print(f"[png] {path}")
    except Exception as e:
        # Preserve functionality without hard-failing if kaleido isn't present
        print(f"[png skipped] {e} (install 'kaleido' to enable PNG export)")

def _save_html_and_open(fig: go.Figure, path: str, open_in_browser: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pio.write_html(fig, file=path, auto_open=False, include_plotlyjs='cdn')
    print(f"[html] {path}")
    if open_in_browser:
        webbrowser.open_new_tab(f"file://{os.path.abspath(path)}")

def _add_candlestick(fig: go.Figure, df: pd.DataFrame, name: str = 'Candlesticks') -> None:
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name=name
    ))

def _add_pattern_markers_value(fig: go.Figure, df: pd.DataFrame) -> None:
    # Markers whose y-values are the pattern outputs (±100/0) — matches 013 behavior
    for pat in SELECTED_PATTERNS:
        fig.add_trace(go.Scatter(x=df.index, y=df[pat], mode='markers', name=pat[3:]))

def _add_pattern_markers_at_close(fig: go.Figure, df: pd.DataFrame) -> None:
    # Markers positioned at the close price on signal dates — matches 014 behavior
    for pat in SELECTED_PATTERNS:
        idx = df.index[df[pat] != 0]
        if len(idx) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=idx,
            y=df.loc[idx, 'Close'],
            mode='markers',
            name=pat[3:]
        ))

def plot_full_trend_patterns_value(df: pd.DataFrame, ticker: str, html_dir: str, img_dir: str,
                                   open_tabs: bool = True, save_png: bool = True) -> None:
    fig = go.Figure()
    _add_candlestick(fig, df)

    # 6-month peak line to last point
    six_months_ago = df.index[-1] - timedelta(days=180)
    last_6m = df.loc[six_months_ago:]
    if not last_6m.empty:
        peak_date = last_6m['Close'].idxmax()
        peak_val = float(last_6m['Close'].max())
        fig.add_trace(go.Scatter(
            x=[peak_date, df.index[-1]],
            y=[peak_val, float(df['Close'].iloc[-1])],
            mode='lines',
            name='Trend Line - Peaks'
        ))

    # 1-year trough line to last point
    one_year_ago = df.index[-1] - timedelta(days=365)
    last_1y = df.loc[one_year_ago:]
    if not last_1y.empty:
        low_date = last_1y['Close'].idxmin()
        low_val = float(last_1y['Close'].min())
        fig.add_trace(go.Scatter(
            x=[low_date, df.index[-1]],
            y=[low_val, float(df['Close'].iloc[-1])],
            mode='lines',
            name='Trend Line - Troughs'
        ))

    _add_pattern_markers_value(fig, df)
    fig.update_layout(
        title=f"{ticker} · Full Period · Trend + Candlestick Patterns (value markers)",
        xaxis_title='Date', yaxis_title='Price'
    )

    html_path = os.path.join(html_dir, f"{ticker}_full_trend_patterns_value.html")
    _save_html_and_open(fig, html_path, open_tabs)
    if save_png:
        _save_png(fig, os.path.join(img_dir, f"{ticker}_full_trend_patterns_value.png"))

def plot_last_n_days(df: pd.DataFrame, ticker: str, n_days: int, html_dir: str, img_dir: str,
                     open_tabs: bool = True, save_png: bool = True) -> None:
    sub = df.loc[df.index[-1] - timedelta(days=n_days):]
    if sub.empty:
        print(f"[warn] No data for last {n_days} days")
        return

    fig = go.Figure()
    _add_candlestick(fig, sub)

    # Peak and trough trendlines (to last point in window)
    peak_date = sub['Close'].idxmax()
    peak_val = float(sub['Close'].max())
    fig.add_trace(go.Scatter(
        x=[peak_date, sub.index[-1]],
        y=[peak_val, float(sub['Close'].iloc[-1])],
        mode='lines', name='Trend Line - Peaks'
    ))

    low_date = sub['Close'].idxmin()
    low_val = float(sub['Close'].min())
    fig.add_trace(go.Scatter(
        x=[low_date, sub.index[-1]],
        y=[low_val, float(sub['Close'].iloc[-1])],
        mode='lines', name='Trend Line - Troughs'
    ))

    _add_pattern_markers_value(fig, sub)
    fig.update_yaxes(range=[0, float(sub['High'].max())])  # matches 013
    fig.update_layout(
        title=f"{ticker} · Last {n_days} Days · Trend + Candlestick Patterns (value markers)",
        xaxis_title='Date', yaxis_title='Price'
    )

    html_path = os.path.join(html_dir, f"{ticker}_last{n_days}_trend_patterns_value.html")
    _save_html_and_open(fig, html_path, open_tabs)
    if save_png:
        _save_png(fig, os.path.join(img_dir, f"{ticker}_last{n_days}_trend_patterns_value.png"))

def plot_full_patterns_at_close(df: pd.DataFrame, ticker: str, html_dir: str, img_dir: str,
                                open_tabs: bool = True, save_png: bool = True) -> None:
    fig = go.Figure()
    _add_candlestick(fig, df)
    _add_pattern_markers_at_close(fig, df)
    fig.update_layout(
        title=f"{ticker} · Full Period · Candlestick Patterns (markers @ close)",
        xaxis_title='Date', yaxis_title='Price'
    )

    html_path = os.path.join(html_dir, f"{ticker}_full_patterns_at_close.html")
    _save_html_and_open(fig, html_path, open_tabs)
    if save_png:
        _save_png(fig, os.path.join(img_dir, f"{ticker}_full_patterns_at_close.png"))

def main():
    ap = argparse.ArgumentParser(
        description="Unified candlestick pattern visualizer using data_retrieval.py"
    )
    ap.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    ap.add_argument("--period", default="max", help="yfinance period string (default: max)")
    ap.add_argument("--no-open", action="store_true", help="Do not open browser tabs for HTML outputs")
    ap.add_argument("--no-png", action="store_true", help="Skip PNG export even if kaleido is installed")
    args = ap.parse_args()

    # Load via data_retrieval.py (preserves caching/normalization)
    df = get_stock_data(args.ticker, period=args.period)
    if df is None or df.empty:
        print(f"[error] No data for {args.ticker}")
        sys.exit(1)

    # Ensure clean index/columns
    df = fix_yfinance_dataframe(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Add TA-Lib pattern columns (same selected set as originals)
    df = add_selected_candlestick_patterns(df)

    # Output dirs under dated base path in /dev/shm
    out_base = create_output_directory(args.ticker)
    html_dir = os.path.join(out_base, "html")
    img_dir = os.path.join(out_base, "images")
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    open_tabs = not args.no_open
    save_png = not args.no_png

    # Plots: preserve all distinct behaviors from originals
    plot_full_trend_patterns_value(df, args.ticker, html_dir, img_dir, open_tabs, save_png)  # 013 full
    plot_last_n_days(df, args.ticker, 90, html_dir, img_dir, open_tabs, save_png)            # 013 last 90
    plot_last_n_days(df, args.ticker, 30, html_dir, img_dir, open_tabs, save_png)            # 013 last 30
    plot_full_patterns_at_close(df, args.ticker, html_dir, img_dir, open_tabs, save_png)     # 014 full @ close

if __name__ == "__main__":
    main()
