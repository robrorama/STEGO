#!/usr/bin/env python3
# SCRIPTNAME: clickable_trends_compare_two.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
from datetime import timedelta
import pandas as pd

# Plotly for browser tabs
import plotly.graph_objects as go
import plotly.io as pio

# Matplotlib for clickable line mode
import matplotlib.pyplot as plt

from scipy.stats import linregress

# **Single data access layer**
# Ensure data_retrieval.py is in the path
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# ---------- utilities ----------

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def _dt_index_to_numeric_seconds(idx: pd.DatetimeIndex):
    # robust conversion across pandas versions
    try:
        return (idx.view('int64') // 10**9)
    except Exception:
        return (idx.astype('int64') // 10**9)

def add_linear_regression_bands(df: pd.DataFrame) -> pd.DataFrame:
    # df['Relative_Close'] must exist
    idx_num = _dt_index_to_numeric_seconds(df.index)
    slope, intercept, _, _, _ = linregress(idx_num, df['Relative_Close'])
    df['Linear_Reg'] = intercept + slope * idx_num
    df['Residuals'] = df['Relative_Close'] - df['Linear_Reg']
    std = df['Residuals'].std()

    vals = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    for i, v in enumerate(vals, start=1):
        df[f'Reg_High_{i}std'] = df['Linear_Reg'] + std * v
        df[f'Reg_Low_{i}std']  = df['Linear_Reg'] - std * v
    return df

def add_ema(df: pd.DataFrame, periods=(20, 50, 100, 200, 300)) -> pd.DataFrame:
    for p in periods:
        df[f'EMA_{p}'] = df['Relative_Close'].ewm(span=p, adjust=False).mean()
    return df

def add_support_resistance(df: pd.DataFrame):
    last = df.tail(30)  # "last 30 days" in the originals -> last 30 observations
    return last['Relative_Close'].max(), last['Relative_Close'].min()

def add_peak_trough_trend_lines(df: pd.DataFrame):
    end = df.index[-1]
    six_months_ago = end - pd.Timedelta(days=180)
    last_six = df.loc[df.index >= six_months_ago]
    hi_date = last_six['Relative_Close'].idxmax()
    hi_val  = last_six.loc[hi_date, 'Relative_Close']

    one_year_ago = end - pd.Timedelta(days=365)
    last_year = df.loc[df.index >= one_year_ago]
    lo_date = last_year['Relative_Close'].idxmin()
    lo_val  = last_year.loc[lo_date, 'Relative_Close']

    return (hi_date, hi_val), (lo_date, lo_val)

def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == 'max':
        return df
    end = df.index.max()
    days = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}.get(period)
    if days is None:
        return df
    start = end - pd.Timedelta(days=days)
    return df.loc[(df.index >= start) & (df.index <= end)]

def _safe_write_image(fig: go.Figure, path: str):
    try:
        # Path is guaranteed to be in /dev/shm via dr.create_output_directory
        fig.write_image(path, width=1920, height=1080, scale=2)
        print(f"[saved] {path}")
    except Exception as e:
        print(f"[warn] write_image failed ({e}) — skipping image export")

# ---------- plotting ----------

def plot_normalized(df1: pd.DataFrame,
                    df2: pd.DataFrame,
                    t1: str, t2: str,
                    tag: str,
                    outdir: str,
                    save_images: bool,
                    renderer: str,
                    show: bool):
    pio.renderers.default = renderer
    s1 = (df1['Close'] - df1['Close'].mean()) / df1['Close'].std()
    s2 = (df2['Close'] - df2['Close'].mean()) / df2['Close'].std()
    s1, s2 = s1.align(s2, join='inner')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s1.index, y=s1, mode='lines', name=t1))
    fig.add_trace(go.Scatter(x=s2.index, y=s2, mode='lines', name=t2))
    fig.update_layout(
        title=f"Normalized Close: {t1} vs {t2} ({tag})",
        xaxis_title='Date', yaxis_title='Z-score',
        xaxis=dict(type='date')
    )
    if save_images:
        os.makedirs(outdir, exist_ok=True)
        _safe_write_image(fig, os.path.join(outdir, f"{t1}_{t2}_normalized_{tag}.png"))
    if show:
        fig.show()

def plot_ratio(df: pd.DataFrame,
               t1: str, t2: str,
               tag: str,
               outdir: str,
               save_images: bool,
               renderer: str,
               show: bool):
    pio.renderers.default = renderer
    fig = go.Figure()

    vals = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    fig.add_trace(go.Scatter(x=df.index, y=df['Relative_Close'], mode='lines', name='Relative Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Reg'], mode='lines', name='Linear Regression'))
    for i, v in enumerate(vals, start=1):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_High_{i}std'], mode='lines', line=dict(dash='dot'), name=f'+{v}σ'))
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_Low_{i}std'],  mode='lines', line=dict(dash='dot'), name=f'-{v}σ'))

    for p in (20, 50, 100, 200, 300):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{p}'], mode='lines', name=f'EMA {p}'))

    res, sup = add_support_resistance(df)
    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[res, res], mode='lines', name='Resistance'))
    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[sup, sup], mode='lines', name='Support'))

    (hd, hv), (ld, lv) = add_peak_trough_trend_lines(df)
    fig.add_trace(go.Scatter(x=[hd, df.index[-1]], y=[hv, df['Relative_Close'].iloc[-1]], mode='lines', line=dict(dash='dash'), name='Trend (peaks)'))
    fig.add_trace(go.Scatter(x=[ld, df.index[-1]], y=[lv, df['Relative_Close'].iloc[-1]], mode='lines', line=dict(dash='dash'), name='Trend (troughs)'))

    fig.update_layout(
        title=f"{t1}/{t2} Relative Price Analysis ({tag})",
        xaxis_title='Date', yaxis_title='Ratio',
        xaxis=dict(rangeslider=dict(visible=True), type='date')
    )

    if save_images:
        os.makedirs(outdir, exist_ok=True)
        _safe_write_image(fig, os.path.join(outdir, f"{t1}_{t2}_ratio_{tag}.png"))
    if show:
        fig.show()

def plot_clickable_matplotlib(df: pd.DataFrame, t1: str, t2: str, tag: str):
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['Relative_Close'], label='Relative Price', linestyle='-')
    vals = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    for i, v in enumerate(vals, start=1):
        plt.plot(df.index, df[f'Reg_High_{i}std'], linestyle=':', label=f'+{v}σ')
        plt.plot(df.index, df[f'Reg_Low_{i}std'],  linestyle=':', label=f'-{v}σ')
    for p in (20, 50, 100, 200, 300):
        plt.plot(df.index, df[f'EMA_{p}'], label=f'EMA {p}')
    res, sup = add_support_resistance(df)
    plt.axhline(y=res, color='r', label='Resistance')
    plt.axhline(y=sup, color='g', label='Support')
    (hd, hv), (ld, lv) = add_peak_trough_trend_lines(df)
    plt.plot([hd, df.index[-1]], [hv, df['Relative_Close'].iloc[-1]], linestyle='--', label='Trend (peaks)')
    plt.plot([ld, df.index[-1]], [lv, df['Relative_Close'].iloc[-1]], linestyle='--', label='Trend (troughs)')

    clicked = []
    def on_click(event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            clicked.append((x, y))
            print(f"clicked: ({x}, {y})")
            if len(clicked) == 2:
                (x0, y0), (x1, y1) = clicked
                plt.plot([x0, x1], [y0, y1], linestyle='--', label='User Line')
                plt.legend()
                plt.draw()
                clicked.clear()

    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.title(f"{t1}/{t2} Ratio (click two points to draw a line) [{tag}]")
    plt.xlabel('Date'); plt.ylabel('Ratio'); plt.legend(loc='best'); plt.grid(True)
    plt.show()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Unified options charting using data_retrieval.py")
    ap.add_argument('ticker1')
    ap.add_argument('ticker2')
    ap.add_argument('--period', default='1y',
                    help='yfinance period (1mo,3mo,6mo,1y,max) OR "YYYY-MM-DD,YYYY-MM-DD" date range')
    ap.add_argument('--renderer', default='browser',
                    help='plotly renderer (e.g., browser, notebook); each fig opens a separate tab/window')
    ap.add_argument('--no-normalized', action='store_true', help='skip normalized overlay plot')
    ap.add_argument('--no-ratio', action='store_true', help='skip main ratio plot')
    ap.add_argument('--all-timeframes', action='store_true',
                    help='also open ratio charts for comma-separated --timeframes')
    ap.add_argument('--timeframes', default='1mo,3mo,6mo,1y,max',
                    help='used with --all-timeframes (default: 1mo,3mo,6mo,1y,max)')
    ap.add_argument('--clickable', action='store_true',
                    help='open a Matplotlib window for interactive user-drawn line on the ratio chart')
    ap.add_argument('--no-show', action='store_true', help='do not open windows/tabs; only save images')
    ap.add_argument('--no-save', action='store_true', help='do not save PNGs')
    args = ap.parse_args()

    t1 = args.ticker1.upper()
    t2 = args.ticker2.upper()
    pr = args.period.strip()
    show = (not args.no_show)
    save_images = (not args.no_save)

    # Output dir: daily folder via data_retrieval (points to /dev/shm/data/...)
    out_base = dr.create_output_directory(f"{t1}_{t2}")
    image_dir = os.path.join(out_base, "images")
    os.makedirs(image_dir, exist_ok=True)

    # --- Normalized overlay ---
    if not args.no_normalized:
        if ',' in pr:
            start, end = [s.strip() for s in pr.split(',')]
            df1 = dr.load_or_download_ticker(t1, start=start, end=end)
            df2 = dr.load_or_download_ticker(t2, start=start, end=end)
            tag = f"{start}_to_{end}"
        else:
            df1 = dr.load_or_download_ticker(t1, period=pr)
            df2 = dr.load_or_download_ticker(t2, period=pr)
            tag = pr
        df1 = ensure_datetime_index(df1)
        df2 = ensure_datetime_index(df2)
        plot_normalized(df1, df2, t1, t2, tag, image_dir, save_images, args.renderer, show)

    # --- Ratio base df ---
    need_ratio = (not args.no_ratio) or args.clickable or args.all_timeframes
    if need_ratio:
        # Uses dr logic to pull data from disk/download
        ratio = dr.get_ratio_dataframe(t1, t2, pr)
        if ratio.empty:
            print("[error] ratio dataframe is empty")
            sys.exit(1)
        ratio = ensure_datetime_index(ratio)
        ratio = add_linear_regression_bands(ratio.copy())
        ratio = add_ema(ratio)

    # --- Main ratio chart (Plotly) ---
    if not args.no_ratio:
        tag = pr.replace(',', '_to_')
        plot_ratio(ratio, t1, t2, tag, image_dir, save_images, args.renderer, show)

    # --- Multi-timeframe ratio charts (Plotly) ---
    if args.all_timeframes:
        tfs = [tf.strip() for tf in args.timeframes.split(',') if tf.strip()]
        for tf in tfs:
            sub = filter_by_period(ratio, tf)
            if not sub.empty:
                plot_ratio(sub, t1, t2, tf, image_dir, save_images, args.renderer, show)

    # --- Clickable ratio (Matplotlib) ---
    if args.clickable:
        tag = pr.replace(',', '_to_')
        plot_clickable_matplotlib(ratio, t1, t2, tag)

if __name__ == '__main__':
    main()
