#!/usr/bin/env python3
# SCRIPTNAME: cool.ichimoku.style.new.bars.viz.interactive.chart.overlays.V7.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# Uses data_retrieval.py for price I/O; writes daily CSV then RELOADS before processing.
# Also caches earnings_dates to CSV via data_retrieval cache path and RELOADS before plotting.

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf  # for earnings_dates only

# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import (
        load_or_download_ticker,
        create_output_directory,
        get_local_cache_path,
        fix_yfinance_dataframe,
    )
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

warnings.filterwarnings('ignore', category=UserWarning)

# ----- helpers: enforced save+reload -----

def retrieve_to_csv_and_reload(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = load_or_download_ticker(ticker, start=start, end=end)
    if df is None or df.empty:
        return pd.DataFrame()

    df = fix_yfinance_dataframe(df)
    if 'Date' in df.columns:
        to_save = df.copy()
    else:
        idx_name = df.index.name or 'Date'
        to_save = df.reset_index().rename(columns={idx_name: 'Date'})
    
    # CONSTRAINT: Output to /dev/shm via data_retrieval logic
    out_dir = create_output_directory(ticker)
    out_csv = os.path.join(out_dir, f"{ticker}.csv")
    to_save.to_csv(out_csv, index=False)
    
    # RELOAD from disk
    re = pd.read_csv(out_csv, parse_dates=['Date'])
    re.sort_values('Date', inplace=True)
    re.set_index('Date', inplace=True)
    return re

def load_earnings_dates_cached(ticker: str) -> pd.DataFrame:
    """Cache and then read-back earnings_dates via data_retrieval's cache path."""
    cache_path = get_local_cache_path(ticker, period="earnings_dates")
    
    # 1. Check disk
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Warning: could not read cached earnings_dates for {ticker}: {e}. Will re-download.")

    # 2. Download if missing
    try:
        df_dl = yf.Ticker(ticker).earnings_dates
    except Exception as e:
        print(f"Warning: yfinance failed to fetch earnings_dates for {ticker}: {e}")
        df_dl = pd.DataFrame()

    # 3. Save to disk
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if df_dl is not None and not df_dl.empty:
        try:
            df_dl.to_csv(cache_path)
        except Exception as e:
            print(f"Warning: failed to write earnings cache for {ticker}: {e}")

    # 4. Read-back (strict reload policy)
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Warning: could not read earnings cache for {ticker}: {e}")
            
    return pd.DataFrame()

# ----- indicators -----

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
    data.sort_index(inplace=True)

    data['20DMA'] = data['Close'].rolling(window=20, min_periods=20).mean()
    data['50DMA'] = data['Close'].rolling(window=50, min_periods=50).mean()
    data['9DMA']  = data['Close'].rolling(window=9,  min_periods=9).mean()

    mid = data['Close'].rolling(window=20, min_periods=20).mean()
    std = data['Close'].rolling(window=20, min_periods=20).std()
    data['BB_Middle']      = mid
    data['BB_Upper_1std']  = mid + std * 1
    data['BB_Upper_2std']  = mid + std * 2
    data['BB_Lower_1std']  = mid - std * 1
    data['BB_Lower_2std']  = mid - std * 2
    return data

# ----- plotting -----

def plot_stock_data(data: pd.DataFrame, ticker: str, out_dir: str) -> None:
    traces = [
        go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlesticks'),
        go.Scatter(x=data.index, y=data['20DMA'], mode='lines', name='20DMA', line=dict(color='blue')),
        go.Scatter(x=data.index, y=data['50DMA'], mode='lines', name='50DMA', line=dict(color='red')),
        go.Scatter(x=data.index, y=data['9DMA'],  mode='lines', name='9DMA',  line=dict(color='purple')),
        go.Scatter(x=data.index, y=data['BB_Middle'],     mode='lines', name='BB_Middle',     line=dict(color='gray', dash='dash')),
        go.Scatter(x=data.index, y=data['BB_Upper_1std'], mode='lines', name='BB_Upper_1std', line=dict(color='green', dash='dot')),
        go.Scatter(x=data.index, y=data['BB_Upper_2std'], mode='lines', name='BB_Upper_2std', line=dict(color='lightgreen', dash='dot')),
        go.Scatter(x=data.index, y=data['BB_Lower_1std'], mode='lines', name='BB_Lower_1std', line=dict(color='orange', dash='dot')),
        go.Scatter(x=data.index, y=data['BB_Lower_2std'], mode='lines', name='BB_Lower_2std', line=dict(color='pink', dash='dot')),
        go.Scatter(x=data.index, y=data['Open'],  mode='markers', name='Open',  marker=dict(color='cyan', size=4)),
        go.Scatter(x=data.index, y=data['Close'], mode='markers', name='Close', marker=dict(color='white', size=4)),
        go.Scatter(x=data.index, y=data['High'],  mode='markers', name='High',  marker=dict(color='green', size=4)),
        go.Scatter(x=data.index, y=data['Low'],   mode='markers', name='Low',   marker=dict(color='yellow', size=4)),
        go.Scatter(x=data.index, y=(data['High'] + data['Low']) / 2.0, mode='markers', name='Midpoint', marker=dict(color='orange', size=4)),
    ]

    with np.errstate(invalid='ignore'):
        mask_hi = data['High'] > data['BB_Upper_2std']
        mask_lo = data['Low']  < data['BB_Lower_2std']

    traces += [
        go.Scatter(x=data.index[mask_hi], y=data['High'][mask_hi], mode='markers',
                   name='High > 2std Upper', marker=dict(color='lawngreen', size=8, line=dict(color='lawngreen', width=2))),
        go.Scatter(x=data.index[mask_lo], y=data['Low'][mask_lo], mode='markers',
                   name='Low < 2std Lower', marker=dict(color='red', size=8, line=dict(color='red', width=2))),
    ]

    # Fill areas between 20DMA and 50DMA
    fill_between = []
    idx = data.index
    for i in range(1, len(data)):
        a20_prev, a20 = data['20DMA'].iloc[i-1], data['20DMA'].iloc[i]
        a50_prev, a50 = data['50DMA'].iloc[i-1], data['50DMA'].iloc[i]
        if pd.notna(a20) and pd.notna(a50):
            fill_between.append(
                go.Scatter(
                    x=[idx[i-1], idx[i], idx[i], idx[i-1]],
                    y=[a50_prev, a50, a20, a20_prev],
                    fill='toself',
                    fillcolor=('rgba(0, 255, 0, 0.2)' if a20 > a50 else 'rgba(255, 0, 0, 0.2)'),
                    line=dict(width=0), mode='lines', showlegend=False, hoverinfo='skip'
                )
            )
    traces += fill_between

    # Horizontal lines for last 6m extremes beyond ±2σ
    if len(data.index) > 0:
        six_months_ago = data.index[-1] - pd.DateOffset(months=6)
        last6 = data[data.index > six_months_ago]
        hi_above = last6['High'][last6['High'] > last6['BB_Upper_2std']].max()
        lo_below = last6['Low'][last6['Low']   < last6['BB_Lower_2std']].min()
        if pd.notna(hi_above):
            traces.append(go.Scatter(x=[data.index[0], data.index[-1]], y=[hi_above, hi_above], mode='lines',
                                     name='Highest Above 2std Upper (Last 6M)', line=dict(color='lawngreen', dash='dash', width=1)))
        if pd.notna(lo_below):
            traces.append(go.Scatter(x=[data.index[0], data.index[-1]], y=[lo_below, lo_below], mode='lines',
                                     name='Lowest Below 2std Lower (Last 6M)', line=dict(color='red', dash='dash', width=1)))

    # Earnings markers/labels (cached + reloaded)
    earnings_markers = earnings_labels = None
    try:
        earnings_dates = load_earnings_dates_cached(ticker)
        if earnings_dates is not None and not earnings_dates.empty:
            try:
                earnings_dates.index = earnings_dates.index.tz_localize(None).normalize()
            except Exception:
                earnings_dates.index = pd.to_datetime(earnings_dates.index).normalize()
            earnings_dates = earnings_dates[(earnings_dates.index >= data.index.min()) &
                                            (earnings_dates.index <= data.index.max())]
            if not earnings_dates.empty:
                ex = earnings_dates.index
                min_low = float(data['Low'].min())
                earnings_markers = go.Scatter(
                    x=ex, y=[min_low * 0.95] * len(ex), mode='markers', name='Earnings Dates',
                    marker=dict(symbol='triangle-up', size=10, color='gold', line=dict(width=1, color='black')),
                    hoverinfo='none'
                )
                earnings_labels = go.Scatter(
                    x=ex, y=[min_low * 0.98] * len(ex), mode='text',
                    text=[d.strftime('%Y-%m-%d') for d in ex], name='Earnings Labels',
                    textfont=dict(size=10, color='white'), showlegend=False, hoverinfo='none'
                )
    except Exception as e:
        print(f"Warning: could not fetch earnings dates for {ticker}: {e}")

    if earnings_markers is not None:
        traces.append(earnings_markers)
    if earnings_labels is not None:
        traces.append(earnings_labels)

    y_min = float(np.nanmin([data['Low'].min(), data['BB_Lower_2std'].min()])) * 0.95
    y_max = float(np.nanmax([data['High'].max(), data['BB_Upper_2std'].max()])) * 1.05

    layout = go.Layout(
        title=f'{ticker} Stock Price and Moving Averages',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price ($)', range=[y_min, y_max]),
        template='plotly_dark',
        legend=dict(x=1.0, y=1.0),
        hovermode='x'
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(legend_title_text='Indicators', showlegend=True)
    
    # Output to /dev/shm
    out_file = os.path.join(out_dir, f"{ticker}_ichimoku_overlay.html")
    fig.write_html(out_file)
    print(f"Chart saved to: {out_file}")
    fig.show()

def main():
    p = argparse.ArgumentParser(description='Plot stock data with indicators; data is saved+reloaded from daily CSV.')
    p.add_argument('ticker', type=str, help='Ticker (e.g., AAPL)')
    p.add_argument('--start_date', type=str, default='2023-01-01', help='YYYY-MM-DD')
    p.add_argument('--end_date',   type=str, default='2023-12-31', help='YYYY-MM-DD')
    args = p.parse_args()

    ticker = args.ticker.upper()
    data = retrieve_to_csv_and_reload(ticker, args.start_date, args.end_date)
    if data is None or data.empty:
        print(f"No data available for {ticker} in {args.start_date}..{args.end_date}")
        sys.exit(1)

    data = calculate_indicators(data)
    out_dir = create_output_directory(ticker)
    plot_stock_data(data, ticker, out_dir)

if __name__ == "__main__":
    main()
