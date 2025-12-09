#!/usr/bin/env python3
# signals.scale.log.purple.stars.py  (enforced save+reload via data_retrieval.py)

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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

'''
The script downloads a ticker’s OHLCV data (caching it to CSV), computes a suite of technical
indicators (log price, returns, multiple moving averages, volatility, momentum), fits
power‑law and LPPL models plus a Fourier‑based frequency check to flag possible bubble or
cyclical patterns, and then renders a multi‑panel Plotly dashboard (candlestick, indicators,
bubble‑signal markers, Fourier flag, and %‑change series). All outputs are saved in a
per‑ticker output folder and displayed interactively.
'''

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
    
    re = pd.read_csv(out_csv, parse_dates=['Date'])
    re.sort_values('Date', inplace=True)
    re.set_index('Date', inplace=True)
    return re

def resolve_price_col(df):
    if 'Adj Close' in df.columns:
        return 'Adj Close'
    if 'Close' in df.columns:
        return 'Close'
    raise KeyError("Neither 'Adj Close' nor 'Close' found in input DataFrame.")

def calculate_indicators(data, price_col):
    data['Log_Price'] = np.log(data[price_col])
    data['ROC'] = data[price_col].pct_change()
    for ma in [9, 20, 50, 100, 200, 300]:
        data[f'MA_{ma}'] = data[price_col].rolling(window=ma).mean()
    data['Volatility'] = data['ROC'].rolling(window=20).std() * np.sqrt(252)
    data['Momentum'] = data[price_col] - data[price_col].shift(20)
    return data

def power_law(x, a, b):
    return a * np.power(x, b)

def identify_potential_bubble_patterns(data):
    data['Bubble_Signal'] = 0
    data['LPPL_Signal'] = 0
    data['LPPL_Confidence'] = 0
    try:
        y_tail = data['Log_Price'].tail(60).values
        if len(y_tail) >= 10 and np.isfinite(y_tail).all():
            x_tail = np.arange(len(y_tail))
            params, _ = curve_fit(power_law, x_tail, y_tail)
            if params[1] > 1.1:
                data.loc[data.index[-60:], 'Bubble_Signal'] = 1
    except Exception as e:
        print(f"Could not fit power-law to recent data: {e}")
    try:
        lppl_window = data['Log_Price'].tail(180)
        t = np.arange(len(lppl_window))
        log_price = lppl_window.values
        if len(log_price) >= 60 and np.isfinite(log_price).all():
            def lppl(t, tc, m, omega, A, B, C, phi):
                return A + B * (tc - t)**m + C * (tc - t)**m * np.cos(omega * np.log(tc - t) - phi)
            p0 = [len(t) + 30, 0.5, 10.0, log_price[-1], -0.5, 0.1, 0.0]
            bounds = ([len(t), 0, 6, -np.inf, -np.inf, -np.inf, -np.inf],
                      [len(t) * 2, 1, 15,  np.inf, 0,       np.inf,  np.inf])
            params, _ = curve_fit(lppl, t, log_price, p0=p0, bounds=bounds)
            tc, m, omega, A, B, C, phi = params
            lppl_conf = abs(tc - len(t))
            if (len(t) < tc < len(t) * 1.5) and (0 < m < 1) and (6 < omega < 15) and (B < 0):
                data.loc[lppl_window.index, 'LPPL_Signal'] = 1
                data.loc[lppl_window.index, 'LPPL_Confidence'] = lppl_conf
                data['LPPL_Fit'] = np.nan
                data.loc[lppl_window.index, 'LPPL_Fit'] = lppl(t, *params)
                print(f"Potential bubble (LPPL): tc={tc:.2f}, m={m:.2f}, omega={omega:.2f}, B={B:.2f}")
    except Exception as e:
        print(f"Could not fit LPPL model to recent data: {e}")
    return data

def signal_processing(data, price_col):
    data['Fourier_Signal'] = 0
    series = data[price_col].values
    if len(series) == 0 or np.isnan(series).all():
        return data
    fft_values = np.fft.fft(series)
    power = np.abs(fft_values) ** 2
    peaks, _ = find_peaks(power, height=np.percentile(power, 95))
    if len(peaks) > 0:
        freqs = peaks / len(series)
        meaningful = [f for f in freqs if 0.01 < f < 0.5]
        if meaningful:
            data['Fourier_Signal'] = 1
            print(f"Dominant Frequencies (cycles/day): {meaningful}")
    return data

def visualize(data, ticker, price_col):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        subplot_titles=[f'{ticker} Stock Price and Moving Averages', 'Log Price and Bubble Signals',
                                        'Volatility and Momentum', 'Fourier Signal', 'Percentage Price Movements with Moving Averages'])

    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlestick'), row=1, col=1)

    for ma in [9, 20, 50, 100, 200, 300]:
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{ma}'], mode='lines', name=f'{ma}-day MA'), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['Log_Price'], mode='lines', name='Log Price'), row=2, col=1)

    pl_mask = data['Bubble_Signal'] == 1
    if pl_mask.any():
        fig.add_trace(go.Scatter(x=data.index[pl_mask], y=data.loc[pl_mask, 'Log_Price'], mode='markers',
                                 name='Potential Bubble (Power Law)', marker=dict(symbol='triangle-up', size=10, line=dict(width=1))), row=2, col=1)

    lppl_mask = data['LPPL_Signal'] == 1
    if lppl_mask.any():
        sig = data.loc[lppl_mask].sort_values('LPPL_Confidence').tail(5)
        fig.add_trace(go.Scatter(x=sig.index, y=sig['Log_Price'], mode='markers', name='Potential Bubble (LPPL)',
                                 marker=dict(symbol='star', size=12, line=dict(width=1))), row=2, col=1)

    if 'LPPL_Fit' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['LPPL_Fit'], mode='lines', name='LPPL Fit', line=dict(dash='dash')), row=2, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Momentum'],   mode='lines', name='Momentum'),   row=3, col=1)

    fs_mask = data['Fourier_Signal'] == 1
    fig.add_trace(go.Scatter(x=data.index[fs_mask], y=[1] * int(fs_mask.sum()), mode='markers', name='Potential Cyclical Pattern (Fourier)'), row=4, col=1)
    fig.update_yaxes(range=[0, 1.1], tickvals=[0, 1], ticktext=['No Signal', 'Signal'], row=4, col=1)

    data['Price_Pct_Change'] = data[price_col].pct_change() * 100
    fig.add_trace(go.Scatter(x=data.index, y=data['Price_Pct_Change'], mode='lines', name=f'{price_col} % Change'), row=5, col=1)
    for ma in [9, 20, 50, 100, 200, 300]:
        data[f'MA_{ma}_Pct_Change'] = data[f'MA_{ma}'].pct_change() * 100
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{ma}_Pct_Change'], mode='lines', name=f'{ma}-day MA % Change'), row=5, col=1)

    fig.update_layout(title_text=f'{ticker} Stock Analysis', title_x=0.5, height=1400, xaxis_rangeslider_visible=False)

    if pl_mask.any():
        first_idx = data.index[pl_mask][0]
        fig.add_annotation(x=first_idx, y=data.loc[first_idx, 'Log_Price'], text="First Power Law Signal",
                           showarrow=True, arrowhead=1, ax=0, ay=-40, row=2, col=1)
    
    # Save chart to /dev/shm
    out_dir = create_output_directory(ticker)
    plot_file = os.path.join(out_dir, f"{ticker}_bubble_analysis.html")
    fig.write_html(plot_file)
    print(f"Chart saved to {plot_file}")
    
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python signals.scale.log.purple.stars.py <ticker>")
        sys.exit(1)

    ticker = sys.argv[1]
    start_date = '2024-01-01'
    end_date = '2024-12-31'

    stock_data = retrieve_to_csv_and_reload(ticker, start=start_date, end=end_date)
    if stock_data is None or stock_data.empty:
        print(f"Error: No data for {ticker} in [{start_date}, {end_date}].")
        sys.exit(1)

    price_col = resolve_price_col(stock_data)
    stock_data = calculate_indicators(stock_data, price_col)
    stock_data = identify_potential_bubble_patterns(stock_data)
    stock_data = signal_processing(stock_data, price_col)
    visualize(stock_data, ticker, price_col)
