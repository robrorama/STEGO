#!/usr/bin/env python3
"""
Unified coiling/support zones script.

- Computes MACD, RSI, peaks, troughs, and coiling/support zones.
- Produces TWO Plotly figures:
  * V1: zone rectangles only (matching V1)
  * V2: zone rectangles + price annotations (matching V2)
- Uses data_retrieval.py for all data loading/caching and output directory creation.

CLI:
    python3 coiling_support_zones_unified.py TICKER ORDER
"""

import sys
import os

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import webbrowser
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# IMPORTANT: use the user's data retrieval/caching layer
try:
    import data_retrieval as dr  # requires data_retrieval.py alongside this script
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# -------- calculations (as in V1/V2) --------

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_peaks_troughs(df, order=5):
    prices = df['Close'].values
    n = len(prices)
    peaks, troughs = [], []
    for i in range(order, n - order):
        window = prices[i - order:i + order + 1]
        center = order
        if np.all(window[center] > window[:center]) and np.all(window[center] > window[center+1:]):
            peaks.append((i, prices[i]))
        if np.all(window[center] < window[:center]) and np.all(window[center] < window[center+1:]):
            troughs.append((i, prices[i]))
    return peaks, troughs

def find_coiling_support_zones(peaks, troughs, df, min_width=5, price_tolerance=0.03):
    """
    Create zones if two peaks (resistance) or two troughs (support)
    are within 'price_tolerance' and separated by at least 'min_width' bars.
    """
    zones = []
    # resistance-like zones from peaks
    for i in range(len(peaks) - 1):
        for j in range(i + 1, len(peaks)):
            if abs(peaks[i][1] - peaks[j][1]) / peaks[i][1] <= price_tolerance:
                start_idx, end_idx = peaks[i][0], peaks[j][0]
                if end_idx - start_idx >= min_width:
                    zones.append({
                        'type': 'coiling/resistance',
                        'start': df.index[start_idx],
                        'end': df.index[end_idx],
                        'price': peaks[i][1]
                    })
    # support-like zones from troughs
    for i in range(len(troughs) - 1):
        for j in range(i + 1, len(troughs)):
            if abs(troughs[i][1] - troughs[j][1]) / troughs[i][1] <= price_tolerance:
                start_idx, end_idx = troughs[i][0], troughs[j][0]
                if end_idx - start_idx >= min_width:
                    zones.append({
                        'type': 'support',
                        'start': df.index[start_idx],
                        'end': df.index[end_idx],
                        'price': troughs[i][1]
                    })
    return zones

# -------- plotting helpers --------

def _build_base_figure(df, ticker, peaks, troughs, title_suffix=""):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlestick'
    ), row=1, col=1)

    # Peaks/Troughs markers
    peak_dates = [df.index[i] for (i, _) in peaks]
    peak_prices = [p for (_, p) in peaks]
    trough_dates = [df.index[i] for (i, _) in troughs]
    trough_prices = [p for (_, p) in troughs]

    fig.add_trace(go.Scatter(x=peak_dates, y=peak_prices, mode='markers',
                             name='Peaks', marker=dict(color='lime', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=trough_dates, y=trough_prices, mode='markers',
                             name='Troughs', marker=dict(color='red', size=10)), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                         marker_color='rgba(158,202,225,0.5)'), row=2, col=1)

    # MACD panel
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker_color='grey'), row=3, col=1)

    # RSI panel
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", annotation_text="Overbought",
                  annotation_position="bottom right", row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", annotation_text="Oversold",
                  annotation_position="top right", row=4, col=1)

    fig.update_layout(
        title=f"{ticker} Stock Analysis (Last 1 Year){title_suffix}",
        xaxis_title='Date', yaxis_title='Price',
        xaxis_rangeslider_visible=False, height=1000
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    return fig

def _add_zones_rectangles(fig, zones, final_date):
    """
    Adds zones. x1 is set to final_date to extend lines to the end of the chart.
    """
    for z in zones:
        fig.add_shape(
            type="rect",
            x0=z['start'], 
            x1=final_date, # Extended to end of plot per request
            y0=z['price'] * (1 - 0.001),
            y1=z['price'] * (1 + 0.001),
            line=dict(color="orange", width=1),
            fillcolor="orange",
            opacity=0.2,
            row=1, col=1
        )

def _add_zone_labels(fig, zones, final_date):
    for z in zones:
        zone_type = "Support" if z['type'] == 'support' else "Resistance"
        # Place annotation at the very end of the chart
        fig.add_annotation(
            x=final_date, y=z['price'],
            text=f"{zone_type}: {z['price']:.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="orange",
            font=dict(size=10, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="orange", borderwidth=1, borderpad=4,
            row=1, col=1
        )

def _write_and_open(fig, path_html):
    # Write self-contained HTML and open a new browser tab
    fig.write_html(path_html, include_plotlyjs="inline", full_html=True)
    webbrowser.open_new_tab("file://" + os.path.abspath(path_html))

# -------- main --------

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 coiling_support_zones_unified.py <ticker> <order>")
        sys.exit(1)

    ticker = sys.argv[1]
    order = int(sys.argv[2])

    # Load data via data_retrieval (no direct yfinance calls here)
    df = dr.get_stock_data(ticker, period="1y")
    if df is None or df.empty:
        print(f"Error: no data for {ticker}.")
        sys.exit(2)

    # Indicators
    df['MACD'], df['Signal'], df['Histogram'] = calculate_macd(df)
    df['RSI'] = calculate_rsi(df)

    # Structure detection
    peaks, troughs = find_peaks_troughs(df, order=order)
    zones = find_coiling_support_zones(peaks, troughs, df)

    # Output directory (dated) from data_retrieval (points to /dev/shm)
    outdir = dr.create_output_directory(ticker)
    
    # Determine last date for extending lines
    last_date = df.index[-1]

    # --- V1 plot: rectangles only ---
    fig_v1 = _build_base_figure(df, ticker, peaks, troughs, title_suffix=" — V1 (zones only)")
    _add_zones_rectangles(fig_v1, zones, last_date)
    html_v1 = os.path.join(outdir, f"{ticker}_coiling_support_zones_V1.html")
    _write_and_open(fig_v1, html_v1)

    # --- V2 plot: rectangles + labels ---
    fig_v2 = _build_base_figure(df, ticker, peaks, troughs, title_suffix=" — V2 (zones + labels)")
    _add_zones_rectangles(fig_v2, zones, last_date)
    _add_zone_labels(fig_v2, zones, last_date)
    html_v2 = os.path.join(outdir, f"{ticker}_coiling_support_zones_V2.html")
    _write_and_open(fig_v2, html_v2)

if __name__ == "__main__":
    main()
