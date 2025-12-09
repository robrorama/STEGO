#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SCRIPTNAME: ok.unified.macd.v3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Unified options/technicals viewer:
- Uses data_retrieval.py for ALL data access (caching, paths).
- Preserves original functionality:
  * Candlesticks + 20/50DMA
  * Green/Red fill between 20DMA and 50DMA (20>50 green, 20<50 red)
  * MACD line/signal + color-coded histogram
  * RSI (14)
  * Plotly dark theme
- Renders separate HTML tabs: Combined (original overlay), Price, MACD, RSI.
- NEW: Opens the generated HTML in a web browser tab by default.
"""

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import warnings
from datetime import datetime
from pathlib import Path
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Silence noisy user warnings (mirrors original script behavior)
warnings.filterwarnings('ignore', category=UserWarning)

# ---- Data access strictly via data_retrieval.py ----
# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import load_or_download_ticker, create_output_directory
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# Optional dependency exactly as original for RSI
try:
    import ta
    _HAS_TA = True
except Exception:
    _HAS_TA = False


# ------------ Indicators ------------

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI(14). Try `ta` for parity with original; fallback if missing."""
    if _HAS_TA:
        return ta.momentum.rsi(series, window=window)
    # Minimal, dependency-free RSI
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # DMAs
    df['20DMA'] = df['Close'].rolling(window=20).mean()
    df['50DMA'] = df['Close'].rolling(window=50).mean()
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
    # RSI
    df['RSI'] = _rsi(df['Close'], window=14)
    return df


# ------------ Plot helpers ------------

def _dma_fill_traces(df: pd.DataFrame):
    """Quadrilateral patches between 20DMA and 50DMA in green/red segments (parity with original)."""
    fill_between = []
    idx = df.index
    for i in range(1, len(df)):
        y50_prev, y50 = df['50DMA'].iloc[i-1], df['50DMA'].iloc[i]
        y20_prev, y20 = df['20DMA'].iloc[i-1], df['20DMA'].iloc[i]
        if pd.isna(y50_prev) or pd.isna(y50) or pd.isna(y20_prev) or pd.isna(y20):
            continue
        is_green = bool(y20 > y50)
        color = 'rgba(0, 255, 0, 0.5)' if is_green else 'rgba(255, 0, 0, 0.5)'
        name = '20DMA > 50DMA' if is_green else '20DMA < 50DMA'
        fill_between.append(go.Scatter(
            x=[idx[i-1], idx[i], idx[i], idx[i-1]],
            y=[y50_prev, y50, y20, y20_prev],
            fill='toself',
            fillcolor=color,
            line=dict(width=0),
            mode='lines',
            name=name,
            showlegend=(i == 1),
            legendgroup=('green_fill' if is_green else 'red_fill'),
            hoverinfo='skip',
            visible=True
        ))
    return fill_between


def figure_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    candles = go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlesticks'
    )
    ma20 = go.Scatter(x=df.index, y=df['20DMA'], mode='lines', name='20DMA')
    ma50 = go.Scatter(x=df.index, y=df['50DMA'], mode='lines', name='50DMA')
    fills = _dma_fill_traces(df)
    layout = go.Layout(
        title=f'{ticker} – Price (Candles) + 20/50DMA',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        template='plotly_dark'
    )
    fig = go.Figure(data=[candles, ma20, ma50] + fills, layout=layout)
    fig.update_layout(legend_title_text='Overlays', showlegend=True)
    return fig


def figure_macd(df: pd.DataFrame, ticker: str) -> go.Figure:
    macd_line = go.Scatter(x=df.index, y=df['MACD_Line'], mode='lines', name='MACD Line')
    macd_signal = go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal')
    macd_hist = go.Bar(
        x=df.index, y=df['MACD_Histogram'], name='MACD Histogram',
        marker_color=['green' if x > 0 else 'red' for x in df['MACD_Histogram']]
    )
    zero = go.Scatter(
        x=[df.index.min(), df.index.max()], y=[0, 0],
        mode='lines', name='Zero', line=dict(dash='dash'), hoverinfo='skip'
    )
    layout = go.Layout(
        title=f'{ticker} – MACD',
        xaxis=dict(title='Date'),
        yaxis=dict(title='MACD'),
        template='plotly_dark',
        barmode='overlay'
    )
    return go.Figure(data=[macd_hist, macd_line, macd_signal, zero], layout=layout)


def figure_rsi(df: pd.DataFrame, ticker: str) -> go.Figure:
    rsi = go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI')
    h70 = go.Scatter(
        x=[df.index.min(), df.index.max()], y=[70, 70],
        mode='lines', name='70', line=dict(dash='dash'), hoverinfo='skip'
    )
    h30 = go.Scatter(
        x=[df.index.min(), df.index.max()], y=[30, 30],
        mode='lines', name='30', line=dict(dash='dash'), hoverinfo='skip'
    )
    layout = go.Layout(
        title=f'{ticker} – RSI(14)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='RSI', range=[0, 100]),
        template='plotly_dark'
    )
    return go.Figure(data=[rsi, h70, h30], layout=layout)


def figure_combined_overlay(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Replicates original single-figure overlay to ensure nothing is missing."""
    candles = go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlesticks'
    )
    ma20 = go.Scatter(x=df.index, y=df['20DMA'], mode='lines', name='20DMA')
    ma50 = go.Scatter(x=df.index, y=df['50DMA'], mode='lines', name='50DMA')
    fills = _dma_fill_traces(df)

    macd_line = go.Scatter(x=df.index, y=df['MACD_Line'], mode='lines', name='MACD Line')
    macd_signal = go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal')
    macd_hist = go.Bar(
        x=df.index, y=df['MACD_Histogram'], name='MACD Histogram',
        marker_color=['green' if x > 0 else 'red' for x in df['MACD_Histogram']]
    )

    rsi = go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', yaxis='y2')

    layout = go.Layout(
        title=f'{ticker} – Price, MACD, RSI (Combined)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 100]),
        template='plotly_dark'
    )
    fig = go.Figure(
        data=[candles, ma20, ma50] + fills + [macd_line, macd_signal, macd_hist, rsi],
        layout=layout
    )
    fig.update_layout(legend_title_text='Indicators & Fills', showlegend=True)
    return fig


# ------------ HTML tabs writer ------------

def write_tabs_html(figs: dict, output_file: str, page_title: str) -> str:
    """
    Write a single HTML with a top tab bar. The first figure includes Plotly.js (CDN).
    """
    first = True
    blocks = []
    for name, fig in figs.items():
        div_id = f"fig_{name.lower()}"
        html = fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False, div_id=div_id)
        blocks.append((name, html))
        first = False

    tabs = "".join([f'<button id="btn-{n}" onclick="openTab(\'{n}\')">{n}</button>' for n in figs.keys()])
    sections = "".join([f'<div id="tab-{n}" class="tabcontent">{h}</div>' for n, h in blocks])

    shell = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{page_title}</title>
<style>
body {{ margin: 0; font-family: sans-serif; }}
.tabbar {{ display: flex; gap: 8px; padding: 10px; border-bottom: 1px solid #444; background: #111; position: sticky; top: 0; z-index: 10; }}
.tabbar button {{ background: #222; color: #ccc; border: 1px solid #444; padding: 8px 12px; cursor: pointer; }}
.tabbar button.active {{ background: #333; color: #fff; border-bottom: 2px solid #08f; }}
.tabcontent {{ display: none; padding: 8px; }}
</style>
<script>
function openTab(name) {{
  document.querySelectorAll('.tabcontent').forEach(el => el.style.display='none');
  document.querySelectorAll('.tabbar button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).style.display = 'block';
  document.getElementById('btn-' + name).classList.add('active');
}}
window.addEventListener('DOMContentLoaded', () => openTab('{list(figs.keys())[0]}'));
</script>
</head>
<body>
  <div class="tabbar">{tabs}</div>
  {sections}
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(shell)
    return output_file


# ------------ CLI ------------

def main():
    p = argparse.ArgumentParser(description="Unified options/technicals plotter (tabs).")
    p.add_argument("-t", "--ticker", default="AAPL")
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--period", default="1y",
                   help="yfinance-style period (used iff start/end not provided), e.g. 1mo, 6mo, 1y, 5y, max")
    p.add_argument("-o", "--output", help="Output HTML file. Default is a dated path from data_retrieval.create_output_directory(...)")
    args = p.parse_args()

    # Pull data strictly via data_retrieval (caching respected)
    if args.start and args.end:
        df = load_or_download_ticker(args.ticker, start=args.start, end=args.end)
    else:
        df = load_or_download_ticker(args.ticker, period=args.period)

    if df is None or df.empty:
        raise SystemExit("No data returned. Check ticker and date range/period.")

    # Compute indicators
    df.index = pd.to_datetime(df.index)
    df = compute_indicators(df)

    # Build figures
    figs = {
        "Combined": figure_combined_overlay(df, args.ticker),
        "Price":    figure_price(df, args.ticker),
        "MACD":     figure_macd(df, args.ticker),
        "RSI":      figure_rsi(df, args.ticker),
    }

    # Output path under /dev/shm/data/YYYY-MM-DD/TICKER by default (from data_retrieval.py)
    if args.output:
        out_path = args.output
    else:
        out_dir = create_output_directory(args.ticker)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"{args.ticker}_{ts}_charts.html")

    final_path = write_tabs_html(figs, out_path, page_title=f"{args.ticker} – Indicators")

    # NEW: Open in a web browser tab by default
    try:
        webbrowser.open_new_tab(Path(final_path).resolve().as_uri())
    except Exception:
        pass

    print(final_path)


if __name__ == "__main__":
    main()
