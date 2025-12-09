#!/usr/bin/env python3
# SCRIPTNAME: spy_sectors_boards.v3.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
"""
Unified SPY sector plotting:
- Produces TWO separate Plotly HTMLs and opens each in its own browser tab:
  1) Normalized performance (base=100 at first bar for each sector)
  2) Actual closing prices (USD)
- Uses data_retrieval.py for ALL data downloading/caching and for output dir creation.

The script pulls daily price data for all S&P 500 sector ETFs (via `data_retrieval.py`),
creates two Plotly figures—one with each sector’s performance normalized to 100 at the start
of the period, and another with the raw closing prices—and saves each as a standalone HTML
file that opens in its own browser tab. All downloads, caching, and output‑folder handling are
managed by `data_retrieval.py`.
"""

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import webbrowser
import pandas as pd
import plotly.graph_objects as go

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr  # must be available on PYTHONPATH / alongside this script
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

SECTORS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

def _write_and_open(fig: go.Figure, path_html: str) -> None:
    fig.write_html(path_html, include_plotlyjs="inline", full_html=True)
    webbrowser.open_new_tab("file://" + os.path.abspath(path_html))

def build_normalized_figure(sectors: dict, period: str) -> go.Figure:
    fig = go.Figure()
    for name, ticker in sectors.items():
        # CONSTRAINT: Use data_retrieval logic
        df = dr.load_or_download_ticker(ticker, period=period)
        if df is None or df.empty or 'Close' not in df.columns:
            print(f"[normalized] skip {name} ({ticker}) – no data")
            continue
        base = df['Close'].iloc[0]
        if pd.isna(base) or base == 0:
            print(f"[normalized] skip {name} ({ticker}) – invalid base")
            continue
        norm = (df['Close'] / base) * 100.0
        fig.add_trace(go.Scatter(x=df.index, y=norm, mode="lines", name=name))

    fig.update_layout(
        title=f"S&P 500 Sector Performance (Normalized, base=100) — period={period}",
        yaxis_title="Normalized Price (Base 100)",
        xaxis_rangeslider_visible=False,
        height=650,
        width=1100,
        legend_title_text="Sectors",
    )
    return fig

def build_price_figure(sectors: dict, period: str) -> go.Figure:
    fig = go.Figure()
    for name, ticker in sectors.items():
        # CONSTRAINT: Use data_retrieval logic
        df = dr.load_or_download_ticker(ticker, period=period)
        if df is None or df.empty or 'Close' not in df.columns:
            print(f"[prices] skip {name} ({ticker}) – no data")
            continue
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode="lines", name=name))

    fig.update_layout(
        title=f"S&P 500 Sector Performance (Actual Prices) — period={period}",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=650,
        width=1100,
        legend_title_text="Sectors",
    )
    return fig

def main():
    period = sys.argv[1] if len(sys.argv) >= 2 else "max"
    
    # CONSTRAINT: Output to /dev/shm via data_retrieval logic
    outdir = dr.create_output_directory("SPY_SECTORS")

    # Build both figures
    fig_norm = build_normalized_figure(SECTORS, period)
    fig_price = build_price_figure(SECTORS, period)

    # Write + open in separate tabs
    norm_path = os.path.join(outdir, f"SPY_sectors_normalized_{period}.html")
    price_path = os.path.join(outdir, f"SPY_sectors_prices_{period}.html")
    _write_and_open(fig_norm, norm_path)
    _write_and_open(fig_price, price_path)

    print(f"Wrote:\n  {norm_path}\n  {price_path}")

if __name__ == "__main__":
    main()
