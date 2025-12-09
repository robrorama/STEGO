#!/usr/bin/env python3
# SCRIPTNAME: commodities.v4.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Unified commodities plotting toolkit.
# Modes (Plotly Only):
#   - plotly-grouped      : 2x2 grid, grouped by sector, normalized prices.
#   - plotly-lrc          : Grid of individual charts with EMAs + Linear Reg Channels.
#   - plotly-combined     : All normalized commodities on one single chart.
#   - all                 : Runs all of the above.

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import webbrowser
import numpy as np
import pandas as pd
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Centralized Data Retrieval
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ---------- Groups ----------
DEFAULT_GROUPS = {
    "Energy": {
        "Light Sweet Crude Oil": "CL=F", "Brent Crude Oil": "BZ=F",
        "Natural Gas": "NG=F", "Heating Oil": "HO=F", "RBOB Gasoline": "RB=F",
    },
    "Metals": {
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F",
        "Platinum": "PL=F", "Palladium": "PA=F",
    },
    "Agriculture": {
        "Corn": "ZC=F", "Wheat": "ZW=F", "Soybeans": "ZS=F",
        "Soybean Oil": "ZL=F", "Sugar": "SB=F", "Coffee": "KC=F",
        "Cocoa": "CC=F", "Cotton": "CT=F", "Orange Juice": "OJ=F",
    },
    "Livestock": {
        "Live Cattle": "LE=F", "Lean Hogs": "HE=F", "Feeder Cattle": "GF=F",
    },
}

# ---------- Math Helpers ----------
def zscore(series: pd.Series) -> pd.Series:
    if series.empty: return series
    sigma = series.std(ddof=0)
    if sigma == 0: return series * 0.0
    return (series - series.mean()) / sigma

def linreg_channels(s: pd.Series, period: int, use_log: bool = True) -> pd.DataFrame:
    s = s[s > 0].copy()
    n = len(s)
    if n < max(5, period): return pd.DataFrame(index=s.index)
    
    y = np.log(s.values) if use_log else s.values
    idx = np.arange(n)
    
    # Fit on last 'period' points
    idx_win = idx[-period:]
    y_win = y[-period:]
    slope, intercept = np.polyfit(idx_win, y_win, 1)
    
    reg = slope * idx + intercept
    resid = y - reg
    std = resid[-period:].std(ddof=0)
    
    df = pd.DataFrame({
        "reg": reg,
        "upper1": reg + std, "lower1": reg - std,
        "upper2": reg + 2*std, "lower2": reg - 2*std
    }, index=s.index)
    
    if use_log:
        return df.apply(np.exp)
    return df

# ---------- Data Helpers ----------
def get_data(ticker, use_dates, period, start, end):
    if use_dates:
        return dr.load_or_download_ticker(ticker, start=start, end=end)
    return dr.load_or_download_ticker(ticker, period=period)

def parse_date_range(period, dates):
    if dates:
        parts = [p.strip() for p in dates.split(",")]
        return True, None, parts[0], parts[1]
    return False, period, None, None

def ensure_outdir():
    # CONSTRAINT: Creates output dir in /dev/shm via data_retrieval logic
    d = dr.create_output_directory("COMMODITIES")
    os.makedirs(d, exist_ok=True)
    return d

# ---------- Modes ----------

def run_plotly_grouped(use_dates, period, start, end, outdir, open_browser):
    """2x2 Subplots of Normalized Prices by Sector"""
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Energy", "Metals", "Agriculture", "Livestock"))
    sectors = ["Energy", "Metals", "Agriculture", "Livestock"]
    
    for i, sector in enumerate(sectors):
        row, col = (i // 2) + 1, (i % 2) + 1
        for name, ticker in DEFAULT_GROUPS.get(sector, {}).items():
            df = get_data(ticker, use_dates, period, start, end)
            if df is None or df.empty: continue
            s = zscore(df["Close"].dropna())
            fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=name, legendgroup=sector), row=row, col=col)

    fig.update_layout(title="Normalized Commodity Prices (Grouped)", height=900, width=1200)
    tag = f"{start}_{end}" if use_dates else period
    path = os.path.join(outdir, f"plotly_grouped_{tag}.html")
    fig.write_html(path)
    print(f"Saved: {path}")
    if open_browser: webbrowser.open("file://" + path)

def run_plotly_lrc(use_dates, period, start, end, outdir, open_browser, use_log):
    """Grid of individual charts with Linear Reg Channels"""
    # Flatten commodities list
    items = []
    for sector, mapping in DEFAULT_GROUPS.items():
        items.extend([(name, ticker) for name, ticker in mapping.items()])
    
    n = len(items)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[x[0] for x in items], vertical_spacing=0.05)
    
    for i, (name, ticker) in enumerate(items):
        row, col = (i // cols) + 1, (i % cols) + 1
        df = get_data(ticker, use_dates, period, start, end)
        if df is None or df.empty: continue
        
        s = df["Close"].dropna()
        
        # Close Price
        fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(color='black', width=1), name=f"{name} Close"), row=row, col=col)
        
        # EMAs
        for span, color in [(50, 'blue'), (200, 'orange')]:
            ema = s.ewm(span=span, adjust=False).mean()
            fig.add_trace(go.Scatter(x=ema.index, y=ema, line=dict(color=color, width=1), name=f"{name} EMA{span}"), row=row, col=col)
            
        # LRC (Short term 50, Long term 144)
        for p, style in [(144, 'dash'), (50, 'dot')]:
            ch = linreg_channels(s, p, use_log)
            if not ch.empty:
                # Just plotting outer bands to reduce clutter
                fig.add_trace(go.Scatter(x=ch.index, y=ch["upper2"], line=dict(color='green', width=1, dash=style), showlegend=False), row=row, col=col)
                fig.add_trace(go.Scatter(x=ch.index, y=ch["lower2"], line=dict(color='red', width=1, dash=style), showlegend=False), row=row, col=col)

    fig.update_layout(title="Commodities: Linear Regression Channels & EMAs", height=300*rows, width=1400, showlegend=False)
    tag = f"{start}_{end}" if use_dates else period
    path = os.path.join(outdir, f"plotly_lrc_{tag}.html")
    fig.write_html(path)
    print(f"Saved: {path}")
    if open_browser: webbrowser.open("file://" + path)

def run_plotly_combined(use_dates, period, start, end, outdir, open_browser):
    """All commodities normalized on one chart"""
    fig = go.Figure()
    for sector, mapping in DEFAULT_GROUPS.items():
        for name, ticker in mapping.items():
            df = get_data(ticker, use_dates, period, start, end)
            if df is None or df.empty: continue
            s = zscore(df["Close"].dropna())
            fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=f"{sector}: {name}"))

    fig.update_layout(title="Combined Normalized Commodities", yaxis_title="Z-Score", height=800)
    tag = f"{start}_{end}" if use_dates else period
    path = os.path.join(outdir, f"plotly_combined_{tag}.html")
    fig.write_html(path)
    print(f"Saved: {path}")
    if open_browser: webbrowser.open("file://" + path)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Unified Commodities (Plotly Only)")
    parser.add_argument("--mode", choices=["plotly-grouped", "plotly-lrc", "plotly-combined", "all"], default="all")
    parser.add_argument("--period", default="1y", help="Period (e.g. 1y, 5y, max)")
    parser.add_argument("--dates", help="YYYY-MM-DD,YYYY-MM-DD (overrides period)")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    parser.add_argument("--log-channels", action="store_true", default=True, help="Use log scale for LRC (default)")
    args = parser.parse_args()

    use_dates, period, start, end = parse_date_range(args.period, args.dates)
    outdir = ensure_outdir()
    open_browser = not args.no_open

    if args.mode in ("plotly-grouped", "all"):
        run_plotly_grouped(use_dates, period, start, end, outdir, open_browser)
    if args.mode in ("plotly-lrc", "all"):
        run_plotly_lrc(use_dates, period, start, end, outdir, open_browser, args.log_channels)
    if args.mode in ("plotly-combined", "all"):
        run_plotly_combined(use_dates, period, start, end, outdir, open_browser)

if __name__ == "__main__":
    main()
