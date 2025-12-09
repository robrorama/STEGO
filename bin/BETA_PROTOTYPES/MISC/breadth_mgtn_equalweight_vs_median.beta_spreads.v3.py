#!/usr/bin/env python3
# SCRIPTNAME: ok.breadth_mgtn_equalweight_vs_median.beta_spreads.v3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

# breadth_mgtn_equalweight_vs_median.beta_spreads.v2.py
#
# Author: Michael Derby
# Framework: STEGO Financial Framework
#
# Purpose:
# - Compare Cap-Weight vs Equal-Weight leadership (Breadth Analysis).
# - Compute Rolling Betas & Spreads.
# - Monitor Cross-Sector Leadership Index & Dispersion.
# - Flag "Narrowing Leadership" Regimes (Red Shading).
#
# V2 Upgrades:
# - FIXED: DeprecationWarning for datetime.utcnow().
# - VISUALS: Combined 3 output files into 1 Master Dark-Mode Dashboard.
# - VISUALS: Added Red Background Shading for Narrowing Leadership Regimes.
# - AUTO-OPEN: Automatically launches browser on completion.
#
# Usage:
#   python3 breadth_mgtn_equalweight_vs_median.beta_spreads.v2.py --window 63 --start 2015-01-01

import os
import sys
import argparse
import webbrowser
from datetime import datetime, timezone 
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

# ---- Import data_retrieval ----
try:
    import data_retrieval as dr
except Exception as e:
    print("[ERROR] Could not import data_retrieval. Make sure it's on PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_close(ticker: str, start: str = "2010-01-01", period: str = "max") -> pd.Series:
    """Use your data_retrieval to get a clean Close series with a name."""
    df = dr.load_or_download_ticker(ticker, period=period)
    # Robust column detection
    for col in ["Adj Close", "Close", "close", "adjclose", "AdjClose"]:
        if col in df.columns:
            s = df[col].copy()
            s.name = ticker.upper()
            s = s[s.index >= pd.to_datetime(start)]
            return s.dropna()
    raise ValueError(f"No suitable price column in DataFrame for {ticker}")

def rolling_beta(y_ret: pd.Series, x_ret: pd.Series, window: int = 63) -> pd.Series:
    """ OLS beta via rolling window: beta_t = Cov(y,x)/Var(x) """
    cov = (y_ret.rolling(window).cov(x_ret))
    var = (x_ret.rolling(window).var())
    beta = cov / var
    return beta.rename(f"beta_{y_ret.name}_to_{x_ret.name}")

def rolling_slope(series: pd.Series, win: int = 63) -> pd.Series:
    """ Calculate rolling slope of log-prices (Momentum) """
    y = np.log(series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    
    def _slope(x):
        if np.isnan(x).any(): return np.nan
        idx = np.arange(len(x))
        xi = idx - idx.mean()
        # slope = cov(idx, x) / var(idx)
        return (xi * x).mean() / (xi**2).mean()
    
    return y.rolling(win, min_periods=win).apply(_slope, raw=False)

def create_master_dashboard(dates, li, disp, disp_med, regime_flag, spread_df, rs_df, out_path):
    """
    Creates a 4-Row Master Dashboard:
    1. Leadership Index + Narrowing Regime Shading
    2. Dispersion vs Median
    3. Beta Spreads (Cap - Equal)
    4. Raw Relative Strength (Log Scale)
    """
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.3, 0.2, 0.25, 0.25],
        subplot_titles=(
            "Leadership Index (Avg Slope) & Narrowing Regimes (Red Shading)", 
            "Dispersion of Sector Breadth", 
            "Beta Spreads (Cap-Weight Beta minus Equal-Weight Beta)",
            "Underlying Relative Strength (Log Scale)"
        )
    )

    # --- ROW 1: Leadership Index + Regime Shading ---
    
    # Add Shading for "Narrowing Leadership" (Regime = True)
    # We create a filled area where Regime == True.
    # To make it visible as a background, we plot it first, or use a separate trace.
    # Since y-axis scale varies, we normalize the fill or use a fixed range if possible.
    # A cleaner hack for "Background" in Plotly is using a filled trace of the Data itself 
    # but masked.
    
    # Mask data for shading
    y_vals = li.values
    mask_narrow = regime_flag.astype(bool).values
    
    # Plot the "Narrowing" (Bad/Risk) Regime in Red tint
    y_narrow = np.where(mask_narrow, y_vals, np.nan)
    fig.add_trace(go.Scatter(
        x=dates, y=y_narrow,
        mode='lines', line=dict(width=0),
        fill='tozeroy', fillcolor='rgba(255, 50, 50, 0.25)',
        name='Regime: Narrowing', hoverinfo='skip', showlegend=True
    ), row=1, col=1)

    # Main Leadership Line
    fig.add_trace(go.Scatter(
        x=dates, y=li, 
        name="Leadership Index", 
        line=dict(color='cyan', width=2)
    ), row=1, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    # --- ROW 2: Dispersion ---
    
    fig.add_trace(go.Scatter(
        x=dates, y=disp, 
        name="Dispersion (IQR)", 
        line=dict(color='yellow', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, y=disp_med, 
        name="6m Median", 
        line=dict(color='orange', dash='dash')
    ), row=2, col=1)

    # --- ROW 3: Beta Spreads ---
    
    # Cycle colors for spreads
    colors = ['#00ff00', '#ff00ff', '#00ccff', '#ffaa00']
    for i, col in enumerate(spread_df.columns):
        c = colors[i % len(colors)]
        # Shorten name for legend
        short_name = col.replace("beta_spread_", "").replace("_minus_", "-")
        fig.add_trace(go.Scatter(
            x=dates, y=spread_df[col], 
            name=short_name, 
            line=dict(color=c, width=1.5),
            opacity=0.8
        ), row=3, col=1)
    
    fig.add_hline(y=0, line_dash="solid", line_color="white", row=3, col=1)

    # --- ROW 4: Relative Strength ---
    
    for i, col in enumerate(rs_df.columns):
        c = colors[i % len(colors)]
        short_name = col.replace("RS_", "").replace("_over_", "/")
        fig.add_trace(go.Scatter(
            x=dates, y=np.log(rs_df[col]), 
            name=short_name, 
            line=dict(color=c, width=1)
        ), row=4, col=1)

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        height=1200,
        title_text="STEGO Framework: Breadth & Beta-Spread Monitor",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.02)
    )
    
    plot(fig, filename=out_path, auto_open=False)
    print(f"[INFO] Saved Dashboard: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="STEGO Breadth/Beta Monitor V2")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--window", type=int, default=63, help="Rolling window (days)")
    parser.add_argument("--median-window", type=int, default=126, help="Median window (days)")
    parser.add_argument("--output-root", type=str, default="/dev/shm/BREADTH_BETA_SPREADS")
    args = parser.parse_args()

    # FIX: Use timezone-aware UTC now
    asof = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ensure_dir(os.path.join(args.output_root, asof))

    # --- Data Loading ---
    pairs = [
        ("XLK", "XITK"),  # Tech: Cap vs Equal Proxy
        ("XLY", "RCD"),   # Disc: Cap vs Equal Proxy
        ("XLF", "RYF"),   # Fin: Cap vs Equal Proxy
        ("XLE", "RYE"),   # Energy: Cap vs Equal Proxy
    ]
    spy = load_close("SPY", start=args.start)

    print("[INFO] Loading tickers...")
    tickers_flat = set(["SPY"])
    for cw, ew in pairs:
        tickers_flat.add(cw)
        tickers_flat.add(ew)
    
    loaded = {}
    for t in sorted(tickers_flat):
        try:
            loaded[t] = load_close(t, start=args.start)
        except Exception as e:
            print(f"[WARNING] Failed to load {t}: {e}")
            
    all_series = [s for s in loaded.values() if s is not None]
    if len(all_series) < 3:
        raise RuntimeError("Insufficient data loaded.")
        
    prices = pd.concat(all_series, axis=1, join="inner").sort_index()
    rets = prices.pct_change()

    # --- Calculations ---
    
    print("[INFO] Computing Metrics...")
    
    # 1. Relative Strength & Betas
    rs_df = pd.DataFrame(index=prices.index)
    spread_df = pd.DataFrame(index=prices.index)
    
    for cw, ew in pairs:
        if cw in prices.columns and ew in prices.columns:
            # RS
            rs = (prices[cw] / prices[ew]).rename(f"RS_{cw}_over_{ew}")
            rs_df[rs.name] = rs
            
            # Betas
            if "SPY" in rets.columns:
                b_cw = rolling_beta(rets[cw], rets["SPY"], window=args.window)
                b_ew = rolling_beta(rets[ew], rets["SPY"], window=args.window)
                spr = (b_cw - b_ew).rename(f"beta_spread_{cw}_minus_{ew}")
                spread_df[spr.name] = spr

    # 2. Leadership Index (Mean Slope of Log RS)
    rs_slope_df = pd.DataFrame(index=rs_df.index)
    for col in rs_df.columns:
        rs_slope_df[col] = rolling_slope(rs_df[col], win=args.window)

    li = rs_slope_df.mean(axis=1).rename("LeadershipIndex")

    # 3. Dispersion (IQR)
    q75 = rs_slope_df.quantile(0.75, axis=1)
    q25 = rs_slope_df.quantile(0.25, axis=1)
    dispersion = (q75 - q25).rename("Dispersion_IQR")

    # 4. Regime Detection
    disp_med6m = dispersion.rolling(args.median_window).median().rename("Median_6m")
    # Regime: True if Dispersion is compressed below median (Narrowing)
    regime_flag = (dispersion < disp_med6m).rename("Regime_Narrowing")

    # --- Exports ---
    
    # CSVs
    print(f"[INFO] Exporting data to {out_dir}...")
    prices.to_csv(os.path.join(out_dir, "prices.csv"))
    pd.concat([li, dispersion, disp_med6m, regime_flag], axis=1).to_csv(os.path.join(out_dir, "leadership_metrics.csv"))

    # Master Dashboard
    html_path = os.path.join(out_dir, "MASTER_BREADTH_DASHBOARD.html")
    create_master_dashboard(prices.index, li, dispersion, disp_med6m, regime_flag, spread_df, rs_df, html_path)

    # --- Console Snapshot ---
    last_row = pd.concat([li.tail(1), dispersion.tail(1), regime_flag.tail(1)], axis=1)
    print("\n[SNAPSHOT] Latest Values:")
    print(last_row.to_string())

    # --- Auto Open ---
    print(f"[INFO] Opening dashboard...")
    webbrowser.open(f"file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    main()
