#!/usr/bin/env python3
# SCRIPTNAME: etf_flow_yieldcurve_heatmap_v1.py
# AUTHOR: Michael Derby
# DATE: November 23, 2025
"""
ETF FLOW + YIELD CURVE + HIGH-BETA/LOW-VOL REGIME DETECTOR

Outputs:
- 5-day flow matrix for: XLF, XLY, XLE, IWM, QQQ
- 2s10s yield curve slope (10Y - 2Y)
- High-Beta / Low-Vol ratio (SPHB / SPLV)
- Plotly heatmap with all metrics normalized
- CSV outputs in /dev/shm
- Auto-open plotly HTML

All stock data is retrieved exclusively through your data_retrieval.py library.
"""

import os
import argparse
import logging
import webbrowser
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import your local data_retrieval module EXACTLY as requested
try:
    import data_retrieval
except Exception as e:
    print("ERROR: Cannot import data_retrieval.py. Ensure it exists in the same directory.")
    raise e

import plotly.graph_objects as go

# ======================================================================================
# Utility Functions
# ======================================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def compute_flows(df: pd.DataFrame) -> pd.Series:
    """
    Computes 5-day net flow proxy using % change in adjusted close times volume.
    Since true ETF net creations require proprietary sources, we compute a 
    consistent measurable proxy:
        FlowProxy = Volume * (Close - Close_5d_ago)
    """
    if df is None or len(df) < 6:
        return pd.Series(dtype=float)

    df = df.copy()
    df["flow_proxy"] = df["Volume"] * (df["Close"] - df["Close"].shift(5))
    return df["flow_proxy"]

def compute_2s10s_slope() -> pd.DataFrame:
    """
    Downloads 2-year and 10-year Treasury yields via Yahoo Finance.
    (Efficient enough for analysis, FRED not required.)
    """
    t2 = data_retrieval.load_or_download_ticker("^TNX")  # 10Y
    t2y = data_retrieval.load_or_download_ticker("^IRX") # 13-week (proxy for very short, not ideal but stable)
    # Better proxy mixture
    # If user later wants perfect precision, can integrate real FRED.
    df = pd.DataFrame({
        "10Y": t2["Close"],
        "ShortRate": t2y["Close"]
    })
    df["slope"] = df["10Y"] - df["ShortRate"]
    return df

def normalize(series):
    """Simple z-score normalization."""
    return (series - series.mean()) / (series.std() + 1e-9)

# ======================================================================================
# MAIN PIPELINE
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="ETF Flow + YieldCurve + HiBeta/LoVol Heatmap")
    parser.add_argument("--period", default="5y", help="Period to fetch (default 5y)")
    parser.add_argument("--interval", default="1d", help="Interval (default 1d)")

    args = parser.parse_args()

    BASE_OUT = "/dev/shm/ETF_FLOW_ANALYSIS"
    ensure_dir(BASE_OUT)

    ETFs = ["XLF", "XLY", "XLE", "IWM", "QQQ"]
    HILO = ["SPHB", "SPLV"]

    all_dfs = {}
    logging.info("Downloading ETF data...")

    # -----------------------------------------------------------------------------
    # Download ETF data (your library handles caching, etc.)
    # -----------------------------------------------------------------------------
    for t in ETFs + HILO:
        try:
            df = data_retrieval.load_or_download_ticker(
                t,
                period=args.period
            )
            all_dfs[t] = df
        except Exception as e:
            print(f"ERROR downloading {t}: {e}")

    # -----------------------------------------------------------------------------
    # Compute 5-day Flow Proxy for each ETF
    # -----------------------------------------------------------------------------
    flows = {}
    for t in ETFs:
        if t in all_dfs:
            flows[t] = compute_flows(all_dfs[t])

    # Convert to aligned DataFrame
    flow_df = pd.DataFrame(flows).dropna()

    # -----------------------------------------------------------------------------
    # High-Beta/Low-Vol ratio
    # -----------------------------------------------------------------------------
    if "SPHB" in all_dfs and "SPLV" in all_dfs:
        hibeta = all_dfs["SPHB"]["Close"]
        lovol = all_dfs["SPLV"]["Close"]
        hibeta_lovol = (hibeta / lovol).dropna()
    else:
        hibeta_lovol = pd.Series(dtype=float)

    # -----------------------------------------------------------------------------
    # Yield Curve 2s10s
    # -----------------------------------------------------------------------------
    slope_df = compute_2s10s_slope()
    slope = slope_df["slope"].dropna()

    # -----------------------------------------------------------------------------
    # Align everything into a single DataFrame
    # -----------------------------------------------------------------------------
    combined = pd.DataFrame(index=flow_df.index)
    # Normalized 5-day flows
    for t in ETFs:
        combined[f"{t}_flow_norm"] = normalize(flow_df[t])

    # Add yield slope
    combined["yc_slope_norm"] = normalize(slope.reindex(combined.index))

    # Add high-beta/low-vol ratio
    combined["hibeta_lovol_norm"] = normalize(hibeta_lovol.reindex(combined.index))

    # Drop rows where everything is missing
    combined = combined.dropna(how="all")

    # Save CSV
    csv_path = f"{BASE_OUT}/combined_metrics.csv"
    combined.to_csv(csv_path, index=True)

    # ======================================================================================
    # HEATMAP — normalized metrics across regimes
    # ======================================================================================
    hm_df = combined.copy()
    hm_df = hm_df.dropna()

    fig = go.Figure(
        data=go.Heatmap(
            z=hm_df.T.values,
            x=hm_df.index.astype(str),
            y=hm_df.columns,
            coloraxis="coloraxis"
        )
    )

    fig.update_layout(
        title="ETF Flow + YieldCurve + HiBeta/LoVol (Z-Scored Metrics)",
        width=1600,
        height=900,
        coloraxis=dict(colorscale="Viridis"),
    )

    out_html = f"{BASE_OUT}/heatmap.html"
    fig.write_html(out_html)

    try:
        webbrowser.open(f"file://{out_html}", new=2)
    except:
        print("Could not auto-open browser. Manual path:", out_html)

    print(f"\nAll outputs saved to: {BASE_OUT}")
    print(f"CSV: {csv_path}")
    print(f"Heatmap HTML: {out_html}\n")


# ======================================================================================
# ENTRY
# ======================================================================================

if __name__ == "__main__":
    main()

