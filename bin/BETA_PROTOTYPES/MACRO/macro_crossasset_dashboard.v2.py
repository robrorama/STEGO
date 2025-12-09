#!/usr/bin/env python3
# SCRIPTNAME: macro_crossasset_dashboard.v1.py
# AUTHOR: Michael Derby (framework wiring by ChatGPT)
# DATE: November 25, 2025
#
# PURPOSE
# -------
# Build a cross-asset macro dashboard using local library `data_retrieval`.
#
# OUTPUTS
# -------
# Location: /dev/shm/MACRO_DASHBOARD/YYYY-MM-DD/
# 1. HTML interactive charts (Plotly)
# 2. CSV data files (STEGO pipeline compliant)
#
# USAGE
# -----
#   python3 macro_crossasset_dashboard.v1.py --start-date 2010-01-01

import argparse
import datetime as dt
import logging
import os
import sys
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Your existing data framework
import data_retrieval as dr

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def ensure_output_dir(root: str = "/dev/shm/MACRO_DASHBOARD") -> str:
    """Create a dated output directory under /dev/shm and return its path."""
    today = dt.date.today().strftime("%Y-%m-%d")
    outdir = os.path.join(root, today)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def parse_date(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%Y-%m-%d").date()

def save_stego_csv(df: pd.DataFrame, filename: str, outdir: str):
    """Saves DataFrame to CSV for STEGO pipelines."""
    if df is None or df.empty:
        logging.warning(f"Skipping CSV save for {filename}: Data is empty.")
        return
    path = os.path.join(outdir, filename)
    df.to_csv(path)
    logging.info(f"Saved STEGO CSV: {path}")

def load_ohlcv(ticker: str, period: str = "max") -> pd.DataFrame:
    """
    Wrapper around data_retrieval.load_or_download_ticker.
    """
    logging.info(f"Loading OHLCV for {ticker} via data_retrieval (period={period})...")
    try:
        # data_retrieval.py signature: load_or_download_ticker(ticker, period="1y", ...)
        df = dr.load_or_download_ticker(ticker, period=period)
    except TypeError:
        df = dr.load_or_download_ticker(ticker)
    
    if df is None or df.empty:
        logging.warning(f"No data returned for {ticker}")
        return pd.DataFrame()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    return df.sort_index()

def pick_price_series(df: pd.DataFrame, name: str = "Close") -> pd.Series:
    """Choose 'Adj Close', 'Close', or first numeric column."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    candidates = ["Adj Close", "adjclose", "AdjClose", "Close", "close", "Price", "price"]
    for col in candidates:
        if col in df.columns:
            s = df[col].astype(float).copy()
            s.name = name
            return s

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        logging.warning("No numeric columns found for price series.")
        return pd.Series(dtype=float)
    s = df[num_cols[0]].astype(float).copy()
    s.name = name
    return s

def restrict_date_range(s: pd.Series, start: dt.date, end: dt.date) -> pd.Series:
    if s.empty:
        return s
    mask = (s.index.date >= start) & (s.index.date <= end)
    return s.loc[mask]

def align_series(series_list):
    """Align a list of Series on a common date index intersection."""
    if not series_list:
        return series_list
    # Start with the first non-empty series index
    valid_series = [s for s in series_list if not s.empty]
    if not valid_series:
        return series_list

    idx = valid_series[0].dropna().index
    for s in valid_series[1:]:
        idx = idx.intersection(s.dropna().index)
    
    aligned = [s.loc[idx] for s in series_list]
    return aligned

def normalize_to_base(s: pd.Series, base_value: float = 100.0) -> pd.Series:
    if s.empty:
        return s
    first_val = s.iloc[0]
    if first_val == 0 or np.isnan(first_val):
        return s
    return (s / first_val) * base_value

def moving_average(s: pd.Series, window: int) -> pd.Series:
    if s.empty:
        return s
    return s.rolling(window).mean()

def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num_aligned, den_aligned = align_series([num, den])
    if len(num_aligned) == 0:
        return pd.Series(dtype=float)
    
    # Check alignment success
    if num_aligned.empty or den_aligned.empty:
        return pd.Series(dtype=float)

    raw = num_aligned / den_aligned
    return raw.replace([np.inf, -np.inf], np.nan).dropna()

# ---------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------

def build_yield_curve_figure(start_date: dt.date, end_date: dt.date, outdir: str) -> go.Figure:
    """
    Yield curve proxy: 10Y (^TNX) - 3M (^IRX).
    Saves CSV: yield_curve_data.csv
    """
    t10_df = load_ohlcv("^TNX")
    t3_df = load_ohlcv("^IRX")

    t10 = restrict_date_range(pick_price_series(t10_df, name="10Y"), start_date, end_date)
    t3 = restrict_date_range(pick_price_series(t3_df, name="3M"), start_date, end_date)

    t10, t3 = align_series([t10, t3])
    
    if len(t10) == 0:
        logging.warning("No overlapping data for Yield Curve.")
        return go.Figure()

    spread = (t10 - t3)
    spread.name = "10Y_minus_3M"

    # STEGO Export
    export_df = pd.DataFrame({'TNX_10Y': t10, 'IRX_3M': t3, 'Spread': spread})
    save_stego_csv(export_df, "yield_curve_data.csv", outdir)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=["10Y vs 3M Treasury Yields", "Yield Curve Slope: 10Y - 3M"]
    )

    fig.add_trace(go.Scatter(x=t10.index, y=t10.values, mode="lines", name="10Y (^TNX)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t3.index, y=t3.values, mode="lines", name="3M (^IRX)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, mode="lines", name="10Y - 3M"), row=2, col=1)

    # Zero-line
    fig.add_trace(go.Scatter(
        x=spread.index, y=[0.0] * len(spread), mode="lines",
        name="Flat", line=dict(dash="dash", color='gray'), showlegend=False
    ), row=2, col=1)

    # Inversion shading
    inv_mask = spread < 0
    if inv_mask.any():
        inv = spread.copy()
        inv[~inv_mask] = np.nan
        fig.add_trace(go.Scatter(
            x=inv.index, y=inv.values, mode="lines", name="Inversion",
            fill="tozeroy", fillcolor="rgba(255,0,0,0.2)", line=dict(width=0), showlegend=True
        ), row=2, col=1)

    fig.update_layout(title="Yield Curve Proxy (10Y vs 3M)", hovermode="x unified")
    return fig


def build_credit_spread_figure(start_date: dt.date, end_date: dt.date, outdir: str) -> go.Figure:
    """
    HYG vs LQD normalized + Spread.
    Saves CSV: credit_spread_data.csv
    """
    hyg_df = load_ohlcv("HYG")
    lqd_df = load_ohlcv("LQD")

    hyg = restrict_date_range(pick_price_series(hyg_df, "HYG"), start_date, end_date)
    lqd = restrict_date_range(pick_price_series(lqd_df, "LQD"), start_date, end_date)
    hyg, lqd = align_series([hyg, lqd])

    if len(hyg) == 0:
        logging.warning("No overlapping data for Credit Spreads.")
        return go.Figure()

    hyg_norm = normalize_to_base(hyg, 100.0)
    lqd_norm = normalize_to_base(lqd, 100.0)
    spread = hyg_norm - lqd_norm
    spread.name = "HYG_minus_LQD_Norm"

    # STEGO Export
    export_df = pd.DataFrame({
        'HYG_Raw': hyg, 'LQD_Raw': lqd,
        'HYG_Norm': hyg_norm, 'LQD_Norm': lqd_norm,
        'Spread_Norm': spread
    })
    save_stego_csv(export_df, "credit_spread_data.csv", outdir)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=["High Yield (HYG) vs IG (LQD) Normalized", "Relative Stress: HYG - LQD"]
    )

    fig.add_trace(go.Scatter(x=hyg_norm.index, y=hyg_norm.values, mode="lines", name="HYG (Norm)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=lqd_norm.index, y=lqd_norm.values, mode="lines", name="LQD (Norm)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, mode="lines", name="Spread"), row=2, col=1)
    fig.add_hline(y=0.0, line=dict(dash="dash", color='gray'), row=2, col=1)

    fig.update_layout(title="Credit Spread Proxy (HYG vs LQD)", hovermode="x unified")
    return fig


def find_dxy_proxy(start_date: dt.date, end_date: dt.date):
    candidates = ["DX-Y.NYB", "DXY", "UUP"]
    for tkr in candidates:
        df = load_ohlcv(tkr)
        s = pick_price_series(df, name=tkr)
        s = restrict_date_range(s, start_date, end_date)
        if not s.empty:
            logging.info(f"Using {tkr} as DXY proxy.")
            return s, tkr
    logging.warning("Failed to load any DXY proxy.")
    return pd.Series(dtype=float), None


def build_dxy_figure(start_date: dt.date, end_date: dt.date, outdir: str) -> go.Figure:
    """
    DXY Proxy with 200d MA.
    Saves CSV: dxy_data.csv
    """
    dxy, ticker_used = find_dxy_proxy(start_date, end_date)
    if dxy.empty or ticker_used is None:
        return go.Figure()

    ma200 = moving_average(dxy, 200)
    ma200.name = "MA_200"

    # STEGO Export
    export_df = pd.DataFrame({'DXY_Proxy': dxy, 'MA_200': ma200})
    save_stego_csv(export_df, "dxy_data.csv", outdir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dxy.index, y=dxy.values, mode="lines", name=f"{ticker_used}"))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, mode="lines", name="200d MA"))

    fig.update_layout(title=f"USD Dollar Index Proxy ({ticker_used})", hovermode="x unified")
    return fig


def build_breadth_figure(start_date: dt.date, end_date: dt.date, outdir: str) -> go.Figure:
    """
    Equity Breadth: A/D Line and % > 50 SMA.
    Saves CSV: market_breadth_data.csv
    """
    universe = [
        "SPY", "QQQ", "IWM", "DIA",
        "XLK", "XLF", "XLE", "XLY", "XLV", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    ]

    price_map = {}
    for tkr in universe:
        df = load_ohlcv(tkr)
        s = restrict_date_range(pick_price_series(df, name=tkr), start_date, end_date)
        if not s.empty:
            price_map[tkr] = s

    if not price_map:
        logging.warning("No breadth universe data available.")
        return go.Figure()

    aligned = align_series(list(price_map.values()))
    if not aligned:
        return go.Figure()
        
    tickers_used = list(price_map.keys())
    price_df = pd.DataFrame({tkr: s for tkr, s in zip(tickers_used, aligned)}).sort_index()

    # Calculations
    daily_diff = price_df.diff()
    advances = (daily_diff > 0).sum(axis=1)
    declines = (daily_diff < 0).sum(axis=1)
    ad_line = (advances - declines).cumsum()
    ad_line.name = "AD_Line"

    sma_50 = price_df.rolling(50).mean()
    above_50 = (price_df > sma_50).sum(axis=1)
    pct_above_50 = 100.0 * above_50 / float(price_df.shape[1])
    pct_above_50.name = "Pct_Above_50d"

    if "SPY" in price_df.columns:
        spy = price_df["SPY"]
    else:
        spy = price_df.iloc[:, 0]
    spy.name = "Ref_Price"

    # STEGO Export
    export_df = pd.DataFrame({
        'Ref_Price': spy,
        'AD_Line': ad_line,
        'Pct_Above_50d': pct_above_50
    })
    save_stego_csv(export_df, "market_breadth_data.csv", outdir)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=["SPY Price", "Advance/Decline Line", "% Above 50d SMA"]
    )

    fig.add_trace(go.Scatter(x=spy.index, y=spy.values, mode="lines", name="SPY"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ad_line.index, y=ad_line.values, mode="lines", name="A/D Line"), row=2, col=1)
    fig.add_trace(go.Scatter(x=pct_above_50.index, y=pct_above_50.values, mode="lines", name="% > 50d"), row=3, col=1)

    fig.update_layout(title="Equity Breadth (ETF Universe)", hovermode="x unified")
    return fig


def build_commodity_curve_figure(start_date: dt.date, end_date: dt.date, outdir: str) -> go.Figure:
    """
    Oil (USO/USL) and NatGas (UNG/UNL) curve proxies.
    Saves CSV: commodity_curve_data.csv
    """
    pairs = [
        ("USO", "USL"),
        ("UNG", "UNL"),
    ]

    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.1, horizontal_spacing=0.08,
        subplot_titles=[
            "Oil: USO vs USL", "Nat Gas: UNG vs UNL",
            "Oil Ratio (Front/Long)", "Nat Gas Ratio (Front/Long)"
        ]
    )

    # Dictionary to collect data for STEGO export
    export_data = {}

    for col_idx, (front_tkr, long_tkr) in enumerate(pairs, start=1):
        front_df = load_ohlcv(front_tkr)
        long_df = load_ohlcv(long_tkr)

        front = restrict_date_range(pick_price_series(front_df, front_tkr), start_date, end_date)
        long = restrict_date_range(pick_price_series(long_df, long_tkr), start_date, end_date)

        if front.empty or long.empty:
            logging.warning(f"Missing data for {front_tkr}/{long_tkr}")
            continue

        front, long = align_series([front, long])
        front_norm = normalize_to_base(front, 100.0)
        long_norm = normalize_to_base(long, 100.0)
        ratio = safe_ratio(front, long)

        # Collect for CSV
        export_data[f"{front_tkr}_Raw"] = front
        export_data[f"{long_tkr}_Raw"] = long
        export_data[f"{front_tkr}_Norm"] = front_norm
        export_data[f"{long_tkr}_Norm"] = long_norm
        export_data[f"{front_tkr}_{long_tkr}_Ratio"] = ratio

        # Top row: Normalized
        fig.add_trace(go.Scatter(x=front_norm.index, y=front_norm.values, mode="lines", name=f"{front_tkr}"), row=1, col=col_idx)
        fig.add_trace(go.Scatter(x=long_norm.index, y=long_norm.values, mode="lines", name=f"{long_tkr}"), row=1, col=col_idx)

        # Bottom row: Ratio
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode="lines", name="Ratio"), row=2, col=col_idx)
        fig.add_hline(y=1.0, line=dict(dash="dash", color='gray'), row=2, col=col_idx)

    # STEGO Export
    if export_data:
        export_df = pd.DataFrame(export_data).sort_index()
        save_stego_csv(export_df, "commodity_curve_data.csv", outdir)

    fig.update_layout(title="Commodity Curve Structure Proxies", hovermode="x unified")
    return fig


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Macro Cross-Asset Dashboard")
    parser.add_argument("--start-date", type=str, default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=dt.date.today().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser tabs")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    
    if start_date > end_date:
        logging.error("start-date must be <= end-date.")
        sys.exit(1)

    outdir = ensure_output_dir()
    logging.info(f"Output directory: {outdir}")
    
    # 1) Yield Curve
    logging.info("Building Yield Curve...")
    fig_yield = build_yield_curve_figure(start_date, end_date, outdir)
    path_yield = os.path.join(outdir, "yield_curve.html")
    pio.write_html(fig_yield, file=path_yield, auto_open=False, include_plotlyjs="cdn")

    # 2) Credit Spreads
    logging.info("Building Credit Spreads...")
    fig_credit = build_credit_spread_figure(start_date, end_date, outdir)
    path_credit = os.path.join(outdir, "credit_spreads.html")
    pio.write_html(fig_credit, file=path_credit, auto_open=False, include_plotlyjs="cdn")

    # 3) DXY
    logging.info("Building DXY...")
    fig_dxy = build_dxy_figure(start_date, end_date, outdir)
    path_dxy = os.path.join(outdir, "dxy_dashboard.html")
    pio.write_html(fig_dxy, file=path_dxy, auto_open=False, include_plotlyjs="cdn")

    # 4) Breadth
    logging.info("Building Breadth...")
    fig_breadth = build_breadth_figure(start_date, end_date, outdir)
    path_breadth = os.path.join(outdir, "breadth_dashboard.html")
    pio.write_html(fig_breadth, file=path_breadth, auto_open=False, include_plotlyjs="cdn")

    # 5) Commodities
    logging.info("Building Commodities...")
    fig_comms = build_commodity_curve_figure(start_date, end_date, outdir)
    path_comms = os.path.join(outdir, "commodity_curve_dashboard.html")
    pio.write_html(fig_comms, file=path_comms, auto_open=False, include_plotlyjs="cdn")

    logging.info("Processing complete. All artifacts saved.")

    if not args.no_open:
        for p in [path_yield, path_credit, path_dxy, path_breadth, path_comms]:
            if os.path.exists(p):
                webbrowser.open("file://" + os.path.abspath(p), new=2)

if __name__ == "__main__":
    main()
