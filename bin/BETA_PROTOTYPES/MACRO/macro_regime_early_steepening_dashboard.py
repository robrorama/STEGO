#!/usr/bin/env python3
# SCRIPTNAME: ok.macro_regime_early_steepening_dashboard.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
SCRIPTNAME: macro_regime_early_steepening_dashboard.py

Description:
    Standalone macro-regime dashboard focused on:
      - Yield-curve re-steepening (US 2Y vs 10Y)
      - Low-to-moderate vol in equities (VIX) and bonds (MOVE)
      - Tight credit (HYG as HY-proxy)
      - Dollar strength (DXY)

    It builds multiple creative Plotly visualizations:
      1) Multi-panel time-series dashboard:
         - 2s10s curve steepness with inversion shading
         - Normalized VIX and MOVE
         - Credit & USD regime z-scores
      2) Regime scatter "cloud":
         - X: 2s10s spread, Y: MOVE
         - Color: HY credit z-score (tight vs stressed)
         - Size: VIX (equity vol)
         - Latest point highlighted
      3) Quadrant regime map for TODAY:
         - X: 2s10s spread, Y: MOVE
         - Labeled quadrants ("Fragile risk-on", etc.)
      4) 2D heatmap of history: MOVE vs 2s10s
      5) Summary table of current macro regime metrics and a composite "Risk-On Score"

Requirements:
    - Python 3.9+
    - pip install:
        yfinance
        pandas
        numpy
        plotly

Usage:
    python3 macro_regime_early_steepening_dashboard.py
    python3 macro_regime_early_steepening_dashboard.py --years 5
    python3 macro_regime_early_steepening_dashboard.py --start 2018-01-01 --end 2025-11-30
"""

import argparse
import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf


# -----------------------------
# Helpers
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Early re-steepening / low-vol / tight-credit macro-regime dashboard (standalone, yfinance-based)."
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, derived from --years.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Lookback in years if --start is not provided (default: 3).",
    )
    return parser.parse_args()


def compute_date_range(args: argparse.Namespace) -> Tuple[str, str]:
    today = dt.date.today()
    if args.end:
        end = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end = today

    if args.start:
        start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start = end - dt.timedelta(days=365 * args.years)

    return start.isoformat(), end.isoformat()


def download_series(ticker: str, start: str, end: str) -> pd.Series:
    """
    Download a single ticker from yfinance and return its Adj Close as a Series.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if "Adj Close" in df.columns:
        s = df["Adj Close"].copy()
    else:
        # fallback if only 'Close'
        s = df["Close"].copy()
    s.name = ticker
    return s


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling z-score over a given window. Designed for daily data.
    """
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    z = (series - roll_mean) / roll_std
    return z


def tanh_normalize(x: pd.Series, scale: float) -> pd.Series:
    """
    Smooth normalization using tanh(x / scale).
    """
    return np.tanh(x / scale)


# -----------------------------
# Core regime calculations
# -----------------------------

def build_macro_dataframe(start: str, end: str) -> pd.DataFrame:
    """
    Download all inputs and construct a unified DataFrame.
    """
    tickers = [
        "^VIX",      # Equity vol
        "^MOVE",    # Bond vol (ICE BofAML MOVE Index)
        "US2Y",     # 2-year Treasury yield
        "US10Y",    # 10-year Treasury yield
        "DX-Y.NYB", # Dollar Index (DXY)
        "HYG",      # HY credit ETF proxy
        "SPY",      # Risk asset proxy
    ]

    series_list = []
    for t in tickers:
        try:
            s = download_series(t, start, end)
            series_list.append(s)
        except Exception as e:
            print(f"[WARN] Could not download {t}: {e}")

    if not series_list:
        raise RuntimeError("No data downloaded. Check tickers or internet connection.")

    df = pd.concat(series_list, axis=1).sort_index()
    df = df.ffill().bfill()  # fill gaps sensibly
    return df


def enrich_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived regime features.
    """
    df = df.copy()

    # Yield curve steepness: 10Y - 2Y
    if "US10Y" in df.columns and "US2Y" in df.columns:
        df["2s10s_spread"] = df["US10Y"] - df["US2Y"]
    else:
        df["2s10s_spread"] = np.nan

    # Daily returns for risk assets and HYG
    for col in ["SPY", "HYG"]:
        if col in df.columns:
            df[f"{col}_ret"] = df[col].pct_change()

    # Z-scores
    if "^VIX" in df.columns:
        df["VIX_z"] = zscore(df["^VIX"])
    if "^MOVE" in df.columns:
        df["MOVE_z"] = zscore(df["^MOVE"])

    if "HYG" in df.columns:
        df["HYG_z"] = zscore(df["HYG"])
    if "DX-Y.NYB" in df.columns:
        df["DXY_z"] = zscore(df["DX-Y.NYB"])

    if "2s10s_spread" in df.columns:
        df["2s10s_z"] = zscore(df["2s10s_spread"])

    # Composite "Risk-On Score"
    #   + Steeper curve -> +score
    #   + Lower VIX/MOVE -> +score
    #   + Higher HYG (tight spreads) -> +score
    #   + Weaker DXY -> +score
    spread_norm = tanh_normalize(df.get("2s10s_spread", pd.Series(index=df.index)), scale=1.5)

    vix_centered = df.get("^VIX", pd.Series(index=df.index)) - df.get("^VIX", pd.Series(index=df.index)).median()
    vix_norm = -tanh_normalize(vix_centered, scale=10.0)

    move_centered = df.get("^MOVE", pd.Series(index=df.index)) - df.get("^MOVE", pd.Series(index=df.index)).median()
    move_norm = -tanh_normalize(move_centered, scale=20.0)

    hyg_z = df.get("HYG_z", pd.Series(index=df.index))
    hyg_norm = tanh_normalize(hyg_z, scale=2.0)

    dxy_centered = df.get("DX-Y.NYB", pd.Series(index=df.index)) - df.get("DX-Y.NYB", pd.Series(index=df.index)).median()
    dxy_norm = -tanh_normalize(dxy_centered, scale=5.0)

    df["risk_on_score"] = (
        0.30 * spread_norm +
        0.20 * vix_norm +
        0.20 * move_norm +
        0.20 * hyg_norm +
        0.10 * dxy_norm
    )

    return df


# -----------------------------
# Visualizations
# -----------------------------

def plot_time_series_dashboard(df: pd.DataFrame) -> None:
    """
    Multi-panel time-series:
      - 2s10s with inversion shading
      - VIX & MOVE z-scores
      - Credit & USD z-scores
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=(
            "Yield Curve: 2s10s Spread (US10Y - US2Y)",
            "Volatility Regime: VIX & MOVE (Rolling Z-Scores)",
            "Credit & Dollar Regime: HYG & DXY (Z-Scores)"
        ),
    )

    idx = df.index

    # --- Row 1: 2s10s spread ---
    if "2s10s_spread" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=df["2s10s_spread"],
                mode="lines",
                name="2s10s Spread (bp)",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Spread: %{y:.2f} bp<extra></extra>",
            ),
            row=1, col=1
        )

        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_width=1,
            line_color="gray",
            row=1, col=1
        )

        # Shaded regions for inversion vs steepening
        # Inversion (spread < 0)
        inv_mask = df["2s10s_spread"] < 0
        if inv_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=idx[inv_mask],
                    y=df["2s10s_spread"][inv_mask],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    fill="tozeroy",
                    fillcolor="rgba(255,0,0,0.1)",
                ),
                row=1, col=1
            )

        # Steepening (spread > 0)
        steep_mask = df["2s10s_spread"] > 0
        if steep_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=idx[steep_mask],
                    y=df["2s10s_spread"][steep_mask],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    fill="tozeroy",
                    fillcolor="rgba(0,255,0,0.1)",
                ),
                row=1, col=1
            )

    # --- Row 2: VIX and MOVE z-scores ---
    if "VIX_z" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=df["VIX_z"],
                mode="lines",
                name="VIX z-score",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>VIX z: %{y:.2f}<extra></extra>",
            ),
            row=2, col=1
        )

    if "MOVE_z" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=df["MOVE_z"],
                mode="lines",
                name="MOVE z-score",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>MOVE z: %{y:.2f}<extra></extra>",
            ),
            row=2, col=1
        )

    fig.add_hline(y=0, line_dash="dash", line_width=1, line_color="gray", row=2, col=1)

    # --- Row 3: Credit & Dollar z-scores ---
    if "HYG_z" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=df["HYG_z"],
                mode="lines",
                name="HYG z-score (credit)",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>HYG z: %{y:.2f}<extra></extra>",
            ),
            row=3, col=1
        )
    if "DXY_z" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=df["DXY_z"],
                mode="lines",
                name="DXY z-score",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>DXY z: %{y:.2f}<extra></extra>",
            ),
            row=3, col=1
        )

    fig.add_hline(y=0, line_dash="dash", line_width=1, line_color="gray", row=3, col=1)

    # Layout
    fig.update_layout(
        title="Macro Regime Dashboard: Curve, Vol, Credit & Dollar",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=900,
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.show()


def plot_regime_scatter(df: pd.DataFrame) -> None:
    """
    Regime scatter "cloud":
      - X: 2s10s spread
      - Y: MOVE
      - Color: HYG z-score
      - Size: VIX
    Highlight latest point.
    """
    if "2s10s_spread" not in df.columns or "^MOVE" not in df.columns:
        print("[WARN] Missing 2s10s or MOVE for scatter plot.")
        return

    scatter_df = df.dropna(subset=["2s10s_spread", "^MOVE", "HYG_z", "^VIX"]).copy()
    if scatter_df.empty:
        print("[WARN] Not enough data for regime scatter.")
        return

    latest_idx = scatter_df.index[-1]

    fig = go.Figure()

    # Full history
    fig.add_trace(
        go.Scatter(
            x=scatter_df["2s10s_spread"],
            y=scatter_df["^MOVE"],
            mode="markers",
            name="History",
            marker=dict(
                size=np.clip(5 + scatter_df["^VIX"] / 5.0, 4, 18),
                color=scatter_df["HYG_z"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="HYG z (credit)"),
                opacity=0.6,
            ),
            text=[d.strftime("%Y-%m-%d") for d in scatter_df.index],
            hovertemplate=(
                "Date: %{text}<br>"
                "2s10s: %{x:.2f} bp<br>"
                "MOVE: %{y:.2f}<br>"
                "VIX (size): %{marker.size:.1f}<extra></extra>"
            ),
        )
    )

    # Latest point
    latest_row = scatter_df.loc[latest_idx]
    fig.add_trace(
        go.Scatter(
            x=[latest_row["2s10s_spread"]],
            y=[latest_row["^MOVE"]],
            mode="markers+text",
            name="Latest",
            marker=dict(
                size=24,
                symbol="star",
                line=dict(width=2),
            ),
            text=[latest_idx.strftime("%Y-%m-%d")],
            textposition="top center",
            hovertemplate=(
                "<b>Latest</b><br>"
                "Date: %{text}<br>"
                "2s10s: %{x:.2f} bp<br>"
                "MOVE: %{y:.2f}<br><extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Regime Scatter: Curve Steepness vs Bond Vol (MOVE)",
        xaxis_title="2s10s Spread (US10Y - US2Y, bp)",
        yaxis_title="MOVE Index",
    )
    fig.show()


def plot_regime_quadrant(df: pd.DataFrame) -> None:
    """
    Quadrant map for today:
      X: 2s10s spread
      Y: MOVE
    Quadrants labeled:
      - Q1: Steep + High MOVE -> 'Volatile Reflation'
      - Q2: Steep + Low MOVE -> 'Fragile Risk-On'
      - Q3: Inverted + Low MOVE -> 'Complacent Late-Cycle'
      - Q4: Inverted + High MOVE -> 'Stress / Risk-Off'
    """
    if "2s10s_spread" not in df.columns or "^MOVE" not in df.columns:
        print("[WARN] Missing 2s10s or MOVE for quadrant map.")
        return

    last_valid = df.dropna(subset=["2s10s_spread", "^MOVE"])
    if last_valid.empty:
        print("[WARN] No valid latest point for quadrant map.")
        return

    latest_idx = last_valid.index[-1]
    latest = last_valid.loc[latest_idx]

    # Use median MOVE as dividing line between low/high vol
    move_med = last_valid["^MOVE"].median()

    fig = go.Figure()

    # Vertical line at 0 bp (inversion vs steepening)
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=last_valid["^MOVE"].min() * 0.9,
        y1=last_valid["^MOVE"].max() * 1.1,
        line=dict(dash="dash", width=1),
    )

    # Horizontal line at median MOVE
    fig.add_shape(
        type="line",
        x0=last_valid["2s10s_spread"].min() * 1.1,
        x1=last_valid["2s10s_spread"].max() * 1.1,
        y0=move_med, y1=move_med,
        line=dict(dash="dash", width=1),
    )

    # Quadrant labels
    x_mid_pos = last_valid["2s10s_spread"].max() * 0.7
    x_mid_neg = last_valid["2s10s_spread"].min() * 0.7
    y_high = last_valid["^MOVE"].max() * 0.95
    y_low = last_valid["^MOVE"].min() * 1.05

    fig.add_annotation(x=x_mid_pos, y=y_high, text="Volatile Reflation", showarrow=False)
    fig.add_annotation(x=x_mid_pos, y=y_low, text="Fragile Risk-On", showarrow=False)
    fig.add_annotation(x=x_mid_neg, y=y_low, text="Complacent Late-Cycle", showarrow=False)
    fig.add_annotation(x=x_mid_neg, y=y_high, text="Stress / Risk-Off", showarrow=False)

    # Latest point
    fig.add_trace(
        go.Scatter(
            x=[latest["2s10s_spread"]],
            y=[latest["^MOVE"]],
            mode="markers+text",
            marker=dict(size=20, symbol="star", line=dict(width=2)),
            text=[latest_idx.strftime("%Y-%m-%d")],
            textposition="top center",
            name="Latest",
            hovertemplate=(
                "<b>Latest</b><br>"
                "Date: %{text}<br>"
                "2s10s: %{x:.2f} bp<br>"
                "MOVE: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Current Macro Regime Quadrant (2s10s vs MOVE)",
        xaxis_title="2s10s Spread (bp, >0 = Steepening)",
        yaxis_title="MOVE Index (Low vs High Bond Vol)",
    )

    fig.show()


def plot_heatmap(df: pd.DataFrame) -> None:
    """
    2D histogram heatmap of MOVE vs 2s10s over history.
    """
    if "2s10s_spread" not in df.columns or "^MOVE" not in df.columns:
        print("[WARN] Missing 2s10s or MOVE for heatmap.")
        return

    heat_df = df.dropna(subset=["2s10s_spread", "^MOVE"]).copy()
    if heat_df.empty:
        print("[WARN] Not enough data for heatmap.")
        return

    fig = go.Figure(
        data=go.Histogram2d(
            x=heat_df["2s10s_spread"],
            y=heat_df["^MOVE"],
            nbinsx=40,
            nbinsy=40,
            colorscale="Viridis",
            colorbar=dict(title="Frequency"),
        )
    )
    fig.update_layout(
        title="Regime Density: 2s10s Spread vs MOVE (Full History)",
        xaxis_title="2s10s Spread (bp)",
        yaxis_title="MOVE Index",
    )
    fig.show()


def plot_summary_table(df: pd.DataFrame) -> None:
    """
    Summary table of last observation.
    """
    last = df.iloc[-1]

    fields = [
        ("Date", last.name.strftime("%Y-%m-%d")),
        ("2s10s Spread (bp)", f"{last.get('2s10s_spread', np.nan):.2f}"),
        ("US2Y", f"{last.get('US2Y', np.nan):.2f}"),
        ("US10Y", f"{last.get('US10Y', np.nan):.2f}"),
        ("VIX", f"{last.get('^VIX', np.nan):.2f}"),
        ("MOVE", f"{last.get('^MOVE', np.nan):.2f}"),
        ("HYG Price", f"{last.get('HYG', np.nan):.2f}"),
        ("DXY", f"{last.get('DX-Y.NYB', np.nan):.2f}"),
        ("VIX z", f"{last.get('VIX_z', np.nan):.2f}"),
        ("MOVE z", f"{last.get('MOVE_z', np.nan):.2f}"),
        ("HYG z", f"{last.get('HYG_z', np.nan):.2f}"),
        ("DXY z", f"{last.get('DXY_z', np.nan):.2f}"),
        ("Risk-On Score (−1..+1)", f"{last.get('risk_on_score', np.nan):.2f}"),
    ]

    header = ["Metric", "Value"]
    cells = [list(zip(*fields))[0], list(zip(*fields))[1]]  # metrics, values

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header, fill_color="lightgrey", align="left"),
                cells=dict(values=cells, align="left"),
            )
        ]
    )
    fig.update_layout(title="Macro Regime Snapshot (Latest)")
    fig.show()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    start, end = compute_date_range(args)

    print(f"[INFO] Loading data from {start} to {end} via yfinance...")
    df_raw = build_macro_dataframe(start, end)

    print("[INFO] Computing regime features...")
    df = enrich_macro(df_raw)

    print("[INFO] Building multi-panel time-series dashboard...")
    plot_time_series_dashboard(df)

    print("[INFO] Building regime scatter visualization...")
    plot_regime_scatter(df)

    print("[INFO] Building current regime quadrant...")
    plot_regime_quadrant(df)

    print("[INFO] Building regime density heatmap...")
    plot_heatmap(df)

    print("[INFO] Building summary table...")
    plot_summary_table(df)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

