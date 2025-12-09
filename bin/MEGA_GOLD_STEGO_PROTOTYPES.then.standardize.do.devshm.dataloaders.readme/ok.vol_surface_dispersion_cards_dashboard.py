#!/usr/bin/env python3
"""
SCRIPTNAME: vol_surface_dispersion_cards_dashboard.py
AUTHOR: ChatGPT (for Michael Derby)
DATE: November 26, 2025

PURPOSE
-------
Standalone Python script that implements:

1) Vol-Surface Rotation "Card"
   - Uses CBOE indices as liquid proxies for term-structure and skew:
     * ^VIX9D (9-day implied volatility)
     * ^VIX   (30-day implied volatility)
     * ^VIX3M (3-month implied volatility)
     * ^SKEW  (CBOE SKEW Index; proxy for tail/skew / RR25-like behavior)
   - Computes term-structure slopes and daily changes.
   - Computes daily change in skew.
   - Computes rolling z-scores for these changes.
   - Flags a "Vol Surface Rotation" regime when:
       |z(ΔSlope_front)| >= z_thresh OR
       |z(ΔSKEW)|       >= z_thresh OR
       sign flip in slope (regime flip).
   - Outputs a text "card" summarizing latest regime.
   - Visualizes:
       * Term structure levels over time.
       * Term-structure slope & Δ-slope with elevated regimes highlighted.
       * Skew (^SKEW) over time.

2) Cross-Sector Flow / Dispersion "Card"
   - Uses SPDR sector ETFs as proxies:
     [XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY, XLC]
   - Optionally SPY as market reference.
   - Computes:
       * Daily % returns for each sector ETF.
       * Cross-sectional dispersion = std across sector returns each day.
       * Breadth = fraction of sectors with positive return each day.
       * Rolling z-scores for dispersion and breadth.
     Also:
       * Vol-of-vol proxy: ^VVIX (if available) or realized vol of ^VIX.
   - Flags a "Cross-Sector Dispersion / Flow" regime when:
       z_dispersion >= disp_z_thresh OR
       z_breadth    >= breadth_z_thresh
     (optionally conditioned on elevated vol-of-vol).
   - Outputs a text "card" summarizing latest regime and possible implications.
   - Visualizes:
       * Time series of dispersion + z-score.
       * Breadth over time.
       * Heatmap of recent sector daily returns.
       * Scatter: dispersion vs VIX level.

IMPLEMENTATION NOTES
--------------------
- Uses yfinance for all data.
- Uses pandas/numpy for calculations.
- Uses plotly for an interactive dashboard (single HTML file with subplots).
- No external custom loaders; completely standalone.
- Designed to be reasonably robust to missing tickers: skips gracefully.
"""

import argparse
import datetime as dt
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

try:
    import yfinance as yf
except ImportError as e:
    print("ERROR: yfinance is required. Install via: pip install yfinance", file=sys.stderr)
    raise e


# -----------------------------
# Utility Functions
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vol-Surface Rotation + Cross-Sector Dispersion Cards Dashboard (Standalone)."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Start date for historical data (YYYY-MM-DD). Default: 2010-01-01",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=dt.date.today().strftime("%Y-%m-%d"),
        help="End date for historical data (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=60,
        help="Rolling window (days) for vol-surface z-scores. Default: 60",
    )
    parser.add_argument(
        "--disp-window",
        type=int,
        default=60,
        help="Rolling window (days) for dispersion/breadth z-scores. Default: 60",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=1.28,
        help="Z-score threshold for vol-surface card triggers (approx 90th percentile). Default: 1.28",
    )
    parser.add_argument(
        "--disp-z-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for dispersion card triggers. Default: 2.0",
    )
    parser.add_argument(
        "--breadth-z-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for breadth card triggers. Default: 2.0",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="vol_surface_dispersion_dashboard.html",
        help="Output HTML file name. Default: vol_surface_dispersion_dashboard.html",
    )
    return parser.parse_args()


def safe_download(
    tickers: List[str], start: str, end: str, column: str = "Adj Close"
) -> pd.DataFrame:
    """
    Download historical data for a list of tickers using yfinance.
    Returns a DataFrame with dates as index and tickers as columns.
    Skips tickers that fully fail or have no valid price column.
    """
    data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        # Standard yfinance multi-index
        out = {}
        for t in tickers:
            try:
                series = data[t][column].dropna()
                if not series.empty:
                    out[t] = series
            except Exception:
                continue
        if not out:
            return pd.DataFrame()
        df = pd.DataFrame(out)
    else:
        # Single ticker
        df = data[column].to_frame()
        df.columns = [tickers[0]]
    return df.sort_index()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score of a Series.
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z


# -----------------------------
# Vol-Surface Rotation Card
# -----------------------------

def compute_vol_surface_features(
    start: str,
    end: str,
    window: int,
) -> Dict[str, pd.DataFrame]:
    """
    Download and compute vol-surface related features:
    - VIX9D, VIX, VIX3M, SKEW, VVIX
    - Term-structure slopes, deltas, z-scores
    - Skew deltas, z-scores
    - Vol-of-vol proxy
    Returns dict of DataFrames / Series.
    """

    vol_tickers = ["^VIX9D", "^VIX", "^VIX3M", "^SKEW", "^VVIX"]
    vol_df = safe_download(vol_tickers, start, end, column="Adj Close")
    if vol_df.empty:
        raise RuntimeError("Failed to download any volatility index data. Check network or ticker availability.")

    # Ensure we have at least VIX
    if "^VIX" not in vol_df.columns:
        raise RuntimeError("VIX (^VIX) data not available. Cannot compute vol-surface metrics.")

    # Term-structure slope: front (VIX - VIX9D), mid (VIX3M - VIX)
    vix = vol_df.get("^VIX")
    vix9d = vol_df.get("^VIX9D")
    vix3m = vol_df.get("^VIX3M")
    skew = vol_df.get("^SKEW")
    vvix = vol_df.get("^VVIX")

    # Align all
    combined_index = vol_df.index
    vix = vix.reindex(combined_index)
    if vix9d is None or vix9d.empty:
        # Fallback: approximate short-term vol via VIX itself; slope_front = 0
        vix9d = vix.copy()
    if vix3m is None or vix3m.empty:
        # Fallback: approximate 3m vol via VIX; slope_mid = 0
        vix3m = vix.copy()
    if skew is None:
        skew = pd.Series(index=combined_index, dtype=float)
    if vvix is None:
        vvix = pd.Series(index=combined_index, dtype=float)

    slope_front = vix - vix9d
    slope_mid = vix3m - vix
    delta_slope_front = slope_front.diff()
    delta_slope_mid = slope_mid.diff()
    delta_skew = skew.diff()

    z_delta_slope_front = rolling_zscore(delta_slope_front, window)
    z_delta_slope_mid = rolling_zscore(delta_slope_mid, window)
    z_delta_skew = rolling_zscore(delta_skew, window)

    # Vol-of-vol proxy: z-score of VVIX (level) if available, else z of |ΔVIX|
    if vvix.notna().sum() > 10:
        z_vvix = rolling_zscore(vvix, window)
        vol_of_vol_proxy = z_vvix
    else:
        dvix = vix.pct_change().abs()
        vol_of_vol_proxy = rolling_zscore(dvix, window)

    result = {
        "vol_df": vol_df,
        "vix": vix,
        "vix9d": vix9d,
        "vix3m": vix3m,
        "skew": skew,
        "vvix": vvix,
        "slope_front": slope_front,
        "slope_mid": slope_mid,
        "delta_slope_front": delta_slope_front,
        "delta_slope_mid": delta_slope_mid,
        "delta_skew": delta_skew,
        "z_delta_slope_front": z_delta_slope_front,
        "z_delta_slope_mid": z_delta_slope_mid,
        "z_delta_skew": z_delta_skew,
        "vol_of_vol_proxy": vol_of_vol_proxy,
    }
    return result


def generate_vol_surface_card(
    features: Dict[str, pd.DataFrame],
    z_threshold: float,
) -> str:
    """
    Build a human-readable "Vol-Surface Rotation Card" string for the latest date.
    """
    idx = features["vix"].dropna().index
    if len(idx) == 0:
        return "Vol-Surface Rotation Card: No VIX data available.\n"

    last_date = idx[-1]
    prev_date = idx[-2] if len(idx) > 1 else None

    vix = features["vix"].loc[last_date]
    vix9d = features["vix9d"].loc[last_date]
    vix3m = features["vix3m"].loc[last_date]
    skew = features["skew"].loc[last_date] if not np.isnan(features["skew"].loc[last_date]) else None
    vvix = features["vvix"].loc[last_date] if not np.isnan(features["vvix"].loc[last_date]) else None

    slope_front = features["slope_front"].loc[last_date]
    slope_mid = features["slope_mid"].loc[last_date]
    delta_slope_front = features["delta_slope_front"].loc[last_date]
    delta_slope_mid = features["delta_slope_mid"].loc[last_date]
    delta_skew = features["delta_skew"].loc[last_date]

    z_delta_slope_front = features["z_delta_slope_front"].loc[last_date]
    z_delta_slope_mid = features["z_delta_slope_mid"].loc[last_date]
    z_delta_skew = features["z_delta_skew"].loc[last_date]
    vol_of_vol_proxy = features["vol_of_vol_proxy"].loc[last_date]

    # Determine triggers
    triggers = []
    regime = "Calm / Neutral"

    if abs(z_delta_slope_front) >= z_threshold:
        triggers.append(f"|z(ΔSlope_front)| >= {z_threshold:.2f}")
    if abs(z_delta_skew) >= z_threshold:
        triggers.append(f"|z(ΔSkew)| >= {z_threshold:.2f}")
    if prev_date is not None:
        prev_slope = features["slope_front"].loc[prev_date]
        if slope_front * prev_slope < 0:
            triggers.append("Sign flip in front-term slope (regime flip)")

    if triggers:
        regime = "Rotation / Stress in Vol Surface"

    card_lines = []
    card_lines.append("========== VOL-SURFACE ROTATION CARD ==========")
    card_lines.append(f"As-of: {last_date.date()}")
    card_lines.append("")
    card_lines.append("Vol Levels (approx 30d / 9d / 3m):")
    card_lines.append(f"  VIX   (30d): {vix:6.2f}")
    card_lines.append(f"  VIX9D ( 9d): {vix9d:6.2f}")
    card_lines.append(f"  VIX3M ( 3m): {vix3m:6.2f}")
    if skew is not None:
        card_lines.append(f"  SKEW Index: {skew:6.2f}")
    if vvix is not None and not np.isnan(vvix):
        card_lines.append(f"  VVIX (vol-of-vol): {vvix:6.2f}")
    card_lines.append("")
    card_lines.append("Term-Structure Slopes (Level):")
    card_lines.append(f"  Slope_front (VIX - VIX9D): {slope_front:7.3f}")
    card_lines.append(f"  Slope_mid   (VIX3M - VIX): {slope_mid:7.3f}")
    card_lines.append("")
    card_lines.append("Daily Changes (Δ) and Rolling Z-scores:")
    card_lines.append(
        f"  ΔSlope_front: {delta_slope_front: .4f} | z(ΔSlope_front): {z_delta_slope_front: .2f}"
    )
    card_lines.append(
        f"  ΔSlope_mid:   {delta_slope_mid: .4f} | z(ΔSlope_mid):   {z_delta_slope_mid: .2f}"
    )
    if delta_skew is not None and not np.isnan(delta_skew):
        card_lines.append(
            f"  ΔSkew (SKEW): {delta_skew: .4f} | z(ΔSkew):         {z_delta_skew: .2f}"
        )
    card_lines.append("")
    card_lines.append(f"Vol-of-Vol Proxy (latest z): {vol_of_vol_proxy: .2f}")
    card_lines.append("")
    card_lines.append(f"Regime Assessment: {regime}")
    if triggers:
        card_lines.append("Triggers:")
        for t in triggers:
            card_lines.append(f"  - {t}")
    else:
        card_lines.append("Triggers: None (within threshold)")

    card_lines.append("===============================================")
    return "\n".join(card_lines) + "\n"


# -----------------------------
# Cross-Sector Dispersion Card
# -----------------------------

def compute_sector_dispersion_features(
    start: str,
    end: str,
    window: int,
) -> Dict[str, pd.DataFrame]:
    """
    Compute cross-sector dispersion, breadth, and vol-of-vol proxy.
    """
    sector_tickers = [
        "XLB", "XLE", "XLF", "XLI", "XLK",
        "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC"
    ]
    market_ticker = "SPY"

    price_df = safe_download(sector_tickers + [market_ticker], start, end, column="Adj Close")
    if price_df.empty:
        raise RuntimeError("Failed to download sector / SPY data.")

    # Keep only sectors that actually downloaded
    sectors_available = [t for t in sector_tickers if t in price_df.columns]
    if not sectors_available:
        raise RuntimeError("No sector data available from yfinance for specified tickers.")

    sectors_df = price_df[sectors_available]
    market_series = price_df[market_ticker] if market_ticker in price_df.columns else None

    # Daily returns
    sector_ret = sectors_df.pct_change().dropna()
    if market_series is not None:
        market_ret = market_series.pct_change().reindex(sector_ret.index)
    else:
        market_ret = None

    # Cross-sectional dispersion: std dev across sectors each day
    dispersion = sector_ret.std(axis=1)

    # Breadth: fraction of sectors with positive returns each day
    breadth = (sector_ret > 0).sum(axis=1) / sector_ret.shape[1]

    # Z-scores
    z_dispersion = rolling_zscore(dispersion, window)
    z_breadth = rolling_zscore(breadth, window)

    # Vol-of-vol proxy: use VVIX if available, else realized vol of VIX (from vol-surface features)
    # We'll just re-download ^VVIX quickly (or reuse if already downloaded in calling code).
    vvix_df = safe_download(["^VVIX"], start, end, column="Adj Close")
    if not vvix_df.empty and "^VVIX" in vvix_df.columns and vvix_df["^VVIX"].notna().sum() > 10:
        vvix_aligned = vvix_df["^VVIX"].reindex(sector_ret.index)
        vol_of_vol_proxy = rolling_zscore(vvix_aligned, window)
    else:
        vix_df = safe_download(["^VIX"], start, end, column="Adj Close")
        if not vix_df.empty and "^VIX" in vix_df.columns:
            vix_series = vix_df["^VIX"].reindex(sector_ret.index)
            dvix_abs = vix_series.pct_change().abs()
            vol_of_vol_proxy = rolling_zscore(dvix_abs, window)
        else:
            vol_of_vol_proxy = pd.Series(index=sector_ret.index, dtype=float)

    result = {
        "price_df": price_df,
        "sectors_df": sectors_df,
        "sector_ret": sector_ret,
        "market_ret": market_ret,
        "dispersion": dispersion,
        "breadth": breadth,
        "z_dispersion": z_dispersion,
        "z_breadth": z_breadth,
        "vol_of_vol_proxy": vol_of_vol_proxy,
        "sectors_available": sectors_available,
    }
    return result


def generate_dispersion_card(
    features: Dict[str, pd.DataFrame],
    disp_z_threshold: float,
    breadth_z_threshold: float,
) -> str:
    """
    Build a human-readable "Cross-Sector Dispersion / Flow Card" string for the latest date.
    """
    idx = features["dispersion"].dropna().index
    if len(idx) == 0:
        return "Cross-Sector Dispersion Card: No dispersion data available.\n"

    last_date = idx[-1]

    dispersion = features["dispersion"].loc[last_date]
    breadth = features["breadth"].loc[last_date]
    z_dispersion = features["z_dispersion"].loc[last_date]
    z_breadth = features["z_breadth"].loc[last_date]
    vol_of_vol_proxy = features["vol_of_vol_proxy"].loc[last_date]

    regime = "Normal / Low-Dispersion"
    triggers = []

    if z_dispersion >= disp_z_threshold:
        triggers.append(f"Dispersion z >= {disp_z_threshold:.2f}")
    if z_breadth >= breadth_z_threshold:
        triggers.append(f"Breadth z >= {breadth_z_threshold:.2f}")

    if triggers:
        regime = "High Dispersion / Rotation Environment"

    # Approx "opportunity commentary"
    implications = []
    if z_dispersion >= disp_z_threshold:
        implications.append("Elevated cross-sectional volatility: wider spread in sector returns (good for relative value / rotation, riskier for passive beta).")
    if z_breadth >= breadth_z_threshold:
        if breadth > 0.5:
            implications.append("Strong positive breadth: many sectors up together (broad risk-on or melt-up conditions).")
        else:
            implications.append("Strong negative breadth: many sectors down together (broad risk-off).")
    if abs(vol_of_vol_proxy) > 1.0:
        implications.append("Vol-of-vol elevated: options markets pricing unstable volatility regimes (consider hedging / vol-structure trades).")

    card_lines = []
    card_lines.append("====== CROSS-SECTOR DISPERSION / FLOW CARD =====")
    card_lines.append(f"As-of: {last_date.date()}")
    card_lines.append("")
    card_lines.append("Latest Cross-Section Metrics:")
    card_lines.append(f"  Dispersion (σ across sector daily returns): {dispersion: .4f}")
    card_lines.append(f"  Dispersion z-score (rolling):             {z_dispersion: .2f}")
    card_lines.append(f"  Breadth (fraction sectors up):            {breadth: .2f}")
    card_lines.append(f"  Breadth z-score (rolling):                {z_breadth: .2f}")
    card_lines.append(f"  Vol-of-Vol Proxy (latest z):              {vol_of_vol_proxy: .2f}")
    card_lines.append("")
    card_lines.append(f"Regime Assessment: {regime}")
    if triggers:
        card_lines.append("Triggers:")
        for t in triggers:
            card_lines.append(f"  - {t}")
    else:
        card_lines.append("Triggers: None (within thresholds)")
    card_lines.append("")
    if implications:
        card_lines.append("Heuristic Implications / Playbook Hooks:")
        for imp in implications:
            card_lines.append(f"  - {imp}")
    else:
        card_lines.append("Heuristic Implications / Playbook Hooks:")
        card_lines.append("  - Stable / low-dispersion regime; relative-value edges may be thinner.")
    card_lines.append("===============================================")

    return "\n".join(card_lines) + "\n"


# -----------------------------
# Dashboard Construction
# -----------------------------

def build_dashboard_figure(
    vol_feats: Dict[str, pd.DataFrame],
    disp_feats: Dict[str, pd.DataFrame],
) -> go.Figure:
    """
    Build a Plotly figure with multiple subplots:
      Row1: VIX term structure + SKEW
      Row2: ΔSlope_front z-score + Skew z and highlights
      Row3: Dispersion & z, Breadth, Heatmap snapshot, Scatter
    To keep it manageable, we use a 3x2 layout:
      (1,1): VIX, VIX9D, VIX3M over time
      (1,2): SKEW over time
      (2,1): z(ΔSlope_front) over time, threshold bands
      (2,2): z(ΔSKEW) over time, threshold bands
      (3,1): Dispersion + its z-score
      (3,2): Sector return heatmap (recent window)
    """
    # Common index merges
    vol_index = vol_feats["vix"].index
    disp_index = disp_feats["dispersion"].index

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=False,
        subplot_titles=[
            "VIX Term Structure (9D / 30D / 3M)",
            "CBOE SKEW Index (Tail Risk / Skew Proxy)",
            "z(ΔSlope_front) - Vol-Surface Rotation",
            "z(ΔSkew) - Skew Rotation",
            "Sector Return Dispersion & z-score",
            "Sector Return Heatmap (Recent)",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # ----- (1,1) VIX Term Structure -----
    fig.add_trace(
        go.Scatter(
            x=vol_index,
            y=vol_feats["vix9d"],
            name="VIX9D (9d)",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vol_index,
            y=vol_feats["vix"],
            name="VIX (30d)",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vol_index,
            y=vol_feats["vix3m"],
            name="VIX3M (3m)",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    # ----- (1,2) SKEW -----
    fig.add_trace(
        go.Scatter(
            x=vol_index,
            y=vol_feats["skew"],
            name="SKEW Index",
            mode="lines",
        ),
        row=1,
        col=2,
    )

    # ----- (2,1) z(ΔSlope_front) -----
    fig.add_trace(
        go.Scatter(
            x=vol_index,
            y=vol_feats["z_delta_slope_front"],
            name="z(ΔSlope_front)",
            mode="lines",
        ),
        row=2,
        col=1,
    )

    # ----- (2,2) z(ΔSkew) -----
    fig.add_trace(
        go.Scatter(
            x=vol_index,
            y=vol_feats["z_delta_skew"],
            name="z(ΔSkew)",
            mode="lines",
        ),
        row=2,
        col=2,
    )

    # ----- (3,1) Dispersion & z -----
    disp_idx = disp_feats["dispersion"].index
    fig.add_trace(
        go.Scatter(
            x=disp_idx,
            y=disp_feats["dispersion"],
            name="Dispersion (σ across sectors)",
            mode="lines",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=disp_idx,
            y=disp_feats["z_dispersion"],
            name="z(Dispersion)",
            mode="lines",
            yaxis="y5",
        ),
        row=3,
        col=1,
    )

    # Create secondary y-axis manually for row3,col1
    # Plotly's make_subplots supports secondary_y per subplot, but it's more complex.
    # Instead we just overlay them and interpret qualitatively.

    # ----- (3,2) Sector Return Heatmap (recent) -----
    sector_ret = disp_feats["sector_ret"]
    sectors = disp_feats["sectors_available"]
    # Take last 90 days (or fewer)
    recent_days = 90
    sector_ret_recent = sector_ret.tail(recent_days)
    # Build heatmap with rows = sectors, columns = dates
    heatmap_z = sector_ret_recent[sectors].T.values
    heatmap_x = sector_ret_recent.index
    heatmap_y = sectors

    heatmap_trace = go.Heatmap(
        x=heatmap_x,
        y=heatmap_y,
        z=heatmap_z,
        colorbar=dict(title="Daily Return"),
        zmid=0.0,
    )
    fig.add_trace(heatmap_trace, row=3, col=2)

    # Layout tweaks
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=2)

    fig.update_yaxes(title_text="Vol Level", row=1, col=1)
    fig.update_yaxes(title_text="SKEW", row=1, col=2)
    fig.update_yaxes(title_text="z-score", row=2, col=1)
    fig.update_yaxes(title_text="z-score", row=2, col=2)
    fig.update_yaxes(title_text="Dispersion / z", row=3, col=1)
    fig.update_yaxes(title_text="Sector", row=3, col=2)

    fig.update_layout(
        title="Vol-Surface Rotation & Cross-Sector Dispersion Dashboard",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=80, b=80),
    )

    return fig


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    start = args.start
    end = args.end
    vol_window = args.vol_window
    disp_window = args.disp_window
    z_threshold = args.z_threshold
    disp_z_threshold = args.disp_z_threshold
    breadth_z_threshold = args.breadth_z_threshold
    output_html = args.output_html

    print("===============================================")
    print("Vol-Surface & Cross-Sector Dispersion Dashboard")
    print("===============================================")
    print(f"Date range: {start} -> {end}")
    print(f"Vol-surface rolling window: {vol_window} days (z-threshold={z_threshold:.2f})")
    print(
        f"Dispersion rolling window: {disp_window} days "
        f"(disp_z_threshold={disp_z_threshold:.2f}, breadth_z_threshold={breadth_z_threshold:.2f})"
    )
    print("Fetching data from yfinance...")

    try:
        vol_feats = compute_vol_surface_features(start, end, vol_window)
    except Exception as e:
        print(f"ERROR while computing vol-surface features: {e}", file=sys.stderr)
        return

    try:
        disp_feats = compute_sector_dispersion_features(start, end, disp_window)
    except Exception as e:
        print(f"ERROR while computing dispersion features: {e}", file=sys.stderr)
        return

    # Generate and print cards
    vol_card = generate_vol_surface_card(vol_feats, z_threshold)
    disp_card = generate_dispersion_card(disp_feats, disp_z_threshold, breadth_z_threshold)

    print()
    print(vol_card)
    print(disp_card)

    # Build dashboard figure
    fig = build_dashboard_figure(vol_feats, disp_feats)

    # Save to HTML
    import plotly.io as pio

    pio.write_html(fig, file=output_html, auto_open=True)
    print(f"Dashboard written to: {output_html}")


if __name__ == "__main__":
    main()

