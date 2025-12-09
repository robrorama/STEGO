#!/usr/bin/env python3
"""
SCRIPTNAME: synthetic_l2_bookmap_heatmap.v1.py

Author: Michael Derby
Framework: STEGO Financial Framework

Description:
    Synthetic "Level-2 style" liquidity and order-flow visualization built solely
    from OHLCV data loaded via data_retrieval.load_or_download_ticker().

    This script constructs:
      - Candle-level synthetic depth ladder (Version B)
      - Pseudo-tick microstructure reconstruction (Version C)
      - Bookmap-like Plotly heatmaps with trade bubbles
      - Microprice, delta, and hidden-liquidity analytics

    Requirements:
      - data_retrieval.py available on PYTHONPATH (unchanged)
      - plotly, numpy, pandas installed
      - yfinance for underlying data via data_retrieval

    All outputs:
      - CSV + HTML written under /dev/shm/SYNTHETIC_L2_BOOKMAP/<TICKER>/<YYYY-MM-DD>
"""

import os
import sys
import math
import argparse
import webbrowser
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio

# Import user's immutable library
import data_retrieval as dr


# ---------------------------------------------------------------------------
# Utility: paths
# ---------------------------------------------------------------------------

def get_output_dir(ticker: str) -> str:
    # Fix DeprecationWarning by using timezone-aware UTC
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    root = os.path.join("/dev/shm", "SYNTHETIC_L2_BOOKMAP", ticker.upper(), today)
    os.makedirs(root, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Synthetic microstructure metrics (Version B core)
# ---------------------------------------------------------------------------

def enrich_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds microstructure features inferred from OHLCV:
      - true_range
      - body_range
      - direction (sign of close-open)
      - microprice proxy
      - imbalance (microprice vs close)
      - hidden_liquidity_ratio (range / body)
      - volatility_proxy (ATR-style)
      - volume_normalized
    """
    df = df.copy()

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in OHLCV frame.")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # Basic ranges
    df["true_range"] = (df["High"] - df["Low"]).abs()
    df["body_range"] = (df["Close"] - df["Open"]).abs()
    df["direction"] = np.sign(df["Close"] - df["Open"]).replace(0, np.nan)

    # Microprice proxy: weighted blend of OHLC
    df["microprice"] = (2.0 * df["Close"] + df["Open"] + df["High"] + df["Low"]) / 5.0

    # Imbalance: where is close vs microprice inside bar range
    eps = 1e-9
    df["imbalance"] = (df["Close"] - df["microprice"]) / (df["true_range"] + eps)

    # Hidden liquidity: large range but small body suggests absorption
    df["hidden_liquidity_ratio"] = df["true_range"] / (df["body_range"] + eps)

    # Simple volatility proxy (ATR-like)
    df["atr_like"] = df["true_range"].rolling(window=14, min_periods=1).mean()

    # Volume normalization
    vol_mean = df["Volume"].replace(0, np.nan).mean()
    if pd.isna(vol_mean) or vol_mean == 0:
        df["volume_normalized"] = 0.0
    else:
        df["volume_normalized"] = df["Volume"] / vol_mean

    # Delta proxy: sign * volume
    df["volume_delta"] = df["direction"].fillna(0.0) * df["Volume"]

    return df


# ---------------------------------------------------------------------------
# Synthetic depth ladder (Version B)
# ---------------------------------------------------------------------------

def build_price_grid(df: pd.DataFrame, n_levels: int) -> np.ndarray:
    """
    Builds a global price grid spanning the whole dataset, used for the depth heatmap.
    """
    price_min = df["Low"].min()
    price_max = df["High"].max()
    if not np.isfinite(price_min) or not np.isfinite(price_max):
        raise ValueError("Invalid price range for grid construction.")

    # small padding
    pad = (price_max - price_min) * 0.05 if price_max > price_min else max(price_min * 0.01, 0.1)
    grid_min = price_min - pad
    grid_max = price_max + pad

    if grid_max <= grid_min:
        grid_max = grid_min + max(price_min * 0.01, 0.1)

    grid = np.linspace(grid_min, grid_max, n_levels)
    return grid


def build_candle_depth_matrix(df: pd.DataFrame, price_grid: np.ndarray) -> np.ndarray:
    """
    For each bar, distribute its volume across the price grid using a Gaussian-like kernel
    centered at the microprice, clipped to the bar's high/low. This approximates where
    volume likely traded and acts as a synthetic liquidity map.

    Returns:
        depth_matrix: shape (n_bars, n_price_levels)
    """
    n_bars = df.shape[0]
    n_levels = price_grid.shape[0]
    depth = np.zeros((n_bars, n_levels), dtype=float)

    eps = 1e-9

    for i, (_, row) in enumerate(df.iterrows()):
        high = float(row["High"])
        low = float(row["Low"])
        open_ = float(row["Open"])
        close = float(row["Close"])
        vol = float(row["Volume"])
        micro = float(row.get("microprice", (open_ + close) * 0.5))
        tr = max(float(row.get("true_range", high - low)), eps)

        if vol <= 0 or not np.isfinite(vol) or high <= low:
            continue

        # Base Gaussian around microprice
        sigma = tr / 3.0
        if sigma <= 0:
            sigma = max(0.01, abs(close) * 0.001)

        dist = (price_grid - micro) / sigma
        kernel = np.exp(-0.5 * dist * dist)

        # Mask outside bar range (less likely traded there)
        mask_outside = (price_grid < low) | (price_grid > high)
        kernel[mask_outside] *= 0.1

        # Directional skew: if bar is up, slightly overweight prices above microprice, etc.
        direction = math.copysign(1.0, close - open_) if close != open_ else 0.0
        if direction != 0:
            skew = 1.0 + 0.3 * direction * np.sign(price_grid - micro)
            kernel *= np.clip(skew, 0.2, 1.8)

        kernel = np.clip(kernel, 0.0, None)
        s = kernel.sum()
        if s <= 0:
            continue

        depth[i, :] = vol * kernel / s

    return depth


# ---------------------------------------------------------------------------
# Pseudo-tick reconstruction (Version C)
# ---------------------------------------------------------------------------

def simulate_ticks_for_bar(ts: pd.Timestamp,
                           row: pd.Series,
                           n_ticks: int,
                           rng: np.random.RandomState) -> pd.DataFrame:
    """
    Simulate pseudo-ticks inside a single OHLCV bar.

    Produces:
        timestamp, price, volume, direction
    """
    high = float(row["High"])
    low = float(row["Low"])
    open_ = float(row["Open"])
    close = float(row["Close"])
    vol = float(row["Volume"])

    if not np.isfinite(vol) or vol <= 0 or high <= low:
        return pd.DataFrame(columns=["timestamp", "price", "volume", "direction"])

    # Time grid: 0..(n_ticks-1) seconds within the bar
    offsets = np.arange(n_ticks, dtype=float)
    t_index = ts + pd.to_timedelta(offsets, unit="s")

    # Base deterministic path open -> close
    base_line = np.linspace(open_, close, n_ticks)

    tr = max(high - low, 1e-6)
    # Random noise around the line
    noise = rng.normal(loc=0.0, scale=tr * 0.1, size=n_ticks)
    prices = base_line + noise

    # Ensure prices stay within [low, high], and force at least one touch of low & high
    prices = np.clip(prices, low, high)
    # Force extremes
    idx_low = rng.randint(0, n_ticks)
    idx_high = rng.randint(0, n_ticks)
    prices[idx_low] = low
    prices[idx_high] = high

    # Volume per tick (log-normal perturbation)
    mean_vol = vol / float(n_ticks)
    vol_noise = rng.lognormal(mean=0.0, sigma=0.7, size=n_ticks)
    tick_volumes = mean_vol * vol_noise
    # Renormalize to total volume
    tick_volumes *= (vol / tick_volumes.sum())

    # Direction per tick from price changes
    price_diff = np.diff(prices, prepend=prices[0])
    direction = np.sign(price_diff)
    direction[0] = np.sign(close - open_) if close != open_ else 0.0

    df_ticks = pd.DataFrame(
        {
            "timestamp": t_index,
            "price": prices,
            "volume": tick_volumes,
            "direction": direction,
        }
    )

    return df_ticks


def simulate_ticks(df: pd.DataFrame,
                   n_ticks_per_bar: int,
                   seed: int = 1337) -> pd.DataFrame:
    """
    Simulate pseudo-ticks for the entire OHLCV history.
    """
    rng = np.random.RandomState(seed)
    all_ticks = []

    for ts, row in df.iterrows():
        ticks = simulate_ticks_for_bar(ts, row, n_ticks_per_bar, rng)
        if not ticks.empty:
            all_ticks.append(ticks)

    if not all_ticks:
        return pd.DataFrame(columns=["timestamp", "price", "volume", "direction"])

    ticks_df = pd.concat(all_ticks, ignore_index=True)
    ticks_df.sort_values("timestamp", inplace=True)
    ticks_df.reset_index(drop=True, inplace=True)
    return ticks_df


def build_tick_depth_matrix(ticks_df: pd.DataFrame,
                            price_grid: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Construct a pseudo-tick-based depth matrix.

    Returns:
        tick_times: np.ndarray of timestamps
        tick_depth: 2D array shape (n_ticks, n_price_levels)
    """
    if ticks_df.empty:
        return np.array([]), np.zeros((0, price_grid.shape[0]), dtype=float)

    n_levels = price_grid.shape[0]
    n_ticks = ticks_df.shape[0]

    tick_times = ticks_df["timestamp"].values.astype("datetime64[ns]")
    depth = np.zeros((n_ticks, n_levels), dtype=float)

    # Precompute interval edges for price grid
    # We'll use nearest index via argmin on absolute diff for each tick
    for i, (_, row) in enumerate(ticks_df.iterrows()):
        price = float(row["price"])
        vol = float(row["volume"])
        if not np.isfinite(price) or not np.isfinite(vol) or vol <= 0:
            continue

        # Find nearest price level
        idx = int(np.argmin(np.abs(price_grid - price)))
        depth[i, idx] += vol

    return tick_times, depth


# ---------------------------------------------------------------------------
# Plotly visualizations
# ---------------------------------------------------------------------------

def make_candle_heatmap_figure(df: pd.DataFrame,
                               price_grid: np.ndarray,
                               depth_matrix: np.ndarray) -> go.Figure:
    """
    Bookmap-style heatmap at bar resolution.
    """
    fig = go.Figure()

    x_vals = df.index
    y_vals = price_grid
    z_vals = depth_matrix.T  # shape: (price_levels, time)

    heat = go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale="Turbo",
        colorbar=dict(title="Synthetic Depth"),
        zsmooth="best",
    )
    fig.add_trace(heat)

    # Overlay closing price - use White for contrast on Dark background
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(width=1.5, color="white"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Synthetic L2 Heatmap (Candle-Level)",
        xaxis_title="Time",
        yaxis_title="Price",
        height=800,
    )
    return fig


def make_tick_heatmap_figure(tick_times: np.ndarray,
                             price_grid: np.ndarray,
                             tick_depth: np.ndarray,
                             ticks_df: pd.DataFrame) -> go.Figure:
    """
    Bookmap-style heatmap at pseudo-tick resolution, with bubble prints.
    """
    fig = go.Figure()

    if tick_depth.shape[0] == 0:
        fig.update_layout(
            template="plotly_dark",
            title="Synthetic L2 Heatmap (Pseudo-Ticks) - No Data",
            xaxis_title="Time",
            yaxis_title="Price",
        )
        return fig

    x_vals = tick_times
    y_vals = price_grid
    z_vals = tick_depth.T

    heat = go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale="Turbo",
        colorbar=dict(title="Synthetic Depth"),
        zsmooth="best",
    )
    fig.add_trace(heat)

    # Bubble prints (trades)
    size_scale = 20.0
    vol = ticks_df["volume"].values
    vol_norm = vol / (np.percentile(vol, 95) + 1e-9)
    marker_sizes = np.clip(vol_norm * size_scale, 2.0, 40.0)

    # Color by direction (buy/sell/neutral) - Distinct Green/Red
    colors = np.where(ticks_df["direction"].values >= 0, "rgba(0, 255, 127, 0.8)", "rgba(255, 69, 0, 0.8)")

    fig.add_trace(
        go.Scatter(
            x=ticks_df["timestamp"],
            y=ticks_df["price"],
            mode="markers",
            name="Synthetic Trades",
            marker=dict(size=marker_sizes, color=colors, line=dict(width=0)),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Synthetic L2 Heatmap (Pseudo-Ticks + Bubble Prints)",
        xaxis_title="Time",
        yaxis_title="Price",
        height=800,
    )
    return fig


def make_microprice_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(width=1.5, color="white"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["microprice"],
            mode="lines",
            name="Microprice",
            line=dict(width=1.5, dash="dot", color="cyan"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume_delta"],
            name="Volume Delta",
            opacity=0.4,
            yaxis="y2",
            marker_color="yellow"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Price vs Microprice & Volume Delta",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Price"),
        yaxis2=dict(
            title="Volume Delta",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        barmode="overlay",
        height=600,
    )
    return fig


def make_hidden_liquidity_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["hidden_liquidity_ratio"],
            mode="lines",
            name="Hidden Liquidity Ratio",
            line=dict(color="magenta")
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Hidden Liquidity Ratio Over Time",
        xaxis_title="Time",
        yaxis_title="Hidden Liquidity (Range / Body)",
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# Tabbed HTML assembly
# ---------------------------------------------------------------------------

def build_tabbed_html(figures, titles, output_path: str):
    """
    Given a list of Plotly figures, render them into a single HTML file with simple
    JS-powered tabs.
    """
    if len(figures) != len(titles):
        raise ValueError("figures and titles must have same length")

    divs = []
    for i, (fig, title) in enumerate(zip(figures, titles)):
        full_html = (i == 0)
        div_id = f"fig{i}"
        html_fragment = pio.to_html(
            fig,
            include_plotlyjs=full_html,
            full_html=False,
            default_width="100%",
            default_height="100%",
            div_id=div_id,
        )
        # Ensure initial hidden state (we'll show tab 0 via JS)
        divs.append(html_fragment)

    # Build the tabs header
    buttons_html = []
    for i, title in enumerate(titles):
        buttons_html.append(
            f'<button class="tab-btn" onclick="showTab({i})">{title}</button>'
        )

    # NOTE: CSS braces are double-escaped {{ }} so Python format() doesn't break
    tabs_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Synthetic L2 Bookmap Heatmap</title>
<style>
body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background-color: #111; color: #eee; }}
#tab-bar {{ padding: 10px; background-color: #000; border-bottom: 1px solid #333; }}
.tab-btn {{
    margin-right: 4px;
    padding: 8px 16px;
    border: 1px solid #333;
    background-color: #222;
    color: #aaa;
    cursor: pointer;
    font-size: 14px;
    border-radius: 4px;
    transition: all 0.2s;
}}
.tab-btn:hover {{ background-color: #333; color: #fff; }}
.tab-btn.active {{ background-color: #007acc; color: #fff; border-color: #007acc; }}
.plot-container {{ width: 100%; height: 90vh; }}
</style>
</head>
<body>
<div id="tab-bar">
{buttons}
</div>
<div id="plots">
{plots}
</div>
<script>
function showTab(idx) {{
    var figs = document.querySelectorAll('div[id^="fig"]');
    for (var i = 0; i < figs.length; i++) {{
        figs[i].style.display = (i === idx) ? 'block' : 'none';
    }}
    var btns = document.getElementsByClassName('tab-btn');
    for (var j = 0; j < btns.length; j++) {{
        if (j === idx) btns[j].classList.add('active');
        else btns[j].classList.remove('active');
    }}
}}
window.onload = function() {{
    showTab(0);
}};
</script>
</body>
</html>
""".format(
        buttons="\n".join(buttons_html),
        plots="\n".join(divs),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tabs_html)


# ---------------------------------------------------------------------------
# CSV Writers
# ---------------------------------------------------------------------------

def save_csv_outputs(output_dir: str,
                     ticker: str,
                     df_enriched: pd.DataFrame,
                     price_grid: np.ndarray,
                     depth_matrix: np.ndarray,
                     ticks_df: pd.DataFrame,
                     tick_depth: np.ndarray):
    """
    Save all relevant artifacts as CSV for STEGO pipelines.
    """
    # Enriched OHLCV
    enriched_path = os.path.join(output_dir, f"{ticker}_enriched_ohlcv.csv")
    df_enriched.to_csv(enriched_path, index=True)

    # Candle-level depth (wide format)
    depth_cols = [f"p_{i}_{price_grid[i]:.6f}" for i in range(price_grid.shape[0])]
    depth_df = pd.DataFrame(depth_matrix, index=df_enriched.index, columns=depth_cols)
    depth_path = os.path.join(output_dir, f"{ticker}_synthetic_candle_depth_wide.csv")
    depth_df.to_csv(depth_path, index=True)

    # Long format depth (optional)
    long_records = []
    for ti, ts in enumerate(df_enriched.index):
        row = depth_matrix[ti]
        for pi, price in enumerate(price_grid):
            vol = row[pi]
            if vol <= 0:
                continue
            long_records.append((ts, float(price), float(vol)))

    long_df = pd.DataFrame(long_records, columns=["timestamp", "price", "synthetic_depth"])
    long_path = os.path.join(output_dir, f"{ticker}_synthetic_candle_depth_long.csv")
    long_df.to_csv(long_path, index=False)

    # Tick-level data + depth
    ticks_path = os.path.join(output_dir, f"{ticker}_synthetic_ticks.csv")
    ticks_df.to_csv(ticks_path, index=False)

    if tick_depth.size > 0:
        tick_depth_cols = [f"p_{i}_{price_grid[i]:.6f}" for i in range(price_grid.shape[0])]
        tick_depth_df = pd.DataFrame(tick_depth, columns=tick_depth_cols)
        tick_depth_df["timestamp"] = ticks_df["timestamp"].values
        tick_depth_path = os.path.join(output_dir, f"{ticker}_synthetic_tick_depth_wide.csv")
        tick_depth_df.to_csv(tick_depth_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Synthetic L2 Bookmap-Style Heatmap from OHLCV (STEG0 / Michael Derby)"
    )
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., SPY, IWM, NVDA)")
    parser.add_argument("--period", type=str, default="6mo", help="Period for data_retrieval.load_or_download_ticker (default: 6mo)")
    parser.add_argument("--n-price-levels", type=int, default=80, help="Number of price levels in synthetic depth ladder (default: 80)")
    parser.add_argument("--n-ticks-per-bar", type=int, default=24, help="Number of pseudo-ticks per bar for Version C (default: 24)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for pseudo-tick simulation")
    parser.add_argument("--no-open-browser", action="store_true", help="Do not auto-open the HTML in a browser")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ticker = args.ticker.upper()

    output_dir = get_output_dir(ticker)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Loading OHLCV for {ticker} via data_retrieval.load_or_download_ticker(period='{args.period}')...")

    # Load via user's data_retrieval (unchanged)
    # This handles the requirement of downloading to local disk and reading from CSV
    df = dr.load_or_download_ticker(ticker, period=args.period)
    if df is None or df.empty:
        raise SystemExit(f"[ERROR] No data returned for ticker {ticker} and period {args.period}.")

    # Ensure DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        else:
            df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)

    # Enrich with microstructure metrics (Version B)
    print("[INFO] Enriching OHLCV with microstructure metrics...")
    df_enriched = enrich_microstructure(df)

    # Build global price grid
    print(f"[INFO] Building price grid with {args.n_price_levels} levels...")
    price_grid = build_price_grid(df_enriched, args.n_price_levels)

    # Build candle-level synthetic depth matrix
    print("[INFO] Building candle-level synthetic depth matrix...")
    depth_matrix = build_candle_depth_matrix(df_enriched, price_grid)

    # Pseudo-tick reconstruction (Version C)
    print(f"[INFO] Simulating pseudo-ticks with {args.n_ticks_per_bar} ticks per bar (seed={args.seed})...")
    ticks_df = simulate_ticks(df_enriched, args.n_ticks_per_bar, seed=args.seed)
    print(f"[INFO] Generated {len(ticks_df)} synthetic ticks.")

    print("[INFO] Building pseudo-tick depth matrix...")
    tick_times, tick_depth = build_tick_depth_matrix(ticks_df, price_grid)

    # Save CSV outputs for STEGO pipelines
    print("[INFO] Saving CSV outputs...")
    save_csv_outputs(output_dir, ticker, df_enriched, price_grid, depth_matrix, ticks_df, tick_depth)

    # Build Plotly figures with Dark Theme
    print("[INFO] Building Plotly figures...")
    fig_candle = make_candle_heatmap_figure(df_enriched, price_grid, depth_matrix)
    fig_ticks = make_tick_heatmap_figure(tick_times, price_grid, tick_depth, ticks_df)
    fig_micro = make_microprice_figure(df_enriched)
    fig_hidden = make_hidden_liquidity_figure(df_enriched)

    figures = [fig_candle, fig_ticks, fig_micro, fig_hidden]
    titles = [
        "Candle Heatmap",
        "Pseudo-Tick Heatmap",
        "Price vs Microprice & Delta",
        "Hidden Liquidity Ratio",
    ]

    html_path = os.path.join(output_dir, f"{ticker}_synthetic_l2_bookmap.html")
    print(f"[INFO] Writing tabbed Plotly HTML to {html_path} ...")
    build_tabbed_html(figures, titles, html_path)

    if not args.no_open_browser:
        try:
            print(f"[INFO] Opening main dashboard in default web browser: {html_path}")
            webbrowser.open(f"file://{html_path}")
        except Exception as exc:
            print(f"[WARN] Could not auto-open browser: {exc}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
