#!/usr/bin/env python3
SCRIPTNAME = "lrc_alignment_multiwindow.v1.py"
"""
Multi-window Linear Regression Channel (LRC) alignment detector.
Uses data_retrieval.py as the ONLY data source.
Computes rolling LRCs on 252/126/63 day windows.
Extracts slope + smoothed slope for each window.
Detects slope inflections.
Flags "concordance" when ≥2 windows inflect together (within tolerance).

Plots:
- Candlesticks
- 3x LRC midlines + channels
- Shaded regions for bullish / bearish concordance

Outputs:
- Plotly HTML (auto-opens in browser)
- Optional PNG (if kaleido installed)
- Console table of last N signals
"""

import argparse
import logging
import os
import sys
from datetime import datetime, date
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import webbrowser

try:
    import plotly.graph_objects as go
except ImportError as e:
    print("ERROR: plotly is required for this script. Install with:\n pip install plotly", file=sys.stderr)
    raise

# --- Try to import your canonical data layer (DO NOT MODIFY THIS FILE) ---
try:
    import data_retrieval  # your existing module
except ImportError as e:
    print("ERROR: Could not import data_retrieval.py; ensure it is in PYTHONPATH or same directory.", file=sys.stderr)
    raise


# ======================================================================
# Logging
# ======================================================================
def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ======================================================================
# Data access via your data_retrieval.py
# ======================================================================
def get_price_dataframe(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    ratio: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches OHLCV dataframe using your canonical data_retrieval.py.
    If ratio is provided, uses get_ratio_dataframe(ticker, ratio, ...).
    Otherwise uses load_or_download_ticker(ticker, ...).
    """
    if start is None:
        start = "2010-01-01"
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    # Try to use ratio logic if requested
    if ratio is not None:
        if not hasattr(data_retrieval, "get_ratio_dataframe"):
            raise RuntimeError("ratio requested but data_retrieval.get_ratio_dataframe is not available.")
        logging.info("Loading RATIO data: %s / %s from %s to %s", ticker, ratio, start, end)
        
        # Check if get_ratio_dataframe accepts interval, otherwise omit it
        # Assuming similar API to load_or_download_ticker based on error
        try:
             df = data_retrieval.get_ratio_dataframe(ticker, ratio, start=start, end=end, interval=interval)
        except TypeError:
             logging.warning("API mismatch: get_ratio_dataframe does not accept 'interval'. Retrying without it.")
             df = data_retrieval.get_ratio_dataframe(ticker, ratio, start=start, end=end)

    else:
        if not hasattr(data_retrieval, "load_or_download_ticker"):
            raise RuntimeError("data_retrieval.load_or_download_ticker not found.")
        logging.info("Loading TICKER data: %s from %s to %s", ticker, start, end)
        
        # FIX: Removed interval=interval because the local library does not support it
        df = data_retrieval.load_or_download_ticker(ticker, start=start, end=end)

    # Standardize index & columns
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            raise RuntimeError("Dataframe has no DatetimeIndex or 'Date' column.")

    df = df.sort_index()

    # Require OHLC + volume; ratio may only have 'Close'
    if "Close" not in df.columns:
        raise RuntimeError("Dataframe missing 'Close' column.")

    # If open/high/low missing, synthesize from close
    for col in ("Open", "High", "Low"):
        if col not in df.columns:
            df[col] = df["Close"]

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    return df


# ======================================================================
# LRC computation
# ======================================================================
def compute_rolling_lrc(
    prices: pd.Series,
    window: int,
    k: float = 2.0,
) -> Dict[str, pd.Series]:
    """
    Compute rolling Linear Regression Channel for a given window.

    Returns dict with:
      - 'mid': midline at the last x of each window
      - 'slope': regression slope per-bar
      - 'sigma': residual std
      - 'upper': mid + k*sigma
      - 'lower': mid - k*sigma
    """
    x = np.arange(window, dtype=float)

    def _reg_mid(vals: np.ndarray) -> float:
        m, b = np.polyfit(x, vals, 1)
        return m * x[-1] + b

    def _reg_slope(vals: np.ndarray) -> float:
        m, b = np.polyfit(x, vals, 1)
        return m

    def _reg_sigma(vals: np.ndarray) -> float:
        m, b = np.polyfit(x, vals, 1)
        y_hat = m * x + b
        resid = vals - y_hat
        return float(np.std(resid))

    logging.info("Computing LRC for window=%d", window)

    mid = prices.rolling(window).apply(_reg_mid, raw=True)
    slope = prices.rolling(window).apply(_reg_slope, raw=True)
    sigma = prices.rolling(window).apply(_reg_sigma, raw=True)

    upper = mid + k * sigma
    lower = mid - k * sigma

    return {
        "mid": mid,
        "slope": slope,
        "sigma": sigma,
        "upper": upper,
        "lower": lower,
    }


# ======================================================================
# Inflection + concordance
# ======================================================================
def smooth_slope(slope: pd.Series, span: int = 5) -> pd.Series:
    return slope.ewm(span=span, adjust=False).mean()


def compute_eps_nearzero(slope: pd.Series, factor: float = 0.1) -> float:
    med = slope.abs().median()
    if np.isnan(med) or med == 0:
        med = slope.abs().mean()
    if np.isnan(med) or med == 0:
        med = 1e-6
    return float(factor * med)


def detect_inflections(
    slope_smooth: pd.Series,
    eps: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect up/down inflections based on first difference of smoothed slope.
    Returns: inflect_up, inflect_down (bool Series)
    """
    diff = slope_smooth.diff()
    # Ignore tiny wiggles
    diff_pos = diff > eps
    diff_neg = diff < -eps
    # Transition of diff from <=0 to >0  => up-inflection
    inflect_up = diff_pos & (~diff_pos.shift(1).fillna(False))
    # Transition from >=0 to <0 => down-inflection
    inflect_down = diff_neg & (~diff_neg.shift(1).fillna(False))

    return inflect_up, inflect_down


def build_concordance_signals(
    df: pd.DataFrame,
    windows: List[int],
    tol_bars: int = 5,
    k_channel: Dict[int, float] = None,
) -> pd.DataFrame:
    """
    Given DataFrame with:
    slope_smooth_{w}, inflect_up_{w}, inflect_down_{w},
    LRC columns for 63-day (mid_63, sigma_63),
    compute bullish/bearish concordance masks.
    """
    if k_channel is None:
        k_channel = {252: 1.8, 126: 1.9, 63: 2.1}

    # Recent-inflection logic via rolling max over tol_bars
    recent_up_cols = []
    recent_down_cols = []

    for w in windows:
        up_col = f"inflect_up_{w}"
        down_col = f"inflect_down_{w}"
        recent_up_col = f"recent_up_{w}"
        recent_down_col = f"recent_down_{w}"

        df[recent_up_col] = df[up_col].rolling(tol_bars).max().astype(bool)
        df[recent_down_col] = df[down_col].rolling(tol_bars).max().astype(bool)

        recent_up_cols.append(recent_up_col)
        recent_down_cols.append(recent_down_col)

    df["recent_up_count"] = df[recent_up_cols].sum(axis=1)
    df["recent_down_count"] = df[recent_down_cols].sum(axis=1)

    up_concord = (df["recent_up_count"] >= 2) & (df["recent_down_count"] == 0)
    down_concord = (df["recent_down_count"] >= 2) & (df["recent_up_count"] == 0)

    # Apply 63-day channel proximity filter (avoid chasing extremes)
    mid_63 = df["mid_63"]
    sigma_63 = df["sigma_63"]
    k63 = k_channel.get(63, 2.0)

    close = df["Close"]

    bull_zone = (close >= mid_63) & (close <= mid_63 + k63 * sigma_63)
    bear_zone = (close <= mid_63) & (close >= mid_63 - k63 * sigma_63)

    df["bull_concord"] = up_concord & bull_zone
    df["bear_concord"] = down_concord & bear_zone

    return df


# ======================================================================
# Plotting
# ======================================================================
def make_lrc_alignment_figure(
    df: pd.DataFrame,
    ticker: str,
    ratio: Optional[str],
    windows: List[int],
) -> go.Figure:
    """
    Build Plotly figure with:
    - Candles
    - 3x LRC midlines + channels
    - Shaded vertical regions for bull/bear concordance
    """
    title = f"{ticker}"
    if ratio:
        title = f"{ticker}/{ratio}"
    title += " – Multi-Window LRC Alignment"

    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color=None,
            decreasing_line_color=None,
            showlegend=True,
        )
    )

    # LRCs
    for w in windows:
        mid_col = f"mid_{w}"
        up_col = f"upper_{w}"
        low_col = f"lower_{w}"

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[mid_col],
                mode="lines",
                name=f"LRC mid {w}",
                line=dict(width=1.3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[up_col],
                mode="lines",
                name=f"LRC upper {w}",
                line=dict(width=0.8, dash="dot"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[low_col],
                mode="lines",
                name=f"LRC lower {w}",
                line=dict(width=0.8, dash="dot"),
                showlegend=False,
            )
        )

    # Shaded concordance bands
    shapes = []

    def _add_regions(mask: pd.Series, color: str) -> None:
        in_region = False
        start = None
        last_idx = None
        # Note: using items() for Pandas 2.0+ compatibility (iterkv/iteritems deprecated)
        for idx, val in mask.items():
            if val and not in_region:
                in_region = True
                start = idx
            if in_region and (not val):
                end = last_idx if last_idx is not None else idx
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=start,
                        x1=end,
                        y0=0,
                        y1=1,
                        fillcolor=color,
                        opacity=0.15,
                        line=dict(width=0),
                        layer="below",
                    )
                )
                in_region = False
                start = None
            last_idx = idx
        # Handle region extending to end
        if in_region and start is not None and last_idx is not None:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start,
                    x1=last_idx,
                    y0=0,
                    y1=1,
                    fillcolor=color,
                    opacity=0.15,
                    line=dict(width=0),
                    layer="below",
                )
            )

    _add_regions(df["bull_concord"].fillna(False), "rgba(0,200,0,0.3)")
    _add_regions(df["bear_concord"].fillna(False), "rgba(200,0,0,0.3)")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        shapes=shapes,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ======================================================================
# Core analysis pipeline
# ======================================================================
def run_lrc_alignment(
    ticker: str,
    ratio: Optional[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    outdir: Optional[str],
    tol_bars: int = 5,
    smooth_span: int = 5,
    persist_signals_display: int = 20,
) -> None:
    windows = [252, 126, 63]
    k_channel = {252: 1.8, 126: 1.9, 63: 2.1}

    df = get_price_dataframe(ticker, start=start, end=end, interval=interval, ratio=ratio)
    close = df["Close"]

    # Compute LRCs for each window
    for w in windows:
        res = compute_rolling_lrc(close, window=w, k=k_channel.get(w, 2.0))
        df[f"mid_{w}"] = res["mid"]
        df[f"slope_{w}"] = res["slope"]
        df[f"sigma_{w}"] = res["sigma"]
        df[f"upper_{w}"] = res["upper"]
        df[f"lower_{w}"] = res["lower"]

    # Smooth slopes & inflections
    for w in windows:
        slope_col = f"slope_{w}"
        smooth_col = f"slope_smooth_{w}"
        df[smooth_col] = smooth_slope(df[slope_col], span=smooth_span)
        eps = compute_eps_nearzero(df[smooth_col], factor=0.1)
        up_col = f"inflect_up_{w}"
        down_col = f"inflect_down_{w}"
        df[up_col], df[down_col] = detect_inflections(df[smooth_col], eps=eps)

    # Concordance
    df = build_concordance_signals(df, windows=windows, tol_bars=tol_bars, k_channel=k_channel)

    # Drop rows before all windows are valid
    max_window = max(windows)
    df = df.iloc[max_window:]

    # Build plot
    fig = make_lrc_alignment_figure(df, ticker=ticker, ratio=ratio, windows=windows)

    # Output directory
    if outdir is None:
        outdir = os.path.join(os.getcwd(), "LRC_ALIGNMENT_OUTPUTS")

    os.makedirs(outdir, exist_ok=True)

    ratio_suffix = f"_{ratio}" if ratio else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{ticker}{ratio_suffix}_lrc_alignment_{interval}_{timestamp}"

    html_path = os.path.join(outdir, base_name + ".html")
    png_path = os.path.join(outdir, base_name + ".png")

    logging.info("Writing Plotly HTML to: %s", html_path)
    fig.write_html(html_path, auto_open=False)

    # Try PNG export if kaleido available
    try:
        fig.write_image(png_path, width=1600, height=900)
        logging.info("PNG snapshot saved to: %s", png_path)
    except Exception as e:
        logging.warning("Could not save PNG (kaleido missing?): %s", e)

    # Auto-open HTML in default browser
    try:
        webbrowser.open("file://" + html_path)
    except Exception as e:
        logging.warning("Could not auto-open browser: %s", e)

    # Summarize signals at the end
    sig_df = df[(df["bull_concord"]) | (df["bear_concord"])].copy()
    sig_df["signal"] = np.where(sig_df["bull_concord"], "BULL", "BEAR")

    if not sig_df.empty:
        logging.info("Last %d concordance signals:", persist_signals_display)
        print("=" * 80)
        print(f"{'Date':<12} {'Signal':<6} {'Close':>10}  recent_up_count recent_down_count")
        print("=" * 80)
        tail = sig_df.tail(persist_signals_display)
        for idx, row in tail.iterrows():
            dt_str = idx.strftime("%Y-%m-%d")
            print(
                f"{dt_str:<12} {row['signal']:<6} {row['Close']:>10.2f}  "
                f"{int(row['recent_up_count']):>5}            {int(row['recent_down_count']):>5}"
            )
        print("=" * 80)
    else:
        logging.info("No concordance signals in sample.")


# ======================================================================
# CLI
# ======================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-window LRC alignment + concordance detector (252/126/63)."
    )

    p.add_argument("--ticker", required=True, help="Primary ticker (e.g., SPY, NVDA).")
    p.add_argument("--ratio", default=None, help="Optional ratio ticker (e.g., QQQ to analyze TICKER/QQQ).")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD). Default: 2010-01-01.")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Default: today.")
    p.add_argument("--interval", default="1d", help="Data interval (default: 1d).")
    p.add_argument("--outdir", default=None, help="Output directory for HTML/PNG. Default: ./LRC_ALIGNMENT_OUTPUTS")
    p.add_argument("--tol_bars", type=int, default=5, help="Concordance tolerance window in bars (default: 5).")
    p.add_argument("--smooth_span", type=int, default=5, help="EMA span for slope smoothing (default: 5).")
    p.add_argument(
        "--signals_to_show",
        type=int,
        default=20,
        help="Number of most recent concordance signals to print (default: 20).",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        run_lrc_alignment(
            ticker=args.ticker,
            ratio=args.ratio,
            start=args.start,
            end=args.end,
            interval=args.interval,
            outdir=args.outdir,
            tol_bars=args.tol_bars,
            smooth_span=args.smooth_span,
            persist_signals_display=args.signals_to_show,
        )
    except Exception as e:
        logging.exception("Fatal error in LRC alignment pipeline: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
