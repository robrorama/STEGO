#!/usr/bin/env python3
# SCRIPTNAME: rolling_corr_jitter_dashboard.v1.py
# AUTHOR: Michael Derby
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# DATE: November 24, 2025
#
# PURPOSE
# -------
# Standalone rolling-correlation "jitter" dashboard using your canonical data_retrieval.py.
#
# FEATURES
# --------
# - Uses data_retrieval.load_or_download_ticker exclusively for price data.
# - Writes all outputs into /dev/shm via BASE_CACHE_PATH() (default: /dev/shm/cache).
# - Base ticker (default: SPY) vs:
#     * All primary SPDR sector ETFs (S and P 500 sectors).
#     * A broad basket of thematic / style / factor / credit / housing / clean energy proxies.
# - Computes, for each window:
#     * Rolling correlation of each proxy vs base.
#     * "Jitter" = rolling standard deviation of daily changes in that rolling correlation.
# - Outputs:
#     * CSVs of rolling correlations and jitter per window.
#     * Summary CSV of latest correlation and jitter across windows.
#     * Plotly HTML dashboards:
#         - Rolling correlation time-series (per window).
#         - Rolling jitter time-series (per window).
#         - Latest correlation matrix heatmap (tickers x windows).
#         - Latest jitter matrix heatmap (tickers x windows).
# - Automatically opens all HTML outputs in the default web browser as separate tabs.
#
# USAGE (examples)
# ----------------
#   python3 rolling_corr_jitter_dashboard.v1.py
#   python3 rolling_corr_jitter_dashboard.v1.py --base SPY --period 5y --windows 20 60 120
#   python3 rolling_corr_jitter_dashboard.v1.py --base QQQ --period 3y --windows 10 30 90
#
# NOTES
# -----
# - This script assumes your data_retrieval.py exposes:
#       load_or_download_ticker(ticker, period="5y", ...),
#       BASE_CACHE_PATH().
#   It MUST NOT pass unsupported keywords like "interval" to that loader.

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import webbrowser

# ---------------------------------------------------------------------------
# Attempt to import your canonical data_retrieval module
# ---------------------------------------------------------------------------
try:
    import data_retrieval as dr
except ImportError as e:
    print(f"[FATAL] Could not import data_retrieval.py: {e}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG: default universe
# ---------------------------------------------------------------------------

# Primary SPDR S and P 500 sector ETFs
SECTOR_ETFS: List[str] = [
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLE",  # Energy
    "XLF",  # Financials
    "XLV",  # Health Care
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLRE", # Real Estate
    "XLU",  # Utilities
    "XLK",  # Technology
    "XLC",  # Communication Services
]

# A broad basket of thematic / style / factor / credit / housing / clean energy proxies
THEMATIC_ETFS: List[str] = [
    # Style / factor / size
    "IWM",   # Russell 2000 (small caps)
    "QQQ",   # Tech / growth heavy
    "DIA",   # Dow
    "MTUM",  # Momentum
    "VLUE",  # Value
    "QUAL",  # Quality
    # Tech / semis / innovation
    "SMH",   # Semiconductors
    "SOXX",  # Semiconductors
    "BOTZ",  # Robotics / AI
    "ARKK",  # Innovation / disruptive
    # Credit / rates
    "HYG",   # High yield credit
    "LQD",   # Investment grade credit
    "IEF",   # 7 to 10 year Treasuries
    "TLT",   # Long duration Treasuries
    "SHY",   # Short Treasuries
    # Financials / regional / housing
    "KRE",   # Regional banks
    "XHB",   # Homebuilders
    "ITB",   # Home construction
    # Biotech / healthcare
    "XBI",   # Biotech
    "IBB",   # Biotech
    # Commodities / resources
    "GLD",   # Gold
    "SLV",   # Silver
    "GDX",   # Gold miners
    "XME",   # Metals and mining
    # Clean energy / ESG
    "TAN",   # Solar
    "ICLN",  # Clean energy
    "PBW",   # Clean energy
    # Misc liquid macro proxies
    "EEM",   # EM equities
    "EFA",   # DM ex US
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_output_root(subdir: str = "ROLLING_CORR_JITTER") -> str:
    """
    Resolve output root directory under /dev/shm using your BASE_CACHE_PATH()
    from data_retrieval, then append our dashboard subdirectory.
    """
    try:
        base_cache = dr.BASE_CACHE_PATH()
    except Exception:
        # Fallback if BASE_CACHE_PATH is not available for any reason
        base_cache = "/dev/shm"

    out_root = os.path.join(base_cache, subdir)
    os.makedirs(out_root, exist_ok=True)
    return out_root


def load_price_series(
    ticker: str,
    period: str,
) -> pd.Series:
    """
    Use your data_retrieval.load_or_download_ticker to fetch a price series.
    Attempts to select 'Adj Close' or 'adjclose' or 'Close' as a fallback.
    Returns a pandas Series indexed by datetime.
    """
    logging.info(f"Loading data for {ticker} (period={period}) via data_retrieval...")
    try:
        df = dr.load_or_download_ticker(ticker, period=period)
    except TypeError:
        # In case signature is different, fall back to minimal args
        df = dr.load_or_download_ticker(ticker)
    except Exception as e:
        logging.error(f"Failed to load {ticker}: {e}")
        raise

    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {ticker}")

    # Try a few common column names
    for col in ["Adj Close", "adjclose", "Close", "close"]:
        if col in df.columns:
            s = df[col].copy()
            s.name = ticker
            return s

    # Last-ditch: if only one non-date column, use it
    non_dt_cols = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.datetime64)]
    if len(non_dt_cols) == 1:
        s = df[non_dt_cols[0]].copy()
        s.name = ticker
        return s

    raise ValueError(f"Could not identify price column for {ticker}. Columns: {df.columns.tolist()}")


def build_price_matrix(
    base: str,
    period: str,
    sectors: List[str],
    thematics: List[str],
) -> pd.DataFrame:
    """
    Build a joint price matrix (DataFrame) for the base + sector + thematic ETFs.
    Only tickers that successfully load and have sufficient data are included.
    """
    tickers = sorted(set([base] + sectors + thematics))
    series_list: List[pd.Series] = []
    included: List[str] = []

    for t in tickers:
        try:
            s = load_price_series(t, period)
            series_list.append(s)
            included.append(t)
        except Exception as e:
            logging.warning(f"Skipping {t} due to load error: {e}")

    if not series_list:
        raise RuntimeError("No tickers successfully loaded. Cannot proceed.")

    # Inner join on dates
    prices = pd.concat(series_list, axis=1, join="inner")
    prices = prices.sort_index()
    logging.info(f"Price matrix built with tickers: {list(prices.columns)} and {len(prices)} rows.")
    return prices


def compute_rolling_corr_and_jitter(
    prices: pd.DataFrame,
    base: str,
    windows: List[int],
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Compute rolling correlation and jitter vs base for each non-base ticker.

    Returns:
        corr_dict[window]  -> DataFrame (index=dates, columns=tickers_ex_base)
        jitter_dict[window]-> DataFrame (index=dates, columns=tickers_ex_base)
    """
    rets = prices.pct_change().dropna()
    if base not in rets.columns:
        raise ValueError(f"Base {base} not found in returns columns: {list(rets.columns)}")

    peers = [c for c in rets.columns if c != base]

    corr_dict: Dict[int, pd.DataFrame] = {}
    jitter_dict: Dict[int, pd.DataFrame] = {}

    for w in windows:
        logging.info(f"Computing rolling correlation and jitter for window={w}...")
        frames = []
        for t in peers:
            s = rets[t].rolling(w).corr(rets[base])
            s.name = t
            frames.append(s)

        corr_df = pd.concat(frames, axis=1)
        corr_df = corr_df.dropna(how="all")
        corr_dict[w] = corr_df

        # Jitter = rolling std of delta correlation, using same window length
        dc = corr_df.diff()
        jitter_df = dc.rolling(w).std()
        jitter_df = jitter_df.dropna(how="all")
        jitter_dict[w] = jitter_df

    return corr_dict, jitter_dict


def save_dataframes_to_csv(
    out_root: str,
    corr_dict: Dict[int, pd.DataFrame],
    jitter_dict: Dict[int, pd.DataFrame],
) -> None:
    """
    Write per-window correlation and jitter DataFrames to CSV files.
    """
    for w, df in corr_dict.items():
        path = os.path.join(out_root, f"rolling_corr_w{w}.csv")
        df.to_csv(path)
        logging.info(f"Wrote rolling correlation CSV for window={w} to {path}")

    for w, df in jitter_dict.items():
        path = os.path.join(out_root, f"rolling_jitter_w{w}.csv")
        df.to_csv(path)
        logging.info(f"Wrote rolling jitter CSV for window={w} to {path}")


def build_latest_summary(
    corr_dict: Dict[int, pd.DataFrame],
    jitter_dict: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a single DataFrame summarizing the latest correlation and jitter
    across all windows for each ticker.

    Columns will look like:
        ticker, corr_w20, corr_w60, jitter_w20, jitter_w60, ...
    """
    all_tickers = set()
    for df in corr_dict.values():
        all_tickers.update(df.columns)

    all_tickers = sorted(all_tickers)
    rows = []

    for t in all_tickers:
        row = {"ticker": t}
        for w, df in corr_dict.items():
            if t in df.columns and not df[t].dropna().empty:
                row[f"corr_w{w}"] = df[t].dropna().iloc[-1]
        for w, df in jitter_dict.items():
            if t in df.columns and not df[t].dropna().empty:
                row[f"jitter_w{w}"] = df[t].dropna().iloc[-1]
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("ticker").sort_index()
    return summary


def plot_timeseries_corr(
    out_root: str,
    corr_dict: Dict[int, pd.DataFrame],
) -> List[str]:
    """
    For each window, create a Plotly line chart of rolling correlations
    vs the base for all tickers. Returns list of HTML paths.
    """
    html_paths: List[str] = []

    for w, df in corr_dict.items():
        if df.empty:
            logging.warning(f"No correlation data for window={w}, skipping timeseries plot.")
            continue

        long_df = df.reset_index().melt(id_vars=df.index.name or "index", var_name="ticker", value_name="corr")
        time_col = df.index.name or "index"

        fig = px.line(
            long_df,
            x=time_col,
            y="corr",
            color="ticker",
            title=f"Rolling Correlation vs Base (window={w})",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Correlation",
            hovermode="x unified",
        )
        html_path = os.path.join(out_root, f"rolling_corr_timeseries_w{w}.html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        html_paths.append(html_path)
        logging.info(f"Wrote rolling correlation timeseries HTML for window={w} to {html_path}")

    return html_paths


def plot_timeseries_jitter(
    out_root: str,
    jitter_dict: Dict[int, pd.DataFrame],
) -> List[str]:
    """
    For each window, create a Plotly line chart of rolling jitter
    vs the base for all tickers. Returns list of HTML paths.
    """
    html_paths: List[str] = []

    for w, df in jitter_dict.items():
        if df.empty:
            logging.warning(f"No jitter data for window={w}, skipping timeseries plot.")
            continue

        long_df = df.reset_index().melt(id_vars=df.index.name or "index", var_name="ticker", value_name="jitter")
        time_col = df.index.name or "index"

        fig = px.line(
            long_df,
            x=time_col,
            y="jitter",
            color="ticker",
            title=f"Rolling Correlation Jitter vs Base (window={w})",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Jitter (Std of ΔCorr)",
            hovermode="x unified",
        )
        html_path = os.path.join(out_root, f"rolling_jitter_timeseries_w{w}.html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        html_paths.append(html_path)
        logging.info(f"Wrote rolling jitter timeseries HTML for window={w} to {html_path}")

    return html_paths


def plot_latest_heatmaps(
    out_root: str,
    corr_dict: Dict[int, pd.DataFrame],
    jitter_dict: Dict[int, pd.DataFrame],
) -> List[str]:
    """
    Build latest correlation and jitter matrices (tickers x windows) and
    plot them as heatmaps.
    """
    html_paths: List[str] = []

    # Identify tickers and windows
    all_tickers = set()
    for df in corr_dict.values():
        all_tickers.update(df.columns)
    for df in jitter_dict.values():
        all_tickers.update(df.columns)
    all_tickers = sorted(all_tickers)

    windows_corr = sorted(corr_dict.keys())
    windows_jitter = sorted(jitter_dict.keys())

    # Correlation matrix
    if windows_corr:
        corr_matrix = pd.DataFrame(index=all_tickers, columns=windows_corr, dtype=float)
        for w in windows_corr:
            df = corr_dict[w]
            for t in all_tickers:
                if t in df.columns and not df[t].dropna().empty:
                    corr_matrix.loc[t, w] = df[t].dropna().iloc[-1]

        fig_corr = px.imshow(
            corr_matrix,
            x=windows_corr,
            y=all_tickers,
            aspect="auto",
            title="Latest Rolling Correlation vs Base (tickers x window)",
            labels={"x": "Window (days)", "y": "Ticker", "color": "Correlation"},
            origin="lower",
        )
        html_path_corr = os.path.join(out_root, "latest_corr_heatmap.html")
        fig_corr.write_html(html_path_corr, include_plotlyjs="cdn")
        html_paths.append(html_path_corr)
        logging.info(f"Wrote latest correlation heatmap HTML to {html_path_corr}")

    # Jitter matrix
    if windows_jitter:
        jitter_matrix = pd.DataFrame(index=all_tickers, columns=windows_jitter, dtype=float)
        for w in windows_jitter:
            df = jitter_dict[w]
            for t in all_tickers:
                if t in df.columns and not df[t].dropna().empty:
                    jitter_matrix.loc[t, w] = df[t].dropna().iloc[-1]

        fig_jitter = px.imshow(
            jitter_matrix,
            x=windows_jitter,
            y=all_tickers,
            aspect="auto",
            title="Latest Rolling Correlation Jitter vs Base (tickers x window)",
            labels={"x": "Window (days)", "y": "Ticker", "color": "Jitter (Std of ΔCorr)"},
            origin="lower",
        )
        html_path_jitter = os.path.join(out_root, "latest_jitter_heatmap.html")
        fig_jitter.write_html(html_path_jitter, include_plotlyjs="cdn")
        html_paths.append(html_path_jitter)
        logging.info(f"Wrote latest jitter heatmap HTML to {html_path_jitter}")

    return html_paths


def open_html_tabs(html_paths: List[str]) -> None:
    """
    Open each HTML file in the default web browser as a new tab.
    """
    for path in html_paths:
        abs_path = os.path.abspath(path)
        url = "file://" + abs_path
        try:
            webbrowser.open(url, new=2)
        except Exception as e:
            logging.warning(f"Could not open {url} in web browser: {e}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling correlation and jitter dashboard vs base (SPY by default), using data_retrieval.py and writing to /dev/shm.",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="SPY",
        help="Base ticker symbol (default: SPY).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Lookback period string for yfinance-style loader (default: 5y).",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[20, 60, 120],
        help="Rolling windows (in days) for correlation and jitter (default: 20 60 120).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="ROLLING_CORR_JITTER",
        help="Subdirectory under BASE_CACHE_PATH (default: ROLLING_CORR_JITTER).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("=== Rolling Correlation and Jitter Dashboard ===")
    logging.info(f"Base: {args.base}, Period: {args.period}, Windows: {args.windows}")

    out_root = get_output_root(args.output_subdir)
    logging.info(f"Output root: {out_root}")

    # Build price matrix
    prices = build_price_matrix(
        base=args.base,
        period=args.period,
        sectors=SECTOR_ETFS,
        thematics=THEMATIC_ETFS,
    )

    # Compute rolling correlation and jitter
    corr_dict, jitter_dict = compute_rolling_corr_and_jitter(
        prices=prices,
        base=args.base,
        windows=args.windows,
    )

    # Save CSV outputs
    save_dataframes_to_csv(out_root, corr_dict, jitter_dict)

    # Build and save latest summary CSV
    summary = build_latest_summary(corr_dict, jitter_dict)
    summary_path = os.path.join(out_root, "latest_corr_jitter_summary.csv")
    summary.to_csv(summary_path)
    logging.info(f"Wrote latest correlation/jitter summary CSV to {summary_path}")

    # Plot and save dashboards
    html_paths: List[str] = []
    html_paths += plot_timeseries_corr(out_root, corr_dict)
    html_paths += plot_timeseries_jitter(out_root, jitter_dict)
    html_paths += plot_latest_heatmaps(out_root, corr_dict, jitter_dict)

    # Open everything in browser tabs
    open_html_tabs(html_paths)

    logging.info("Dashboard generation complete.")


if __name__ == "__main__":
    main(sys.argv[1:])

