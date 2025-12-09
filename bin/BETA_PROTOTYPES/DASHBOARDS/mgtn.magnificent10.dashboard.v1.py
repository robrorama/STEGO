#!/usr/bin/env python3
# SCRIPTNAME: mgtn.magnificent10.dashboard.v1.py
# AUTHOR: Michael Derby
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# DATE: November 25, 2025
#
# PURPOSE
# -------
# New, MGTN-specific dashboard functionality (without duplicating your existing scripts):
#
# A) MGTN Index / Synthetic Equal-Weight Dashboard
#    - Builds a synthetic equal-weight "Magnificent 10" index from the 10 constituents.
#    - Optionally overlays any real index ticker you specify (e.g., "MGTN") if available via data_retrieval.
#    - Plots normalized (100 = start) curves for each constituent and the synthetic EQW index.
#
# B) Magnificent 10 Constituents Panel
#    - Wide price panel for the 10 names.
#    - Correlation heatmap of daily returns between all 10 tickers.
#    - All underlying series saved for STEGO pipelines as CSVs.
#
# E) Sector Mosaic Page (Plotly, multi-panel style similar to the sector image)
#    - SPY + SPDR sectors (XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLK, XLU, XLC).
#    - Each sector shown in a small multiple as relative performance vs SPY (ratio - 1).
#    - Outputs to /dev/shm via data_retrieval path helpers.
#
# DESIGN NOTES
# ------------
# - Uses your existing data_retrieval.py and options_data_retrieval.py AS-IS (no modification).
# - All outputs (CSVs + HTML dashboards) are written under /dev/shm/data/YYYY-MM-DD/MGTN_DASHBOARD.
# - Each major page is a separate Plotly figure:
#       1) MGTN Index & Synthetic Equal-Weight
#       2) MGTN Constituents Correlation Panel
#       3) Sector Relative-Performance Mosaic (SPY vs sectors)
#   Each figure is:
#       - Saved to HTML
#       - Shown via fig.show() so it opens in separate browser tabs.
#
# USAGE EXAMPLES
# --------------
#   python3 mgtn.magnificent10.dashboard.v1.py
#   python3 mgtn.magnificent10.dashboard.v1.py --index-ticker MGTN --period 2y
#   python3 mgtn.magnificent10.dashboard.v1.py --start 2023-01-01 --end 2025-11-25
#
# NOTES
# -----
# - If the specified index ticker (e.g. MGTN) is not available via yfinance, the script will still run
#   and will only plot the synthetic equal-weight index + constituents.
# - If any individual ticker fails to load, it is skipped, and processing continues for the rest.
# - This script focuses ONLY on new functionality (A, B, E) and does NOT attempt to replicate existing
#   options surfaces, macro dashboards, or other previous scripts.

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as po

# Your canonical loaders (must be present on PYTHONPATH / same dir when you run this)
import data_retrieval
import options_data_retrieval  # imported for completeness / future use, but not used heavily here


# ------------------------------------------------------------------------------------
# CONFIG & CONSTANTS
# ------------------------------------------------------------------------------------

MAGNIFICENT10_TICKERS: List[str] = [
    # "Magnificent 7" + AMD, AVGO, PLTR
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "PLTR",
]

SECTOR_TICKERS: List[str] = [
    "SPY",  # baseline
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLV",
    "XLI",
    "XLB",
    "XLK",
    "XLU",
    "XLC",
]


# ------------------------------------------------------------------------------------
# PATH / OUTPUT HELPERS (WIRED INTO data_retrieval)
# ------------------------------------------------------------------------------------

def init_output_root(label: str = "MGTN_DASHBOARD") -> str:
    """
    Use data_retrieval.create_dated_directory() to maintain your standard
    /dev/shm/data/YYYY-MM-DD/LABEL structure.

    Returns the root output directory for this run.
    """
    root = data_retrieval.create_dated_directory(label)
    os.makedirs(root, exist_ok=True)
    return root


def ensure_subdir(root: str, sub: str) -> str:
    path = os.path.join(root, sub)
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------------------------
# DATA LOADING HELPERS
# ------------------------------------------------------------------------------------

def _choose_price_column(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Pick a 'price' column from a yfinance-style DataFrame.
    Prefer 'Adj Close', then 'Close'; otherwise returns None.
    """
    for col in ["Adj Close", "Close"]:
        if col in df.columns:
            return df[col]
    return None


def load_single_ticker(
    ticker: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Light wrapper over data_retrieval.load_or_download_ticker with safe error handling.
    """
    try:
        if start or end:
            df = data_retrieval.load_or_download_ticker(
                ticker=ticker,
                start=start,
                end=end,
            )
        else:
            df = data_retrieval.load_or_download_ticker(
                ticker=ticker,
                period=period or "1y",
            )
        if df is None or df.empty:
            logging.warning(f"[{ticker}] Loaded empty DataFrame; skipping.")
            return None
        # Ensure DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                logging.warning(f"[{ticker}] Could not convert index to datetime; skipping.")
                return None
        df = df.sort_index()
        return df
    except Exception as e:
        logging.warning(f"Failed to load ticker {ticker}: {e}")
        return None


def load_constituent_panel(
    tickers: List[str],
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load OHLCV data for each constituent, align them on a common index, and
    return:
        - wide price panel (Close/Adj Close) with columns as tickers
        - dict of raw per-ticker DataFrames
    """
    series_dict: Dict[str, pd.Series] = {}
    raw_dict: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        df = load_single_ticker(t, period=period, start=start, end=end)
        if df is None:
            continue
        px = _choose_price_column(df)
        if px is None:
            logging.warning(f"[{t}] No price column found; skipping.")
            continue
        series_dict[t] = px.rename(t)
        raw_dict[t] = df

    if not series_dict:
        logging.error("No valid constituents loaded. Exiting panel creation.")
        return pd.DataFrame(), raw_dict

    # Align on common index (inner join)
    panel = pd.concat(series_dict.values(), axis=1, join="inner")
    panel = panel.dropna(how="all")
    return panel, raw_dict


# ------------------------------------------------------------------------------------
# SYNTHETIC EQUAL-WEIGHT INDEX CONSTRUCTION
# ------------------------------------------------------------------------------------

def build_synthetic_eqw_index(price_panel: pd.DataFrame) -> pd.Series:
    """
    Given a wide panel of prices (rows = dates, cols = tickers),
    construct an equal-weight synthetic index:
      - Normalize each column to 100 at the first non-NaN date.
      - Take the simple average across tickers at each date.
    """
    if price_panel.empty:
        return pd.Series(dtype=float)

    norm = price_panel / price_panel.iloc[0] * 100.0
    eqw = norm.mean(axis=1)
    eqw.name = "MGTN_EQW_SYNTH"
    return eqw


# ------------------------------------------------------------------------------------
# FIGURE BUILDERS
# ------------------------------------------------------------------------------------

def make_index_figure(
    eqw_index: pd.Series,
    price_panel: pd.DataFrame,
    real_index: Optional[pd.Series] = None,
    index_ticker: Optional[str] = None,
) -> go.Figure:
    """
    Build a Plotly figure showing:
      - Normalized (100) curves for each constituent
      - Synthetic equal-weight index
      - Optional overlay of real MGTN index (if available)
    """
    fig = go.Figure()

    if not price_panel.empty:
        norm_panel = price_panel / price_panel.iloc[0] * 100.0
        for col in norm_panel.columns:
            fig.add_trace(
                go.Scatter(
                    x=norm_panel.index,
                    y=norm_panel[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>" + col + "</extra>",
                )
            )

    if eqw_index is not None and not eqw_index.empty:
        fig.add_trace(
            go.Scatter(
                x=eqw_index.index,
                y=eqw_index.values,
                mode="lines",
                name="MGTN_EQW_SYNTH",
                line=dict(width=3, dash="solid"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>MGTN_EQW_SYNTH</extra>",
            )
        )

    if real_index is not None and not real_index.empty:
        label = index_ticker or "INDEX"
        fig.add_trace(
            go.Scatter(
                x=real_index.index,
                y=real_index.values,
                mode="lines",
                name=f"{label}_REAL",
                line=dict(width=2, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>" + f"{label}_REAL" + "</extra>",
            )
        )

    fig.update_layout(
        title="Magnificent 10 Synthetic Equal-Weight Index (and Constituents)",
        xaxis_title="Date",
        yaxis_title="Normalized Price (100 = first date)",
        hovermode="x unified",
    )
    return fig


def make_constituent_corr_figure(price_panel: pd.DataFrame) -> go.Figure:
    """
    Build a Plotly figure showing:
      - A correlation heatmap of daily returns between all Magnificent 10 tickers.
    """
    if price_panel.empty:
        # Empty figure placeholder
        fig = go.Figure()
        fig.update_layout(
            title="Magnificent 10 Correlation Heatmap (NO DATA)",
        )
        return fig

    # Compute daily returns (simple)
    returns = price_panel.pct_change().dropna(how="all")
    corr = returns.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar=dict(title="Correlation"),
            hovertemplate="Pair: %{y} vs %{x}<br>Corr: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Magnificent 10 – Daily Return Correlation Matrix",
        xaxis_title="Ticker",
        yaxis_title="Ticker",
    )
    return fig


def make_sector_mosaic_figure(
    sector_panel: pd.DataFrame,
    spy_series: pd.Series,
) -> go.Figure:
    """
    Build a sector mosaic (multi-panel) figure:
      - Each sector shows its relative performance vs SPY:
          rel = (sector_norm / spy_norm) - 1
      - Layout: small multiples style in a grid.
    """
    # Build relative performance table
    rel_dict: Dict[str, pd.Series] = {}
    # Align SPY with sectors
    if spy_series is None or spy_series.empty:
        # Can't compute relative; just return blank figure
        fig = go.Figure()
        fig.update_layout(
            title="Sector Mosaic (SPY baseline missing; no data)",
        )
        return fig

    # Normalize SPY
    spy_norm = spy_series / spy_series.iloc[0] * 100.0

    # For each sector (excluding SPY itself), compute rel perf
    for col in sector_panel.columns:
        if col == "SPY":
            continue
        s = sector_panel[col].dropna()
        # Align with spy_norm
        df_join = pd.concat([spy_norm.rename("SPY"), s.rename(col)], axis=1, join="inner").dropna()
        if df_join.empty:
            continue
        sector_norm = df_join[col] / df_join[col].iloc[0] * 100.0
        spy_norm_aligned = df_join["SPY"]
        rel = sector_norm / spy_norm_aligned - 1.0
        rel.name = col
        rel_dict[col] = rel

    if not rel_dict:
        fig = go.Figure()
        fig.update_layout(
            title="Sector Mosaic (no valid sector vs SPY pairs)",
        )
        return fig

    rel_panel = pd.concat(rel_dict.values(), axis=1, join="outer")

    # Build mosaic grid
    sectors = list(rel_dict.keys())
    n = len(sectors)
    # Reasonable grid: up to 4 columns
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        subplot_titles=sectors,
    )

    # Add each sector's rel perf as a separate subplot
    row = 1
    col = 1
    for sector in sectors:
        series = rel_panel[sector].dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=sector,
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>Rel vs SPY: %{y:.2%}<extra>" + sector + "</extra>",
            ),
            row=row,
            col=col,
        )

        # Add zero-line
        fig.add_hline(
            y=0.0,
            line=dict(width=1, dash="dot"),
            row=row,
            col=col,
        )

        col += 1
        if col > ncols:
            col = 1
            row += 1

    fig.update_layout(
        title="Sector Relative Performance vs SPY (Mosaic)",
        hovermode="x unified",
        showlegend=False,
    )
    # Tighten subplot spacing
    fig.update_layout(
        margin=dict(l=40, r=20, t=80, b=40),
    )
    return fig


# ------------------------------------------------------------------------------------
# SECTOR PANEL LOADING
# ------------------------------------------------------------------------------------

def load_sector_panel(
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load SPY + sector ETFs into a wide price panel and return:
        - sector_panel: prices for all tickers
        - spy_series: price for SPY
    """
    series_dict: Dict[str, pd.Series] = {}

    for t in SECTOR_TICKERS:
        df = load_single_ticker(t, period=period, start=start, end=end)
        if df is None:
            continue
        px = _choose_price_column(df)
        if px is None:
            logging.warning(f"[{t}] No price column for sector panel.")
            continue
        series_dict[t] = px.rename(t)

    if not series_dict:
        logging.error("No sector data loaded.")
        return pd.DataFrame(), None

    panel = pd.concat(series_dict.values(), axis=1, join="inner").dropna(how="all")
    spy_series = panel["SPY"] if "SPY" in panel.columns else None
    return panel, spy_series


# ------------------------------------------------------------------------------------
# CSV OUTPUT HELPERS
# ------------------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path)
        logging.info(f"Saved CSV: {path}")
    except Exception as e:
        logging.warning(f"Failed to save CSV {path}: {e}")


def save_series_csv(series: pd.Series, path: str) -> None:
    try:
        series.to_frame(name=series.name if series.name else "value").to_csv(path)
        logging.info(f"Saved CSV: {path}")
    except Exception as e:
        logging.warning(f"Failed to save CSV {path}: {e}")


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MGTN Magnificent 10 Dashboard (Index, Constituents, Sector Mosaic)"
    )
    p.add_argument(
        "--index-ticker",
        type=str,
        default="MGTN",
        help="Real index ticker to try to overlay (default: MGTN). If not found, only synthetic EQW is used.",
    )
    p.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Period string for yfinance-style downloads (ignored if --start/--end provided).",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD). If provided, overrides --period together with --end.",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD). If provided, overrides --period together with --start.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="If set, do NOT call fig.show() (useful for headless batch runs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("=== MGTN Magnificent 10 Dashboard (A + B + E) ===")
    logging.info(f"Index ticker: {args.index_ticker}")
    logging.info(f"Period: {args.period}, Start: {args.start}, End: {args.end}")

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------
    root = init_output_root("MGTN_DASHBOARD")
    index_dir = ensure_subdir(root, "INDEX")
    corr_dir = ensure_subdir(root, "CONSTITUENTS")
    sector_dir = ensure_subdir(root, "SECTORS")

    # -------------------------------------------------------------------------
    # Load Magnificent 10 panel
    # -------------------------------------------------------------------------
    price_panel, raw_constituents = load_constituent_panel(
        MAGNIFICENT10_TICKERS,
        period=args.period,
        start=args.start,
        end=args.end,
    )
    if price_panel.empty:
        logging.error("No data for Magnificent 10. Exiting.")
        return

    # Save underlying panel
    save_csv(price_panel, os.path.join(corr_dir, "mgtn_constituents_price_panel.csv"))

    # -------------------------------------------------------------------------
    # Build synthetic equal-weight index
    # -------------------------------------------------------------------------
    eqw_index = build_synthetic_eqw_index(price_panel)
    if not eqw_index.empty:
        save_series_csv(eqw_index, os.path.join(index_dir, "mgtn_eqw_synthetic_index.csv"))

    # Try to load real index ticker (e.g. MGTN) if available
    real_index_series = None
    if args.index_ticker:
        idx_df = load_single_ticker(
            args.index_ticker,
            period=args.period,
            start=args.start,
            end=args.end,
        )
        if idx_df is not None:
            idx_px = _choose_price_column(idx_df)
            if idx_px is not None and not idx_px.empty:
                # Normalize to same scale (100 at first date)
                idx_px = idx_px.sort_index()
                idx_px_norm = idx_px / idx_px.iloc[0] * 100.0
                idx_px_norm.name = f"{args.index_ticker}_REAL"
                real_index_series = idx_px_norm
                save_series_csv(
                    real_index_series,
                    os.path.join(index_dir, f"{args.index_ticker}_real_index_normalized.csv"),
                )
            else:
                logging.info(f"Index ticker {args.index_ticker} loaded but no price column; skipping real overlay.")
        else:
            logging.info(f"Index ticker {args.index_ticker} not available; running EQW-only.")

    # -------------------------------------------------------------------------
    # FIGURE 1: MGTN Index & Synthetic Equal-Weight (A)
    # -------------------------------------------------------------------------
    fig_index = make_index_figure(
        eqw_index=eqw_index,
        price_panel=price_panel,
        real_index=real_index_series,
        index_ticker=args.index_ticker,
    )
    index_html = os.path.join(index_dir, "mgtn_index_synthetic_eqw.html")
    po.plot(fig_index, filename=index_html, auto_open=False)
    logging.info(f"Saved INDEX dashboard HTML: {index_html}")
    if not args.no_show:
        fig_index.show()

    # -------------------------------------------------------------------------
    # FIGURE 2: Magnificent 10 Correlation Panel (B)
    # -------------------------------------------------------------------------
    fig_corr = make_constituent_corr_figure(price_panel)
    corr_html = os.path.join(corr_dir, "mgtn_constituents_correlation.html")
    po.plot(fig_corr, filename=corr_html, auto_open=False)
    logging.info(f"Saved CONSTITUENTS correlation HTML: {corr_html}")
    if not args.no_show:
        fig_corr.show()

    # -------------------------------------------------------------------------
    # FIGURE 3: Sector Mosaic Page (E)
    # -------------------------------------------------------------------------
    sector_panel, spy_series = load_sector_panel(
        period=args.period,
        start=args.start,
        end=args.end,
    )
    if not sector_panel.empty and spy_series is not None and not spy_series.empty:
        # Save raw sector panel
        save_csv(sector_panel, os.path.join(sector_dir, "sector_price_panel.csv"))
        fig_sector = make_sector_mosaic_figure(sector_panel=sector_panel, spy_series=spy_series)
        sector_html = os.path.join(sector_dir, "sector_mosaic_relative_vs_spy.html")
        po.plot(fig_sector, filename=sector_html, auto_open=False)
        logging.info(f"Saved SECTOR mosaic HTML: {sector_html}")
        if not args.no_show:
            fig_sector.show()
    else:
        logging.warning("Sector data or SPY series missing; sector mosaic (E) not generated.")

    logging.info("=== MGTN Magnificent 10 Dashboard DONE ===")


if __name__ == "__main__":
    main()

