#!/usr/bin/env python3
# SCRIPTNAME: commodities_fx_usd_dashboard.py
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# AUTHOR: Michael Derby (with ChatGPT assistant)
# DATE: 2025-11-25
#
# PURPOSE
# -------
# Build an interactive Plotly dashboard that:
#   - Pulls GSCI-style broad commodities and USD index (DXY proxy) using your
#     existing data_retrieval.py loader (unchanged).
#   - Recreates a "Commodities vs US Dollar" dual-axis chart similar to the
#     reference image (commodities vs DXY).
#   - Computes rolling correlations between:
#         * DXY vs broad commodities and key raw materials (oil, copper, gold).
#         * Commodity FX pairs vs their "home" commodities (AUD–Copper,
#           CAD–WTI, NOK–Brent, ZAR–Gold) where data is available.
#   - Writes all raw price panels and correlation series to CSV under
#     /dev/shm/data/YYYY-MM-DD/COMMODITY_FX_USD_DASHBOARD
#   - Writes a single Plotly HTML dashboard to the same directory and opens it
#     in the browser.
#
# IMPORTANT
# ---------
# - Uses data_retrieval.py exactly as-is (no modifications).
# - All disk I/O goes through the /dev/shm-driven BASE_DATA_PATH()/BASE_CACHE_PATH().
# - Designed to be a standalone script you can drop into your STEGO ecosystem.
#
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# --- YOUR EXISTING DATA LAYERS (unchanged) ---
import data_retrieval as dr


def ensure_output_root(subdir_name: str = "COMMODITY_FX_USD_DASHBOARD") -> str:
    """
    Create a dated output directory under BASE_DATA_PATH, e.g.:
        /dev/shm/data/YYYY-MM-DD/COMMODITY_FX_USD_DASHBOARD
    and return its path.
    """
    base = dr.BASE_DATA_PATH()
    today = datetime.today().strftime("%Y-%m-%d")
    root = os.path.join(base, today, subdir_name)
    os.makedirs(root, exist_ok=True)
    return root


def try_load_close_series(ticker: str, period: str = "5y") -> pd.Series:
    """
    Use dr.load_or_download_ticker() to load a single ticker and return its
    'Close' price series. Returns an empty Series on failure.
    """
    try:
        df = dr.load_or_download_ticker(ticker, period=period)
        if df is None or len(df) == 0:
            logging.warning(f"No data returned for ticker {ticker} (period={period}).")
            return pd.Series(dtype=float)
        if "Close" not in df.columns:
            logging.warning(f"'Close' column not found for ticker {ticker}.")
            return pd.Series(dtype=float)
        s = pd.to_numeric(df["Close"], errors="coerce")
        s.name = ticker
        return s.dropna()
    except Exception as e:
        logging.warning(f"Failed to load ticker {ticker}: {e}")
        return pd.Series(dtype=float)


def load_first_available_close(label: str,
                               ticker_candidates,
                               period: str = "5y") -> tuple[str, pd.Series]:
    """
    Given a human-readable label and a list of possible tickers, try each in
    order until we get a non-empty 'Close' series.

    Returns
    -------
    chosen_ticker : str
    series        : pd.Series (index=dates, name=label)
    """
    for tkr in ticker_candidates:
        s = try_load_close_series(tkr, period=period)
        if not s.empty:
            logging.info(f"{label}: using ticker {tkr} (period={period}, rows={len(s)})")
            s = s.rename(label)
            return tkr, s

    raise RuntimeError(f"Could not load any ticker for {label}. Tried: {ticker_candidates}")


def build_price_panel(period: str = "5y") -> tuple[pd.DataFrame, dict]:
    """
    Load all relevant instruments and build a unified Close-price panel
    (inner-joined on the date index).

    Returns
    -------
    panel : pd.DataFrame   columns = labels (e.g., 'DXY', 'GSCI', 'WTI Crude')
    meta  : dict           label -> {'ticker': chosen_ticker}
    """
    instruments = {
        # USD broad index
        "DXY": ["DX-Y.NYB", "DXY", "UUP"],

        # Broad commodities (GSCI-style proxy)
        "GSCI Commodities": ["^SPGSCI", "GSG", "DBC"],

        # Key raw materials
        "WTI Crude": ["CL=F"],
        "Brent Crude": ["BZ=F"],
        "Copper": ["HG=F"],
        "Gold": ["GC=F"],

        # Commodity FX pairs (or close proxies)
        "AUDUSD": ["AUDUSD=X"],
        "USDCAD": ["USDCAD=X", "CAD=X"],
        "NOKUSD": ["NOKUSD=X", "NOK=X"],
        "ZARUSD": ["ZARUSD=X", "ZAR=X"],
    }

    series_list = []
    meta: dict[str, dict] = {}

    for label, cands in instruments.items():
        try:
            chosen, s = load_first_available_close(label, cands, period=period)
            series_list.append(s)
            meta[label] = {"ticker": chosen}
        except Exception as e:
            logging.warning(f"Skipping {label}: {e}")

    if not series_list:
        raise RuntimeError("No instruments could be loaded; aborting.")

    panel = pd.concat(series_list, axis=1, join="inner").sort_index()
    panel = panel.loc[~panel.index.duplicated(keep="first")]

    logging.info(f"Price panel built with shape {panel.shape} and columns {list(panel.columns)}")
    return panel, meta


def compute_log_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns for each column in the price panel.
    """
    log_prices = np.log(price_panel.replace(0, np.nan))
    rets = log_prices.diff().dropna(how="all")
    return rets


def rolling_correlation(rets: pd.DataFrame,
                        base: str,
                        others: list[str],
                        window: int = 60) -> pd.DataFrame:
    """
    Compute rolling window correlations between one base series and a list of
    others. Any missing pair is skipped.

    Returns a DataFrame whose columns are "base vs other".
    """
    cols = {}
    if base not in rets.columns:
        logging.warning(f"Base series {base} not in returns panel; cannot compute correlations.")
        return pd.DataFrame(index=rets.index)

    for other in others:
        if other not in rets.columns:
            logging.warning(f"Skipping correlation {base} vs {other}: missing in returns panel.")
            continue
        col_name = f"{base} vs {other}"
        cols[col_name] = rets[base].rolling(window).corr(rets[other])

    if not cols:
        return pd.DataFrame(index=rets.index)

    corr_df = pd.DataFrame(cols).dropna(how="all")
    logging.info(f"Rolling correlation {base} vs {others}: shape {corr_df.shape}")
    return corr_df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to CSV with logging.
    """
    try:
        df.to_csv(path)
        logging.info(f"Saved CSV: {path} (rows={len(df)})")
    except Exception as e:
        logging.warning(f"Failed to save CSV {path}: {e}")


def build_dashboard_figure(price_panel: pd.DataFrame,
                           meta: dict,
                           dxy_label: str = "DXY",
                           comm_label: str = "GSCI Commodities",
                           window: int = 60) -> tuple[go.Figure, dict, pd.DataFrame, pd.DataFrame]:
    """
    Construct a multi-row Plotly figure:

    Row 1: Normalized Commodities vs US Dollar (dual-axis).
    Row 2: Rolling correlations of DXY vs key commodities.
    Row 3: Rolling correlations of commodity FX vs "home" commodities.

    Returns (fig, diagnostics_dict, dxy_corr_df, fx_corr_df).
    """
    diagnostics = {}

    # --- Normalize prices for Row 1 ---
    needed = [dxy_label, comm_label]
    missing = [c for c in needed if c not in price_panel.columns]
    if missing:
        raise RuntimeError(f"Missing required columns for main chart: {missing}")

    norm_panel = price_panel[needed].copy()
    norm_panel = norm_panel / norm_panel.iloc[0] * 100.0

    # --- Compute log returns and correlations ---
    rets = compute_log_returns(price_panel)

    dxy_others = [c for c in ["GSCI Commodities", "WTI Crude", "Brent Crude", "Copper", "Gold"]
                  if c in rets.columns]
    dxy_corr = rolling_correlation(rets, base=dxy_label, others=dxy_others, window=window)

    fx_pairs = [
        ("AUDUSD", "Copper"),
        ("USDCAD", "WTI Crude"),
        ("NOKUSD", "Brent Crude"),
        ("ZARUSD", "Gold"),
    ]
    fx_corr_cols = {}
    for fx, com in fx_pairs:
        if fx not in rets.columns or com not in rets.columns:
            logging.warning(f"Skipping FX/commodity pair {fx} vs {com}: data missing.")
            continue
        col_name = f"{fx} vs {com}"
        fx_corr_cols[col_name] = rets[fx].rolling(window).corr(rets[com])
    fx_corr = pd.DataFrame(fx_corr_cols).dropna(how="all")

    diagnostics["dxy_corr_columns"] = list(dxy_corr.columns)
    diagnostics["fx_corr_columns"] = list(fx_corr.columns)

    # --- Build Plotly figure with 3 rows ---
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.06,
    )

    # Row 1: Commodities vs US Dollar (dual axis)
    fig.add_trace(
        go.Scatter(
            x=norm_panel.index,
            y=norm_panel[comm_label],
            mode="lines",
            name="GSCI Commodities (indexed)",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=norm_panel.index,
            y=norm_panel[dxy_label],
            mode="lines",
            name="DXY [indexed, RHS]",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Commodities (Index = 100)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="DXY (Index = 100)", row=1, col=1, secondary_y=True)

    # Row 2: DXY vs commodity rolling correlations
    if not dxy_corr.empty:
        for col in dxy_corr.columns:
            fig.add_trace(
                go.Scatter(
                    x=dxy_corr.index,
                    y=dxy_corr[col],
                    mode="lines",
                    name=col,
                ),
                row=2,
                col=1,
            )
        x0 = dxy_corr.index.min()
        x1 = dxy_corr.index.max()
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=0.0,
            y1=0.0,
            line=dict(dash="dash"),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text=f"{window}-day Rolling Corr (DXY vs Commodities)", row=2, col=1)
    else:
        logging.warning("DXY vs commodities rolling correlation DataFrame is empty; row 2 will be blank.")

    # Row 3: FX vs commodity rolling correlations
    if not fx_corr.empty:
        for col in fx_corr.columns:
            fig.add_trace(
                go.Scatter(
                    x=fx_corr.index,
                    y=fx_corr[col],
                    mode="lines",
                    name=col,
                ),
                row=3,
                col=1,
            )
        x0 = fx_corr.index.min()
        x1 = fx_corr.index.max()
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=0.0,
            y1=0.0,
            line=dict(dash="dash"),
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text=f"{window}-day Rolling Corr (FX vs Commodities)", row=3, col=1)
    else:
        logging.warning("FX vs commodity rolling correlation DataFrame is empty; row 3 will be blank.")

    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.update_layout(
        title="Commodities vs US Dollar & Commodity FX – STEGO Macro Dashboard",
        legend=dict(orientation="h", x=0, y=1.02),
        hovermode="x unified",
    )

    return fig, diagnostics, dxy_corr, fx_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Commodities vs US Dollar dashboard using STEGO data_retrieval loaders."
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="History period passed to data_retrieval.load_or_download_ticker (e.g. '3y', '5y', '10y', 'max').",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling window length (in trading days) for correlations.",
    )
    parser.add_argument(
        "--no-auto-open",
        action="store_true",
        help="If set, do not auto-open the Plotly HTML in a browser.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("=== COMMODITIES vs US DOLLAR DASHBOARD ===")
    logging.info(f"Period: {args.period}, window: {args.window}")

    # 1) Prepare output directory
    out_root = ensure_output_root()
    logging.info(f"Output root directory: {out_root}")

    # 2) Build price panel
    price_panel, meta = build_price_panel(period=args.period)

    # 3) Save prices and meta
    price_csv_path = os.path.join(out_root, "price_panel.csv")
    save_csv(price_panel, price_csv_path)

    meta_path = os.path.join(out_root, "instrument_meta.csv")
    meta_df = (
        pd.DataFrame.from_dict(meta, orient="index")
        .rename_axis("label")
        .reset_index()
    )
    save_csv(meta_df, meta_path)

    # 4) Build dashboard figure and correlation DataFrames
    fig, diagnostics, dxy_corr, fx_corr = build_dashboard_figure(
        price_panel=price_panel,
        meta=meta,
        dxy_label="DXY",
        comm_label="GSCI Commodities",
        window=args.window,
    )

    # Save correlation CSVs
    dxy_corr_path = os.path.join(out_root, f"dxy_vs_commodities_corr_{args.window}d.csv")
    fx_corr_path = os.path.join(out_root, f"fx_vs_commodities_corr_{args.window}d.csv")
    if not dxy_corr.empty:
        save_csv(dxy_corr, dxy_corr_path)
    if not fx_corr.empty:
        save_csv(fx_corr, fx_corr_path)

    # 5) Save Plotly HTML dashboard (and PNG if possible)
    html_path = os.path.join(out_root, "commodities_fx_usd_dashboard.html")
    try:
        pio.write_html(fig, file=html_path, auto_open=not args.no_auto_open, include_plotlyjs="cdn")
        logging.info(f"Dashboard HTML written to: {html_path}")
    except Exception as e:
        logging.warning(f"Failed to write or auto-open HTML dashboard: {e}")

    png_path = os.path.join(out_root, "commodities_fx_usd_dashboard.png")
    try:
        fig.write_image(png_path, width=1400, height=900, scale=2)
        logging.info(f"Dashboard PNG written to: {png_path}")
    except Exception as e:
        logging.warning(f"Could not export PNG (kaleido may not be installed): {e}")

    # 6) Log diagnostics
    logging.info(f"Diagnostics: {diagnostics}")


if __name__ == "__main__":
    main()

