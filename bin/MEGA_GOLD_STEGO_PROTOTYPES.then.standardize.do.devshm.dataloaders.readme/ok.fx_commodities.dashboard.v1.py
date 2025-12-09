#!/usr/bin/env python3
# SCRIPTNAME: fx_commodity_equity_bond_dashboard.v1.py
# AUTHOR: Michael Derby (STEGO FINANCIAL FRAMEWORK)
# DATE:   2025-11-24
#
# PURPOSE
# -------
# Unified Plotly dashboard generator for:
#   1) FX–Commodity Couplings & Betas
#       - USD/CAD (CAD=X) vs WTI Crude (CL=F)
#       - AUD/USD (AUDUSD=X) vs Copper (HG=F)
#   2) Equity–Bond Co-moves & Correlations (ETF Proxies)
#       - SPY vs IEF (10y proxy)
#       - SPY vs SHY (2y proxy)
#       - QQQ vs IEF
#
# DESIGN
# ------
# - Uses your existing data_retrieval.py (unchanged, imported as a module).
# - NEVER passes "interval" into load_or_download_ticker (you’ve seen that error).
# - All outputs (CSVs + HTML) go into /dev/shm-based trees:
#       BASE_PLOTS_PATH (default: /dev/shm/plots)
#       BASE_CACHE_PATH (default: /dev/shm/cache)
# - Each major view is a separate Plotly HTML file and is opened in a NEW browser tab:
#       FX:   USD/CAD vs WTI  (prices + rolling betas/dispersion)
#             AUD/USD vs Copper
#       EQ/B: SPY / Bonds correlations
#             QQQ / Bonds correlations
#
# - Everything is function-based (no classes), ready to drop into your script arsenal.

import argparse
import logging
import os
import sys
import webbrowser
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    print("ERROR: plotly is required for this dashboard. Install via: pip install plotly", file=sys.stderr)
    raise

# ---- LOAD YOUR DATA LAYER (unchanged) ----
try:
    import data_retrieval as dr
except ImportError as e:
    print("ERROR: Could not import data_retrieval.py. Ensure it is on PYTHONPATH or in the same directory.", file=sys.stderr)
    raise


# =========================
# PATH / ENV HELPERS
# =========================

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def get_base_plots_path() -> str:
    # Where all Plotly HTMLs go by default
    return _env("BASE_PLOTS_PATH", "/dev/shm/plots")


def get_base_cache_path() -> str:
    # Where CSVs / numeric outputs go by default
    return _env("BASE_CACHE_PATH", "/dev/shm/cache")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================
# DATA LOADING
# =========================

def load_price_series(
    ticker: str,
    period: str = "10y",
    price_col: str = "Close"
) -> pd.Series:
    """
    Wrapper around your data_retrieval.load_or_download_ticker.

    IMPORTANT:
    - Does NOT pass 'interval' (we've seen that create unexpected keyword errors).
    - Expects a DataFrame with a price_col (default 'Close').
    """
    logging.info(f"Loading ticker {ticker} for period={period} via data_retrieval...")
    try:
        df = dr.load_or_download_ticker(ticker, period=period)
    except TypeError:
        # Fallback if your function doesn't accept 'period' either:
        logging.warning(
            f"load_or_download_ticker signature may not accept 'period'; retrying without it for {ticker}..."
        )
        df = dr.load_or_download_ticker(ticker)

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"data_retrieval.load_or_download_ticker({ticker}) did not return a DataFrame")

    if price_col not in df.columns:
        raise ValueError(f"Expected column '{price_col}' in data for {ticker}, got columns: {df.columns}")

    s = df[price_col].copy()
    s = s.dropna()
    s.name = ticker
    return s


def align_pair(
    s1: pd.Series,
    s2: pd.Series,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Align two price series on common dates.
    Returns a DataFrame with columns [s1.name, s2.name].
    """
    df = pd.concat([s1, s2], axis=1)
    if dropna:
        df = df.dropna()
    return df


# =========================
# STATS HELPERS
# =========================

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple log returns for each column.
    """
    return np.log(df / df.shift(1)).dropna()


def rolling_beta_and_corr(
    df_returns: pd.DataFrame,
    y_col: str,
    x_col: str,
    window: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling beta and correlation of Y vs X over a given window.

    Beta = Cov(Y, X) / Var(X)
    Corr = Cov(Y, X) / (std(Y)*std(X))
    """
    if y_col not in df_returns.columns or x_col not in df_returns.columns:
        raise ValueError(f"Columns {y_col} and/or {x_col} not in returns DataFrame.")

    y = df_returns[y_col]
    x = df_returns[x_col]

    rolling_cov = y.rolling(window).cov(x)
    rolling_var = x.rolling(window).var()
    beta = rolling_cov / rolling_var

    rolling_corr = y.rolling(window).corr(x)

    beta.name = f"beta_{y_col}_vs_{x_col}_{window}"
    rolling_corr.name = f"corr_{y_col}_vs_{x_col}_{window}"
    return beta, rolling_corr


def compute_multi_window_stats(
    df_prices: pd.DataFrame,
    base_col: str,
    other_col: str,
    windows: List[int]
) -> Dict[str, pd.Series]:
    """
    Compute log returns & rolling betas / corrs across multiple windows.
    Returns dict:
        {
            'returns': DataFrame,
            'beta_<window>': Series,
            'corr_<window>': Series,
            'dispersion_beta_<w_short>_<w_long>': Series (long - short betas)
        }
    """
    results: Dict[str, pd.Series] = {}

    rets = compute_log_returns(df_prices[[base_col, other_col]])
    results["returns"] = rets

    betas = {}
    corrs = {}
    for w in windows:
        beta_w, corr_w = rolling_beta_and_corr(rets, y_col=base_col, x_col=other_col, window=w)
        betas[w] = beta_w
        corrs[w] = corr_w
        results[f"beta_{w}"] = beta_w
        results[f"corr_{w}"] = corr_w

    # Compute dispersions for consecutive window pairs (e.g., 21 vs 63)
    if len(windows) >= 2:
        sorted_w = sorted(windows)
        for i in range(len(sorted_w) - 1):
            w_short = sorted_w[i]
            w_long = sorted_w[i + 1]
            disp = betas[w_long] - betas[w_short]
            disp.name = f"dispersion_beta_{w_short}_{w_long}"
            results[disp.name] = disp

    return results


# =========================
# PLOTTING HELPERS
# =========================

def save_and_open_fig(fig: go.Figure, out_html_path: str, title: str) -> None:
    """
    Save a Plotly figure to HTML and open in a new browser tab.
    """
    logging.info(f"Writing Plotly HTML: {out_html_path}")
    fig.write_html(out_html_path, include_plotlyjs="cdn", full_html=True)
    # Ensure we open via file://
    url = "file://" + os.path.abspath(out_html_path)
    logging.info(f"Opening in web browser: {url}")
    webbrowser.open_new_tab(url)


def plot_fx_commodity_panel(
    pair_name: str,
    df_prices: pd.DataFrame,
    stats: Dict[str, pd.Series],
    base_label: str,
    other_label: str,
    windows: List[int],
    out_html_path: str
) -> None:
    """
    Create a multi-panel Plotly figure for an FX–Commodity pair:
      - Top: normalized prices
      - Middle: rolling betas for all windows
      - Bottom: dispersion (long - short) if present
    """
    # Normalize prices (start at 100)
    norm = df_prices / df_prices.iloc[0] * 100.0

    # Build figure with 3 rows sharing x-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.35, 0.20],
        subplot_titles=(
            f"{pair_name}: Normalized Prices",
            f"{pair_name}: Rolling Betas ({', '.join([str(w) + 'd' for w in windows])})",
            f"{pair_name}: Beta Dispersion (long - short)"
        )
    )

    # --- Row 1: prices ---
    fig.add_trace(
        go.Scatter(
            x=norm.index,
            y=norm.iloc[:, 0],
            mode="lines",
            name=base_label
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=norm.index,
            y=norm.iloc[:, 1],
            mode="lines",
            name=other_label
        ),
        row=1,
        col=1
    )

    # --- Row 2: betas ---
    for w in sorted(windows):
        beta_key = f"beta_{w}"
        if beta_key in stats:
            beta = stats[beta_key]
            fig.add_trace(
                go.Scatter(
                    x=beta.index,
                    y=beta.values,
                    mode="lines",
                    name=f"Beta {base_label} vs {other_label} ({w}d)"
                ),
                row=2,
                col=1
            )

    # --- Row 3: dispersion (if computed) ---
    if len(windows) >= 2:
        sorted_w = sorted(windows)
        for i in range(len(sorted_w) - 1):
            w_short = sorted_w[i]
            w_long = sorted_w[i + 1]
            disp_key = f"dispersion_beta_{w_short}_{w_long}"
            if disp_key in stats:
                disp = stats[disp_key]
                fig.add_trace(
                    go.Scatter(
                        x=disp.index,
                        y=disp.values,
                        mode="lines",
                        name=f"Dispersion Beta ({w_long}d - {w_short}d)"
                    ),
                    row=3,
                    col=1
                )

    fig.update_layout(
        title=pair_name,
        showlegend=True,
        hovermode="x unified",
        height=900
    )

    save_and_open_fig(fig, out_html_path, pair_name)


def plot_equity_bond_correlation_panel(
    title: str,
    df_prices: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    windows: List[int],
    out_html_path: str
) -> None:
    """
    Panel for equity-bond rolling correlations.

    df_prices: DataFrame with all needed tickers as columns.
    pairs: list of (equity_ticker, bond_ticker)
    windows: rolling windows (e.g., [21, 63])
    """
    # Compute returns for all series
    rets = compute_log_returns(df_prices)

    fig = make_subplots(
        rows=len(pairs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[
            f"{eq} vs {bd}: Rolling Correlation ({', '.join([str(w) + 'd' for w in windows])})"
            for (eq, bd) in pairs
        ]
    )

    # For each pair, compute corr windows and plot
    for i, (eq, bd) in enumerate(pairs, start=1):
        if eq not in rets.columns or bd not in rets.columns:
            logging.warning(f"Missing {eq} or {bd} in returns for equity-bond panel; skipping {eq}-{bd}.")
            continue

        for w in sorted(windows):
            corr_s = rets[eq].rolling(w).corr(rets[bd])
            corr_s.name = f"corr_{eq}_vs_{bd}_{w}"
            fig.add_trace(
                go.Scatter(
                    x=corr_s.index,
                    y=corr_s.values,
                    mode="lines",
                    name=f"{eq} vs {bd} ({w}d)",
                    showlegend=True if i == 1 else False  # show legend only in top subplot
                ),
                row=i,
                col=1
            )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=400 * len(pairs)
    )

    save_and_open_fig(fig, out_html_path, title)


# =========================
# CSV OUTPUT
# =========================

def write_stats_to_csv(
    stats: Dict[str, pd.Series],
    out_csv_dir: str,
    prefix: str
) -> None:
    """
    For a stats dict from compute_multi_window_stats, dump each Series
    into a combined wide CSV and one long-form CSV (for convenient use).
    """
    ensure_dir(out_csv_dir)

    # Wide format
    wide_df = pd.DataFrame()
    for k, v in stats.items():
        if isinstance(v, pd.Series):
            wide_df[k] = v
        elif isinstance(v, pd.DataFrame):
            # e.g., 'returns'
            for col in v.columns:
                wide_df[f"{k}_{col}"] = v[col]

    wide_path = os.path.join(out_csv_dir, f"{prefix}_wide.csv")
    logging.info(f"Writing wide CSV stats: {wide_path}")
    wide_df.to_csv(wide_path, index=True)

    # Long format (key, date, value)
    long_records = []
    for k, v in stats.items():
        if isinstance(v, pd.Series):
            temp = pd.DataFrame({"date": v.index, "metric": k, "value": v.values})
            long_records.append(temp)
        elif isinstance(v, pd.DataFrame):
            for col in v.columns:
                temp = pd.DataFrame(
                    {
                        "date": v.index,
                        "metric": f"{k}_{col}",
                        "value": v[col].values,
                    }
                )
                long_records.append(temp)

    if long_records:
        long_df = pd.concat(long_records, axis=0, ignore_index=True)
        long_path = os.path.join(out_csv_dir, f"{prefix}_long.csv")
        logging.info(f"Writing long CSV stats: {long_path}")
        long_df.to_csv(long_path, index=False)


# =========================
# MAIN WORKFLOW
# =========================

def build_fx_commodity_dashboards(
    period: str,
    windows: List[int],
    plots_base: str,
    cache_base: str
) -> None:
    """
    Construct dashboards for:
      - USD/CAD vs WTI
      - AUD/USD vs Copper
    """
    fx_plots_dir = os.path.join(plots_base, "FX_COMMODITY_DASHBOARD")
    fx_cache_dir = os.path.join(cache_base, "FX_COMMODITY_DASHBOARD")
    ensure_dir(fx_plots_dir)
    ensure_dir(fx_cache_dir)

    # ---- 1) USD/CAD vs WTI ----
    usdcad_ticker = "CAD=X"       # USD per 1 CAD (yfinance naming)
    wti_ticker = "CL=F"           # WTI front-month futures
    logging.info("=== FX–Commodity: USD/CAD vs WTI ===")
    s_usdcad = load_price_series(usdcad_ticker, period=period)
    s_wti = load_price_series(wti_ticker, period=period)

    df_usd_wti = align_pair(s_usdcad, s_wti)
    df_usd_wti.columns = ["USD/CAD", "WTI"]

    stats_usd_wti = compute_multi_window_stats(
        df_usd_wti,
        base_col="USD/CAD",
        other_col="WTI",
        windows=windows
    )

    write_stats_to_csv(
        stats_usd_wti,
        out_csv_dir=fx_cache_dir,
        prefix="USDCAD_WTI"
    )

    usd_wti_html = os.path.join(fx_plots_dir, "USDCAD_vs_WTI_dashboard.html")
    plot_fx_commodity_panel(
        pair_name="USD/CAD vs WTI",
        df_prices=df_usd_wti,
        stats=stats_usd_wti,
        base_label="USD/CAD",
        other_label="WTI",
        windows=windows,
        out_html_path=usd_wti_html
    )

    # ---- 2) AUD/USD vs Copper ----
    audusd_ticker = "AUDUSD=X"
    copper_ticker = "HG=F"
    logging.info("=== FX–Commodity: AUD/USD vs Copper ===")
    s_audusd = load_price_series(audusd_ticker, period=period)
    s_copper = load_price_series(copper_ticker, period=period)

    df_aud_cu = align_pair(s_audusd, s_copper)
    df_aud_cu.columns = ["AUD/USD", "Copper"]

    stats_aud_cu = compute_multi_window_stats(
        df_aud_cu,
        base_col="AUD/USD",
        other_col="Copper",
        windows=windows
    )

    write_stats_to_csv(
        stats_aud_cu,
        out_csv_dir=fx_cache_dir,
        prefix="AUDUSD_COPPER"
    )

    aud_cu_html = os.path.join(fx_plots_dir, "AUDUSD_vs_Copper_dashboard.html")
    plot_fx_commodity_panel(
        pair_name="AUD/USD vs Copper",
        df_prices=df_aud_cu,
        stats=stats_aud_cu,
        base_label="AUD/USD",
        other_label="Copper",
        windows=windows,
        out_html_path=aud_cu_html
    )


def build_equity_bond_dashboards(
    period: str,
    windows: List[int],
    plots_base: str,
    cache_base: str
) -> None:
    """
    Construct dashboards for equity–bond co-moves using ETF proxies:
      - SPY vs IEF (10y proxy)
      - SPY vs SHY (2y proxy)
      - QQQ vs IEF
    """
    eq_plots_dir = os.path.join(plots_base, "EQUITY_BOND_DASHBOARD")
    eq_cache_dir = os.path.join(cache_base, "EQUITY_BOND_DASHBOARD")
    ensure_dir(eq_plots_dir)
    ensure_dir(eq_cache_dir)

    logging.info("=== Equity–Bond: loading ETF proxies ===")
    spy = load_price_series("SPY", period=period)
    qqq = load_price_series("QQQ", period=period)
    # Bond ETFs as yield proxies:
    #   SHY ~ 1-3y Treasuries (short end / 2y-ish)
    #   IEF ~ 7-10y Treasuries (10y-ish)
    shy = load_price_series("SHY", period=period)
    ief = load_price_series("IEF", period=period)

    df_all = pd.concat([spy, qqq, shy, ief], axis=1).dropna()
    df_all.columns = ["SPY", "QQQ", "SHY", "IEF"]

    # Save raw aligned prices for reference
    raw_prices_path = os.path.join(eq_cache_dir, "EQUITY_BOND_prices.csv")
    logging.info(f"Writing equity-bond raw prices: {raw_prices_path}")
    df_all.to_csv(raw_prices_path, index=True)

    # ---- Build SPY / Bonds panel ----
    spy_pairs = [("SPY", "IEF"), ("SPY", "SHY")]
    spy_html = os.path.join(eq_plots_dir, "SPY_bonds_correlation_dashboard.html")
    plot_equity_bond_correlation_panel(
        title="SPY vs Bonds: Rolling Correlations",
        df_prices=df_all[["SPY", "SHY", "IEF"]],
        pairs=spy_pairs,
        windows=windows,
        out_html_path=spy_html
    )

    # ---- Build QQQ / Bonds panel ----
    qqq_pairs = [("QQQ", "IEF"), ("QQQ", "SHY")]
    qqq_html = os.path.join(eq_plots_dir, "QQQ_bonds_correlation_dashboard.html")
    plot_equity_bond_correlation_panel(
        title="QQQ vs Bonds: Rolling Correlations",
        df_prices=df_all[["QQQ", "SHY", "IEF"]],
        pairs=qqq_pairs,
        windows=windows,
        out_html_path=qqq_html
    )

    # Optional: write rolling corr data as CSV for further analysis
    rets = compute_log_returns(df_all)
    corr_records = []
    for (eq, bd) in spy_pairs + qqq_pairs:
        for w in sorted(windows):
            corr_s = rets[eq].rolling(w).corr(rets[bd])
            temp = pd.DataFrame(
                {
                    "date": corr_s.index,
                    "equity": eq,
                    "bond": bd,
                    "window": w,
                    "corr": corr_s.values,
                }
            )
            corr_records.append(temp)
    if corr_records:
        corr_df = pd.concat(corr_records, axis=0, ignore_index=True)
        corr_path = os.path.join(eq_cache_dir, "EQUITY_BOND_rolling_correlations_long.csv")
        logging.info(f"Writing equity-bond rolling correlations: {corr_path}")
        corr_df.to_csv(corr_path, index=False)


# =========================
# CLI / MAIN
# =========================

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FX–Commodity & Equity–Bond Macro Dashboard (STEGO Framework)"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="10y",
        help="Period for historical data (passed to data_retrieval.load_or_download_ticker when supported, default: 10y)."
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[21, 63],
        help="Rolling windows in trading days (e.g., 21 63). Default: 21 63."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO."
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    plots_base = get_base_plots_path()
    cache_base = get_base_cache_path()

    logging.info("=== FX–Commodity & Equity–Bond Macro Dashboard ===")
    logging.info(f"Period: {args.period}")
    logging.info(f"Rolling windows: {args.windows}")
    logging.info(f"Plots base directory: {plots_base}")
    logging.info(f"Cache base directory: {cache_base}")

    # Build FX–Commodity tabs (two HTMLs in /dev/shm/plots/FX_COMMODITY_DASHBOARD)
    build_fx_commodity_dashboards(
        period=args.period,
        windows=args.windows,
        plots_base=plots_base,
        cache_base=cache_base
    )

    # Build Equity–Bond tabs (two HTMLs in /dev/shm/plots/EQUITY_BOND_DASHBOARD)
    build_equity_bond_dashboards(
        period=args.period,
        windows=args.windows,
        plots_base=plots_base,
        cache_base=cache_base
    )

    logging.info("All dashboards generated and opened in browser tabs.")


if __name__ == "__main__":
    main(sys.argv[1:])

