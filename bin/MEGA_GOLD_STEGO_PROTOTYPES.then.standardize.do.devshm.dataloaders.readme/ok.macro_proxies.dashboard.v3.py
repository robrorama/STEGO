#!/usr/bin/env python3
# SCRIPTNAME: macro_proxies.dashboard.v2.py
# AUTHOR:    Michael Derby (generated with ChatGPT)
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# DATE:      2025-11-24
#
# PURPOSE
# -------
# Expanded macro "one-screen" style dashboard using TRADEABLE / INDEX PROXIES
# wired into your existing data_retrieval.py loader.
#
# PANELS / PROXY FAMILIES
# -----------------------
# 1) RATES / YIELD CURVE PROXIES
#    - Treasury yield index proxies (Yahoo):
#        * Short / front end: ^IRX (13-week T-bill)
#        * 5Y: ^FVX
#        * 10Y: ^TNX
#        * 30Y: ^TYX
#      (Also attempts ^UST5Y, ^UST10Y, ^UST30Y but safely ignores failures.)
#    - Slopes:
#        * 10s-2s    ~ 10Y (^TNX) - front-end proxy (^IRX)
#        * 30s-5s    ~ 30Y (^TYX) - 5Y (^FVX)
#        * 30s-10s   ~ 30Y (^TYX) - 10Y (^TNX)
#        * 10s-3m    ~ 10Y (^TNX) - 3M (^IRX)
#    - Term-premium style ETF ratios:
#        * IEF / SHY
#        * TLT / SHY
#
# 2) LIQUIDITY PROXIES
#    - SPY vs cash-equivalents: BIL, SHV, SGOV
#        * Normalized levels
#        * Ratios: SPY/BIL, SPY/SHV, SPY/SGOV
#    - Dollar liquidity:
#        * DXY index (^DXY)
#        * UUP ETF
#    - Global-liquidity proxy:
#        * ACWI / BIL
#    - ETF "flow" proxy:
#        * 60d rolling z-score of log(Volume) for SPY, QQQ, IWM, HYG
#
# 3) CREDIT SPREAD PROXIES
#    - ETFs: HYG, JNK, LQD, IEF, IEI, EMB, PCY, HYGH (best effort)
#    - Normalized ETF levels
#    - Spread proxies (log-ratios):
#        * HY vs Treasuries: log(HYG / IEF), log(JNK / IEF)
#        * IG vs Treasuries: log(LQD / IEF)
#        * HY vs IG:        log(HYG / LQD), log(JNK / LQD)
#        * EM vs Treas:     log(EMB / IEF), log(PCY / IEF)
#        * HY vs IEI:       log(HYG / IEI)
#        * IG vs IEI:       log(LQD / IEI)
#
# 4) BREADTH & ROTATION PROXIES
#    - Breadth:
#        * RSP / SPY (equal vs cap-weight S&P)
#        * QQQ / QQQE (concentration in Nasdaq 100)
#        * ACWI / RSP
#        * VT / VTI
#    - Style & size rotation:
#        * SPY, QQQ, IWM normalized
#    - Factor ETFs:
#        * MTUM, VLUE, QUAL, SIZE, USMV (normalized)
#
# 5) INFLATION & GROWTH PROXIES (COMMODITIES & TIPS)
#    - Commodities / sectors:
#        * DBC (broad), USO (oil), GLD, SLV, COPX, XLE, XLB
#        * Normalized levels + key ratios: GLD/SLV, DBC/SPY, XLE/SPY, COPX/SPY
#    - Inflation breakeven proxies:
#        * TIP / IEF
#        * TIP / SHY
#
# 6) RISK APPETITE & VOLATILITY
#    - VIX complex:
#        * ^VIX, ^VIX9D, ^VVIX (best effort)
#    - Cross-asset risk appetite:
#        * ARKK / QQQ
#        * ARKK / SPY
#        * BTC-USD / GLD
#        * SMH / SPY
#    - Vol risk premium proxy:
#        * SPY 21d realized vol vs VIX index
#
# 7) GROWTH vs VALUE / THEMATIC
#    - Growth vs small/value:
#        * QQQ / IWM
#    - Growth vs financials / energy:
#        * QQQ / XLF
#        * QQQ / XLE
#    - Growth innovation:
#        * ARKK / SPY
#        * IGV / SPY
#
# IMPLEMENTATION NOTES
# --------------------
# - ALWAYS uses your existing data_retrieval.py (never modified here).
# - Never passes "interval" to load_or_download_ticker (only period="max").
# - All outputs are written to: /dev/shm/MACRO_PROXIES_DASHBOARD (default).
# - Each figure is saved as its own Plotly HTML file and opened
#   in a separate browser tab when the script runs (unless --no-open).
#
# CLI EXAMPLES
# ------------
#   python3 macro_proxies.dashboard.v2.py
#   python3 macro_proxies.dashboard.v2.py --start 2015-01-01 --end 2025-11-24
#   python3 macro_proxies.dashboard.v2.py --no-open --verbose
#

import argparse
import logging
import os
import sys
import webbrowser
from datetime import date

import numpy as np
import pandas as pd

try:
    import data_retrieval
except Exception as e:
    print("ERROR: Failed to import data_retrieval.py. "
          "Ensure it is on PYTHONPATH or in the same directory.")
    print(f"Underlying error: {e}")
    sys.exit(1)

import plotly.graph_objects as go


# ======================================================================
# PATH / LOGGING HELPERS
# ======================================================================

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ======================================================================
# DATA LOADING HELPERS (using your data_retrieval.py)
# ======================================================================

def detect_price_column(df: pd.DataFrame) -> str:
    """Return appropriate price column name from a typical yfinance-style DataFrame."""
    candidates = ["Adj Close", "AdjClose", "Close", "close", "adj_close"]
    for col in candidates:
        if col in df.columns:
            return col
    # fallback: first numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    raise ValueError("No suitable price column found in DataFrame.")


def safe_load_ohlcv(ticker: str) -> pd.DataFrame | None:
    """
    Safely load OHLCV using your canonical loader.
    - Uses period='max'
    - Catches errors and returns None on failure.
    """
    try:
        logging.info(f"Loading OHLCV for {ticker} via data_retrieval.load_or_download_ticker(period='max')")
        df = data_retrieval.load_or_download_ticker(ticker, period="max")
        if df is None or len(df) == 0:
            logging.warning(f"{ticker}: received empty DataFrame.")
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        logging.warning(f"Failed to load {ticker}: {e}")
        return None


def safe_load_price_series(ticker: str,
                           start: pd.Timestamp | None,
                           end: pd.Timestamp | None) -> pd.Series | None:
    """
    Safely load a price series for a given ticker and slice to [start, end].
    Returns a pandas Series or None.
    """
    df = safe_load_ohlcv(ticker)
    if df is None:
        return None
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    if df.empty:
        logging.warning(f"{ticker}: empty after date filter.")
        return None
    try:
        price_col = detect_price_column(df)
    except ValueError as e:
        logging.warning(f"{ticker}: {e}")
        return None
    s = df[price_col].astype(float).copy()
    s.name = ticker
    return s


# ======================================================================
# PLOTTING HELPERS
# ======================================================================

def write_and_open_figure(fig: go.Figure, outdir: str, basename: str, open_html: bool = True) -> str:
    """
    Save a Plotly figure as HTML under outdir with basename and optionally open in browser.
    Returns the full path to the HTML file.
    """
    ensure_dir(outdir)
    html_path = os.path.join(outdir, f"{basename}.html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    logging.info(f"Wrote HTML: {html_path}")
    if open_html:
        url = "file://" + os.path.abspath(html_path)
        try:
            webbrowser.open_new_tab(url)
        except Exception as e:
            logging.warning(f"Could not open browser for {html_path}: {e}")
    return html_path


def normalize_to_100(series: pd.Series) -> pd.Series:
    """Normalize series so that the first valid point = 100."""
    s = series.dropna()
    if s.empty:
        return series * np.nan
    base = s.iloc[0]
    return (s / base) * 100.0


def rolling_zscore(s: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score for a series."""
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std()
    z = (s - rolling_mean) / rolling_std
    return z


def ann_realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualized realized volatility using rolling standard deviation of returns."""
    return np.sqrt(252.0) * returns.rolling(window).std()


# ======================================================================
# 1) RATES / YIELD CURVE PROXIES
# ======================================================================

def build_rates_panel(start: pd.Timestamp,
                      end: pd.Timestamp,
                      outdir: str,
                      open_html: bool) -> None:
    """
    Build rates / yield curve proxy charts:
      - Yield index levels: ^IRX, ^FVX, ^TNX, ^TYX (+ UST* if available).
      - Slope series: 10s-2s, 30s-5s, 30s-10s, 10s-3m.
      - Term-premium proxies: IEF/SHY, TLT/SHY.
    """
    ensure_dir(outdir)

    # --- Load yield index proxies ---
    label_to_ticker = {
        "3M (^IRX)": "^IRX",
        "5Y (^FVX)": "^FVX",
        "10Y (^TNX)": "^TNX",
        "30Y (^TYX)": "^TYX",
        # best-effort additional indices; ok if they fail
        "5Y (^UST5Y)": "^UST5Y",
        "10Y (^UST10Y)": "^UST10Y",
        "30Y (^UST30Y)": "^UST30Y",
    }

    yield_frames = []
    for label, t in label_to_ticker.items():
        s = safe_load_price_series(t, start, end)
        if s is not None:
            s.name = label
            yield_frames.append(s)

    if not yield_frames:
        logging.error("No yield indices loaded; skipping rates panel.")
        return

    ydf = pd.concat(yield_frames, axis=1).dropna(how="all")

    # --- Plot yield levels (index units) ---
    fig_yields = go.Figure()
    for col in ydf.columns:
        fig_yields.add_trace(
            go.Scatter(
                x=ydf.index,
                y=ydf[col],
                mode="lines",
                name=col
            )
        )
    fig_yields.update_layout(
        title="Yield Curve Proxies (Index Levels)",
        xaxis_title="Date",
        yaxis_title="Index Level (approx bps)",
        legend_title="Tenor"
    )
    write_and_open_figure(fig_yields, outdir, "01_rates_yield_indices", open_html=open_html)

    # --- Slopes ---
    slopes = pd.DataFrame(index=ydf.index)

    # Identify front-end / 2Y proxy: ^IRX as front-end, or UST5Y etc. We'll use IRX as "short" and FVX as 5Y.
    short_label = "3M (^IRX)"
    ten_label = "10Y (^TNX)"
    five_label = "5Y (^FVX)"
    thirty_label = "30Y (^TYX)"

    if ten_label in ydf.columns and short_label in ydf.columns:
        slopes["10s-3m (TNX - IRX)"] = ydf[ten_label] - ydf[short_label]
    if ten_label in ydf.columns and five_label in ydf.columns:
        # not exactly 2s-10s but we can at least show 10s-5s if we lack true 2Y
        pass
    if thirty_label in ydf.columns and five_label in ydf.columns:
        slopes["30s-5s (TYX - FVX)"] = ydf[thirty_label] - ydf[five_label]
    if thirty_label in ydf.columns and ten_label in ydf.columns:
        slopes["30s-10s (TYX - TNX)"] = ydf[thirty_label] - ydf[ten_label]

    slopes = slopes.dropna(how="all")
    if not slopes.empty:
        fig_slopes = go.Figure()
        for col in slopes.columns:
            fig_slopes.add_trace(
                go.Scatter(
                    x=slopes.index,
                    y=slopes[col],
                    mode="lines",
                    name=col
                )
            )
        fig_slopes.add_hline(y=0, line_width=1, line_dash="dash", annotation_text="0 line")
        fig_slopes.update_layout(
            title="Yield Curve Slopes (Proxy Spreads)",
            xaxis_title="Date",
            yaxis_title="Index Level Difference",
            legend_title="Slope"
        )
        write_and_open_figure(fig_slopes, outdir, "02_rates_yield_slopes", open_html=open_html)
    else:
        logging.warning("No slopes computed; skipping slope chart.")

    # --- Term premium style ETF ratios: IEF/SHY, TLT/SHY ---
    etf_pairs = [
        ("IEF", "SHY", "IEF / SHY (Term Premium Proxy)"),
        ("TLT", "SHY", "TLT / SHY (Long Term Premium Proxy)"),
    ]

    term_df = pd.DataFrame()
    for num, den, label in etf_pairs:
        num_s = safe_load_price_series(num, start, end)
        den_s = safe_load_price_series(den, start, end)
        if num_s is None or den_s is None:
            continue
        aligned = pd.concat([num_s, den_s], axis=1, keys=[num, den]).dropna()
        if aligned.empty:
            continue
        term_df[label] = aligned[num] / aligned[den]

    if not term_df.empty:
        fig_term = go.Figure()
        for col in term_df.columns:
            fig_term.add_trace(
                go.Scatter(
                    x=term_df.index,
                    y=term_df[col],
                    mode="lines",
                    name=col
                )
            )
        fig_term.update_layout(
            title="Term Premium Proxies (ETF Ratios)",
            xaxis_title="Date",
            yaxis_title="Ratio",
            legend_title="Term Premium Proxies"
        )
        write_and_open_figure(fig_term, outdir, "03_rates_term_premium_proxies", open_html=open_html)
    else:
        logging.warning("No term premium ETF ratios computed; skipping that chart.")


# ======================================================================
# 2) LIQUIDITY PROXIES
# ======================================================================

def build_liquidity_panel(start: pd.Timestamp,
                          end: pd.Timestamp,
                          outdir: str,
                          open_html: bool) -> None:
    """
    Build liquidity proxy charts:
      - SPY vs BIL/SHV/SGOV normalized
      - SPY/BIL, SPY/SHV, SPY/SGOV ratios
      - DXY (^DXY) & UUP
      - ACWI / BIL
      - 60d z-score of log(Volume) for SPY, QQQ, IWM, HYG
    """
    ensure_dir(outdir)

    # --------------------------
    # A) SPY vs cash equivalents
    # --------------------------
    spy_df = safe_load_ohlcv("SPY")
    bil_df = safe_load_ohlcv("BIL")
    shv_df = safe_load_ohlcv("SHV")
    sgov_df = safe_load_ohlcv("SGOV")

    if spy_df is not None:
        if start is not None:
            spy_df = spy_df[spy_df.index >= start]
        if end is not None:
            spy_df = spy_df[spy_df.index <= end]

    # Build combined price frame
    price_cols = {}
    if spy_df is not None:
        price_cols["SPY"] = spy_df[detect_price_column(spy_df)]
    if bil_df is not None:
        if start is not None:
            bil_df = bil_df[bil_df.index >= start]
        if end is not None:
            bil_df = bil_df[bil_df.index <= end]
        price_cols["BIL"] = bil_df[detect_price_column(bil_df)]
    if shv_df is not None:
        if start is not None:
            shv_df = shv_df[shv_df.index >= start]
        if end is not None:
            shv_df = shv_df[shv_df.index <= end]
        price_cols["SHV"] = shv_df[detect_price_column(shv_df)]
    if sgov_df is not None:
        if start is not None:
            sgov_df = sgov_df[sgov_df.index >= start]
        if end is not None:
            sgov_df = sgov_df[sgov_df.index <= end]
        price_cols["SGOV"] = sgov_df[detect_price_column(sgov_df)]

    if price_cols:
        liq_df = pd.concat(price_cols.values(), axis=1)
        liq_df.columns = list(price_cols.keys())
        liq_df = liq_df.dropna(how="any", subset=["SPY"])  # ensure SPY is valid

        # Normalized
        for col in liq_df.columns:
            liq_df[col + "_norm"] = normalize_to_100(liq_df[col])

        fig_liq_norm = go.Figure()
        for base in ["SPY", "BIL", "SHV", "SGOV"]:
            col_norm = base + "_norm"
            if col_norm in liq_df.columns:
                fig_liq_norm.add_trace(
                    go.Scatter(
                        x=liq_df.index,
                        y=liq_df[col_norm],
                        mode="lines",
                        name=f"{base} (norm=100)"
                    )
                )
        fig_liq_norm.update_layout(
            title="Liquidity Proxies: SPY vs Cash-like ETFs (Normalized)",
            xaxis_title="Date",
            yaxis_title="Index (Start=100)",
            legend_title="Asset"
        )
        write_and_open_figure(fig_liq_norm, outdir, "01_liquidity_spy_vs_cash_normalized", open_html=open_html)

        # Ratios
        for base, cash_name in [("SPY", "BIL"), ("SPY", "SHV"), ("SPY", "SGOV")]:
            if base in liq_df.columns and cash_name in liq_df.columns:
                ratio = liq_df[base] / liq_df[cash_name]
                fig_ratio = go.Figure()
                fig_ratio.add_trace(
                    go.Scatter(
                        x=ratio.index,
                        y=ratio,
                        mode="lines",
                        name=f"{base}/{cash_name}"
                    )
                )
                fig_ratio.update_layout(
                    title=f"Liquidity Ratio: {base} / {cash_name}",
                    xaxis_title="Date",
                    yaxis_title="Ratio",
                    legend_title="Series"
                )
                write_and_open_figure(fig_ratio, outdir, f"02_liquidity_{base}_{cash_name}_ratio", open_html=open_html)
    else:
        logging.warning("No SPY or cash-equivalent ETFs loaded for liquidity panel A.")

    # --------------------------
    # B) Dollar liquidity & global ratios
    # --------------------------
    dxy = safe_load_price_series("^DXY", start, end)
    uup = safe_load_price_series("UUP", start, end)
    acwi = safe_load_price_series("ACWI", start, end)
    if bil_df is not None and "BIL" in price_cols:
        bil = price_cols["BIL"]
        acwi_bil = None
        if acwi is not None and not bil.empty:
            align = pd.concat([acwi, bil], axis=1, keys=["ACWI", "BIL"]).dropna()
            if not align.empty:
                acwi_bil = align["ACWI"] / align["BIL"]
    else:
        acwi_bil = None

    # Dollar index & UUP
    if dxy is not None or uup is not None:
        fig_dollar = go.Figure()
        if dxy is not None:
            fig_dollar.add_trace(
                go.Scatter(
                    x=dxy.index,
                    y=normalize_to_100(dxy),
                    mode="lines",
                    name="^DXY (norm=100)"
                )
            )
        if uup is not None:
            fig_dollar.add_trace(
                go.Scatter(
                    x=uup.index,
                    y=normalize_to_100(uup),
                    mode="lines",
                    name="UUP (norm=100)"
                )
            )
        fig_dollar.update_layout(
            title="Dollar Liquidity Proxies: DXY & UUP (Normalized)",
            xaxis_title="Date",
            yaxis_title="Index (Start=100)",
            legend_title="Series"
        )
        write_and_open_figure(fig_dollar, outdir, "03_liquidity_dollar_proxies", open_html=open_html)

    if acwi_bil is not None:
        fig_acwi_bil = go.Figure()
        fig_acwi_bil.add_trace(
            go.Scatter(
                x=acwi_bil.index,
                y=acwi_bil,
                mode="lines",
                name="ACWI / BIL"
            )
        )
        fig_acwi_bil.update_layout(
            title="Global Liquidity Proxy: ACWI / BIL",
            xaxis_title="Date",
            yaxis_title="Ratio",
            legend_title="Series"
        )
        write_and_open_figure(fig_acwi_bil, outdir, "04_liquidity_acwi_bil_ratio", open_html=open_html)

    # --------------------------
    # C) Volume z-score flows
    # --------------------------
    flow_tickers = ["SPY", "QQQ", "IWM", "HYG"]
    vol_frames = []
    for t in flow_tickers:
        df = safe_load_ohlcv(t)
        if df is None or "Volume" not in df.columns:
            continue
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        if df.empty:
            continue
        series = df["Volume"].astype(float)
        series.name = t
        vol_frames.append(series)

    if vol_frames:
        vdf = pd.concat(vol_frames, axis=1).dropna(how="all")
        log_vdf = np.log(vdf.replace(0, np.nan))
        zdf = log_vdf.copy()
        for col in log_vdf.columns:
            zdf[col] = rolling_zscore(log_vdf[col], window=60)

        fig_flows = go.Figure()
        for col in zdf.columns:
            fig_flows.add_trace(
                go.Scatter(
                    x=zdf.index,
                    y=zdf[col],
                    mode="lines",
                    name=f"{col} logVol z-score (60d)"
                )
            )
        fig_flows.add_hline(y=0, line_width=1, line_dash="dash", annotation_text="0")
        fig_flows.update_layout(
            title="ETF Flow Proxy: 60d Rolling z-score of log(Volume)",
            xaxis_title="Date",
            yaxis_title="Z-score",
            legend_title="Ticker"
        )
        write_and_open_figure(fig_flows, outdir, "05_liquidity_volume_zscore_flows", open_html=open_html)
    else:
        logging.warning("No volume data for SPY/QQQ/IWM/HYG; skipping flow z-score chart.")


# ======================================================================
# 3) CREDIT SPREAD PROXIES
# ======================================================================

def build_credit_panel(start: pd.Timestamp,
                       end: pd.Timestamp,
                       outdir: str,
                       open_html: bool) -> None:
    """
    Build credit spread proxy charts:
      - ETF levels: HYG, JNK, LQD, IEF, IEI, EMB, PCY, HYGH (best effort)
      - Normalized ETF curves
      - Log-ratio spread proxies (HY vs IG vs Treas vs EM)
    """
    ensure_dir(outdir)

    tickers = ["HYG", "JNK", "LQD", "IEF", "IEI", "EMB", "PCY", "HYGH"]
    price_data = {}
    for t in tickers:
        s = safe_load_price_series(t, start, end)
        if s is not None:
            price_data[t] = s

    if len(price_data) < 2:
        logging.error("Not enough credit-related ETFs loaded; skipping credit panel.")
        return

    df = pd.concat(price_data.values(), axis=1)
    df.columns = list(price_data.keys())
    df = df.dropna(how="all")

    # --- Normalized levels ---
    for col in df.columns:
        df[col + "_norm"] = normalize_to_100(df[col])

    fig_levels = go.Figure()
    for col in df.columns:
        if col.endswith("_norm"):
            base = col.replace("_norm", "")
            fig_levels.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=f"{base} (norm=100)"
                )
            )
    fig_levels.update_layout(
        title="Credit & Duration ETFs (Normalized Levels)",
        xaxis_title="Date",
        yaxis_title="Index (Start=100)",
        legend_title="ETF"
    )
    write_and_open_figure(fig_levels, outdir, "01_credit_etf_levels_normalized", open_html=open_html)

    # --- Spread proxies (log-ratios) ---
    spread_df = pd.DataFrame(index=df.index)

    def add_spread(label: str, num: str, den: str) -> None:
        nonlocal spread_df
        if num in df.columns and den in df.columns:
            spread_df[label] = np.log(df[num] / df[den])

    add_spread("HY_vs_Treasuries (log HYG/IEF)", "HYG", "IEF")
    add_spread("HY2_vs_Treasuries (log JNK/IEF)", "JNK", "IEF")
    add_spread("IG_vs_Treasuries (log LQD/IEF)", "LQD", "IEF")
    add_spread("HY_vs_IG (log HYG/LQD)", "HYG", "LQD")
    add_spread("HY2_vs_IG (log JNK/LQD)", "JNK", "LQD")
    add_spread("EMB_vs_Treasuries (log EMB/IEF)", "EMB", "IEF")
    add_spread("PCY_vs_Treasuries (log PCY/IEF)", "PCY", "IEF")
    add_spread("HY_vs_IEI (log HYG/IEI)", "HYG", "IEI")
    add_spread("IG_vs_IEI (log LQD/IEI)", "LQD", "IEI")

    spread_df = spread_df.dropna(how="all")
    if spread_df.empty:
        logging.warning("No credit log-ratio spreads computed; skipping spread chart.")
        return

    fig_spreads = go.Figure()
    for col in spread_df.columns:
        fig_spreads.add_trace(
            go.Scatter(
                x=spread_df.index,
                y=spread_df[col],
                mode="lines",
                name=col
            )
        )
    fig_spreads.update_layout(
        title="Credit Spread Proxies (log-ratios of ETF prices)",
        xaxis_title="Date",
        yaxis_title="log(Price Ratio)",
        legend_title="Spread Proxy"
    )
    write_and_open_figure(fig_spreads, outdir, "02_credit_spread_proxies_log_ratios", open_html=open_html)


# ======================================================================
# 4) BREADTH & STYLE / SIZE ROTATION
# ======================================================================

def build_breadth_rotation_panel(start: pd.Timestamp,
                                 end: pd.Timestamp,
                                 outdir: str,
                                 open_html: bool) -> None:
    """
    Build breadth and rotation proxies:
      - Breadth:
          * RSP / SPY
          * QQQ / QQQE
          * ACWI / RSP
          * VT / VTI
      - Style & size rotation:
          * SPY, QQQ, IWM normalized
      - Factor ETFs:
          * MTUM, VLUE, QUAL, SIZE, USMV normalized
    """
    ensure_dir(outdir)

    # --- Breadth: RSP / SPY ---
    rsp = safe_load_price_series("RSP", start, end)
    spy = safe_load_price_series("SPY", start, end)
    qqq = safe_load_price_series("QQQ", start, end)
    qqqe = safe_load_price_series("QQQE", start, end)
    acwi = safe_load_price_series("ACWI", start, end)
    vt = safe_load_price_series("VT", start, end)
    vti = safe_load_price_series("VTI", start, end)

    # RSP / SPY
    if rsp is not None and spy is not None:
        df_rsp_spy = pd.concat([rsp, spy], axis=1, keys=["RSP", "SPY"]).dropna()
        df_rsp_spy["RSP_SPY_ratio"] = df_rsp_spy["RSP"] / df_rsp_spy["SPY"]
        df_rsp_spy["RSP_SPY_ratio_norm"] = normalize_to_100(df_rsp_spy["RSP_SPY_ratio"])

        fig_rsp_spy = go.Figure()
        fig_rsp_spy.add_trace(
            go.Scatter(
                x=df_rsp_spy.index,
                y=df_rsp_spy["RSP_SPY_ratio_norm"],
                mode="lines",
                name="RSP/SPY (norm=100)"
            )
        )
        fig_rsp_spy.update_layout(
            title="Breadth Proxy: RSP / SPY Ratio (Normalized)",
            xaxis_title="Date",
            yaxis_title="Index (Start=100)",
            legend_title="Breadth Ratio"
        )
        write_and_open_figure(fig_rsp_spy, outdir, "01_breadth_rsp_spy_ratio_normalized", open_html=open_html)

    # QQQ / QQQE (concentration)
    if qqq is not None and qqqe is not None:
        df_qqq = pd.concat([qqq, qqqe], axis=1, keys=["QQQ", "QQQE"]).dropna()
        df_qqq["QQQ_QQQE_ratio"] = df_qqq["QQQ"] / df_qqq["QQQE"]
        df_qqq["QQQ_QQQE_ratio_norm"] = normalize_to_100(df_qqq["QQQ_QQQE_ratio"])

        fig_qqq_qqqe = go.Figure()
        fig_qqq_qqqe.add_trace(
            go.Scatter(
                x=df_qqq.index,
                y=df_qqq["QQQ_QQQE_ratio_norm"],
                mode="lines",
                name="QQQ/QQQE (norm=100)"
            )
        )
        fig_qqq_qqqe.update_layout(
            title="Concentration Proxy: QQQ / QQQE Ratio (Normalized)",
            xaxis_title="Date",
            yaxis_title="Index (Start=100)",
            legend_title="Concentration Ratio"
        )
        write_and_open_figure(fig_qqq_qqqe, outdir, "02_breadth_qqq_qqqe_ratio_normalized", open_html=open_html)

    # ACWI / RSP & VT / VTI
    if acwi is not None and rsp is not None:
        df_acwi_rsp = pd.concat([acwi, rsp], axis=1, keys=["ACWI", "RSP"]).dropna()
        df_acwi_rsp["ACWI_RSP_ratio"] = df_acwi_rsp["ACWI"] / df_acwi_rsp["RSP"]
        fig_acwi_rsp = go.Figure()
        fig_acwi_rsp.add_trace(
            go.Scatter(
                x=df_acwi_rsp.index,
                y=df_acwi_rsp["ACWI_RSP_ratio"],
                mode="lines",
                name="ACWI/RSP"
            )
        )
        fig_acwi_rsp.update_layout(
            title="Global vs Equal-Weight Breadth: ACWI / RSP",
            xaxis_title="Date",
            yaxis_title="Ratio",
            legend_title="Series"
        )
        write_and_open_figure(fig_acwi_rsp, outdir, "03_breadth_acwi_rsp_ratio", open_html=open_html)

    if vt is not None and vti is not None:
        df_vt_vti = pd.concat([vt, vti], axis=1, keys=["VT", "VTI"]).dropna()
        df_vt_vti["VT_VTI_ratio"] = df_vt_vti["VT"] / df_vt_vti["VTI"]
        fig_vt_vti = go.Figure()
        fig_vt_vti.add_trace(
            go.Scatter(
                x=df_vt_vti.index,
                y=df_vt_vti["VT_VTI_ratio"],
                mode="lines",
                name="VT/VTI"
            )
        )
        fig_vt_vti.update_layout(
            title="Global vs US: VT / VTI",
            xaxis_title="Date",
            yaxis_title="Ratio",
            legend_title="Series"
        )
        write_and_open_figure(fig_vt_vti, outdir, "04_breadth_vt_vti_ratio", open_html=open_html)

    # --- Style & size rotation: SPY, QQQ, IWM normalized ---
    rotation_tickers = ["SPY", "QQQ", "IWM"]
    rotation_data = {}
    for t in rotation_tickers:
        s = safe_load_price_series(t, start, end)
        if s is not None:
            rotation_data[t] = s

    if rotation_data:
        df_rot = pd.concat(rotation_data.values(), axis=1)
        df_rot.columns = list(rotation_data.keys())
        df_rot = df_rot.dropna()
        for col in df_rot.columns:
            df_rot[col + "_norm"] = normalize_to_100(df_rot[col])

        fig_rot = go.Figure()
        for base in rotation_tickers:
            col_norm = base + "_norm"
            if col_norm in df_rot.columns:
                fig_rot.add_trace(
                    go.Scatter(
                        x=df_rot.index,
                        y=df_rot[col_norm],
                        mode="lines",
                        name=f"{base} (norm=100)"
                    )
                )
        fig_rot.update_layout(
            title="Style / Size Rotation: SPY vs QQQ vs IWM (Normalized)",
            xaxis_title="Date",
            yaxis_title="Index (Start=100)",
            legend_title="ETF"
        )
        write_and_open_figure(fig_rot, outdir, "05_rotation_spy_qqq_iwm_normalized", open_html=open_html)

    # --- Factor ETFs: MTUM, VLUE, QUAL, SIZE, USMV ---
    factors = ["MTUM", "VLUE", "QUAL", "SIZE", "USMV"]
    factor_data = {}
    for f in factors:
        s = safe_load_price_series(f, start, end)
        if s is not None:
            factor_data[f] = s

    if factor_data:
        df_fac = pd.concat(factor_data.values(), axis=1)
        df_fac.columns = list(factor_data.keys())
        df_fac = df_fac.dropna()
        for col in df_fac.columns:
            df_fac[col + "_norm"] = normalize_to_100(df_fac[col])

        fig_fac = go.Figure()
        for col in df_fac.columns:
            if col.endswith("_norm"):
                base = col.replace("_norm", "")
                fig_fac.add_trace(
                    go.Scatter(
                        x=df_fac.index,
                        y=df_fac[col],
                        mode="lines",
                        name=f"{base} (norm=100)"
                    )
                )
        fig_fac.update_layout(
            title="Factor ETFs: MTUM / VLUE / QUAL / SIZE / USMV (Normalized)",
            xaxis_title="Date",
            yaxis_title="Index (Start=100)",
            legend_title="Factor ETF"
        )
        write_and_open_figure(fig_fac, outdir, "06_breadth_factor_etfs_normalized", open_html=open_html)


# ======================================================================
# 5) INFLATION & GROWTH PROXIES (COMMODITIES & TIPS)
# ======================================================================

def build_inflation_growth_panel(start: pd.Timestamp,
                                 end: pd.Timestamp,
                                 outdir: str,
                                 open_html: bool) -> None:
    """
    Build inflation & growth proxy charts:
      - Commodities & cyclicals: DBC, USO, GLD, SLV, COPX, XLE, XLB
      - Ratios: GLD/SLV, DBC/SPY, XLE/SPY, COPX/SPY
      - TIPS breakeven proxies: TIP/IEF, TIP/SHY
    """
    ensure_dir(outdir)

    # --- Commodities & cyclical sectors ---
    tickers = ["DBC", "USO", "GLD", "SLV", "COPX", "XLE", "XLB", "SPY"]
    com_data = {}
    for t in tickers:
        s = safe_load_price_series(t, start, end)
        if s is not None:
            com_data[t] = s

    if com_data:
        df_com = pd.concat(com_data.values(), axis=1)
        df_com.columns = list(com_data.keys())
        df_com = df_com.dropna(how="all")
        for col in df_com.columns:
            df_com[col + "_norm"] = normalize_to_100(df_com[col])

        # Normalized chart
        fig_com = go.Figure()
        for base in ["DBC", "USO", "GLD", "SLV", "COPX", "XLE", "XLB", "SPY"]:
            col_norm = base + "_norm"
            if col_norm in df_com.columns:
                fig_com.add_trace(
                    go.Scatter(
                        x=df_com.index,
                        y=df_com[col_norm],
                        mode="lines",
                        name=f"{base} (norm=100)"
                    )
                )
        fig_com.update_layout(
            title="Inflation & Growth Proxies: Commodities & Cyclicals (Normalized)",
            xaxis_title="Date",
            yaxis_title="Index (Start=100)",
            legend_title="ETF"
        )
        write_and_open_figure(fig_com, outdir, "01_inflation_commodities_normalized", open_html=open_html)

        # Ratios vs GLD, SPY
        def ratio_chart(num: str, den: str, label: str, basename: str) -> None:
            if num in df_com.columns and den in df_com.columns:
                r = df_com[num] / df_com[den]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=r.index,
                        y=r,
                        mode="lines",
                        name=label
                    )
                )
                fig.update_layout(
                    title=label,
                    xaxis_title="Date",
                    yaxis_title="Ratio",
                    legend_title="Series"
                )
                write_and_open_figure(fig, outdir, basename, open_html=open_html)

        ratio_chart("GLD", "SLV", "Precious Spread: GLD / SLV", "02_inflation_gld_slv_ratio")
        ratio_chart("DBC", "SPY", "Commodity vs Equities: DBC / SPY", "03_inflation_dbc_spy_ratio")
        ratio_chart("XLE", "SPY", "Energy vs Equities: XLE / SPY", "04_inflation_xle_spy_ratio")
        ratio_chart("COPX", "SPY", "Copper Miners vs SPY: COPX / SPY", "05_inflation_copx_spy_ratio")
    else:
        logging.warning("No commodity / cyclicals data loaded; skipping inflation panel A.")

    # --- TIPS breakeven proxies: TIP / IEF, TIP / SHY ---
    tip = safe_load_price_series("TIP", start, end)
    ief = safe_load_price_series("IEF", start, end)
    shy = safe_load_price_series("SHY", start, end)

    be_df = pd.DataFrame()
    if tip is not None and ief is not None:
        tmp = pd.concat([tip, ief], axis=1, keys=["TIP", "IEF"]).dropna()
        if not tmp.empty:
            be_df["TIP_IEF"] = tmp["TIP"] / tmp["IEF"]
    if tip is not None and shy is not None:
        tmp = pd.concat([tip, shy], axis=1, keys=["TIP", "SHY"]).dropna()
        if not tmp.empty:
            be_df["TIP_SHY"] = tmp["TIP"] / tmp["SHY"]

    if not be_df.empty:
        fig_be = go.Figure()
        for col in be_df.columns:
            fig_be.add_trace(
                go.Scatter(
                    x=be_df.index,
                    y=be_df[col],
                    mode="lines",
                    name=col.replace("_", " / ")
                )
            )
        fig_be.update_layout(
            title="Inflation Breakeven Proxies: TIP / IEF and TIP / SHY",
            xaxis_title="Date",
            yaxis_title="Ratio",
            legend_title="Breakeven Proxy"
        )
        write_and_open_figure(fig_be, outdir, "06_inflation_tip_breakeven_proxies", open_html=open_html)
    else:
        logging.warning("No TIP/IEF/SHY breakeven data; skipping that chart.")


# ======================================================================
# 6) RISK APPETITE & VOLATILITY
# ======================================================================

def build_risk_appetite_panel(start: pd.Timestamp,
                              end: pd.Timestamp,
                              outdir: str,
                              open_html: bool) -> None:
    """
    Build risk appetite & volatility charts:
      - VIX complex: ^VIX, ^VIX9D, ^VVIX (best effort)
      - Cross-asset risk appetite:
          * ARKK / QQQ, ARKK / SPY, BTC-USD / GLD, SMH / SPY
      - SPY 21d realized vol vs VIX index
    """
    ensure_dir(outdir)

    # --- VIX complex ---
    vix = safe_load_price_series("^VIX", start, end)
    vix9d = safe_load_price_series("^VIX9D", start, end)
    vvix = safe_load_price_series("^VVIX", start, end)

    if vix is not None or vix9d is not None or vvix is not None:
        fig_vix = go.Figure()
        if vix is not None:
            fig_vix.add_trace(
                go.Scatter(
                    x=vix.index,
                    y=vix,
                    mode="lines",
                    name="VIX"
                )
            )
        if vix9d is not None:
            fig_vix.add_trace(
                go.Scatter(
                    x=vix9d.index,
                    y=vix9d,
                    mode="lines",
                    name="VIX9D"
                )
            )
        if vvix is not None:
            fig_vix.add_trace(
                go.Scatter(
                    x=vvix.index,
                    y=vvix,
                    mode="lines",
                    name="VVIX"
                )
            )
        fig_vix.update_layout(
            title="Volatility Indices: VIX / VIX9D / VVIX",
            xaxis_title="Date",
            yaxis_title="Index Level",
            legend_title="Vol Index"
        )
        write_and_open_figure(fig_vix, outdir, "01_risk_vix_complex", open_html=open_html)
    else:
        logging.warning("No VIX complex indices loaded; skipping VIX chart.")

    # --- Cross-asset risk appetite ratios ---
    arkk = safe_load_price_series("ARKK", start, end)
    qqq = safe_load_price_series("QQQ", start, end)
    spy = safe_load_price_series("SPY", start, end)
    btc = safe_load_price_series("BTC-USD", start, end)
    gld = safe_load_price_series("GLD", start, end)
    smh = safe_load_price_series("SMH", start, end)

    def ratio_series(num: pd.Series | None, den: pd.Series | None) -> pd.Series | None:
        if num is None or den is None:
            return None
        df = pd.concat([num, den], axis=1, keys=["num", "den"]).dropna()
        if df.empty:
            return None
        return df["num"] / df["den"]

    arkk_qqq = ratio_series(arkk, qqq)
    arkk_spy = ratio_series(arkk, spy)
    btc_gld = ratio_series(btc, gld)
    smh_spy = ratio_series(smh, spy)

    # Plot cross-asset appetite ratios combined
    appetite_df = pd.DataFrame()
    if arkk_qqq is not None:
        appetite_df["ARKK/QQQ"] = arkk_qqq
    if arkk_spy is not None:
        appetite_df["ARKK/SPY"] = arkk_spy
    if btc_gld is not None:
        appetite_df["BTC-USD/GLD"] = btc_gld
    if smh_spy is not None:
        appetite_df["SMH/SPY"] = smh_spy

    if not appetite_df.empty:
        fig_appetite = go.Figure()
        for col in appetite_df.columns:
            fig_appetite.add_trace(
                go.Scatter(
                    x=appetite_df.index,
                    y=appetite_df[col],
                    mode="lines",
                    name=col
                )
            )
        fig_appetite.update_layout(
            title="Cross-Asset Risk Appetite Ratios",
            xaxis_title="Date",
            yaxis_title="Ratio",
            legend_title="Series"
        )
        write_and_open_figure(fig_appetite, outdir, "02_risk_appetite_cross_asset_ratios", open_html=open_html)
    else:
        logging.warning("No cross-asset risk appetite ratios; skipping that chart.")

    # --- SPY realized vol vs VIX ---
    spy_df = safe_load_ohlcv("SPY")
    if spy_df is not None:
        if start is not None:
            spy_df = spy_df[spy_df.index >= start]
        if end is not None:
            spy_df = spy_df[spy_df.index <= end]
        if not spy_df.empty:
            price_col = detect_price_column(spy_df)
            px = spy_df[price_col].astype(float)
            rets = np.log(px / px.shift(1)).dropna()
            rv_21 = ann_realized_vol(rets, window=21)

            rv_vix_df = pd.DataFrame(index=rv_21.index)
            rv_vix_df["RealizedVol_21d"] = rv_21

            # Align VIX to RV dates
            if vix is not None:
                rv_vix_df = rv_vix_df.join(vix.rename("VIX"), how="left")

            rv_vix_df = rv_vix_df.dropna(how="all")
            if not rv_vix_df.empty:
                fig_rv_vix = go.Figure()
                fig_rv_vix.add_trace(
                    go.Scatter(
                        x=rv_vix_df.index,
                        y=rv_vix_df["RealizedVol_21d"],
                        mode="lines",
                        name="SPY 21d Realized Vol (ann.)"
                    )
                )
                if "VIX" in rv_vix_df.columns and not rv_vix_df["VIX"].isna().all():
                    fig_rv_vix.add_trace(
                        go.Scatter(
                            x=rv_vix_df.index,
                            y=rv_vix_df["VIX"],
                            mode="lines",
                            name="VIX"
                        )
                    )
                fig_rv_vix.update_layout(
                    title="Volatility Risk Premium Proxy: SPY Realized Vol vs VIX",
                    xaxis_title="Date",
                    yaxis_title="Volatility / Index",
                    legend_title="Series"
                )
                write_and_open_figure(fig_rv_vix, outdir, "03_risk_realized_vs_vix", open_html=open_html)
            else:
                logging.warning("RV/VIX frame empty; skipping realized vs VIX chart.")
        else:
            logging.warning("SPY data empty for realized vol; skipping realized vs VIX chart.")
    else:
        logging.warning("SPY OHLCV not loaded; skipping realized vs VIX chart.")


# ======================================================================
# 7) GROWTH vs VALUE / THEMATIC
# ======================================================================

def build_growth_value_panel(start: pd.Timestamp,
                             end: pd.Timestamp,
                             outdir: str,
                             open_html: bool) -> None:
    """
    Build growth vs value / sector rotation charts:
      - QQQ / IWM
      - QQQ / XLF
      - QQQ / XLE
      - ARKK / SPY
      - IGV / SPY
    """
    ensure_dir(outdir)

    qqq = safe_load_price_series("QQQ", start, end)
    iwm = safe_load_price_series("IWM", start, end)
    xlf = safe_load_price_series("XLF", start, end)
    xle = safe_load_price_series("XLE", start, end)
    arkk = safe_load_price_series("ARKK", start, end)
    spy = safe_load_price_series("SPY", start, end)
    igv = safe_load_price_series("IGV", start, end)

    def ratio(num: pd.Series | None, den: pd.Series | None, label: str) -> tuple[str, pd.Series] | None:
        if num is None or den is None:
            return None
        df = pd.concat([num, den], axis=1, keys=["num", "den"]).dropna()
        if df.empty:
            return None
        return label, df["num"] / df["den"]

    ratios = []
    r1 = ratio(qqq, iwm, "QQQ/IWM")
    if r1 is not None:
        ratios.append(r1)
    r2 = ratio(qqq, xlf, "QQQ/XLF")
    if r2 is not None:
        ratios.append(r2)
    r3 = ratio(qqq, xle, "QQQ/XLE")
    if r3 is not None:
        ratios.append(r3)
    r4 = ratio(arkk, spy, "ARKK/SPY")
    if r4 is not None:
        ratios.append(r4)
    r5 = ratio(igv, spy, "IGV/SPY")
    if r5 is not None:
        ratios.append(r5)

    if not ratios:
        logging.warning("No growth vs value / thematic ratios computed; skipping this panel.")
        return

    # Build combined DataFrame
    gv_df = pd.DataFrame()
    for label, series in ratios:
        gv_df[label] = series

    fig_gv = go.Figure()
    for col in gv_df.columns:
        fig_gv.add_trace(
            go.Scatter(
                x=gv_df.index,
                y=gv_df[col],
                mode="lines",
                name=col
            )
        )
    fig_gv.update_layout(
        title="Growth vs Value / Thematic Ratios",
        xaxis_title="Date",
        yaxis_title="Ratio",
        legend_title="Series"
    )
    write_and_open_figure(fig_gv, outdir, "01_growth_value_thematic_ratios", open_html=open_html)


# ======================================================================
# MAIN
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Macro proxies dashboard v2: rates, liquidity, credit, breadth, "
                    "inflation, risk appetite, growth/value using data_retrieval.py "
                    "and Plotly HTML outputs in /dev/shm."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date (YYYY-MM-DD). Default: 2015-01-01",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/dev/shm/MACRO_PROXIES_DASHBOARD",
        help="Root output directory (default: /dev/shm/MACRO_PROXIES_DASHBOARD)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="If set, do NOT auto-open HTML files in the browser.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    # Parse dates
    try:
        start_ts = pd.to_datetime(args.start) if args.start else None
    except Exception:
        logging.error(f"Invalid --start date: {args.start}")
        sys.exit(1)

    if args.end:
        try:
            end_ts = pd.to_datetime(args.end)
        except Exception:
            logging.error(f"Invalid --end date: {args.end}")
            sys.exit(1)
    else:
        end_ts = pd.to_datetime(date.today())

    open_html = not args.no_open

    root_out = os.path.abspath(args.output_root)
    ensure_dir(root_out)
    logging.info(f"Output root directory: {root_out}")
    logging.info(f"Date range: {start_ts.date() if start_ts is not None else 'None'} -> {end_ts.date()}")

    # Subdirectories per "panel family" for organization
    rates_outdir = os.path.join(root_out, "RATES")
    liq_outdir = os.path.join(root_out, "LIQUIDITY")
    credit_outdir = os.path.join(root_out, "CREDIT")
    breadth_outdir = os.path.join(root_out, "BREADTH_ROTATION")
    infl_outdir = os.path.join(root_out, "INFLATION_GROWTH")
    risk_outdir = os.path.join(root_out, "RISK_APPETITE")
    gv_outdir = os.path.join(root_out, "GROWTH_VALUE")

    # Build all panels
    build_rates_panel(start_ts, end_ts, rates_outdir, open_html=open_html)
    build_liquidity_panel(start_ts, end_ts, liq_outdir, open_html=open_html)
    build_credit_panel(start_ts, end_ts, credit_outdir, open_html=open_html)
    build_breadth_rotation_panel(start_ts, end_ts, breadth_outdir, open_html=open_html)
    build_inflation_growth_panel(start_ts, end_ts, infl_outdir, open_html=open_html)
    build_risk_appetite_panel(start_ts, end_ts, risk_outdir, open_html=open_html)
    build_growth_value_panel(start_ts, end_ts, gv_outdir, open_html=open_html)

    logging.info("Macro proxies dashboard v2 construction complete.")


if __name__ == "__main__":
    main()

