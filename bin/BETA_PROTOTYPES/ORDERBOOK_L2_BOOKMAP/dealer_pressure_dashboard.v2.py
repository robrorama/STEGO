#!/usr/bin/env python3
# SCRIPTNAME: dealer_pressure_dashboard.v2.py
# AUTHOR: Michael Derby
# DATE:   2025-11-26
#
# DEALER PRESSURE DASHBOARD (GEX + VANNA + OI STRUCTURE)
# ------------------------------------------------------
# - Uses your existing data_retrieval.py and options_data_retrieval.py as-is.
# - Pulls underlying via data_retrieval.load_or_download_ticker().
# - Uses options_data_retrieval to:
#       * get_available_remote_expirations()
#       * ensure_option_chains_cached()
#       * load_all_cached_option_chains() for a front set of expiries.
# - Computes:
#       * Per-strike GEX and cumulative GEX with flip zones
#       * A practical vanna proxy from Black–Scholes vega and moneyness
#       * OI and (placeholder) ΔOI heatmaps
#       * IV term structure (9d / 30d / 3m buckets) from OI-weighted IV
#       * A composite dealer-pressure score by strike
# - Writes ALL outputs (HTML + CSV) to:
#       /dev/shm/DEALER_PRESSURE_DASHBOARD/<TICKER>/
# - Opens each Plotly figure in a separate browser tab.

import os
import sys
import logging
import datetime
from math import log, sqrt, exp, pi

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import data_retrieval
import options_data_retrieval


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ---------------------------------------------------------------------------
# Black–Scholes helper utilities (for greeks & vanna proxy)
# ---------------------------------------------------------------------------

SQRT_2PI = sqrt(2.0 * pi)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / SQRT_2PI


def bs_d1_d2(S, K, T, r, q, sigma):
    """
    Vectorized Black–Scholes d1, d2.

    Inputs:
        S, K, T, r, q, sigma – numpy arrays or scalars
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = float(r)
    q = float(q)

    out_shape = np.broadcast(S, K, T, sigma).shape
    d1 = np.full(out_shape, np.nan, dtype=float)
    d2 = np.full(out_shape, np.nan, dtype=float)

    valid = (S > 0) & (K > 0) & (T > 0) & (sigma > 0)
    if not np.any(valid):
        return d1, d2

    Sv = S[valid]
    Kv = K[valid]
    Tv = T[valid]
    sigv = sigma[valid]

    with np.errstate(divide="ignore", invalid="ignore"):
        d1v = (np.log(Sv / Kv) + (r - q + 0.5 * sigv * sigv) * Tv) / (sigv * np.sqrt(Tv))
        d2v = d1v - sigv * np.sqrt(Tv)

    d1[valid] = d1v
    d2[valid] = d2v
    return d1, d2


def bs_gamma(S, K, T, r, q, sigma):
    """
    Black–Scholes gamma (same for calls and puts).
    """
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = float(q)

    out = np.zeros_like(d1)
    valid = (S > 0) & (T > 0) & (sigma > 0)
    if np.any(valid):
        Sv = S[valid]
        Tv = T[valid]
        sigv = sigma[valid]
        d1v = d1[valid]
        out[valid] = (np.exp(-q * Tv) * _norm_pdf(d1v)) / (Sv * sigv * np.sqrt(Tv))
    return out


def bs_vega(S, K, T, r, q, sigma):
    """
    Black–Scholes vega (per 1.0 change in vol, not per 1%).
    """
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = float(q)

    out = np.zeros_like(d1)
    valid = (S > 0) & (T > 0) & (sigma > 0)
    if np.any(valid):
        Sv = S[valid]
        Tv = T[valid]
        d1v = d1[valid]
        out[valid] = Sv * np.exp(-q * Tv) * _norm_pdf(d1v) * np.sqrt(Tv)
    return out


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

def compute_gex(df: pd.DataFrame, spot: float) -> tuple:
    """
    Compute per-strike GEX and cumulative GEX with flip zones.

    df must have:
        - 'strike'
        - 'type'  ("call" / "put")
        - 'gamma'
        - 'oi'
    """
    work = df.copy()
    work["signed_gamma"] = np.where(work["type"] == "call", 1.0, -1.0) * work["gamma"] * work["oi"] * 100.0
    gex_by_strike = work.groupby("strike")["signed_gamma"].sum().sort_index()
    cumulative = gex_by_strike.cumsum()
    flips = cumulative[(cumulative.shift(1) * cumulative < 0)]
    return gex_by_strike, cumulative, flips


def compute_vanna_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot vanna exposure to (strike x expiration) grid.

    df must have:
        - 'strike'
        - 'expiration'
        - 'vanna'
        - 'oi'
    """
    work = df.copy()
    work["vanna_exposure"] = work["vanna"] * work["oi"] * 100.0
    grid = work.pivot_table(
        index="strike",
        columns="expiration",
        values="vanna_exposure",
        aggfunc="sum"
    ).sort_index(axis=0)
    return grid


def compute_oi_grids(df: pd.DataFrame) -> tuple:
    """
    Return OI and ΔOI pivoted as (strike x expiration) heatmaps.

    df must have:
        - 'strike'
        - 'expiration'
        - 'oi'
        - 'oi_change'
    """
    work = df.copy()
    oi_grid = work.pivot_table(
        index="strike",
        columns="expiration",
        values="oi",
        aggfunc="sum"
    )
    doi_grid = work.pivot_table(
        index="strike",
        columns="expiration",
        values="oi_change",
        aggfunc="sum"
    )
    return oi_grid, doi_grid


def compute_term_structure_from_chain(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Build a simple IV term structure (9d / 30d / 3m) from OI-weighted IV per expiration.

    df must have:
        - 'expiration' (Timestamp)
        - 'iv'
        - 'oi'
    """
    work = df.copy()
    work["dte_days"] = (work["expiration"] - asof.normalize()).dt.days
    work = work[work["dte_days"] > 0]

    if work.empty:
        return pd.DataFrame({"tenor": ["9d", "30d", "3m"], "iv": [np.nan, np.nan, np.nan]})

    grouped = work.groupby("expiration").apply(
        lambda g: pd.Series({
            "dte": (g["expiration"].iloc[0] - asof.normalize()).days,
            "iv_oi_weighted": np.average(g["iv"], weights=np.maximum(g["oi"].values, 1.0))
            if np.any(g["oi"].values > 0) else g["iv"].mean()
        })
    )

    def _bucket_iv(min_dte, max_dte):
        sel = grouped[(grouped["dte"] >= min_dte) & (grouped["dte"] <= max_dte)]
        if sel.empty:
            return np.nan
        return sel["iv_oi_weighted"].mean()

    iv_9d = _bucket_iv(1, 9)
    iv_30d = _bucket_iv(10, 30)
    iv_3m = _bucket_iv(31, 90)

    ts = pd.DataFrame({
        "tenor": ["9d", "30d", "3m"],
        "iv": [iv_9d, iv_30d, iv_3m]
    })
    return ts


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(arr)
    return (arr - m) / s


def compute_composite_score(
    strikes: np.ndarray,
    cumulative_gex: pd.Series,
    flips: pd.Series,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a composite dealer-pressure score per strike.

    Ingredients:
        - GEX slope
        - Flip proximity indicator
        - Aggregated vanna per strike
        - Aggregated OI (used as ΔOI proxy)
    """
    strikes = np.asarray(strikes, dtype=float)

    # GEX slope
    gex_values = cumulative_gex.reindex(strikes, method="nearest").values
    gex_slope = np.gradient(gex_values) if strikes.size > 1 else np.zeros_like(gex_values)
    gex_slope_z = normalize(gex_slope)

    # Flip risk: 1 if that strike is near a flip zone
    flip_strikes = np.asarray(flips.index.values, dtype=float) if flips is not None and len(flips) > 0 else np.array([])
    flip_risk = np.zeros_like(strikes, dtype=float)
    if flip_strikes.size > 0:
        for i, k in enumerate(strikes):
            # nearest flip within 1% of strike
            if np.any(np.isfinite(flip_strikes)):
                dist = np.min(np.abs(flip_strikes - k))
                if dist <= 0.01 * k:
                    flip_risk[i] = 1.0
    flip_risk_z = normalize(flip_risk)

    # Vanna aggregated per strike
    vanna_per_strike = df.groupby("strike")["vanna"].sum().reindex(strikes).fillna(0.0).values
    vanna_z = normalize(vanna_per_strike)

    # OI aggregated per strike – used as proxy ΔOI intensity
    doi_proxy = df.groupby("strike")["oi"].sum().reindex(strikes).fillna(0.0).values
    doi_z = normalize(doi_proxy)

    # Weights
    w1, w2, w3, w4 = 1.0, 2.0, 1.0, 1.0
    composite = w1 * gex_slope_z + w2 * flip_risk_z + w3 * vanna_z + w4 * doi_z

    out = pd.DataFrame({
        "strike": strikes,
        "gex_slope_z": gex_slope_z,
        "flip_risk_z": flip_risk_z,
        "vanna_z": vanna_z,
        "doi_z": doi_z,
        "composite_score": composite,
    })
    return out


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def build_gex_plot(cumulative: pd.Series, flips: pd.Series, ticker: str, outdir: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative.index.values,
        y=cumulative.values,
        mode="lines",
        name="Cumulative GEX"
    ))
    if flips is not None and len(flips) > 0:
        fig.add_trace(go.Scatter(
            x=flips.index.values,
            y=flips.values,
            mode="markers",
            name="Flip zones",
            marker=dict(size=9)
        ))

    fig.update_layout(
        title=f"{ticker} – Cumulative Dealer GEX (by Strike)",
        xaxis_title="Strike",
        yaxis_title="Cumulative GEX (arb units)"
    )
    html_path = os.path.join(outdir, "gex.html")
    pio.write_html(fig, html_path, auto_open=True)


def build_vanna_plot(vanna_grid: pd.DataFrame, ticker: str, outdir: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=vanna_grid.values,
            x=[str(x.date()) if hasattr(x, "date") else str(x) for x in vanna_grid.columns],
            y=vanna_grid.index.values
        )
    )
    fig.update_layout(
        title=f"{ticker} – Vanna Exposure Heatmap (strike × expiry)",
        xaxis_title="Expiration",
        yaxis_title="Strike"
    )
    html_path = os.path.join(outdir, "vanna_heatmap.html")
    pio.write_html(fig, html_path, auto_open=True)


def build_oi_plots(oi_grid: pd.DataFrame, doi_grid: pd.DataFrame, ticker: str, outdir: str):
    fig1 = go.Figure(
        data=go.Heatmap(
            z=oi_grid.values,
            x=[str(x.date()) if hasattr(x, "date") else str(x) for x in oi_grid.columns],
            y=oi_grid.index.values
        )
    )
    fig1.update_layout(
        title=f"{ticker} – Open Interest Heatmap",
        xaxis_title="Expiration",
        yaxis_title="Strike"
    )
    p1 = os.path.join(outdir, "oi_heatmap.html")
    pio.write_html(fig1, p1, auto_open=True)

    fig2 = go.Figure(
        data=go.Heatmap(
            z=doi_grid.values,
            x=[str(x.date()) if hasattr(x, "date") else str(x) for x in doi_grid.columns],
            y=doi_grid.index.values
        )
    )
    fig2.update_layout(
        title=f"{ticker} – ΔOI Heatmap (placeholder ΔOI from OI proxy)",
        xaxis_title="Expiration",
        yaxis_title="Strike"
    )
    p2 = os.path.join(outdir, "doi_heatmap.html")
    pio.write_html(fig2, p2, auto_open=True)


def build_term_structure_plot(ts: pd.DataFrame, ticker: str, outdir: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["tenor"].values,
        y=ts["iv"].values,
        mode="lines+markers",
        name="IV"
    ))
    fig.update_layout(
        title=f"{ticker} – IV Term Structure (OI-weighted buckets)",
        xaxis_title="Tenor",
        yaxis_title="Implied Volatility"
    )
    html_path = os.path.join(outdir, "term_structure.html")
    pio.write_html(fig, html_path, auto_open=True)


def build_composite_plot(comp_df: pd.DataFrame, ticker: str, outdir: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comp_df["strike"].values,
        y=comp_df["composite_score"].values,
        mode="lines+markers",
        name="Composite score"
    ))
    fig.update_layout(
        title=f"{ticker} – Composite Dealer-Pressure Score (by Strike)",
        xaxis_title="Strike",
        yaxis_title="Score (z-scored components)"
    )
    html_path = os.path.join(outdir, "composite_score.html")
    pio.write_html(fig, html_path, auto_open=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dealer_pressure_dashboard.v2.py TICKER")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    # Output directory under /dev/shm
    outdir = f"/dev/shm/DEALER_PRESSURE_DASHBOARD/{ticker}"
    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Output directory: {outdir}")

    # Underlying spot via your canonical loader
    logging.info(f"Loading underlying via data_retrieval: {ticker}")
    ohlcv = data_retrieval.load_or_download_ticker(ticker)
    if ohlcv is None or ohlcv.empty:
        logging.error(f"No OHLCV data for {ticker}.")
        sys.exit(1)

    spot = float(ohlcv["Close"].iloc[-1])
    asof = pd.to_datetime(ohlcv.index[-1]).normalize()
    logging.info(f"Spot price: {spot:.4f} (as of {asof.date()})")

    # Options expirations: use options_data_retrieval public API
    logging.info("Fetching available remote expirations via options_data_retrieval.get_available_remote_expirations()...")
    remote_exps = options_data_retrieval.get_available_remote_expirations(ticker, source="yfinance")
    if not remote_exps:
        logging.error(f"No remote expirations available for {ticker}.")
        sys.exit(1)

    # Choose front N expirations
    front_n = 8
    selected_exps = remote_exps[:front_n]
    logging.info(f"Selected front {len(selected_exps)} expirations for analysis: {[str(e.date()) for e in selected_exps]}")

    # Ensure they are cached to disk (uses load_or_download_option_chain() internally)
    logging.info("Ensuring selected option chains are cached...")
    options_data_retrieval.ensure_option_chains_cached(
        ticker=ticker,
        expirations=selected_exps,
        source="yfinance",
        force_refresh=False
    )

    # Load all cached chains for selected expirations
    logging.info("Loading cached option chains via load_all_cached_option_chains()...")
    chain_df = options_data_retrieval.load_all_cached_option_chains(
        ticker=ticker,
        source="yfinance",
        expirations=selected_exps
    )

    if chain_df is None or chain_df.empty:
        logging.error("Option chain DataFrame is empty after load_all_cached_option_chains().")
        sys.exit(1)

    # Standardize & compute greeks and vanna proxy
    df = chain_df.copy()

    # Required base columns from yfinance: 'strike', 'type', 'expiration', 'openInterest', 'impliedVolatility'
    required_base = ["strike", "type", "expiration", "openInterest"]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        logging.error(f"Missing required columns in options chain: {missing}")
        sys.exit(1)

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df["oi"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0.0)

    if "impliedVolatility" in df.columns:
        df["iv"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    else:
        logging.warning("No impliedVolatility column; setting IV to NaN.")
        df["iv"] = np.nan

    # Time to expiry in years, based on underlying asof date
    df["T"] = (df["expiration"] - asof).dt.days / 365.0
    df = df[df["T"] > 0.0]

    # Drop rows with invalid core fields
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["strike", "T", "iv", "oi", "type"])
    if df.empty:
        logging.error("No valid option rows after basic cleaning (strike/T/iv/oi/type).")
        sys.exit(1)

    # Greeks (r, q set to 0 for now; can be wired to macro later)
    r = 0.0
    q = 0.0
    S_arr = np.full(df.shape[0], spot, dtype=float)
    K_arr = df["strike"].values.astype(float)
    T_arr = df["T"].values.astype(float)
    iv_arr = df["iv"].values.astype(float)

    logging.info("Computing gamma and vega greeks...")
    gamma_arr = bs_gamma(S_arr, K_arr, T_arr, r, q, iv_arr)
    vega_arr = bs_vega(S_arr, K_arr, T_arr, r, q, iv_arr)

    df["gamma"] = gamma_arr
    df["vega"] = vega_arr

    # Vanna proxy: use moneyness * vega (captures sign/focus across strikes)
    df["vanna"] = ((df["strike"] - spot) / max(spot, 1e-8)) * df["vega"]

    # Placeholder ΔOI (no historical OI series in the loader; keep as zero but keep pipeline intact)
    df["oi_change"] = 0.0

    # Final clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["gamma", "vega", "vanna"])
    if df.empty:
        logging.error("No valid rows after greek/vanna computation.")
        sys.exit(1)

    # Save full enriched chain snapshot for STEGO pipelines
    chain_csv_path = os.path.join(outdir, f"{ticker}_options_enriched.csv")
    df.to_csv(chain_csv_path, index=False)
    logging.info(f"Saved enriched options snapshot to {chain_csv_path}")

    # -----------------------------------------------------------------------
    # Dealer-flow analytics
    # -----------------------------------------------------------------------
    logging.info("Computing GEX and cumulative GEX...")
    gex_by_strike, cumulative_gex, flips = compute_gex(df, spot)

    logging.info("Computing vanna exposure grid...")
    vanna_grid = compute_vanna_grid(df)

    logging.info("Computing OI and ΔOI grids...")
    oi_grid, doi_grid = compute_oi_grids(df)

    logging.info("Computing IV term structure from OI-weighted expiries...")
    ts = compute_term_structure_from_chain(df, asof=asof)

    # Composite score per strike
    strikes_sorted = cumulative_gex.index.values.astype(float)
    logging.info("Computing composite dealer-pressure score...")
    comp_df = compute_composite_score(strikes_sorted, cumulative_gex, flips, df)

    # -----------------------------------------------------------------------
    # Persist numeric outputs as CSVs (for STEGO pipelines)
    # -----------------------------------------------------------------------
    gex_by_strike.to_csv(os.path.join(outdir, f"{ticker}_gex_by_strike.csv"))
    cumulative_gex.to_csv(os.path.join(outdir, f"{ticker}_gex_cumulative.csv"))
    vanna_grid.to_csv(os.path.join(outdir, f"{ticker}_vanna_grid.csv"))
    oi_grid.to_csv(os.path.join(outdir, f"{ticker}_oi_grid.csv"))
    doi_grid.to_csv(os.path.join(outdir, f"{ticker}_doi_grid.csv"))
    ts.to_csv(os.path.join(outdir, f"{ticker}_iv_term_structure.csv"), index=False)
    comp_df.to_csv(os.path.join(outdir, f"{ticker}_composite_score.csv"), index=False)

    logging.info("Saved all CSV outputs for STEGO pipelines.")

    # -----------------------------------------------------------------------
    # Build Plotly dashboards (auto-open each in a new browser tab)
    # -----------------------------------------------------------------------
    logging.info("Building GEX plot...")
    build_gex_plot(cumulative_gex, flips, ticker, outdir)

    logging.info("Building vanna heatmap...")
    build_vanna_plot(vanna_grid, ticker, outdir)

    logging.info("Building OI and ΔOI heatmaps...")
    build_oi_plots(oi_grid, doi_grid, ticker, outdir)

    logging.info("Building IV term structure plot...")
    build_term_structure_plot(ts, ticker, outdir)

    logging.info("Building composite score plot...")
    build_composite_plot(comp_df, ticker, outdir)

    logging.info("Dealer pressure dashboard complete.")


if __name__ == "__main__":
    # Ensure pandas is imported (used in main)
    import pandas as pd  # noqa: F401
    main()

