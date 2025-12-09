#!/usr/bin/env python3
# SCRIPTNAME: options_surface_dashboard.v3.py
# AUTHOR: Michael Derby
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# DATE: November 24, 2025
#
# PURPOSE
# -------
# Full options surface dashboard for a single underlying:
#   • RR25 (25Δ risk reversal)
#   • BF25 (25Δ butterfly)
#   • Smile slices per expiry
#   • ATM IV term structure
#   • IV–RV (10/20/30d)
#   • Dealer Gamma & Vanna by expiry
#   • IV surface heatmap (Strike × Expiration)
#   • Skew term-structure panel (RR25 with sign coloring)
#   • BF25 curvature panel (BF25 with sign coloring)
#   • Gamma/Vanna sign regime heatmap
#
# DESIGN
# ------
# - Uses ONLY your canonical data layer:
#       data_retrieval.py
#       options_data_retrieval.py
# - Does NOT modify those files.
# - Follows your filesystem layout and caching policy.
# - Defaults to the next 6 nearest expirations.
# - Allows:
#       --max-expirations N
#       --all
#       --expirations YYYY-MM-DD YYYY-MM-DD ...
# - Prefers cached expirations; falls back to remote listing if needed.
#
# OUTPUTS
# -------
#   /dev/shm/options_surface/{TICKER}/
#       rr25.html
#       bf25.html
#       rr25_termstructure_signed.html
#       bf25_curvature_signed.html
#       gamma_vanna_regime_map.html
#       smile_YYYY-MM-DD.html
#       term_structure.html
#       iv_rv.csv
#       iv_rv.html
#       dealer_gamma.html
#       dealer_vanna.html
#       iv_surface.html
#

import os
import sys
import argparse
import logging
from datetime import datetime
from math import log, sqrt, exp, pi, erf

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

# ----------------------------------------------------------------------
# Import your canonical data modules
# ----------------------------------------------------------------------
try:
    import data_retrieval
    import options_data_retrieval
except Exception as e:
    print("ERROR importing data_retrieval / options_data_retrieval:", e)
    sys.exit(1)


# ----------------------------------------------------------------------
# Filesystem helpers
# ----------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# Math / distributions
# ----------------------------------------------------------------------
def norm_pdf(x):
    return 1.0 / sqrt(2.0 * pi) * np.exp(-0.5 * x * x)


def norm_cdf(x):
    """
    Normal CDF using math.erf, vectorized for numpy arrays.
    Avoids np.erf dependency (not present in some NumPy builds).
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


# ----------------------------------------------------------------------
# Greeks & IV processing
# ----------------------------------------------------------------------
def add_iv_and_greeks(
    df: pd.DataFrame,
    spot: float,
    today: pd.Timestamp,
    r: float = 0.05,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    Add 'iv', 'delta', 'gamma', 'vanna' to the option chain DataFrame.

    - Uses 'impliedVolatility' from yfinance if present.
    - Computes Black–Scholes greeks for calls/puts.
    - Treats any T <= 0 as 1/365 year (avoid div-by-zero).
    """
    if df.empty:
        return df

    df = df.copy()

    # Standardize IV column to 'iv'
    if "iv" not in df.columns:
        if "impliedVolatility" in df.columns:
            df["iv"] = df["impliedVolatility"].astype(float)
        else:
            logging.warning("No impliedVolatility column found; cannot compute greeks.")
            df["iv"] = np.nan

    # Ensure expiration is Timestamp
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()

    # Time to expiration in years
    T = (df["expiration"] - today).dt.days.astype(float) / 365.0
    T = T.clip(lower=1.0 / 365.0)  # avoid zero or negative

    K = df["strike"].astype(float)
    sigma = df["iv"].astype(float).clip(lower=1e-6)  # avoid zero

    # Black–Scholes d1, d2
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

    # Normal pdf/cdf
    N_d1 = norm_cdf(d1)
    N_minus_d1 = norm_cdf(-d1)
    phi_d1 = norm_pdf(d1)

    disc_q = np.exp(-q * T)

    # Delta
    delta_call = disc_q * N_d1
    delta_put = -disc_q * N_minus_d1

    # Gamma
    gamma = disc_q * phi_d1 / (spot * sigma * np.sqrt(T))

    # Vanna (∂²V / ∂S∂σ) ~ S * e^{-qT} * sqrt(T) * phi(d1)
    vanna = spot * disc_q * np.sqrt(T) * phi_d1

    # Assign based on type
    df["type"] = df["type"].str.lower()
    df["delta"] = np.where(df["type"] == "call", delta_call, delta_put)
    df["gamma"] = gamma
    df["vanna"] = np.where(df["type"] == "call", vanna, -vanna)

    return df


# ----------------------------------------------------------------------
# RR25 / BF25 helpers
# ----------------------------------------------------------------------
def nearest_delta(df: pd.DataFrame, target: float = 0.25, is_call: bool = True) -> float:
    """Return IV at nearest delta to ±target."""
    if df.empty or "delta" not in df.columns or "iv" not in df.columns:
        return np.nan

    df = df.copy()
    if is_call:
        df["d_err"] = (df["delta"] - target).abs()
    else:
        df["d_err"] = (np.abs(df["delta"]) - target).abs()

    idx = df["d_err"].idxmin()
    if pd.isna(idx):
        return np.nan

    return float(df.loc[idx, "iv"])


def compute_rr25(ocdf: pd.DataFrame) -> float:
    """RR25 = 25Δ call IV - 25Δ put IV."""
    calls = ocdf[ocdf["type"] == "call"]
    puts = ocdf[ocdf["type"] == "put"]

    c25 = nearest_delta(calls, 0.25, True)
    p25 = nearest_delta(puts, 0.25, False)
    if np.isnan(c25) or np.isnan(p25):
        return np.nan
    return c25 - p25


def compute_bf25(ocdf: pd.DataFrame) -> float:
    """
    Approx BF25:
        BF25 ≈ avg(25Δ call IV, 25Δ put IV) – ATM IV
    where ATM is approx |delta| < 0.05.
    """
    if ocdf.empty:
        return np.nan

    atm_slice = ocdf.loc[ocdf["delta"].abs() < 0.05, "iv"]
    atm_iv = atm_slice.mean() if not atm_slice.empty else np.nan
    if np.isnan(atm_iv):
        return np.nan

    calls = ocdf[ocdf["type"] == "call"]
    puts = ocdf[ocdf["type"] == "put"]
    c25 = nearest_delta(calls, 0.25, True)
    p25 = nearest_delta(puts, 0.25, False)
    if np.isnan(c25) or np.isnan(p25):
        return np.nan

    wings = 0.5 * (c25 + p25)
    return wings - atm_iv


# ----------------------------------------------------------------------
# Realized vol
# ----------------------------------------------------------------------
def realized_vol(ohlc: pd.DataFrame, window: int) -> float:
    logret = np.log(ohlc["Close"]).diff()
    rv = logret.rolling(window).std() * np.sqrt(252.0)
    return float(rv.iloc[-1])


# ----------------------------------------------------------------------
# Dealer gamma / vanna aggregation
# ----------------------------------------------------------------------
def aggregate_dealer_greeks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "gamma" not in df.columns or "vanna" not in df.columns:
        return pd.DataFrame(columns=["expiration", "gamma", "vanna"])

    g = (
        df.groupby("expiration")[["gamma", "vanna"]]
        .sum()
        .reset_index()
        .sort_values("expiration")
    )
    return g


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def save_and_open(fig: go.Figure, path: str) -> None:
    plot(fig, filename=path, auto_open=True)


def colorful_surface_fig(x, y, z, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                colorscale="Turbo",
                colorbar=dict(title="IV"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Strike",
        yaxis_title="Expiration",
        template="plotly_dark",
    )
    return fig


# ----------------------------------------------------------------------
# Expiration selection logic
# ----------------------------------------------------------------------
def determine_expirations(
    ticker: str,
    all_flag: bool,
    max_expirations: int,
    explicit_exps,
) -> list:
    """
    Determine which expirations to use, honoring:
        1) explicit_exps (if given)
        2) --all
        3) --max-expirations or default 6

    Preference:
        - Use cached expirations if available.
        - If not enough cached, pull remote list and extend.
    """
    # Explicit expirations override everything
    if explicit_exps:
        exps = [pd.to_datetime(e).normalize() for e in explicit_exps]
        exps = sorted(set(exps))
        logging.info(f"Using explicit expirations: {[e.date() for e in exps]}")
        return exps

    # Cached expirations
    cached = options_data_retrieval.list_cached_option_expirations(ticker)
    cached = sorted(set(cached))

    # Remote expirations (only if needed)
    def load_remote():
        try:
            return options_data_retrieval.get_available_remote_expirations(ticker)
        except Exception as e:
            logging.error(f"Failed to fetch remote expirations for {ticker}: {e}")
            return []

    if all_flag:
        if cached:
            logging.info(f"Using ALL cached expirations ({len(cached)}) for {ticker}")
            return cached
        remote = load_remote()
        logging.info(f"Using ALL remote expirations ({len(remote)}) for {ticker}")
        return remote

    # Default / max-expirations path
    N = max_expirations if max_expirations is not None else 6

    if cached:
        exps = cached[:N]
        logging.info(
            f"Using first {len(exps)} cached expirations for {ticker}: {[e.date() for e in exps]}"
        )
        return exps

    # No cached expirations -> fall back to remote
    remote = load_remote()
    exps = remote[:N]
    logging.info(
        f"Using first {len(exps)} remote expirations for {ticker}: {[e.date() for e in exps]}"
    )
    return exps


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Options surface dashboard wired to data_retrieval / options_data_retrieval."
    )
    parser.add_argument("--ticker", type=str, required=True, help="Underlying ticker (e.g. SPY)")
    parser.add_argument(
        "--max-expirations",
        type=int,
        default=6,
        help="Use the nearest N expirations (default: 6). Ignored if --all or --expirations is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use ALL available expirations (cached if possible, else remote).",
    )
    parser.add_argument(
        "--expirations",
        nargs="*",
        help="Explicit list of expirations (YYYY-MM-DD). Overrides --all and --max-expirations.",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()

    outdir = f"/dev/shm/options_surface/{ticker}"
    ensure_dir(outdir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # --------------------------------------------------------------
    # Load underlying spot + realized vol base
    # --------------------------------------------------------------
    logging.info(f"Loading underlying for {ticker} via data_retrieval...")
    ohlc = data_retrieval.load_or_download_ticker(ticker)
    if ohlc.empty:
        logging.error(f"No OHLC data for {ticker}")
        sys.exit(1)

    spot = float(ohlc["Close"].iloc[-1])
    today = pd.to_datetime(ohlc.index[-1]).normalize()
    logging.info(f"Spot for {ticker}: {spot:.4f} (as of {today.date()})")

    # --------------------------------------------------------------
    # Determine expirations to use
    # --------------------------------------------------------------
    expirations = determine_expirations(
        ticker=ticker,
        all_flag=args.all,
        max_expirations=args.max_expirations,
        explicit_exps=args.expirations,
    )
    if not expirations:
        logging.error("No expirations available.")
        sys.exit(1)

    # Ensure chains are cached (but do not force refresh)
    logging.info("Ensuring option chains are cached...")
    try:
        options_data_retrieval.ensure_option_chains_cached(
            ticker=ticker,
            expirations=expirations,
            source="yfinance",
            force_refresh=False,
        )
    except Exception as e:
        logging.error(f"Failed to ensure option chains cached: {e}")

    # Load all selected chains
    logging.info("Loading option chains from cache...")
    oc = options_data_retrieval.load_all_cached_option_chains(
        ticker=ticker,
        source="yfinance",
        expirations=expirations,
    )
    if oc.empty:
        logging.error("Loaded option chain DataFrame is empty.")
        sys.exit(1)

    # Add IV + greeks
    logging.info("Computing IV and greeks (delta, gamma, vanna)...")
    oc = add_iv_and_greeks(oc, spot=spot, today=today)

    # Normalize types and expiration
    oc["type"] = oc["type"].str.lower()
    oc["expiration"] = pd.to_datetime(oc["expiration"]).dt.normalize()

    # Sorted unique expiration list (only those actually present)
    exp_list = sorted(oc["expiration"].unique())
    exp_dates = [e.date() for e in exp_list]

    # ==============================================================
    # 1) RR25 & BF25 by expiry
    # ==============================================================
    rr25_vals = []
    bf25_vals = []

    for e in exp_list:
        sub = oc[oc["expiration"] == e]
        rr25_vals.append(compute_rr25(sub))
        bf25_vals.append(compute_bf25(sub))

    # --- Basic RR25 line panel ---
    fig_rr = go.Figure()
    fig_rr.add_trace(
        go.Scatter(
            x=exp_dates,
            y=rr25_vals,
            mode="lines+markers",
            name="RR25",
        )
    )
    fig_rr.update_layout(
        title=f"{ticker} 25Δ Risk Reversal (RR25) by Expiry",
        xaxis_title="Expiration",
        yaxis_title="RR25 (Call25 IV - Put25 IV)",
        template="plotly_dark",
    )
    save_and_open(fig_rr, os.path.join(outdir, "rr25.html"))

    # --- Basic BF25 line panel ---
    fig_bf = go.Figure()
    fig_bf.add_trace(
        go.Scatter(
            x=exp_dates,
            y=bf25_vals,
            mode="lines+markers",
            name="BF25",
        )
    )
    fig_bf.update_layout(
        title=f"{ticker} 25Δ Butterfly (BF25) by Expiry",
        xaxis_title="Expiration",
        yaxis_title="BF25 (Wings - ATM IV)",
        template="plotly_dark",
    )
    save_and_open(fig_bf, os.path.join(outdir, "bf25.html"))

    # ==============================================================
    # 1b) Extra panels: Skew term-structure & BF curvature (signed)
    # ==============================================================
    # RR25 with diverging color by sign
    fig_rr_signed = go.Figure()
    fig_rr_signed.add_trace(
        go.Bar(
            x=exp_dates,
            y=rr25_vals,
            marker=dict(
                color=rr25_vals,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="RR25"),
            ),
            name="RR25",
        )
    )
    fig_rr_signed.update_layout(
        title=f"{ticker} Skew Term Structure (RR25, sign-colored)",
        xaxis_title="Expiration",
        yaxis_title="RR25",
        template="plotly_dark",
    )
    save_and_open(fig_rr_signed, os.path.join(outdir, "rr25_termstructure_signed.html"))

    # BF25 curvature map (bars with sign coloring)
    fig_bf_signed = go.Figure()
    fig_bf_signed.add_trace(
        go.Bar(
            x=exp_dates,
            y=bf25_vals,
            marker=dict(
                color=bf25_vals,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="BF25"),
            ),
            name="BF25",
        )
    )
    fig_bf_signed.update_layout(
        title=f"{ticker} BF25 Curvature (sign-colored)",
        xaxis_title="Expiration",
        yaxis_title="BF25 (wings - ATM IV)",
        template="plotly_dark",
    )
    save_and_open(fig_bf_signed, os.path.join(outdir, "bf25_curvature_signed.html"))

    # ==============================================================
    # 2) Smile slices (per expiry)
    # ==============================================================
    for e in exp_list:
        sub = oc[oc["expiration"] == e].sort_values("strike")

        if sub.empty:
            continue

        fig_s = go.Figure()
        fig_s.add_trace(
            go.Scatter(
                x=sub["strike"],
                y=sub["iv"],
                mode="lines+markers",
                text=[
                    f"type={t}, Δ={d:.2f}, γ={g:.4f}"
                    for t, d, g in zip(sub["type"], sub["delta"], sub["gamma"])
                ],
                hovertemplate="Strike=%{x}<br>IV=%{y:.4f}<br>%{text}<extra></extra>",
                name=str(e.date()),
            )
        )
        fig_s.update_layout(
            title=f"{ticker} Smile Slice – {e.date()}",
            xaxis_title="Strike",
            yaxis_title="Implied Volatility",
            template="plotly_dark",
        )
        save_and_open(fig_s, os.path.join(outdir, f"smile_{e.date()}.html"))

    # ==============================================================
    # 3) ATM IV term structure
    # ==============================================================
    atm_iv = []
    for e in exp_list:
        sub = oc[oc["expiration"] == e]
        atm_slice = sub.loc[sub["delta"].abs() < 0.05, "iv"]
        atm_iv.append(float(atm_slice.mean()) if not atm_slice.empty else np.nan)

    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=exp_dates,
            y=atm_iv,
            mode="lines+markers",
            name="ATM IV",
        )
    )
    fig_ts.update_layout(
        title=f"{ticker} ATM IV Term Structure",
        xaxis_title="Expiration",
        yaxis_title="ATM IV (|Δ| < 0.05)",
        template="plotly_dark",
    )
    save_and_open(fig_ts, os.path.join(outdir, "term_structure.html"))

    # ==============================================================
    # 4) IV – RV (10/20/30d)
    # ==============================================================
    logging.info("Computing realized vol windows for IV–RV...")
    rv10 = realized_vol(ohlc, 10)
    rv20 = realized_vol(ohlc, 20)
    rv30 = realized_vol(ohlc, 30)

    current_atm = atm_iv[0] if atm_iv else np.nan

    iv_rv_table = pd.DataFrame(
        {
            "Window": [10, 20, 30],
            "RV": [rv10, rv20, rv30],
            "ATM_IV": [current_atm] * 3,
            "IV_minus_RV": [
                current_atm - rv10,
                current_atm - rv20,
                current_atm - rv30,
            ],
        }
    )
    iv_rv_csv = os.path.join(outdir, "iv_rv.csv")
    iv_rv_table.to_csv(iv_rv_csv, index=False)

    fig_ivrv = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(iv_rv_table.columns),
                    fill_color="#222222",
                    font=dict(color="white"),
                    align="center",
                ),
                cells=dict(
                    values=[iv_rv_table[c] for c in iv_rv_table.columns],
                    fill_color="#333333",
                    align="center",
                ),
            )
        ]
    )
    fig_ivrv.update_layout(
        title=f"{ticker} IV–RV Snapshot (as of {today.date()})",
        template="plotly_dark",
    )
    save_and_open(fig_ivrv, os.path.join(outdir, "iv_rv.html"))

    # ==============================================================
    # 5) Dealer Gamma & Vanna by expiry
    # ==============================================================
    logging.info("Aggregating dealer gamma & vanna by expiry...")
    gv = aggregate_dealer_greeks(oc)

    if not gv.empty:
        gv_dates = [d.date() for d in gv["expiration"]]

        fig_g = go.Figure()
        fig_g.add_trace(
            go.Bar(
                x=gv_dates,
                y=gv["gamma"],
                name="Gamma",
            )
        )
        fig_g.update_layout(
            title=f"{ticker} Dealer Gamma by Expiry",
            xaxis_title="Expiration",
            yaxis_title="Net Gamma (approx)",
            template="plotly_dark",
        )
        save_and_open(fig_g, os.path.join(outdir, "dealer_gamma.html"))

        fig_v = go.Figure()
        fig_v.add_trace(
            go.Bar(
                x=gv_dates,
                y=gv["vanna"],
                name="Vanna",
            )
        )
        fig_v.update_layout(
            title=f"{ticker} Dealer Vanna by Expiry",
            xaxis_title="Expiration",
            yaxis_title="Net Vanna (approx)",
            template="plotly_dark",
        )
        save_and_open(fig_v, os.path.join(outdir, "dealer_vanna.html"))

        # ----------------------------------------------------------
        # Extra panel: Gamma/Vanna sign regime map (heatmap)
        # ----------------------------------------------------------
        z = np.vstack([gv["gamma"].values, gv["vanna"].values])
        fig_regime = go.Figure(
            data=[
                go.Heatmap(
                    x=gv_dates,
                    y=["Gamma", "Vanna"],
                    z=z,
                    colorscale="RdBu",
                    reversescale=True,
                    colorbar=dict(title="Exposure"),
                )
            ]
        )
        fig_regime.update_layout(
            title=f"{ticker} Dealer Gamma/Vanna Regime Map",
            xaxis_title="Expiration",
            yaxis_title="Greek",
            template="plotly_dark",
        )
        save_and_open(fig_regime, os.path.join(outdir, "gamma_vanna_regime_map.html"))
    else:
        logging.warning("No gamma/vanna columns available to aggregate.")

    # ==============================================================
    # 6) IV Surface Heatmap (Strike × Expiration)
    # ==============================================================
    logging.info("Building IV surface heatmap...")
    pivot = (
        oc.pivot_table(
            index="expiration",
            columns="strike",
            values="iv",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if not pivot.empty:
        fig_heat = colorful_surface_fig(
            x=pivot.columns.astype(float),
            y=[idx.date() for idx in pivot.index],
            z=pivot.values,
            title=f"{ticker} IV Surface (Strike × Expiration)",
        )
        save_and_open(fig_heat, os.path.join(outdir, "iv_surface.html"))
    else:
        logging.warning("IV surface pivot is empty; skipping heatmap.")

    logging.info("Options surface dashboard completed.")


if __name__ == "__main__":
    main()

