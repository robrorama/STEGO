#!/usr/bin/env python3
"""
options.gamma_vanna_rotation_dashboard.v1.py

Cross-index dealer gamma / vanna "regime rotation" dashboard for SPY / QQQ / IWM.

- Uses *only* the provided data_retrieval.py and options_data_retrieval.py (must be importable).
- Pulls short-dated option chains, computes Black–Scholes gamma & vanna, and aggregates exposures.
- Saves all derived data as CSVs under /dev/shm for STEGO / MICRA pipelines.
- Builds Plotly HTML dashboards per ticker (heatmaps + strike curves) and opens them in the browser.

Usage examples
--------------
    python3 options.gamma_vanna_rotation_dashboard.v1.py
    python3 options.gamma_vanna_rotation_dashboard.v1.py --tickers SPY,QQQ,IWM
    python3 options.gamma_vanna_rotation_dashboard.v1.py --tickers SPY --max-expiries 6 --max-dte 14

Notes
-----
- Assumes customers are net long options, dealers net short.
- For "dealer gamma exposure" GEX we use:
      position_sign = +1 for calls, -1 for puts (customer delta sign)
      dealer_position = -position_sign
      gamma_exposure = dealer_position * gamma * open_interest * 100 * spot**2
- For vanna exposure we use:
      vanna_exposure = dealer_position * vanna * open_interest * 100 * spot
- Sign conventions can be adjusted later; sign-flip *locations* are usually robust to a global sign change.

"""

import argparse
import os
from datetime import datetime, date
import math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline

# Local loaders – MUST exist in the same directory or Python path
import data_retrieval
import options_data_retrieval


# -----------------------------
# Math / Black–Scholes helpers
# -----------------------------

SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_d1_d2(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    r: float = 0.0,
    q: float = 0.0,
) -> Tuple[float, float]:
    """
    Black–Scholes d1, d2 with basic safety checks.
    """
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return float("nan"), float("nan")
    vsqrt = vol * math.sqrt(t)
    if vsqrt <= 0:
        return float("nan"), float("nan")
    num = math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t
    d1 = num / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


def bs_gamma(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """
    Black–Scholes gamma for equity/options (same for calls and puts).
    """
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return 0.0
    d1, _ = bs_d1_d2(spot, strike, t, vol, r, q)
    if not math.isfinite(d1):
        return 0.0
    denom = spot * vol * math.sqrt(t)
    if denom <= 0:
        return 0.0
    return math.exp(-q * t) * norm_pdf(d1) / denom


def bs_vanna(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    r: float = 0.0,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """
    Approximate vanna: d(Delta)/d(vol).

    One common closed form:
        vanna = -exp(-q t) * phi(d1) * d2 / vol
    (same for calls/puts in BS under some conventions).

    We keep it symmetric and let sign come from dealer position sign.
    """
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return 0.0
    d1, d2 = bs_d1_d2(spot, strike, t, vol, r, q)
    if not (math.isfinite(d1) and math.isfinite(d2)):
        return 0.0
    return -math.exp(-q * t) * norm_pdf(d1) * d2 / max(vol, 1e-8)


# -----------------------------
# Core gamma/vanna computation
# -----------------------------

def compute_gamma_vanna_by_option(
    ticker: str,
    spot: float,
    options_df: pd.DataFrame,
    today: date,
    r: float = 0.0,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    For each option row in the chain, compute:
        - days_to_expiry
        - t_years
        - gamma
        - vanna
        - dealer gamma exposure
        - dealer vanna exposure

    Returns a per-option DataFrame suitable for CSV export.
    """
    df = options_df.copy()

    # Basic columns sanity
    req_cols = ["type", "expiration", "strike", "impliedVolatility", "openInterest"]
    for c in req_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize types
    df["type"] = df["type"].str.lower().replace({"c": "call", "p": "put"})

    # Time to expiry
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    df["days_to_expiry"] = df["expiration"].apply(lambda d: (d - today).days)
    df = df[df["days_to_expiry"] >= 0].copy()

    # Continuous time in years (floor at 1 trading day for stability)
    df["t_years"] = df["days_to_expiry"].astype(float) / 365.0
    df.loc[df["t_years"] <= 0, "t_years"] = 1.0 / 365.0

    # Implied vol and OI
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0.0)

    # Option contract multiplier (US equity options)
    contract_mult = 100.0

    gamma_list = []
    vanna_list = []
    gex_list = []
    vanna_exposure_list = []

    for idx, row in df.iterrows():
        k = float(row["strike"])
        vol = float(row["impliedVolatility"]) if not pd.isna(row["impliedVolatility"]) else 0.0
        t = float(row["t_years"])
        oi = float(row["openInterest"])
        opt_type = str(row["type"]).lower()

        if not (spot > 0 and k > 0 and t > 0 and vol > 0 and oi > 0):
            gamma_val = 0.0
            vanna_val = 0.0
        else:
            gamma_val = bs_gamma(spot, k, t, vol, r=r, q=q)
            vanna_val = bs_vanna(spot, k, t, vol, r=r, q=q, option_type=opt_type)

        # Customer delta sign: calls +1, puts -1 -> dealer is opposite
        if opt_type == "call":
            customer_sign = 1.0
        elif opt_type == "put":
            customer_sign = -1.0
        else:
            customer_sign = 0.0

        dealer_position_sign = -customer_sign

        # Dealer gamma / vanna exposure
        # S^2 scaling for gamma, S scaling for vanna (for interpretability).
        gamma_exposure = dealer_position_sign * gamma_val * oi * contract_mult * (spot ** 2)
        vanna_exposure = dealer_position_sign * vanna_val * oi * contract_mult * spot

        gamma_list.append(gamma_val)
        vanna_list.append(vanna_val)
        gex_list.append(gamma_exposure)
        vanna_exposure_list.append(vanna_exposure)

    df["gamma"] = gamma_list
    df["vanna"] = vanna_list
    df["dealer_gamma_exposure"] = gex_list
    df["dealer_vanna_exposure"] = vanna_exposure_list

    df["ticker"] = ticker
    df["spot"] = spot

    return df


def aggregate_by_strike_expiry(
    df_opt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-option gamma/vanna exposures to strike x expiry grid.
    """
    if df_opt.empty:
        return pd.DataFrame()

    group_cols = ["ticker", "spot", "expiration", "days_to_expiry", "strike"]
    agg_cols = {
        "dealer_gamma_exposure": "sum",
        "dealer_vanna_exposure": "sum",
        "openInterest": "sum",
    }
    agg = df_opt.groupby(group_cols, as_index=False).agg(agg_cols)

    # Convenience: moneyness
    agg["moneyness"] = agg["strike"] / agg["spot"] - 1.0

    return agg


def summarize_spot_window(
    df_strike_exp: pd.DataFrame,
    window_pct: float = 0.02,
) -> pd.DataFrame:
    """
    Around-spot window summary: sum exposures for strikes within +/- window_pct of spot
    for each expiration.
    """
    if df_strike_exp.empty:
        return pd.DataFrame()

    out_rows = []
    for (ticker, spot), df_grp in df_strike_exp.groupby(["ticker", "spot"]):
        low = spot * (1.0 - window_pct)
        high = spot * (1.0 + window_pct)
        mask = (df_grp["strike"] >= low) & (df_grp["strike"] <= high)
        df_win = df_grp[mask]

        if df_win.empty:
            continue

        grouped = df_win.groupby(["expiration", "days_to_expiry"], as_index=False).agg(
            {
                "dealer_gamma_exposure": "sum",
                "dealer_vanna_exposure": "sum",
                "openInterest": "sum",
            }
        )
        grouped["ticker"] = ticker
        grouped["spot"] = spot
        grouped["window_pct"] = window_pct

        out_rows.append(grouped)

    if not out_rows:
        return pd.DataFrame()

    return pd.concat(out_rows, ignore_index=True)


def compute_signflip_levels(
    df_strike_exp: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each ticker x expiration, approximate strike levels where dealer_gamma_exposure crosses zero.
    """
    if df_strike_exp.empty:
        return pd.DataFrame()

    rows = []
    for (ticker, exp), df_grp in df_strike_exp.groupby(["ticker", "expiration"]):
        df_sorted = df_grp.sort_values("strike")
        strikes = df_sorted["strike"].values
        gex = df_sorted["dealer_gamma_exposure"].values

        if len(strikes) < 2:
            continue

        signflips = []
        for i in range(len(strikes) - 1):
            g1 = gex[i]
            g2 = gex[i + 1]
            if g1 == 0:
                signflips.append(strikes[i])
            elif g1 * g2 < 0:
                # Linear interpolation for approximate root
                s1 = strikes[i]
                s2 = strikes[i + 1]
                # g(s) ~ g1 + (g2 - g1) * (s - s1) / (s2 - s1)
                # solve for g(s) = 0
                if s2 != s1:
                    s0 = s1 - g1 * (s2 - s1) / (g2 - g1)
                else:
                    s0 = s1
                signflips.append(s0)

        if not signflips:
            continue

        for s0 in signflips:
            rows.append(
                {
                    "ticker": ticker,
                    "expiration": exp,
                    "approx_signflip_strike": float(s0),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# -----------------------------
# Plotly visualization helpers
# -----------------------------

def make_heatmap_figure(
    ticker: str,
    spot: float,
    df_strike_exp: pd.DataFrame,
) -> go.Figure:
    """
    Create a 2-panel heatmap figure:
        Row 1: dealer_gamma_exposure
        Row 2: dealer_vanna_exposure
    Axes: x = strike, y = days_to_expiry.
    """
    if df_strike_exp.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{ticker}: No gamma/vanna data available",
        )
        return fig

    # Pivot for heatmaps
    df = df_strike_exp.copy()
    # Ensure numeric
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["days_to_expiry"] = pd.to_numeric(df["days_to_expiry"], errors="coerce")

    gamma_pivot = df.pivot_table(
        index="days_to_expiry",
        columns="strike",
        values="dealer_gamma_exposure",
        aggfunc="sum",
    ).sort_index(axis=0).sort_index(axis=1)

    vanna_pivot = df.pivot_table(
        index="days_to_expiry",
        columns="strike",
        values="dealer_vanna_exposure",
        aggfunc="sum",
    ).sort_index(axis=0).sort_index(axis=1)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"{ticker} Dealer Gamma Exposure Heatmap",
            f"{ticker} Dealer Vanna Exposure Heatmap",
        ),
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=gamma_pivot.values,
            x=gamma_pivot.columns,
            y=gamma_pivot.index,
            colorbar_title="Gamma Exp.",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=vanna_pivot.values,
            x=vanna_pivot.columns,
            y=vanna_pivot.index,
            colorbar_title="Vanna Exp.",
        ),
        row=2,
        col=1,
    )

    # Mark spot as a vertical line
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_width=2,
        annotation_text=f"Spot {spot:.2f}",
        annotation_position="top left",
        row="all",
        col=1,
    )

    fig.update_xaxes(title_text="Strike", row=2, col=1)
    fig.update_yaxes(title_text="Days to Expiry", row=1, col=1)
    fig.update_yaxes(title_text="Days to Expiry", row=2, col=1)

    fig.update_layout(
        title=f"{ticker}: Dealer Gamma/Vanna Exposure vs Strike/Expiry",
        height=700,
    )

    return fig


def make_strike_curves_figure(
    ticker: str,
    spot: float,
    df_strike_exp: pd.DataFrame,
) -> go.Figure:
    """
    Create a figure with gamma & vanna vs strike curves for the nearest expirations.
    """
    if df_strike_exp.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{ticker}: No gamma/vanna data for curves",
        )
        return fig

    df = df_strike_exp.copy()
    df = df.sort_values(["expiration", "strike"])

    unique_exps = (
        df[["expiration", "days_to_expiry"]]
        .drop_duplicates()
        .sort_values(["expiration"])
        .reset_index(drop=True)
    )

    # Take up to 4 nearest expiries
    unique_exps = unique_exps.head(4)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"{ticker} Dealer Gamma Exposure vs Strike (nearest expiries)",
            f"{ticker} Dealer Vanna Exposure vs Strike (nearest expiries)",
        ),
        vertical_spacing=0.12,
    )

    for _, row in unique_exps.iterrows():
        exp = row["expiration"]
        dte = row["days_to_expiry"]
        df_e = df[df["expiration"] == exp].sort_values("strike")

        if df_e.empty:
            continue

        label = f"{exp} ({int(dte)}d)"

        fig.add_trace(
            go.Scatter(
                x=df_e["strike"],
                y=df_e["dealer_gamma_exposure"],
                mode="lines",
                name=f"Gamma {label}",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_e["strike"],
                y=df_e["dealer_vanna_exposure"],
                mode="lines",
                name=f"Vanna {label}",
            ),
            row=2,
            col=1,
        )

    # Mark spot
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_width=2,
        annotation_text=f"Spot {spot:.2f}",
        annotation_position="top left",
        row="all",
        col=1,
    )

    fig.update_xaxes(title_text="Strike", row=2, col=1)
    fig.update_yaxes(title_text="Gamma Exp.", row=1, col=1)
    fig.update_yaxes(title_text="Vanna Exp.", row=2, col=1)

    fig.update_layout(
        title=f"{ticker}: Dealer Gamma/Vanna vs Strike (nearest expiries)",
        height=700,
    )

    return fig


# -----------------------------
# Main orchestration
# -----------------------------

def ensure_output_dir(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    return root


def select_expirations_short_dated(
    all_exps: List[pd.Timestamp],
    today: date,
    max_dte: int,
    max_expiries: int,
) -> List[pd.Timestamp]:
    """
    Select a short-dated subset of expirations (by calendar days to expiry).
    """
    # Normalize to date
    exps = [pd.to_datetime(e).normalize().date() for e in all_exps]
    exps_with_dte = []
    for e in exps:
        dte = (e - today).days
        if dte >= 0:
            exps_with_dte.append((e, dte))

    if not exps_with_dte:
        return []

    exps_with_dte.sort(key=lambda x: x[0])

    # Filter by max_dte first
    filtered = [e for e, dte in exps_with_dte if dte <= max_dte]

    if not filtered:
        # If none fall in window, fall back to first max_expiries expirations
        filtered = [e for e, _ in exps_with_dte[:max_expiries]]

    # Limit to max_expiries
    filtered = filtered[:max_expiries]

    # Convert back to Timestamps for options_data_retrieval
    return [pd.to_datetime(e).normalize() for e in filtered]


def process_ticker(
    ticker: str,
    output_root: str,
    max_expiries: int,
    max_dte: int,
) -> Dict[str, str]:
    """
    Full pipeline for a single ticker:
        - Load underlying and spot
        - Get short-dated expirations
        - Ensure option chains cached
        - Compute per-option gamma/vanna
        - Aggregations, summaries, CSVs
        - Plotly HTML dashboards

    Returns:
        dict of key -> path for outputs (CSVs, HTMLs).
    """
    print(f"[INFO] Processing {ticker} ...")

    today = datetime.now().date()

    # Output dirs
    date_str = today.strftime("%Y-%m-%d")
    ticker_dir = os.path.join(output_root, date_str, ticker.upper())
    os.makedirs(ticker_dir, exist_ok=True)

    # Load underlying
    print(f"[INFO] Loading underlying OHLCV for {ticker} via data_retrieval.load_or_download_ticker(period='1y')...")
    df_px = data_retrieval.load_or_download_ticker(ticker, period="1y")
    if df_px.empty:
        print(f"[ERROR] Failed to load underlying data for {ticker}. Skipping.")
        return {}

    spot = float(df_px["Close"].iloc[-1])
    print(f"[INFO] Spot for {ticker}: {spot:.4f}")

    # Get remote expirations
    print(f"[INFO] Querying available remote expirations for {ticker} via options_data_retrieval.get_available_remote_expirations ...")
    try:
        all_exps = options_data_retrieval.get_available_remote_expirations(ticker)
    except Exception as e:
        print(f"[ERROR] Failed to load remote expirations for {ticker}: {e}")
        return {}

    if not all_exps:
        print(f"[ERROR] No remote expirations available for {ticker}.")
        return {}

    chosen_exps = select_expirations_short_dated(
        all_exps,
        today=today,
        max_dte=max_dte,
        max_expiries=max_expiries,
    )

    if not chosen_exps:
        print(f"[ERROR] No suitable short-dated expirations for {ticker}.")
        return {}

    print(f"[INFO] Selected expirations for {ticker}: {[str(e.date()) for e in chosen_exps]}")

    # Ensure chains cached
    print(f"[INFO] Ensuring option chains cached for {ticker} ...")
    try:
        options_data_retrieval.ensure_option_chains_cached(
            ticker,
            expirations=chosen_exps,
            source="yfinance",
            force_refresh=False,
        )
    except Exception as e:
        print(f"[ERROR] Failed to ensure option chains cached for {ticker}: {e}")
        return {}

    # Load all chosen expirations from cache
    print(f"[INFO] Loading cached option chains for {ticker} ...")
    try:
        chains = options_data_retrieval.load_all_cached_option_chains(
            ticker,
            expirations=chosen_exps,
            source="yfinance",
        )
    except Exception as e:
        print(f"[ERROR] Failed to load cached option chains for {ticker}: {e}")
        return {}

    if chains.empty:
        print(f"[ERROR] Cached option chains empty for {ticker}.")
        return {}

    print(f"[INFO] Loaded {len(chains)} option rows for {ticker} across {len(chosen_exps)} expirations.")

    # Compute gamma/vanna per option
    print(f"[INFO] Computing per-option gamma & vanna exposures for {ticker} ...")
    df_opt = compute_gamma_vanna_by_option(
        ticker=ticker,
        spot=spot,
        options_df=chains,
        today=today,
        r=0.0,
        q=0.0,
    )

    # Aggregate by strike x expiry
    print(f"[INFO] Aggregating by strike x expiry for {ticker} ...")
    df_strike_exp = aggregate_by_strike_expiry(df_opt)

    # Spot-window summary
    df_spot_window = summarize_spot_window(df_strike_exp, window_pct=0.02)

    # Sign-flip summary
    df_signflips = compute_signflip_levels(df_strike_exp)

    # -----------------------------
    # CSV outputs
    # -----------------------------
    csv_opt = os.path.join(ticker_dir, f"{ticker.upper()}_per_option_gamma_vanna.csv")
    csv_strike_exp = os.path.join(ticker_dir, f"{ticker.upper()}_strike_expiry_gamma_vanna.csv")
    csv_spot_window = os.path.join(ticker_dir, f"{ticker.upper()}_spot_window_gamma_vanna.csv")
    csv_signflips = os.path.join(ticker_dir, f"{ticker.upper()}_gamma_signflip_levels.csv")

    print(f"[INFO] Writing CSVs for {ticker} into {ticker_dir} ...")
    df_opt.to_csv(csv_opt, index=False)
    df_strike_exp.to_csv(csv_strike_exp, index=False)
    df_spot_window.to_csv(csv_spot_window, index=False)
    df_signflips.to_csv(csv_signflips, index=False)

    # -----------------------------
    # Plotly HTML dashboards
    # -----------------------------
    print(f"[INFO] Building Plotly figures for {ticker} ...")

    fig_hm = make_heatmap_figure(ticker, spot, df_strike_exp)
    fig_curves = make_strike_curves_figure(ticker, spot, df_strike_exp)

    html_hm = os.path.join(ticker_dir, f"{ticker.upper()}_gamma_vanna_heatmaps.html")
    html_curves = os.path.join(ticker_dir, f"{ticker.upper()}_gamma_vanna_strike_curves.html")

    print(f"[INFO] Saving Plotly HTML for {ticker} and opening in browser ...")
    plot_offline(fig_hm, filename=html_hm, auto_open=True)
    plot_offline(fig_curves, filename=html_curves, auto_open=True)

    return {
        "csv_per_option": csv_opt,
        "csv_strike_expiry": csv_strike_exp,
        "csv_spot_window": csv_spot_window,
        "csv_signflips": csv_signflips,
        "html_heatmaps": html_hm,
        "html_curves": html_curves,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dealer gamma/vanna rotation dashboard for SPY/QQQ/IWM (or custom tickers).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="SPY,QQQ,IWM",
        help="Comma-separated list of tickers (default: SPY,QQQ,IWM)",
    )
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=8,
        help="Max number of short-dated expirations per ticker (default: 8)",
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=21,
        help="Max calendar days-to-expiry for 'short-dated' selection (default: 21)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/dev/shm/GAMMA_VANNA_ROTATION",
        help="Root output directory (default: /dev/shm/GAMMA_VANNA_ROTATION)",
    )

    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    output_root = ensure_output_dir(args.output_root)

    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Tickers: {tickers}")
    print(f"[INFO] max_expiries={args.max_expiries}, max_dte={args.max_dte}")

    for ticker in tickers:
        try:
            outputs = process_ticker(
                ticker=ticker,
                output_root=output_root,
                max_expiries=args.max_expiries,
                max_dte=args.max_dte,
            )
            if outputs:
                print(f"[INFO] Completed {ticker}. Outputs:")
                for k, v in outputs.items():
                    print(f"    {k}: {v}")
            else:
                print(f"[WARN] No outputs generated for {ticker}.")
        except Exception as e:
            print(f"[ERROR] Unhandled exception while processing {ticker}: {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

