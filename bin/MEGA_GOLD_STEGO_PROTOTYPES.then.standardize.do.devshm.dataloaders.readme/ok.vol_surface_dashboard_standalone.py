#!/usr/bin/env python3
"""
vol_surface_dashboard_standalone.py

Author: (you)
Description:
    Standalone volatility surface and skew/term-structure visualizer
    for a single underlying using yfinance and Plotly.

    What this script does:
      - Downloads option chains for a given ticker from yfinance
      - Computes:
          * ATM IV (approx 50-delta) per expiry
          * 25-delta call IV and 25-delta put IV per expiry
          * 25-delta risk reversal (RR25 = IV25C - IV25P)
          * 25-delta butterfly (BF25 = 0.5*(IV25C+IV25P) - IV50)
      - Builds a mid IV surface across strikes and expiries

    Creative visualizations included:
      1) 3D IV Surface:
         - Z: mid implied volatility (avg of call/put IV)
         - X: moneyness (K / spot)
         - Y: time to expiry (years)

      2) 2D IV Heatmap:
         - Color map of IV across moneyness vs tenor

      3) ATM Term Structure:
         - ATM (50-delta) IV vs tenor

      4) Smile Overlay:
         - IV vs moneyness for several representative expiries (short/mid/long)

      5) RR/BF Microstructure Plot:
         - RR25 and BF25 per expiry to see skew direction and curvature

      6) Delta Buckets Term Structure:
         - IV vs tenor for 50-delta, 25-delta call, 25-delta put

    All plots are combined into a single self-contained HTML dashboard
    and opened automatically in your default browser.

Usage:
    python3 vol_surface_dashboard_standalone.py --ticker SPY --max-expiries 10
"""

import argparse
import datetime as dt
import math
import os
import sys
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf


# --------------- Math helpers ---------------

def norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black–Scholes call delta (European)."""
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return float("nan")
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    except Exception:
        return float("nan")
    return norm_cdf(d1)


def sanitize_iv(iv):
    """Convert IV from yfinance to float and treat non-positive as NaN."""
    try:
        iv = float(iv)
    except Exception:
        return float("nan")
    if not (iv > 0 and iv < 5):  # basic sanity: < 500% IV
        return float("nan")
    return iv


# --------------- Core data loader ---------------

def load_underlying_and_options(ticker: str, max_expiries: int = 15):
    """
    Load underlying spot and option chains across expiries using yfinance.
    Returns:
        spot (float),
        expiry_dates (list of datetime.date),
        option_chains (dict: expiry_str -> (calls_df, puts_df))
    """
    print(f"[INFO] Loading underlying data for {ticker} via yfinance...")
    yf_ticker = yf.Ticker(ticker)

    # Spot from last close
    hist = yf_ticker.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker}.")
    spot = float(hist["Close"].dropna().iloc[-1])
    print(f"[INFO] Spot for {ticker}: {spot:.4f}")

    all_expiries = list(yf_ticker.options)
    if not all_expiries:
        raise RuntimeError(f"No listed options found for {ticker} via yfinance.")

    all_expiries = sorted(all_expiries)
    if max_expiries is not None and max_expiries > 0:
        all_expiries = all_expiries[:max_expiries]

    print(f"[INFO] Using up to {len(all_expiries)} expiries: {all_expiries}")

    option_chains = {}
    for exp_str in all_expiries:
        print(f"[INFO] Downloading option chain for expiry {exp_str}...")
        chain = yf_ticker.option_chain(exp_str)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        option_chains[exp_str] = (calls, puts)

    expiry_dates = [dt.datetime.strptime(e, "%Y-%m-%d").date() for e in all_expiries]

    return spot, expiry_dates, option_chains


# --------------- Surface & skew computations ---------------

def build_surface_and_metrics(spot, expiry_dates, option_chains, risk_free: float = 0.03):
    """
    Build a mid-IV surface and compute per-expiry metrics (ATM IV, RR, BF, delta term structures).

    Returns:
        df_surface: rows of (T_years, expiry, strike, moneyness, iv_mid)
        df_expiry_metrics: per-expiry metrics (ATM, RR, BF, 25C, 25P, 50)
    """
    today = dt.date.today()

    surface_rows = []
    expiry_metrics_rows = []

    for exp_date in expiry_dates:
        exp_str = exp_date.strftime("%Y-%m-%d")
        calls, puts = option_chains[exp_str]

        # Ensure basic columns exist
        if "strike" not in calls.columns or "impliedVolatility" not in calls.columns:
            print(f"[WARN] Calls for {exp_str} missing required columns; skipping.")
            continue
        if "strike" not in puts.columns or "impliedVolatility" not in puts.columns:
            print(f"[WARN] Puts for {exp_str} missing required columns; skipping.")
            continue

        # Time to expiry (year fraction)
        days_to_exp = (exp_date - today).days
        T_years = max(days_to_exp / 365.25, 0.001)  # small epsilon if 0

        calls = calls.copy()
        puts = puts.copy()

        calls["impliedVolatility"] = calls["impliedVolatility"].apply(sanitize_iv)
        puts["impliedVolatility"] = puts["impliedVolatility"].apply(sanitize_iv)

        # Merge calls/puts by strike for mid IV surface
        merged = pd.merge(
            calls[["strike", "impliedVolatility"]],
            puts[["strike", "impliedVolatility"]],
            on="strike",
            how="outer",
            suffixes=("_c", "_p"),
        )

        # Compute mid IV per strike
        def mid_iv(row):
            ivs = []
            if not math.isnan(row.get("impliedVolatility_c", float("nan"))):
                ivs.append(row["impliedVolatility_c"])
            if not math.isnan(row.get("impliedVolatility_p", float("nan"))):
                ivs.append(row["impliedVolatility_p"])
            if not ivs:
                return float("nan")
            return float(np.mean(ivs))

        merged["iv_mid"] = merged.apply(mid_iv, axis=1)
        merged = merged.dropna(subset=["iv_mid"])

        if merged.empty:
            print(f"[WARN] No valid mid IVs for expiry {exp_str}; skipping.")
            continue

        # Populate surface rows
        for _, row in merged.iterrows():
            K = float(row["strike"])
            iv_m = float(row["iv_mid"])
            moneyness = K / spot if spot > 0 else float("nan")
            surface_rows.append(
                {
                    "expiry": exp_str,
                    "T_years": T_years,
                    "strike": K,
                    "moneyness": moneyness,
                    "iv_mid": iv_m,
                }
            )

        # ---- Per-expiry ATM, RR25, BF25, and delta-bucket IVs ----

        # ATM approx: strike with min |K-spot| using mid IV
        merged["dist_atm"] = (merged["strike"] - spot).abs()
        merged_atm = merged.sort_values("dist_atm").iloc[0]
        atm_iv = float(merged_atm["iv_mid"])

        # Delta bucket search
        best_50_iv = float("nan")
        best_25c_iv = float("nan")
        best_25p_iv = float("nan")
        best_50_dist = float("inf")
        best_25c_dist = float("inf")
        best_25p_dist = float("inf")

        # Search calls
        for _, row in calls.iterrows():
            K = float(row["strike"])
            iv = row["impliedVolatility"]
            iv = sanitize_iv(iv)
            if math.isnan(iv):
                continue
            delta_c = bs_delta_call(spot, K, risk_free, iv, T_years)
            if math.isnan(delta_c):
                continue

            # 50-delta (call)
            dist_50 = abs(delta_c - 0.5)
            if dist_50 < best_50_dist:
                best_50_dist = dist_50
                best_50_iv = iv

            # 25-delta call
            dist_25c = abs(delta_c - 0.25)
            if dist_25c < best_25c_dist:
                best_25c_dist = dist_25c
                best_25c_iv = iv

        # Search puts
        for _, row in puts.iterrows():
            K = float(row["strike"])
            iv = row["impliedVolatility"]
            iv = sanitize_iv(iv)
            if math.isnan(iv):
                continue
            # call delta from BS, then put delta = call_delta - 1
            delta_c = bs_delta_call(spot, K, risk_free, iv, T_years)
            if math.isnan(delta_c):
                continue
            delta_p = delta_c - 1.0  # negative

            # 25-delta put ~ delta = -0.25
            dist_25p = abs(delta_p + 0.25)
            if dist_25p < best_25p_dist:
                best_25p_dist = dist_25p
                best_25p_iv = iv

        # Fallback: if best_50_iv is NaN, use ATM
        if math.isnan(best_50_iv):
            best_50_iv = atm_iv

        # RR25 and BF25
        if not (math.isnan(best_25c_iv) or math.isnan(best_25p_iv) or math.isnan(best_50_iv)):
            rr25 = best_25c_iv - best_25p_iv
            bf25 = 0.5 * (best_25c_iv + best_25p_iv) - best_50_iv
        else:
            rr25 = float("nan")
            bf25 = float("nan")

        expiry_metrics_rows.append(
            {
                "expiry": exp_str,
                "T_years": T_years,
                "ATM_IV": atm_iv,
                "IV_50": best_50_iv,
                "IV_25C": best_25c_iv,
                "IV_25P": best_25p_iv,
                "RR25": rr25,
                "BF25": bf25,
            }
        )

    if not surface_rows:
        raise RuntimeError("No valid surface rows constructed. Possibly no IV data.")

    df_surface = pd.DataFrame(surface_rows)
    df_expiry_metrics = pd.DataFrame(expiry_metrics_rows).sort_values("T_years")

    return df_surface, df_expiry_metrics


# --------------- Plot builders ---------------

def build_surface_figs(df_surface, ticker: str):
    """
    Build 3D IV surface and 2D heatmap from df_surface.
    df_surface columns: expiry, T_years, strike, moneyness, iv_mid
    """
    df = df_surface.copy()
    df["m_bucket"] = df["moneyness"].round(2)

    pivot = df.pivot_table(
        index="T_years", columns="m_bucket", values="iv_mid", aggfunc="mean"
    ).sort_index(axis=0).sort_index(axis=1)

    Z = pivot.values
    T_vals = pivot.index.values
    M_vals = pivot.columns.values

    # 3D Surface
    fig3d = go.Figure(
        data=[
            go.Surface(
                z=Z,
                x=M_vals,
                y=T_vals,
                colorbar={"title": "IV"},
            )
        ]
    )
    fig3d.update_layout(
        title=f"{ticker} Implied Volatility Surface (Mid IV)",
        scene=dict(
            xaxis_title="Moneyness (K / Spot)",
            yaxis_title="Tenor (years)",
            zaxis_title="IV (decimal)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # 2D Heatmap
    fig_hm = go.Figure(
        data=[
            go.Heatmap(
                z=Z,
                x=M_vals,
                y=T_vals,
                colorbar={"title": "IV"},
            )
        ]
    )
    fig_hm.update_layout(
        title=f"{ticker} IV Heatmap (Mid IV)",
        xaxis_title="Moneyness (K / Spot)",
        yaxis_title="Tenor (years)",
        margin=dict(l=60, r=10, b=40, t=40),
    )

    return fig3d, fig_hm


def build_term_structure_fig(df_expiry_metrics, ticker: str):
    """ATM and delta-bucket term structure."""
    df = df_expiry_metrics.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["T_years"],
            y=df["ATM_IV"],
            mode="lines+markers",
            name="ATM IV (approx 50Δ)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["T_years"],
            y=df["IV_25C"],
            mode="lines+markers",
            name="25Δ Call IV",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["T_years"],
            y=df["IV_25P"],
            mode="lines+markers",
            name="25Δ Put IV",
        )
    )

    fig.update_layout(
        title=f"{ticker} Term Structure by Delta Buckets",
        xaxis_title="Tenor (years)",
        yaxis_title="Implied Volatility (decimal)",
        legend=dict(orientation="h", x=0.0, y=1.1),
        margin=dict(l=60, r=10, b=40, t=60),
    )
    return fig


def build_smiles_fig(df_surface, ticker: str, max_smiles: int = 4):
    """
    Plot IV smiles (IV vs moneyness) for a few representative expiries.
    """
    df = df_surface.copy()
    unique_T = sorted(df["T_years"].unique())
    if not unique_T:
        raise RuntimeError("No tenors available for smiles.")

    # Choose up to max_smiles evenly spaced tenors
    idxs = np.linspace(0, len(unique_T) - 1, num=min(max_smiles, len(unique_T)), dtype=int)
    chosen_T = [unique_T[i] for i in sorted(set(idxs))]

    fig = go.Figure()
    for T in chosen_T:
        slice_df = df[df["T_years"] == T].copy()
        slice_df = slice_df.sort_values("moneyness")
        fig.add_trace(
            go.Scatter(
                x=slice_df["moneyness"],
                y=slice_df["iv_mid"],
                mode="lines+markers",
                name=f"T={T:.3f}y",
            )
        )

    fig.update_layout(
        title=f"{ticker} IV Smiles (Mid IV vs Moneyness)",
        xaxis_title="Moneyness (K / Spot)",
        yaxis_title="Implied Volatility (decimal)",
        legend=dict(orientation="h", x=0.0, y=1.1),
        margin=dict(l=60, r=10, b=40, t=60),
    )
    return fig


def build_rr_bf_fig(df_expiry_metrics, ticker: str):
    """Plot RR25 and BF25 by tenor as a quick skew/curvature lens."""
    df = df_expiry_metrics.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["T_years"],
            y=df["RR25"],
            mode="lines+markers",
            name="RR25 (25ΔC - 25ΔP)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["T_years"],
            y=df["BF25"],
            mode="lines+markers",
            name="BF25 (wing richness vs ATM)",
        )
    )

    fig.update_layout(
        title=f"{ticker} Skew & Curvature (RR25 / BF25)",
        xaxis_title="Tenor (years)",
        yaxis_title="IV difference (decimal)",
        legend=dict(orientation="h", x=0.0, y=1.1),
        margin=dict(l=60, r=10, b=40, t=60),
    )
    return fig


def build_rr_vs_bf_scatter(df_expiry_metrics, ticker: str):
    """Scatter of RR25 vs BF25 colored by tenor."""
    df = df_expiry_metrics.copy()

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["RR25"],
                y=df["BF25"],
                mode="markers+text",
                text=[f"{t:.2f}y" for t in df["T_years"]],
                textposition="top center",
                marker=dict(size=10),
                name="Expiries",
            )
        ]
    )

    fig.update_layout(
        title=f"{ticker} RR25 vs BF25 Map (per expiry)",
        xaxis_title="RR25 (25Δ call IV - 25Δ put IV)",
        yaxis_title="BF25 (wing richness vs ATM)",
        margin=dict(l=60, r=10, b=40, t=60),
    )
    return fig


# --------------- Main Dashboard Builder ---------------

def build_dashboard_html(figures, ticker: str, outfile: str):
    """
    Combine multiple Plotly figures into a single HTML file, and open it.
    """
    print(f"[INFO] Writing dashboard to {outfile} ...")
    parts = []
    for i, fig in enumerate(figures):
        parts.append(
            pio.to_html(
                fig,
                include_plotlyjs=(i == 0),
                full_html=False,
                default_width="100%",
                default_height="600px",
            )
        )
    html = "<html><head><meta charset='utf-8'><title>{}</title></head><body>".format(
        f"{ticker} Vol Surface Dashboard"
    )
    html += "\n".join(parts)
    html += "</body></html>"

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(outfile)
    print(f"[INFO] Opening in browser: file://{abs_path}")
    webbrowser.open("file://" + abs_path)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone volatility surface & skew dashboard using yfinance and Plotly."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Underlying ticker symbol (e.g., SPY, AAPL, QQQ, NVDA).",
    )
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=15,
        help="Maximum number of expiries to pull for the surface.",
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.03,
        help="Risk-free rate used for delta calculations (annualized, e.g., 0.03 = 3%%).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML filename (default: vol_surface_dashboard_<TICKER>.html).",
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()
    max_expiries = args.max_expiries
    r = args.risk_free
    outfile = args.output or f"vol_surface_dashboard_{ticker}.html"

    try:
        spot, expiry_dates, option_chains = load_underlying_and_options(
            ticker, max_expiries=max_expiries
        )
        df_surface, df_expiry_metrics = build_surface_and_metrics(
            spot, expiry_dates, option_chains, risk_free=r
        )

        fig3d, fig_hm = build_surface_figs(df_surface, ticker)
        fig_term = build_term_structure_fig(df_expiry_metrics, ticker)
        fig_smiles = build_smiles_fig(df_surface, ticker)
        fig_rr_bf = build_rr_bf_fig(df_expiry_metrics, ticker)
        fig_rr_vs_bf = build_rr_vs_bf_scatter(df_expiry_metrics, ticker)

        figures = [
            fig3d,
            fig_hm,
            fig_term,
            fig_smiles,
            fig_rr_bf,
            fig_rr_vs_bf,
        ]

        build_dashboard_html(figures, ticker, outfile)

        print("[INFO] Done.")

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

