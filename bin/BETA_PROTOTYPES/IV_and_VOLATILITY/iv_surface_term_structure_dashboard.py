#!/usr/bin/env python3
# SCRIPTNAME: ok.iv_surface_term_structure_dashboard.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
SCRIPTNAME: iv_surface_term_structure_dashboard.py

Description:
    Standalone IV surface and term-structure visualizer that implements
    the "3-number read" (Level, Skew, Curvature) across expiries and
    multiple creative visualizations for a professional options trader.

    Features:
      - Downloads options chains via yfinance for a single ticker.
      - Computes approximate Black–Scholes delta per option.
      - Normalizes by spot to get moneyness buckets.
      - Computes, per expiry:
          * Level (L): ATM IV
          * Skew (S): 25Δ Put IV - 25Δ Call IV
          * Curvature (C): average(25Δ Put IV, 25Δ Call IV) - ATM IV
      - Builds Plotly dashboards:
          1) 3D IV Surface (moneyness × time-to-expiry × IV)
          2) IV Heatmap (expiry vs moneyness buckets)
          3) Smile Comparison (short / medium / long expiry)
          4) Term Structure (ATM, 25Δ P, 25Δ C)
          5) Level / Skew / Curvature Regime Panel

    Usage:
        python3 iv_surface_term_structure_dashboard.py --ticker SPY
        python3 iv_surface_term_structure_dashboard.py --ticker NVDA --max-expiries 8

    Dependencies:
        pip install yfinance plotly pandas numpy

    Notes:
        - Intended as a self-contained, "just works" script.
        - No external data_retrieval / STEGO requirements.
"""

import argparse
import os
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po
import plotly.express as px
import yfinance as yf


# -----------------------------
# Black–Scholes Delta Helpers
# -----------------------------
def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erf; avoids SciPy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(spot: float,
             strike: float,
             t_years: float,
             r: float,
             iv: float,
             option_type: str) -> float:
    """
    Approximate Black–Scholes delta for a single option.
    option_type: 'C' or 'P'
    """
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        # Fallback: rough delta by simple moneyness
        if option_type.upper() == "C":
            return max(0.0, min(1.0, 0.5 * (spot / strike)))
        else:
            return max(-1.0, min(0.0, -0.5 * (strike / spot)))

    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    except Exception:
        if option_type.upper() == "C":
            return 0.5
        else:
            return -0.5

    if option_type.upper() == "C":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


# -----------------------------
# Data Loading & Processing
# -----------------------------
def load_options_surface(ticker: str, max_expiries: int = 8) -> pd.DataFrame:
    """
    Download options chains for a ticker and build a unified DataFrame:
    columns: ['ticker','expiry','option_type','strike','iv','spot',
              'ttm_days','ttm_years','moneyness','delta', ...]
    """
    tk = yf.Ticker(ticker)
    # Spot from latest close
    hist = tk.history(period="2d")
    if hist.empty:
        raise RuntimeError(f"No price history available for {ticker}")
    spot = float(hist["Close"].dropna().iloc[-1])

    expiries = tk.options
    if not expiries:
        raise RuntimeError(f"No listed options expiries for {ticker}")

    expiries = expiries[:max_expiries]
    rows = []
    utc_now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    for e in expiries:
        try:
            chain = tk.option_chain(e)
        except Exception as exc:
            print(f"[WARN] Failed to load chain for {e}: {exc}")
            continue

        expiry_dt = pd.to_datetime(e).to_pydatetime().replace(tzinfo=timezone.utc)
        ttm_days = max((expiry_dt - utc_now).days, 0)
        ttm_years = max(ttm_days / 365.0, 0.0)

        for df_opt, opt_type in [(chain.calls, "C"), (chain.puts, "P")]:
            if df_opt is None or df_opt.empty:
                continue

            df = df_opt.copy()
            df["option_type"] = opt_type
            df["expiry"] = expiry_dt
            df["ticker"] = ticker
            df["spot"] = spot
            df["ttm_days"] = ttm_days
            df["ttm_years"] = ttm_years

            # yfinance column name is impliedVolatility
            if "impliedVolatility" in df.columns:
                df["iv"] = df["impliedVolatility"].astype(float)
            else:
                # Fallback: synthetic 20% if missing (unlikely)
                df["iv"] = 0.20

            rows.append(df)

    if not rows:
        raise RuntimeError(f"No option rows collected for {ticker}")

    full = pd.concat(rows, ignore_index=True)

    # Basic filters & derived fields
    full = full.dropna(subset=["iv", "strike"])
    full = full[full["iv"] > 0]
    full = full[full["strike"] > 0]

    full["moneyness"] = full["strike"] / full["spot"]
    full["moneyness_bucket"] = full["moneyness"].round(2)

    # Compute deltas
    r = 0.03  # simple flat rate assumption
    deltas = []
    for _, row in full.iterrows():
        d = bs_delta(
            spot=float(row["spot"]),
            strike=float(row["strike"]),
            t_years=float(row["ttm_years"]),
            r=r,
            iv=float(row["iv"]),
            option_type=row["option_type"],
        )
        deltas.append(d)
    full["delta"] = deltas

    return full


def compute_l_s_c_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Level (L), Skew (S), Curvature (C) for each expiry.

    Definitions:
      L = ATM IV
      S = IV_25Δ_put - IV_25Δ_call
      C = 0.5 * (IV_25Δ_put + IV_25Δ_call) - IV_ATM
    """
    metrics = []

    for expiry, sub in df.groupby("expiry"):
        sub = sub.copy()
        spot = float(sub["spot"].iloc[0])
        ttm_days = float(sub["ttm_days"].iloc[0])
        ttm_years = float(sub["ttm_years"].iloc[0])

        # ATM: strike closest to spot
        sub["dist_atm"] = (sub["strike"] - spot).abs()
        atm_row = sub.loc[sub["dist_atm"].idxmin()]
        iv_atm = float(atm_row["iv"])

        # 25D Call (closest delta to +0.25)
        calls = sub[sub["option_type"] == "C"].copy()
        puts = sub[sub["option_type"] == "P"].copy()

        iv_25c = np.nan
        iv_25p = np.nan

        if not calls.empty:
            calls["d_call"] = (calls["delta"] - 0.25).abs()
            row_25c = calls.loc[calls["d_call"].idxmin()]
            iv_25c = float(row_25c["iv"])

        if not puts.empty:
            puts["d_put"] = (puts["delta"] + 0.25).abs()  # target -0.25
            row_25p = puts.loc[puts["d_put"].idxmin()]
            iv_25p = float(row_25p["iv"])

        # Level / Skew / Curvature
        L = iv_atm
        S = np.nan
        C = np.nan

        if not np.isnan(iv_25c) and not np.isnan(iv_25p):
            S = iv_25p - iv_25c
            C = 0.5 * (iv_25p + iv_25c) - iv_atm

        metrics.append(
            {
                "expiry": expiry,
                "ttm_days": ttm_days,
                "ttm_years": ttm_years,
                "L_atm_iv": L,
                "iv_25d_put": iv_25p,
                "iv_25d_call": iv_25c,
                "S_skew": S,
                "C_curvature": C,
            }
        )

    met_df = pd.DataFrame(metrics).sort_values("expiry").reset_index(drop=True)
    return met_df


def suggest_trade_from_metrics(met: pd.DataFrame) -> str:
    """
    Very simple rule-based trade suggestion based on front/back L/S/C.
    This is intentionally high-level; you would tailor sizing, strikes, etc.
    """
    if met.empty:
        return "No metrics available to suggest a trade."

    # front = shortest expiry, back = longest
    met_sorted = met.sort_values("ttm_days")
    front = met_sorted.iloc[0]
    back = met_sorted.iloc[-1]

    L_front = front["L_atm_iv"]
    L_back = back["L_atm_iv"]
    S_front = front["S_skew"]
    C_front = front["C_curvature"]

    text_lines = []
    text_lines.append(f"Front expiry T+{int(front['ttm_days'])}d vs Back expiry T+{int(back['ttm_days'])}d.")
    text_lines.append(f"ATM IV front={L_front:.2%}, back={L_back:.2%}. Skew front={S_front:.2%}, Curvature front={C_front:.2%}.")

    # Rules (very simplified)
    if L_front < L_back and abs(L_back - L_front) > 0.03:
        text_lines.append(
            "- Back-month IV richer than front by >3 vol points → "
            "consider ATM call/put calendars or diagonals (buy back, sell front)."
        )
    elif L_front > L_back and abs(L_front - L_back) > 0.03:
        text_lines.append(
            "- Front-month IV richer than back by >3 vol points → "
            "consider front-loaded spreads or short-term premium selling with defined risk."
        )

    if not np.isnan(S_front):
        if S_front < -0.05:
            text_lines.append(
                "- Downside skew is very negative (puts rich vs calls) → "
                "bear/bull put spreads or risk-reversals (sell call, buy put) if directionally bearish."
            )
        elif S_front > 0.02:
            text_lines.append(
                "- Skew is positive (calls rich vs puts) → "
                "call spreads / call diagonals, or selling upside skew vs long delta."
            )

    if not np.isnan(C_front):
        if C_front > 0.03:
            text_lines.append(
                "- Curvature high: wings expensive vs ATM → "
                "favor ATM-centric structures (calendars/straddles) over long wings."
            )
        elif C_front < -0.01:
            text_lines.append(
                "- Curvature low/negative: wings relatively cheap → "
                "consider flies or condors to buy cheap wings and sell the middle."
            )

    if len(text_lines) == 2:
        text_lines.append(
            "- No extreme regime flags. Consider vanilla directional spreads aligned with your macro/flow view."
        )

    return "\n".join(text_lines)


# -----------------------------
# Plotting Helpers
# -----------------------------
def make_iv_surface_3d(df: pd.DataFrame, outdir: str, ticker: str) -> str:
    """
    3D surface: x = moneyness bucket, y = ttm_days, z = IV
    """
    pivot = df.pivot_table(
        index="ttm_days",
        columns="moneyness_bucket",
        values="iv",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)

    x = pivot.columns.values
    y = pivot.index.values
    z = pivot.values

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Viridis",
                colorbar=dict(title="IV"),
                contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            )
        ]
    )
    fig.update_layout(
        title=f"{ticker} Implied Volatility Surface (Moneyness vs Days to Expiry)",
        scene=dict(
            xaxis_title="Moneyness (K / S)",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility",
        ),
        template="plotly_dark",
    )

    fname = os.path.join(outdir, f"{ticker}_iv_surface_3d.html")
    po.plot(fig, filename=fname, auto_open=False)
    return fname


def make_iv_heatmap(df: pd.DataFrame, outdir: str, ticker: str) -> str:
    """
    Heatmap of IV vs expiry & moneyness bucket.
    """
    pivot = df.pivot_table(
        index="expiry",
        columns="moneyness_bucket",
        values="iv",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.values,
                y=[d.date().isoformat() for d in pivot.index],
                coloraxis="coloraxis",
            )
        ]
    )
    fig.update_layout(
        title=f"{ticker} IV Heatmap (Expiry vs Moneyness)",
        xaxis_title="Moneyness (K / S)",
        yaxis_title="Expiry Date",
        coloraxis=dict(colorscale="Turbo", colorbar=dict(title="IV")),
        template="plotly_dark",
    )

    fname = os.path.join(outdir, f"{ticker}_iv_heatmap.html")
    po.plot(fig, filename=fname, auto_open=False)
    return fname


def make_smile_comparison(df: pd.DataFrame, outdir: str, ticker: str) -> str:
    """
    Compare smiles for short/mid/long expiries.
    """
    expiries_sorted = sorted(df["expiry"].unique())
    if len(expiries_sorted) == 0:
        return ""

    # Pick short, mid, long (allow duplicates if few expiries)
    short_exp = expiries_sorted[0]
    mid_exp = expiries_sorted[len(expiries_sorted) // 2]
    long_exp = expiries_sorted[-1]

    smiles = []
    label_map = {
        short_exp: f"Short ({short_exp.date()})",
        mid_exp: f"Mid ({mid_exp.date()})",
        long_exp: f"Long ({long_exp.date()})",
    }

    for exp in [short_exp, mid_exp, long_exp]:
        sub = df[df["expiry"] == exp].copy()
        sub = sub.sort_values("moneyness")
        sub["expiry_label"] = label_map[exp]
        smiles.append(sub[["moneyness", "iv", "expiry_label"]])

    smiles_df = pd.concat(smiles, ignore_index=True)

    fig = px.line(
        smiles_df,
        x="moneyness",
        y="iv",
        color="expiry_label",
        title=f"{ticker} Smile Comparison (Short / Mid / Long)",
    )
    fig.update_layout(
        xaxis_title="Moneyness (K / S)",
        yaxis_title="Implied Volatility",
        template="plotly_dark",
    )

    fname = os.path.join(outdir, f"{ticker}_smile_comparison.html")
    po.plot(fig, filename=fname, auto_open=False)
    return fname


def make_term_structure(met: pd.DataFrame, outdir: str, ticker: str) -> str:
    """
    Term structure lines: ATM IV, 25Δ Put, 25Δ Call vs time-to-expiry.
    """
    met = met.sort_values("ttm_days")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=met["ttm_days"],
            y=met["L_atm_iv"],
            mode="lines+markers",
            name="ATM IV (L)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=met["ttm_days"],
            y=met["iv_25d_put"],
            mode="lines+markers",
            name="25Δ Put IV",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=met["ttm_days"],
            y=met["iv_25d_call"],
            mode="lines+markers",
            name="25Δ Call IV",
        )
    )

    fig.update_layout(
        title=f"{ticker} Term Structure: ATM vs 25Δ Wings",
        xaxis_title="Days to Expiry",
        yaxis_title="Implied Volatility",
        template="plotly_dark",
    )

    fname = os.path.join(outdir, f"{ticker}_term_structure.html")
    po.plot(fig, filename=fname, auto_open=False)
    return fname


def make_lsc_regime_panel(met: pd.DataFrame, outdir: str, ticker: str) -> str:
    """
    Panel-style figure with L, S, C stacked for regime reading.
    """
    met = met.sort_values("ttm_days")

    fig = make_subplots_lsc(met, ticker)
    fname = os.path.join(outdir, f"{ticker}_l_s_c_regimes.html")
    po.plot(fig, filename=fname, auto_open=False)
    return fname


def make_subplots_lsc(met: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Helper: L/S/C stacked subplot figure.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Level (ATM IV)", "Skew (25Δ Put - 25Δ Call)", "Curvature"),
    )

    x = met["ttm_days"]

    fig.add_trace(
        go.Scatter(x=x, y=met["L_atm_iv"], mode="lines+markers", name="Level L"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=met["S_skew"], mode="lines+markers", name="Skew S"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=met["C_curvature"], mode="lines+markers", name="Curvature C"),
        row=3,
        col=1,
    )

    fig.update_xaxes(title_text="Days to Expiry", row=3, col=1)
    fig.update_yaxes(title_text="IV", row=1, col=1)
    fig.update_yaxes(title_text="Skew (ΔIV)", row=2, col=1)
    fig.update_yaxes(title_text="Curvature (ΔIV)", row=3, col=1)

    fig.update_layout(
        title=f"{ticker} Level / Skew / Curvature Regime View",
        showlegend=False,
        template="plotly_dark",
        height=800,
    )

    return fig


def build_main_dashboard(df: pd.DataFrame, met: pd.DataFrame, suggestion: str, outdir: str, ticker: str) -> str:
    """
    Build a compact "main" dashboard combining:
      - Term Structure (L, 25Δ P, 25Δ C)
      - L/S/C small multiples
      - Text box with rule-based trade suggestion
    """
    from plotly.subplots import make_subplots

    met = met.sort_values("ttm_days")
    x = met["ttm_days"]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"colspan": 2, "type": "domain"}, None],
        ],
        subplot_titles=(
            "Term Structure (ATM & Wings)",
            "L / S / C Snapshots",
            "Rule-based Trade Commentary",
        ),
        vertical_spacing=0.12,
    )

    # Term structure
    fig.add_trace(
        go.Scatter(x=x, y=met["L_atm_iv"], mode="lines+markers", name="ATM IV"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=met["iv_25d_put"], mode="lines+markers", name="25Δ Put IV"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=met["iv_25d_call"], mode="lines+markers", name="25Δ Call IV"),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Days to Expiry", row=1, col=1)
    fig.update_yaxes(title_text="IV", row=1, col=1)

    # L/S/C "spark" lines
    fig.add_trace(
        go.Scatter(x=x, y=met["L_atm_iv"], mode="lines+markers", name="L (Level)"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=x, y=met["S_skew"], mode="lines+markers", name="S (Skew)"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=x, y=met["C_curvature"], mode="lines+markers", name="C (Curvature)"),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Days to Expiry", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    # Text box for suggestion: use annotation in "domain" subplot
    fig.add_annotation(
        text="<br>".join(suggestion.splitlines()),
        x=0.5,
        y=0.0,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12),
    )

    fig.update_layout(
        title=f"{ticker} IV Term Structure & Regime Dashboard",
        height=900,
        template="plotly_dark",
    )

    fname = os.path.join(outdir, f"{ticker}_main_dashboard.html")
    po.plot(fig, filename=fname, auto_open=True)
    return fname


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="IV Surface & Term Structure Dashboard")
    parser.add_argument("--ticker", type=str, default="SPY", help="Underlying ticker symbol (default: SPY)")
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=8,
        help="Maximum number of expiries to load (default: 8)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for HTML plots (default: ./iv_surface_dashboard_<TICKER>/)",
    )

    args = parser.parse_args()
    ticker = args.ticker.upper()
    max_expiries = args.max_expiries

    # Output directory
    if args.outdir is None:
        outdir = os.path.join(os.getcwd(), f"iv_surface_dashboard_{ticker}")
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Loading options surface for {ticker} (up to {max_expiries} expiries)...")
    df = load_options_surface(ticker, max_expiries=max_expiries)
    print(f"[INFO] Loaded {len(df)} option rows.")

    # Compute metrics
    print("[INFO] Computing Level / Skew / Curvature metrics...")
    met = compute_l_s_c_metrics(df)

    # Save raw data and metrics to CSV for future use
    df.to_csv(os.path.join(outdir, f"{ticker}_options_surface_raw.csv"), index=False)
    met.to_csv(os.path.join(outdir, f"{ticker}_expiry_metrics_L_S_C.csv"), index=False)

    # Generate rule-based suggestion string
    suggestion = suggest_trade_from_metrics(met)
    print("\n=== RULE-BASED TRADE COMMENTARY ===")
    print(suggestion)
    print("===================================\n")

    # Build visualizations
    print("[INFO] Building 3D IV surface...")
    make_iv_surface_3d(df, outdir, ticker)

    print("[INFO] Building IV heatmap...")
    make_iv_heatmap(df, outdir, ticker)

    print("[INFO] Building smile comparison (short/mid/long)...")
    make_smile_comparison(df, outdir, ticker)

    print("[INFO] Building term structure view...")
    make_term_structure(met, outdir, ticker)

    print("[INFO] Building L/S/C regime panel...")
    make_lsc_regime_panel(met, outdir, ticker)

    print("[INFO] Building main dashboard (will auto-open in browser)...")
    main_html = build_main_dashboard(df, met, suggestion, outdir, ticker)

    print(f"[DONE] All plots written under: {outdir}")
    print(f"[DONE] Main dashboard: {main_html}")


if __name__ == "__main__":
    main()

