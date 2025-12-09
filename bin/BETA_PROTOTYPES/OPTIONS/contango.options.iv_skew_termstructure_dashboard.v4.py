#!/usr/bin/env python3
# SCRIPTNAME: options.iv_skew_termstructure_dashboard.v1.py
# AUTHOR:    Michael Derby
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# DATE:      2025-11-26
#
# PURPOSE
# -------
# IV / Skew / Term-Structure dashboard using your standard loaders.
#
# - Uses YOUR existing data_retrieval.py and options_data_retrieval.py (never modifies them).
# - Pulls underlying spot from load_or_download_ticker().
# - Uses options_data_retrieval.ensure_option_chains_cached() +
#   load_all_cached_option_chains() to build a flat chain across expiries.
# - Computes per-expiry:
#     * IV_atm (closest strike to spot)
#     * 25Δ Call IV (IV_25C)
#     * 25Δ Put IV (IV_25P)
#     * RR25  = IV_25C - IV_25P
#     * BF25  = 0.5*(IV_25C + IV_25P) - IV_atm
# - Computes term-structure on IV_atm:
#     * Slope & convexity vs sqrt(TTM)
# - Classifies a simple volatility regime and suggests structure buckets.
# - Writes all numeric outputs to /dev/shm/OPTIONS_IV_SKEW_TERMSTRUCTURE/<TICKER>/<DATE>/
# - Generates Plotly HTML dashboards and opens each in a separate browser tab.
#
# USAGE
# -----
#   python3 options.iv_skew_termstructure_dashboard.v1.py SPY
#
# REQUIREMENTS
# ------------
# - plotly
# - pandas
# - numpy
# - Your data_retrieval.py and options_data_retrieval.py in PYTHONPATH.


import argparse
import datetime as dt
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# -------------------------------------------------------------------------
# Your standard loaders (DO NOT MODIFY THESE MODULES)
# -------------------------------------------------------------------------

try:
    from data_retrieval import load_or_download_ticker
except ImportError:
    print("[ERROR] Could not import data_retrieval. Ensure it is on PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

try:
    from options_data_retrieval import (
        ensure_option_chains_cached,
        load_all_cached_option_chains,
        list_cached_option_expirations,
    )
except ImportError:
    print("[ERROR] Could not import options_data_retrieval. Ensure it is on PYTHONPATH.", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def infer_option_type(row: pd.Series) -> Optional[str]:
    """
    Infer option type ('C' or 'P') from common columns.
    Your options_data_retrieval standardizes a 'type' column with 'call'/'put'.
    """
    # Prefer canonical 'type' column
    if "type" in row.index:
        v = str(row["type"]).lower()
        if v.startswith("c"):
            return "C"
        if v.startswith("p"):
            return "P"

    # Fallbacks if needed
    for col in ["optionType"]:
        if col in row.index:
            v = str(row[col]).upper()
            if v.startswith("C"):
                return "C"
            if v.startswith("P"):
                return "P"

    for col in ["contractSymbol", "contract_symbol", "symbol"]:
        if col in row.index:
            sym = str(row[col]).upper()
            if "C" in sym[-3:]:
                return "C"
            if "P" in sym[-3:]:
                return "P"

    return None


def parse_expiration_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has a proper 'expiration' datetime64 column.
    """
    if "expiration" in df.columns:
        col = "expiration"
    elif "expiry" in df.columns:
        col = "expiry"
    elif "expirationDate" in df.columns:
        col = "expirationDate"
    else:
        raise ValueError("No expiration column found (expected 'expiration'/'expiry'/'expirationDate').")

    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    if col != "expiration":
        df = df.rename(columns={col: "expiration"})
    return df


def compute_ttm_years(expiration: pd.Timestamp, asof: dt.date) -> float:
    """Compute time-to-maturity in years using ACT/365."""
    delta = (expiration.date() - asof).days
    return max(delta, 0) / 365.0


def compute_bs_delta(
    spot: float,
    strike: float,
    ttm: float,
    iv: float,
    opt_type: str,
    r: float = 0.02,
    q: float = 0.0,
) -> float:
    """
    Black-Scholes delta with continuous rates, given IV.

    opt_type: 'C' or 'P'
    """
    if spot <= 0 or strike <= 0 or ttm <= 0 or iv <= 0:
        return float("nan")

    try:
        sigma = iv
        sqrt_t = math.sqrt(ttm)
        d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * ttm) / (sigma * sqrt_t)
        if opt_type.upper() == "C":
            return math.exp(-q * ttm) * norm_cdf(d1)
        elif opt_type.upper() == "P":
            return -math.exp(-q * ttm) * norm_cdf(-d1)
        else:
            return float("nan")
    except Exception:
        return float("nan")


# -------------------------------------------------------------------------
# Data loading wrappers
# -------------------------------------------------------------------------


def load_spot_and_asof(ticker: str) -> Tuple[float, dt.date]:
    """
    Load underlying via data_retrieval.load_or_download_ticker(period='max')
    and return (spot, asof_date).
    """
    logging.info("Loading underlying OHLCV for %s via data_retrieval.load_or_download_ticker(period='max')...", ticker)
    df = load_or_download_ticker(ticker, period="max")
    if df is None or len(df) == 0:
        raise ValueError(f"Failed to load underlying data for {ticker}.")

    if "Close" not in df.columns:
        raise ValueError(f"Underlying data for {ticker} has no 'Close' column.")

    spot = float(df["Close"].iloc[-1])
    asof = df.index[-1].date()
    logging.info("Spot for %s: %.4f (as of %s)", ticker, spot, asof.isoformat())
    return spot, asof


def load_flat_option_chain(
    ticker: str,
    source: str = "yfinance",
    max_expiries: Optional[int] = None,
    ensure_remote: bool = True,
) -> pd.DataFrame:
    """
    Canonical way to build a flat chain across expirations using YOUR API:

      1) optionally ensure all remote expirations are cached:
            ensure_option_chains_cached(ticker, expirations=None, source=source)
      2) list_cached_option_expirations(ticker)
      3) load_all_cached_option_chains(ticker, expirations=...)

    Returns a single DataFrame with an 'expiration' column.
    """
    if ensure_remote:
        logging.info("Ensuring option chains are cached for %s via ensure_option_chains_cached...", ticker)
        ensure_option_chains_cached(
            ticker=ticker,
            expirations=None,     # means: all remote expirations
            source=source,
            force_refresh=False,
        )

    logging.info("Listing cached expirations for %s...", ticker)
    exps = list_cached_option_expirations(ticker, source=source)
    if not exps:
        raise ValueError(f"No cached option expirations found for {ticker} after ensure_option_chains_cached().")

    exps = sorted(exps)
    if max_expiries is not None and len(exps) > max_expiries:
        exps = exps[:max_expiries]
        logging.info("Using first %d expiries for analytics.", max_expiries)
    else:
        logging.info("Using all %d cached expiries for analytics.", len(exps))

    logging.info("Loading concatenated cached option chains for %s...", ticker)
    df = load_all_cached_option_chains(
        ticker=ticker,
        source=source,
        expirations=exps,
    )

    if df is None or len(df) == 0:
        raise ValueError(f"load_all_cached_option_chains() returned empty DataFrame for {ticker}.")

    df = parse_expiration_column(df)
    return df


# -------------------------------------------------------------------------
# Core analytics: IVatm / RR25 / BF25 / term structure
# -------------------------------------------------------------------------


def compute_per_expiry_smile_metrics(
    chain: pd.DataFrame,
    spot: float,
    asof: dt.date,
    r: float = 0.02,
    q: float = 0.0,
) -> pd.DataFrame:
    """
    For each expiration:
      - Compute TTM
      - Compute BS deltas (using current IV)
      - Find:
          IVatm  = IV at strike closest to spot
          IV25C  = IV at call with delta ~= +0.25
          IV25P  = IV at put  with delta ~= -0.25
          RR25   = IV25C - IV25P
          BF25   = 0.5 * (IV25C + IV25P) - IVatm
    """

    df = chain.copy()

    # Sanity: we need implied volatility and strike
    iv_cols = [c for c in df.columns if c.lower() in ("impliedvolatility", "iv", "implied_vol")]
    if not iv_cols:
        raise ValueError("No implied vol column found (expected 'impliedVolatility' / 'iv' / 'implied_vol').")
    iv_col = iv_cols[0]

    if "strike" not in df.columns:
        raise ValueError("Options chain has no 'strike' column.")

    df = parse_expiration_column(df)

    # Compute TTM and option type, then deltas
    df["ttm_years"] = df["expiration"].apply(lambda x: compute_ttm_years(x, asof))
    df["opt_type"] = df.apply(infer_option_type, axis=1)

    logging.info("Computing deltas from IV for each option row...")
    deltas = []
    for _, row in df.iterrows():
        opt_type = row["opt_type"]
        ttm = float(row["ttm_years"])
        strike = float(row["strike"])
        iv = float(row[iv_col]) if not pd.isna(row[iv_col]) else float("nan")
        if opt_type is None or np.isnan(iv) or ttm <= 0:
            deltas.append(float("nan"))
            continue
        d = compute_bs_delta(spot=spot, strike=strike, ttm=ttm, iv=iv, opt_type=opt_type, r=r, q=q)
        deltas.append(d)
    df["delta_bs"] = deltas

    metrics = []
    for exp, sub in df.groupby("expiration"):
        ttm = float(sub["ttm_years"].iloc[0])

        # Skip already expired
        if ttm <= 0:
            continue

        sub_valid = sub.dropna(subset=[iv_col, "strike"])
        if sub_valid.empty:
            continue

        sub_valid = sub_valid.copy()
        sub_valid["abs_moneyness"] = (sub_valid["strike"] - spot).abs()
        atm_row = sub_valid.loc[sub_valid["abs_moneyness"].idxmin()]
        iv_atm = float(atm_row[iv_col])

        # 25Δ call: opt_type == 'C', delta ~ +0.25
        calls = sub_valid[(sub_valid["opt_type"] == "C") & sub_valid["delta_bs"].notna()]
        puts = sub_valid[(sub_valid["opt_type"] == "P") & sub_valid["delta_bs"].notna()]

        iv_25c = float("nan")
        iv_25p = float("nan")

        if not calls.empty:
            calls = calls.copy()
            calls["delta_target_dist"] = (calls["delta_bs"] - 0.25).abs()
            c_row = calls.loc[calls["delta_target_dist"].idxmin()]
            iv_25c = float(c_row[iv_col])

        if not puts.empty:
            puts = puts.copy()
            puts["delta_target_dist"] = (puts["delta_bs"] + 0.25).abs()
            p_row = puts.loc[puts["delta_target_dist"].idxmin()]
            iv_25p = float(p_row[iv_col])

        rr25 = float("nan")
        bf25 = float("nan")
        if not np.isnan(iv_25c) and not np.isnan(iv_25p):
            rr25 = iv_25c - iv_25p
            bf25 = 0.5 * (iv_25c + iv_25p) - iv_atm

        metrics.append(
            {
                "expiration": exp,
                "ttm_years": ttm,
                "sqrt_ttm": math.sqrt(ttm) if ttm > 0 else 0.0,
                "IV_atm": iv_atm,
                "IV_25C": iv_25c,
                "IV_25P": iv_25p,
                "RR25": rr25,
                "BF25": bf25,
            }
        )

    if not metrics:
        raise ValueError("No valid per-expiry smile metrics computed (check input chain).")

    metrics_df = pd.DataFrame(metrics).sort_values("expiration").reset_index(drop=True)
    return metrics_df


def compute_term_structure_stats(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """
    Fit IV_atm as a function of sqrt(TTM):
        - linear fit: IV = a1 * sqrt_ttm + b1
        - quadratic fit: IV = a2 * sqrt_ttm^2 + b2 * sqrt_ttm + c2

    Returns:
        {
            "slope_linear": a1,
            "intercept_linear": b1,
            "slope_quad_at_zero": b2,  # derivative at 0
            "convexity_quad": a2
        }
    """
    stats = {
        "slope_linear": float("nan"),
        "intercept_linear": float("nan"),
        "slope_quad_at_zero": float("nan"),
        "convexity_quad": float("nan"),
    }

    df = metrics_df.dropna(subset=["sqrt_ttm", "IV_atm"])
    if len(df) < 2:
        return stats

    x = df["sqrt_ttm"].values
    y = df["IV_atm"].values

    # Linear
    try:
        a1, b1 = np.polyfit(x, y, 1)
        stats["slope_linear"] = float(a1)
        stats["intercept_linear"] = float(b1)
    except Exception:
        pass

    # Quadratic
    if len(df) >= 3:
        try:
            a2, b2, c2 = np.polyfit(x, y, 2)
            stats["slope_quad_at_zero"] = float(b2)
            stats["convexity_quad"] = float(a2)
        except Exception:
            pass

    return stats


def classify_vol_regime(metrics_df: pd.DataFrame, ts_stats: Dict[str, float]) -> Dict[str, Union[str, float]]:
    """
    Rough regime classifier:
      - Term structure: contango / backwardation / flat
      - Skew magnitude: tame / elevated / extreme
      - Curvature: low / medium / high
    """
    df = metrics_df.copy()
    result: Dict[str, Union[str, float, List[str]]] = {}

    # Term structure: near vs far IV_atm
    df_sorted = df.sort_values("ttm_years")
    if len(df_sorted) >= 2:
        iv_near = float(df_sorted["IV_atm"].iloc[0])
        iv_far = float(df_sorted["IV_atm"].iloc[-1])
        diff = iv_far - iv_near

        if diff > 0.01:
            term_regime = "contango (carry-friendly: far IV > near IV)"
        elif diff < -0.01:
            term_regime = "backwardation (stress: near IV > far IV)"
        else:
            term_regime = "flat / mixed"
    else:
        term_regime = "unknown (insufficient expiries)"

    result["term_structure_regime"] = term_regime

    # Skew magnitude (RR25)
    rr = df["RR25"].dropna()
    if len(rr) > 0:
        rr_mean = float(rr.mean())
        rr_abs_mean = float(rr.abs().mean())
        if rr_abs_mean < 0.03:
            skew_regime = "tame skew"
        elif rr_abs_mean < 0.07:
            skew_regime = "elevated skew"
        else:
            skew_regime = "extreme skew"
    else:
        rr_mean = float("nan")
        rr_abs_mean = float("nan")
        skew_regime = "unknown"

    result["rr25_mean"] = rr_mean
    result["rr25_abs_mean"] = rr_abs_mean
    result["skew_regime"] = skew_regime

    # Curvature via BF25
    bf = df["BF25"].dropna()
    if len(bf) > 0:
        bf_mean = float(bf.mean())
        if abs(bf_mean) < 0.01:
            curvature_regime = "low curvature"
        elif abs(bf_mean) < 0.03:
            curvature_regime = "medium curvature"
        else:
            curvature_regime = "high curvature"
    else:
        bf_mean = float("nan")
        curvature_regime = "unknown"

    result["bf25_mean"] = bf_mean
    result["curvature_regime"] = curvature_regime

    # Store term-structure stats as well
    for k, v in ts_stats.items():
        result[k] = v

    # Map to rough structure buckets
    suggestions: List[str] = []

    # Term-structure based
    if "contango" in term_regime:
        suggestions.append(
            "Term: contango -> consider calendars/diagonals (long far, short near) and other long-theta structures."
        )
    elif "backwardation" in term_regime:
        suggestions.append(
            "Term: backwardation -> near IV rich; be cautious with naked long gamma, prefer defined-risk spreads."
        )

    # Skew based
    if "extreme skew" in skew_regime:
        suggestions.append(
            "Skew: extreme -> puts rich vs calls; prefer put spreads or collars over naked puts; "
            "skew-selling risk reversals only with tail risk controls."
        )
    elif "elevated skew" in skew_regime:
        suggestions.append(
            "Skew: elevated -> consider put-spread collars, limited-risk risk reversals, or skew-financed call structures."
        )
    elif "tame skew" in skew_regime:
        suggestions.append(
            "Skew: tame -> less edge from skew trades; focus more on term-structure and directional gamma/theta trades."
        )

    # Curvature based
    if "high curvature" in curvature_regime:
        suggestions.append("Curvature: high -> butterflies/flies/ratio flies to monetize curvature.")
    elif "medium curvature" in curvature_regime:
        suggestions.append("Curvature: medium -> standard flies and broken-wing flies can be attractive.")
    elif "low curvature" in curvature_regime:
        suggestions.append("Curvature: low -> less edge from flies; simple verticals and calendars dominate.")

    result["structure_suggestions"] = suggestions
    return result


# -------------------------------------------------------------------------
# Plotly visuals
# -------------------------------------------------------------------------


def make_term_structure_figure(
    ticker: str,
    metrics_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics_df["expiration"],
            y=metrics_df["IV_atm"],
            mode="lines+markers",
            name="IV_atm",
        )
    )
    fig.update_layout(
        title=f"{ticker} - Term Structure (IV_atm vs Expiration)",
        xaxis_title="Expiration",
        yaxis_title="IV_atm",
        hovermode="x unified",
    )
    return fig


def make_rr25_figure(ticker: str, metrics_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics_df["expiration"],
            y=metrics_df["RR25"],
            name="RR25 (IV_25C - IV_25P)",
        )
    )
    fig.update_layout(
        title=f"{ticker} - 25Δ Risk Reversal (RR25) by Expiration",
        xaxis_title="Expiration",
        yaxis_title="RR25 (vol points)",
        hovermode="x unified",
    )
    return fig


def make_bf25_figure(ticker: str, metrics_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics_df["expiration"],
            y=metrics_df["BF25"],
            name="BF25 (0.5*(IV25C+IV25P) - IVatm)",
        )
    )
    fig.update_layout(
        title=f"{ticker} - 25Δ Butterfly (BF25) by Expiration",
        xaxis_title="Expiration",
        yaxis_title="BF25 (vol points)",
        hovermode="x unified",
    )
    return fig


def make_regime_text_figure(ticker: str, regime_info: Dict[str, Union[str, float, List[str]]]) -> go.Figure:
    """
    Simple text panel summarizing the regime classification and suggestions.
    """
    lines: List[str] = []
    lines.append(f"Ticker: {ticker}")
    lines.append("")
    lines.append(f"Term Structure Regime: {regime_info.get('term_structure_regime', 'n/a')}")
    lines.append(f"Skew Regime: {regime_info.get('skew_regime', 'n/a')}")
    lines.append(f"Curvature Regime: {regime_info.get('curvature_regime', 'n/a')}")

    rr_mean = regime_info.get("rr25_mean", float("nan"))
    rr_abs_mean = regime_info.get("rr25_abs_mean", float("nan"))
    bf_mean = regime_info.get("bf25_mean", float("nan"))

    def fmt(x: Union[float, str]) -> str:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "n/a"
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    lines.append("")
    lines.append(f"RR25 mean: {fmt(rr_mean)}")
    lines.append(f"|RR25| mean: {fmt(rr_abs_mean)}")
    lines.append(f"BF25 mean: {fmt(bf_mean)}")

    slope_lin = regime_info.get("slope_linear", float("nan"))
    conv_quad = regime_info.get("convexity_quad", float("nan"))
    lines.append("")
    lines.append(f"Term-Structure Slope (linear fit): {fmt(slope_lin)}")
    lines.append(f"Term-Structure Convexity (quadratic fit): {fmt(conv_quad)}")

    suggestions = regime_info.get("structure_suggestions", [])
    if suggestions:
        lines.append("")
        lines.append("Structure Buckets / Ideas:")
        for s in suggestions:
            lines.append(f"- {s}")

    text = "<br>".join(lines)
    fig = go.Figure()
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
    )
    fig.update_layout(
        title=f"{ticker} - Vol Regime Summary & Structure Buckets",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def build_output_dir(ticker: str) -> str:
    today_str = dt.date.today().isoformat()
    out_dir = os.path.join(
        "/dev/shm",
        "OPTIONS_IV_SKEW_TERMSTRUCTURE",
        ticker.upper(),
        today_str,
    )
    ensure_dir(out_dir)
    return out_dir


def save_figure(fig: go.Figure, out_dir: str, filename_prefix: str, auto_open: bool = True) -> None:
    html_path = os.path.join(out_dir, f"{filename_prefix}.html")
    png_path = os.path.join(out_dir, f"{filename_prefix}.png")

    pio.write_html(fig, file=html_path, auto_open=auto_open)
    try:
        fig.write_image(png_path)
    except Exception as e:
        logging.warning("Could not write PNG (%s): %s", png_path, e)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="IV / Skew / Term-Structure Dashboard using your standard data loaders."
    )
    parser.add_argument("ticker", type=str, help="Underlying ticker (e.g., IWM, SPY, NVDA)")
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.02,
        help="Risk-free rate for BS delta calc (default: 0.02).",
    )
    parser.add_argument(
        "--div-yield",
        type=float,
        default=0.0,
        help="Continuous dividend yield for BS delta calc (default: 0.0).",
    )
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=12,
        help="Max number of expiries to use (closest in time). Default: 12.",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    r = float(args.risk_free)
    q = float(args.div_yield)
    max_expiries = None if args.max_expiries <= 0 else args.max_expiries

    logging.info("=== OPTIONS_IV_SKEW_TERMSTRUCTURE Dashboard for %s ===", ticker)
    out_dir = build_output_dir(ticker)
    logging.info("Output directory: %s", out_dir)

    # 1) Load underlying & as-of
    spot, asof = load_spot_and_asof(ticker)

    # 2) Load options chain and flatten via your canonical helpers
    chain_df = load_flat_option_chain(ticker, source="yfinance", max_expiries=max_expiries, ensure_remote=True)
    logging.info("Loaded options chain: %d rows, %d columns", chain_df.shape[0], chain_df.shape[1])

    # 3) Compute per-expiry smile metrics
    logging.info("Computing per-expiry IVatm / RR25 / BF25 metrics...")
    metrics_df = compute_per_expiry_smile_metrics(chain=chain_df, spot=spot, asof=asof, r=r, q=q)
    logging.info("Computed smile metrics for %d expirations.", len(metrics_df))

    # 4) Compute term-structure statistics
    ts_stats = compute_term_structure_stats(metrics_df)

    # 5) Classify regime & build suggestions
    regime_info = classify_vol_regime(metrics_df, ts_stats)

    # 6) Persist metrics table
    metrics_csv_path = os.path.join(out_dir, f"{ticker}_iv_smile_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info("Saved metrics CSV to %s", metrics_csv_path)

    # 7) Plotly figures
    logging.info("Building Plotly figures...")
    fig_term = make_term_structure_figure(ticker, metrics_df)
    fig_rr25 = make_rr25_figure(ticker, metrics_df)
    fig_bf25 = make_bf25_figure(ticker, metrics_df)
    fig_regime = make_regime_text_figure(ticker, regime_info)

    # 8) Save & open each as HTML (separate browser tabs)
    logging.info("Writing HTML dashboards and opening in browser tabs...")
    save_figure(fig_term, out_dir, f"{ticker}_term_structure", auto_open=True)
    save_figure(fig_rr25, out_dir, f"{ticker}_rr25", auto_open=True)
    save_figure(fig_bf25, out_dir, f"{ticker}_bf25", auto_open=True)
    save_figure(fig_regime, out_dir, f"{ticker}_regime_summary", auto_open=True)

    logging.info("Done.")


if __name__ == "__main__":
    main()

