#!/usr/bin/env python3
# SCRIPTNAME: gamma.charmm.visualizer.v4.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Gamma & Vanna Exposure Visualizer using Black-Scholes.
#   - Uses options_data_retrieval.py to load cached options chains.
#   - Option D spot estimation (nearest ±0.5 delta) per expiration.
#   - Calculates Gamma & Vanna exposure (Greeks * Open Interest * Contract Size).
#   - Outputs interactive surfaces, heatmaps, and term structure slices.

import argparse
import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import math
import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# CONSTRAINT: Import local data retrieval module
try:
    import options_data_retrieval as optdr
except ImportError:
    print("Error: options_data_retrieval.py not found.")
    sys.exit(1)


# =============================================================================
# Helpers: timezone-normalization
# =============================================================================

def _to_naive_date(ts) -> pd.Timestamp:
    """
    Convert any timestamp-like object to timezone-naive, normalized date.
    Works for tz-aware and tz-naive inputs.
    """
    t = pd.to_datetime(ts)
    # If tz-aware, drop timezone (tz_convert(None) only if tz-aware)
    if getattr(t, "tzinfo", None) is not None:
        t = t.tz_convert(None)
    return t.normalize()


# =============================================================================
# Normal PDF / CDF (robust: no np.erf dependency)
# =============================================================================

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / SQRT_2PI


# Try SciPy first (fast and tested), fallback to math.erf + np.vectorize
try:
    from scipy.stats import norm as _norm_dist

    def _norm_cdf(x):
        return _norm_dist.cdf(x)

except Exception:
    def _norm_cdf(x):
        """
        Robust, dependency-light CDF using math.erf.
        math.erf is scalar, so we vectorize it.
        """
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


# =============================================================================
# Spot Estimation (Option D: nearest 0.5-delta)
# =============================================================================

def _mid_price(bid: pd.Series, ask: pd.Series, last: pd.Series) -> pd.Series:
    """
    Compute mid price: (bid+ask)/2 if possible, else fallback to last.
    """
    mid = (bid + ask) / 2.0
    mid = mid.where(~mid.isna(), last)
    return mid


def _estimate_spot_for_expiration(
    df_e: pd.DataFrame,
    exp_ts: pd.Timestamp,
    risk_free: float,
    div_yield: float,
    contract_size: int,
    today: pd.Timestamp,
) -> float:
    """
    Estimate spot price for a single expiration.
    Uses:
       1. Strike with minimal |call_mid - put_mid| (parity-based ATM)
       2. Nearest +0.5 delta call and -0.5 delta put
    """

    exp_ts = _to_naive_date(exp_ts)
    today = _to_naive_date(today)

    calls = df_e[df_e["type"] == "call"].copy()
    puts = df_e[df_e["type"] == "put"].copy()

    if calls.empty and puts.empty:
        return float(df_e["strike"].median())

    # Pair calls & puts by strike
    calls = calls.rename(columns={"bid": "bid_c", "ask": "ask_c", "lastPrice": "last_c"})
    puts = puts.rename(columns={"bid": "bid_p", "ask": "ask_p", "lastPrice": "last_p"})

    pair = pd.merge(
        calls[["strike", "bid_c", "ask_c", "last_c"]],
        puts[["strike", "bid_p", "ask_p", "last_p"]],
        on="strike",
        how="inner",
    )

    if pair.empty:
        return float(df_e["strike"].median())

    call_mid = _mid_price(pair["bid_c"], pair["ask_c"], pair["last_c"])
    put_mid = _mid_price(pair["bid_p"], pair["ask_p"], pair["last_p"])

    diff = (call_mid - put_mid).abs()
    idx_min = diff.idxmin()
    if pd.isna(idx_min):
        return float(df_e["strike"].median())

    S0 = float(pair.loc[idx_min, "strike"])

    # Compute deltas at S0
    T_days = max((exp_ts - today).days, 1)
    T = T_days / 365.0

    df_tmp = df_e.copy()
    df_tmp = df_tmp[df_tmp["impliedVolatility"].notna()]
    if df_tmp.empty:
        return S0

    S = S0
    K = df_tmp["strike"].astype(float).to_numpy()
    sigma = np.clip(df_tmp["impliedVolatility"].astype(float).to_numpy(), 1e-4, None)
    T_arr = np.full_like(sigma, T, dtype=float)
    S_arr = np.full_like(sigma, S, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = (np.log(S_arr / K) + (risk_free - div_yield + 0.5 * sigma ** 2) * T_arr) / (
            sigma * np.sqrt(T_arr)
        )

    phi1 = _norm_cdf(d1)
    is_call = (df_tmp["type"].to_numpy() == "call")

    disc_q = math.exp(-div_yield * T)
    delta = np.empty_like(sigma)
    delta[is_call] = disc_q * phi1[is_call]          # call delta
    delta[~is_call] = disc_q * (phi1[~is_call] - 1)  # put delta

    strike_call = None
    strike_put = None

    if np.any(is_call):
        d_call = np.abs(delta[is_call] - 0.5)
        idx_rel = int(np.nanargmin(d_call))
        strike_call = float(df_tmp.loc[is_call].iloc[idx_rel]["strike"])

    if np.any(~is_call):
        d_put = np.abs(delta[~is_call] + 0.5)
        idx_rel = int(np.nanargmin(d_put))
        strike_put = float(df_tmp.loc[~is_call].iloc[idx_rel]["strike"])

    candidates = [s for s in [strike_call, strike_put] if s is not None]
    if not candidates:
        return S0

    return float(np.mean(candidates))


def estimate_spot_per_expiration(
    df: pd.DataFrame,
    risk_free: float,
    div_yield: float,
    contract_size: int,
    today: Optional[pd.Timestamp] = None,
) -> Dict[pd.Timestamp, float]:
    """
    Estimate spot for each expiration using Option D.
    """

    if today is None:
        today = pd.Timestamp.utcnow()
    today = _to_naive_date(today)

    spots: Dict[pd.Timestamp, float] = {}
    df = df.copy()
    # Make expiration tz-naive dates
    df["expiration"] = pd.to_datetime(df["expiration"]).map(_to_naive_date)

    for exp_ts in sorted(df["expiration"].unique()):
        exp_ts = _to_naive_date(exp_ts)
        df_e = df[df["expiration"] == exp_ts]
        if df_e.empty:
            continue

        S_est = _estimate_spot_for_expiration(
            df_e=df_e,
            exp_ts=exp_ts,
            risk_free=risk_free,
            div_yield=div_yield,
            contract_size=contract_size,
            today=today,
        )
        spots[exp_ts] = S_est

    return spots


# =============================================================================
# Greeks + Exposure
# =============================================================================

def compute_greeks_and_exposure(
    df: pd.DataFrame,
    spots: Dict[pd.Timestamp, float],
    risk_free: float,
    div_yield: float,
    contract_size: int,
    today: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:

    if today is None:
        today = pd.Timestamp.utcnow()
    today = _to_naive_date(today)

    df = df.copy()
    df["expiration"] = pd.to_datetime(df["expiration"]).map(_to_naive_date)

    df["spot_est"] = df["expiration"].map(spots)
    df = df[df["spot_est"].notna()]

    if df.empty:
        return pd.DataFrame(columns=["expiration", "strike", "gamma_exposure", "vanna_exposure"])

    T_days = (df["expiration"] - today).dt.days.astype(float).clip(lower=1.0)
    T = T_days / 365.0

    S = df["spot_est"].astype(float).to_numpy()
    K = df["strike"].astype(float).to_numpy()
    sigma = np.clip(df["impliedVolatility"].astype(float).to_numpy(), 1e-4, None)
    T_arr = T.to_numpy()

    valid = (S > 0) & (K > 0) & (sigma > 0) & (T_arr > 0)
    if not np.any(valid):
        return pd.DataFrame(columns=["expiration", "strike", "gamma_exposure", "vanna_exposure"])

    S_v = S[valid]
    K_v = K[valid]
    sig_v = sigma[valid]
    T_v = T_arr[valid]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = (np.log(S_v / K_v) + (risk_free - div_yield + 0.5 * sig_v ** 2) * T_v) / (
            sig_v * np.sqrt(T_v)
        )
        d2 = d1 - sig_v * np.sqrt(T_v)

    phi = _norm_pdf(d1)
    disc_q = np.exp(-div_yield * T_v)

    gamma = disc_q * phi / (S_v * sig_v * np.sqrt(T_v))
    vanna = -disc_q * phi * d2 / sig_v

    gamma_full = np.zeros_like(sigma)
    vanna_full = np.zeros_like(sigma)

    gamma_full[valid] = gamma
    vanna_full[valid] = vanna

    df["gamma"] = gamma_full
    df["vanna"] = vanna_full

    if "openInterest" in df.columns:
        oi = df["openInterest"].fillna(0).astype(float)
    else:
        oi = pd.Series(0.0, index=df.index)

    df["gamma_exposure"] = df["gamma"] * oi * contract_size
    df["vanna_exposure"] = df["vanna"] * oi * contract_size

    agg = (
        df.groupby(["expiration", "strike"], as_index=False)[["gamma_exposure", "vanna_exposure"]]
        .sum()
    )

    return agg


# =============================================================================
# Plotting
# =============================================================================

def pivot_surface(df: pd.DataFrame, value_col: str):
    exps = np.sort(df["expiration"].unique())
    strikes = np.sort(df["strike"].unique())
    Z = np.zeros((len(exps), len(strikes)))

    df_idx = df.set_index(["expiration", "strike"])[value_col]

    for i, e in enumerate(exps):
        for j, k in enumerate(strikes):
            Z[i, j] = df_idx.get((e, k), np.nan)

    return exps, strikes, Z


def plot_surface(exps, strikes, Z, title: str, outdir: str, fname: str):
    Y = np.arange(len(exps))
    X, Yg = np.meshgrid(strikes, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Yg, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiration index")
    ax.set_zlabel("Exposure")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.6, aspect=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=160)
    plt.close(fig)


def plot_heatmap(exps, strikes, Z, title: str, outdir: str, fname: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(Z, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(exps)))
    # Robust conversion: numpy.datetime64 / Timestamp / others
    ax.set_yticklabels([str(pd.Timestamp(e).date()) for e in exps])
    step = max(1, len(strikes) // 12)
    ax.set_xticks(np.arange(len(strikes))[::step])
    ax.set_xticklabels([str(int(s)) for s in strikes[::step]])
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiration")
    ax.set_title(title)
    fig.colorbar(im, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=160)
    plt.close(fig)


def plot_term_slice(df: pd.DataFrame, value_col: str, strike: float, outdir: str, fname: str):
    strikes = np.sort(df["strike"].unique())
    if len(strikes) == 0:
        return
    nearest = float(strikes[np.argmin(np.abs(strikes - strike))])
    dsub = df[df["strike"] == nearest].sort_values("expiration")
    if dsub.empty:
        return

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(dsub["expiration"], dsub[value_col], marker="o")
    ax.set_title(f"{value_col} term structure (nearest {nearest})")
    ax.set_xlabel("Expiration")
    ax.set_ylabel("Exposure")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=160)
    plt.close(fig)


def animate_gamma_sweep(exps, strikes, Z, outdir: str):
    try:
        import matplotlib.animation as animation
    except Exception as e:
        print("Animation skipped:", e)
        return

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    line, = ax.plot([], [])
    ax.set_xlim(min(strikes), max(strikes))
    gmin, gmax = np.nanmin(Z), np.nanmax(Z)
    pad = 0.1 * (gmax - gmin if gmax > gmin else 1)
    ax.set_ylim(gmin - pad, gmax + pad)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Gamma Exposure")
    title = ax.set_title("Gamma by Strike — ")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        y = Z[frame, :]
        line.set_data(strikes, y)
        title.set_text(f"Gamma by Strike — {str(pd.Timestamp(exps[frame]).date())}")
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=len(exps), init_func=init, blit=True
    )
    ani.save(os.path.join(outdir, "gamma_sweep.mp4"), fps=2)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--ensure_remote", action="store_true")
    ap.add_argument("--max_expirations", type=int, default=None)
    ap.add_argument("--risk_free", type=float, default=0.05)
    ap.add_argument("--div_yield", type=float, default=0.0)
    ap.add_argument("--contract_size", type=int, default=100)
    ap.add_argument("--strike_slice", type=float, default=None)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--outdir", default=None)

    args = ap.parse_args()
    ticker = args.ticker.upper()

    # Determine output directory
    # CONSTRAINT: Ensures output is in /dev/shm (BASE_DATA_PATH defaults to /dev/shm/data)
    if args.outdir is None:
        base_data = os.environ.get("BASE_DATA_PATH", "/dev/shm/data")
        today_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        outdir = os.path.join(base_data, today_str, f"{ticker}_OPTIONS_GAMMA_VANNA")
    else:
        outdir = args.outdir

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Optional: download & cache all remote expirations
    if args.ensure_remote:
        optdr.ensure_option_chains_cached(
            ticker=ticker,
            expirations=None,
            source=args.source,
            force_refresh=False,
        )

    # Load cached chains
    df = optdr.load_all_cached_option_chains(ticker, source=args.source)
    if df.empty:
        print("No cached options found. Try --ensure_remote.")
        sys.exit(1)

    df["expiration"] = pd.to_datetime(df["expiration"]).map(_to_naive_date)
    exps_sorted = np.sort(df["expiration"].unique())

    if args.max_expirations is not None and len(exps_sorted) > args.max_expirations:
        keep = set(exps_sorted[: args.max_expirations])
        df = df[df["expiration"].isin(keep)]

    if df.empty:
        print("No data after expiration filter.")
        sys.exit(1)

    # Estimate spot
    spots = estimate_spot_per_expiration(
        df,
        risk_free=args.risk_free,
        div_yield=args.div_yield,
        contract_size=args.contract_size,
    )
    if not spots:
        print("Failed to estimate spot for all expirations.")
        sys.exit(1)

    # Greeks & exposures
    agg = compute_greeks_and_exposure(
        df,
        spots=spots,
        risk_free=args.risk_free,
        div_yield=args.div_yield,
        contract_size=args.contract_size,
    )
    if agg.empty:
        print("No exposures computed.")
        sys.exit(1)

    # Surfaces
    exps, strikes, G = pivot_surface(agg, "gamma_exposure")
    _, _, V = pivot_surface(agg, "vanna_exposure")

    # Plots
    plot_surface(exps, strikes, G, f"{ticker} Gamma Exposure Surface", outdir, "gamma_surface.png")
    plot_heatmap(exps, strikes, G, f"{ticker} Gamma Exposure Heatmap", outdir, "gamma_heatmap.png")
    plot_surface(exps, strikes, V, f"{ticker} Vanna Exposure Surface", outdir, "vanna_surface.png")
    plot_heatmap(exps, strikes, V, f"{ticker} Vanna Exposure Heatmap", outdir, "vanna_heatmap.png")

    if args.strike_slice is not None:
        plot_term_slice(agg, "gamma_exposure", args.strike_slice, outdir, f"gamma_term_{args.strike_slice}.png")
        plot_term_slice(agg, "vanna_exposure", args.strike_slice, outdir, f"vanna_term_{args.strike_slice}.png")

    if args.animate:
        animate_gamma_sweep(exps, strikes, G, outdir)

    print(f"Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
