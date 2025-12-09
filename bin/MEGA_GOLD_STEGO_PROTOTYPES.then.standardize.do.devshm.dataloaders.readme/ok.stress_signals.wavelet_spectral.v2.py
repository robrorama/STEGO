#!/usr/bin/env python3
# SCRIPTNAME: stress_signals.wavelet_spectral.v1.2.py
# AUTHOR: Michael Derby
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# PURPOSE:
#   Wavelet-change-point fusion + Spectral tilt stress signals
#   - Uses YOUR existing data_retrieval.py (unchanged).
#   - Pulls OHLCV for a single ticker.
#   - Computes:
#       * Wavelet energy (CWT) on log-returns + change-points (ruptures).
#       * Spectral "tilt" (high-frequency power share) + vol-of-vol in sliding windows.
#       * Robust z-scores via MAD (no statsmodels needed).
#   - Flags alerts when both high-frequency share and vol-of-vol are extreme.
#   - Optionally confirms with ΔVIX z-score (if ^VIX is available via data_retrieval).
#   - Writes all outputs into /dev/shm/STRESS_SIGNALS/{TICKER}/...
#   - Opens three Plotly HTML dashboards in separate browser tabs:
#       * Price + stress markers
#       * Wavelet energy + change-points
#       * Spectral tilt + alerts

import os
import sys
import argparse
import json
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

import pywt
import ruptures as rpt

from scipy.signal import welch, get_window
from scipy.integrate import trapezoid

import plotly.graph_objs as go

# --- local imports (your canonical loader) ---
sys.path.append(os.getcwd())
import data_retrieval as dr  # do not modify this file


# ==============================
# Utility & Robust Statistics
# ==============================

def ensure_dirs(base: str) -> None:
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "csv"), exist_ok=True)
    os.makedirs(os.path.join(base, "html"), exist_ok=True)


def robust_z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Robust z-score using median and MAD (no external dependencies).

    z_i = (x_i - median(x)) / MAD,  MAD = median(|x - median(x)|) * 1.4826

    If MAD == 0 (flat series), fall back to std deviation.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < eps:
        std = np.nanstd(x)
        if not np.isfinite(std) or std < eps:
            std = eps
        return (x - med) / std
    return (x - med) / mad


# ==============================
# Wavelet Energy + Change-Points
# ==============================

def cwt_energy(returns: np.ndarray, wavelet: str = "morl", scales=None):
    """
    Continuous Wavelet Transform (CWT) energy over returns.

    returns: 1D array of returns.
    wavelet: mother wavelet (default "morl").
    scales:  array of scales; if None, use dyadic-like [4, 8, ..., 128].
    """
    if scales is None:
        scales = np.unique((2 ** np.arange(2, 8)).astype(int))

    if returns.size == 0:
        return np.array([]), scales

    coefs, _ = pywt.cwt(returns, scales, wavelet)  # shape (S, T)
    energy = np.nanmean(coefs ** 2, axis=0)        # average energy across scales -> length T
    return energy, scales


def detect_changes(signal: np.ndarray, pen: float = 5.0, method: str = "pelt") -> np.ndarray:
    """
    Change-point detection on a 1D signal via ruptures.

    Returns indices where changes are detected (excluding the final endpoint).
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 3:
        return np.array([], dtype=int)
    if method == "pelt":
        algo = rpt.Pelt(model="rbf").fit(x)
        idx = algo.predict(pen=pen)
    else:
        algo = rpt.Binseg(model="rbf").fit(x)
        idx = algo.predict(pen=pen)
    if len(idx) == 0:
        return np.array([], dtype=int)
    return np.array(idx[:-1], dtype=int)  # exclude trailing endpoint


# ==============================
# Spectral Tilt (Short-Time)
# ==============================

def spectral_tilt(returns: np.ndarray,
                  fs: float = 1.0,
                  wlen: int = 128,
                  step: int = 16,
                  split: float = 0.15):
    """
    Sliding-window spectral tilt via Welch periodogram.

    returns: 1D array of log-returns.
    fs:      sampling frequency (1.0 for daily bars).
    wlen:    window length in bars.
    step:    stride between windows.
    split:   fraction of frequency band that defines low vs high freq (e.g. 0.15).

    Returns:
        centers: indices (in returns) of window centers (int array).
        hi_share: high-frequency power share in each window.
        vov:      local vol-of-vol proxy (std of first differences of segment).
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n == 0 or wlen <= 1:
        return np.array([], dtype=int), np.array([]), np.array([])

    wlen = int(wlen)
    step = int(step)
    if wlen > n:
        # If data is shorter than window, shrink window
        wlen = n

    win = get_window("hann", wlen)

    hi_share = []
    vov = []
    centers = []

    for start in range(0, n - wlen + 1, step):
        seg = r[start:start + wlen]
        if seg.size < 2:
            continue

        # Welch: window array length must match nperseg
        f, Pxx = welch(
            seg,
            fs=fs,
            window=win,
            nperseg=wlen,
            noverlap=0,
            detrend="constant"
        )

        if f.size < 2:
            share = 0.5
        else:
            cutoff = int(np.floor(split * f.size))
            # ensure cutoff is inside (0, len(f))
            if cutoff <= 0 or cutoff >= f.size:
                share = 0.5
            else:
                low_power = trapezoid(Pxx[:cutoff], f[:cutoff])
                high_power = trapezoid(Pxx[cutoff:], f[cutoff:])
                denom = high_power + low_power
                if denom <= 0:
                    share = 0.5
                else:
                    share = high_power / denom

        hi_share.append(share)
        vov.append(np.std(np.diff(seg)) if seg.size > 1 else 0.0)
        centers.append(start + wlen // 2)

    return np.array(centers, dtype=int), np.array(hi_share, dtype=float), np.array(vov, dtype=float)


# ==============================
# ΔVIX Robust Z
# ==============================

def vix_delta_z(vix_series: pd.Series) -> np.ndarray:
    """
    Compute robust z-score of ΔVIX using MAD-based robust_z.
    """
    if vix_series is None or vix_series.size == 0:
        return np.array([])
    dvix = vix_series.diff().fillna(0.0).values
    return robust_z(dvix)


# ==============================
# Plotly Dashboards
# ==============================

def make_plot_price(df: pd.DataFrame,
                    ticker: str,
                    outdir: str,
                    wave_points: np.ndarray,
                    tilt_alert_idx: np.ndarray) -> str:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))

    if wave_points is not None and wave_points.size > 0:
        valid_wp = wave_points[wave_points < len(df)]
        if valid_wp.size > 0:
            fig.add_trace(go.Scatter(
                x=df.index[valid_wp],
                y=df['Close'].iloc[valid_wp],
                mode="markers",
                name="Wavelet breaks",
                marker=dict(size=8, symbol="x")
            ))

    if tilt_alert_idx is not None and tilt_alert_idx.size > 0:
        valid_ta = tilt_alert_idx[tilt_alert_idx < len(df)]
        if valid_ta.size > 0:
            fig.add_trace(go.Scatter(
                x=df.index[valid_ta],
                y=df['Close'].iloc[valid_ta],
                mode="markers",
                name="Spectral tilt alerts",
                marker=dict(size=8, symbol="triangle-up")
            ))

    fig.update_layout(
        title=f"{ticker} | Price with Stress Signals",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    html = os.path.join(outdir, "html", f"{ticker}_price_signals.html")
    fig.write_html(html, auto_open=False, include_plotlyjs="cdn")
    webbrowser.open("file://" + os.path.abspath(html))
    return html


def make_plot_energy(df: pd.DataFrame,
                     energy: np.ndarray,
                     cps: np.ndarray,
                     outdir: str,
                     ticker: str) -> str:
    if energy.size == 0:
        # dummy flat line if no energy
        energy_plot = np.zeros(len(df))
    else:
        energy_plot = energy

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[:len(energy_plot)],
        y=energy_plot,
        mode="lines",
        name="CWT energy"
    ))

    if cps is not None and cps.size > 0:
        valid_cps = cps[cps < len(energy_plot)]
        if valid_cps.size > 0:
            fig.add_trace(go.Scatter(
                x=df.index[valid_cps],
                y=energy_plot[valid_cps],
                mode="markers",
                name="Change points",
                marker=dict(size=7)
            ))

    fig.update_layout(
        title=f"{ticker} | Wavelet Energy & Change Points",
        xaxis_title="Date",
        yaxis_title="Energy"
    )

    html = os.path.join(outdir, "html", f"{ticker}_energy.html")
    fig.write_html(html, auto_open=False, include_plotlyjs="cdn")
    webbrowser.open("file://" + os.path.abspath(html))
    return html


def make_plot_tilt(df: pd.DataFrame,
                   centers: np.ndarray,
                   hi_share: np.ndarray,
                   vov: np.ndarray,
                   z_hi: np.ndarray,
                   z_vov: np.ndarray,
                   alerts_idx: np.ndarray,
                   outdir: str,
                   ticker: str) -> str:
    if centers.size == 0:
        # Nothing to plot; create dummy arrays
        t = df.index[:1]
        hi_share = np.array([0.5])
        vov = np.array([0.0])
        z_hi = np.array([0.0])
        z_vov = np.array([0.0])
        alerts_idx = np.array([], dtype=int)
    else:
        t = df.index[centers]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=hi_share,
        mode="lines",
        name="High-freq share"
    ))
    fig.add_trace(go.Scatter(
        x=t,
        y=vov,
        mode="lines",
        name="Vol-of-Vol"
    ))
    fig.add_trace(go.Scatter(
        x=t,
        y=z_hi,
        mode="lines",
        name="z(High-freq share)",
        yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=t,
        y=z_vov,
        mode="lines",
        name="z(Vol-of-Vol)",
        yaxis="y2"
    ))

    if alerts_idx is not None and alerts_idx.size > 0 and centers.size > 0:
        valid_ai = alerts_idx[alerts_idx < len(hi_share)]
        if valid_ai.size > 0:
            fig.add_trace(go.Scatter(
                x=t[valid_ai],
                y=hi_share[valid_ai],
                mode="markers",
                name="Tilt alerts",
                marker=dict(size=7, symbol="triangle-up")
            ))

    fig.update_layout(
        title=f"{ticker} | Spectral Tilt & Alerts",
        xaxis_title="Date",
        yaxis=dict(title="Level"),
        yaxis2=dict(title="z-score", overlaying='y', side='right')
    )

    html = os.path.join(outdir, "html", f"{ticker}_tilt.html")
    fig.write_html(html, auto_open=False, include_plotlyjs="cdn")
    webbrowser.open("file://" + os.path.abspath(html))
    return html


# ==============================
# Main CLI
# ==============================

def main():
    ap = argparse.ArgumentParser(
        description="Wavelet-change-point + Spectral tilt stress signals (MAD-robust, no statsmodels)."
    )
    ap.add_argument("--ticker", required=True, help="Underlying ticker symbol (e.g. NVDA, IWM, SPY).")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--wave_pen", type=float, default=6.0, help="PELT penalty for wavelet change-points.")
    ap.add_argument("--wlen", type=int, default=128, help="STFT/Welch window length (bars).")
    ap.add_argument("--step", type=int, default=16, help="Stride between windows.")
    ap.add_argument("--hi_split", type=float, default=0.15, help="Low/high freq split as fraction of band.")
    ap.add_argument("--alert_z", type=float, default=3.0, help="z threshold for tilt alerts.")
    ap.add_argument("--persist", type=int, default=2, help="Consecutive windows needed to trigger alert.")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    outdir = os.path.join("/dev/shm", "STRESS_SIGNALS", ticker)
    ensure_dirs(outdir)

    # --- Load OHLCV via your canonical data loader ---
    df = dr.load_or_download_ticker(ticker, start=args.start, end=args.end)
    df = df.dropna(subset=["Close"])
    df = df.sort_index()

    if df.shape[0] < 5:
        print(f"[ERROR] Not enough data for {ticker} after filtering.")
        return

    # --- Log returns ---
    ret = np.log(df["Close"]).diff().fillna(0.0).values

    # --- Wavelet energy + change-points ---
    energy, scales = cwt_energy(ret, wavelet="morl")
    cps = detect_changes(energy, pen=args.wave_pen, method="pelt")

    # --- Spectral tilt (short-time) ---
    centers, hi_share, vov = spectral_tilt(
        ret,
        fs=1.0,
        wlen=args.wlen,
        step=args.step,
        split=args.hi_split
    )

    z_hi = robust_z(hi_share)
    z_vov = robust_z(vov)

    # --- Alert when both z streams exceed threshold with persistence >= N ---
    both = (z_hi >= args.alert_z) & (z_vov >= args.alert_z)
    alerts_idx = []
    run = 0
    for i, flag in enumerate(both):
        if flag:
            run += 1
        else:
            run = 0
        if run == args.persist:
            alerts_idx.append(i)
    alerts_idx = np.array(alerts_idx, dtype=int)

    # Map window centers back to nearest bars for price markers
    if centers.size > 0 and alerts_idx.size > 0:
        price_alert_idx = centers[alerts_idx]
        price_alert_idx = price_alert_idx[price_alert_idx < len(df)]
    else:
        price_alert_idx = np.array([], dtype=int)

    # --- Optional VIX confirm ---
    try:
        vix = dr.load_or_download_ticker(
            "^VIX",
            start=df.index[0].strftime("%Y-%m-%d"),
            end=df.index[-1].strftime("%Y-%m-%d")
        )["Close"].reindex(df.index).fillna(method="ffill")
        z_dvix = vix_delta_z(vix)
    except Exception:
        vix = None
        z_dvix = np.zeros(len(df))

    # --- Save CSV artifacts for reproducibility ---
    # Base / energy-level data
    base_df = pd.DataFrame({
        "date": df.index,
        "close": df["Close"].values,
        "ret": ret,
        "cwt_energy": energy if energy.size == df.shape[0] else np.pad(
            energy,
            (df.shape[0] - energy.size, 0),
            mode="constant",
            constant_values=np.nan
        )[-df.shape[0]:],
        "z_dVIX": z_dvix
    })
    base_df.to_csv(os.path.join(outdir, "csv", f"{ticker}_base_energy.csv"), index=False)

    # Tilt data (only for valid centers)
    if centers.size > 0:
        tilt_df = pd.DataFrame({
            "date": df.index[centers],
            "hi_share": hi_share,
            "vov": vov,
            "z_hi": z_hi,
            "z_vov": z_vov
        })
    else:
        tilt_df = pd.DataFrame(columns=["date", "hi_share", "vov", "z_hi", "z_vov"])
    tilt_df.to_csv(os.path.join(outdir, "csv", f"{ticker}_tilt.csv"), index=False)

    # Meta info
    meta = {
        "ticker": ticker,
        "scales": [int(s) for s in np.atleast_1d(scales)],
        "wave_pen": args.wave_pen,
        "wlen": args.wlen,
        "step": args.step,
        "hi_split": args.hi_split,
        "alert_z": args.alert_z,
        "persist": args.persist,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(outdir, "csv", f"{ticker}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Plotly dashboards (each opens in new browser tab) ---
    price_html = make_plot_price(df, ticker, outdir, cps, price_alert_idx)
    energy_html = make_plot_energy(df, energy, cps, outdir, ticker)
    tilt_html = make_plot_tilt(df, centers, hi_share, vov, z_hi, z_vov, alerts_idx, outdir, ticker)

    print(f"[OK] Outputs in: {outdir}")
    print(f"[HTML] {price_html}")
    print(f"[HTML] {energy_html}")
    print(f"[HTML] {tilt_html}")


if __name__ == "__main__":
    main()

