#!/usr/bin/env python3
# SCRIPTNAME: vix.spike_detector.wavelet_welch_pelt.v3.py
# AUTHOR: Michael Derby (framework wiring by ChatGPT)
# DATE:   November 25, 2025

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from scipy.signal import welch
    from scipy.ndimage import median_filter
except ImportError:
    print("ERROR: scipy is required for this script.", file=sys.stderr)
    raise

try:
    import pywt
except ImportError:
    pywt = None

try:
    import ruptures as rpt
except ImportError:
    rpt = None

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import data_retrieval as dr


def ensure_output_dir(root="/dev/shm/VIX_SPIKE_DETECT"):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    outdir = os.path.join(root, date_str)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def select_price_series(df):
    for col in ["Close", "Adj Close", "close", "adj_close"]:
        if col in df.columns:
            s = df[col].copy()
            s.name = col
            return s
    num = df.select_dtypes(include=[np.number]).columns
    if len(num) == 0:
        raise ValueError("No numeric columns found in VIX dataframe")
    s = df[num[0]].copy()
    s.name = num[0]
    return s


def load_vix_series(ticker, start=None, end=None):
    logging.info(f"Loading VIX series for {ticker} via data_retrieval...")
    df = dr.load_or_download_ticker(ticker=ticker, period="max")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    s = select_price_series(df)

    if start is not None:
        s = s[s.index >= pd.to_datetime(start)]
    if end is not None:
        s = s[s.index <= pd.to_datetime(end)]

    s = s.dropna()
    logging.info(f"Loaded VIX: {len(s)} rows from {s.index[0].date()} to {s.index[-1].date()}")
    return s


def denoise_series(s, kernel_size=3):
    if kernel_size < 2:
        return s.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    arr = median_filter(s.values, size=kernel_size)
    return pd.Series(arr, index=s.index, name=f"{s.name}_denoised")


def compute_welch_features(s, window=252, step=5, cutoff_frac=0.25):
    if len(s) < window:
        raise ValueError(f"Not enough data ({len(s)}) for window={window}")

    vals = s.values.astype(float)
    idx = s.index
    rec = []

    for end in range(window, len(vals) + 1, step):
        seg = vals[end - window:end]
        t = idx[end - 1]

        freqs, Pxx = welch(seg, nperseg=min(256, window), scaling="spectrum")
        if len(freqs) < 2:
            continue

        cut = max(1, int(len(freqs) * cutoff_frac))
        cut = min(cut, len(freqs) - 1)

        low = np.trapezoid(Pxx[:cut], freqs[:cut])
        high = np.trapezoid(Pxx[cut:], freqs[cut:])

        tot = low + high
        if tot > 0:
            p = Pxx / Pxx.sum()
            ent = -np.sum(p * np.log(p + 1e-12))
        else:
            ent = 0

        rec.append({
            "timestamp": t,
            "total_power": tot,
            "low_power": low,
            "high_power": high,
            "low_high_ratio": low / (high + 1e-12),
            "spectral_entropy": ent,
        })

    return pd.DataFrame.from_records(rec).set_index("timestamp")


def compute_wavelet_edge_intensity(s, wavelet="gaus1", num_scales=32):
    if pywt is None:
        logging.warning("pywt missing -> edge intensity = NaN")
        return pd.Series(np.nan, index=s.index, name="edge_intensity")

    x = s.values.astype(float)
    scales = np.linspace(1, num_scales, num_scales)
    coef, _ = pywt.cwt(x, scales, wavelet)
    ridges = np.abs(coef).max(axis=0)
    return pd.Series(ridges, index=s.index, name="edge_intensity")


def zscore_series(x):
    m = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=x.index, name=f"{x.name}_z")
    z = (x - m) / sd
    z.name = f"{x.name}_z"
    return z


def compute_pelt_changepoints(mat, penalty=5.0):
    if rpt is None:
        logging.warning("ruptures missing -> no CP")
        return [], []

    X = mat.values.astype(float)
    n = X.shape[0]
    if n < 10:
        return [], []

    algo = rpt.Pelt(model="rbf").fit(X)
    bkpts = algo.predict(pen=penalty)
    inds = [b for b in bkpts if b < n]

    dates = []
    for b in inds:
        ts = mat.index[b - 1]
        if hasattr(ts, "to_pydatetime"):
            dates.append(ts.to_pydatetime())
        else:
            dates.append(ts)
    return inds, dates


def build_composite_score(wdf, edge):
    e = edge.reindex(wdf.index, method="nearest")

    r = zscore_series(wdf["low_high_ratio"])
    h = zscore_series(wdf["spectral_entropy"])
    ez = zscore_series(e)

    comp = (r + h + ez) / 3
    comp.name = "composite_z"

    return pd.concat([wdf, e.rename("edge_intensity"), r, h, ez, comp], axis=1)


def detect_spikes(df, th=1.5):
    out = df["composite_z"] > th
    out.name = "is_spike"
    return out


def build_plotly_dashboard(vix_raw, feats, spikes, break_dates, out_html, ticker, z_threshold):

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.3, 0.25, 0.2, 0.25],
        subplot_titles=(
            f"{ticker} Price & Spikes",
            "Welch: Ratio & Entropy",
            "Wavelet Edge Intensity",
            "Composite Z & Threshold"
        )
    )

    # Row 1
    fig.add_trace(go.Scatter(x=vix_raw.index, y=vix_raw.values, mode="lines",
                             name=f"{ticker} Price"), row=1, col=1)

    spike_dates = feats.index[spikes.reindex(feats.index, fill_value=False)]
    spike_prices = vix_raw.reindex(spike_dates)

    fig.add_trace(go.Scatter(
        x=spike_dates,
        y=spike_prices,
        mode="markers",
        name="Spike Signal",
        marker=dict(size=8, symbol="circle-open")
    ), row=1, col=1)

    # *** FIXED: Add PELT CPs safely using SHAPES ***
    for d in break_dates:
        if hasattr(d, "to_pydatetime"):
            d = d.to_pydatetime()
        fig.add_shape(
            type="line",
            x0=d, x1=d,
            y0=0, y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dot", width=1, color="black"),
            opacity=0.4
        )

    # Row 2
    fig.add_trace(go.Scatter(x=feats.index, y=feats["low_high_ratio"],
                             mode="lines", name="Low/High Ratio"), row=2, col=1)
    fig.add_trace(go.Scatter(x=feats.index, y=feats["spectral_entropy"],
                             mode="lines", name="Spectral Entropy"), row=2, col=1)

    # Row 3
    fig.add_trace(go.Scatter(
        x=feats.index, y=feats["edge_intensity"],
        mode="lines", name="Edge Intensity"
    ), row=3, col=1)

    # Row 4
    fig.add_trace(go.Scatter(
        x=feats.index, y=feats["composite_z"],
        mode="lines", name="Composite Z"
    ), row=4, col=1)

    fig.add_hline(y=z_threshold, line=dict(dash="dash"),
                  annotation_text=f"Z={z_threshold}")

    fig.update_layout(
        title=f"VIX Spike Early-Warning ({ticker})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1)
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    fig.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="^VIX")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--cutoff-frac", type=float, default=0.25)
    p.add_argument("--z-threshold", type=float, default=1.5)
    p.add_argument("--pelt-penalty", type=float, default=5.0)
    p.add_argument("--median-kernel", type=int, default=3)
    p.add_argument("--out-root", type=str, default="/dev/shm/VIX_SPIKE_DETECT")
    return p.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    configure_logging()
    args = parse_args()

    outdir = ensure_output_dir(args.out_root)
    logging.info(f"Output directory: {outdir}")

    vix_raw = load_vix_series(args.ticker, args.start_date, args.end_date)
    vix = denoise_series(vix_raw, args.median_kernel)

    wdf = compute_welch_features(vix, args.window, args.step, args.cutoff_frac)
    edges = compute_wavelet_edge_intensity(vix)
    feats = build_composite_score(wdf, edges)

    zcols = ["low_high_ratio_z", "spectral_entropy_z", "edge_intensity_z"]
    mat = feats[zcols]

    _, break_dates = compute_pelt_changepoints(mat, args.pelt_penalty)

    spikes = detect_spikes(feats, args.z_threshold)

    csv_path = os.path.join(outdir, "vix_spike_features.csv")
    feats.assign(is_spike=spikes).to_csv(csv_path)

    sig_path = os.path.join(outdir, "vix_spike_signals.csv")
    pd.DataFrame({
        "date": feats.index[spikes],
        "composite_z": feats.loc[spikes, "composite_z"]
    }).to_csv(sig_path, index=False)

    html_path = os.path.join(outdir, "vix_spike_dashboard.html")

    build_plotly_dashboard(
        vix_raw, feats, spikes, break_dates, html_path,
        args.ticker, args.z_threshold
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()

