#!/usr/bin/env python3
# SCRIPTNAME: mega.lrc.multi.v1.py

import argparse
import sys
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

# ---------------------------------------------------------
# Canonical data access (do not modify this module)
# ---------------------------------------------------------
try:
    import data_retrieval
except Exception as e:
    print(f"[ERROR] Could not import data_retrieval.py: {e}")
    sys.exit(1)


# ---------------------------------------------------------
# Core regression + fusion logic
# ---------------------------------------------------------
def compute_regression(df, W):
    """
    Compute linear regression over the last W bars.
    Returns (m, b, sigma, r2, residuals, fitted) or None if not enough data.
    """
    if len(df) < W:
        return None

    y = df["Close"].iloc[-W:].values
    x = np.arange(W)

    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b

    residuals = y - y_pred
    sigma = np.std(residuals)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return m, b, sigma, r2, residuals, y_pred


def compute_vol_regime(df):
    """
    Compute 63d rolling volatility and classify into low/med/high regime.
    Robust to short histories.
    """
    if len(df) < 10:
        return np.nan, "med"

    win = min(63, max(5, len(df) // 2))
    v = df["Close"].pct_change().rolling(win).std() * np.sqrt(252)
    latest_vol = v.iloc[-1]
    v_nonan = v.dropna()
    if len(v_nonan) < 5:
        return latest_vol, "med"

    q33 = v_nonan.quantile(0.33)
    q66 = v_nonan.quantile(0.66)

    if latest_vol <= q33:
        regime = "low"
    elif latest_vol <= q66:
        regime = "med"
    else:
        regime = "high"

    return latest_vol, regime


def compute_weights(reg_outputs, vol, regime, gamma=1.5, lam=1.0):
    """
    Weight each window by R^gamma, regime preference, and volatility damping.
    reg_outputs: {W: (m,b,sigma,r2,residuals,fitted)}
    """
    pref = {
        "low":  {252: 0.6, 126: 0.3, 63: 0.1},
        "med":  {252: 0.4, 126: 0.4, 63: 0.2},
        "high": {252: 0.2, 126: 0.35, 63: 0.45},
    }
    if regime not in pref:
        regime = "med"

    # R² weights
    wR = {}
    for W, (m, b, sigma, r2, _, _) in reg_outputs.items():
        wR[W] = (max(r2, 0.0)) ** gamma

    # Regime preference, fallback to equal if missing
    pW = {W: pref[regime].get(W, 1.0 / len(reg_outputs)) for W in reg_outputs}

    # Volatility damping
    d = 1.0 / (1.0 + lam * (0.0 if np.isnan(vol) else vol))

    raw = {W: wR[W] * pW[W] * d for W in reg_outputs}
    total = sum(raw.values())

    if total == 0:
        n = len(raw)
        return {W: 1.0 / n for W in reg_outputs}

    return {W: raw[W] / total for W in reg_outputs}


def fuse_channels(reg_outputs, weights):
    """
    Fuse multi-window regression outputs into single slope, intercept, width.
    """
    m_star = sum(weights[W] * reg_outputs[W][0] for W in weights)
    b_star = sum(weights[W] * reg_outputs[W][1] for W in weights)
    sigma_star = np.sqrt(sum(weights[W] * (reg_outputs[W][2] ** 2) for W in weights))
    return m_star, b_star, sigma_star


# ---------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------
def compute_bollinger(df, length=20, k=2.0):
    close = df["Close"]
    ma = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = ma + k * std
    lower = ma - k * std

    out = df.copy()
    out["BB_MID"] = ma
    out["BB_UPPER"] = upper
    out["BB_LOWER"] = lower
    return out


# ---------------------------------------------------------
# Darvas boxes (approximate)
# ---------------------------------------------------------
def find_pivots(df, left=3, right=3):
    highs = df["High"].values
    lows = df["Low"].values
    pivots_hi = []
    pivots_lo = []
    n = len(df)

    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i - left : i + right + 1]
        if highs[i] == window_h.max():
            pivots_hi.append(i)
        if lows[i] == window_l.min():
            pivots_lo.append(i)

    return pivots_hi, pivots_lo


def build_darvas_boxes(df, pivots_hi, pivots_lo):
    """
    Very simplified Darvas: pair each pivot high with next pivot low below it.
    """
    boxes = []
    piv_hi_sorted = sorted(pivots_hi)
    piv_lo_sorted = sorted(pivots_lo)
    j = 0

    for hi_idx in piv_hi_sorted:
        hi_price = df["High"].iloc[hi_idx]
        while j < len(piv_lo_sorted) and piv_lo_sorted[j] <= hi_idx:
            j += 1
        if j >= len(piv_lo_sorted):
            break

        lo_idx = piv_lo_sorted[j]
        lo_price = df["Low"].iloc[lo_idx]
        if lo_price < hi_price:
            boxes.append(
                {
                    "top": hi_price,
                    "bottom": lo_price,
                    "start": hi_idx,
                    "end": lo_idx,
                }
            )
    return boxes


# ---------------------------------------------------------
# Anchored VWAP lines
# ---------------------------------------------------------
def compute_avwap_lines(df, anchor_indices, price_col="Close"):
    """
    Compute Anchored VWAP lines from multiple anchor indices.
    Returns {anchor_idx: vwap_array_from_anchor}.
    """
    if "Volume" not in df.columns:
        return {}

    price = df[price_col].values
    vol = df["Volume"].values
    n = len(df)

    pv = price * vol
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(vol)

    lines = {}
    for idx in anchor_indices:
        if idx < 0 or idx >= n:
            continue
        pv_start = cum_pv[idx - 1] if idx > 0 else 0.0
        vol_start = cum_vol[idx - 1] if idx > 0 else 0.0
        vwap = (cum_pv[idx:] - pv_start) / (cum_vol[idx:] - vol_start + 1e-9)
        lines[idx] = vwap

    return lines


# ---------------------------------------------------------
# Hurst exponent / fractal dimension
# ---------------------------------------------------------
def estimate_hurst_rs(series, max_lag=64):
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < 20:
        return np.nan

    max_lag = min(max_lag, n // 2)
    lags = np.arange(2, max_lag)
    rs = []

    for lag in lags:
        segs = n // lag
        if segs < 2:
            continue
        rs_vals = []
        for i in range(segs):
            segment = s[i * lag : (i + 1) * lag]
            mean = segment.mean()
            dev = segment - mean
            cum = np.cumsum(dev)
            R = cum.max() - cum.min()
            S = segment.std()
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            rs.append(np.mean(rs_vals))

    if len(rs) < 2:
        return np.nan

    log_lags = np.log(lags[: len(rs)])
    log_rs = np.log(rs)
    H, _ = np.polyfit(log_lags, log_rs, 1)
    return H


# ---------------------------------------------------------
# Rolling slope map for z-score heatmap
# ---------------------------------------------------------
def compute_rolling_slopes(df, windows):
    close = df["Close"].values
    n = len(close)
    data = {W: [] for W in windows}
    idxs = []
    maxW = max(windows)

    for t in range(maxW - 1, n):
        idxs.append(df.index[t])
        for W in windows:
            if t - W + 1 < 0:
                data[W].append(np.nan)
                continue
            y = close[t - W + 1 : t + 1]
            x = np.arange(W)
            m, _ = np.polyfit(x, y, 1)
            data[W].append(m)

    slopes_df = pd.DataFrame(data, index=idxs)
    return slopes_df


# ---------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------
def plot_fused_lrc(df, ticker, m_star, b_star, sigma_star,
                   hurst, fractal_dim, vol_regime, outpath):
    N = len(df)
    x = np.arange(N)

    mid = m_star * x + b_star
    upper = mid + 2.0 * sigma_star
    lower = mid - 2.0 * sigma_star

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )
    fig.add_trace(go.Scatter(x=df.index, y=mid, mode="lines", name="Fused Mid"))
    fig.add_trace(go.Scatter(x=df.index, y=upper, mode="lines", name="Upper Band"))
    fig.add_trace(go.Scatter(x=df.index, y=lower, mode="lines", name="Lower Band"))

    title = f"{ticker} – Fused LRC | H={hurst:.2f} D={fractal_dim:.2f} Regime={vol_regime}"
    fig.update_layout(title=title, width=1400, height=800)
    plot(fig, filename=outpath, auto_open=True)


def plot_bollinger(df_bb, ticker, outpath):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_bb.index,
            open=df_bb["Open"],
            high=df_bb["High"],
            low=df_bb["Low"],
            close=df_bb["Close"],
            name="Price",
        )
    )
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb["BB_MID"], mode="lines", name="BB Mid"))
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb["BB_UPPER"], mode="lines", name="BB Upper"))
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb["BB_LOWER"], mode="lines", name="BB Lower"))

    fig.update_layout(title=f"{ticker} – Bollinger Bands", width=1400, height=800)
    plot(fig, filename=outpath, auto_open=True)


def plot_darvas(df, boxes, ticker, outpath):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    for box in boxes:
        start = df.index[box["start"]]
        end = df.index[box["end"]]
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=box["bottom"],
            y1=box["top"],
            line=dict(width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        )

    fig.update_layout(title=f"{ticker} – Darvas Boxes (approx)", width=1400, height=800)
    plot(fig, filename=outpath, auto_open=True)


def plot_avwap(df, avwap_lines, ticker, outpath):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    for idx, line in avwap_lines.items():
        label = f"AVWAP@{df.index[idx].date()}"
        fig.add_trace(go.Scatter(x=df.index[idx:], y=line, mode="lines", name=label))

    fig.update_layout(title=f"{ticker} – Anchored VWAP Lines", width=1400, height=800)
    plot(fig, filename=outpath, auto_open=True)


def plot_residuals(reg_outputs, windows, outpath, ticker):
    fig = go.Figure()
    for W in windows:
        if W not in reg_outputs:
            continue
        residuals = reg_outputs[W][4]
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name=f"{W}-day residuals",
                opacity=0.6,
            )
        )

    fig.update_layout(
        barmode="overlay",
        title=f"{ticker} – Regression Residual Distributions",
        width=1200,
        height=700,
    )
    plot(fig, filename=outpath, auto_open=True)


def plot_slope_heatmap(slopes_df, outpath, ticker):
    if slopes_df.empty:
        return

    z_df = slopes_df.copy()
    for col in z_df.columns:
        mu = z_df[col].mean()
        sigma = z_df[col].std()
        if sigma == 0 or np.isnan(sigma):
            z_df[col] = 0.0
        else:
            z_df[col] = (z_df[col] - mu) / sigma

    x = z_df.index
    y = [str(c) for c in z_df.columns]
    z = z_df.values.T

    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorbar=dict(title="Slope Z"),
        )
    )
    fig.update_layout(
        title=f"{ticker} – Rolling Slope Z-Score Heatmap",
        width=1400,
        height=600,
    )
    plot(fig, filename=outpath, auto_open=True)


# ---------------------------------------------------------
# Main entry
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", default="2010-01-01")
    args = parser.parse_args()

    # Data
    try:
        df = data_retrieval.load_or_download_ticker(args.ticker, start=args.start)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    df = df.dropna().copy()

    # Multi-window regressions
    windows = [252, 126, 63]
    reg_outputs = {}

    print("\nRegression windows availability:")
    for W in windows:
        res = compute_regression(df, W)
        if res is None:
            print(f"  {W}-day: NOT AVAILABLE")
        else:
            print(f"  {W}-day: OK")
            reg_outputs[W] = res

    if not reg_outputs:
        print("[FATAL] No regression windows available.")
        sys.exit(1)

    # Vol regime + weights + fused channel
    vol, regime = compute_vol_regime(df)
    weights = compute_weights(reg_outputs, vol, regime)
    m_star, b_star, sigma_star = fuse_channels(reg_outputs, weights)

    # Hurst / fractal
    close_series = df["Close"]
    hurst = estimate_hurst_rs(close_series, max_lag=64)
    fractal_dim = 2.0 - hurst if not np.isnan(hurst) else np.nan

    # Bollinger
    df_bb = compute_bollinger(df, length=20, k=2.0)

    # Darvas pivots / boxes
    piv_hi, piv_lo = find_pivots(df, left=3, right=3)
    boxes = build_darvas_boxes(df, piv_hi, piv_lo)

    # Anchors for AVWAP: last 3 pivot lows and last 3 pivot highs
    anchors = sorted(piv_lo[-3:] + piv_hi[-3:])
    anchors = [a for a in anchors if 0 <= a < len(df)]
    avwap_lines = compute_avwap_lines(df, anchors, price_col="Close")

    # Rolling slope z heatmap
    slopes_df = compute_rolling_slopes(df, [W for W in windows if W in reg_outputs])

    base = args.ticker.upper()

    # Plots (each opens in its own HTML tab)
    plot_fused_lrc(
        df,
        base,
        m_star,
        b_star,
        sigma_star,
        hurst if not np.isnan(hurst) else 0.0,
        fractal_dim if not np.isnan(fractal_dim) else 0.0,
        regime,
        outpath=f"{base}_FUSED_LRC.html",
    )

    plot_bollinger(df_bb, base, outpath=f"{base}_BBANDS.html")

    if boxes:
        plot_darvas(df, boxes, base, outpath=f"{base}_DARVAS.html")

    if avwap_lines:
        plot_avwap(df, avwap_lines, base, outpath=f"{base}_AVWAP.html")

    plot_residuals(reg_outputs, windows, outpath=f"{base}_RESIDUALS.html", ticker=base)

    plot_slope_heatmap(slopes_df, outpath=f"{base}_SLOPE_Z_HEATMAP.html", ticker=base)

    # Console summary
    print("\nDONE.")
    print("Windows used:")
    for W in reg_outputs:
        print(f"  {W}-day (weight {weights.get(W, 0):.4f})")
    print(f"Hurst: {hurst:.3f}  Fractal dimension: {fractal_dim:.3f}")
    print(f"Vol regime: {regime}")
    print("HTML outputs created for:")
    print("  - Fused LRC")
    print("  - Bollinger Bands")
    print("  - Darvas Boxes (if boxes exist)")
    print("  - Anchored VWAP (if anchors exist)")
    print("  - Regression residual distributions")
    print("  - Rolling slope Z-score heatmap")


if __name__ == "__main__":
    main()

