#!/usr/bin/env python3
# SCRIPTNAME: head.shoulders.v4.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Unified script that builds two figures from a single data load (via data_retrieval.py):
#   - Figure 1: Candlesticks + wick-centered peaks/troughs + pivot-anchored lines
#               (green for peaks / red for troughs) + black "touch" trough line.
#   - Figure 2: Multiscale linear-regression (short/medium/long) lines for peaks & troughs.
#
# Both figures are optionally saved and, by default, opened in separate browser tabs.
#
# Additionally, detects head and shoulders (top/bottom) and triple tops/bottoms patterns
# based on identified peaks and troughs, prints them to console, and renders each
# detected pattern on its own separate chart.

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ----------------------------
# Import centralized data layer
# ----------------------------
try:
    import data_retrieval  # cache-aware IO
except ImportError:
    print("Error: 'data_retrieval.py' not found on PYTHONPATH.")
    sys.exit(1)

# ----------------------------
# Peak/Trough Detection (wick-centered High/Low)
# ----------------------------
def find_peaks_troughs(df: pd.DataFrame, order: int = 5):
    """
    Detect wick-centered peaks and troughs using simple local-extrema rules.
    Returns: (peaks, troughs) where each is a list of (i, price) tuples.
    """
    highs, lows = df['High'].to_numpy(), df['Low'].to_numpy()
    peaks, troughs = [], []
    n = len(df)

    for i in range(order, n - order):
        if highs[i] == np.max(highs[i - order:i + order + 1]):
            peaks.append((i, highs[i]))
        if lows[i] == np.min(lows[i - order:i + order + 1]):
            troughs.append((i, lows[i]))
    return peaks, troughs

# ----------------------------
# Pattern Detection Functions
# ----------------------------
def detect_head_shoulders_top(peaks, troughs, df_index, tolerance=0.05, min_distance=5):
    """
    Detect head and shoulders top (bearish) patterns.
    Returns list of dicts with pattern components.
    """
    patterns = []
    peaks = sorted(peaks, key=lambda x: x[0])
    troughs = sorted(troughs, key=lambda x: x[0])
    for j in range(2, len(peaks)):
        left_shoulder = peaks[j-2]
        head = peaks[j-1]
        right_shoulder = peaks[j]
        i_ls, p_ls = left_shoulder
        i_h, p_h = head
        i_rs, p_rs = right_shoulder
        if i_h - i_ls < min_distance or i_rs - i_h < min_distance:
            continue
        if p_h <= p_ls or p_h <= p_rs:
            continue
        mean_shoulder = (p_ls + p_rs) / 2
        if abs(p_ls - p_rs) / mean_shoulder > tolerance:
            continue
        trough1 = [t for t in troughs if i_ls < t[0] < i_h]
        trough2 = [t for t in troughs if i_h < t[0] < i_rs]
        if not trough1 or not trough2:
            continue
        t1 = min(trough1, key=lambda x: x[1])
        t2 = min(trough2, key=lambda x: x[1])
        mean_neck = (t1[1] + t2[1]) / 2
        if abs(t1[1] - t2[1]) / mean_neck > tolerance:
            continue
        patterns.append({
            'type': 'Head and Shoulders Top',
            'left_shoulder': (df_index[i_ls], p_ls),
            'head': (df_index[i_h], p_h),
            'right_shoulder': (df_index[i_rs], p_rs),
            'left_trough': (df_index[t1[0]], t1[1]),
            'right_trough': (df_index[t2[0]], t2[1])
        })
    return patterns

def detect_head_shoulders_bottom(troughs, peaks, df_index, tolerance=0.05, min_distance=5):
    """
    Detect inverse head and shoulders bottom (bullish) patterns.
    Returns list of dicts with pattern components.
    """
    patterns = []
    troughs = sorted(troughs, key=lambda x: x[0])
    peaks = sorted(peaks, key=lambda x: x[0])
    for j in range(2, len(troughs)):
        left_shoulder = troughs[j-2]
        head = troughs[j-1]
        right_shoulder = troughs[j]
        i_ls, p_ls = left_shoulder
        i_h, p_h = head
        i_rs, p_rs = right_shoulder
        if i_h - i_ls < min_distance or i_rs - i_h < min_distance:
            continue
        if p_h >= p_ls or p_h >= p_rs:
            continue
        mean_shoulder = (p_ls + p_rs) / 2
        if abs(p_ls - p_rs) / mean_shoulder > tolerance:
            continue
        peak1 = [p for p in peaks if i_ls < p[0] < i_h]
        peak2 = [p for p in peaks if i_h < p[0] < i_rs]
        if not peak1 or not peak2:
            continue
        p1 = max(peak1, key=lambda x: x[1])
        p2 = max(peak2, key=lambda x: x[1])
        mean_peak = (p1[1] + p2[1]) / 2
        if abs(p1[1] - p2[1]) / mean_peak > tolerance:
            continue
        patterns.append({
            'type': 'Inverse Head and Shoulders Bottom',
            'left_shoulder': (df_index[i_ls], p_ls),
            'head': (df_index[i_h], p_h),
            'right_shoulder': (df_index[i_rs], p_rs),
            'left_peak': (df_index[p1[0]], p1[1]),
            'right_peak': (df_index[p2[0]], p2[1])
        })
    return patterns

def detect_triple_top(peaks, troughs, df_index, tolerance=0.05, min_distance=5):
    """
    Detect triple top (bearish) patterns.
    Returns list of dicts with pattern components.
    """
    patterns = []
    peaks = sorted(peaks, key=lambda x: x[0])
    troughs = sorted(troughs, key=lambda x: x[0])
    for j in range(2, len(peaks)):
        peak1 = peaks[j-2]
        peak2 = peaks[j-1]
        peak3 = peaks[j]
        i1, v1 = peak1
        i2, v2 = peak2
        i3, v3 = peak3
        if i2 - i1 < min_distance or i3 - i2 < min_distance:
            continue
        avg_peak = (v1 + v2 + v3) / 3
        if any(abs(v - avg_peak) / avg_peak > tolerance for v in [v1, v2, v3]):
            continue
        trough1 = [t for t in troughs if i1 < t[0] < i2]
        trough2 = [t for t in troughs if i2 < t[0] < i3]
        if not trough1 or not trough2:
            continue
        t1 = min(trough1, key=lambda x: x[1])
        t2 = min(trough2, key=lambda x: x[1])
        mean_trough = (t1[1] + t2[1]) / 2
        if abs(t1[1] - t2[1]) / mean_trough > tolerance:
            continue
        patterns.append({
            'type': 'Triple Top',
            'peak1': (df_index[i1], v1),
            'peak2': (df_index[i2], v2),
            'peak3': (df_index[i3], v3),
            'trough1': (df_index[t1[0]], t1[1]),
            'trough2': (df_index[t2[0]], t2[1])
        })
    return patterns

def detect_triple_bottom(troughs, peaks, df_index, tolerance=0.05, min_distance=5):
    """
    Detect triple bottom (bullish) patterns.
    Returns list of dicts with pattern components.
    """
    patterns = []
    troughs = sorted(troughs, key=lambda x: x[0])
    peaks = sorted(peaks, key=lambda x: x[0])
    for j in range(2, len(troughs)):
        trough1 = troughs[j-2]
        trough2 = troughs[j-1]
        trough3 = troughs[j]
        i1, v1 = trough1
        i2, v2 = trough2
        i3, v3 = trough3
        if i2 - i1 < min_distance or i3 - i2 < min_distance:
            continue
        avg_trough = (v1 + v2 + v3) / 3
        if any(abs(v - avg_trough) / avg_trough > tolerance for v in [v1, v2, v3]):
            continue
        peak1 = [p for p in peaks if i1 < p[0] < i2]
        peak2 = [p for p in peaks if i2 < p[0] < i3]
        if not peak1 or not peak2:
            continue
        p1 = max(peak1, key=lambda x: x[1])
        p2 = max(peak2, key=lambda x: x[1])
        mean_peak = (p1[1] + p2[1]) / 2
        if abs(p1[1] - p2[1]) / mean_peak > tolerance:
            continue
        patterns.append({
            'type': 'Triple Bottom',
            'trough1': (df_index[i1], v1),
            'trough2': (df_index[i2], v2),
            'trough3': (df_index[i3], v3),
            'peak1': (df_index[p1[0]], p1[1]),
            'peak2': (df_index[p2[0]], p2[1])
        })
    return patterns

def print_patterns(patterns):
    if not patterns:
        return
    for pat in patterns:
        print(f"{pat['type']}:")
        for key, value in pat.items():
            if key != 'type':
                print(f"  {key}: Date={value[0].date()}, Price={value[1]:.2f}")
        print("---")

# ----------------------------
# Pattern Drawing Functions
# ----------------------------
def create_base_chart(df: pd.DataFrame, ticker: str, title: str) -> go.Figure:
    """Creates a base figure with Candlesticks and Volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candlestick"
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color="lightslategrey"), row=2, col=1)
    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False, height=900, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

def draw_head_shoulders_top(fig: go.Figure, pattern: dict):
    """Adds shapes for a Head and Shoulders Top pattern to a figure."""
    dates = [pattern['left_shoulder'][0], pattern['head'][0], pattern['right_shoulder'][0]]
    prices = [pattern['left_shoulder'][1], pattern['head'][1], pattern['right_shoulder'][1]]
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode='lines+markers', name='H&S Top Outline',
        line=dict(color='blue', width=2), marker=dict(size=10, color='blue')
    ), row=1, col=1)
    neck_dates = [pattern['left_trough'][0], pattern['right_trough'][0]]
    neck_prices = [pattern['left_trough'][1], pattern['right_trough'][1]]
    fig.add_trace(go.Scatter(
        x=neck_dates, y=neck_prices, mode='lines', name='Neckline',
        line=dict(color='purple', width=2, dash='dash')
    ), row=1, col=1)

def draw_head_shoulders_bottom(fig: go.Figure, pattern: dict):
    """Adds shapes for an Inverse Head and Shoulders Bottom pattern to a figure."""
    dates = [pattern['left_shoulder'][0], pattern['head'][0], pattern['right_shoulder'][0]]
    prices = [pattern['left_shoulder'][1], pattern['head'][1], pattern['right_shoulder'][1]]
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode='lines+markers', name='Inv. H&S Outline',
        line=dict(color='orange', width=2), marker=dict(size=10, color='orange')
    ), row=1, col=1)
    neck_dates = [pattern['left_peak'][0], pattern['right_peak'][0]]
    neck_prices = [pattern['left_peak'][1], pattern['right_peak'][1]]
    fig.add_trace(go.Scatter(
        x=neck_dates, y=neck_prices, mode='lines', name='Neckline',
        line=dict(color='purple', width=2, dash='dash')
    ), row=1, col=1)

def draw_triple_top(fig: go.Figure, pattern: dict):
    """Adds shapes for a Triple Top pattern to a figure."""
    dates = [pattern['peak1'][0], pattern['peak2'][0], pattern['peak3'][0]]
    prices = [pattern['peak1'][1], pattern['peak2'][1], pattern['peak3'][1]]
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode='lines+markers', name='Triple Top Resistance',
        line=dict(color='cyan', width=2), marker=dict(size=10, color='cyan')
    ), row=1, col=1)
    support_dates = [pattern['trough1'][0], pattern['trough2'][0]]
    support_prices = [pattern['trough1'][1], pattern['trough2'][1]]
    fig.add_trace(go.Scatter(
        x=support_dates, y=support_prices, mode='lines', name='Support',
        line=dict(color='purple', width=2, dash='dash')
    ), row=1, col=1)

def draw_triple_bottom(fig: go.Figure, pattern: dict):
    """Adds shapes for a Triple Bottom pattern to a figure."""
    dates = [pattern['trough1'][0], pattern['trough2'][0], pattern['trough3'][0]]
    prices = [pattern['trough1'][1], pattern['trough2'][1], pattern['trough3'][1]]
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode='lines+markers', name='Triple Bottom Support',
        line=dict(color='magenta', width=2), marker=dict(size=10, color='magenta')
    ), row=1, col=1)
    resistance_dates = [pattern['peak1'][0], pattern['peak2'][0]]
    resistance_prices = [pattern['peak1'][1], pattern['peak2'][1]]
    fig.add_trace(go.Scatter(
        x=resistance_dates, y=resistance_prices, mode='lines', name='Resistance',
        line=dict(color='purple', width=2, dash='dash')
    ), row=1, col=1)

# ----------------------------
# Pivot/Touch trendline helpers (Figure 1)
# ----------------------------
def _anchor_line(points, kind, df_index):
    """
    Build TA-style line by connecting an early pivot with a recent pivot.
    kind: "peak" -> choose highest in first half, then most recent in second half
          "trough" -> choose lowest in first half, then most recent in second half
    Returns (x_dates, y_vals) possibly empty if not enough points.
    """
    if len(points) < 2:
        return [], []

    half = len(points) // 2
    if kind == "peak":
        first = max(points[:half], key=lambda x: x[1])
        last  = max(points[half:], key=lambda x: x[0])
    else:  # trough
        first = min(points[:half], key=lambda x: x[1])
        last  = max(points[half:], key=lambda x: x[0])

    x_idx = [first[0], last[0]]
    y_vals = [first[1], last[1]]
    x_dates = [df_index[i] for i in x_idx]

    # Extend to chart end
    slope = (y_vals[1] - y_vals[0]) / (x_idx[1] - x_idx[0]) if (x_idx[1] - x_idx[0]) != 0 else 0.0
    x_dates.append(df_index[-1])
    y_vals.append(y_vals[0] + slope * (len(df_index) - 1 - x_idx[0]))
    return x_dates, y_vals

def _touch_trough_trendline(troughs, df_index):
    """
    Build a 'touch' support line starting from the absolute lowest trough with slope
    chosen to touch as many subsequent troughs as possible (within ~1% tolerance).
    """
    if len(troughs) < 2:
        return [], []

    anchor_idx, anchor_val = min(troughs, key=lambda t: t[1])
    candidates = []
    for (i, val) in troughs:
        if i <= anchor_idx:
            continue
        denom = (i - anchor_idx)
        if denom == 0:
            continue
        slope = (val - anchor_val) / denom
        count = 0
        for (j, v) in troughs:
            if j >= anchor_idx:
                pred = anchor_val + slope * (j - anchor_idx)
                if abs(pred - v) / max(abs(v), 1e-12) < 0.01:  # within 1%
                    count += 1
        candidates.append((slope, count, i))

    if not candidates:
        return [], []
    slope = max(candidates, key=lambda x: (x[1], x[2]))[0]

    x_idx = [anchor_idx, len(df_index) - 1]
    y_vals = [anchor_val, anchor_val + slope * (len(df_index) - 1 - anchor_idx)]
    x_dates = [df_index[i] for i in x_idx]
    return x_dates, y_vals

def create_figure_pivot_touch(df: pd.DataFrame, ticker: str, peaks, troughs) -> go.Figure:
    """Figure 1: Candlesticks + markers + pivot-anchored lines + black 'touch' line."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candlestick"
    ), row=1, col=1)

    # Markers
    if peaks:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in peaks], y=[p for (_, p) in peaks],
            mode="markers", name="Peaks",
            marker=dict(symbol="circle", size=40,
                        color="rgba(0,200,0,0.3)",
                        line=dict(color="darkgreen", width=2))
        ), row=1, col=1)
    if troughs:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in troughs], y=[t for (_, t) in troughs],
            mode="markers", name="Troughs",
            marker=dict(symbol="circle", size=40,
                        color="rgba(200,0,0,0.3)",
                        line=dict(color="darkred", width=2))
        ), row=1, col=1)

    # Trendlines
    x_peak, y_peak = _anchor_line(peaks, "peak", df.index)
    if x_peak:
        fig.add_trace(go.Scatter(x=x_peak, y=y_peak, mode="lines",
                                 name="Peak Trend", line=dict(color="green", width=2)), row=1, col=1)

    x_trough, y_trough = _anchor_line(troughs, "trough", df.index)
    if x_trough:
        fig.add_trace(go.Scatter(x=x_trough, y=y_trough, mode="lines",
                                 name="Trough Trend", line=dict(color="red", width=2)), row=1, col=1)

    x_touch, y_touch = _touch_trough_trendline(troughs, df.index)
    if x_touch:
        fig.add_trace(go.Scatter(x=x_touch, y=y_touch, mode="lines",
                                 name="Touch Trend", line=dict(color="black", width=2, dash="dot")),
                      row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color="lightslategrey"), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Peaks & Troughs with Pivot/Touch Trendlines",
        xaxis_rangeslider_visible=False, height=900, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ----------------------------
# Multiscale LR helpers (Figure 2)
# ----------------------------
def _select_points_in_window(points, last_index: int, window_bars: int):
    if window_bars <= 0:
        return []
    start_idx = max(0, last_index - window_bars + 1)
    return [(i, y) for (i, y) in points if i >= start_idx]

def _fit_linear_regression(points_subset):
    x = np.array([i for (i, _) in points_subset], dtype=float)
    y = np.array([y for (_, y) in points_subset], dtype=float)
    m, b = np.polyfit(x, y, 1)
    return m, b, int(x.min())

def _build_lr_line(points, window_bars: int, n_bars: int, df_index):
    if not points or window_bars <= 0:
        return [], []
    inwin = _select_points_in_window(points, n_bars - 1, window_bars)
    if len(inwin) < 2:
        return [], []
    m, b, x0 = _fit_linear_regression(inwin)
    x_start = x0
    x_end   = n_bars - 1
    y_start = m * x_start + b
    y_end   = m * x_end + b
    return [df_index[x_start], df_index[x_end]], [y_start, y_end]

def create_figure_multiscale(df: pd.DataFrame,
                             ticker: str,
                             peaks, troughs,
                             short_days: int,
                             medium_days: int,
                             long_days: int) -> go.Figure:
    """Figure 2: Candlesticks + multiscale LR lines for peaks/troughs."""
    n = len(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candlestick"
    ), row=1, col=1)

    # Markers
    if peaks:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in peaks], y=[p for (_, p) in peaks],
            mode="markers", name="Peaks",
            marker=dict(symbol="circle", size=34,
                        color="rgba(0,160,0,0.28)",
                        line=dict(color="darkgreen", width=1.5))
        ), row=1, col=1)
    if troughs:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in troughs], y=[t for (_, t) in troughs],
            mode="markers", name="Troughs",
            marker=dict(symbol="circle", size=34,
                        color="rgba(200,0,0,0.28)",
                        line=dict(color="darkred", width=1.5))
        ), row=1, col=1)

    # Peak LR lines (greens)
    for label, win, style in [
        (f"Peak LR (short {short_days}d)",  short_days,  dict(color="green", width=1.5, dash="dot")),
        (f"Peak LR (med {medium_days}d)",   medium_days, dict(color="green", width=2.0, dash="dash")),
        (f"Peak LR (long {long_days}d)",    long_days,   dict(color="green", width=2.8)),
    ]:
        xs, ys = _build_lr_line(peaks, win, n, df.index)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label, line=style), row=1, col=1)

    # Trough LR lines (reds)
    for label, win, style in [
        (f"Trough LR (short {short_days}d)", short_days,  dict(color="red", width=1.5, dash="dot")),
        (f"Trough LR (med {medium_days}d)",  medium_days, dict(color="red", width=2.0, dash="dash")),
        (f"Trough LR (long {long_days}d)",   long_days,   dict(color="red", width=2.8)),
    ]:
        xs, ys = _build_lr_line(troughs, win, n, df.index)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label, line=style), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color="lightslategrey"), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Multiscale Peak/Trough Linear-Regression Lines",
        xaxis_rangeslider_visible=False, height=900, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ----------------------------
# Save helpers
# ----------------------------
def _resolve_output_dir(ticker: str) -> str:
    return data_retrieval.create_output_directory(ticker)

def _maybe_save(fig: go.Figure, out_dir: str, out_name: str, save_png: bool, save_html: bool):
    # Constraint check: data_retrieval handles the directory, we just use it.
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, out_name)
    if save_html:
        fig.write_html(base + ".html", include_plotlyjs="cdn")
        print(f"[saved] {base}.html")
    if save_png:
        try:
            fig.write_image(base + ".png", width=1920, height=1080, scale=2)
            print(f"[saved] {base}.png")
        except Exception as e:
            print(f"[warn] PNG export requires 'kaleido': {e}")

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Dual peak/trough plotting (pivot/touch & multiscale LR) via data_retrieval.")
    p.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    date = p.add_mutually_exclusive_group()
    date.add_argument("--period", default="1y", help="yfinance period if --date-range not provided (default: 1y)")
    date.add_argument("--date-range", metavar="START,END", help="Explicit date range (YYYY-MM-DD,YYYY-MM-DD)")

    p.add_argument("--order", type=int, default=5, help="Local-extrema half-window (default: 5)")
    p.add_argument("--short-days",  type=int, default=60,  help="Short lookback (bars). Default: 60")
    p.add_argument("--medium-days", type=int, default=120, help="Medium lookback (bars). Default: 120")
    p.add_argument("--long-days",   type=int, default=252, help="Long lookback (bars). Default: 252")
    p.add_argument("--save-png",  action="store_true", help="Save PNG (requires kaleido)")
    p.add_argument("--save-html", action="store_true", help="Save HTML")
    p.add_argument("--no-show",   action="store_true", help="Do not open figures in browser")
    args = p.parse_args()

    # --- Load data strictly through data_retrieval ---
    if args.date_range:
        try:
            start, end = [s.strip() for s in args.date_range.split(",")]
        except ValueError:
            sys.exit("Error: --date-range must be 'YYYY-MM-DD,YYYY-MM-DD'")
        df = data_retrieval.load_or_download_ticker(args.ticker, start=start, end=end)
    else:
        df = data_retrieval.get_stock_data(args.ticker, period=args.period)

    if df is None or df.empty:
        sys.exit("No data found.")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Basic column sanity
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        sys.exit(f"Data missing required columns: {missing}")

    # Peaks/Troughs
    peaks, troughs = find_peaks_troughs(df, order=args.order)

    # Pattern Detection
    hs_tops = detect_head_shoulders_top(peaks, troughs, df.index, min_distance=args.order * 2)
    hs_bottoms = detect_head_shoulders_bottom(troughs, peaks, df.index, min_distance=args.order * 2)
    triple_tops = detect_triple_top(peaks, troughs, df.index, min_distance=args.order * 2)
    triple_bottoms = detect_triple_bottom(troughs, peaks, df.index, min_distance=args.order * 2)

    # Print detected patterns
    print("\nDetected Patterns:")
    print_patterns(hs_tops)
    print_patterns(hs_bottoms)
    print_patterns(triple_tops)
    print_patterns(triple_bottoms)

    # --- Build, Save, and Prepare All Figures for Showing ---
    
    # Build the two original figures
    fig1 = create_figure_pivot_touch(df, args.ticker, peaks, troughs)
    fig2 = create_figure_multiscale(df, args.ticker, peaks, troughs,
                                    args.short_days, args.medium_days, args.long_days)

    # CONSTRAINT: Use Output Directory in /dev/shm
    out_dir = _resolve_output_dir(args.ticker)
    all_figures = [fig1, fig2]

    # Save original figures
    _maybe_save(fig1, out_dir, f"{args.ticker}_peaks_troughs_pivot_touch", args.save_png, args.save_html)
    _maybe_save(fig2, out_dir, f"{args.ticker}_peaks_troughs_multiscale", args.save_png, args.save_html)

    # Define all pattern types and their corresponding drawing functions
    all_pattern_groups = {
        'hs_top': (hs_tops, draw_head_shoulders_top),
        'hs_bottom': (hs_bottoms, draw_head_shoulders_bottom),
        'triple_top': (triple_tops, draw_triple_top),
        'triple_bottom': (triple_bottoms, draw_triple_bottom)
    }

    # Iterate through each pattern group, create and save figures
    for pat_key, (patterns, draw_func) in all_pattern_groups.items():
        if not patterns:
            continue
        for i, pattern in enumerate(patterns):
            title = f"{args.ticker} - {pattern['type']} #{i + 1}"
            fig_pat = create_base_chart(df, args.ticker, title)
            draw_func(fig_pat, pattern)
            
            # Save the new pattern figure if requested
            _maybe_save(fig_pat, out_dir, f"{args.ticker}_{pat_key}_{i+1}", args.save_png, args.save_html)
            
            all_figures.append(fig_pat)

    # Show all figures (open in separate browser tabs)
    if not args.no_show:
        pio.renderers.default = "browser"
        for fig in all_figures:
            fig.show()

if __name__ == "__main__":
    main()
