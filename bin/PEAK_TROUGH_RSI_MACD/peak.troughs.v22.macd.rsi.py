#!/usr/bin/env python3
# SCRIPTNAME: peak.troughs.v21.with.macd.rsi.works.just.ignores.most.recent.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Enhancements:
#   - Volume bars color-coded (green on up days, red on down days) in the original price/volume figure.
#   - Adds a SECOND figure with:
#       * MACD(12,26,9): MACD line, signal line, histogram
#       * RSI(14) with 70/30 guides
#     For BOTH MACD and RSI panes:
#       * Peak & trough markers (local extrema on the series)
#       * THREE LR lines from those extrema (short/med/long), same style as price figure.
#
# Usage:
#   python3 peaks_troughs_multiscale_with_indicators.py TICKER \
#       --period 1y \
#       --order 5 \
#       --short-days 60 --medium-days 120 --long-days 252 \
#       [--date-range START,END] \
#       [--save-png] [--save-html] [--no-show]
#
# Notes:
#   - Requires: plotly, numpy, pandas (and kaleido if saving PNG)
#   - Must have data_retrieval.py in PYTHONPATH or same dir.

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
# Import data_retrieval (REQUIRED)  (uses your cached loader)
# ----------------------------
try:
    import data_retrieval
except ImportError:
    print("Error: 'data_retrieval.py' not found in directory.")
    sys.exit(1)


# ----------------------------
# Peak/Trough Detection (generic)
# ----------------------------
def find_peaks_troughs_ohlc(df: pd.DataFrame, order: int = 5):
    """
    Wick-centered local extrema for OHLC:
    Returns:
        peaks   : list of (index_i, high_value)
        troughs : list of (index_i, low_value)
    """
    highs = df['High'].to_numpy()
    lows  = df['Low'].to_numpy()
    peaks, troughs = [], []
    n = len(df)
    for i in range(order, n - order):
        if highs[i] == np.max(highs[i - order:i + order + 1]):
            peaks.append((i, highs[i]))
        if lows[i] == np.min(lows[i - order:i + order + 1]):
            troughs.append((i, lows[i]))
    return peaks, troughs


def find_peaks_troughs_series(series: pd.Series, order: int = 5):
    """
    Local extrema on a single series.
    Returns:
        peaks   : list of (i, y)
        troughs : list of (i, y)
    """
    y = series.to_numpy()
    peaks, troughs = [], []
    n = len(series)
    for i in range(order, n - order):
        seg = y[i - order:i + order + 1]
        if np.isfinite(y[i]):
            if y[i] == np.nanmax(seg):
                peaks.append((i, y[i]))
            if y[i] == np.nanmin(seg):
                troughs.append((i, y[i]))
    return peaks, troughs


# ----------------------------
# Helpers for multiscale regression lines
# ----------------------------
def _select_points_in_window(points, last_index: int, window_days: int):
    if window_days <= 0:
        return []
    start_idx = max(0, last_index - window_days + 1)
    return [(i, y) for (i, y) in points if i >= start_idx]


def _fit_linear_regression(points_subset):
    x = np.array([i for (i, _) in points_subset], dtype=float)
    y = np.array([y for (_, y) in points_subset], dtype=float)
    m, b = np.polyfit(x, y, 1)
    return m, b, int(x.min())


def build_lr_line(points, window_days: int, n_bars: int, df_index):
    """
    Returns (x_dates, y_vals) for a 2-point line spanning from earliest point
    used in the fit to the last bar in the chart.
    """
    if not points or window_days <= 0:
        return [], []
    inwin = _select_points_in_window(points, n_bars - 1, window_days)
    if len(inwin) < 2:
        return [], []
    m, b, x0 = _fit_linear_regression(inwin)
    x_start = x0
    x_end   = n_bars - 1
    y_start = m * x_start + b
    y_end   = m * x_end + b
    return [df_index[x_start], df_index[x_end]], [y_start, y_end]


# ----------------------------
# Indicators: MACD & RSI
# ----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series_close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series_close, fast)
    ema_slow = ema(series_close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series_close: pd.Series, period: int = 14):
    # Wilder's RSI
    delta = series_close.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


# ----------------------------
# PRICE + VOLUME (with colored volume) FIGURE
# ----------------------------
def create_price_volume_chart(df: pd.DataFrame,
                              ticker: str,
                              peaks_pt, troughs_pt,
                              short_days: int, medium_days: int, long_days: int) -> go.Figure:
    n = len(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candlestick"
    ), row=1, col=1)

    # Peaks on price
    if peaks_pt:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in peaks_pt],
            y=[p for (_, p) in peaks_pt],
            mode="markers",
            name="Peaks",
            marker=dict(symbol="circle", size=34,
                        color="rgba(0,160,0,0.28)",
                        line=dict(color="darkgreen", width=1.5))
        ), row=1, col=1)

    # Troughs on price
    if troughs_pt:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in troughs_pt],
            y=[t for (_, t) in troughs_pt],
            mode="markers",
            name="Troughs",
            marker=dict(symbol="circle", size=34,
                        color="rgba(200,0,0,0.28)",
                        line=dict(color="darkred", width=1.5))
        ), row=1, col=1)

    # Multiscale LR lines on price (green for peaks, red for troughs)
    # Peak lines
    for days, width, dash, label in [
        (short_days, 1.5, "dot",  f"Peak LR (short {short_days}d)"),
        (medium_days, 2.0, "dash", f"Peak LR (med {medium_days}d)"),
        (long_days, 2.8, None,     f"Peak LR (long {long_days}d)"),
    ]:
        xs, ys = build_lr_line(peaks_pt, days, n, df.index)
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=label,
                line=dict(color="green", width=width, dash=dash) if dash else dict(color="green", width=width)
            ), row=1, col=1)
    # Trough lines
    for days, width, dash, label in [
        (short_days, 1.5, "dot",  f"Trough LR (short {short_days}d)"),
        (medium_days, 2.0, "dash", f"Trough LR (med {medium_days}d)"),
        (long_days, 2.8, None,     f"Trough LR (long {long_days}d)"),
    ]:
        xs, ys = build_lr_line(troughs_pt, days, n, df.index)
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=label,
                line=dict(color="red", width=width, dash=dash) if dash else dict(color="red", width=width)
            ), row=1, col=1)

    # Colored volume: green for up-day (Close >= Open), red otherwise
    vol_colors = np.where(df['Close'] >= df['Open'], "rgba(0,160,0,0.75)", "rgba(200,0,0,0.75)")
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.9
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"{ticker} — Multiscale Peak/Trough LR (Price) + Colored Volume",
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# ----------------------------
# INDICATOR FIGURE (MACD + RSI) with peaks/troughs & LR lines
# ----------------------------
def create_indicator_chart(df: pd.DataFrame,
                           ticker: str,
                           order: int,
                           short_days: int, medium_days: int, long_days: int) -> go.Figure:
    n = len(df)

    # Compute indicators
    macd_line, signal_line, macd_hist = macd(df['Close'])
    rsi_line = rsi(df['Close'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.55, 0.45],
                        subplot_titles=(f"MACD(12,26,9) — {ticker}", f"RSI(14) — {ticker}"))

    # ---- MACD pane ----
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_line, name="MACD", mode="lines"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=signal_line, name="Signal", mode="lines"
    ), row=1, col=1)
    # Histogram (bar)
    fig.add_trace(go.Bar(
        x=df.index, y=macd_hist, name="Hist"
    ), row=1, col=1)

    # Peaks/troughs on MACD line
    macd_peaks, macd_troughs = find_peaks_troughs_series(macd_line, order=order)
    if macd_peaks:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in macd_peaks],
            y=[v for (_, v) in macd_peaks],
            mode="markers", name="MACD Peaks",
            marker=dict(symbol="circle", size=20,
                        color="rgba(0,160,0,0.25)",
                        line=dict(color="darkgreen", width=1.2))
        ), row=1, col=1)
    if macd_troughs:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in macd_troughs],
            y=[v for (_, v) in macd_troughs],
            mode="markers", name="MACD Troughs",
            marker=dict(symbol="circle", size=20,
                        color="rgba(200,0,0,0.25)",
                        line=dict(color="darkred", width=1.2))
        ), row=1, col=1)

    # LR lines on MACD peaks/troughs
    for pts, color, label_base in [
        (macd_peaks,   "green", "MACD Peak LR"),
        (macd_troughs, "red",   "MACD Trough LR"),
    ]:
        for days, width, dash in [
            (short_days, 1.5, "dot"),
            (medium_days, 2.0, "dash"),
            (long_days, 2.8, None),
        ]:
            xs, ys = build_lr_line(pts, days, n, df.index)
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", name=f"{label_base} ({days}d)",
                    line=dict(color=color, width=width, dash=dash) if dash else dict(color=color, width=width)
                ), row=1, col=1)

    fig.update_yaxes(title_text="MACD", row=1, col=1)

    # ---- RSI pane ----
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi_line, name="RSI(14)", mode="lines"
    ), row=2, col=1)
    # 70/30 guides
    fig.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]], y=[70, 70],
        mode="lines", name="RSI 70", line=dict(dash="dash")
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]], y=[30, 30],
        mode="lines", name="RSI 30", line=dict(dash="dash")
    ), row=2, col=1)

    # Peaks/troughs on RSI
    rsi_peaks, rsi_troughs = find_peaks_troughs_series(rsi_line, order=order)
    if rsi_peaks:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in rsi_peaks],
            y=[v for (_, v) in rsi_peaks],
            mode="markers", name="RSI Peaks",
            marker=dict(symbol="circle", size=16,
                        color="rgba(0,160,0,0.25)",
                        line=dict(color="darkgreen", width=1.0))
        ), row=2, col=1)
    if rsi_troughs:
        fig.add_trace(go.Scatter(
            x=[df.index[i] for (i, _) in rsi_troughs],
            y=[v for (_, v) in rsi_troughs],
            mode="markers", name="RSI Troughs",
            marker=dict(symbol="circle", size=16,
                        color="rgba(200,0,0,0.25)",
                        line=dict(color="darkred", width=1.0))
        ), row=2, col=1)

    # LR lines on RSI peaks/troughs
    for pts, color, label_base in [
        (rsi_peaks,   "green", "RSI Peak LR"),
        (rsi_troughs, "red",   "RSI Trough LR"),
    ]:
        for days, width, dash in [
            (short_days, 1.5, "dot"),
            (medium_days, 2.0, "dash"),
            (long_days, 2.8, None),
        ]:
            xs, ys = build_lr_line(pts, days, n, df.index)
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", name=f"{label_base} ({days}d)",
                    line=dict(color=color, width=width, dash=dash) if dash else dict(color=color, width=width)
                ), row=2, col=1)

    fig.update_yaxes(title_text="RSI", row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — MACD & RSI with Peaks/Troughs + Multiscale LR",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    return fig


# ----------------------------
# Save helpers
# ----------------------------
def save_figure(fig: go.Figure, base_path: str, save_png: bool, save_html: bool):
    if not (save_png or save_html):
        return
    if save_html:
        fig.write_html(base_path + ".html", include_plotlyjs="cdn")
        print(f"[+] Saved HTML: {base_path}.html")
    if save_png:
        try:
            fig.write_image(base_path + ".png", width=1920, height=1080, scale=2)
            print(f"[+] Saved PNG: {base_path}.png")
        except Exception as e:
            print("PNG export failed — install kaleido. Error:", e)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Price/Volume with colored volume + separate MACD/RSI figure, all using data_retrieval."
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--period", default="1y",
                        help="yfinance period if --date-range is not provided (default: 1y)")
    parser.add_argument("--date-range", metavar="START,END",
                        help="Explicit date range (YYYY-MM-DD,YYYY-MM-DD)")
    parser.add_argument("--order", type=int, default=5,
                        help="Local-extrema half-window for peak/trough detection (default: 5)")
    parser.add_argument("--short-days", type=int, default=60,
                        help="Short lookback window in trading days (default: 60)")
    parser.add_argument("--medium-days", type=int, default=120,
                        help="Medium lookback window in trading days (default: 120)")
    parser.add_argument("--long-days", type=int, default=252,
                        help="Long lookback window in trading days (default: 252)")
    parser.add_argument("--save-png", action="store_true", help="Save PNG (requires kaleido)")
    parser.add_argument("--save-html", action="store_true", help="Save HTML")
    parser.add_argument("--no-show", action="store_true", help="Do not open a browser window")
    args = parser.parse_args()

    # --- Load data strictly through data_retrieval ---
    if args.date_range:
        start, end = [s.strip() for s in args.date_range.split(",")]
        df = data_retrieval.load_or_download_ticker(args.ticker, start=start, end=end)
    else:
        df = data_retrieval.get_stock_data(args.ticker, period=args.period)

    if df.empty:
        sys.exit("No data found.")

    # Ensure expected columns
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        sys.exit(f"Data is missing required columns: {missing}")

    # Peaks/troughs for PRICE (wick-based)
    peaks_pt, troughs_pt = find_peaks_troughs_ohlc(df, order=args.order)

    # Figure 1: price + colored volume
    fig_price = create_price_volume_chart(
        df, args.ticker, peaks_pt, troughs_pt,
        args.short_days, args.medium_days, args.long_days
    )

    # Figure 2: indicators (MACD + RSI) with peaks/troughs & LR lines
    fig_ind = create_indicator_chart(
        df, args.ticker, args.order,
        args.short_days, args.medium_days, args.long_days
    )

    # Save both
    # CONSTRAINT: Output must be in /dev/shm via data_retrieval
    out_dir = data_retrieval.create_output_directory(args.ticker)
    save_figure(fig_price, os.path.join(out_dir, f"{args.ticker}_price_volume_multiscale"), args.save_png, args.save_html)
    save_figure(fig_ind,   os.path.join(out_dir, f"{args.ticker}_indicators_macd_rsi_multiscale"), args.save_png, args.save_html)

    # Show both
    if not args.no_show:
        pio.renderers.default = "browser"
        fig_price.show()
        fig_ind.show()


if __name__ == "__main__":
    main()
