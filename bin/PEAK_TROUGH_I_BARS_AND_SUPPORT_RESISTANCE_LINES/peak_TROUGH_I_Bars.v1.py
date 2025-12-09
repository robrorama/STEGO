#!/usr/bin/env python3
# SCRIPTNAME: peak_TROUGH_I_Bars.v1.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Unified technical analysis script:
# - Figure A: Candlestick + peak/trough markers + I-bar drawdowns + recovery boxes
#             + % drawdown/recovery labels + Volume + MACD + RSI
# - Figure B: Support/Resistance (last N sessions) with optional PNG/HTML export
#
# Data source: data_retrieval.py (cache-aware).

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

# Ensure we can import the user's data_retrieval module (adjust if needed)
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if "/mnt/data" not in sys.path:
    sys.path.insert(0, "/mnt/data")

# Import centralized data layer (required by user)
try:
    import data_retrieval
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ----------------------------
# Technical indicator routines
# ----------------------------

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """
    MACD family: returns (macd, signal, histogram)
    Mirrors logic in V9/V10 scripts.
    """
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_rsi(df: pd.DataFrame, period: int = 14):
    """
    RSI as in V9/V10: simple rolling-average version.
    """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def find_peaks_troughs(df: pd.DataFrame, order: int = 5):
    """
    Sliding-window extrema detection (NumPy-based) as in V9/V10.
    Returns lists of (idx, price) tuples referencing df index positions.
    """
    prices = df['Close'].to_numpy()
    n = len(prices)
    peaks, troughs = [], []
    for i in range(order, n - order):
        window = prices[i - order:i + order + 1]
        c = order
        if np.all(window[c] > window[:c]) and np.all(window[c] > window[c + 1:]):
            peaks.append((i, prices[i]))
        if np.all(window[c] < window[:c]) and np.all(window[c] < window[c + 1:]):
            troughs.append((i, prices[i]))
    return peaks, troughs

def _pair_peaks_troughs(peaks, troughs):
    """
    Pair each peak with the subsequent trough (I-bar pairing), consistent with V9/V10.
    Returns list of tuples: (peak_index, peak_price, trough_index, trough_price).
    """
    p_idx = [p[0] for p in peaks]
    p_val = [p[1] for p in peaks]
    t_idx = [t[0] for t in troughs]
    t_val = [t[1] for t in troughs]

    pairs = []
    i = j = 0
    while i < len(p_idx) and j < len(t_idx):
        if p_idx[i] < t_idx[j]:
            pairs.append((p_idx[i], p_val[i], t_idx[j], t_val[j]))
            i += 1
        else:
            j += 1
    # If peaks remain, pair with last seen trough (as in source logic)
    while i < len(p_idx) - 1 and j > 0:
        pairs.append((p_idx[i], p_val[i], t_idx[j - 1], t_val[j - 1]))
        i += 1
    return pairs

# ----------------------------
# Plot construction
# ----------------------------

def build_i_bar_figure(df: pd.DataFrame, ticker: str, peaks, troughs) -> go.Figure:
    """
    Create the 4-row figure with:
    row1: Candlestick + peak/trough markers + I-bar drawdowns + recovery boxes + % labels
    row2: Volume
    row3: MACD family
    row4: RSI with 70/30 reference lines
    """
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.2, 0.15, 0.15])

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlestick'
    ), row=1, col=1)

    # Markers for peaks/troughs
    peak_dates  = [df.index[i] for (i, _) in peaks]
    peak_prices = [p for (_, p) in peaks]
    trough_dates  = [df.index[i] for (i, _) in troughs]
    trough_prices = [p for (_, p) in troughs]

    if peaks:
        fig.add_trace(go.Scatter(x=peak_dates, y=peak_prices, mode='markers',
                                 name='Peaks', marker=dict(color='lime', size=10)), row=1, col=1)
    if troughs:
        fig.add_trace(go.Scatter(x=trough_dates, y=trough_prices, mode='markers',
                                 name='Troughs', marker=dict(color='red', size=10)), row=1, col=1)

    # I-bar drawdowns + recovery boxes + % labels (V10 behavior)
    pairs = _pair_peaks_troughs(peaks, troughs)
    for (pi, pprice, ti, tprice) in pairs:
        px = df.index[pi]
        tx = df.index[ti]
        # Top horizontal at peak
        fig.add_shape(type="line", x0=px, y0=pprice, x1=tx, y1=pprice,
                      line=dict(color="RoyalBlue", width=2), row=1, col=1)
        # Bottom horizontal at trough
        fig.add_shape(type="line", x0=px, y0=tprice, x1=tx, y1=tprice,
                      line=dict(color="RoyalBlue", width=2), row=1, col=1)
        # Vertical connector at midpoint
        midx = px + (tx - px) / 2
        fig.add_shape(type="line", x0=midx, y0=pprice, x1=midx, y1=tprice,
                      line=dict(color="RoyalBlue", width=2), row=1, col=1)

        # Drawdown % label at trough
        dd_pct = (tprice - pprice) / pprice * 100.0
        fig.add_annotation(x=tx, y=tprice,
                           text=f"{dd_pct:.1f}%", showarrow=False,
                           font=dict(size=10, color="black"),
                           row=1, col=1, yshift=-10)

    # Recovery boxes and % labels between trough_i and next peak_{i+1}
    for k in range(len(pairs) - 1):
        _, _, tr_i, tr_p = pairs[k]
        nxt_p_i, nxt_p_p, _, _ = pairs[k + 1]
        tr_x = df.index[tr_i]
        np_x = df.index[nxt_p_i]

        fig.add_shape(type="rect", x0=tr_x, y0=tr_p, x1=np_x, y1=nxt_p_p,
                      line=dict(color="gold", width=0),
                      fillcolor="gold", opacity=0.30, row=1, col=1)

        rec_pct = (nxt_p_p - tr_p) / tr_p * 100.0
        fig.add_annotation(x=np_x, y=nxt_p_p,
                           text=f"{rec_pct:.1f}%", showarrow=False,
                           font=dict(size=10, color="black"),
                           row=1, col=1, yshift=10)

    # Volume
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

    # MACD family
    macd, signal, hist = calculate_macd(df)
    fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'),   row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=hist, name='Histogram'),   row=3, col=1)

    # RSI
    rsi = calculate_rsi(df)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI'), row=4, col=1)

    # Horizontal lines at 70/30 (using add_hline if available)
    try:
        fig.add_hline(y=70, line_dash="dot", annotation_text="Overbought",
                      annotation_position="bottom right", row=4, col=1)
        fig.add_hline(y=30, line_dash="dot", annotation_text="Oversold",
                      annotation_position="top right", row=4, col=1)
    except Exception:
        # Fallback: shapes
        xmin, xmax = df.index.min(), df.index.max()
        for yval in (70, 30):
            fig.add_shape(type="line", x0=xmin, x1=xmax, y0=yval, y1=yval,
                          line=dict(dash="dot"), row=4, col=1)

    fig.update_layout(title=f"{ticker} — Peaks/Troughs I‑Bars + Indicators",
                      xaxis_rangeslider_visible=False, height=1000)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    return fig

def build_support_resistance_figure(df: pd.DataFrame, ticker: str, last_n: int = 30) -> go.Figure:
    """
    Support/Resistance based on min/max Close over the last N sessions,
    matching the logic from the separate trend-line script.
    """
    if last_n <= 0 or last_n > len(df):
        last_n = min(30, len(df))

    recent = df.iloc[-last_n:]
    sup_val = recent['Close'].min()
    res_val = recent['Close'].max()

    fig = go.Figure(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlesticks'
    ))

    # Horizontal lines for S/R across full chart range
    x0, x1 = df.index[0], df.index[-1]
    fig.add_trace(go.Scatter(x=[x0, x1], y=[res_val, res_val],
                             mode="lines", name="Resistance"))
    fig.add_trace(go.Scatter(x=[x0, x1], y=[sup_val, sup_val],
                             mode="lines", name="Support"))

    fig.update_layout(title=f"{ticker} — Support/Resistance (last {last_n} sessions)",
                      xaxis_title="Date", yaxis_title="Price",
                      xaxis_rangeslider_visible=False)
    return fig

# ----------------------------
# I/O helpers
# ----------------------------

def _resolve_output_dir(ticker: str) -> str:
    # Use the date-stamped directory structure from data_retrieval
    return data_retrieval.create_output_directory(ticker)

def _maybe_save(fig: go.Figure, out_dir: str, out_name: str, save_png: bool, save_html: bool):
    # CONSTRAINT: Path is guaranteed to be in /dev/shm via _resolve_output_dir
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
            print(f"[warn] PNG export needs 'kaleido' installed: {e}")

# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Unified technical analysis using data_retrieval.")
    p.add_argument("ticker", help="Symbol, e.g., AAPL")
    date = p.add_mutually_exclusive_group()
    date.add_argument("--period", default="1y",
                      help="yfinance period string (e.g., 6mo, 1y, 2y, max). Default: 1y")
    date.add_argument("--date-range", metavar=("START,END"),
                      help="Explicit date range as 'YYYY-MM-DD,YYYY-MM-DD'")

    p.add_argument("--order", type=int, default=5, help="Peak/trough window half-size. Default: 5")
    p.add_argument("--sr-last-n", type=int, default=30,
                   help="Sessions for S/R window. Default: 30")
    p.add_argument("--save-png", action="store_true", help="Save PNGs (requires kaleido).")
    p.add_argument("--save-html", action="store_true", help="Save HTML.")
    p.add_argument("--no-show", action="store_true", help="Do not open figures in browser.")
    args = p.parse_args()

    # Data retrieval strictly via data_retrieval.py
    if args.date_range:
        try:
            start, end = [s.strip() for s in args.date_range.split(",")]
        except ValueError:
            raise SystemExit("Error: --date-range must be 'YYYY-MM-DD,YYYY-MM-DD'")
        df = data_retrieval.load_or_download_ticker(args.ticker, start=start, end=end)
    else:
        df = data_retrieval.get_stock_data(args.ticker, period=args.period)

    if df is None or df.empty:
        raise SystemExit("No data retrieved.")

    # Ensure time index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Compute extrema
    peaks, troughs = find_peaks_troughs(df, order=args.order)

    # Build figures
    fig_a = build_i_bar_figure(df, args.ticker, peaks, troughs)
    fig_b = build_support_resistance_figure(df, args.ticker, last_n=args.sr_last_n)

    out_dir = _resolve_output_dir(args.ticker)
    _maybe_save(fig_a, out_dir, f"{args.ticker}_ibars_indicators", args.save_png, args.save_html)
    _maybe_save(fig_b, out_dir, f"{args.ticker}_support_resistance", args.save_png, args.save_html)

    # Open different plots in different browser tabs/windows
    if not args.no_show:
        pio.renderers.default = "browser"
        fig_a.show()
        fig_b.show()

if __name__ == "__main__":
    main()
