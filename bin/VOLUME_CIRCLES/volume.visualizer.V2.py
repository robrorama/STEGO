#!/usr/bin/env python3
# SCRIPTNAME: volume.visualizer.V2.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Unified interesting-volume visualizer (00++)
#
# Key additions:
# - CLI options: --zscore, --hv-mode {zonly,strict}, --rec-threshold, --vol-color {up,down},
#                --no-tabs, --no-gif, --extra-tabs
# - High-volume overlays: bubbles sized by volume; optional strict criterion (Z>thr AND Vol>μ+2σ)
# - Seasonal recurrence marking (>= N occurrences of the same MM-DD) with gold stars
# - "Max" time window
# - Animated GIF across windows (price-panel PNGs)
# - Optional extra tabs: separate Price+Volume and Oscillators
# - Daily CSV snapshot to the canonical dated output directory

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

from ta.trend import MACD
from ta.momentum import StochasticOscillator, RSIIndicator

import imageio.v2 as imageio

# Canonical data & dirs (cache-first, env-driven)
# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import (
        load_or_download_ticker,
        create_output_directory,
        IMAGES_SUBDIR,
    )
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ------------------------------- period utils --------------------------------
VALID_PERIODS = {
    "1d": "1d", "5d": "5d",
    "1m": "1mo", "1mo": "1mo",
    "3m": "3mo", "3mo": "3mo",
    "6m": "6mo", "6mo": "6mo",
    "1y": "1y", "2y": "2y", "5y": "5y", "10y": "10y",
    "ytd": "ytd", "max": "max",
}
def normalize_period(token: str | None) -> str | None:
    if not token:
        return None
    return VALID_PERIODS.get(token.strip().lower(), token.strip().lower())


# ------------------------------- data access ---------------------------------
def fetch_data(ticker: str, period: str | None = None,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """All paths go through data_retrieval.load_or_download_ticker."""
    if period:
        return load_or_download_ticker(ticker, period=period)
    return load_or_download_ticker(ticker, start=start, end=end)


# ------------------------------ calculations ---------------------------------
def add_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Price-based indicators (MAs, MACD, Stochastic, RSI, Bollinger)."""
    out = df.copy()
    for s in [50, 100, 150, 200, 300]:
        out[f"MA{s}"] = out["Close"].rolling(window=s, min_periods=1).mean()

    macd = MACD(close=out["Close"])
    out["MACD"]        = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    out["MACD_Diff"]   = macd.macd_diff()

    stoch = StochasticOscillator(high=out["High"], close=out["Close"], low=out["Low"])
    out["Stoch"]        = stoch.stoch()
    out["Stoch_Signal"] = stoch.stoch_signal()

    out["RSI"] = RSIIndicator(close=out["Close"]).rsi()

    ma20  = out["Close"].rolling(window=20, min_periods=1).mean()
    std20 = out["Close"].rolling(window=20, min_periods=1).std(ddof=0)
    out["Bollinger_High"] = ma20 + 2 * std20
    out["Bollinger_Low"]  = ma20 - 2 * std20
    return out


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-series indicators (MAs, MACD, Stochastic, RSI, Bollinger on Volume)."""
    out = df.copy()
    for s in [50, 100, 150, 200, 300]:
        out[f"MA{s}"] = out["Volume"].rolling(window=s, min_periods=1).mean()

    macd = MACD(close=out["Volume"])
    out["MACD"]        = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    out["MACD_Diff"]   = macd.macd_diff()

    stoch = StochasticOscillator(high=out["Volume"], close=out["Volume"], low=out["Volume"])
    out["Stoch"]        = stoch.stoch()
    out["Stoch_Signal"] = stoch.stoch_signal()

    out["RSI"] = RSIIndicator(close=out["Volume"]).rsi()

    ma20  = out["Volume"].rolling(window=20, min_periods=1).mean()
    std20 = out["Volume"].rolling(window=20, min_periods=1).std(ddof=0)
    out["Bollinger_High"] = ma20 + 2 * std20
    out["Bollinger_Low"]  = ma20 - 2 * std20
    return out


def compute_high_volume_days(df: pd.DataFrame, z_thresh: float = 2.0,
                             mode: str = "zonly") -> pd.DataFrame:
    """
    High-volume detection:
      mode='zonly'  -> Vol_ZScore > z_thresh
      mode='strict' -> (Vol_ZScore > z_thresh) and (Volume > MA20 + 2*STD20)
    """
    out = df.copy()
    out["Vol_MA20"]  = out["Volume"].rolling(window=20, min_periods=1).mean()
    vol_std20        = out["Volume"].rolling(window=20, min_periods=1).std(ddof=0)
    vol_std20_safe   = vol_std20.replace(0, np.nan)
    out["Vol_ZScore"] = ((out["Volume"] - out["Vol_MA20"]) / vol_std20_safe).fillna(0.0)

    if mode == "strict":
        hv = out[(out["Vol_ZScore"] > z_thresh) & (out["Volume"] > (out["Vol_MA20"] + 2 * vol_std20))]
    else:
        hv = out[out["Vol_ZScore"] > z_thresh]

    hv = hv.copy()
    if hv.empty:
        hv["MarkerSize"] = []
        return hv

    # Bubble sizing
    min_marker, max_marker = 50.0, 300.0
    min_vol = float(hv["Volume"].min())
    max_vol = float(hv["Volume"].max())
    rng     = max(1.0, max_vol - min_vol)
    hv["MarkerSize"] = ((hv["Volume"] - min_vol) / rng) * (max_marker - min_marker) + min_marker

    # For recurrence analysis
    hv["MonthDay"] = hv.index.strftime("%m-%d")
    return hv


def identify_recurring_high_volume_days(hv: pd.DataFrame, min_count: int = 3) -> dict[str, int]:
    """Month-day recurrences among high-volume dates; returns {MM-DD: count}."""
    if hv.empty:
        return {}
    counts = Counter(hv["MonthDay"])
    return {k: v for k, v in counts.items() if v >= min_count}


# -------------------------------- plotting -----------------------------------
def _volume_colors(df: pd.DataFrame, rule: str = "up"):
    """
    rule='up'   -> green if Close >= Open (00's logic)
    rule='down' -> green if (Open - Close) >= 0 (matches 01's legacy logic)
    """
    if rule == "down":
        return ["green" if (o - c) >= 0 else "red" for o, c in zip(df["Open"], df["Close"])]
    return ["green" if (c - o) >= 0 else "red" for o, c in zip(df["Open"], df["Close"])]


def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "-")


def _write_html_and_png(fig, outdir: str, ticker: str, period_name: str, suffix: str):
    """Write HTML+PNG; return png path if succeeded (else None)."""
    os.makedirs(outdir, exist_ok=True)
    safe = _safe_name(period_name)
    html_fp = os.path.join(outdir, f"{ticker}_{safe}_{suffix}.html")
    png_fp  = os.path.join(outdir, f"{ticker}_{safe}_{suffix}.png")
    fig.write_html(html_fp)
    ok = False
    try:
        fig.write_image(png_fp, width=1920, height=1080, scale=4)
        ok = True
    except Exception as e:
        print(f"[warn] write_image failed for {png_fp}: {e}")
        png_fp = None
    return png_fp


def plot_price_panel(df: pd.DataFrame, hv: pd.DataFrame, recurring: dict[str, int],
                     ticker: str, period_name: str, outdir: str,
                     vol_color_rule: str = "up", show: bool = True) -> str | None:
    """5-row price panel (candles+Bollinger, MACD, Volume, Stoch, RSI) + HV overlays and recurring stars."""
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("Stock Data", "MACD", "Volume", "Stochastic", "RSI"),
        row_heights=[0.60, 0.10, 0.10, 0.10, 0.10]
    )

    # Price panel + Bollinger
    fig.add_trace(
        go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                       low=df["Low"], close=df["Close"], name="Price"),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_High"], name="Boll High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_Low"],  name="Boll Low"),  row=1, col=1)

    # MACD panel
    macd_colors = ["green" if v >= 0 else "red" for v in df["MACD_Diff"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Diff"], marker_color=macd_colors, name="MACD Diff"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        name="MACD"),        row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"),      row=2, col=1)

    # Volume panel (choose coloring rule)
    vol_colors = _volume_colors(df, rule=vol_color_rule)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"), row=3, col=1)

    # High-volume day markers on price (bubbles + text labels only)
    if not hv.empty:
        fig.add_trace(
            go.Scatter(x=hv.index, y=hv["Close"], mode="markers+text",
                       marker=dict(size=hv["MarkerSize"], opacity=0.5, line=dict(width=2), symbol="circle"),
                       text=[f"{v/1e6:.1f}M" for v in hv["Volume"]],
                       textposition="middle center", name="High Vol (Z)"),
            row=1, col=1
        )
        # Gold stars for recurring MM-DDs
        for mmdd, _cnt in recurring.items():
            dates = hv.index[hv["MonthDay"] == mmdd]
            if len(dates) > 0:
                fig.add_trace(go.Scatter(
                    x=dates, y=df.loc[df.index.isin(dates), "Close"],
                    mode="markers", marker=dict(size=15, color="gold", symbol="star"),
                    name=f"Recurring {mmdd}"), row=1, col=1)

    # Stochastic + RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch"],        name="Stoch"),     row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_Signal"], name="Stoch Sig"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],          name="RSI"),       row=5, col=1)

    fig.update_layout(
        title=f"{ticker} — {period_name} Price Analysis",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        height=1000, template="plotly_dark"
    )

    if show:
        fig.show()
    return _write_html_and_png(fig, outdir, ticker, period_name, suffix="price")


def plot_volume_analytics(df: pd.DataFrame, ticker: str, period_name: str,
                          outdir: str, show: bool = True) -> None:
    """5-row volume analytics figure (Bollinger/Vol, MACD, MA50, Stoch, RSI)."""
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("Volume Data", "MACD", "MA50", "Stochastic", "RSI"),
        row_heights=[0.60, 0.10, 0.10, 0.10, 0.10]
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Volume"], name="Volume"),            row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_High"], name="Boll High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_Low"],  name="Boll Low"),  row=1, col=1)

    macd_colors = ["green" if v >= 0 else "red" for v in df["MACD_Diff"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Diff"], marker_color=macd_colors, name="MACD Diff"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        name="MACD"),        row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"),      row=2, col=1)

    if "MA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50 (Vol)"),     row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch"],        name="Stoch"),      row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_Signal"], name="Stoch Sig"),  row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],          name="RSI"),        row=5, col=1)

    fig.update_layout(
        title=f"{ticker} — {period_name} Volume Analysis",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        height=1000, template="plotly_dark"
    )

    if show:
        fig.show()
    _write_html_and_png(fig, outdir, ticker, period_name, suffix="volume")


# --------- optional extra tabs (from script 01) ----------
def make_price_volume_tab(df: pd.DataFrame, hv: pd.DataFrame, ticker: str, period_name: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("Price (Bollinger & High-Vol Bubbles)", "Volume"),
                        row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="Market Data"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_High"], name="Boll High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_Low"],  name="Boll Low"),  row=1, col=1)
    if not hv.empty:
        fig.add_trace(go.Scatter(x=hv.index, y=hv["Close"], mode="markers+text",
                                 marker=dict(size=hv["MarkerSize"], line=dict(width=2), symbol="circle"),
                                 text=[f"{v/1e6:.1f}M" for v in hv["Volume"]],
                                 textposition="middle center", name="High Vol"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(height=900, width=1200, showlegend=False, xaxis_rangeslider_visible=False,
                      title=f"{ticker} — Price & Volume — {period_name}")
    return fig


def make_oscillators_tab(df: pd.DataFrame, ticker: str, period_name: str):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("MACD", "Stochastic", "RSI"),
                        row_heights=[0.34, 0.33, 0.33])
    macd_diff = df["MACD_Diff"].fillna(0.0)
    macd_colors = ["green" if v >= 0 else "red" for v in macd_diff]
    fig.add_trace(go.Bar(x=df.index, y=macd_diff, marker_color=macd_colors, name="MACD Diff"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        name="MACD"),        row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"),      row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch"],        name="Stoch"),      row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_Signal"], name="Stoch Sig"),  row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],          name="RSI"),        row=3, col=1)
    fig.update_layout(height=900, width=1200, showlegend=False, xaxis_rangeslider_visible=False,
                      title=f"{ticker} — Oscillators — {period_name}")
    return fig


# ----------------------------------- CLI -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Canonical 00 visualizer with 01/02 features merged in (no 'Z' text).")
    ap.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    ap.add_argument("date_args", nargs="*", help="[PERIOD | START_DATE END_DATE]")
    ap.add_argument("--zscore", type=float, default=2.0, help="Z-score threshold for high-volume bubbles (default: 2.0)")
    ap.add_argument("--hv-mode", choices=["zonly", "strict"], default="zonly",
                    help="High-volume criterion: zonly=Z>thr; strict=Z>thr and Volume>μ+2σ (default: zonly)")
    ap.add_argument("--rec-threshold", type=int, default=3,
                    help="Minimum occurrences for recurring MM-DD stars (default: 3)")
    ap.add_argument("--vol-color", choices=["up", "down"], default="up",
                    help="Volume bar coloring rule: up=green if Close>=Open; down=green if Open>=Close (default: up)")
    ap.add_argument("--no-tabs", action="store_true", help="Do not open interactive browser tabs.")
    ap.add_argument("--no-gif", action="store_true", help="Do not build animated GIF from price PNGs.")
    ap.add_argument("--extra-tabs", action="store_true",
                    help="Also show extra tabs: (Price+Volume) and (Oscillators).")

    args = ap.parse_args()
    ticker = args.ticker.upper()

    # Period vs (start,end) parsing (keeps 00's UX)
    period = None
    start = end = None
    if len(args.date_args) == 1:
        period = normalize_period(args.date_args[0])
        if period not in set(VALID_PERIODS.values()):
            sys.exit(f"Unrecognized period: {args.date_args[0]}")
    elif len(args.date_args) >= 2:
        start, end = args.date_args[0], args.date_args[1]

    # Renderer for tabs (only if tabs enabled)
    if not args.no_tabs:
        pio.renderers.default = "browser"

    # Fetch data via canonical loader
    df = fetch_data(ticker, period=period, start=start, end=end)
    if df is None or df.empty:
        sys.exit("No data retrieved.")

    # Dated canonical output dir + per-ticker images dir (env-driven)
    outdir = create_output_directory(ticker)
    images_dir = os.path.join(IMAGES_SUBDIR(), ticker)
    os.makedirs(images_dir, exist_ok=True)

    # Daily snapshot CSV (full series)
    snapshot_csv = os.path.join(outdir, f"{ticker}.csv")
    try:
        df.to_csv(snapshot_csv, index_label="Date")
    except Exception as e:
        print(f"[warn] snapshot CSV not written: {e}")

    # Indicators
    price_df = add_price_indicators(df.copy())
    vol_df   = add_volume_indicators(df.copy())

    # HV detection + recurrence (configurable)
    hv_all   = compute_high_volume_days(price_df, z_thresh=args.zscore, mode=args.hv_mode)
    recurring = identify_recurring_high_volume_days(hv_all, min_count=args.rec_threshold)

    # Anchor to last trading bar to avoid empty windows on non-trading days
    anchor = pd.Timestamp(price_df.index.max())

    # Time windows (adds 'Max')
    ranges = {
        "Last 1 day":    timedelta(days=1),
        "Last 7 days":   timedelta(days=7),
        "Last 1 Month":  timedelta(days=30),
        "Last 3 Months": timedelta(days=90),
        "Last 6 Months": timedelta(days=180),
        "Last Year":     timedelta(days=365),
        "Last 2 Years":  timedelta(days=365*2),
        "Last 5 Years":  timedelta(days=365*5),
        "Max":           None,  # special case handled below
    }

    # Build figures per window; collect price PNGs for GIF
    price_pngs_for_gif: list[str] = []

    for name, delta in ranges.items():
        if name == "Max":
            start_cut = price_df.index.min()
        else:
            start_cut = anchor - delta

        price_slice = price_df[price_df.index >= start_cut]
        vol_slice   = vol_df[vol_df.index >= start_cut]
        hv_slice    = hv_all[hv_all.index >= start_cut]

        if price_slice.empty:
            continue

        # Price panel (with HV bubbles and recurring stars; NO 'Z' overlay)
        png_path = plot_price_panel(price_slice, hv_slice, recurring, ticker, name, outdir,
                                    vol_color_rule=args.vol_color, show=not args.no_tabs)
        if png_path:
            price_pngs_for_gif.append(png_path)

        # Volume analytics (indicators computed on Volume)
        if not vol_slice.empty:
            plot_volume_analytics(vol_slice, ticker, name, outdir, show=not args.no_tabs)

        # Optional extra tabs
        if args.extra_tabs:
            try:
                price_vol_fig = make_price_volume_tab(price_slice, hv_slice, ticker, name)
                osc_fig       = make_oscillators_tab(price_slice, ticker, name)
                if not args.no_tabs:
                    price_vol_fig.show()
                    osc_fig.show()
                # Save these optional tabs as well
                try:
                    price_vol_fig.write_image(os.path.join(images_dir, f"{ticker}_{_safe_name(name)}_price.png"),
                                              width=1920, height=1080, scale=4)
                    osc_fig.write_image(os.path.join(images_dir, f"{ticker}_{_safe_name(name)}_osc.png"),
                                        width=1920, height=1080, scale=4)
                except Exception as e:
                    print(f"[warn] extra tab image save failed: {e}")
            except Exception as e:
                print(f"[warn] extra tabs failed: {e}")

    # GIF across windows from price-panel PNGs
    if (not args.no_gif) and price_pngs_for_gif:
        gif_path = os.path.join(images_dir, f"{ticker}_macd.gif")
        try:
            with imageio.get_writer(gif_path, mode="I", duration=1, loop=0) as writer:
                for png in price_pngs_for_gif:
                    try:
                        writer.append_data(imageio.imread(png))
                    except Exception as e:
                        print(f"[warn] skipping {png} in GIF: {e}")
            print(f"GIF created at {gif_path}")
        except Exception as e:
            print(f"[warn] GIF creation failed: {e}")

if __name__ == "__main__":
    main()
