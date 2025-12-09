#!/usr/bin/env python3
# SCRIPTNAME: ok.corr_sentinel_usd_gold_oil.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
corr_sentinel_usd_gold_oil.py

Standalone "USD–Gold–Oil Correlation Sentinel" script.

Author: (you)
Description:
    - Downloads daily prices for:
        * US Dollar Index (default: DX-Y.NYB)
        * Gold futures continuous (GC=F)
        * WTI crude oil futures continuous (CL=F)
    - Computes daily returns and 60-day rolling Pearson correlations:
        * corr_usd_gold = corr(DXY, Gold)
        * corr_usd_oil  = corr(DXY, Oil)
    - Defines a "VOL-REGIME WARNING" when:
        * corr_usd_gold > usd_gold_hi   (default: -0.4)
        * corr_usd_oil  < usd_oil_lo    (default:  0.2)
      i.e., warning = (corr_usd_gold > -0.4) & (corr_usd_oil < 0.2)

Outputs:
    - CSV with prices, returns, correlations, and warning signal.
    - JSON summary with latest correlations and warning flag.
    - Plotly HTML dashboard with three tabs:
        1) USD–Gold 60D rolling correlation
        2) USD–Oil  60D rolling correlation
        3) Regime scatter: corr_usd_gold (x) vs corr_usd_oil (y)
      plus a big banner indicating WARNING vs NORMAL.

Usage (examples):
    python3 corr_sentinel_usd_gold_oil.py
    python3 corr_sentinel_usd_gold_oil.py --window 60 --smooth 5
    python3 corr_sentinel_usd_gold_oil.py --start 2015-01-01 --end 2025-11-30
    python3 corr_sentinel_usd_gold_oil.py --no-browser

Requirements:
    pip install yfinance plotly pandas numpy
"""

import argparse
import json
import os
import sys
from datetime import datetime
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


# ----------------------------
# Helpers
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USD–Gold–Oil Correlation Sentinel (standalone)."
    )
    parser.add_argument(
        "--usd-ticker",
        type=str,
        default="DX-Y.NYB",
        help="Ticker for USD index (default: DX-Y.NYB for DXY via ICE).",
    )
    parser.add_argument(
        "--gold-ticker",
        type=str,
        default="GC=F",
        help="Ticker for Gold futures continuous (default: GC=F).",
    )
    parser.add_argument(
        "--oil-ticker",
        type=str,
        default="CL=F",
        help="Ticker for WTI crude oil futures continuous (default: CL=F).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Start date (YYYY-MM-DD). Default: 2010-01-01.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling window size in days (default: 60).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Optional EMA smoothing window (days) for display only. 0 = no smoothing.",
    )
    parser.add_argument(
        "--usd-gold-hi",
        type=float,
        default=-0.4,
        help="Threshold for corr_usd_gold > this => part of warning (default: -0.4).",
    )
    parser.add_argument(
        "--usd-oil-lo",
        type=float,
        default=0.2,
        help="Threshold for corr_usd_oil < this => part of warning (default: 0.2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="corr_sentinel_output",
        help="Directory to write CSV, JSON, and HTML (default: ./corr_sentinel_output).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the HTML dashboard in the default browser.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_single_ticker(ticker: str, start: str, end: str = None) -> pd.Series:
    """
    Download daily prices for a ticker using yfinance.
    Returns the 'Adj Close' if available, otherwise 'Close' as a Series.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    print(f"[INFO] Downloading data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise RuntimeError(f"No data returned for ticker {ticker}.")

    if "Adj Close" in df.columns:
        s = df["Adj Close"].copy()
    elif "Close" in df.columns:
        s = df["Close"].copy()
    else:
        raise RuntimeError(f"Ticker {ticker}: missing 'Adj Close' / 'Close' columns.")

    s.name = ticker
    print(f"[INFO] {ticker}: {len(s)} rows downloaded.")
    return s


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from price DataFrame.
    """
    ret = price_df.pct_change().dropna()
    return ret


def rolling_corr(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling Pearson correlation for two aligned series.
    """
    return series_a.rolling(window=window).corr(series_b)


def apply_smoothing(series: pd.Series, smooth: int) -> pd.Series:
    """
    Optional EMA smoothing for display.
    """
    if smooth and smooth > 1:
        return series.ewm(span=smooth, adjust=False).mean()
    return series


def build_banner_text(warn_latest: bool) -> str:
    return "VOL-REGIME WARNING" if warn_latest else "NORMAL REGIME"


def build_banner_color(warn_latest: bool) -> str:
    return "red" if warn_latest else "green"


# ----------------------------
# Plotly Figure Builders
# ----------------------------

def build_corr_time_figure(
    dates: pd.DatetimeIndex,
    corr_series: pd.Series,
    threshold: float,
    title: str,
    yaxis_title: str,
    threshold_label: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=corr_series,
            mode="lines",
            name="Rolling correlation",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text=threshold_label,
        annotation_position="top left",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def build_regime_scatter_figure(
    corr_usd_gold: pd.Series,
    corr_usd_oil: pd.Series,
    usd_gold_hi: float,
    usd_oil_lo: float,
) -> go.Figure:
    # Align indexes and drop rows where either is NaN
    df = pd.DataFrame(
        {
            "corr_usd_gold": corr_usd_gold,
            "corr_usd_oil": corr_usd_oil,
        }
    ).dropna()

    if df.empty:
        raise RuntimeError("No non-NaN data for regime scatter plot.")

    last_idx = df.index[-1]
    last_row = df.iloc[-1]

    fig = go.Figure()

    # Historical points
    fig.add_trace(
        go.Scatter(
            x=df["corr_usd_gold"],
            y=df["corr_usd_oil"],
            mode="markers",
            name="History",
            opacity=0.5,
        )
    )

    # Last point highlighted
    fig.add_trace(
        go.Scatter(
            x=[last_row["corr_usd_gold"]],
            y=[last_row["corr_usd_oil"]],
            mode="markers+text",
            name="Last",
            marker=dict(size=12, symbol="diamond"),
            text=[last_idx.strftime("%Y-%m-%d")],
            textposition="top center",
        )
    )

    # Warning quadrant shading: corr_usd_gold > usd_gold_hi AND corr_usd_oil < usd_oil_lo
    x_min = df["corr_usd_gold"].min()
    x_max = df["corr_usd_gold"].max()
    y_min = df["corr_usd_oil"].min()
    y_max = df["corr_usd_oil"].max()

    # Expand ranges slightly for nicer view
    x_range = [x_min - 0.05, x_max + 0.05]
    y_range = [y_min - 0.05, y_max + 0.05]

    fig.add_shape(
        type="rect",
        x0=usd_gold_hi,
        x1=x_range[1],
        y0=y_range[0],
        y1=usd_oil_lo,
        fillcolor="rgba(255,0,0,0.1)",
        line=dict(width=0),
        layer="below",
    )

    fig.add_vline(
        x=usd_gold_hi,
        line_dash="dash",
        annotation_text="USD–Gold threshold",
        annotation_position="top left",
    )
    fig.add_hline(
        y=usd_oil_lo,
        line_dash="dash",
        annotation_text="USD–Oil threshold",
        annotation_position="top right",
    )

    fig.update_layout(
        title="USD–Gold vs USD–Oil Correlation Regime Grid",
        xaxis_title="corr(USD, Gold)",
        yaxis_title="corr(USD, Oil)",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ----------------------------
# HTML Dashboard Builder
# ----------------------------

def build_tabbed_html(fig1: go.Figure, fig2: go.Figure, fig3: go.Figure,
                      banner_text: str, banner_color: str) -> str:
    """
    Build a simple tabbed HTML page with three Plotly figures.
    Uses include_plotlyjs='cdn' on the first figure; the others omit it.
    """
    from plotly.io import to_html

    fig1_html = to_html(fig1, include_plotlyjs="cdn", full_html=False, div_id="fig1")
    fig2_html = to_html(fig2, include_plotlyjs=False, full_html=False, div_id="fig2")
    fig3_html = to_html(fig3, include_plotlyjs=False, full_html=False, div_id="fig3")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>USD–Gold–Oil Correlation Sentinel</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        .banner {{
            padding: 16px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: white;
            background-color: {banner_color};
        }}
        .tabs {{
            overflow: hidden;
            background-color: #f1f1f1;
            border-bottom: 1px solid #ccc;
        }}
        .tab-button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 16px;
        }}
        .tab-button:hover {{
            background-color: #ddd;
        }}
        .tab-button.active {{
            background-color: #ccc;
        }}
        .tabcontent {{
            display: none;
            padding: 16px;
        }}
    </style>
</head>
<body>

<div class="banner">{banner_text}</div>

<div class="tabs">
    <button class="tab-button" onclick="openTab(event, 'tab1')" id="defaultOpen">USD–Gold Corr</button>
    <button class="tab-button" onclick="openTab(event, 'tab2')">USD–Oil Corr</button>
    <button class="tab-button" onclick="openTab(event, 'tab3')">Regime Grid</button>
</div>

<div id="tab1" class="tabcontent">
    {fig1_html}
</div>

<div id="tab2" class="tabcontent">
    {fig2_html}
</div>

<div id="tab3" class="tabcontent">
    {fig3_html}
</div>

<script>
function openTab(evt, tabName) {{
    var i, tabcontent, tabbuttons;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {{
        tabcontent[i].style.display = "none";
    }}
    tabbuttons = document.getElementsByClassName("tab-button");
    for (i = 0; i < tabbuttons.length; i++) {{
        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
    }}
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}}
document.getElementById("defaultOpen").click();
</script>

</body>
</html>
"""
    return html


# ----------------------------
# Main Logic
# ----------------------------

def main():
    args = parse_args()
    end_date = args.end or datetime.today().strftime("%Y-%m-%d")

    ensure_dir(args.output_dir)

    # 1) Download prices
    usd = download_single_ticker(args.usd_ticker, args.start, end_date)
    gold = download_single_ticker(args.gold_ticker, args.start, end_date)
    oil = download_single_ticker(args.oil_ticker, args.start, end_date)

    # 2) Combine and align by date
    prices = pd.concat([usd, gold, oil], axis=1).dropna()
    prices.columns = ["USD", "GOLD", "OIL"]

    print(f"[INFO] Combined price frame: {prices.shape[0]} rows.")

    # 3) Daily returns
    rets = compute_returns(prices)
    print(f"[INFO] Returns frame: {rets.shape[0]} rows.")

    # 4) Rolling correlations
    corr_usd_gold_raw = rolling_corr(rets["USD"], rets["GOLD"], window=args.window)
    corr_usd_oil_raw = rolling_corr(rets["USD"], rets["OIL"], window=args.window)

    corr_usd_gold = apply_smoothing(corr_usd_gold_raw, args.smooth)
    corr_usd_oil = apply_smoothing(corr_usd_oil_raw, args.smooth)

    # Align correlation series and drop NaNs
    corr_df = pd.DataFrame(
        {
            "corr_usd_gold": corr_usd_gold,
            "corr_usd_oil": corr_usd_oil,
        }
    ).dropna()

    if corr_df.empty:
        raise RuntimeError("No non-NaN values after rolling correlation; "
                           "consider reducing window size or checking data.")

    # 5) Warning logic
    warn_series = (corr_df["corr_usd_gold"] > args.usd_gold_hi) & (
        corr_df["corr_usd_oil"] < args.usd_oil_lo
    )
    warn_series.name = "warning"

    # 6) Combine everything into one output DataFrame
    # Reindex prices/returns to correlation index for cleaner CSV
    out_df = pd.concat(
        [
            prices.reindex(corr_df.index),
            rets.reindex(corr_df.index).add_suffix("_ret"),
            corr_df,
            warn_series,
        ],
        axis=1,
    )

    # 7) Latest summary
    latest_date = corr_df.index[-1]
    latest_gold_corr = float(corr_df["corr_usd_gold"].iloc[-1])
    latest_oil_corr = float(corr_df["corr_usd_oil"].iloc[-1])
    latest_warn = bool(warn_series.iloc[-1])

    banner_text = build_banner_text(latest_warn)
    banner_color = build_banner_color(latest_warn)

    print(
        f"[SUMMARY] {latest_date.strftime('%Y-%m-%d')} "
        f"warning={latest_warn} "
        f"corr_usd_gold={latest_gold_corr:.3f} "
        f"corr_usd_oil={latest_oil_corr:.3f}"
    )

    # 8) Write CSV and JSON
    csv_path = os.path.join(args.output_dir, "corr_sentinel_usd_gold_oil.csv")
    json_path = os.path.join(args.output_dir, "corr_sentinel_usd_gold_oil_latest.json")
    html_path = os.path.join(args.output_dir, "corr_sentinel_usd_gold_oil_dashboard.html")

    out_df.to_csv(csv_path)
    print(f"[INFO] Wrote CSV: {csv_path}")

    summary = {
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "latest_corr_usd_gold": latest_gold_corr,
        "latest_corr_usd_oil": latest_oil_corr,
        "warning": latest_warn,
        "window": args.window,
        "smooth": args.smooth,
        "usd_gold_hi": args.usd_gold_hi,
        "usd_oil_lo": args.usd_oil_lo,
        "usd_ticker": args.usd_ticker,
        "gold_ticker": args.gold_ticker,
        "oil_ticker": args.oil_ticker,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Wrote JSON: {json_path}")

    # 9) Build figures
    # Use corr_df index as the time axis
    dates = corr_df.index

    fig1 = build_corr_time_figure(
        dates=dates,
        corr_series=corr_df["corr_usd_gold"],
        threshold=args.usd_gold_hi,
        title="USD–Gold 60D Rolling Correlation",
        yaxis_title="corr(USD, Gold)",
        threshold_label=f"Threshold (>{args.usd_gold_hi})",
    )

    fig2 = build_corr_time_figure(
        dates=dates,
        corr_series=corr_df["corr_usd_oil"],
        threshold=args.usd_oil_lo,
        title="USD–Oil 60D Rolling Correlation",
        yaxis_title="corr(USD, Oil)",
        threshold_label=f"Threshold (<{args.usd_oil_lo})",
    )

    fig3 = build_regime_scatter_figure(
        corr_usd_gold=corr_df["corr_usd_gold"],
        corr_usd_oil=corr_df["corr_usd_oil"],
        usd_gold_hi=args.usd_gold_hi,
        usd_oil_lo=args.usd_oil_lo,
    )

    # 10) Optional PNG snapshots (requires kaleido)
    try:
        fig1.write_image(os.path.join(args.output_dir, "corr_usd_gold.png"))
        fig2.write_image(os.path.join(args.output_dir, "corr_usd_oil.png"))
        fig3.write_image(os.path.join(args.output_dir, "regime_grid.png"))
        print("[INFO] PNG snapshots saved (requires kaleido; if not installed, ignore any warnings).")
    except Exception as e:
        print(f"[WARN] Could not write PNGs (likely missing 'kaleido'): {e}")

    # 11) Build tabbed HTML and write to disk
    html = build_tabbed_html(fig1, fig2, fig3, banner_text, banner_color)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Wrote HTML dashboard: {html_path}")

    # 12) Auto-open in browser (unless disabled)
    if not args.no_browser:
        try:
            webbrowser.open("file://" + os.path.abspath(html_path))
            print("[INFO] Opened dashboard in default web browser.")
        except Exception as e:
            print(f"[WARN] Could not open browser: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"[ERROR] {ex}", file=sys.stderr)
        sys.exit(1)

