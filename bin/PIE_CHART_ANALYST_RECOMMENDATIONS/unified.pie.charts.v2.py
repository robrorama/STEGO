#!/usr/bin/env python3
# SCRIPTNAME: unified.pie.charts.v2.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Fetches and visualizes Analyst Recommendations (Buy/Hold/Sell) for a ticker.
#   - Uses data_retrieval.py for canonical output directory management.
#   - Persists summary and time-series data to CSV before plotting.
#   - Generates an interactive Plotly Pie Chart.

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import webbrowser
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr  # uses BASE_DATA_PATH, BASE_CACHE_PATH, IMAGES_SUBDIR, PNGS_SUBDIR
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ------------------------------- utils --------------------------------
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# -------------------- recommendations retrieval -----------------------
def _load_or_fetch_recommendations(ticker: str, out_dir: str):
    """
    Returns (rec_summary_df, recommendations_df). Either may be None.
    Persists CSV(s) under the daily out_dir created by data_retrieval.py.
    """
    _ensure_dir(out_dir)
    sum_csv = os.path.join(out_dir, f"{ticker}_recommendations_summary.csv")
    rec_csv = os.path.join(out_dir, f"{ticker}_recommendations.csv")

    rec_summary = None
    if os.path.exists(sum_csv):
        try:
            rec_summary = pd.read_csv(sum_csv)
            print(f"[info] loaded recommendations_summary from {sum_csv}")
        except Exception as e:
            print(f"[warn] could not read {sum_csv}: {e}")

    recs = None
    if os.path.exists(rec_csv):
        try:
            recs = pd.read_csv(rec_csv)
            print(f"[info] loaded recommendations (time series) from {rec_csv}")
        except Exception as e:
            print(f"[warn] could not read {rec_csv}: {e}")

    # If either is missing, fetch from yfinance
    if rec_summary is None or recs is None:
        try:
            tkr = yf.Ticker(ticker)
        except Exception as e:
            print(f"[error] yfinance Ticker({ticker}) failed: {e}")
            return (rec_summary, recs)

        if rec_summary is None:
            try:
                rs = getattr(tkr, "recommendations_summary", None)
                # yfinance may return dict-like; normalize to DataFrame
                if rs is not None and not isinstance(rs, pd.DataFrame):
                    rs = pd.DataFrame([rs])
                if rs is not None and not rs.empty:
                    rec_summary = rs
                    rec_summary.to_csv(sum_csv, index=False)
                    print(f"[info] saved recommendations_summary -> {sum_csv}")
            except Exception as e:
                print(f"[warn] fetching recommendations_summary failed: {e}")

        if recs is None:
            try:
                r = getattr(tkr, "recommendations", None)
                # Some yfinance versions return None here
                if isinstance(r, pd.DataFrame) and not r.empty:
                    recs = r
                    recs.to_csv(rec_csv)
                    print(f"[info] saved recommendations (time series) -> {rec_csv}")
            except Exception as e:
                print(f"[warn] fetching recommendations failed: {e}")

    return (rec_summary, recs)

# --------------------- counting / bucketing logic ---------------------
def _counts_from_summary(rec_summary: pd.DataFrame):
    """Prefer exact columns if provided by yfinance: strongBuy,buy,hold,sell,strongSell"""
    base = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    if rec_summary is None or rec_summary.empty:
        return base
    cols = {c.lower(): c for c in rec_summary.columns}
    def read(col_key: str) -> int:
        if col_key in cols:
            try:
                v = rec_summary[cols[col_key]].iloc[0]
                return int(v) if pd.notna(v) else 0
            except Exception:
                return 0
        return 0
    return {
        "strongBuy":  read("strongbuy"),
        "buy":        read("buy"),
        "hold":       read("hold"),
        "sell":       read("sell"),
        "strongSell": read("strongsell"),
    }

def _counts_from_time_series(recs: pd.DataFrame):
    """
    Map time-series 'To Grade' (or variants) into counts.
    Buckets include synonyms commonly seen in Street vernacular.
    """
    base = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    if recs is None or recs.empty:
        return base

    # Locate target column
    lower_cols = [c.lower() for c in recs.columns]
    if "to grade" in lower_cols:
        col = recs.columns[lower_cols.index("to grade")]
    elif "tograde" in lower_cols:
        col = recs.columns[lower_cols.index("tograde")]
    elif "to_grade" in lower_cols:
        col = recs.columns[lower_cols.index("to_grade")]
    elif "grade" in lower_cols:
        col = recs.columns[lower_cols.index("grade")]
    else:
        # Nothing usable
        return base

    def bucket(v: str):
        if not isinstance(v, str):
            return None
        s = v.strip().lower()
        # Strong Buy / Conviction Buy
        if "strong" in s and "buy" in s:
            return "strongBuy"
        # Buy cluster
        if any(k in s for k in ["buy", "outperform", "overweight", "accumulate", "add", "positive"]):
            return "buy"
        # Strong Sell
        if "strong" in s and "sell" in s:
            return "strongSell"
        # Sell cluster
        if any(k in s for k in ["sell", "underperform", "underweight", "reduce", "negative"]):
            return "sell"
        # Hold cluster
        if any(k in s for k in ["hold", "neutral", "market perform", "sector perform",
                                "peer perform", "equal", "in-line"]):
            return "hold"
        return None

    counts = dict(base)
    for v in recs[col].astype(str):
        b = bucket(v)
        if b:
            counts[b] += 1
    return counts

# --------------------------- figure build -----------------------------
def _build_pie_figure(ticker: str, counts: dict) -> go.Figure:
    labels = ["Buy", "Hold", "Sell"]
    sizes = [
        counts.get("buy", 0) + counts.get("strongBuy", 0),
        counts.get("hold", 0),
        counts.get("sell", 0) + counts.get("strongSell", 0),
    ]
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=sizes,
            hole=0.3,
            textinfo="percent+label",
            pull=[0.1, 0, 0],
        )]
    )
    fig.update_layout(
        title=f"Analyst Recommendations for {ticker}",
        showlegend=True
    )
    return fig

# ------------------------ persist + open tabs -------------------------
def _save_and_open_fig(fig: go.Figure, ticker: str, out_dir: str, open_browser: bool = True) -> str:
    """
    Saves HTML to the daily output dir (from data_retrieval.py),
    saves PNG(s) to IMAGES_SUBDIR()/TICKER and PNGS_SUBDIR(), also copies into out_dir,
    opens the HTML in a new browser tab (one tab per figure).
    """
    _ensure_dir(out_dir)

    # HTML (open per-figure in its own tab)
    html_path = os.path.join(out_dir, f"{ticker}_analyst_recommendations.html")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)

    # PNGs (use env-driven subdirs from data_retrieval.py)
    ts = _today_str()
    images_root = _ensure_dir(dr.IMAGES_SUBDIR())
    image_dir = _ensure_dir(os.path.join(images_root, ticker))
    image_file = os.path.join(image_dir, f"{ticker}_analyst_recommendations.png")

    pngs_root = _ensure_dir(dr.PNGS_SUBDIR())
    pngs_file = os.path.join(pngs_root, f"{ticker}_{ts}_analyst_recommendations.png")

    out_png = os.path.join(out_dir, f"{ticker}_analyst_recommendations.png")

    try:
        # Requires 'kaleido' to be installed.
        fig.write_image(image_file)
        try:
            import shutil
            for dest in (pngs_file, out_png):
                try:
                    shutil.copy2(image_file, dest)
                except Exception as ce:
                    print(f"[warn] copy -> {dest} failed: {ce}")
        except Exception as e:
            print(f"[warn] PNG duplication failed: {e}")
    except Exception as e:
        print(f"[warn] PNG export skipped (install 'kaleido'): {e}")

    if open_browser:
        webbrowser.open_new_tab("file://" + os.path.abspath(html_path))
    return html_path

# -------------------------------- main --------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Unified analysts visualization (uses data_retrieval.py for outputs)"
    )
    p.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    p.add_argument("--no-browser", action="store_true", help="Do not open browser tabs")
    args = p.parse_args()

    ticker = args.ticker.upper()

    # Ensure we use the provided retrieval module for output dir
    out_dir = dr.create_output_directory(ticker)  # BASE_DATA_PATH/YYYY-MM-DD/TICKER

    # Load or fetch recommendations; write CSV(s) under out_dir
    rec_summary, recs = _load_or_fetch_recommendations(ticker, out_dir)

    # Prefer summary counts; fall back to mapped time-series counts
    counts = _counts_from_summary(rec_summary)
    if sum(counts.values()) == 0:
        counts = _counts_from_time_series(recs)

    # Build figure list (each will open in its own tab if not --no-browser)
    figs = []
    figs.append(_build_pie_figure(ticker, counts))

    # Persist & open each figure in its own tab
    for fig in figs:
        path = _save_and_open_fig(fig, ticker, out_dir, open_browser=not args.no_browser)
        print(f"[ok] wrote {path}")

if __name__ == "__main__":
    main()
