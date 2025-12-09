#!/usr/bin/env python3

# SCRIPTNAME: ok.macro.liquidity.dashboard.v3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

SCRIPTNAME = "macro.liquidity.dashboard.v3.py"
AUTHOR = "ChatGPT (for Michael Derby)"
DATE = "2025-11-23"
PURPOSE = """
Build a multi-panel macro / liquidity / vol / credit dashboard.
v3 Update:
- SAVES to local folder (./MACRO_DASHBOARD) to fix browser permission errors.
- PRINTS clickable links in terminal.
- Uses direct CSV fallback if pandas_datareader is missing.
"""

import argparse
import logging
import os
import sys
import webbrowser
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --- TRY IMPORTS FOR DATA ---
try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

# Optional: your own stock loader
try:
    import data_retrieval  # must not be modified
except ImportError:
    data_retrieval = None

# Plotly
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None


# --- PATH CONFIG ---
# FIX: Default to current directory instead of /dev/shm to avoid browser sandbox issues
def BASE_MACRO_PATH() -> str:
    return os.path.join(os.getcwd(), "MACRO_DASHBOARD")


# --- FRED SERIES DEFINITIONS ---
FRED_SERIES: Dict[str, str] = {
    # Yields (nominal)
    "DGS1MO": "1M Treasury Yield",
    "DGS3MO": "3M Treasury Yield",
    "DGS2": "2Y Treasury Yield",
    "DGS5": "5Y Treasury Yield",
    "DGS10": "10Y Treasury Yield",
    "DGS30": "30Y Treasury Yield",
    # Real yields / breakevens
    "DFII5": "5Y TIPS Yield",
    "DFII10": "10Y TIPS Yield",
    "T10YIE": "10Y Breakeven Inflation",
    # Credit spreads
    "BAMLH0A0HYM2": "HY OAS (ICE BofA US High Yield)",
    "BAMLCC0A1AAABBEY": "IG OAS (ICE BofA US Corp IG)",
    # Liquidity / balance sheet
    "RRPONTSYD": "ON RRP Balance",
    "WALCL": "Fed Total Assets (WALCL)",
    "WTREGEN": "Treasury General Account (TGA)",
    # Volatility
    "VIXCLS": "VIX Index (Close)",
    "MOVE": "MOVE Treasury Vol Index",
}


# --- LOGGING ---
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --- DIRECTORIES ---
def init_run_directories() -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(BASE_MACRO_PATH(), timestamp)
    paths = {
        "base": base,
        "data": os.path.join(base, "data"),
        "charts_html": os.path.join(base, "charts_html"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    logging.info(f"Run directory initialized at: {base}")
    return paths


# --- DATA LOADING: FRED (With Fallback) ---
def fetch_fred_series(series_ids: List[str],
                      start: Optional[str],
                      end: Optional[str]) -> pd.DataFrame:
    if start is None: start = "1990-01-01"
    if end is None: end = datetime.today().strftime("%Y-%m-%d")

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    all_data = []
    idx_name = None

    for sid in series_ids:
        try:
            logging.info(f"Fetching FRED series: {sid}")
            if pdr is not None:
                s = pdr.DataReader(sid, "fred", start=start, end=end)
                s.columns = [sid]
            else:
                # Fallback: Direct CSV download
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
                s = pd.read_csv(url, index_col=0, parse_dates=True)
                s.columns = [sid]
                s = s[(s.index >= start_dt) & (s.index <= end_dt)]

            s[sid] = pd.to_numeric(s[sid], errors="coerce")
            all_data.append(s)
            if idx_name is None: idx_name = s.index.name

        except Exception as e:
            logging.warning(f"Failed to fetch {sid}: {e}")

    if not all_data:
        logging.error("No FRED data could be fetched.")
        return pd.DataFrame()

    df = pd.concat(all_data, axis=1)
    df.index.name = idx_name or "Date"
    df = df.ffill().dropna(how="all")
    return df


# --- DATA LOADING: EQUITY ---
def fetch_ticker_ohlc(ticker: str, start: Optional[str], end: Optional[str]) -> Optional[pd.DataFrame]:
    if data_retrieval is None: return None
    try:
        df = data_retrieval.load_or_download_ticker(ticker, period="max")
        df = df.sort_index()
        if start: df = df[df.index >= pd.to_datetime(start)]
        if end: df = df[df.index <= pd.to_datetime(end)]
        return df
    except Exception:
        return None

def fetch_multiple_tickers(tickers: List[str], start: Optional[str], end: Optional[str]) -> Dict[str, pd.DataFrame]:
    result = {}
    for t in tickers:
        df = fetch_ticker_ohlc(t, start, end)
        if df is not None: result[t.upper()] = df
    return result


# --- PLOT HELPERS ---
def save_and_open_plotly(fig: "go.Figure",
                         name: str,
                         paths: Dict[str, str],
                         open_browser: bool = True) -> None:
    if fig is None: return

    safe_name = name.replace(" ", "_").replace("/", "_")
    html_path = os.path.join(paths["charts_html"], f"{safe_name}.html")

    try:
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        logging.info(f"Saved HTML: {html_path}")
        
        # Print clickable link for user convenience
        print(f"  >> Chart Ready: file://{os.path.abspath(html_path)}")
        
    except Exception as e:
        logging.error(f"Failed to write HTML for {name}: {e}")

    if open_browser:
        try:
            url = "file://" + os.path.abspath(html_path)
            webbrowser.open_new_tab(url)
        except Exception as e:
            logging.error(f"Failed to auto-open browser: {e}")


# --- FIGURE BUILDERS ---
def build_yield_curve_panel(df: pd.DataFrame) -> Optional["go.Figure"]:
    if go is None or df.empty: return None
    cols = [c for c in ["DGS1MO", "DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"] if c in df.columns]
    if not cols: return None
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=FRED_SERIES.get(c, c)))
    fig.update_layout(title="Treasury Yield Curve (Levels)", xaxis_title="Date", yaxis_title="Yield (%)", hovermode="x unified")
    return fig

def build_curve_spreads(df: pd.DataFrame) -> Optional["go.Figure"]:
    if go is None or df.empty: return None
    spreads = pd.DataFrame(index=df.index)
    if "DGS10" in df.columns and "DGS2" in df.columns: spreads["2s10s"] = df["DGS10"] - df["DGS2"]
    if "DGS10" in df.columns and "DGS3MO" in df.columns: spreads["3m10y"] = df["DGS10"] - df["DGS3MO"]
    if spreads.empty: return None
    fig = go.Figure()
    for c in spreads.columns:
        fig.add_trace(go.Scatter(x=spreads.index, y=spreads[c], mode="lines", name=c))
    fig.add_shape(type="line", x0=spreads.index.min(), x1=spreads.index.max(), y0=0, y1=0, line=dict(color="white", width=1, dash="dot"))
    fig.update_layout(title="Yield Curve Spreads", hovermode="x unified")
    return fig

def build_real_vs_nominal(df: pd.DataFrame) -> Optional["go.Figure"]:
    if go is None or df.empty: return None
    fig = go.Figure()
    if "DGS10" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["DGS10"], mode="lines", name="10Y Nominal"))
    if "DFII10" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["DFII10"], mode="lines", name="10Y Real"))
    if "T10YIE" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["T10YIE"], mode="lines", name="10Y Breakeven"))
    if not fig.data: return None
    fig.update_layout(title="Nominal vs Real vs Breakeven (10Y)", hovermode="x unified")
    return fig

def build_credit_spreads(df: pd.DataFrame) -> Optional["go.Figure"]:
    if go is None or df.empty: return None
    cols = [c for c in ["BAMLH0A0HYM2", "BAMLCC0A1AAABBEY"] if c in df.columns]
    if not cols: return None
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=FRED_SERIES.get(c, c)))
    fig.update_layout(title="Credit Spreads (HY & IG OAS)", hovermode="x unified")
    return fig

def build_liquidity_panel(df: pd.DataFrame) -> Optional["go.Figure"]:
    if go is None or df.empty: return None
    cols = [c for c in ["RRPONTSYD", "WALCL", "WTREGEN"] if c in df.columns]
    if not cols: return None
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if make_subplots else go.Figure()
    for c in cols:
        is_bs = (c == "WALCL") 
        if make_subplots:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=FRED_SERIES.get(c, c)), secondary_y=is_bs)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=FRED_SERIES.get(c, c)))
    fig.update_layout(title="Liquidity: Fed Assets, RRP, TGA", hovermode="x unified")
    return fig

def build_vol_panel(df: pd.DataFrame) -> Optional["go.Figure"]:
    if go is None or df.empty: return None
    cols = [c for c in ["VIXCLS", "MOVE"] if c in df.columns]
    if not cols: return None
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=FRED_SERIES.get(c, c)))
    fig.update_layout(title="Volatility Proxies (VIX, MOVE)", hovermode="x unified")
    return fig

def build_liquidity_composite(df: pd.DataFrame) -> Optional["go.Figure"]:
    if go is None or df.empty: return None
    comps = [c for c in ["RRPONTSYD", "WALCL", "WTREGEN"] if c in df.columns]
    if not comps: return None
    comp_df = df[comps].dropna()
    if comp_df.empty: return None
    z = (comp_df - comp_df.mean()) / comp_df.std(ddof=0)
    composite = pd.Series(0, index=comp_df.index)
    if "WALCL" in comp_df: composite += z["WALCL"]
    if "WTREGEN" in comp_df: composite -= z["WTREGEN"]
    if "RRPONTSYD" in comp_df: composite -= z["RRPONTSYD"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=composite.index, y=composite, mode="lines", name="Net Liquidity Z-Score"))
    fig.add_shape(type="line", x0=composite.index.min(), x1=composite.index.max(), y0=0, y1=0, line=dict(color="white", dash="dot"))
    fig.update_layout(title="Net Liquidity Composite Z-Score (Assets - TGA - RRP)", hovermode="x unified")
    return fig

def build_ticker_price_panel(ticker_data: Dict[str, pd.DataFrame]) -> Optional["go.Figure"]:
    if go is None or not ticker_data: return None
    fig = go.Figure()
    for ticker, df in ticker_data.items():
        if "Close" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=ticker))
    if not fig.data: return None
    fig.update_layout(title="Price Overlay", hovermode="x unified")
    return fig


# --- MAIN ---
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Macro Dashboard")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start Date")
    parser.add_argument("--end", type=str, default=None, help="End Date")
    parser.add_argument("--tickers", nargs="*", default=[], help="Tickers")
    parser.add_argument("--no-open-browser", action="store_true", help="Skip browser open")
    parser.add_argument("--verbose", action="store_true", help="Debug logs")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    if go is None:
        logging.error("Plotly is not installed. Install via 'pip install plotly'")
        sys.exit(1)

    # Initialize directories
    paths = init_run_directories()
    
    if pdr is None:
        logging.warning("pandas_datareader not installed. Using direct CSV fallback.")

    # 1) Fetch FRED Data
    fred_ids = list(FRED_SERIES.keys())
    fred_df = fetch_fred_series(fred_ids, args.start, args.end)

    # 2) Fetch Tickers
    tickers = [t.upper() for t in args.tickers] if args.tickers else []
    ticker_data = fetch_multiple_tickers(tickers, args.start, args.end)

    # 3) Dump Data
    try:
        fred_df.to_csv(os.path.join(paths["data"], "fred_macro.csv"))
    except Exception: pass

    # 4) Build Charts
    figs = [
        (build_yield_curve_panel(fred_df), "yield_curve"),
        (build_curve_spreads(fred_df), "yield_spreads"),
        (build_real_vs_nominal(fred_df), "real_vs_nominal"),
        (build_credit_spreads(fred_df), "credit_spreads"),
        (build_liquidity_panel(fred_df), "liquidity_panel"),
        (build_vol_panel(fred_df), "volatility"),
        (build_liquidity_composite(fred_df), "liquidity_composite"),
        (build_ticker_price_panel(ticker_data), "prices")
    ]

    open_browser = not args.no_open_browser
    count = 0
    print("\n" + "="*60)
    for fig, name in figs:
        if fig:
            save_and_open_plotly(fig, name, paths, open_browser)
            count += 1
    print("="*60 + "\n")
    
    if count == 0:
        logging.error("No charts generated. Data fetch likely failed.")
    else:
        logging.info(f"Generated {count} charts.")

if __name__ == "__main__":
    main()
