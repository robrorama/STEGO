#!/usr/bin/env python3
"""
Fractal convergence / Hurst "collapse" visualizer (v7).
Uses the canonical stock data loader: data_retrieval.load_or_download_ticker()

Enhancements v7:
- INPUT FLEXIBILITY: Supports command line args (python script.py AAPL) 
  OR Interactive Input (python script.py -> "Enter ticker: ")
- v6 Features: Dual-pane (Price/Hurst), Full Y-Axis (0-1.05), Zoom Dropdown.
- Convergence Logic: Pure spread calculation (tightest knots).
"""

import os
import sys
import argparse
from typing import Sequence, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser

# --- Optional Plotly support (interactive HTML) ---
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    go = None
    pio = None
    _HAS_PLOTLY = False
    print("Warning: Plotly not installed. HTML charts will be skipped.")

# --- Import canonical data layer ---
try:
    import data_retrieval as dr
except ImportError:
    print("ERROR: Could not import data_retrieval.py. Ensure it is in the same directory.", file=sys.stderr)
    sys.exit(1)

# ----------------------------------------------------------------------
# Config & CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fractal Convergence/Collapse Visualizer")
    
    # Allow positional arguments (e.g., "python script.py SPY NVDA")
    parser.add_argument("tickers_pos", type=str, nargs="*", 
                        help="Ticker(s) to analyze (space-separated)")
    
    # Keep legacy flag support just in case
    parser.add_argument("--ticker", type=str, nargs="+", default=[], 
                        help="Ticker(s) via flag")
    
    parser.add_argument("--period", type=str, default="10y", 
                        help="Data period to load (default: 10y)")
    parser.add_argument("--outdir", type=str, default=None, 
                        help="Output directory (default: /dev/shm/fractal_collapse)")
    
    return parser.parse_args()

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_ticker(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = dr.load_or_download_ticker(ticker, period=period)
    except TypeError:
        print(f"  (Note: 'period' argument not supported by local data_retrieval; fetching full history for {ticker})")
        df = dr.load_or_download_ticker(ticker)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    if "Date" not in df.columns:
        df = df.reset_index()
        if "Date" not in df.columns:
            first_col = df.columns[0]
            if first_col != "Date":
                df = df.rename(columns={first_col: "Date"})

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            raise KeyError(f"'Close' column not found for {ticker}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ----------------------------------------------------------------------
# Hurst R/S estimation
# ----------------------------------------------------------------------
def hurst_rs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20: return np.nan
    y = x - x.mean()
    z = np.cumsum(y)
    R = z.max() - z.min()
    S = y.std(ddof=1)
    if S == 0: return np.nan
    return np.log((R / S) + 1e-12) / np.log(n)

def rolling_hurst(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).apply(lambda v: hurst_rs(v.values), raw=False)

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_hurst_and_spread(
    df: pd.DataFrame,
    windows: Sequence[int] = (63, 126, 252),
) -> Tuple[pd.DataFrame, pd.Series]:
    close = df["Close"].astype(float)
    Hs = [rolling_hurst(close, w) for w in windows]
    H_stack = pd.concat(Hs, axis=1)
    H_stack.columns = [f"H_{w}" for w in windows]

    score = H_stack.max(axis=1) - H_stack.min(axis=1)
    score = score.replace([np.inf, -np.inf], np.nan)
    return H_stack, score

def build_hfd_metrics(
    df: pd.DataFrame,
    windows: Sequence[int],
    H_stack: pd.DataFrame,
    score: pd.Series,
) -> pd.DataFrame:
    dates = df["Date"]
    H_df = H_stack.copy()
    H_df.index = dates
    score_series = score.copy()
    score_series.index = dates

    valid_count = H_df.notna().sum(axis=1)
    H_spread = H_df.max(axis=1) - H_df.min(axis=1)
    H_spread[valid_count < 2] = np.nan
    
    # Pure spread logic (v6)
    H_convergence_score = H_spread 

    FD_df = 2.0 - H_df
    FD_df.columns = [c.replace("H_", "FD_") for c in H_df.columns]

    metrics_df = pd.DataFrame(index=dates)
    for col in H_df.columns: metrics_df[col] = H_df[col]
    for col in FD_df.columns: metrics_df[col] = FD_df[col]

    metrics_df["H_spread"] = H_spread
    metrics_df["H_convergence_score"] = H_convergence_score
    metrics_df["H_valid_count"] = valid_count
    
    return metrics_df

def detect_convergence_events(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    windows: Sequence[int],
    quantile: float = 0.05, 
    min_separation: int = 20,
) -> List[int]:
    if "H_convergence_score" not in metrics_df.columns: return []

    dates = df["Date"]
    pos_by_date = pd.Series(np.arange(len(df)), index=dates)
    valid_mask = metrics_df["H_valid_count"] >= len(windows)
    conv_series = metrics_df.loc[valid_mask, "H_convergence_score"].dropna()

    if conv_series.empty: return []

    # Filter out nonsense perfect 0s
    conv_series = conv_series[conv_series > 0.0001]
    
    threshold = conv_series.quantile(quantile)
    candidate_dates = conv_series.index[conv_series <= threshold]
    if len(candidate_dates) == 0: return []

    candidate_positions = sorted(set(pos_by_date[candidate_dates].dropna().astype(int).tolist()))
    events: List[int] = []
    i = 0
    while i < len(candidate_positions):
        cluster = [candidate_positions[i]]
        j = i + 1
        while j < len(candidate_positions) and (candidate_positions[j] - cluster[-1] <= min_separation):
            cluster.append(candidate_positions[j])
            j += 1
        
        best_pos = cluster[0]
        best_score = None
        for p in cluster:
            d = df.loc[p, "Date"]
            s_val = metrics_df.loc[d, "H_convergence_score"]
            if best_score is None or (s_val < best_score):
                best_score = s_val
                best_pos = p
        events.append(int(best_pos))
        i = j
    return events

def find_global_collapse(df, metrics_df, windows):
    if "H_convergence_score" not in metrics_df.columns: return None, None
    dates = df["Date"]
    pos_by_date = pd.Series(np.arange(len(df)), index=dates)
    valid_mask = metrics_df["H_valid_count"] >= len(windows)
    conv_series = metrics_df.loc[valid_mask, "H_convergence_score"].dropna()
    if conv_series.empty: return None, None
    best_date = conv_series.idxmin()
    if best_date not in pos_by_date: return None, None
    return best_date, int(pos_by_date[best_date])

# ----------------------------------------------------------------------
# TA columns
# ----------------------------------------------------------------------
def add_ta_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    rolling = df["Close"].rolling(20)
    df["BB_mid"] = df["SMA20"]
    df["BB_up"] = df["SMA20"] + 2 * rolling.std()
    df["BB_dn"] = df["SMA20"] - 2 * rolling.std()
    return df

# ----------------------------------------------------------------------
# Plotly Chart (Subplots)
# ----------------------------------------------------------------------
def make_plotly_chart(
    df: pd.DataFrame,
    df_ta: pd.DataFrame,
    metrics_df: pd.DataFrame,
    ticker: str,
    outdir: str,
    windows: Sequence[int],
    convergence_indices: List[int],
) -> None:
    if not _HAS_PLOTLY: return
    if metrics_df.empty: return

    dates = df["Date"]
    metrics_aligned = metrics_df.reindex(dates)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Price & TA", "Hurst Exponent (0.0 to 1.05)")
    )

    # --- ROW 1: Price ---
    hover_texts = []
    for i in range(len(df)):
        dt_str = dates.iloc[i].strftime('%Y-%m-%d')
        c = df["Close"].iloc[i]
        hover_texts.append(f"<b>{dt_str}</b><br>Close: {c:.2f}")

    fig.add_trace(go.Candlestick(
        x=dates, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC", text=hover_texts, hoverinfo="text"
    ), row=1, col=1)

    colors_sma = {"SMA20": "blue", "SMA50": "orange", "SMA200": "red"}
    for sma, color in colors_sma.items():
        if sma in df_ta.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=df_ta[sma], name=sma, mode='lines',
                line=dict(color=color, width=1), opacity=0.8
            ), row=1, col=1)

    if "BB_up" in df_ta.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df_ta["BB_up"], name="BB Upper", mode='lines',
            line=dict(color="gray", width=0.5, dash="dot"), hoverinfo="skip", showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df_ta["BB_dn"], name="BB Lower", mode='lines',
            line=dict(color="gray", width=0.5, dash="dot"), hoverinfo="skip", showlegend=False,
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
        ), row=1, col=1)

    # --- ROW 2: Hurst Lines ---
    h_colors = {63: "#00cc00", 126: "#ff9900", 252: "#3366ff"} 
    
    for w in windows:
        col_name = f"H_{w}"
        if col_name in metrics_aligned.columns:
            valid_h = metrics_aligned[col_name]
            fig.add_trace(go.Scatter(
                x=dates, y=valid_h, name=f"H ({w})", mode='lines',
                line=dict(color=h_colors.get(w, "black"), width=1.5),
                hovertemplate=f"H({w}): %{{y:.3f}}<extra></extra>" 
            ), row=2, col=1)

    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1, annotation_text="Random Walk (0.5)")

    # --- Highlights & Zoom ---
    for idx in convergence_indices:
        if 0 <= idx < len(df):
            event_date = df.loc[idx, "Date"]
            x0 = event_date - pd.Timedelta(days=2)
            x1 = event_date + pd.Timedelta(days=2)
            
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor="rgba(255, 0, 0, 0.15)",
                layer="below", line_width=0, row=1, col=1
            )
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor="rgba(255, 0, 0, 0.15)",
                layer="below", line_width=0, row=2, col=1
            )

    buttons = []
    buttons.append(dict(
        label="All History", method="relayout",
        args=[{"xaxis.range": [dates.min(), dates.max()]}]
    ))

    for idx in convergence_indices:
        if 0 <= idx < len(df):
            evt_date = df.loc[idx, "Date"]
            date_str = evt_date.strftime("%Y-%m-%d")
            zoom_start = evt_date - pd.Timedelta(days=120)
            zoom_end = evt_date + pd.Timedelta(days=120)
            
            buttons.append(dict(
                label=f"Event: {date_str}", method="relayout",
                args=[{"xaxis.range": [zoom_start, zoom_end]}]
            ))

    fig.update_layout(
        title=f"{ticker} Fractal Analysis (Top: Price, Bottom: Hurst Clustering)",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        height=800, 
        updatemenus=[
            dict(
                type="dropdown", direction="down", active=0, x=0, y=1.02,
                showactive=True, buttons=buttons
            )
        ]
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Hurst (H)", range=[0.0, 1.05], row=2, col=1)

    html_path = os.path.join(outdir, f"{ticker}_fractal_convergence_plotly.html")
    pio.write_html(fig, file=html_path, auto_open=False, include_plotlyjs="cdn")
    print(f"Saved Interactive Chart: {html_path}")

    try:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception:
        pass

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    outdir = args.outdir if args.outdir else os.environ.get("FRACTAL_COLLAPSE_OUTDIR", "/dev/shm/fractal_collapse")
    os.makedirs(outdir, exist_ok=True)
    windows = (63, 126, 252)

    # Consolidate tickers from positional args and flags
    target_tickers = args.tickers_pos + args.ticker
    
    # Remove duplicates and empty strings
    target_tickers = list(set([t for t in target_tickers if t]))

    # Interactive Fallback: If no tickers provided, ask user.
    if not target_tickers:
        print("No tickers provided.")
        user_input = input("Enter ticker symbol(s) (space-separated, e.g., SPY NVDA): ").strip()
        if user_input:
            target_tickers = user_input.split()
        else:
            print("No input provided. Using defaults: SPY, NVDA")
            target_tickers = ["SPY", "NVDA"]

    for ticker in target_tickers:
        ticker = ticker.upper()
        print(f"\n=== Processing {ticker} ===")
        
        try:
            df = load_ticker(ticker, period=args.period)
        except Exception as e:
            print(f"Failed to load {ticker}: {e}")
            continue

        df_ta = add_ta_columns(df)
        H_stack, score = compute_hurst_and_spread(df, windows=windows)
        metrics_df = build_hfd_metrics(df, windows, H_stack, score)
        
        collapse_date, collapse_idx = find_global_collapse(df, metrics_df, windows)
        convergence_indices = detect_convergence_events(df, metrics_df, windows)
        print(f"  > Events found: {len(convergence_indices)} tightest knots.")

        if not H_stack.empty: 
            H_stack.to_csv(os.path.join(outdir, f"{ticker}_hurst_windows.csv"), index_label="Date")
        if not metrics_df.empty: 
            metrics_df.to_csv(os.path.join(outdir, f"{ticker}_hurst_fd_metrics.csv"), index_label="Date")

        make_plotly_chart(df, df_ta, metrics_df, ticker, outdir, windows, convergence_indices)

if __name__ == "__main__":
    main()
