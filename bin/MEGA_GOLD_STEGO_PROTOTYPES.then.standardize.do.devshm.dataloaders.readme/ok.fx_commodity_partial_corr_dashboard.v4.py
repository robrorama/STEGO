#!/usr/bin/env python3
"""
fx_commodity_partial_corr_dashboard.v3.py

Author:    Michael Derby
Framework: STEGO Financial Framework

Description
-----------
FX-Commodity Partial Correlation Dashboard with ROBUST DATA ENGINEERING.
Fixes timezone mismatches, yfinance column inconsistencies, and index alignment.

Three Logic Pattern Enforcements:
1. Force Timezone-Naive Indices.
2. Robust Column Extraction (Close vs Adj Close vs iloc[0]).
3. Pre-Calculation Intersection (Strict Alignment).

Usage
-----
    python3 fx_commodity_partial_corr_dashboard.v3.py \
        --commodity "CL=F" \
        --fx "CAD=X" \
        --usd-index "UUP" \
        --period "2y"
"""

import os
import argparse
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 1. Import User Libraries
# ---------------------------------------------------------------------------
try:
    import data_retrieval as dr
except ImportError:
    print("CRITICAL: Failed to import data_retrieval.py. Ensure it is in the path.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Configuration & Styling
# ---------------------------------------------------------------------------

STYLING = {
    "cyan": "#00E5FF",
    "pink": "#FF4081",
    "lime": "#CCFF00",
    "bg": "#111111",
    "paper": "#1E1E1E",
    "text": "#EEEEEE",
    "grid": "#333333"
}

def get_output_dir(commodity: str, fx: str) -> str:
    root = "/dev/shm/STEGO_FX_COMMODITY"
    today_str = datetime.now().strftime("%Y-%m-%d")
    c_safe = commodity.replace("=", "").replace("^", "")
    f_safe = fx.replace("=", "").replace("^", "")
    out_dir = os.path.join(root, f"{c_safe}_{f_safe}", today_str)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ---------------------------------------------------------------------
# 3. Data Engineering Logic (The 3 Fixes)
# ---------------------------------------------------------------------

def robust_extract_and_normalize(df_input, name: str) -> pd.Series:
    """
    Applies Logic Pattern #1 (Timezone Fix) and #2 (Robust Extraction).
    """
    if df_input is None or df_input.empty:
        print(f"[WARN] Empty data for {name}")
        return pd.Series(dtype=float)

    # --- Logic Pattern #2: Robust Column Extraction ---
    # Handle yfinance variance (MultiIndex, Series, or diff col names)
    if isinstance(df_input, pd.DataFrame):
        # Flatten MultiIndex if present
        if isinstance(df_input.columns, pd.MultiIndex):
            df_input.columns = df_input.columns.droplevel(1)
        
        if 'Close' in df_input.columns:
            s = df_input['Close']
        elif 'Adj Close' in df_input.columns:
            s = df_input['Adj Close']
        else:
            # Fallback: Grab first column
            s = df_input.iloc[:, 0] 
    else:
        # It's already a series
        s = df_input

    # Ensure it's a Series
    s = s.copy()
    s.name = name

    # --- Logic Pattern #1: Force Timezone-Naive Indices ---
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    
    # Normalize to midnight to ensure clean intersection
    s.index = s.index.normalize()
    
    # Handle duplicates (multiple ticks per day) by taking the last
    s = s.groupby(s.index).last()
    
    return s

def align_market_data(raw_c, raw_f, raw_u) -> pd.DataFrame:
    """
    Applies Logic Pattern #3: Pre-Calculation Intersection.
    """
    # 1. Normalize individual series first
    s_c = robust_extract_and_normalize(raw_c, "C_close")
    s_f = robust_extract_and_normalize(raw_f, "F_close")
    s_u = robust_extract_and_normalize(raw_u, "U_close")

    # 2. Find Intersection of ALL loaded data first
    common = s_c.index
    if not s_f.empty: common = common.intersection(s_f.index)
    if not s_u.empty: common = common.intersection(s_u.index)

    if len(common) < 10:
        print(f"[CRITICAL] Only {len(common)} common dates found. Alignment failed.")
        return pd.DataFrame()

    # 3. Hard-filter everything to this common index
    s_c = s_c.loc[common]
    s_f = s_f.loc[common]
    s_u = s_u.loc[common]

    # Combine
    df = pd.concat([s_c, s_f, s_u], axis=1)
    df.sort_index(inplace=True)
    
    print(f"[INFO] aligned data shape: {df.shape}")
    return df

# ---------------------------------------------------------------------
# 4. Math Logic (OLS & Rolling)
# ---------------------------------------------------------------------

def _ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 3: return np.full_like(y, np.nan)
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    y_fit = y[mask]
    try:
        beta_hat, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
        fitted = beta_hat[0] + beta_hat[1] * x
        return y - fitted
    except: 
        return np.full_like(y, np.nan)

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3: return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])

def rolling_partial_corr(c_ret: pd.Series, f_ret: pd.Series, u_ret: pd.Series, window: int) -> pd.Series:
    """
    Calculates partial correlation of Commodity vs FX, controlling for USD.
    """
    idx = c_ret.index
    out = np.full(len(idx), np.nan, dtype=float)
    c, f, u = c_ret.values, f_ret.values, u_ret.values
    
    for i in range(window - 1, len(idx)):
        sl = slice(i - window + 1, i + 1)
        # Calculate residuals for THIS window
        eps_c = _ols_residuals(c[sl], u[sl])
        eps_f = _ols_residuals(f[sl], u[sl])
        out[i] = _safe_corr(eps_c, eps_f)
    return pd.Series(out, index=idx)

def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window=window, min_periods=window//2).corr(b)

# ---------------------------------------------------------------------
# 5. Pipeline
# ---------------------------------------------------------------------

def run_analysis(commodity, fx, usd, period, windows, out_dir):
    # 1. Fetch Data (Using user's data_retrieval lib)
    print(f"[INFO] Loading Data: {commodity}, {fx}, {usd} over {period}")
    raw_c = dr.load_or_download_ticker(commodity, period)
    raw_f = dr.load_or_download_ticker(fx, period)
    raw_u = dr.load_or_download_ticker(usd, period)

    # 2. Robust Alignment
    df_prices = align_market_data(raw_c, raw_f, raw_u)
    
    # Save aligned prices
    df_prices.to_csv(os.path.join(out_dir, "aligned_prices.csv"))

    # 3. Calculate Log Returns
    # Using numpy log to avoid division by zero issues
    log_px = np.log(df_prices)
    df_ret = log_px.diff().dropna()
    df_ret.columns = ["C_ret", "F_ret", "U_ret"]
    
    # Merge prices and returns
    df = pd.concat([df_prices, df_ret], axis=1).dropna()

    # 4. Compute Metrics
    print("[INFO] Computing Rolling Metrics...")
    roll_data = {}
    for w in windows:
        # Standard Correlation
        roll_data[f"corr_CF_w{w}"] = rolling_corr(df["C_ret"], df["F_ret"], w)
        roll_data[f"corr_CU_w{w}"] = rolling_corr(df["C_ret"], df["U_ret"], w)
        # Partial Correlation (The 'pure' relationship)
        roll_data[f"partial_w{w}"] = rolling_partial_corr(df["C_ret"], df["F_ret"], df["U_ret"], w)
    
    rolling_df = pd.DataFrame(roll_data, index=df.index)
    
    # 5. Calculate Full-Sample Residuals (Idiosyncratic components)
    df["eps_C"] = _ols_residuals(df["C_ret"].values, df["U_ret"].values)
    df["eps_F"] = _ols_residuals(df["F_ret"].values, df["U_ret"].values)

    return df, rolling_df

# ---------------------------------------------------------------------
# 6. Visualization (Plotly)
# ---------------------------------------------------------------------

def generate_dashboard(df, rolling_df, args, out_dir):
    # Create Subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Normalized Price History (Base 100)", 
            f"Correlation Dynamics (Window={args.base_window}d)", 
            "Idiosyncratic Residual Scatter (Net of USD)"
        ),
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]]
    )

    # --- Plot 1: Prices ---
    norm = df[["C_close", "F_close", "U_close"]] / df[["C_close", "F_close", "U_close"]].iloc[0] * 100
    
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm["C_close"], name=f"{args.commodity} (Comm)",
        line=dict(color=STYLING["cyan"], width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm["F_close"], name=f"{args.fx} (FX)",
        line=dict(color=STYLING["pink"], width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm["U_close"], name=f"{args.usd_index} (USD)",
        line=dict(color=STYLING["lime"], width=1, dash='dot')
    ), row=1, col=1)

    # --- Plot 2: Correlations ---
    w = args.base_window
    if f"corr_CF_w{w}" in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df.index, y=rolling_df[f"partial_w{w}"], 
            name="Partial Corr (Net of USD)",
            line=dict(color="#FFFFFF", width=3),
            fill='tozeroy', fillcolor='rgba(255,255,255,0.1)'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=rolling_df.index, y=rolling_df[f"corr_CF_w{w}"], 
            name="Raw Corr(Comm, FX)",
            line=dict(color=STYLING["cyan"], width=1.5, dash='dash')
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=rolling_df.index, y=rolling_df[f"corr_CU_w{w}"], 
            name="Corr(Comm, USD)",
            line=dict(color=STYLING["lime"], width=1, dash='dot')
        ), row=2, col=1)

    # --- Plot 3: Scatter of Residuals ---
    # We color the markers by time (index) to show regime evolution
    fig.add_trace(go.Scatter(
        x=df["eps_C"], y=df["eps_F"],
        mode="markers",
        name="Residuals",
        marker=dict(
            color=df.index.astype(int), 
            colorscale="Viridis", 
            opacity=0.6, 
            size=6,
            line=dict(width=1, color=STYLING["bg"])
        ),
        text=df.index.strftime("%Y-%m-%d")
    ), row=3, col=1)

    # --- Layout Update ---
    fig.update_layout(
        title=f"<b>STEGO Framework</b> | {args.commodity} vs {args.fx} | Regressed on {args.usd_index}",
        template="plotly_dark",
        height=1000,
        plot_bgcolor=STYLING["bg"],
        paper_bgcolor=STYLING["paper"],
        font=dict(family="Monospace", color=STYLING["text"]),
        hovermode="x unified"
    )

    # Axis tweaks
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1, gridcolor=STYLING["grid"])
    fig.update_yaxes(title_text="Correlation", row=2, col=1, range=[-1.1, 1.1], gridcolor=STYLING["grid"])
    fig.update_yaxes(title_text=f"{args.fx} Residual", row=3, col=1, gridcolor=STYLING["grid"])
    fig.update_xaxes(title_text=f"{args.commodity} Residual", row=3, col=1, gridcolor=STYLING["grid"])

    # --- HTML Write ---
    html_path = os.path.join(out_dir, "dashboard.html")
    pio.write_html(fig, file=html_path, auto_open=False)
    
    return html_path

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEGO Framework: Partial Correlation Dashboard")
    parser.add_argument("--commodity", required=True, help="Ticker 1 (e.g., CL=F)")
    parser.add_argument("--fx", required=True, help="Ticker 2 (e.g., CAD=X)")
    parser.add_argument("--usd-index", default="UUP", help="Proxy for USD strength (e.g., UUP or DX-Y.NYB)")
    parser.add_argument("--period", default="2y", help="Lookback period")
    parser.add_argument("--base-window", type=int, default=63, help="Rolling window size (days)")
    args = parser.parse_args()

    out_dir = get_output_dir(args.commodity, args.fx)
    print(f"[INIT] Output Directory: {out_dir}")
    
    # Run Pipeline
    try:
        df, rolling_df = run_analysis(
            args.commodity, args.fx, args.usd_index, 
            args.period, [21, 63, 126, 252], out_dir
        )
        
        # Visualize
        path = generate_dashboard(df, rolling_df, args, out_dir)
        print(f"[SUCCESS] Dashboard generated at: {path}")
        
        # Open in Browser
        webbrowser.open(f"file://{path}")
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
