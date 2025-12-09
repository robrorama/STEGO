#!/usr/bin/env python3
# SCRIPTNAME: ok.fx_commodity_partial_corr_dashboard_yf.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
FX-Commodity Partial Correlation Dashboard
------------------------------------------
A standalone CLI tool to analyze the relationship between a commodity and a currency pair,
controlling for the USD index (DXY/UUP).

Usage:
    python3 fx_commodity_partial_corr_dashboard_yf.py --commodity "CL=F" --fx "CAD=X" --usd-index "UUP"
    python3 fx_commodity_partial_corr_dashboard_yf.py --commodity "GC=F" --fx "AUD=X" --period "5y"

Requirements:
    pip install yfinance pandas numpy plotly
    # (optional) pip install scipy
"""

import argparse
import os
import sys
import datetime
import webbrowser
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------------
# 1. ROBUSTNESS HELPERS (Rules 4a, 4b, 4c)
# ------------------------------------------------------------------------------

def force_tz_naive_and_normalize(series: pd.Series) -> pd.Series:
    """
    Rule 4(a): Timezone Fix.
    Converts index to datetime, strips timezone, sorts, and normalizes to midnight.
    Handles duplicate indices by keeping the last observation.
    """
    if series.empty:
        return series
        
    # Ensure datetime index
    series.index = pd.to_datetime(series.index)
    
    # Strip timezone if present
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
        
    # Normalize to midnight (date only)
    series.index = series.index.normalize()
    
    # Sort
    series = series.sort_index()
    
    # Handle duplicates (keep last)
    series = series[~series.index.duplicated(keep='last')]
    
    return series

def robust_extract_close(df: pd.DataFrame) -> pd.Series:
    """
    Rule 4(b): Robust column extraction.
    extracts 'Close' > 'Adj Close' > First Column.
    """
    if df.empty:
        return pd.Series(dtype=float)

    # Flatten MultiIndex columns if necessary
    # yfinance sometimes returns (Price, Ticker) or (Ticker, Price) or just Price
    if isinstance(df.columns, pd.MultiIndex):
        # Try to find a level named 'Close' or 'Adj Close'
        # Or just drop the ticker level if it's a single ticker download
        try:
            df = df.droplevel(1, axis=1) 
        except:
            pass # Keep trying other methods if droplevel fails or isn't right

    # Brute-force logic as requested
    if isinstance(df, pd.DataFrame):
        if 'Close' in df.columns:
            s = df['Close']
        elif 'Adj Close' in df.columns:
            s = df['Adj Close']
        else:
            s = df.iloc[:, 0]  # fallback
    else:
        s = df
        
    return s

def align_intersection(c: pd.Series, f: pd.Series, u: pd.Series):
    """
    Rule 4(c): Alignment Fix.
    Strict intersection of dates across all three assets.
    """
    common = c.index
    common = common.intersection(f.index)
    common = common.intersection(u.index)
    
    # Hard-filter
    c_aligned = c.loc[common]
    f_aligned = f.loc[common]
    u_aligned = u.loc[common]
    
    return c_aligned, f_aligned, u_aligned

# ------------------------------------------------------------------------------
# 2. DATA DOWNLOADER
# ------------------------------------------------------------------------------

def download_ticker(ticker: str, period: str) -> pd.Series:
    """
    Strictly downloads one ticker using yfinance.
    Applies column extraction and timezone fixing immediately.
    """
    print(f"[INFO] Downloading {ticker} (period={period})...")
    try:
        df = yf.download(
            ticker, 
            period=period, 
            interval="1d", 
            auto_adjust=False, 
            actions=False, 
            progress=False, 
            threads=False
        )
        
        if df is None or df.empty:
            print(f"[ERROR] Download returned empty data for {ticker}")
            return pd.Series(dtype=float)
            
        # Extract Close
        s = robust_extract_close(df)
        
        # Apply Timezone Fix
        s = force_tz_naive_and_normalize(s)
        
        # Rename series for clarity (optional, mainly for debug)
        s.name = ticker
        
        return s
        
    except Exception as e:
        print(f"[ERROR] Exception downloading {ticker}: {e}")
        return pd.Series(dtype=float)

# ------------------------------------------------------------------------------
# 3. ANALYTICS
# ------------------------------------------------------------------------------

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """ Computes log returns: ln(P_t / P_{t-1}) """
    # Filter 0 or negative prices to avoid log errors
    valid = prices[prices > 0]
    return np.log(valid).diff().dropna()

def compute_rolling_stats(df_rets, comm_col, fx_col, usd_col, window):
    """
    Computes rolling correlations and partial correlations.
    """
    # Rolling window object
    r = df_rets.rolling(window=window)
    
    # Pairwise Correlations
    corr_cf = r.corr(pairwise=True)[comm_col].xs(fx_col, level=1) # This syntax depends on pandas version, easier to do manual columns
    
    # Let's do it explicitly to be safe across pandas versions
    # df_rets has columns [comm, fx, usd]
    
    rho_cf = df_rets[comm_col].rolling(window).corr(df_rets[fx_col])
    rho_cu = df_rets[comm_col].rolling(window).corr(df_rets[usd_col])
    rho_fu = df_rets[fx_col].rolling(window).corr(df_rets[usd_col])
    
    # Partial Correlation Formula:
    # r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    # x = Comm, y = FX, z = USD
    
    numerator = rho_cf - (rho_cu * rho_fu)
    denominator = np.sqrt((1 - rho_cu**2) * (1 - rho_fu**2))
    
    partial_corr = numerator / denominator
    
    return rho_cf, partial_corr, rho_cu

def get_ols_residuals(y: np.array, x: np.array):
    """
    Regress y on x (y = a + bx + e), return residuals e.
    Uses numpy.linalg.lstsq
    """
    # A = [x, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    residuals = y - (m * x + c)
    return residuals

# ------------------------------------------------------------------------------
# 4. MAIN SCRIPT
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FX-Commodity Partial Correlation Dashboard")
    parser.add_argument("--commodity", type=str, required=True, help="Ticker for commodity (e.g., CL=F)")
    parser.add_argument("--fx", type=str, required=True, help="Ticker for FX pair (e.g., CAD=X)")
    parser.add_argument("--usd-index", type=str, default="UUP", help="Ticker for USD index proxy (default UUP)")
    parser.add_argument("--period", type=str, default="2y", help="Data period (default 2y)")
    parser.add_argument("--base-window", type=int, default=63, help="Rolling window size (default 63)")
    
    args = parser.parse_args()
    
    # 1. Download
    s_comm = download_ticker(args.commodity, args.period)
    s_fx = download_ticker(args.fx, args.period)
    s_usd = download_ticker(args.usd_index, args.period)
    
    # 2. Check Empty
    if s_comm.empty or s_fx.empty or s_usd.empty:
        sys.stderr.write("[FATAL] One or more datasets is empty. Exiting.\n")
        sys.exit(1)
        
    # 3. Align (Rule 4c)
    s_comm, s_fx, s_usd = align_intersection(s_comm, s_fx, s_usd)
    
    if len(s_comm) < args.base_window:
        sys.stderr.write(f"[FATAL] Not enough aligned data ({len(s_comm)} rows) for window {args.base_window}.\n")
        sys.exit(1)
        
    # 4. Build Price DataFrame
    df_prices = pd.DataFrame({
        'C_close': s_comm,
        'F_close': s_fx,
        'U_close': s_usd
    })
    
    # 5. Compute Returns
    # Using numpy log diff
    df_rets = pd.DataFrame()
    df_rets['C_ret'] = np.log(s_comm).diff()
    df_rets['F_ret'] = np.log(s_fx).diff()
    df_rets['U_ret'] = np.log(s_usd).diff()
    
    # Drop NaN from diff
    df_rets.dropna(inplace=True)
    
    # Re-align prices to returns (lose first row)
    df_prices = df_prices.loc[df_rets.index]
    
    # 6. Rolling Metrics (Base Window)
    # Using analytic formula for speed and stability
    raw_corr, partial_corr, comm_usd_corr = compute_rolling_stats(
        df_rets, 'C_ret', 'F_ret', 'U_ret', args.base_window
    )
    
    # 7. Full Sample Residuals
    # Regress C_ret on U_ret -> eps_C
    eps_C = get_ols_residuals(df_rets['C_ret'].values, df_rets['U_ret'].values)
    # Regress F_ret on U_ret -> eps_F
    eps_F = get_ols_residuals(df_rets['F_ret'].values, df_rets['U_ret'].values)
    
    df_residuals = pd.DataFrame({
        'eps_C': eps_C,
        'eps_F': eps_F
    }, index=df_rets.index)
    
    # 8. Visuals
    
    # Normalize Prices (base 100)
    df_norm = df_prices.div(df_prices.iloc[0]) * 100
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], 
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Normalized Prices (Aligned)", 
            f"Rolling {args.base_window}-Day Correlations", 
            "Full Sample Residuals (Net of USD)"
        ),
        vertical_spacing=0.15
    )
    
    # -- Top Row: Prices --
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['C_close'], name=f"Comm ({args.commodity})", line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['F_close'], name=f"FX ({args.fx})", line=dict(color='magenta')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['U_close'], name=f"USD ({args.usd_index})", line=dict(color='gray', dash='dot')), row=1, col=1)
    
    # -- Bottom Left: Rolling Corrs --
    fig.add_trace(go.Scatter(x=raw_corr.index, y=raw_corr, name="Raw Corr(Comm, FX)", line=dict(color='yellow')), row=2, col=1)
    fig.add_trace(go.Scatter(x=partial_corr.index, y=partial_corr, name="Partial Corr (Net USD)", line=dict(color='lime', width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=comm_usd_corr.index, y=comm_usd_corr, name="Corr(Comm, USD)", line=dict(color='red', width=1)), row=2, col=1)
    
    # Add zero line for correlations
    fig.add_shape(type="line", x0=raw_corr.index[0], x1=raw_corr.index[-1], y0=0, y1=0, line=dict(color="white", width=1, dash="dash"), row=2, col=1)
    
    # -- Bottom Right: Scatter --
    fig.add_trace(go.Scatter(
        x=df_residuals['eps_F'], 
        y=df_residuals['eps_C'], 
        mode='markers', 
        name='Residuals',
        marker=dict(color='rgba(100, 200, 255, 0.6)', size=5)
    ), row=2, col=2)
    
    # Update Axes
    fig.update_xaxes(title_text="Residual FX (Net of USD)", row=2, col=2)
    fig.update_yaxes(title_text="Residual Comm (Net of USD)", row=2, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        title_text=f"FX-Commodity Linkage: {args.commodity} vs {args.fx} (Control: {args.usd_index})",
        height=900,
        showlegend=True
    )
    
    # 9. Output
    # Determine directory
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    sanitized_pair = f"{args.commodity}_{args.fx}".replace("=", "").replace("^", "")
    out_dir = os.path.join(base_dir, "FX_COMM_DASH", sanitized_pair, date_str)
    
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "aligned_prices.csv")
    html_path = os.path.join(out_dir, "dashboard.html")
    
    # Save CSV
    df_prices.to_csv(csv_path)
    
    # Save HTML
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
    
    print(f"\n[SUCCESS] Dashboard generated.")
    print(f"Data saved to: {csv_path}")
    print(f"Dashboard at:  {html_path}")
    
    # Attempt to open
    try:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
