#!/usr/bin/env python3
# SCRIPTNAME: stochastics_unified.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Integrated analytics + ML from:
# machine.learning.a.py, second.try.model.v1.py, third.a.py, third.b.py
# All data access routed through data_retrieval.py.

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import math
import webbrowser
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr  # <- single source of data / output dirs
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ----------------------------- utils -----------------------------

def write_and_open(fig, out_dir: str, filename: str, open_tabs: bool = True) -> str:
    # CONSTRAINT: out_dir comes from data_retrieval, ensuring /dev/shm
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    if open_tabs:
        webbrowser.open_new_tab("file://" + os.path.abspath(path))
    return path

def returns_and_vol(close: pd.Series):
    ret = close.pct_change()
    vol_ann = ret.rolling(252).std() * np.sqrt(252)
    return ret, vol_ann

def normal_pdf(x, mu, sigma):
    if sigma <= 0:  # guard
        return np.zeros_like(x)
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def compute_acf(series: pd.Series, nlags: int = 60):
    x = series.dropna().values
    if x.size < 2:
        return np.array([]), np.array([])
    x = x - x.mean()
    denom = (x**2).sum()
    lags = np.arange(nlags + 1)
    acf_vals = []
    for k in lags:
        if k >= len(x):
            acf_vals.append(np.nan)
        else:
            acf_vals.append(np.dot(x[:-k] if k > 0 else x, x[k:]) / denom)
    return lags, np.array(acf_vals)

def gbm_paths(s0: float, mu_annual: float, sig_annual: float, steps: int, sims: int, dt: float = 1/252.0):
    if s0 <= 0 or sig_annual < 0 or dt <= 0 or steps <= 0 or sims <= 0:
        raise ValueError("Invalid GBM parameters")
    z = np.random.standard_normal((steps, sims))
    drift = (mu_annual - 0.5 * sig_annual**2) * dt
    shock = sig_annual * np.sqrt(dt) * z
    log_rel = drift + shock
    log_rel = np.vstack([np.zeros((1, sims)), log_rel]).cumsum(axis=0)  # (steps+1, sims)
    return s0 * np.exp(log_rel)  # price paths

def trendline(y: np.ndarray):
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return yhat, r2

# ----------------------------- figures -----------------------------

def make_dashboard_fig(ticker: str, df: pd.DataFrame, vix_df: pd.DataFrame,
                       steps: int, sims: int, acf_lags: int):
    close = df['Close'].astype(float)
    ret, vol_ann = returns_and_vol(close)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # (4) histogram + normal fit
    ret_clean = ret.dropna()
    mu, sigma = float(ret_clean.mean()), float(ret_clean.std(ddof=1))
    xmin, xmax = (ret_clean.min(), ret_clean.max()) if not ret_clean.empty else (-0.05, 0.05)
    x_pdf = np.linspace(xmin, xmax, 200)
    y_pdf = normal_pdf(x_pdf, mu, sigma)

    # (5) autocorrelation
    lags, acf_vals = compute_acf(ret, nlags=acf_lags)

    # (6,7) GBM + 95% CI
    s0 = float(close.iloc[-1])
    mu_ann = ret_clean.mean() * 252 if ret_clean.size else 0.0
    sig_ann = ret_clean.std(ddof=1) * np.sqrt(252) if ret_clean.size else 0.0
    paths = gbm_paths(s0, float(mu_ann), float(sig_ann), steps=steps, sims=sims)
    final_prices = paths[-1]
    ci_lo, ci_hi = np.percentile(final_prices, [2.5, 97.5])

    # (8) rolling corr with VIX
    vix_ret = vix_df['Close'].pct_change() if not vix_df.empty else pd.Series(index=df.index, dtype=float)
    corr = pd.concat([ret, vix_ret], axis=1, join='inner').dropna()
    corr.columns = ['SPY_R', 'VIX_R']
    roll_corr = corr['SPY_R'].rolling(252).corr(corr['VIX_R'])

    # (9) trendline
    yhat, r2 = trendline(close.values)

    # (10) expanding vol
    expanding_vol = ret.expanding().std() * np.sqrt(252)

    fig = make_subplots(rows=3, cols=3, vertical_spacing=0.09, horizontal_spacing=0.06,
                        subplot_titles=(
                            "Close", "Close + MA20/MA50", "Rolling Vol (Ann, 252d)",
                            "Return Histogram + Normal Fit",
                            f"Autocorrelation (lags≤{acf_lags})",
                            f"GBM {steps} steps / {sims} sims (95% CI: {ci_lo:.2f}-{ci_hi:.2f})",
                            "Rolling Corr(SPY,VIX)",
                            f"Trendline (R²={r2:.2f})",
                            "Expanding Vol (Ann)"
                        ))

    # (1) Close
    fig.add_trace(go.Scatter(x=df.index, y=close, mode="lines", name="Close"),
                  row=1, col=1)

    # (2) Close + MAs
    fig.add_trace(go.Scatter(x=df.index, y=close, mode="lines", name="Close"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=ma20, mode="lines", name="MA20"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=ma50, mode="lines", name="MA50"), row=1, col=2)

    # (3) Rolling Vol
    fig.add_trace(go.Scatter(x=vol_ann.index, y=vol_ann, mode="lines", name="Vol(Ann)"),
                  row=1, col=3)

    # (4) Histogram + Normal fit
    fig.add_trace(go.Histogram(x=ret_clean, histnorm='probability density', name="Returns",
                               nbinsx=60, opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_pdf, y=y_pdf, mode="lines", name=f"N({mu:.4f},{sigma:.4f})"),
                  row=2, col=1)

    # (5) ACF
    if lags.size:
        fig.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF"), row=2, col=2)

    # (6) GBM paths (subsample display for clarity if many sims)
    draw = min(sims, 200)
    for j in range(draw):
        fig.add_trace(go.Scatter(x=np.arange(paths.shape[0]), y=paths[:, j],
                                 mode="lines", name=None, showlegend=False,
                                 hoverinfo="skip", line=dict(width=1)),
                      row=2, col=3)

    # (8) Rolling corr with VIX
    if not roll_corr.empty:
        fig.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, mode="lines", name="Roll Corr"),
                      row=3, col=1)

    # (9) Trendline
    fig.add_trace(go.Scatter(x=df.index, y=close, mode="lines", name="Close"),
                  row=3, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=yhat, mode="lines", name="Trend", line=dict(dash="dash")),
                  row=3, col=2)

    # (10) Expanding vol
    fig.add_trace(go.Scatter(x=expanding_vol.index, y=expanding_vol, mode="lines", name="Exp Vol"),
                  row=3, col=3)

    fig.update_layout(height=1100, width=1500, title_text=f"{ticker} — Stochastics Dashboard")
    return fig, (ci_lo, ci_hi)

def make_ewma_fig(ticker: str, df: pd.DataFrame):
    ret = df['Close'].pct_change()
    ewma_vol = ret.ewm(span=252).std() * np.sqrt(252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol, mode="lines", name="EWMA Vol (Ann)"))
    fig.update_layout(title=f"{ticker} — EWMA Volatility (span=252)",
                      xaxis_title="Date", yaxis_title="Vol (annualized)")
    return fig

def make_ml_fig_and_stats(ticker: str, df: pd.DataFrame, test_size: float = 0.2, next_n: int = 5):
    # Features as in machine.learning.a.py
    xclose = df['Close'].astype(float)
    ret = xclose.pct_change()
    fe = pd.DataFrame(index=df.index)
    fe['Daily_Return'] = ret
    fe['MA_5'] = xclose.rolling(5).mean()
    fe['MA_20'] = xclose.rolling(20).mean()
    fe['Volatility_20'] = ret.rolling(20).std()
    data = pd.concat([fe, xclose.rename('Close')], axis=1).dropna()

    X = data[['Daily_Return', 'MA_5', 'MA_20', 'Volatility_20']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))

    # Plot (actual vs predicted over the test window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=y_test.index, y=preds, mode="lines", name="Predicted"))
    fig.update_layout(title=f"{ticker} — Linear Regression (MSE={mse:.4f})",
                      xaxis_title="Date", yaxis_title="Price")

    # “Next-N” predictions (using last N rows of available features)
    n = max(1, int(next_n))
    tail_feats = X.tail(n)
    next_preds = model.predict(tail_feats) if not tail_feats.empty else np.array([])

    return fig, mse, pd.Series(next_preds, index=tail_feats.index, name="Predicted_Close")

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified stochastics + ML script (HTML tabs). Uses data_retrieval.py.")
    ap.add_argument("ticker", nargs="?", default="SPY", help="Ticker (default: SPY)")
    ap.add_argument("-p", "--period", default="max", help="Period for data_retrieval (e.g., 1y, 2y, max). Default: max")
    ap.add_argument("--steps", type=int, default=252, help="GBM steps (default 252)")
    ap.add_argument("--sims", type=int, default=1000, help="GBM simulations (default 1000)")
    ap.add_argument("--acf-lags", type=int, default=60, help="ACF lags (default 60)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test size for ML (default 0.2)")
    ap.add_argument("--next-n", type=int, default=5, help="Next-N predictions (default 5)")
    ap.add_argument("--no-open", action="store_true", help="Do not auto-open browser tabs")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    period = args.period

    # Load data via data_retrieval
    # CONSTRAINT: Use data_retrieval logic
    print(f"Loading {tkr} (period={period}) via data_retrieval ...")
    df = dr.get_stock_data(tkr, period=period)
    if df is None or df.empty or 'Close' not in df.columns:
        sys.exit(f"ERROR: No usable data for {tkr}.")

    # Also load VIX for rolling correlation panel
    vix = dr.get_stock_data("^VIX", period=period)
    
    # CONSTRAINT: Use data_retrieval for output path (/dev/shm)
    out_dir = dr.create_output_directory(tkr)

    # Print basic describe (as in second/third scripts)
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(df.describe())

    # Dashboard (3x3)
    fig_dash, (ci_lo, ci_hi) = make_dashboard_fig(tkr, df, vix, steps=args.steps, sims=args.sims, acf_lags=args.acf_lags)
    dash_path = write_and_open(fig_dash, out_dir, "dashboard_3x3.html", open_tabs=not args.no_open)

    # EWMA volatility (separate)
    fig_ewma = make_ewma_fig(tkr, df)
    ewma_path = write_and_open(fig_ewma, out_dir, "ewma_volatility.html", open_tabs=not args.no_open)

    # Machine learning regression (separate)
    fig_ml, mse, next_preds = make_ml_fig_and_stats(tkr, df, test_size=args.test_size, next_n=args.next_n)
    ml_path = write_and_open(fig_ml, out_dir, "ml_regression.html", open_tabs=not args.no_open)

    # Console outputs that originals printed
    print(f"\n[GBM] 95% Confidence Interval for {tkr} in {args.steps} trading days: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"[ML]  Mean Squared Error on test window: {mse:.4f}")
    if not next_preds.empty:
        print(f"[ML]  Next-{len(next_preds)} predictions (indexed by last feature dates):")
        print(next_preds)

    print("\nSaved HTML:")
    for p in (dash_path, ewma_path, ml_path):
        print("  " + os.path.abspath(p))

if __name__ == "__main__":
    main()
