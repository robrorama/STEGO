#!/usr/bin/env python3
# SCRIPTNAME: ok.probabilities_explorer.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Unified probabilities & exploratory analysis script.

- Integrates functionality from probabilities.v1..v4.*:
  * Descriptive stats
  * Plotly candlestick
  * MAs (20, 50), MA crossover (50/200)
  * Daily returns, distribution w/ normal fit, empirical CDF and probability helpers
  * Rolling vol (annualized), expanding vol, EWMA vol
  * Autocorrelation (daily returns)
  * GBM paths + Monte Carlo final-price CI
  * Rolling correlation with VIX (^VIX)
  * Linear regression trendline (Close vs time, R^2)
  * Histograms (Close, Volume), Box plots (OHLC), Correlation heatmap (OHLC+Volume)
  * Scatter: Volume vs Close
  * Rolling mean/std (252d) alongside price
  * Pair plot (scatter matrix) for OHLC
- Opens each logical group in its own Plotly HTML **tab**.
- All data comes from data_retrieval.py (caching + output dir).

Usage:
    python3 probabilities_unified.py TICKER [period] [--steps 252] [--sims 1000] [--lags 30]
"""

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import math
import webbrowser
import numpy as np
import pandas as pd
from scipy.stats import norm, linregress
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr  # must be in PYTHONPATH / same dir
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ----------------------- CLI -----------------------

def parse_args(argv):
    if len(argv) < 2:
        print("Usage: python3 probabilities_unified.py TICKER [period] [--steps N] [--sims N] [--lags L]")
        sys.exit(1)
    ticker = argv[1]
    period = argv[2] if len(argv) >= 3 and not argv[2].startswith("--") else "max"
    steps = 252
    sims = 1000
    lags = 30
    i = 3 if period != "max" or (len(argv) >= 3 and not argv[2].startswith("--")) else 2
    while i < len(argv):
        if argv[i] == "--steps" and i + 1 < len(argv):
            steps = int(argv[i+1]); i += 2
        elif argv[i] == "--sims" and i + 1 < len(argv):
            sims = int(argv[i+1]); i += 2
        elif argv[i] == "--lags" and i + 1 < len(argv):
            lags = int(argv[i+1]); i += 2
        else:
            i += 1
    return ticker, period, steps, sims, lags

# ----------------------- helpers -----------------------

def write_and_open(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.write_html(path, include_plotlyjs="inline", full_html=True)
    webbrowser.open_new_tab("file://" + os.path.abspath(path))
    return path

def gbm_paths(s0, mu, sigma, dt, steps, n_sims, seed=None):
    if s0 <= 0 or sigma < 0 or dt <= 0 or steps <= 0 or n_sims <= 0:
        raise ValueError("Invalid GBM parameters")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((steps, n_sims))
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    log_returns = drift + vol * z
    # cumulative log-returns
    log_price = np.vstack([np.zeros((1, n_sims)), np.cumsum(log_returns, axis=0)])
    paths = s0 * np.exp(log_price)
    return paths  # shape: (steps+1, n_sims)

def autocorr_series(x, max_lag):
    x = pd.Series(x).dropna()
    return [x.autocorr(lag=k) for k in range(1, max_lag + 1)]

# ----------------------- main -----------------------

def main():
    ticker, period, steps, sims, lags = parse_args(sys.argv)
    
    # CONSTRAINT: Output to /dev/shm via data_retrieval logic
    outdir = dr.create_output_directory(ticker)

    # ---- data ----
    # CONSTRAINT: Use data_retrieval logic
    df = dr.load_or_download_ticker(ticker, period=period)
    if df is None or df.empty:
        print(f"Error: no data for {ticker} (period={period}).")
        sys.exit(2)

    # basic prints (v2/v3)
    print("Descriptive statistics:")
    print(df.describe())

    # derive series
    close = df['Close'].astype(float)
    volume = df['Volume'].astype(float) if 'Volume' in df.columns else pd.Series(index=df.index, dtype=float)
    ret = close.pct_change().dropna()
    ret_pct = 100.0 * ret  # in percent for human readability

    # MAs & crossovers (v2, v4a)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    # rolling vol (annualized), expanding vol, EWMA vol (v2)
    rolling_vol = ret.rolling(252).std() * math.sqrt(252)
    expanding_vol = ret.expanding().std() * math.sqrt(252)
    ewma_vol = ret.ewm(span=252).std() * math.sqrt(252)

    # rolling stats (v3/v4b)
    rolling_mean_252 = close.rolling(252).mean()
    rolling_std_252 = close.rolling(252).std()

    # ----- FIG 1: Candlestick + MAs (v3/v4) -----
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=close, name="OHLC"
    ))
    fig_candle.add_trace(go.Scatter(x=df.index, y=ma20, name="MA 20", mode="lines"))
    fig_candle.add_trace(go.Scatter(x=df.index, y=ma50, name="MA 50", mode="lines"))
    fig_candle.add_trace(go.Scatter(x=df.index, y=ma200, name="MA 200", mode="lines"))
    fig_candle.update_layout(
        title=f"{ticker} — Candlestick with MAs (20/50/200) — period={period}",
        xaxis_rangeslider_visible=False, height=700
    )
    write_and_open(fig_candle, outdir, f"{ticker}_candlestick_MAs.html")

    # ----- FIG 2: Distribution & CDF of daily returns (v2 + v1) -----
    # fit normal on ret_pct for display
    mu_p = ret_pct.mean()
    std_p = ret_pct.std(ddof=0)
    x_grid = np.linspace(ret_pct.min(), ret_pct.max(), 400)
    pdf_fit = norm.pdf(x_grid, mu_p, std_p)

    sorted_ret = np.sort(ret_pct.values)
    cdf_y = np.arange(1, len(sorted_ret) + 1) / len(sorted_ret)

    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("Daily Returns Distribution", "Empirical CDF"))
    fig_dist.add_trace(go.Histogram(x=ret_pct, nbinsx=60, histnorm="probability density", name="Returns"), row=1, col=1)
    fig_dist.add_trace(go.Scatter(x=x_grid, y=pdf_fit, mode="lines", name=f"Normal fit (μ={mu_p:.3f}%, σ={std_p:.3f}%)"), row=1, col=1)
    fig_dist.add_trace(go.Scatter(x=sorted_ret, y=cdf_y, mode="lines", name="Empirical CDF"), row=1, col=2)
    fig_dist.update_layout(title=f"{ticker} — Returns Distribution & CDF", height=600, showlegend=True)
    write_and_open(fig_dist, outdir, f"{ticker}_distribution_CDF.html")

    # ----- Probability helpers (v1) -----
    def get_probability(percentage_change):
        if len(sorted_ret) == 0:
            return float('nan')
        if percentage_change < sorted_ret[0]:
            return 0.0
        if percentage_change > sorted_ret[-1]:
            return 1.0
        # percentileofscore on sorted array equivalent
        idx = np.searchsorted(sorted_ret, percentage_change, side="right")
        return idx / len(sorted_ret)

    def get_probability_range(lower_bound, upper_bound):
        return max(0.0, get_probability(upper_bound) - get_probability(lower_bound))

    # example prints (mirroring v1)
    example_percentage = -2.5
    print(f"P(Δ ≤ {example_percentage:.2f}%): {get_probability(example_percentage):.4f}")
    example_lower, example_upper = 1.0, 2.0
    print(f"P({example_lower:.2f}% ≤ Δ ≤ {example_upper:.2f}%): {get_probability_range(example_lower, example_upper):.4f}")

    # ----- FIG 3: Autocorrelation of daily returns (v2) -----
    acf_vals = autocorr_series(ret, max(lags, 1))
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=list(range(1, len(acf_vals)+1)), y=acf_vals, name="ACF"))
    fig_acf.update_layout(title=f"{ticker} — Autocorrelation of Daily Returns (lags={len(acf_vals)})", xaxis_title="Lag", yaxis_title="Autocorr", height=500)
    write_and_open(fig_acf, outdir, f"{ticker}_autocorrelation.html")

    # ----- FIG 4: Volatility suite (rolling, expanding, EWMA) (v2) -----
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, name="Rolling Vol (252d)", mode="lines"))
    fig_vol.add_trace(go.Scatter(x=expanding_vol.index, y=expanding_vol, name="Expanding Vol", mode="lines"))
    fig_vol.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol, name="EWMA Vol (span=252)", mode="lines"))
    fig_vol.update_layout(title=f"{ticker} — Annualized Volatility (Rolling/Expanding/EWMA)", height=550, yaxis_title="Vol (annualized)")
    write_and_open(fig_vol, outdir, f"{ticker}_volatility_suite.html")

    # ----- FIG 5: GBM simulations + final-price distribution (v2) -----
    s0 = float(close.iloc[-1])
    mu_ann = ret.mean() * 252.0
    sigma_ann = ret.std(ddof=0) * math.sqrt(252.0)
    dt = 1.0 / 252.0
    paths = gbm_paths(s0, mu_ann, sigma_ann, dt, steps, sims)
    final_prices = paths[-1, :]
    ci_lo, ci_hi = np.percentile(final_prices, [2.5, 97.5])

    fig_gbm = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.08,
                            subplot_titles=(f"GBM Paths (μ={mu_ann:.3f}, σ={sigma_ann:.3f})",
                                            f"Final Price Distribution — 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]"))
    # show up to 100 paths to keep it readable
    show_n = min(100, sims)
    for j in range(show_n):
        fig_gbm.add_trace(go.Scatter(x=list(range(paths.shape[0])), y=paths[:, j], mode="lines", showlegend=False), row=1, col=1)
    fig_gbm.add_trace(go.Histogram(x=final_prices, nbinsx=60, name="Final Prices"), row=2, col=1)
    fig_gbm.add_vline(x=ci_lo, line_dash="dash", line_color="red", row=2, col=1)
    fig_gbm.add_vline(x=ci_hi, line_dash="dash", line_color="red", row=2, col=1)
    fig_gbm.update_layout(height=800)
    write_and_open(fig_gbm, outdir, f"{ticker}_gbm_monte_carlo.html")

    # ----- FIG 6: Rolling correlation with VIX (v2) -----
    try:
        vix = dr.load_or_download_ticker("^VIX", period=period)
        vix_ret = vix['Close'].pct_change()
        joined = pd.concat([ret, vix_ret], axis=1, join="inner").dropna()
        joined.columns = ["RET", "VIX_RET"]
        rol_corr = joined['RET'].rolling(252).corr(joined['VIX_RET'])
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=rol_corr.index, y=rol_corr, mode="lines", name="Rolling Corr (252d)"))
        fig_corr.update_layout(title=f"{ticker} vs VIX — Rolling Correlation (252d)", yaxis_title="Correlation", height=500)
        write_and_open(fig_corr, outdir, f"{ticker}_rolling_corr_VIX.html")
    except Exception as e:
        print(f"Rolling correlation with VIX failed: {e}")

    # ----- FIG 7: Linear regression trendline (v2) -----
    x = np.arange(len(close))
    slope, intercept, r, p, se = linregress(x, close.values)
    trend = slope * x + intercept
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df.index, y=close, name="Close", mode="lines"))
    fig_trend.add_trace(go.Scatter(x=df.index, y=trend, name=f"Trendline (R²={r**2:.2f})", mode="lines"))
    fig_trend.update_layout(title=f"{ticker} — Trendline via Linear Regression", height=550)
    write_and_open(fig_trend, outdir, f"{ticker}_trendline.html")

    # ----- FIG 8: Histograms Close/Volume (v3/v4b) -----
    fig_hist = make_subplots(rows=1, cols=2, subplot_titles=("Close Histogram", "Volume Histogram"))
    fig_hist.add_trace(go.Histogram(x=close, nbinsx=60, name="Close"), row=1, col=1)
    if not volume.isna().all():
        fig_hist.add_trace(go.Histogram(x=volume, nbinsx=60, name="Volume"), row=1, col=2)
    fig_hist.update_layout(title=f"{ticker} — Histograms", height=500)
    write_and_open(fig_hist, outdir, f"{ticker}_histograms.html")

    # ----- FIG 9: Box plots OHLC (v3/v4b) -----
    fig_box = go.Figure()
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            fig_box.add_trace(go.Box(y=df[col], name=col))
    fig_box.update_layout(title=f"{ticker} — Box Plots (OHLC)", height=500)
    write_and_open(fig_box, outdir, f"{ticker}_boxplots_OHLC.html")

    # ----- FIG 10: Correlation heatmap (v3/v4b) -----
    corr = df[["Open", "High", "Low", "Close", "Volume"]].dropna().corr()
    fig_heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu"))
    fig_heat.update_layout(title=f"{ticker} — Correlation Heatmap (OHLC+Volume)", height=600)
    write_and_open(fig_heat, outdir, f"{ticker}_correlation_heatmap.html")

    # ----- FIG 11: Rolling mean/std with price (v3/v4b) -----
    fig_rollstats = go.Figure()
    fig_rollstats.add_trace(go.Scatter(x=df.index, y=close, name="Close", mode="lines"))
    fig_rollstats.add_trace(go.Scatter(x=df.index, y=rolling_mean_252, name="252d Rolling Mean", mode="lines"))
    fig_rollstats.add_trace(go.Scatter(x=df.index, y=rolling_std_252, name="252d Rolling Std", mode="lines"))
    fig_rollstats.update_layout(title=f"{ticker} — Close with 252d Rolling Mean/Std", height=550)
    write_and_open(fig_rollstats, outdir, f"{ticker}_rolling_mean_std.html")

    # ----- FIG 12: Volume vs Close scatter (v3/v4b) -----
    if not volume.isna().all():
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=volume, y=close, mode="markers", name="Pts", opacity=0.5))
        fig_scatter.update_layout(title=f"{ticker} — Volume vs Close", xaxis_title="Volume", yaxis_title="Close", height=550)
        write_and_open(fig_scatter, outdir, f"{ticker}_scatter_volume_close.html")

    # ----- FIG 13: Pair plot (scatter matrix) OHLC (v4a) -----
    ohlc = df[["Open", "High", "Low", "Close"]].dropna()
    # downsample if very large for responsiveness
    max_points = 1500
    if len(ohlc) > max_points:
        step = int(len(ohlc) / max_points)
        ohlc = ohlc.iloc[::step]
    fig_pair = px.scatter_matrix(ohlc, dimensions=["Open", "High", "Low", "Close"], title=f"{ticker} — Scatter Matrix (OHLC)")
    fig_pair.update_traces(diagonal_visible=True, showupperhalf=True)
    fig_pair.update_layout(height=900, width=900)
    write_and_open(fig_pair, outdir, f"{ticker}_pairplot_OHLC.html")

    # ----- FIG 14: Cumulative returns curve (v4a) -----
    cumret = (1.0 + ret).cumprod()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=cumret.index, y=cumret, mode="lines", name="Cumulative Return"))
    fig_cum.update_layout(title=f"{ticker} — Cumulative Returns", height=500, yaxis_title="Multiple of Initial")
    write_and_open(fig_cum, outdir, f"{ticker}_cumulative_returns.html")

    print(f"HTML outputs written to: {outdir}")

if __name__ == "__main__":
    main()
