#!/usr/bin/env python3
# SCRIPTNAME: ok.multi_factor_volatility_dashboard.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
multi_factor_volatility_dashboard.py

A professional-grade multi-factor analytics dashboard for options traders.
Compares three market-regime detectors:
1. Wavelet Short-Scale Volatility Energy (High-frequency turbulence)
2. Welch Power Spectral Density Slope (1/f Noise structure / Spectral drift)
3. PELT-style Variance Structural Breaks (Regime change detection)

Dependencies:
    pip install pandas numpy scipy plotly yfinance
    Optional: pip install PyWavelets ruptures (Automatic fallbacks provided if missing)

Usage:
    python multi_factor_volatility_dashboard.py --ticker SPY --years 2
    python multi_factor_volatility_dashboard.py --ticker QQQ --years 5
"""

import os
import sys
import argparse
import logging
import datetime
import warnings
import webbrowser
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# Optional Libraries & Fallbacks
# -----------------------------------------------------------------------------
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    import ruptures as rpt
    HAS_RPT = True
except ImportError:
    HAS_RPT = False

# -----------------------------------------------------------------------------
# Logging & Config
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MultiFactorVol")
warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------------------------------------------------------
# 1. Robust Data Helpers
# -----------------------------------------------------------------------------

def force_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe index is timezone-naive."""
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def smart_get_close(df: pd.DataFrame) -> pd.Series:
    """Robustly extract Close column from yfinance result."""
    if df.empty:
        return pd.Series(dtype=float)
    
    # 1. Standard 'Close'
    if 'Close' in df.columns:
        return df['Close']
    
    # 2. MultiIndex (e.g. ('Close', 'SPY'))
    if isinstance(df.columns, pd.MultiIndex):
        try:
            return df.xs('Close', axis=1, level=0).iloc[:, 0]
        except KeyError:
            pass
            
    # 3. Fallback: Last column? No, 'Adj Close' usually
    if 'Adj Close' in df.columns:
        return df['Adj Close']
        
    # 4. Fallback: iloc
    return df.iloc[:, 0]

def download_data(ticker: str, years: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download Ticker and VIX data.
    Returns (df_ticker, df_vix) with 'Close', 'LogRet' columns.
    """
    start_date = (datetime.datetime.now() - datetime.timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    # Ticker
    logger.info(f"Downloading {ticker} from {start_date}...")
    df_t = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
    df_t = force_naive_index(df_t)
    
    # VIX (Implicit Volatility Regime)
    logger.info(f"Downloading ^VIX from {start_date}...")
    df_v = yf.download("^VIX", start=start_date, progress=False, auto_adjust=False)
    df_v = force_naive_index(df_v)
    
    # Process Ticker
    close_t = smart_get_close(df_t)
    df_t_clean = pd.DataFrame(index=df_t.index)
    df_t_clean['Close'] = close_t
    df_t_clean['LogRet'] = np.log(df_t_clean['Close'] / df_t_clean['Close'].shift(1))
    df_t_clean.dropna(inplace=True)
    
    # Process VIX
    close_v = smart_get_close(df_v)
    df_v_clean = pd.DataFrame(index=df_v.index)
    df_v_clean['Close'] = close_v
    # For VIX, log returns measure "vol of vol" changes
    df_v_clean['LogRet'] = np.log(df_v_clean['Close'] / df_v_clean['Close'].shift(1))
    df_v_clean.dropna(inplace=True)
    
    # Align
    common = df_t_clean.index.intersection(df_v_clean.index)
    return df_t_clean.loc[common], df_v_clean.loc[common]

# -----------------------------------------------------------------------------
# 2. Mathematical Detectors
# -----------------------------------------------------------------------------

def compute_wavelet_energy(series: pd.Series, window=10) -> pd.Series:
    """
    Detector A: Wavelet Short-Scale Volatility Energy.
    Measures high-frequency turbulence often preceding shocks.
    Fallback: Manual Haar DWT if PyWavelets not installed.
    """
    arr = series.values
    
    if HAS_PYWT:
        # Discrete Wavelet Transform using 'db1' (Haar)
        # We want Detail coefficients (cD) at level 1 for highest freq
        try:
            (cA, cD) = pywt.dwt(arr, 'db1')
            # Pad cD to match length (DWT downsamples by 2)
            # We want a time-aligned series.
            # Use Stationary Wavelet Transform (SWT) for no downsampling if possible
            # But simple approximation: High Pass Filter.
            
            # Re-implementation for time-alignment without downsampling:
            # Haar High Pass is essentially x[t] - x[t-1] scaled.
            pass
        except Exception:
            pass

    # Manual Rolling Haar Energy Proxy (Robust & Fast)
    # Energy ~ Square of high-freq diffs
    # Haar Detail ~ (x[i] - x[i-1]) / sqrt(2)
    
    diffs = np.diff(arr, prepend=arr[0])
    energy = diffs ** 2
    
    # Smooth energy to get a readable signal
    energy_series = pd.Series(energy, index=series.index)
    energy_smooth = energy_series.rolling(window=window).mean()
    
    # Normalize (0-1) for comparison
    return (energy_smooth - energy_smooth.min()) / (energy_smooth.max() - energy_smooth.min())

def compute_psd_slope(series: pd.Series, window=60) -> pd.Series:
    """
    Detector B: Welch Power Spectral Density Slope.
    Estimates the exponent alpha in 1/f^alpha noise.
    Slope close to 0 = White Noise (Random Walk).
    Slope << 0 = Persistent Trends / Pink Noise.
    Rapid changes in slope indicate structural memory shifts.
    """
    res = []
    idx = []
    
    vals = series.values
    
    for i in range(window, len(vals)):
        segment = vals[i-window:i]
        
        # Detrend to focus on fluctuations
        segment = signal.detrend(segment)
        
        # Welch's method
        freqs, psd = signal.welch(segment, nperseg=window, scaling='density')
        
        # Fit Log-Log slope (exclude 0 freq)
        valid = (freqs > 0) & (psd > 0)
        if np.sum(valid) > 5:
            log_f = np.log(freqs[valid])
            log_p = np.log(psd[valid])
            
            # Linear regression
            slope, intercept, r_val, p_val, std_err = stats.linregress(log_f, log_p)
            res.append(slope)
        else:
            res.append(np.nan)
        
        idx.append(series.index[i])
        
    # Align to original index
    slope_series = pd.Series(res, index=idx)
    # Forward fill gaps
    slope_series = slope_series.reindex(series.index).ffill()
    return slope_series

def detect_regimes_pelt(series: pd.Series, penalty=10) -> List[pd.Timestamp]:
    """
    Detector C: PELT-style Structural Break Detection.
    Detects shifts in Variance (Volatility Regimes).
    Fallback: Rolling Variance Change Detector.
    """
    bkps_dates = []
    
    if HAS_RPT:
        try:
            # Change in variance (standard deviation)
            # Ruptures expects numpy array
            signal_arr = series.values.reshape(-1, 1)
            algo = rpt.Pelt(model="rbf").fit(signal_arr)
            result = algo.predict(pen=penalty)
            
            # Convert indices to dates
            for r in result:
                if r < len(series):
                    bkps_dates.append(series.index[r])
            return bkps_dates
        except Exception as e:
            logger.warning(f"Ruptures failed ({e}), using fallback.")
            
    # Fallback: Rolling Variance Gradient
    # If rolling variance jumps significantly, mark breakpoint
    roll_std = series.rolling(30).std()
    grad = np.abs(np.gradient(roll_std.fillna(0)))
    
    # Thresholding for fallback
    threshold = np.percentile(grad, 98) # Top 2% of shifts
    peaks, _ = signal.find_peaks(grad, height=threshold, distance=60) # Min 60 days between regimes
    
    for p in peaks:
        bkps_dates.append(series.index[p])
        
    return bkps_dates

# -----------------------------------------------------------------------------
# 3. Visualization Generator
# -----------------------------------------------------------------------------

def save_plot(fig, filename, out_dir, auto_open=False):
    path = os.path.join(out_dir, filename)
    fig.write_html(path)
    logger.info(f"Saved {filename}")
    if auto_open:
        webbrowser.open(f"file://{os.path.abspath(path)}")

def generate_dashboards(df_t: pd.DataFrame, df_v: pd.DataFrame, 
                        t_bkps: List[pd.Timestamp], v_bkps: List[pd.Timestamp],
                        ticker: str, out_dir: str):
    
    # Calculations
    logger.info("Computing Wavelet Energy...")
    df_t['WaveletEnergy'] = compute_wavelet_energy(df_t['LogRet'])
    df_v['WaveletEnergy'] = compute_wavelet_energy(df_v['LogRet']) # On VIX returns!
    
    logger.info("Computing PSD Slope (Spectral Drift)...")
    df_t['PSDSlope'] = compute_psd_slope(df_t['LogRet'])
    
    logger.info("Computing Rolling Vol & Z-Score...")
    df_t['RollVol'] = df_t['LogRet'].rolling(20).std() * np.sqrt(252)
    df_t['ZScore'] = (df_t['LogRet'] - df_t['LogRet'].rolling(60).mean()) / df_t['LogRet'].rolling(60).std()
    
    # Common layout
    layout_dark = dict(template="plotly_dark", height=700)
    
    # --- 1. SPY Price + Regime Markers ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_t.index, y=df_t['Close'], mode='lines', name=ticker, line=dict(color='cyan')))
    for bkp in t_bkps:
        fig1.add_vline(x=bkp, line_dash="dash", line_color="yellow", opacity=0.7)
    fig1.update_layout(title=f"{ticker} Price Structure & Variance Regimes (PELT)", **layout_dark)
    save_plot(fig1, "1_price_regimes.html", out_dir, auto_open=True)
    
    # --- 2. VIX Price + Regime Markers ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_v.index, y=df_v['Close'], mode='lines', name="VIX", line=dict(color='orange')))
    for bkp in v_bkps:
        fig2.add_vline(x=bkp, line_dash="dash", line_color="red", opacity=0.7)
    fig2.update_layout(title="VIX Levels & Implied Volatility Regimes", **layout_dark)
    save_plot(fig2, "2_vix_regimes.html", out_dir)
    
    # --- 3. Wavelet Energy (SPY) ---
    # Highlight high stress
    cutoff = df_t['WaveletEnergy'].quantile(0.90)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_t.index, y=df_t['WaveletEnergy'], mode='lines', name='Wavelet Energy', line=dict(color='magenta')))
    fig3.add_hline(y=cutoff, line_dash="dot", annotation_text="90th % Stress")
    fig3.update_layout(title=f"{ticker} Short-Scale Wavelet Energy (High-Freq Turbulence)", **layout_dark)
    save_plot(fig3, "3_wavelet_energy.html", out_dir)
    
    # --- 4. Welch PSD Slope ---
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_t.index, y=df_t['PSDSlope'], mode='lines', name='Spectral Slope', line=dict(color='lime')))
    fig4.add_hline(y=0, line_dash="solid", line_color="white", annotation_text="Random Walk (White Noise)")
    fig4.update_layout(title=f"{ticker} Spectral Slope (Memory Drift detector)", 
                       yaxis_title="Slope (Alpha)", 
                       annotations=[dict(x=0.5, y=-0.1, xref='paper', yref='paper', text="Lower = More Persistence/Drift", showarrow=False)],
                       **layout_dark)
    save_plot(fig4, "4_psd_slope.html", out_dir)
    
    # --- 5. Combined Regime Matrix (Heatmap) ---
    # Normalize features for heatmap
    features = df_t[['WaveletEnergy', 'PSDSlope', 'RollVol', 'ZScore']].dropna().copy()
    # MinMax Scale
    features = (features - features.min()) / (features.max() - features.min())
    
    fig5 = go.Figure(data=go.Heatmap(
        z=features.T.values,
        x=features.index,
        y=features.columns,
        colorscale='Inferno'
    ))
    fig5.update_layout(title="Multi-Factor Regime Convergence Matrix", **layout_dark)
    save_plot(fig5, "5_regime_matrix.html", out_dir)
    
    # --- 6. SPY vs VIX Comparative Timeline ---
    fig6 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"{ticker} Wavelet Energy", "VIX Wavelet Energy"))
    fig6.add_trace(go.Scatter(x=df_t.index, y=df_t['WaveletEnergy'], name=f'{ticker} Energy'), row=1, col=1)
    fig6.add_trace(go.Scatter(x=df_v.index, y=df_v['WaveletEnergy'], name='VIX Energy', line=dict(color='orange')), row=2, col=1)
    fig6.update_layout(title="Volatility Transmission: Realized vs Implied Turbulence", **layout_dark)
    save_plot(fig6, "6_comparative_timeline.html", out_dir)
    
    # --- 7. Returns Distribution Shift Panel ---
    # Rolling 60d distribution snapshot (Last 60d vs 60d from 1 year ago)
    recent = df_t['LogRet'].tail(60)
    old = df_t['LogRet'].shift(252).tail(60).dropna()
    
    fig7 = go.Figure()
    fig7.add_trace(go.Histogram(x=recent, name='Recent 60d', opacity=0.75, marker_color='cyan'))
    if not old.empty:
        fig7.add_trace(go.Histogram(x=old, name='1 Year Ago', opacity=0.5, marker_color='gray'))
    fig7.update_layout(title="Distribution Fat-Tail Check (Recent vs Past)", barmode='overlay', **layout_dark)
    save_plot(fig7, "7_distribution_shift.html", out_dir)
    
    # --- 8. Three-Detector Alignment Gauge ---
    # Normalize and Overlay
    norm_slope = (df_t['PSDSlope'] - df_t['PSDSlope'].mean()) / df_t['PSDSlope'].std()
    norm_energy = (df_t['WaveletEnergy'] - df_t['WaveletEnergy'].mean()) / df_t['WaveletEnergy'].std()
    
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=df_t.index, y=norm_energy, name='Wavelet Energy (Z)', line=dict(width=1, color='magenta')))
    fig8.add_trace(go.Scatter(x=df_t.index, y=norm_slope, name='PSD Slope (Z)', line=dict(width=1, color='lime')))
    
    # Add Regime Bands
    # Create a binary series for regime breaks
    # We just shade regions between breakpoints?
    # Let's use vertical lines for PELT
    for bkp in t_bkps:
        fig8.add_vline(x=bkp, line_width=1, line_color='yellow')
        
    fig8.update_layout(title="Detector Alignment Gauge (Z-Scores)", **layout_dark)
    save_plot(fig8, "8_alignment_gauge.html", out_dir)

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Factor Volatility Dashboard")
    parser.add_argument("--ticker", type=str, default="SPY", help="Equity Ticker")
    parser.add_argument("--years", type=int, default=2, help="Years of history")
    args = parser.parse_args()
    
    # 1. Setup Output
    today_str = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
    out_dir = os.path.join("output_multifact", f"{args.ticker}_{today_str}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Download Data
    df_t, df_v = download_data(args.ticker, args.years)
    
    if df_t.empty:
        logger.error("No data found.")
        sys.exit(1)
        
    # 3. Detect Regimes (PELT)
    logger.info("Running PELT Structural Break Detection...")
    t_bkps = detect_regimes_pelt(df_t['LogRet'])
    v_bkps = detect_regimes_pelt(df_v['LogRet'])
    logger.info(f"Detected {len(t_bkps)} variance regimes in {args.ticker}")
    
    # 4. Generate Dashboards
    generate_dashboards(df_t, df_v, t_bkps, v_bkps, args.ticker, out_dir)
    
    logger.info(f"Done. Results in {out_dir}")

if __name__ == "__main__":
    main()
