#!/usr/bin/env python3
"""
vix_wavelet_psd_changepoint_dashboard.py

HIGH-LEVEL PURPOSE:
This script builds a professional early-volatility-signal dashboard for any market series 
(default: VIX, but usable for any equity/ETF/index).

Core goals:
1. Detect “pre-spike” behavior and early-warning conditions in volatility.
2. Combine:
   - Continuous Wavelet Transform (CWT) short-scale energy (Custom robust implementation),
   - Welch Power Spectral Density (PSD) band ratios,
   - A composite early-warning signal,
   - Changepoint detection on this composite signal.
3. Produce a multi-tab Plotly HTML dashboard for interactive analysis.
4. Serialize all intermediate metrics to CSV so they can be reused in other pipelines.

USAGE:
    python vix_wavelet_psd_changepoint_dashboard.py --ticker ^VIX --period 2y
"""

import os
import sys
import argparse
import logging
import pathlib
import datetime
import webbrowser
from typing import Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import zscore
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ruptures as rpt

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = "/dev/shm/VIX_WAVELET_PSD_CP"

# ---------------------------------------------------------------------------
# ROBUST WAVELET IMPLEMENTATION (Version Independent)
# ---------------------------------------------------------------------------
def morlet_kernel(M: int, s: float, w: float = 5.0) -> np.ndarray:
    """
    Generate a complex Morlet wavelet kernel.
    M: length of the window
    s: scaling factor
    w: omega0 (frequency parameter)
    """
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    # Complex Morlet formula
    # psi(t) = pi^(-0.25) * exp(i*w*x) * exp(-0.5*x^2)
    # Note: We include normalization for energy conservation consistency
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * (np.pi**(-0.25))
    return wavelet

def robust_cwt(data: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Manual implementation of CWT using fftconvolve.
    Replaces scipy.signal.cwt to avoid version/attribute errors.
    """
    output = np.zeros((len(scales), len(data)), dtype=complex)
    
    for i, s in enumerate(scales):
        # Determine window size. Scipy convention is often min(10*s, len(data))
        # We use a window large enough to capture the wavelet decay.
        M = int(min(10 * s, len(data)))
        if M % 2 == 0: M += 1 # Ensure odd length for centering
        
        # Generate wavelet
        wavelet = morlet_kernel(M, s)
        
        # Convolve
        # 'same' returns output of length max(M, N). Boundary effects exist.
        output[i, :] = signal.fftconvolve(data, wavelet, mode='same')
        
    return output

# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------
def load_data(ticker: str, period: str, interval: str, output_dir: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Downloads data from yfinance, computes log returns, drops NaNs.
    Saves 'prices.csv'.
    """
    logger.info(f"Downloading data for {ticker} (Period: {period}, Interval: {interval})...")
    
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:
        logger.error(f"Failed to download data via yfinance: {e}")
        sys.exit(1)

    if df.empty:
        logger.error("Downloaded DataFrame is empty. Exiting.")
        sys.exit(1)

    # Standardize columns (handle MultiIndex if present from recent yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # Drop ticker level

    # Ensure we have Close
    if 'Close' not in df.columns:
        # Fallback for some assets
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            logger.error(f"No 'Close' column found in data. Columns: {df.columns}")
            sys.exit(1)

    # Compute Log Returns
    # log_return = ln(P_t / P_{t-1})
    df['LogReturn'] = np.log(df['Close']).diff()
    
    # Drop initial NaNs created by diff
    df.dropna(subset=['LogReturn'], inplace=True)
    
    # Save raw data
    csv_path = output_dir / "prices.csv"
    df.to_csv(csv_path)
    logger.info(f"Saved prices to {csv_path}")

    return df, df['LogReturn']

# ---------------------------------------------------------------------------
# WAVELET ANALYSIS
# ---------------------------------------------------------------------------
def compute_cwt_short_energy(
    log_returns: pd.Series, 
    output_dir: pathlib.Path
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Performs CWT using custom Morlet implementation. Computes short-scale energy.
    Saves 'cwt_scalogram.csv' and 'cwt_short_energy.csv'.
    """
    logger.info("Computing Continuous Wavelet Transform (CWT)...")
    
    data = log_returns.values
    # Scales: 1 to 10 (Short term focus)
    scales = np.arange(1, 11)
    
    # Use robust local CWT implementation
    cwt_matrix_complex = robust_cwt(data, scales)
    cwt_matrix = np.abs(cwt_matrix_complex)
    
    # Create DataFrame for Scalogram
    # Columns: scale_1, scale_2, ...
    cwt_df = pd.DataFrame(
        cwt_matrix.T, 
        index=log_returns.index, 
        columns=[f"scale_{s}" for s in scales]
    )
    cwt_df.to_csv(output_dir / "cwt_scalogram.csv")
    
    # Compute Short-Scale Energy
    # Sum of squared magnitudes of the first few scales (e.g., 1-4)
    short_scales_idx = slice(0, 4) 
    energy_values = np.sum(cwt_matrix[short_scales_idx, :] ** 2, axis=0)
    
    short_energy_series = pd.Series(energy_values, index=log_returns.index, name="ShortScaleEnergy")
    
    se_df = short_energy_series.to_frame()
    se_df.to_csv(output_dir / "cwt_short_energy.csv")
    
    return cwt_df, short_energy_series

# ---------------------------------------------------------------------------
# WELCH PSD ANALYSIS
# ---------------------------------------------------------------------------
def compute_welch_psd_ratio(
    log_returns: pd.Series, 
    output_dir: pathlib.Path
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Computes Welch PSD ratio (High Freq / Low Freq).
    Uses Spectrogram to create a time-varying estimate of the PSD.
    Saves 'welch_psd.csv' (average) and 'welch_ratio.csv'.
    """
    logger.info("Computing Welch PSD and Band Ratios...")
    
    fs = 1.0  # Normalized frequency (1 sample per time step)
    data = log_returns.values
    
    # 1. Global Welch (for reference snapshot)
    freqs, psd_global = signal.welch(data, fs=fs, nperseg=min(len(data), 256))
    psd_df = pd.DataFrame({'frequency': freqs, 'psd': psd_global})
    psd_df.to_csv(output_dir / "welch_psd.csv", index=False)
    
    # 2. Time-Varying PSD via Spectrogram to get a rolling ratio
    nperseg = 64
    if len(data) < nperseg:
        nperseg = max(len(data) // 2, 4)
    
    if nperseg < 4:
        # Fallback for extremely short series
        logger.warning("Series too short for spectrogram. Using static ratio.")
        mask_low = freqs < 0.05
        mask_high = freqs > 0.20
        p_low = np.sum(psd_global[mask_low])
        p_high = np.sum(psd_global[mask_high])
        ratio = p_high / p_low if p_low > 0 else 0
        ratio_series = pd.Series(ratio, index=log_returns.index, name="WelchRatio")
        ratio_df = ratio_series.to_frame()
        ratio_df.to_csv(output_dir / "welch_ratio.csv")
        return ratio_series, psd_df, ratio_df

    f_spec, t_spec, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=nperseg-1)
    
    # Calculate Band Powers for each time slice
    # Bands: Low < 0.05, High > 0.20
    mask_low = f_spec < 0.05
    mask_high = f_spec > 0.20
    
    ratios = []
    # Iterate columns (time steps)
    for i in range(Sxx.shape[1]):
        spectrum = Sxx[:, i]
        p_low = np.sum(spectrum[mask_low])
        p_high = np.sum(spectrum[mask_high])
        
        if p_low == 0:
            ratios.append(0.0)
        else:
            ratios.append(p_high / p_low)
            
    # Align spectrogram indices to original DataFrame index
    valid_indices = np.floor(t_spec).astype(int)
    
    # Handle potentially out-of-bounds indices if spectrogram estimation pads
    valid_indices = np.clip(valid_indices, 0, len(log_returns) - 1)

    temp_series = pd.Series(ratios, index=log_returns.index[valid_indices])
    
    # Reindex to full timeframe, ffill/bfill to cover edges
    ratio_series = temp_series.reindex(log_returns.index).ffill().bfill()
    ratio_series.name = "WelchRatio"
    
    ratio_df = ratio_series.to_frame()
    ratio_df.to_csv(output_dir / "welch_ratio.csv")
    
    return ratio_series, psd_df, ratio_df

# ---------------------------------------------------------------------------
# COMPOSITE SIGNAL BUILDER
# ---------------------------------------------------------------------------
def build_composite_signal(
    short_energy: pd.Series, 
    welch_ratio: pd.Series, 
    output_dir: pathlib.Path
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs CompositeZ signal:
    Z(ShortEnergy) + Z(WelchRatio).
    Computes rolling 95th percentile threshold.
    Flags pre-spike conditions.
    """
    logger.info("Building Composite Early-Warning Signal...")
    
    # Align
    df = pd.DataFrame({
        'ShortEnergy': short_energy,
        'WelchRatio': welch_ratio
    }).dropna()
    
    # Z-Score
    df['Z_SSE'] = zscore(df['ShortEnergy'])
    df['Z_PSD'] = zscore(df['WelchRatio'])
    
    # Composite
    df['CompositeZ'] = df['Z_SSE'] + df['Z_PSD']
    
    # Rolling Threshold (Window = 50)
    window_size = 50
    df['RollingQ95'] = df['CompositeZ'].rolling(window=window_size, min_periods=10).quantile(0.95)
    
    # Flag
    df['FlagPreSpike'] = df['CompositeZ'] > df['RollingQ95']
    
    # Save
    df.to_csv(output_dir / "composite_signal.csv")
    df[['RollingQ95']].to_csv(output_dir / "thresholds.csv")
    
    return df, df['CompositeZ']

# ---------------------------------------------------------------------------
# CHANGEPOINT DETECTION
# ---------------------------------------------------------------------------
def detect_changepoints(
    composite_z: pd.Series, 
    output_dir: pathlib.Path
) -> pd.DataFrame:
    """
    Uses Ruptures (PELT) to detect regime changes in the CompositeZ signal.
    """
    logger.info("Detecting Changepoints using PELT...")
    
    signal_values = composite_z.dropna().values
    dates = composite_z.dropna().index
    
    if len(signal_values) < 10:
        logger.warning("Signal too short for changepoint detection.")
        return pd.DataFrame(columns=['index', 'date'])

    # PELT with RBF kernel
    model = rpt.Pelt(model="rbf").fit(signal_values)
    breakpoints = model.predict(pen=3)
    
    # Convert to dates
    cp_data = []
    for bp in breakpoints:
        idx = bp - 1 # 0-based index
        if 0 <= idx < len(dates):
            date_val = dates[idx]
            cp_data.append({'index': idx, 'date': date_val})
            
    cp_df = pd.DataFrame(cp_data)
    cp_df.to_csv(output_dir / "changepoints.csv", index=False)
    
    return cp_df

# ---------------------------------------------------------------------------
# DASHBOARD GENERATION
# ---------------------------------------------------------------------------
def build_and_save_dashboard(
    output_dir: pathlib.Path,
    df_prices: pd.DataFrame,
    cwt_df: pd.DataFrame,
    short_energy: pd.Series,
    psd_df: pd.DataFrame,
    welch_ratio: pd.Series,
    df_composite: pd.DataFrame,
    cp_df: pd.DataFrame
):
    """
    Generates a Plotly HTML dashboard with interactive visibility toggles (Tabs).
    """
    logger.info("Generating Plotly Dashboard...")
    
    # Create Figure
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=("Primary Analysis View", "Secondary/Context View")
    )
    
    # --- PREPARE TRACES ---
    
    # GROUP 1: Price & Returns
    trace_price = go.Scatter(x=df_prices.index, y=df_prices['Close'], name='Close Price', line=dict(color='#00F0FF'))
    trace_logret = go.Scatter(x=df_prices.index, y=df_prices['LogReturn'], name='Log Returns', line=dict(color='gray', width=1))
    
    # GROUP 2: CWT
    # Heatmap needs Transpose
    trace_heatmap = go.Heatmap(
        z=cwt_df.T.values,
        x=cwt_df.index,
        y=cwt_df.columns,
        colorscale='Viridis',
        name='CWT Scalogram',
        showscale=True
    )
    trace_energy = go.Scatter(x=short_energy.index, y=short_energy.values, name='Short Scale Energy', line=dict(color='#FF8C00'))
    
    # GROUP 3: Welch PSD
    trace_wratio = go.Scatter(x=welch_ratio.index, y=welch_ratio.values, name='Welch Ratio (High/Low)', line=dict(color='#BD00FF'))
    
    # GROUP 4: Composite
    trace_comp = go.Scatter(x=df_composite.index, y=df_composite['CompositeZ'], name='Composite Z', line=dict(color='white', width=1.5))
    trace_thresh = go.Scatter(x=df_composite.index, y=df_composite['RollingQ95'], name='95% Threshold', line=dict(color='red', dash='dash'))
    
    anom_dates = df_composite[df_composite['FlagPreSpike']].index
    anom_vals = df_composite[df_composite['FlagPreSpike']]['CompositeZ']
    trace_flags = go.Scatter(
        x=anom_dates, y=anom_vals, mode='markers', 
        name='Pre-Spike Signal', marker=dict(color='#FF0000', size=8, symbol='x')
    )
    
    # GROUP 5: Regime/Changepoints
    cp_dates = pd.to_datetime(cp_df['date']) if not cp_df.empty else []
    
    x_cp_lines = []
    y_cp_lines = []
    if not df_composite.empty:
        y_min, y_max = df_composite['CompositeZ'].min(), df_composite['CompositeZ'].max()
        for d in cp_dates:
            x_cp_lines.extend([d, d, None])
            y_cp_lines.extend([y_min, y_max, None])
        
    trace_cp = go.Scatter(
        x=x_cp_lines, y=y_cp_lines, mode='lines', 
        name='Regime Change (PELT)', line=dict(color='#00FF00', width=2, dash='dot')
    )

    # --- ADD TRACES ---
    # Index 0: Price
    fig.add_trace(trace_price, row=1, col=1)
    # Index 1: Returns
    fig.add_trace(trace_logret, row=2, col=1)
    
    # Index 2: Heatmap (CWT)
    fig.add_trace(trace_heatmap, row=2, col=1)
    # Index 3: Energy
    fig.add_trace(trace_energy, row=1, col=1)
    
    # Index 4: Welch Ratio
    fig.add_trace(trace_wratio, row=2, col=1)
    # Index 5: Price Copy (for Welch context)
    fig.add_trace(trace_price, row=1, col=1) 
    
    # Index 6: Composite
    fig.add_trace(trace_comp, row=1, col=1)
    # Index 7: Thresh
    fig.add_trace(trace_thresh, row=1, col=1)
    # Index 8: Flags
    fig.add_trace(trace_flags, row=1, col=1)
    
    # Index 9: Composite Copy (Regime)
    fig.add_trace(trace_comp, row=1, col=1) 
    # Index 10: Changepoints
    fig.add_trace(trace_cp, row=1, col=1)

    # --- UPDATE MENUS (TABS) ---
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            x=0.5,
            y=1.15,
            xanchor='center',
            yanchor='top',
            bgcolor="#222",
            bordercolor="#444",
            font=dict(color="white"),
            buttons=[
                dict(
                    label="1. Price/Returns",
                    method="update",
                    args=[{"visible": [True, True, False, False, False, False, False, False, False, False, False]},
                          {"title": "Price Action & Log Returns"}]
                ),
                dict(
                    label="2. CWT Analysis",
                    method="update",
                    args=[{"visible": [False, False, True, True, False, False, False, False, False, False, False]},
                          {"title": "Wavelet Short-Scale Energy (Top) & Scalogram (Bottom)"}]
                ),
                dict(
                    label="3. Welch PSD Ratio",
                    method="update",
                    args=[{"visible": [False, False, False, False, True, True, False, False, False, False, False]},
                          {"title": "Price (Top) & Welch High/Low Frequency Ratio (Bottom)"}]
                ),
                dict(
                    label="4. Composite Signal",
                    method="update",
                    args=[{"visible": [False, False, False, False, False, False, True, True, True, False, False]},
                          {"title": "Composite Early-Warning Signal (Z-Score Sum)"}]
                ),
                dict(
                    label="5. Regime Changes",
                    method="update",
                    args=[{"visible": [False, False, False, False, False, False, False, False, False, True, True]},
                          {"title": "Composite Signal with Detected Changepoints (PELT)"}]
                ),
            ],
        )
    ]
    
    fig.update_layout(
        updatemenus=updatemenus,
        title_text=f"Volatility Signal Dashboard",
        template="plotly_dark",
        height=800,
        hovermode="x unified"
    )
    
    # Default visibility
    for i in range(len(fig.data)):
        fig.data[i].visible = (i in [0, 1])

    output_path = output_dir / "dashboard.html"
    fig.write_html(str(output_path))
    logger.info(f"Dashboard saved to {output_path}")
    
    try:
        webbrowser.open(f"file://{output_path.resolve()}")
    except:
        pass

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VIX Wavelet PSD Changepoint Dashboard")
    parser.add_argument("--ticker", default="^VIX", help="Ticker symbol (default: ^VIX)")
    parser.add_argument("--period", default="2y", help="Data period (default: 2y)")
    parser.add_argument("--interval", default="1d", help="Data interval (default: 1d)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="Base output directory")
    
    args = parser.parse_args()
    
    # Setup Paths
    base_dir = pathlib.Path(args.output_dir)
    clean_ticker = args.ticker.replace("^", "").upper()
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    
    run_dir = base_dir / clean_ticker / date_str
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting analysis for {args.ticker} -> {run_dir}")
    
    # 1. Load Data
    df_prices, log_returns = load_data(args.ticker, args.period, args.interval, run_dir)
    
    # 2. Wavelet Analysis (Robust CWT)
    cwt_df, short_energy = compute_cwt_short_energy(log_returns, run_dir)
    
    # 3. Welch PSD
    welch_ratio, psd_df, ratio_df = compute_welch_psd_ratio(log_returns, run_dir)
    
    # 4. Composite Signal
    df_composite, composite_z = build_composite_signal(short_energy, welch_ratio, run_dir)
    
    # 5. Changepoints
    cp_df = detect_changepoints(composite_z, run_dir)
    
    # 6. Dashboard
    build_and_save_dashboard(
        run_dir, 
        df_prices, 
        cwt_df, 
        short_energy, 
        psd_df, 
        welch_ratio, 
        df_composite, 
        cp_df
    )
    
    logger.info("Analysis Complete.")

if __name__ == "__main__":
    main()
