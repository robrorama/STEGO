#!/usr/bin/env python3
"""
SCRIPTNAME: early_vix_wavelet_welch_dashboard.py

Author: Michael Derby
Framework: STEGO Financial Framework

Description:
    Standalone multi-tab Plotly HTML dashboard for early volatility and VIX regime-shift detection,
    using 1-minute realized volatility, wavelet energy entropy, Welch PSD band ratios, and a
    composite early-warning signal across multiple wavelet families (Haar, db4, Coiflet).

    Features:
    - Auto-downloads 1m intraday data and Daily VIX data via yfinance.
    - Persists all raw and calculated data to CSV in /dev/shm for pipeline integration.
    - Computes Rolling Wavelet Energy Entropy (Haar, db4, coif1).
    - Computes Welch PSD Low/High Frequency Ratios.
    - specific Z-Score normalization for regime change detection.
    - Generates a static, interactive HTML dashboard.
"""

import os
import sys
import argparse
import datetime
import math
import pathlib
import webbrowser
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import pywt
from scipy import signal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION & ARGS
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="STEGO Early VIX Wavelet Dashboard")
    parser.add_argument("--ticker", type=str, required=True, help="Underlying ticker (e.g., SPY, NVDA)")
    parser.add_argument("--period", type=str, default="5d", help="Intraday download period (default: 5d)")
    parser.add_argument("--interval", type=str, default="1m", help="Intraday resolution (default: 1m)")
    parser.add_argument("--vix-ticker", type=str, default="^VIX", help="Volatility index ticker (default: ^VIX)")
    parser.add_argument("--wavelet-levels", type=int, default=6, help="DWT decomposition levels")
    parser.add_argument("--entropy-window", type=int, default=240, help="Rolling window for entropy (minutes)")
    parser.add_argument("--welch-window", type=int, default=256, help="Window length for Welch PSD")
    parser.add_argument("--output-root", type=str, default="/dev/shm/EARLY_VIX_WAVELET_WELCH", help="Base output directory")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# DATA IO & CACHING
# ---------------------------------------------------------------------------

def ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)

def get_session_dir(root: str, ticker: str) -> pathlib.Path:
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    path = pathlib.Path(root) / ticker.upper() / today_str
    ensure_dir(path)
    return path

def download_and_cache_intraday(ticker: str, period: str, interval: str, save_dir: pathlib.Path) -> pd.DataFrame:
    """
    Downloads intraday data, saves to CSV, and reads it back to ensure persistence workflow.
    """
    print(f"[INFO] Downloading {ticker} intraday ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    
    # Cleaning
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Flatten multi-index columns if present (yfinance update quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # Ensure Adj Close or Close
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            raise ValueError(f"Missing Close data for {ticker}")

    df.dropna(inplace=True)

    # Persist
    csv_path = save_dir / f"{ticker}_intraday_ohlcv_{interval}.csv"
    df.to_csv(csv_path)
    print(f"[INFO] Saved intraday data to {csv_path}")

    # Reload from disk (Strict STEGO requirement)
    df_loaded = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df_loaded

def download_and_cache_daily(ticker: str, save_dir: pathlib.Path) -> pd.DataFrame:
    """
    Downloads daily context data (VIX), saves to CSV, reads back.
    """
    print(f"[INFO] Downloading {ticker} daily context...")
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False, progress=False)
    
    if df.empty:
        print(f"[WARN] No data found for {ticker}, context plots may be empty.")
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    df.dropna(inplace=True)
    
    csv_path = save_dir / f"{ticker.replace('^', '')}_daily_ohlcv.csv"
    df.to_csv(csv_path)
    print(f"[INFO] Saved daily context data to {csv_path}")
    
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)

# ---------------------------------------------------------------------------
# CORE CALCULATIONS: RV, WAVELETS, WELCH
# ---------------------------------------------------------------------------

def compute_returns_rv(df: pd.DataFrame, ticker: str, save_dir: pathlib.Path) -> pd.DataFrame:
    """
    Computes Log Returns and 30-min Rolling Realized Volatility.
    """
    # Use Adj Close
    price = df['Adj Close']
    log_ret = np.log(price).diff().fillna(0)
    
    # Rolling RV (sum of squared log returns)
    rv_window = 30
    rv = (log_ret ** 2).rolling(window=rv_window, min_periods=5).sum()
    
    # Save Returns
    ret_df = pd.DataFrame(log_ret)
    ret_df.columns = ['log_return']
    ret_df.to_csv(save_dir / f"{ticker}_returns_1m.csv")
    
    # Save RV
    rv_df = pd.DataFrame(rv)
    rv_df.columns = ['rv_30m']
    rv_df.to_csv(save_dir / f"{ticker}_rv_rolling30m.csv")
    
    return rv_df

def calculate_wavelet_entropy_window(data_segment, wavelet='db4', level=6):
    """
    Computes Shannon entropy of wavelet energy for a single window.
    """
    if len(data_segment) < 2**level:
        return np.nan, []

    # Decompose
    coeffs = pywt.wavedec(data_segment, wavelet, level=level, mode='periodization')
    
    # Ignore approximation (index 0), keep details
    detail_coeffs = coeffs[1:]
    
    # Calculate energy per level
    energies = np.array([np.sum(c**2) for c in detail_coeffs])
    total_energy = np.sum(energies)
    
    if total_energy == 0:
        return 0.0, np.zeros_like(energies)
        
    # Probabilities
    probs = energies / total_energy
    
    # Shannon Entropy
    # Handle 0 probabilities to avoid log(0)
    p_nz = probs[probs > 0]
    entropy = -np.sum(p_nz * np.log(p_nz))
    
    return entropy, probs

def compute_rolling_wavelet_metrics(rv_series: pd.Series, ticker: str, save_dir: pathlib.Path, 
                                   window: int, levels: int):
    """
    Computes rolling entropy for Haar, db4, coif1.
    Also saves band probabilities for db4.
    """
    families = ['haar', 'db4', 'coif1']
    results = {fam: [] for fam in families}
    db4_bands = []
    timestamps = []
    
    values = rv_series.values
    idx = rv_series.index
    
    print(f"[INFO] Computing rolling wavelet entropy (w={window}, l={levels})...")
    
    # Iterate through rolling windows
    # Note: Starting from 'window' to end
    for i in range(window, len(values)):
        seg = values[i-window : i]
        ts = idx[i]
        
        # We need a small check to ensure segment isn't all NaNs
        if np.isnan(seg).all():
            for fam in families: results[fam].append(np.nan)
            db4_bands.append([np.nan]*levels)
            timestamps.append(ts)
            continue
            
        seg = np.nan_to_num(seg) # Handle NaNs in RV by zeroing them for transform
        
        # 1. Haar
        h_ent, _ = calculate_wavelet_entropy_window(seg, 'haar', levels)
        results['haar'].append(h_ent)
        
        # 2. db4
        d_ent, d_probs = calculate_wavelet_entropy_window(seg, 'db4', levels)
        results['db4'].append(d_ent)
        db4_bands.append(d_probs)
        
        # 3. coif1
        c_ent, _ = calculate_wavelet_entropy_window(seg, 'coif1', levels)
        results['coif1'].append(c_ent)
        
        timestamps.append(ts)

    # Save Entropies
    df_ent = pd.DataFrame(results, index=timestamps)
    for fam in families:
        df_ent[[fam]].to_csv(save_dir / f"{ticker}_wavelet_entropy_{fam}.csv")
        
    # Save db4 Bands
    # Columns are D_level. If level=6, we have D6, D5, D4, D3, D2, D1 (Standard pywt order usually A, D_n...D1)
    # pywt.wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # We sliced [1:], so we have [cD_n, ..., cD_1]
    # Let's label them D{levels} down to D1
    band_cols = [f"D{levels - i}" for i in range(levels)]
    df_bands = pd.DataFrame(db4_bands, index=timestamps, columns=band_cols)
    df_bands.to_csv(save_dir / f"{ticker}_wavelet_band_probabilities_db4.csv")
    
    return df_ent, df_bands

def compute_welch_ratio(rv_series: pd.Series, ticker: str, save_dir: pathlib.Path, window: int):
    """
    Computes rolling Welch PSD Low/High frequency ratio.
    Low: < 0.02 cycles/min (Periods > 50 min)
    High: 0.02 - 0.5 cycles/min (Periods 2 - 50 min)
    """
    print(f"[INFO] Computing rolling Welch PSD ratio (w={window})...")
    values = rv_series.values
    idx = rv_series.index
    
    ratios = []
    timestamps = []
    
    f_low_cut = 0.02
    
    for i in range(window, len(values)):
        seg = values[i-window : i]
        ts = idx[i]
        
        if np.isnan(seg).all():
            ratios.append(np.nan)
            timestamps.append(ts)
            continue
            
        seg = np.nan_to_num(seg)
        
        # Welch PSD
        freqs, psd = signal.welch(seg, fs=1.0, nperseg=window)
        
        # Integration indices
        # Low band: 0 to 0.02
        idx_low = (freqs > 0) & (freqs <= f_low_cut)
        # High band: 0.02 to 0.5
        idx_high = (freqs > f_low_cut)
        
        power_low = np.sum(psd[idx_low])
        power_high = np.sum(psd[idx_high])
        
        if power_high == 0:
            ratios.append(np.nan) # Avoid div/0
        else:
            ratios.append(power_low / power_high)
            
        timestamps.append(ts)
        
    df_welch = pd.DataFrame({'welch_ratio': ratios}, index=timestamps)
    df_welch.to_csv(save_dir / f"{ticker}_welch_ratio.csv")
    
    return df_welch

def compute_composite_signal(df_ent, df_welch, ticker, save_dir):
    """
    Z-score normalization and composite signal generation.
    """
    # Align DataFrames
    common_idx = df_ent.index.intersection(df_welch.index)
    
    if len(common_idx) == 0:
        return pd.DataFrame()

    df_final = pd.DataFrame(index=common_idx)
    
    # 1. Entropy db4 Z-Score (Higher entropy -> usually higher randomness/volatility transition)
    # Using expanding window for Z-score to respect causality, or a long rolling window.
    # We will use a long rolling window (e.g., 500 bars) for Z-score baseline.
    z_window = 500
    
    vals_ent = df_ent.loc[common_idx, 'db4']
    z_ent = (vals_ent - vals_ent.rolling(z_window).mean()) / vals_ent.rolling(z_window).std()
    
    # 2. Welch Ratio Z-Score
    vals_welch = df_welch.loc[common_idx, 'welch_ratio']
    z_welch = (vals_welch - vals_welch.rolling(z_window).mean()) / vals_welch.rolling(z_window).std()
    
    # 3. Haar Entropy Z-Score (Auxiliary)
    vals_haar = df_ent.loc[common_idx, 'haar']
    z_haar = (vals_haar - vals_haar.rolling(z_window).mean()) / vals_haar.rolling(z_window).std()
    
    df_final['z_entropy_db4'] = z_ent
    df_final['z_welch_ratio'] = z_welch
    df_final['z_entropy_haar'] = z_haar
    
    # Composite: Average Z-scores
    df_final['composite_z'] = df_final[['z_entropy_db4', 'z_welch_ratio', 'z_entropy_haar']].mean(axis=1)
    
    df_final.to_csv(save_dir / f"{ticker}_composite_signal.csv")
    return df_final

# ---------------------------------------------------------------------------
# VISUALIZATION: PLOTLY & HTML DASHBOARD
# ---------------------------------------------------------------------------

def generate_html_dashboard(ticker, save_dir, 
                            df_price, df_rv, df_ent, df_bands, df_welch, df_comp, df_vix):
    
    print("[INFO] Generating Plotly Dashboard...")
    
    # Common layout settings
    layout_cfg = dict(
        template="plotly_dark",
        autosize=True,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left")
    )
    
    # --- Tab 1: Price & RV ---
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                              row_heights=[0.7, 0.3], subplot_titles=(f"{ticker} Price", "Realized Volatility (30m)"))
    fig_price.add_trace(go.Scatter(x=df_price.index, y=df_price['Adj Close'], name="Price", line=dict(color='#00F0FF')), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=df_rv.index, y=df_rv['rv_30m'], name="RV 30m", line=dict(color='#FF0055')), row=2, col=1)
    fig_price.update_layout(**layout_cfg)
    html_price = pio.to_html(fig_price, full_html=False, include_plotlyjs='cdn')

    # --- Tab 2: Wavelet Entropy ---
    fig_ent = go.Figure()
    fig_ent.add_trace(go.Scatter(x=df_ent.index, y=df_ent['db4'], name="Entropy (db4)", line=dict(color='cyan', width=2)))
    fig_ent.add_trace(go.Scatter(x=df_ent.index, y=df_ent['haar'], name="Entropy (haar)", line=dict(color='yellow', width=1, dash='dot')))
    fig_ent.add_trace(go.Scatter(x=df_ent.index, y=df_ent['coif1'], name="Entropy (coif1)", line=dict(color='magenta', width=1, dash='dot')))
    fig_ent.update_layout(title="Wavelet Energy Entropy (Rolling)", yaxis_title="Shannon Entropy", **layout_cfg)
    html_ent = pio.to_html(fig_ent, full_html=False, include_plotlyjs=False)

    # --- Tab 3: Band Probabilities (db4) ---
    fig_band = go.Figure()
    # Stacked Area
    for col in df_bands.columns:
        fig_band.add_trace(go.Scatter(x=df_bands.index, y=df_bands[col], name=col, stackgroup='one', mode='none'))
    fig_band.update_layout(title="Wavelet Detail Energy Distribution (db4)", yaxis_title="Probability", **layout_cfg)
    html_band = pio.to_html(fig_band, full_html=False, include_plotlyjs=False)

    # --- Tab 4: Welch Ratio ---
    fig_welch = go.Figure()
    fig_welch.add_trace(go.Scatter(x=df_welch.index, y=df_welch['welch_ratio'], name="Low/High Ratio", line=dict(color='#00FF00')))
    fig_welch.update_layout(title="Welch PSD Ratio (Low Band / High Band)", yaxis_title="Power Ratio", **layout_cfg)
    html_welch = pio.to_html(fig_welch, full_html=False, include_plotlyjs=False)

    # --- Tab 5: Composite Signal ---
    fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7])
    fig_comp.add_trace(go.Scatter(x=df_comp.index, y=df_comp['z_entropy_db4'], name="Z-Entropy", line=dict(color='gray', width=1)), row=1, col=1)
    fig_comp.add_trace(go.Scatter(x=df_comp.index, y=df_comp['z_welch_ratio'], name="Z-Welch", line=dict(color='gray', width=1)), row=1, col=1)
    
    # Color composite based on threshold
    c_vals = df_comp['composite_z']
    fig_comp.add_trace(go.Scatter(x=df_comp.index, y=c_vals, name="Composite Z", line=dict(color='white', width=2)), row=2, col=1)
    
    # Add threshold lines
    fig_comp.add_hline(y=1.5, line_dash="dash", line_color="red", row=2, col=1)
    fig_comp.add_hline(y=-1.5, line_dash="dash", line_color="green", row=2, col=1)
    
    fig_comp.update_layout(title="Composite Early Warning Signal (Z-Scores)", **layout_cfg)
    html_comp = pio.to_html(fig_comp, full_html=False, include_plotlyjs=False)

    # --- Tab 6: VIX Context ---
    # Daily context plot
    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True)
    if not df_vix.empty:
        fig_vix.add_trace(go.Scatter(x=df_vix.index, y=df_vix['Adj Close'], name="VIX Daily", line=dict(color='orange')), row=1, col=1)
    
    # Resample composite to daily for comparison? Or just plot all intraday points
    # Plotting intraday composite against daily VIX is tricky on same x-axis if zoom differs.
    # We just plot the Intraday Composite on row 2, user can zoom to see alignment.
    fig_vix.add_trace(go.Scatter(x=df_comp.index, y=df_comp['composite_z'], name="Intraday Composite", line=dict(color='white')), row=2, col=1)
    fig_vix.update_layout(title="VIX Daily Context vs Intraday Signal", **layout_cfg)
    html_vix = pio.to_html(fig_vix, full_html=False, include_plotlyjs=False)

    # --- Construct Full HTML ---
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>STEGO Early VIX: {ticker}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background-color: #111; color: #eee; margin: 0; }}
        .tab {{ overflow: hidden; border-bottom: 1px solid #333; background-color: #222; }}
        .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888; font-size: 16px; }}
        .tab button:hover {{ background-color: #333; color: #fff; }}
        .tab button.active {{ background-color: #444; color: #00F0FF; border-bottom: 2px solid #00F0FF; }}
        .tabcontent {{ display: none; padding: 20px; border-top: none; animation: fadeEffect 1s; }}
        @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
        h1 {{ padding: 20px; margin: 0; font-weight: 300; }}
        .meta {{ font-size: 0.8em; color: #666; padding-left: 20px; }}
    </style>
    </head>
    <body>

    <h1>STEGO <span style="color:#00F0FF">Early VIX</span> Dashboard: {ticker}</h1>
    <div class="meta">Generated: {datetime.datetime.now()} | Path: {save_dir}</div>

    <div class="tab">
      <button class="tablinks" onclick="openTab(event, 'PriceRV')" id="defaultOpen">Price & RV</button>
      <button class="tablinks" onclick="openTab(event, 'Entropy')">Wavelet Entropy</button>
      <button class="tablinks" onclick="openTab(event, 'Bands')">Wavelet Bands (db4)</button>
      <button class="tablinks" onclick="openTab(event, 'Welch')">Welch PSD Ratio</button>
      <button class="tablinks" onclick="openTab(event, 'Composite')">Composite Signal</button>
      <button class="tablinks" onclick="openTab(event, 'VIX')">VIX Context</button>
    </div>

    <div id="PriceRV" class="tabcontent">{html_price}</div>
    <div id="Entropy" class="tabcontent">{html_ent}</div>
    <div id="Bands" class="tabcontent">{html_band}</div>
    <div id="Welch" class="tabcontent">{html_welch}</div>
    <div id="Composite" class="tabcontent">{html_comp}</div>
    <div id="VIX" class="tabcontent">{html_vix}</div>

    <script>
    function openTab(evt, tabName) {{
      var i, tabcontent, tablinks;
      tabcontent = document.getElementsByClassName("tabcontent");
      for (i = 0; i < tabcontent.length; i++) {{
        tabcontent[i].style.display = "none";
      }}
      tablinks = document.getElementsByClassName("tablinks");
      for (i = 0; i < tablinks.length; i++) {{
        tablinks[i].className = tablinks[i].className.replace(" active", "");
      }}
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }}
    document.getElementById("defaultOpen").click();
    </script>
    
    </body>
    </html>
    """
    
    output_html = save_dir / "dashboard.html"
    with open(output_html, "w", encoding='utf-8') as f:
        f.write(html_template)
        
    print(f"[SUCCESS] Dashboard written to {output_html}")
    return output_html

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    
    # 1. Setup Session Directory
    save_dir = get_session_dir(args.output_root, args.ticker)
    print(f"[INFO] STEGO Early VIX Framework initializing for {args.ticker}...")
    print(f"[INFO] Output Directory: {save_dir}")
    
    # 2. Data Retrieval (Download -> Cache -> Load)
    try:
        df_intraday = download_and_cache_intraday(args.ticker, args.period, args.interval, save_dir)
        df_vix = download_and_cache_daily(args.vix_ticker, save_dir)
    except Exception as e:
        print(f"[ERROR] Data retrieval failed: {e}")
        sys.exit(1)
        
    # 3. Core Calculations
    # Returns & RV
    df_rv = compute_returns_rv(df_intraday, args.ticker, save_dir)
    
    # Wavelet Entropy & Bands
    df_ent, df_bands = compute_rolling_wavelet_metrics(
        df_rv['rv_30m'], args.ticker, save_dir, 
        window=args.entropy_window, levels=args.wavelet_levels
    )
    
    # Welch Ratio
    df_welch = compute_welch_ratio(
        df_rv['rv_30m'], args.ticker, save_dir, 
        window=args.welch_window
    )
    
    # Composite Signal
    df_comp = compute_composite_signal(df_ent, df_welch, args.ticker, save_dir)
    
    # 4. Generate Dashboard
    if df_comp.empty:
        print("[WARN] Not enough data generated for composite signal. Dashboard may be incomplete.")
        
    html_path = generate_html_dashboard(
        args.ticker, save_dir,
        df_intraday, df_rv, df_ent, df_bands, df_welch, df_comp, df_vix
    )
    
    # 5. Launch
    print(f"[INFO] Opening {html_path} in browser...")
    webbrowser.open(f"file://{html_path}")

if __name__ == "__main__":
    main()
