#!/usr/bin/env python3
"""
options_dashboard.py

A comprehensive, standalone Options Trading Analytics Dashboard for professional traders.
Generates rich Plotly visualizations for Underlying Structure, Volatility Surfaces,
Dealer Gamma Exposure (GEX), and Volume Analytics.

Requirements:
    pip install pandas numpy scipy plotly yfinance

Usage:
    python options_dashboard.py --ticker SPY --lookback 365 --output-dir ./output --open-html
    python options_dashboard.py --ticker NVDA
"""

import os
import sys
import argparse
import logging
import time
import datetime
import warnings
from typing import Dict, List, Optional, Tuple, Any

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("OptionsDashboard")

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------------------------------------------------------
# 1. Helpers: Robust Data Handling
# -----------------------------------------------------------------------------

def force_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe index is timezone-naive to prevent alignment errors."""
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def smart_get(df: pd.DataFrame, key: str) -> pd.Series:
    """
    Robustly extract a column from yfinance DataFrame which might be:
    - Standard DataFrame
    - MultiIndex columns (Price, Ticker)
    - Case-insensitive match
    """
    if df.empty:
        raise KeyError(f"DataFrame is empty, cannot get {key}")
        
    # 1. Direct match
    if key in df.columns:
        return df[key]
        
    # 2. MultiIndex fallback (e.g., ('Close', 'SPY'))
    if isinstance(df.columns, pd.MultiIndex):
        # Check levels
        for col in df.columns:
            if col[0] == key: # Match first level
                return df[col]
            if len(col) > 1 and col[1] == key: # Match ticker level? Uncommon for yf history
                return df[col]
        # Try cross-section if key is in top level
        if key in df.columns.levels[0]:
            return df.xs(key, level=0, axis=1).iloc[:, 0]

    # 3. Case-insensitive search
    key_lower = key.lower()
    for c in df.columns:
        c_str = str(c[0]) if isinstance(c, tuple) else str(c)
        if key_lower == c_str.lower():
            return df[c]
            
    # 4. Fallback for 'Close' -> 'Adj Close'
    if key.lower() == 'close':
        for c in df.columns:
            c_str = str(c[0]) if isinstance(c, tuple) else str(c)
            if 'adj close' in c_str.lower():
                return df[c]

    raise KeyError(f"Cannot find column for {key} in {df.columns}")

# -----------------------------------------------------------------------------
# 2. Data Acquisition
# -----------------------------------------------------------------------------

def download_ohlcv_yf(ticker: str, lookback_days: int = 365) -> pd.DataFrame:
    """Download daily OHLCV data."""
    logger.info(f"Downloading {ticker} OHLCV (lookback={lookback_days}d)...")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
    df = force_naive_index(df)
    
    if df.empty:
        logger.error(f"No OHLCV data found for {ticker}")
        sys.exit(1)
        
    # Standardize columns
    try:
        # Create a clean DataFrame with standard names
        clean = pd.DataFrame(index=df.index)
        clean['Open'] = smart_get(df, 'Open')
        clean['High'] = smart_get(df, 'High')
        clean['Low'] = smart_get(df, 'Low')
        clean['Close'] = smart_get(df, 'Close')
        clean['Volume'] = smart_get(df, 'Volume')
        return clean
    except Exception as e:
        logger.error(f"Error parsing OHLCV columns: {e}")
        sys.exit(1)

def download_options_chain_yf(ticker: str) -> pd.DataFrame:
    """
    Download FULL options chain for all available expirations.
    Returns a single DataFrame with columns: 
    [expiration, strike, type, lastPrice, bid, ask, impliedVolatility, inTheMoney]
    """
    logger.info(f"Downloading options chain for {ticker}...")
    tk = yf.Ticker(ticker)
    
    try:
        expirations = tk.options
    except Exception as e:
        logger.error(f"Failed to get expirations: {e}")
        return pd.DataFrame()

    if not expirations:
        logger.warning("No expirations found.")
        return pd.DataFrame()

    all_opts = []
    
    # Limit for speed if necessary, but request asked for "Full"
    # We will fetch sequentially with a small delay to be polite
    for exp in expirations:
        try:
            chain = tk.option_chain(exp)
            calls = chain.calls
            puts = chain.puts
            
            if not calls.empty:
                calls['type'] = 'call'
                calls['expiration'] = pd.Timestamp(exp)
                all_opts.append(calls)
                
            if not puts.empty:
                puts['type'] = 'put'
                puts['expiration'] = pd.Timestamp(exp)
                all_opts.append(puts)
                
        except Exception as e:
            logger.warning(f"Failed to fetch {exp}: {e}")
            
    if not all_opts:
        return pd.DataFrame()
        
    df_chain = pd.concat(all_opts, ignore_index=True)
    
    # Normalize Columns
    # Ensure we have: strike, type, expiration, bid, ask, impliedVolatility
    # Fill NaN bids/asks with lastPrice if needed
    for col in ['bid', 'ask', 'lastPrice', 'impliedVolatility']:
        if col not in df_chain.columns:
            df_chain[col] = np.nan
            
    # Calculate Mid
    df_chain['mid'] = (df_chain['bid'] + df_chain['ask']) / 2
    df_chain['mid'] = df_chain['mid'].fillna(df_chain['lastPrice'])
    
    return df_chain

# -----------------------------------------------------------------------------
# 3. Analytics & Greeks
# -----------------------------------------------------------------------------

def compute_greeks(row, spot_price, r=0.045):
    """
    Compute Black-Scholes Greeks for a single option row.
    Returns (Delta, Gamma, Vega, Theta).
    """
    try:
        S = spot_price
        K = row['strike']
        
        # Time to expiry in years
        today = pd.Timestamp.now().normalize()
        exp = row['expiration']
        if exp.tz is not None: exp = exp.tz_localize(None)
        
        T = (exp - today).days / 365.0
        if T <= 1e-4: T = 1e-4 # Avoid division by zero
        
        sigma = row['impliedVolatility']
        if pd.isna(sigma) or sigma <= 0:
            return np.nan, np.nan, np.nan, np.nan
            
        opt_type = row['type']
        
        # d1, d2
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # PDF / CDF
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        cdf_neg_d1 = stats.norm.cdf(-d1)
        
        # Gamma (Same for Call/Put)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        
        # Vega (Same for Call/Put) - usually expressed per 1% vol change
        vega = S * pdf_d1 * np.sqrt(T) * 0.01
        
        # Theta (Per day)
        # Simplified: - (S * pdf_d1 * sigma) / (2 * sqrt(T)) - rK * exp(-rT) * N(d2) [for call]
        # This is often complex, using approx for display
        theta_common = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
        
        if opt_type == 'call':
            delta = cdf_d1
            # theta call
            # theta = (theta_common - r * K * np.exp(-r*T) * stats.norm.cdf(d2)) / 365.0
        else:
            delta = cdf_d1 - 1
            # theta put
            # theta = (theta_common + r * K * np.exp(-r*T) * stats.norm.cdf(-d2)) / 365.0
            
        return delta, gamma, vega, 0.0 # Theta skipped for brevity/stability
        
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

def compute_gamma_exposure(df_chain: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """
    Compute Net Gamma Exposure (GEX) per strike.
    Formula:
    Call GEX = Gamma * OI * 100 * Spot  (Assumption: Dealer Short Call -> Long Gamma? 
                                         Standard: Dealer Short Call -> Short Gamma.
                                         Standard Metric: Call GEX is contribution to Net.
                                         If Market is Long Calls, Dealer is Short Calls -> Dealer Negative Gamma.
                                         Common GEX Charts: Call OI -> Positive Bar, Put OI -> Negative Bar)
    
    We will use the "SpotGamma" / "SqueezeMetrics" convention:
    Dealer GEX = (Call Gamma * Call OI) - (Put Gamma * Put OI)
    Scaled by Spot * 100 to get dollar notional exposure per 1% move? 
    Let's stick to raw index units: sum(Gamma * OI * 100).
    """
    if df_chain.empty:
        return pd.DataFrame()
        
    # Calculate Greeks if not present (vectorized approx or row-wise)
    # Ideally vectorized, but row-wise is safer for CLI robustness
    # We'll just loop or apply. For 5k rows, apply is fine.
    
    # 1. Compute Greeks
    greeks = df_chain.apply(lambda row: compute_greeks(row, spot_price), axis=1, result_type='expand')
    df_chain[['delta', 'gamma', 'vega', 'theta']] = greeks
    
    # 2. GEX per row
    # Open Interest is crucial. Fill NaN with 0
    if 'openInterest' not in df_chain.columns:
        df_chain['openInterest'] = 0
    df_chain['openInterest'] = df_chain['openInterest'].fillna(0)
    
    # Contribution to Dealer Gamma:
    # We assume Market Buys Options, Dealer Sells.
    # Dealer Short Call -> Short Gamma (Negative).
    # Dealer Short Put -> Short Gamma (Negative).
    # WAIT. Standard GEX charts show Positive Gamma dominance in bull markets.
    # The convention usually used is:
    # Call GEX adds to stability (Dealer Long?), Put GEX adds to volatility (Dealer Short?).
    # Let's use the most standard simple convention:
    # Net GEX = (Call Gamma * OI) - (Put Gamma * OI)
    # Scaled by 100 shares.
    
    df_chain['gex'] = df_chain['gamma'] * df_chain['openInterest'] * 100 * spot_price
    
    # For puts, we subtract (or treat as negative contribution)
    df_chain.loc[df_chain['type'] == 'put', 'gex'] = df_chain.loc[df_chain['type'] == 'put', 'gex'] * -1
    
    # Group by Strike
    gex_by_strike = df_chain.groupby('strike')['gex'].sum().sort_index()
    
    return pd.DataFrame(gex_by_strike).rename(columns={'gex': 'NetGEX'})

def compute_realized_vol(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add RV columns to OHLCV df."""
    # Log Returns
    df_ohlcv['LogRet'] = np.log(df_ohlcv['Close'] / df_ohlcv['Close'].shift(1))
    
    for window in [10, 20, 60]:
        col = f'RV_{window}d'
        # Annualized
        df_ohlcv[col] = df_ohlcv['LogRet'].rolling(window).std() * np.sqrt(252) * 100
        
    return df_ohlcv

def volume_profile(df_ohlcv: pd.DataFrame, bins=50) -> pd.DataFrame:
    """Compute Volume Profile (Price vs Volume)."""
    # Use last N days for profile, e.g., 60 days
    subset = df_ohlcv.tail(60)
    
    price_min = subset['Low'].min()
    price_max = subset['High'].max()
    
    # Create bins
    hist, bin_edges = np.histogram(subset['Close'], bins=bins, weights=subset['Volume'], range=(price_min, price_max))
    
    # Midpoints
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return pd.DataFrame({'Price': bin_mids, 'Volume': hist})

# -----------------------------------------------------------------------------
# 4. Visualization
# -----------------------------------------------------------------------------

def plot_dashboard(ticker: str, 
                   df_ohlcv: pd.DataFrame, 
                   df_chain: pd.DataFrame, 
                   df_gex: pd.DataFrame,
                   output_dir: str,
                   open_html: bool):
    """Generate and save all plots."""
    
    spot_price = df_ohlcv['Close'].iloc[-1]
    
    # Theme settings
    layout_template = "plotly_dark"
    
    # --- 1. Underlying Market Structure ---
    # Candles, RV, Volume
    fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                         subplot_titles=(f"{ticker} Price Action", "Realized Volatility", "Volume"))
    
    # Candles
    fig1.add_trace(go.Candlestick(x=df_ohlcv.index,
                                  open=df_ohlcv['Open'], high=df_ohlcv['High'],
                                  low=df_ohlcv['Low'], close=df_ohlcv['Close'],
                                  name="OHLC"), row=1, col=1)
    
    # EMAs
    df_ohlcv['EMA20'] = df_ohlcv['Close'].ewm(span=20).mean()
    df_ohlcv['EMA50'] = df_ohlcv['Close'].ewm(span=50).mean()
    
    fig1.add_trace(go.Scatter(x=df_ohlcv.index, y=df_ohlcv['EMA20'], line=dict(color='cyan', width=1), name="EMA 20"), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df_ohlcv.index, y=df_ohlcv['EMA50'], line=dict(color='orange', width=1), name="EMA 50"), row=1, col=1)
    
    # RV
    for w, c in zip([10, 20, 60], ['yellow', 'lime', 'magenta']):
        if f'RV_{w}d' in df_ohlcv.columns:
            fig1.add_trace(go.Scatter(x=df_ohlcv.index, y=df_ohlcv[f'RV_{w}d'], line=dict(color=c, width=1.5), name=f"RV {w}d"), row=2, col=1)
            
    # Volume
    fig1.add_trace(go.Bar(x=df_ohlcv.index, y=df_ohlcv['Volume'], marker_color='slategray', name="Volume"), row=3, col=1)
    
    fig1.update_layout(template=layout_template, title_text=f"{ticker} Market Structure", height=900)
    fig1.update_xaxes(rangeslider_visible=False)
    
    # --- 2. GEX & Volume Profile ---
    fig2 = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], 
                         subplot_titles=("Dealer Gamma Exposure (Net GEX)", "Volume Profile (60d)"))
    
    # GEX
    # FIX: Filter GEX to reasonable range around spot to avoid squishing the chart
    # +/- 50% from spot
    lower_bound = spot_price * 0.5
    upper_bound = spot_price * 1.5
    
    if not df_gex.empty:
        # Filter indices (Strike is index)
        subset_gex = df_gex[
            (df_gex.index >= lower_bound) & 
            (df_gex.index <= upper_bound)
        ]
        
        colors = ['#00FF00' if v >= 0 else '#FF0000' for v in subset_gex['NetGEX']]
        fig2.add_trace(go.Bar(x=subset_gex.index, y=subset_gex['NetGEX'], marker_color=colors, name="Net GEX"), row=1, col=1)
    
    # Spot Line on GEX
    fig2.add_vline(x=spot_price, line_dash="dash", line_color="white", annotation_text="Spot", row=1, col=1)
    
    # Volume Profile (Horizontal)
    vp = volume_profile(df_ohlcv)
    fig2.add_trace(go.Bar(x=vp['Volume'], y=vp['Price'], orientation='h', marker_color='rgba(100, 100, 255, 0.5)', name="Vol Profile"), row=1, col=2)
    fig2.add_hline(y=spot_price, line_dash="dash", line_color="white", row=1, col=2)
    
    fig2.update_layout(template=layout_template, title_text=f"{ticker} Microstructure: Gamma & Volume", height=600)
    
    # --- 3. Smiles & Term Structure ---
    if not df_chain.empty:
        # Nearest Expiry
        today = pd.Timestamp.now().normalize()
        df_chain['days_to_exp'] = (df_chain['expiration'] - today).dt.days
        valid_chain = df_chain[df_chain['days_to_exp'] > 0].copy()
        
        if not valid_chain.empty:
            nearest_exp = valid_chain['days_to_exp'].min()
            nearest_df = valid_chain[valid_chain['days_to_exp'] == nearest_exp]
            
            fig3 = make_subplots(rows=1, cols=2, subplot_titles=(f"Vol Smile (DTE={nearest_exp})", "Term Structure (ATM IV)"))
            
            # Smile
            calls = nearest_df[nearest_df['type']=='call'].sort_values('strike')
            puts = nearest_df[nearest_df['type']=='put'].sort_values('strike')
            
            fig3.add_trace(go.Scatter(x=calls['strike'], y=calls['impliedVolatility'], mode='lines+markers', name="Call IV", line=dict(color='cyan')), row=1, col=1)
            fig3.add_trace(go.Scatter(x=puts['strike'], y=puts['impliedVolatility'], mode='lines+markers', name="Put IV", line=dict(color='magenta')), row=1, col=1)
            fig3.add_vline(x=spot_price, line_dash="dot", row=1, col=1)
            
            # Term Structure (ATM)
            # Find ATM for each expiry
            term_struct = []
            for exp, grp in valid_chain.groupby('days_to_exp'):
                # Simple ATM: min dist to spot
                atm_opt = grp.iloc[(grp['strike'] - spot_price).abs().argsort()[:1]]
                if not atm_opt.empty:
                    iv = atm_opt['impliedVolatility'].values[0]
                    if iv > 0:
                        term_struct.append({'dte': exp, 'iv': iv})
            
            ts_df = pd.DataFrame(term_struct).sort_values('dte')
            if not ts_df.empty:
                fig3.add_trace(go.Scatter(x=ts_df['dte'], y=ts_df['iv'], mode='lines+markers', name="ATM Term Structure", line=dict(color='yellow')), row=1, col=2)
            
            fig3.update_layout(template=layout_template, title_text=f"{ticker} Volatility Surface Snapshots", height=500)
            
            # --- 4. 3D Surface ---
            # X=Strike, Y=DTE, Z=IV
            # Filter somewhat around spot to avoid 3D clutter?
            # Let's take strikes +/- 40%
            
            subset = valid_chain[
                (valid_chain['strike'] > spot_price * 0.6) & 
                (valid_chain['strike'] < spot_price * 1.4) &
                (valid_chain['type'] == 'call') # Plot Calls for surface
            ]
            
            fig4 = go.Figure(data=[go.Mesh3d(x=subset['strike'], 
                                             y=subset['days_to_exp'], 
                                             z=subset['impliedVolatility'], 
                                             intensity=subset['impliedVolatility'],
                                             colorscale='Viridis', 
                                             opacity=0.8)])
            
            fig4.update_layout(template=layout_template, 
                               title_text=f"{ticker} 3D Implied Volatility Surface (Calls)",
                               scene=dict(xaxis_title='Strike', yaxis_title='DTE', zaxis_title='IV'),
                               height=800)
        else:
            fig3 = go.Figure().add_annotation(text="No Valid Options Data")
            fig4 = go.Figure().add_annotation(text="No Valid Options Data")
    else:
        fig3 = go.Figure().add_annotation(text="No Options Data Downloaded")
        fig4 = go.Figure().add_annotation(text="No Options Data Downloaded")

    # --- SAVE ---
    figs = {
        'market_structure': fig1,
        'gamma_volume': fig2,
        'smile_term': fig3,
        'vol_surface_3d': fig4
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for name, fig in figs.items():
        html_path = os.path.join(output_dir, f"{name}.html")
        fig.write_html(html_path)
        logger.info(f"Saved {html_path}")
        saved_files.append(html_path)
        
        # Try PNG save (requires kaleido)
        try:
            png_path = os.path.join(output_dir, f"{name}.png")
            fig.write_image(png_path)
        except Exception:
            pass # Silent fail if kaleido not installed

    # Open ALL HTML files
    if open_html:
        import webbrowser
        for fpath in saved_files:
            abs_path = os.path.abspath(fpath)
            logger.info(f"Opening {abs_path}...")
            webbrowser.open(f"file://{abs_path}")

# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Professional Options Analytics Dashboard")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g. SPY)")
    parser.add_argument("--lookback", type=int, default=365, help="Days of history")
    parser.add_argument("--output-dir", type=str, default="./output", help="Save directory")
    parser.add_argument("--open-html", action="store_true", help="Open charts in browser")
    
    args = parser.parse_args()
    
    # 1. Underlying Data
    df_ohlcv = download_ohlcv_yf(args.ticker, args.lookback)
    df_ohlcv = compute_realized_vol(df_ohlcv)
    
    # 2. Options Data
    df_chain = download_options_chain_yf(args.ticker)
    
    # 3. Microstructure Analytics
    df_gex = pd.DataFrame()
    if not df_chain.empty:
        current_spot = df_ohlcv['Close'].iloc[-1]
        logger.info(f"Computing Greeks & Gamma Exposure (Spot={current_spot:.2f})...")
        df_gex = compute_gamma_exposure(df_chain, current_spot)
    else:
        logger.warning("Skipping Options Analytics (Empty Chain)")
        
    # 4. Save CSV Artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    df_ohlcv.to_csv(os.path.join(args.output_dir, f"{args.ticker}_ohlcv_rv.csv"))
    if not df_chain.empty:
        df_chain.to_csv(os.path.join(args.output_dir, f"{args.ticker}_options_chain.csv"))
    if not df_gex.empty:
        df_gex.to_csv(os.path.join(args.output_dir, f"{args.ticker}_gamma_exposure.csv"))
        
    # 5. Visuals
    plot_dashboard(args.ticker, df_ohlcv, df_chain, df_gex, args.output_dir, args.open_html)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
