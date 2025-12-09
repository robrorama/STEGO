#!/usr/bin/env python3
"""
Pre-Market & Volatility Dashboard (Standalone)
----------------------------------------------
A production-ready, standalone Python script for options traders.
It gathers market data, derived volatility metrics, and macro correlations,
generating a comprehensive Plotly dashboard.

Usage:
    python3 premarket_vol_dashboard.py --ticker SPY --period 1y
    python3 premarket_vol_dashboard.py --ticker NVDA --open-html

Requirements:
    pip install yfinance pandas numpy plotly scipy
"""

import argparse
import os
import sys
import datetime
import webbrowser
import warnings
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# Suppress minor pandas warnings for clean CLI output
warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------------------------------------------------------
# 1. Robust Data Helpers
# ------------------------------------------------------------------------------

def force_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the index is a timezone-naive DatetimeIndex.
    """
    if df is None or df.empty:
        return df
    
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Sort and remove duplicates
    df = df.sort_index()
    return df[~df.index.duplicated(keep='last')]

def clean_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens MultiIndex columns (Price, Ticker) to single level (Price)
    if yfinance returns them.
    """
    if df is None or df.empty:
        return df
        
    if isinstance(df.columns, pd.MultiIndex):
        # If we have (Price, Ticker), drop level 1
        # Check if common columns are in level 0
        level0 = df.columns.get_level_values(0)
        if 'Close' in level0 or 'Open' in level0 or 'Adj Close' in level0:
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass
    return df

def robust_extract_close(df: pd.DataFrame) -> pd.Series:
    """
    Extracts a clean 'Close' series from yfinance result.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Clean multi-index first if passed raw
    df = clean_yf_columns(df)

    # Priority extraction
    if 'Close' in df.columns:
        s = df['Close']
    elif 'Adj Close' in df.columns:
        s = df['Adj Close']
    else:
        # Fallback to first column
        s = df.iloc[:, 0]
    
    # Ensure numeric
    s = pd.to_numeric(s, errors='coerce')
    return s

def get_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Wrapper for yfinance download with standard cleaning.
    """
    try:
        df = yf.download(
            ticker, 
            period=period, 
            interval=interval, 
            progress=False, 
            threads=False,
            auto_adjust=False
        )
        df = clean_yf_columns(df)
        return force_tz_naive(df)
    except Exception as e:
        print(f"[WARN] Failed to download {ticker}: {e}")
        return pd.DataFrame()

def get_intraday_data(ticker: str) -> pd.DataFrame:
    """
    Fetches 5-minute intraday data (max 60 days allowed by YF).
    """
    try:
        # 1mo is a safe window for 5m data
        df = yf.download(
            ticker, 
            period="1mo", 
            interval="5m", 
            progress=False, 
            threads=False,
            auto_adjust=False
        )
        df = clean_yf_columns(df)
        return force_tz_naive(df)
    except Exception as e:
        print(f"[WARN] Failed to download intraday {ticker}: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# 2. Options & Volatility Logic
# ------------------------------------------------------------------------------

def get_options_chain_data(ticker: str) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
    """
    Fetches the nearest expiration option chain to compute Skew and ATM IV.
    Returns: (ATM_IV, Skew_DataFrame)
    """
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None, None
        
        # Pick the next standard monthly if possible, or just the 2nd expiration 
        # to avoid expiring-today noise.
        target_date = exps[1] if len(exps) > 1 else exps[0]
        
        chain = tk.option_chain(target_date)
        calls = chain.calls
        puts = chain.puts
        
        # Current Price estimate
        hist = tk.history(period='1d')
        if hist.empty:
            return None, None
        current_price = hist['Close'].iloc[-1]
        
        # Calculate ATM IV (average of Call/Put IV near spot)
        # Find strikes closest to current price
        calls['abs_diff'] = abs(calls['strike'] - current_price)
        puts['abs_diff'] = abs(puts['strike'] - current_price)
        
        atm_call = calls.sort_values('abs_diff').iloc[0]
        atm_put = puts.sort_values('abs_diff').iloc[0]
        
        atm_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2.0
        
        # Build Skew Data (IV across strikes)
        # Merge calls and puts on strike
        skew_df = pd.merge(
            calls[['strike', 'impliedVolatility']].rename(columns={'impliedVolatility': 'iv_call'}),
            puts[['strike', 'impliedVolatility']].rename(columns={'impliedVolatility': 'iv_put'}),
            on='strike', how='outer'
        )
        skew_df['avg_iv'] = skew_df[['iv_call', 'iv_put']].mean(axis=1)
        skew_df['moneyness'] = skew_df['strike'] / current_price
        
        # Filter for reasonable moneyness (e.g. 0.8 to 1.2)
        skew_df = skew_df[(skew_df['moneyness'] > 0.7) & (skew_df['moneyness'] < 1.3)]
        
        return atm_iv, skew_df.sort_values('strike')
        
    except Exception as e:
        print(f"[WARN] Options data failure: {e}")
        return None, None

def compute_realized_volatility(prices: pd.Series, windows=[10, 20, 30]) -> pd.DataFrame:
    """
    Computes annualized realized volatility.
    """
    log_ret = np.log(prices / prices.shift(1))
    rv_df = pd.DataFrame(index=prices.index)
    
    for w in windows:
        # Std dev * sqrt(252)
        rv_df[f'RV_{w}d'] = log_ret.rolling(window=w).std() * np.sqrt(252) * 100
        
    return rv_df

def compute_intraday_volume_zscore(df_intra: pd.DataFrame) -> pd.DataFrame:
    """
    Computes volume Z-scores for intraday buckets.
    """
    if df_intra.empty:
        return pd.DataFrame()
    
    # Extract time component
    df = df_intra.copy()
    df['time'] = df.index.time
    df['date'] = df.index.date
    
    # Calculate mean and std of Volume per time bucket over the last N days
    stats = df.groupby('time')['Volume'].agg(['mean', 'std'])
    
    # Map back
    df = df.merge(stats, on='time', how='left')
    
    # Z-score
    df['vol_zscore'] = (df['Volume'] - df['mean']) / (df['std'] + 1e-9)
    
    # Pivot for heatmap: Index=Date, Cols=Time
    # We aggregate just in case of duplicates, though 5m shouldn't have them
    heatmap_data = df.pivot_table(index='date', columns='time', values='vol_zscore', aggfunc='mean')
    
    return heatmap_data

# ------------------------------------------------------------------------------
# 3. Visualization Generator
# ------------------------------------------------------------------------------

def generate_dashboard(
    ticker: str,
    df_daily: pd.DataFrame,
    df_intra: pd.DataFrame,
    rv_df: pd.DataFrame,
    macro_corr: pd.DataFrame,
    skew_df: pd.DataFrame,
    atm_iv: float,
    vol_term_structure: pd.DataFrame,
    gap_data: pd.DataFrame,
    output_dir: str
):
    """
    Creates the comprehensive Plotly dashboard.
    """
    
    # --- Setup Subplots ---
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],           # Price+RV, IV/RV Divergence
            [{"type": "xy"}, {"type": "xy"}],           # Term Struct, Skew
            [{"type": "heatmap"}, {"type": "polar"}],   # Volume Heatmap, Macro Radar
            [{"type": "xy"}, {"type": "indicator"}]     # Gap Risk, IV Gauge
        ],
        subplot_titles=(
            f"{ticker} Price & Realized Volatility", "Realized Volatility History",
            "Volatility Term Structure (VIX Proxies)", f"Options Skew (Next Exp)",
            "Intraday Volume Z-Score Heatmap", "Macro Correlation Radar (90d)",
            "Overnight Gap Distribution", "Current ATM IV vs RV Range"
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # 1. Price + RV Bands
    # We simulate bands: Price +/- 20d RV equivalent move? 
    # Or just Price candles. Let's do Price Candles + BB
    if not df_daily.empty:
        # Standard OHLC
        fig.add_trace(go.Candlestick(
            x=df_daily.index,
            open=df_daily['Open'], high=df_daily['High'],
            low=df_daily['Low'], close=df_daily['Close'],
            name='OHLC'
        ), row=1, col=1)
    
    # 2. RV History
    if not rv_df.empty:
        fig.add_trace(go.Scatter(x=rv_df.index, y=rv_df['RV_20d'], name='RV 20d', line=dict(color='orange')), row=1, col=2)
        fig.add_trace(go.Scatter(x=rv_df.index, y=rv_df['RV_10d'], name='RV 10d', line=dict(color='cyan', width=1)), row=1, col=2)
        # Add Current IV line if available
        if atm_iv:
            fig.add_hline(y=atm_iv * 100, line_dash="dash", line_color="red", annotation_text=f"Current IV: {atm_iv*100:.1f}%", row=1, col=2)

    # 3. Vol Term Structure
    if not vol_term_structure.empty:
        # Current snapshot
        latest = vol_term_structure.iloc[-1]
        fig.add_trace(go.Scatter(
            x=['Spot (VIX)', '3-Month', '6-Month'],
            y=[latest['^VIX'], latest['^VIX3M'], latest['^VIX6M']],
            mode='lines+markers',
            name='Term Structure',
            line=dict(color='purple', width=3)
        ), row=2, col=1)
        
        # Check regime
        is_contango = latest['^VIX'] < latest['^VIX3M']
        regime = "Contango (Normal)" if is_contango else "Backwardation (Fear)"
        fig.add_annotation(
            xref="x3", yref="y3",
            text=regime,
            x=1, y=latest['^VIX'],
            showarrow=True, arrowhead=1
        )

    # 4. Skew Surface
    if skew_df is not None and not skew_df.empty:
        fig.add_trace(go.Scatter(
            x=skew_df['strike'], y=skew_df['avg_iv'] * 100,
            mode='lines', name='Implied Vol',
            line=dict(color='yellow')
        ), row=2, col=2)
        
        # Mark ATM
        current_price = df_daily['Close'].iloc[-1]
        fig.add_vline(x=current_price, line_dash="dot", line_color="gray", row=2, col=2)

    # 5. Volume Heatmap
    heatmap_data = compute_intraday_volume_zscore(df_intra)
    if not heatmap_data.empty:
        # Heatmap requires strings for axes usually in plotly express, but GO works with index
        # Format dates for Y axis text
        y_dates = [d.strftime('%Y-%m-%d') for d in heatmap_data.index]
        x_times = [t.strftime('%H:%M') for t in heatmap_data.columns]
        
        fig.add_trace(go.Heatmap(
            z=heatmap_data.values,
            x=x_times,
            y=y_dates,
            colorscale='RdBu_r', # Red = High Vol, Blue = Low Vol
            zmid=0,
            colorbar=dict(len=0.4, y=0.25) # adjust position
        ), row=3, col=1)

    # 6. Macro Radar
    if not macro_corr.empty:
        # Radial chart
        r_vals = macro_corr['Correlation'].values
        theta_vals = macro_corr['Asset'].values
        
        # Close the loop
        r_vals = np.concatenate([r_vals, [r_vals[0]]])
        theta_vals = np.concatenate([theta_vals, [theta_vals[0]]])
        
        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill='toself',
            name='90d Correlation'
        ), row=3, col=2)

    # 7. Gap Distribution
    if not gap_data.empty:
        fig.add_trace(go.Histogram(
            x=gap_data['gap_pct'],
            nbinsx=30,
            name='Gap %',
            marker_color='teal'
        ), row=4, col=1)

    # 8. IV Gauge
    # Use RV_30d range as the 'bounds' for the gauge
    if not rv_df.empty and atm_iv:
        current_iv_val = atm_iv * 100
        rv_vals = rv_df['RV_30d'].dropna()
        min_rv, max_rv = rv_vals.min(), rv_vals.max()
        avg_rv = rv_vals.mean()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = current_iv_val,
            domain = {'row': 0, 'column': 1}, # This needs specific layout adjustment in make_subplots? 
            # Subplots with 'indicator' type handle domain automatically relative to grid cell
            title = {'text': "IV vs Hist RV"},
            delta = {'reference': avg_rv, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [min(0, min_rv*0.8), max(current_iv_val*1.2, max_rv*1.2)]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, min_rv], 'color': "green"},
                    {'range': [min_rv, max_rv], 'color': "gray"},
                    {'range': [max_rv, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': current_iv_val
                }
            }
        ), row=4, col=2)

    # --- Layout ---
    fig.update_layout(
        template='plotly_dark',
        height=1400,
        width=1600,
        title_text=f"Pro Options Dashboard: {ticker} | {datetime.date.today()}",
        showlegend=False
    )
    
    # Save
    html_path = os.path.join(output_dir, "dashboard.html")
    fig.write_html(html_path)
    return html_path

# ------------------------------------------------------------------------------
# 4. Main Controller
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Professional Pre-Market Options Dashboard")
    parser.add_argument("--ticker", type=str, default="SPY", help="Equity Ticker (default: SPY)")
    parser.add_argument("--period", type=str, default="1y", help="Historical Data Period")
    parser.add_argument("--open-html", action="store_true", help="Auto-open browser")
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    print(f"[*] Initializing Dashboard for {ticker}...")
    
    # Create output dir
    out_dir = f"dashboard_output_{ticker}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1. Download Core Data
    print(f"[*] Downloading Daily & Intraday Data...")
    df_daily = get_historical_data(ticker, period=args.period)
    df_intra = get_intraday_data(ticker)
    
    if df_daily.empty:
        print("[!] Error: No daily data found. Exiting.")
        sys.exit(1)

    # 2. Macro Data
    print(f"[*] Downloading Macro & Volatility Proxies...")
    # Map: Symbol -> Name
    macro_map = {
        '^VIX': 'VIX',
        '^VIX3M': 'VIX3M', 
        '^VIX6M': 'VIX6M',
        '^TNX': 'Rates (10y)',
        'DX-Y.NYB': 'Dollar'
    }
    
    macro_data = {}
    for sym, name in macro_map.items():
        m_df = get_historical_data(sym, period=args.period)
        if not m_df.empty:
            s = robust_extract_close(m_df)
            s.name = sym
            macro_data[sym] = s
            
    # 3. Compute Metrics
    print(f"[*] Computing Volatility Metrics...")
    s_close = robust_extract_close(df_daily)
    
    # Realized Vol
    rv_df = compute_realized_volatility(s_close)
    
    # Gap Analysis
    # Gap = (Open - PrevClose) / PrevClose
    # Need to align Open and Close properly
    prev_close = s_close.shift(1)
    s_open = df_daily['Open'] if 'Open' in df_daily.columns else s_close # Fallback
    gap_series = (s_open - prev_close) / prev_close
    
    # FIX: Explicitly pass index to DataFrame constructor to handle potential scalar broadcasting issues
    gap_data = pd.DataFrame({'gap_pct': gap_series * 100}, index=gap_series.index).dropna()
    
    # 4. Correlations (Macro)
    print(f"[*] Computing Correlations...")
    # Align dates
    base_rets = s_close.pct_change()
    corr_results = []
    
    for sym, s_macro in macro_data.items():
        name = macro_map[sym]
        if sym in ['^VIX3M', '^VIX6M']: continue # Skip term structure tickers for correlation
        
        macro_rets = s_macro.pct_change()
        # Intersection
        common = base_rets.index.intersection(macro_rets.index)
        if len(common) > 30:
            c = base_rets.loc[common].corr(macro_rets.loc[common])
            corr_results.append({'Asset': name, 'Correlation': c})
            
    corr_df = pd.DataFrame(corr_results)
    
    # 5. Vol Term Structure Data
    term_syms = ['^VIX', '^VIX3M', '^VIX6M']
    term_data = []
    for ts in term_syms:
        if ts in macro_data:
            term_data.append(macro_data[ts])
            
    if len(term_data) == 3:
        vol_ts_df = pd.concat(term_data, axis=1).dropna()
    else:
        vol_ts_df = pd.DataFrame()

    # 6. Options Data
    print(f"[*] Fetching Options Chain (ATM IV & Skew)...")
    atm_iv, skew_df = get_options_chain_data(ticker)
    if atm_iv:
        print(f"    ATM IV: {atm_iv*100:.2f}%")
    else:
        print("    [!] Options data unavailable or failed.")

    # 7. Generate Output
    print(f"[*] Generating Plots...")
    
    # Save CSVs
    df_daily.to_csv(os.path.join(out_dir, "ohlcv.csv"))
    rv_df.to_csv(os.path.join(out_dir, "realized_vol.csv"))
    if skew_df is not None: skew_df.to_csv(os.path.join(out_dir, "skew_curve.csv"))
    
    html_file = generate_dashboard(
        ticker, df_daily, df_intra, rv_df, corr_df, 
        skew_df, atm_iv, vol_ts_df, gap_data, out_dir
    )
    
    print(f"[SUCCESS] Dashboard saved to: {html_file}")
    if args.open_html:
        webbrowser.open(f"file://{os.path.abspath(html_file)}")

if __name__ == "__main__":
    main()
