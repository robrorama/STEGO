#!/usr/bin/env python3
"""
equity_vol_dashboard.py

A standalone CLI tool to generate a volatility dashboard for equities/ETFs.
Downloads underlying OHLCV and Option Chains via yfinance, computes
Realized Volatility (RV), Implied Volatility (IV), and Volatility Risk Premium (VRP),
and produces a tabbed Plotly HTML dashboard.

Prerequisites:
    pip install pandas numpy scipy plotly yfinance

Usage:
    python equity_vol_dashboard.py "SPY, QQQ" --period 2y --max-expiries 6
    python equity_vol_dashboard.py NVDA --no-open
    python equity_vol_dashboard.py IWM --outdir ./my_dashboard --datadir ./my_cache

Features:
    - Robust Timezone handling (Naive enforcement).
    - Robust Column extraction.
    - Strict Index Alignment.
    - Black-Scholes IV solver with Bisection fallback.
    - Options Chain caching (Read-After-Write discipline).
    - Retry logic for network calls.
"""

import os
import sys
import argparse
import logging
import time
import datetime
import math
import pathlib
import json
import webbrowser
from typing import List, Dict, Optional, Tuple, Union, Any
from functools import reduce

# Third-party imports
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Attempt SciPy import, handle gracefully for fallback
try:
    import scipy.stats as stats
    import scipy.optimize as optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Mandatory Data Handling Fixes
# ==========================================

def force_naive_index(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    The Timezone Fix: Force all time indices to be timezone-naive.
    """
    if df is None or df.empty:
        return df
    
    # Handle Index
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    
    return df

def robust_close_extraction(df: pd.DataFrame) -> pd.Series:
    """
    The yfinance Fix: Robust column extraction for price.
    """
    if isinstance(df, pd.DataFrame):
        if 'Close' in df.columns:
            s = df['Close']
        elif 'Adj Close' in df.columns:
            s = df['Adj Close']
        else:
            # Fallback: Grab first column
            s = df.iloc[:, 0]
    else:
        s = df
    
    return force_naive_index(s)

def align_and_filter(*dfs: Union[pd.DataFrame, pd.Series]) -> List[Union[pd.DataFrame, pd.Series]]:
    """
    The Alignment Fix: Pre-calculation intersection.
    """
    valid_dfs = [d for d in dfs if d is not None and not d.empty]
    if not valid_dfs:
        return list(dfs)
    
    # Find intersection
    common = valid_dfs[0].index
    for d in valid_dfs[1:]:
        common = common.intersection(d.index)
    
    # Hard-filter
    result = []
    for d in dfs:
        if d is not None and not d.empty:
            res = d.loc[common]
            result.append(res)
        else:
            result.append(d)
            
    return result

# ==========================================
# 2. Math & Financial Models
# ==========================================

def norm_cdf(x):
    """Standard normal CDF. Uses math.erf if scipy unavailable."""
    if HAS_SCIPY:
        return stats.norm.cdf(x)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x):
    """Standard normal PDF."""
    if HAS_SCIPY:
        return stats.norm.pdf(x)
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Compute Black-Scholes price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return price

def implied_volatility(price, S, K, T, r, option_type='call'):
    """
    Solve for IV using Brentq (SciPy) or Bisection (Fallback).
    Range: [1e-4, 5.0]
    """
    if price <= 0 or T <= 0:
        return np.nan

    # Intrinsic value check
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if price < intrinsic:
        return np.nan

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price

    low, high = 1e-4, 5.0
    
    # Try SciPy Brentq
    if HAS_SCIPY:
        try:
            # Check bounds first to avoid error
            if objective(low) * objective(high) < 0:
                return optimize.brentq(objective, low, high)
        except Exception:
            pass # Fall through to bisection

    # Bisection Fallback
    for _ in range(50):
        mid = (low + high) / 2
        diff = objective(mid)
        if abs(diff) < 1e-5:
            return mid
        if diff < 0:
            low = mid
        else:
            high = mid
            
    return (low + high) / 2

# ==========================================
# 3. Data Acquisition (YFinance)
# ==========================================

def get_underlying_data(ticker: str, period='max', start=None, end=None) -> pd.DataFrame:
    """
    Download underlying OHLCV. Enforce rules.
    """
    logger.info(f"Downloading underlying data for {ticker}...")
    t = yf.Ticker(ticker)
    
    # yfinance specific: period overrides start/end usually, but we handle logic
    if start or end:
        df = t.history(start=start, end=end, auto_adjust=False, actions=False)
    else:
        df = t.history(period=period, auto_adjust=False, actions=False)
    
    if df.empty:
        logger.warning(f"No underlying data found for {ticker}")
        return df

    # 1. Timezone Fix
    df = force_naive_index(df)
    
    # 2. De-duplicate and Sort
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    
    logger.info(f"Downloaded {len(df)} rows for {ticker}. Last date: {df.index[-1]}")
    return df

def get_available_expirations(ticker: str, max_expiries=12) -> List[pd.Timestamp]:
    """
    Get expiration dates, convert to Timestamps, sort, filter.
    """
    t = yf.Ticker(ticker)
    exps = t.options
    if not exps:
        return []
    
    # Parse to timestamps
    dates = []
    for e in exps:
        try:
            dt = pd.to_datetime(e)
            # Timezone Fix for scalar
            if dt.tz is not None:
                dt = dt.tz_localize(None)
            dates.append(dt)
        except:
            continue
            
    dates.sort()
    return dates[:max_expiries]

def load_or_download_option_chain_csv(ticker: str, expiration: pd.Timestamp, datadir: str) -> pd.DataFrame:
    """
    Download chain, save to CSV, Re-read (Read-After-Write).
    Handles retries.
    """
    exp_str = expiration.strftime('%Y-%m-%d')
    base_dir = pathlib.Path(datadir) / "options" / "yfinance" / ticker
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / f"{exp_str}.csv"
    
    # We do NOT use cached data if the goal is to get fresh options.
    # However, the prompt implies "Save to CSV... Immediately re-read".
    # We will strictly follow: Download -> Save -> Read -> Return.
    
    t = yf.Ticker(ticker)
    
    max_retries = 3
    df_chain = pd.DataFrame()
    
    for attempt in range(max_retries):
        try:
            chain = t.option_chain(exp_str)
            calls = chain.calls
            puts = chain.puts
            
            if calls is None and puts is None:
                break
                
            # Tag and Combine
            if calls is not None:
                calls['type'] = 'call'
                calls['expiration'] = expiration
            if puts is not None:
                puts['type'] = 'put'
                puts['expiration'] = expiration
                
            df_chain = pd.concat([calls, puts], ignore_index=True)
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {exp_str}: {e}")
            time.sleep(2 ** attempt)
            
    if df_chain.empty:
        return pd.DataFrame()
        
    # Save
    df_chain.to_csv(csv_path, index=False)
    
    # Re-read (Strict requirement)
    df_read = pd.read_csv(csv_path)
    
    # Fix timestamp type after read
    if 'expiration' in df_read.columns:
        df_read['expiration'] = pd.to_datetime(df_read['expiration'])
        # Scalar Timezone Fix if needed (though usually read as naive)
        
    return df_read

# ==========================================
# 4. Computation Logic
# ==========================================

def compute_realized_vol(df_price: pd.DataFrame, windows=[10, 20, 60]) -> pd.DataFrame:
    """
    Compute Annualized Realized Volatility.
    Input df_price must contain 'Close' (robustly extracted).
    """
    s_close = robust_close_extraction(df_price)
    
    # Log Returns
    log_ret = np.log(s_close / s_close.shift(1))
    
    rv_dict = {}
    rv_dict['Close'] = s_close
    
    for w in windows:
        # Annualize: std * sqrt(252)
        roll_std = log_ret.rolling(window=w).std()
        rv_dict[f'RV_{w}'] = roll_std * np.sqrt(252)
        
    df_rv = pd.DataFrame(rv_dict)
    # Ensure TZ naive
    df_rv = force_naive_index(df_rv)
    return df_rv

def build_option_metrics(df_chain: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    Compute ATM IV, RR, BF for a single expiration chain.
    """
    if df_chain.empty:
        return {}

    # Ensure required columns
    required = ['strike', 'type', 'lastPrice', 'bid', 'ask']
    for r in required:
        if r not in df_chain.columns:
            return {}
            
    # Normalize IV column
    if 'impliedVolatility' not in df_chain.columns:
        df_chain['impliedVolatility'] = np.nan
    
    # Compute IV if missing
    # T = DTE / 252
    # We need 'expiration' from df_chain to calc DTE, or pass it in.
    # df_chain has 'expiration' column.
    exp_date = df_chain['expiration'].iloc[0]
    now = pd.Timestamp.now().normalize()
    # Timezone Fix for calculation
    if exp_date.tz is not None: exp_date = exp_date.tz_localize(None)
    if now.tz is not None: now = now.tz_localize(None)
    
    dte_days = (exp_date - now).days
    T = max(dte_days, 0) / 252.0
    
    # Calculate IV row-by-row where needed
    # (Doing this via loop for clarity and safety, though vectorized is faster)
    # Using mid price
    df_chain['mid'] = (df_chain['bid'] + df_chain['ask']) / 2
    # Fallback to lastPrice
    df_chain['mid'] = df_chain['mid'].fillna(df_chain['lastPrice'])
    
    def get_iv(row):
        val = row.get('impliedVolatility', np.nan)
        if pd.notnull(val) and val > 0:
            return val
        # Compute BS IV
        return implied_volatility(row['mid'], current_price, row['strike'], T, 0.0, row['type'])

    # Apply IV calc
    df_chain['calc_iv'] = df_chain.apply(get_iv, axis=1)
    
    # Filter valid IVs
    valid = df_chain[df_chain['calc_iv'] > 0].copy()
    if valid.empty:
        return {}

    # Separate Calls/Puts
    calls = valid[valid['type'] == 'call']
    puts = valid[valid['type'] == 'put']
    
    if calls.empty or puts.empty:
        return {}

    # --- ATM IV ---
    # Nearest strike to current price
    atm_strike_c = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
    atm_strike_p = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
    
    iv_atm_c = atm_strike_c['calc_iv'].values[0] if not atm_strike_c.empty else np.nan
    iv_atm_p = atm_strike_p['calc_iv'].values[0] if not atm_strike_p.empty else np.nan
    
    atm_iv = np.nanmean([iv_atm_c, iv_atm_p])
    
    # --- 25 Delta Proxies ---
    # Call 25D ~ Strike = Spot * 1.15
    # Put 25D ~ Strike = Spot * 0.85
    target_c = current_price * 1.15
    target_p = current_price * 0.85
    
    c_25 = calls.iloc[(calls['strike'] - target_c).abs().argsort()[:1]]
    p_25 = puts.iloc[(puts['strike'] - target_p).abs().argsort()[:1]]
    
    iv_25c = c_25['calc_iv'].values[0] if not c_25.empty else np.nan
    iv_25p = p_25['calc_iv'].values[0] if not p_25.empty else np.nan
    
    # RR = C25 - P25
    rr = iv_25c - iv_25p
    
    # BF = 0.5 * (C25 + P25) - ATM
    bf = 0.5 * (iv_25c + iv_25p) - atm_iv
    
    return {
        'expiration': exp_date,
        'dte': dte_days,
        'atm_iv': atm_iv,
        'iv_25c': iv_25c,
        'iv_25p': iv_25p,
        'rr_25': rr,
        'bf_25': bf
    }

def build_iv_rv_vrp_summary(df_metrics: pd.DataFrame, df_rv: pd.DataFrame) -> pd.DataFrame:
    """
    VRP Snapshot logic.
    Targets [7, 30, 60] DTE.
    RV Windows [10, 20, 60].
    Mapping: 7->10, 30->20, 60->60.
    """
    if df_metrics.empty or df_rv.empty:
        return pd.DataFrame()
        
    targets = [7, 30, 60]
    rv_map = {7: 10, 30: 20, 60: 60}
    
    results = []
    
    # Get latest RV values
    last_rv = df_rv.iloc[-1]
    
    for t in targets:
        # Find closest DTE in metrics
        # Use simple absolute difference
        df_metrics['diff'] = (df_metrics['dte'] - t).abs()
        closest = df_metrics.sort_values('diff').iloc[0]
        
        atm_iv = closest['atm_iv']
        
        # Get corresponding RV
        rv_win = rv_map[t]
        rv_col = f'RV_{rv_win}'
        if rv_col not in last_rv:
            continue
            
        real_vol = last_rv[rv_col]
        
        vrp = atm_iv - real_vol
        
        results.append({
            'Target_DTE': t,
            'Actual_DTE': closest['dte'],
            'RV_Window': rv_win,
            'ATM_IV': atm_iv,
            'RV': real_vol,
            'VRP': vrp
        })
        
    return pd.DataFrame(results)

# ==========================================
# 5. Plotly Dashboard
# ==========================================

def make_dashboard_html(ticker: str, 
                        df_rv: pd.DataFrame, 
                        df_metrics: pd.DataFrame, 
                        df_vrp: pd.DataFrame, 
                        out_file: str):
    """
    Create Tabbed HTML Dashboard using Plotly.
    Tab 1: RV Timeseries + VRP Bar
    Tab 2: Skew (RR & BF) vs DTE
    Tab 3: Term Structure (ATM IV vs DTE)
    """
    
    # --- Colors ---
    colors = {
        'price': 'rgba(255, 255, 255, 0.2)',
        'rv10': '#00F0F0',
        'rv20': '#00FF00',
        'rv60': '#FF00FF',
        'iv': '#FFFF00',
        'bar_iv': '#FFFF00',
        'bar_rv': '#00FF00',
        'bar_vrp': '#00BFFF'
    }

    # ==========================
    # TAB 1: RV + VRP
    # ==========================
    fig1 = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], 
                         subplot_titles=("Realized Volatility History", "VRP Snapshot"),
                         vertical_spacing=0.15)
    
    # RV Time Series
    if not df_rv.empty:
        # Price (Secondary, hidden axis or just faint overlay scaled?)
        # Let's put price on secondary y
        fig1.add_trace(go.Scatter(x=df_rv.index, y=df_rv['Close'], name='Close', 
                                  line=dict(color=colors['price'], width=2), yaxis='y3'), row=1, col=1)
        
        for w, c in zip([10, 20, 60], [colors['rv10'], colors['rv20'], colors['rv60']]):
            col = f'RV_{w}'
            if col in df_rv.columns:
                fig1.add_trace(go.Scatter(x=df_rv.index, y=df_rv[col], name=col,
                                          line=dict(color=c, width=1.5)), row=1, col=1)

    # VRP Bar Chart
    if not df_vrp.empty:
        x_cats = [f"Target {r['Target_DTE']}d" for _, r in df_vrp.iterrows()]
        
        fig1.add_trace(go.Bar(x=x_cats, y=df_vrp['ATM_IV'], name='Implied Vol', marker_color=colors['bar_iv']), row=2, col=1)
        fig1.add_trace(go.Bar(x=x_cats, y=df_vrp['RV'], name='Realized Vol', marker_color=colors['bar_rv']), row=2, col=1)
        fig1.add_trace(go.Bar(x=x_cats, y=df_vrp['VRP'], name='VRP', marker_color=colors['bar_vrp']), row=2, col=1)

    fig1.update_layout(
        template="plotly_dark", 
        title_text=f"{ticker} - Volatility Overview",
        barmode='group',
        yaxis3=dict(overlaying='y', side='right', showgrid=False, title='Price')
    )

    # ==========================
    # TAB 2: Skew Metrics
    # ==========================
    fig2 = make_subplots(rows=2, cols=1, subplot_titles=("Risk Reversal (25d Call - 25d Put)", "Butterfly (Wings - ATM)"))
    
    if not df_metrics.empty:
        # Sort by DTE
        df_m_sorted = df_metrics.sort_values('dte')
        
        # RR
        fig2.add_trace(go.Scatter(x=df_m_sorted['dte'], y=df_m_sorted['rr_25'], mode='lines+markers',
                                  name='RR 25d', line=dict(color='#FF6347')), row=1, col=1)
        
        # BF
        fig2.add_trace(go.Scatter(x=df_m_sorted['dte'], y=df_m_sorted['bf_25'], mode='lines+markers',
                                  name='BF 25d', line=dict(color='#8A2BE2')), row=2, col=1)
        
    fig2.update_layout(template="plotly_dark", title_text=f"{ticker} - Skew Structure")
    fig2.update_xaxes(title_text="Days to Expiration")

    # ==========================
    # TAB 3: Term Structure
    # ==========================
    fig3 = go.Figure()
    if not df_metrics.empty:
        df_m_sorted = df_metrics.sort_values('dte')
        fig3.add_trace(go.Scatter(x=df_m_sorted['dte'], y=df_m_sorted['atm_iv'], mode='lines+markers',
                                  name='ATM IV', line=dict(color='#FFFF00')))
        
    fig3.update_layout(template="plotly_dark", title_text=f"{ticker} - ATM IV Term Structure",
                       xaxis_title="Days to Expiration", yaxis_title="Annualized IV")

    # --- Generate HTML with Tabs ---
    # We will embed the div strings and a small JS script to toggle them
    
    div1 = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn')
    div2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)
    div3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{ticker} Volatility Dashboard</title>
        <style>
            body {{ font-family: sans-serif; background-color: #111; color: #fff; margin: 0; padding: 20px; }}
            .tab {{ overflow: hidden; border: 1px solid #444; background-color: #222; }}
            .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-size: 17px; }}
            .tab button:hover {{ background-color: #444; }}
            .tab button.active {{ background-color: #007bff; color: white; }}
            .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #444; border-top: none; animation: fadeEffect 1s; }}
            @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
        </style>
    </head>
    <body>

    <h2>{ticker} Volatility Dashboard</h2>
    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="tab">
      <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Overview (RV/VRP)</button>
      <button class="tablinks" onclick="openTab(event, 'Skew')">Skew (RR/BF)</button>
      <button class="tablinks" onclick="openTab(event, 'TermStructure')">Term Structure</button>
    </div>

    <div id="Overview" class="tabcontent">
      {div1}
    </div>

    <div id="Skew" class="tabcontent">
      {div2}
    </div>

    <div id="TermStructure" class="tabcontent">
      {div3}
    </div>

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
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

# ==========================================
# 6. Main Orchestration
# ==========================================

def process_ticker(ticker: str, args):
    """
    Main pipeline for a single ticker.
    """
    logger.info(f"--- Processing {ticker} ---")
    
    # Setup paths
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    out_dir = pathlib.Path(args.outdir) / ticker / today_str
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Underlying
    df_price = get_underlying_data(ticker, period=args.period, start=args.start, end=args.end)
    if df_price.empty:
        logger.error(f"Failed to get price data for {ticker}. Skipping.")
        return

    # 2. Realized Vol
    df_rv = compute_realized_vol(df_price)
    # Save RV CSV
    rv_path = out_dir / f"{ticker}_realized_vol_timeseries.csv"
    df_rv.to_csv(rv_path)
    logger.info(f"Saved RV Data: {rv_path}")

    # 3. Options
    # Get last price for ATM calc
    last_close = robust_close_extraction(df_price).iloc[-1]
    
    expirations = get_available_expirations(ticker, max_expiries=args.max_expiries)
    
    metrics_list = []
    
    if not expirations:
        logger.warning(f"No expirations found for {ticker}.")
    else:
        logger.info(f"Processing {len(expirations)} expirations for {ticker}...")
        for exp in expirations:
            # Download/Load
            df_chain = load_or_download_option_chain_csv(ticker, exp, args.datadir)
            if df_chain.empty:
                continue
            
            # Compute Metrics
            m = build_option_metrics(df_chain, last_close)
            if m:
                metrics_list.append(m)
                
    df_metrics = pd.DataFrame(metrics_list)
    
    # Save Option Metrics CSV
    opt_path = out_dir / f"{ticker}_option_metrics_by_expiration.csv"
    df_metrics.to_csv(opt_path, index=False)
    logger.info(f"Saved Option Metrics: {opt_path}")
    
    # 4. VRP Summary
    df_vrp = build_iv_rv_vrp_summary(df_metrics, df_rv)
    vrp_path = out_dir / f"{ticker}_iv_rv_vrp_summary.csv"
    df_vrp.to_csv(vrp_path, index=False)
    logger.info(f"Saved VRP Summary: {vrp_path}")
    
    # 5. Dashboard
    dash_path = out_dir / f"{ticker}_iv_rvrp_skew_dashboard.html"
    try:
        make_dashboard_html(ticker, df_rv, df_metrics, df_vrp, str(dash_path))
        logger.info(f"Saved Dashboard: {dash_path}")
        
        if not args.no_open:
            webbrowser.open(f"file://{os.path.abspath(dash_path)}")
            
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")

def main():
    parser = argparse.ArgumentParser(description="Equity Volatility Dashboard Generator")
    parser.add_argument("tickers", type=str, help="Comma/space separated tickers (e.g. 'SPY, QQQ')")
    parser.add_argument("--period", type=str, default="2y", help="History period (default: 2y)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--max-expiries", type=int, default=12, help="Max option expirations to process")
    parser.add_argument("--outdir", type=str, default="./out", help="Output directory")
    parser.add_argument("--datadir", type=str, default="./data", help="Data cache directory")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    
    args = parser.parse_args()
    
    # Parse tickers
    raw_tickers = args.tickers.replace(',', ' ').split()
    tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
    
    if not tickers:
        logger.error("No tickers provided.")
        sys.exit(1)
        
    for t in tickers:
        try:
            process_ticker(t, args)
        except Exception as e:
            logger.error(f"Critical error processing {t}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
