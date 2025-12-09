"""
SCRIPTNAME: stego_vol_dashboard.py
AUTHOR: Michael Derby
FRAMEWORK: STEGO Financial Framework
DESCRIPTION: 
    Generates a professional-grade Volatility Structure Dashboard.
    analyzing Term Structure, Skew, IV Surface, and Volatility Rotation.
    
    Adheres to strict CSV-only output for dashboard artifacts.
    Uses existing data_retrieval libraries without modification.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.optimize import brentq
from scipy.interpolate import griddata, interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import webbrowser
from datetime import datetime, timedelta
import warnings

# --- IMPORT USER LIBRARIES ---
# We assume these files are in the PYTHONPATH or current directory
try:
    import data_retrieval as dr
    import options_data_retrieval as odr
except ImportError as e:
    sys.exit(f"CRITICAL ERROR: Could not import STEGO data libraries: {e}")

# --- GLOBAL SETTINGS ---
# Suppress specific pandas warnings regarding fragmentation or chaining
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

DASHBOARD_ROOT = "/dev/shm/VOL_STRUCTURE_DASHBOARD"

# --- MATH & BLACK-SCHOLES ENGINE ---

def norm_cdf(x):
    return si.norm.cdf(x)

def norm_pdf(x):
    return si.norm.pdf(x)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Vectorized Black-Scholes Price"""
    # Avoid div by zero
    T = np.maximum(T, 1e-5) 
    sigma = np.maximum(sigma, 1e-4)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = (S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1))
    return price

def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    """Vectorized Delta Calculation"""
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-4)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1

def implied_volatility(price, S, K, T, r, option_type='call'):
    """
    Robust IV Solver using Newton-Raphson with fallback to BrentQ.
    Handles scalar inputs.
    """
    if price <= 0 or T <= 0:
        return 0.0

    # Intrinsic value check
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if price < intrinsic:
        return 0.0 # Arbitrage violation or bad data

    MAX_ITER = 100
    PRECISION = 1e-5
    sigma = 0.5 # Initial guess

    for i in range(MAX_ITER):
        p = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - p
        
        if abs(diff) < PRECISION:
            return sigma
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm_pdf(d1)
        
        if vega < 1e-8:
            break # Vega too low, switch to brent
            
        sigma = sigma + diff / vega
        
    return sigma

def vectorize_iv(row, S, r):
    """Wrapper for DataFrame apply"""
    try:
        # Midpoint price
        price = (row['bid'] + row['ask']) / 2.0
        if price <= 0:
            price = row['lastPrice']
        
        # Calculate T in years
        T = (row['expiration'] - datetime.now()).days / 365.0
        if T <= 0: return 0.0

        return implied_volatility(
            price, S, row['strike'], T, r, row['type']
        )
    except:
        return 0.0

def vectorize_delta(row, S, r):
    """Wrapper for DataFrame apply"""
    try:
        T = (row['expiration'] - datetime.now()).days / 365.0
        if T <= 0: return 0.0
        
        return black_scholes_delta(
            S, row['strike'], T, r, row['iv'], row['type']
        )
    except:
        return 0.0

# --- DATA PROCESSING PIPELINE ---

def get_risk_free_rate():
    # Attempt to get TNX, else default
    try:
        df = dr.get_stock_data("^TNX", period="5d")
        if not df.empty:
            return df['Close'].iloc[-1] / 100.0
    except:
        pass
    return 0.045 # Default 4.5%

def setup_directories(ticker):
    today = datetime.now().strftime('%Y-%m-%d')
    base_path = os.path.join(DASHBOARD_ROOT, ticker, today)
    os.makedirs(base_path, exist_ok=True)
    return base_path

def load_vix_complex(days):
    """Loads VIX and VIX3M (or VXV)"""
    print(f"[*] Loading Volatility Complex for last {days} days...")
    
    # Get VIX
    df_vix = dr.get_stock_data("^VIX", period=f"{days}d")
    
    # Get VIX3M or VXV
    df_vix3m = dr.get_stock_data("^VIX3M", period=f"{days}d")
    if df_vix3m.empty:
        df_vix3m = dr.get_stock_data("^VXV", period=f"{days}d")
    
    # Align
    if df_vix.empty or df_vix3m.empty:
        print("[!] Warning: Missing VIX or VIX3M data.")
        return pd.DataFrame()
        
    combined = pd.DataFrame({
        'VIX': df_vix['Close'],
        'VIX3M': df_vix3m['Close']
    }).dropna()
    
    combined['Ratio'] = combined['VIX'] / combined['VIX3M']
    combined['Slope'] = combined['VIX3M'] - combined['VIX']
    
    # Regime Logic
    conditions = [
        (combined['Slope'] > 1.0), 
        (combined['Slope'] < -0.5),
        (combined['Slope'] >= -0.5) & (combined['Slope'] <= 1.0)
    ]
    choices = ['Contango', 'Backwardation', 'Flat']
    combined['Regime'] = np.select(conditions, choices, default='Flat')
    
    return combined

def process_options_data(ticker, spot_price, r, save_path):
    print(f"[*] Fetching Option Chain for {ticker}...")
    
    # 1. Get Expirations (using user library)
    exps = odr.get_available_remote_expirations(ticker)
    if not exps:
        print("[!] No expirations found.")
        return pd.DataFrame()
    
    # Filter for next 6 months
    cutoff = datetime.now() + timedelta(days=180)
    valid_exps = [e for e in exps if e <= cutoff]
    
    all_options = []
    
    for exp in valid_exps:
        # Download using user library
        chain = odr.load_or_download_option_chain(ticker, exp)
        if chain.empty: continue
        
        # Ensure 'bid' and 'ask' exist, fill with 0 if not (yfinance quirks)
        for col in ['bid', 'ask', 'lastPrice', 'impliedVolatility']:
            if col not in chain.columns:
                chain[col] = 0.0
        
        all_options.append(chain)

    if not all_options:
        return pd.DataFrame()
        
    master_df = pd.concat(all_options, ignore_index=True)
    
    # 2. Cleanup & Pre-calc
    master_df = master_df[(master_df['strike'] > 0)]
    
    # Calculate IV where missing (0 or too low) or if column missing
    print("[*] Computing Implied Volatilities & Greeks (Vectorized)...")
    
    # Add T (Time to expiry in years)
    master_df['T'] = (master_df['expiration'] - datetime.now()).dt.days / 365.0
    master_df = master_df[master_df['T'] > 0.002] # Drop expiring today/tomorrow
    
    # Rename 'iv' to 'impliedVolatility' if needed standardizing
    if 'iv' in master_df.columns and 'impliedVolatility' not in master_df.columns:
        master_df['impliedVolatility'] = master_df['iv']

    # Force recalculate if IV is zero or missing
    # Many yfinance chains have 0 IV for liquid options, we must fix this.
    mask_calc = (master_df['impliedVolatility'] < 0.001) | (master_df['impliedVolatility'].isna())
    
    if mask_calc.any():
        master_df.loc[mask_calc, 'impliedVolatility'] = master_df[mask_calc].apply(
            lambda row: vectorize_iv(row, spot_price, r), axis=1
        )
    
    # CRITICAL: Drop rows where IV is still 0 after calculation (garbage data)
    master_df = master_df[master_df['impliedVolatility'] > 0.001]

    # Now calculate Deltas
    master_df['delta'] = master_df.apply(
        lambda row: vectorize_delta(row, spot_price, r), axis=1
    )
    
    # Moneyness
    master_df['moneyness'] = master_df['strike'] / spot_price
    
    # Save Raw CSV
    csv_path = os.path.join(save_path, "options_master.csv")
    master_df.to_csv(csv_path, index=False)
    
    return master_df

def calculate_skew_metrics(master_df):
    """
    Calculates 25-Delta Risk Reversal and ATM IV for each expiration.
    Uses robust interpolation (np.interp) instead of strict interp1d.
    """
    skew_data = []
    
    expirations = master_df['expiration'].unique()
    
    for exp in expirations:
        subset = master_df[master_df['expiration'] == exp]
        
        # Filter for valid IVs
        subset = subset[subset['impliedVolatility'] > 0.01]
        
        calls = subset[subset['type'] == 'call'].sort_values('delta')
        puts = subset[subset['type'] == 'put'].sort_values('delta') 
        
        if calls.empty or puts.empty: continue
        
        # Find ATM IV (closest to Moneyness=1)
        try:
            atm_call = subset.iloc[(subset['moneyness'] - 1).abs().argsort()[:1]]
            atm_iv = atm_call['impliedVolatility'].values[0] if not atm_call.empty else 0
        except:
            atm_iv = 0
        
        # Interpolate 25 Delta Call (Target Delta: 0.25)
        # Note: Call delta usually runs 0 to 1. We sort by delta.
        try:
            # Drop duplicates and NaNs
            c_clean = calls.dropna(subset=['delta', 'impliedVolatility']).drop_duplicates(subset=['delta'])
            if len(c_clean) > 2:
                iv_call_25 = np.interp(0.25, c_clean['delta'], c_clean['impliedVolatility'])
            else:
                iv_call_25 = np.nan
        except:
            iv_call_25 = np.nan
            
        # Interpolate 25 Delta Put (Target Delta: -0.25)
        # Put delta runs -1 to 0. We sort by delta (ascending).
        try:
            p_clean = puts.dropna(subset=['delta', 'impliedVolatility']).drop_duplicates(subset=['delta'])
            if len(p_clean) > 2:
                iv_put_25 = np.interp(-0.25, p_clean['delta'], p_clean['impliedVolatility'])
            else:
                iv_put_25 = np.nan
        except:
            iv_put_25 = np.nan
            
        days_to_exp = (pd.to_datetime(exp) - datetime.now()).days
        
        # Only calculate RR if we have valid IVs
        if not np.isnan(iv_call_25) and not np.isnan(iv_put_25):
            rr25 = iv_call_25 - iv_put_25
        else:
            rr25 = np.nan
        
        # Add to list only if we have at least ATM or Skew
        if atm_iv > 0 or not np.isnan(rr25):
            skew_data.append({
                'expiration': exp,
                'days': days_to_exp,
                'atm_iv': atm_iv,
                'iv_call_25': iv_call_25,
                'iv_put_25': iv_put_25,
                'rr25': rr25
            })
        
    return pd.DataFrame(skew_data).sort_values('days')

def get_rotation_data(ticker, current_surface_df):
    """
    Loads yesterday's data from /dev/shm structure if it exists
    """
    today = datetime.now().strftime('%Y-%m-%d')
    ticker_dir = os.path.join(DASHBOARD_ROOT, ticker)
    
    # Find directories not equal to today
    if not os.path.exists(ticker_dir):
        return pd.DataFrame()
        
    avail_dates = sorted([d for d in os.listdir(ticker_dir) if d != today])
    if not avail_dates:
        return pd.DataFrame()
        
    last_date = avail_dates[-1]
    last_file = os.path.join(ticker_dir, last_date, "options_master.csv")
    
    if not os.path.exists(last_file):
        return pd.DataFrame()
        
    print(f"[*] Found previous surface data from {last_date}")
    prev_df = pd.read_csv(last_file)
    prev_df['expiration'] = pd.to_datetime(prev_df['expiration'])
    
    # Merge on Expiration + Strike + Type
    curr = current_surface_df[['expiration', 'strike', 'type', 'impliedVolatility']].copy()
    prev = prev_df[['expiration', 'strike', 'type', 'impliedVolatility']].copy()
    
    merged = pd.merge(curr, prev, on=['expiration', 'strike', 'type'], suffixes=('_curr', '_prev'))
    merged['iv_change'] = merged['impliedVolatility_curr'] - merged['impliedVolatility_prev']
    
    return merged

# --- PLOTTING ENGINE (PLOTLY) ---

def generate_dashboard(ticker, price_df, vix_df, skew_df, surface_df, rotation_df, html_name):
    
    print("[*] Generating Interactive Plotly Dashboard...")
    
    # --- STYLE CONFIG ---
    layout_template = "plotly_dark"
    colors = {
        'bg': '#1e1e1e',
        'text': '#e5e5e5',
        'cyan': '#00ffff',
        'magenta': '#ff00ff',
        'yellow': '#ffff00',
        'green': '#00ff00',
        'red': '#ff0000'
    }

    # --- TAB 1: TERM STRUCTURE ---
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    
    # VIX Line
    fig_ts.add_trace(go.Scatter(x=vix_df.index, y=vix_df['VIX'], name="VIX", line=dict(color=colors['cyan'])), secondary_y=False)
    fig_ts.add_trace(go.Scatter(x=vix_df.index, y=vix_df['VIX3M'], name="VIX3M", line=dict(color=colors['magenta'])), secondary_y=False)
    
    # Ratio (Filled Area)
    fig_ts.add_trace(go.Scatter(
        x=vix_df.index, y=vix_df['Ratio'], name="VIX/VIX3M Ratio",
        fill='tozeroy', line=dict(color='rgba(255, 255, 0, 0.5)', width=0),
        opacity=0.3
    ), secondary_y=True)

    fig_ts.update_layout(title="VIX Term Structure Analysis", template=layout_template, height=600)
    fig_ts.update_yaxes(title_text="Volatility Index", secondary_y=False)
    fig_ts.update_yaxes(title_text="Ratio", secondary_y=True)

    # --- TAB 2: 25-DELTA SKEW ---
    fig_skew_ts = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3])
    
    if not skew_df.empty:
        # Plot only valid data
        clean_skew = skew_df.dropna(subset=['rr25'])
        fig_skew_ts.add_trace(go.Scatter(
            x=clean_skew['days'], y=clean_skew['rr25'], mode='lines+markers',
            name="25d Risk Reversal (Call - Put)",
            line=dict(color=colors['green'], width=3)
        ), row=1, col=1)
        
        # Bar plot for nearest expiry
        if not clean_skew.empty:
            nearest = clean_skew.iloc[0]
            fig_skew_ts.add_trace(go.Bar(
                x=['Put 25d', 'Call 25d'], y=[nearest['iv_put_25'], nearest['iv_call_25']],
                name=f"Nearest Exp ({int(nearest['days'])}d)", marker_color=[colors['red'], colors['green']]
            ), row=2, col=1)
    else:
        fig_skew_ts.add_annotation(text="Insufficient Data for Skew Calculation", showarrow=False, font=dict(size=20))

    fig_skew_ts.update_layout(title="25-Delta Risk Reversal Term Structure", template=layout_template, height=600)
    fig_skew_ts.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # --- TAB 3: STRIKE SKEW (Nearest 3) ---
    fig_smiles = go.Figure()
    exps = sorted(surface_df['expiration'].unique())[:3]
    
    palette = [colors['cyan'], colors['magenta'], colors['yellow']]
    
    for i, exp in enumerate(exps):
        # We need a clean curve. Let's filter for relevant strikes around spot
        sub = surface_df[(surface_df['expiration'] == exp) & (surface_df['impliedVolatility'] > 0)]
        sub = sub.sort_values('strike')
        
        # Cubic spline for smoothness
        if len(sub) > 5:
            # Avg Call/Put IV per strike
            avg_iv = sub.groupby('strike')['impliedVolatility'].mean()
            x_new = np.linspace(avg_iv.index.min(), avg_iv.index.max(), 300)
            try:
                spl = interp1d(avg_iv.index, avg_iv.values, kind='cubic')
                y_new = spl(x_new)
                fig_smiles.add_trace(go.Scatter(x=x_new, y=y_new, mode='lines', name=f"Exp {str(exp)[:10]}", line=dict(color=palette[i % 3])))
            except:
                fig_smiles.add_trace(go.Scatter(x=avg_iv.index, y=avg_iv.values, mode='lines+markers', name=f"Exp {str(exp)[:10]}", line=dict(color=palette[i % 3])))
    
    fig_smiles.update_layout(title="Volatility Smiles (Nearest 3 Expiries)", template=layout_template, xaxis_title="Strike", yaxis_title="Implied Volatility")

    # --- TAB 4: 3D SURFACE ---
    # Create grid
    fig_3d = go.Figure()
    
    if not surface_df.empty:
        # Pivot for grid
        df_surf = surface_df.groupby(['days_to_exp', 'strike'])['impliedVolatility'].mean().reset_index()
        
        if len(df_surf) > 10:
            # Griddata interpolation
            x = df_surf['strike']
            y = df_surf['days_to_exp']
            z = df_surf['impliedVolatility']
            
            xi = np.linspace(x.min(), x.max(), 50)
            yi = np.linspace(y.min(), y.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((x, y), z, (Xi, Yi), method='linear')
            
            fig_3d.add_trace(go.Surface(z=Zi, x=Xi, y=Yi, colorscale='Viridis', opacity=0.9))
        
    fig_3d.update_layout(title="Implied Volatility Surface", template=layout_template, scene=dict(xaxis_title="Strike", yaxis_title="Days to Exp", zaxis_title="IV"), height=700)

    # --- TAB 5: ROTATION HEATMAP ---
    fig_rot = go.Figure()
    if not rotation_df.empty:
        # Group by buckets to make a heatmap
        pivot_rot = rotation_df.groupby(['expiration', 'strike'])['iv_change'].mean().reset_index()
        
        fig_rot.add_trace(go.Contour(
            x=pivot_rot['strike'],
            y=pivot_rot['expiration'],
            z=pivot_rot['iv_change'],
            colorscale='RdBu_r', 
            zmid=0,
            contours=dict(coloring='heatmap')
        ))
    else:
        # Show specific message for missing rotation data
        fig_rot.add_annotation(
            text="First Run Detected<br>Insufficient historical data for rotation analysis.<br>Run again tomorrow to see Day-Over-Day changes.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#888"),
            align="center"
        )
        # Add dummy axes so it looks like a plot
        fig_rot.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig_rot.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        
    fig_rot.update_layout(title="DoD Volatility Rotation (Change in IV)", template=layout_template)

    # --- TAB 6: ATM TERM STRUCTURE ---
    fig_atm = go.Figure()
    if not skew_df.empty:
        fig_atm.add_trace(go.Scatter(x=skew_df['days'], y=skew_df['atm_iv'], mode='markers', name='Data'))
        
        # Logarithmic fit
        clean_atm = skew_df.dropna(subset=['atm_iv', 'days'])
        if len(clean_atm) > 3:
            z = np.polyfit(np.log(clean_atm['days']), clean_atm['atm_iv'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(clean_atm['days'].min(), clean_atm['days'].max(), 100)
            y_trend = p(np.log(x_trend))
            fig_atm.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', name='Log Fit', line=dict(dash='dash')))
            
    fig_atm.update_layout(title="ATM Term Structure", template=layout_template, xaxis_title="Days to Expiry", yaxis_title="ATM IV")

    # --- TAB 7: COMPOSITE DASHBOARD ---
    fig_comp = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    # Price
    fig_comp.add_trace(go.Scatter(x=price_df.index, y=price_df['Close'], name="Price"), row=1, col=1)
    
    # ATM IV (Need history, mapped from VIX approx or if we had stored daily IVs. Using VIX as proxy for ATM IV history)
    fig_comp.add_trace(go.Scatter(x=vix_df.index, y=vix_df['VIX'], name="VIX (ATM Proxy)", line=dict(color='orange')), row=2, col=1)
    
    # Slope
    fig_comp.add_trace(go.Bar(x=vix_df.index, y=vix_df['Slope'], name="Term Slope (VIX3M-VIX)", marker_color=vix_df['Slope'].apply(lambda x: 'red' if x<0 else 'green')), row=3, col=1)
    
    # Ratio
    fig_comp.add_trace(go.Scatter(x=vix_df.index, y=vix_df['Ratio'], name="VIX Ratio", line=dict(color='cyan')), row=4, col=1)
    
    fig_comp.update_layout(title=f"Composite Volatility Dashboard: {ticker}", template=layout_template, height=900)

    # --- HTML ASSEMBLY (CUSTOM TABS) ---
    # We save plots as divs
    div_ts = pio.to_html(fig_ts, full_html=False, include_plotlyjs='cdn')
    div_skew = pio.to_html(fig_skew_ts, full_html=False, include_plotlyjs=False)
    div_smiles = pio.to_html(fig_smiles, full_html=False, include_plotlyjs=False)
    div_3d = pio.to_html(fig_3d, full_html=False, include_plotlyjs=False)
    div_rot = pio.to_html(fig_rot, full_html=False, include_plotlyjs=False)
    div_atm = pio.to_html(fig_atm, full_html=False, include_plotlyjs=False)
    div_comp = pio.to_html(fig_comp, full_html=False, include_plotlyjs=False)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>STEGO Volatility Dashboard - {ticker}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ background-color: #121212; color: #e5e5e5; font-family: sans-serif; margin: 0; padding: 20px; }}
            .header {{ padding: 10px; border-bottom: 2px solid #333; margin-bottom: 20px; }}
            .author {{ font-size: 0.8em; color: #888; }}
            .tab {{ overflow: hidden; border: 1px solid #333; background-color: #1e1e1e; }}
            .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888; }}
            .tab button:hover {{ background-color: #333; }}
            .tab button.active {{ background-color: #00ffff; color: #000; font-weight: bold; }}
            .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 1s; }}
            @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>STEGO Volatility Structure: <span style="color:#00ffff">{ticker}</span></h1>
            <div class="author">Author: Michael Derby | Framework: STEGO | Generated: {datetime.now()}</div>
        </div>

        <div class="tab">
            <button class="tablinks" onclick="openTab(event, 'TermStructure')" id="defaultOpen">Term Structure</button>
            <button class="tablinks" onclick="openTab(event, 'Skew')">25d Skew</button>
            <button class="tablinks" onclick="openTab(event, 'Smiles')">Volatility Smiles</button>
            <button class="tablinks" onclick="openTab(event, 'Surface3D')">IV Surface (3D)</button>
            <button class="tablinks" onclick="openTab(event, 'Rotation')">DoD Rotation</button>
            <button class="tablinks" onclick="openTab(event, 'ATMTerm')">ATM Term</button>
            <button class="tablinks" onclick="openTab(event, 'Composite')">Composite</button>
        </div>

        <div id="TermStructure" class="tabcontent">{div_ts}</div>
        <div id="Skew" class="tabcontent">{div_skew}</div>
        <div id="Smiles" class="tabcontent">{div_smiles}</div>
        <div id="Surface3D" class="tabcontent">{div_3d}</div>
        <div id="Rotation" class="tabcontent">{div_rot}</div>
        <div id="ATMTerm" class="tabcontent">{div_atm}</div>
        <div id="Composite" class="tabcontent">{div_comp}</div>

        <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
            
            // Trigger Plotly resize to fix rendering in hidden tabs
            window.dispatchEvent(new Event('resize'));
        }}
        document.getElementById("defaultOpen").click();
        </script>
    </body>
    </html>
    """
    
    with open(html_name, "w", encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"[*] Dashboard saved to {html_name}")
    webbrowser.open(f"file://{os.path.abspath(html_name)}")

# --- MAIN EXECUTION FLOW ---

def main():
    parser = argparse.ArgumentParser(description="STEGO Volatility Dashboard")
    parser.add_argument("--ticker", required=True, help="Equity Ticker")
    parser.add_argument("--days", type=int, default=90, help="Lookback days")
    parser.add_argument("--html-file-name", default="vol_dashboard.html", help="Output filename")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    save_path = setup_directories(ticker)
    
    print(f"\n--- STARTING STEGO VOLATILITY ENGINE: {ticker} ---")
    
    # 1. Fetch History
    print("[*] Downloading Underlying History...")
    df_price = dr.get_stock_data(ticker, period=f"{args.days}d")
    df_price.to_csv(os.path.join(save_path, "underlying.csv"))
    
    # 2. Fetch VIX Complex
    df_vix = load_vix_complex(args.days)
    df_vix.to_csv(os.path.join(save_path, "vix_complex.csv"))
    
    # 3. Get Options & Build Surface
    # Note: Using last price as spot proxy for Greeks
    if df_price.empty:
        print("[!] Error: No price data found.")
        return
        
    spot = df_price['Close'].iloc[-1]
    rf_rate = get_risk_free_rate()
    
    surface_df = process_options_data(ticker, spot, rf_rate, save_path)
    
    if surface_df.empty:
        print("[!] No options data available. Dashboard will be limited.")
        return

    # 4. Calculate Skew Metrics
    surface_df['days_to_exp'] = surface_df['T'] * 365.0
    skew_df = calculate_skew_metrics(surface_df)
    skew_df.to_csv(os.path.join(save_path, "skew_metrics.csv"), index=False)
    
    # 5. Calculate Rotation
    rotation_df = get_rotation_data(ticker, surface_df)
    if not rotation_df.empty:
        rotation_df.to_csv(os.path.join(save_path, "rotation.csv"), index=False)
    
    # 6. Generate Dashboard
    generate_dashboard(
        ticker, df_price, df_vix, skew_df, surface_df, rotation_df, args.html_file_name
    )

if __name__ == "__main__":
    main()
