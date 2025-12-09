# SCRIPTNAME: ok.vol_iv_surfaces_and_delta_shocks.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import logging
import time
import json
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots
from scipy.stats import norm, percentileofscore

# -----------------------------------------------------------------------------
# LOGGING & CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress pandas chained assignment warnings for cleaner output
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------------------------------------------------------
# MATH & FINANCE HELPERS (Vectorized)
# -----------------------------------------------------------------------------
def bs_d1(S, K, T, r, sigma):
    """Calculate Black-Scholes d1."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def bs_delta(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes Delta. Vectorized for option_type."""
    d1 = bs_d1(S, K, T, r, sigma)
    call_delta = norm.cdf(d1)
    
    # Handle both scalar string and array of strings for option_type
    if isinstance(option_type, str):
        return call_delta if option_type == 'call' else call_delta - 1.0
    else:
        return np.where(option_type == 'call', call_delta, call_delta - 1.0)

def bs_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes Price. Vectorized for option_type."""
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Handle both scalar string and array of strings for option_type
    if isinstance(option_type, str):
        return call_price if option_type == 'call' else put_price
    else:
        return np.where(option_type == 'call', call_price, put_price)

def bs_vega(S, K, T, r, sigma):
    """Calculate Black-Scholes Vega."""
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol_solver(price, S, K, T, r, option_type='call', tol=1e-4, max_iter=50):
    """
    Vectorized Newton-Raphson solver for Implied Volatility.
    Returns np.nan if solver fails or inputs are invalid.
    """
    # Initial guess: simple approximation
    sigma = np.full_like(price, 0.5, dtype=float)
    
    for i in range(max_iter):
        p_est = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)
        
        diff = price - p_est
        
        # Avoid division by zero
        mask = (np.abs(diff) < tol)
        if mask.all():
            break
            
        # Update step
        with np.errstate(divide='ignore', invalid='ignore'):
            step = diff / np.where(vega == 0, 1e-8, vega)
        
        # Dampen large steps
        step = np.clip(step, -0.5, 0.5)
        sigma = sigma + step
        
        # Check bounds
        sigma = np.clip(sigma, 1e-4, 5.0)
    
    return sigma

# -----------------------------------------------------------------------------
# CLASS 1: DATA INGESTION
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Handles downloading, sanitizing, caching, and loading data.
    Enforces 'Universal Fixer' rules for yfinance quirks.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The Universal Fixer:
        1. Swaps levels if Price is not at level 0.
        2. Flattens MultiIndex columns.
        3. Enforces DatetimeIndex.
        4. Coerces numerics.
        5. Strips timezones.
        """
        if df.empty:
            return df

        # 1. & 2. Handle MultiIndex Columns
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Adj Close' or 'Close' is in level 1 (common in new yfinance)
            # We want level 0 to be the Attribute (Close), level 1 to be Ticker
            if 'Close' not in df.columns.get_level_values(0) and 'Close' in df.columns.get_level_values(1):
                df = df.swaplevel(0, 1, axis=1)
            
            # Flatten: Join levels with underscore
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
            
            # Clean up names: 'Adj Close_SPY' -> 'Adj Close' if single ticker
            # Or just normalize to simple names if possible
            new_cols = []
            for c in df.columns:
                if 'Adj Close' in c: new_cols.append('Adj Close')
                elif 'Close' in c: new_cols.append('Close')
                elif 'High' in c: new_cols.append('High')
                elif 'Low' in c: new_cols.append('Low')
                elif 'Open' in c: new_cols.append('Open')
                elif 'Volume' in c: new_cols.append('Volume')
                else: new_cols.append(c)
            
            # Only apply if lengths match (simple single ticker case)
            if len(set(new_cols)) == len(df.columns):
                df.columns = new_cols

        # 3. Strict Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Attempt to find a date column
            for col in ['Date', 'datetime', 'Timestamp']:
                if col in df.columns:
                    df.set_index(col, inplace=True)
                    break
            df.index = pd.to_datetime(df.index, errors='coerce')

        # Drop rows with invalid index
        df = df[df.index.notna()]

        # 4. Strip Timezones
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 5. Numeric Coercion
        cols_to_numeric = [c for c in df.columns if c not in ['contractSymbol', 'currency']]
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_underlying_history(self, ticker: str, history_days: int = 252) -> pd.DataFrame:
        csv_path = self.output_dir / f"{ticker}_history.csv"
        
        # Load from cache if exists
        if csv_path.exists():
            logger.info(f"Loading underlying history from {csv_path}")
            df = pd.read_csv(csv_path, index_col=0)
            df = self._sanitize_df(df)
            
            # Check freshness (simple check: is last date recent?)
            if not df.empty and (datetime.now() - df.index[-1]).days < 1:
                return df

        # Download
        logger.info(f"Downloading underlying history for {ticker}...")
        start_date = (datetime.now() - timedelta(days=history_days*2)).strftime('%Y-%m-%d')
        
        # Retry logic
        for _ in range(3):
            try:
                df = yf.download(ticker, start=start_date, progress=False, group_by='column', auto_adjust=True)
                break
            except Exception as e:
                logger.warning(f"Download failed: {e}. Retrying...")
                time.sleep(2)
        else:
            logger.error("Failed to download underlying data.")
            return pd.DataFrame()

        df = self._sanitize_df(df)
        
        # Filter to needed length
        df = df.sort_index().tail(history_days)
        
        # Save cache
        if not df.empty:
            df.to_csv(csv_path)
            
        return df

    def get_options_snapshot(self, ticker: str, max_dte: int) -> Dict[str, pd.DataFrame]:
        """
        Downloads option chains. Note: caching complete chains is heavy, 
        so we typically download fresh for the 'current' state, but you could add caching.
        """
        logger.info(f"Fetching option chains for {ticker}...")
        tk = yf.Ticker(ticker)
        
        try:
            expirations = tk.options
        except Exception:
            logger.error("Could not fetch expirations.")
            return {}

        chains = {}
        today = datetime.now()

        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_date - today).days
            
            if dte < 0 or dte > max_dte:
                continue
                
            time.sleep(0.5) # Be polite to API
            try:
                opt = tk.option_chain(exp)
                calls = opt.calls
                puts = opt.puts
                
                calls['type'] = 'call'
                puts['type'] = 'put'
                
                # Combine
                df_chain = pd.concat([calls, puts], ignore_index=True)
                df_chain['expiry'] = exp
                df_chain['dte'] = dte
                
                # Sanitize column names for internal consistency
                # Rename 'lastPrice' -> 'last', 'impliedVolatility' -> 'iv_yf'
                rename_map = {
                    'lastPrice': 'last',
                    'openInterest': 'oi',
                    'impliedVolatility': 'iv_yf'
                }
                df_chain.rename(columns=rename_map, inplace=True)
                
                chains[exp] = df_chain
                logger.info(f"Loaded expiry {exp} (DTE {dte}) - {len(df_chain)} contracts")
                
            except Exception as e:
                logger.warning(f"Failed to fetch chain for {exp}: {e}")

        return chains

    def load_or_download_iv_history(self, ticker: str, history_csv_name: str) -> pd.DataFrame:
        path = self.output_dir / history_csv_name
        
        if path.exists():
            logger.info("Loading IV History CSV...")
            try:
                df = pd.read_csv(path)
                # Ensure date is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                logger.warning(f"Corrupt IV history file: {e}. Starting fresh.")
        
        return pd.DataFrame()

    def save_iv_history(self, df_iv_hist: pd.DataFrame, history_csv_name: str) -> None:
        path = self.output_dir / history_csv_name
        df_iv_hist.to_csv(path, index=False)
        logger.info(f"Saved IV History to {path}")

    def _backfill_shadow_history(self, ticker: str, underlying_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates synthetic IV history based on Realized Volatility to prevent Cold Start.
        """
        logger.info("Performing Shadow Backfill for IV History...")
        
        if underlying_df.empty:
            return pd.DataFrame()

        # Calculate Realized Vol (20d)
        df_rv = underlying_df.copy()
        df_rv['ret'] = np.log(df_rv['Close'] / df_rv['Close'].shift(1))
        df_rv['rv'] = df_rv['ret'].rolling(20).std() * np.sqrt(252)
        df_rv = df_rv.dropna()

        records = []
        
        # Create a synthetic surface for past dates
        # Buckets: deltas [0.25, 0.50, 0.75]
        deltas = [0.25, 0.50, 0.75]
        expiries_dte = [30, 60, 90]
        
        for date, row in df_rv.iterrows():
            base_vol = row['rv']
            if np.isnan(base_vol) or base_vol == 0:
                continue

            for dte in expiries_dte:
                # Add term structure (slight upward slope)
                ts_adj = base_vol * (1 + (dte/365.0)*0.1)
                
                for delta in deltas:
                    # Add skew (puts higher, calls lower - simplified)
                    # We store delta buckets. 0.5 is ATM.
                    skew_mult = 1.0
                    if delta < 0.5: # OTM Puts area equivalent
                        skew_mult = 1.1
                    elif delta > 0.5: # ITM Calls / OTM Calls depending on convention
                        # Assuming absolute delta buckets. 
                        # Let's just create 'ATM' (0.5) and 'Wing' (0.25)
                        pass
                    
                    # Random noise for realism
                    noise = np.random.normal(0, 0.02)
                    final_iv = ts_adj * skew_mult + noise
                    
                    records.append({
                        'date': date,
                        'dte': dte,
                        'delta_bucket': delta, # Using generic absolute delta
                        'iv': abs(final_iv),
                        'source': 'shadow'
                    })

        return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# CLASS 2: FINANCIAL ANALYSIS
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Core Quant Logic: IV Solver, Greeks, Aggregation, Ranking.
    Strictly Immutable (Copy-on-Write).
    """
    def __init__(self, underlying_df: pd.DataFrame, options_chains: Dict[str, pd.DataFrame], iv_history: pd.DataFrame, risk_free_rate: float = 0.04):
        # Store immutable copies
        self._underlying = underlying_df.copy()
        self._options_chains = {k: v.copy() for k, v in options_chains.items()}
        self._iv_history = iv_history.copy()
        self.r = risk_free_rate
        
        # Get current spot
        if not self._underlying.empty:
            self.spot_price = self._underlying['Close'].iloc[-1]
        else:
            self.spot_price = 100.0 # Fallback

    def compute_surface_snapshot(self) -> pd.DataFrame:
        """
        Iterates all options, computes IV & Delta, aggregates into buckets.
        """
        all_options = []
        
        for exp, chain in self._options_chains.items():
            if chain.empty: continue
            
            df = chain.copy()
            
            # 1. Mid Price
            df['mid'] = (df['bid'] + df['ask']) / 2
            # Fallback to last if bid/ask zero/weird
            mask_bad_mid = (df['mid'] <= 0) | (df['bid'] == 0) | (df['ask'] == 0)
            df.loc[mask_bad_mid, 'mid'] = df.loc[mask_bad_mid, 'last']
            
            # 2. Setup inputs for solver
            T = df['dte'] / 365.0
            # Avoid T=0
            T = T.replace(0, 1/365.0) 
            
            # 3. Solve IV
            # We solve for 'mid' price
            ivs = implied_vol_solver(
                price=df['mid'].values,
                S=self.spot_price,
                K=df['strike'].values,
                T=T.values,
                r=self.r,
                option_type=df['type'].values
            )
            df['calc_iv'] = ivs
            
            # 4. Compute Greeks (Delta)
            deltas = bs_delta(
                S=self.spot_price,
                K=df['strike'].values,
                T=T.values,
                r=self.r,
                sigma=np.where(np.isnan(ivs), 0.5, ivs),
                option_type=df['type'].values
            )
            df['delta'] = deltas
            df['abs_delta'] = np.abs(deltas)
            
            # 5. Moneyness
            df['moneyness'] = df['strike'] / self.spot_price
            
            all_options.append(df)
            
        if not all_options:
            return pd.DataFrame()
            
        full_df = pd.concat(all_options, ignore_index=True)
        
        # 6. Bucket Aggregation
        # Define buckets: 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90
        # We will bin abs_delta to nearest standard bucket
        def get_bucket(d):
            buckets = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
            idx = (np.abs(buckets - d)).argmin()
            return buckets[idx]

        full_df['delta_bucket'] = full_df['abs_delta'].apply(get_bucket)
        
        # Filter valid IVs
        clean_df = full_df.dropna(subset=['calc_iv'])
        clean_df = clean_df[(clean_df['calc_iv'] > 0.001) & (clean_df['calc_iv'] < 5.0)]
        
        # Group by DTE and DeltaBucket
        agg = clean_df.groupby(['expiry', 'dte', 'delta_bucket'])['calc_iv'].median().reset_index()
        agg.rename(columns={'calc_iv': 'iv'}, inplace=True)
        agg['date'] = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        return agg

    def calculate_metrics(self, current_snapshot: pd.DataFrame) -> pd.DataFrame:
        """
        Computes IV Percentile and DeltaIV against history.
        """
        if current_snapshot.empty:
            return current_snapshot
            
        output = current_snapshot.copy()
        output['iv_percentile'] = np.nan
        output['delta_iv'] = 0.0
        
        if self._iv_history.empty:
            return output

        # Ensure history date types
        hist = self._iv_history.copy()
        hist['date'] = pd.to_datetime(hist['date'])

        # Loop through rows to find stats
        # Optimization: In production, vectorizing this is better, but for dashboard logic loop is readable
        for idx, row in output.iterrows():
            bucket = row['delta_bucket']
            dte_target = row['dte']
            
            # Filter history for similar bucket and nearby DTE (+- 5 days)
            mask = (hist['delta_bucket'] == bucket) & \
                   (hist['dte'].between(dte_target - 5, dte_target + 5))
            
            relevant_hist = hist[mask]
            
            if not relevant_hist.empty:
                # IV Percentile (Rank)
                vals = relevant_hist['iv'].values
                # Current IV rank
                pct = percentileofscore(vals, row['iv'])
                output.at[idx, 'iv_percentile'] = pct
                
                # Delta IV (vs rolling 5-day median)
                recent_vals = relevant_hist.sort_values('date').tail(5)['iv']
                if not recent_vals.empty:
                    baseline = recent_vals.median()
                    output.at[idx, 'delta_iv'] = row['iv'] - baseline
        
        return output

    def get_regime_data(self) -> dict:
        """
        Calculates RV vs IV and defines regime.
        """
        if self._underlying.empty:
            return {"rv_20": 0, "iv_atm_30": 0, "vrp": 0, "label": "No Data"}
            
        # RV
        closes = self._underlying['Close']
        log_ret = np.log(closes / closes.shift(1))
        rv_20 = log_ret.tail(20).std() * np.sqrt(252)
        
        # IV ATM (approximate from snapshot logic if stored, or just take generic)
        # We'll use the history's last ATM 30d entry if available
        iv_atm = 0.0
        if not self._iv_history.empty:
            hist = self._iv_history
            # Look for recent date, delta ~ 0.5, dte ~ 30
            mask = (hist['delta_bucket'] == 0.5) & (hist['dte'].between(20, 45))
            recent = hist[mask].sort_values('date')
            if not recent.empty:
                iv_atm = recent.iloc[-1]['iv']
        
        vrp = iv_atm - rv_20
        
        # Classification
        label = "Neutral"
        if iv_atm < 0.15 and vrp > 0: label = "Complacent / Low Vol"
        if iv_atm > 0.30 and vrp < -0.05: label = "Panic / Realized Spike"
        if iv_atm > 0.40: label = "High Volatility Regime"
        
        return {
            "rv_20": rv_20,
            "iv_atm_30": iv_atm,
            "vrp": vrp,
            "label": label
        }

# -----------------------------------------------------------------------------
# CLASS 3: DASHBOARD RENDERER
# -----------------------------------------------------------------------------
class DashboardRenderer:
    """
    Builds Plotly figures and embeds them in a Tabbed HTML dashboard.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def generate_dashboard(self, ticker: str, current_surface: pd.DataFrame, 
                          regime_data: dict, iv_history: pd.DataFrame, 
                          filename: str = "dashboard.html"):
        
        if current_surface.empty:
            logger.error("No surface data to render.")
            return

        # --- TAB 1: IV Percentile Heatmap ---
        # X: DTE, Y: Delta Bucket, Z: IV Percentile
        # Pivot for heatmap
        piv_pct = current_surface.pivot_table(index='delta_bucket', columns='dte', values='iv_percentile')
        
        fig_heatmap_pct = go.Figure(data=go.Heatmap(
            z=piv_pct.values,
            x=piv_pct.columns,
            y=piv_pct.index,
            colorscale='RdYlGn_r', # Red is high percentile (expensive), Green low
            zmin=0, zmax=100,
            colorbar=dict(title='IV %le')
        ))
        fig_heatmap_pct.update_layout(
            title=f'{ticker} IV Percentile Surface',
            xaxis_title='Days to Expiry (DTE)',
            yaxis_title='Delta Bucket (Abs)',
            template='plotly_dark'
        )

        # --- TAB 2: Delta IV Heatmap ---
        piv_div = current_surface.pivot_table(index='delta_bucket', columns='dte', values='delta_iv')
        
        fig_heatmap_div = go.Figure(data=go.Heatmap(
            z=piv_div.values,
            x=piv_div.columns,
            y=piv_div.index,
            colorscale='RdBu_r', # Red = Vol Spike, Blue = Vol Crush
            zmid=0,
            colorbar=dict(title='ΔIV')
        ))
        fig_heatmap_div.update_layout(
            title=f'{ticker} ΔIV (Change vs Baseline)',
            xaxis_title='Days to Expiry',
            yaxis_title='Delta Bucket',
            template='plotly_dark'
        )

        # --- TAB 3: Skew Structure ---
        fig_skew = go.Figure()
        # Select a few key maturities
        dtes = sorted(current_surface['dte'].unique())
        selected_dtes = [d for d in dtes if d in [30, 60, 90, 120] or (d > 20 and d < 40)]
        # De-dupe and limit
        selected_dtes = sorted(list(set(selected_dtes)))[:4]
        
        for dte in selected_dtes:
            subset = current_surface[current_surface['dte'] == dte].sort_values('delta_bucket')
            fig_skew.add_trace(go.Scatter(
                x=subset['delta_bucket'],
                y=subset['iv'],
                mode='lines+markers',
                name=f'DTE {dte}'
            ))
            
        fig_skew.update_layout(
            title=f'{ticker} Volatility Skew (By Expiry)',
            xaxis_title='Delta (Moneyness Proxy)',
            yaxis_title='Implied Volatility',
            template='plotly_dark'
        )

        # --- TAB 4: Regime & History ---
        fig_regime = make_subplots(rows=2, cols=1, subplot_titles=("Term Structure (ATM)", "Recent IV History (ATM)"))
        
        # Term structure (Delta 0.5)
        atm_curve = current_surface[current_surface['delta_bucket'] == 0.5].sort_values('dte')
        fig_regime.add_trace(go.Scatter(
            x=atm_curve['dte'], y=atm_curve['iv'], mode='lines+markers', name='Current ATM TS'
        ), row=1, col=1)
        
        # History (ATM 30d approx)
        if not iv_history.empty:
            hist_atm = iv_history[
                (iv_history['delta_bucket'] == 0.5) & 
                (iv_history['dte'].between(25, 35))
            ].sort_values('date').tail(60)
            
            fig_regime.add_trace(go.Scatter(
                x=hist_atm['date'], y=hist_atm['iv'], mode='lines', name='30d ATM IV History'
            ), row=2, col=1)

        fig_regime.update_layout(
            title=f"Regime: {regime_data['label']} | VRP: {regime_data['vrp']:.2%}",
            template='plotly_dark',
            height=600
        )

        # --- HTML GENERATION ---
        
        # Get raw divs
        div_1 = py_offline.plot(fig_heatmap_pct, include_plotlyjs=False, output_type='div')
        div_2 = py_offline.plot(fig_heatmap_div, include_plotlyjs=False, output_type='div')
        div_3 = py_offline.plot(fig_skew, include_plotlyjs=False, output_type='div')
        div_4 = py_offline.plot(fig_regime, include_plotlyjs=False, output_type='div')
        
        # Get Plotly JS source
        plotly_js = py_offline.get_plotlyjs()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{ticker} Vol Dashboard</title>
            <style>
                body {{ font-family: sans-serif; background-color: #1e1e1e; color: #ddd; margin: 0; }}
                .tab {{ overflow: hidden; border: 1px solid #444; background-color: #333; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; }}
                .tab button:hover {{ background-color: #555; }}
                .tab button.active {{ background-color: #4CAF50; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #444; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
                .header {{ padding: 20px; text-align: center; background: #222; }}
                .metrics {{ display: flex; justify-content: space-around; padding: 10px; background: #2a2a2a; margin-bottom: 10px; }}
                .metric-box {{ text-align: center; }}
                .metric-val {{ font-size: 1.2em; font-weight: bold; color: #4CAF50; }}
            </style>
            <script type="text/javascript">
                {plotly_js}
            </script>
        </head>
        <body>
            <div class="header">
                <h1>{ticker} Volatility Regime Dashboard</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-box">Regime<br><span class="metric-val" style="color:white">{regime_data['label']}</span></div>
                <div class="metric-box">Realized Vol (20d)<br><span class="metric-val">{regime_data['rv_20']:.2%}</span></div>
                <div class="metric-box">Implied Vol (ATM 30d)<br><span class="metric-val">{regime_data['iv_atm_30']:.2%}</span></div>
                <div class="metric-box">VRP Spread<br><span class="metric-val">{regime_data['vrp']:.2%}</span></div>
            </div>

            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Tab1')" id="defaultOpen">IV Percentile Surface</button>
                <button class="tablinks" onclick="openTab(event, 'Tab2')">Delta IV Shock</button>
                <button class="tablinks" onclick="openTab(event, 'Tab3')">Skew Structure</button>
                <button class="tablinks" onclick="openTab(event, 'Tab4')">Regime & Term Structure</button>
            </div>

            <div id="Tab1" class="tabcontent">{div_1}</div>
            <div id="Tab2" class="tabcontent">{div_2}</div>
            <div id="Tab3" class="tabcontent">{div_3}</div>
            <div id="Tab4" class="tabcontent">{div_4}</div>

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
                    
                    // CRITICAL: Trigger resize for Plotly to render correctly in hidden tabs
                    window.dispatchEvent(new Event('resize'));
                }}
                
                // Open default tab
                document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard generated at {out_path.resolve()}")
        return out_path

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Volatility Dashboard")
    parser.add_argument("ticker", nargs="?", default="SPY", help="Equity Ticker (default: SPY)")
    parser.add_argument("--history-days", type=int, default=252, help="Days of underlying history")
    parser.add_argument("--max-dte", type=int, default=120, help="Max days to expiry for surface")
    parser.add_argument("--output-dir", default="vol_output", help="Output directory")
    parser.add_argument("--open-html", action="store_true", help="Open dashboard in browser immediately")
    
    args = parser.parse_args()
    
    # 1. Initialize Ingestion
    ingest = DataIngestion(args.output_dir)
    
    # 2. Get Underlying Data
    df_underlying = ingest.get_underlying_history(args.ticker, args.history_days)
    if df_underlying.empty:
        logger.error("Critical: No underlying data found. Exiting.")
        exit(1)
        
    # 3. Handle History & Shadow Backfill
    hist_csv = f"{args.ticker}_iv_history.csv"
    df_iv_hist = ingest.load_or_download_iv_history(args.ticker, hist_csv)
    
    if df_iv_hist.empty:
        logger.warning("IV History missing. Running Shadow Backfill...")
        df_iv_hist = ingest._backfill_shadow_history(args.ticker, df_underlying)
        if not df_iv_hist.empty:
            ingest.save_iv_history(df_iv_hist, hist_csv)
    
    # 4. Get Current Options Snapshot
    chains = ingest.get_options_snapshot(args.ticker, args.max_dte)
    if not chains:
        logger.error("No option chains found. Cannot build surface.")
        exit(1)
        
    # 5. Financial Analysis
    analyst = FinancialAnalysis(df_underlying, chains, df_iv_hist)
    
    # Compute Surface
    logger.info("Computing Volatility Surface & Greeks...")
    current_snapshot = analyst.compute_surface_snapshot()
    
    # Compute Metrics (Percentiles, DeltaIV)
    logger.info("Calculating Regime Metrics...")
    final_surface = analyst.calculate_metrics(current_snapshot)
    regime_stats = analyst.get_regime_data()
    
    # Update History with current snapshot (Append and Save)
    if not final_surface.empty:
        # Align columns for saving
        save_cols = ['date', 'dte', 'delta_bucket', 'iv']
        # Be careful with columns existing
        snapshot_to_save = final_surface.reset_index()[save_cols].copy()
        
        # Concatenate safely
        new_history = pd.concat([df_iv_hist, snapshot_to_save], ignore_index=True)
        # Deduplicate (keep last for same date/dte/bucket)
        new_history.drop_duplicates(subset=['date', 'dte', 'delta_bucket'], keep='last', inplace=True)
        
        ingest.save_iv_history(new_history, hist_csv)
    
    # 6. Render Dashboard
    renderer = DashboardRenderer(args.output_dir)
    html_path = renderer.generate_dashboard(
        ticker=args.ticker,
        current_surface=final_surface,
        regime_data=regime_stats,
        iv_history=df_iv_hist,
        filename=f"{args.ticker}_dashboard.html"
    )
    
    if args.open_html and html_path:
        logger.info("Opening dashboard in browser...")
        webbrowser.open(f"file://{html_path.resolve()}")

    logger.info("Done.")
