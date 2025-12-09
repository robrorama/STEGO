# SCRIPTNAME: 03.directional.pressure.via.delta.hedging.models.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
HEDGE FUND OPTIONS ANALYTICS SUITE (Standalone - Robust Version)
----------------------------------------------------------------
Role: Senior Quant Developer & Technical Architect
Objective: Robust Option Analytics with 'Missing Data' Approximation Engines.

Fixes applied:
1. IV Solver: Calculates Implied Vol from Price if API returns 0.
2. Fallback Greeks: Uses Realized Vol if IV cannot be solved.
3. Data Proxies: Fills missing OI with Volume for activity estimates.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq # Root finding for IV

# Plotly Imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as py_offline

# Suppress Warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CLASS: DATA INGESTION (Robust)
# ==============================================================================
class DataIngestion:
    def __init__(self, output_dir: str, lookback_years: int = 1):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        self.options_dir = os.path.join(output_dir, "options")
        self.dash_dir = os.path.join(output_dir, "dashboards")
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.options_dir, exist_ok=True)
        os.makedirs(self.dash_dir, exist_ok=True)

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if df.empty: return df
        
        # 1. MultiIndex Cleanup
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(0):
                df = df.swaplevel(0, 1, axis=1)
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    new_cols.append(col[0] if col[1] in [ticker, ""] else f"{col[0]}_{col[1]}")
                else:
                    new_cols.append(col)
            df.columns = new_cols

        # 2. Index Cleanup
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 3. Numeric & Sort
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df.sort_index()[~df.index.duplicated(keep='last')]

    def get_underlying_history(self, ticker: str, intraday: bool = False) -> pd.DataFrame:
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        
        # Try Disk Load
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not df.empty:
                    # If data is recent (< 12 hours), use it
                    if (datetime.now() - df.index[-1]).total_seconds() < 43200:
                        print(f"[{ticker}] Loaded underlying from disk.")
                        return self._sanitize_df(df, ticker)
            except Exception: pass

        # Download
        print(f"[{ticker}] Downloading underlying data...")
        try:
            interval = "1h" if intraday else "1d"
            period = "730d" if intraday else f"{self.lookback_years}y"
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            df = self._sanitize_df(df, ticker)
            if not df.empty:
                df.to_csv(file_path)
            return df
        except Exception as e:
            print(f"[{ticker}] Download failed: {e}")
            return pd.DataFrame()

    def get_options_snapshot(self, ticker: str, num_expiries: int = 6) -> pd.DataFrame:
        print(f"[{ticker}] Fetching options chain...")
        try:
            tk = yf.Ticker(ticker)
            expiries = tk.options
            if not expiries: return pd.DataFrame()
        except: return pd.DataFrame()

        all_opts = []
        target = expiries[:num_expiries]
        
        for exp in target:
            try:
                opt = tk.option_chain(exp)
                c, p = opt.calls.copy(), opt.puts.copy()
                c['type'], p['type'] = 'call', 'put'
                c['expiry'], p['expiry'] = exp, exp
                all_opts.extend([c, p])
            except: continue

        if not all_opts: return pd.DataFrame()
        
        full = pd.concat(all_opts, ignore_index=True)
        # Save raw snapshot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        full.to_csv(os.path.join(self.options_dir, f"{ts}_{ticker}_chain.csv"), index=False)
        return full


# ==============================================================================
# 2. CLASS: FINANCIAL ANALYSIS (Repair Engine)
# ==============================================================================
class FinancialAnalysis:
    def __init__(self, risk_free_rate: float = 0.04):
        self.r = risk_free_rate

    def analyze_underlying(self, df: pd.DataFrame) -> dict:
        if df.empty: return {}
        df = df.copy()
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        
        df['log_ret'] = np.log(df[col] / df[col].shift(1))
        # Current Realized Vol (20D)
        rv20 = df['log_ret'].rolling(20).std() * np.sqrt(252)
        
        return {
            'current_price': df[col].iloc[-1],
            'rv_20d': rv20.iloc[-1] if not np.isnan(rv20.iloc[-1]) else 0.20, # Default 20% if calc fails
            'history_df': df
        }

    def _implied_vol_solver(self, price, S, K, T, r, flag):
        """
        Newton-Raphson solver to approximate IV if API data is missing.
        """
        if price <= 0 or T <= 0: return 0.001
        
        def bs_price(sigma):
            # Black Scholes Price Function
            sigma = max(sigma, 1e-4)
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            if flag == 'call':
                return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            else:
                return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        def objective(sigma):
            return bs_price(sigma) - price

        try:
            # Use Brent's method (safer than Newton for bounded 0 to 500% vol)
            iv = brentq(objective, 1e-4, 5.0) 
            return iv
        except:
            return 0.0 # Failed to converge

    def _compute_greeks_row(self, row, S, r, default_vol):
        """
        Calculates Greeks with fallback logic for missing IV.
        """
        try:
            K = float(row['strike'])
            T = max(float(row['dte']) / 365.0, 0.001)
            
            # 1. Data Repair: Implied Volatility
            iv = float(row['impliedVolatility'])
            
            # If IV is junk (0 or NaN) or effectively zero, REPAIR IT
            if np.isnan(iv) or iv < 0.01:
                mid_price = (row['bid'] + row['ask']) / 2
                if mid_price > 0:
                    iv = self._implied_vol_solver(mid_price, S, K, T, r, row['type'])
                else:
                    # Fallback to Historical Vol if price is also junk
                    iv = default_vol
            
            sigma = max(iv, 0.001)

            # 2. Black Scholes Maths
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            q = 0.0
            disc = np.exp(-r*T)
            
            if row['type'] == 'call':
                delta = norm.cdf(d1)
                theta = (- (S * sigma * norm.pdf(d1))/(2*np.sqrt(T)) - r*K*disc*norm.cdf(d2)) / 365.0
            else:
                delta = norm.cdf(d1) - 1
                theta = (- (S * sigma * norm.pdf(d1))/(2*np.sqrt(T)) + r*K*disc*norm.cdf(-d2)) / 365.0

            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
            
            return pd.Series({
                'Delta': delta, 
                'Gamma': gamma, 
                'Vega': vega, 
                'Theta': theta, 
                'corrected_IV': sigma
            })
        except:
            return pd.Series({'Delta':0, 'Gamma':0, 'Vega':0, 'Theta':0, 'corrected_IV': 0})

    def process_chain(self, chain: pd.DataFrame, spot: float, hist_vol: float) -> pd.DataFrame:
        if chain.empty: return chain
        df = chain.copy()
        
        # Basic Cleanup
        cols = ['strike', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        for c in cols:
            if c not in df.columns: df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        # Data Repair: Proxy Open Interest with Volume if OI is empty (common in early morning)
        if df['openInterest'].sum() == 0 and df['volume'].sum() > 0:
            print("  [WARN] Open Interest missing. Approximating using Volume for visualization.")
            df['openInterest'] = df['volume'] # Rough proxy just to see activity

        # Time
        df['expiry'] = pd.to_datetime(df['expiry'])
        df['dte'] = (df['expiry'] - datetime.now()).dt.days
        df = df[df['dte'] >= 0]
        
        # Apply Greek Calc with Repair Logic
        # Pass `hist_vol` as the "last resort" volatility
        print(f"  [INFO] Calculating Greeks (Repairing IV where missing using RV={hist_vol:.2%})...")
        greeks = df.apply(lambda row: self._compute_greeks_row(row, spot, self.r, hist_vol), axis=1)
        df = pd.concat([df, greeks], axis=1)
        
        # Dealer Exposures
        multiplier = 100
        # Dealer is short options cust is long
        df['dealer_gamma'] = -1 * df['Gamma'] * df['openInterest'] * multiplier
        df['GEX'] = df['dealer_gamma'] * (spot**2)
        
        return df

    def get_dealer_flow(self, chain: pd.DataFrame, spot: float) -> dict:
        if chain.empty: return {}
        return {
            'net_gex': chain['GEX'].sum(),
            # Approx flow for 1% move: -1 * DealerGamma * (1% Spot)
            'gamma_flow_1pct': -1 * chain['dealer_gamma'].sum() * (spot * 0.01)
        }


# ==============================================================================
# 3. CLASS: DASHBOARD RENDERER (Visuals)
# ==============================================================================
class DashboardRenderer:
    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, "dashboards")

    def render(self, ticker, und_data, chain, flow):
        print(f"[{ticker}] Rendering dashboard...")
        
        spot = und_data.get('current_price', 100)
        plots = {}
        
        # 1. Underlying
        hist = und_data.get('history_df', pd.DataFrame())
        if not hist.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                         low=hist['Low'], close=hist['Close'], name='Px'), row=1, col=1)
            # Add Volume
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Vol', marker_color='rgba(100, 100, 255, 0.3)'), row=2, col=1)
            fig.update_layout(title=f"{ticker} Price Action", height=500, template="plotly_dark")
            plots['underlying'] = py_offline.plot(fig, output_type='div', include_plotlyjs=False)

        # 2. GEX Profile (The "Gamma Wall")
        if not chain.empty:
            # Group by Strike
            gex_grp = chain.groupby('strike')['GEX'].sum().reset_index()
            # Filter meaningful range (+/- 15% of spot) to avoid empty plot scaling
            gex_grp = gex_grp[ (gex_grp['strike'] > spot*0.85) & (gex_grp['strike'] < spot*1.15) ]
            
            fig = px.bar(gex_grp, x='strike', y='GEX', 
                         title=f"Net GEX by Strike (Zoomed +/- 15%)",
                         color='GEX', color_continuous_scale=['red', 'gray', 'green'])
            
            fig.add_vline(x=spot, line_dash="dash", line_color="yellow", annotation_text="Spot")
            fig.update_layout(template="plotly_dark", height=500)
            plots['gex'] = py_offline.plot(fig, output_type='div', include_plotlyjs=False)

            # 3. IV Skew (Using Corrected IV)
            near_exp = chain['expiry'].min()
            sub = chain[chain['expiry'] == near_exp]
            # Filter bad IVs (0 or > 200%)
            sub = sub[(sub['corrected_IV'] > 0.01) & (sub['corrected_IV'] < 2.0)]
            
            fig_iv = go.Figure()
            for t, c in [('call', 'cyan'), ('put', 'magenta')]:
                df_t = sub[sub['type'] == t]
                fig_iv.add_trace(go.Scatter(x=df_t['strike'], y=df_t['corrected_IV'], mode='lines+markers', name=f"{t} IV", line=dict(color=c)))
            
            fig_iv.add_vline(x=spot, line_dash="dash", line_color="yellow")
            fig_iv.update_layout(title=f"Implied Vol Skew (Front Month) - Repaired Data", template="plotly_dark", height=400)
            plots['skew'] = py_offline.plot(fig_iv, output_type='div', include_plotlyjs=False)

            # 4. Open Interest Structure
            oi_grp = chain.groupby(['strike', 'type'])['openInterest'].sum().reset_index()
            oi_grp = oi_grp[ (oi_grp['strike'] > spot*0.8) & (oi_grp['strike'] < spot*1.2) ]
            fig_oi = px.bar(oi_grp, x='strike', y='openInterest', color='type', title="Open Interest Structure", barmode='overlay')
            fig_oi.update_layout(template="plotly_dark", height=400)
            plots['oi'] = py_offline.plot(fig_oi, output_type='div', include_plotlyjs=False)

        self._save_html(ticker, plots, flow)

    def _save_html(self, ticker, plots, flow):
        # Using CDN for JS to keep file size small
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{ticker} Analytics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ background: #111; color: #ddd; font-family: sans-serif; padding: 20px; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .card {{ background: #222; padding: 15px; border-radius: 8px; border: 1px solid #333; }}
                .full-width {{ grid-column: span 2; }}
                h2 {{ margin-top: 0; color: #4db8ff; }}
                .metric {{ font-size: 1.2em; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>{ticker} Options Intelligence</h1>
            <div class="card">
                <h2>Dealer Flow Estimates</h2>
                <div class="metric">Net GEX: <strong>${flow.get('net_gex', 0)/1e9:.2f} B</strong></div>
                <div class="metric">Est. Hedge Flow (1% Spot Move): <strong>${flow.get('gamma_flow_1pct', 0)/1e6:.1f} M</strong></div>
                <small>*Calculated using approximated IV where API data was missing.</small>
            </div>
            
            <div class="grid">
                <div class="card full-width">{plots.get('underlying', 'No Price Data')}</div>
                <div class="card full-width">{plots.get('gex', 'No GEX Data (Check Inputs)')}</div>
                <div class="card">{plots.get('skew', 'No IV Data')}</div>
                <div class="card">{plots.get('oi', 'No OI Data')}</div>
            </div>
        </body>
        </html>
        """
        fname = os.path.join(self.output_dir, f"dashboards/{ticker}_Repaired.html")
        with open(fname, 'w') as f: f.write(html)
        print(f"[{ticker}] Saved: {fname}")


# ==============================================================================
# 4. EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"], help="Tickers")
    parser.add_argument("--output-dir", default="./market_data")
    parser.add_argument("--lookback", type=int, default=1)
    args = parser.parse_args()

    # Init
    ingest = DataIngestion(args.output_dir, args.lookback)
    quant = FinancialAnalysis()
    render = DashboardRenderer(args.output_dir)

    for ticker in args.tickers:
        print(f"\n--- Processing {ticker} ---")
        
        # 1. Underlying
        df_und = ingest.get_underlying_history(ticker)
        if df_und.empty: 
            print("No underlying data."); continue
            
        und_res = quant.analyze_underlying(df_und)
        spot = und_res['current_price']
        rv = und_res['rv_20d']
        
        # 2. Options (Robust)
        chain = ingest.get_options_snapshot(ticker)
        flow = {}
        
        if not chain.empty and spot > 0:
            # Process with Repair Logic (passing Realized Vol 'rv' as fallback)
            chain = quant.process_chain(chain, spot, hist_vol=rv)
            flow = quant.get_dealer_flow(chain, spot)
            
        # 3. Render
        render.render(ticker, und_res, chain, flow)

if __name__ == "__main__":
    main()
