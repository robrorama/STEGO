#!/usr/bin/env python3
"""
Research-Grade Options Analytics Dashboard
------------------------------------------
A standalone tool for professional options research.
Features:
- Daily/Intraday equity analysis
- Volatility surface construction
- Skew & Structure metrics
- Finite-difference Numerical Greeks
- Interactive Plotly Dashboard

Usage:
    python3 options_research_dashboard.py --ticker NVDA --days 60 --compute-greeks --open-html
    python3 options_research_dashboard.py --ticker SPY --intraday 15m --days 5

Requirements:
    pip install yfinance pandas numpy plotly scipy
"""

import argparse
import os
import sys
import datetime
import time
import webbrowser
import warnings
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm, linregress

# Suppress warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. MATH & PRICING ENGINE (Black-Scholes & Numerical Greeks)
# ------------------------------------------------------------------------------

class PricingEngine:
    """
    Black-Scholes analytical pricer and Finite Difference Greeks engine.
    """
    @staticmethod
    def black_scholes_price(S, K, T, r, sigma, opt_type='call'):
        """Analytical Black-Scholes Price"""
        if T <= 0 or sigma <= 0:
            return np.maximum(0, S - K) if opt_type == 'call' else np.maximum(0, K - S)
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if opt_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    @staticmethod
    def calculate_numerical_greeks(row, S_spot, r=0.045):
        """
        Computes Greeks using Finite Differences.
        row must contain: strike, time_to_expiry, impliedVolatility, type
        """
        K = row['strike']
        T = row['time_to_expiry']
        sigma = row['impliedVolatility']
        opt_type = row['type']
        
        if pd.isna(sigma) or sigma <= 0 or T <= 0:
            return pd.Series(0, index=['delta','gamma','vega','theta','vomma','vanna','charm'])

        # Bump sizes
        dS = S_spot * 0.0025  # 0.25% spot bump
        dVol = 0.01           # 1% vol bump
        dT = 1/365.0          # 1 day time decay
        
        # Base Price
        P0 = PricingEngine.black_scholes_price(S_spot, K, T, r, sigma, opt_type)
        
        # --- Delta & Gamma (Spot Bumps) ---
        P_up = PricingEngine.black_scholes_price(S_spot + dS, K, T, r, sigma, opt_type)
        P_dn = PricingEngine.black_scholes_price(S_spot - dS, K, T, r, sigma, opt_type)
        
        delta = (P_up - P_dn) / (2 * dS)
        gamma = (P_up - 2*P0 + P_dn) / (dS ** 2)
        
        # --- Vega & Vomma (Vol Bumps) ---
        P_v_up = PricingEngine.black_scholes_price(S_spot, K, T, r, sigma + dVol, opt_type)
        P_v_dn = PricingEngine.black_scholes_price(S_spot, K, T, r, sigma - dVol, opt_type)
        
        vega = (P_v_up - P_v_dn) / (2 * dVol) # Sensitivity to 100% vol change mathematically, often scaled /100
        # Usually Vega is reported as change per 1% vol. The finite diff gives change per unit (1.0).
        # We will scale Vega to be "change per 1% vol"
        vega = vega / 100.0 
        
        vomma = (P_v_up - 2*P0 + P_v_dn) / (dVol ** 2) / 100.0 # Vomma per 1%? Let's keep consistent.
        
        # --- Theta (Time Decay) ---
        # Theta is change as time PASSES. T becomes T - dT.
        T_new = max(1e-5, T - dT)
        P_t = PricingEngine.black_scholes_price(S_spot, K, T_new, r, sigma, opt_type)
        theta = (P_t - P0) / dT # Annualized theta? Usually per day.
        # Since dT is 1/365, this results in change per year? 
        # (P_new - P_old) is the change for 1 day. 
        # If we divide by dT (years), we get annualized Theta.
        # Traders usually want "Theta per day".
        theta_per_day = (P_t - P0) 
        
        # --- Vanna (dDelta / dVol) ---
        # Calculate Delta at sigma + dVol
        # P(S+dS, vol+dVol)
        P_up_v_up = PricingEngine.black_scholes_price(S_spot + dS, K, T, r, sigma + dVol, opt_type)
        P_dn_v_up = PricingEngine.black_scholes_price(S_spot - dS, K, T, r, sigma + dVol, opt_type)
        delta_v_up = (P_up_v_up - P_dn_v_up) / (2 * dS)
        
        # P(S+dS, vol-dVol)
        P_up_v_dn = PricingEngine.black_scholes_price(S_spot + dS, K, T, r, sigma - dVol, opt_type)
        P_dn_v_dn = PricingEngine.black_scholes_price(S_spot - dS, K, T, r, sigma - dVol, opt_type)
        delta_v_dn = (P_up_v_dn - P_dn_v_dn) / (2 * dS)
        
        vanna = (delta_v_up - delta_v_dn) / (2 * dVol) / 100.0 # Scaled for 1% vol
        
        # --- Charm (dDelta / dTime) ---
        # Calculate Delta at T - dT
        P_up_t = PricingEngine.black_scholes_price(S_spot + dS, K, T_new, r, sigma, opt_type)
        P_dn_t = PricingEngine.black_scholes_price(S_spot - dS, K, T_new, r, sigma, opt_type)
        delta_t = (P_up_t - P_dn_t) / (2 * dS)
        
        charm = (delta_t - delta) # Change in delta per day
        
        return pd.Series([delta, gamma, vega, theta_per_day, vomma, vanna, charm], 
                         index=['delta','gamma','vega','theta','vomma','vanna','charm'])

# ------------------------------------------------------------------------------
# 2. DATA UTILITIES
# ------------------------------------------------------------------------------

def robust_extract_close(df: pd.DataFrame) -> pd.Series:
    if df.empty: return pd.Series()
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.droplevel(1)
        except: pass
    if 'Close' in df.columns: return df['Close']
    if 'Adj Close' in df.columns: return df['Adj Close']
    return df.iloc[:, 0]

def force_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

class DataFetcher:
    def __init__(self, ticker, days, intraday_interval=None):
        self.ticker = ticker.upper()
        self.days = days
        self.intraday_interval = intraday_interval
        self.spot_price = 0.0
        self.hist_data = pd.DataFrame()
        self.options_data = pd.DataFrame()
        self.ticker_obj = yf.Ticker(self.ticker)

    def fetch_equity(self):
        print(f"[*] Fetching Equity Data for {self.ticker}...")
        interval = self.intraday_interval if self.intraday_interval else "1d"
        
        # Handle "max days" constraints for intraday roughly
        # 1m = 7d, 5m = 60d, 15m = 60d usually on YF
        period_arg = f"{self.days}d"
        
        try:
            df = self.ticker_obj.history(period=period_arg, interval=interval)
            df = force_tz_naive(df)
            
            if df.empty:
                # Fallback
                df = yf.download(self.ticker, period=period_arg, interval=interval, progress=False)
                df = force_tz_naive(df)
            
            if df.empty:
                print("[!] Equity download failed.")
                return

            df['Close'] = robust_extract_close(df)
            
            # Metrics
            df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Return'] = df['Close'].pct_change()
            
            # Realized Vol
            for w in [10, 20, 60]:
                df[f'RV_{w}'] = df['LogRet'].rolling(w).std() * np.sqrt(252 if not self.intraday_interval else 252*78) * 100
                
            # ATR
            high = df['High'] if 'High' in df.columns else df['Close']
            low = df['Low'] if 'Low' in df.columns else df['Close']
            prev_close = df['Close'].shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
            
            # Volume Z-Score
            if 'Volume' in df.columns:
                v_mean = df['Volume'].rolling(20).mean()
                v_std = df['Volume'].rolling(20).std()
                df['Vol_Z'] = (df['Volume'] - v_mean) / (v_std + 1e-9)
            
            self.hist_data = df
            self.spot_price = df['Close'].iloc[-1]
            print(f"    > Spot: {self.spot_price:.2f} | Rows: {len(df)}")
            
        except Exception as e:
            print(f"[!] Error fetching equity: {e}")

    def fetch_options(self):
        print(f"[*] Fetching Option Chain...")
        try:
            exps = self.ticker_obj.options
            if not exps:
                print("[!] No expirations found.")
                return
        except:
            return

        all_opts = []
        now = datetime.datetime.now()
        
        # Limit to first 12 expirations to ensure speed/reliability
        target_exps = exps[:12]
        
        for e_str in target_exps:
            try:
                time.sleep(0.1) # Rate limit protection
                chain = self.ticker_obj.option_chain(e_str)
                exp_date = pd.to_datetime(e_str)
                days_to_exp = (exp_date - now).days
                if days_to_exp < 0: days_to_exp = 0
                T_years = (days_to_exp + 1) / 365.0
                
                # Process Calls
                c = chain.calls.copy()
                c['type'] = 'call'
                
                # Process Puts
                p = chain.puts.copy()
                p['type'] = 'put'
                
                combined = pd.concat([c, p])
                combined['expiry'] = exp_date
                combined['expiry_str'] = e_str
                combined['time_to_expiry'] = T_years
                combined['mid_price'] = (combined['bid'] + combined['ask']) / 2
                combined['moneyness'] = combined['strike'] / self.spot_price
                
                # Clean IV
                combined['impliedVolatility'] = pd.to_numeric(combined['impliedVolatility'], errors='coerce')
                combined = combined[combined['impliedVolatility'] > 0.001]
                
                all_opts.append(combined)
            except:
                continue
                
        if all_opts:
            self.options_data = pd.concat(all_opts)
            print(f"    > Loaded {len(self.options_data)} contracts.")

# ------------------------------------------------------------------------------
# 3. ANALYTICS ENGINE
# ------------------------------------------------------------------------------

class AnalyticsEngine:
    def __init__(self, df_opts, spot):
        self.df = df_opts
        self.spot = spot
        self.vol_surface = pd.DataFrame()
        self.term_structure = pd.DataFrame()
        
    def compute_structure_metrics(self):
        if self.df.empty: return
        
        print("[*] Computing Vol Surface & Structure Metrics...")
        
        # 1. Delta Computation (Analytical for sorting/grouping)
        # We need generic delta for buckets, even if we do numerical Greeks later
        r = 0.045
        # Removed redundant call to PricingEngine.black_scholes_price for delta estimation
        
        # Re-use d1 from pricing engine logic? Let's just do a quick analytical delta here.
        def quick_delta(row):
            # Define d1 locally
            S, K, T, sigma = self.spot, row['strike'], row['time_to_expiry'], row['impliedVolatility']
            if T<=0 or sigma<=0: return 0
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return norm.cdf(d1) if row['type'] == 'call' else norm.cdf(d1) - 1
            
        self.df['bs_delta'] = self.df.apply(quick_delta, axis=1)
        
        # 2. Term Structure & Smile Metrics
        groups = self.df.groupby('expiry')
        ts_res = []
        
        for exp, grp in groups:
            # ATM IV
            grp['dist'] = abs(grp['strike'] - self.spot)
            atm = grp.loc[grp['dist'].idxmin()]
            atm_iv = atm['impliedVolatility']
            
            # 25 Delta Risk Reversal & Fly
            calls = grp[grp['type'] == 'call']
            puts = grp[grp['type'] == 'put']
            
            # Find 25d Call
            c25 = calls.iloc[(calls['bs_delta'] - 0.25).abs().argsort()[:1]]
            # Find 25d Put (Delta is -0.25)
            p25 = puts.iloc[(puts['bs_delta'] - (-0.25)).abs().argsort()[:1]]
            
            rr = np.nan
            fly = np.nan
            
            if not c25.empty and not p25.empty:
                iv_c = c25['impliedVolatility'].values[0]
                iv_p = p25['impliedVolatility'].values[0]
                rr = iv_c - iv_p
                fly = ((iv_c + iv_p)/2) - atm_iv
                
            # Curvature (Quadratic fit of IV vs Strike)
            # IV ~ a*K^2 + b*K + c. 'a' is curvature.
            try:
                z = np.polyfit(grp['strike'], grp['impliedVolatility'], 2)
                curvature = z[0] * 10000 # Scale up
            except:
                curvature = 0
                
            ts_res.append({
                'expiry': exp,
                'TTE': grp['time_to_expiry'].iloc[0],
                'ATM_IV': atm_iv,
                'RR_25d': rr,
                'Fly_25d': fly,
                'Curvature': curvature
            })
            
        self.term_structure = pd.DataFrame(ts_res).sort_values('expiry')

    def compute_numerical_greeks(self):
        print("[*] Computing Numerical Greeks (Finite Difference)...")
        # Apply row-wise
        greeks = self.df.apply(
            lambda row: PricingEngine.calculate_numerical_greeks(row, self.spot), axis=1
        )
        self.df = pd.concat([self.df, greeks], axis=1)

# ------------------------------------------------------------------------------
# 4. VISUALIZATION ENGINE
# ------------------------------------------------------------------------------

class VisualizationEngine:
    def __init__(self, data_fetcher: DataFetcher, analytics: AnalyticsEngine, output_dir="."):
        self.df_hist = data_fetcher.hist_data
        self.df_opts = analytics.df
        self.ts = analytics.term_structure
        self.ticker = data_fetcher.ticker
        self.spot = data_fetcher.spot_price
        self.out_dir = output_dir

    def generate_dashboard(self, filename="options_research.html"):
        print("[*] Generating Plotly Dashboard...")
        
        fig = make_subplots(
            rows=4, cols=3,
            specs=[
                [{"type": "xy", "colspan": 2}, None, {"type": "xy"}], # Price (Wide), Vol
                [{"type": "surface", "rowspan": 2}, {"type": "xy"}, {"type": "xy"}], # 3D Surface, Smile, Term Struct
                [None, {"type": "xy"}, {"type": "xy"}], # (Surface cont.), RR, Curvature
                [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "xy"}] # Greeks Heatmaps, Greek Line
            ],
            subplot_titles=(
                f"{self.ticker} Price Structure", "Realized Volatility",
                "Volatility Surface (Strike vs TTE)", "Volatility Smiles (by Expiry)", "ATM Term Structure",
                "25d Risk Reversal (Skew)", "Smile Curvature",
                "Gamma Heatmap", "Vanna Heatmap", "Delta Exposure Profile"
            ),
            vertical_spacing=0.08, horizontal_spacing=0.05
        )
        
        # 1. Price Chart
        fig.add_trace(go.Candlestick(
            x=self.df_hist.index,
            open=self.df_hist['Open'], high=self.df_hist['High'],
            low=self.df_hist['Low'], close=self.df_hist['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add SMAs
        for w in [20, 50]:
            sma = self.df_hist['Close'].rolling(w).mean()
            fig.add_trace(go.Scatter(x=self.df_hist.index, y=sma, name=f'SMA {w}', line=dict(width=1)), row=1, col=1)

        # 2. Realized Vol
        if 'RV_20' in self.df_hist.columns:
            fig.add_trace(go.Scatter(x=self.df_hist.index, y=self.df_hist['RV_20'], name='RV 20', line=dict(color='orange')), row=1, col=3)

        # 3. 3D Vol Surface
        # Pivot table: Index=Strike, Cols=TTE, Values=IV
        # We need to bin strikes to make a regular grid or use mesh3d
        if not self.df_opts.empty:
            surf_data = self.df_opts[self.df_opts['moneyness'].between(0.8, 1.2)]
            fig.add_trace(go.Mesh3d(
                x=surf_data['strike'],
                y=surf_data['time_to_expiry'],
                z=surf_data['impliedVolatility'],
                intensity=surf_data['impliedVolatility'],
                colorscale='Viridis',
                opacity=0.8,
                name='Vol Surface'
            ), row=2, col=1)

            # 4. Smiles (Line Plot)
            # Plot top 5 liquid expiries
            exps = self.df_opts['expiry_str'].unique()[:5]
            for e in exps:
                d = self.df_opts[self.df_opts['expiry_str'] == e]
                d = d.sort_values('strike')
                # Filter weird IVs
                d = d[d['moneyness'].between(0.7, 1.3)]
                fig.add_trace(go.Scatter(
                    x=d['strike'], y=d['impliedVolatility'],
                    mode='lines', name=f'Smile {e}'
                ), row=2, col=2)

        # 5. Term Structure
        if not self.ts.empty:
            fig.add_trace(go.Scatter(
                x=self.ts['expiry'], y=self.ts['ATM_IV'],
                mode='lines+markers', name='ATM Term', line=dict(color='cyan')
            ), row=2, col=3)

            # 6. Risk Reversal
            fig.add_trace(go.Scatter(
                x=self.ts['expiry'], y=self.ts['RR_25d'],
                mode='lines+markers', name='25d RR', line=dict(color='red')
            ), row=3, col=2)

            # 7. Curvature
            fig.add_trace(go.Scatter(
                x=self.ts['expiry'], y=self.ts['Curvature'],
                mode='lines+markers', name='Curvature', line=dict(color='purple')
            ), row=3, col=3)

        # 8. Greeks Heatmaps (if computed)
        if 'gamma' in self.df_opts.columns:
            # Aggregate Gamma by Strike & Expiry
            # Bin strikes for cleaner map
            self.df_opts['strike_bin'] = (self.df_opts['strike'] / 5).round() * 5
            
            # Gamma Map
            g_map = self.df_opts.groupby(['expiry_str', 'strike_bin'])['gamma'].mean().unstack(level=0)
            # Limit range
            spot_r = self.spot
            g_map = g_map.loc[spot_r*0.8 : spot_r*1.2]
            
            fig.add_trace(go.Heatmap(
                z=g_map.values, x=g_map.columns, y=g_map.index,
                colorscale='Plasma', name='Gamma Map', showscale=False
            ), row=4, col=1)

            # Vanna Map
            v_map = self.df_opts.groupby(['expiry_str', 'strike_bin'])['vanna'].mean().unstack(level=0)
            v_map = v_map.loc[spot_r*0.8 : spot_r*1.2]
            
            fig.add_trace(go.Heatmap(
                z=v_map.values, x=v_map.columns, y=v_map.index,
                colorscale='RdBu', zmid=0, name='Vanna Map', showscale=False
            ), row=4, col=2)

            # Delta Exposure (Line) - Sum of Delta per strike bucket? 
            # Or just Plot Delta vs Strike for nearest expiry
            near_exp = exps[0] if len(exps) > 0 else None
            if near_exp:
                d_near = self.df_opts[self.df_opts['expiry_str'] == near_exp].sort_values('strike')
                fig.add_trace(go.Scatter(
                    x=d_near['strike'], y=d_near['delta'],
                    mode='lines', name=f'Delta ({near_exp})'
                ), row=4, col=3)

        fig.update_layout(
            template='plotly_white',
            height=1400, width=1600,
            title_text=f"Research Dashboard: {self.ticker} | Spot: {self.spot:.2f}",
            showlegend=True
        )
        
        out_path = os.path.join(self.out_dir, filename)
        fig.write_html(out_path)
        print(f"[SUCCESS] Dashboard saved to {out_path}")
        return out_path

# ------------------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Options Research Dashboard")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--intraday", type=str, default=None, help="e.g. 5m, 15m")
    parser.add_argument("--compute-greeks", action="store_true")
    parser.add_argument("--open-html", action="store_true")
    
    args = parser.parse_args()
    
    # 1. Fetch
    fetcher = DataFetcher(args.ticker, args.days, args.intraday)
    fetcher.fetch_equity()
    fetcher.fetch_options()
    
    if fetcher.options_data.empty or fetcher.hist_data.empty:
        print("[!] Insufficient data to proceed.")
        sys.exit(1)
        
    # 2. Analyze
    analytics = AnalyticsEngine(fetcher.options_data, fetcher.spot_price)
    analytics.compute_structure_metrics()
    
    if args.compute_greeks:
        analytics.compute_numerical_greeks()
        
    # 3. Visualize
    viz = VisualizationEngine(fetcher, analytics)
    html_path = viz.generate_dashboard(f"OPTIONS_DASHBOARD_{args.ticker}.html")
    
    # 4. Save Data
    fetcher.hist_data.to_csv(f"{args.ticker}_ohlcv.csv")
    analytics.df.to_csv(f"{args.ticker}_options_chain.csv")
    analytics.term_structure.to_csv(f"{args.ticker}_vol_structure.csv")
    
    if args.open_html:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    main()

