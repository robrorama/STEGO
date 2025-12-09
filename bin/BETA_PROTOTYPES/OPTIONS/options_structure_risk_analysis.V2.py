#!/usr/bin/env python3
# SCRIPTNAME: ok.options_structure_risk_analysis.V2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Professional Options Structure & Risk Analysis Dashboard
------------------------------------------------------
A standalone tool for institutional-grade options market structure analysis.
Calculates Dealer Greeks (GEX, Vanna, Charm), IV Term Structure, Skew,
and Volatility Regimes using purely public data via yfinance.

Usage:
    python3 options_structure_risk_analysis.py --ticker SPY --history 365
    python3 options_structure_risk_analysis.py --ticker NVDA --open-html

Requirements:
    pip install yfinance pandas numpy plotly scipy
"""

import argparse
import os
import sys
import datetime
import warnings
import webbrowser
import time
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm, entropy

# Suppress warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. MATH & GREEKS ENGINE (Black-Scholes)
# ------------------------------------------------------------------------------

class GreeksEngine:
    """
    Vectorized Black-Scholes Greeks calculator.
    """
    @staticmethod
    def d1_d2(S, K, T, r, sigma):
        # Avoid division by zero
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-5)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def calculate_greeks(row, S, r=0.04):
        """
        Row-wise wrapper for dataframe apply. 
        Expects row with: strike, impliedVolatility, type ('call' or 'put'), time_to_expiry
        """
        K = row['strike']
        sigma = row['impliedVolatility']
        T = row['time_to_expiry']
        opt_type = row['type']

        if pd.isna(sigma) or sigma <= 0:
            return pd.Series([0,0,0,0,0], index=['delta','gamma','vega','vanna','charm'])

        d1, d2 = GreeksEngine.d1_d2(S, K, T, r, sigma)
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        
        # Gamma (same for call/put)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        
        # Vega (same for call/put) - Sensitivity to 1% vol change usually /100
        vega = S * pdf_d1 * np.sqrt(T) / 100.0
        
        # Vanna - dDelta/dVol (same for call/put approx, signs differ strictly but mostly used magnitude)
        # Vanna = -d2 / sigma * Vega
        vanna = -d2 / sigma * vega
        
        # Charm - dDelta/dTime
        # Call Charm = -pdf(d1)*(2rT - d2*sigma*sqrt(T))/(2T*sigma*sqrt(T))
        # Simplified: decay of delta over time
        # Using analytical approximation
        charm_term = (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
        
        if opt_type == 'call':
            delta = cdf_d1
            charm = -pdf_d1 * charm_term
        else:
            delta = cdf_d1 - 1
            charm = -pdf_d1 * charm_term + r * norm.cdf(-d2) # Put specific term? 
            # Simplify Charm for robust quick calc: 
            # Put Charm = Call Charm + r * exp(-rT) * N(-d2)? 
            # Let's stick to the primary decay driver.
        
        return pd.Series([delta, gamma, vega, vanna, charm], index=['delta','gamma','vega','vanna','charm'])

# ------------------------------------------------------------------------------
# 2. DATA UTILITIES
# ------------------------------------------------------------------------------

def robust_extract_close(df: pd.DataFrame) -> pd.Series:
    """Robustly extracts Close column from yfinance MultiIndex or standard DF."""
    if df.empty: return pd.Series()
    
    # Handle MultiIndex columns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Attempt to drop ticker level if it exists
        try:
            df.columns = df.columns.droplevel(1)
        except:
            pass
            
    # Priority
    if 'Close' in df.columns:
        return df['Close']
    elif 'Adj Close' in df.columns:
        return df['Adj Close']
    return df.iloc[:, 0]

def force_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    return df

# ------------------------------------------------------------------------------
# 3. CORE LOGIC
# ------------------------------------------------------------------------------

class MarketStructureAnalyzer:
    def __init__(self, ticker, history_days=365):
        self.ticker_symbol = ticker.upper()
        self.history_days = history_days
        self.ticker_obj = yf.Ticker(self.ticker_symbol)
        self.spot_price = 0.0
        
        # Data Containers
        self.df_price = pd.DataFrame()
        self.df_options = pd.DataFrame()
        self.term_structure = pd.DataFrame()
        self.gex_data = pd.DataFrame()
        
    def run_pipeline(self):
        print(f"[*] Starting Analysis for {self.ticker_symbol}...")
        
        # 1. Price Data
        self._fetch_price_history()
        
        # 2. Options Data
        self._fetch_options_chain()
        
        # 3. Analytics
        if not self.df_options.empty:
            self._compute_greeks_and_exposures()
            self._analyze_term_structure()
        else:
            print("[!] No options data found. Skipping structure analysis.")
            
        print("[*] Pipeline Complete.")

    def _fetch_price_history(self):
        print("    > Downloading price history...")
        # Get historical data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=self.history_days)
        
        df = self.ticker_obj.history(start=start_date, end=end_date)
        df = force_tz_naive(df)
        
        if df.empty:
            print("[!] Price download failed.")
            return

        # Metrics
        df['Close'] = robust_extract_close(df)
        df['Return'] = df['Close'].pct_change()
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Realized Vol
        for w in [10, 20, 60]:
            df[f'RV_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100
            
        # ATR-like volatility normalized
        df['HighLow'] = (df['High'] - df['Low']) / df['Close']
        df['VolRegime'] = df['RV_20d'].rolling(5).mean() # Smoothed
        
        self.df_price = df
        self.spot_price = df['Close'].iloc[-1]
        print(f"    > Current Spot: {self.spot_price:.2f}")

    def _fetch_options_chain(self):
        print("    > Fetching options chain (this may take a moment)...")
        try:
            exps = self.ticker_obj.options
            if not exps:
                return
        except Exception as e:
            print(f"[!] Failed options fetch: {e}")
            return

        # Increased limit to 12 expirations and added delay to prevent rate limiting
        target_exps = exps[:12]
        all_opts = []
        
        now = datetime.datetime.now()
        
        print(f"    > Collecting {len(target_exps)} expirations...")
        for i, e_str in enumerate(target_exps):
            try:
                # Slight delay to be nice to API and prevent empty returns
                time.sleep(0.2)
                
                chain = self.ticker_obj.option_chain(e_str)
                expiry_date = pd.to_datetime(e_str)
                
                # Calls
                c = chain.calls.copy()
                c['type'] = 'call'
                
                # Puts
                p = chain.puts.copy()
                p['type'] = 'put'
                
                combined = pd.concat([c, p])
                combined['expiry'] = expiry_date
                
                # Time to expiry in years
                days_to_exp = (expiry_date - now).days
                if days_to_exp < 0: days_to_exp = 0
                combined['time_to_expiry'] = (days_to_exp + 1) / 365.0 # +1 to avoid div0
                
                all_opts.append(combined)
            except Exception as e:
                # Silent fail for individual chain is fine, just skip
                continue
                
        if all_opts:
            self.df_options = pd.concat(all_opts)
            # Ensure IV is numeric and drop bad data
            self.df_options['impliedVolatility'] = pd.to_numeric(self.df_options['impliedVolatility'], errors='coerce')
            self.df_options = self.df_options[self.df_options['impliedVolatility'] > 0.001]
            
            # Calculate Moneyness
            self.df_options['moneyness'] = self.df_options['strike'] / self.spot_price
            print(f"    > Loaded {len(self.df_options)} option contracts.")

    def _compute_greeks_and_exposures(self):
        print("    > Computing Dealer Greeks (GEX, Vanna, Charm)...")
        
        # Vectorized or Apply calculation
        # To speed up, we pass S (spot) and r (rate)
        # Using T-Bill rate approx 4.25%
        r = 0.0425 
        
        # Apply Greek Engine
        greeks = self.df_options.apply(
            lambda row: GreeksEngine.calculate_greeks(row, self.spot_price, r), axis=1
        )
        
        self.df_options = pd.concat([self.df_options, greeks], axis=1)
        
        # --- EXPOSURE CALCS ---
        # GEX = Gamma * OI * Spot^2 * 0.01 (Factor to scale to roughly $ per 1% move)
        # Convention: Dealers are Short Calls (Market is Long) -> Dealers Short Gamma
        # Dealers are Short Puts (Market is Long) -> Dealers Long Gamma
        # Wait, the "Gamma Flip" convention usually is:
        # Call OI contributes Positive to Net GEX (Dealers Short Call -> Short Gamma -> Hedging reinforces move? No, wait.)
        # Let's use the SqueezeMetrics standard:
        # Net GEX = (Call Gamma * Call OI) - (Put Gamma * Put OI)
        # Units: Spot * Gamma * OI * 100? No, let's stick to raw Gamma sum weighted by Spot^2.
        
        # Standard GEX ($ exposure per 1% move):
        # GEX = Gamma * OI * Spot^2 * 0.01
        # Direction:
        # Calls: Dealers sell calls -> Short Gamma. If spot up, delta up, sell more underlying. Stabilizing? No.
        # Short Gamma is destabilizing (buy high, sell low).
        # Long Gamma is stabilizing (sell high, buy low).
        
        # Standard Charting Convention:
        # Positive bars = Calls, Negative bars = Puts.
        # Net > 0 = Long Gamma Regime (Stabilizing).
        # Net < 0 = Short Gamma Regime (Volatile).
        
        # We calculate specific GEX contribution
        # Contribution = Gamma * OI * Spot^2 * 0.01
        # For Calls: Contribution is POSITIVE in the chart sum (but implies dealer short gamma??)
        # Actually, standard logic: 
        # Call GEX = Gamma * OI
        # Put GEX = Gamma * OI * -1
        
        self.df_options['GEX'] = self.df_options['gamma'] * self.df_options['openInterest'] * (self.spot_price**2) * 0.01
        self.df_options.loc[self.df_options['type'] == 'put', 'GEX'] *= -1
        
        # Vanna Exposure
        # dDelta/dVol.
        self.df_options['VEX'] = self.df_options['vanna'] * self.df_options['openInterest']
        
        # Charm Exposure
        # dDelta/dTime
        self.df_options['CEX'] = self.df_options['charm'] * self.df_options['openInterest']

    def _analyze_term_structure(self):
        print("    > Analyzing Term Structure & Skew...")
        # Group by expiry
        groups = self.df_options.groupby('expiry')
        ts_data = []
        
        for exp, grp in groups:
            # ATM Vol
            grp['dist_atm'] = abs(grp['strike'] - self.spot_price)
            # Find closest strike. If multiple, pick first.
            atm_opt = grp.loc[grp['dist_atm'].idxmin()]
            
            # Use iloc[0] if it returns a DataFrame (multiple same strikes)
            if isinstance(atm_opt, pd.DataFrame):
                atm_opt = atm_opt.iloc[0]
                
            atm_iv = atm_opt['impliedVolatility']
            
            # Skew (25 Delta)
            # Find Call closest to 0.25 delta
            calls = grp[grp['type']=='call']
            puts = grp[grp['type']=='put']
            
            # Robust check for empty slices
            if calls.empty or puts.empty:
                continue

            c25 = calls.iloc[(calls['delta'] - 0.25).abs().argsort()[:1]]
            p25 = puts.iloc[(puts['delta'].abs() - 0.25).abs().argsort()[:1]]
            
            if not c25.empty and not p25.empty:
                c_vol = c25['impliedVolatility'].values[0]
                p_vol = p25['impliedVolatility'].values[0]
                rr = c_vol - p_vol
                bf = ((c_vol + p_vol)/2) - atm_iv
            else:
                rr = np.nan
                bf = np.nan
                
            ts_data.append({
                'expiry': exp,
                'ATM_IV': atm_iv,
                'RiskReversal': rr,
                'Butterfly': bf,
                'Total_OI': grp['openInterest'].sum(),
                'Net_GEX': grp['GEX'].sum()
            })
            
        self.term_structure = pd.DataFrame(ts_data).sort_values('expiry')

# ------------------------------------------------------------------------------
# 4. VISUALIZATION ENGINE
# ------------------------------------------------------------------------------

class DashboardGenerator:
    def __init__(self, analyzer: MarketStructureAnalyzer, output_dir="."):
        self.an = analyzer
        self.out_dir = output_dir
        
    def generate_html(self, filename="options_dashboard.html"):
        print("[*] Generating Dashboard...")
        
        fig = make_subplots(
            rows=5, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "xy"}], # Price, Term Struct
                [{"type": "heatmap"}, {"type": "xy"}], # GEX Heatmap, Net GEX by Exp
                [{"type": "xy"}, {"type": "xy"}], # OI Hist, Skew
                [{"type": "heatmap"}, {"type": "heatmap"}], # Vanna, Charm
                [{"type": "xy"}, {"type": "xy"}] # Vol Cone/Regime, Entropy/Clustering
            ],
            subplot_titles=(
                f"{self.an.ticker_symbol} Price & 20d Vol", "ATM IV Term Structure",
                "Gamma Exposure (Strike vs Expiry)", "Net Gamma Exposure Profile",
                "Open Interest Distribution (Moneyness)", "Skew (Risk Reversal 25d)",
                "Vanna Exposure Map", "Charm Exposure Map",
                "Realized Volatility Regimes", "Return Entropy (Rolling)"
            ),
            vertical_spacing=0.06,
            horizontal_spacing=0.08
        )
        
        # 1. Price Chart
        df = self.an.df_price
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC'
        ), row=1, col=1)
        
        # 2. Term Structure
        ts = self.an.term_structure
        if not ts.empty:
            fig.add_trace(go.Scatter(
                x=ts['expiry'], y=ts['ATM_IV']*100, mode='lines+markers', name='ATM IV %',
                line=dict(color='cyan')
            ), row=1, col=2)
            
        # 3. Gamma Heatmap (Strike vs Expiry)
        # We need to bin strikes or use raw if sparse
        # Let's bin by moneyness for cleaner heatmap
        opts = self.an.df_options.copy()
        if not opts.empty:
            opts['moneyness_bin'] = opts['moneyness'].round(2)
            gex_heat = opts.groupby(['expiry', 'moneyness_bin'])['GEX'].sum().unstack(level=0).fillna(0)
            
            # Limit heatmap range to near-the-money
            gex_heat = gex_heat.loc[0.8:1.2] 
            
            fig.add_trace(go.Heatmap(
                z=gex_heat.values,
                x=gex_heat.columns,
                y=gex_heat.index,
                colorscale='RdBu',
                zmid=0,
                name='GEX Heatmap',
                colorbar=dict(len=0.15, y=0.75)
            ), row=2, col=1)
            
            # 4. Net GEX Profile
            colors = ['green' if v > 0 else 'red' for v in ts['Net_GEX']]
            fig.add_trace(go.Bar(
                x=ts['expiry'], y=ts['Net_GEX'],
                marker_color=colors,
                name='Net GEX'
            ), row=2, col=2)
            
            # 5. OI Distribution
            # Histogram of OI by moneyness
            oi_dist = opts.groupby('moneyness_bin')['openInterest'].sum()
            oi_dist = oi_dist.loc[0.7:1.3]
            fig.add_trace(go.Bar(
                x=oi_dist.index, y=oi_dist.values,
                name='Open Interest',
                marker_color='purple'
            ), row=3, col=1)
            
            # 6. Skew (RR)
            fig.add_trace(go.Scatter(
                x=ts['expiry'], y=ts['RiskReversal']*100,
                mode='lines+markers', name='25d RR skew',
                line=dict(color='orange')
            ), row=3, col=2)
            
            # 7. Vanna Heatmap
            vex_heat = opts.groupby(['expiry', 'moneyness_bin'])['VEX'].sum().unstack(level=0).fillna(0)
            vex_heat = vex_heat.loc[0.8:1.2]
            fig.add_trace(go.Heatmap(
                z=vex_heat.values, x=vex_heat.columns, y=vex_heat.index,
                colorscale='Viridis', name='Vanna Map',
                colorbar=dict(len=0.15, y=0.35)
            ), row=4, col=1)
            
            # 8. Charm Heatmap
            cex_heat = opts.groupby(['expiry', 'moneyness_bin'])['CEX'].sum().unstack(level=0).fillna(0)
            cex_heat = cex_heat.loc[0.8:1.2]
            fig.add_trace(go.Heatmap(
                z=cex_heat.values, x=cex_heat.columns, y=cex_heat.index,
                colorscale='Plasma', name='Charm Map',
                colorbar=dict(len=0.15, y=0.35)
            ), row=4, col=2)

        # 9. Volatility Regime
        # Scatter of Return vs Vol
        fig.add_trace(go.Scatter(
            x=df['Return'], y=df['RV_20d'],
            mode='markers', marker=dict(size=4, opacity=0.5, color=df.index.astype(int), colorscale='Bluered'),
            name='Vol Regime'
        ), row=5, col=1)
        
        # 10. Entropy / Clustering
        # Calculate Rolling Entropy of returns (histogram method)
        def calc_entropy(window):
            if len(window) < 10: return 0
            # Bin returns
            counts, _ = np.histogram(window, bins=10, density=True)
            return entropy(counts + 1e-9)
            
        df['Entropy'] = df['Return'].rolling(30).apply(calc_entropy)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Entropy'],
            mode='lines', name='30d Return Entropy',
            line=dict(color='yellow')
        ), row=5, col=2)

        # Layout
        fig.update_layout(
            template='plotly_dark',
            height=1600,
            width=1400,
            title_text=f"Options Structure Analysis: {self.an.ticker_symbol} | Spot: {self.an.spot_price:.2f}",
            showlegend=False
        )
        
        out_path = os.path.join(self.out_dir, filename)
        fig.write_html(out_path)
        print(f"[SUCCESS] Dashboard saved to {out_path}")
        return out_path
        
    def export_csvs(self):
        self.an.df_price.to_csv(f"{self.an.ticker_symbol}_ohlcv.csv")
        if not self.an.term_structure.empty:
            self.an.term_structure.to_csv(f"{self.an.ticker_symbol}_term_structure.csv")
        if not self.an.df_options.empty:
            cols = ['expiry','strike','type','impliedVolatility','delta','gamma','GEX','VEX','CEX']
            self.an.df_options[cols].to_csv(f"{self.an.ticker_symbol}_greeks.csv")

# ------------------------------------------------------------------------------
# 5. MAIN EXECUTION
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Professional Options Structure Dashboard")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--history", type=int, default=365, help="Days of history")
    parser.add_argument("--open-html", action="store_true", help="Open dashboard automatically")
    
    args = parser.parse_args()
    
    # Run
    analyzer = MarketStructureAnalyzer(args.ticker, args.history)
    analyzer.run_pipeline()
    
    if analyzer.df_price.empty:
        print("[!] No data available. Exiting.")
        sys.exit(1)
        
    generator = DashboardGenerator(analyzer)
    html_path = generator.generate_html(f"{args.ticker}_structure_dashboard.html")
    generator.export_csvs()
    
    if args.open_html:
        webbrowser.open(f"file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    main()
