# SCRIPTNAME: ok.awesome.black.scholes.V2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import sys
import os
import math
import datetime
import webbrowser
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio

# -----------------------------------------------------------------------------
# 1. MATH & BLACK-SCHOLES ENGINE (No Scipy Dependency)
# -----------------------------------------------------------------------------

class QuantMath:
    """
    Standalone implementation of probability functions and Black-Scholes 
    to avoid Scipy dependency as requested.
    """
    
    @staticmethod
    def norm_pdf(x: float) -> float:
        """Standard Normal Probability Density Function."""
        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Standard Normal Cumulative Distribution Function using Error Function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    @classmethod
    def black_scholes_price(cls, S, K, T, r, sigma, option_type='call') -> float:
        if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1, d2 = cls.d1_d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            return S * cls.norm_cdf(d1) - K * math.exp(-r * T) * cls.norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * cls.norm_cdf(-d2) - S * cls.norm_cdf(-d1)

    @classmethod
    def calculate_greeks(cls, S, K, T, r, sigma, option_type='call') -> Dict[str, float]:
        """Computes Delta, Gamma, Vega, Theta, Rho."""
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}

        d1, d2 = cls.d1_d2(S, K, T, r, sigma)
        pdf_d1 = cls.norm_pdf(d1)
        cdf_d1 = cls.norm_cdf(d1)
        cdf_neg_d1 = cls.norm_cdf(-d1)
        
        # Gamma and Vega are same for Call and Put
        gamma = pdf_d1 / (S * sigma * math.sqrt(T))
        vega = S * pdf_d1 * math.sqrt(T) / 100.0 # Scaled for 1% vol change
        
        if option_type == 'call':
            delta = cdf_d1
            theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T)) 
                     - r * K * math.exp(-r * T) * cls.norm_cdf(d2)) / 365.0
            rho = (K * T * math.exp(-r * T) * cls.norm_cdf(d2)) / 100.0
        else:
            delta = cdf_d1 - 1
            theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T)) 
                     + r * K * math.exp(-r * T) * cls.norm_cdf(-d2)) / 365.0
            rho = (-K * T * math.exp(-r * T) * cls.norm_cdf(-d2)) / 100.0

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

# -----------------------------------------------------------------------------
# 2. DATA INGESTION & PROCESSING
# -----------------------------------------------------------------------------

class OptionsAnalytics:
    def __init__(self, ticker: str, days_lookback: int = 365):
        self.ticker_symbol = ticker.upper()
        self.days_lookback = days_lookback
        self.ticker = yf.Ticker(ticker)
        
        # State containers
        self.spot_price = 0.0
        self.risk_free_rate = 0.02 # Default fallback
        self.history = pd.DataFrame()
        self.chains = pd.DataFrame()
        self.term_structure = pd.DataFrame()
        self.greeks_surface = pd.DataFrame()
        self.microstructure = pd.DataFrame()
        
    def fetch_market_data(self):
        print(f"[*] Fetching OHLCV data for {self.ticker_symbol}...")
        self.history = self.ticker.history(period=f"{self.days_lookback}d")
        
        if self.history.empty:
            raise ValueError(f"No history found for {self.ticker_symbol}")
            
        self.spot_price = self.history['Close'].iloc[-1]
        print(f"[*] Spot Price: {self.spot_price:.2f}")

        # Risk Free Rate Proxy (^IRX is 13 week treasury yield index)
        try:
            irx = yf.Ticker("^IRX")
            irx_hist = irx.history(period="5d")
            if not irx_hist.empty:
                # IRX 4.50 means 4.5%
                self.risk_free_rate = irx_hist['Close'].iloc[-1] / 100.0
                print(f"[*] Risk-Free Rate (^IRX): {self.risk_free_rate:.2%}")
        except Exception:
            print("[!] Could not fetch ^IRX, using default risk-free rate of 4.5%")
            self.risk_free_rate = 0.045

    def fetch_option_chains(self, num_expiries=6):
        print(f"[*] Fetching option chains (next {num_expiries} expiries)...")
        expirations = self.ticker.options
        if not expirations:
            raise ValueError("No option expirations found.")
            
        target_expiries = expirations[:num_expiries]
        all_chains = []
        
        today = datetime.datetime.now().date()
        
        for exp_date_str in target_expiries:
            try:
                # YF returns a named tuple usually, handle carefully
                chain = self.ticker.option_chain(exp_date_str)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                
                calls['type'] = 'call'
                puts['type'] = 'put'
                
                df = pd.concat([calls, puts])
                df['expiry'] = pd.to_datetime(exp_date_str)
                
                # Calculate T (Time to expiry in years)
                exp_date = pd.to_datetime(exp_date_str).date()
                days_to_exp = (exp_date - today).days
                
                # Avoid T=0 division by zero
                if days_to_exp <= 0:
                    days_to_exp = 0.5 
                
                df['dte'] = days_to_exp
                df['T'] = days_to_exp / 365.0
                
                all_chains.append(df)
                print(f"    -> Loaded {exp_date_str} ({len(df)} contracts)")
            except Exception as e:
                print(f"    [!] Failed to load {exp_date_str}: {e}")
                
        if not all_chains:
            raise ValueError("Could not load any option chains.")
            
        self.chains = pd.concat(all_chains, ignore_index=True)
        self.chains['mid'] = (self.chains['bid'] + self.chains['ask']) / 2
        
        # Fill NaN IVs or zeros with a basic approximation if needed, 
        # but YF usually provides decent 'impliedVolatility'
        self.chains = self.chains[self.chains['impliedVolatility'] > 0.001].copy()

    def compute_greeks_and_iv(self):
        print("[*] Computing Greeks and cleaning data...")
        
        greeks_list = []
        
        # Iterate efficiently
        for idx, row in self.chains.iterrows():
            g = QuantMath.calculate_greeks(
                S=self.spot_price,
                K=row['strike'],
                T=row['T'],
                r=self.risk_free_rate,
                sigma=row['impliedVolatility'],
                option_type=row['type']
            )
            greeks_list.append(g)
            
        greeks_df = pd.DataFrame(greeks_list)
        self.chains = pd.concat([self.chains.reset_index(drop=True), greeks_df], axis=1)
        
        # GEX (Gamma Exposure) Approximation: Gamma * OpenInterest * Spot * 100
        # Call GEX is positive, Put GEX is negative
        self.chains['gex'] = np.where(
            self.chains['type'] == 'call',
            self.chains['gamma'] * self.chains['openInterest'] * 100 * self.spot_price,
            -1 * self.chains['gamma'] * self.chains['openInterest'] * 100 * self.spot_price
        )

    def analyze_term_structure(self):
        print("[*] Analyzing Term Structure & Skew...")
        
        results = []
        expiries = self.chains['expiry'].unique()
        
        for exp in expiries:
            # FIX: Explicitly copy the slice to avoid SettingWithCopyWarning
            df_exp = self.chains[self.chains['expiry'] == exp].copy()
            if df_exp.empty: continue
            
            T = df_exp['T'].iloc[0]
            dte = df_exp['dte'].iloc[0]
            
            # 1. ATM IV (Strike closest to Spot)
            # Average of Call and Put IV at the closest strike
            df_exp['dist_to_spot'] = abs(df_exp['strike'] - self.spot_price)
            atm_strike = df_exp.loc[df_exp['dist_to_spot'].idxmin()]['strike']
            
            atm_calls = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['type'] == 'call')]
            atm_puts = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['type'] == 'put')]
            
            iv_atm = (atm_calls['impliedVolatility'].mean() + atm_puts['impliedVolatility'].mean()) / 2
            
            # 2. Skew Points (25D, 10D)
            # We need to interpolate IV based on Delta
            
            calls = df_exp[df_exp['type'] == 'call'].sort_values('delta')
            puts = df_exp[df_exp['type'] == 'put'].sort_values('delta') # deltas are negative
            
            def get_iv_at_delta(sub_df, target_delta):
                # Simple finding of closest or linear interp
                # Given strict dependencies, we use numpy interp
                if sub_df.empty: return np.nan
                return np.interp(target_delta, sub_df['delta'], sub_df['impliedVolatility'])

            iv_25c = get_iv_at_delta(calls, 0.25)
            iv_10c = get_iv_at_delta(calls, 0.10)
            
            iv_25p = get_iv_at_delta(puts, -0.25)
            iv_10p = get_iv_at_delta(puts, -0.10)
            
            # Metrics
            rr_25 = iv_25c - iv_25p # Risk Reversal
            rr_10 = iv_10c - iv_10p
            fly = ((iv_25c + iv_25p) / 2) - iv_atm # Butterfly (Wings vs ATM)
            
            # Expected Move = Spot * ATM_IV * sqrt(T)
            expected_move = self.spot_price * iv_atm * math.sqrt(T)
            
            results.append({
                'expiry': exp,
                'dte': dte,
                'T': T,
                'iv_atm': iv_atm,
                'iv_25c': iv_25c,
                'iv_25p': iv_25p,
                'iv_10c': iv_10c,
                'iv_10p': iv_10p,
                'rr_25': rr_25,
                'rr_10': rr_10,
                'fly': fly,
                'expected_move': expected_move
            })
            
        self.term_structure = pd.DataFrame(results).sort_values('expiry')

    def calculate_synthetic_microstructure(self):
        print("[*] Calculating Synthetic Microstructure Proxies...")
        df = self.history.copy()
        
        # 1. Log Returns
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Realized Volatility Windows
        df['rv_10'] = df['log_ret'].rolling(10).std() * np.sqrt(252)
        df['rv_21'] = df['log_ret'].rolling(21).std() * np.sqrt(252)
        df['rv_63'] = df['log_ret'].rolling(63).std() * np.sqrt(252)
        
        # 3. Microprice Approximation: (H + L + 2C) / 4
        # Gives more weight to the close
        df['microprice'] = (df['High'] + df['Low'] + 2*df['Close']) / 4
        
        # 4. Volume Imbalance Proxy
        # (Close - Open) / (High - Low) scaled by Volume
        # Normalized -1 to 1
        range_len = (df['High'] - df['Low']).replace(0, 0.01)
        df['vol_imbalance'] = ((df['Close'] - df['Open']) / range_len) * np.log(df['Volume'])
        
        # 5. Liquidity Proxy / Amihud-like
        # Abs(Ret) / Volume. Lower is more liquid.
        # Here we inverse it to make "Higher = More Liquid"
        # Using High-Low range as volatility proxy for intraday
        df['liquidity_proxy'] = df['Volume'] / (df['High'] - df['Low']).replace(0, 0.01)
        
        self.microstructure = df.dropna()

    def get_gtc_helper_data(self):
        """Prepares data for the Trade Planner tab (Nearest Expiry)"""
        if self.chains.empty: return None
        
        # Get nearest expiry
        near_exp = self.chains['expiry'].min()
        df = self.chains[self.chains['expiry'] == near_exp].copy()
        
        # Focus on a window around spot
        df = df[ (df['strike'] > self.spot_price * 0.8) & (df['strike'] < self.spot_price * 1.2) ]
        
        return df, near_exp

# -----------------------------------------------------------------------------
# 3. VISUALIZATION ENGINE (Plotly -> HTML)
# -----------------------------------------------------------------------------

class DashboardGenerator:
    def __init__(self, analytics: OptionsAnalytics):
        self.oa = analytics
        
    def _style_fig(self, fig, title):
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(family="Roboto, sans-serif"),
            hovermode="x unified"
        )
        return fig

    def make_tab1_context(self):
        # Subplots: OHLCV + RV
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=self.oa.microstructure.index,
            open=self.oa.microstructure['Open'],
            high=self.oa.microstructure['High'],
            low=self.oa.microstructure['Low'],
            close=self.oa.microstructure['Close'],
            name='OHLC'
        ), row=1, col=1)
        
        # Expected Move Cone (Forward looking projected from last price)
        # Just simple lines for next 30 days based on current ATM IV
        last_price = self.oa.spot_price
        # Check if term structure exists
        if not self.oa.term_structure.empty:
            avg_iv = self.oa.term_structure['iv_atm'].mean()
            days_fwd = np.arange(1, 31)
            upper = last_price * (1 + avg_iv * np.sqrt(days_fwd/365))
            lower = last_price * (1 - avg_iv * np.sqrt(days_fwd/365))
            dates_fwd = [self.oa.microstructure.index[-1] + datetime.timedelta(days=int(x)) for x in days_fwd]
            
            fig.add_trace(go.Scatter(x=dates_fwd, y=upper, mode='lines', 
                                     line=dict(dash='dot', color='gray'), name='+1 SD Exp Move'), row=1, col=1)
            fig.add_trace(go.Scatter(x=dates_fwd, y=lower, mode='lines', 
                                     line=dict(dash='dot', color='gray'), name='-1 SD Exp Move'), row=1, col=1)

        # RV
        fig.add_trace(go.Scatter(x=self.oa.microstructure.index, y=self.oa.microstructure['rv_10'], name='RV 10d'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.oa.microstructure.index, y=self.oa.microstructure['rv_21'], name='RV 21d'), row=2, col=1)
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", tickformat=".1%", row=2, col=1)
        
        return self._style_fig(fig, f"{self.oa.ticker_symbol} Price Context & Realized Volatility")

    def make_tab2_term_structure(self):
        ts = self.oa.term_structure
        
        fig = sp.make_subplots(rows=2, cols=2, 
                               subplot_titles=("Term Structure (IV)", "Skew (Risk Reversal)", "Smile Structure (Fly)", "IV vs Delta Scatter"))
        
        # 1. Term Structure
        fig.add_trace(go.Scatter(x=ts['dte'], y=ts['iv_atm'], mode='lines+markers', name='ATM IV'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts['dte'], y=ts['iv_25c'], mode='lines+markers', name='25D Call IV', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts['dte'], y=ts['iv_25p'], mode='lines+markers', name='25D Put IV', line=dict(dash='dash')), row=1, col=1)
        
        # 2. Skew (RR)
        fig.add_trace(go.Bar(x=ts['dte'], y=ts['rr_25'], name='RR 25 (Skew)'), row=1, col=2)
        
        # 3. Fly
        fig.add_trace(go.Scatter(x=ts['dte'], y=ts['fly'], mode='lines+markers', name='Butterfly (Wings)'), row=2, col=1)
        
        # 4. Scatter IV vs Delta (Nearest Expiry)
        near_exp = self.oa.chains['expiry'].min()
        near_chain = self.oa.chains[self.oa.chains['expiry'] == near_exp]
        calls = near_chain[near_chain['type']=='call']
        
        fig.add_trace(go.Scatter(x=calls['delta'], y=calls['impliedVolatility'], mode='markers', name=f'Smile ({near_exp.date()})'), row=2, col=2)
        
        fig.update_xaxes(title_text="DTE", row=1, col=1)
        fig.update_xaxes(title_text="DTE", row=1, col=2)
        fig.update_yaxes(tickformat=".1%", row=1, col=1)
        
        return self._style_fig(fig, "Volatility Term Structure & Skew Analysis")

    def make_tab3_surface(self):
        # 3D Surface Plot of IV
        # X: Strike, Y: DTE, Z: IV
        
        # We need to pivot the data
        # Filter for Calls for surface
        df = self.oa.chains[self.oa.chains['type'] == 'call'].copy()
        
        # Pivot table
        # We bin strikes to make the surface mesh cleaner
        df['strike_bin'] = df['strike'].round(1)
        pivot = df.pivot_table(index='dte', columns='strike_bin', values='impliedVolatility')
        
        # Interpolate to fill gaps
        pivot = pivot.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0)
        
        fig = go.Figure(data=[go.Surface(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            opacity=0.9
        )])
        
        fig.update_layout(
            title='Implied Volatility Surface (Call Strikes)',
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='DTE',
                zaxis_title='Implied Volatility'
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    def make_tab4_greeks(self):
        # Focus on nearest expiry for detail
        expiries = self.oa.chains['expiry'].unique()[:3] # First 3
        
        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Delta", "Gamma", "Vega", "Gamma Exposure (GEX)"))
        
        colors = ['#00e676', '#2979ff', '#ff1744']
        
        for i, exp in enumerate(expiries):
            df = self.oa.chains[(self.oa.chains['expiry'] == exp) & (self.oa.chains['type'] == 'call')]
            df_put = self.oa.chains[(self.oa.chains['expiry'] == exp) & (self.oa.chains['type'] == 'put')]
            
            name = str(exp.date())
            c = colors[i % len(colors)]
            
            # Delta
            fig.add_trace(go.Scatter(x=df['strike'], y=df['delta'], mode='lines', line=dict(color=c), name=f'{name} Delta'), row=1, col=1)
            
            # Gamma
            fig.add_trace(go.Scatter(x=df['strike'], y=df['gamma'], mode='lines', line=dict(color=c), name=f'{name} Gamma'), row=1, col=2)
            
            # Vega
            fig.add_trace(go.Scatter(x=df['strike'], y=df['vega'], mode='lines', line=dict(color=c), name=f'{name} Vega'), row=2, col=1)
            
            # GEX (Total GEX per strike = Call GEX + Put GEX)
            # Need to align strikes
            gex_call = df.set_index('strike')['gex']
            gex_put = df_put.set_index('strike')['gex']
            total_gex = gex_call.add(gex_put, fill_value=0)
            
            fig.add_trace(go.Bar(x=total_gex.index, y=total_gex.values, marker_color=c, opacity=0.5, name=f'{name} GEX'), row=2, col=2)

        # Add Spot Line
        for r in [1, 2]:
            for c in [1, 2]:
                fig.add_vline(x=self.oa.spot_price, line_dash="dash", line_color="white", row=r, col=c)

        return self._style_fig(fig, "Option Greeks Profile")

    def make_tab5_microstructure(self):
        df = self.oa.microstructure.tail(252) # Last year
        
        fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, 
                               subplot_titles=("Synthetic Microprice vs Close", "Volume Imbalance Pressure", "Liquidity Proxy"))
        
        # 1. Microprice
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['microprice'], name='Microprice', line=dict(color='cyan', dash='dot')), row=1, col=1)
        
        # 2. Imbalance
        colors = np.where(df['vol_imbalance'] > 0, 'green', 'red')
        fig.add_trace(go.Bar(x=df.index, y=df['vol_imbalance'], marker_color=colors, name='Vol Imbalance'), row=2, col=1)
        
        # 3. Liquidity
        fig.add_trace(go.Scatter(x=df.index, y=df['liquidity_proxy'], fill='tozeroy', name='Liquidity Proxy'), row=3, col=1)
        
        return self._style_fig(fig, "Synthetic Microstructure Signals (OHLCV Derived)")

    def make_tab6_gtc_helper(self):
        df, exp_date = self.oa.get_gtc_helper_data()
        
        # Theoretical Calculations
        # Scenario: Weekend Theta Decay (assume 2 days pass)
        # Scenario: Vol Shock (-5% IV)
        
        df = df[df['type'] == 'call'].copy() # Focus on calls for simplicity in grid
        
        T_now = df['T'].iloc[0]
        T_weekend = max(0, T_now - (3/365.0))
        
        # Recalculate Prices
        df['theo_weekend'] = df.apply(lambda r: QuantMath.black_scholes_price(
            self.oa.spot_price, r['strike'], T_weekend, self.oa.risk_free_rate, r['impliedVolatility'], 'call'), axis=1)
        
        df['theo_vol_drop'] = df.apply(lambda r: QuantMath.black_scholes_price(
            self.oa.spot_price, r['strike'], T_now, self.oa.risk_free_rate, max(0.01, r['impliedVolatility'] - 0.05), 'call'), axis=1)
        
        # Suggested GTC Limit Buy: 
        # Target the lower of weekend decay or vol drop, minus a safety buffer (e.g., 5%)
        df['gtc_limit_buy'] = np.minimum(df['theo_weekend'], df['theo_vol_drop']) * 0.95
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df['strike'], y=df['lastPrice'], name='Current Market Price', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=df['strike'], y=df['theo_weekend'], name='Theoretical (Post-Weekend)', line=dict(dash='dot', color='orange')))
        fig.add_trace(go.Scatter(x=df['strike'], y=df['gtc_limit_buy'], name='Rec. GTC Buy Limit (Conservative)', line=dict(color='#00e676')))
        
        fig.add_vline(x=self.oa.spot_price, line_dash="dash", line_color="gray", annotation_text="Spot")
        
        fig.update_layout(
            title=f"GTC Order Planner (Expiry: {exp_date.date()})",
            xaxis_title="Strike Price",
            yaxis_title="Option Premium ($)",
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e"
        )
        
        return fig

    def generate_html(self, filename):
        print(f"[*] Generating Dashboard: {filename}...")
        
        # Generate Plotly Divs (Raw HTML strings)
        config = {'displayModeBar': False, 'responsive': True}
        
        div1 = pio.to_html(self.make_tab1_context(), full_html=False, include_plotlyjs='cdn', config=config)
        div2 = pio.to_html(self.make_tab2_term_structure(), full_html=False, include_plotlyjs=False, config=config)
        div3 = pio.to_html(self.make_tab3_surface(), full_html=False, include_plotlyjs=False, config=config)
        div4 = pio.to_html(self.make_tab4_greeks(), full_html=False, include_plotlyjs=False, config=config)
        div5 = pio.to_html(self.make_tab5_microstructure(), full_html=False, include_plotlyjs=False, config=config)
        div6 = pio.to_html(self.make_tab6_gtc_helper(), full_html=False, include_plotlyjs=False, config=config)
        
        # Custom HTML Template with JS Tabs
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.oa.ticker_symbol} Options Master Dashboard</title>
            <style>
                body {{ font-family: 'Roboto', sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
                h1 {{ text-align: center; color: #00e676; }}
                .stats-bar {{ display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; font-size: 1.1em; }}
                .stat-box {{ background: #1e1e1e; padding: 10px 20px; border-radius: 5px; border: 1px solid #333; }}
                
                /* Tabs Styling */
                .tab {{ overflow: hidden; border-bottom: 1px solid #333; margin-bottom: 20px; display: flex; justify-content: center; }}
                .tab button {{
                    background-color: #1e1e1e; color: #888; float: left; border: none; outline: none;
                    cursor: pointer; padding: 14px 20px; transition: 0.3s; font-size: 16px; margin: 0 5px;
                    border-radius: 5px 5px 0 0;
                }}
                .tab button:hover {{ background-color: #333; color: white; }}
                .tab button.active {{ background-color: #00e676; color: #121212; font-weight: bold; }}
                
                .tabcontent {{ display: none; padding: 6px 12px; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h1>{self.oa.ticker_symbol} Options Analytics</h1>
            
            <div class="stats-bar">
                <div class="stat-box">Spot: {self.oa.spot_price:.2f}</div>
                <div class="stat-box">Risk Free: {self.oa.risk_free_rate:.2%}</div>
                <div class="stat-box">Data Date: {datetime.datetime.now().strftime('%Y-%m-%d')}</div>
            </div>

            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'Tab1')">Price & Context</button>
                <button class="tablinks" onclick="openTab(event, 'Tab2')">Term Structure</button>
                <button class="tablinks" onclick="openTab(event, 'Tab3')">IV Surface</button>
                <button class="tablinks" onclick="openTab(event, 'Tab4')">Greeks</button>
                <button class="tablinks" onclick="openTab(event, 'Tab5')">Microstructure</button>
                <button class="tablinks" onclick="openTab(event, 'Tab6')">Trade Planner</button>
            </div>

            <div id="Tab1" class="tabcontent" style="display:block;">{div1}</div>
            <div id="Tab2" class="tabcontent">{div2}</div>
            <div id="Tab3" class="tabcontent">{div3}</div>
            <div id="Tab4" class="tabcontent">{div4}</div>
            <div id="Tab5" class="tabcontent">{div5}</div>
            <div id="Tab6" class="tabcontent">{div6}</div>

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
                    
                    // Trigger Plotly resize to fix rendering in hidden tabs
                    window.dispatchEvent(new Event('resize'));
                }}
            </script>
        </body>
        </html>
        """
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(html_content)

# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Professional Options Analytics Dashboard")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--days", type=int, default=365, help="Lookback days for OHLCV (default: 365)")
    parser.add_argument("--open-html", action="store_true", help="Open HTML dashboard automatically")
    
    args = parser.parse_args()
    
    try:
        # Pipeline
        oa = OptionsAnalytics(args.ticker, args.days)
        oa.fetch_market_data()
        oa.calculate_synthetic_microstructure()
        oa.fetch_option_chains(num_expiries=6)
        oa.compute_greeks_and_iv()
        oa.analyze_term_structure()
        
        # Exports
        print("[*] Saving CSV Data...")
        oa.term_structure.to_csv(f"{args.ticker}_term_structure.csv", index=False)
        oa.chains.to_csv(f"{args.ticker}_greeks.csv", index=False)
        oa.microstructure.to_csv(f"{args.ticker}_synthetic_microstructure.csv")
        
        # Dashboard
        filename = f"{args.ticker}_options_dashboard.html"
        dash = DashboardGenerator(oa)
        dash.generate_html(filename)
        
        print(f"\n[SUCCESS] Dashboard generated: {os.path.abspath(filename)}")
        
        if args.open_html:
            webbrowser.open('file://' + os.path.abspath(filename))
            
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
