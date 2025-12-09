# SCRIPTNAME: 09.volatility.surface.top.down.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.interpolate import CubicSpline, griddata
from datetime import datetime, date

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Directory Structure
    DATA_DIR = "data_warehouse"
    OUTPUT_DIR = "reports"
    
    # Financial Constants
    RISK_FREE_RATE_DEFAULT = 0.045
    TRADING_DAYS = 252
    
    # Visuals
    PLOTLY_THEME = "plotly_dark"
    COLOR_CALL = "#00ff00"  # Green
    COLOR_PUT = "#ff0000"   # Red
    COLOR_ATM = "#00ccff"   # Blue
    
    @staticmethod
    def setup_dirs(output_dir=None):
        if output_dir:
            Config.OUTPUT_DIR = output_dir
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    @staticmethod
    def get_market_iv_column():
        # yfinance column name for IV
        return 'impliedVolatility'

# ==========================================
# 2. DATA INGESTION (DISK-FIRST)
# ==========================================
class DataIngestion:
    def __init__(self, ticker):
        self.ticker_symbol = ticker.upper()
        self.ticker_obj = yf.Ticker(self.ticker_symbol)
        self.timestamp = datetime.now().strftime("%Y%m%d")

    def get_underlying_price(self):
        """Fetches current spot price with fallback."""
        try:
            # Try fast fetch
            price = self.ticker_obj.fast_info.last_price
            if price: return price
            
            # Fallback to history
            hist = self.ticker_obj.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
            return 0.0
        except Exception as e:
            print(f"Error fetching spot: {e}")
            return 0.0

    def get_options_chain(self, max_expiries=None):
        """
        Iterates through expiries, downloads chains, aggregates, and caches to disk.
        """
        filename = f"{Config.DATA_DIR}/{self.ticker_symbol}_options_{self.timestamp}.csv"
        
        # 1. Disk-First Check
        if os.path.exists(filename):
            print(f"[Data] Loading Options Chain from disk: {filename}")
            full_chain = pd.read_csv(filename)
            full_chain['expirationDate'] = pd.to_datetime(full_chain['expirationDate'])
            return full_chain

        # 2. Download from API
        print(f"[Data] Downloading Options Chain from API...")
        expirations = self.ticker_obj.options
        if not expirations:
            print("No expirations found.")
            return pd.DataFrame()

        all_opts = []
        count = 0
        
        for exp in expirations:
            if max_expiries and count >= max_expiries:
                break
                
            try:
                opt = self.ticker_obj.option_chain(exp)
                
                # Process Calls
                calls = opt.calls.copy()
                calls['type'] = 'call'
                
                # Process Puts
                puts = opt.puts.copy()
                puts['type'] = 'put'
                
                combined = pd.concat([calls, puts])
                combined['expirationDate'] = pd.to_datetime(exp)
                all_opts.append(combined)
                
                count += 1
                time.sleep(0.3) # Gentle rate limiting
            except Exception as e:
                print(f"Failed to fetch {exp}: {e}")

        if not all_opts:
            return pd.DataFrame()

        full_chain = pd.concat(all_opts, ignore_index=True)
        full_chain = self._sanitize_options_df(full_chain)
        
        # 3. Save to Disk
        full_chain.to_csv(filename, index=False)
        return full_chain

    def _sanitize_options_df(self, df):
        # Convert numeric columns safely
        cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # Hedge Fund Approximation: If volume is 0/missing, use a fraction of OI to prevent void charts
        if df['volume'].sum() == 0:
            df['volume'] = df['openInterest'] * 0.05 
            
        return df

# ==========================================
# 3. FINANCIAL ANALYSIS (MATH ENGINE)
# ==========================================
class FinancialAnalysis:
    def __init__(self, spot, risk_free_rate, options_df):
        self.S = spot
        self.r = risk_free_rate
        self.df = options_df.copy()
        self.process_time_to_expiry()
        
    def process_time_to_expiry(self):
        """Calculates T (years) for every option."""
        today = datetime.now()
        self.df['T'] = (self.df['expirationDate'] - today).dt.days / 365.0
        # Filter expired or today's options to avoid DivByZero and weird noise
        self.df = self.df[self.df['T'] > 0.002].copy()

    def calculate_greeks(self):
        """
        Computes Delta, Gamma, Vega, Vanna, Charm, and GEX.
        Uses vectorized NumPy operations for performance.
        """
        if self.df.empty: return

        S = self.S
        K = self.df['strike'].values
        T = self.df['T'].values
        r = self.r
        sigma = self.df[Config.get_market_iv_column()].values
        
        # Prevent division by zero in math
        sigma = np.maximum(sigma, 0.001)
        T = np.maximum(T, 0.0001)
        
        # Black-Scholes Terms
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        
        # Delta
        delta_call = cdf_d1
        delta_put = cdf_d1 - 1
        self.df['delta'] = np.where(self.df['type'] == 'call', delta_call, delta_put)
        
        # Gamma (Same for Call & Put)
        self.df['gamma'] = pdf_d1 / (S * sigma * np.sqrt(T))
        
        # Vega (Sensitivity to 1% vol change, usually scaled /100)
        self.df['vega'] = S * pdf_d1 * np.sqrt(T) / 100 
        
        # Vanna (dDelta/dVol)
        self.df['vanna'] = -pdf_d1 * (d2 / sigma) 
        
        # Charm (dDelta/dTime) - Simplified Call Charm for visual proxy
        charm_call = -pdf_d1 * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
        self.df['charm'] = charm_call 
        
        # --- DEALER EXPOSURE (GEX) ---
        # GEX Notional = Gamma * OI * 100 * Spot
        # Convention: 
        # Call OI => Positive Gamma (Market Stability)
        # Put OI => Negative Gamma (Market Instability)
        self.df['GEX_Notional'] = (
            self.df['gamma'] * self.df['openInterest'] * 100 * S * np.where(self.df['type']=='call', 1, -1)
        )

    def build_vol_surface(self):
        """
        Constructs a smoothed 3D surface using Cubic Splines.
        """
        # Filter deep OTM/ITM noise
        mask = (self.df['strike'] > self.S * 0.5) & (self.df['strike'] < self.S * 1.5)
        subset = self.df[mask].copy()
        
        if subset.empty: return None, None, None

        expiries = sorted(subset['T'].unique())
        strikes = np.linspace(subset['strike'].min(), subset['strike'].max(), 50)
        iv_grid = []
        
        for t in expiries:
            df_t = subset[subset['T'] == t]
            # Average Call/Put IV
            df_t = df_t.groupby('strike')[Config.get_market_iv_column()].mean().reset_index()
            df_t = df_t.sort_values('strike')
            
            if len(df_t) < 4:
                # Fallback if sparse data
                iv_interpolated = np.interp(strikes, df_t['strike'], df_t[Config.get_market_iv_column()])
            else:
                cs = CubicSpline(df_t['strike'], df_t[Config.get_market_iv_column()])
                iv_interpolated = cs(strikes)
            
            iv_grid.append(iv_interpolated)
            
        return strikes, expiries, np.array(iv_grid)

    def get_term_structure(self):
        """Extracts ATM IV vs Expiry."""
        term_struct = []
        for t in sorted(self.df['T'].unique()):
            df_t = self.df[self.df['T'] == t]
            # Find strike closest to spot
            if df_t.empty: continue
            
            idx = (df_t['strike'] - self.S).abs().idxmin()
            iv = df_t.loc[idx, Config.get_market_iv_column()]
            term_struct.append({'T': t, 'IV': iv})
        return pd.DataFrame(term_struct)

    def get_skew_metrics(self):
        """Calculates 25-Delta Risk Reversal and Butterfly."""
        results = []
        for t in sorted(self.df['T'].unique()):
            df_t = self.df[self.df['T'] == t]
            
            try:
                # Calls (target delta 0.25)
                calls = df_t[df_t['type'] == 'call'].sort_values('delta')
                if len(calls) > 3:
                    f_call = CubicSpline(calls['delta'], calls[Config.get_market_iv_column()])
                    iv_25c = f_call(0.25)
                else: continue

                # Puts (target delta -0.25)
                puts = df_t[df_t['type'] == 'put'].sort_values('delta')
                if len(puts) > 3:
                    f_put = CubicSpline(puts['delta'], puts[Config.get_market_iv_column()])
                    iv_25p = f_put(-0.25)
                else: continue
                
                # ATM
                idx = (df_t['strike'] - self.S).abs().idxmin()
                iv_atm = df_t.loc[idx, Config.get_market_iv_column()]

                rr = iv_25c - iv_25p
                bf = iv_25c + iv_25p - 2 * iv_atm

                results.append({'T': t, 'RR': rr, 'BF': bf, 'ATM': iv_atm})
            except Exception:
                continue
                
        return pd.DataFrame(results)

    def scenario_stress_test(self, pct_move=0.05):
        """
        Estimates Exposure change if Spot moves +/- 5%.
        """
        scenarios = {}
        shocks = [-pct_move, 0, pct_move]
        
        # Pre-calculation values
        K = self.df['strike'].values
        T = self.df['T'].values
        sigma = self.df[Config.get_market_iv_column()].values
        OI = self.df['openInterest'].values
        types = np.where(self.df['type']=='call', 1, -1)
        
        for shock in shocks:
            new_spot = self.S * (1 + shock)
            
            # Recalculate Gamma at new spot (assuming sticky strike/vol)
            d1_new = (np.log(new_spot / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            pdf_d1_new = norm.pdf(d1_new)
            gamma_new = pdf_d1_new / (new_spot * sigma * np.sqrt(T))
            
            gex_new = gamma_new * OI * 100 * new_spot * types
            total_gex = np.nansum(gex_new)
            
            lbl = f"Spot {shock:+.0%}"
            scenarios[lbl] = total_gex
            
        return scenarios

# ==========================================
# 4. DASHBOARD RENDERER (PLOTLY OFFLINE)
# ==========================================
class DashboardRenderer:
    def __init__(self, analysis_obj, output_path):
        self.an = analysis_obj
        self.path = output_path
        self.figs = [] # Stores (id, title, div)

    def generate_dashboard(self):
        self._make_summary_tab()
        self._make_surface_tab()
        self._make_smile_tab()
        self._make_exposure_tab()
        self._make_scenario_tab()
        self._compile_html()

    def _make_summary_tab(self):
        ts = self.an.get_term_structure()
        
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Term Structure (ATM IV)", "Spot Price Context"),
            specs=[[{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        if not ts.empty:
            fig.add_trace(go.Scatter(x=ts['T'], y=ts['IV'], mode='lines+markers', name='ATM IV', line=dict(color=Config.COLOR_ATM)), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode = "number+gauge", value = self.an.S,
            title = {"text": "Spot Price"},
            gauge = {'axis': {'range': [self.an.S*0.8, self.an.S*1.2]}, 'bar': {'color': "white"}}
        ), row=1, col=2)
        
        fig.update_layout(template=Config.PLOTLY_THEME, height=450)
        self.figs.append({"id": "summary", "title": "Summary", "div": pyo.plot(fig, include_plotlyjs=False, output_type='div')})

    def _make_surface_tab(self):
        strikes, expiries, iv_grid = self.an.build_vol_surface()
        
        if strikes is not None:
            fig = go.Figure(data=[go.Surface(z=iv_grid, x=strikes, y=expiries, colorscale='Viridis')])
            fig.update_layout(
                title='Volatility Surface', 
                scene = dict(
                    xaxis_title='Strike', 
                    yaxis_title='Expiry (Yrs)', 
                    zaxis_title='Implied Volatility'
                ),
                template=Config.PLOTLY_THEME,
                height=600,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            self.figs.append({"id": "volsurf", "title": "Vol Surface 3D", "div": pyo.plot(fig, include_plotlyjs=False, output_type='div')})

    def _make_smile_tab(self):
        skew = self.an.get_skew_metrics()
        
        if not skew.empty:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Risk Reversal (Skew)", "Butterfly (Kurtosis)"))
            
            fig.add_trace(go.Scatter(x=skew['T'], y=skew['RR'], mode='lines', name='25d RR', line=dict(color='#ff00ff')), row=1, col=1)
            fig.add_trace(go.Scatter(x=skew['T'], y=skew['BF'], mode='lines', name='25d Fly', line=dict(color='#ffff00')), row=1, col=2)
            
            fig.update_layout(template=Config.PLOTLY_THEME, height=450)
            self.figs.append({"id": "skew", "title": "Skew & Kurtosis", "div": pyo.plot(fig, include_plotlyjs=False, output_type='div')})

    def _make_exposure_tab(self):
        # Group GEX by strike
        gex_df = self.an.df.groupby('strike')['GEX_Notional'].sum().reset_index()
        # Zoom in on spot
        gex_df = gex_df[(gex_df['strike'] > self.an.S * 0.75) & (gex_df['strike'] < self.an.S * 1.25)]
        
        fig = go.Figure()
        colors = np.where(gex_df['GEX_Notional'] < 0, Config.COLOR_PUT, Config.COLOR_CALL)
        
        fig.add_trace(go.Bar(
            x=gex_df['strike'], 
            y=gex_df['GEX_Notional'],
            marker_color=colors,
            name='Net GEX'
        ))
        
        fig.add_vline(x=self.an.S, line_width=2, line_dash="dash", line_color="white", annotation_text="Spot")
        
        fig.update_layout(
            title="Net Gamma Exposure Profile (Dealer Perspective)", 
            xaxis_title="Strike", 
            yaxis_title="Notional GEX ($)",
            template=Config.PLOTLY_THEME, 
            height=550
        )
        self.figs.append({"id": "gex", "title": "Dealer Exposure", "div": pyo.plot(fig, include_plotlyjs=False, output_type='div')})

    def _make_scenario_tab(self):
        scenarios = self.an.scenario_stress_test()
        
        x_vals = list(scenarios.keys())
        y_vals = list(scenarios.values())
        colors = ['#ff0000', '#aaaaaa', '#00ff00']
        
        fig = go.Figure(go.Bar(x=x_vals, y=y_vals, marker_color=colors))
        fig.update_layout(
            title="Total GEX Sensitivity to Spot Shock",
            yaxis_title="Total Network Gamma ($)",
            template=Config.PLOTLY_THEME,
            height=450
        )
        self.figs.append({"id": "scenario", "title": "Stress Test", "div": pyo.plot(fig, include_plotlyjs=False, output_type='div')})

    def _compile_html(self):
        tabs_html = ""
        content_html = ""
        
        for i, item in enumerate(self.figs):
            active_class = "active" if i == 0 else ""
            display_style = "block" if i == 0 else "none"
            
            tabs_html += f"""
            <button class="tablinks {active_class}" onclick="openTab(event, '{item['id']}')">{item['title']}</button>
            """
            content_html += f"""
            <div id="{item['id']}" class="tabcontent" style="display: {display_style};">
                {item['div']}
            </div>
            """

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hedge Fund Vol Analytics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; }}
                .header {{ padding: 20px; background: #1f1f1f; border-bottom: 1px solid #333; }}
                .tab {{ overflow: hidden; background-color: #1f1f1f; border-bottom: 1px solid #333; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 20px; transition: 0.3s; color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
                .tab button:hover {{ background-color: #333; color: #fff; }}
                .tab button.active {{ background-color: #333; color: #4db6ac; border-bottom: 2px solid #4db6ac; }}
                .tabcontent {{ padding: 20px; animation: fadeEffect 0.5s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
            <script>
            function openTab(evt, cityName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
                document.getElementById(cityName).style.display = "block";
                evt.currentTarget.className += " active";
                window.dispatchEvent(new Event('resize')); 
            }}
            </script>
        </head>
        <body>
            <div class="header">
                <h2>VolAnalytics <span style="font-size:0.6em; color:#666">HEDGE FUND EDITION</span></h2>
            </div>
            <div class="tab">
                {tabs_html}
            </div>
            {content_html}
        </body>
        </html>
        """
        
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"[Dashboard] Saved successfully to {self.path}")

# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Options Analytics Engine")
    parser.add_argument("--ticker", type=str, required=True, help="Stock Ticker (e.g. SPY, NVDA)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--risk-free", type=float, default=0.045, help="Risk Free Rate (decimal, e.g. 0.045)")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore disk cache and force download")
    
    args = parser.parse_args()
    
    # Setup
    Config.setup_dirs(args.output_dir)
    print(f"\n=== VolAnalytics Engine: {args.ticker} ===")
    
    # Ingestion
    ingest = DataIngestion(args.ticker)
    
    # Force refresh handling
    if args.force_refresh:
        f_path = f"{Config.DATA_DIR}/{ingest.ticker_symbol}_options_{ingest.timestamp}.csv"
        if os.path.exists(f_path):
            os.remove(f_path)
            print("[System] Cache cleared.")

    spot = ingest.get_underlying_price()
    print(f"[Market] Spot Price: ${spot:.2f}")
    
    options_df = ingest.get_options_chain()
    
    if options_df.empty:
        print("[Error] No options data found. Exiting.")
        sys.exit(1)
        
    # Analysis
    print("[Math] calculating Greeks, Surfaces, and GEX...")
    analyzer = FinancialAnalysis(spot, args.risk_free, options_df)
    analyzer.calculate_greeks()
    
    # Visualization
    output_file = f"{Config.OUTPUT_DIR}/{args.ticker}_dashboard.html"
    print(f"[Viz] Generating dashboard at {output_file}...")
    renderer = DashboardRenderer(analyzer, output_file)
    renderer.generate_dashboard()
    
    print("=== Complete ===\n")
