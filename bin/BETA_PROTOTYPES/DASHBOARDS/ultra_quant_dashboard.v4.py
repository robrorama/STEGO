# SCRIPTNAME: ok.ultra_quant_dashboard.v4.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import datetime
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import griddata
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import yfinance as yf

# Visualization
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.offline as py_offline

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. DATA INGESTION LAYER (Robust, Persistent, Sanitized)
# -----------------------------------------------------------------------------
class DataIngestion:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.cache_dir = "data_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _sanitize_df(self, df):
        """
        The 'Bulletproof' Sanitize Method:
        Handles yfinance MultiIndex variations, forces dates, and strips timezones.
        """
        if df.empty: return df
        
        # 1. Handle MultiIndex (Swap levels if needed)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' is in Level 0
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            # Check if 'Close' is in Level 1 (Swap needed)
            elif 'Close' in df.columns.get_level_values(1):
                df = df.swaplevel(0, 1, axis=1)
                df.columns = df.columns.get_level_values(0)
            # Fallback: Flatten
            else:
                df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # 2. Strict Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find a Date column
            for col in ['Date', 'Datetime', 'date', 'datetime']:
                if col in df.columns:
                    df = df.set_index(col)
                    break
        
        # 3. Force Index to Datetime and Strip Timezone
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        
        # 4. Numeric Coercion
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df.dropna(how='all')

    def get_underlying(self, period="2y"):
        """Check cache -> Download -> Sanitize -> Save"""
        cache_path = f"{self.cache_dir}/{self.ticker}_daily.csv"
        
        # A. Try Cache
        if os.path.exists(cache_path):
            # Check if file is fresh (modified today)
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_path))
            if mod_time.date() == datetime.datetime.now().date():
                print(f"   [CACHE] Loading {self.ticker} prices from local file...")
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return df

        # B. Download
        print(f"   [API] Downloading {self.ticker} prices...")
        time.sleep(0.5) # Rate limit protection
        try:
            df = yf.download(self.ticker, period=period, interval="1d", progress=False, group_by='column', auto_adjust=False)
            df = self._sanitize_df(df)
            
            if not df.empty:
                df.to_csv(cache_path)
            return df
        except Exception as e:
            print(f"   [ERROR] Download failed: {e}")
            return pd.DataFrame()

    def get_intraday(self):
        """No caching for intraday (too volatile), direct fetch."""
        print(f"   [API] Downloading {self.ticker} intraday (60d)...")
        time.sleep(0.5)
        try:
            # yfinance often needs auto_adjust=False to get OHLC correctly
            tick = yf.Ticker(self.ticker)
            df = tick.history(period="60d", interval="5m", auto_adjust=False)
            df = self._sanitize_df(df)
            return df
        except Exception as e:
            print(f"   [ERROR] Intraday failed: {e}")
            return pd.DataFrame()

    def get_options_chain(self):
        """Fetches options. Complex object, saving as Pickle if needed, or just re-downloading."""
        print(f"   [API] Scanning options chain for {self.ticker}...")
        tick = yf.Ticker(self.ticker)
        try:
            exps = tick.options
            if not exps: return pd.DataFrame()
        except: return pd.DataFrame()

        all_opts = []
        # Limit to next 8 expirations to save time
        for date in exps[:8]:
            try:
                time.sleep(0.3) # Gentle rate limiting
                chain = tick.option_chain(date)
                calls = chain.calls
                calls['type'] = 'call'
                puts = chain.puts
                puts['type'] = 'put'
                
                df = pd.concat([calls, puts])
                df['expiry'] = pd.to_datetime(date)
                
                # Cleanup
                for col in ['impliedVolatility', 'strike', 'bid', 'ask', 'openInterest']:
                     if col in df.columns:
                         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                all_opts.append(df)
                print(f"     -> Fetched {date} ({len(df)} contracts)")
            except: pass
        
        if all_opts:
            return pd.concat(all_opts, ignore_index=True)
        return pd.DataFrame()

    def get_risk_free_rate(self):
        try:
            tnx = yf.Ticker("^TNX")
            h = tnx.history(period="5d")
            return h['Close'].iloc[-1] / 100.0
        except:
            return 0.045

# -----------------------------------------------------------------------------
# 2. FINANCIAL ENGINE LAYER (Math, Models, Greeks)
# -----------------------------------------------------------------------------
class FinancialEngine:
    def __init__(self, df_daily, df_intraday, df_options, rfr):
        # COPY ON WRITE: Never mutate inputs in place
        self.daily = df_daily.copy()
        self.intraday = df_intraday.copy()
        self.options = df_options.copy()
        self.rfr = rfr
        self.spot = self.daily['Close'].iloc[-1] if not self.daily.empty else 0
        self.alerts = []

    # --- Black Scholes Logic ---
    @staticmethod
    def _d1_d2(S, K, T, r, sigma):
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-5)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def compute_greeks(self):
        if self.options.empty: return
        
        print("   [MATH] Calculating Greeks & GEX...")
        today = datetime.datetime.now()
        
        # Calculate T (Time to expiry)
        self.options['T'] = (self.options['expiry'] - today).dt.total_seconds() / (365*24*3600)
        self.options = self.options[self.options['T'] > 0.001] # Filter expired
        
        # Vectorized Greeks
        S = self.spot
        K = self.options['strike'].values
        T = self.options['T'].values
        r = self.rfr
        sigma = self.options['impliedVolatility'].values
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        norm_pdf_d1 = stats.norm.pdf(d1)
        norm_cdf_d1 = stats.norm.cdf(d1)
        
        # Delta
        self.options['delta'] = np.where(self.options['type'] == 'call', norm_cdf_d1, norm_cdf_d1 - 1)
        
        # Gamma
        self.options['gamma'] = norm_pdf_d1 / (S * sigma * np.sqrt(T))
        
        # Vanna (Simplification)
        self.options['vanna'] = -norm_pdf_d1 * d2 / sigma
        
        # GEX (Gamma Exposure)
        # GEX = Gamma * OI * 100 * Spot * (+1/-1)
        self.options['GEX'] = self.options['gamma'] * self.options['openInterest'] * 100 * S
        self.options.loc[self.options['type']=='put', 'GEX'] *= -1

    def compute_pca_surface(self):
        """Extracts Level, Skew, Curve factors from IV surface."""
        if self.options.empty: return None
        
        print("   [MATH] Computing Vol Surface PCA...")
        df = self.options.copy()
        df = df[df['impliedVolatility'] > 0]
        df['moneyness'] = round(df['strike'] / self.spot, 2)
        df = df[(df['moneyness'] >= 0.8) & (df['moneyness'] <= 1.2)]
        
        pivot = df.pivot_table(index='expiry', columns='moneyness', values='impliedVolatility', aggfunc='mean')
        pivot = pivot.interpolate(axis=1).ffill(axis=1).bfill(axis=1).dropna()
        
        if pivot.shape[0] < 3 or pivot.shape[1] < 3: return None
        
        # SVD
        vals = pivot.values - pivot.values.mean(axis=0)
        try:
            U, s, Vt = np.linalg.svd(vals, full_matrices=False)
            return {
                'scree': s**2 / np.sum(s**2),
                'loadings': Vt[:3],
                'columns': pivot.columns
            }
        except: return None

    def compute_regimes(self):
        """HMM for Volatility Regimes."""
        if len(self.daily) < 100: return None
        
        print("   [MATH] Fitting Regime Switching Model...")
        df = self.daily.copy()
        df['log_ret'] = np.log(df['Close']/df['Close'].shift(1))
        df['RV'] = df['log_ret'].rolling(20).std() * np.sqrt(252) * 100
        data = df['RV'].dropna()
        
        try:
            model = MarkovRegression(data, k_regimes=2, trend='c', switching_variance=True)
            res = model.fit()
            return {
                'probs': res.smoothed_marginal_probabilities,
                'index': data.index
            }
        except: return None

# -----------------------------------------------------------------------------
# 3. RENDERING LAYER (Offline-Capable, Resizable)
# -----------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self, ticker):
        self.ticker = ticker

    def generate_html(self, engine, filename):
        print("   [RENDER] Building HTML Dashboard...")
        
        # A. Embed Plotly JS (No CDN)
        plotly_js = py_offline.get_plotlyjs()
        
        # --- PLOT 1: Regimes ---
        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig1.add_trace(go.Scatter(x=engine.daily.index, y=engine.daily['Close'], name='Price'), row=1, col=1)
        
        regimes = engine.compute_regimes()
        if regimes:
            idx = regimes['index']
            probs = regimes['probs']
            fig1.add_trace(go.Scatter(x=idx, y=probs[0], name='Low Vol Prob', fill='tozeroy', line=dict(width=0), opacity=0.3), row=2, col=1)
            fig1.add_trace(go.Scatter(x=idx, y=probs[1], name='High Vol Prob', fill='tonexty', line=dict(width=0), opacity=0.3), row=2, col=1)
        fig1.update_layout(title="Price & Volatility Regimes", template="plotly_dark")

        # --- PLOT 2: GEX Profile ---
        fig2 = go.Figure()
        if not engine.options.empty:
            gex = engine.options.groupby('strike')['GEX'].sum() / 1e9
            fig2.add_trace(go.Bar(x=gex.index, y=gex.values, marker_color=np.where(gex.values<0, 'red', 'green'), name='GEX ($Bn)'))
            fig2.add_vline(x=engine.spot, line_dash="dash", annotation_text="Spot")
        fig2.update_layout(title="Gamma Exposure (GEX) Profile", template="plotly_dark")

        # --- PLOT 3: Surface PCA ---
        fig3 = go.Figure()
        pca = engine.compute_pca_surface()
        if pca:
            x = pca['columns']
            fig3.add_trace(go.Scatter(x=x, y=pca['loadings'][0], name='Level (PC1)'))
            fig3.add_trace(go.Scatter(x=x, y=pca['loadings'][1], name='Skew (PC2)'))
            fig3.add_trace(go.Scatter(x=x, y=pca['loadings'][2], name='Curve (PC3)'))
        fig3.update_layout(title="Volatility Surface Factors (PCA)", template="plotly_dark")
        
        # --- Generate HTML ---
        plots = {'Regimes': fig1, 'GEX': fig2, 'PCA': fig3}
        divs = {k: pio.to_html(v, full_html=False, include_plotlyjs=False) for k,v in plots.items()}
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.ticker} Quant Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ background: #111; color: #ddd; font-family: sans-serif; margin: 0; padding: 20px; }}
                .tab {{ overflow: hidden; border-bottom: 1px solid #333; }}
                .tab button {{ background-color: #222; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; color: #ccc; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #007bff; color: white; }}
                .tabcontent {{ display: none; padding: 20px; animation: fadeEffect 0.5s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h1>{self.ticker} Ultra-Quant Dashboard</h1>
            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Regimes')" id="defaultOpen">Regimes</button>
                <button class="tablinks" onclick="openTab(event, 'GEX')">GEX Profile</button>
                <button class="tablinks" onclick="openTab(event, 'PCA')">Vol PCA</button>
            </div>

            <div id="Regimes" class="tabcontent">{divs['Regimes']}</div>
            <div id="GEX" class="tabcontent">{divs['GEX']}</div>
            <div id="PCA" class="tabcontent">{divs['PCA']}</div>

            <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
                
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
                
                // CRITICAL FIX: Trigger Resize for Plotly
                window.dispatchEvent(new Event('resize'));
            }}
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(html)
        print(f"✅ Dashboard saved to: {os.path.abspath(filename)}")

# -----------------------------------------------------------------------------
# 4. MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="SPY")
    args = parser.parse_args()
    
    # 1. Ingestion
    ingest = DataIngestion(args.ticker)
    df_daily = ingest.get_underlying()
    df_intra = ingest.get_intraday()
    df_opts = ingest.get_options_chain()
    rfr = ingest.get_risk_free_rate()
    
    # 2. Financial Logic
    engine = FinancialEngine(df_daily, df_intra, df_opts, rfr)
    engine.compute_greeks()
    
    # 3. Rendering
    renderer = DashboardRenderer(args.ticker)
    renderer.generate_html(engine, f"{args.ticker}_dashboard.html")

if __name__ == "__main__":
    main()
