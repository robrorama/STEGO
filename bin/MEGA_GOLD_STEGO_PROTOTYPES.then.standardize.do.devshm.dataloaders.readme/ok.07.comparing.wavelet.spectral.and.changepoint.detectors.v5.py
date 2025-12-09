import argparse
import os
import time
import logging
import warnings
import json
import ast
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.offline as py_offline
import plotly.express as px
from plotly.subplots import make_subplots

# ==============================================================================
# SECTION 0: CONFIGURATION & UTILS
# ==============================================================================

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quant_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
N_DAYS_YEAR = 252

# ==============================================================================
# SECTION 1: DATA INGESTION (MANDATORY DISK-FIRST)
# ==============================================================================

class DataIngestion:
    """
    Handles data acquisition, sanitization, and storage.
    NO analysis logic allowed here.
    """
    def __init__(self, tickers: List[str], output_dir: str, lookback_years: int):
        self.tickers = tickers
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        self.options_dir = os.path.join(output_dir, 'options')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.options_dir, exist_ok=True)

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """The Universal Fixer: Enforces strict data quality rules."""
        if df.empty:
            return df

        # 1. Flatten MultiIndex unconditionally
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.to_flat_index()

        # 2. Nuclear Rename Strategy
        # Map whatever column names we have to standard OHLCV tags by analyzing string content
        new_map = {}
        for c in df.columns:
            # Create a clean lowercase string representation of the column
            # e.g. "('Close', 'DIS')" -> "close dis" -> "close"
            # e.g. "Adj Close" -> "adj close"
            c_str = str(c).lower()
            c_str = c_str.replace("'", "").replace('"', "").replace("(", "").replace(")", "").replace(",", "")
            
            # Remove ticker name to avoid confusion (e.g. if ticker is 'OPEN')
            c_str = c_str.replace(ticker.lower(), "").strip()
            
            if 'adj close' in c_str or 'adj_close' in c_str:
                new_map[c] = 'Adj Close'
            elif 'close' in c_str:
                new_map[c] = 'Close'
            elif 'open' in c_str:
                new_map[c] = 'Open'
            elif 'high' in c_str:
                new_map[c] = 'High'
            elif 'low' in c_str:
                new_map[c] = 'Low'
            elif 'volume' in c_str:
                new_map[c] = 'Volume'
        
        # Apply the renaming
        df = df.rename(columns=new_map)
        
        # 3. Filter and Deduplicate
        # Only keep the standard columns we identified
        valid_cols = [c for c in df.columns if c in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        if valid_cols:
            df = df[valid_cols]
        
        # Remove duplicate columns (e.g. if multiple columns mapped to 'Close')
        df = df.loc[:, ~df.columns.duplicated()]

        # 4. Handle Adj Close fallback
        # If we have Adj Close but no Close, use Adj Close as Close
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Close'})

        # 5. Enforce DatetimeIndex (Remove Timezone)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 6. Coerce Numerics
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 7. Strict Sort & Dedup Rows
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        return df

    def _fetch_option_chain(self, ticker: str) -> pd.DataFrame:
        """Downloads current option chain snapshot to disk."""
        tk = yf.Ticker(ticker)
        all_opts = []
        try:
            exps = tk.options
            # Limit to next 6 expirations to save time/bandwidth for this demo
            for e in exps[:6]:
                opt = tk.option_chain(e)
                calls = opt.calls
                calls['type'] = 'call'
                puts = opt.puts
                puts['type'] = 'put'
                df = pd.concat([calls, puts])
                df['expiration'] = e
                all_opts.append(df)
            
            if all_opts:
                full_chain = pd.concat(all_opts)
                # Save to disk
                fpath = os.path.join(self.options_dir, f"{ticker}_options.csv")
                full_chain.to_csv(fpath, index=False)
                return full_chain
        except Exception as e:
            logger.warning(f"Could not fetch options for {ticker}: {e}")
        return pd.DataFrame()

    def run(self):
        """Main ingestion pipeline."""
        start_date = (datetime.now() - timedelta(days=self.lookback_years*365)).strftime('%Y-%m-%d')
        
        for ticker in self.tickers:
            fpath = os.path.join(self.output_dir, f"{ticker}.csv")
            needs_download = True

            # Step A: Check Disk
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                    
                    # Sanitize immediately upon loading to fix bad CSVs from previous runs
                    df = self._sanitize_df(df, ticker)
                    
                    # Validation: Check if 'Close' exists and we have enough data
                    if 'Close' not in df.columns:
                        logger.warning(f"{ticker}: Corrupted data on disk (missing Close). Redownloading.")
                        needs_download = True
                    elif len(df) < (self.lookback_years * 200):
                        logger.info(f"{ticker}: Data incomplete on disk. Redownloading.")
                        needs_download = True
                    else:
                        # Save the sanitized version back to disk to fix the file permanently
                        df.to_csv(fpath)
                        logger.info(f"{ticker}: Loaded and sanitized from disk.")
                        needs_download = False
                except Exception as e:
                    logger.warning(f"{ticker}: Error loading from disk ({e}). Redownloading.")
                    needs_download = True

            # Step B/C: Download if needed
            if needs_download:
                logger.info(f"{ticker}: Downloading...")
                # auto_adjust=True makes Close = Adj Close. We rely on 'Close' key.
                # using group_by='ticker' ensures consistent structure even for single ticker
                df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True, group_by='ticker')
                time.sleep(1) # Critical Sleep
                
                df = self._sanitize_df(df, ticker)
                
                if 'Close' in df.columns:
                    df.to_csv(fpath)
                else:
                    logger.error(f"{ticker}: Download failed to produce 'Close' column. Found: {df.columns.tolist()}")
                
                # Fetch options only during live download phase
                self._fetch_option_chain(ticker)

        logger.info("Ingestion Complete.")

# ==============================================================================
# SECTION 2: FINANCIAL ANALYSIS (PURE LOGIC)
# ==============================================================================

class FinancialAnalysis:
    """
    Computes all quant metrics.
    NO downloading allowed.
    NO plotting allowed.
    """
    def __init__(self, tickers: List[str], data_dir: str, risk_free_rate: float, flags: argparse.Namespace):
        self.tickers = tickers
        self.data_dir = data_dir
        self.r = risk_free_rate
        self.flags = flags
        self.market_data = {}
        self.option_data = {}
        self.results = {}

    def _load_data(self):
        for t in self.tickers:
            # Load Spot
            fpath = os.path.join(self.data_dir, f"{t}.csv")
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                # Verify column integrity one last time
                if 'Close' in df.columns:
                    self.market_data[t] = df
                else:
                    logger.warning(f"Skipping {t}: Missing 'Close' column in analysis load.")
            
            # Load Options
            opath = os.path.join(self.data_dir, 'options', f"{t}_options.csv")
            if os.path.exists(opath):
                self.option_data[t] = pd.read_csv(opath)

    # --- MATH HELPERS ---
    def _bs_greeks(self, S, K, T, r, sigma, type_='call'):
        """Vectorized Black-Scholes Greeks"""
        # Avoid divide by zero for very short timestamps
        T = np.maximum(T, 1e-5)
        
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate Delta vectorized using np.where
        delta = norm.cdf(d1)
        if isinstance(type_, (pd.Series, np.ndarray, list)):
             # Vectorized adjustment for puts (Call Delta - 1)
             is_put = (type_ == 'put')
             delta = np.where(is_put, delta - 1, delta)
        elif type_ == 'put':
             # Scalar fallback
             delta -= 1
            
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100 # Scaled
        
        return delta, gamma, vega

    # --- CORE ANALYSIS ---
    def analyze_market_structure(self, df):
        if 'Close' not in df.columns:
            return df

        # MAs
        for w in [5, 20, 50, 100, 200]:
            df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility (Realized)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['HV_20'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
        
        return df

    def analyze_options(self, ticker, spot_price):
        if ticker not in self.option_data:
            return None
        
        df = self.option_data[ticker].copy()
        
        # Basic cleanup
        if 'lastTradeDate' in df.columns:
            df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate']).dt.tz_localize(None)
        
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'])
            today = datetime.now()
            df['T'] = (df['expiration'] - today).dt.days / 365.0
            df = df[df['T'] > 0.001] # Remove expired/today
            
            # GEX Calculation
            # We need implied vol.
            if 'impliedVolatility' in df.columns and 'strike' in df.columns:
                df['delta'], df['gamma'], df['vega'] = self._bs_greeks(
                    spot_price, df['strike'], df['T'], self.r, df['impliedVolatility'], df['type']
                )
                
                # GEX = Gamma * OpenInterest * Spot * 100
                if 'openInterest' in df.columns:
                    df['GEX'] = df['gamma'] * df['openInterest'] * spot_price * 100
                    df.loc[df['type'] == 'put', 'GEX'] *= -1
        
        return df

    def analyze_regimes(self, df):
        """Regime detection using GMM on Volatility and Returns."""
        if 'Log_Ret' not in df.columns or 'HV_20' not in df.columns:
            return df
            
        data = df[['Log_Ret', 'HV_20']].dropna()
        if len(data) < 100:
            return df
        
        # GMM
        try:
            gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
            gmm.fit(data)
            df.loc[data.index, 'Regime'] = gmm.predict(data)
            
            # Signal Processing: Hilbert Envelope of Volatility
            analytic_signal = hilbert(df['HV_20'].fillna(0))
            amplitude_envelope = np.abs(analytic_signal)
            df['Vol_Envelope'] = amplitude_envelope
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
        
        return df

    def analyze_macro_correlations(self):
        # Requires SPY and others to be loaded.
        # Calculate rolling correlation matrix
        common_idx = None
        prices = {}
        for t, df in self.market_data.items():
            if 'Close' not in df.columns: continue
            
            if common_idx is None:
                common_idx = df.index
            else:
                common_idx = common_idx.intersection(df.index)
            prices[t] = df['Close']
        
        if not prices:
            return None
            
        price_df = pd.DataFrame(prices).loc[common_idx]
        if price_df.empty: return None
        
        returns = price_df.pct_change().dropna()
        
        # Rolling correlation (60d)
        roll_corr = returns.rolling(window=60).corr()
        
        return roll_corr

    def run(self):
        self._load_data()
        
        for t, df in self.market_data.items():
            logger.info(f"Analyzing {t}...")
            
            # 1. Market Structure
            df = self.analyze_market_structure(df)
            
            # 2. Options
            opt_res = None
            if not df.empty and 'Close' in df.columns:
                last_price = df['Close'].iloc[-1]
                opt_res = self.analyze_options(t, last_price)
            
            # 3. Regimes
            if self.flags.hmm_regimes:
                df = self.analyze_regimes(df)
            
            # Store
            self.results[t] = {
                'market': df,
                'options': opt_res
            }
            
        # 4. Macro
        self.macro_corr = self.analyze_macro_correlations()

# ==============================================================================
# SECTION 3: DASHBOARD RENDERER (OFFLINE PLOTLY)
# ==============================================================================

class DashboardRenderer:
    """
    Renders HTML.
    NO Math allowed here.
    """
    def __init__(self, analysis_results, macro_corr, flags):
        self.results = analysis_results
        self.macro_corr = macro_corr
        self.flags = flags

    def _get_plotly_js(self):
        return py_offline.get_plotlyjs()

    def _plot_market_overview(self, ticker):
        if ticker not in self.results: return "<div>No Data</div>"
        data = self.results[ticker]['market']
        if data is None or data.empty or 'Close' not in data.columns: return "<div>No Market Data</div>"
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Candlestick
        fig.add_trace(go.Candlestick(x=data.index,
                        open=data.get('Open', data['Close']), 
                        high=data.get('High', data['Close']),
                        low=data.get('Low', data['Close']), 
                        close=data['Close'], name='OHLC'), row=1, col=1)
        
        # MAs
        colors = ['cyan', 'orange', 'yellow']
        for i, ma in enumerate([20, 50, 200]):
            if f'MA_{ma}' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{ma}'], 
                                         line=dict(width=1, color=colors[i]), name=f'MA {ma}'), row=1, col=1)

        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", row=2, col=1)

        fig.update_layout(template="plotly_dark", height=600, title=f"{ticker} Market Overview")
        fig.update_xaxes(rangeslider_visible=False)
        return py_offline.plot(fig, output_type='div', include_plotlyjs=False)

    def _plot_vol_surface(self, ticker):
        if ticker not in self.results: return "<div>No Data</div>"
        opt = self.results[ticker]['options']
        if opt is None or opt.empty: return "<div>No Options Data</div>"
        
        # Filter for reasonable range
        try:
            spot = self.results[ticker]['market']['Close'].iloc[-1]
            opt = opt[(opt['strike'] > spot*0.7) & (opt['strike'] < spot*1.3)]
            
            x = opt['strike']
            y = opt['T'] * 365
            z = opt['impliedVolatility'] * 100
            
            # Grid
            xi = np.linspace(x.min(), x.max(), 50)
            yi = np.linspace(y.min(), y.max(), 50)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata((x, y), z, (XI, YI), method='linear')
            
            fig = go.Figure(data=[go.Surface(z=ZI, x=XI, y=YI, colorscale='Viridis')])
            fig.update_layout(template="plotly_dark", title=f"{ticker} IV Surface", 
                              scene=dict(xaxis_title='Strike', yaxis_title='Days to Exp', zaxis_title='IV %'),
                              height=600)
            
            return py_offline.plot(fig, output_type='div', include_plotlyjs=False)
        except Exception as e:
             return f"<div>Error plotting surface: {e}</div>"

    def _plot_gex_profile(self, ticker):
        if ticker not in self.results: return "<div>No Data</div>"
        opt = self.results[ticker]['options']
        if opt is None or opt.empty or 'GEX' not in opt.columns: return "<div>No GEX Data</div>"
        
        try:
            # Aggregate GEX by Strike
            gex_profile = opt.groupby('strike')['GEX'].sum() / 1e9 # In Billions
            
            spot = self.results[ticker]['market']['Close'].iloc[-1]
            
            fig = go.Figure()
            colors = np.where(gex_profile.values > 0, 'green', 'red')
            
            fig.add_trace(go.Bar(x=gex_profile.index, y=gex_profile.values, 
                                 marker_color=colors, name='GEX ($B)'))
            
            fig.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="Spot")
            
            # Find Zero Gamma Flip (approx)
            try:
                zero_gamma = gex_profile.rolling(2).apply(lambda x: x[0]*x[1] < 0).idxmax()
                fig.add_vline(x=zero_gamma, line_dash="dot", line_color="yellow", annotation_text="Zero Gamma")
            except:
                pass

            fig.update_layout(template="plotly_dark", title=f"{ticker} Dealer Gamma Exposure Profile ($B)", height=600)
            return py_offline.plot(fig, output_type='div', include_plotlyjs=False)
        except Exception as e:
            return f"<div>Error plotting GEX: {e}</div>"

    def _plot_regimes(self, ticker):
        if ticker not in self.results: return "<div>No Data</div>"
        df = self.results[ticker]['market']
        if 'Regime' not in df.columns: return "<div>Regimes not calculated</div>"
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'), row=1, col=1)
        
        fig.add_trace(go.Heatmap(
            x=df.index, y=[1], z=[df['Regime'].values], 
            colorscale='Hot', showscale=False
        ), row=2, col=1)
        
        if 'Vol_Envelope' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['Vol_Envelope'], name='Hilbert Vol Env', line=dict(color='yellow')), row=1, col=1)

        fig.update_layout(template="plotly_dark", title=f"{ticker} Regime Detection (HMM + Signal Proc)", height=600)
        return py_offline.plot(fig, output_type='div', include_plotlyjs=False)

    def generate_html(self, output_path):
        # Gather all plots
        tabs = {}
        for t in self.results.keys():
            tabs[f"{t}_Overview"] = self._plot_market_overview(t)
            tabs[f"{t}_VolSurface"] = self._plot_vol_surface(t)
            tabs[f"{t}_GEX"] = self._plot_gex_profile(t)
            if self.flags.hmm_regimes:
                tabs[f"{t}_Regimes"] = self._plot_regimes(t)

        # Basic HTML Template
        js_lib = self._get_plotly_js()
        
        # CSS & JS for Tabs
        css = """
        body { font-family: sans-serif; background-color: #111; color: #eee; margin: 0; }
        .tab { overflow: hidden; border-bottom: 1px solid #333; background-color: #222; }
        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888; }
        .tab button:hover { background-color: #333; color: white; }
        .tab button.active { background-color: #444; color: white; border-bottom: 2px solid cyan; }
        .tabcontent { display: none; padding: 6px 12px; border-top: none; animation: fadeEffect 1s; }
        @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
        """
        
        script = """
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
            
            // Fix Plotly resize bug on tab switch
            window.dispatchEvent(new Event('resize'));
        }
        document.addEventListener("DOMContentLoaded", function() {
            var btn = document.querySelector('.tablinks');
            if(btn) btn.click();
        });
        """

        html_parts = [
            f"<html><head><script>{js_lib}</script><style>{css}</style></head><body>",
            "<h2>QD-MDS: Quantitative Derivatives & Macro Diagnostic System</h2>",
            "<div class='tab'>"
        ]
        
        # Tab Buttons
        for name in tabs.keys():
            html_parts.append(f"<button class='tablinks' onclick=\"openTab(event, '{name}')\">{name}</button>")
        html_parts.append("</div>")
        
        # Tab Content
        for name, plot_div in tabs.items():
            html_parts.append(f"<div id='{name}' class='tabcontent'>{plot_div}</div>")
            
        html_parts.append(f"<script>{script}</script></body></html>")
        
        full_html = "\n".join(html_parts)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        logger.info(f"Dashboard saved to {output_path}")

# ==============================================================================
# SECTION 4: MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantitative Derivatives System")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'])
    parser.add_argument('--output-dir', type=str, default='./market_data')
    parser.add_argument('--lookback', type=int, default=1)
    parser.add_argument('--risk-free-rate', type=float, default=0.04)
    
    # Flags
    parser.add_argument('--mode', choices=['daily', 'intraday'], default='daily')
    parser.add_argument('--surface-overlays', action='store_true')
    parser.add_argument('--delta-pressure', action='store_true')
    parser.add_argument('--dispersion-models', action='store_true')
    parser.add_argument('--hmm-regimes', action='store_true')
    parser.add_argument('--lstm-panels', action='store_true')
    parser.add_argument('--microstructure', action='store_true')
    parser.add_argument('--neutral-signals', action='store_true')
    parser.add_argument('--monte-carlo', action='store_true')
    
    args = parser.parse_args()
    
    # 1. Ingestion
    ingestor = DataIngestion(args.tickers, args.output_dir, args.lookback)
    ingestor.run()
    
    # 2. Analysis
    analyst = FinancialAnalysis(args.tickers, args.output_dir, args.risk_free_rate, args)
    analyst.run()
    
    # 3. Rendering
    renderer = DashboardRenderer(analyst.results, analyst.macro_corr, args)
    renderer.generate_html("dashboard.html")
    
    logger.info("System execution complete.")
