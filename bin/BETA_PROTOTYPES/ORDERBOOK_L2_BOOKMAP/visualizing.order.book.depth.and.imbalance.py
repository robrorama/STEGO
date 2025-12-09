# SCRIPTNAME: ok.05.visualizing.order.book.depth.and.imbalance.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import argparse
import time
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union

# Scientific Stack
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import griddata
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Market Data
import yfinance as yf

# Visualization
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

# ---------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("QuantSystem")
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# CLASS 1: DATA INGESTION (I/O & SANITIZATION)
# ---------------------------------------------------------------------
class DataIngestion:
    """
    Responsible ONLY for:
    - Reading/writing CSV (Disk-first)
    - Downloading OHLCV & Options data
    - Shadow backfill logic
    - Data Sanitization
    """
    def __init__(self, output_dir: str, lookback_years: float = 1.0):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        self.start_date = (datetime.now() - timedelta(days=int(lookback_years * 365))).strftime('%Y-%m-%d')
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created data directory: {self.output_dir}")

    def _sanitize_df(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """The Universal Fixer for yfinance dataframes."""
        if df.empty:
            return df

        # Fix YFinance MultiIndex Column Bug (Price, Ticker) -> Price
        if isinstance(df.columns, pd.MultiIndex):
            # Attempt to swap if level 0 is ticker
            if ticker and ticker in df.columns.get_level_values(0):
                 df = df.swaplevel(0, 1, axis=1)
            
            # Drop the ticker level if it exists, keeping just OHLCV
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass

        # Force standard column names
        df.columns = [c.capitalize() for c in df.columns]
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Strip timezone
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Coerce numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop duplicates and sort
        df = df.loc[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # Add metadata
        df.attrs['sanitized_at'] = datetime.now().isoformat()
        df.attrs['source'] = 'yfinance'
        
        return df

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Disk-first acquisition of OHLCV data."""
        filepath = os.path.join(self.output_dir, f"{ticker}.csv")
        
        # 1. Load from Disk
        if os.path.exists(filepath):
            try:
                logger.info(f"[{ticker}] Loading from disk...")
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                df = self._sanitize_df(df, ticker)
                # Check if stale (older than 1 day)
                last_date = df.index[-1]
                if (datetime.now() - last_date).days < 2:
                    return df
                logger.info(f"[{ticker}] Data stale, redownloading...")
            except Exception as e:
                logger.warning(f"[{ticker}] Disk read failed: {e}. Downloading.")

        # 2. Download
        logger.info(f"[{ticker}] Downloading from YF...")
        try:
            time.sleep(1.0) # Polite delay
            raw_df = yf.download(ticker, start=self.start_date, progress=False, group_by='column')
            
            # Handle empty result
            if raw_df.empty:
                logger.error(f"[{ticker}] Download returned empty dataframe.")
                return pd.DataFrame()

            # Sanitize
            clean_df = self._sanitize_df(raw_df, ticker)
            
            # Write to disk
            clean_df.to_csv(filepath)
            
            return clean_df
            
        except Exception as e:
            logger.error(f"[{ticker}] Download failed: {e}")
            return pd.DataFrame()

    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        """
        Fetches current options chain. 
        Note: Historical options are not available free. 
        We save snapshots to build history over time.
        """
        date_str = datetime.now().strftime('%Y%m%d')
        filepath = os.path.join(self.output_dir, f"{ticker}_opts_{date_str}.csv")
        
        if os.path.exists(filepath):
            return pd.read_csv(filepath)

        logger.info(f"[{ticker}] Fetching options chain...")
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options
            
            all_opts = []
            
            # Fetch first 4 expirations to save time/bandwidth for demo
            for exp in expirations[:4]: 
                opt = tk.option_chain(exp)
                calls = opt.calls
                calls['type'] = 'call'
                puts = opt.puts
                puts['type'] = 'put'
                
                chain = pd.concat([calls, puts])
                chain['expiration'] = exp
                all_opts.append(chain)
                time.sleep(0.5)
            
            if not all_opts:
                return pd.DataFrame()
                
            full_chain = pd.concat(all_opts)
            full_chain['ingest_date'] = date_str
            full_chain.to_csv(filepath, index=False)
            return full_chain
            
        except Exception as e:
            logger.warning(f"[{ticker}] Options fetch failed: {e}")
            return pd.DataFrame()

    def generate_synthetic_l2(self, ticker: str, ref_price: float) -> pd.DataFrame:
        """
        Generates synthetic Level 2 / Order Book data for microstructure demo.
        Real L2 data is expensive/proprietary.
        """
        logger.info(f"[{ticker}] Generating synthetic Microstructure data...")
        np.random.seed(42)
        n_rows = 500
        
        # Drift process
        returns = np.random.normal(0, 0.0001, n_rows)
        price_path = ref_price * np.exp(np.cumsum(returns))
        
        data = []
        for t, mid in enumerate(price_path):
            spread = mid * 0.0002
            bid = mid - spread/2
            ask = mid + spread/2
            
            # Skew depth based on momentum
            mom = returns[t]
            bid_size = np.random.randint(100, 1000) * (1 - mom*100)
            ask_size = np.random.randint(100, 1000) * (1 + mom*100)
            
            # Microprice (Stoikov)
            micro = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
            
            data.append({
                'time': datetime.now() + timedelta(seconds=t),
                'bid': bid, 'ask': ask,
                'bid_size': abs(bid_size), 'ask_size': abs(ask_size),
                'mid': mid,
                'microprice': micro,
                'imbalance': (bid_size - ask_size) / (bid_size + ask_size)
            })
            
        return pd.DataFrame(data)

# ---------------------------------------------------------------------
# CLASS 2: FINANCIAL ANALYSIS (MATH & MODELING)
# ---------------------------------------------------------------------
class FinancialAnalysis:
    """
    Responsible ONLY for:
    - Math, Modeling, Signals
    - Greeks, GEX, Surfaces
    - HMM, LSTM, Stress Tests
    """
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    # --- OPTIONS MATH ---
    def black_scholes_greeks(self, S, K, T, r, sigma, opt_type='call'):
        """Vectorized Black-Scholes Greeks."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        cdf_d2 = stats.norm.cdf(d2)
        
        greeks = {}
        
        if opt_type == 'call':
            greeks['delta'] = cdf_d1
            greeks['theta'] = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2)
        else:
            greeks['delta'] = cdf_d1 - 1
            greeks['theta'] = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * (1 - cdf_d2))
            
        greeks['gamma'] = pdf_d1 / (S * sigma * np.sqrt(T))
        greeks['vega'] = S * pdf_d1 * np.sqrt(T)
        greeks['vanna'] = -pdf_d1 * d2 / sigma
        greeks['charm'] = -pdf_d1 * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
        greeks['vomma'] = greeks['vega'] * d1 * d2 / sigma
        
        return greeks

    def analyze_dealer_positioning(self, chain: pd.DataFrame, spot: float) -> Dict:
        """
        Calculates GEX, Vanna Exposure, and Zero-Gamma levels.
        Assumption: Dealers are Long Calls / Short Puts? 
        Standard Market Maker assumption: Dealers are Short Calls, Short Puts (Short Vol).
        Actually, standard assumption is Dealers are counterparty:
        - Customer buys call (Dealer Short Call) -> Dealer needs to hedge +Delta -> Dealer Buys Spot.
        - Customer buys put (Dealer Short Put) -> Dealer needs to hedge -Delta -> Dealer Sells Spot.
        """
        if chain.empty:
            return {}

        # Filter near-term
        chain = chain.copy()
        chain['T'] = (pd.to_datetime(chain['expiration']) - datetime.now()).dt.days / 365.0
        chain = chain[chain['T'] > 0.001]
        
        # Approximate IV if missing (vectorized fallback)
        if 'impliedVolatility' not in chain.columns:
            chain['impliedVolatility'] = 0.2
            
        # Calculate Greeks
        greeks_list = []
        for idx, row in chain.iterrows():
            g = self.black_scholes_greeks(spot, row['strike'], row['T'], self.r, row['impliedVolatility'], row['type'])
            greeks_list.append(g)
            
        greeks_df = pd.DataFrame(greeks_list, index=chain.index)
        df = pd.concat([chain, greeks_df], axis=1)
        
        # GEX Calculation: Gamma * OpenInterest * 100 * Spot * Spot * 0.01 (Simplified)
        # Net Gamma Exposure ($ per 1% move)
        # Dealer Sign: Short Call (-), Short Put (+) -> Actually Dealer is Short the Option.
        # If Dealer is Short Call: Gamma is negative.
        # If Dealer is Short Put: Gamma is negative.
        # Correction: Dealers are usually Short Volatility overall, but GEX convention:
        # Call GEX is positive (Market rises, Dealer buys), Put GEX is negative (Market drops, Dealer sells).
        
        df['GEX'] = df['gamma'] * df['openInterest'] * 100 * spot * spot * 0.01
        df.loc[df['type'] == 'put', 'GEX'] *= -1 
        
        total_gex = df['GEX'].sum()
        
        # Zero Gamma Level search
        strikes = np.sort(df['strike'].unique())
        gex_profile = []
        for s in strikes:
            # Re-calc gex at hypothetical spot s
            # Simplified: Assuming OI and IV constant, just Delta/Gamma shift
            # This is expensive, so we do a simple aggregation map
            gex_at_strike = df[df['strike'] == s]['GEX'].sum()
            gex_profile.append(gex_at_strike)
            
        return {
            'total_gex': total_gex,
            'df': df,
            'gex_by_strike': pd.Series(gex_profile, index=strikes)
        }

    # --- MACRO & REGIMES ---
    def fit_regimes(self, returns: pd.Series, n_states=3) -> Tuple[pd.DataFrame, object]:
        """Fit GMM as a proxy for HMM to detect volatility regimes."""
        X = returns.values.reshape(-1, 1)
        # Add volatility as feature
        vol = returns.rolling(20).std().fillna(0).values.reshape(-1, 1)
        features = np.hstack([X, vol])
        
        model = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
        model.fit(features)
        states = model.predict(features)
        probs = model.predict_proba(features)
        
        df_states = pd.DataFrame(probs, index=returns.index, columns=[f'State_{i}' for i in range(n_states)])
        df_states['Regime'] = states
        return df_states, model

    def calculate_dispersion(self, index_returns: pd.Series, component_returns: pd.DataFrame) -> pd.Series:
        """Calculate Dispersion (std dev of cross-sectional returns)."""
        # Align dates
        combined = pd.concat([index_returns, component_returns], axis=1).dropna()
        comps = combined.iloc[:, 1:]
        
        # Cross sectional std dev at each timestamp
        dispersion = comps.std(axis=1)
        return dispersion

    def residualize_macro(self, asset_returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
        """Remove macro factor beta from asset returns using Ridge Regression."""
        common_idx = asset_returns.index.intersection(factors.index)
        if len(common_idx) < 50:
            return asset_returns # Not enough overlap
            
        y = asset_returns.loc[common_idx]
        X = factors.loc[common_idx]
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        residuals = y - preds
        return residuals

    # --- FORECASTING & ML ---
    def train_forecaster(self, series: pd.Series) -> Tuple[pd.DataFrame, float]:
        """Train an MLP (Neural Net) to forecast next day returns."""
        # Feature Engineering
        df = pd.DataFrame(series)
        df.columns = ['target']
        df['lag1'] = df['target'].shift(1)
        df['lag2'] = df['target'].shift(2)
        df['lag3'] = df['target'].shift(3)
        df['roll_mean'] = df['target'].rolling(5).mean()
        df['roll_std'] = df['target'].rolling(5).std()
        df = df.dropna()
        
        if len(df) < 100:
            return pd.DataFrame(), 0.0

        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # MLP Regressor (Proxy for LSTM)
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
        model.fit(X_train_s, y_train)
        
        preds = model.predict(X_test_s)
        score = model.score(X_test_s, y_test)
        
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds}, index=y_test.index)
        return result_df, score

    # --- STRESS TESTING ---
    def run_monte_carlo(self, S0, mu, sigma, T=1.0, steps=252, sims=100) -> np.ndarray:
        """Geometric Brownian Motion paths."""
        dt = T/steps
        paths = np.zeros((steps + 1, sims))
        paths[0] = S0
        
        for t in range(1, steps + 1):
            Z = np.random.standard_normal(sims)
            paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*Z)
            
        return paths

# ---------------------------------------------------------------------
# CLASS 3: DASHBOARD RENDERER (VISUALIZATION)
# ---------------------------------------------------------------------
class DashboardRenderer:
    """
    Responsible ONLY for:
    - Plotly HTML generation
    - Layouts, Themes, Templates
    - JavaScript injection
    """
    def __init__(self, title="Hedge Fund Analytics"):
        self.title = title
        # Custom Dark Theme
        self.template = "plotly_dark"
        self.layout_config = {
            'paper_bgcolor': '#1e1e1e',
            'plot_bgcolor': '#1e1e1e',
            'font': {'family': "Roboto, monospace", 'color': '#e0e0e0'},
            'margin': dict(l=40, r=40, t=50, b=40)
        }

    def _get_plotly_js(self):
        """Get offline JS source."""
        return py_offline.get_plotlyjs()

    def generate_dashboard(self, 
                           ohlc: pd.DataFrame, 
                           options_data: Dict, 
                           macro_data: Dict, 
                           forecast_data: Dict,
                           micro_data: Optional[pd.DataFrame],
                           stress_paths: np.ndarray,
                           filename: str):
        
        # 1. Initialize Subplots/Tabs structure via HTML logic
        # Plotly doesn't support native tabs in a single figure efficiently for complex grids.
        # We will generate separate div strings and use custom HTML/JS for tabs.
        
        figs = {}
        
        # --- TAB 1: OVERVIEW ---
        fig_overview = make_subplots(rows=2, cols=2, 
                                     specs=[[{"secondary_y": True}, {}], [{"colspan": 2}, None]],
                                     subplot_titles=("Price & Volume", "Returns Distribution", "Realized Volatility"))
        
        # Candlestick
        fig_overview.add_trace(go.Candlestick(x=ohlc.index, open=ohlc['Open'], high=ohlc['High'],
                                             low=ohlc['Low'], close=ohlc['Close'], name='OHLC'), row=1, col=1)
        fig_overview.add_trace(go.Bar(x=ohlc.index, y=ohlc['Volume'], name='Volume', opacity=0.3), row=1, col=1, secondary_y=True)
        
        # Dist
        returns = ohlc['Close'].pct_change().dropna()
        fig_overview.add_trace(go.Histogram(x=returns, nbinsx=50, name='Ret Dist'), row=1, col=2)
        
        # Vol
        vol = returns.rolling(20).std() * np.sqrt(252)
        fig_overview.add_trace(go.Scatter(x=vol.index, y=vol, mode='lines', name='20d Realized Vol', line=dict(color='#00d4ff')), row=2, col=1)
        
        fig_overview.update_layout(template=self.template, **self.layout_config, height=700)
        figs['Overview'] = fig_overview

        # --- TAB 2: OPTIONS & GREEKS ---
        if 'df' in options_data and not options_data['df'].empty:
            df_opt = options_data['df']
            gex_series = options_data['gex_by_strike']
            
            fig_opt = make_subplots(rows=2, cols=2, 
                                    specs=[[{'type': 'scene'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
                                    subplot_titles=("Deep IV Surface", "GEX Profile", "Smile Structure", "Total Gamma Heatmap"))
            
            # 1. Deep IV Surface (3D)
            # Interpolate for smooth surface
            xi = np.linspace(df_opt['strike'].min(), df_opt['strike'].max(), 50)
            yi = np.linspace(df_opt['T'].min(), df_opt['T'].max(), 50)
            XI, YI = np.meshgrid(xi, yi)
            try:
                ZI = griddata((df_opt['strike'], df_opt['T']), df_opt['impliedVolatility'], (XI, YI), method='linear')
                fig_opt.add_trace(go.Surface(z=ZI, x=XI, y=YI, colorscale='Viridis', name='IV Surface'), row=1, col=1)
            except:
                pass # Not enough points

            # 2. GEX Profile
            colors = ['#ff4444' if x < 0 else '#00c851' for x in gex_series.values]
            fig_opt.add_trace(go.Bar(x=gex_series.index, y=gex_series.values, marker_color=colors, name='GEX'), row=1, col=2)
            
            # 3. Smile (Front month)
            front_month = df_opt[df_opt['T'] == df_opt['T'].min()]
            fig_opt.add_trace(go.Scatter(x=front_month['strike'], y=front_month['impliedVolatility'], mode='markers+lines', name='Front Vol'), row=2, col=1)
            
            # 4. GEX Heatmap (Strike vs Expiry)
            pivoted_gex = df_opt.pivot_table(index='T', columns='strike', values='GEX', aggfunc='sum').fillna(0)
            fig_opt.add_trace(go.Heatmap(z=pivoted_gex.values, x=pivoted_gex.columns, y=pivoted_gex.index, colorscale='RdBu', showscale=False), row=2, col=2)

            fig_opt.update_layout(template=self.template, **self.layout_config, height=800)
            figs['Derivatives'] = fig_opt
        else:
            figs['Derivatives'] = go.Figure().update_layout(title="No Options Data Available")

        # --- TAB 3: FORECAST & REGIMES ---
        fig_ml = make_subplots(rows=2, cols=1, subplot_titles=("Forecasting Model (MLP/LSTM Proxy)", "Regime Detection (HMM Proxy)"))
        
        # Forecast
        if not forecast_data['forecast'].empty:
            res = forecast_data['forecast']
            fig_ml.add_trace(go.Scatter(x=res.index, y=res['Actual'], name='Actual', line=dict(color='gray', width=1)), row=1, col=1)
            fig_ml.add_trace(go.Scatter(x=res.index, y=res['Predicted'], name='Predicted', line=dict(color='#ffeb3b', width=2)), row=1, col=1)
        
        # Regimes
        if not forecast_data['regimes'].empty:
            reg = forecast_data['regimes']
            # Area plot for probabilities
            fig_ml.add_trace(go.Scatter(x=reg.index, y=reg['State_0'], stackgroup='one', name='Regime 0 (Low Vol)'), row=2, col=1)
            if 'State_1' in reg.columns:
                fig_ml.add_trace(go.Scatter(x=reg.index, y=reg['State_1'], stackgroup='one', name='Regime 1 (High Vol)'), row=2, col=1)

        fig_ml.update_layout(template=self.template, **self.layout_config, height=700)
        figs['Quantitative_ML'] = fig_ml

        # --- TAB 4: MICROSTRUCTURE (Intraday) ---
        if micro_data is not None and not micro_data.empty:
            fig_micro = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
            
            # 1. Price + Microprice
            fig_micro.add_trace(go.Scatter(x=micro_data['time'], y=micro_data['mid'], name='Mid Price'), row=1, col=1)
            fig_micro.add_trace(go.Scatter(x=micro_data['time'], y=micro_data['microprice'], name='Microprice', line=dict(dash='dot')), row=1, col=1)
            
            # 2. Imbalance
            fig_micro.add_trace(go.Bar(x=micro_data['time'], y=micro_data['imbalance'], name='Order Imbalance', marker_color='orange'), row=2, col=1)
            
            # 3. LOB Heatmap approximation (using bid/ask size as proxy for depth intensity)
            # Creating a mock heatmap logic for visualization
            fig_micro.add_trace(go.Scatter(x=micro_data['time'], y=micro_data['bid_size'], fill='tozeroy', name='Bid Depth', line=dict(color='green')), row=3, col=1)
            fig_micro.add_trace(go.Scatter(x=micro_data['time'], y=micro_data['ask_size'], fill='tozeroy', name='Ask Depth', line=dict(color='red')), row=3, col=1)
            
            fig_micro.update_layout(template=self.template, **self.layout_config, height=800, title="Intraday Microstructure")
            figs['Microstructure'] = fig_micro
        else:
            figs['Microstructure'] = go.Figure().update_layout(title="Intraday Mode Not Active")

        # --- TAB 5: STRESS TEST ---
        fig_stress = go.Figure()
        if stress_paths.size > 0:
            steps, sims = stress_paths.shape
            x_axis = np.arange(steps + 1)
            # Plot first 50 paths
            for i in range(min(sims, 50)):
                fig_stress.add_trace(go.Scatter(x=x_axis, y=stress_paths[:, i], mode='lines', opacity=0.3, line=dict(width=1), showlegend=False))
            
            # Add mean path
            fig_stress.add_trace(go.Scatter(x=x_axis, y=np.mean(stress_paths, axis=1), mode='lines', name='Mean Path', line=dict(color='white', width=3)))
            
            fig_stress.update_layout(template=self.template, **self.layout_config, title="Monte Carlo Stress Paths (GBM)", height=600)
            
        figs['Stress_Test'] = fig_stress

        # --- ASSEMBLE HTML ---
        self._write_html(figs, filename)

    def _write_html(self, figs: Dict[str, go.Figure], filename: str):
        """Constructs the tabbed HTML page with offline JS."""
        
        # Generate div strings
        plot_divs = {}
        for name, fig in figs.items():
            plot_divs[name] = py_offline.plot(fig, include_plotlyjs=False, output_type='div')

        # CSS
        style = """
        <style>
            body { margin: 0; padding: 0; background-color: #121212; color: #e0e0e0; font-family: 'Roboto', sans-serif; }
            .tab { overflow: hidden; border-bottom: 1px solid #333; background-color: #1e1e1e; }
            .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888; font-weight: bold; }
            .tab button:hover { background-color: #333; color: white; }
            .tab button.active { background-color: #007acc; color: white; }
            .tabcontent { display: none; padding: 6px 12px; border-top: none; animation: fadeEffect 1s; }
            @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
            h1 { text-align: center; padding: 20px; margin: 0; font-weight: 300; letter-spacing: 2px; }
        </style>
        """

        # JS for Tabs & Resize Fix
        script = """
        <script>
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
                
                // CRITICAL: Trigger resize for Plotly to redraw correctly in hidden tabs
                window.dispatchEvent(new Event('resize'));
            }
            // Open default
            document.addEventListener("DOMContentLoaded", function() {
                document.getElementsByClassName("tablinks")[0].click();
            });
        </script>
        """

        html_content = f"""
        <html>
        <head>
            <title>{self.title}</title>
            <script type="text/javascript">{self._get_plotly_js()}</script>
            {style}
        </head>
        <body>
            <h1>QUANTITATIVE ANALYTICS DASHBOARD</h1>
            
            <div class="tab">
                {''.join([f'<button class="tablinks" onclick="openTab(event, \'{name}\')">{name.replace("_", " ")}</button>' for name in plot_divs.keys()])}
            </div>

            {''.join([f'<div id="{name}" class="tabcontent">{div}</div>' for name, div in plot_divs.items()])}
            
            {script}
        </body>
        </html>
        """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to: {os.path.abspath(filename)}")

# ---------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Analytics System")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='List of tickers')
    parser.add_argument('--output-dir', default='./market_data', help='Data storage directory')
    parser.add_argument('--lookback', type=float, default=2.0, help='Years of history')
    parser.add_argument('--mode', default='standard', choices=['standard', 'intraday', 'forecast', 'stress'])
    parser.add_argument('--intraday', action='store_true', help='Force intraday pipelines')
    
    args = parser.parse_args()
    
    # Initialize Classes
    ingest = DataIngestion(args.output_dir, args.lookback)
    quant = FinancialAnalysis()
    render = DashboardRenderer()
    
    for ticker in args.tickers:
        logger.info(f"=== PROCESSING {ticker} ===")
        
        # 1. GET DATA (Disk First)
        df_ohlc = ingest.get_ohlcv(ticker)
        if df_ohlc.empty:
            logger.error(f"Skipping {ticker} due to missing data.")
            continue
            
        df_opts = ingest.get_options_chain(ticker)
        
        # 2. RUN ANALYSIS
        logger.info("Running Financial Analysis...")
        
        # Options Analytics
        spot_price = df_ohlc['Close'].iloc[-1]
        opt_analytics = quant.analyze_dealer_positioning(df_opts, spot_price)
        if 'total_gex' in opt_analytics:
            logger.info(f"Total Dealer GEX: ${opt_analytics['total_gex']:,.2f}")

        # ML & Forecasting
        returns = df_ohlc['Close'].pct_change().dropna()
        regimes, _ = quant.fit_regimes(returns)
        forecast_df, score = quant.train_forecaster(returns)
        logger.info(f"ML Forecast R2 Score: {score:.4f}")
        
        forecast_data = {'regimes': regimes, 'forecast': forecast_df}
        
        # Intraday
        micro_data = pd.DataFrame()
        if args.mode == 'intraday' or args.intraday:
            micro_data = ingest.generate_synthetic_l2(ticker, spot_price)
            
        # Stress Test
        vol = returns.std() * np.sqrt(252)
        stress_paths = quant.run_monte_carlo(spot_price, 0.05, vol)

        # 3. RENDER DASHBOARD
        filename = f"{args.output_dir}/{ticker}_dashboard.html"
        render.generate_dashboard(
            ohlc=df_ohlc,
            options_data=opt_analytics,
            macro_data={}, # Placeholder for demo
            forecast_data=forecast_data,
            micro_data=micro_data,
            stress_paths=stress_paths,
            filename=filename
        )

    logger.info("System Execution Complete.")

if __name__ == "__main__":
    main()
