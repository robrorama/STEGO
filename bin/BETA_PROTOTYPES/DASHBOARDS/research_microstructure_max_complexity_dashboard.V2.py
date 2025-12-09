# SCRIPTNAME: ok.research_microstructure_max_complexity_dashboard.V2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

# Author: Michael Derby
# Framework: Research-Grade STEGO-Inspired Standalone System (Max Complexity)
# Description:
#   Comprehensive intraday microstructure, synthetic L2, UVOL, wavelet/PSD,
#   options vol structure, AVWAP families, fractal/entropy diagnostics,
#   TA-Lib/statsmodels/sklearn factor analytics, and ML/HMM regime detection.
#   Produces a multi-tab Plotly HTML research interface with CSV exports.

import os
import sys
import argparse
import logging
import warnings
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.special import ndtr

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Quant / ML
try:
    import talib
except ImportError:
    logging.warning("TA-Lib not found. Some indicators will be skipped.")
    talib = None

import pywt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.garch import arch_model
except ImportError:
    logging.warning("Statsmodels not found. GARCH/ARIMA will be skipped.")
    sm = None

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# --- IMPORT USER LIBRARIES ---
# Assuming these are in the python path or same directory
try:
    import data_retrieval as dr
    import options_data_retrieval as odr
except ImportError:
    print("CRITICAL: data_retrieval.py or options_data_retrieval.py not found.")
    sys.exit(1)

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# --- CONSTANTS ---
BASE_OUTPUT_DIR = Path("/dev/shm/RESEARCH_MICROSTRUCTURE_MAX")

# --- MATH & PHYSICS HELPERS ---

def hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series vector ts"""
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def approx_entropy(U, m=2, r=0.2):
    """Approximate Entropy (ApEn)"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)
    if N < m + 1: return 0
    return abs(_phi(m) - _phi(m + 1))

def fractal_dimension(Z, k_max=None):
    """Higuchi Fractal Dimension"""
    L = []
    x = []
    N = len(Z)
    if k_max is None: k_max = min(10, N // 2)
    for k in range(1, k_max):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int((N - m) / k)):
                Lmk += abs(Z[m + i * k] - Z[m + (i - 1) * k])
            Lmk = Lmk * (N - 1) / (((N - m) / k) * k)
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
        x.append([np.log(1.0 / k), 1])
    (p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=None)
    return p[0]

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Greeks for a single option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = ndtr(d1)
        # Gamma is same for call/put
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        # Vanna (dDelta/dSigma)
        vanna = -stats.norm.pdf(d1) * d2 / sigma
    else:
        delta = ndtr(d1) - 1
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vanna = -stats.norm.pdf(d1) * d2 / sigma
        
    return delta, gamma, vanna

# --- CORE ANALYTICS MODULES ---

class MicrostructureEngine:
    def __init__(self, df, model_type='C'):
        self.df = df.copy()
        self.model_type = model_type
        
    def generate_synthetic_depth(self):
        """
        Generates a synthetic L2 orderbook based on OHLCV and model type.
        """
        logging.info(f"Generating Synthetic Depth Model {self.model_type}...")
        
        # Define grid relative to close
        pct_range = np.linspace(-0.02, 0.02, 40) # +/- 2%
        
        depth_data = []
        timestamps = []
        
        for idx, row in self.df.iterrows():
            mid = row['Close']
            vol = row['Volume']
            high_low_range = row['High'] - row['Low']
            
            # Base liquidity shape
            if self.model_type == 'A': # Uniform
                shape = np.ones_like(pct_range)
            elif self.model_type == 'B': # Power law
                shape = 1 / (np.abs(pct_range) + 0.001)
            else: # 'C' - Volatility Weighted Exponential
                sigma = high_low_range / mid if mid > 0 else 0.01
                shape = np.exp(-np.abs(pct_range) / (sigma * 10))
                
            # Normalize shape to represent volume distribution
            shape = shape / np.sum(shape) * vol
            
            levels = mid * (1 + pct_range)
            depth_data.append(shape)
            timestamps.append(idx)
            
        self.depth_matrix = np.array(depth_data).T # Rows = Price Levels, Cols = Time
        self.price_levels = pct_range
        self.timestamps = timestamps
        return self.depth_matrix

    def compute_order_flow_metrics(self):
        """Approximates VPIN, Microprice, and Pressure."""
        df = self.df
        
        # Simple tick rule proxy: Close > Open -> Buy, else Sell
        df['delta_p'] = df['Close'] - df['Open']
        df['trade_dir'] = np.where(df['delta_p'] > 0, 1, -1)
        
        df['up_vol'] = np.where(df['trade_dir'] == 1, df['Volume'], 0)
        df['down_vol'] = np.where(df['trade_dir'] == -1, df['Volume'], 0)
        
        # Volume Imbalance
        df['vol_imbalance'] = (df['up_vol'] - df['down_vol']) / (df['up_vol'] + df['down_vol'] + 1e-9)
        
        # Microprice proxy (Close adjusted by imbalance)
        # Using simple assumption that imbalance pushes price within the spread
        spread_proxy = (df['High'] - df['Low']).rolling(10).mean()
        df['microprice'] = df['Close'] + (df['vol_imbalance'] * spread_proxy * 0.5)
        
        # Microprice drift (velocity) and acceleration
        df['mp_velocity'] = df['microprice'].diff()
        df['mp_accel'] = df['mp_velocity'].diff()
        
        # Pressure Index
        # PI = Velocity * Imbalance * (Volume / AvgVolume)
        avg_vol = df['Volume'].rolling(20).mean()
        df['pressure_index'] = df['mp_velocity'] * df['vol_imbalance'] * (df['Volume'] / (avg_vol + 1))
        
        return df

    def compute_depth_imbalance(self, top_n=[1, 3, 5]):
        """Calculates imbalances at theoretical synthetic levels."""
        # Assuming depth_matrix is (levels, time) and center is index ~20
        center_idx = len(self.price_levels) // 2
        
        results = pd.DataFrame(index=self.timestamps)
        
        for n in top_n:
            # Slices for bids (below center) and asks (above center)
            # We take n levels below and n levels above
            bids = self.depth_matrix[center_idx-n:center_idx, :]
            asks = self.depth_matrix[center_idx+1:center_idx+1+n, :]
            
            cum_bid = np.sum(bids, axis=0)
            cum_ask = np.sum(asks, axis=0)
            
            imb = (cum_bid - cum_ask) / (cum_bid + cum_ask + 1e-9)
            results[f'depth_imb_{n}'] = imb
            
            # Rolling Z-score
            rolling_mean = pd.Series(imb).rolling(30).mean()
            rolling_std = pd.Series(imb).rolling(30).std()
            results[f'z_depth_imb_{n}'] = (imb - rolling_mean) / (rolling_std + 1e-9)
            
        return results

class UVOLEngine:
    def __init__(self, intraday_df, hist_df):
        self.df = intraday_df
        self.hist_df = hist_df
        
    def detect_shocks(self):
        logging.info("Running UVOL Shock Detection...")
        # Add minute of day
        self.df['min_of_day'] = self.df.index.hour * 60 + self.df.index.minute
        
        # Calculate baseline from history (simplified)
        # Ideally we group by min_of_day across days
        if not self.hist_df.empty:
            self.hist_df['min_of_day'] = self.hist_df.index.hour * 60 + self.hist_df.index.minute
            baseline = self.hist_df.groupby('min_of_day')['Volume'].agg(['mean', 'std'])
        else:
            # Fallback to rolling if no history
            baseline = pd.DataFrame()
            baseline['mean'] = self.df['Volume'].rolling(50).mean()
            baseline['std'] = self.df['Volume'].rolling(50).std()
            
        # Map baseline to current
        if not self.hist_df.empty:
            self.df = self.df.join(baseline, on='min_of_day', rsuffix='_base')
        else:
            self.df['mean_base'] = baseline['mean']
            self.df['std_base'] = baseline['std']
            
        # Current rolling stats
        roll_mean = self.df['Volume'].rolling(10).mean()
        roll_std = self.df['Volume'].rolling(10).std()
        
        # Z-scores
        self.df['z_anchor'] = (self.df['Volume'] - self.df['mean']) / (self.df['std'] + 1)
        self.df['z_roll'] = (self.df['Volume'] - roll_mean) / (roll_std + 1)
        self.df['z_uvol'] = np.maximum(self.df['z_anchor'].fillna(0), self.df['z_roll'].fillna(0))
        
        # Classification
        conditions = [
            (self.df['z_uvol'] >= 4),
            (self.df['z_uvol'] >= 3),
            (self.df['z_uvol'] >= 2)
        ]
        choices = ['SHOCK', 'BURST', 'WATCH']
        self.df['uvol_status'] = np.select(conditions, choices, default='NORMAL')
        return self.df

class WaveletEngine:
    def __init__(self, df, wavelet='morlet'):
        self.df = df
        self.wavelet = wavelet
        
    def run_analysis(self):
        logging.info("Computing Wavelet Transforms & PSD...")
        prices = self.df['Close'].values
        # Normalize
        prices = (prices - np.mean(prices)) / np.std(prices)
        
        # Map user-friendly names to pywt names
        # 'morlet' -> 'morl' (Real valued Morlet)
        wavelet_name = self.wavelet
        if wavelet_name == 'morlet':
            wavelet_name = 'morl'
        
        # CWT
        scales = np.arange(1, 64)
        coeffs, freqs = pywt.cwt(prices, scales, wavelet_name)
        power = (np.abs(coeffs)) ** 2
        
        # Edge Intensity (Sum of power at high freqs)
        edge_intensity = np.sum(power[:10, :], axis=0) # Top 10 smallest scales
        
        # Welch PSD (Rolling)
        # Simplified: using return variance ratios as proxy for PSD shift
        returns = self.df['Close'].pct_change().fillna(0)
        
        # Low band vs High Band Ratio (WLR)
        # High band: 2-5 bars, Low band: 10-30 bars
        std_fast = returns.rolling(5).std()
        std_slow = returns.rolling(20).std()
        wlr = std_slow / (std_fast + 1e-9)
        
        return power, edge_intensity, wlr

class OptionsEngine:
    def __init__(self, ticker):
        self.ticker = ticker
        
    def analyze_structure(self):
        logging.info("Fetching & Analyzing Options Chain...")
        try:
            # Use provided library to get expirations
            exps = odr.get_available_remote_expirations(self.ticker)
            if not exps:
                return pd.DataFrame(), pd.DataFrame()
                
            # Take nearest 2 expirations
            target_exps = exps[:2]
            
            all_options = []
            
            # Get spot price approx
            stock_df = dr.get_stock_data(self.ticker, period="1d")
            spot = stock_df['Close'].iloc[-1] if not stock_df.empty else 100
            
            for exp in target_exps:
                chain = odr.load_or_download_option_chain(self.ticker, exp, force_refresh=True)
                chain['days_to_exp'] = (exp - datetime.now()).days
                all_options.append(chain)
                
            if not all_options:
                return pd.DataFrame(), pd.DataFrame()
                
            full_chain = pd.concat(all_options)
            
            # Calculate GEX/Vanna proxies
            # Assumptions: r=0.05, sigma=impliedVolatility (if avail) or 0.3
            r = 0.05
            T = full_chain['days_to_exp'] / 365.0
            full_chain['T'] = T.apply(lambda x: max(x, 0.001)) # Avoid div/0
            
            if 'impliedVolatility' not in full_chain.columns:
                full_chain['impliedVolatility'] = 0.3
            
            # Vectorized Greeks
            # Note: yfinance data often has 'strike', 'lastPrice', 'impliedVolatility', 'openInterest'
            
            greeks = full_chain.apply(
                lambda row: black_scholes_greeks(
                    spot, row['strike'], row['T'], r, row['impliedVolatility'], row['type']
                ), axis=1
            )
            
            full_chain[['delta', 'gamma', 'vanna']] = pd.DataFrame(greeks.tolist(), index=full_chain.index)
            
            # Total GEX per strike
            # Gamma Exposure = Gamma * OpenInterest * 100 * Spot * Spot * 0.01 (Simplified: Gamma * OI * Spot)
            # Standard Dealer GEX formula: GEX = Gamma * OI * 100 * Spot^2 * 0.01 roughly
            # Here we use: Directional Gamma = Gamma * OI * 100 * Spot
            # Calls are long dealer gamma (positive), Puts are short (negative) ?? 
            # Actually: Dealers long calls -> +Gamma. Dealers short puts -> +Gamma. 
            # Convention: Dealers sell calls (short gamma), sell puts (short gamma).
            # Let's assume Dealers are Short OTM options.
            # GEX = Gamma * OI * 100 * Spot * (+1 for Call, -1 for Put) -- Standard "Spot Gamma" model
            
            full_chain['GEX'] = full_chain['gamma'] * full_chain['openInterest'] * 100 * spot * np.where(full_chain['type'] == 'call', 1, -1)
            full_chain['VannaExposure'] = full_chain['vanna'] * full_chain['openInterest'] * 100 * np.where(full_chain['type'] == 'call', 1, -1)
            
            # Aggregate by strike
            gex_surface = full_chain.groupby(['strike', 'expiration'])[['GEX', 'VannaExposure']].sum().reset_index()
            
            return full_chain, gex_surface
            
        except Exception as e:
            logging.error(f"Options analysis failed: {e}")
            return pd.DataFrame(), pd.DataFrame()

# --- MAIN DASHBOARD BUILDER ---

def run_dashboard(ticker, period, interval, depth_model, wavelet_type, n_clusters, use_hmm, lookback_days):
    
    # 1. SETUP DIRS
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = BASE_OUTPUT_DIR / ticker / today_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting analysis for {ticker}. Output: {output_dir}")
    
    # 2. DATA INGESTION
    # 2.1 Intraday
    logging.info("Fetching Intraday OHLCV...")
    # Map period/interval to yfinance format if needed, but dr handles it
    # dr.get_stock_data expects period
    # To get intraday '1m', we need download params. 
    # data_retrieval.get_stock_data uses period='1y' default.
    # We'll use yfinance directly via dr's wrapper style or just direct call to ensure 1m data
    # dr.load_or_download_ticker caches 1y. We need shorter for 1m.
    import yfinance as yf # Direct fallback for specific intraday args
    
    intraday_df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if intraday_df.empty:
        logging.error("No intraday data found.")
        return
        
    intraday_df = dr.fix_yfinance_dataframe(intraday_df)
    intraday_df.to_csv(output_dir / "raw_intraday_prices.csv")
    
    # 2.2 Historical (for baselines)
    logging.info("Fetching Historical Baseline...")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    hist_df = dr.load_or_download_ticker(ticker, start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
    hist_df.to_csv(output_dir / "historical_profile.csv")
    
    # 3. ANALYTICS EXECUTION
    
    # 3.1 Microstructure
    micro_eng = MicrostructureEngine(intraday_df, depth_model)
    depth_matrix = micro_eng.generate_synthetic_depth()
    df_orderflow = micro_eng.compute_order_flow_metrics()
    df_depth_imb = micro_eng.compute_depth_imbalance()
    
    # Save CSVs
    pd.DataFrame(depth_matrix).to_csv(output_dir / f"synthetic_depth_{depth_model}.csv")
    df_orderflow.to_csv(output_dir / "order_flow_metrics.csv")
    df_depth_imb.to_csv(output_dir / "depth_imbalance.csv")
    
    # 3.2 UVOL
    uvol_eng = UVOLEngine(df_orderflow, hist_df)
    df_uvol = uvol_eng.detect_shocks()
    df_uvol.to_csv(output_dir / "uvol_signals.csv")
    
    # 3.3 Wavelets
    wav_eng = WaveletEngine(df_uvol, wavelet_type)
    power_spectrum, edge_intensity, wlr = wav_eng.run_analysis()
    df_uvol['wavelet_edge'] = edge_intensity
    df_uvol['wlr'] = wlr
    
    # 3.4 Options
    opt_eng = OptionsEngine(ticker)
    opt_chain, gex_surface = opt_eng.analyze_structure()
    if not gex_surface.empty:
        gex_surface.to_csv(output_dir / "dealer_gex_vanna.csv")
    
    # 3.5 AVWAP
    logging.info("Computing AVWAPs...")
    # Anchor to High/Low of session
    high_idx = df_uvol['High'].idxmax()
    low_idx = df_uvol['Low'].idxmin()
    
    def compute_vwap(df, start_idx):
        sub = df.loc[start_idx:]
        v = sub['Volume']
        p = sub['Close']
        return (p * v).cumsum() / v.cumsum()
        
    df_uvol['avwap_high'] = compute_vwap(df_uvol, high_idx)
    df_uvol['avwap_low'] = compute_vwap(df_uvol, low_idx)
    df_uvol.to_csv(output_dir / "avwap_levels.csv")
    
    # 3.6 Fractal & Entropy
    logging.info("Computing Fractal Metrics...")
    # Rolling 50 period
    closes = df_uvol['Close'].values
    hursts = [np.nan]*50
    frac_dims = [np.nan]*50
    entropies = [np.nan]*50
    
    for i in range(50, len(closes)):
        window = closes[i-50:i]
        hursts.append(hurst_exponent(window))
        frac_dims.append(fractal_dimension(window))
        entropies.append(approx_entropy(window))
        
    df_uvol['hurst'] = hursts + [np.nan] * (len(closes) - len(hursts))
    df_uvol['frac_dim'] = frac_dims + [np.nan] * (len(closes) - len(frac_dims))
    df_uvol['entropy'] = entropies + [np.nan] * (len(closes) - len(entropies))
    
    # 3.7 TA-Lib
    if talib:
        df_uvol['RSI'] = talib.RSI(df_uvol['Close'])
        df_uvol['ATR'] = talib.ATR(df_uvol['High'], df_uvol['Low'], df_uvol['Close'])
        df_uvol['BB_UP'], _, df_uvol['BB_LO'] = talib.BBANDS(df_uvol['Close'])
        
    # 3.8 ML Regimes
    logging.info("Running ML Regime Detection...")
    features = df_uvol[['vol_imbalance', 'pressure_index', 'z_uvol', 'wavelet_edge', 'wlr', 'hurst', 'entropy']].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = RobustScaler()
    feat_scaled = scaler.fit_transform(features)
    
    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(feat_scaled)
    df_uvol['pca_1'] = components[:,0]
    df_uvol['pca_2'] = components[:,1]
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_uvol['cluster_regime'] = kmeans.fit_predict(feat_scaled)
    
    # HMM
    if use_hmm and HMM_AVAILABLE:
        logging.info("Fitting HMM...")
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        model.fit(feat_scaled)
        df_uvol['hmm_regime'] = model.predict(feat_scaled)
    else:
        df_uvol['hmm_regime'] = 0

    df_uvol.to_csv(output_dir / "final_master_dataset.csv")

    # 4. DASHBOARD VISUALIZATION
    logging.info("Constructing Plotly Dashboard...")
    
    # We will use Updatemenus to create 'Tabs'
    # Tabs: [Microstructure, OrderFlow, UVOL, Wavelets, Options, Fractal, ML]
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
    )
    
    # --- TRACE GROUPS ---
    # We add ALL traces to the figure, but set 'visible' to False for those not in the default tab.
    
    traces_registry = {} # key: tab_name, value: list of trace indices
    trace_counter = 0
    
    def add_trace_to_tab(trace, tab_name, row=1, col=1, secondary_y=False):
        nonlocal trace_counter
        fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)
        if tab_name not in traces_registry:
            traces_registry[tab_name] = []
        traces_registry[tab_name].append(trace_counter)
        trace_counter += 1

    # TAB 1: SYNTHETIC L2 & LIQUIDITY
    # Main Heatmap (approximated by scatter points for density or contour)
    # Using Heatmap for depth matrix is tricky on shared time axis with simple traces.
    # We will use image or dense scatter. Let's use Contour for performance if small, or Heatmap
    
    # Midprice
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['Close'], name='Close', line=dict(color='white')), 'Microstructure')
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['microprice'], name='Microprice', line=dict(color='cyan', dash='dot')), 'Microstructure')
    
    # Microstructure heatmap (simplified as bands)
    # Visualizing the depth matrix:
    # We can plot the 'center of mass' of liquidity as a fill
    # Or just top-N imbalance as bars in subplots
    add_trace_to_tab(go.Bar(x=df_uvol.index, y=df_uvol['Volume'], name='Volume', marker_color='rgba(100,100,100,0.5)'), 'Microstructure', row=4)
    
    # TAB 2: ORDER FLOW
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['Close'], name='Price'), 'OrderFlow')
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['pressure_index'], name='Pressure Idx', line=dict(color='orange')), 'OrderFlow', row=2)
    add_trace_to_tab(go.Bar(x=df_uvol.index, y=df_uvol['vol_imbalance'], name='Vol Imbalance', marker_color=np.where(df_uvol['vol_imbalance']>0, 'green', 'red')), 'OrderFlow', row=3)
    
    # TAB 3: UVOL
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['Close'], name='Price'), 'UVOL')
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['z_uvol'], name='Z-UVOL', line=dict(color='yellow')), 'UVOL', row=2)
    # Mark shocks
    shocks = df_uvol[df_uvol['uvol_status'] == 'SHOCK']
    add_trace_to_tab(go.Scatter(x=shocks.index, y=shocks['Close'], mode='markers', marker=dict(color='red', size=10, symbol='star'), name='Shock'), 'UVOL')
    
    # TAB 4: WAVELETS
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['Close'], name='Price'), 'Wavelets')
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['wavelet_edge'], name='Edge Intensity', line=dict(color='magenta')), 'Wavelets', row=2)
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['wlr'], name='WLR Ratio', line=dict(color='cyan')), 'Wavelets', row=3)
    
    # TAB 5: OPTIONS (Static mostly)
    if not gex_surface.empty:
        # Plot GEX vs Strike (Last Expiry)
        latest_exp = gex_surface['expiration'].unique()[0]
        subset = gex_surface[gex_surface['expiration'] == latest_exp]
        add_trace_to_tab(go.Bar(x=subset['strike'], y=subset['GEX'], name=f'GEX {latest_exp}'), 'Options')
    else:
         add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['Close'], name='No Options Data'), 'Options')

    # TAB 6: ML REGIMES
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['Close'], mode='markers', marker=dict(color=df_uvol['cluster_regime'], colorscale='Turbo', size=4), name='Cluster Regime'), 'ML_Regimes')
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['pca_1'], name='PCA 1'), 'ML_Regimes', row=2)
    add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['pca_2'], name='PCA 2'), 'ML_Regimes', row=2)
    if use_hmm:
        add_trace_to_tab(go.Scatter(x=df_uvol.index, y=df_uvol['hmm_regime'], name='HMM State', line_shape='hv'), 'ML_Regimes', row=3)

    # --- BUTTON LOGIC ---
    # Create buttons to toggle visibility
    tabs = list(traces_registry.keys())
    buttons = []
    
    for tab in tabs:
        # Visibility array: True for this tab's traces, False otherwise
        visible = [False] * trace_counter
        for idx in traces_registry[tab]:
            visible[idx] = True
            
        buttons.append(dict(
            label=tab,
            method="update",
            args=[{"visible": visible},
                  {"title": f"STEGO Framework: {tab} Analysis - {ticker}"}]
        ))
        
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.5,
                y=1.15,
                xanchor='center',
                yanchor='top',
                buttons=buttons
            )
        ],
        template="plotly_dark",
        title=f"STEGO Framework: {tabs[0]} Analysis - {ticker}",
        height=900
    )
    
    # Set initial visibility (Tab 1)
    initial_visible = [False] * trace_counter
    for idx in traces_registry[tabs[0]]:
        initial_visible[idx] = True
        
    # Apply initial visibility manually to the data in fig
    for i, trace in enumerate(fig.data):
        trace.visible = initial_visible[i]

    # SAVE
    html_path = output_dir / f"research_microstructure_max_complexity_dashboard.html"
    logging.info(f"Saving HTML to {html_path}...")
    fig.write_html(str(html_path))
    
    logging.info("Opening dashboard in browser...")
    webbrowser.open(f"file://{html_path}")
    
    logging.info("DONE.")

# --- ENTRY POINT ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEGO Research-Grade Microstructure Dashboard")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., NVDA)")
    parser.add_argument("--period", type=str, default="5d", help="Data period (e.g. 5d)")
    parser.add_argument("--interval", type=str, default="1m", help="Intraday interval (e.g. 1m)")
    parser.add_argument("--depth-model", type=str, choices=['A', 'B', 'C'], default='C', help="Synthetic depth model")
    parser.add_argument("--wavelet", type=str, choices=['morlet', 'mexh'], default='morlet', help="Wavelet type")
    parser.add_argument("--clusters", type=int, default=4, help="Number of ML clusters")
    parser.add_argument("--use-hmm", action='store_true', help="Enable Hidden Markov Models")
    parser.add_argument("--max-lookback-days", type=int, default=60, help="Historical baseline lookback")
    
    args = parser.parse_args()
    
    run_dashboard(
        args.ticker,
        args.period,
        args.interval,
        args.depth_model,
        args.wavelet,
        args.clusters,
        args.use_hmm,
        args.max_lookback_days
    )
