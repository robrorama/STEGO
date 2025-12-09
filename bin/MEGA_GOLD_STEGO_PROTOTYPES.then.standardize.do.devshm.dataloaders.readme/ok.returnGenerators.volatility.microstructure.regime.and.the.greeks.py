import os
import sys
import time
import argparse
import logging
import warnings
import json
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
import scipy.signal as signal
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Visualization
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Suppress minor warnings for production output
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [QUANT_ENGINE] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CLASS 1: DataIngestion
# ==============================================================================
class DataIngestion:
    """
    Handles strict I/O, downloading, sanitization, and persistence.
    No financial logic allowed here.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_dataframe(self, df, ticker):
        """
        Enforce strict data hygiene:
        1. Flatten MultiIndex columns.
        2. Coerce numeric.
        3. Drop timezone.
        4. standard column naming.
        """
        # Flatten MultiIndex if present (yfinance specific)
        if isinstance(df.columns, pd.MultiIndex):
            # Check level swap (Price, Ticker) vs (Ticker, Price)
            # We want just the price/metric name
            if ticker in df.columns.get_level_values(0):
                 df.columns = df.columns.get_level_values(1)
            else:
                 df.columns = df.columns.get_level_values(0)
        
        # If columns are just the ticker name (sometimes happens with single column request), fix it
        if len(df.columns) == 1 and df.columns[0] == ticker:
            df.columns = ['Close'] # Fallback

        # Standardize generic names to specific names if needed, or keep generic for single ticker
        # Ideally, we return a generic OHLCV frame
        
        # Coerce numeric
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Timezone naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Drop rows with NaN in critical columns
        df.dropna(subset=['Close'], inplace=True)
        
        return df

    def download_ticker(self, ticker, start_date, end_date, interval='1d'):
        """Download from API with retry logic and delays."""
        logger.info(f"Downloading {ticker} from {start_date} to {end_date} (Interval: {interval})...")
        try:
            # Explicitly set auto_adjust=False to ensure deterministic column names ('Close' vs 'Adj Close')
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                interval=interval, 
                auto_adjust=False, 
                progress=False,
                threads=False
            )
            time.sleep(1.0) # API throttling compliance
            return df
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def get_data(self, ticker, lookback_years=1.0, intraday=False, force_refresh=False):
        """
        Disk-first pipeline:
        Check CSV -> Load -> If missing/stale -> Download -> Sanitize -> Save -> Load
        """
        filename = f"{ticker}_{'intraday' if intraday else 'daily'}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculate required date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(365 * lookback_years))
        
        data_exists = os.path.exists(filepath)
        
        if data_exists and not force_refresh:
            logger.info(f"Loading {ticker} from disk...")
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # VALIDATION: Check if critical columns exist. If not, corrupt file.
                if 'Close' not in df.columns:
                    logger.warning(f"Corrupted data found for {ticker} (missing 'Close'). Triggering refresh.")
                    data_exists = False
                else:
                    # Shadow-backfill check: is data stale? (Older than 3 days)
                    last_date = df.index[-1]
                    if (end_date - last_date).days > 3:
                        logger.warning(f"Data for {ticker} is stale (Last: {last_date}). Triggering refresh.")
                        data_exists = False
                    else:
                        return df
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}. Triggering refresh.")
                data_exists = False

        if not data_exists or force_refresh:
            interval = '60m' if intraday else '1d' 
            # Note: yfinance limits 1m data to 7 days, 60m to 730 days. 
            # For intraday lookback > 7 days, we use 60m or accept API limits.
            
            raw_df = self.download_ticker(ticker, start_date, end_date, interval)
            
            if raw_df.empty:
                logger.warning(f"No data returned for {ticker}.")
                return raw_df
                
            clean_df = self._sanitize_dataframe(raw_df, ticker)
            
            if clean_df.empty:
                logger.warning(f"Data cleaned to empty for {ticker}.")
                return clean_df

            # Save to disk
            clean_df.to_csv(filepath)
            logger.info(f"Persisted clean data for {ticker} to {filepath}")
            return clean_df

    def get_option_chain(self, ticker):
        """Fetch option chain for nearest expiration (for Greeks lab)."""
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            if not exps:
                return None
            
            # Get nearest expiry
            expiry = exps[0]
            chain = tk.option_chain(expiry)
            return {'calls': chain.calls, 'puts': chain.puts, 'expiry': expiry}
        except Exception as e:
            logger.error(f"Could not fetch options for {ticker}: {e}")
            return None


# ==============================================================================
# CLASS 2: FinancialAnalysis
# ==============================================================================
class FinancialAnalysis:
    """
    Performs ALL mathematics. No I/O. No Plotly.
    """
    def __init__(self, risk_free_rate=0.04):
        self.rfr = risk_free_rate

    # --- Core Volatility & Returns ---
    def add_returns(self, df):
        # Validate 'Close' exists before math
        if 'Close' not in df.columns:
             logger.warning("Missing 'Close' column for returns calculation.")
             return df
             
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Simple_Ret'] = df['Close'].pct_change()
        return df

    def add_volatility_metrics(self, df, windows=[20, 60]):
        # Realized Volatility (Annualized)
        for w in windows:
            df[f'Realized_Vol_{w}d'] = df['Log_Ret'].rolling(window=w).std() * np.sqrt(252)
        
        # Parkinson Volatility (High-Low)
        if 'High' in df.columns and 'Low' in df.columns:
            const = 1.0 / (4.0 * np.log(2.0))
            df['Parkinson_Vol'] = np.sqrt(const * (np.log(df['High'] / df['Low'])**2)).rolling(20).mean() * np.sqrt(252)
        return df

    def add_technical_overlays(self, df):
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands for Vol Structure
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df['BB_Lower'] = rolling_mean - (rolling_std * 2)
        return df

    # --- Microstructure Proxies ---
    def calculate_microstructure(self, df):
        """
        Estimates microprice and order imbalance using OHLCV proxies.
        """
        # Ensure required columns exist
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in req_cols):
            return df

        # Order Imbalance Proxy: (Close - Open) / (High - Low)
        # Scaled by Volume
        range_len = df['High'] - df['Low']
        range_len = range_len.replace(0, np.nan) # Avoid div by zero
        
        df['Imbalance_Signal'] = ((2 * df['Close'] - df['High'] - df['Low']) / range_len) * df['Volume']
        
        # Normalize Imbalance (Rolling Z-Score)
        roll_mu = df['Imbalance_Signal'].rolling(20).mean()
        roll_std = df['Imbalance_Signal'].rolling(20).std()
        df['Imbalance_Z'] = (df['Imbalance_Signal'] - roll_mu) / roll_std
        
        # Microprice Estimate (Mid + Adjustment)
        # Mid is approx (High+Low)/2
        mid = (df['High'] + df['Low']) / 2
        # Adjustment coefficient (theoretical alpha)
        alpha = 0.5 * (df['High'] - df['Low']).mean() 
        # Sigmoid activation of imbalance Z to bound the adjustment
        adj = alpha * np.tanh(df['Imbalance_Z'].fillna(0))
        
        df['Microprice'] = mid + adj
        df['Buy_Pressure'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
        df['Sell_Pressure'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)
        
        return df

    # --- Signal Processing / Regime Detection ---
    def wavelet_energy_detector(self, df, column='Log_Ret', widths=np.arange(1, 10)):
        """
        Uses Ricker wavelet (Mexican Hat) via CWT to detect short-scale energy bursts.
        """
        if column not in df.columns:
            return df
            
        data = df[column].fillna(0).values
        # Continuous Wavelet Transform
        try:
            # Check availability dynamically to avoid crashes on limited scipy installs
            if not hasattr(signal, 'cwt') or not hasattr(signal, 'ricker'):
                raise AttributeError("scipy.signal.cwt or ricker not found")

            cwtmatr = signal.cwt(data, signal.ricker, widths)
            # Energy = squared magnitude
            energy = np.mean(cwtmatr**2, axis=0)
            df['Wavelet_Energy'] = energy
        except Exception as e:
            logger.warning(f"Wavelet transform failed (Scipy version issue?): {e}. Skipping wavelet metrics.")
            df['Wavelet_Energy'] = 0.0
            
        return df

    def welch_psd_slope_detector(self, df, column='Log_Ret', window=64):
        """
        Rolling Welch PSD slope. 
        High slope (negative) = Pink noise (Trends).
        Flat slope = White noise (Mean reversion).
        """
        if column not in df.columns:
            return df

        slopes = [np.nan] * len(df)
        vals = df[column].fillna(0).values
        
        for i in range(window, len(df)):
            segment = vals[i-window : i]
            freqs, psd = signal.welch(segment)
            # Log-Log regression for slope
            # Avoid log(0)
            valid = (freqs > 0) & (psd > 0)
            if np.sum(valid) > 5:
                slope, _, _, _, _ = stats.linregress(np.log(freqs[valid]), np.log(psd[valid]))
                slopes[i] = slope
            else:
                slopes[i] = 0
        
        df['PSD_Slope'] = slopes
        return df

    def detect_change_points(self, df, column='Realized_Vol_20d'):
        """
        Simple statistical change point detection using rolling gradient/diff spikes.
        """
        if column not in df.columns:
            return df

        # Calculate rate of change of the metric
        roc = df[column].diff().abs()
        # Z-score of ROC
        z_roc = (roc - roc.rolling(60).mean()) / roc.rolling(60).std()
        
        df['Regime_Change_Flag'] = np.where(z_roc > 3.0, 1, 0)
        return df

    # --- Options Analytics ---
    def black_scholes_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Analytical Greeks.
        """
        if T <= 0 or sigma <= 0:
            return {k: 0.0 for k in ['delta', 'gamma', 'vega', 'theta', 'rho']}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        pdf_d1 = stats.norm.pdf(d1)

        greeks = {}
        
        if option_type == 'call':
            greeks['delta'] = N_d1
            greeks['rho'] = K * T * np.exp(-r * T) * N_d2
            greeks['theta'] = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365.0
        else:
            greeks['delta'] = N_d1 - 1
            greeks['rho'] = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)
            greeks['theta'] = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365.0

        greeks['gamma'] = pdf_d1 / (S * sigma * np.sqrt(T))
        greeks['vega'] = S * np.sqrt(T) * pdf_d1 / 100.0 # Per 1% vol change
        
        return greeks

    def process_options_chain(self, chain_dict, underlying_price):
        """
        Enhance raw chain with Greeks and IV Surface data.
        """
        if not chain_dict:
            return None, None
            
        calls = chain_dict['calls'].copy()
        expiry_str = chain_dict['expiry']
        
        # Calculate time to expiry in years
        expiry_date = pd.to_datetime(expiry_str)
        now = datetime.now()
        T = (expiry_date - now).days / 365.0
        if T < 1e-5: T = 1e-5 # Avoid div by zero
        
        # Add Greeks
        greeks_list = []
        for _, row in calls.iterrows():
            iv = row['impliedVolatility']
            K = row['strike']
            # Only compute if IV is reasonable
            if iv > 0 and iv < 5:
                g = self.black_scholes_greeks(underlying_price, K, T, self.rfr, iv, 'call')
                g.update({'strike': K, 'iv': iv, 'maturity': T})
                greeks_list.append(g)
        
        greeks_df = pd.DataFrame(greeks_list)
        return calls, greeks_df

# ==============================================================================
# CLASS 3: DashboardRenderer
# ==============================================================================
class DashboardRenderer:
    """
    Handles ONLY:
    - Offline Plotly figure construction
    - HTML multi-tab layout
    - Embedding plotly.js
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def _get_plotly_config(self):
        return {'displayModeBar': True, 'responsive': True}

    def create_price_vol_tab(self, data_dict):
        """Tab 1: Price, MAs, Volatility, Correlations."""
        # Create a subplot for each ticker (Price) + 1 common for Correlation
        tickers = list(data_dict.keys())
        rows = len(tickers) + 2 # +1 for vol comparison, +1 for corr
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            subplot_titles=[f"{t} Price Action" for t in tickers] + ["Volatility Term Structure", "Rolling Correlation Heatmap"]
        )

        row_idx = 1
        for t, df in data_dict.items():
            # Price + MAs
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f"{t} Close", line=dict(width=1)), row=row_idx, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name=f"{t} SMA50", line=dict(width=1, dash='dot')), row=row_idx, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name=f"{t} BB Up", line=dict(width=0.5, color='gray'), showlegend=False), row=row_idx, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name=f"{t} BB Low", line=dict(width=0.5, color='gray'), fill='tonexty', showlegend=False), row=row_idx, col=1)
            row_idx += 1

        # Volatility Comparison
        for t, df in data_dict.items():
            fig.add_trace(go.Scatter(x=df.index, y=df['Realized_Vol_20d'], name=f"{t} RV 20d"), row=row_idx, col=1)
        row_idx += 1

        # Correlation Heatmap (Last 60 days avg)
        # Construct correlation matrix
        close_df = pd.DataFrame({t: df['Close'] for t, df in data_dict.items()})
        corr = close_df.corr()
        
        fig.add_trace(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ), row=row_idx, col=1)

        fig.update_layout(height=400*rows, title_text="Price & Volatility Overview", template="plotly_dark")
        return fig

    def create_options_tab(self, greeks_data):
        """Tab 2: IV Smile, Surface, Greeks."""
        if not greeks_data:
            fig = go.Figure()
            fig.add_annotation(text="No Options Data Available (Use --compute-greeks)", showarrow=False)
            return fig

        ticker, df = greeks_data
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=["IV Smile", "Delta vs Strike", "IV Surface (Simulated Extrapolation)", "Vega Exposure"]
        )

        # 1. IV Smile
        fig.add_trace(go.Scatter(x=df['strike'], y=df['iv'], mode='lines+markers', name='Implied Vol'), row=1, col=1)
        
        # 2. Delta
        fig.add_trace(go.Scatter(x=df['strike'], y=df['delta'], mode='lines', name='Delta', line=dict(color='orange')), row=1, col=2)

        # 3. Surface (Simulated expansion for visuals since we only pulled 1 expiry)
        # Create a meshgrid
        strikes = np.linspace(df['strike'].min(), df['strike'].max(), 20)
        maturities = np.linspace(0.01, 1.0, 20) # Simulated term structure
        X, Y = np.meshgrid(strikes, maturities)
        # Dummy smile function: IV increases away from ATM
        atm = df['strike'].median()
        Z = 0.2 + 0.0001 * (X - atm)**2 + 0.05 / np.sqrt(Y) 
        
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Plasma', name='IV Surface'), row=2, col=1)
        fig.update_scenes(xaxis_title='Strike', yaxis_title='Maturity', zaxis_title='IV', row=2, col=1)

        # 4. Vega
        fig.add_trace(go.Bar(x=df['strike'], y=df['vega'], name='Vega'), row=2, col=2)

        fig.update_layout(height=900, title_text=f"Options Lab: {ticker}", template="plotly_dark")
        return fig

    def create_microstructure_tab(self, data_dict):
        """Tab 3: Microprice, Imbalance."""
        tickers = list(data_dict.keys())
        fig = make_subplots(rows=len(tickers), cols=2, subplot_titles=[f"{t} Micro vs Mid" for t in tickers] + [f"{t} Order Imbalance" for t in tickers])
        
        for i, (t, df) in enumerate(data_dict.items()):
            row = i + 1
            # Slice last 100 points for detail
            subset = df.iloc[-100:]
            
            # Microprice
            mid = (subset['High'] + subset['Low']) / 2
            fig.add_trace(go.Scatter(x=subset.index, y=mid, name="Mid Price", line=dict(dash='dot', color='gray')), row=row, col=1)
            fig.add_trace(go.Scatter(x=subset.index, y=subset['Microprice'], name="Microprice", line=dict(color='cyan')), row=row, col=1)
            
            # Imbalance
            colors = np.where(subset['Imbalance_Z'] > 0, 'green', 'red')
            fig.add_trace(go.Bar(x=subset.index, y=subset['Imbalance_Z'], marker_color=colors, name="Imbalance Z"), row=row, col=2)

        fig.update_layout(height=400*len(tickers), title_text="Microstructure Lab (Last 100 periods)", template="plotly_dark")
        return fig

    def create_regime_tab(self, data_dict):
        """Tab 4: Wavelets, PSD, Change Points."""
        tickers = list(data_dict.keys())
        fig = make_subplots(rows=len(tickers), cols=1, subplot_titles=[f"{t} Regime Diagnostics" for t in tickers])
        
        for i, (t, df) in enumerate(data_dict.items()):
            row = i + 1
            # Normalized prices
            norm_price = (df['Close'] - df['Close'].mean()) / df['Close'].std()
            fig.add_trace(go.Scatter(x=df.index, y=norm_price, name='Norm Price', opacity=0.3), row=row, col=1)
            
            # Wavelet Energy (Secondary axis or scaled)
            fig.add_trace(go.Scatter(x=df.index, y=df['Wavelet_Energy'], name='Wavelet Energy', line=dict(color='yellow')), row=row, col=1)
            
            # PSD Slope
            fig.add_trace(go.Scatter(x=df.index, y=df['PSD_Slope'], name='Welch Slope (Noise Color)', line=dict(color='magenta')), row=row, col=1)
            
            # Regime Change Markers
            changes = df[df['Regime_Change_Flag'] == 1]
            fig.add_trace(go.Scatter(x=changes.index, y=changes['Wavelet_Energy'], mode='markers', marker=dict(symbol='star', size=10, color='red'), name='Regime Change'), row=row, col=1)

        fig.update_layout(height=400*len(tickers), title_text="Volatility Regime Detectors", template="plotly_dark")
        return fig

    def generate_html(self, figs_dict):
        """
        Assembles standalone HTML with internal CSS tabs.
        No external JS/CSS files except Plotly embed.
        """
        # Convert figures to HTML divs (include plotly.js = True ensures offline)
        # To reduce size, we can include plotly.js only once in the head, 
        # but standard to_html(include_plotlyjs=True) puts it in the div. 
        # We will use include_plotlyjs='cdn' for lightweight or True for strict offline. 
        # Prompt: "Offline Plotly with local JS inclusion". 
        # So we include_plotlyjs=True in the first plot, and False in others to save space?
        # Actually, best practice for single file is to put script in head.
        
        plot_divs = {}
        for key, fig in figs_dict.items():
            plot_divs[key] = pio.to_html(fig, full_html=False, include_plotlyjs='cdn') # Using CDN for reasonable file size, else it's 3MB per file. 
            # Note: User prompt asked for "Avoid CDNs entirely". 
            # To strictly follow, we must use include_plotlyjs=True. 
            # But putting it in every div crashes browser. We put it in the first one only.
        
        # Re-generating with smart JS inclusion
        keys = list(figs_dict.keys())
        plot_divs = {}
        for i, key in enumerate(keys):
            include_js = True if i == 0 else False
            # If include_js is False, the plot assumes Plotly is already loaded.
            # But standard `to_html` with False doesn't emit the script tag.
            # We must manually ensure the library is present.
            # Ideally, we use include_plotlyjs=True for the first one.
            # However, `to_html(include_plotlyjs=True)` embeds the FULL library (~3MB).
            # If we do it once, subsequent plots can use it? No, `to_html` usually namespaces it.
            # For simplicity and strict adherence to "Offline", we will use include_plotlyjs=True on ALL,
            # or use a CDN if the user allows. The prompt says "Avoid CDNs entirely". 
            # We will use 'directory' or True. Let's use True but be aware of file size.
            
            # Optimization: Just use CDN for the text response to avoid a 10MB text block, 
            # but add a comment that for strict offline, change to True.
            # Actually, let's use a small JS trick.
            plot_divs[key] = pio.to_html(figs_dict[key], full_html=False, include_plotlyjs=(i==0))

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Hedge Fund Analytics Dashboard</title>
            <style>
                body {{ font-family: "Open Sans", sans-serif; background: #111; color: #eee; margin: 0; }}
                .tab {{ overflow: hidden; border-bottom: 1px solid #333; background-color: #222; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888; font-weight: bold; }}
                .tab button:hover {{ background-color: #333; color: #fff; }}
                .tab button.active {{ background-color: #444; color: #4db8ff; border-bottom: 2px solid #4db8ff; }}
                .tabcontent {{ display: none; padding: 6px 12px; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>

        <div class="tab">
          <button class="tablinks" onclick="openTab(event, 'PriceVol')" id="defaultOpen">1. Price & Volatility</button>
          <button class="tablinks" onclick="openTab(event, 'Options')">2. Options & Greeks</button>
          <button class="tablinks" onclick="openTab(event, 'Microstructure')">3. Microstructure</button>
          <button class="tablinks" onclick="openTab(event, 'Regimes')">4. Regime Detection</button>
        </div>

        <div id="PriceVol" class="tabcontent">
          {plot_divs['tab1']}
        </div>

        <div id="Options" class="tabcontent">
          {plot_divs['tab2']}
        </div>

        <div id="Microstructure" class="tabcontent">
          {plot_divs['tab3']}
        </div>

        <div id="Regimes" class="tabcontent">
          {plot_divs['tab4']}
        </div>

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
            
            // Dispatch resize event to fix Plotly rendering in hidden tabs
            window.dispatchEvent(new Event('resize'));
        }}
        // Get the element with id="defaultOpen" and click on it
        document.getElementById("defaultOpen").click();
        </script>
        
        </body>
        </html>
        """
        
        output_path = os.path.join(self.output_dir, "dashboard.html")
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_template)
        logger.info(f"Dashboard generated at: {output_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Analytics Engine")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='List of tickers')
    parser.add_argument('--output-dir', default='./market_data', help='Output directory')
    parser.add_argument('--lookback', type=float, default=1.0, help='Years of history')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate')
    parser.add_argument('--intraday', action='store_true', help='Use intraday data')
    parser.add_argument('--compute-greeks', action='store_true', help='Compute numerical options Greeks')
    
    args = parser.parse_args()
    
    # 1. Setup
    ingestion = DataIngestion(args.output_dir)
    analysis = FinancialAnalysis(risk_free_rate=args.risk_free_rate)
    renderer = DashboardRenderer(args.output_dir)
    
    processed_data = {}
    greeks_data = None
    
    # 2. Ingestion & Analysis Loop
    for ticker in args.tickers:
        logger.info(f"Processing {ticker}...")
        df = ingestion.get_data(ticker, lookback_years=args.lookback, intraday=args.intraday)
        
        if df.empty:
            continue
            
        # Analysis Pipeline
        df = analysis.add_returns(df)
        df = analysis.add_volatility_metrics(df)
        df = analysis.add_technical_overlays(df)
        df = analysis.calculate_microstructure(df)
        df = analysis.wavelet_energy_detector(df)
        df = analysis.welch_psd_slope_detector(df)
        df = analysis.detect_change_points(df)
        
        processed_data[ticker] = df
        
        # Optional: Greeks (Only for first ticker to save time/requests for this demo)
        if args.compute_greeks and greeks_data is None:
            logger.info(f"Fetching option chain for {ticker}...")
            chain = ingestion.get_option_chain(ticker)
            if chain:
                calls, greeks_df = analysis.process_options_chain(chain, df['Close'].iloc[-1])
                greeks_data = (ticker, greeks_df)
    
    # 3. Visualization
    if not processed_data:
        logger.error("No data processed. Exiting.")
        return

    logger.info("Generating Dashboard...")
    figs = {
        'tab1': renderer.create_price_vol_tab(processed_data),
        'tab2': renderer.create_options_tab(greeks_data),
        'tab3': renderer.create_microstructure_tab(processed_data),
        'tab4': renderer.create_regime_tab(processed_data)
    }
    
    renderer.generate_html(figs)
    logger.info("Complete.")

if __name__ == "__main__":
    main()
