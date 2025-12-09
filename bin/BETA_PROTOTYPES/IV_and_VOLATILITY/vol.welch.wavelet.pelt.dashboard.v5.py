# SCRIPTNAME: ok.vol.welch.wavelet.pelt.dashboard.v5.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
vol_signal_dashboard.py

Objective:
    Production-grade Quantitative Analysis Dashboard for Volatility Signal Processing.
    Decouples Data Ingestion (IO/Sanitization) from Analytics (Math/Signal Processing).

Architecture:
    1. VolDataIngestion: Handles yfinance, disk caching (raw_data/), and strict sanitization.
    2. VolAnalytics: Implements Welch PSD, Wavelet CWT, PELT, and Black-Scholes IV.
    3. Dashboard: Generates interactive Plotly HTML visualizations.

Dependencies:
    pandas, numpy, yfinance, plotly, scipy
    Optional: pywt (PyWavelets), ruptures (for regime detection)

Author: Senior Quant Developer
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats, signal
from datetime import datetime, timedelta

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional High-Performance Libraries with Graceful Degradation
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("Warning: 'pywt' not found. Wavelet analysis will be skipped.")

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("Warning: 'ruptures' not found. PELT regime detection will be skipped.")

# --- Configuration ---
RAW_DATA_DIR = "raw_data"
RISK_FREE_RATE = 0.04
DEFAULT_TICKER = "^VIX"  # For volatility analysis
DEFAULT_OPTION_TICKER = "SPY"  # For IV Surface

class VolDataIngestion:
    """
    Handles all IO operations, API interaction, local caching, and data sanitization.
    Enforces the 'Clean Data' philosophy.
    """
    def __init__(self, base_dir=RAW_DATA_DIR):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"[IO] Created cache directory: {self.base_dir}")

    def _sanitize_df(self, df: pd.DataFrame, is_ohlc=True) -> pd.DataFrame:
        """
        Private Janitor Method:
        1. Flattens MultiIndex columns (yfinance quirk).
        2. Strips Timezones (Plotly/Scipy compatibility).
        3. Regularizes Frequency (Business Days) to ensure uniform time steps.
        """
        df = df.copy()

        # 1. Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Timezone Stripping
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 3. Regularization (Strictly required for Welch/Wavelet)
        if is_ohlc:
            # Resample to Business Day, forward fill missing (e.g. holidays)
            df = df.asfreq('B').ffill().dropna()
            
            # Ensure numeric types
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

        return df

    def get_history(self, ticker: str, period="2y") -> pd.DataFrame:
        """
        Check Disk -> Load -> Sanitize -> Return.
        Else: Download -> Sanitize -> Save -> Return.
        """
        file_path = os.path.join(self.base_dir, f"{ticker}_history.csv")

        if os.path.exists(file_path):
            print(f"[IO] Loading historical data from cache: {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return self._sanitize_df(df)
        
        print(f"[IO] Downloading fresh data for {ticker}...")
        df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        df = self._sanitize_df(df)
        df.to_csv(file_path)
        return df

    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        """
        Fetches option chains.
        Cache Strategy: Caches based on Ticker + Today's Date.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(self.base_dir, f"{ticker}_{today_str}_chain.csv")

        if os.path.exists(file_path):
            print(f"[IO] Loading options chain from cache: {file_path}")
            df = pd.read_csv(file_path, index_col=0)
            # Ensure expiration is datetime
            df['expirationDate'] = pd.to_datetime(df['expirationDate'])
            return df

        print(f"[IO] Downloading options chain for {ticker} (This may take time)...")
        tk = yf.Ticker(ticker)
        expirations = tk.options
        all_opts = []

        # Rate limiting logic
        for exp in expirations[:12]: # Limit to nearest 12 expirations for speed
            try:
                opt = tk.option_chain(exp)
                calls = opt.calls
                calls['type'] = 'call'
                calls['expirationDate'] = pd.to_datetime(exp)
                all_opts.append(calls)
                
                puts = opt.puts
                puts['type'] = 'put'
                puts['expirationDate'] = pd.to_datetime(exp)
                all_opts.append(puts)
                
                time.sleep(0.2) # Be polite to API
            except Exception as e:
                print(f"[IO] Failed to fetch {exp}: {e}")

        if not all_opts:
            raise ValueError("Could not fetch any options data.")

        df = pd.concat(all_opts)
        
        # Basic Type Enforcement before save
        df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce')
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        
        df.to_csv(file_path)
        return df


class VolAnalytics:
    """
    Mathematical Kernels for Signal Processing and Derivatives Pricing.
    Enforces Immutability: Methods return NEW objects, never modify self._data.
    """
    def __init__(self, price_series: pd.Series, risk_free_rate=RISK_FREE_RATE):
        self._clean_data = price_series.copy()
        self.r = risk_free_rate

    def _get_zscore_vol(self, window=20):
        """Helper to get standardized volatility series."""
        # Calculate Log Returns
        log_ret = np.log(self._clean_data / self._clean_data.shift(1)).dropna()
        # Realized Volatility
        vol = log_ret.rolling(window=window).std() * np.sqrt(252)
        vol = vol.dropna()
        # Z-Score Normalization for Signal Processing
        vol_z = (vol - vol.mean()) / vol.std()
        return vol, vol_z

    def compute_welch_features(self, window=60) -> pd.DataFrame:
        """
        Performs Rolling Welch PSD to determine frequency dominance.
        """
        print("[Math] Computing Welch Spectral Density...")
        _, vol_z = self._get_zscore_vol()
        
        results = {'LF_Power': [], 'HF_Power': [], 'Ratio_HF_LF': []}
        dates = []

        # Rolling Window Analysis
        for i in range(window, len(vol_z)):
            segment = vol_z.iloc[i-window:i].values
            
            # Welch Method
            freqs, psd = signal.welch(segment, nperseg=window//2)
            
            # Integrate Power Bands
            # Low Freq: Trend/Persistence. High Freq: Noise/Mean Reversion
            lf_mask = (freqs < 0.1)
            hf_mask = (freqs >= 0.1)
            
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
            
            results['LF_Power'].append(lf_power)
            results['HF_Power'].append(hf_power)
            # Avoid DivZero
            results['Ratio_HF_LF'].append(hf_power / (lf_power + 1e-9))
            dates.append(vol_z.index[i])

        return pd.DataFrame(results, index=dates)

    def detect_pelt_regimes(self) -> pd.DataFrame:
        """
        Uses Ruptures (PELT) to detect structural breaks in volatility mean/variance.
        """
        if not HAS_RUPTURES:
            return pd.DataFrame()

        print("[Math] Detecting Regime Changes (PELT)...")
        vol, _ = self._get_zscore_vol()
        
        # Convert to numpy for ruptures
        signal_arr = vol.values.reshape(-1, 1)
        
        # PELT Algorithm with RBF kernel
        algo = rpt.Pelt(model="rbf").fit(signal_arr)
        # Penalty optimization is subjective, 10 is a decent start for financial time series
        bkps = algo.predict(pen=10) 
        
        # Map breakpoints to dates
        regime_df = pd.DataFrame(index=vol.index)
        regime_df['Regime_ID'] = 0
        
        current_id = 0
        last_bkp = 0
        for bkp in bkps:
            # Prevent index out of bounds
            bkp = min(bkp, len(vol) - 1)
            date_idx = vol.index[last_bkp:bkp]
            regime_df.loc[date_idx, 'Regime_ID'] = current_id
            current_id += 1
            last_bkp = bkp
            
        return regime_df

    def compute_wavelet_edges(self) -> pd.Series:
        """
        Continuous Wavelet Transform (CWT) for edge detection in volatility.
        """
        if not HAS_PYWT:
            return pd.Series()
            
        print("[Math] Computing Wavelet Transform...")
        _, vol_z = self._get_zscore_vol()
        
        # Using Morlet wavelet
        scales = np.arange(1, 31)
        coeffs, freqs = pywt.cwt(vol_z.values, scales, 'morl')
        
        # Sum of coefficients across scales often highlights discontinuities
        cwt_power = np.sum(np.abs(coeffs), axis=0)
        
        return pd.Series(cwt_power, index=vol_z.index, name="Wavelet_Power")

    @staticmethod
    def _black_scholes_call(S, K, T, r, sigma):
        """Helper for Newton-Raphson."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    @staticmethod
    def _vega(S, K, T, r, sigma):
        """Vega calculation for Newton-Raphson."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * stats.norm.pdf(d1) * np.sqrt(T)

    def build_iv_surface(self, options_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """
        Filters options and backs out IV using Newton-Raphson.
        """
        print("[Math] Constructing IV Surface (Newton-Raphson)...")
        
        # 1. Filters
        df = options_df.copy()
        df['DTE'] = (df['expirationDate'] - datetime.now()).dt.days
        df = df[df['DTE'] > 7]  # Exclude strictly short dated
        
        df['Moneyness'] = df['strike'] / spot_price
        df = df[(df['Moneyness'] > 0.8) & (df['Moneyness'] < 1.2)]
        
        # Liquidity Check (Bid/Ask)
        # Note: Yahoo data often has 0 bid. We use lastPrice as proxy if bid/ask missing
        # but ideally we filter wide spreads.
        df = df[df['volume'] > 10] # Basic volume filter
        
        results = []
        
        # Vectorization is hard for root-finding without external libs like scipy.optimize
        # Using a robust iterative loop.
        for idx, row in df.iterrows():
            market_price = row['lastPrice']
            K = row['strike']
            T = row['DTE'] / 365.0
            r = self.r
            
            # Initial Guess
            sigma = 0.5
            max_iter = 100
            tol = 1e-5
            
            for _ in range(max_iter):
                price = self._black_scholes_call(spot_price, K, T, r, sigma)
                diff = market_price - price
                
                if abs(diff) < tol:
                    break
                
                v = self._vega(spot_price, K, T, r, sigma)
                
                if v == 0: # Avoid division by zero
                    break
                    
                sigma = sigma + diff / v
            
            results.append({
                'Strike': K,
                'DTE': row['DTE'],
                'IV': sigma,
                'Moneyness': row['Moneyness']
            })
            
        iv_df = pd.DataFrame(results)
        # Filter insane IVs (solver artifacts)
        iv_df = iv_df[(iv_df['IV'] > 0.01) & (iv_df['IV'] < 3.0)]
        return iv_df


def generate_dashboard(price_data, welch_df, wavelet_series, regime_df, iv_surface_df):
    """
    Constructs the Plotly HTML Dashboard.
    """
    print("[Vis] Generating HTML Dashboard...")
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
               [{"type": "xy"}, {"type": "surface"}]],
        subplot_titles=("Volatility Regimes & Wavelet Intensity", "Spectral Power Ratio (HF/LF)", "Implied Volatility Surface"),
        vertical_spacing=0.1
    )

    # --- Chart 1: Volatility + Regimes + Wavelet ---
    # Plot Volatility
    vol = price_data['Close'] # Just using price for visual context, or calculate vol
    
    # Calculate simple HV for display
    hv = np.log(price_data['Close'] / price_data['Close'].shift(1)).rolling(20).std() * np.sqrt(252)
    
    fig.add_trace(
        go.Scatter(x=hv.index, y=hv.values, name="Historical Vol (20d)", line=dict(color='white', width=1)),
        row=1, col=1
    )

    # Add Regime Backgrounds
    if not regime_df.empty:
        # We need to group by regime ID to make colored shapes
        # This is a simplified approach using scatter markers for regime changes
        changes = regime_df['Regime_ID'].diff().fillna(0)
        change_dates = regime_df.index[changes != 0]
        
        for date in change_dates:
            fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="cyan", row=1, col=1)

    # Add Wavelet Intensity on secondary y-axis logic (or just overlay normalized)
    if not wavelet_series.empty:
        # Align indexes
        w_aligned = wavelet_series.reindex(hv.index).fillna(0)
        # Normalize for visualization
        w_norm = (w_aligned - w_aligned.min()) / (w_aligned.max() - w_aligned.min()) * hv.max()
        
        fig.add_trace(
            go.Scatter(x=w_aligned.index, y=w_norm, name="Wavelet Discontinuity", 
                       line=dict(color='yellow', width=1), fill='tozeroy', opacity=0.3),
            row=1, col=1
        )

    # --- Chart 2: Spectral Monitor ---
    fig.add_trace(
        go.Scatter(x=welch_df.index, y=welch_df['Ratio_HF_LF'], name="HF/LF Power Ratio",
                   line=dict(color='magenta')),
        row=2, col=1
    )
    # Add simple threshold line
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=2, col=1)

    # --- Chart 3: IV Surface ---
    if not iv_surface_df.empty:
        # Pivot for meshgrid
        # We need a regular grid for plot_surface. We can use interpolation or just scatter 3d if sparse.
        # For robustness in a script, scatter3d is safer, but Surface is requested.
        # Let's try to interpolate or use Mesh3d.
        
        fig.add_trace(
            go.Mesh3d(
                x=iv_surface_df['Strike'],
                y=iv_surface_df['DTE'],
                z=iv_surface_df['IV'],
                intensity=iv_surface_df['IV'],
                colorscale='Viridis',
                opacity=0.8,
                name='IV Surface'
            ),
            row=2, col=2
        )

    # Styling
    fig.update_layout(
        template="plotly_dark",
        title_text="Advanced Volatility Signal Dashboard",
        height=900,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.write_html("dashboard.html")
    print("[Vis] Saved to dashboard.html")


def main():
    print("--- Volatility Signal Dashboard Initialization ---")
    
    # 1. Instantiate Ingestion
    ingest = VolDataIngestion()
    
    try:
        # 2. Get Data
        # For Volatility Analysis, we usually look at VIX or the underlying Asset
        ticker_hist = DEFAULT_TICKER 
        df_hist = ingest.get_history(ticker_hist)
        
        # For Surface, we need the option chain ticker (e.g. SPY)
        ticker_opt = DEFAULT_OPTION_TICKER
        df_chain = ingest.get_options_chain(ticker_opt)
        
        # Get Spot price for IV calcs
        # We need current spot, which is last close of history if tickers match, 
        # but here we have ^VIX vs SPY. Need SPY history for spot.
        df_spot_hist = ingest.get_history(ticker_opt)
        spot_price = df_spot_hist['Close'].iloc[-1]
        print(f"[Main] Detected Spot Price for {ticker_opt}: {spot_price:.2f}")

        # 3. Instantiate Analytics
        # We analyze the history of VIX (or the asset) for signals
        analytics = VolAnalytics(df_hist['Close'])
        
        # 4. Run Pipeline
        welch_df = analytics.compute_welch_features()
        wavelet_series = analytics.compute_wavelet_edges()
        regime_df = analytics.detect_pelt_regimes()
        
        # 5. Build Surface
        iv_surface_df = analytics.build_iv_surface(df_chain, spot_price)
        
        # 6. Visualize
        generate_dashboard(df_hist, welch_df, wavelet_series, regime_df, iv_surface_df)
        
        print("--- Execution Complete ---")
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
