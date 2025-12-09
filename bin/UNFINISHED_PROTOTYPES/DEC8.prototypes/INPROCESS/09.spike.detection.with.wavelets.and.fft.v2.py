import numpy as np
import pandas as pd
import yfinance as yf
import scipy.signal as signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    'TICKER': '^VIX',          # We use VIX as the proxy for "Panic/Spike" detection
    'LOOKBACK_YEARS': 1,       # Amount of data to fetch
    'INTERVAL': '1d',          # Daily bars
    
    # Wavelet (CWT) Params
    'CWT_SCALES_HF': (2, 6),   # High Frequency Scales (Fast moves)
    'CWT_SCALES_LF': (10, 30), # Low Frequency Scales (Background noise)
    'MEDIAN_FILTER_WIN': 3,    # Denoising window
    'Z_SCORE_WIN': 200,        # Rolling window for Z-score normalization
    
    # FFT (STFT) Params
    'STFT_WIN': 64,
    'STFT_HOP': 8,
    
    # Triggers
    'THRESH_Z': 2.0,           # Z-score threshold
    'THRESH_RATIO': 1.4,       # HF/LF Energy Ratio
    'THRESH_BETA': -0.25,      # Spectral Slope drop threshold
}

# ==========================================
# CLASS: DATA INGESTION & APPROXIMATION
# ==========================================
class DataEngine:
    def __init__(self, ticker):
        self.ticker = ticker
        
    def fetch_data(self):
        """
        Fetches REAL data. No fake generation.
        """
        print(f"--- 📡 CONNECTING TO EXCHANGE: FETCHING {self.ticker} ---")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365 * CONFIG['LOOKBACK_YEARS'])
        
        # FIXED: Added auto_adjust=True to silence warning and ensure clean price data
        df = yf.download(self.ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if df.empty:
            raise ValueError("No data returned. Check ticker or internet connection.")
            
        # Standardize column names (Handle MultiIndex if present)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
            
        # Rename for consistency
        rename_map = {'Close': 'Close', 'Volume': 'Vol'}
        # Handle case where Yahoo returns 'Adj Close' or just 'Close' due to auto_adjust
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            rename_map['Adj Close'] = 'Close'
            
        df = df.rename(columns=rename_map)
        
        # Ensure we have the columns we need
        if 'Vol' not in df.columns: 
            df['Vol'] = 0 # Create dummy if missing
            
        return df[['Close', 'Vol']]

    def approximate_missing_data(self, df):
        """
        CRITICAL: Fills missing data gaps using Volume-Weighted Interpolation.
        If volume is missing, we infer it from local volatility.
        """
        print("--- 🔧 APPROXIMATING MISSING DATA STRUCTURES ---")
        
        # 1. Fill Volume NaNs with rolling median (conservative approximation)
        df['Vol'] = df['Vol'].fillna(df['Vol'].rolling(20, min_periods=1).median())
        
        # 2. Identify Price Gaps
        if df['Close'].isnull().sum() > 0:
            # Use Cubic Spline interpolation for price to maintain organic curve
            df['Close'] = df['Close'].interpolate(method='cubic')
            
        # 3. Create 'Proxy Volume' if real volume is 0 (common in indices like VIX)
        # We approximate volume activity using Absolute Price Velocity
        # If sum is low, assume data feed is price only
        if df['Vol'].sum() < 100: 
            # Proxy liquidity = Price Velocity
            df['Vol'] = np.abs(df['Close'].diff()) * 1e6 
            
        # 4. Detrending and Z-Scoring (Pre-processing for Wavelets)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        
        # Rolling Z-Score of returns (Standardization)
        rolling_mean = df['Log_Ret'].rolling(window=CONFIG['Z_SCORE_WIN']).mean()
        rolling_std = df['Log_Ret'].rolling(window=CONFIG['Z_SCORE_WIN']).std()
        df['Signal_Norm'] = (df['Log_Ret'] - rolling_mean) / rolling_std
        
        return df.fillna(0)

# ==========================================
# CLASS: SIGNAL PROCESSING CORE (FIXED)
# ==========================================
class QuantitativeCore:
    
    @staticmethod
    def complex_morlet_cwt(signal_data, scales):
        """
        FIXED: Replaced signal.cwt with explicit fftconvolve loop.
        This is faster (O(NlogN)) and avoids AttributeError on older Scipy versions.
        """
        n = len(signal_data)
        # Pre-allocate output matrix (Scales x Time)
        cwt_matrix = np.zeros((len(scales), n), dtype=complex)
        
        for i, s in enumerate(scales):
            # 1. Generate Wavelet
            # Length M needs to be large enough to capture the wavelet tails.
            # 10 * s is a standard heuristic for Morlet.
            M = int(s * 10.0)
            if M > n: M = n # Cap at signal length
            
            # w=5 is the omega0 (central frequency)
            wavelet = signal.morlet2(M, s, w=5)
            
            # 2. Convolve (FFT based)
            # mode='same' ensures output length matches input length
            cwt_matrix[i, :] = signal.fftconvolve(signal_data, wavelet, mode='same')
        
        # Power = Magnitude Squared
        power = np.abs(cwt_matrix) ** 2
        return power

    @staticmethod
    def compute_wavelet_features(df):
        print("--- 🌊 CALCULATING WAVELET ENERGY (CWT) ---")
        sig = df['Signal_Norm'].values
        
        # Define Scale Maps
        hf_scales = np.arange(CONFIG['CWT_SCALES_HF'][0], CONFIG['CWT_SCALES_HF'][1])
        lf_scales = np.arange(CONFIG['CWT_SCALES_LF'][0], CONFIG['CWT_SCALES_LF'][1])
        all_scales = np.concatenate([hf_scales, lf_scales])
        
        # 1. Compute Power (Using the new robust function)
        power_matrix = QuantitativeCore.complex_morlet_cwt(sig, all_scales)
        
        # 2. Denoise Power (Median Filter)
        power_matrix = ndimage.median_filter(power_matrix, size=(1, CONFIG['MEDIAN_FILTER_WIN']))
        
        # 3. Split Bands
        idx_hf = range(len(hf_scales))
        idx_lf = range(len(hf_scales), len(all_scales))
        
        # 4. Calculate Energies
        P_hf = np.sum(power_matrix[idx_hf, :], axis=0)
        P_lf = np.sum(power_matrix[idx_lf, :], axis=0)
        
        # Avoid div by zero
        P_lf[P_lf == 0] = 1e-6
        
        # 5. Feature: HF/LF Ratio
        df['CWT_Ratio'] = P_hf / P_lf
        
        # 6. Feature: Z-Score of HF Derivative (Acceleration)
        hf_deriv = np.gradient(P_hf)
        
        # Rolling Z of derivative
        hf_deriv_series = pd.Series(hf_deriv)
        roll_mean = hf_deriv_series.rolling(CONFIG['Z_SCORE_WIN']).mean()
        roll_std = hf_deriv_series.rolling(CONFIG['Z_SCORE_WIN']).std()
        df['CWT_Accel_Z'] = (hf_deriv_series - roll_mean) / roll_std
        
        return df, power_matrix

    @staticmethod
    def compute_stft_features(df):
        print("--- 📉 FITTING SPECTRAL SLOPES (FFT) ---")
        sig = df['Signal_Norm'].values
        nperseg = CONFIG['STFT_WIN']
        
        # Compute STFT
        f, t, Zxx = signal.stft(sig, nperseg=nperseg, noverlap=nperseg-CONFIG['STFT_HOP'])
        
        # Power Spectrum
        Sxx = np.abs(Zxx)**2
        
        # Calculate Spectral Slope (Beta) per time window
        betas = []
        
        # Avoid log(0)
        f_log = np.log(f[1:] + 1e-9) 
        
        for i in range(Sxx.shape[1]):
            spec_slice = Sxx[1:, i] # Skip DC component
            spec_log = np.log(spec_slice + 1e-9)
            
            # Linear Fit: y = mx + c -> spec_log = beta * f_log + c
            try:
                slope, _ = np.polyfit(f_log, spec_log, 1)
            except:
                slope = 0
            betas.append(slope)
            
        # Expand betas back to original dataframe size
        beta_series = pd.Series(betas)
        t_indices = np.linspace(0, len(df)-1, len(betas)).astype(int)
        
        # Fill into DF
        df['Spectral_Beta'] = np.nan
        df.iloc[t_indices, df.columns.get_loc('Spectral_Beta')] = beta_series.values
        df['Spectral_Beta'] = df['Spectral_Beta'].interpolate(method='linear')
        
        # Delta Beta
        df['Delta_Beta'] = df['Spectral_Beta'] - df['Spectral_Beta'].rolling(200).median()
        
        return df

# ==========================================
# VISUALIZATION ENGINE
# ==========================================
def render_dashboard(df, power_matrix):
    print("--- 🎨 RENDERING TRADER DASHBOARD ---")
    
    # Setup styling (Hedge Fund Dark Mode)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1.5, 1.5, 1], hspace=0.05)
    
    # Color Maps
    cmap_energy = LinearSegmentedColormap.from_list("custom_magma", ["#000000", "#3b0f70", "#8c2981", "#fe9f6d", "#fcfdbf"])
    
    # ------------------------------------------------
    # PANEL 1: PRICE & SIGNALS
    # ------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    
    # Price Line
    ax1.plot(df.index, df['Close'], color='#d1d4dc', lw=1, alpha=0.8, label='Price')
    
    # Identify Triggers
    mask_trigger = (
        (df['CWT_Accel_Z'] > CONFIG['THRESH_Z']) & 
        (df['CWT_Ratio'] > CONFIG['THRESH_RATIO']) &
        (df['Delta_Beta'] < CONFIG['THRESH_BETA'])
    )
    
    triggers = df[mask_trigger]
    
    # Plot Triggers
    ax1.scatter(triggers.index, triggers['Close'], color='#00ff41', s=100, marker='^', 
                edgecolor='black', zorder=5, label='QUANT SPIKE ALERT')
    
    ax1.set_ylabel(f"{CONFIG['TICKER']} Price", fontsize=12, color='white')
    ax1.legend(loc='upper left', frameon=False)
    ax1.set_title(f"INSTITUTIONAL SPIKE DETECTOR: {CONFIG['TICKER']}", fontsize=16, fontweight='bold', color='#00ff41')
    ax1.grid(True, color='#333333', linestyle='--', alpha=0.3)

    # ------------------------------------------------
    # PANEL 2: WAVELET ENERGY SCALOGRAM
    # ------------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Clip power for better contrast
    p_robust = np.clip(power_matrix, 0, np.percentile(power_matrix, 99))
    
    ax2.imshow(p_robust, aspect='auto', cmap=cmap_energy, origin='lower', 
               extent=[0, len(df), 2, 30])
    
    ax2.text(0.01, 0.9, "CWT: LATENT ENERGY BUILDUP", transform=ax2.transAxes, 
             color='#fe9f6d', fontweight='bold')
    ax2.set_ylabel("Wavelet Scale", color='gray')
    ax2.set_yticks([])

    # ------------------------------------------------
    # PANEL 3: SPECTRAL BETA
    # ------------------------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    beta_col = '#00d4ff'
    ax3.plot(df.index, df['Spectral_Beta'], color=beta_col, lw=1.5)
    
    # Draw Threshold Line
    thresh_line = df['Spectral_Beta'].rolling(200).median() + CONFIG['THRESH_BETA']
    ax3.plot(df.index, thresh_line, color='red', linestyle='--', alpha=0.5, lw=1)
    
    ax3.fill_between(df.index, df['Spectral_Beta'], thresh_line, 
                     where=(df['Spectral_Beta'] < thresh_line),
                     color='red', alpha=0.2)

    ax3.text(0.01, 0.9, "STFT: SPECTRAL FLATTENING (Beta)", transform=ax3.transAxes, 
             color=beta_col, fontweight='bold')
    ax3.set_ylabel("Beta Slope", color='gray')

    # ------------------------------------------------
    # PANEL 4: SIGNAL CONFIDENCE
    # ------------------------------------------------
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    ax4.plot(df.index, df['CWT_Ratio'], color='#fcfdbf', lw=1)
    ax4.axhline(CONFIG['THRESH_RATIO'], color='white', linestyle=':', alpha=0.5)
    
    ax4.fill_between(df.index, df['CWT_Ratio'], CONFIG['THRESH_RATIO'], 
                     where=(df['CWT_Ratio'] > CONFIG['THRESH_RATIO']),
                     color='#fcfdbf', alpha=0.3)
    
    ax4.text(0.01, 0.8, "HF/LF ENERGY RATIO", transform=ax4.transAxes, 
             color='#fcfdbf', fontweight='bold')
    ax4.set_xlabel("Date", fontsize=12)

    # ------------------------------------------------
    # FINAL FORMATTING
    # ------------------------------------------------
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    # Add actionable stats box
    last_row = df.iloc[-1]
    status = "WAIT"
    col = "gray"
    
    if (last_row['CWT_Accel_Z'] > CONFIG['THRESH_Z'] and 
        last_row['CWT_Ratio'] > CONFIG['THRESH_RATIO']):
        status = "ALERT: HIGH FREQ ENERGY"
        col = "#00ff41"
        
    stats = (f"LATEST READINGS:\n"
             f"Vol Regime: {last_row['Vol']:.0f}\n"
             f"Energy Ratio: {last_row['CWT_Ratio']:.2f}\n"
             f"Spectral Beta: {last_row['Spectral_Beta']:.2f}\n"
             f"STATUS: {status}")
    
    props = dict(boxstyle='round', facecolor='#222222', alpha=0.8, edgecolor=col)
    ax1.text(0.02, 0.05, stats, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='bottom', bbox=props, color='white', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
def run_quant_analysis():
    try:
        # 1. Acquire
        engine = DataEngine(CONFIG['TICKER'])
        raw_df = engine.fetch_data()
        
        # 2. Approximate & Clean
        clean_df = engine.approximate_missing_data(raw_df)
        
        # 3. Compute
        df_wav, power_matrix = QuantitativeCore.compute_wavelet_features(clean_df)
        df_final = QuantitativeCore.compute_stft_features(df_wav)
        
        # 4. Filter for plotting
        plot_df = df_final.iloc[CONFIG['Z_SCORE_WIN']:]
        plot_power = power_matrix[:, CONFIG['Z_SCORE_WIN']:]
        
        # 5. Visualize
        render_dashboard(plot_df, plot_power)
        
        # 6. Actionable Output
        latest = df_final.iloc[-1]
        print("\n==========================================")
        print(f"TRADING REPORT: {CONFIG['TICKER']}")
        print("==========================================")
        print(f"Date: {latest.name.date()}")
        print(f"Close Price: {latest['Close']:.2f}")
        print(f"HF/LF Energy Ratio: {latest['CWT_Ratio']:.4f} (Threshold: {CONFIG['THRESH_RATIO']})")
        print(f"Spectral Beta:      {latest['Spectral_Beta']:.4f}")
        print("------------------------------------------")
        if (latest['CWT_Accel_Z'] > CONFIG['THRESH_Z']) and (latest['CWT_Ratio'] > CONFIG['THRESH_RATIO']):
            print(">>> ACTION: POTENTIAL VOLATILITY SPIKE IMMINENT <<<")
            print("    Recommendation: Long Gamma / Debit Spreads")
        else:
            print(">>> ACTION: NO SIGNAL / MEAN REVERSION <<<")
            print("    Recommendation: Short Vol / Theta Collection")
        print("==========================================\n")
        
    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_quant_analysis()
