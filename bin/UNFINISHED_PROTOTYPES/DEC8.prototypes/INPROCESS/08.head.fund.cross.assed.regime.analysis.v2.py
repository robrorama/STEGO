import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # Mandatory for IterativeImputer
from sklearn.impute import IterativeImputer
from scipy.stats import skew
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'sans-serif'

# ==============================================================================
# 1. DATA INGESTION & ROBUST IMPUTATION ENGINE (FIXED)
# ==============================================================================

class HedgeFundDataLoader:
    def __init__(self, lookback_period="2y"):
        self.lookback = lookback_period
        self.tickers = {
            # FX
            'USDCAD': 'CAD=X',
            'AUDUSD': 'AUDUSD=X',
            'DXY': 'DX-Y.NYB',
            
            # Commodities & Proxies
            'WTI_Front': 'CL=F',
            'Brent': 'BZ=F',
            'IronOre_Proxy_1': 'RIO', # Rio Tinto (Heavy Iron Ore exposure)
            'IronOre_Proxy_2': 'VALE', # Vale SA
            
            # Rates & Bonds
            'US10Y': '^TNX',
            'US02Y': '^IRX',
            'TIPS': 'TIP', # Proxy for Real Yields
            'HYG': 'HYG', # High Yield Corp Bond
            
            # Equity & Vol
            'SPX': '^GSPC',
            'VIX': '^VIX',
            'VVIX': '^VVIX', 
        }

    def fetch_data(self):
        print(f"--- Fetching Real-Time Market Data ({self.lookback}) ---")
        
        # FIX: We explicitly set auto_adjust=False to force yfinance to behave predictably,
        # or we handle the 'Close' fallback dynamically.
        data = yf.download(list(self.tickers.values()), period=self.lookback, progress=False, auto_adjust=False)
        
        # Check if data is empty
        if data.empty:
            raise ValueError("YFinance returned no data. Check your internet connection.")

        # --- ROBUST COLUMN EXTRACTION ---
        # yfinance returns a MultiIndex (Price, Ticker). We need to extract the price safely.
        
        # Try 'Adj Close' first (standard for historical analysis), fallback to 'Close'
        if 'Adj Close' in data.columns.get_level_values(0):
            price = data['Adj Close'].copy()
        elif 'Close' in data.columns.get_level_values(0):
            print(">> Note: 'Adj Close' not found. Using 'Close'.")
            price = data['Close'].copy()
        else:
            # Fallback: Attempt to grab whatever close column is there
            try:
                price = data.xs('Close', axis=1, level=0)
            except KeyError:
                raise KeyError(f"Could not find Price columns. Available: {data.columns.levels[0]}")

        # Extract Volume
        if 'Volume' in data.columns.get_level_values(0):
            volume = data['Volume'].copy()
        else:
            # If volume is missing (rare), create a dummy dataframe of zeros
            volume = pd.DataFrame(0, index=price.index, columns=price.columns)

        # Rename columns to friendly names (USDCAD, etc.)
        # Filter out tickers that might have failed download
        valid_tickers = [t for t in self.tickers.values() if t in price.columns]
        price = price[valid_tickers]
        volume = volume[valid_tickers]
        
        inv_map = {v: k for k, v in self.tickers.items()}
        price.rename(columns=inv_map, inplace=True)
        volume.rename(columns=inv_map, inplace=True)

        return price, volume

    def approximate_missing_data(self, df_price, df_vol):
        """
        Uses Multivariate Imputation to fill gaps.
        """
        print("--- Approximating Missing/Gap Data via Iterative Imputation ---")
        
        # Combine Price and Volume
        combined = df_price.copy()
        for col in df_vol.columns:
            combined[f"{col}_Vol"] = df_vol[col]
            
        # Impute
        imputer = IterativeImputer(max_iter=10, random_state=42)
        try:
            imputed_data = imputer.fit_transform(combined)
        except Exception as e:
            print(f"Imputation Warning: {e}. Falling back to forward fill.")
            return df_price.fillna(method='ffill').fillna(method='bfill')
        
        df_imputed = pd.DataFrame(imputed_data, columns=combined.columns, index=combined.index)
        
        # Extract back just the prices
        clean_prices = df_imputed[df_price.columns]
        
        # Math safety: Ensure no negative prices
        clean_prices = clean_prices.clip(lower=0.01)
        
        return clean_prices

# ==============================================================================
# 2. QUANTITATIVE FEATURE ENGINEERING
# ==============================================================================

class QuantEngine:
    def __init__(self, data):
        self.df = data.copy()
        
    def build_fx_commodity_features(self):
        df = self.df.copy()
        
        # Returns
        df['r_USDCAD'] = np.log(df['USDCAD'] / df['USDCAD'].shift(1))
        
        # --- PROXIES ---
        # WTI Slope Proxy: Momentum of Spot vs 60D MA
        df['WTI_MA60'] = df['WTI_Front'].rolling(60).mean()
        df['WTI_Slope_Proxy'] = (df['WTI_Front'] - df['WTI_MA60']) / df['WTI_MA60']
        
        # Volatility Regimes (State Filter)
        df['CAD_RV'] = df['r_USDCAD'].rolling(window=60).std() * np.sqrt(252)
        
        # Define Regimes (Low/High Vol) based on 66th percentile
        vol_threshold = df['CAD_RV'].quantile(0.66)
        df['CAD_Regime'] = np.where(df['CAD_RV'] > vol_threshold, 'High Vol', 'Low Vol')
        
        return df.dropna()

    def build_eq_bond_features(self):
        df = self.df.copy()
        
        # Log Returns
        df['r_SPX'] = np.log(df['SPX'] / df['SPX'].shift(1))
        # Real Yield Proxy: Inverse of TIPS
        df['r_RealYield'] = -1 * np.log(df['TIPS'] / df['TIPS'].shift(1)) 
        
        # Rolling Correlation (The Target)
        df['Corr_SPX_Yield'] = df['r_SPX'].rolling(60).corr(df['r_RealYield'])
        
        # Predictors
        df['Realized_Skew'] = df['r_SPX'].rolling(22).apply(lambda x: skew(x, nan_policy='omit'))
        
        # Handle missing VVIX if it wasn't in the download or imputation
        if 'VVIX' in df.columns:
            df['Vol_of_Vol'] = df['VVIX']
        else:
            df['Vol_of_Vol'] = df['VIX'].rolling(20).std()
        
        return df.dropna()

# ==============================================================================
# 3. VISUALIZATION LAYER
# ==============================================================================

def plot_fx_comm_linkage(df):
    # Rolling Beta
    window = 60
    rolling_cov = df['r_USDCAD'].rolling(window).cov(df['WTI_Slope_Proxy'])
    rolling_var = df['WTI_Slope_Proxy'].rolling(window).var()
    df['Beta_CAD_Oil'] = rolling_cov / rolling_var
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    # Plot 1: Price & Regime
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("FX-Commodity Regime: USD/CAD vs WTI Slope Proxy", fontsize=14, fontweight='bold', color='#333333')
    ax1.plot(df.index, df['USDCAD'], color='#007acc', label='USD/CAD', linewidth=1.5)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['WTI_Slope_Proxy'], color='#ffa500', label='WTI Slope (Proxy)', alpha=0.6, linestyle='--')
    
    # Shade High Vol
    y_min, y_max = ax1.get_ylim()
    ax1.fill_between(df.index, y_min, y_max, where=(df['CAD_Regime']=='High Vol'), 
                     color='red', alpha=0.1, label='High Vol Regime')
    
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Beta
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_title("Rolling 60D Beta (USD/CAD sensitivity to Oil)", fontsize=12)
    colors = np.where(df['Beta_CAD_Oil'] < 0, 'green', 'red') 
    ax2.bar(df.index, df['Beta_CAD_Oil'], color=colors, width=2.0)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel("Beta")
    
    # Plot 3: Signal
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_title("Trader Signal: Decoupling Alert", fontsize=12)
    signal = (df['Beta_CAD_Oil'] > 0) & (df['CAD_Regime'] == 'High Vol')
    ax3.plot(df.index, signal.astype(int), color='orange', drawstyle='steps-post')
    ax3.fill_between(df.index, 0, signal.astype(int), color='orange', alpha=0.3)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Normal', 'ALERT'])
    
    plt.tight_layout()
    return fig

def plot_eq_bond_regime(df):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    # Plot 1: Correlation
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Equity-Bond Correlation Regime (SPX vs Real Yields)", fontsize=14, fontweight='bold', color='#333333')
    
    curr_corr = df['Corr_SPX_Yield'].iloc[-1]
    color_line = 'green' if curr_corr < 0 else 'red'
    
    ax1.plot(df.index, df['Corr_SPX_Yield'], color='gray', linewidth=1, alpha=0.5)
    ax1.plot(df.index, df['Corr_SPX_Yield'].rolling(10).mean(), color=color_line, linewidth=2, label='Smoothed Corr')
    
    ax1.axhline(0.3, linestyle='--', color='red', alpha=0.5)
    ax1.axhline(-0.3, linestyle='--', color='green', alpha=0.5)
    ax1.set_ylabel("60D Correlation")
    
    # Plot 2: Skew
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Realized Skew (Downside Risk)", fontsize=10)
    ax2.plot(df.index, df['Realized_Skew'], color='purple')
    ax2.axhline(0, color='black', linestyle=':')
    ax2.fill_between(df.index, df['Realized_Skew'], 0, where=(df['Realized_Skew']< -0.5), color='red', alpha=0.3)

    # Plot 3: Vol of Vol
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Vol-of-Vol (Tail Risk Pricing)", fontsize=10)
    ax3.plot(df.index, df['Vol_of_Vol'], color='magenta')
    
    plt.tight_layout()
    return fig

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("HEDGE FUND ANALYTICS: CROSS-ASSET REGIME MAPPER")
    print("="*50 + "\n")
    
    loader = HedgeFundDataLoader()
    
    # 1. Fetch & Clean
    try:
        raw_prices, raw_volumes = loader.fetch_data()
        clean_prices = loader.approximate_missing_data(raw_prices, raw_volumes)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit()

    # 2. Process
    qe = QuantEngine(clean_prices)
    df_fx = qe.build_fx_commodity_features()
    df_eq = qe.build_eq_bond_features()

    print(f"Models Built. FX samples: {len(df_fx)}, EQ samples: {len(df_eq)}")
    print("Rendering plots...")

    # 3. Visualize
    plot_fx_comm_linkage(df_fx)
    plot_eq_bond_regime(df_eq)
    
    print("Done. Displaying...")
    plt.show()
