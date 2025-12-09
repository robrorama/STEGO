# SCRIPTNAME: 07.bayesian.heatmaps.for.options.trading.v4.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.ndimage import convolve1d
import warnings
import os
import time
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'tickers': [
        'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',  # Macro
        'AAPL', 'MSFT', 'NVDA', 'AMZN',     # Mega Cap
        'XOM', 'CVX',                       # Energy
        'JPM', 'BAC',                       # Financials
        'JNJ', 'PFE',                       # Health
        'TSLA', 'AMD', 'COIN', 'MSTR'       # High Beta / Crypto Proxies
    ],
    # Fixed Bin Edges (11 edges = 10 bins)
    'bins_return': [-np.inf, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, np.inf],
    # Fixed Labels (10 labels)
    'bin_labels': [
        '<-4%', '-4%:-3%', '-3%:-2%', '-2%:-1%', '-1%:0%', 
        '0%:1%', '1%:2%', '2%:3%', '3%:4%', '>4%'
    ],
    'data_dir': './market_data_cache',
    'core_lookback_years': 10,
    'recent_lookback_years': 3,
    'lambda_shrinkage': 300,
    'smoothing_kernel': [0.25, 0.50, 0.25], 
    'beta_prior_a': 2,
    'beta_prior_b': 2
}

# Ensure cache directory exists
os.makedirs(CONFIG['data_dir'], exist_ok=True)

# ==========================================
# 2. DISK-FIRST DATA PIPELINE (CSV EDITION)
# ==========================================
def get_todays_filename():
    """Returns the filename for today's data cache."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(CONFIG['data_dir'], f"market_data_{today_str}.csv")

def download_and_cache_data():
    """
    Downloads data ticker-by-ticker with a delay to avoid throttling.
    Saves to CSV. Does NOT return the data.
    """
    filename = get_todays_filename()
    
    # If already exists, skip download
    if os.path.exists(filename):
        print(f"[System] Cache found: {filename}. Skipping download.")
        return

    print(f"[System] No cache found. Initiating download for {len(CONFIG['tickers'])} tickers...")
    print("[System] Enforcing 1.0s delay between requests...")
    
    all_dfs = []
    
    for ticker in CONFIG['tickers']:
        try:
            print(f"   -> Downloading {ticker}...")
            # Download max available history
            df = yf.download(ticker, period="10y", progress=False, auto_adjust=True)
            
            if not df.empty:
                # Basic cleanup
                df['Ticker'] = ticker
                
                # Flatten MultiIndex columns if they exist (YF update behavior)
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        df.columns = df.columns.get_level_values(0)
                    except:
                        pass 
                
                # Approximate Missing Values immediately
                df['Close'] = df['Close'].ffill()
                df['Open'] = df['Open'].ffill()
                
                # Approx Volume: If 0 or NaN, use 30d rolling avg
                df['Volume'] = df['Volume'].replace(0, np.nan)
                df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(30, min_periods=1).mean())
                
                all_dfs.append(df)
            
            # THROTTLE CONTROL
            time.sleep(1.1) 
            
        except Exception as e:
            print(f"   [!] Failed to download {ticker}: {e}")

    if all_dfs:
        print(f"[System] Saving raw data to disk: {filename}")
        full_df = pd.concat(all_dfs)
        # Reset index to ensure Date is a column before saving to CSV
        full_df = full_df.reset_index()
        full_df.to_csv(filename, index=False)
    else:
        raise ValueError("Critical: No data could be downloaded.")

def load_data_from_disk():
    """
    Strictly reads from the disk (CSV).
    """
    filename = get_todays_filename()
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file {filename} is missing.")
    
    print(f"[System] Loading data from disk: {filename}")
    # PARSE DATES IS CRITICAL FOR CSVs
    df = pd.read_csv(filename, parse_dates=['Date'])
    return df

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def process_features(df):
    """
    Calculates Returns, Buckets, and Streaks.
    """
    # 1. Ensure Date is a proper column and data is sorted
    if 'Date' not in df.columns:
        df = df.reset_index()
    
    # Ensure we are working with a clean RangeIndex to prevent alignment errors
    df = df.reset_index(drop=True)
    df = df.sort_values(['Ticker', 'Date'])

    # 2. Basic Features
    df['Return'] = df['Close'].pct_change()
    df['Dollar_Vol'] = df['Close'] * df['Volume']
    
    # Target: Next Day Up?
    df['Target_Up'] = (df.groupby('Ticker')['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop rows where we can't calculate return or target
    df = df.dropna(subset=['Return', 'Dollar_Vol'])

    # 3. Bucketing
    try:
        df['Ret_Bucket'] = pd.cut(
            df['Return'], 
            bins=CONFIG['bins_return'], 
            labels=CONFIG['bin_labels'], 
            include_lowest=True
        )
    except ValueError as e:
        print("CRITICAL ERROR IN BUCKETING.")
        raise e

    # 4. Streak Calculation (Vectorized Fix)
    df['Sign'] = np.sign(df['Return'])
    df['Sign'] = df['Sign'].replace(0, 1) # Treat flat as up
    
    # Vectorized Streak Logic
    prev_sign = df.groupby('Ticker')['Sign'].shift(1)
    condition = (df['Sign'] != prev_sign)
    df['Streak_ID'] = condition.cumsum()
    
    df['Streak_Len'] = df.groupby(['Streak_ID']).cumcount() + 1
    
    # Apply direction and Cap
    df['Streak_Signed'] = df['Streak_Len'] * df['Sign']
    df['Streak_Capped'] = df['Streak_Signed'].clip(lower=-5, upper=5).astype(int)
    
    return df

# ==========================================
# 4. BAYESIAN MATH ENGINE (FIXED)
# ==========================================
def get_posterior(subset_df, weight_col=None):
    """
    Calculates Beta-Binomial posterior with Kernel Smoothing.
    """
    mux = pd.MultiIndex.from_product(
        [CONFIG['bin_labels'], range(-5, 6)], 
        names=['Ret_Bucket', 'Streak_Capped']
    )

    if weight_col:
        # Weighted Aggregation
        subset_df['w_norm'] = subset_df[weight_col] / subset_df[weight_col].mean()
        subset_df['w_up'] = subset_df['Target_Up'] * subset_df['w_norm']
        
        agg = subset_df.groupby(['Ret_Bucket', 'Streak_Capped']).agg(
            n_up=('w_up', 'sum'),
            n_total=('w_norm', 'sum')
        )
    else:
        agg = subset_df.groupby(['Ret_Bucket', 'Streak_Capped']).agg(
            n_up=('Target_Up', 'sum'),
            n_total=('Target_Up', 'count')
        )

    agg = agg.reindex(mux, fill_value=0)
    
    # Smoothing
    n_up_mat = agg['n_up'].unstack(level=0).fillna(0)
    n_tot_mat = agg['n_total'].unstack(level=0).fillna(0)
    
    k = CONFIG['smoothing_kernel']
    
    # --- FIX: result_type='broadcast' ensures we get a DataFrame back ---
    n_up_smooth = n_up_mat.apply(lambda x: convolve1d(x, k, mode='nearest'), axis=1, result_type='broadcast')
    n_tot_smooth = n_tot_mat.apply(lambda x: convolve1d(x, k, mode='nearest'), axis=1, result_type='broadcast')
    
    agg['n_up_smooth'] = n_up_smooth.stack()
    agg['n_tot_smooth'] = n_tot_smooth.stack()
    
    agg['alpha_post'] = CONFIG['beta_prior_a'] + agg['n_up_smooth']
    agg['beta_post'] = CONFIG['beta_prior_b'] + (agg['n_tot_smooth'] - agg['n_up_smooth'])
    
    return agg

def run_hierarchical_model(df):
    """
    Pools Recent Data (Fast) with Core Data (Slow).
    """
    last_date = df['Date'].max()
    date_recent_cutoff = last_date - pd.DateOffset(years=CONFIG['recent_lookback_years'])
    
    df_core = df[df['Date'] <= last_date]
    df_recent = df[df['Date'] > date_recent_cutoff]
    
    post_core = get_posterior(df_core, weight_col='Dollar_Vol')
    post_recent = get_posterior(df_recent, weight_col='Dollar_Vol')
    
    merged = post_recent.join(post_core, lsuffix='_rec', rsuffix='_core')
    
    # Shrinkage
    merged['w'] = merged['n_tot_smooth_rec'] / (merged['n_tot_smooth_rec'] + CONFIG['lambda_shrinkage'])
    
    # Means
    mu_rec = merged['alpha_post_rec'] / (merged['alpha_post_rec'] + merged['beta_post_rec'])
    mu_core = merged['alpha_post_core'] / (merged['alpha_post_core'] + merged['beta_post_core'])
    
    merged['pooled_mean'] = (merged['w'] * mu_rec) + ((1 - merged['w']) * mu_core)
    
    # Intervals
    merged['n_eff'] = merged['n_tot_smooth_rec'] + CONFIG['lambda_shrinkage']
    merged['std_error'] = np.sqrt((merged['pooled_mean'] * (1 - merged['pooled_mean'])) / merged['n_eff'])
    
    merged['lower'] = merged['pooled_mean'] - merged['std_error']
    merged['upper'] = merged['pooled_mean'] + merged['std_error']
    
    return merged

# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_dashboard(stats):
    """
    Creates the Hedge Fund style dashboard.
    """
    mean_pivot = stats['pooled_mean'].unstack(level=0).sort_index(ascending=False)
    lower_pivot = stats['lower'].unstack(level=0).sort_index(ascending=False)
    upper_pivot = stats['upper'].unstack(level=0).sort_index(ascending=False)
    
    uncertainty_pivot = (upper_pivot - lower_pivot)
    
    fig = plt.figure(figsize=(20, 12), facecolor='#f0f0f0')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    
    # --- PLOT 1: Alpha Heatmap ---
    ax1 = fig.add_subplot(gs[:, 0])
    
    annot_data = mean_pivot.applymap(lambda x: f"{x:.0%}")
    
    sns.heatmap(mean_pivot, annot=annot_data, fmt="", 
                cmap="RdYlGn", center=0.5, vmin=0.40, vmax=0.60,
                cbar_kws={'label': 'Probability Next Day UP', 'orientation': 'horizontal'},
                linewidths=1, linecolor='#333333', ax=ax1)
    
    ax1.set_title("Bayesian Probability of 'Green' Tomorrow\n(Weighted by Volume + Hierarchical Shrinkage)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Current Streak (Days)", fontsize=12)
    ax1.set_xlabel("Yesterday's Return Bucket", fontsize=12)

    # --- PLOT 2: Risk / Confidence ---
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(uncertainty_pivot, cmap="Greys", ax=ax2, cbar=True, vmin=0, vmax=0.25)
    ax2.set_title("Model Uncertainty (Interval Width)\nDarker = Less Data = LOWER POSITION SIZE", fontsize=12)
    ax2.set_ylabel("")
    ax2.set_yticks([])

    # --- PLOT 3: Actionable Skew ---
    ax3 = fig.add_subplot(gs[1, 1])
    
    flat_stats = stats.reset_index()
    actionable = flat_stats[
        ((flat_stats['pooled_mean'] > 0.55) | (flat_stats['pooled_mean'] < 0.45)) &
        ((flat_stats['upper'] - flat_stats['lower']) < 0.15)
    ].sort_values('pooled_mean')
    
    if not actionable.empty:
        top_picks = actionable.tail(5)
        bot_picks = actionable.head(5)
        picks = pd.concat([bot_picks, top_picks])
        
        colors = ['red' if x < 0.5 else 'green' for x in picks['pooled_mean']]
        
        y_pos = range(len(picks))
        ax3.barh(y_pos, picks['pooled_mean'] - 0.5, left=0.5, color=colors)
        
        labels = [f"Streak {r.Streak_Capped} / {r.Ret_Bucket}" for r in picks.itertuples()]
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels)
        ax3.set_xlim(0.3, 0.7)
        ax3.axvline(0.5, color='black')
        ax3.set_title("Top Statistical Edges (Low Uncertainty Only)", fontsize=12)
        ax3.grid(True, axis='x', linestyle='--', alpha=0.5)
    else:
        ax3.text(0.5, 0.5, "No High-Confidence Edges Found Today", ha='center')

    plt.tight_layout()
    return fig

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    print("--- Bayesian Options Analytics Initialized ---")
    
    # 1. Manage Data
    download_and_cache_data()
    
    # 2. Read from Disk (CSV)
    df = load_data_from_disk()
    
    print(f"[System] Data loaded from disk. Rows: {len(df)}")
    
    # 3. Process
    print("[Analytics] Calculating Buckets and Streaks...")
    df_processed = process_features(df)
    
    print("[Analytics] Running Hierarchical Bayesian Pooling...")
    stats = run_hierarchical_model(df_processed)
    
    # 4. Visuals
    print("[Analytics] Rendering Dashboard...")
    fig = plot_dashboard(stats)
    
    elapsed = time.time() - start_time
    print(f"--- Done in {elapsed:.2f} seconds ---")
    
    plt.show()
