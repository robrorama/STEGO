import os
import time
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
RAW_DATA_DIR = "raw_data"
API_SLEEP_SECONDS = 1.0  # Rate limit prevention
DEFAULT_TICKER = "SPY"

class OptionsDataIngestion:
    """
    Handles all I/O operations: API calls, Disk R/W, and Data Sanitization.
    This class is the 'Gateway' ensuring only clean data enters the system.
    """

    def __init__(self):
        self._ensure_storage()

    def _ensure_storage(self):
        """Creates the raw_data directory if it doesn't exist."""
        if not os.path.exists(RAW_DATA_DIR):
            os.makedirs(RAW_DATA_DIR)
            print(f"[SYSTEM] Created storage directory: {RAW_DATA_DIR}")

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The Gateway Method.
        Enforces types, flattens indexes, and removes timezones.
        """
        # 1. MultiIndex Flattening (Handle yfinance quirks)
        if isinstance(df.columns, pd.MultiIndex):
            print("[DATA] Detected MultiIndex columns. Flattening...")
            df.columns = df.columns.get_level_values(0)

        # 2. Strict Datetime Indexing
        # Identify the date column. Usually 'lastTradeDate'.
        target_date_col = None
        for col in ['lastTradeDate', 'date', 'Date']:
            if col in df.columns:
                target_date_col = col
                break
        
        if target_date_col:
            # Coerce to datetime objects first
            df[target_date_col] = pd.to_datetime(df[target_date_col], utc=True)
            df.set_index(target_date_col, inplace=True)
        
        # Ensure index is DatetimeIndex and remove Timezone for Plotly/Excel compatibility
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_convert(None)
        
        # 3. Aggressive Type Casting
        numeric_cols = ['strike', 'lastPrice', 'impliedVolatility', 'openInterest', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ensure expirationDate is datetime for calculations
        if 'expirationDate' in df.columns:
            df['expirationDate'] = pd.to_datetime(df['expirationDate'])

        return df

    def get_chain(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves options chain. 
        Strategy: CHECK DISK -> IF MISSING -> DOWNLOAD & SAVE.
        """
        file_path = os.path.join(RAW_DATA_DIR, f"{ticker}_options.csv")

        # --- PATH A: Check Disk ---
        if os.path.exists(file_path):
            print(f"[IO] Cache hit: Loading {file_path}...")
            try:
                # Load without index first, let sanitizer handle it
                df = pd.read_csv(file_path)
                return self._sanitize_df(df)
            except Exception as e:
                print(f"[ERROR] Corrupt cache file. Re-downloading. Error: {e}")

        # --- PATH B: Download Workflow ---
        print(f"[API] Downloading full chain for {ticker} (This may take time)...")
        yf_ticker = yf.Ticker(ticker)
        
        try:
            expirations = yf_ticker.options
        except Exception as e:
            print(f"[ERROR] Failed to fetch expirations for {ticker}: {e}")
            return pd.DataFrame()

        all_options = []

        print(f"[API] Found {len(expirations)} expiration dates.")
        
        for exp_date in expirations:
            print(f"   > Fetching chain: {exp_date} ...")
            try:
                # Fetch chain
                chain = yf_ticker.option_chain(exp_date)
                
                # Process Calls
                calls = chain.calls
                calls['type'] = 'call'
                calls['expirationDate'] = exp_date
                
                # Process Puts
                puts = chain.puts
                puts['type'] = 'put'
                puts['expirationDate'] = exp_date
                
                all_options.append(pd.concat([calls, puts]))
                
                # RATE LIMITING
                time.sleep(API_SLEEP_SECONDS)
                
            except Exception as e:
                print(f"   [WARN] Failed to fetch {exp_date}: {e}")

        if not all_options:
            print("[ERROR] No data retrieved.")
            return pd.DataFrame()

        # Concatenate
        full_df = pd.concat(all_options)
        
        # Sanitize BEFORE saving to ensure clean CSV
        clean_df = self._sanitize_df(full_df)
        
        # Save to disk
        clean_df.to_csv(file_path)
        print(f"[IO] Saved cached data to {file_path}")

        return clean_df


class OptionsAnalytics:
    """
    Handles Math, Filtering, and Visualization.
    Core Philosophy: Immutability. self._raw_df is READ ONLY.
    """

    def __init__(self, df: pd.DataFrame, ticker: str):
        if df.empty:
            raise ValueError("OptionsAnalytics initialized with empty DataFrame")
        self._raw_df = df
        self.ticker = ticker

    def _get_working_copy(self) -> pd.DataFrame:
        """Create a deep copy for analysis to protect source of truth."""
        return self._raw_df.copy()

    def _calculate_dte(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Days To Expiration (DTE) column."""
        now = datetime.now()
        df['dte'] = (df['expirationDate'] - now).dt.days
        # Filter out expired or bugged negative dates
        df = df[df['dte'] >= 0]
        return df

    def plot_volatility_skew(self, expiration_date=None):
        """
        Plots IV vs Strike for a specific expiration.
        If no date provided, picks the nearest monthly expiration.
        """
        df = self._get_working_copy()
        
        # Filter for Calls only for standard skew analysis (or both)
        # Let's visualize Calls and Puts for comparison
        
        if expiration_date is None:
            # Default to the 3rd available expiration to avoid 0 DTE noise
            unique_exps = sorted(df['expirationDate'].unique())
            if len(unique_exps) > 2:
                expiration_date = unique_exps[2]
            else:
                expiration_date = unique_exps[0]
        
        # Filter Data
        mask = df['expirationDate'] == expiration_date
        subset = df[mask]

        if subset.empty:
            print(f"[VIZ] No data for expiration {expiration_date}")
            return

        calls = subset[subset['type'] == 'call']
        puts = subset[subset['type'] == 'put']

        fig = go.Figure()

        # Calls Line
        fig.add_trace(go.Scatter(
            x=calls['strike'], y=calls['impliedVolatility'],
            mode='lines+markers', name='Calls IV',
            line=dict(color='green')
        ))

        # Puts Line
        fig.add_trace(go.Scatter(
            x=puts['strike'], y=puts['impliedVolatility'],
            mode='lines+markers', name='Puts IV',
            line=dict(color='red')
        ))

        exp_str = pd.to_datetime(expiration_date).strftime('%Y-%m-%d')
        fig.update_layout(
            title=f"Volatility Skew: {self.ticker} (Exp: {exp_str})",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility",
            template="plotly_dark"
        )
        
        print(f"[VIZ] Generating Skew Plot for {exp_str}...")
        fig.show()

    def plot_volatility_surface(self, option_type='call'):
        """
        Plots 3D Surface: X=Strike, Y=DTE, Z=Implied Volatility.
        """
        df = self._get_working_copy()
        
        # Pre-processing
        df = self._calculate_dte(df)
        df = df[df['type'] == option_type]

        # Pivot to create a grid (Strike x DTE)
        # We aggregate using mean in case of duplicate strikes (unlikely but possible in dirty data)
        pivot_df = df.pivot_table(
            index='dte', 
            columns='strike', 
            values='impliedVolatility', 
            aggfunc='mean'
        )

        # Plotly Surface expects 2D arrays
        # We fill NaNs with 0 or interpolate. For visual clarity in surface, 
        # linear interpolation is often best, but raw data is safer.
        # Let's keep NaNs as is; Plotly handles them by creating gaps.
        
        x_strikes = pivot_df.columns
        y_dte = pivot_df.index
        z_iv = pivot_df.values

        fig = go.Figure(data=[go.Surface(
            z=z_iv, 
            x=x_strikes, 
            y=y_dte,
            colorscale='Viridis',
            opacity=0.9
        )])

        fig.update_layout(
            title=f"Volatility Surface ({option_type.upper()}): {self.ticker}",
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiration (DTE)',
                zaxis_title='Implied Volatility'
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=50)
        )

        print(f"[VIZ] Generating Volatility Surface for {option_type}s...")
        fig.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Financial Engineering Options Analyzer")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Stock Ticker (e.g., SPY, AAPL)")
    parser.add_argument("--force-download", action="store_true", help="Ignore local cache and force new download")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    # 1. Ingestion
    ingestor = OptionsDataIngestion()
    
    # Optional: Clear cache if force download requested
    if args.force_download:
        cache_path = os.path.join(RAW_DATA_DIR, f"{ticker}_options.csv")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("[SYSTEM] Cache cleared.")

    df = ingestor.get_chain(ticker)

    if df.empty:
        print("[ERROR] Analysis aborted due to empty data.")
        return

    # 2. Analytics
    try:
        analyzer = OptionsAnalytics(df, ticker)
        
        # 3. Visualizations
        analyzer.plot_volatility_skew()
        analyzer.plot_volatility_surface(option_type='call')
        # analyzer.plot_volatility_surface(option_type='put') # Uncomment if needed

    except ValueError as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
