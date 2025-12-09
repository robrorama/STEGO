# SCRIPTNAME: ok.fx_commodity_partial_corr_dashboard.v7.DUPE.maybe.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
fx_commodity_partial_corr_dashboard.py

Role: Senior Quantitative Developer
Objective: Partial Correlation Analysis (Commodity vs FX | USD Control)
Framework: Standalone, Modular, No External Loaders (yfinance/plotly only)

Usage:
    python fx_commodity_partial_corr_dashboard.py --commodity CL=F --fx CAD=X --usd-index DX-Y.NYB --period 2y
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. DATA INGESTION LAYER (I/O & SANITIZATION)
# ==========================================

class DataIngestion:
    """
    Responsible solely for I/O, downloading, sanitization, and caching.
    Strictly decoupled from analysis logic.
    """
    def __init__(self, storage_dir='raw_data'):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def fetch_data(self, ticker: str, period: str = '2y') -> pd.DataFrame:
        """
        Retrieves data with a Persistence Layer (Local Caching).
        """
        # clean ticker for filename safety
        safe_ticker = ticker.replace("=", "_").replace("^", "")
        file_path = os.path.join(self.storage_dir, f"{safe_ticker}_{period}.csv")

        if os.path.exists(file_path):
            print(f"[DataIngestion] Cache Hit: Loading {ticker} from {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            print(f"[DataIngestion] Cache Miss: Downloading {ticker} from yfinance...")
            try:
                # Rate Limiting
                time.sleep(1) 
                df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
                
                if df.empty:
                    raise ValueError(f"No data found for ticker: {ticker}")
                
                # Sanitize BEFORE saving to ensure cache is clean
                df = self._sanitize_df(df)
                
                # Save to cache
                df.to_csv(file_path)
            except Exception as e:
                print(f"[DataIngestion] Critical Error downloading {ticker}: {e}")
                return pd.DataFrame()

        # Sanitize again on load to ensure type safety (e.g. recovering DatetimeIndex from CSV)
        return self._sanitize_df(df)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressive Data Sanitization strictly handling yfinance edge cases.
        """
        # 1. MultiIndex Flattening
        # yfinance v0.2+ often returns columns as MultiIndex (Price, Ticker).
        if isinstance(df.columns, pd.MultiIndex):
            # If we downloaded a single ticker, drop the Ticker level
            try:
                df.columns = df.columns.droplevel(1)
            except IndexError:
                pass # Already flat or unexpected structure

        # 2. Strict Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Attempt to find a date column if index is RangeIndex
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)

        # 3. Timezone Removal (Plotly Compatibility)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 4. Numeric Coercion
        # Force core columns to numeric, coercing errors to NaN
        core_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in core_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where all core data is NaN
        df.dropna(how='all', inplace=True)
        
        return df

# ==========================================
# 2. FINANCIAL ANALYSIS LAYER (MATH & LOGIC)
# ==========================================

class FinancialAnalysis:
    """
    Responsible solely for math, logic, and dataframe manipulations.
    Implements Immutability & State Management.
    """
    def __init__(self, commodity_df, fx_df, usd_df):
        # Source of Truth: Private dictionary
        self._raw_data = {
            'commodity': commodity_df,
            'fx': fx_df,
            'usd': usd_df
        }

    def _get_prices(self):
        """Helper to extract Close prices safely."""
        return pd.DataFrame({
            'commodity': self._raw_data['commodity']['Adj Close'],
            'fx': self._raw_data['fx']['Adj Close'],
            'usd': self._raw_data['usd']['Adj Close']
        })

    def get_display_data(self):
        """
        Alignment: Display View (Outer Join + Forward Fill).
        Used for Price Charts.
        """
        df = self._get_prices().copy() # Copy-on-Write
        # Resample to daily to unify timelines if mixed frequencies
        df = df.resample('D').last()
        df = df.ffill().dropna()
        
        # Normalize to 100
        return df / df.iloc[0] * 100

    def calculate_rolling_correlations(self, window=63):
        """
        Alignment: Math View (Inner Join).
        Calculates Rolling Correlations and Partial Correlation.
        """
        # Copy-on-Write
        prices = self._get_prices().copy()
        
        # Math View: Drop NaNs (Intersection only)
        prices = prices.dropna()

        # Log Returns
        returns = np.log(prices / prices.shift(1)).dropna()

        # Rolling Pairwise Correlations
        r_xy = returns['commodity'].rolling(window).corr(returns['fx'])   # Comm vs FX
        r_xz = returns['commodity'].rolling(window).corr(returns['usd'])  # Comm vs USD
        r_yz = returns['fx'].rolling(window).corr(returns['usd'])         # FX vs USD

        # Rolling Partial Correlation Formula
        # r_xy.z = (r_xy - r_xz * r_yz) / sqrt( (1 - r_xz^2) * (1 - r_yz^2) )
        
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        # Avoid division by zero
        partial_corr = numerator / denominator.replace(0, np.nan)

        return pd.DataFrame({
            'Raw Correlation (Comm vs FX)': r_xy,
            'Partial Correlation (Net USD)': partial_corr
        })

    def calculate_residuals(self):
        """
        Residual Analysis: Full-sample OLS.
        y = alpha + beta*x + epsilon
        """
        prices = self._get_prices().copy()
        prices = prices.dropna()
        returns = np.log(prices / prices.shift(1)).dropna()

        if returns.empty:
            return pd.DataFrame()

        # OLS Helper
        def get_residuals(y, x):
            # numpy.polyfit(x, y, 1) returns [slope, intercept]
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
            return y - predicted

        resid_comm = get_residuals(returns['commodity'], returns['usd'])
        resid_fx = get_residuals(returns['fx'], returns['usd'])

        return pd.DataFrame({
            'Commodity Residuals': resid_comm,
            'FX Residuals': resid_fx
        })

# ==========================================
# 3. VISUALIZATION LAYER
# ==========================================

def generate_dashboard(fa: FinancialAnalysis, tickers: dict):
    """
    Generates a self-contained HTML dashboard using Plotly.
    """
    # 1. Get Data
    df_norm = fa.get_display_data()
    df_corr = fa.calculate_rolling_correlations(window=63)
    df_resid = fa.calculate_residuals()

    # 2. Create Subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
        subplot_titles=(
            "Normalized Price Performance (Base=100)",
            "Rolling Correlation Analysis (63-Day Window)",
            "Residual Analysis (Idiosyncratic Returns vs USD)"
        ),
        vertical_spacing=0.15
    )

    # Plot 1: Normalized Prices (Top Row, Full Width)
    colors = {'commodity': '#1f77b4', 'fx': '#ff7f0e', 'usd': '#2ca02c'}
    names = {'commodity': tickers['commodity'], 'fx': tickers['fx'], 'usd': tickers['usd']}
    
    for key in ['commodity', 'fx', 'usd']:
        fig.add_trace(
            go.Scatter(x=df_norm.index, y=df_norm[key], name=names[key],
                       line=dict(color=colors[key], width=1.5)),
            row=1, col=1
        )

    # Plot 2: Correlation Analysis (Bottom Left)
    fig.add_trace(
        go.Scatter(x=df_corr.index, y=df_corr['Raw Correlation (Comm vs FX)'],
                   name="Raw Corr (Comm/FX)",
                   line=dict(color='gray', width=1, dash='dot')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_corr.index, y=df_corr['Partial Correlation (Net USD)'],
                   name="Partial Corr (Net USD)",
                   line=dict(color='firebrick', width=2)),
        row=2, col=1
    )

    # Plot 3: Residual Scatter (Bottom Right)
    fig.add_trace(
        go.Scatter(
            x=df_resid['Commodity Residuals'],
            y=df_resid['FX Residuals'],
            mode='markers',
            name='Residuals',
            marker=dict(
                size=5,
                color=df_resid['FX Residuals'], # Color by Y value
                colorscale='Viridis',
                opacity=0.6
            ),
            text=df_resid.index.strftime('%Y-%m-%d'),
            hovertemplate='<b>Date</b>: %{text}<br><b>Comm Resid</b>: %{x:.4f}<br><b>FX Resid</b>: %{y:.4f}'
        ),
        row=2, col=2
    )

    # 3. Layout Updates
    fig.update_layout(
        title_text=f"Quantitative Partial Correlation Dashboard: {names['commodity']} vs {names['fx']} (Control: {names['usd']})",
        height=900,
        template="plotly_white",
        hovermode="x unified"
    )

    # Axes Titles
    fig.update_yaxes(title_text="Rebased Price", row=1, col=1)
    fig.update_yaxes(title_text="Correlation Coeff", range=[-1, 1], row=2, col=1)
    fig.update_xaxes(title_text="Commodity Residuals (Net USD)", row=2, col=2)
    fig.update_yaxes(title_text="FX Residuals (Net USD)", row=2, col=2)

    # Add Zero line to Correlation Plot
    fig.add_shape(type="line", x0=df_corr.index[0], y0=0, x1=df_corr.index[-1], y1=0,
                  line=dict(color="black", width=1), row=2, col=1)

    # Save
    output_file = "dashboard.html"
    fig.write_html(output_file)
    print(f"\n[Success] Dashboard generated: {os.path.abspath(output_file)}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Partial Correlation Quant Dashboard")
    parser.add_argument('--commodity', type=str, required=True, help="Commodity Ticker (e.g. CL=F)")
    parser.add_argument('--fx', type=str, required=True, help="Forex Ticker (e.g. CAD=X)")
    parser.add_argument('--usd-index', type=str, default='DX-Y.NYB', help="USD Index Ticker")
    parser.add_argument('--period', type=str, default='2y', help="Lookback period")
    
    args = parser.parse_args()

    print("--- Starting Quant Analysis ---")
    
    # 1. Initialize Ingestion
    ingest = DataIngestion()
    
    # 2. Fetch Data (with caching)
    print(f"Fetching data for {args.commodity}...")
    comm_df = ingest.fetch_data(args.commodity, args.period)
    
    print(f"Fetching data for {args.fx}...")
    fx_df = ingest.fetch_data(args.fx, args.period)
    
    print(f"Fetching data for {args.usd_index}...")
    usd_df = ingest.fetch_data(args.usd_index, args.period)

    # Validation
    if comm_df.empty or fx_df.empty or usd_df.empty:
        print("[Error] One or more datasets could not be retrieved. Exiting.")
        return

    # 3. Initialize Analysis
    analysis = FinancialAnalysis(comm_df, fx_df, usd_df)
    
    # 4. Generate Dashboard
    tickers = {
        'commodity': args.commodity,
        'fx': args.fx,
        'usd': args.usd_index
    }
    generate_dashboard(analysis, tickers)

if __name__ == "__main__":
    main()
