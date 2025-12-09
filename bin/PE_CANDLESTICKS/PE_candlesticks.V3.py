#!/usr/bin/env python3
# SCRIPTNAME: PE_candlesticks.V3.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Plots Candlesticks with P/E Ratio and Linear Regression Channels.
#   - Uses data_retrieval.py for price data.
#   - Uses local caching + yfinance for fundamental data (Net Income/Shares Outstanding).
#   - Uses Plotly for visualization.
#   - Uses argparse for inputs.

import os
import sys
import json

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import pandas as pd
from scipy.stats import linregress
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Optional yfinance for fundamentals (guarded import)
try:
    import yfinance as yf
except ImportError:
    yf = None

# Centralized Data Retrieval
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

def get_price_data(ticker, period):
    """
    Retrieves price data using data_retrieval.py logic.
    Supports period strings (e.g. 'max') or comma-separated dates.
    """
    if "," in period:
        start_date, end_date = [d.strip() for d in period.split(',')]
        df = dr.load_or_download_ticker(ticker, start=start_date, end=end_date)
        print(f"Loaded data for {ticker} from {start_date} to {end_date}")
    else:
        df = dr.load_or_download_ticker(ticker, period=period)
        print(f"Loaded data for {ticker} (period={period})")
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

def add_ema(df, periods):
    if 'Close' not in df.columns: return df
    for p in periods:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    return df

def _get_cached_fundamentals(ticker):
    """
    Local cache handler for fundamentals since data_retrieval.py 
    only handles OHLCV. Stores in BASE_CACHE_PATH/fundamentals.
    """
    cache_dir = os.path.join(dr.BASE_CACHE_PATH(), "fundamentals")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_fundamentals.json")

    # 1. Try loading from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                # Simple check: if data is recent enough? 
                # For now, just return cached data to satisfy "read from disk" preference.
                print(f"Loaded fundamentals for {ticker} from cache.")
                return data
        except Exception as e:
            print(f"Error reading fundamental cache: {e}")

    # 2. Download if missing
    if yf is None:
        print("Warning: yfinance not installed, cannot download fundamentals.")
        return None

    print(f"Downloading fundamentals for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        # Quarterly financials for Net Income
        inc = stock.quarterly_financials
        if inc.empty or "Net Income" not in inc.index:
            return None
        
        # Sum last 4 quarters (TTM)
        net_income_ttm = float(inc.loc["Net Income"].iloc[:4].sum())
        shares = stock.info.get("sharesOutstanding", 0)

        data = {
            "net_income_ttm": net_income_ttm,
            "shares_outstanding": shares
        }

        # 3. Save to cache
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        return data

    except Exception as e:
        print(f"Error downloading fundamentals: {e}")
        return None

def calculate_pe_ratio(ticker, df):
    """
    Calculates trailing P/E ratio using cached fundamentals.
    """
    fund_data = _get_cached_fundamentals(ticker)
    
    if not fund_data:
        print(f"Warning: Could not retrieve fundamentals for {ticker}; skipping P/E.")
        df["PE_Ratio"] = pd.NA
        return df

    net_income_ttm = fund_data.get("net_income_ttm", 0)
    shares_outstanding = fund_data.get("shares_outstanding", 0)

    if shares_outstanding <= 0:
        print(f"Warning: Shares outstanding <= 0 for {ticker}; skipping P/E.")
        df["PE_Ratio"] = pd.NA
        return df

    eps_ttm = net_income_ttm / shares_outstanding
    if eps_ttm <= 0:
        print(f"Warning: EPS is zero/negative for {ticker}; skipping P/E.")
        df["PE_Ratio"] = pd.NA
        return df

    # Calculate P/E
    df["PE_Ratio"] = df["Close"] / eps_ttm
    return df

def add_pe_linear_regression_bands(df):
    if 'PE_Ratio' not in df.columns or df['PE_Ratio'].isna().all():
        return df
        
    # Drop NaNs for regression calculation, but map back to original index
    valid_pe = df['PE_Ratio'].dropna()
    if valid_pe.empty: return df

    # Use numeric index for regression
    idx_numeric = valid_pe.index.astype('int64') // 10**9
    slope, intercept, _, _, _ = linregress(idx_numeric, valid_pe.values)
    
    # Apply to whole dataframe (trend line)
    full_idx_numeric = df.index.astype('int64') // 10**9
    df['PE_Linear_Reg'] = intercept + slope * full_idx_numeric
    
    # Calculate residuals and std dev on valid data
    residuals = valid_pe - (intercept + slope * idx_numeric)
    std_dev = residuals.std()
    
    # Create bands
    desired_values = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    for i, val in enumerate(desired_values):
        df[f'PE_Reg_High_{val}std'] = df['PE_Linear_Reg'] + val * std_dev
        df[f'PE_Reg_Low_{val}std']  = df['PE_Linear_Reg'] - val * std_dev
        
    return df

def plot_data(df, ticker, outdir, show):
    fig = make_subplots(rows=1, cols=1)

    # 1. Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Candlesticks'
    ))

    # 2. P/E Ratio
    if 'PE_Ratio' in df.columns and not df['PE_Ratio'].isna().all():
        fig.add_trace(go.Scatter(
            x=df.index, y=df['PE_Ratio'], 
            line=dict(color='black', width=2), name='P/E Ratio'
        ))
        
        if 'PE_Linear_Reg' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['PE_Linear_Reg'], 
                line=dict(color='blue', width=2), name='P/E LinReg'
            ))
            
            colors = ['grey', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'blue']
            desired_values = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
            
            for i, val in enumerate(desired_values):
                col_idx = i % len(colors)
                # High Band
                if f'PE_Reg_High_{val}std' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[f'PE_Reg_High_{val}std'],
                        line=dict(color=colors[col_idx], width=1, dash='dot'),
                        name=f'+{val} std'
                    ))
                # Low Band
                if f'PE_Reg_Low_{val}std' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[f'PE_Reg_Low_{val}std'],
                        line=dict(color=colors[col_idx], width=1, dash='dot'),
                        name=f'-{val} std'
                    ))

    fig.update_layout(
        title=f"{ticker} - Candlesticks & P/E Ratio Analysis",
        xaxis_title="Date", yaxis_title="Price / PE",
        height=900,
        hovermode="x unified"
    )

    # Save
    html_file = os.path.join(outdir, f"{ticker}_pe_regression.html")
    fig.write_html(html_file)
    print(f"Saved interactive HTML: {html_file}")
    
    if show:
        fig.show()

def main():
    parser = argparse.ArgumentParser(description="P/E Ratio and Linear Regression Analysis (Plotly)")
    parser.add_argument("ticker", help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--period", default="max", help="Period (e.g. 1y, 5y, max) OR 'YYYY-MM-DD,YYYY-MM-DD'")
    parser.add_argument("--no-show", action="store_true", help="Do not open browser tab")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    
    # 1. Get Price Data (uses dr - cached)
    df = get_price_data(ticker, args.period)
    if df.empty:
        sys.exit(f"Error: No price data found for {ticker}")

    # 2. Add Indicators
    df = add_ema(df, [20, 50, 100])
    
    # 3. Add P/E Data (uses local cache for fundamentals)
    print("Fetching fundamental data for P/E calculation...")
    df = calculate_pe_ratio(ticker, df)
    
    # 4. Add Regression Bands
    df = add_pe_linear_regression_bands(df)
    
    # 5. Plot (outputs to /dev/shm via dr)
    outdir = dr.create_output_directory(ticker)
    plot_data(df.dropna(), ticker, outdir, not args.no_show)

if __name__ == "__main__":
    main()
