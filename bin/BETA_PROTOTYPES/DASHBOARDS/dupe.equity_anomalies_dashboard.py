#!/usr/bin/env python3
# SCRIPTNAME: ok.dupe.equity_anomalies_dashboard.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
equity_anomalies_dashboard.py

Description:
    A standalone, production-grade dashboard for analyzing classic equity anomalies 
    (Momentum, Reversal, Turn-of-Month, PEAD) and Options Volatility.
    
    Adheres to strict engineering standards:
    - No external custom loaders.
    - Check-First Caching (raw_data/ directory).
    - Immutable Data Processing (No in-place modifications).
    - Aggressive Sanitization of yfinance data.

Usage:
    python equity_anomalies_dashboard.py SPY AAPL --mode C
"""

import os
import sys
import math
import time
import argparse
import random
import datetime
import warnings
from typing import List, Dict, Tuple, Optional

# --- Dependencies ---
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError as e:
    print("CRITICAL ERROR: Missing required dependencies.")
    print(f"Details: {e}")
    print("Please run: pip install yfinance pandas numpy plotly")
    sys.exit(1)

# Suppress warnings for cleaner CLI output
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)


# ==========================================
# 1. Math Kernel (No External Libs)
# ==========================================
class MathUtils:
    """
    Standalone financial math library.
    Replaces scipy/quantlib dependencies.
    """
    @staticmethod
    def norm_cdf(x: float) -> float:
        """Standard Normal CDF using error function."""
        return 0.5 * (1.0 + math.erf(x / 1.41421356))

    @staticmethod
    def norm_pdf(x: float) -> float:
        """Standard Normal PDF."""
        return (1.0 / 2.50662827463) * math.exp(-0.5 * x * x)

    @staticmethod
    def bs_price_and_greeks(S: float, K: float, T: float, r: float, sigma: float, type_="call") -> Tuple[float, float]:
        """
        Calculates Black-Scholes Price and Delta.
        Returns: (price, delta)
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0, 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        nd1 = MathUtils.norm_cdf(d1)

        if type_ == "call":
            price = S * nd1 - K * math.exp(-r * T) * MathUtils.norm_cdf(d2)
            delta = nd1
        else:
            price = K * math.exp(-r * T) * MathUtils.norm_cdf(-d2) - S * MathUtils.norm_cdf(-d1)
            delta = nd1 - 1.0

        return price, delta

    @staticmethod
    def impl_vol_newton(market_price: float, S: float, K: float, T: float, r: float, type_="call") -> float:
        """
        Newton-Raphson solver for Implied Volatility.
        Bounds: 0.01 to 5.0. Max Iterations: 10.
        """
        if market_price <= 0:
            return 0.0

        sigma = 0.5  # Initial guess
        for _ in range(10):
            price, _ = MathUtils.bs_price_and_greeks(S, K, T, r, sigma, type_)
            diff = market_price - price

            if abs(diff) < 1e-4:
                return sigma

            # Calculate Vega (Raw)
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            vega = S * MathUtils.norm_pdf(d1) * math.sqrt(T)

            if vega < 1e-8:  # Avoid zero division
                return sigma

            sigma = sigma + (diff / vega)

            # Clamp bounds
            sigma = max(0.01, min(5.0, sigma))

        return sigma


# ==========================================
# 2. Anomalies Analyzer (Core Architecture)
# ==========================================
class AnomaliesAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.raw_data_dir = "./raw_data"
        self._ensure_directories()
        
    def _ensure_directories(self):
        if not os.path.exists(self.raw_data_dir):
            os.makedirs(self.raw_data_dir)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressively sanitizes a DataFrame immediately upon loading.
        Enforces Immutability by operating on a copy.
        """
        df = df.copy()

        # 1. Flatten MultiIndex (yfinance often returns ('Close', 'AAPL'))
        if isinstance(df.columns, pd.MultiIndex):
            # Keep level 0 (Price Type) and drop Level 1 (Ticker)
            df.columns = df.columns.get_level_values(0)

        # 2. Enforce DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif df.index.name == 'Date': 
                 # Sometimes index is named Date but is object type
                 df.index = pd.to_datetime(df.index)

        # 3. Strip Timezones
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

        # 4. Coerce Numerics
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Surprise', 
                        'strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. Auto-Calculate Returns if Price exists
        target_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if target_col in df.columns:
            # We use 'price' as the standardized column for analysis
            df['price'] = df[target_col]
            df['ret'] = np.log(df['price'] / df['price'].shift(1))

        # 6. Fill NaNs for Options
        if 'impliedVolatility' in df.columns:
            df['impliedVolatility'] = df['impliedVolatility'].fillna(0.0)

        return df

    # --- Data Acquisition (Persistence Layer) ---

    def get_price_history(self) -> pd.DataFrame:
        """Check-First Caching for Price History."""
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        cache_path = os.path.join(self.raw_data_dir, f"{self.ticker}_price_{today_str}.csv")

        # A. Check Cache
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return self._sanitize_df(df)
            except Exception:
                pass # Fallback to download on read error

        # B. Download & Save
        print(f"[*] Downloading Price History for {self.ticker}...")
        try:
            # yfinance returns MultiIndex if auto_adjust=False, we handle that in sanitize
            df = yf.download(self.ticker, period="2y", progress=False, auto_adjust=False)
            if df.empty:
                return pd.DataFrame()
            
            # Save raw-ish data (sanitized first to ensure index is correct for CSV)
            df_clean = self._sanitize_df(df)
            df_clean.to_csv(cache_path)
            return df_clean
        except Exception as e:
            print(f"[ERROR] Price download failed: {e}")
            return pd.DataFrame()

    def get_earnings_data(self) -> pd.DataFrame:
        """Check-First Caching for Earnings Dates."""
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        cache_path = os.path.join(self.raw_data_dir, f"{self.ticker}_earnings_{today_str}.csv")

        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return self._sanitize_df(df)
            except Exception:
                pass

        print(f"[*] Downloading Earnings Dates for {self.ticker}...")
        try:
            t = yf.Ticker(self.ticker)
            df = t.get_earnings_dates(limit=24)
            if df is None or df.empty:
                return pd.DataFrame()
            
            df_clean = self._sanitize_df(df)
            df_clean.to_csv(cache_path)
            return df_clean
        except Exception as e:
            print(f"[WARN] Earnings download failed (may not be supported for this ticker): {e}")
            return pd.DataFrame()

    def get_atm_options_chain(self, spot_price: float) -> pd.DataFrame:
        """
        Fetches next monthly expiration chain.
        Calculates robust IV/Greeks. Returns filtered ATM data.
        """
        print(f"[*] Fetching Option Chain for {self.ticker}...")
        
        # Rate Limiting
        time.sleep(random.uniform(1.0, 2.0))

        try:
            t = yf.Ticker(self.ticker)
            exps = t.options
            if not exps:
                return pd.DataFrame()

            # Find next valid expiration (at least 7 days out)
            today = datetime.date.today()
            target_exp = None
            for e in exps:
                d = datetime.datetime.strptime(e, "%Y-%m-%d").date()
                if (d - today).days > 7:
                    target_exp = e
                    break
            
            if not target_exp:
                return pd.DataFrame()

            # Cache check for this specific expiry
            cache_path = os.path.join(self.raw_data_dir, f"{self.ticker}_chain_{target_exp}.csv")
            
            if os.path.exists(cache_path):
                 df = pd.read_csv(cache_path)
                 df = self._sanitize_df(df)
            else:
                 opt = t.option_chain(target_exp)
                 calls = opt.calls
                 calls['type'] = 'call'
                 puts = opt.puts
                 puts['type'] = 'put'
                 df = pd.concat([calls, puts])
                 df['expiration'] = target_exp
                 
                 # Save Cache
                 df.to_csv(cache_path, index=False)
                 df = self._sanitize_df(df)

            if df.empty: return pd.DataFrame()

            # --- Robust Processing (Immutable) ---
            df_proc = df.copy()
            
            # Ref Price
            df_proc['mid'] = (df_proc['bid'] + df_proc['ask']) / 2
            df_proc['price_ref'] = np.where(df_proc['mid'] > 0, df_proc['mid'], df_proc['lastPrice'])

            # Time to Expiry
            exp_date = pd.to_datetime(target_exp).date()
            T = (exp_date - today).days / 365.0
            if T < 0.001: T = 0.001

            # Calculate Greeks if missing
            r = 0.045 # Risk free assumption
            
            rows = []
            for _, row in df_proc.iterrows():
                K = row['strike']
                price = row['price_ref']
                otype = row['type']
                iv = row['impliedVolatility']

                # Solve IV if bad data
                if iv <= 0.001 or pd.isna(iv):
                    iv = MathUtils.impl_vol_newton(price, spot_price, K, T, r, otype)
                
                # Calc Delta
                _, delta = MathUtils.bs_price_and_greeks(spot_price, K, T, r, iv, otype)
                
                row['calc_iv'] = iv
                row['calc_delta'] = delta
                rows.append(row)

            df_final = pd.DataFrame(rows)

            # Filter for ATM
            # ATM Call: Strike closest to spot
            # ATM Put: Strike closest to spot
            df_final['dist'] = (df_final['strike'] - spot_price).abs()
            
            atm_call = df_final[df_final['type'] == 'call'].sort_values('dist').iloc[:1]
            atm_put = df_final[df_final['type'] == 'put'].sort_values('dist').iloc[:1]
            
            return pd.concat([atm_call, atm_put])

        except Exception as e:
            print(f"[ERROR] Options processing failed: {e}")
            return pd.DataFrame()

    # --- Analysis Logic (Pure Functions Logic) ---

    def compute_momentum(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Calculates 3m, 6m, 12m momentum."""
        df = df_in.copy()
        px = df['price']
        
        # 3m = 63d, 6m = 126d, 12m = 252d
        df['mom_3m'] = px / px.shift(63) - 1.0
        df['mom_6m'] = px / px.shift(126) - 1.0
        df['mom_12m'] = px / px.shift(252) - 1.0
        
        return df.dropna(subset=['mom_3m'])

    def compute_reversal(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Weekly Reversal Analysis."""
        df = df_in.copy()
        
        # Calculate Past 5d return and Next 5d return
        df['past_5d'] = df['ret'].rolling(5).sum()
        df['next_5d'] = df['ret'].shift(-5).rolling(5).sum()
        
        df = df.dropna()
        
        df['label'] = np.where(df['past_5d'] > 0, "Up", "Down")
        
        summary = df.groupby('label')['next_5d'].mean()
        # Ensure both labels exist for plotting
        if 'Up' not in summary: summary['Up'] = 0.0
        if 'Down' not in summary: summary['Down'] = 0.0
        
        return pd.DataFrame(summary).rename(columns={'next_5d': 'mean_next_return'})

    def compute_tom(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Turn-of-Month Analysis."""
        df = df_in.copy()
        
        # Identify TOM Days (Last 5 trading days of month)
        # Hack: Group by Year-Month, take last 5 indices
        df['ym'] = df.index.to_period('M')
        tom_indices = []
        
        for _, group in df.groupby('ym'):
            if len(group) >= 5:
                tom_indices.extend(group.index[-5:])
        
        df['is_tom'] = df.index.isin(tom_indices)
        
        tom_mean = df[df['is_tom']]['ret'].mean()
        rest_mean = df[~df['is_tom']]['ret'].mean()
        
        return pd.DataFrame({
            'Category': ['TOM Days', 'Rest of Month'],
            'Mean_Daily_Ret': [tom_mean, rest_mean]
        })

    def compute_pead(self, df_price: pd.DataFrame, df_earn: pd.DataFrame) -> pd.DataFrame:
        """PEAD Event Study."""
        if df_price.empty or df_earn.empty:
            return pd.DataFrame()
        
        price = df_price['price']
        events = []
        
        # Iterate earnings dates
        for dt, row in df_earn.iterrows():
            # Normalize timezone naive
            dt_naive = dt.replace(tzinfo=None)
            
            # Find closest trading day (Market might be closed on earnings date)
            try:
                # Use searchsorted logic via indexer
                idx_loc = price.index.get_indexer([dt_naive], method='bfill')[0]
                
                if idx_loc == -1 or (idx_loc + 20) >= len(price):
                    continue
                
                # Prices
                p0 = price.iloc[idx_loc]
                p20 = price.iloc[idx_loc + 20]
                drift = (p20 / p0) - 1.0
                
                surprise = row.get('Surprise', 0)
                if pd.isna(surprise): surprise = 0
                
                etype = "Positive" if surprise > 0 else "Negative"
                
                events.append({
                    'Date': price.index[idx_loc],
                    'Drift_20d': drift,
                    'Surprise': surprise,
                    'Type': etype
                })
            except Exception:
                continue
                
        return pd.DataFrame(events)


# ==========================================
# 3. Visualization Logic (Plotly)
# ==========================================
def generate_dashboard(ticker: str, mom_df: pd.DataFrame, rev_df: pd.DataFrame, 
                       tom_df: pd.DataFrame, pead_df: pd.DataFrame, opt_df: pd.DataFrame):
    
    # Initialize Subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"secondary_y": True}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "table"}]],
        subplot_titles=(
            f"{ticker} Momentum & Price", 
            "Structural Anomalies (Reversal & TOM)",
            "PEAD (20-Day Drift)",
            "ATM Options Snapshot"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # 1. Momentum Chart (Row 1, Col 1)
    # Price on secondary Y, Momentum on Primary
    fig.add_trace(
        go.Scatter(x=mom_df.index, y=mom_df['mom_3m'], name="3m Mom", line=dict(color='#00E5FF')),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=mom_df.index, y=mom_df['mom_12m'], name="12m Mom", line=dict(color='#FF4081')),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=mom_df.index, y=mom_df['price'], name="Price", line=dict(color='white', width=1, dash='dot')),
        row=1, col=1, secondary_y=True
    )

    # 2. Structural Anomalies (Row 1, Col 2)
    # Since we have two different concepts (Reversal vs TOM), we can plot them side-by-side on one chart 
    # or just pick one. The prompt asks for bar charts. Let's combine them cleverly.
    
    # Reversal Bars
    rev_x = ["Rev: After Up Wk", "Rev: After Down Wk"]
    rev_y = [rev_df.loc['Up', 'mean_next_return'], rev_df.loc['Down', 'mean_next_return']]
    
    # TOM Bars
    tom_x = tom_df['Category'].tolist()
    tom_y = tom_df['Mean_Daily_Ret'].tolist()
    
    x_combined = rev_x + tom_x
    y_combined = rev_y + tom_y
    colors = ['#CCFF00' if y > 0 else '#FF4081' for y in y_combined]
    
    fig.add_trace(
        go.Bar(x=x_combined, y=y_combined, marker_color=colors, name="Avg Return"),
        row=1, col=2
    )

    # 3. PEAD Scatter (Row 2, Col 1)
    if not pead_df.empty:
        pos = pead_df[pead_df['Type'] == 'Positive']
        neg = pead_df[pead_df['Type'] == 'Negative']
        
        fig.add_trace(
            go.Scatter(x=pos['Date'], y=pos['Drift_20d'], mode='markers', 
                       marker=dict(color='#CCFF00', symbol='triangle-up', size=10), name="Pos Surprise"),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=neg['Date'], y=neg['Drift_20d'], mode='markers', 
                       marker=dict(color='#FF4081', symbol='triangle-down', size=10), name="Neg Surprise"),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # 4. Options Table (Row 2, Col 2)
    if not opt_df.empty:
        # Format columns for display
        tbl = opt_df[['type', 'expiration', 'strike', 'price_ref', 'calc_iv', 'calc_delta']].copy()
        tbl['calc_iv'] = tbl['calc_iv'].apply(lambda x: f"{x:.1%}")
        tbl['calc_delta'] = tbl['calc_delta'].apply(lambda x: f"{x:.2f}")
        tbl['price_ref'] = tbl['price_ref'].apply(lambda x: f"{x:.2f}")
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(tbl.columns), fill_color='#333', font=dict(color='white')),
                cells=dict(values=[tbl[k].tolist() for k in tbl.columns], fill_color='#222', font=dict(color='white'))
            ),
            row=2, col=2
        )

    # Layout Styling
    fig.update_layout(
        template="plotly_dark",
        title_text=f"Equity Anomalies Dashboard: {ticker}",
        height=900,
        showlegend=True
    )
    
    # Save
    filename = f"{ticker}_dashboard.html"
    print(f"[*] Saving Dashboard to {filename}...")
    fig.write_html(filename)
    print(f"[SUCCESS] Open {filename} in your browser.")


# ==========================================
# 4. Main Execution
# ==========================================
def process_ticker(ticker: str, mode: str):
    print(f"\n--- Processing {ticker} (Mode: {mode}) ---")
    analyzer = AnomaliesAnalyzer(ticker)
    
    # 1. Fetch Data
    df_price = analyzer.get_price_history()
    if df_price.empty:
        print(f"[SKIP] No price data for {ticker}")
        return

    # 2. Compute Price Anomalies
    print("[*] Computing Momentum...")
    mom_df = analyzer.compute_momentum(df_price)
    
    print("[*] Computing Reversal...")
    rev_df = analyzer.compute_reversal(df_price)
    
    print("[*] Computing Turn-of-Month...")
    tom_df = analyzer.compute_tom(df_price)
    
    # 3. Compute PEAD
    print("[*] Computing PEAD...")
    df_earn = analyzer.get_earnings_data()
    pead_df = analyzer.compute_pead(df_price, df_earn)
    
    # 4. Options (If Mode B or C)
    opt_df = pd.DataFrame()
    if mode in ['B', 'C']:
        current_spot = df_price['price'].iloc[-1]
        opt_df = analyzer.get_atm_options_chain(current_spot)

    # 5. Generate Dashboard
    generate_dashboard(ticker, mom_df, rev_df, tom_df, pead_df, opt_df)


def main():
    parser = argparse.ArgumentParser(description="Equity Anomalies Dashboard")
    parser.add_argument("tickers", nargs="+", help="List of tickers (e.g. SPY AAPL)")
    parser.add_argument("--mode", default="C", choices=["A", "B", "C"], help="Analysis Mode")
    args = parser.parse_args()

    # Iterate Sequentially (Strict Isolation)
    for ticker in args.tickers:
        # Strip potential commas if user typed "SPY,AAPL"
        clean_ticker = ticker.replace(',', '').strip()
        if clean_ticker:
            try:
                process_ticker(clean_ticker, args.mode)
            except Exception as e:
                print(f"[ERROR] Failed to process {clean_ticker}: {e}")

if __name__ == "__main__":
    main()
