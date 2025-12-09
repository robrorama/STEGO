"""
equity_anomalies_dashboard.py

Role: Lead Quantitative Developer
Specialization: Factor Investing and Market Anomalies
Description: A modular, immutable, and caching-enabled dashboard for analyzing 
             classic equity anomalies (Momentum, Reversal, TOM, PEAD) and 
             integrating robust Options data.

Usage:
    python equity_anomalies_dashboard.py --tickers AAPL MSFT SPY
"""

import os
import argparse
import logging
import math
import datetime
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress warnings usually associated with pandas chained assignments or yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Math Kernel (No Scipy/QuantLib) ---

def _norm_cdf(x):
    """Standard Normal CDF using Error Function approximation."""
    return 0.5 * (1.0 + math.erf(x / 1.41421356))

def _norm_pdf(x):
    """Standard Normal PDF."""
    return (1.0 / 2.50662827463) * math.exp(-0.5 * x * x)

def bs_price_and_greeks(S, K, T, r, sigma, type_="call"):
    """
    Calculate Black-Scholes Price and Delta.
    
    Args:
        S (float): Spot Price
        K (float): Strike Price
        T (float): Time to maturity (in years)
        r (float): Risk-free rate
        sigma (float): Volatility (annualized)
        type_ (str): "call" or "put"
        
    Returns:
        tuple: (price, delta)
    """
    # Guard against expiration or zero vol
    if T <= 0:
        intrinsic = max(0, S - K) if type_ == "call" else max(0, K - S)
        return intrinsic, 0.0
    if sigma <= 0:
        sigma = 0.0001

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if type_ == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0

    return price, delta

def impl_vol_newton(market_price, S, K, T, r, type_="call"):
    """
    Newton-Raphson solver for Implied Volatility.
    Bounds: 0.01 to 5.0. Max Iterations: 10.
    """
    sigma = 0.5  # Initial guess
    tol = 1e-4
    max_iter = 10

    for i in range(max_iter):
        price, _ = bs_price_and_greeks(S, K, T, r, sigma, type_)
        
        diff = market_price - price
        if abs(diff) < tol:
            return sigma
            
        # Vega calculation
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * _norm_pdf(d1) * math.sqrt(T)

        if abs(vega) < 1e-8:
            break # Avoid division by zero
            
        sigma = sigma + diff / vega
        
        # Enforce bounds
        if sigma < 0.01: sigma = 0.01
        if sigma > 5.0: sigma = 5.0
        
    return sigma


# --- Core Analyzer Class ---

class AnomaliesAnalyzer:
    """
    Central engine for analyzing equity anomalies.
    Enforces Strict Single-Ticker Processing and Data Immutability.
    """
    
    DATA_DIR = "./raw_data"

    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.price_history = None  # Source of Truth for OHLCV
        self.earnings_data = None  # Source of Truth for Earnings
        
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)

    def load_data(self):
        """Orchestrates data loading: Check Cache -> Download -> Sanitize -> Save."""
        logger.info(f"[{self.ticker}] Loading Price History...")
        self.price_history = self._fetch_or_load("prices", self._download_prices)
        
        logger.info(f"[{self.ticker}] Loading Earnings Data...")
        self.earnings_data = self._fetch_or_load("earnings", self._download_earnings)

    def _fetch_or_load(self, data_type, download_func):
        """Generic caching logic."""
        filename = f"{self.ticker}_{data_type}.csv"
        filepath = os.path.join(self.DATA_DIR, filename)

        if os.path.exists(filepath):
            logger.info(f"[{self.ticker}] Found cache for {data_type}.")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return self._sanitize_df(df)
        else:
            logger.info(f"[{self.ticker}] No cache. Downloading {data_type}...")
            df = download_func()
            if df is not None and not df.empty:
                # Sanitize BEFORE saving to ensure cache is clean
                df = self._sanitize_df(df)
                df.to_csv(filepath)
            return df

    def _download_prices(self):
        """Downloads max history via yfinance."""
        try:
            # period="max" can return extensive history
            df = yf.download(self.ticker, period="10y", progress=False, multi_level_index=False)
            return df
        except Exception as e:
            logger.error(f"[{self.ticker}] Price download failed: {e}")
            return pd.DataFrame()

    def _download_earnings(self):
        """Downloads earnings dates via yfinance."""
        try:
            t = yf.Ticker(self.ticker)
            df = t.get_earnings_dates(limit=30) # Get last 30 entries
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"[{self.ticker}] Earnings download failed: {e}")
            return pd.DataFrame()

    def _sanitize_df(self, df):
        """
        Aggressive Data Sanitization.
        1. Flatten MultiIndex (if any remains).
        2. Enforce DatetimeIndex & Remove Timezones.
        3. Coerce Numerics.
        4. Calculate Log Returns.
        """
        if df is None or df.empty:
            return df
            
        # 1. Flatten MultiIndex Columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            # If MultiIndex exists, we likely have ('Close', 'AAPL'). 
            # We just want the first level generally if it's OHLCV
            df.columns = df.columns.get_level_values(0)

        # 2. Enforce DatetimeIndex & Remove TZ
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        
        # Remove timezone info to normalize comparisons
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 3. Coerce Numerics
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Surprise(%)']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Calculate Log Returns (if Price data)
        if 'Close' in df.columns:
            # Sort index just in case
            df = df.sort_index()
            # Handle potential zeros before log
            safe_close = df['Close'].replace(0, np.nan)
            df['ret'] = np.log(safe_close / safe_close.shift(1))

        return df

    # --- Analysis Methods (Immutability Enforced) ---

    def analyze_momentum(self):
        """Calculates 3M, 6M, 12M Momentum."""
        if self.price_history is None: return None
        
        df = self.price_history.copy() # Immutable copy
        
        # trading days approximation: 21 days/mo
        df['Mom_3M'] = df['Close'] / df['Close'].shift(63) - 1.0
        df['Mom_6M'] = df['Close'] / df['Close'].shift(126) - 1.0
        df['Mom_12M'] = df['Close'] / df['Close'].shift(252) - 1.0
        
        return df[['Close', 'Mom_3M', 'Mom_6M', 'Mom_12M']].dropna()

    def analyze_reversal(self):
        """Weekly Reversal: Past 5d vs Next 5d."""
        if self.price_history is None: return None
        
        df = self.price_history.copy()
        
        # Calculate rolling 5-day returns (approx 1 week)
        df['5d_ret'] = df['Close'].pct_change(5)
        
        # Past 5d is just the current row's 5d_ret
        # Next 5d is the 5d_ret shifted BACKWARDS by 5 days
        df['Next_5d_ret'] = df['5d_ret'].shift(-5)
        
        df = df.dropna()
        
        df['Direction'] = np.where(df['5d_ret'] > 0, 'Up', 'Down')
        
        # Groupby
        stats = df.groupby('Direction')['Next_5d_ret'].mean()
        return stats # Series: Index=[Down, Up], Values=Mean Return

    def analyze_turn_of_month(self):
        """TOM: Last 5 trading days of month vs Rest."""
        if self.price_history is None: return None
        
        df = self.price_history.copy()
        
        # Add auxiliary columns for grouping
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        
        # Identify last 5 days of every month
        # We assign a reverse day count per month group
        # Day 1 = Last day of month
        df['Reverse_Day'] = df.groupby(['Year', 'Month']).cumcount(ascending=False) + 1
        
        df['Is_TOM'] = df['Reverse_Day'] <= 5
        
        stats = df.groupby('Is_TOM')['ret'].mean()
        # Transform log returns back to simple arithmetic mean for display if desired, 
        # or keep as mean daily log return
        return stats

    def analyze_pead(self):
        """PEAD: Drift T+0 to T+20 post earnings."""
        if self.price_history is None or self.earnings_data is None: return None
        
        prices = self.price_history.copy()
        earnings = self.earnings_data.copy()
        
        results = []
        
        # Iterate through earnings events
        for date, row in earnings.iterrows():
            if date not in prices.index:
                # Try finding nearest trading day if exact match fails
                try:
                    loc = prices.index.get_indexer([date], method='nearest')[0]
                    date = prices.index[loc]
                except:
                    continue
            
            # Get integer location of the date
            try:
                idx_loc = prices.index.get_loc(date)
            except KeyError:
                continue

            # Need T+0 to T+20 (20 trading days forward)
            if idx_loc + 20 >= len(prices):
                continue
                
            price_t0 = prices['Close'].iloc[idx_loc]
            price_t20 = prices['Close'].iloc[idx_loc + 20]
            
            drift = (price_t20 / price_t0) - 1.0
            
            # Check for surprise
            if 'Surprise(%)' in row and pd.notnull(row['Surprise(%)']):
                surprise = row['Surprise(%)']
                sType = 'Positive' if surprise > 0 else 'Negative'
                results.append({
                    'Date': date,
                    'Drift_20d': drift,
                    'Surprise': surprise,
                    'Type': sType
                })
                
        return pd.DataFrame(results)

    def analyze_options_snapshot(self):
        """
        Fetches option chain for next monthly exp, calculates robust IV if missing,
        returns ATM Call/Put details.
        """
        try:
            t = yf.Ticker(self.ticker)
            exps = t.options
            if not exps:
                return None
            
            # Find next monthly expiration (heuristic: usually 3rd Friday)
            # Simplification: Just take the 2nd or 3rd available to avoid immediate expiry noise
            target_exp = exps[1] if len(exps) > 1 else exps[0]
            
            chain = t.option_chain(target_exp)
            calls = chain.calls
            puts = chain.puts
            
            # Get Spot Price (approximate from latest history or live fetch)
            # using price_history latest close as proxy if live fetch fails
            spot = self.price_history['Close'].iloc[-1]
            
            # Risk free rate approx
            r = 0.045 
            
            # Time to maturity
            exp_date = pd.to_datetime(target_exp)
            today = pd.to_datetime(datetime.date.today())
            days_to_exp = (exp_date - today).days
            T = max(days_to_exp / 365.0, 0.001)

            # Find ATM
            # Calculate absolute difference between strike and spot
            calls['abs_diff'] = abs(calls['strike'] - spot)
            puts['abs_diff'] = abs(puts['strike'] - spot)
            
            atm_call = calls.loc[calls['abs_diff'].idxmin()].copy()
            atm_put = puts.loc[puts['abs_diff'].idxmin()].copy()
            
            # Helper to process row
            def process_option(row, type_):
                mid_price = (row['bid'] + row['ask']) / 2
                if mid_price == 0: mid_price = row['lastPrice']
                
                # Check provided IV
                iv = row['impliedVolatility']
                
                # If IV is suspicious (very low or zero), calculate it
                if pd.isna(iv) or iv < 0.01:
                    iv = impl_vol_newton(mid_price, spot, row['strike'], T, r, type_)
                
                # Calculate Delta
                _, delta = bs_price_and_greeks(spot, row['strike'], T, r, iv, type_)
                
                return {
                    'Type': type_.capitalize(),
                    'Strike': row['strike'],
                    'Bid': row['bid'],
                    'Ask': row['ask'],
                    'Mid': mid_price,
                    'IV': round(iv, 4),
                    'Delta': round(delta, 4),
                    'Expiry': target_exp
                }

            res_call = process_option(atm_call, "call")
            res_put = process_option(atm_put, "put")
            
            return pd.DataFrame([res_call, res_put])

        except Exception as e:
            logger.error(f"[{self.ticker}] Options analysis failed: {e}")
            return None


# --- Visualization Engine ---

def generate_dashboard(ticker, momentum, reversal, tom, pead, options):
    """Creates a comprehensive Plotly HTML dashboard."""
    
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2, "secondary_y": True}, None],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        subplot_titles=(
            f"{ticker} Price & Momentum (3M, 12M)", 
            "Weekly Reversal (Next 5d Return)", 
            "Turn of Month Effect (Daily Mean Return)", 
            "PEAD: Post-Earnings Drift (20d)",
            "ATM Options Snapshot"
        ),
        vertical_spacing=0.1
    )

    # 1. Momentum Chart (Row 1, Col 1-2)
    # Price
    fig.add_trace(
        go.Scatter(x=momentum.index, y=momentum['Close'], name='Price', line=dict(color='black', width=1)),
        row=1, col=1, secondary_y=True
    )
    # Mom 3M
    fig.add_trace(
        go.Scatter(x=momentum.index, y=momentum['Mom_3M'], name='Mom 3M', line=dict(color='blue', width=1, dash='dot')),
        row=1, col=1, secondary_y=False
    )
    # Mom 12M
    fig.add_trace(
        go.Scatter(x=momentum.index, y=momentum['Mom_12M'], name='Mom 12M', line=dict(color='orange', width=1.5)),
        row=1, col=1, secondary_y=False
    )

    # 2. Reversal (Row 2, Col 1)
    colors_rev = ['red' if idx == 'Down' else 'green' for idx in reversal.index]
    fig.add_trace(
        go.Bar(x=reversal.index, y=reversal.values, name='Reversal', marker_color=colors_rev),
        row=2, col=1
    )

    # 3. TOM (Row 2, Col 2)
    # Index is boolean (True=TOM, False=Rest)
    tom_labels = ['Rest of Month', 'TOM Days']
    tom_vals = [tom.get(False, 0), tom.get(True, 0)]
    fig.add_trace(
        go.Bar(x=tom_labels, y=tom_vals, name='TOM', marker_color=['gray', 'purple']),
        row=2, col=2
    )

    # 4. PEAD (Row 3, Col 1)
    if not pead.empty:
        colors_pead = pead['Type'].map({'Positive': 'green', 'Negative': 'red'})
        fig.add_trace(
            go.Scatter(
                x=pead['Date'], y=pead['Drift_20d'], 
                mode='markers', 
                marker=dict(color=colors_pead, size=8),
                text=pead['Surprise'],
                name='Earnings Drift'
            ),
            row=3, col=1
        )

    # 5. Options Table (Row 3, Col 2)
    if options is not None and not options.empty:
        fig.add_trace(
            go.Table(
                header=dict(values=list(options.columns), fill_color='paleturquoise', align='left'),
                cells=dict(values=[options[k].tolist() for k in options.columns], fill_color='lavender', align='left')
            ),
            row=3, col=2
        )

    fig.update_layout(height=1000, width=1200, title_text=f"Equity Anomalies Dashboard: {ticker}", showlegend=True)
    
    filename = f"dashboard_{ticker}.html"
    fig.write_html(filename)
    logger.info(f"[{ticker}] Dashboard saved to {filename}")


# --- Main Execution Flow ---

def main():
    parser = argparse.ArgumentParser(description="Equity Anomalies Dashboard")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers (e.g. AAPL MSFT)")
    args = parser.parse_args()

    for ticker in args.tickers:
        logger.info(f"--- Processing {ticker} ---")
        try:
            # 1. Initialize & Load
            analyzer = AnomaliesAnalyzer(ticker)
            analyzer.load_data()

            if analyzer.price_history is None or analyzer.price_history.empty:
                logger.warning(f"Skipping {ticker}: No price data found.")
                continue

            # 2. Analyze
            mom_df = analyzer.analyze_momentum()
            rev_res = analyzer.analyze_reversal()
            tom_res = analyzer.analyze_turn_of_month()
            pead_res = analyzer.analyze_pead()
            opt_res = analyzer.analyze_options_snapshot()

            # 3. Visualize
            generate_dashboard(ticker, mom_df, rev_res, tom_res, pead_res, opt_res)

        except Exception as e:
            logger.error(f"Critical error processing {ticker}: {e}")
            continue

if __name__ == "__main__":
    main()
