import sys
import os
import time
import datetime
import webbrowser
import warnings
import math
from typing import Tuple, Optional, List, Dict

# Third-party imports
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress pandas chained assignment warnings for cleaner output
pd.options.mode.chained_assignment = None

# ==============================================================================
# PART 1: STANDALONE QUANTITATIVE FUNCTIONS (PURE MATH)
# ==============================================================================

def norm_cdf(x: float) -> float:
    """Standard Normal Cumulative Distribution Function."""
    return norm.cdf(x)

def norm_pdf(x: float) -> float:
    """Standard Normal Probability Density Function."""
    return norm.pdf(x)

def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """Calculates d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Computes Black-Scholes price for a European option.
    """
    if T <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    
    d1, d2 = d1_d2(S, K, T, r, sigma)
    
    if option_type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        
    return price

def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Computes the Delta of an option.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1, _ = d1_d2(S, K, T, r, sigma)
    
    if option_type == 'call':
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0

def implied_volatility_solver(market_price: float, S: float, K: float, T: float, r: float, option_type: str = 'call') -> float:
    """
    Calculates Implied Volatility using a Bisection method.
    Returns NaN if price is below intrinsic value or did not converge.
    """
    # 1. Check Intrinsic Value
    intrinsic = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    if market_price <= intrinsic:
        return np.nan

    # 2. Bisection setup
    low = 0.001  # 0.1%
    high = 5.0   # 500%
    tolerance = 1e-5
    max_iter = 100

    for _ in range(max_iter):
        mid = (low + high) / 2
        price_mid = black_scholes_price(S, K, T, r, mid, option_type)
        
        diff = price_mid - market_price
        
        if abs(diff) < tolerance:
            return mid
        
        if diff > 0:
            high = mid
        else:
            low = mid
            
    return np.nan

# ==============================================================================
# PART 2: MARKET DATA GATEWAY (I/O & SANITIZATION)
# ==============================================================================

class MarketDataGateway:
    """
    Handles all interactions with yfinance, disk I/O, and caching.
    Strictly forbids financial calculations.
    """
    
    def __init__(self, cache_dir: str = "./market_data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, ticker: str, data_type: str) -> str:
        """Generates a filename based on ticker and today's date."""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.cache_dir, f"{ticker}_{data_type}_{today}.csv")

    def _sanitize_payload(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mandatory Data Sanitization Pipeline.
        1. Flattens MultiIndexes.
        2. Enforces DatetimeIndex/Numeric types.
        3. Strips Timezones.
        """
        if df.empty:
            return df
        
        # 1. Flatten MultiIndexes
        if isinstance(df.columns, pd.MultiIndex):
            # If columns are like ('Close', 'AAPL'), take level 0
            df.columns = df.columns.get_level_values(0)
            
        # 2. Enforce DatetimeIndex
        # If 'Date' is a column, set it as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # If index is already datetime-like, ensure it's strictly DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass # Keep as is if conversion fails, though likely problematic
        
        # 3. Strip Timezones
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 4. Type Coercion for known financial columns
        numeric_cols = [
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'strike', 'lastPrice', 'bid', 'ask', 'change', 'openInterest', 'impliedVolatility'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_spot_history(self, ticker: str) -> pd.DataFrame:
        """Fetches OHLCV data. Read-Through Caching."""
        cache_path = self._get_cache_path(ticker, "spot")
        
        if os.path.exists(cache_path):
            print(f"[Gateway] Loading spot data from cache: {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return self._sanitize_payload(df)
        
        print(f"[Gateway] Downloading spot data for {ticker}...")
        # Get 1 year of history for volatility calculation
        df = yf.download(ticker, period="1y", progress=False)
        
        df = self._sanitize_payload(df)
        df.to_csv(cache_path)
        return df

    def get_full_option_chain(self, ticker: str) -> pd.DataFrame:
        """
        Iterates through all expirations to build a volatility surface.
        Includes Rate Limiting. Read-Through Caching.
        """
        cache_path = self._get_cache_path(ticker, "options")
        
        if os.path.exists(cache_path):
            print(f"[Gateway] Loading option chain from cache: {cache_path}")
            df = pd.read_csv(cache_path)
            # Ensure contractSymbol is string
            df['contractSymbol'] = df['contractSymbol'].astype(str)
            return self._sanitize_payload(df)
        
        print(f"[Gateway] Downloading option chain for {ticker} (this may take time)...")
        yf_ticker = yf.Ticker(ticker)
        
        try:
            expirations = yf_ticker.options
        except Exception as e:
            print(f"[Error] Could not fetch expirations: {e}")
            return pd.DataFrame()

        all_options = []
        
        for expiry in expirations:
            print(f"  -> Fetching expiry: {expiry}")
            try:
                # API Rate Limiting
                time.sleep(0.5) 
                
                opt = yf_ticker.option_chain(expiry)
                calls = opt.calls
                puts = opt.puts
                
                calls['optionType'] = 'call'
                puts['optionType'] = 'put'
                
                # Tag with expiry date
                calls['expirationDate'] = expiry
                puts['expirationDate'] = expiry
                
                all_options.append(calls)
                all_options.append(puts)
                
            except Exception as e:
                print(f"  [Warning] Failed to fetch {expiry}: {e}")
                continue
        
        if not all_options:
            return pd.DataFrame()
            
        master_df = pd.concat(all_options, ignore_index=True)
        master_df = self._sanitize_payload(master_df)
        
        # Save cache
        master_df.to_csv(cache_path, index=False)
        return master_df

# ==============================================================================
# PART 3: VOLATILITY ANALYZER (LOGIC LAYER)
# ==============================================================================

class VolatilityAnalyzer:
    """
    Pure Logic Layer. Stateless regarding retrieval.
    Calculates RV, IV, and Greeks.
    """
    
    def __init__(self, risk_free_rate: float = 0.045):
        # 4.5% Risk Free Rate assumption (approx current T-Bill)
        self.r = risk_free_rate

    def calculate_realized_volatility(self, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes 10d, 20d, 60d Realized Volatility.
        Copy-on-Write.
        """
        df = spot_df.copy()
        
        # Log Returns
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Annualized Volatility (Window std * sqrt(252))
        df['RV_10d'] = df['log_ret'].rolling(window=10).std() * np.sqrt(252)
        df['RV_20d'] = df['log_ret'].rolling(window=20).std() * np.sqrt(252)
        df['RV_60d'] = df['log_ret'].rolling(window=60).std() * np.sqrt(252)
        
        return df

    def compute_surface_metrics(self, options_df: pd.DataFrame, current_spot: float) -> pd.DataFrame:
        """
        Augments option chain with T (time to expiry), calculated IV, and Delta.
        Copy-on-Write.
        """
        df = options_df.copy()
        
        # 1. Calculate DTE and T (Years)
        today = datetime.datetime.now()
        df['expirationDate'] = pd.to_datetime(df['expirationDate'])
        df['dte'] = (df['expirationDate'] - today).dt.days
        # Filter expired options
        df = df[df['dte'] > 0]
        df['T'] = df['dte'] / 365.0
        
        # 2. Mid Price
        df['midPrice'] = (df['bid'] + df['ask']) / 2
        # Fallback to lastPrice if bid/ask is broken (zero width)
        mask_bad_quote = (df['midPrice'] == 0) | (df['bid'] == 0) | (df['ask'] == 0)
        df.loc[mask_bad_quote, 'midPrice'] = df.loc[mask_bad_quote, 'lastPrice']
        
        # 3. Calculate IV & Delta per row
        # (Vectorization of Bisection is hard, looping is acceptable for typical chain sizes < 5k rows)
        
        ivs = []
        deltas = []
        
        print("[Analyzer] Solving Implied Volatility & Greeks...")
        
        for idx, row in df.iterrows():
            # Extract params
            K = row['strike']
            T = row['T']
            price = row['midPrice']
            opt_type = row['optionType']
            
            # Solve IV
            iv = implied_volatility_solver(price, current_spot, K, T, self.r, opt_type)
            ivs.append(iv)
            
            # Solve Delta (use solved IV if valid, else NaN)
            if not np.isnan(iv):
                delta = calculate_delta(current_spot, K, T, self.r, iv, opt_type)
            else:
                delta = np.nan
            deltas.append(delta)
            
        df['calc_IV'] = ivs
        df['delta'] = deltas
        
        # Clean up failed calculations
        df.dropna(subset=['calc_IV', 'delta'], inplace=True)
        return df

    def extract_term_structure_and_skew(self, processed_chain: pd.DataFrame, current_spot: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregates metrics by Expiry.
        1. ATM IV
        2. 25-Delta Risk Reversal (Call IV - Put IV)
        3. 25-Delta Butterfly (Avg(Call, Put) - ATM)
        """
        results = []
        
        # Group by Expiration
        groups = processed_chain.groupby('expirationDate')
        
        for expiry, group in groups:
            dte = group['dte'].iloc[0]
            if dte < 2: continue # Skip 0-1 DTE noise
            
            calls = group[group['optionType'] == 'call']
            puts = group[group['optionType'] == 'put']
            
            if calls.empty or puts.empty: continue
            
            # --- 1. Find ATM IV (Strike closest to Spot) ---
            # Combine calls/puts to find contract nearest to spot
            # (Ideally uses Forward, but Spot is acceptable for Dashboard approx)
            atm_idx = (group['strike'] - current_spot).abs().idxmin()
            atm_iv = group.loc[atm_idx, 'calc_IV']
            
            # --- 2. Find 25-Delta Options ---
            # Find Call with Delta closest to 0.25
            call_25d_idx = (calls['delta'] - 0.25).abs().idxmin()
            iv_call_25 = calls.loc[call_25d_idx, 'calc_IV']
            
            # Find Put with Delta closest to -0.25
            put_25d_idx = (puts['delta'] - (-0.25)).abs().idxmin()
            iv_put_25 = puts.loc[put_25d_idx, 'calc_IV']
            
            # --- 3. Compute Metrics ---
            # Risk Reversal: Call Skew vs Put Skew (Positive = Bullish Skew, Negative = Bearish Skew)
            risk_reversal = iv_call_25 - iv_put_25
            
            # Butterfly: Convexity / Kurtosis
            butterfly = ((iv_call_25 + iv_put_25) / 2) - atm_iv
            
            results.append({
                'expirationDate': expiry,
                'dte': dte,
                'ATM_IV': atm_iv,
                'RR_25d': risk_reversal,
                'Fly_25d': butterfly,
                'IV_Call_25': iv_call_25,
                'IV_Put_25': iv_put_25
            })
            
        skew_df = pd.DataFrame(results).sort_values('dte')
        return skew_df

# ==============================================================================
# PART 4: MAIN EXECUTION & VISUALIZATION
# ==============================================================================

def main():
    # 1. Parse Arguments
    if len(sys.argv) < 2:
        print("Usage: python options_dashboard.py <TICKER>")
        sys.exit(1)
        
    ticker = sys.argv[1].upper()
    print(f"=== Starting Analysis for {ticker} ===")
    
    # 2. Initialize Gateway & Analyzer
    gateway = MarketDataGateway()
    analyzer = VolatilityAnalyzer(risk_free_rate=0.045) # 4.5% assumption
    
    # 3. Process Spot Data (Regime)
    spot_df = gateway.get_spot_history(ticker)
    if spot_df.empty:
        print("Error: No spot data found.")
        sys.exit(1)
        
    spot_df = analyzer.calculate_realized_volatility(spot_df)
    current_spot = spot_df['Close'].iloc[-1]
    print(f"Current Spot Price: ${current_spot:.2f}")
    
    # 4. Process Option Data (Surface)
    options_df = gateway.get_full_option_chain(ticker)
    if options_df.empty:
        print("Error: No options data found.")
        sys.exit(1)
        
    processed_ops = analyzer.compute_surface_metrics(options_df, current_spot)
    skew_metrics = analyzer.extract_term_structure_and_skew(processed_ops, current_spot)
    
    # 5. Build Dashboard (Plotly)
    print("Generating Dashboard...")
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            f"Volatility Regime: Realized (Hist) vs Current ATM IV Levels ({ticker})",
            "Skew Structure: 25-Delta Risk Reversal",
            "Term Structure: ATM Implied Volatility"
        ),
        vertical_spacing=0.15
    )
    
    # --- Chart 1: Regime Overlay (RV History) ---
    # Plot 20d RV
    fig.add_trace(
        go.Scatter(
            x=spot_df.index, y=spot_df['RV_20d'], 
            mode='lines', name='20d Realized Vol',
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1
    )
    
    # Plot 60d RV
    fig.add_trace(
        go.Scatter(
            x=spot_df.index, y=spot_df['RV_60d'], 
            mode='lines', name='60d Realized Vol',
            line=dict(color='gray', width=1, dash='dot')
        ),
        row=1, col=1
    )
    
    # Add horizontal line for Current Avg ATM IV to see where we stand relative to history
    curr_avg_iv = skew_metrics['ATM_IV'].mean() if not skew_metrics.empty else 0
    fig.add_hline(
        y=curr_avg_iv, line_dash="dash", line_color="red", 
        annotation_text=f"Current Avg ATM IV: {curr_avg_iv:.1%}", 
        row=1, col=1
    )

    # --- Chart 2: Skew Structure (Risk Reversal vs DTE) ---
    # Positive RR = Calls Expensive (Bullish), Negative = Puts Expensive (Bearish)
    fig.add_trace(
        go.Scatter(
            x=skew_metrics['dte'], y=skew_metrics['RR_25d'],
            mode='markers+lines', name='25d Risk Reversal',
            marker=dict(size=8, color='purple')
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_color='black', line_width=1, row=2, col=1)

    # --- Chart 3: Term Structure (ATM IV vs DTE) ---
    fig.add_trace(
        go.Scatter(
            x=skew_metrics['dte'], y=skew_metrics['ATM_IV'],
            mode='markers+lines', name='ATM IV',
            marker=dict(size=8, color='orange')
        ),
        row=2, col=2
    )

    # Formatting
    fig.update_layout(
        title_text=f"Volatility & Skew Dashboard: {ticker}",
        height=900,
        showlegend=True,
        template="plotly_white"
    )
    
    # Y-Axis Formatting to Percent
    fig.update_yaxes(tickformat=".1%", title="Volatility", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", title="Call IV - Put IV", row=2, col=1)
    fig.update_yaxes(tickformat=".1%", title="Implied Volatility", row=2, col=2)
    fig.update_xaxes(title="Days To Expiry", row=2, col=1)
    fig.update_xaxes(title="Days To Expiry", row=2, col=2)

    # 6. Save & Open
    output_file = "dashboard.html"
    fig.write_html(output_file)
    print(f"Success! Dashboard saved to {output_file}")
    
    abs_path = os.path.abspath(output_file)
    webbrowser.open(f"file://{abs_path}")

if __name__ == "__main__":
    main()
