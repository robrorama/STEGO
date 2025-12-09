# SCRIPTNAME: ok.sudden_drop_gtc_dashboard.V3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
sudden_drop_gtc_dashboard.py

A hedge-fund-grade, standalone Python 3 script for professional options traders.
It performs intraday sudden-drop detection, computes Greeks/IV via robust numerical methods,
builds GTC exit scenarios, and renders a fully offline Plotly HTML dashboard.

Usage:
    python sudden_drop_gtc_dashboard.py --ticker SPY --period 5d --interval 1m --open-html

Requirements:
    pip install yfinance pandas numpy plotly scipy
"""

import argparse
import datetime
import os
import sys
import time
import warnings
import webbrowser
from pathlib import Path

# --- Third-Party Imports ---
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline
import yfinance as yf

# Check for scipy
try:
    from scipy.stats import norm
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback for norm.cdf and norm.pdf will be defined in FinancialAnalysis

# --- Global Config & Safety ---
warnings.filterwarnings("ignore")  # Silence yfinance/pandas warnings for cleaner CLI output

# --- 1. Data Ingestion Class ---

class DataIngestion:
    """
    Handles robust data downloading, caching, and aggressive sanitization.
    Strictly follows yfinance column normalization and timezone stripping rules.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_intraday_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Loads intraday data from local cache if available/fresh, otherwise downloads from yfinance.
        """
        filename = f"intraday_{ticker}_{period}_{interval}.csv"
        filepath = self.output_dir / filename

        # 1. Try Load Cache
        if filepath.exists() and filepath.stat().st_size > 100:
            print(f"[INFO] Loading cached intraday data from {filepath}...")
            df = pd.read_csv(filepath, index_col=0)
            df = self._sanitize_df(df)
            if not df.empty:
                return df

        # 2. Download
        print(f"[INFO] Downloading intraday data for {ticker} (period={period}, interval={interval})...")
        time.sleep(1.0)  # Rate limit protection
        # Force group_by='column' to handle multi-index consistently
        df = yf.download(ticker, period=period, interval=interval, group_by='column', progress=False)
        
        # 3. Sanitize
        df = self._sanitize_df(df)

        # 4. Cache
        if not df.empty:
            df.to_csv(filepath)
            print(f"[INFO] Cached intraday data to {filepath}")
        else:
            print(f"[WARN] Downloaded data for {ticker} is empty.")

        return df

    def get_shadow_history(self, ticker: str) -> pd.DataFrame:
        """
        Loads or downloads 1 year of daily history for 'Shadow' context (long-term vol regime).
        """
        filename = f"shadow_daily_{ticker}.csv"
        filepath = self.output_dir / filename

        # 1. Try Load Cache
        if filepath.exists() and filepath.stat().st_size > 100:
            print(f"[INFO] Loading shadow history from {filepath}...")
            df = pd.read_csv(filepath, index_col=0)
            df = self._sanitize_df(df)
            if not df.empty:
                return df
        
        # 2. Backfill Logic
        return self._backfill_shadow_history(ticker, filepath)

    def _backfill_shadow_history(self, ticker: str, filepath: Path) -> pd.DataFrame:
        print(f"[INFO] Backfilling shadow history (1y daily) for {ticker}...")
        time.sleep(1.0)
        df = yf.download(ticker, period="1y", interval="1d", group_by='column', progress=False)
        df = self._sanitize_df(df)

        if df.empty:
            print("[WARN] Shadow history download failed/empty.")
            return pd.DataFrame()

        # Compute Shadow Metrics (e.g., Shadow GEX proxy)
        # Proxy: (Neutral Vol - Realized Vol) * Notional
        # We assume Neutral Vol is constant 20% (0.20) for this proxy, or 60-day MA.
        # Notional = Close * 1 (normalized).
        
        # Scalar safety wrapper
        def safe_float(x):
            if isinstance(x, pd.Series): return float(x.iloc[0])
            return float(x)

        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
        df['RealizedVol20d'] = df['LogRet'].rolling(window=20).std() * np.sqrt(252)
        
        # Simple Proxy Logic:
        # If Realized Vol < 15%, we are accumulating (positive GEX environment).
        # If Realized Vol > 25%, we are in fear (negative GEX environment).
        df['ShadowGEX_Proxy'] = (0.20 - df['RealizedVol20d']) * df['Close'] * 1000 # Scaling factor

        df.to_csv(filepath)
        return df

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressive sanitization logic to handle yfinance MultiIndex messes and ensure
        clean, flat, numeric, timezone-naive DataFrames.
        """
        if df.empty:
            return df

        # 1. Yfinance Column Normalization (MultiIndex Handling)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' is in level 1 (common with group_by='column' and single ticker)
            # We want columns like 'Close', 'Open', etc. 
            # Often yf returns (Ticker, PriceType) or (PriceType, Ticker).
            
            # If level 0 contains the ticker, and level 1 contains 'Close'
            if 'Close' in df.columns.get_level_values(1):
                df = df.swaplevel(0, 1, axis=1)
            
            # If level 0 contains 'Close' now (or originally), we flatten.
            # We prefer just the PriceType if it's unique, otherwise Ticker_PriceType.
            new_cols = []
            for col in df.columns:
                # col is a tuple
                c0 = str(col[0]).strip() # Should be PriceType (Close, Open...)
                c1 = str(col[1]).strip() # Should be Ticker or empty
                
                if c0 in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                    new_cols.append(c0) # Keep it simple: "Close"
                else:
                    new_cols.append(f"{c0}_{c1}") # Fallback
            df.columns = new_cols

        # 2. Strict Datetime Index
        # If index is currently generic RangeIndex, try to find a Date column
        if not isinstance(df.index, pd.DatetimeIndex):
            potential_dates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if potential_dates:
                df = df.set_index(potential_dates[0])

        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notnull()] # Drop NaT

        # 3. Strip Timezones
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        # 4. Numeric Coercion
        # Identify core columns and force numeric
        cols_to_coerce = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for c in cols_to_coerce:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Drop rows where Close is NaN (essential for calcs)
        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])

        return df

    def get_options_chain(self, ticker: str, expiry: str = None) -> tuple:
        """
        Fetches options chain. If expiry None, picks nearest non-expired.
        Returns: (calls_df, puts_df, expiry_date_str)
        """
        tk = yf.Ticker(ticker)
        try:
            expirations = tk.options
        except Exception as e:
            print(f"[ERROR] Could not fetch expirations for {ticker}: {e}")
            return pd.DataFrame(), pd.DataFrame(), None

        if not expirations:
            print(f"[WARN] No expirations found for {ticker}.")
            return pd.DataFrame(), pd.DataFrame(), None

        # Select Expiry
        target_expiry = expirations[0]
        if expiry:
            if expiry in expirations:
                target_expiry = expiry
            else:
                print(f"[WARN] Requested expiry {expiry} not found. Defaulting to {target_expiry}.")

        print(f"[INFO] Fetching options chain for expiry: {target_expiry}")
        try:
            opt = tk.option_chain(target_expiry)
            calls = opt.calls
            puts = opt.puts
            return calls, puts, target_expiry
        except Exception as e:
            print(f"[ERROR] Failed to fetch chain: {e}")
            return pd.DataFrame(), pd.DataFrame(), None


# --- 2. Financial Analysis Class ---

class FinancialAnalysis:
    """
    Core math logic: Sudden Drops, Black-Scholes, Greeks (Finite Diff), Scenarios.
    Immutable: never changes input DFs in place.
    """
    def __init__(self, intraday_df: pd.DataFrame, shadow_df: pd.DataFrame, riskfree: float = 0.04):
        self._intraday_df = intraday_df.copy()
        self._shadow_df = shadow_df.copy()
        self.r = riskfree

    # --- Math Helpers ---
    
    @staticmethod
    def _safe_scalar(val):
        """Extract scalar from 1-element Series or return val."""
        if isinstance(val, (pd.Series, np.ndarray)):
            if hasattr(val, 'iloc'):
                return float(val.iloc[0])
            elif hasattr(val, 'item'):
                return val.item()
        return float(val)

    @staticmethod
    def norm_cdf(x):
        if SCIPY_AVAILABLE:
            return norm.cdf(x)
        # Fallback approximation
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def norm_pdf(x):
        if SCIPY_AVAILABLE:
            return norm.pdf(x)
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    def black_scholes_price(self, S, K, T, r, sigma, flag='c'):
        """
        S: Spot, K: Strike, T: Time (years), r: Rate, sigma: Vol
        flag: 'c' or 'p'
        """
        S = float(S)
        K = float(K)
        T = float(T)
        sigma = float(sigma)
        
        if T <= 0: return max(0.0, S - K) if flag == 'c' else max(0.0, K - S)
        if sigma <= 0: return max(0.0, S - K) if flag == 'c' else max(0.0, K - S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if flag == 'c':
            price = S * self.norm_cdf(d1) - K * np.exp(-r * T) * self.norm_cdf(d2)
        else:
            price = K * np.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
        
        return price

    def implied_volatility(self, price, S, K, T, r, flag='c'):
        """
        Robust IV solver. Uses Newton-Raphson via Scipy if available, 
        else specific logic.
        """
        if T <= 0 or price <= 0: return 0.0

        # Objective function
        def obj_func(sigma):
            return self.black_scholes_price(S, K, T, r, sigma, flag) - price

        # Bounds
        LOW, HIGH = 1e-4, 5.0

        # Try Newton/Brentq if Scipy
        if SCIPY_AVAILABLE:
            try:
                # brentq is robust for bracketing
                return optimize.brentq(obj_func, LOW, HIGH, xtol=1e-4)
            except:
                pass # Fallback to manual

        # Manual Bisection Fallback
        low, high = LOW, HIGH
        for _ in range(50):
            mid = (low + high) / 2
            diff = obj_func(mid)
            if abs(diff) < 1e-4:
                return mid
            if diff < 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def calculate_greeks_fd(self, S, K, T, r, sigma, flag='c'):
        """
        Computes Greeks via Central Finite Differences.
        """
        # Perturbation sizes
        dS = S * 0.001
        dSigma = 0.001
        dT = 1 / 365.0
        dR = 0.0001

        base_price = self.black_scholes_price(S, K, T, r, sigma, flag)
        
        # Delta & Gamma
        p_up = self.black_scholes_price(S + dS, K, T, r, sigma, flag)
        p_down = self.black_scholes_price(S - dS, K, T, r, sigma, flag)
        delta = (p_up - p_down) / (2 * dS)
        gamma = (p_up - 2 * base_price + p_down) / (dS ** 2)

        # Vega (sensitivity to 1% vol change usually, but BS uses unit vol. Standard is unit change / 100)
        p_vol_up = self.black_scholes_price(S, K, T, r, sigma + dSigma, flag)
        p_vol_down = self.black_scholes_price(S, K, T, r, sigma - dSigma, flag)
        vega = (p_vol_up - p_vol_down) / (2 * dSigma) / 100.0 # Standard Vega definition

        # Theta (time decay per day)
        # Note: T is reducing as time passes. So we look at T - dT
        if T > dT:
            p_time_less = self.black_scholes_price(S, K, T - dT, r, sigma, flag)
            theta = (p_time_less - base_price) / 1.0 # Per day
        else:
            theta = 0.0

        # Rho
        p_r_up = self.black_scholes_price(S, K, T, r + dR, sigma, flag)
        p_r_down = self.black_scholes_price(S, K, T, r - dR, sigma, flag)
        rho = (p_r_up - p_r_down) / (2 * dR) / 100.0

        return {
            'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho, 
            'price': base_price
        }

    # --- Domain Logic ---

    def analyze_sudden_drops(self, jump_window_mins=5, rv_window_mins=60):
        """
        Detects sudden drops and jumps.
        """
        df = self._intraday_df.copy()
        if df.empty: return pd.DataFrame(), pd.DataFrame()

        # Resample logic (ensure uniform intervals approx)
        # Since we just loaded sanitized data, we assume it's clean 1m or 5m
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Rolling Realized Volatility (Annualized)
        # Assumes 'interval' was passed correctly in ingestion. 
        # We approximate bars per year based on index freq
        
        # Determine frequency in minutes roughly
        if len(df) > 1:
            diffs = df.index.to_series().diff().dropna()
            freq_mins = diffs.median().total_seconds() / 60.0
            if freq_mins == 0: freq_mins = 1
        else:
            freq_mins = 1

        rv_window_bars = int(rv_window_mins / freq_mins)
        jump_window_bars = int(jump_window_mins / freq_mins)
        
        # Rolling RV
        df['RV'] = df['LogRet'].rolling(window=rv_window_bars).std() * np.sqrt(252 * 6.5 * 60 / freq_mins) # Approx bars per year
        
        # Rolling Bipower Variation (Robust to jumps)
        # BV ~ (pi/2) * sum(|r_t|*|r_t-1|)
        abs_ret = df['LogRet'].abs()
        df['BV'] = (np.pi / 2) * (abs_ret * abs_ret.shift(1)).rolling(window=rv_window_bars).mean() 
        # Scale BV to Volatility scale for comparison
        df['BV_Vol'] = np.sqrt(df['BV'] * 252 * 6.5 * 60 / freq_mins)

        # Jump Statistic (RV / BV). If > 1.0 significant, > 1.5-2.0 is jump
        df['JumpStat'] = np.where(df['BV_Vol'] > 0, df['RV'] / df['BV_Vol'], 1.0)

        # Drawdown in jump window
        df['RollingMax'] = df['Close'].rolling(window=jump_window_bars).max()
        df['Drawdown'] = (df['Close'] - df['RollingMax']) / df['RollingMax']

        # Z-Score of returns over window
        win_mean = df['LogRet'].rolling(window=jump_window_bars).mean()
        win_std = df['LogRet'].rolling(window=jump_window_bars).std()
        df['ZScore'] = (df['LogRet'] - win_mean) / win_std

        # Identify Drop Events
        # Criteria: Drawdown < -0.2% (intraday sudden) AND JumpStat > 1.2
        drops = df[ (df['Drawdown'] < -0.002) & (df['JumpStat'] > 1.1) ].copy()
        
        return df, drops
    def generate_gtc_grid(self, spot, target_expiry_str, calls_df, puts_df):
        """
        Builds the GTC Exit Scenario Grid.
        Updated to prevent duplicate strikes (fixes ValueError on Pivot).
        """
        # Parse Expiry
        try:
            exp_date = pd.to_datetime(target_expiry_str)
            now = pd.Timestamp.now()
            dte_days = (exp_date - now).days
            T = max(dte_days / 365.0, 1/365.0) # Avoid T=0
        except:
            return pd.DataFrame() # Fail gracefully

        grid_rows = []
        
        # Use a dictionary to ensure uniqueness: key = (opt_type, strike)
        unique_options = {}

        # Sanitize Options Data
        for opt_type, data in [('call', calls_df), ('put', puts_df)]:
            if data.empty: continue
            
            # Create a clean copy to avoid SettingWithCopy warnings
            data = data.copy()
            data['mid'] = (data['bid'] + data['ask']) / 2
            data = data[data['mid'] > 0]
            
            # Simple Moneyness filter for "Key Options"
            data['moneyness'] = data['strike'] / spot
            
            if not data.empty:
                # 1. Select ATM (closest to 1.0)
                atm_idx = (data['moneyness'] - 1.0).abs().idxmin()
                row = data.loc[atm_idx]
                unique_options[(opt_type, row['strike'])] = row
                
                # 2. Select Wings (approx +/- 2% moves)
                down_wing_df = data.iloc[(data['moneyness'] - 0.98).abs().argsort()[:1]]
                up_wing_df = data.iloc[(data['moneyness'] - 1.02).abs().argsort()[:1]]
                
                if not down_wing_df.empty:
                    r = down_wing_df.iloc[0]
                    unique_options[(opt_type, r['strike'])] = r
                if not up_wing_df.empty:
                    r = up_wing_df.iloc[0]
                    unique_options[(opt_type, r['strike'])] = r

        # Definition of Scenarios
        spot_shifts = [-0.01, -0.005, 0.0, 0.005, 0.01]
        iv_shifts = [-0.03, -0.01, 0.0, 0.01, 0.03] # Vol points

        # Iterate over UNIQUE options only
        for (opt_type, K), row in unique_options.items():
            market_price = row['mid']
            
            # Estimate Base IV
            base_iv = self.implied_volatility(market_price, spot, K, T, self.r, flag='c' if opt_type=='call' else 'p')
            if base_iv == 0: base_iv = 0.20 # Fallback

            for s_sh in spot_shifts:
                for v_sh in iv_shifts:
                    sim_spot = spot * (1 + s_sh)
                    sim_vol = max(0.01, base_iv + v_sh)
                    
                    res = self.calculate_greeks_fd(
                        sim_spot, K, T, self.r, sim_vol, flag='c' if opt_type=='call' else 'p'
                    )
                    
                    grid_rows.append({
                        'type': opt_type,
                        'strike': K,
                        'scenario_spot_chg': s_sh,
                        'scenario_vol_chg': v_sh,
                        'sim_price': res['price'],
                        'sim_delta': res['delta'],
                        'sim_gamma': res['gamma'],
                        'sim_vega': res['vega'],
                        'base_iv': base_iv
                    })
        
        return pd.DataFrame(grid_rows)

    def generate_gtc_gridOLD(self, spot, target_expiry_str, calls_df, puts_df):
        """
        Builds the GTC Exit Scenario Grid.
        """
        # Parse Expiry
        try:
            exp_date = pd.to_datetime(target_expiry_str)
            now = pd.Timestamp.now()
            dte_days = (exp_date - now).days
            T = max(dte_days / 365.0, 1/365.0) # Avoid T=0
        except:
            return pd.DataFrame() # Fail gracefully

        grid_rows = []

        # Filter for ATM and +/- 25 Delta (Approx via Strike/Spot)
        # Crude selection: ATM, 95%, 105% Moneyness
        # 0.98 < K/S < 1.02 -> ATM
        # < 0.95 -> OTM Put / ITM Call
        
        relevant_strikes = []
        
        # Sanitize Options Data
        for opt_type, data in [('call', calls_df), ('put', puts_df)]:
            if data.empty: continue
            
            data['mid'] = (data['bid'] + data['ask']) / 2
            data = data[data['mid'] > 0]
            
            # Simple Moneyness filter for "Key Options"
            # We want strikes around spot
            data['moneyness'] = data['strike'] / spot
            
            # Select ATM and wings
            # ATM: closest to 1.0
            if not data.empty:
                atm_idx = (data['moneyness'] - 1.0).abs().idxmin()
                relevant_strikes.append( (opt_type, data.loc[atm_idx]) )
                
                # Wings (approx +/- 2% moves)
                down_wing = data.iloc[(data['moneyness'] - 0.98).abs().argsort()[:1]]
                up_wing = data.iloc[(data['moneyness'] - 1.02).abs().argsort()[:1]]
                
                if not down_wing.empty: relevant_strikes.append((opt_type, down_wing.iloc[0]))
                if not up_wing.empty: relevant_strikes.append((opt_type, up_wing.iloc[0]))

        # Definition of Scenarios
        spot_shifts = [-0.01, -0.005, 0.0, 0.005, 0.01]
        iv_shifts = [-0.03, -0.01, 0.0, 0.01, 0.03] # Vol points

        for opt_type, row in relevant_strikes:
            K = row['strike']
            market_price = row['mid']
            
            # Estimate Base IV
            base_iv = self.implied_volatility(market_price, spot, K, T, self.r, flag='c' if opt_type=='call' else 'p')
            if base_iv == 0: base_iv = 0.20 # Fallback

            for s_sh in spot_shifts:
                for v_sh in iv_shifts:
                    sim_spot = spot * (1 + s_sh)
                    sim_vol = max(0.01, base_iv + v_sh)
                    
                    res = self.calculate_greeks_fd(
                        sim_spot, K, T, self.r, sim_vol, flag='c' if opt_type=='call' else 'p'
                    )
                    
                    grid_rows.append({
                        'type': opt_type,
                        'strike': K,
                        'scenario_spot_chg': s_sh,
                        'scenario_vol_chg': v_sh,
                        'sim_price': res['price'],
                        'sim_delta': res['delta'],
                        'sim_gamma': res['gamma'],
                        'sim_vega': res['vega'],
                        'base_iv': base_iv
                    })
        
        return pd.DataFrame(grid_rows)


# --- 3. Dashboard Renderer Class ---

class DashboardRenderer:
    def __init__(self, ticker: str):
        self.ticker = ticker

    def render_html(self, intraday_df, drops_df, shadow_df, gtc_grid, filename):
        """
        Constructs the offline HTML dashboard.
        """
        # 1. Plotly Figures
        
        # Fig 1: Intraday Price & Sudden Drops
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=intraday_df.index,
            open=intraday_df['Open'], high=intraday_df['High'],
            low=intraday_df['Low'], close=intraday_df['Close'],
            name='OHLC'
        ))
        # Add Markers for drops
        if not drops_df.empty:
            fig_price.add_trace(go.Scatter(
                x=drops_df.index, y=drops_df['Low'],
                mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Sudden Drop'
            ))
        fig_price.update_layout(title=f"{self.ticker} Intraday & Drops", template="plotly_dark", height=500)

        # Fig 2: Volatility Regime (RV/BV)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=intraday_df.index, y=intraday_df['RV'], name='Realized Vol (Ann)'))
        fig_vol.add_trace(go.Scatter(x=intraday_df.index, y=intraday_df['BV_Vol'], name='Bipower Vol (Ann)'))
        fig_vol.add_trace(go.Scatter(x=intraday_df.index, y=intraday_df['JumpStat'], name='Jump Stat', yaxis='y2', line=dict(dash='dot', color='gray')))
        fig_vol.update_layout(
            title="Intraday Volatility & Jump Stats", template="plotly_dark", height=400,
            yaxis2=dict(title="Jump Ratio", overlaying='y', side='right')
        )

        # Fig 3: Shadow History
        fig_shadow = go.Figure()
        if not shadow_df.empty:
            fig_shadow.add_trace(go.Scatter(x=shadow_df.index, y=shadow_df['Close'], name='Close'))
            fig_shadow.add_trace(go.Bar(x=shadow_df.index, y=shadow_df['ShadowGEX_Proxy'], name='Shadow GEX Proxy', yaxis='y2', opacity=0.3))
            fig_shadow.update_layout(
                title="1-Year Shadow History & GEX Proxy", template="plotly_dark", height=400,
                yaxis2=dict(title="GEX Proxy", overlaying='y', side='right')
            )

        # Fig 4: GTC Exit Heatmap (Price vs Spot/Vol Shift) for ATM Call (Example)
        fig_heat = go.Figure()
        if not gtc_grid.empty:
            # Pivot for Heatmap: Filter for first call option found (likely ATM)
            # Find the strike with the most rows (should be uniform, but safety first)
            sample_strike = gtc_grid['strike'].mode()[0]
            subset = gtc_grid[ (gtc_grid['strike'] == sample_strike) & (gtc_grid['type'] == 'call') ]
            
            if not subset.empty:
                pivot = subset.pivot(index='scenario_vol_chg', columns='scenario_spot_chg', values='sim_price')
                fig_heat.add_trace(go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='Viridis',
                    texttemplate="%{z:.2f}",
                    showscale=True
                ))
                fig_heat.update_layout(
                    title=f"GTC Exit Scenario: Call Strike {sample_strike} (Price)",
                    xaxis_title="Spot Shift", yaxis_title="Vol Shift",
                    template="plotly_dark", height=500
                )

        # 2. HTML Construction (Offline & Tabs)
        
        # Get raw JS
        plotly_js = plotly.offline.get_plotlyjs()
        
        # Get Divs
        div_price = plotly.offline.plot(fig_price, include_plotlyjs=False, output_type='div')
        div_vol = plotly.offline.plot(fig_vol, include_plotlyjs=False, output_type='div')
        div_shadow = plotly.offline.plot(fig_shadow, include_plotlyjs=False, output_type='div')
        div_heat = plotly.offline.plot(fig_heat, include_plotlyjs=False, output_type='div')

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{self.ticker} Options Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: sans-serif; background-color: #1e1e1e; color: #ddd; margin: 0; padding: 20px; }}
                .tab {{ overflow: hidden; border: 1px solid #444; background-color: #333; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ddd; font-weight: bold; }}
                .tab button:hover {{ background-color: #555; }}
                .tab button.active {{ background-color: #007bff; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #444; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h2>{self.ticker} Sudden Drop & GTC Dashboard</h2>
            
            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Intraday')" id="defaultOpen">Intraday Analysis</button>
                <button class="tablinks" onclick="openTab(event, 'Scenarios')">GTC Scenarios</button>
                <button class="tablinks" onclick="openTab(event, 'Shadow')">Shadow History</button>
            </div>

            <div id="Intraday" class="tabcontent">
                <h3>Price Action & Volatility</h3>
                {div_price}
                {div_vol}
            </div>

            <div id="Scenarios" class="tabcontent">
                <h3>GTC Exit Planning (Heatmaps)</h3>
                <p>Simulated Call Price under Spot/Vol Shocks</p>
                {div_heat}
            </div>

            <div id="Shadow" class="tabcontent">
                <h3>Historical Context</h3>
                {div_shadow}
            </div>

            <script>
            function openTab(evt, cityName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(cityName).style.display = "block";
                evt.currentTarget.className += " active";
                
                // Trigger resize for Plotly
                window.dispatchEvent(new Event('resize'));
            }}
            
            // Open default
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """

        with open(filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[INFO] Dashboard generated: {filename}")
        return filename


# --- 4. Main Execution Flow ---

def main():
    parser = argparse.ArgumentParser(description="Sudden Drop GTC Dashboard")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--period", type=str, default="5d", help="Intraday period (1d, 5d)")
    parser.add_argument("--interval", type=str, default="1m", help="Intraday interval (1m, 5m)")
    parser.add_argument("--riskfree", type=float, default=0.045, help="Risk-free rate")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--open-html", action="store_true", help="Auto-open HTML in browser")
    parser.add_argument("--expiry", type=str, default=None, help="Specific expiry YYYY-MM-DD")
    
    args = parser.parse_args()

    # 1. Instantiate Classes
    ingestor = DataIngestion(args.output_dir)
    renderer = DashboardRenderer(args.ticker)

    # 2. Ingest Data
    intraday_df = ingestor.get_intraday_data(args.ticker, args.period, args.interval)
    shadow_df = ingestor.get_shadow_history(args.ticker)
    
    # 3. Analyze
    if intraday_df.empty:
        print("[ERROR] No intraday data available. Exiting.")
        sys.exit(1)

    analyzer = FinancialAnalysis(intraday_df, shadow_df, args.riskfree)
    
    print("[INFO] Analyzing sudden drops...")
    analyzed_df, drops_df = analyzer.analyze_sudden_drops()
    if not drops_df.empty:
        print(f"[ALERT] {len(drops_df)} sudden drop(s) detected!")
        print(drops_df[['Close', 'Drawdown', 'JumpStat']].tail())
    else:
        print("[INFO] No significant sudden drops detected in window.")

    # 4. GTC Grid (Options)
    print("[INFO] Building GTC Exit Scenario Grid...")
    calls, puts, exp_date = ingestor.get_options_chain(args.ticker, args.expiry)
    
    gtc_grid = pd.DataFrame()
    if not calls.empty and not puts.empty:
        # Get last price for spot reference
        current_spot = FinancialAnalysis._safe_scalar(analyzed_df['Close'].iloc[-1])
        gtc_grid = analyzer.generate_gtc_grid(current_spot, exp_date, calls, puts)
    else:
        print("[WARN] Skipping GTC Grid due to missing options data.")

    # 5. Render
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sudden_drop_dashboard_{args.ticker}_{timestamp}.html"
    filepath = os.path.join(args.output_dir, filename)
    
    renderer.render_html(analyzed_df, drops_df, shadow_df, gtc_grid, filepath)

    if args.open_html:
        webbrowser.open(f"file://{os.path.abspath(filepath)}")

if __name__ == "__main__":
    main()
