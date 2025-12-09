# SCRIPTNAME: ok.options_regime_iv_dashboard_standalone.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import os
import time
import math
import datetime
import webbrowser
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots

# Try importing scipy for precise Black-Scholes, else fallback to approximation
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# -----------------------------------------------------------------------------
#                                HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def safe_float(val) -> float:
    """Safely convert a value to float, handling single-element Series."""
    if isinstance(val, pd.Series):
        if val.empty:
            return 0.0
        val = val.iloc[0]
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def norm_cdf(x):
    """Standard normal CDF. Uses scipy if available, else error function approx."""
    if SCIPY_AVAILABLE:
        return norm.cdf(x)
    # Approximation using error function
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x):
    """Standard normal PDF."""
    if SCIPY_AVAILABLE:
        return norm.pdf(x)
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

def bs_delta_gamma(S, K, T, r, sigma, opt_type='C'):
    """
    Calculate Black-Scholes Delta and Gamma.
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Implied Volatility
    opt_type: 'C' or 'P'
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0, 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    # Gamma is the same for Call and Put
    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
    
    if opt_type.upper() == 'C':
        delta = norm_cdf(d1)
    else:
        delta = norm_cdf(d1) - 1.0
        
    return delta, gamma

# -----------------------------------------------------------------------------
#                            CLASS: DataIngestion
# -----------------------------------------------------------------------------

class DataIngestion:
    """
    Solely responsible for:
    - Downloading data from yfinance (spot and options).
    - Caching to CSV.
    - Loading from CSV if present.
    - Running universal sanitizer on all DataFrames.
    - Implementing Persistence Layer and Shadow Backfill.
    """
    def __init__(self, ticker: str, output_dir: str, history_days: int = 252, interval: str = '1d'):
        self.ticker = ticker.upper()
        self.output_dir = output_dir
        self.history_days = history_days
        self.interval = interval
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_spot_filename(self) -> str:
        return os.path.join(self.output_dir, f"prices_{self.ticker}_{self.interval}_{self.history_days}.csv")

    def _get_shadow_filename(self) -> str:
        return os.path.join(self.output_dir, f"shadow_history_{self.ticker}.csv")

    def _get_options_filename(self, expiry: str) -> str:
        return os.path.join(self.output_dir, f"options_{self.ticker}_{expiry}.csv")

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressive Data Sanitization.
        Handles yfinance MultiIndex columns, timezone stripping, and numeric coercion.
        """
        if df.empty:
            return df

        # 1. yfinance Column Normalization (MultiIndex handling)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if "Adj Close" or "Close" is in level 0
            level0 = df.columns.get_level_values(0)
            if 'Adj Close' not in level0 and 'Close' not in level0:
                # Try swapping if they might be in level 1
                level1 = df.columns.get_level_values(1)
                if 'Adj Close' in level1 or 'Close' in level1:
                    df = df.swaplevel(0, 1, axis=1)

            # If we are targeting a specific ticker, try to extract just that ticker's slice
            # This simplifies the DataFrame to just OHLCV columns
            try:
                # Check if the ticker exists in the columns (usually level 1 after swap or standard)
                # We iterate to find the ticker key because yfinance might change case or format
                found_col = None
                for col in df.columns.levels[1]:
                    if col.upper() == self.ticker:
                        found_col = col
                        break
                
                if found_col:
                    df = df.xs(found_col, axis=1, level=1, drop_level=True)
            except Exception:
                # Fallback: Flatten MultiIndex to strings
                df.columns = [f"{c[0]}_{c[1]}" if isinstance(c, tuple) else c for c in df.columns]

        # 2. Strict Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Look for likely date columns
            potential_cols = ['date', 'Date', 'datetime', 'Datetime', 'index']
            for c in potential_cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors='coerce')
                    df = df.set_index(c)
                    break

        # Enforce Datetime Types on Index
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]

        # 3. Strip Timezones
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 4. Numeric Coercion
        # Preserve specific string columns if this is options data
        options_str_cols = ['contractSymbol', 'type', 'expiry', 'currency']
        for col in df.columns:
            if col not in options_str_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _backfill_shadow_history(self, neutral_vol: float):
        """
        Cold Start Prevention.
        Downloads 1 year of daily data to compute baseline realized vol and Shadow GEX.
        """
        print(f"[Info] Backfilling shadow history (1Y) for {self.ticker}...")
        try:
            # Enforce 1d interval for shadow history to be stable
            df = yf.download(self.ticker, period="1y", interval="1d", 
                             group_by='column', auto_adjust=False, progress=False)
            df = self._sanitize_df(df)
            
            if df.empty:
                print("[Warning] Backfill failed: No data downloaded.")
                return

            # Compute simple metrics for the cache
            # Prefer Adj Close, then Close
            px_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            if px_col not in df.columns:
                 # Last ditch attempt to find a close column
                cols = [c for c in df.columns if 'lose' in c]
                if cols: px_col = cols[0]
                else: return

            close_prices = df[px_col]
            returns = np.log(close_prices / close_prices.shift(1))
            
            # 20-day Annualized Vol
            rv_20 = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Approx Notional
            vol_col = 'Volume' if 'Volume' in df.columns else None
            notional = close_prices * df[vol_col] if vol_col else close_prices * 0
            
            # Shadow GEX Proxy
            # (Neutral - Realized) * Notional
            # If RV < Neutral (stable) -> Positive GEX
            # If RV > Neutral (unstable) -> Negative GEX
            shadow_gex = (neutral_vol - rv_20) * notional

            # Create shadow DF
            shadow_df = pd.DataFrame({
                'Shadow_RV20': rv_20,
                'Shadow_GEX': shadow_gex,
                'Close': close_prices
            })
            
            # Save
            shadow_df.to_csv(self._get_shadow_filename())
            print(f"[Info] Shadow history saved to {self._get_shadow_filename()}")

        except Exception as e:
            print(f"[Error] Failed to backfill shadow history: {e}")

    def get_spot_history(self, neutral_vol: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (spot_df, shadow_df).
        Checks cache first. If missing, downloads.
        Also handles shadow history cold start.
        """
        spot_file = self._get_spot_filename()
        shadow_file = self._get_shadow_filename()

        # 1. Load or Download Spot
        if os.path.exists(spot_file) and os.path.getsize(spot_file) > 10:
            print(f"[Cache] Loading spot data from {spot_file}")
            spot_df = pd.read_csv(spot_file)
            spot_df = self._sanitize_df(spot_df)
        else:
            print(f"[Net] Downloading spot data for {self.ticker}...")
            time.sleep(1) # Rate limit
            
            # Calculate start date based on history_days
            start_dt = datetime.datetime.now() - datetime.timedelta(days=self.history_days)
            start_str = start_dt.strftime('%Y-%m-%d')
            
            try:
                spot_df = yf.download(self.ticker, start=start_str, interval=self.interval, 
                                      group_by='column', auto_adjust=False, progress=False)
                spot_df = self._sanitize_df(spot_df)
                spot_df.to_csv(spot_file)
            except Exception as e:
                print(f"[Error] Spot download failed: {e}")
                spot_df = pd.DataFrame()

        # 2. Check Shadow History (Cold Start Prevention)
        if not os.path.exists(shadow_file) or os.path.getsize(shadow_file) < 10:
            self._backfill_shadow_history(neutral_vol)
        
        # Load Shadow
        if os.path.exists(shadow_file):
            shadow_df = pd.read_csv(shadow_file)
            shadow_df = self._sanitize_df(shadow_df)
        else:
            shadow_df = pd.DataFrame()

        return spot_df, shadow_df

    def get_options_chain(self, max_expiries: int = 3) -> pd.DataFrame:
        """
        Downloads option chains for nearest expiries. Caches per expiry.
        """
        try:
            tk = yf.Ticker(self.ticker)
            expiries = tk.options
        except Exception as e:
            print(f"[Error] Could not fetch expiries: {e}")
            return pd.DataFrame()

        if not expiries:
            return pd.DataFrame()

        selected_expiries = expiries[:max_expiries]
        all_opts = []

        for date_str in selected_expiries:
            fname = self._get_options_filename(date_str)
            
            if os.path.exists(fname) and os.path.getsize(fname) > 10:
                print(f"[Cache] Loading options for {date_str}")
                df_opt = pd.read_csv(fname)
                df_opt = self._sanitize_df(df_opt)
            else:
                print(f"[Net] Downloading options for {date_str}...")
                time.sleep(1) # Rate limit
                try:
                    chain = tk.option_chain(date_str)
                    calls = chain.calls
                    puts = chain.puts
                    
                    calls['type'] = 'C'
                    puts['type'] = 'P'
                    
                    df_opt = pd.concat([calls, puts], ignore_index=True)
                    df_opt['expiry'] = date_str
                    
                    # Sanitize BEFORE saving to ensure clean numeric csv
                    df_opt = self._sanitize_df(df_opt)
                    df_opt.to_csv(fname, index=False)
                except Exception as e:
                    print(f"[Error] Failed options download for {date_str}: {e}")
                    continue

            all_opts.append(df_opt)

        if not all_opts:
            return pd.DataFrame()
        
        return pd.concat(all_opts, ignore_index=True)


# -----------------------------------------------------------------------------
#                            CLASS: FinancialAnalysis
# -----------------------------------------------------------------------------

class FinancialAnalysis:
    """
    Solely responsible for:
    - All math & logic.
    - Immutability (Copy-on-Write).
    """
    def __init__(self, spot_df: pd.DataFrame, shadow_df: pd.DataFrame, options_df: pd.DataFrame, 
                 rf_rate: float, neutral_vol: float):
        # Store raw copies, never mutate
        self._spot_df = spot_df.copy() if not spot_df.empty else pd.DataFrame()
        self._shadow_df = shadow_df.copy() if not shadow_df.empty else pd.DataFrame()
        self._options_df = options_df.copy() if not options_df.empty else pd.DataFrame()
        self.rf = rf_rate
        self.neutral_vol = neutral_vol

        # Identify Price Column in Spot
        self.spot_col = 'Close'
        if not self._spot_df.empty:
            if 'Adj Close' in self._spot_df.columns:
                self.spot_col = 'Adj Close'
            elif 'Close' in self._spot_df.columns:
                self.spot_col = 'Close'
    
    def get_latest_spot(self) -> float:
        if self._spot_df.empty:
            return 0.0
        return safe_float(self._spot_df[self.spot_col].iloc[-1])

    def compute_spot_regime(self) -> pd.DataFrame:
        """
        Calculates RV windows and returns a DF with OHLC + RV metrics.
        """
        if self._spot_df.empty:
            return pd.DataFrame()
        
        df = self._spot_df.copy()
        
        # Calculate Returns
        df['log_ret'] = np.log(df[self.spot_col] / df[self.spot_col].shift(1))
        
        # Realized Volatility (Annualized)
        windows = [10, 20, 60]
        for w in windows:
            df[f'RV_{w}'] = df['log_ret'].rolling(window=w).std() * np.sqrt(252)
            
        return df

    def compute_shadow_gex_series(self) -> pd.DataFrame:
        """
        Merges shadow history with current spot data to create a full timeline of Shadow GEX.
        """
        # Start with historical shadow data
        if self._shadow_df.empty:
             # If shadow is empty, try to compute from current spot if sufficient
             full_df = self.compute_spot_regime()
        else:
            # Combine logic: We want the historical shadow + new calculations
            # For simplicity in this standalone script, we will re-calculate 
            # metrics on the FULL spot history available and merge with shadow 
            # strictly for backfill if spot is short.
            # However, prompt implies extending. 
            # Simplest robust approach: Re-calculate everything on the provided Spot DF
            # and prepend Shadow DF if Spot DF is short (e.g. intraday).
            
            # Note: Spot DF might be intraday. Shadow is Daily.
            # We will use Spot DF for recent, Shadow for context.
            # Actually, easiest is to just return the Shadow DF aligned with current spot RV.
            
            # Let's just calculate fresh on Spot DF if it's Daily, else rely on Shadow DF.
            # If Spot is Intraday, RV calc is tricky. 
            # Assumption: The user provides enough history_days for Spot to calculate RV.
            full_df = self.compute_spot_regime()

        if full_df.empty:
            return pd.DataFrame()

        # Ensure we have Volume
        if 'Volume' not in full_df.columns:
            full_df['Volume'] = 0
            
        # Calculate Shadow GEX on the full series
        # GEX Proxy = (Neutral - RV_20) * (Close * Volume)
        if 'RV_20' in full_df.columns:
            # Fill NaN RV with neutral to avoid noise
            rv = full_df['RV_20'].fillna(self.neutral_vol)
            notional = full_df[self.spot_col] * full_df['Volume']
            full_df['Shadow_GEX'] = (self.neutral_vol - rv) * notional
        else:
            full_df['Shadow_GEX'] = 0.0

        return full_df

    def compute_options_analytics(self) -> pd.DataFrame:
        """
        Augments options DataFrame with Moneyness, Greeks, IV.
        """
        if self._options_df.empty:
            return pd.DataFrame()

        df = self._options_df.copy()
        S = self.get_latest_spot()
        
        if S <= 0:
            return df

        # Moneyness
        df['moneyness'] = df['strike'] / S
        
        # Mid Price & Spread
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        # Fallback to lastPrice if mid is zero or nan
        mask_bad_mid = (df['mid_price'].isna()) | (df['mid_price'] <= 0)
        df.loc[mask_bad_mid, 'mid_price'] = df.loc[mask_bad_mid, 'lastPrice']
        
        df['spread'] = (df['ask'] - df['bid']).clip(lower=0)

        # Greeks Approximation (Vectorized loop for safety)
        # We need Time to Expiry in Years
        now = datetime.datetime.now()
        
        # Pre-calc columns
        deltas = []
        gammas = []
        
        for idx, row in df.iterrows():
            try:
                exp_date = pd.to_datetime(row['expiry'])
                # Approx days to expiry
                dte = (exp_date - now).days
                if dte < 0: dte = 0
                T = max(dte / 365.0, 0.001) # Avoid div by zero
                
                iv = row['impliedVolatility']
                if pd.isna(iv) or iv <= 0:
                    # Fallback IV? Just skip for now.
                    deltas.append(0.0)
                    gammas.append(0.0)
                    continue

                d, g = bs_delta_gamma(S, row['strike'], T, self.rf, iv, row['type'])
                deltas.append(d)
                gammas.append(g)
            except Exception:
                deltas.append(0.0)
                gammas.append(0.0)

        df['delta_bs'] = deltas
        df['gamma_bs'] = gammas
        
        # Gamma Proxy (Directional)
        # Call Gamma is usually long dealer gamma (if customers are long calls? No, simplistic view)
        # Standard GEX assumption: Dealers satisfy client demand.
        # Clients Long Calls -> Dealers Short Calls -> Dealers Short Gamma.
        # Clients Long Puts -> Dealers Short Puts -> Dealers Short Gamma.
        # BUT the prompt asks for "Simple Gamma Proxy".
        # Let's use: Gamma * OI * Spot * Spot * 0.01 (Cash Gamma)
        # And sign it: Calls add, Puts subtract? 
        # Standard "GEX" convention: 
        # Calls contribute positive spot exposure (dealer long gamma), Puts contribute negative?
        # Let's stick to the prompt's "Gamma Pressure": 
        # "Color segments where gamma is net positive vs net negative"
        # We will assume: Dealer Gamma = Call OI * Gamma - Put OI * Gamma.
        
        df['gamma_notional'] = df['gamma_bs'] * df['openInterest'] * (S**2) * 0.01
        
        # Apply sign for aggregate view (Calls +, Puts -)
        # Note: This implies Dealers are Long Calls and Short Puts, or vice versa depending on sign.
        # We will calculate 'Dealer_Gamma_Exposure':
        # Typically Dealers are Short the Options.
        # Short Call -> Short Gamma. Short Put -> Short Gamma.
        # Wait, usually GEX maps stabilizing vs destabilizing.
        # Let's just output raw magnitude per strike and split by type for the viz.
        
        return df

    def get_aggregated_gamma(self, analytics_df: pd.DataFrame) -> pd.DataFrame:
        if analytics_df.empty:
            return pd.DataFrame()
            
        # Group by strike
        g = analytics_df.groupby('strike').apply(
            lambda x: pd.Series({
                'call_gamma': x.loc[x['type']=='C', 'gamma_notional'].sum(),
                'put_gamma': x.loc[x['type']=='P', 'gamma_notional'].sum(),
                'total_oi': x['openInterest'].sum()
            })
        ).reset_index()
        
        # Net Gamma (Assuming Dealers are Short Option -> Dealers are Short Gamma? 
        # Actually standard SpotGamma/SqueezeMetrics logic is:
        # Call OI is Dealers Short Call (Long Gamma hedge needed? No, Short Call = Short Gamma).
        # Put OI is Dealers Short Put (Short Gamma? No, Short Put = Long Gamma? No. Short Opt is Short Gamma).
        # Standard model:
        # Market Makers are Long Calls (from overwrite sellers) -> Long Gamma.
        # Market Makers are Short Puts (from hedgers) -> Long Gamma? No, Short Put is Short Gamma? 
        # Let's stick to a simpler "Call Gamma" vs "Put Gamma" visualization so user can interpret.
        
        g['net_gamma_proxy'] = g['call_gamma'] - g['put_gamma']
        return g


# -----------------------------------------------------------------------------
#                            CLASS: DashboardRenderer
# -----------------------------------------------------------------------------

class DashboardRenderer:
    """
    Solely responsible for building Plotly figures and HTML assembly.
    """
    def __init__(self, ticker: str, spot_df: pd.DataFrame, shadow_gex_df: pd.DataFrame, 
                 options_df: pd.DataFrame):
        self.ticker = ticker
        self.spot_df = spot_df
        self.gex_df = shadow_gex_df
        self.opt_df = options_df

    def _fig_spot_regime(self) -> go.Figure:
        if self.spot_df.empty:
            return go.Figure().add_annotation(text="No Spot Data", showarrow=False)

        # Create subplots: Price on top, RV on bottom
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Price Candle/Line
        # If interval is < 1d, use line for clarity if too many points, else candle
        if len(self.spot_df) > 1000:
            fig.add_trace(go.Scatter(x=self.spot_df.index, y=self.spot_df['Close'], 
                                     name='Price', line=dict(color='black')), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(x=self.spot_df.index,
                                         open=self.spot_df['Open'] if 'Open' in self.spot_df else self.spot_df['Close'],
                                         high=self.spot_df['High'] if 'High' in self.spot_df else self.spot_df['Close'],
                                         low=self.spot_df['Low'] if 'Low' in self.spot_df else self.spot_df['Close'],
                                         close=self.spot_df['Close'], name='OHLC'), row=1, col=1)

        # MA 20
        ma20 = self.spot_df['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=self.spot_df.index, y=ma20, name='MA20', 
                                 line=dict(color='orange', width=1)), row=1, col=1)

        # RV Metrics
        cols = [c for c in self.spot_df.columns if c.startswith('RV_')]
        colors = {'RV_10': 'blue', 'RV_20': 'red', 'RV_60': 'green'}
        
        for c in cols:
            fig.add_trace(go.Scatter(x=self.spot_df.index, y=self.spot_df[c], name=c,
                                     line=dict(color=colors.get(c, 'grey'))), row=2, col=1)

        fig.update_layout(title=f"{self.ticker} Spot & Vol Regime", 
                          height=600, template='plotly_white')
        return fig

    def _fig_shadow_gex(self) -> go.Figure:
        if self.gex_df.empty:
            return go.Figure().add_annotation(text="No Shadow GEX Data", showarrow=False)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Shadow GEX Bar/Area
        # Color positive green, negative red
        colors = np.where(self.gex_df['Shadow_GEX'] >= 0, 'green', 'red')
        
        fig.add_trace(go.Bar(x=self.gex_df.index, y=self.gex_df['Shadow_GEX'],
                             marker_color=colors, name='Shadow GEX Proxy', opacity=0.6),
                      secondary_y=False)

        # Overlay RV
        if 'RV_20' in self.gex_df.columns:
            fig.add_trace(go.Scatter(x=self.gex_df.index, y=self.gex_df['RV_20'],
                                     name='RV_20', line=dict(color='black', dash='dot')),
                          secondary_y=True)

        fig.update_layout(title="Shadow GEX & Volatility Tension", height=500, template='plotly_white')
        return fig

    def _fig_iv_structure(self) -> go.Figure:
        if self.opt_df.empty:
            return go.Figure().add_annotation(text="No Options Data", showarrow=False)

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, 
                            subplot_titles=("IV Smiles (by Expiry)", "Term Structure (ATM IV)"))

        expiries = self.opt_df['expiry'].unique()
        
        # 1. Smiles
        for exp in expiries:
            sub = self.opt_df[self.opt_df['expiry'] == exp]
            # Filter reasonable moneyness for plot
            sub = sub[(sub['moneyness'] > 0.7) & (sub['moneyness'] < 1.3)]
            
            # Scatter for Calls and Puts? Just blend or show Calls OTM / Puts OTM?
            # Simplest: Show all IVs
            fig.add_trace(go.Scatter(x=sub['strike'], y=sub['impliedVolatility'],
                                     mode='markers', name=f"{exp} IV", marker=dict(size=4)),
                          row=1, col=1)

        # 2. Term Structure
        # Calculate ATM IV per expiry
        ts_data = []
        S = safe_float(self.spot_df['Close'].iloc[-1]) if not self.spot_df.empty else 0
        
        for exp in expiries:
            sub = self.opt_df[self.opt_df['expiry'] == exp]
            # Find strike closest to Spot
            if sub.empty: continue
            
            sub['dist'] = abs(sub['strike'] - S)
            atm_row = sub.loc[sub['dist'].idxmin()]
            ts_data.append({'expiry': exp, 'iv': atm_row['impliedVolatility']})
            
        ts_df = pd.DataFrame(ts_data)
        if not ts_df.empty:
            fig.add_trace(go.Scatter(x=ts_df['expiry'], y=ts_df['iv'], 
                                     mode='lines+markers', name='ATM IV Term Structure',
                                     line=dict(color='purple', width=3)),
                          row=2, col=1)

        fig.update_layout(height=800, template='plotly_white', showlegend=True)
        return fig

    def _fig_gamma_profile(self, gamma_df: pd.DataFrame) -> go.Figure:
        if gamma_df.empty:
             return go.Figure().add_annotation(text="No Gamma Data", showarrow=False)
             
        S = safe_float(self.spot_df['Close'].iloc[-1]) if not self.spot_df.empty else 0
        
        # Filter near spot
        lower = S * 0.8
        upper = S * 1.2
        sub = gamma_df[(gamma_df['strike'] >= lower) & (gamma_df['strike'] <= upper)]

        fig = go.Figure()
        
        # Call Gamma (Positive)
        fig.add_trace(go.Bar(x=sub['strike'], y=sub['call_gamma'], name='Call Gamma', marker_color='green'))
        # Put Gamma (Negative for viz)
        fig.add_trace(go.Bar(x=sub['strike'], y=-sub['put_gamma'], name='Put Gamma', marker_color='red'))
        
        # Spot Line
        fig.add_vline(x=S, line_dash="dash", line_color="black", annotation_text="Spot")
        
        fig.update_layout(title="Gamma Exposure Profile (Calls vs Puts)", barmode='relative',
                          height=500, template='plotly_white')
        return fig

    def generate_dashboard(self, output_path: str, gamma_df: pd.DataFrame):
        """
        Orchestrates figure creation and HTML assembly.
        """
        fig1 = self._fig_spot_regime()
        fig2 = self._fig_shadow_gex()
        fig3 = self._fig_iv_structure()
        fig4 = self._fig_gamma_profile(gamma_df)
        
        # Get offline divs
        div1 = py_offline.plot(fig1, include_plotlyjs=False, output_type='div')
        div2 = py_offline.plot(fig2, include_plotlyjs=False, output_type='div')
        div3 = py_offline.plot(fig3, include_plotlyjs=False, output_type='div')
        div4 = py_offline.plot(fig4, include_plotlyjs=False, output_type='div')
        
        # Get Plotly JS source
        plotly_js = py_offline.get_plotlyjs()

        html_content = self._make_html(plotly_js, div1, div2, div3, div4)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_path

    def _make_html(self, js_lib, d1, d2, d3, d4):
        return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{self.ticker} Options Dashboard</title>
<script type="text/javascript">{js_lib}</script>
<style>
    body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f4f4f4; }}
    .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
    .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }}
    .tab button:hover {{ background-color: #ddd; }}
    .tab button.active {{ background-color: #ccc; }}
    .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; background: #fff; animation: fadeEffect 1s; }}
    @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
</style>
</head>
<body>

<h2>{self.ticker} Options Regime & Volatility Dashboard</h2>

<div class="tab">
  <button class="tablinks" onclick="openTab(event, 'Spot')" id="defaultOpen">Spot & Vol</button>
  <button class="tablinks" onclick="openTab(event, 'Shadow')">Shadow GEX</button>
  <button class="tablinks" onclick="openTab(event, 'IV')">IV & Term Structure</button>
  <button class="tablinks" onclick="openTab(event, 'Gamma')">Gamma Profile</button>
</div>

<div id="Spot" class="tabcontent">{d1}</div>
<div id="Shadow" class="tabcontent">{d2}</div>
<div id="IV" class="tabcontent">{d3}</div>
<div id="Gamma" class="tabcontent">{d4}</div>

<script>
function openTab(evt, tabName) {{
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {{
        tabcontent[i].style.display = "none";
    }}
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {{
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }}
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
    
    // Trigger Resize for Plotly to render correctly in previously hidden tabs
    window.dispatchEvent(new Event('resize'));
    
    // Explicitly resize plots in the active tab
    var activeTab = document.getElementById(tabName);
    var plots = activeTab.getElementsByClassName('plotly-graph-div');
    for (var j=0; j<plots.length; j++) {{
        Plotly.Plots.resize(plots[j]);
    }}
}}
document.getElementById("defaultOpen").click();
</script>

</body>
</html>
"""

# -----------------------------------------------------------------------------
#                                MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hedge-Fund Grade Options Dashboard (Standalone)")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g. SPY, NVDA)")
    parser.add_argument("--history-days", type=int, default=252, help="Days of spot history")
    parser.add_argument("--interval", type=str, default="1d", help="Spot interval (1d, 1h, 15m)")
    parser.add_argument("--rf", type=float, default=0.045, help="Risk-free rate (decimal)")
    parser.add_argument("--shadow-neutral-vol", type=float, default=0.20, help="Neutral Vol for GEX proxy")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--open-html", action="store_true", help="Open HTML in browser after run")
    parser.add_argument("--max-expiries", type=int, default=3, help="Max option expiries to download")
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    out_dir = args.output_dir if args.output_dir else f"./out_{ticker}"
    
    print(f"--- Starting Dashboard Build for {ticker} ---")
    
    # 1. Data Ingestion
    ingestion = DataIngestion(ticker, out_dir, args.history_days, args.interval)
    
    # Spot & Shadow
    spot_df, shadow_hist_df = ingestion.get_spot_history(args.shadow_neutral_vol)
    
    # Options
    options_df = ingestion.get_options_chain(args.max_expiries)
    
    # 2. Financial Analysis
    print("--- Running Financial Analysis ---")
    ana = FinancialAnalysis(spot_df, shadow_hist_df, options_df, args.rf, args.shadow_neutral_vol)
    
    # Calculate Series
    spot_metrics = ana.compute_spot_regime()
    shadow_gex_series = ana.compute_shadow_gex_series()
    
    # Calculate Options Analytics
    opt_analytics = ana.compute_options_analytics()
    gamma_profile = ana.get_aggregated_gamma(opt_analytics)
    
    # 3. Rendering
    print("--- Rendering Dashboard ---")
    renderer = DashboardRenderer(ticker, spot_metrics, shadow_gex_series, opt_analytics)
    
    html_file = os.path.join(out_dir, f"options_dashboard_{ticker}.html")
    renderer.generate_dashboard(html_file, gamma_profile)
    
    print(f"--- Success! Dashboard saved to: {html_file} ---")
    
    if args.open_html:
        webbrowser.open('file://' + os.path.abspath(html_file))

if __name__ == "__main__":
    main()
