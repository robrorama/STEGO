"""
vol_surface_pca.py

Senior Quantitative Developer System
------------------------------------
A standalone, production-ready script for Implied Volatility Surface Analysis,
Principal Component Analysis (PCA) of Term Structures, and "Sticky-Delta" vs
"Sticky-Strike" shock modeling.

Architecture:
    1. DataIngestion: Disk-First pipeline with rate limiting and strict sanitization.
    2. FinancialAnalysis: Pure math core for Black-Scholes, Surface Metrics, and PCA.
    3. DashboardRenderer: Offline-ready Plotly HTML generation.

Usage:
    python vol_surface_pca.py --tickers SPY QQQ --output-dir ./market_data --lookback 1

Dependencies:
    pip install pandas numpy scipy yfinance scikit-learn plotly
"""

import os
import time
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress pandas/yfinance future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)


# -----------------------------------------------------------------------------
# 1. Data Ingestion (Disk-First Pipeline)
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Handles IO, API interactions, Rate Limiting, and Data Sanitization.
    Enforces a strict Disk-First policy.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created data directory: {self.output_dir}")

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal Fixer for yfinance multi-index and typing bugs.
        """
        if df.empty:
            return df

        # Fix MultiIndex Swap: If 'Close' is in level 1, swap levels
        if isinstance(df.columns, pd.MultiIndex):
            # Heuristic: Check if 'Close' or 'Adj Close' is in the second level
            if 'Close' in df.columns.get_level_values(1):
                df = df.swaplevel(0, 1, axis=1)
                logger.debug("Swapped MultiIndex levels.")

            # Flatten columns to Snake Case or Ticket_Metric
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Join with underscore and clean
                    c = f"{col[0]}_{col[1]}".strip()
                else:
                    c = str(col)
                new_cols.append(c.replace(" ", "_").lower())
            df.columns = new_cols

        # Ensure Index is Datetime (tz-naive)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Force Numeric Coercion on known price columns if they exist (flattened)
        for col in df.columns:
            if any(x in col for x in ['close', 'open', 'high', 'low', 'adj_close']):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_spot_data(self, ticker: str, lookback_years: float = 1.0) -> pd.DataFrame:
        """
        Retrieves historical spot price data.
        """
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")

        # 1. Check Disk
        if os.path.exists(file_path):
            logger.info(f"[{ticker}] Spot data found on disk. Loading...")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return self._sanitize_df(df)

        # 2. If Missing, Download
        logger.info(f"[{ticker}] Spot data missing. Downloading...")
        
        start_date = (datetime.now() - timedelta(days=int(lookback_years*365))).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, progress=False, group_by='column')
        
        # 3. Sanitize
        df = self._sanitize_df(df)

        # 4. Save
        df.to_csv(file_path)
        logger.info(f"[{ticker}] Spot data saved to disk.")
        
        return df

    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves current options chain. 
        Note: yfinance options data is complex. We flatten it to a standardized CSV.
        """
        file_path = os.path.join(self.output_dir, f"{ticker}_options.csv")

        # 1. Check Disk
        if os.path.exists(file_path):
            logger.info(f"[{ticker}] Options chain found on disk. Loading...")
            return pd.read_csv(file_path)

        # 2. If Missing, Download
        logger.info(f"[{ticker}] Options chain missing. Downloading...")
        
        try:
            yf_ticker = yf.Ticker(ticker)
            expirations = yf_ticker.options
            
            all_options = []
            
            for exp in expirations:
                # STAGGER: Sleep between expiration calls is polite but usually unnecessary 
                # for single ticker objects in yfinance, but we add a tiny buffer.
                time.sleep(0.2) 
                
                try:
                    chain = yf_ticker.option_chain(exp)
                    calls = chain.calls
                    puts = chain.puts
                    
                    calls['type'] = 'call'
                    puts['type'] = 'put'
                    calls['expiration'] = exp
                    puts['expiration'] = exp
                    
                    all_options.append(pd.concat([calls, puts], sort=False))
                except Exception as e:
                    logger.warning(f"Failed to fetch {exp} for {ticker}: {e}")

            if not all_options:
                logger.error(f"No options data found for {ticker}")
                return pd.DataFrame()

            df_chain = pd.concat(all_options, ignore_index=True)
            
            # Simple Sanitization for CSV storage
            df_chain['lastTradeDate'] = pd.to_datetime(df_chain['lastTradeDate']).apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) else x)
            
            # 4. Save
            df_chain.to_csv(file_path, index=False)
            logger.info(f"[{ticker}] Options chain saved ({len(df_chain)} rows).")
            return df_chain

        except Exception as e:
            logger.error(f"Critical error fetching options for {ticker}: {e}")
            return pd.DataFrame()

    def process_tickers(self, tickers: List[str], lookback: float) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Master loop with enforced staggering.
        """
        data_store = {}
        for i, ticker in enumerate(tickers):
            if i > 0:
                logger.info("STAGGER: Sleeping 1s to respect API rate limits...")
                time.sleep(1)
            
            spot = self.get_spot_data(ticker, lookback)
            chain = self.get_options_chain(ticker)
            data_store[ticker] = (spot, chain)
            
        return data_store


# -----------------------------------------------------------------------------
# 2. Financial Analysis (Pure Math)
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Pure Logic. No API calls.
    Handles BS Math, Surface Metrics (RR/BF), PCA, and Shock Scenarios.
    """
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    # --- Black-Scholes Primitives ---
    def _d1(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0: return 0
        return (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def bs_call_delta(self, S, K, T, sigma):
        return norm.cdf(self._d1(S, K, T, sigma))

    def bs_put_delta(self, S, K, T, sigma):
        return norm.cdf(self._d1(S, K, T, sigma)) - 1.0

    def bs_strike_from_delta(self, target_delta, S, T, sigma, is_call=True):
        """
        Inverse Delta: Solves for K given a Delta.
        Uses Brent's method for root finding.
        """
        def obj_func(K_guess):
            if is_call:
                return self.bs_call_delta(S, K_guess, T, sigma) - target_delta
            else:
                return self.bs_put_delta(S, K_guess, T, sigma) - target_delta
        
        # Bound search space reasonably (0.1*Spot to 3.0*Spot)
        try:
            return brentq(obj_func, S * 0.1, S * 3.0)
        except ValueError:
            return np.nan

    # --- Surface Construction ---
    def build_surface_metrics(self, spot_price: float, chain_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates ATM Vol, RR25, BF25 for each expiration.
        """
        if chain_df.empty: return pd.DataFrame()

        # Convert Expiration to DTE (Days to Expiry) / T (Years)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        chain_df['expiration'] = pd.to_datetime(chain_df['expiration'])
        chain_df['days_to_expiry'] = (chain_df['expiration'] - today).dt.days
        chain_df['T'] = chain_df['days_to_expiry'] / 365.0
        
        # Filter stale data or weird expiries
        chain_df = chain_df[chain_df['days_to_expiry'] > 0].copy()
        
        metrics = []

        for exp_date, grp in chain_df.groupby('expiration'):
            T = grp['T'].iloc[0]
            dte = grp['days_to_expiry'].iloc[0]
            
            calls = grp[grp['type'] == 'call'].set_index('strike').sort_index()
            puts = grp[grp['type'] == 'put'].set_index('strike').sort_index()
            
            # 1. ATM Vol (Average IV of Call/Put closest to Spot)
            # Find closest strike to spot
            strikes = calls.index.values
            if len(strikes) == 0: continue
            
            atm_strike_idx = (np.abs(strikes - spot_price)).argmin()
            atm_strike = strikes[atm_strike_idx]
            
            try:
                iv_call_atm = calls.loc[atm_strike]['impliedVolatility']
                iv_put_atm = puts.loc[atm_strike]['impliedVolatility'] if atm_strike in puts.index else iv_call_atm
                atm_vol = (iv_call_atm + iv_put_atm) / 2.0
            except:
                continue

            # 2. Identify 25-Delta Strikes
            # We iterate strikes to find actual deltas, then interpolate or pick closest
            
            # Calculate Deltas for all calls/puts in this expiry to find the specific strikes
            # Using the ATM vol as a baseline proxy for delta calculation is standard for quick surface construction,
            # though using row-specific IV is better. Here we use the row IV provided by yfinance.
            
            call_deltas = []
            for k, row in calls.iterrows():
                d = self.bs_call_delta(spot_price, k, T, row['impliedVolatility'])
                call_deltas.append((k, d, row['impliedVolatility']))
            
            put_deltas = []
            for k, row in puts.iterrows():
                d = self.bs_put_delta(spot_price, k, T, row['impliedVolatility'])
                put_deltas.append((k, d, row['impliedVolatility']))
                
            df_calls = pd.DataFrame(call_deltas, columns=['K', 'Delta', 'IV'])
            df_puts = pd.DataFrame(put_deltas, columns=['K', 'Delta', 'IV'])
            
            # Find closest to 0.25 and -0.25
            try:
                c25_row = df_calls.iloc[(df_calls['Delta'] - 0.25).abs().argsort()[:1]]
                p25_row = df_puts.iloc[(df_puts['Delta'] - (-0.25)).abs().argsort()[:1]]
                
                if c25_row.empty or p25_row.empty: continue
                
                iv_call_25 = c25_row['IV'].values[0]
                iv_put_25 = p25_row['IV'].values[0]
                
                # 3. Calculate Metrics
                rr25 = iv_call_25 - iv_put_25
                bf25 = 0.5 * (iv_call_25 + iv_put_25) - atm_vol
                
                metrics.append({
                    'expiration': exp_date,
                    'T': T,
                    'days_to_expiry': dte,
                    'atm_vol': atm_vol,
                    'rr25': rr25,
                    'bf25': bf25
                })

            except Exception as e:
                continue
                
        return pd.DataFrame(metrics).sort_values('days_to_expiry')

    def run_pca(self, surface_df: pd.DataFrame):
        """
        Runs PCA on the [ATM, RR, BF] matrix.
        Returns explained variance and components.
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("Scikit-learn not found. Skipping PCA.")
            return None, None

        if surface_df.empty or len(surface_df) < 3:
            return None, None

        features = ['atm_vol', 'rr25', 'bf25']
        X = surface_df[features].dropna()
        
        if X.empty: return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=min(3, len(X)))
        pca.fit(X_scaled)

        explained_var = pca.explained_variance_ratio_
        components = pd.DataFrame(pca.components_, columns=features, index=[f'PC{i+1}' for i in range(len(explained_var))])
        
        return explained_var, components

    def calculate_shocks(self, spot_price: float, chain_df: pd.DataFrame, shock_pct: float = 0.01):
        """
        Sticky-Delta vs Sticky-Strike logic.
        Focuses on Mid-Tenor (closest to 90 days/3m).
        """
        if chain_df.empty: return None

        # Find Mid-Tenor
        chain_df['expiration'] = pd.to_datetime(chain_df['expiration'])
        today = datetime.now()
        chain_df['dte'] = (chain_df['expiration'] - today).dt.days
        
        # Filter for ~3 months (90 days)
        target_dte = 90
        unique_dtes = chain_df['dte'].unique()
        if len(unique_dtes) == 0: return None
        
        closest_dte = unique_dtes[(np.abs(unique_dtes - target_dte)).argmin()]
        
        # Get curve for this tenor
        curve = chain_df[(chain_df['dte'] == closest_dte) & (chain_df['type'] == 'call')].copy()
        # FIX: Filter out garbage data (IV is zero or effectively zero)
        curve = curve[curve['impliedVolatility'] > 0.001].copy()
        if curve.empty: return None

        # Current State
        T = closest_dte / 365.0
        curve['delta'] = curve.apply(lambda row: self.bs_call_delta(spot_price, row['strike'], T, row['impliedVolatility']), axis=1)
        
        # Scenario Setups
        spot_up = spot_price * (1 + shock_pct)
        spot_down = spot_price * (1 - shock_pct)
        
        results = []
        
        for idx, row in curve.iterrows():
            k_orig = row['strike']
            iv_orig = row['impliedVolatility']
            delta_orig = row['delta']
            
            # --- Sticky Strike ---
            # Vol stays same at Strike K, regardless of Spot move.
            # Effectively, IV(K, S_new) = IV(K, S_old)
            # We just plot the original smile line essentially.
            
            # --- Sticky Delta (Spot UP) ---
            # Vol stays same at Delta.
            # We need to find the NEW Strike K_new that gives 'delta_orig' given S_up.
            # Logic: We assume the volatility associated with that *moneyness/delta* moves with spot.
            
            # 1. Calculate new Strike for Spot Up that maintains original Delta using original Vol
            # Note: Strict Sticky Delta implies the volatility depends on moneyness (S/K).
            # If we fix Delta, we are essentially fixing moneyness.
            k_sticky_delta_up = self.bs_strike_from_delta(delta_orig, spot_up, T, iv_orig, is_call=True)
            
            # 2. Calculate new Strike for Spot Down
            k_sticky_delta_down = self.bs_strike_from_delta(delta_orig, spot_down, T, iv_orig, is_call=True)

            results.append({
                'strike': k_orig,
                'iv_orig': iv_orig,
                'delta': delta_orig,
                # For plotting Sticky Delta, we plot the point (K_new, IV_orig)
                'k_sd_up': k_sticky_delta_up,
                'k_sd_down': k_sticky_delta_down,
            })
            
        return pd.DataFrame(results), closest_dte


# -----------------------------------------------------------------------------
# 3. Dashboard Renderer (Offline Plotly)
# -----------------------------------------------------------------------------
class DashboardRenderer:
    """
    Generates static HTML files. No CDN dependencies.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        
        # Load Plotly JS for offline embedding
        import plotly.offline as py_offline
        self.plotly_js = py_offline.get_plotlyjs()

    def _wrap_html(self, ticker: str, figures: Dict[str, object], title_suffix: str) -> str:
        """
        Wraps Plotly divs in a clean HTML skeleton with resize fix.
        """
        import plotly.io as pio
        
        html_parts = [
            f"<html><head><title>{ticker} - {title_suffix}</title>",
            f"<script>{self.plotly_js}</script>",
            "<style>body{font-family: sans-serif; margin: 20px;} .chart-container{margin-bottom: 50px;}</style>",
            "</head><body>",
            f"<h1>{ticker} Analysis - {title_suffix}</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        for name, fig in figures.items():
            if fig is None: continue
            div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            html_parts.append(f"<h2>{name}</h2><div class='chart-container'>{div}</div>")

        # Tab Resize Fix
        html_parts.append("<script>window.dispatchEvent(new Event('resize'));</script>")
        html_parts.append("</body></html>")
        
        return "\n".join(html_parts)

    def generate_missing_data_report(self, ticker: str):
        path = os.path.join(self.output_dir, f"{ticker}_report.html")
        with open(path, "w") as f:
            f.write(f"<html><body><h1>Data Missing for {ticker}</h1><p>Please check logs or try again later.</p></body></html>")
        logger.warning(f"Generated missing data report for {ticker}")

    def create_dashboard(self, ticker: str, surface_df: pd.DataFrame, pca_res, shock_res):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if surface_df.empty:
            self.generate_missing_data_report(ticker)
            return

        figures = {}

        # 1. Term Structure & Skew
        fig_ts = make_subplots(rows=1, cols=3, subplot_titles=("Term Structure (ATM)", "Skew (RR25)", "Kurtosis (BF25)"))
        
        fig_ts.add_trace(go.Scatter(x=surface_df['days_to_expiry'], y=surface_df['atm_vol'], mode='lines+markers', name='ATM Vol'), row=1, col=1)
        fig_ts.add_trace(go.Scatter(x=surface_df['days_to_expiry'], y=surface_df['rr25'], mode='lines+markers', name='RR 25'), row=1, col=2)
        fig_ts.add_trace(go.Scatter(x=surface_df['days_to_expiry'], y=surface_df['bf25'], mode='lines+markers', name='BF 25'), row=1, col=3)
        
        fig_ts.update_layout(height=400, title_text="Volatility Surface Metrics")
        figures['Surface Metrics'] = fig_ts

        # 2. Scatter RR vs BF
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=surface_df['rr25'], y=surface_df['bf25'],
            mode='markers+text',
            text=surface_df['days_to_expiry'],
            textposition="top center",
            name='Tenors'
        ))
        fig_scatter.update_layout(title="Smile Geometry: Skew (RR) vs Kurtosis (BF)", xaxis_title="Risk Reversal (Skew)", yaxis_title="Butterfly (Kurtosis)")
        figures['Smile Geometry'] = fig_scatter

        # 3. PCA
        explained_var, components = pca_res
        if explained_var is not None:
            fig_pca = make_subplots(rows=1, cols=2, subplot_titles=("Explained Variance", "Factor Loadings (PC1)"))
            
            fig_pca.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained_var))], y=explained_var, name='Var Exp'), row=1, col=1)
            
            # Loadings for PC1 only for brevity
            fig_pca.add_trace(go.Bar(x=components.columns, y=components.iloc[0], name='PC1 Loadings'), row=1, col=2)
            
            fig_pca.update_layout(height=400, title_text="Principal Component Analysis")
            figures['PCA'] = fig_pca

        # 4. Shock Panel
        if shock_res:
            df_shock, dte_used = shock_res
            if not df_shock.empty:
                fig_shock = go.Figure()
                
                # Original Smile (Sticky Strike)
                fig_shock.add_trace(go.Scatter(
                    x=df_shock['strike'], y=df_shock['iv_orig'],
                    mode='lines+markers', name=f'Current / Sticky Strike (Spot)',
                    line=dict(dash='dash', color='gray')
                ))
                
                # Sticky Delta (Spot UP)
                # We plot IV_orig against the NEW strikes calculated for Spot Up
                fig_shock.add_trace(go.Scatter(
                    x=df_shock['k_sd_up'], y=df_shock['iv_orig'],
                    mode='lines', name='Sticky Delta (Spot +1%)',
                    line=dict(color='green')
                ))

                # Sticky Delta (Spot DOWN)
                fig_shock.add_trace(go.Scatter(
                    x=df_shock['k_sd_down'], y=df_shock['iv_orig'],
                    mode='lines', name='Sticky Delta (Spot -1%)',
                    line=dict(color='red')
                ))
                
                fig_shock.update_layout(
                    title=f"Shock Scenarios: {dte_used} DTE",
                    xaxis_title="Strike Price",
                    yaxis_title="Implied Volatility",
                    hovermode="x"
                )
                figures['Shock Analysis'] = fig_shock

        # Write to File
        html_content = self._wrap_html(ticker, figures, "Volatility Report")
        file_path = os.path.join(self.output_dir, f"{ticker}_analysis.html")
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"[{ticker}] Dashboard generated: {file_path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Volatility Surface & PCA Engine")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='List of tickers')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Data/HTML storage path')
    parser.add_argument('--lookback', type=float, default=1.0, help='Years of spot history')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate (decimal)')
    parser.add_argument('--shock-pct', type=float, default=0.01, help='Shock size (decimal, e.g., 0.01 for 1%)')
    
    args = parser.parse_args()

    # 1. Initialize Components
    ingestion = DataIngestion(args.output_dir)
    analyst = FinancialAnalysis(risk_free_rate=args.risk_free_rate)
    renderer = DashboardRenderer(args.output_dir)

    # 2. Data Loop
    logger.info("Starting Data Ingestion Pipeline...")
    raw_data = ingestion.process_tickers(args.tickers, args.lookback)

    # 3. Analysis & Rendering Loop
    for ticker, (spot_df, chain_df) in raw_data.items():
        logger.info(f"[{ticker}] Starting Financial Analysis...")
        
        if spot_df.empty or chain_df.empty:
            logger.error(f"[{ticker}] Insufficient data. Skipping analysis.")
            renderer.generate_missing_data_report(ticker)
            continue
            
        # Get latest spot price
        # Try to get the very last price, prioritizing Close
        try:
            # Handle flattened columns, looking for 'close'
            close_col = [c for c in spot_df.columns if 'close' in c.lower()][0]
            current_spot = spot_df[close_col].iloc[-1]
        except IndexError:
            logger.error(f"[{ticker}] Could not identify close price column.")
            continue

        # A. Build Surface
        surface_df = analyst.build_surface_metrics(current_spot, chain_df)
        
        # B. PCA
        pca_res = analyst.run_pca(surface_df)
        
        # C. Shocks
        shock_res = analyst.calculate_shocks(current_spot, chain_df, args.shock_pct)
        
        # D. Render
        renderer.create_dashboard(ticker, surface_df, pca_res, shock_res)

    logger.info("Batch Processing Complete.")

if __name__ == "__main__":
    main()
