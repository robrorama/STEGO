# SCRIPTNAME: ok.05.intraday.volatility.drift.v3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import datetime
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as po
import scipy.stats as si
import yfinance as yf

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("QuantDashboard")
warnings.filterwarnings("ignore")  # Suppress yfinance/pandas warnings for cleaner CLI

# ------------------------------------------------------------------------------
# 1. DataIngestion Class
# ------------------------------------------------------------------------------
class DataIngestion:
    """
    Handles strict disk-first IO, path management, and data sanitization.
    """

    def __init__(self, output_dir: str, lookback_years: int):
        self.output_dir = Path(output_dir)
        self.prices_dir = self.output_dir / "prices"
        self.options_dir = self.output_dir / "options"
        self.lookback_years = lookback_years

        # Ensure directories exist
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        self.options_dir.mkdir(parents=True, exist_ok=True)

    def get_prices(self, ticker: str) -> pd.DataFrame:
        """
        Disk-first pipeline for underlying OHLCV data.
        Performs shadow backfill if historical data is insufficient.
        """
        file_path = self.prices_dir / f"{ticker}.csv"
        start_date = datetime.datetime.now() - datetime.timedelta(days=self.lookback_years * 365)
        
        # 1. Try reading from disk
        if file_path.exists():
            logger.info(f"[{ticker}] Found price data on disk.")
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = self._sanitize_df(df)
                
                # Shadow Backfill Check: Is the data old enough?
                if df.index.min() > start_date + datetime.timedelta(days=10):
                    logger.info(f"[{ticker}] History too short. Triggering shadow backfill.")
                    return self._download_and_save_prices(ticker, start_date, file_path)
                
                # Check if data is stale (older than 2 days)
                if df.index.max() < datetime.datetime.now() - datetime.timedelta(days=2):
                     logger.info(f"[{ticker}] Data stale. Updating...")
                     return self._download_and_save_prices(ticker, start_date, file_path)
                     
                return df
            except Exception as e:
                logger.error(f"[{ticker}] Error reading CSV: {e}. Re-downloading.")
        
        # 2. Fallback to download
        return self._download_and_save_prices(ticker, start_date, file_path)

    def _download_and_save_prices(self, ticker: str, start_date: datetime.datetime, file_path: Path) -> pd.DataFrame:
        logger.info(f"[{ticker}] Downloading OHLCV data...")
        time.sleep(1.0)  # Rate limit respect
        
        # yfinance download
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
        
        # Sanitize immediately
        df_clean = self._sanitize_df(df)
        
        # Save to disk
        df_clean.to_csv(file_path)
        return df_clean

    def get_options_snapshot(self, ticker: str) -> pd.DataFrame:
        """
        Disk-first pipeline for options chains. 
        Reuses 'today's' snapshot if available, otherwise downloads current chain.
        """
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        filename = f"{ticker}_options_{today_str}.csv"
        file_path = self.options_dir / filename

        if file_path.exists():
            logger.info(f"[{ticker}] Found today's options snapshot on disk.")
            return pd.read_csv(file_path)

        logger.info(f"[{ticker}] Downloading options chain...")
        time.sleep(1.0) # Rate limit respect
        
        try:
            yf_ticker = yf.Ticker(ticker)
            expirations = yf_ticker.options
            if not expirations:
                logger.warning(f"[{ticker}] No expirations found.")
                return pd.DataFrame()

            all_opts = []
            # Grab all expiries to ensure we find a valid front-month
            # Limiting to first 6 to save time/bandwidth for this example, or all if needed.
            for expiry in expirations[:6]: 
                try:
                    chain = yf_ticker.option_chain(expiry)
                    calls = chain.calls.copy()
                    calls['type'] = 'call'
                    puts = chain.puts.copy()
                    puts['type'] = 'put'
                    
                    combined = pd.concat([calls, puts])
                    combined['expiry'] = expiry
                    all_opts.append(combined)
                    time.sleep(0.5) # Gentle staggering
                except Exception as e:
                    logger.warning(f"Failed to fetch expiry {expiry} for {ticker}: {e}")

            if not all_opts:
                return pd.DataFrame()

            full_chain = pd.concat(all_opts, ignore_index=True)
            
            # Basic cleaning for CSV storage
            cols_to_keep = ['expiry', 'strike', 'type', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
            # Intersection to avoid errors if yfinance changes schema
            cols_present = [c for c in cols_to_keep if c in full_chain.columns]
            full_chain = full_chain[cols_present]
            
            full_chain.to_csv(file_path, index=False)
            return full_chain

        except Exception as e:
            logger.error(f"[{ticker}] Options download failed: {e}")
            return pd.DataFrame()

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal Fixer for yfinance DataFrame quirks (MultiIndex, Timezones, etc).
        """
        if df.empty:
            return df

        # 1. Handle MultiIndex Columns
        if isinstance(df.columns, pd.MultiIndex):
            # Check if Level 0 contains OHLCV or Ticker
            # Usually yfinance gives (Price, Ticker) -> Level 0 is Price
            # But sometimes if group_by='ticker', it's (Ticker, Price)
            
            # Heuristic: Look for 'Close' in level 0
            if 'Close' in df.columns.get_level_values(0):
                # Format is likely (Price, Ticker). We want to drop the ticker level if it exists.
                df.columns = df.columns.droplevel(1)
            elif 'Close' in df.columns.get_level_values(1):
                # Format is likely (Ticker, Price). Swap levels then drop ticker.
                df = df.swaplevel(0, 1, axis=1)
                df.columns = df.columns.droplevel(1)
        
        # 2. Flatten Columns (Just in case duplicate names or remaining complexity)
        df.columns = [str(c).strip() for c in df.columns]
        
        # 3. Index Normalization
        df.index = pd.to_datetime(df.index).normalize() # Remove time component
        if df.index.tz is not None:
             df.index = df.index.tz_localize(None)
        
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        # 4. Numeric Coercion
        cols_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for c in cols_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Fill naive gaps in price data
        df = df.ffill().bfill()

        return df


# ------------------------------------------------------------------------------
# 2. FinancialAnalysis Class (Updated with Smart Backfill)
# ------------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Computes IV, Skew, Regimes, and signals using sanitized data.
    Implements 'Smart Backfill' to approximate missing options history using Price Skew.
    """
    def __init__(self, risk_free_rate: float, output_dir: Path):
        self.r = risk_free_rate
        self.output_dir = output_dir

    def run_for_ticker(self, ticker: str, prices_df: pd.DataFrame, options_df: pd.DataFrame) -> dict:
        """
        Main runner for a single ticker. Returns a comprehensive dictionary of metrics.
        """
        if prices_df.empty:
            return None

        # --- A. Price Metrics ---
        # Returns
        prices_df['log_ret'] = np.log(prices_df['Close'] / prices_df['Close'].shift(1))
        
        # Realized Volatility (Annualized)
        prices_df['rv_21'] = prices_df['log_ret'].rolling(window=21).std() * np.sqrt(252)
        prices_df['rv_63'] = prices_df['log_ret'].rolling(window=63).std() * np.sqrt(252)
        
        spot = prices_df['Close'].iloc[-1]
        
        # --- B. Options Metrics ---
        iv_metrics = self._analyze_iv(ticker, spot, options_df, prices_df)
        
        # Pass prices_df to analyze_skew for the Proxy Backfill logic
        skew_metrics = self._analyze_skew(ticker, spot, options_df, iv_metrics['atm_iv_front'], prices_df)

        # --- C. Regimes ---
        regimes = self._determine_regimes(iv_metrics, skew_metrics)

        return {
            "ticker": ticker,
            "spot": spot,
            "history_df": prices_df,  # Needed for charts
            "iv_metrics": iv_metrics,
            "skew_metrics": skew_metrics,
            "regimes": regimes
        }

    def _analyze_iv(self, ticker: str, spot: float, options_df: pd.DataFrame, prices_df: pd.DataFrame) -> dict:
        """Computes ATM IV and manages IV history."""
        
        # 1. Select Front Month Expiry (>= 7 days out)
        if options_df.empty:
             return {"atm_iv_front": np.nan, "iv_percentile": np.nan, "iv_history": []}

        today = datetime.datetime.now()
        options_df['expiry_dt'] = pd.to_datetime(options_df['expiry'])
        options_df['days_to_expiry'] = (options_df['expiry_dt'] - today).dt.days

        # Filter for >= 7 days
        valid_opts = options_df[options_df['days_to_expiry'] >= 7]
        if valid_opts.empty:
            # Fallback to any future date
            valid_opts = options_df[options_df['days_to_expiry'] > 0]
        
        if valid_opts.empty:
             return {"atm_iv_front": np.nan, "iv_percentile": np.nan, "iv_history": []}

        # Pick nearest expiry
        target_expiry = valid_opts['expiry'].min()
        target_chain = valid_opts[valid_opts['expiry'] == target_expiry]
        
        # Time to expiry in years
        days_to_exp = target_chain['days_to_expiry'].iloc[0]
        T = max(days_to_exp / 365.0, 0.001)

        # 2. Find ATM Option
        # We generally look for the strike closest to Spot
        target_chain['dist'] = abs(target_chain['strike'] - spot)
        atm_row = target_chain.loc[target_chain['dist'].idxmin()]
        
        # Use simple logic: if impliedVol is present and non-zero, use it.
        # Otherwise, invert BS.
        atm_iv = atm_row.get('impliedVolatility', 0)
        
        # If yfinance returned bad IV (0 or NaN), we calculate it.
        if pd.isna(atm_iv) or atm_iv < 0.01:
            mid_price = (atm_row.get('bid', 0) + atm_row.get('ask', 0)) / 2
            if mid_price == 0: mid_price = atm_row['lastPrice']
            
            atm_iv = self._implied_vol_solver(
                price=mid_price, S=spot, K=atm_row['strike'], T=T, r=self.r, 
                option_type=atm_row['type']
            )

        # 3. IV History & Percentile
        # Load external IV history or create from RV proxy
        iv_hist_path = self.output_dir / f"iv_history_{ticker}.csv"
        
        # Construct Proxy History (Realized Vol) to fill gaps
        proxy_series = prices_df['rv_21'].rename("iv_30")

        if iv_hist_path.exists():
            iv_history = pd.read_csv(iv_hist_path, index_col=0, parse_dates=True)
            # Smart Backfill: Fill NaN/Missing dates in history with Proxy
            iv_history = iv_history.combine_first(pd.DataFrame(proxy_series))
        else:
            iv_history = pd.DataFrame(proxy_series)

        # Append today's REAL value
        if not pd.isna(atm_iv):
            today_ts = pd.Timestamp.now().normalize()
            iv_history.loc[today_ts, 'iv_30'] = atm_iv

        # Save back to disk
        iv_history.to_csv(iv_hist_path)

        # Compute Percentile
        current_val = iv_history['iv_30'].iloc[-1]
        # Use last 252 days for percentile ranking
        window = iv_history['iv_30'].tail(252).dropna()
        if not window.empty:
            rank = si.percentileofscore(window, current_val) / 100.0
        else:
            rank = 0.5 

        return {
            "atm_iv_front": atm_iv,
            "iv_history": iv_history,
            "iv_percentile": rank,
            "T": T,
            "expiry": target_expiry,
            "chain_df": target_chain
        }

    def _analyze_skew(self, ticker: str, spot: float, options_df: pd.DataFrame, atm_iv: float, prices_df: pd.DataFrame) -> dict:
        """
        Computes 25-Delta Risk Reversal.
        Uses 'Statistical Return Skewness' as a proxy for historical gaps.
        """
        # --- 1. Compute Real Skew (Today) ---
        rr_25d = np.nan
        
        if not options_df.empty and not pd.isna(atm_iv):
            today = datetime.datetime.now()
            options_df['expiry_dt'] = pd.to_datetime(options_df['expiry'])
            options_df['days_to_expiry'] = (options_df['expiry_dt'] - today).dt.days
            valid_opts = options_df[options_df['days_to_expiry'] >= 7]
            if valid_opts.empty: valid_opts = options_df[options_df['days_to_expiry'] > 0]
            
            if not valid_opts.empty:
                target_expiry = valid_opts['expiry'].min()
                chain = valid_opts[valid_opts['expiry'] == target_expiry].copy()
                
                T = chain['days_to_expiry'].iloc[0] / 365.0
                if T < 0.001: T = 0.001

                # Calculate Deltas
                def get_delta(row):
                    iv = row.get('impliedVolatility')
                    if pd.isna(iv) or iv < 0.01: iv = atm_iv # Fallback
                    return self._black_scholes_delta(spot, row['strike'], T, self.r, iv, row['type'])

                chain['calc_delta'] = chain.apply(get_delta, axis=1)

                calls = chain[chain['type'] == 'call']
                puts = chain[chain['type'] == 'put']

                if not calls.empty and not puts.empty:
                    call_25 = calls.iloc[(calls['calc_delta'] - 0.25).abs().argmin()]
                    put_25 = puts.iloc[(puts['calc_delta'] + 0.25).abs().argmin()]

                    iv_call = call_25.get('impliedVolatility')
                    if pd.isna(iv_call) or iv_call < 0.01: 
                        mid = (call_25['bid'] + call_25['ask'])/2 or call_25['lastPrice']
                        iv_call = self._implied_vol_solver(mid, spot, call_25['strike'], T, self.r, 'call')

                    iv_put = put_25.get('impliedVolatility')
                    if pd.isna(iv_put) or iv_put < 0.01:
                        mid = (put_25['bid'] + put_25['ask'])/2 or put_25['lastPrice']
                        iv_put = self._implied_vol_solver(mid, spot, put_25['strike'], T, self.r, 'put')

                    rr_25d = iv_call - iv_put

        # --- 2. Smart Backfill Logic ---
        
        # A. Create Proxy Series (Statistical Skew scaled to Vol Points)
        # 63-day rolling skew of returns
        rolling_stat_skew = prices_df['log_ret'].rolling(window=63).skew()
        
        # Scaling Heuristic: 
        # Realized Skew of -1.0 roughly maps to -0.05 Risk Reversal (Put Skew).
        # So we multiply stat_skew by 0.05.
        proxy_rr = rolling_stat_skew * 0.05
        
        # B. Load or Init History
        rr_hist_path = self.output_dir / f"rr_history_{ticker}.csv"
        
        if rr_hist_path.exists():
            rr_history = pd.read_csv(rr_hist_path, index_col=0, parse_dates=True)
        else:
            rr_history = pd.DataFrame(columns=['rr_25d'])

        # C. Combine: Fill gaps in Real History with Proxy
        # combine_first prefers "self" (rr_history), fills nulls from "other" (proxy_rr).
        rr_history = rr_history.combine_first(pd.DataFrame(proxy_rr, columns=['rr_25d']))

        # D. Append Today's Real Data (Overwriting proxy if present for today)
        today_ts = pd.Timestamp.now().normalize()
        if not pd.isna(rr_25d):
            rr_history.loc[today_ts, 'rr_25d'] = rr_25d
        
        # Trim to price history length (avoid infinite old tails)
        rr_history = rr_history[rr_history.index >= prices_df.index.min()]
        
        # Save
        rr_history.to_csv(rr_hist_path)

        # --- 3. Compute Percentiles ---
        current_rr = rr_history['rr_25d'].iloc[-1]
        
        # Use last 252 days for robust percentile
        history_window = rr_history['rr_25d'].tail(252).dropna()
        
        if not history_window.empty:
            rr_rank = si.percentileofscore(history_window, current_rr) / 100.0
            
            # Change Percentile
            daily_changes = history_window.diff().abs()
            current_change = abs(current_rr - rr_history['rr_25d'].iloc[-2]) if len(rr_history) > 1 else 0
            change_rank = si.percentileofscore(daily_changes.dropna(), current_change) / 100.0
        else:
            rr_rank = 0.5
            change_rank = 0.0

        return {
            "rr_25d": current_rr,
            "rr_history": rr_history,
            "rr_percentile": rr_rank,
            "rr_change_percentile": change_rank
        }

    def _determine_regimes(self, iv_m: dict, skew_m: dict) -> dict:
        iv_pct = iv_m.get('iv_percentile', 0.5)
        rr_pct = skew_m.get('rr_percentile', 0.5)
        rr_chg_pct = skew_m.get('rr_change_percentile', 0.0)
        rr_val = skew_m.get('rr_25d', 0)

        # IV Regime
        if pd.isna(iv_pct): iv_regime = "Unknown"
        elif iv_pct < 0.2: iv_regime = "IV_Very_Low"
        elif iv_pct > 0.8: iv_regime = "IV_Very_High"
        else: iv_regime = "IV_Normal"

        # Skew Regime
        if pd.isna(rr_pct): skew_regime = "Unknown"
        elif rr_val < 0 and rr_pct < 0.2: skew_regime = "Deep_Put_Skew"
        elif rr_val > 0 and rr_pct > 0.8: skew_regime = "Call_Skew"
        else: skew_regime = "Neutral_Skew"

        # Shock
        skew_shock_flag = False
        skew_shock_desc = ""
        if rr_chg_pct > 0.95:
            skew_shock_flag = True
            skew_shock_desc = "25Δ RR change > 95th %ile: SKEW SHOCK."

        # Combined
        combined = f"{iv_regime}_{skew_regime}"
        if skew_shock_flag:
            combined += "_SHOCK"

        return {
            "iv_regime": iv_regime,
            "skew_regime": skew_regime,
            "skew_shock_flag": skew_shock_flag,
            "skew_shock_desc": skew_shock_desc,
            "combined_regime": combined
        }

    # --- Math Helpers ---
    
    def _black_scholes_delta(self, S, K, T, r, sigma, option_type):
        if T <= 0 or sigma <= 0: return 0
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        if option_type == 'call':
            return si.norm.cdf(d1)
        else:
            return si.norm.cdf(d1) - 1

    def _implied_vol_solver(self, price, S, K, T, r, option_type):
        """Newton-Raphson Solver"""
        sigma = 0.3 # Initial guess
        for i in range(10):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                val = S*si.norm.cdf(d1) - K*np.exp(-r*T)*si.norm.cdf(d2)
                vega = S*np.sqrt(T)*si.norm.pdf(d1)
            else:
                val = K*np.exp(-r*T)*si.norm.cdf(-d2) - S*si.norm.cdf(-d1)
                vega = S*np.sqrt(T)*si.norm.pdf(d1)
            
            diff = val - price
            if abs(diff) < 1e-4:
                return sigma
            
            if abs(vega) < 1e-6: # Avoid div/0
                break
                
            sigma = sigma - diff/vega
            if sigma <= 0.01: sigma = 0.01 # Floor
            if sigma > 3.0: sigma = 3.0 # Cap
            
        return sigma


# ------------------------------------------------------------------------------
# 3. DashboardRenderer Class
# ------------------------------------------------------------------------------
class DashboardRenderer:
    """
    Generates offline HTML with embedded Plotly JS and tab logic.
    """
    def render(self, analysis_summary: dict, output_path: Path):
        
        # 1. Prepare Figures
        figs_html = {}
        
        # Cross-Ticker Comparison
        comp_fig = self._build_comparison_heatmap(analysis_summary)
        figs_html['comparison'] = po.plot(comp_fig, include_plotlyjs=False, output_type='div')

        # Per-Ticker Figures
        for ticker, data in analysis_summary.items():
            if data is None: continue
            
            # IV Chart
            iv_fig = self._build_iv_chart(data)
            figs_html[f"{ticker}_iv"] = po.plot(iv_fig, include_plotlyjs=False, output_type='div')
            
            # Skew Chart
            skew_fig = self._build_skew_chart(data)
            figs_html[f"{ticker}_skew"] = po.plot(skew_fig, include_plotlyjs=False, output_type='div')

        # 2. Build HTML Content
        plotly_js = po.get_plotlyjs()
        
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Quant Options Dashboard</title>
            <script>{plotly_js}</script>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #1e1e1e; color: #ddd; margin: 0; }}
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #2e2e2e; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-weight: bold; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #007bff; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 0.5s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
                .panel {{ background: #252525; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                .metric-box {{ display: inline-block; padding: 10px; background: #333; margin-right: 10px; border-radius: 4px; min-width: 120px; text-align: center; }}
                .metric-val {{ font-size: 1.2em; font-weight: bold; color: #00bcd4; }}
                .metric-label {{ font-size: 0.8em; color: #aaa; }}
                .alert {{ color: #ff5252; font-weight: bold; border: 1px solid #ff5252; padding: 5px; display: inline-block; margin-top: 5px; }}
                h2, h3 {{ color: #f0f0f0; }}
            </style>
        </head>
        <body>

        <div class="tab">
          <button class="tablinks" onclick="openTab(event, 'Comparison')" id="defaultOpen">Market Overview</button>
          {''.join([f'<button class="tablinks" onclick="openTab(event, \'{t}\')">{t}</button>' for t in analysis_summary.keys() if analysis_summary[t]])}
        </div>

        <div id="Comparison" class="tabcontent">
            <div class="panel">
                <h3>Cross-Asset Regime Heatmap</h3>
                {figs_html.get('comparison', 'No Data')}
            </div>
        </div>

        """

        # Add Ticker Tabs
        for ticker, data in analysis_summary.items():
            if data is None: continue
            
            iv_m = data['iv_metrics']
            sk_m = data['skew_metrics']
            reg = data['regimes']
            
            alert_html = f"<div class='alert'>{reg['skew_shock_desc']}</div>" if reg['skew_shock_flag'] else ""
            
            html_content += f"""
            <div id="{ticker}" class="tabcontent">
                <h2>{ticker} Analysis <span style="font-size:0.6em; color:#888">Spot: {data['spot']:.2f}</span></h2>
                <div class="panel">
                    <div class="metric-box"><div class="metric-val">{iv_m['atm_iv_front']:.1%}</div><div class="metric-label">ATM IV</div></div>
                    <div class="metric-box"><div class="metric-val">{iv_m['iv_percentile']:.0%}</div><div class="metric-label">IV %Rank</div></div>
                    <div class="metric-box"><div class="metric-val">{sk_m['rr_25d']:.1%}</div><div class="metric-label">25d Skew</div></div>
                    <div class="metric-box"><div class="metric-val">{sk_m['rr_percentile']:.0%}</div><div class="metric-label">Skew %Rank</div></div>
                    <br>{alert_html}
                </div>
                
                <div class="panel">
                    <h3>Implied Volatility Regime</h3>
                    <p>Regime: <b>{reg['iv_regime']}</b></p>
                    {figs_html.get(f"{ticker}_iv", "")}
                </div>
                
                <div class="panel">
                    <h3>Skew Structure (Risk Reversal)</h3>
                    <p>Regime: <b>{reg['skew_regime']}</b> | Change Intensity: {sk_m['rr_change_percentile']:.0%}</p>
                    {figs_html.get(f"{ticker}_skew", "")}
                </div>
            </div>
            """

        # JS for Tabs + Resize
        html_content += """
        <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            if (evt) evt.currentTarget.className += " active";
            
            // Trigger Plotly Resize
            window.dispatchEvent(new Event('resize'));
        }
        document.getElementById("defaultOpen").click();
        </script>
        </body>
        </html>
        """

        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Dashboard written to {output_path}")

    def _build_iv_chart(self, data):
        df = data['iv_metrics']['iv_history']
        if df.empty: return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['iv_30'], mode='lines', name='IV30 / Proxy', line=dict(color='#00e676')))
        
        # Add percentile bands
        high = df['iv_30'].quantile(0.8)
        low = df['iv_30'].quantile(0.2)
        
        fig.add_hline(y=high, line_dash="dash", line_color="red", annotation_text="80th %ile")
        fig.add_hline(y=low, line_dash="dash", line_color="green", annotation_text="20th %ile")
        
        fig.update_layout(
            title="IV History (vs Realized Proxy)", 
            template="plotly_dark", 
            height=350,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    def _build_skew_chart(self, data):
        df = data['skew_metrics']['rr_history']
        if df.empty or 'rr_25d' not in df.columns: return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df['rr_25d'], name='25d RR', marker_color='#29b6f6'))
        
        fig.update_layout(
            title="25-Delta Risk Reversal History (Smart Backfill)", 
            template="plotly_dark", 
            height=350,
            yaxis_title="Call IV - Put IV",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    def _build_comparison_heatmap(self, analysis):
        tickers = []
        iv_ranks = []
        skew_ranks = []
        skew_chg = []
        
        for t, d in analysis.items():
            if d is None: continue
            tickers.append(t)
            iv_ranks.append(d['iv_metrics']['iv_percentile'])
            skew_ranks.append(d['skew_metrics']['rr_percentile'])
            skew_chg.append(d['skew_metrics']['rr_change_percentile'])

        if not tickers: return go.Figure()

        z_data = [iv_ranks, skew_ranks, skew_chg]
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=tickers,
            y=['IV Rank', 'Skew Rank', 'Skew Change'],
            colorscale='RdBu_r',
            zmin=0, zmax=1,
            text=[[f"{v:.0%}" for v in row] for row in z_data],
            texttemplate="%{text}",
            showscale=True
        ))
        
        fig.update_layout(
            title="Market Regime Scanner",
            template="plotly_dark",
            height=400
        )
        return fig


# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Standalone Quantitative Options Dashboard")
    parser.add_argument("--tickers", type=str, default="SPY,QQQ,IWM", help="Comma-separated tickers")
    parser.add_argument("--output-dir", type=str, default="./market_data", help="Data directory")
    parser.add_argument("--lookback", type=int, default=1, help="Years of history")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk free rate (0.04 = 4%)")
    
    args = parser.parse_args()
    
    ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    
    # 1. Instantiate Modules
    ingestor = DataIngestion(args.output_dir, args.lookback)
    analyzer = FinancialAnalysis(args.risk_free_rate, ingestor.output_dir)
    renderer = DashboardRenderer()
    
    summary_results = {}

    print("=== Starting Quantitative Pipeline ===")
    
    # 2. Execution Loop
    for ticker in ticker_list:
        try:
            print(f"Processing {ticker}...")
            
            # Step A: Ingest
            prices = ingestor.get_prices(ticker)
            options = ingestor.get_options_snapshot(ticker)
            
            # Step B: Analyze
            result = analyzer.run_for_ticker(ticker, prices, options)
            summary_results[ticker] = result
            
        except Exception as e:
            logger.error(f"Critical failure for {ticker}: {e}", exc_info=True)
            summary_results[ticker] = None

    # 3. Render
    output_html = Path(args.output_dir) / "dashboard.html"
    print(f"Rendering dashboard to {output_html}...")
    renderer.render(summary_results, output_html)
    
    print("=== Done. Run 'open {}' to view. ===".format(output_html))

if __name__ == "__main__":
    main()
