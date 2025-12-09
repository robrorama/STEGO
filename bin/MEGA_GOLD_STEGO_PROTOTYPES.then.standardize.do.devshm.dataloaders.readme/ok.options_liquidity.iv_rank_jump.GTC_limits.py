"""
GTC Limit Recommendation & Execution Dashboard
==============================================
Role: Senior Quantitative Developer
Author: AI Assistant
Date: 2025-12-03
Context: Liquidity-aware and Volatility-aware execution logic for options.

Architecture:
    1. DataIngestion: Disk-first, yfinance-based, robust sanitization.
    2. FinancialAnalysis: Metric computation (IV Rank, Liquidity Scores), GTC limits.
    3. DashboardRenderer: Offline Plotly, Multi-tab HTML, Resize fixes.

Usage:
    python execution_dashboard.py --tickers SPY QQQ IBIT --output-dir ./market_data
"""

import argparse
import os
import time
import datetime
import logging
import warnings
import webbrowser  # Added for auto-opening browser
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from scipy import stats

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress pandas/future warnings for cleaner CLI output
warnings.simplefilter(action='ignore', category=FutureWarning)


# ==============================================================================
# 1. Data Ingestion Layer
# ==============================================================================

class DataIngestion:
    """
    Handles all IO and data retrieval.
    Responsibility: Disk-first pipeline, yfinance downloads, data sanitization.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal sanitizer for yfinance DataFrames.
        Handles MultiIndex level swapping, flattening, and type coercion.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df_clean = df.copy()

        # 1. Normalize MultiIndex Levels
        # yfinance can return (Price, Ticker) or (Ticker, Price).
        if isinstance(df_clean.columns, pd.MultiIndex):
            # Check if Level 0 contains price attributes
            level_0_vals = df_clean.columns.get_level_values(0).unique().tolist()
            common_attrs = {'Close', 'Adj Close', 'Open', 'High', 'Low', 'Volume'}
            
            # If attributes are in level 1, swap them to level 0
            # Heuristic: if level 0 has no overlap with common attributes but level 1 does
            l0_set = set(str(x) for x in level_0_vals)
            if not common_attrs.intersection(l0_set):
                 df_clean.columns = df_clean.columns.swaplevel(0, 1)

            # 2. Flatten MultiIndex
            # Format: Attribute_Ticker (e.g., Close_SPY)
            new_cols = []
            for col in df_clean.columns:
                if isinstance(col, tuple):
                    # col[0] is attribute, col[1] is ticker
                    attr = str(col[0]).strip()
                    tick = str(col[1]).strip()
                    if tick:
                        new_cols.append(f"{attr}_{tick}")
                    else:
                        new_cols.append(attr)
                else:
                    new_cols.append(str(col))
            df_clean.columns = new_cols

        # 3. Standardize Index
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = pd.to_datetime(df_clean.index)
        
        # Remove timezone info for consistency
        if df_clean.index.tz is not None:
            df_clean.index = df_clean.index.tz_localize(None)
            
        df_clean.sort_index(inplace=True)
        
        # 4. Coerce Numerics
        for col in df_clean.columns:
            # Skip if it's explicitly non-numeric (unlikely in yf price data but possible)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        return df_clean

    def get_underlying_history(self, ticker: str, lookback_years: float, interval: str) -> pd.DataFrame:
        """
        Disk-first retrieval of underlying price history.
        """
        filename = os.path.join(self.output_dir, f"{ticker}_prices.csv")
        
        # Try loading from disk
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                if not df.empty:
                    logger.info(f"Loaded {ticker} history from disk.")
                    return df
            except Exception as e:
                logger.warning(f"Failed to read {filename}, forcing redownload. Error: {e}")

        # Download if missing or empty
        return self._download_underlying(ticker, lookback_years, interval, filename)

    def _download_underlying(self, ticker: str, lookback_years: float, interval: str, filename: str) -> pd.DataFrame:
        logger.info(f"Downloading underlying history for {ticker}...")
        
        # Rate limit
        time.sleep(1.0)
        
        start_date = (datetime.datetime.now() - datetime.timedelta(days=int(lookback_years*365))).strftime('%Y-%m-%d')
        
        try:
            # group_by='column' usually puts Attributes in Level 0
            df = yf.download(
                [ticker], 
                start=start_date, 
                interval=interval, 
                group_by='column', 
                progress=False,
                threads=False # Disable threads to be gentle on API
            )
            
            sanitized_df = self._sanitize_df(df)
            
            # Save to disk
            sanitized_df.to_csv(filename)
            return sanitized_df
            
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {e}")
            return pd.DataFrame()

    def get_options_snapshot(self, ticker: str, min_dte: int, max_dte: int) -> pd.DataFrame:
        """
        Disk-first retrieval of options chain snapshot.
        """
        # We append today's date to filename to ensure freshness per day
        today_str = datetime.datetime.now().strftime('%Y%m%d')
        filename = os.path.join(self.output_dir, f"{ticker}_options_chain_{today_str}.csv")

        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                if not df.empty:
                    logger.info(f"Loaded {ticker} options chain from disk.")
                    return df
            except Exception as e:
                logger.warning(f"Failed to read {filename}, forcing redownload. Error: {e}")

        return self._download_options(ticker, min_dte, max_dte, filename)

    def _download_options(self, ticker: str, min_dte: int, max_dte: int, filename: str) -> pd.DataFrame:
        logger.info(f"Downloading options chain for {ticker} (DTE {min_dte}-{max_dte})...")
        
        time.sleep(1.0)
        yf_ticker = yf.Ticker(ticker)
        
        try:
            expirations = yf_ticker.options
        except Exception as e:
            logger.error(f"Could not fetch expirations for {ticker}: {e}")
            return pd.DataFrame()

        all_contracts = []
        today = datetime.datetime.now()

        for exp_date_str in expirations:
            exp_date = pd.to_datetime(exp_date_str)
            dte = (exp_date - today).days

            if min_dte <= dte <= max_dte:
                time.sleep(1.0) # Strict rate limiting per expiration
                try:
                    chain = yf_ticker.option_chain(exp_date_str)
                    
                    for opt_type, df in [('call', chain.calls), ('put', chain.puts)]:
                        df['optionType'] = opt_type
                        df['expirationDate'] = exp_date
                        df['dte'] = dte
                        df['underlying'] = ticker
                        all_contracts.append(df)
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch chain for {ticker} exp {exp_date_str}: {e}")
                    continue

        if not all_contracts:
            logger.warning(f"No options found for {ticker} within DTE range.")
            return pd.DataFrame()

        full_df = pd.concat(all_contracts, ignore_index=True)
        
        # Standardize basic columns
        rename_map = {
            'lastPrice': 'last_price',
            'openInterest': 'open_interest',
            'impliedVolatility': 'iv'
        }
        full_df.rename(columns=rename_map, inplace=True)
        
        # Handle bid/ask size if missing (YF often doesn't provide them)
        if 'bidSize' not in full_df.columns:
            full_df['bidSize'] = np.nan
        if 'askSize' not in full_df.columns:
            full_df['askSize'] = np.nan

        full_df.to_csv(filename, index=False)
        return full_df


# ==============================================================================
# 2. Financial Analysis Layer
# ==============================================================================

class FinancialAnalysis:
    """
    Core Logic. No IO. No Plotly.
    Calculates Liquidity, Volatility, Execution Risks, and GTC Recommendations.
    """
    
    def __init__(self, 
                 base_order_size: float, 
                 liquidity_threshold: float,
                 iv_jump_threshold: float,
                 risk_free_rate: float):
        self.base_order_size = base_order_size
        self.liquidity_threshold = liquidity_threshold
        self.iv_jump_threshold = iv_jump_threshold
        self.rfr = risk_free_rate
        
        # Risk Multiplier Configurations (Asset Specific)
        self.risk_configs = {
            'SPY': {'min': 1.5, 'max': 3.0},
            'QQQ': {'min': 1.75, 'max': 3.5},
            'IWM': {'min': 1.75, 'max': 3.5},
            'NVDA': {'min': 2.0, 'max': 6.0},
            'DEFAULT': {'min': 1.5, 'max': 4.0}
        }

    def _get_risk_config(self, ticker: str):
        return self.risk_configs.get(ticker, self.risk_configs['DEFAULT'])

    def _find_column(self, df_columns, target_sub: str, ticker: str) -> Optional[str]:
        """
        Robustly find a column given potential variable naming (e.g. Volume_IBIT vs Volume).
        """
        # 1. Exact ticker match (e.g., Volume_IBIT)
        target_exact = f"{target_sub}_{ticker}"
        if target_exact in df_columns:
            return target_exact
            
        # 2. Generic match (e.g., Volume)
        if target_sub in df_columns:
            return target_sub
            
        # 3. Fuzzy match (case insensitive)
        # e.g. "volume" or "Volume_ibit"
        for col in df_columns:
            if target_sub.lower() in str(col).lower():
                return col
                
        return None

    def analyze_ticker(self, 
                       ticker: str, 
                       df_price: pd.DataFrame, 
                       df_chain: pd.DataFrame) -> Dict[str, Any]:
        """
        Main analysis pipeline for a single ticker.
        """
        if df_price.empty or df_chain.empty:
            return {}

        # 1. Underlying Liquidity Analysis - Robust Column Detection
        cols = df_price.columns
        
        vol_col = self._find_column(cols, "Volume", ticker)
        close_col = self._find_column(cols, "Close", ticker)
        
        # Fallback for Close if only Adj Close exists
        if not close_col:
            close_col = self._find_column(cols, "Adj Close", ticker)

        if not vol_col or not close_col:
            logger.error(f"Missing required columns (Volume/Close) for {ticker}. Available: {cols.tolist()}")
            return {}

        # Calculate Rolling Volume Percentiles
        try:
            recent_vol = df_price[vol_col].iloc[-1]
            vol_history = df_price[vol_col].iloc[:-1] # exclude today
        except Exception as e:
            logger.error(f"Error accessing volume data for {ticker}: {e}")
            return {}
        
        if len(vol_history) > 30:
            ticker_liq_pct = stats.percentileofscore(vol_history.tail(30), recent_vol)
        else:
            ticker_liq_pct = 50.0 # Neutral default

        # 2. Volatility Analysis (IV Rank / Jump)
        df_price['log_ret'] = np.log(df_price[close_col] / df_price[close_col].shift(1))
        df_price['hv_30d'] = df_price['log_ret'].rolling(window=30).std() * np.sqrt(252) * 100
        
        # SAFEGUARD: Drop NaNs to calculate median properly
        hv_clean = df_price['hv_30d'].dropna()
        if not hv_clean.empty:
            median_hv_30d = hv_clean.rolling(window=252, min_periods=1).median().iloc[-1]
            if pd.isna(median_hv_30d):
                median_hv_30d = hv_clean.median() 
        else:
            median_hv_30d = 0.0
        
        # Identify Front-Month ATM IV
        valid_chain = df_chain[df_chain['dte'] >= 1].copy()
        if valid_chain.empty:
            return {}

        min_dte = valid_chain['dte'].min()
        front_chain = valid_chain[valid_chain['dte'] == min_dte]
        
        # Find ATM (Strike closest to spot)
        spot = df_price[close_col].iloc[-1]
        front_chain['dist'] = abs(front_chain['strike'] - spot)
        atm_contract = front_chain.loc[front_chain['dist'].idxmin()]
        
        # SAFEGUARD: Explicitly check for NaN in YF IV data
        raw_iv = atm_contract['iv']
        if pd.isna(raw_iv) or raw_iv is None:
             front_atm_iv = 0.0 # Force numeric
        else:
             front_atm_iv = raw_iv * 100 # Convert to percentage
        
        # Term Slope (Front vs Back)
        back_dte = valid_chain[valid_chain['dte'] > min_dte]['dte'].min()
        if np.isnan(back_dte):
             back_dte = min_dte
             back_atm_iv = front_atm_iv
        else:
             back_chain = valid_chain[valid_chain['dte'] == back_dte]
             back_chain['dist'] = abs(back_chain['strike'] - spot)
             
             back_atm_iv_raw = back_chain.loc[back_chain['dist'].idxmin()]['iv']
             if pd.isna(back_atm_iv_raw):
                 back_atm_iv = front_atm_iv
             else:
                 back_atm_iv = back_atm_iv_raw * 100
             
        term_slope = back_atm_iv - front_atm_iv
        
        # IV Jump: Current IV vs Historical Median Volatility
        iv_jump = front_atm_iv - median_hv_30d
        
        # IV Rank
        hv_history = df_price['hv_30d'].dropna().tail(252)
        if not hv_history.empty:
            iv_rank = stats.percentileofscore(hv_history, front_atm_iv)
        else:
            iv_rank = 50.0

        # 3. Contract-Level Analysis
        df_chain_processed = valid_chain.copy()
        
        # Fill NaNs for calculation safety
        df_chain_processed['bid'] = df_chain_processed['bid'].fillna(0)
        df_chain_processed['ask'] = df_chain_processed['ask'].fillna(0)
        df_chain_processed['volume'] = df_chain_processed['volume'].fillna(0)
        df_chain_processed['open_interest'] = df_chain_processed['open_interest'].fillna(0)
        
        # Metrics
        df_chain_processed['spread'] = df_chain_processed['ask'] - df_chain_processed['bid']
        
        # Mid Estimate (Size weighted if possible)
        def calc_mid(row):
            if row['bid'] > 0 and row['ask'] > 0:
                if row['bidSize'] > 0 and row['askSize'] > 0 and not np.isnan(row['bidSize']):
                     return (row['bid'] * row['askSize'] + row['ask'] * row['bidSize']) / (row['bidSize'] + row['askSize'])
                return 0.5 * (row['bid'] + row['ask'])
            return row['last_price']
            
        df_chain_processed['mid_est'] = df_chain_processed.apply(calc_mid, axis=1)
        # Handle zero mid
        df_chain_processed['mid_est'] = df_chain_processed['mid_est'].replace(0, 0.01) 
        
        df_chain_processed['relative_spread'] = df_chain_processed['spread'] / df_chain_processed['mid_est']

        # Liquidity Score per contract
        df_chain_processed['liq_score_raw'] = np.log1p(df_chain_processed['volume']) + 0.5 * np.log1p(df_chain_processed['open_interest'])
        
        # Calculate percentile within this ticker's chain
        df_chain_processed['liq_pct_contract'] = df_chain_processed['liq_score_raw'].rank(pct=True) * 100

        # 4. Execution Risk & Recommendations
        is_liq_poor = (ticker_liq_pct < self.liquidity_threshold)
        is_vol_stressed = (iv_jump > self.iv_jump_threshold)
        
        risk_config = self._get_risk_config(ticker)
        
        results = []
        for idx, row in df_chain_processed.iterrows():
            contract_liq_poor = row['liq_pct_contract'] < self.liquidity_threshold
            
            # Risk Score (0 to 1, higher is worse)
            w_liq = 0.5
            w_vol = 0.3
            w_spd = 0.2
            
            spd_norm = min(row['relative_spread'] / 0.20, 1.0)
            if pd.isna(spd_norm): spd_norm = 1.0
            
            liq_pct = row['liq_pct_contract']
            if pd.isna(liq_pct): liq_pct = 0.0
            
            risk_score = (w_liq * (1 - liq_pct/100.0)) + \
                         (w_vol * (1.0 if is_vol_stressed else 0.0)) + \
                         (w_spd * spd_norm)
                         
            widen_factor = np.interp(risk_score, [0, 1], [risk_config['min'], risk_config['max']])
            
            if not is_liq_poor and not is_vol_stressed and not contract_liq_poor:
                 limit_buy = row['mid_est'] - (0.25 * row['spread'])
                 limit_sell = row['mid_est'] + (0.25 * row['spread'])
                 rec_size = self.base_order_size
            else:
                 offset = widen_factor * 0.25 * row['spread']
                 limit_buy = row['mid_est'] - offset
                 limit_sell = row['mid_est'] + offset
                 
                 size_scalar = np.interp(risk_score, [0, 1], [1.0, 0.25])
                 rec_size = max(1, round(self.base_order_size * size_scalar, 0))

            do_not_trade = False
            if row['relative_spread'] > 0.50 or (row['liq_pct_contract'] < 5.0):
                do_not_trade = True
                rec_size = 0

            row['risk_score'] = round(risk_score, 2)
            row['rec_limit_buy'] = round(limit_buy, 2)
            row['rec_limit_sell'] = round(limit_sell, 2)
            row['rec_size'] = int(rec_size)
            row['do_not_trade'] = do_not_trade
            results.append(row)
            
        df_results = pd.DataFrame(results)
        
        # FINAL SANITIZATION: Force Valid Floats for scalar metrics
        summary = {
            'ticker': ticker,
            'spot': float(np.nan_to_num(spot, nan=0.0)),
            'ticker_liq_pct': float(np.nan_to_num(ticker_liq_pct, nan=50.0)),
            'front_atm_iv': float(np.nan_to_num(front_atm_iv, nan=0.0)),
            'back_atm_iv': float(np.nan_to_num(back_atm_iv, nan=0.0)),
            'term_slope': float(np.nan_to_num(term_slope, nan=0.0)),
            'iv_jump': float(np.nan_to_num(iv_jump, nan=0.0)),
            'iv_rank': float(np.nan_to_num(iv_rank, nan=50.0)),
            'contracts': df_results
        }
        return summary


# ==============================================================================
# 3. Dashboard Renderer Layer (Modified: Removed Overview Tab)
# ==============================================================================

class DashboardRenderer:
    """
    Visualization Logic.
    Generates offline Plotly HTML with custom JS for tab resizing.
    """
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.html_parts = []
        self.tabs_js = """
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
            evt.currentTarget.className += " active";
            
            // DISPATCH RESIZE EVENT TO FIX PLOTLY
            window.dispatchEvent(new Event('resize'));
        }
        // Open default tab
        document.addEventListener("DOMContentLoaded", function() {
           if(document.getElementById("defaultOpen")){
               document.getElementById("defaultOpen").click();
           }
        });
        </script>
        <style>
        body {font-family: Arial;}
        .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }
        .tab button:hover { background-color: #ddd; }
        .tab button.active { background-color: #ccc; }
        .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
        </style>
        """

    def generate_dashboard(self, analysis_results: List[Dict[str, Any]]):
        if not analysis_results:
            logger.warning("No results to render.")
            return

        # Get Plotly JS source
        plotly_js = py_offline.get_plotlyjs()
        
        tickers = [r['ticker'] for r in analysis_results]
        
        # Construct HTML
        html = f"<html><head><script>{plotly_js}</script>{self.tabs_js}</head><body>"
        html += "<h1>Quantitative Execution Dashboard</h1>"
        
        # Tab Buttons (Removed Overview)
        html += '<div class="tab">'
        
        # Set first ticker as default active tab
        for i, t in enumerate(tickers):
            default_id = 'id="defaultOpen"' if i == 0 else ""
            html += f'<button class="tablinks" onclick="openTab(event, \'{t}\')" {default_id}>{t}</button>'
        html += '</div>'

        # Per-Ticker Tab Content
        for res in analysis_results:
            ticker_div = self._render_ticker_tab(res)
            html += f'<div id="{res["ticker"]}" class="tabcontent">{ticker_div}</div>'

        html += "</body></html>"
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Dashboard saved to {self.output_path}")

    def _render_ticker_tab(self, res):
        ticker = res['ticker']
        df = res['contracts']
        
        # Layout: 2x2 grid basically
        
        # A. Liquidity vs Spread
        fig_liq = go.Figure(data=go.Scatter(
            x=df['liq_pct_contract'],
            y=df['relative_spread'],
            mode='markers',
            text=df['strike'].astype(str) + df['optionType'].str.upper() + " DTE:" + df['dte'].astype(str),
            marker=dict(
                color=df['risk_score'],
                colorscale='Viridis',
                colorbar=dict(title="Risk Score"),
                size=8
            )
        ))
        fig_liq.update_layout(title=f"{ticker}: Contract Liquidity vs Relative Spread",
                              xaxis_title="Contract Liquidity Percentile",
                              yaxis_title="Relative Spread")
        
        # B. Limit Offset Curve (Buy)
        # Filter for a specific expiry range to make it readable, e.g., front month
        min_dte = df['dte'].min()
        subset = df[df['dte'] == min_dte].sort_values('strike')
        
        fig_offset = go.Figure()
        
        # Buy Offset trace
        buy_offsets = subset['mid_est'] - subset['rec_limit_buy']
        fig_offset.add_trace(go.Scatter(x=subset['strike'], y=buy_offsets, name='Buy Offset', mode='lines+markers'))
        
        # Theoretical "Benign" Offset (0.25 * spread)
        benign_offset = 0.25 * subset['spread']
        fig_offset.add_trace(go.Scatter(x=subset['strike'], y=benign_offset, name='Baseline (Benign)', line=dict(dash='dash')))
        
        fig_offset.update_layout(title=f"{ticker}: Rec. Limit Offset (Front Month DTE {min_dte})",
                                 xaxis_title="Strike", yaxis_title="Offset form Mid ($)")

        # C. Size Heatmap
        # Aggregate size recommendations by DTE bucket and Moneyness
        df['moneyness'] = df['strike'] / res['spot']
        df['dte_bucket'] = (df['dte'] // 7) * 7 # Weekly buckets
        
        pivot = df.pivot_table(index='moneyness', columns='dte_bucket', values='rec_size', aggfunc='mean')
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Magma'
        ))
        fig_heat.update_layout(title=f"{ticker}: Rec. Order Size Heatmap", xaxis_title="DTE (Approx Weeks)", yaxis_title="Moneyness")

        # D. IV Panel (Gauge)
        fig_iv = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = res['iv_rank'],
            title = {'text': "IV Rank (%)"},
            delta = {'reference': 50},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': "black"},
                     'steps': [
                         {'range': [0, 30], 'color': "green"},
                         {'range': [30, 70], 'color': "yellow"},
                         {'range': [70, 100], 'color': "red"}],
                     'threshold': {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': res['iv_rank']}}
        ))
        fig_iv.update_layout(height=300)

        # Generate Divs
        div_liq = py_offline.plot(fig_liq, include_plotlyjs=False, output_type='div')
        div_offset = py_offline.plot(fig_offset, include_plotlyjs=False, output_type='div')
        div_heat = py_offline.plot(fig_heat, include_plotlyjs=False, output_type='div')
        div_iv = py_offline.plot(fig_iv, include_plotlyjs=False, output_type='div')
        
        # Simple Stats Table
        stats_html = f"""
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Spot</th><th>Liq Pct</th><th>IV Rank</th><th>IV Jump</th><th>Front IV</th>
            </tr>
            <tr>
                <td>{res['spot']:.2f}</td>
                <td>{res['ticker_liq_pct']:.1f}%</td>
                <td>{res['iv_rank']:.1f}%</td>
                <td>{res['iv_jump']:.2f}</td>
                <td>{res['front_atm_iv']:.2f}%</td>
            </tr>
        </table>
        <br>
        """

        # Assemble Ticker HTML
        # Using Flexbox for grid
        html = f"""
        <h3>{ticker} Analysis</h3>
        {stats_html}
        <div style="display: flex; flex-wrap: wrap;">
            <div style="flex: 50%;">{div_iv}</div>
            <div style="flex: 50%;">{div_heat}</div>
            <div style="flex: 50%;">{div_liq}</div>
            <div style="flex: 50%;">{div_offset}</div>
        </div>
        """
        return html


# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="GTC Limit & Liquidity Dashboard")
    
    # Required Args
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help="List of tickers")
    parser.add_argument('--output-dir', type=str, default='./market_data', help="Data storage directory")
    parser.add_argument('--lookback', type=float, default=1.0, help="Years of history for stats")
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help="Risk free rate")
    
    # Logic Args
    parser.add_argument('--interval', type=str, default='1d', choices=['1d', '1h', '30m'], help="Price interval")
    parser.add_argument('--min-dte', type=int, default=5, help="Min days to expiry")
    parser.add_argument('--max-dte', type=int, default=45, help="Max days to expiry")
    parser.add_argument('--base-order-size', type=float, default=1.0, help="Base lot size")
    parser.add_argument('--liquidity-threshold-pct', type=float, default=30.0, help="Liquidity percentile warning")
    parser.add_argument('--iv-jump-threshold', type=float, default=5.0, help="Vol jump threshold")
    parser.add_argument('--html-filename', type=str, default='gtc_liquidity_execution_dashboard.html', help="Output filename")
    
    args = parser.parse_args()
    
    # CRITICAL FIX: Enforce Uppercase for Tickers immediately to match YF data
    tickers = [t.upper() for t in args.tickers]
    
    # Initialize Modules
    data_ingest = DataIngestion(args.output_dir)
    fin_analysis = FinancialAnalysis(
        base_order_size=args.base_order_size,
        liquidity_threshold=args.liquidity_threshold_pct,
        iv_jump_threshold=args.iv_jump_threshold,
        risk_free_rate=args.risk_free_rate
    )
    full_output_path = os.path.abspath(os.path.join(args.output_dir, args.html_filename))
    renderer = DashboardRenderer(full_output_path)
    
    results = []
    
    print(f"Starting analysis for: {tickers}")
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        # 1. Ingestion
        try:
            df_prices = data_ingest.get_underlying_history(ticker, args.lookback, args.interval)
            df_options = data_ingest.get_options_snapshot(ticker, args.min_dte, args.max_dte)
        except Exception as e:
            logger.error(f"Critical error fetching data for {ticker}: {e}")
            continue
            
        # 2. Analysis
        try:
            ticker_res = fin_analysis.analyze_ticker(ticker, df_prices, df_options)
            if ticker_res:
                results.append(ticker_res)
            else:
                logger.warning(f"Analysis produced no results for {ticker} (possible missing options data).")
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)

    # 3. Visualization
    if results:
        print("Generating Dashboard...")
        renderer.generate_dashboard(results)
        print(f"Opening dashboard in browser: {full_output_path}")
        webbrowser.open('file://' + full_output_path)
    else:
        print("No results to display.")
    
    print("Done.")

if __name__ == "__main__":
    main()
