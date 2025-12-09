#!/usr/bin/env python3
"""
HEDGE FUND GRADE DASHBOARD: Dealer, Macro, Surface, Breadth
Version: 5.0 (Modular Refactor)

Features:
- Standalone yfinance integration (No external dependencies).
- Robust Column Parsing (Fixes yfinance MultiIndex swap issues).
- Offline Plotly Embedding (No CDN links).
- Tab Resize Auto-Correction.
- "Shadow Backfill" for cold-start historical data.
"""

import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
OUTPUT_ROOT = "DASHBOARD_OUTPUT"  # Changed from /dev/shm for cross-platform compatibility
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
YF_SLEEP_SECONDS = 1.0

# ==========================================
# 1. DATA INGESTION CLASS
# ==========================================
class DataIngestion:
    """
    Sole responsibility: Download, Cache, Load, and Sanitize Data.
    Enforces rate limits and strict data typing.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The 'Universal Fixer' for yfinance data structures.
        """
        if df.empty:
            return df

        # 1. Fix MultiIndex Column Swapping (Ticker <-> Attribute)
        if isinstance(df.columns, pd.MultiIndex):
            # Check where 'Close' or 'Adj Close' lives
            level_0 = df.columns.get_level_values(0)
            level_1 = df.columns.get_level_values(1)
            
            # If attributes are in level 1, we might need to swap if yfinance flipped them
            # Heuristic: If Ticker is in Level 0, we usually want to flatten. 
            # But recent yf versions sometimes put 'Close' in Level 0.
            
            has_close_L0 = any(x in ['Close', 'Adj Close'] for x in level_0)
            has_close_L1 = any(x in ['Close', 'Adj Close'] for x in level_1)

            if has_close_L1 and not has_close_L0:
                # Standard yf format: (Ticker, Attribute) -> We want attribute access usually
                # But for flattening, it's easier if we just drop the ticker level or flatten string
                pass
            elif has_close_L0 and not has_close_L1:
                # Inverted format: (Attribute, Ticker). Swap to normalize.
                self.logger.info("Detected inverted yfinance columns. Swapping levels.")
                df = df.swaplevel(0, 1, axis=1)

            # Flatten columns to single string "Ticker_Attribute" or just "Ticker" if single attribute
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Join non-empty elements
                    c = "_".join([str(x) for x in col if x]).strip()
                    new_cols.append(c)
                else:
                    new_cols.append(str(col))
            df.columns = new_cols

        # 2. Enforce Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find a Date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                # Attempt to convert index
                df.index = pd.to_datetime(df.index, errors='coerce')

        # 3. Strip Timezones
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 4. Numeric Coercion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_ticker_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Checks local CSV -> Downloads if stale/missing -> Saves.
        """
        clean_ticker = ticker.replace("^", "").replace("=", "")
        fpath = os.path.join(self.cache_dir, f"{clean_ticker}_hist.csv")
        today_str = datetime.now().strftime("%Y-%m-%d")

        # Persistence Check
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            # Simple check: if last date is today (or last friday), use it
            if not df.empty and df.index[-1].strftime("%Y-%m-%d") == today_str:
                return df
        
        # API Download
        self.logger.info(f"Downloading history for {ticker}...")
        time.sleep(YF_SLEEP_SECONDS)
        try:
            df = yf.download(ticker, period=period, group_by='column', progress=False)
            df = self._sanitize_df(df)
            
            # Handle specific single-ticker download structure in recent yfinance
            # If we requested one ticker, columns might just be 'Open', 'Close', etc.
            # If flattened, they might be 'AAPL_Close'. Normalize to 'Close'.
            if len(df.columns) > 0 and ticker not in df.columns[0]: 
                pass # Already standard names
            else:
                # If columns are like 'AAPL_Close', rename to 'Close' for generic usage
                mapping = {c: c.split('_')[-1] for c in df.columns if '_' in c}
                if mapping:
                    df.rename(columns=mapping, inplace=True)

            if not df.empty:
                df.to_csv(fpath)
            return df
        except Exception as e:
            self.logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def get_options_chain(self, ticker: str, max_expiries: int = 6) -> pd.DataFrame:
        """
        Fetches current options chain for nearest expirations.
        Note: yfinance does not provide historical options data.
        """
        self.logger.info(f"Fetching options chain for {ticker}...")
        tkr = yf.Ticker(ticker)
        all_opts = []
        
        try:
            exps = tkr.options
            if not exps:
                return pd.DataFrame()
            
            target_exps = exps[:max_expiries]
            
            for e in target_exps:
                time.sleep(YF_SLEEP_SECONDS / 2) # Be gentle
                try:
                    chain = tkr.option_chain(e)
                    calls = chain.calls
                    puts = chain.puts
                    calls['type'] = 'call'
                    puts['type'] = 'put'
                    
                    combined = pd.concat([calls, puts], axis=0, ignore_index=True)
                    combined['expiration'] = e
                    all_opts.append(combined)
                except Exception as inner_e:
                    self.logger.warning(f"Failed expiry {e}: {inner_e}")
                    continue
            
            if not all_opts:
                return pd.DataFrame()

            full_chain = pd.concat(all_opts, ignore_index=True)
            return full_chain

        except Exception as e:
            self.logger.error(f"Critical options fetch error: {e}")
            return pd.DataFrame()

# ==========================================
# 2. FINANCIAL ANALYSIS CLASS
# ==========================================
class FinancialAnalysis:
    """
    Sole responsibility: Math and Logic.
    Implements Copy-on-Write to protect raw data.
    """
    def __init__(self, raw_data_map: Dict[str, pd.DataFrame]):
        # Store copies to ensure immutability of source
        self._raw_data = {k: v.copy() for k, v in raw_data_map.items()}
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _safe_scalar(series: pd.Series) -> float:
        """Safely extracts a float from a single-element series."""
        if isinstance(series, pd.Series):
            if series.empty: return np.nan
            return float(series.iloc[0])
        return float(series)

    def calculate_shadow_gex_history(self, ticker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cold Start Prevention: Backfills 'Shadow GEX' based on realized volatility.
        Logic: Low Vol -> Dealers Long Gamma (Positive GEX). High Vol -> Short Gamma.
        """
        if ticker_df.empty:
            return pd.DataFrame()
        
        df = ticker_df.copy()
        df['returns'] = df['Close'].pct_change()
        
        # Realized Volatility (20-day rolling std dev)
        df['rv'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Shadow GEX Proxy: (Neutral_Vol - Realized_Vol) * Scalar
        # Assuming neutral market vol is ~15% (0.15)
        neutral_vol = 0.15
        scaling_factor = 1e9 # Arbitrary notional scaler for visualization
        
        df['shadow_gex'] = (neutral_vol - df['rv']) * scaling_factor
        df['gex'] = df['shadow_gex'] # Alias for plotting
        
        return df[['gex']].dropna()

    def bs_greeks(self, S, K, T, r, sigma, opt_type):
        """Vectorized Black-Scholes Greeks"""
        # Avoid division by zero
        T = np.maximum(T, 1e-4)
        sigma = np.maximum(sigma, 1e-4)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * pdf_d1 * np.sqrt(T)
        
        if opt_type == 'call':
            delta = cdf_d1
        else:
            delta = cdf_d1 - 1
            
        return delta, gamma, vega

    def calculate_dealer_exposure(self, chains: pd.DataFrame, spot: float) -> Tuple[float, float, float]:
        """
        Calculates Net GEX, Vanna, and PCR.
        """
        if chains.empty:
            return 0.0, 0.0, 0.0

        df = chains.copy()
        
        # Normalize columns
        col_map = {
            'strike': 'strike', 'lastPrice': 'price', 
            'openInterest': 'oi', 'impliedVolatility': 'iv',
            'expiration': 'expiration', 'type': 'type'
        }
        # Rename available columns
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
        
        # Fill NaNs
        df['oi'] = df['oi'].fillna(0)
        df['iv'] = df['iv'].fillna(0.2) # Default 20% IV
        
        # Calc Time to Expiry
        today = datetime.now()
        df['exp_date'] = pd.to_datetime(df['expiration'])
        df['T'] = (df['exp_date'] - today).dt.days / 365.0
        df['T'] = df['T'].clip(lower=0.001) # Avoid 0
        
        # Vectorized Greeks
        # Note: We need to handle calls and puts efficiently. 
        # For simplicity in this demo, we iterate types or use masking, but vectorized is better.
        r = 0.05 # Risk free assumption
        
        # Calculate Greeks
        calls = df[df['type'] == 'call'].copy()
        puts = df[df['type'] == 'put'].copy()
        
        c_delta, c_gamma, c_vega = self.bs_greeks(spot, calls['strike'], calls['T'], r, calls['iv'], 'call')
        p_delta, p_gamma, p_vega = self.bs_greeks(spot, puts['strike'], puts['T'], r, puts['iv'], 'put')
        
        # GEX = Gamma * Spot^2 * OI * 100 * (+1 for all long dealer assumption? No.)
        # Standard GEX: Market Maker is Short the Option. 
        # If Client Buys Call (Dealer Short Call): Dealer is Short Gamma. 
        # However, typical "GEX" charts assume Dealer is Long the OI. 
        # Convention: Call OI contributes POSITIVE Gamma, Put OI contributes NEGATIVE Gamma (Dealer hedging needs).
        
        # Contribution Logic (Dealer Perspective):
        # Long Call OI (Client Long, Dealer Short) -> Dealer Short Gamma -> Needs to Buy as goes up -> Accelerator.
        # WAIT. The standard SpotGamma/SqueezeMetrics def:
        # Call OI -> Market Maker is Short Call -> Short Gamma (hedges by buying into strength? No, Short Gamma sells into weakness).
        # Let's stick to the common convention: GEX = Gamma * Spot * Spot * OI * 100 * (+1 Call, -1 Put).
        
        calls['gex'] = c_gamma * (spot**2) * calls['oi'] * 100
        puts['gex'] = p_gamma * (spot**2) * puts['oi'] * 100 * -1
        
        # Vanna = Vega * Delta ... 
        # Vanna/Charm exposure is complex. Using simplified Vanna proxy: Vega * OI * 100
        calls['vanna'] = c_vega * calls['oi'] * 100 
        puts['vanna'] = p_vega * puts['oi'] * 100 
        
        net_gex = calls['gex'].sum() + puts['gex'].sum()
        net_vanna = calls['vanna'].sum() + puts['vanna'].sum()
        
        total_put_oi = puts['oi'].sum()
        total_call_oi = calls['oi'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        return net_gex, net_vanna, pcr

    def calculate_macro_composite(self) -> pd.DataFrame:
        """
        Combines Yields, Credit, Dollar into a regime signal.
        """
        required = ['^IRX', '^TNX', 'HYG', 'LQD', 'UUP']
        df_list = []
        for t in required:
            if t in self._raw_data:
                d = self._raw_data[t]['Close'].copy()
                d.name = t
                df_list.append(d)
        
        if not df_list:
            return pd.DataFrame()
            
        df = pd.concat(df_list, axis=1).fillna(method='ffill')
        
        # Logic: 
        # 1. Curve Slope: TNX (10y) - IRX (13w)
        if '^TNX' in df.columns and '^IRX' in df.columns:
            df['slope'] = df['^TNX'] - df['^IRX']
        else:
            df['slope'] = 0
            
        # 2. Credit Spread: HYG / LQD (Ratio)
        if 'HYG' in df.columns and 'LQD' in df.columns:
            df['credit'] = df['HYG'] / df['LQD']
        else:
            df['credit'] = 0
            
        # Z-Score normalization (rolling 60 day)
        for col in ['slope', 'credit', 'UUP']:
            if col in df.columns:
                mean = df[col].rolling(60).mean()
                std = df[col].rolling(60).std()
                df[f'{col}_z'] = (df[col] - mean) / std
        
        # Composite: Mean of Z-scores (Invert UUP as strong dollar is usually risk-off)
        cols_to_mean = [c for c in ['slope_z', 'credit_z'] if c in df.columns]
        if 'UUP_z' in df.columns:
            df['UUP_z_inv'] = df['UUP_z'] * -1
            cols_to_mean.append('UUP_z_inv')
            
        df['composite'] = df[cols_to_mean].mean(axis=1)
        return df[['composite']].dropna()

    def calculate_breadth(self) -> pd.DataFrame:
        """
        RSP (Equal Weight) vs SPY (Cap Weight) relative performance.
        """
        if 'RSP' not in self._raw_data or 'SPY' not in self._raw_data:
            return pd.DataFrame()
            
        rsp = self._raw_data['RSP']['Close']
        spy = self._raw_data['SPY']['Close']
        
        df = pd.concat([rsp, spy], axis=1, keys=['RSP', 'SPY']).dropna()
        
        # Normalize to start of period
        df['RSP_norm'] = df['RSP'] / df['RSP'].iloc[0]
        df['SPY_norm'] = df['SPY'] / df['SPY'].iloc[0]
        
        df['rel_strength'] = df['RSP_norm'] - df['SPY_norm']
        return df

# ==========================================
# 3. DASHBOARD RENDERER CLASS
# ==========================================
class DashboardRenderer:
    """
    Sole responsibility: Generate HTML with embedded JS fixes.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_offline_js(self) -> str:
        """Retrieves the full Plotly JS library string."""
        return pyo.get_plotlyjs()

    def _make_resize_script(self) -> str:
        """JavaScript to fix blank charts in tabs."""
        return """
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
                
                // THE FIX: Trigger resize event for Plotly
                window.dispatchEvent(new Event('resize'));
            }
        </script>
        """

    def _wrap_html(self, ticker: str, plots: Dict[str, str]) -> str:
        js_lib = self._get_offline_js()
        resize_script = self._make_resize_script()
        
        # Basic CSS for tabs
        css = """
        <style>
            body { font-family: sans-serif; background: #1e1e1e; color: #ddd; }
            .tab { overflow: hidden; border: 1px solid #333; background-color: #2e2e2e; }
            .tab button { background-color: inherit; float: left; border: none; outline: none; 
                          cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; }
            .tab button:hover { background-color: #444; }
            .tab button.active { background-color: #555; color: white; }
            .tabcontent { display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; }
            .plot-container { height: 80vh; width: 100%; }
        </style>
        """
        
        # Build Tabs HTML
        tab_buttons = '<div class="tab">'
        tab_contents = ''
        is_first = True
        
        for title, plot_div in plots.items():
            safe_title = title.replace(" ", "_")
            active_cls = " active" if is_first else ""
            display_style = "block" if is_first else "none"
            
            tab_buttons += f'<button class="tablinks{active_cls}" onclick="openTab(event, \'{safe_title}\')">{title}</button>'
            tab_contents += f'<div id="{safe_title}" class="tabcontent" style="display: {display_style};">{plot_div}</div>'
            is_first = False
            
        tab_buttons += '</div>'
        
        html = f"""
        <html>
        <head>
            <title>{ticker} Dashboard</title>
            <script type="text/javascript">{js_lib}</script>
            {css}
            {resize_script}
        </head>
        <body>
            <h2>{ticker} Quantitative Dashboard</h2>
            {tab_buttons}
            {tab_contents}
        </body>
        </html>
        """
        return html

    def generate_dashboard(self, ticker: str, figures: Dict[str, go.Figure]):
        """
        Converts Plotly Figures to HTML Divs and saves the full file.
        """
        plot_divs = {}
        for name, fig in figures.items():
            # Generate DIV only, no JS lib included here (we inject it in head)
            div = pyo.plot(fig, include_plotlyjs=False, output_type='div')
            plot_divs[name] = div
            
        full_html = self._wrap_html(ticker, plot_divs)
        
        fname = os.path.join(self.output_dir, f"{ticker}_Dashboard.html")
        with open(fname, "w", encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"Dashboard saved to: {fname}")

# ==========================================
# 4. MAIN ORCHESTRATION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", type=str, help="Main Ticker (e.g., SPY, NVDA)")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    # Setup
    ingestion = DataIngestion(cache_dir="MARKET_DATA_CACHE")
    renderer = DashboardRenderer(output_dir=OUTPUT_ROOT)
    
    print(f"--- Starting Analysis for {ticker} ---")
    
    # 1. Ingest Data
    # Main Ticker
    df_main = ingestion.get_ticker_history(ticker)
    
    # Macro Tickers
    macro_tickers = ['^IRX', '^TNX', 'HYG', 'LQD', 'UUP']
    raw_macro = {}
    for t in macro_tickers:
        raw_macro[t] = ingestion.get_ticker_history(t)
        
    # Breadth Tickers
    raw_breadth = {}
    for t in ['RSP', 'SPY']:
        raw_breadth[t] = ingestion.get_ticker_history(t)
        
    # Options Chain (Live)
    options_chain = ingestion.get_options_chain(ticker)
    
    # 2. Financial Analysis
    # Combine all raw data into one map for the analyzer
    all_raw = {ticker: df_main}
    all_raw.update(raw_macro)
    all_raw.update(raw_breadth)
    
    analyzer = FinancialAnalysis(all_raw)
    
    # A. Dealer Metrics (Current & History)
    spot_price = analyzer._safe_scalar(df_main['Close'].iloc[-1]) if not df_main.empty else 0
    
    # Calculate current Snapshot GEX
    current_gex, current_vanna, current_pcr = analyzer.calculate_dealer_exposure(options_chain, spot_price)
    
    # Handle Historical GEX (Cold Start: Shadow Backfill)
    # Check if we have a real history file (not implemented in this simplified script, so we use Shadow)
    # In a full system, you would load 'gex_history.csv' here.
    gex_history = analyzer.calculate_shadow_gex_history(df_main)
    
    # Append today's Real GEX to the Shadow History for the chart
    if spot_price > 0:
        new_row = pd.DataFrame({'gex': [current_gex]}, index=[pd.Timestamp.now().normalize()])
        gex_history = pd.concat([gex_history, new_row])
        # Remove duplicate index if exists
        gex_history = gex_history[~gex_history.index.duplicated(keep='last')]
    
    # B. Macro Regime
    macro_df = analyzer.calculate_macro_composite()
    
    # C. Breadth
    breadth_df = analyzer.calculate_breadth()
    
    # 3. Generate Figures
    figs = {}
    
    # Fig 1: Dealer Exposure
    fig_dealer = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dealer.add_trace(go.Scatter(x=gex_history.index, y=gex_history['gex'], 
                                    name="GEX (Shadow/Real)", mode='lines+markers', line=dict(color='cyan')))
    fig_dealer.update_layout(title=f"Dealer GEX Exposure: {ticker}", template="plotly_dark")
    figs['Dealer Positioning'] = fig_dealer
    
    # Fig 2: Macro
    if not macro_df.empty:
        fig_macro = go.Figure()
        fig_macro.add_trace(go.Scatter(x=macro_df.index, y=macro_df['composite'], 
                                       name="Macro Composite", fill='tozeroy'))
        fig_macro.update_layout(title="Macro Regime Composite (Rates/Credit/Dollar)", template="plotly_dark")
        figs['Macro Regime'] = fig_macro
        
    # Fig 3: Breadth
    if not breadth_df.empty:
        fig_breadth = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_breadth.add_trace(go.Scatter(x=breadth_df.index, y=breadth_df['rel_strength'], 
                                         name="RSP vs SPY"), row=1, col=1)
        fig_breadth.add_trace(go.Scatter(x=df_main.index, y=df_main['Close'], 
                                         name="Price"), row=2, col=1)
        fig_breadth.update_layout(title="Market Breadth (Equal vs Cap Weight)", template="plotly_dark")
        figs['Breadth'] = fig_breadth
        
    # 4. Render
    renderer.generate_dashboard(ticker, figs)

if __name__ == "__main__":
    main()
