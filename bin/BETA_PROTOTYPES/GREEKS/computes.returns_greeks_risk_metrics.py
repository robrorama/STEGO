#!/usr/bin/env python3
# SCRIPTNAME: ok.computes.returns_greeks_risk_metrics.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Production-Grade Quantitative Dashboard Generator
-------------------------------------------------
Role: Senior Quantitative Developer
Objective: Ingest market data, compute financial metrics/greeks, render offline dashboard.

Architecture:
1. DataIngestion: Disk-first IO, Sanatization.
2. FinancialAnalysis: Math, Greeks, Signals.
3. DashboardRenderer: Plotly Visualization.

Usage:
    python quant_dashboard.py --tickers SPY QQQ --lookback 2
"""

import os
import sys
import time
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# LOGGING & CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("QuantEngine")
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. DATA INGESTION CLASS
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Responsibilities:
    - IO & data acquisition ONLY
    - Strict disk-first pipeline
    - Universal Fixer sanitization
    """
    def __init__(self, output_dir: str, interval: str, lookback_years: float, intraday: bool):
        self.output_dir = output_dir
        self.interval = interval
        self.lookback_years = lookback_years
        self.intraday = intraday
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_or_download_all(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Orchestrates the disk-first loading pipeline."""
        data_store = {}
        
        for ticker in tickers:
            file_path = self._get_file_path(ticker)
            
            # 1. Check disk
            if os.path.exists(file_path):
                logger.info(f"Loading {ticker} from disk: {file_path}")
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    df = self._sanitize_df(df, ticker) # Re-sanitize ensures types/index are correct
                    data_store[ticker] = df
                    continue
                except Exception as e:
                    logger.error(f"Corrupt file for {ticker}, re-downloading. Error: {e}")
            
            # 2. Download if missing or corrupt
            logger.info(f"Downloading {ticker} via yfinance...")
            raw_df = self._download_ticker(ticker)
            
            # 3. Sanitize
            clean_df = self._sanitize_df(raw_df, ticker)
            
            # 4. Save to disk
            clean_df.to_csv(file_path)
            
            # 5. Reload (Shadow backfill guarantee)
            data_store[ticker] = clean_df
            time.sleep(1.0) # Rate limit
            
        return data_store

    def fetch_options_chain(self, ticker: str) -> Any:
        """Fetches raw option chain object (IO task)."""
        try:
            return yf.Ticker(ticker)
        except Exception as e:
            logger.warning(f"Could not fetch options for {ticker}: {e}")
            return None

    def _download_ticker(self, ticker: str) -> pd.DataFrame:
        """Downloads data from yfinance with strict group_by rules."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(365 * self.lookback_years))
        
        # Adjust for intraday limits if necessary
        if self.intraday and self.lookback_years > 0.16: # ~60 days
            logger.warning(f"Intraday lookback capped at 60 days for {ticker}")
            start_date = end_date - timedelta(days=59)

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=self.interval,
                group_by='column',
                auto_adjust=False,
                progress=False,
                threads=False 
            )
            return df
        except Exception as e:
            logger.error(f"Download failed for {ticker}: {e}")
            return pd.DataFrame()

    def _get_file_path(self, ticker: str) -> str:
        safe_ticker = ticker.replace("^", "")
        mode = "intraday" if self.intraday else "daily"
        return os.path.join(self.output_dir, f"{safe_ticker}_{mode}.csv")

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        THE UNIVERSAL FIXER
        """
        if df.empty:
            return df

        # 1. Group-by enforcement & Level Check
        # yfinance often returns MultiIndex columns (Price, Ticker) or (Ticker, Price)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if Ticker is level 0 or level 1
            if ticker in df.columns.get_level_values(0):
                # Standard: (Ticker, Price) -> Drop level 0
                df = df.droplevel(0, axis=1)
            elif ticker in df.columns.get_level_values(1):
                # Swapped: (Price, Ticker) -> Drop level 1
                df = df.droplevel(1, axis=1)
            
            # If multiindex still exists (e.g. strict group_by='column' output from recent yf versions)
            # Sometimes it comes back as just Price columns if single ticker download.
        
        # 2. Flatten Columns (Standardize names)
        # Ensure we just have strings like "Open", "Close", etc.
        # Then map to "{TICKER}_{FIELD}"
        new_cols = []
        for col in df.columns:
            col_name = col if isinstance(col, str) else col[0]
            # Clean weird chars
            col_name = col_name.strip()
            new_cols.append(f"{ticker}_{col_name}")
        df.columns = new_cols

        # 3. Timezone normalization
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 4. Sort & Dedupe
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # 5. Numeric Coercion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where everything is NaN (failed download artifacts)
        df.dropna(how='all', inplace=True)

        return df

# -----------------------------------------------------------------------------
# 2. FINANCIAL ANALYSIS CLASS
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Responsibilities:
    - Math, Signals, Greeks, Transforms
    - NO IO, NO Plotting
    """
    def __init__(self, data_store: Dict[str, pd.DataFrame], risk_free_rate: float):
        self.data_store = data_store
        self.rf = risk_free_rate
        self.results = {}

    def compute_all_metrics(self) -> Dict[str, Any]:
        """Driver for all calculations."""
        for ticker, df in self.data_store.items():
            if df.empty:
                continue
            
            # Setup container
            self.results[ticker] = {
                'df': df.copy(),
                'latest': {},
                'options': {}
            }
            
            # 4.1 Returns
            close_col = f"{ticker}_Close"
            if close_col not in df.columns:
                # Try Adj Close fallback
                close_col = f"{ticker}_Adj Close"
            
            if close_col in df.columns:
                self._compute_returns(ticker, close_col)
                self._compute_volatility(ticker)
                self._compute_sharpe(ticker)
                self._compute_trend_indicators(ticker, close_col)
                self._compute_microstructure(ticker) # Checks for bid/ask cols inside
            
            # Update latest snapshot
            self._update_snapshot(ticker, close_col)

        return self.results

    def compute_options_greeks(self, ticker: str, yf_ticker_obj: Any):
        """
        Computes numerical Greeks for options.
        Note: This is computationally intensive. Limiting to near-term expirations.
        """
        if not yf_ticker_obj:
            return

        try:
            expirations = yf_ticker_obj.options
            if not expirations:
                return
            
            # Select first 2 expirations to save time/memory
            target_exps = expirations[:2]
            
            current_price = self.results[ticker]['df'][f"{ticker}_Close"].iloc[-1]
            chain_data = []

            for date_str in target_exps:
                opt_chain = yf_ticker_obj.option_chain(date_str)
                calls = opt_chain.calls
                puts = opt_chain.puts
                
                # Calculate Time to Expiry (T)
                expiry_date = pd.to_datetime(date_str)
                T = (expiry_date - datetime.now()).days / 365.0
                if T < 0.001: T = 0.001 # Avoid div by zero

                # Process Calls
                self._process_chain_subset(calls, 'call', current_price, T, chain_data)
                # Process Puts
                self._process_chain_subset(puts, 'put', current_price, T, chain_data)

            self.results[ticker]['options'] = pd.DataFrame(chain_data)
            
        except Exception as e:
            logger.error(f"Error computing Greeks for {ticker}: {e}")

    def _process_chain_subset(self, df, opt_type, S, T, collector):
        """Helper to iterate rows and calc greeks."""
        for _, row in df.iterrows():
            K = row['strike']
            sigma = row['impliedVolatility']
            price = row['lastPrice']
            
            # Skip invalid data
            if sigma is None or sigma < 0.01 or K <= 0:
                continue

            greeks = self._numerical_greeks(S, K, T, self.rf, sigma, opt_type)
            
            entry = {
                'type': opt_type,
                'strike': K,
                'expiry_T': T,
                'iv': sigma,
                'lastPrice': price,
                **greeks
            }
            collector.append(entry)

    def _numerical_greeks(self, S, K, T, r, sigma, opt_type):
        """
        Finite Difference Method for Greeks.
        """
        ds = S * 0.01 # 1% step for Delta/Gamma
        dt = 1 / 365.0 # 1 day step for Theta
        dv = 0.01 # 1% step for Vega
        dr = 0.0001 # 1 bps for Rho

        # Base Price
        V = self._black_scholes(S, K, T, r, sigma, opt_type)

        # Delta & Gamma (Central Difference)
        V_up = self._black_scholes(S + ds, K, T, r, sigma, opt_type)
        V_down = self._black_scholes(S - ds, K, T, r, sigma, opt_type)
        
        delta = (V_up - V_down) / (2 * ds)
        gamma = (V_up - 2 * V + V_down) / (ds ** 2)

        # Theta (Forward Difference, usually negative)
        # Time decays, so T becomes T-dt
        V_t_minus = self._black_scholes(S, K, T - dt, r, sigma, opt_type)
        theta = (V_t_minus - V) / dt # Daily theta

        # Vega (Central Difference)
        V_vol_up = self._black_scholes(S, K, T, r, sigma + dv, opt_type)
        V_vol_down = self._black_scholes(S, K, T, r, sigma - dv, opt_type)
        vega = (V_vol_up - V_vol_down) / (2 * dv) / 100 # Scaled

        # Rho
        V_r_up = self._black_scholes(S, K, T, r + dr, sigma, opt_type)
        rho = (V_r_up - V) / dr / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def _black_scholes(self, S, K, T, r, sigma, opt_type):
        """Pricing Model."""
        if T <= 0: return max(0, S - K) if opt_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if opt_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    def _compute_returns(self, ticker, close_col):
        df = self.results[ticker]['df']
        df['Returns'] = df[close_col].pct_change()
        df['LogReturns'] = np.log(df[close_col] / df[close_col].shift(1))

    def _compute_volatility(self, ticker):
        df = self.results[ticker]['df']
        for window in [10, 20, 60]:
            # Annualized RV
            df[f'RV_{window}'] = df['LogReturns'].rolling(window).std() * np.sqrt(252)

    def _compute_sharpe(self, ticker):
        df = self.results[ticker]['df']
        # Rolling Sharpe (Ex-post)
        # Assuming 20-day window for local regime
        window = 20
        excess_ret = df['Returns'] - (self.rf / 252)
        df['RollingSharpe'] = excess_ret.rolling(window).mean() / (df['Returns'].rolling(window).std() + 1e-9)

    def _compute_trend_indicators(self, ticker, col):
        df = self.results[ticker]['df']
        # SMAs
        df['SMA_20'] = df[col].rolling(20).mean()
        df['SMA_50'] = df[col].rolling(50).mean()
        
        # MACD
        ema12 = df[col].ewm(span=12, adjust=False).mean()
        ema26 = df[col].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']

    def _compute_microstructure(self, ticker):
        """Microprice and Order Imbalance (if Bid/Ask data exists)."""
        df = self.results[ticker]['df']
        bid_col = f"{ticker}_Bid"
        ask_col = f"{ticker}_Ask"
        bs_col = f"{ticker}_Bid Size" # Note: yfinance spaces usage varies
        as_col = f"{ticker}_Ask Size" # Sometimes "AskSize", checking logic needed if using other providers

        # Basic check for column existence (YFinance history rarely returns Size, but we support logic)
        # Adjusting names to sanitized output
        # Assuming sanitized as: TICKER_Bid, TICKER_Ask
        
        if bid_col in df.columns and ask_col in df.columns:
            # VWAP approximation (using High/Low/Close and Volume)
            if f"{ticker}_Volume" in df.columns:
                 v = df[f"{ticker}_Volume"]
                 tp = (df[f"{ticker}_High"] + df[f"{ticker}_Low"] + df[f"{ticker}_Close"]) / 3
                 df['VWAP'] = (tp * v).cumsum() / v.cumsum()
            
            # Microprice requires sizes. If not present, skip
            # Mocking logic for size columns if they don't exist in standard free data
            # to demonstrate the math
            pass 

    def _update_snapshot(self, ticker, close_col):
        df = self.results[ticker]['df']
        if df.empty: return
        
        last_row = df.iloc[-1]
        
        # Vol Regime
        rv20 = last_row.get('RV_20', 0)
        if rv20 < 0.10: v_reg = "Low Vol"
        elif rv20 < 0.20: v_reg = "Normal Vol"
        else: v_reg = "High Vol"

        # Trend Regime
        sma20 = last_row.get('SMA_20', 0)
        sma50 = last_row.get('SMA_50', 0)
        price = last_row[close_col]
        
        if price > sma20 > sma50: t_reg = "Bullish"
        elif price < sma20 < sma50: t_reg = "Bearish"
        else: t_reg = "Neutral/Chop"

        self.results[ticker]['latest'] = {
            'date': df.index[-1],
            'price': price,
            'return': last_row.get('Returns', 0),
            'rv20': rv20,
            'vol_regime': v_reg,
            'trend_regime': t_reg
        }

# -----------------------------------------------------------------------------
# 3. DASHBOARD RENDERER CLASS
# -----------------------------------------------------------------------------
class DashboardRenderer:
    """
    Responsibilities:
    - ALL Plotly visualization
    - Single HTML output
    - Offline JS embedding
    """
    def __init__(self, analysis_results: Dict[str, Any], output_dir: str, filename: str):
        self.data = analysis_results
        self.output_path = os.path.join(output_dir, filename)

    def generate_dashboard(self):
        """Creates the multi-tab HTML dashboard."""
        if not self.data:
            logger.warning("No data to render.")
            return

        # Prepare Tabs
        # We will create one big figure with dropdowns OR separate HTML sections?
        # Plotly "Tabs" are usually handled via Dash. For static HTML, we use
        # Bootstrap tabs + Plotly divs, or a single Plotly figure with Updatemenus (complex).
        # Better approach for "Hedge Fund Grade":
        # Create separate Plotly HTML divs and stitch them into a custom HTML template with Tabs.
        
        html_parts = []
        tab_headers = []
        
        tickers = list(self.data.keys())
        
        # We will generate one set of tabs PER TICKER or one global dashboard?
        # Specification implies a dashboard. We will focus on the FIRST ticker 
        # or aggregate. Let's build a dashboard for the first ticker in list 
        # or create a dropdown in Plotly.
        # To keep it robust: We visualize the FIRST ticker provided as the "Active" one,
        # or just stack them. Let's do the first ticker for the detailed view.
        
        primary_ticker = tickers[0]
        t_data = self.data[primary_ticker]
        df = t_data['df']
        
        # --- TAB 1: PRICE & SIGNALS ---
        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        # Candlestick (Simulated with OHLC if avail, else Line)
        open_c = f"{primary_ticker}_Open"
        high_c = f"{primary_ticker}_High"
        low_c = f"{primary_ticker}_Low"
        close_c = f"{primary_ticker}_Close"
        
        if open_c in df.columns:
            fig1.add_trace(go.Candlestick(x=df.index, open=df[open_c], high=df[high_c],
                                          low=df[low_c], close=df[close_c], name='Price'), row=1, col=1)
        else:
            fig1.add_trace(go.Scatter(x=df.index, y=df[close_c], name='Price'), row=1, col=1)

        # SMAs
        if 'SMA_20' in df.columns:
            fig1.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(width=1)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(width=1)), row=1, col=1)

        # MACD
        if 'MACD_Hist' in df.columns:
            fig1.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist'), row=2, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['MACD_Line'], name='MACD'), row=2, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'), row=2, col=1)

        fig1.update_layout(title=f"{primary_ticker} - Price & Core Signals", template="plotly_dark", height=700)
        div1 = py_offline.plot(fig1, include_plotlyjs=False, output_type='div')
        
        # --- TAB 2: VOLATILITY ---
        fig2 = go.Figure()
        if 'RV_20' in df.columns:
            fig2.add_trace(go.Scatter(x=df.index, y=df['RV_10'], name='RV 10d'))
            fig2.add_trace(go.Scatter(x=df.index, y=df['RV_20'], name='RV 20d'))
            fig2.add_trace(go.Scatter(x=df.index, y=df['RV_60'], name='RV 60d'))
        fig2.update_layout(title=f"{primary_ticker} - Realized Volatility", template="plotly_dark", height=700)
        div2 = py_offline.plot(fig2, include_plotlyjs=False, output_type='div')
        
        # --- TAB 3: OPTIONS SURFACE (If data exists) ---
        opt_df = t_data.get('options')
        div3 = "<div>No Options Data Available</div>"
        
        if isinstance(opt_df, pd.DataFrame) and not opt_df.empty:
            calls = opt_df[opt_df['type'] == 'call']
            if not calls.empty:
                # 3D Surface: Strike x Expiry x IV
                fig3 = go.Figure(data=[go.Scatter3d(
                    x=calls['strike'],
                    y=calls['expiry_T']*365,
                    z=calls['iv'],
                    mode='markers',
                    marker=dict(size=3, color=calls['iv'], colorscale='Viridis'),
                    text=calls['iv']
                )])
                fig3.update_layout(title="Call IV Surface (Strike vs Days vs IV)", scene=dict(
                    xaxis_title='Strike', yaxis_title='Days to Exp', zaxis_title='IV'
                ), template="plotly_dark", height=700)
                div3 = py_offline.plot(fig3, include_plotlyjs=False, output_type='div')

        # --- TAB 4: DISTRIBUTIONS ---
        fig4 = make_subplots(rows=1, cols=2)
        fig4.add_trace(go.Histogram(x=df['Returns'], nbinsx=50, name='Returns Dist'), row=1, col=1)
        if 'RV_20' in df.columns:
            fig4.add_trace(go.Box(y=df['RV_20'], name='Vol Dist'), row=1, col=2)
        fig4.update_layout(title="Distributions", template="plotly_dark", height=700)
        div4 = py_offline.plot(fig4, include_plotlyjs=False, output_type='div')

        # --- CONSTRUCT HTML ---
        # We need to embed plotly.js manually or via CDN (Offline mode implies inline usually, 
        # but the file is 3MB. plot(include_plotlyjs=True) puts it in every div. 
        # We will put it once in the head.)
        
        plotly_js = py_offline.get_plotlyjs()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quant Dashboard: {primary_ticker}</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: sans-serif; background-color: #111; color: #ddd; margin: 0; }}
                .tab {{ overflow: hidden; border: 1px solid #444; background-color: #222; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ddd; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #007bff; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #444; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h1 style="padding: 10px;">Institutional Dashboard: {primary_ticker}</h1>
            
            <div class="tab">
              <button class="tablinks" onclick="openTab(event, 'Price')" id="defaultOpen">Price & Signals</button>
              <button class="tablinks" onclick="openTab(event, 'Vol')">Volatility</button>
              <button class="tablinks" onclick="openTab(event, 'Options')">Options Analysis</button>
              <button class="tablinks" onclick="openTab(event, 'Dist')">Distributions</button>
            </div>

            <div id="Price" class="tabcontent">{div1}</div>
            <div id="Vol" class="tabcontent">{div2}</div>
            <div id="Options" class="tabcontent">{div3}</div>
            <div id="Dist" class="tabcontent">{div4}</div>

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
                if (evt) evt.currentTarget.className += " active";
                
                // MANDATORY RESIZE FIX FOR PLOTLY IN TABS
                window.dispatchEvent(new Event('resize'));
            }}
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        with open(self.output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        return self.output_path

# -----------------------------------------------------------------------------
# 4. MAIN & CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Institutional Quant Dashboard")
    parser.add_argument("--tickers", nargs="+", default=['SPY', 'QQQ', 'IWM'], help="List of tickers")
    parser.add_argument("--output-dir", default="./market_data", help="Data/Output directory")
    parser.add_argument("--lookback", type=float, default=1.0, help="Lookback in years")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk free rate (decimal)")
    parser.add_argument("--intraday", action="store_true", help="Enable intraday logic")
    parser.add_argument("--interval", default="1d", help="Interval (1d, 1h, 5m)")
    parser.add_argument("--dashboard-name", default="dashboard.html", help="Output HTML filename")
    
    args = parser.parse_args()

    # Override interval if intraday flag is set but interval is default
    if args.intraday and args.interval == "1d":
        args.interval = "5m"
        print(f"INFO: Intraday flag set. Switching interval to {args.interval}")

    # 1. Ingestion
    print("\n--- 1. STARTING DATA INGESTION ---")
    ingestor = DataIngestion(args.output_dir, args.interval, args.lookback, args.intraday)
    data_store = ingestor.load_or_download_all(args.tickers)

    # 2. Analysis
    print("\n--- 2. RUNNING FINANCIAL ANALYSIS ---")
    analyzer = FinancialAnalysis(data_store, args.risk_free_rate)
    results = analyzer.compute_all_metrics()
    
    # Compute Options (Separately as it involves network calls handled via ingestion helper)
    print("Computing Greeks (Fetching option chains)...")
    for ticker in args.tickers:
        yf_obj = ingestor.fetch_options_chain(ticker)
        if yf_obj:
            analyzer.compute_options_greeks(ticker, yf_obj)

    # 3. CLI Summary
    print("\n--- 3. SUMMARY REPORT ---")
    print(f"{'Ticker':<10} | {'Last Date':<12} | {'Price':<10} | {'Return':<8} | {'RV20':<8} | {'Regime'}")
    print("-" * 80)
    for ticker, res in results.items():
        latest = res['latest']
        if not latest:
            continue
        print(f"{ticker:<10} | {str(latest['date'].date()):<12} | {latest['price']:.2f}      | {latest['return']:.2%}{' ':<2} | {latest['rv20']:.2%}   | {latest['vol_regime']}")
        
        # Alerts
        if latest['rv20'] > 0.30:
            print(f"  [ALERT] Extreme Volatility detected for {ticker}")
        if abs(latest['return']) > 0.03:
            print(f"  [ALERT] Large Move detected for {ticker}")

    # 4. Visualization
    print("\n--- 4. RENDERING DASHBOARD ---")
    renderer = DashboardRenderer(results, args.output_dir, args.dashboard_name)
    path = renderer.generate_dashboard()
    print(f"Dashboard saved to: {path}")

if __name__ == "__main__":
    main()
