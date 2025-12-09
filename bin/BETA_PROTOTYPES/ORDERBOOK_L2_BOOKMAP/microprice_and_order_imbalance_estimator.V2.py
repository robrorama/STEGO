# SCRIPTNAME: ok.microprice_and_order_imbalance_estimator.V2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import time
import argparse
import logging
import datetime
import warnings
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MicropriceDash")

# Suppress pandas FutureWarnings for clean CLI output
warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------------------------------------------------------------
# CLASS 1: DATA INGESTION (Load vs Process Split)
# --------------------------------------------------------------------------------
class DataIngestion:
    """
    Handles file I/O, API calls, caching, and aggressive data sanitization.
    Strictly separates 'getting data' from 'analyzing data'.
    """
    def __init__(self, ticker: str, interval: str, lookback_days: int, output_dir: str):
        self.ticker = ticker.upper()
        self.interval = interval
        self.lookback_days = lookback_days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_paths(self) -> Dict[str, Path]:
        """Generate file paths for caching."""
        sanitized_ticker = self.ticker.replace("^", "")
        return {
            "intraday": self.output_dir / f"intraday_{sanitized_ticker}_{self.interval}.csv",
            "daily": self.output_dir / f"daily_{sanitized_ticker}.csv",
            "options": self.output_dir / f"options_chain_{sanitized_ticker}.csv",
            "shadow": self.output_dir / f"microprice_history_{sanitized_ticker}.csv"
        }

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        UNIVERSAL FIXER: Aggressive sanitization of yfinance DataFrames.
        Handles MultiIndex flattening, timezone stripping, and numeric coercion.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # 1. Normalize Columns (MultiIndex handling)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if Ticker is level 0 or level 1
            # We want attributes (Open, Close) in columns, not Ticker
            if self.ticker in df.columns.get_level_values(0):
                # Ticker is level 0, swap to make attributes level 0
                df.columns = df.columns.swaplevel(0, 1)
            
            # Now flatten: If column is ('Close', 'SPY') -> 'Close'
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Take the attribute part (usually first after swap, or second originally)
                    # We assume the attribute is one of the standard OHLCV names
                    c_name = col[0] if col[0] in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] else col[1]
                    new_cols.append(c_name)
                else:
                    new_cols.append(col)
            df.columns = new_cols

        # 2. Strict Datetime Index
        # If index is numeric (0,1,2), look for 'Date' or 'Datetime' column
        if not isinstance(df.index, pd.DatetimeIndex):
            for col_name in ['Date', 'Datetime', 'date', 'datetime']:
                if col_name in df.columns:
                    df.set_index(col_name, inplace=True)
                    break
        
        # Coerce index to datetime
        df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
        df = df[df.index.notnull()]

        # 3. Strip Timezones (Make naive for Plotly/Pandas compatibility)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 4. Numeric Coercion
        # Ensure OHLCV are floats
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where critical data is NaN
        if 'Close' in df.columns:
            df.dropna(subset=['Close'], inplace=True)

        return df

    def fetch_intraday(self) -> pd.DataFrame:
        """Cache-first strategy for intraday data."""
        path = self._get_paths()["intraday"]
        
        # Try Cache
        if path.exists() and path.stat().st_size > 0:
            logger.info(f"Loading cached intraday data from {path}")
            try:
                df = pd.read_csv(path, index_col=0)
                return self._sanitize_df(df)
            except Exception as e:
                logger.warning(f"Cache corrupt ({e}), redownloading...")

        # Download
        logger.info(f"Downloading {self.ticker} Intraday ({self.interval})...")
        time.sleep(1) # Rate limit protection
        try:
            # Note: period is calculated roughly from days, capped at yf limits usually
            # for 1m, max is 7d. For dashboard logic, we handle request carefully.
            period_str = f"{self.lookback_days}d" if self.lookback_days < 60 else "max"
            df = yf.download(
                self.ticker, 
                period=period_str, 
                interval=self.interval, 
                group_by='column', 
                progress=False,
                auto_adjust=False
            )
            df = self._sanitize_df(df)
            if not df.empty:
                df.to_csv(path)
            return df
        except Exception as e:
            logger.error(f"Failed to download intraday data: {e}")
            return pd.DataFrame()

    def _backfill_shadow_history(self) -> pd.DataFrame:
        """Ensures we have daily context for 'Cold Start' prevention."""
        path = self._get_paths()["daily"]
        
        # Logic: Always try to refresh shadow history if it's old, or load cache
        need_download = True
        if path.exists():
            # Check modification time? For now, just load to see if valid
            df = pd.read_csv(path, index_col=0)
            df = self._sanitize_df(df)
            if not df.empty:
                need_download = False
                # If last date is old, might want to update, but sticking to cache-first for speed
        
        if need_download:
            logger.info("Backfilling shadow history (Daily OHLC)...")
            time.sleep(1)
            try:
                df = yf.download(self.ticker, period="1y", interval="1d", group_by='column', progress=False)
                df = self._sanitize_df(df)
                if not df.empty:
                    df.to_csv(path)
            except Exception:
                return pd.DataFrame()
        
        return df

    def fetch_options_chain(self) -> pd.DataFrame:
        """Fetches nearest expiry option chain for ATM IV context."""
        # Options data is volatile and hard to cache meaningfully for "Live" use,
        # but we stick to the architecture. We won't cache options to disk to force freshness 
        # unless user specifically wants persistence, but prompts says 'cache first'.
        # We will skip disk caching for options to ensure relevance, as stale options are useless.
        
        logger.info("Fetching Options Chain...")
        time.sleep(1)
        try:
            t = yf.Ticker(self.ticker)
            exps = t.options
            if not exps:
                return pd.DataFrame()
            
            # Get nearest expiry
            chain = t.option_chain(exps[0])
            calls = chain.calls
            puts = chain.puts
            
            # Sanitize minimal
            calls['type'] = 'call'
            puts['type'] = 'put'
            calls['expiry'] = exps[0]
            puts['expiry'] = exps[0]
            
            df = pd.concat([calls, puts], ignore_index=True)
            return df
        except Exception as e:
            logger.warning(f"Options data unavailable: {e}")
            return pd.DataFrame()

# --------------------------------------------------------------------------------
# CLASS 2: FINANCIAL ANALYSIS (Math & Logic)
# --------------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Core math engine. Computes microprice, persistence, imbalance, and regimes.
    Immutable inputs: does not modify raw data in place.
    """
    def __init__(self, intraday_df: pd.DataFrame, daily_df: pd.DataFrame, options_df: pd.DataFrame):
        self._intraday = intraday_df.copy()
        self._daily = daily_df.copy()
        self._options = options_df.copy()
        self.results = {}

    def run_analysis(self):
        """Orchestrator for all analysis steps."""
        if self._intraday.empty:
            logger.warning("Intraday data empty. Skipping intraday analysis.")
        else:
            self._compute_microprice_and_imbalance()
            self._apply_persistence_filter()
            self._compute_directional_pressure()
            self._compute_regimes()

        if not self._daily.empty:
            self._compute_shadow_metrics()
        
        if not self._options.empty and not self._intraday.empty:
            self._compute_options_context()

    def _compute_microprice_and_imbalance(self):
        """
        Estimates Microprice and Imbalance using OHLCV as pseudo-L1 data.
        """
        df = self._intraday.copy()
        
        # 1. Proxies for Bid/Ask Price
        # In a bar, High is approx best Ask, Low is approx best Bid (loose approximation)
        # Midprice
        df['MidPrice'] = (df['High'] + df['Low']) / 2.0
        
        # 2. Proxies for Bid/Ask Size
        # We infer "Buying Volume" vs "Selling Volume" based on Close relative to Open
        # If Close > Open, we assume aggressive buying (Ask Size consumption).
        # We model Order Book:
        #   BidSize ~ Volume * (Relative position of Close to Low)
        #   AskSize ~ Volume * (Relative position of High to Close)
        
        # Normalize Close location (0 to 1)
        range_len = (df['High'] - df['Low']).replace(0, 0.01) # avoid div0
        clv = (df['Close'] - df['Low']) / range_len
        
        # Synthetic Sizes (Smoothing applied to simulate book depth)
        df['SynBidSize'] = df['Volume'] * clv
        df['SynAskSize'] = df['Volume'] * (1 - clv)
        
        # Safety for zero volume bars
        df['SynBidSize'] = df['SynBidSize'].replace(0, 1)
        df['SynAskSize'] = df['SynAskSize'].replace(0, 1)

        # 3. Microprice Calculation (Size Weighted)
        # MP = (AskPx * BidSz + BidPx * AskSz) / (BidSz + AskSz)
        # Using High as AskPx, Low as BidPx
        df['MicroPrice'] = (
            (df['High'] * df['SynBidSize']) + (df['Low'] * df['SynAskSize'])
        ) / (df['SynBidSize'] + df['SynAskSize'])

        # 4. Imbalance
        # Imbalance = (BidSz - AskSz) / (BidSz + AskSz)
        df['Imbalance'] = (df['SynBidSize'] - df['SynAskSize']) / (df['SynBidSize'] + df['SynAskSize'])
        
        # Microprice Spread in Bps
        df['MicroSpread_Bps'] = ((df['MicroPrice'] - df['MidPrice']) / df['MidPrice']) * 10000

        self.results['intraday'] = df

    def _apply_persistence_filter(self):
        """
        Implements Quote Persistence.
        Quote is persistent if 'Price State' remains stable for N bars.
        """
        df = self.results['intraday']
        
        # Define a Quote State: Round MidPrice to nearest 0.05 or similar tick size proxy
        # If the rounded midprice doesn't change, we assume the quote "level" is holding.
        tick_size = 0.05 # Assumption for major ETFs/Stocks
        df['QuoteState'] = (df['MidPrice'] / tick_size).round() * tick_size
        
        # Calculate streak (Persistence Duration)
        # Compare current state to previous state
        df['StateChanged'] = df['QuoteState'] != df['QuoteState'].shift(1)
        # Group by consecutive changes to count duration
        df['Persistence_Group'] = df['StateChanged'].cumsum()
        df['Persistence_Duration'] = df.groupby('Persistence_Group').cumcount() + 1
        
        # Filter: Only consider metrics "Valid" if Persistence > Threshold (e.g., 2 bars)
        # In a real tick feed, this would be milliseconds. Here it's bars.
        persistence_threshold = 2
        
        df['Filtered_Imbalance'] = np.where(
            df['Persistence_Duration'] >= persistence_threshold,
            df['Imbalance'],
            np.nan 
        )
        
        # Forward fill valid imbalance to avoid gaps in plotting, but mark as "Low Confidence" conceptually
        df['Filtered_Imbalance_Filled'] = df['Filtered_Imbalance'].ffill()

        self.results['intraday'] = df

    def _compute_directional_pressure(self):
        """
        Score = Sign(EMA(Imbalance)) * |Delta Microprice|
        """
        df = self.results['intraday']
        
        # EMA of Imbalance (smooth out noise)
        ema_imb = df['Imbalance'].ewm(span=5, adjust=False).mean()
        
        # Delta Microprice
        delta_mp = df['MicroPrice'].diff().fillna(0)
        
        # Directional Pressure
        df['DirectionalPressure'] = np.sign(ema_imb) * np.abs(delta_mp)
        
        # Cumulative Pressure (Tape momentum)
        df['CumPressure'] = df['DirectionalPressure'].cumsum()
        
        self.results['intraday'] = df

    def _compute_regimes(self):
        """
        Classify bars into Regimes based on Filtered Imbalance.
        """
        df = self.results['intraday']
        imb = df['Filtered_Imbalance_Filled'].fillna(0)
        
        conditions = [
            (imb > 0.3),
            (imb > 0.05) & (imb <= 0.3),
            (imb < -0.3),
            (imb >= -0.3) & (imb < -0.05)
        ]
        choices = ['Strong Buy', 'Buy Bias', 'Strong Sell', 'Sell Bias']
        df['Regime'] = np.select(conditions, choices, default='Neutral')
        
        self.results['intraday'] = df

    def _compute_shadow_metrics(self):
        """Daily stats for backfill context."""
        df = self._daily.copy()
        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Realized Vol (20d annualized)
        df['RealizedVol'] = df['LogReturn'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # Shadow Pressure Proxy: Return / Vol (Sharp-ish ratio daily)
        df['ShadowPressure'] = df['LogReturn'] / (df['RealizedVol']/100)
        
        self.results['daily'] = df

    def _compute_options_context(self):
        """
        Calculate ATM IV and simple skew from the option chain.
        """
        # Find current spot price (last close)
        spot = self._intraday['Close'].iloc[-1]
        
        # Filter chain for nearest expiry is already done in ingestion
        chain = self._options
        if chain.empty:
            self.results['atm_iv'] = 0
            self.results['skew'] = 0
            return

        # Find ATM Strike
        chain['abs_diff'] = abs(chain['strike'] - spot)
        atm_row = chain.loc[chain['abs_diff'].idxmin()]
        
        # ATM IV (average of call and put IV at this strike if available)
        # Note: yfinance 'impliedVolatility' is a float (e.g. 0.15)
        self.results['atm_iv'] = atm_row['impliedVolatility'] * 100
        
        # Simple Skew: (Put IV 10% OTM - Call IV 10% OTM) / ATM IV
        # This is a rough approx using available data
        put_otm_strike = spot * 0.95
        call_otm_strike = spot * 1.05
        
        try:
            put_iv = chain.loc[(chain['type']=='put') & (abs(chain['strike'] - put_otm_strike).argsort()[:1])]['impliedVolatility'].values[0]
            call_iv = chain.loc[(chain['type']=='call') & (abs(chain['strike'] - call_otm_strike).argsort()[:1])]['impliedVolatility'].values[0]
            self.results['skew'] = (put_iv - call_iv) * 100
        except:
            self.results['skew'] = 0

# --------------------------------------------------------------------------------
# CLASS 3: DASHBOARD RENDERER (Plotly & HTML)
# --------------------------------------------------------------------------------
class DashboardRenderer:
    """
    Generates self-contained HTML with Plotly Offline.
    Handles Tab logic and Resize events via JS injection.
    """
    def __init__(self, analysis_results: Dict, ticker: str, output_path: Path):
        self.data = analysis_results
        self.ticker = ticker
        self.output_path = output_path
        self.figs = {}

    def build_dashboard(self):
        logger.info("Building Plotly Figures...")
        self._build_tab1_tape()
        self._build_tab2_imbalance()
        self._build_tab3_persistence()
        self._build_tab4_options()
        self._build_tab5_shadow()
        self._generate_html()

    def _build_tab1_tape(self):
        df = self.data.get('intraday', pd.DataFrame())
        if df.empty: return

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Price & Microprice
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Last Trade', line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MicroPrice'], name='MicroPrice', line=dict(color='orange', width=1.5)), row=1, col=1)
        
        # Micro-Mid Spread
        fig.add_trace(go.Scatter(x=df.index, y=df['MicroSpread_Bps'], name='Micro-Mid Spread (bps)', 
                                 line=dict(color='cyan', width=1), fill='tozeroy'), row=2, col=1)

        fig.update_layout(title=f"{self.ticker} Intraday Tape vs Microprice", height=600, xaxis_rangeslider_visible=False)
        self.figs['tab1'] = fig

    def _build_tab2_imbalance(self):
        df = self.data.get('intraday', pd.DataFrame())
        if df.empty: return

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        # Filtered Imbalance
        fig.add_trace(go.Bar(x=df.index, y=df['Filtered_Imbalance_Filled'], name='Filtered Imbalance',
                             marker_color=np.where(df['Filtered_Imbalance_Filled']>0, 'green', 'red')), row=1, col=1)
        
        # Threshold lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=-0.3, line_dash="dash", line_color="gray", row=1, col=1)

        # Directional Pressure
        fig.add_trace(go.Scatter(x=df.index, y=df['CumPressure'], name='Cumulative Dir Pressure', line=dict(color='purple')), row=2, col=1)
        
        fig.update_layout(title="Imbalance & Directional Pressure", height=600)
        self.figs['tab2'] = fig

    def _build_tab3_persistence(self):
        df = self.data.get('intraday', pd.DataFrame())
        if df.empty: return

        # Histogram of durations
        durations = df['Persistence_Duration'].dropna()
        
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]])
        
        fig.add_trace(go.Histogram(x=durations, nbinsx=20, name='Persistence Dist'), row=1, col=1)
        
        # Regime Map (Imbalance vs Delta Price)
        fig.add_trace(go.Scatter(
            x=df['Filtered_Imbalance_Filled'], 
            y=df['MicroPrice'].diff(),
            mode='markers',
            marker=dict(
                color=df['Persistence_Duration'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Dur")
            ),
            text=df['Regime'],
            name='Regime Map'
        ), row=1, col=2)

        fig.update_layout(title="Persistence Diagnostics", height=500)
        self.figs['tab3'] = fig

    def _build_tab4_options(self):
        # Regime vs IV dummy chart
        atm_iv = self.data.get('atm_iv', 0)
        skew = self.data.get('skew', 0)
        df = self.data.get('intraday', pd.DataFrame())
        
        if df.empty: return

        # Count Regimes
        regime_counts = df['Regime'].value_counts()
        
        # Correctly define specs for subplots (xy vs domain)
        fig = make_subplots(
            rows=1, cols=2, 
            specs=[[{"type": "xy"}, {"type": "domain"}]]
        )
        
        fig.add_trace(go.Bar(x=regime_counts.index, y=regime_counts.values, name='Regime Freq'), row=1, col=1)
        
        # Gauge for ATM IV
        fig.add_trace(go.Indicator(
            mode = "number+gauge",
            value = atm_iv,
            title = {'text': "ATM IV"},
            domain = {'x': [0.5, 1], 'y': [0, 1]}
        ), row=1, col=2)

        fig.update_layout(title=f"Options Context: Skew approx {skew:.2f}", height=400)
        self.figs['tab4'] = fig

    def _build_tab5_shadow(self):
        df = self.data.get('daily', pd.DataFrame())
        if df.empty: return

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=df.index, y=df['RealizedVol'], name='Realized Vol (20d)', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Bar(x=df.index, y=df['ShadowPressure'], name='Shadow Pressure', marker_color='gray', opacity=0.3), secondary_y=True)

        fig.update_layout(title="Shadow History (Daily Backfill)", height=500)
        self.figs['tab5'] = fig

    def _generate_html(self):
        logger.info("Generating HTML container...")
        
        # Serialize plots to JSON/HTML divs
        # We start empty, then populate with to_html(..., full_html=False)
        divs = {}
        for key, fig in self.figs.items():
            # Standard serialization for all tabs
            # We will handle the JS dependency by embedding it ONCE in the first active tab (or just the header)
            divs[key] = pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'responsive': True})

        # FIX: We need to inject the Plotly JS lib for the charts to work offline.
        # Instead of risky string splitting, we use include_plotlyjs=True on the first chart.
        # This writes ~3MB of JS into the HTML string of the first chart.
        if 'tab1' in self.figs:
            divs['tab1'] = pio.to_html(self.figs['tab1'], include_plotlyjs=True, full_html=False, config={'responsive': True})
        elif 'tab5' in self.figs: # Fallback if tab1 empty
             divs['tab5'] = pio.to_html(self.figs['tab5'], include_plotlyjs=True, full_html=False, config={'responsive': True})
        
        # If absolutely NO charts generated, this will just produce a blank dashboard with buttons, which is fine.

        # HTML Template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Microprice Dashboard: {self.ticker}</title>
            <style>
                body {{ font-family: sans-serif; background: #1e1e1e; color: #ddd; margin: 0; }}
                .tab {{ overflow: hidden; border-bottom: 1px solid #444; background-color: #333; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ddd; font-weight: bold; }}
                .tab button:hover {{ background-color: #555; }}
                .tab button.active {{ background-color: #007bff; color: white; }}
                .tabcontent {{ display: none; padding: 20px; border-top: none; height: 90vh; }}
                h1 {{ padding: 10px; margin: 0; font-size: 1.2rem; }}
            </style>
        </head>
        <body>
            <h1>{self.ticker} Intraday Microprice & Quote Persistence</h1>
            
            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'tab1')" id="defaultOpen">Tape & Microprice</button>
                <button class="tablinks" onclick="openTab(event, 'tab2')">Imbalance</button>
                <button class="tablinks" onclick="openTab(event, 'tab3')">Persistence</button>
                <button class="tablinks" onclick="openTab(event, 'tab4')">Options</button>
                <button class="tablinks" onclick="openTab(event, 'tab5')">Shadow History</button>
            </div>

            <div id="tab1" class="tabcontent">{divs.get('tab1', 'No Data')}</div>
            <div id="tab2" class="tabcontent">{divs.get('tab2', 'No Data')}</div>
            <div id="tab3" class="tabcontent">{divs.get('tab3', 'No Data')}</div>
            <div id="tab4" class="tabcontent">{divs.get('tab4', 'No Data')}</div>
            <div id="tab5" class="tabcontent">{divs.get('tab5', 'No Data')}</div>

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
                    
                    // Trigger Resize for Plotly
                    window.dispatchEvent(new Event('resize'));
                    
                    // Specific Plotly resize call for all plots
                    var plots = document.getElementsByClassName('plotly-graph-div');
                    for (var j=0; j<plots.length; j++) {{
                        Plotly.Plots.resize(plots[j]);
                    }}
                }}
                
                // Open default tab
                document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        with open(self.output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Dashboard saved to {self.output_path}")

# --------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hedge-Fund Grade Intraday Microprice Dashboard")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., SPY)")
    parser.add_argument("--interval", type=str, default="1m", help="Intraday interval (1m, 2m, 5m, 15m)")
    parser.add_argument("--lookback-days", type=int, default=5, help="Days of intraday data")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory for CSVs/HTML")
    parser.add_argument("--open-html", action="store_true", help="Open HTML in browser after generation")
    
    args = parser.parse_args()
    
    logger.info(f"Starting analysis for {args.ticker}...")
    
    # 1. Ingestion
    ingest = DataIngestion(args.ticker, args.interval, args.lookback_days, args.output_dir)
    intraday_df = ingest.fetch_intraday()
    daily_df = ingest._backfill_shadow_history()
    options_df = ingest.fetch_options_chain()
    
    # 2. Analysis
    analysis = FinancialAnalysis(intraday_df, daily_df, options_df)
    analysis.run_analysis()
    
    # 3. Rendering
    output_html = Path(args.output_dir) / f"dashboard_{args.ticker}.html"
    renderer = DashboardRenderer(analysis.results, args.ticker, output_html)
    renderer.build_dashboard()
    
    if args.open_html:
        import webbrowser
        webbrowser.open(f"file://{output_html.resolve()}")

if __name__ == "__main__":
    main()
