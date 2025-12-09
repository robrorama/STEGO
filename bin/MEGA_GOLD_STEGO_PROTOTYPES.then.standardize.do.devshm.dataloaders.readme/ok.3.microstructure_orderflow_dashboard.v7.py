#!/usr/bin/env python3
# SCRIPTNAME: production_market_dashboard.py
# ROLE: Senior Quantitative Developer Implementation
# DESCRIPTION: Standalone market dashboard with synthetic order flow and volatility cones.

import argparse
import datetime as dt
import logging
import os
import sys
import time
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio  # <--- FIXED: Added missing import
from plotly.subplots import make_subplots
import plotly.offline as py_offline
import webbrowser

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# CLASS 1: DataIngestion (I/O, API, Sanitization)
# ------------------------------------------------------------------------------
class DataIngestion:
    def __init__(self, ticker: str, days: int = 365):
        self.ticker = ticker.upper()
        self.days = days
        self.filename = f"{self.ticker}_data.csv"

    def get_data(self) -> pd.DataFrame:
        """
        Orchestrates the fetch or load process.
        """
        if os.path.exists(self.filename):
            logger.info(f"Local cache found for {self.ticker}. Loading from CSV...")
            try:
                df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
                # Run sanitization even on loaded CSV to ensure types/formats
                return self._sanitize_df(df)
            except Exception as e:
                logger.error(f"Failed to load CSV: {e}. Falling back to download.")

        return self._fetch_and_save()

    def _fetch_and_save(self) -> pd.DataFrame:
        """
        Fetches data via yfinance, sanitizes, and persists to disk.
        """
        logger.info(f"Fetching {self.days} days of data for {self.ticker} from yfinance...")
        
        # Rate Limiting
        time.sleep(1)

        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=self.days)
        
        # Download
        df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)

        if df is None or df.empty:
            logger.warning("Download returned empty DataFrame. Attempting Shadow Backfill.")
            df = self._backfill_shadow_history()

        df = self._sanitize_df(df)
        
        # Save to CSV
        try:
            df.to_csv(self.filename)
            logger.info(f"Data saved to {self.filename}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

        return df

    def _backfill_shadow_history(self) -> pd.DataFrame:
        """
        Cold Start / Shadow Backfill: Force a 1-year download if primary fails or is empty.
        """
        logger.info("Triggering Shadow Backfill (1 year OHLC)...")
        time.sleep(1)
        return yf.download(self.ticker, period="1y", progress=False)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Critical sanitization: Fixes MultiIndex swap levels and enforces types.
        """
        if df.empty:
            raise ValueError(f"No data available for {self.ticker}")

        # --- The "Swap Levels" Fix ---
        # yfinance often returns MultiIndex columns [('Adj Close', 'AAPL'), ...]
        # or sometimes [('AAPL', 'Adj Close')]. We need 'Adj Close' on top level (0).
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' or 'Adj Close' is in Level 0
            level_0_vals = df.columns.get_level_values(0).unique()
            if 'Close' not in level_0_vals and 'Adj Close' not in level_0_vals:
                # If not in Level 0, assume they are in Level 1 and swap
                logger.info("Detected inverted MultiIndex. Swapping levels...")
                df = df.swaplevel(0, 1, axis=1)
            
            # Drop the Ticker level to flatten columns to ['Open', 'High', ...]
            # We assume the level containing 'Close' is now 0.
            # We drop level 1 which should be the ticker name.
            try:
                df.columns = df.columns.droplevel(1)
            except IndexError:
                pass # Already flat or issue with levels

        # Ensure standard column names
        df.columns = [c.capitalize() for c in df.columns] # Open, High, Low, Close...

        # Strict Typing: Datetime Index
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        # Strict Typing: Numerics
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df.dropna()

# ------------------------------------------------------------------------------
# CLASS 2: FinancialAnalysis (Math, Volatility, GEX, Order Flow)
# ------------------------------------------------------------------------------
class FinancialAnalysis:
    def __init__(self, df: pd.DataFrame):
        # Immutability: Store raw, never modify in place
        self._raw_data = df

    def get_ohlcv(self) -> pd.DataFrame:
        return self._raw_data.copy()

    def calculate_volatility_cone(self) -> pd.DataFrame:
        """
        Calculates Realized Volatility cones (Min, Max, Avg, Current).
        """
        df = self._raw_data.copy()
        
        # Log Returns
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Annualized Realized Volatility Windows
        windows = [10, 30, 60, 90]
        results = {}

        for w in windows:
            # Annualize factor: sqrt(252)
            # Use .iloc[-1] to get current scalar safely
            roll_std = df['log_ret'].rolling(window=w).std() * np.sqrt(252)
            
            current_vol = roll_std.iloc[-1] if not roll_std.empty else 0
            
            # Historic stats for the cone
            min_vol = roll_std.min()
            max_vol = roll_std.max()
            avg_vol = roll_std.mean()
            
            results[w] = {
                'Current': current_vol,
                'Min': min_vol,
                'Max': max_vol,
                'Avg': avg_vol
            }
            
        return pd.DataFrame(results).T # Transpose for easy plotting

    def calculate_shadow_gex(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Estimates "Shadow GEX" levels.
        Proxy: (Neutral Vol - Realized Vol) * Volume
        """
        df = self._raw_data.copy()
        
        # Simple Realized Vol (20d)
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rv'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
        
        # Assume a "Neutral" IV baseline (e.g., average of recent RV)
        # Handle scalar extraction safely
        neutral_vol = df['rv'].mean()
        if isinstance(neutral_vol, pd.Series):
             neutral_vol = neutral_vol.iloc[0]

        # Shadow Gamma Proxy calculation
        # If RV < Neutral, we assume dealers are long gamma (suppressing vol).
        # We weight this by Volume to find significant price levels.
        df['shadow_gex'] = (neutral_vol - df['rv']) * df['Volume']
        
        # Find levels with highest Shadow GEX accumulation
        # Group by price buckets (rounded integers)
        df['price_bucket'] = df['Close'].round(0)
        levels = df.groupby('price_bucket')['shadow_gex'].sum().sort_values(ascending=False).head(5)
        
        return df['shadow_gex'], levels

    def build_synthetic_order_flow(self, num_bins: int = 50) -> Dict:
        """
        Refactored synthetic logic for DOM and Buying Pressure.
        """
        df = self._raw_data.copy()
        
        # We simulate intraday-like structure from daily data for the sandbox
        # Create synthetic time steps
        t = np.linspace(0, len(df), len(df))
        
        # Synthetic Order Book Depth (DOM)
        # Using Price vs Time heatmap data
        # Simply using Close price distribution over time
        
        # Calculate Buying Pressure (Aggressor Buy vs Sell)
        # Proxy: (Close - Open) / (High - Low) scaled by Volume
        # If Close > Open, more buying pressure.
        
        range_len = (df['High'] - df['Low'])
        range_len = range_len.replace(0, 0.01) # Avoid div by zero
        
        df['buy_pressure'] = ((df['Close'] - df['Open']) / range_len) * df['Volume']
        
        # DOM Heatmap simulation: Gaussian smear around Close price
        # This is purely for visualization in this sandbox context
        heatmap_data = []
        
        # Return necessary series
        return {
            'dates': df.index,
            'price': df['Close'],
            'buy_pressure': df['buy_pressure'],
            'volume': df['Volume']
        }

# ------------------------------------------------------------------------------
# CLASS 3: DashboardRenderer (HTML, JS, Plotly)
# ------------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self, ticker: str, data_analysis: FinancialAnalysis):
        self.ticker = ticker
        self.analysis = data_analysis
        self.output_file = f"{ticker}_dashboard.html"

    def render(self):
        """
        Generates the HTML dashboard.
        """
        # 1. Prepare Data
        df = self.analysis.get_ohlcv()
        vol_cone_df = self.analysis.calculate_volatility_cone()
        shadow_gex_series, gex_levels = self.analysis.calculate_shadow_gex()
        order_flow = self.analysis.build_synthetic_order_flow()

        # ---------------------------------------------------------
        # TAB 1: GEX Proxy & Price Action
        # ---------------------------------------------------------
        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                             vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Candlestick
        fig1.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC'
        ), row=1, col=1)

        # Shadow GEX Levels (Horizontal Rectangles)
        for i, (price, strength) in enumerate(gex_levels.items()):
            # Plot a line or shade region
            fig1.add_hrect(y0=price*0.99, y1=price*1.01, 
                           fillcolor="green", opacity=0.1, 
                           annotation_text=f"GEX Level {price}", annotation_position="right",
                           row=1, col=1)

        # Volume
        fig1.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='grey'), row=2, col=1)
        
        fig1.update_layout(title=f"{self.ticker} - Price Action & Shadow GEX Levels", 
                           xaxis_rangeslider_visible=False, template="plotly_dark")

        # ---------------------------------------------------------
        # TAB 2: Synthetic Order Flow (Heatmap & Pressure)
        # ---------------------------------------------------------
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        
        # For Sandbox: Plot Close Price as a Line for context
        fig2.add_trace(go.Scatter(x=order_flow['dates'], y=order_flow['price'], mode='lines', name='Price'), row=1, col=1)
        
        # Buying Pressure
        colors = np.where(order_flow['buy_pressure'] > 0, 'green', 'red')
        fig2.add_trace(go.Bar(x=order_flow['dates'], y=order_flow['buy_pressure'], 
                              name='Net Aggressor Vol', marker_color=colors), row=2, col=1)
        
        fig2.update_layout(title="Synthetic Order Flow & Buying Pressure", template="plotly_dark")

        # ---------------------------------------------------------
        # TAB 3: Volatility Cone
        # ---------------------------------------------------------
        fig3 = go.Figure()
        
        # X-Axis: Lookback Windows (converted to string for categorical plotting)
        x_vals = [str(x) for x in vol_cone_df.index]
        
        fig3.add_trace(go.Scatter(x=x_vals, y=vol_cone_df['Max'], mode='lines+markers', name='Max Vol', line=dict(dash='dash', color='red')))
        fig3.add_trace(go.Scatter(x=x_vals, y=vol_cone_df['Avg'], mode='lines+markers', name='Avg Vol', line=dict(color='yellow')))
        fig3.add_trace(go.Scatter(x=x_vals, y=vol_cone_df['Min'], mode='lines+markers', name='Min Vol', line=dict(dash='dash', color='green')))
        
        fig3.add_trace(go.Scatter(x=x_vals, y=vol_cone_df['Current'], mode='lines+markers', name='Current RV', 
                                  line=dict(color='white', width=4), marker=dict(size=12)))

        fig3.update_layout(title="Volatility Cone (Min/Max/Avg vs Current)", 
                           xaxis_title="Lookback Window (Days)", yaxis_title="Annualized Volatility",
                           template="plotly_dark")

        # ---------------------------------------------------------
        # TAB 4: Microstructure Tiles (Text/Indicator)
        # ---------------------------------------------------------
        # Calculating summary stats safely (handling NaNs)
        last_close = df['Close'].iloc[-1]
        last_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].mean()
        rel_vol = last_vol / avg_vol if avg_vol != 0 else 0
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Indicator(
            mode = "number+delta",
            value = last_close,
            title = {"text": "Last Price"},
            delta = {'reference': df['Close'].iloc[-2], 'relative': False},
            domain = {'row': 0, 'column': 0}))

        fig4.add_trace(go.Indicator(
            mode = "number",
            value = rel_vol,
            title = {"text": "Relative Volume"},
            domain = {'row': 0, 'column': 1}))
            
        fig4.update_layout(
            grid = {'rows': 1, 'columns': 2, 'pattern': "independent"},
            template="plotly_dark"
        )

        # ---------------------------------------------------------
        # HTML GENERATION (The "Offline" & "Tab" Fixes)
        # ---------------------------------------------------------
        
        # 1. Get Div strings
        div1 = pio.to_html(fig1, full_html=False, include_plotlyjs=False, div_id="chart1")
        div2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False, div_id="chart2")
        div3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False, div_id="chart3")
        div4 = pio.to_html(fig4, full_html=False, include_plotlyjs=False, div_id="chart4")

        # 2. Get Offline JS library
        plotly_js = py_offline.get_plotlyjs()

        # 3. Construct HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.ticker} Market Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: sans-serif; background-color: #1e1e1e; color: #ddd; margin: 0; }}
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #2e2e2e; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-size: 17px; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #555; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border-top: none; height: 90vh; }}
                h1 {{ padding: 10px; margin: 0; background-color: #111; }}
            </style>
        </head>
        <body>

        <h1>{self.ticker} Quant Dashboard</h1>

        <div class="tab">
            <button class="tablinks" onclick="openTab(event, 'Tab1')" id="defaultOpen">Price & GEX</button>
            <button class="tablinks" onclick="openTab(event, 'Tab2')">Order Flow</button>
            <button class="tablinks" onclick="openTab(event, 'Tab3')">Vol Cone</button>
            <button class="tablinks" onclick="openTab(event, 'Tab4')">Microstructure</button>
        </div>

        <div id="Tab1" class="tabcontent"> {div1} </div>
        <div id="Tab2" class="tabcontent"> {div2} </div>
        <div id="Tab3" class="tabcontent"> {div3} </div>
        <div id="Tab4" class="tabcontent"> {div4} </div>

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
            
            // THE TAB FIX: Trigger resize to render hidden plotly charts
            window.dispatchEvent(new Event('resize'));
        }}
        
        // Open default tab
        document.getElementById("defaultOpen").click();
        </script>

        </body>
        </html>
        """

        with open(self.output_file, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard generated: {self.output_file}")
        webbrowser.open(f"file://{os.path.abspath(self.output_file)}")

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Options Trading Dashboard")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., SPY, NVDA)")
    args = parser.parse_args()

    try:
        # 1. Ingest
        ingestion = DataIngestion(args.ticker)
        df = ingestion.get_data()

        # 2. Analyze
        analysis = FinancialAnalysis(df)

        # 3. Render
        renderer = DashboardRenderer(args.ticker, analysis)
        renderer.render()

    except Exception as e:
        logger.error(f"Fatal Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
