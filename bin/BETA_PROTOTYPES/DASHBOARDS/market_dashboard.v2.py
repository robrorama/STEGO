# SCRIPTNAME: ok.04.market_dashboard.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys
import os
import time
import argparse
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------

DEFAULT_INDICES = ['SPY', 'QQQ', 'IWM']

# SPDR Sector ETFs
SECTOR_TICKERS = [
    'XLE', 'XLF', 'XLU', 'XLI', 'XLK', 'XLV', 
    'XLY', 'XLP', 'XLB', 'XLRE', 'XLC'
]

# Dow Jones Industrial Average Components (Historical Snapshot)
DOW_30_TICKERS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

ALL_DEFAULT_TICKERS = list(set(DEFAULT_INDICES + SECTOR_TICKERS + DOW_30_TICKERS))

# -----------------------------------------------------------------------------
# 1. DATA INGESTION CLASS
# -----------------------------------------------------------------------------

class DataIngestion:
    """
    Handles strict disk-first data loading, downloading with rate-limiting,
    and data sanitization.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"[System] Created output directory: {self.output_dir}")

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The Universal Fixer: Standardizes incoming DataFrames from yfinance/CSV.
        """
        # 1. Swap Levels if MultiIndex and Close is in Level 1 (yfinance bug fix)
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels > 1:
                # Check if 'Close' is likely in the second level
                if 'Close' in df.columns.get_level_values(1):
                    df = df.swaplevel(0, 1, axis=1)
            
            # 2. Flatten Columns
            # Join MultiIndex tuples into single strings if they remain
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Join non-empty parts
                    new_cols.append('_'.join([str(c) for c in col if c]).strip())
                else:
                    new_cols.append(str(col))
            df.columns = new_cols

        # 3. Strict Index cleaning
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone info to prevent comparison errors
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Coerce numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def get_data(self, tickers: list, lookback_years: int) -> dict:
        """
        Retrieves data for a list of tickers using the disk-first strategy.
        Returns a dictionary: {ticker: pd.DataFrame}
        """
        data_map = {}
        start_date = datetime.datetime.now() - datetime.timedelta(days=lookback_years * 365)
        print(f"[DataIngestion] Processing {len(tickers)} tickers...")

        for ticker in tickers:
            file_path = os.path.join(self.output_dir, f"{ticker}.csv")
            df = None
            
            # 1. Check Disk
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    df = self._sanitize_df(df)
                    
                    # Verify data isn't empty and covers reasonable range
                    if df.empty or df.index[-1] < (datetime.datetime.now() - datetime.timedelta(days=5)):
                        print(f"[Disk] Stale or empty data for {ticker}. Redownloading...")
                        df = None # Trigger download
                    else:
                        # print(f"[Disk] Loaded {ticker}")
                        pass
                except Exception as e:
                    print(f"[Disk] Error reading {ticker}: {e}. Redownloading...")
                    df = None

            # 2. IF MISSING or Stale: Download
            if df is None:
                print(f"[Web] Downloading {ticker}...")
                
                # CRITICAL: Rate Limiting
                time.sleep(1.0)
                
                try:
                    # Download using yfinance
                    # downloading a single ticker usually returns simple columns, 
                    # but we run sanitize regardless.
                    raw_df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
                    
                    if not raw_df.empty:
                        # Run the universal fixer
                        clean_df = self._sanitize_df(raw_df)
                        
                        # Save to disk
                        clean_df.to_csv(file_path)
                        
                        # Read back to ensure consistency
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        df = self._sanitize_df(df)
                    else:
                        print(f"[Web] No data found for {ticker}")
                except Exception as e:
                    print(f"[Web] Failed to download {ticker}: {e}")

            if df is not None:
                # Filter by date explicitly to match lookback
                mask = (df.index >= pd.Timestamp(start_date))
                data_map[ticker] = df.loc[mask]

        return data_map

# -----------------------------------------------------------------------------
# 2. FINANCIAL ANALYSIS CLASS
# -----------------------------------------------------------------------------

class FinancialAnalysis:
    """
    Pure mathematical logic. No plotting code here.
    """
    def __init__(self, data_map: dict):
        self.data_map = data_map

    def _extract_series(self, col_name: str, tickers: list = None) -> pd.DataFrame:
        """Helper to combine specific columns (Close, Volume) from multiple dfs."""
        combined = pd.DataFrame()
        target_tickers = tickers if tickers else self.data_map.keys()
        
        for t in target_tickers:
            if t in self.data_map:
                df = self.data_map[t]
                # Handle cases where 'Close' might be 'Adj Close' or just 'Close'
                # The sanitize step attempts to fix levels, but column names rely on yfinance default
                if col_name in df.columns:
                    combined[t] = df[col_name]
                elif 'Adj Close' in df.columns and col_name == 'Close':
                    combined[t] = df['Adj Close']
                elif 'Close' in df.columns and col_name == 'Adj Close':
                    combined[t] = df['Close']
                    
        return combined

    def get_close_prices(self, tickers: list = None) -> pd.DataFrame:
        return self._extract_series('Close', tickers)

    def get_volumes(self, tickers: list = None) -> pd.DataFrame:
        return self._extract_series('Volume', tickers)

    def calculate_cumulative_returns(self, tickers: list = None) -> pd.DataFrame:
        """ (Price_t / Price_0) - 1 """
        prices = self.get_close_prices(tickers)
        if prices.empty:
            return pd.DataFrame()
        # Forward fill gaps then drop remaining NaNs at start
        prices = prices.ffill().dropna()
        if prices.empty:
            return pd.DataFrame()
        return (prices / prices.iloc[0]) - 1.0

    def calculate_normalized_prices(self, tickers: list = None) -> pd.DataFrame:
        """ Price_t / Price_0 (Base 1.0) """
        prices = self.get_close_prices(tickers)
        if prices.empty:
            return pd.DataFrame()
        prices = prices.ffill().dropna()
        if prices.empty:
            return pd.DataFrame()
        return prices / prices.iloc[0]

    def calculate_correlations(self, tickers: list = None) -> pd.DataFrame:
        prices = self.get_close_prices(tickers)
        if prices.empty:
            return pd.DataFrame()
        # Daily returns
        daily_rets = prices.pct_change().dropna()
        # Remove assets with zero variance (flat lines)
        daily_rets = daily_rets.loc[:, daily_rets.std() > 0]
        return daily_rets.corr()

    def get_last_price_and_change(self, tickers: list) -> pd.DataFrame:
        """
        Returns DataFrame with columns ['Price', 'PctChange'] for the entire lookback period.
        Used for Treemaps.
        """
        prices = self.get_close_prices(tickers)
        if prices.empty:
            return pd.DataFrame(columns=['Price', 'PctChange'])
            
        results = []
        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) > 0:
                start_p = series.iloc[0]
                end_p = series.iloc[-1]
                pct_chg = (end_p - start_p) / start_p
                results.append({
                    'Ticker': col,
                    'Price': end_p,
                    'PctChange': pct_chg
                })
        
        return pd.DataFrame(results).set_index('Ticker')

# -----------------------------------------------------------------------------
# 3. DASHBOARD RENDERER CLASS
# -----------------------------------------------------------------------------

class DashboardRenderer:
    """
    Handles Plotly visualization and HTML generation using offline JS.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def _get_plotly_js(self):
        """Returns the full offline Plotly JS string."""
        return py_offline.get_plotlyjs()

    def plot_price_history(self, df: pd.DataFrame):
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
        fig.update_layout(title="Asset Price History", xaxis_title="Date", yaxis_title="Price ($)")
        return fig

    def plot_cumulative_returns(self, df: pd.DataFrame):
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
        fig.update_layout(
            title="Cumulative Returns (Lookback Period)", 
            xaxis_title="Date", 
            yaxis_title="Return",
            yaxis_tickformat='.1%'
        )
        return fig

    def plot_normalized_comparison(self, df: pd.DataFrame):
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
        fig.update_layout(
            title="Normalized Performance (Base = 1.0)", 
            xaxis_title="Date", 
            yaxis_title="Relative Value"
        )
        return fig

    def plot_sector_volume(self, df: pd.DataFrame):
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
        fig.update_layout(
            title="Sector Volume (Log Scale)",
            xaxis_title="Date",
            yaxis_title="Volume",
            yaxis_type="log"
        )
        return fig

    def plot_dow_treemap(self, metrics_df: pd.DataFrame):
        """
        metrics_df index=Ticker, columns=[Price, PctChange]
        """
        if metrics_df.empty:
            return go.Figure()

        # Formatting text
        text_labels = [
            f"{t}<br>${p:.2f}<br>{c:.2%}" 
            for t, p, c in zip(metrics_df.index, metrics_df['Price'], metrics_df['PctChange'])
        ]
        
        fig = go.Figure(go.Treemap(
            labels=metrics_df.index,
            parents=["Dow 30"] * len(metrics_df),
            values=metrics_df['Price'],  # Size
            text=text_labels,
            textinfo="text",
            marker=dict(
                colors=metrics_df['PctChange'], # Color
                colorscale='RdBu',
                cmid=0  # Center color scale at 0% change
            )
        ))
        fig.update_layout(title="Dow 30 Components (Size=Price, Color=Return)")
        return fig

    def plot_correlation_heatmap(self, corr_df: pd.DataFrame):
        if corr_df.empty:
            return go.Figure()
            
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(title="Asset Correlation Matrix")
        return fig

    def generate_html_report(self, figures: dict, filename: str = "market_dashboard.html"):
        """
        Generates a standalone HTML file with embedded JS and the specific Tab Resize fix.
        figures: dict { 'div_id': figure_object }
        """
        print("[Renderer] Generating HTML report...")
        
        # 1. Generate Divs
        plot_divs = {}
        for key, fig in figures.items():
            # config={'responsive': True} helps with resizing
            plot_divs[key] = py_offline.plot(fig, include_plotlyjs=False, output_type='div')

        # 2. HTML Template
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quantitative Market Dashboard v2</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; padding: 20px; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        
        /* Tab Styles */
        .tab {{ overflow: hidden; border: 1px solid #333; background-color: #2d2d2d; }}
        .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #aaa; font-size: 16px; font-weight: bold; }}
        .tab button:hover {{ background-color: #444; }}
        .tab button.active {{ background-color: #007acc; color: white; }}
        
        /* Tab Content */
        .tabcontent {{ display: none; padding: 20px; border: 1px solid #333; border-top: none; background-color: #252525; height: 80vh; }}
        
        .chart-container {{ width: 100%; height: 100%; }}
    </style>
    <!-- Offline Plotly JS -->
    <script type="text/javascript">
        {self._get_plotly_js()}
    </script>
</head>
<body>

    <h1>Market Dashboard v2.0</h1>

    <div class="tab">
      <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Overview</button>
      <button class="tablinks" onclick="openTab(event, 'Returns')">Returns Analysis</button>
      <button class="tablinks" onclick="openTab(event, 'Correlations')">Correlations</button>
      <button class="tablinks" onclick="openTab(event, 'Sectors')">Sector Volume</button>
    </div>

    <div id="Overview" class="tabcontent">
        <h3>Price History</h3>
        <div class="chart-container">{plot_divs.get('price_chart', 'No Data')}</div>
    </div>

    <div id="Returns" class="tabcontent">
        <div style="height: 50%; width: 100%;">
            <h3>Cumulative Returns</h3>
            {plot_divs.get('returns_chart', 'No Data')}
        </div>
        <div style="height: 50%; width: 100%;">
             <h3>Normalized Comparison (Base=1.0)</h3>
            {plot_divs.get('norm_chart', 'No Data')}
        </div>
    </div>

    <div id="Correlations" class="tabcontent">
        <div style="height: 50%; width: 100%;">
            <h3>Dow 30 Heatmap</h3>
            {plot_divs.get('dow_treemap', 'No Data')}
        </div>
        <div style="height: 50%; width: 100%;">
             <h3>Correlation Matrix</h3>
            {plot_divs.get('corr_heatmap', 'No Data')}
        </div>
    </div>

    <div id="Sectors" class="tabcontent">
        <h3>Sector Liquidity (Log Scale)</h3>
        <div class="chart-container">{plot_divs.get('sector_vol_chart', 'No Data')}</div>
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
        if (evt) {{
            evt.currentTarget.className += " active";
        }}
        
        // --- CRITICAL TAB RESIZE FIX ---
        // Forces Plotly to redraw when a hidden tab becomes visible
        setTimeout(function() {{
            window.dispatchEvent(new Event('resize'));
        }}, 100);
    }}

    // Open default tab
    document.getElementById("defaultOpen").click();
    </script>

</body>
</html>
        """
        
        full_path = os.path.join(self.output_dir, filename)
        with open(full_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[Renderer] Dashboard saved to: {os.path.abspath(full_path)}")

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Financial Market Dashboard Generator v2")
    parser.add_argument('--tickers', nargs='+', default=[], help="List of specific tickers to analyze. Supports keywords: INDICES, SECTORS, DOW")
    parser.add_argument('--output-dir', type=str, default='./market_data', help="Directory for cache and report")
    parser.add_argument('--lookback', type=int, default=1, help="Years of history")
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help="Risk free rate (decimal)")
    
    args = parser.parse_args()

    # 1. Configuration
    if args.tickers:
        expanded_tickers = []
        for t in args.tickers:
            if t.upper() == 'INDICES':
                expanded_tickers.extend(DEFAULT_INDICES)
            elif t.upper() == 'SECTORS':
                expanded_tickers.extend(SECTOR_TICKERS)
            elif t.upper() == 'DOW':
                expanded_tickers.extend(DOW_30_TICKERS)
            else:
                expanded_tickers.append(t)
        target_tickers = list(set(expanded_tickers))
    else:
        target_tickers = ALL_DEFAULT_TICKERS

    # Determine subsets for specific charts based on the final target_tickers list
    # This logic ensures that if you only ask for SECTORS, the Dow chart is empty, etc.
    dow_tickers = [t for t in target_tickers if t in DOW_30_TICKERS]
    sector_tickers = [t for t in target_tickers if t in SECTOR_TICKERS]
    
    # 2. Ingestion
    ingestion = DataIngestion(args.output_dir)
    data_map = ingestion.get_data(target_tickers, args.lookback)
    
    if not data_map:
        print("No data available. Exiting.")
        return

    # 3. Analysis
    analyst = FinancialAnalysis(data_map)
    
    # Prepare dataframes for charts
    # A. General Prices (Use user provided or default indices)
    # If using custom mode (args.tickers present), use those. Otherwise default indices.
    # We want "main display" to show whatever the user requested.
    main_display_tickers = target_tickers if args.tickers else DEFAULT_INDICES
    
    df_prices = analyst.get_close_prices(main_display_tickers)
    
    # B. Returns
    df_cum_returns = analyst.calculate_cumulative_returns(main_display_tickers)
    
    # C. Normalized (Use same as main display)
    df_norm = analyst.calculate_normalized_prices(main_display_tickers)
    
    # D. Dow 30 Treemap Data
    df_dow_metrics = analyst.get_last_price_and_change(dow_tickers)
    
    # E. Correlations (All available data)
    df_corr = analyst.calculate_correlations(list(data_map.keys()))
    
    # F. Sector Volume
    df_sector_vol = analyst.get_volumes(sector_tickers)

    # 4. Rendering
    renderer = DashboardRenderer(args.output_dir)
    
    figures = {
        'price_chart': renderer.plot_price_history(df_prices),
        'returns_chart': renderer.plot_cumulative_returns(df_cum_returns),
        'norm_chart': renderer.plot_normalized_comparison(df_norm),
        'dow_treemap': renderer.plot_dow_treemap(df_dow_metrics),
        'corr_heatmap': renderer.plot_correlation_heatmap(df_corr),
        'sector_vol_chart': renderer.plot_sector_volume(df_sector_vol)
    }
    
    renderer.generate_html_report(figures)

if __name__ == "__main__":
    main()
