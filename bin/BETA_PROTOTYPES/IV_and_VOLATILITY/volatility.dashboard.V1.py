# SCRIPTNAME: ok.3.volatility.dashboard.V1.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py_offline
import webbrowser
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. DATA INGESTION LAYER
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Handles robust fetching of financial data, schema normalization,
    and local caching. Enforces strict column formatting to handle
    yfinance API inconsistencies.
    """
    def __init__(self, tickers, start_date=None):
        self.tickers = tickers
        # Default to 2 years lookback if not provided
        if start_date is None:
            self.start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
        
        self.cache_file = 'market_data_cache.csv'

    def fetch_data(self):
        """
        Orchestrates the data loading process. Tries local cache first,
        falls back to API download if missing or stale.
        """
        if os.path.exists(self.cache_file):
            print("[System] Local cache found. Loading...")
            try:
                df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
                # Verify freshness (simple check: is last date recent?)
                last_date = df.index[-1]
                if (datetime.now() - last_date).days > 1:
                    print("[System] Cache stale. Triggering Shadow Backfill...")
                    return self._backfill_shadow_history()
                return df
            except Exception as e:
                print(f"[Error] Corrupt cache: {e}. Re-downloading.")
                return self._backfill_shadow_history()
        else:
            print("[System] No cache found. Triggering Cold Start Backfill...")
            return self._backfill_shadow_history()

    def _backfill_shadow_history(self):
        """
        Downloads data from yfinance, normalizes the MultiIndex mess,
        and flattens to a clean DataFrame.
        """
        print(f"[Network] Downloading {self.tickers} from {self.start_date}...")
        
        # FIX A: Robust Column Parsing (Group by column)
        raw_df = yf.download(
            self.tickers, 
            start=self.start_date, 
            group_by='column', 
            auto_adjust=False, # We want Close for standard calculation, Adj Close for returns if preferred
            progress=False
        )

        if raw_df.empty:
            raise ValueError("Download failed. Check internet connection or tickers.")

        # FIX A (Cont): MultiIndex Inspection and Swap
        # yfinance output structure changes. We ensure (Attribute, Ticker) format before flattening
        if isinstance(raw_df.columns, pd.MultiIndex):
            # Check level 0 for Attributes like 'Close' or 'Open'
            # If Level 0 contains the Tickers (e.g., 'SPY'), we must swap.
            level_0_vals = raw_df.columns.get_level_values(0).unique()
            common_attrs = {'Close', 'Adj Close', 'Open', 'High', 'Low', 'Volume'}
            
            # If no intersection between level 0 and attributes, assume headers are swapped
            if not common_attrs.intersection(set(level_0_vals)):
                print("[Data] Detected Ticker-First MultiIndex. Swapping levels...")
                raw_df = raw_df.swaplevel(0, 1, axis=1)

        # Flatten columns to clean format: "Ticker_Attribute" -> e.g. "SPY_Close"
        # Note: If single ticker, yf might not return MultiIndex. We force handling.
        if isinstance(raw_df.columns, pd.MultiIndex):
            new_cols = []
            for col in raw_df.columns:
                # col is (Attribute, Ticker) after normalization
                attr, ticker = col
                new_cols.append(f"{ticker}_{attr}")
            raw_df.columns = new_cols
        else:
            # Handle single ticker edge case if necessary, though list input prevents this usually
            pass

        # Timezone sanitization
        if raw_df.index.tz is not None:
            raw_df.index = raw_df.index.tz_localize(None)

        # Save to cache
        raw_df.to_csv(self.cache_file)
        print("[System] Data ingestion complete. Cache updated.")
        return raw_df


# -----------------------------------------------------------------------------
# 2. FINANCIAL ANALYSIS LAYER
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Performs vector calculations for Volatility, Gamma Exposure proxies,
    and Micro-structure analysis. Enforces immutability.
    """
    def __init__(self, raw_data):
        # Immutable source of truth
        self._raw_data = raw_data.copy()
        self.metrics = pd.DataFrame(index=self._raw_data.index)

    def perform_analysis(self):
        """Main driver for all calculations."""
        self._calculate_returns_and_vol()
        self._calculate_gamma_regime()
        self._calculate_correlations()
        self._calculate_microstructure()
        return self.metrics

    def _calculate_returns_and_vol(self):
        """Calculates Returns, RV cones, and IV spreads."""
        df = self._raw_data.copy()
        
        # Prepare Spot and Vol keys (assuming SPY and ^VIX for this demo)
        spot_col = 'SPY_Close'
        vol_col = '^VIX_Close'
        
        # Log Returns
        self.metrics['Spot_Price'] = df[spot_col]
        self.metrics['Vol_Index'] = df[vol_col]
        self.metrics['Log_Ret'] = np.log(df[spot_col] / df[spot_col].shift(1))

        # Realized Volatility Cone (Annualized)
        windows = [10, 30, 60, 90]
        for w in windows:
            # Stdev of log returns * sqrt(252) * 100 for percentage
            self.metrics[f'RV_{w}'] = self.metrics['Log_Ret'].rolling(window=w).std() * np.sqrt(252) * 100

        # Rich/Cheap Indicator (IV - RV30)
        # Using VIX as IV proxy
        self.metrics['Vol_Risk_Premium'] = self.metrics['Vol_Index'] - self.metrics[f'RV_30']

    def _calculate_gamma_regime(self):
        """
        Estimates Gamma Exposure (GEX) Regime using a Dealer Model Proxy.
        Logic: Dealers are Long Gamma (stabilizing) when Spot > Major Moving Average (20d),
        and Short Gamma (accelerating) when Spot < Major Moving Average.
        """
        spot = self.metrics['Spot_Price']
        ma_20 = spot.rolling(window=20).mean()
        
        # +1 for Positive Gamma (Green Zone), -1 for Negative Gamma (Red Zone)
        self.metrics['Gamma_Regime'] = np.where(spot > ma_20, 1, -1)
        self.metrics['Spot_MA_20'] = ma_20

    def _calculate_correlations(self):
        """Calculates rolling correlation for Tail Risk analysis."""
        # Rolling correlation between Spot Returns and VIX changes
        # When Spot drops, VIX usually pops (Negative Corr).
        # A breakdown (Corr -> 0 or Positive) indicates market fragility or rotation.
        
        spot_ret = self.metrics['Log_Ret']
        vix_ret = np.log(self.metrics['Vol_Index'] / self.metrics['Vol_Index'].shift(1))
        
        self.metrics['Spot_Vol_Corr'] = spot_ret.rolling(window=20).corr(vix_ret)

    def _calculate_microstructure(self):
        """
        Simulates Liquidity Gap analysis using Daily Range vs Volume.
        """
        df = self._raw_data.copy()
        
        high = df['SPY_High']
        low = df['SPY_Low']
        close = df['SPY_Close']
        prev_close = close.shift(1)
        volume = df['SPY_Volume']

        # True Range Calculation
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # Element-wise max
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        self.metrics['True_Range'] = true_range
        self.metrics['Volume'] = volume
        
        # Identify "Liquidity Holes": High Range on Low Volume
        # We define Low Volume as < 20th percentile, High Range as > 80th percentile
        vol_thresh = volume.rolling(window=20).quantile(0.20)
        range_thresh = true_range.rolling(window=20).quantile(0.80)
        
        self.metrics['Liquidity_Gap'] = (volume < vol_thresh) & (true_range > range_thresh)


# -----------------------------------------------------------------------------
# 3. DASHBOARD RENDERER LAYER
# -----------------------------------------------------------------------------
class DashboardRenderer:
    """
    Generates the HTML dashboard.
    Crucial: Injects Plotly JS offline and handles the Tab-Resize bug.
    """
    def __init__(self, metrics_df):
        self.df = metrics_df.dropna() # Drop NaN for clean plotting
        self.template = "plotly_dark"

    def generate_dashboard(self, filename="options_dashboard.html"):
        print("[Rendering] Generating High-Performance Visuals...")
        
        # Generate Figures
        fig_gamma = self._create_gamma_vise()
        fig_cone = self._create_vol_cone()
        fig_tail = self._create_tail_heatmap()
        fig_micro = self._create_liquidity_structure()

        # Generate HTML components
        # FIX B: Offline Plotly Injection (Get full JS string)
        plotly_js = py_offline.get_plotlyjs()
        
        # Get Div strings (include_plotlyjs=False because we inject it globally)
        div_gamma = py_offline.plot(fig_gamma, include_plotlyjs=False, output_type='div')
        div_cone = py_offline.plot(fig_cone, include_plotlyjs=False, output_type='div')
        div_tail = py_offline.plot(fig_tail, include_plotlyjs=False, output_type='div')
        div_micro = py_offline.plot(fig_micro, include_plotlyjs=False, output_type='div')

        # Assemble HTML with FIX C (Resize Bug)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quant Volatility Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; padding: 20px; }}
                h1 {{ color: #00d4ff; text-align: center; font-weight: 300; letter-spacing: 2px; }}
                .tab {{ overflow: hidden; border-bottom: 1px solid #333; margin-bottom: 20px; }}
                .tab button {{ background-color: #2b2b2b; color: #888; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 15px; }}
                .tab button:hover {{ background-color: #383838; color: #fff; }}
                .tab button.active {{ background-color: #007bff; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; background-color: #252525; height: 80vh; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>

            <h1>ALPHA VOLATILITY TERMINAL</h1>

            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Gamma')" id="defaultOpen">Gamma Vise</button>
                <button class="tablinks" onclick="openTab(event, 'VolCone')">Volatility Cone</button>
                <button class="tablinks" onclick="openTab(event, 'TailRisk')">Tail Risk Map</button>
                <button class="tablinks" onclick="openTab(event, 'Micro')">Micro-Structure</button>
            </div>

            <div id="Gamma" class="tabcontent">
                {div_gamma}
            </div>

            <div id="VolCone" class="tabcontent">
                {div_cone}
            </div>

            <div id="TailRisk" class="tabcontent">
                {div_tail}
            </div>

            <div id="Micro" class="tabcontent">
                {div_micro}
            </div>

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
                    
                    // FIX C: THE TAB-RESIZE BUG FIX
                    // Dispatch a resize event to force Plotly to redraw within the new container dimensions
                    window.dispatchEvent(new Event('resize'));
                }}
                
                // Open default tab
                document.getElementById("defaultOpen").click();
            </script>

        </body>
        </html>
        """

        with open(filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[System] Dashboard saved to {os.path.abspath(filename)}")
        webbrowser.open('file://' + os.path.abspath(filename))

    def _create_gamma_vise(self):
        """Tab 1: Spot Price with Dealer Gamma Zones."""
        fig = go.Figure()
        
        # Spot Price
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['Spot_Price'],
            mode='lines', name='SPY Spot',
            line=dict(color='white', width=1.5)
        ))
        
        # Add background colors based on Regime
        # Logic: Iterate through regimes and add shapes. 
        # Optimization: We create a scatter plot with 'fill' to simulate bands if complex,
        # but for clean vertical zones, we analyze state changes.
        
        # To keep it performant for Plotly, we plot the MA, then color the area between lines?
        # Better: Visual hack -> Plot the Spot Price line, but color it? 
        # No, prompt asks for background zones.
        
        # We will map the regime to a heat-style scatter behind the line
        # 1 (Green) = Long Gamma, -1 (Red) = Short Gamma
        
        # Scaled visualization: We'll create a colored bar at the bottom
        fig.add_trace(go.Heatmap(
            x=self.df.index,
            y=[self.df['Spot_Price'].min() * 0.98] * len(self.df), # Position at bottom
            z=self.df['Gamma_Regime'],
            colorscale=[[0, 'red'], [1, 'green']],
            showscale=False,
            name='Gamma Regime'
        ))

        # Add Moving Average
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['Spot_MA_20'],
            mode='lines', name='Dealer Gamma Flip (20MA)',
            line=dict(color='#00d4ff', dash='dot', width=1)
        ))

        fig.update_layout(
            title="Dealer Gamma Exposure Proxy (Net Gamma Regime)",
            template=self.template,
            xaxis_title="Date", yaxis_title="Price",
            height=700
        )
        return fig

    def _create_vol_cone(self):
        """Tab 2: RV Cone + Rich/Cheap Subplot."""
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.05)
        
        # RV Lines
        colors = ['#ffe100', '#ff9d00', '#ff5900', '#ff0000']
        for i, w in enumerate([10, 30, 60, 90]):
            fig.add_trace(go.Scatter(
                x=self.df.index, y=self.df[f'RV_{w}'],
                mode='lines', name=f'Realized Vol ({w}d)',
                line=dict(color=colors[i], width=1)
            ), row=1, col=1)

        # Implied Vol (VIX)
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['Vol_Index'],
            mode='lines', name='Implied Vol (VIX)',
            line=dict(color='#00ffcc', width=2)
        ), row=1, col=1)

        # Spread Subplot
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['Vol_Risk_Premium'],
            mode='lines', name='VRP (IV - RV30)',
            fill='tozeroy',
            line=dict(color='#a6a6a6', width=1)
        ), row=2, col=1)

        fig.update_layout(
            title="Volatility Cone & Risk Premium Structure",
            template=self.template,
            height=700
        )
        return fig

    def _create_tail_heatmap(self):
        """Tab 3: Rolling Correlation Heatmap (Spot vs Vol)."""
        # For a true heatmap feel in 2D, we need multiple windows or just the 1D line mapped to color.
        # Let's create a Heatmap where Y axis is Lookback Window and X is Date.
        # This is computationally heavier but visually superior.
        
        windows = list(range(10, 95, 5))
        heatmap_data = []
        
        spot_ret = self.df['Log_Ret']
        vix_ret = np.log(self.df['Vol_Index'] / self.df['Vol_Index'].shift(1))

        for w in windows:
            # Calculate rolling corr for this window
            col = spot_ret.rolling(window=w).corr(vix_ret)
            heatmap_data.append(col.values)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=self.df.index,
            y=windows,
            colorscale='RdBu', # Red = Positive (Bad for hedges), Blue = Negative (Normal)
            zmid=0,
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="Tail Risk Matrix: Spot-Vol Correlation Term Structure",
            template=self.template,
            xaxis_title="Date", yaxis_title="Rolling Window (Days)",
            height=700
        )
        return fig

    def _create_liquidity_structure(self):
        """Tab 4: Intraday Micro-structure (Liquidity Holes)."""
        
        # Base Scatter
        fig = go.Figure()
        
        # Normal Days
        normal = self.df[~self.df['Liquidity_Gap']]
        fig.add_trace(go.Scatter(
            x=normal['Volume'], y=normal['True_Range'],
            mode='markers', name='Normal Liquidity',
            marker=dict(color='#444', size=5, opacity=0.6)
        ))
        
        # Liquidity Gaps
        gaps = self.df[self.df['Liquidity_Gap']]
        fig.add_trace(go.Scatter(
            x=gaps['Volume'], y=gaps['True_Range'],
            mode='markers', name='Liquidity Holes (Gap Risk)',
            marker=dict(color='#ff0044', size=10, symbol='x')
        ))

        fig.update_layout(
            title="Micro-Structure: Liquidity Gap Detection",
            template=self.template,
            xaxis_title="Volume", yaxis_title="True Range",
            height=700
        )
        return fig


# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- STARTING QUANT PIPELINE ---")
    
    # 1. Ingest
    # Using SPY (Market) and ^VIX (Vol)
    ingestor = DataIngestion(tickers=['SPY', '^VIX'])
    raw_df = ingestor.fetch_data()
    
    # 2. Analyze
    analyst = FinancialAnalysis(raw_df)
    processed_df = analyst.perform_analysis()
    
    # 3. Render
    renderer = DashboardRenderer(processed_df)
    renderer.generate_dashboard()
    
    print("--- PIPELINE COMPLETE ---")
