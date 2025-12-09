# SCRIPTNAME: ok.4.hedge_fund_dashboard.V2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
hedge_fund_dashboard_v1.py

Principal Quantitative Software Engineer: Greenfield Implementation
Market Regime & Correlation Analytics Engine

Architecture:
1. DataIngestion: Robust I/O, 'Universal Fixer' for yfinance, Smart Caching.
2. QuantEngine: PCA, Rolling Stats, Partial Correlations, Lag/Lead.
3. DashboardRenderer: Offline Plotly, Dark Mode, JS Injection.

Usage:
    python hedge_fund_dashboard_v1.py [--tickers SPY,TLT,...] [--lookback 5] [--update-cache]
"""

import os
import time
import argparse
import logging
import datetime
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

# Analytics
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Visualization
import plotly.graph_objs as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots
import plotly.express as px

# -----------------------------------------------------------------------------
# LOGGING & CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [QuantEngine] - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
warnings.simplefilter(action='ignore', category=FutureWarning)

CACHE_FILE = "cache_market_data.csv"
HTML_OUTPUT = "Market_Regime_Dashboard.html"

# Default Tickers by Module
DEFAULT_TICKERS = {
    'macro': ['SPY', 'TLT', 'TIP', 'HYG'],
    'sectors': ['XLY', 'XLE', 'XLK', 'XLF', 'XLV', 'XLI', 'XLB', 'XLP', 'XLU', 'XLRE', 'XLC'],
    'commodities': ['GC=F', 'SI=F', 'CL=F', 'XLE'],
    'fx_liquidity': ['EURUSD=X', 'JPY=X', 'DX-Y.NYB', 'BTC-USD']
}
FRED_API_KEY = os.getenv("FRED_API_KEY")  # Optional

# -----------------------------------------------------------------------------
# CLASS 1: DATA INGESTION (The "Universal Fixer" & Smart Cache)
# -----------------------------------------------------------------------------
class DataIngestion:
    def __init__(self, force_update: bool = False):
        self.force_update = force_update
        self.fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The 'Universal Fixer': Handles yfinance multi-index fragility and strict typing.
        """
        if df.empty:
            return df

        # 1. Swap Levels Logic (Legacy fix, though the loop method simplifies this structure significantly)
        if isinstance(df.columns, pd.MultiIndex):
            target_col = 'Adj Close' if 'Adj Close' in df.columns.get_level_values(0) or 'Adj Close' in df.columns.get_level_values(1) else 'Close'
            if target_col in df.columns.get_level_values(0):
                df = df[target_col]
            elif target_col in df.columns.get_level_values(1):
                df = df.swaplevel(0, 1, axis=1)
                df = df[target_col]

        # 2. Flattening: Ensure columns are simple strings
        df.columns = [str(c).upper().strip() for c in df.columns]

        # 3. Strict Typing & Timezone Removal
        df = df.apply(pd.to_numeric, errors='coerce')
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        # Sort index to be sure
        df.sort_index(inplace=True)
        return df

    def _cold_start_protection(self, df: pd.DataFrame, expected_tickers: List[str]) -> pd.DataFrame:
        """
        Generates 'Shadow' datasets (flat line) for failed tickers to prevent crash.
        """
        for ticker in expected_tickers:
            norm_ticker = ticker.upper().strip()
            if norm_ticker not in df.columns:
                logger.warning(f"Cold Start Protection: Generating shadow data for {norm_ticker}")
                # Fill with 1.0 or last valid if completely missing
                df[norm_ticker] = 1.0
            else:
                # Forward fill then backward fill to handle gaps
                df[norm_ticker] = df[norm_ticker].ffill().bfill()
        return df

    def fetch_market_data(self, tickers: List[str], years: int = 5) -> pd.DataFrame:
        """
        Smart Caching Logic: Check local -> Valid? Load. Else -> Download -> Sanitize -> Save.
        STAGGERED DOWNLOAD: 1 request per second strictly enforced.
        """
        now = datetime.datetime.now()
        start_date = now - datetime.timedelta(days=years*365)
        
        # Check Cache
        if not self.force_update and os.path.exists(CACHE_FILE):
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
            if (now - mod_time).total_seconds() < 86400: # 24 hours
                logger.info("Loading data from local cache...")
                df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
                return self._cold_start_protection(df, tickers)

        logger.info(f"Downloading new data for {len(tickers)} tickers (Staggered 1/sec)...")
        
        frames = []
        for ticker in tickers:
            norm_ticker = ticker.upper().strip()
            try:
                # Download strictly ONE ticker
                logger.info(f"Requesting {norm_ticker}...")
                df_tick = yf.download(ticker, start=start_date, end=now, progress=False, threads=False)
                
                if not df_tick.empty:
                    # Isolate Close/Adj Close immediately
                    target_col = 'Adj Close' if 'Adj Close' in df_tick.columns else 'Close'
                    
                    # Handle MultiIndex if yfinance returns (Price, Ticker) structure even for single ticker
                    if isinstance(df_tick.columns, pd.MultiIndex):
                         # If it's multi-index, usually it is (Price, Ticker)
                        try:
                            df_tick = df_tick.xs(target_col, axis=1, level=0, drop_level=True)
                        except KeyError:
                             # Try swapping levels if structure is reversed
                            df_tick = df_tick.swaplevel(0, 1, axis=1)
                            if target_col in df_tick.columns:
                                df_tick = df_tick[target_col]

                    # If it's a Series (single col), convert to DataFrame with ticker name
                    if isinstance(df_tick, pd.Series):
                        df_tick = df_tick.to_frame(name=norm_ticker)
                    elif target_col in df_tick.columns:
                        df_tick = df_tick[[target_col]].rename(columns={target_col: norm_ticker})
                    else:
                        # Fallback: just take the first column if structure is weird
                        df_tick = df_tick.iloc[:, 0].to_frame(name=norm_ticker)
                        
                    frames.append(df_tick)
                
                # STRICT RATE LIMITING
                time.sleep(1.0) 
                
            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")

        if not frames:
            logger.critical("All downloads failed.")
            return pd.DataFrame()

        # Merge all individual frames on Date Index
        # outer join to keep all dates
        raw_df = pd.concat(frames, axis=1, join='outer')
        
        clean_df = self._sanitize_dataframe(raw_df)
        clean_df = self._cold_start_protection(clean_df, tickers)
        
        # Save to Cache
        clean_df.to_csv(CACHE_FILE)
        logger.info("Data saved to cache.")
        
        return clean_df

# -----------------------------------------------------------------------------
# CLASS 2: QUANT ENGINE (Math, PCA, Stats)
# -----------------------------------------------------------------------------
class QuantEngine:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def calculate_rolling_corr(self, asset_a: str, asset_b: str, window: int = 60) -> pd.Series:
        if asset_a in self.data.columns and asset_b in self.data.columns:
            return self.data[asset_a].pct_change().rolling(window).corr(self.data[asset_b].pct_change())
        return pd.Series(dtype=float)

    def calculate_volatility(self, asset: str, window: int = 60) -> pd.Series:
        if asset in self.data.columns:
            return self.data[asset].pct_change().rolling(window).std() * np.sqrt(252)
        return pd.Series(dtype=float)

    def get_lagged_cross_corr(self, asset_a: str, asset_b: str, max_lag: int = 30) -> pd.DataFrame:
        """
        Computes correlation at lags -30 to +30.
        """
        rets_a = self.data[asset_a].pct_change().dropna()
        rets_b = self.data[asset_b].pct_change().dropna()
        
        # Align
        common_idx = rets_a.index.intersection(rets_b.index)
        r_a = rets_a.loc[common_idx]
        r_b = rets_b.loc[common_idx]
        
        lags = range(-max_lag, max_lag + 1)
        corrs = []
        for lag in lags:
            if lag < 0:
                # Shift B forward (B leads A?)
                shifted_b = r_b.shift(abs(lag))
            else:
                shifted_b = r_b.shift(-lag)
            
            # Recalculate correlation on valid valid
            valid = pd.concat([r_a, shifted_b], axis=1).dropna()
            if len(valid) > 100:
                c, _ = pearsonr(valid.iloc[:,0], valid.iloc[:,1])
                corrs.append(c)
            else:
                corrs.append(0)
                
        return pd.DataFrame({'Lag': lags, 'Correlation': corrs})

    def get_partial_correlation(self, asset_a: str, asset_b: str, controls: List[str]) -> float:
        """
        Calculates Partial Correlation of A and B controlling for 'controls'.
        Using residuals method.
        """
        df_subset = self.data[[asset_a, asset_b] + controls].pct_change().dropna()
        
        if df_subset.empty: 
            return 0.0

        # Residuals of A ~ Controls
        X = df_subset[controls].values
        y_a = df_subset[asset_a].values.reshape(-1, 1)
        y_b = df_subset[asset_b].values.reshape(-1, 1)

        reg_a = LinearRegression().fit(X, y_a)
        res_a = y_a - reg_a.predict(X)

        reg_b = LinearRegression().fit(X, y_b)
        res_b = y_b - reg_b.predict(X)

        res_a = res_a.flatten()
        res_b = res_b.flatten()

        return pearsonr(res_a, res_b)[0]

    def extract_pca_factor(self, assets: List[str]) -> pd.Series:
        """
        Extracts 1st Principal Component (Market/Factor) from a basket.
        """
        subset = self.data[assets].pct_change().dropna()
        if subset.empty:
            return pd.Series(dtype=float)
        
        pca = PCA(n_components=1)
        pca.fit(subset)
        first_pc = pca.transform(subset)
        
        return pd.Series(first_pc.flatten(), index=subset.index)

# -----------------------------------------------------------------------------
# CLASS 3: DASHBOARD RENDERER (Offline Plotly, HTML, JS)
# -----------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self):
        # Dark Mode / Cyberpunk Template
        self.layout_template = go.Layout(
            paper_bgcolor='#0b0c10',
            plot_bgcolor='#1f2833',
            font={'color': '#66fcf1', 'family': 'Courier New, monospace'},
            xaxis={'gridcolor': '#45a29e', 'showgrid': True, 'zerolinecolor': '#45a29e'},
            yaxis={'gridcolor': '#45a29e', 'showgrid': True, 'zerolinecolor': '#45a29e'},
            hovermode='x unified',
            margin=dict(l=50, r=50, t=50, b=50)
        )
        self.colors = ['#66fcf1', '#45a29e', '#c5c6c7', '#ff00ff', '#bd93f9']

    def _get_plotly_js(self) -> str:
        """Retrieve full JS library string for offline embedding."""
        return py_offline.get_plotlyjs()

    def create_macro_chart(self, df_corr: pd.DataFrame, inflation_data: pd.Series) -> str:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Rolling Equity/Bond Correlation (Regime Detection)", "Correlation vs Inflation"))

        # 1. Rolling Correlation Line
        fig.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Corr_60d'], name='60d Corr',
                                 line=dict(color=self.colors[0], width=2)), row=1, col=1)
        
        # 2. Regime Overlay (Background Shapes)
        fig.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Corr_60d'].where(df_corr['Corr_60d'] > 0.5),
                                 fill='tozeroy', fillcolor='rgba(255, 0, 255, 0.2)', 
                                 line=dict(width=0), name='Risk Parity Danger'), row=1, col=1)

        # 3. Macro Scatter
        if not inflation_data.empty:
            # Resample inflation to daily to match corr index (ffill)
            inf_aligned = inflation_data.reindex(df_corr.index, method='ffill')
            fig.add_trace(go.Scatter(x=inf_aligned, y=df_corr['Corr_60d'], mode='markers',
                                     marker=dict(color=self.colors[3], size=4, opacity=0.6),
                                     name='Corr vs Inflation'), row=2, col=1)
            fig.update_xaxes(title_text="Inflation Proxy Level", row=2, col=1)
        
        fig.update_layout(self.layout_template, height=800, title_text="<b>Module A: Macro Regimes</b>")
        return py_offline.plot(fig, include_plotlyjs=False, output_type='div')

    def create_sector_decoupling(self, sector_corrs: pd.DataFrame) -> str:
        fig = go.Figure()
        
        for col in sector_corrs.columns:
            # Highlight XLY specifically
            width = 3 if 'XLY' in col else 1
            opacity = 1.0 if 'XLY' in col else 0.3
            color = self.colors[3] if 'XLY' in col else self.colors[1]
            
            fig.add_trace(go.Scatter(x=sector_corrs.index, y=sector_corrs[col], 
                                     name=col, line=dict(width=width, color=color), opacity=opacity))
            
        fig.add_shape(type="line", x0=sector_corrs.index[0], x1=sector_corrs.index[-1], y0=0.6, y1=0.6,
                      line=dict(color="Red", width=2, dash="dash"))

        fig.update_layout(self.layout_template, height=600, title_text="<b>Module B: Sector Decoupling (Threshold < 0.6)</b>")
        return py_offline.plot(fig, include_plotlyjs=False, output_type='div')

    def create_lag_analysis(self, lag_df: pd.DataFrame) -> str:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lag_df['Lag'], y=lag_df['Correlation'], marker_color=self.colors[0]))
        
        # Mark max lag
        if not lag_df.empty:
            max_lag = lag_df.loc[lag_df['Correlation'].idxmax()]
            fig.add_annotation(x=max_lag['Lag'], y=max_lag['Correlation'],
                            text=f"Optimal: {int(max_lag['Lag'])}d", showarrow=True, arrowhead=1)
        
        fig.update_layout(self.layout_template, height=500, title_text="<b>Module C: Lead/Lag Analysis</b>",
                          xaxis_title="Lag (Days) [-B leads, +B lags]", yaxis_title="Correlation")
        return py_offline.plot(fig, include_plotlyjs=False, output_type='div')

    def create_quant_partials(self, raw_corr: float, part_corr: float, asset_pair: str) -> str:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Raw Correlation', 'Partial Correlation (USD Removed)'],
            y=[raw_corr, part_corr],
            marker_color=[self.colors[3], self.colors[0]],
            text=[f"{raw_corr:.2f}", f"{part_corr:.2f}"],
            textposition='auto'
        ))
        
        fig.update_layout(self.layout_template, height=500, title_text=f"<b>Module D: Quant Scope ({asset_pair})</b>")
        return py_offline.plot(fig, include_plotlyjs=False, output_type='div')

    def render_dashboard(self, divs: Dict[str, str]) -> None:
        """
        Generates the final HTML file with Tabs and Resize triggers.
        """
        js_lib = self._get_plotly_js()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8"/>
            <title>Hedge Fund Analytics Dashboard</title>
            <style>
                body {{ background-color: #0b0c10; color: #c5c6c7; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }}
                .tab {{ overflow: hidden; border: 1px solid #45a29e; background-color: #1f2833; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #66fcf1; font-weight: bold; }}
                .tab button:hover {{ background-color: #45a29e; color: #0b0c10; }}
                .tab button.active {{ background-color: #66fcf1; color: #0b0c10; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #45a29e; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
            <script type="text/javascript">{js_lib}</script>
        </head>
        <body>
            <h1>> MARKET_REGIME_DASHBOARD_v1.0</h1>
            <p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Status: READY</p>
            
            <div class="tab">
              <button class="tablinks" onclick="openCity(event, 'Macro')">Module A: Macro</button>
              <button class="tablinks" onclick="openCity(event, 'Sector')">Module B: Sectors</button>
              <button class="tablinks" onclick="openCity(event, 'Commodity')">Module C: Commodity</button>
              <button class="tablinks" onclick="openCity(event, 'Quant')">Module D: Quant</button>
            </div>

            <div id="Macro" class="tabcontent" style="display:block;">
              {divs['macro']}
            </div>

            <div id="Sector" class="tabcontent">
              {divs['sector']}
            </div>

            <div id="Commodity" class="tabcontent">
              {divs['commodity']}
            </div>
            
            <div id="Quant" class="tabcontent">
              {divs['quant']}
            </div>

            <script>
            function openCity(evt, cityName) {{
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
              
              // THE TAB RESIZE TRIGGER
              window.dispatchEvent(new Event('resize'));
            }}
            </script>
        </body>
        </html>
        """
        
        with open(HTML_OUTPUT, "w", encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Dashboard generated successfully: {os.path.abspath(HTML_OUTPUT)}")

# -----------------------------------------------------------------------------
# MAIN ORCHESTRATION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Greenfield Market Regime Engine")
    parser.add_argument("--tickers", type=str, help="Comma separated tickers override")
    parser.add_argument("--lookback", type=int, default=5, help="Years of history")
    parser.add_argument("--update-cache", action="store_true", help="Force refresh of data")
    args = parser.parse_args()

    # 1. Consolidate Tickers
    all_tickers = []
    for k, v in DEFAULT_TICKERS.items():
        all_tickers.extend(v)
    
    if args.tickers:
        all_tickers.extend(args.tickers.split(','))
    
    all_tickers = list(set(all_tickers)) # Dedupe

    # 2. Ingestion
    ingestor = DataIngestion(force_update=args.update_cache)
    df = ingestor.fetch_market_data(all_tickers, years=args.lookback)
    
    if df.empty:
        logger.critical("No data available. Exiting.")
        return

    quant = QuantEngine(df)
    renderer = DashboardRenderer()
    divs = {}

    # --- MODULE A: MACRO ---
    logger.info("Running Module A: Macro Scope...")
    # Calculate SPY/TLT Correlation
    df_macro = pd.DataFrame()
    df_macro['Corr_60d'] = quant.calculate_rolling_corr('SPY', 'TLT', 60)
    
    # Try to get Inflation proxy (TIP) as backup for CPI
    tip_series = df['TIP'] if 'TIP' in df.columns else pd.Series(dtype=float)
    divs['macro'] = renderer.create_macro_chart(df_macro, tip_series)

    # --- MODULE B: SECTORS ---
    logger.info("Running Module B: Sector Scope...")
    sector_corrs = pd.DataFrame()
    for sector in DEFAULT_TICKERS['sectors']:
        if sector in df.columns:
            # Correlation to SPY
            sector_corrs[sector] = quant.calculate_rolling_corr(sector, 'SPY', 60)
    
    divs['sector'] = renderer.create_sector_decoupling(sector_corrs)

    # --- MODULE C: COMMODITIES ---
    logger.info("Running Module C: Commodity Scope...")
    # Gold vs Silver
    lag_df = quant.get_lagged_cross_corr('GC=F', 'SI=F', max_lag=30)
    divs['commodity'] = renderer.create_lag_analysis(lag_df)

    # --- MODULE D: QUANT ---
    logger.info("Running Module D: Quant Scope...")
    # Extract USD Factor from FX
    fx_cols = ['EURUSD=X', 'JPY=X', 'DX-Y.NYB']
    valid_fx = [c for c in fx_cols if c in df.columns]
    
    if len(valid_fx) >= 2:
        usd_factor = quant.extract_pca_factor(valid_fx)
        # Add to df for calculations
        df['USD_FACTOR'] = usd_factor
        
        # Calculate Raw vs Partial for BTC vs Gold (example of liquidity assets)
        target_a = 'BTC-USD'
        target_b = 'GC=F'
        
        if target_a in df.columns and target_b in df.columns:
            raw_c = df[target_a].pct_change().corr(df[target_b].pct_change())
            # Controls must exist
            part_c = quant.get_partial_correlation(target_a, target_b, ['USD_FACTOR'])
            divs['quant'] = renderer.create_quant_partials(raw_c, part_c, f"{target_a} vs {target_b}")
        else:
            divs['quant'] = "<div>Insufficient data for Quant Module targets.</div>"
    else:
        divs['quant'] = "<div>Insufficient FX data for PCA extraction.</div>"

    # 3. Render
    renderer.render_dashboard(divs)

if __name__ == "__main__":
    main()
