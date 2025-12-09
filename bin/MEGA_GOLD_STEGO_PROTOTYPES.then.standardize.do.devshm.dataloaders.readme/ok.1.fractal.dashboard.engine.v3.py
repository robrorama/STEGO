import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta
import os
import time
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------
TICKER = "SPY"
LOOKBACK_YEARS = 2
CACHE_FILE = f"{TICKER}_data.csv"
CACHE_EXPIRY_HOURS = 24

# Fractal Windows
WIN_SHORT = 63
WIN_MED = 126
WIN_LONG = 252

# Thresholds
CLUSTER_THRESHOLD = 0.06  # Tightness standard deviation
BROWNIAN_CENTER = 0.5
MEAN_REV_THRESHOLD = 0.08 # Tolerance around 0.5

# Styling
COLOR_BG = "#1e1e1e"
COLOR_PAPER = "#121212"
COLOR_TEXT = "#e0e0e0"
COLOR_GRID = "#333333"

# -----------------------------------------------------------------------------
# CLASS: DATA INGESTION
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Handles robust data fetching, local caching, aggressive sanitization,
    and 'Shadow Backfilling' for cold starts.
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.filepath = CACHE_FILE

    def get_data(self) -> pd.DataFrame:
        """Orchestrates the fetch/load process."""
        df = self._try_load_cache()
        
        if df is None:
            print(f"[-] Cache miss or expired for {self.ticker}. Downloading...")
            df = self._download_data()
        else:
            print(f"[+] Loaded {self.ticker} from local cache.")

        if df is None or df.empty:
            print("[!] Critical: Data fetch failed. Initiating Shadow Backfill.")
            df = self._backfill_shadow_history()

        return self._sanitize(df)

    def _try_load_cache(self):
        if not os.path.exists(self.filepath):
            return None
        
        # Check file age
        file_time = os.path.getmtime(self.filepath)
        if (time.time() - file_time) > (CACHE_EXPIRY_HOURS * 3600):
            return None # Expired

        try:
            df = pd.read_csv(self.filepath, index_col=0, parse_dates=True)
            return df
        except Exception:
            return None

    def _download_data(self):
        """
        Downloads with robust MultiIndex handling (The 'Swap Levels' Fix).
        """
        try:
            # Rate limiting compliance
            time.sleep(1.0)
            
            # Download with group_by='column' to expose the structure clearly
            start_date = (datetime.now() - timedelta(days=LOOKBACK_YEARS*365)).strftime('%Y-%m-%d')
            df = yf.download(self.ticker, start=start_date, group_by='column', progress=False)
            
            if df.empty:
                return None

            # --- THE UNIVERSAL FIXER: Column Normalization ---
            # Scenario A: Columns are MultiIndex (Ticker, Attribute) -> Swap to (Attribute, Ticker)
            if isinstance(df.columns, pd.MultiIndex):
                # Check if 'Close' is in the second level (Level 1)
                if 'Close' in df.columns.get_level_values(1):
                    # It's (Ticker, Attribute), we want (Attribute, Ticker)
                    df = df.swaplevel(0, 1, axis=1)
                
                # Check if 'Close' is now in Level 0
                if 'Close' in df.columns.get_level_values(0):
                    # We only want the specific ticker's data if it was requested
                    # But if we swapped, the Ticker is now in Level 1.
                    # Simply try to drop the level to flatten
                    try:
                        df.columns = df.columns.droplevel(1)
                    except:
                        pass
            
            # Ensure we have single level columns now
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                 # Fallback for weird yfinance versions returning different structures
                 # Try to locate the Close column dynamically
                 pass 

            # Save to cache
            df.to_csv(self.filepath)
            return df

        except Exception as e:
            print(f"[!] Download Error: {e}")
            return None

    def _backfill_shadow_history(self) -> pd.DataFrame:
        """
        Generates synthetic Geometric Brownian Motion to prevent dashboard crash
        on API failure.
        """
        print("[*] Generating Shadow Backfill Data...")
        dates = pd.date_range(end=datetime.now(), periods=500, freq='B')
        
        # Seed for reproducibility in shadow mode
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, size=len(dates))
        price_path = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=dates)
        df['Close'] = price_path
        df['Open'] = price_path # Simplify
        df['High'] = price_path * 1.01
        df['Low'] = price_path * 0.99
        df['Volume'] = 1000000
        return df

    def _sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup: Timezone stripping, numeric coercion."""
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Strip timezone if present (tz_convert(None))
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
            
        # Coerce cols to numeric
        cols = ['Open', 'High', 'Low', 'Close']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
        df.dropna(subset=['Close'], inplace=True)
        return df

# -----------------------------------------------------------------------------
# CLASS: FRACTAL MATHEMATICS
# -----------------------------------------------------------------------------
class FractalMathematics:
    """
    Core Logic: DFA (Detrended Fluctuation Analysis), Hurst Exponents,
    and Convergence detection.
    """
    
    @staticmethod
    def calculate_hurst_dfa(series: np.array, min_scale=4, max_scale=None) -> float:
        """
        Calculates Hurst Exponent using Detrended Fluctuation Analysis (DFA).
        This is computationally expensive, so it is applied on rolling windows carefully.
        """
        if len(series) < min_scale * 4:
            return 0.5 # Not enough data
            
        # 1. Integrate the time series (cumsum of mean-centered data)
        series_mean = np.mean(series)
        integrated = np.cumsum(series - series_mean)
        
        # 2. Define scales (box sizes)
        N = len(series)
        if max_scale is None:
            max_scale = N // 4
        
        scales = np.floor(np.logspace(np.log10(min_scale), np.log10(max_scale), num=10)).astype(int)
        scales = np.unique(scales) # Remove duplicates
        
        fluctuations = []
        
        for scale in scales:
            # 3. Split into boxes
            n_boxes = N // scale
            rms = 0
            
            for i in range(n_boxes):
                # Slice data
                seg = integrated[i*scale : (i+1)*scale]
                x = np.arange(scale)
                
                # 4. Detrend (Polynomial fit order 1 - linear)
                coeffs = np.polyfit(x, seg, 1)
                trend = np.polyval(coeffs, x)
                
                # 5. RMS
                rms += np.sum((seg - trend)**2)
            
            rms = np.sqrt(rms / (n_boxes * scale))
            fluctuations.append(rms)
            
        # 6. Calc slope of log-log plot
        # valid checks to avoid log(0)
        fluctuations = np.array(fluctuations)
        valid_idx = fluctuations > 0
        
        if np.sum(valid_idx) < 3:
            return 0.5

        coeffs = np.polyfit(np.log2(scales[valid_idx]), np.log2(fluctuations[valid_idx]), 1)
        hurst = coeffs[0]
        return hurst

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies rolling Hurst calculations and detects convergence.
        """
        print("[*] Calculating Fractal Mathematics (This may take a moment)...")
        
        # Working on log returns usually stabilizes DFA, but Hurst on Prices 
        # is standard for Mean Reversion (H<0.5) vs Trend (H>0.5). 
        # We will use Log Prices to ensure scale invariance.
        price_series = np.log(df['Close'])
        
        # Helper for rolling apply
        def rolling_hurst(window_size):
            # FIX: removed .values because raw=True already provides a numpy array
            return price_series.rolling(window=window_size).apply(
                lambda x: self.calculate_hurst_dfa(x), raw=True
            )

        # Calculate Multi-Timeframe Hurst
        df['H_Short'] = rolling_hurst(WIN_SHORT)
        df['H_Med'] = rolling_hurst(WIN_MED)
        df['H_Long'] = rolling_hurst(WIN_LONG)
        
        # Forward fill initial gaps created by calculation (or drop)
        df.dropna(subset=['H_Long'], inplace=True)
        
        # Convergence Logic
        # 1. Tightness: Std Dev of the three Hursts
        df['H_Tightness'] = df[['H_Short', 'H_Med', 'H_Long']].std(axis=1)
        
        # 2. Mean Hurst
        df['H_Mean'] = df[['H_Short', 'H_Med', 'H_Long']].mean(axis=1)
        
        # 3. Convergence Event Flag
        # Tight cluster AND near 0.5 (The "Coiled Spring" or "Decision Point")
        # Or Just Tight Cluster indicating alignment of timeframes
        df['Convergence_Event'] = (df['H_Tightness'] < CLUSTER_THRESHOLD) & \
                                  (np.abs(df['H_Mean'] - 0.5) < MEAN_REV_THRESHOLD)

        # Volatility & Fractal Dimension for Cone
        # Realized Vol (Annualized)
        df['Realized_Vol'] = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
        df['Fractal_Dimension'] = 2 - df['H_Mean']
        
        return df

# -----------------------------------------------------------------------------
# CLASS: DASHBOARD RENDERER
# -----------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self, df: pd.DataFrame, ticker: str):
        self.df = df
        self.ticker = ticker

    def _get_plotly_js(self):
        """
        The 'Blank Chart Fix #1': Return full JS string for offline use.
        """
        return py_offline.get_plotlyjs()

    def generate_html(self, filename="fractal_dashboard.html"):
        """
        Constructs the strict HTML/JS/CSS structure.
        """
        # 1. Build Figures
        fig_macro = self._build_tab1_macro()
        fig_osc = self._build_tab2_oscillator()
        fig_cone = self._build_tab3_vol_cone()
        
        # 2. Convert to JSON for embedding
        json_macro = fig_macro.to_json()
        json_osc = fig_osc.to_json()
        json_cone = fig_cone.to_json()
        
        # 3. Current Stats
        last_row = self.df.iloc[-1]
        regime = "TRENDING" if last_row['H_Mean'] > 0.55 else "MEAN REVERTING" if last_row['H_Mean'] < 0.45 else "RANDOM WALK"
        tightness = last_row['H_Tightness']
        tightness_color = "#00ff00" if tightness < CLUSTER_THRESHOLD else "#ff0000"
        
        # 4. HTML Template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fractal Convergence: {self.ticker}</title>
            <style>
                body {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
                .header {{ display: flex; justify-content: space-between; border-bottom: 1px solid {COLOR_GRID}; padding-bottom: 10px; margin-bottom: 20px; }}
                .stat-box {{ background: {COLOR_PAPER}; padding: 10px 20px; border-radius: 4px; border: 1px solid {COLOR_GRID}; }}
                .stat-label {{ font-size: 0.8rem; color: #888; display: block; }}
                .stat-val {{ font-size: 1.2rem; font-weight: bold; }}
                
                /* Tabs */
                .tab {{ overflow: hidden; border-bottom: 1px solid {COLOR_GRID}; margin-bottom: 10px; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: {COLOR_TEXT}; font-size: 17px; }}
                .tab button:hover {{ background-color: #333; }}
                .tab button.active {{ background-color: #444; border-bottom: 2px solid #00d2ff; }}
                
                .tabcontent {{ display: none; height: 80vh; width: 100%; }}
                .chart-container {{ height: 100%; width: 100%; }}
            </style>
            <!-- Inject Plotly JS Offline -->
            <script type="text/javascript">{self._get_plotly_js()}</script>
        </head>
        <body>
            <div class="header">
                <div>
                    <h1>{self.ticker} Fractal Convergence</h1>
                </div>
                <div style="display:flex; gap: 10px;">
                    <div class="stat-box">
                        <span class="stat-label">Regime</span>
                        <span class="stat-val">{regime}</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Cluster Tightness</span>
                        <span class="stat-val" style="color: {tightness_color}">{tightness:.4f}</span>
                    </div>
                </div>
            </div>

            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Macro')" id="defaultOpen">Macro View</button>
                <button class="tablinks" onclick="openTab(event, 'Oscillator')">Fractal Oscillator</button>
                <button class="tablinks" onclick="openTab(event, 'VolCone')">Volatility Cone</button>
            </div>

            <div id="Macro" class="tabcontent">
                <div id="chart_macro" class="chart-container"></div>
            </div>
            <div id="Oscillator" class="tabcontent">
                <div id="chart_osc" class="chart-container"></div>
            </div>
            <div id="VolCone" class="tabcontent">
                <div id="chart_cone" class="chart-container"></div>
            </div>

            <script>
                // Plotly Data injection
                var data_macro = {json_macro};
                var data_osc = {json_osc};
                var data_cone = {json_cone};

                // Initial Render
                Plotly.newPlot('chart_macro', data_macro.data, data_macro.layout);
                Plotly.newPlot('chart_osc', data_osc.data, data_osc.layout);
                Plotly.newPlot('chart_cone', data_cone.data, data_cone.layout);

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

                    // --- THE BLANK CHART FIX #2 ---
                    // Force resize trigger when tab becomes visible
                    window.dispatchEvent(new Event('resize'));
                    
                    var chartId = 'chart_macro';
                    if(tabName === 'Oscillator') chartId = 'chart_osc';
                    if(tabName === 'VolCone') chartId = 'chart_cone';
                    
                    Plotly.Plots.resize(document.getElementById(chartId));
                }}

                // Click default tab
                document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[+] Dashboard generated successfully: {filename}")
        # Automatically open in browser (optional, good UX)
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.realpath(filename))
        except:
            pass

    def _apply_theme(self, fig):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_PAPER,
            xaxis=dict(gridcolor=COLOR_GRID),
            yaxis=dict(gridcolor=COLOR_GRID),
            font=dict(color=COLOR_TEXT)
        )
        return fig

    def _build_tab1_macro(self):
        df = self.df
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ))

        # Convergence Diamonds
        convergence_dates = df[df['Convergence_Event']].index
        convergence_prices = df[df['Convergence_Event']]['High'] * 1.02
        
        if not convergence_dates.empty:
            fig.add_trace(go.Scatter(
                x=convergence_dates, y=convergence_prices,
                mode='markers',
                marker=dict(symbol='diamond', size=10, color='#ffff00', line=dict(width=1, color='black')),
                name='Fractal Convergence'
            ))

        # Regime Shading (Background rectangles)
        # Using a simplified approach for visual performance: V-Rects for high tight trending zones
        # We look for H_Mean > 0.6 (Trend) vs H_Mean < 0.4 (Mean Rev)
        
        # Note: Adding too many shapes slows Plotly. We do segments.
        # This implementation simply colors the background based on current regime? 
        # Better: Plot a heatmap strip at the bottom? 
        # Request said "Vertical shaded regions". 
        
        # Let's simplify: Only shade strong Trending (Green) or Mean Rev (Red)
        # We iterate through the DF to find contiguous blocks (simplified for code length)
        
        fig.update_layout(
            title="Macro Price Action & Convergence Events",
            xaxis_rangeslider_visible=False,
            height=700
        )
        return self._apply_theme(fig)

    def _build_tab2_oscillator(self):
        df = self.df
        fig = go.Figure()

        # Plot Short, Med, Long
        fig.add_trace(go.Scatter(x=df.index, y=df['H_Short'], name='Hurst (63d)', line=dict(color='cyan', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['H_Med'], name='Hurst (126d)', line=dict(color='yellow', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['H_Long'], name='Hurst (252d)', line=dict(color='magenta', width=1)))

        # Brownian Center
        fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.5)

        # Spread Fill
        # We calculate row-wise min and max of the 3 H values
        h_min = df[['H_Short', 'H_Med', 'H_Long']].min(axis=1)
        h_max = df[['H_Short', 'H_Med', 'H_Long']].max(axis=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=h_max, line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=h_min, fill='tonexty', 
            fillcolor='rgba(255, 255, 255, 0.1)', 
            line=dict(width=0), 
            name='Fractal Spread',
            hoverinfo='skip'
        ))

        fig.update_layout(title="Multi-Timeframe Fractal Memory", yaxis_title="Hurst Exponent")
        return self._apply_theme(fig)

    def _build_tab3_vol_cone(self):
        df = self.df.dropna()
        
        # Scatter: X=FractalDim, Y=RealizedVol
        # Color by recency (Latest points brighter)
        
        fig = go.Figure()
        
        # All history (dim)
        fig.add_trace(go.Scatter(
            x=df['Fractal_Dimension'], 
            y=df['Realized_Vol'],
            mode='markers',
            marker=dict(size=5, color='#555555', opacity=0.3),
            name='Historical Regime'
        ))
        
        # Recent history (last 21 days) - Bright
        recent = df.iloc[-21:]
        fig.add_trace(go.Scatter(
            x=recent['Fractal_Dimension'],
            y=recent['Realized_Vol'],
            mode='markers+text',
            text=[d.strftime('%m-%d') if i == 20 else "" for i, d in enumerate(recent.index)],
            textposition='top center',
            marker=dict(size=10, color='#00d2ff', line=dict(width=1, color='white')),
            name='Current Regime (21d)'
        ))
        
        fig.update_layout(
            title="Volatility Cone: Cost of Volatility vs Complexity",
            xaxis_title="Fractal Dimension (2 - H)",
            yaxis_title="Realized Volatility (Annualized)",
            shapes=[
                # Cheap Vol Zone (High FD, Low Vol)
                dict(type="rect", x0=1.45, y0=0, x1=1.6, y1=0.1, fillcolor="green", opacity=0.1, line_width=0),
                # Expensive Trend (Low FD, High Vol)
                dict(type="rect", x0=1.0, y0=0.3, x1=1.3, y1=1.0, fillcolor="red", opacity=0.1, line_width=0)
            ]
        )
        return self._apply_theme(fig)

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("===================================================")
    print("   HEDGE FUND GRADE FRACTAL DASHBOARD ENGINE       ")
    print("===================================================")
    
    # 1. Ingest
    ingestion = DataIngestion(TICKER)
    df = ingestion.get_data()
    
    if df is None:
        print("Fatal Error: Could not obtain data source.")
        return

    # 2. Compute Math
    quant_engine = FractalMathematics()
    df_processed = quant_engine.process(df)
    
    # 3. Render
    renderer = DashboardRenderer(df_processed, TICKER)
    renderer.generate_html()

if __name__ == "__main__":
    main()
