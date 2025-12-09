import os
import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as si
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = "SPY"
CACHE_DIR = "market_data_cache"
RISK_FREE_RATE = 0.045  # 4.5%
LOOKBACK_YEARS = 2

# ==========================================
# CLASS 1: DATA INGESTION
# ==========================================
class DataIngestion:
    """
    Handles ETL: Downloading, Caching, Sanitizing, and Loading.
    Ensures data types are strict and consistent.
    """
    def __init__(self, ticker, cache_dir):
        self.ticker_symbol = ticker
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        self.ohlcv_path = os.path.join(cache_dir, f"{ticker}_ohlcv.csv")
        self.history_metrics_path = os.path.join(cache_dir, f"{ticker}_metrics_history.csv")

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal Fixer: Normalizes yfinance MultiIndex, types, and timezones.
        """
        if df.empty:
            return df

        # 1. Handle MultiIndex Columns (yfinance update fix)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' is in Level 0 or Level 1
            # If Ticker is Level 0, we might need to swap or just drop level
            if 'Close' not in df.columns.get_level_values(0):
                # Assume standard yf structure: (Price, Ticker) -> swap to (Ticker, Price) ??
                # Actually standard is usually (Price, Ticker). 
                # If we see ('SPY', 'Close') instead of ('Close', 'SPY'), we swap.
                if self.ticker_symbol in df.columns.get_level_values(0):
                     df = df.swaplevel(0, 1, axis=1)
            
            # Flatten columns: ('Close', 'SPY') -> 'Close'
            df.columns = df.columns.droplevel(1)

        # 2. Strict Datetime Index
        df.index = pd.to_datetime(df.index)

        # 3. Strip Timezones
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 4. Numeric Coercion (Force objects to floats)
        cols_to_fix = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols_to_fix:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df

    def get_ohlcv(self) -> pd.DataFrame:
        """
        Fetch OHLCV. Logic: Local CSV -> Download -> Sanitize -> Save.
        """
        # 1. Try Local Load
        if os.path.exists(self.ohlcv_path):
            try:
                print(f"[DataIngestion] Loading cached OHLCV for {self.ticker_symbol}...")
                df = pd.read_csv(self.ohlcv_path, index_col=0)
                df = self._sanitize_df(df)
                
                # Check staleness (if data is older than today's market open)
                last_date = df.index[-1]
                if last_date < pd.Timestamp.now() - pd.Timedelta(days=1):
                    print("[DataIngestion] Cache stale. Refreshing...")
                    raise FileNotFoundError # Force download
                return df
            except Exception as e:
                print(f"[DataIngestion] Cache read failed: {e}. Downloading new data.")

        # 2. Download
        print(f"[DataIngestion] Downloading {self.ticker_symbol} from yfinance...")
        time.sleep(1) # Rate Limit
        try:
            df = yf.download(self.ticker_symbol, period=f"{LOOKBACK_YEARS}y", progress=False)
            df = self._sanitize_df(df)
            
            # Save
            df.to_csv(self.ohlcv_path)
            return df
        except Exception as e:
            print(f"[Critical] Failed to download OHLCV: {e}")
            return pd.DataFrame()

    def get_option_chain(self) -> pd.DataFrame:
        """
        Downloads option chain for near-term expirations using yfinance Ticker object.
        """
        print(f"[DataIngestion] Fetching Option Chain for {self.ticker_symbol}...")
        tkr = yf.Ticker(self.ticker_symbol)
        
        try:
            exps = tkr.options
        except Exception:
            print("[DataIngestion] Failed to get expirations.")
            return pd.DataFrame()

        all_opts = []
        
        # Limit to first 6 expirations to prevent timeouts/throttling
        target_exps = exps[:6] 

        for e in target_exps:
            try:
                time.sleep(0.5) # Rate limit
                print(f"   -> Fetching expiry: {e}")
                opt = tkr.option_chain(e)
                
                calls = opt.calls
                calls['type'] = 'call'
                
                puts = opt.puts
                puts['type'] = 'put'
                
                chain = pd.concat([calls, puts])
                chain['expiration'] = e
                all_opts.append(chain)
            except Exception as err:
                print(f"   -> Failed expiry {e}: {err}")
                continue
        
        if not all_opts:
            return pd.DataFrame()
            
        df_chain = pd.concat(all_opts)
        
        # Basic cleanup for chain
        df_chain['expiration'] = pd.to_datetime(df_chain['expiration'])
        df_chain['lastTradeDate'] = pd.to_datetime(df_chain['lastTradeDate']).dt.tz_convert(None)
        
        return df_chain

    def _backfill_shadow_history(self, ohlcv: pd.DataFrame):
        """
        Cold Start Fix: Creates a synthetic history file if one doesn't exist.
        """
        if os.path.exists(self.history_metrics_path):
            return

        print("[DataIngestion] Cold Start Detected. Generating Shadow Backfill...")
        
        # Create synthetic GEX based on Volume and Volatility
        # Logic: High Vol + Drop = Negative GEX regime approximation
        shadow = ohlcv.copy()
        shadow['returns'] = shadow['Close'].pct_change()
        shadow['realized_vol'] = shadow['returns'].rolling(20).std()
        
        # Synthetic metric: (Close * Volume * 0.01) modulated by trend
        # This is purely to ensure charts are not empty on day 1
        shadow['Net_GEX'] = np.where(shadow['returns'] > 0, 
                                     shadow['Volume'] * 100, 
                                     shadow['Volume'] * -100)
        
        shadow['Vanna_Exposure'] = shadow['realized_vol'] * 1e9 # Dummy scale
        
        save_df = shadow[['Net_GEX', 'Vanna_Exposure']]
        save_df.to_csv(self.history_metrics_path)
        print("[DataIngestion] Shadow history created.")

    def load_historical_metrics(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Loads the history of calculated metrics (GEX, Vanna) for plotting.
        """
        self._backfill_shadow_history(ohlcv)
        
        try:
            df = pd.read_csv(self.history_metrics_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            # Align with OHLCV dates
            combined = ohlcv.join(df, how='left')
            return combined
        except Exception:
            return ohlcv

# ==========================================
# CLASS 2: FINANCIAL ANALYSIS
# ==========================================
class FinancialAnalysis:
    """
    Pure Logic Core. Stateless with respect to data modifications.
    """
    def __init__(self, raw_ohlcv, raw_options):
        self._ohlcv = raw_ohlcv.copy()
        self._options = raw_options.copy() if not raw_options.empty else pd.DataFrame()

    def black_scholes_greeks(self, S, K, T, r, sigma, opt_type):
        """
        Vectorized Black-Scholes Greeks calculation.
        """
        # Safety for divide by zero or negative time
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-5)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if opt_type == 'call':
            delta = si.norm.cdf(d1)
            # Gamma is same for put/call
            gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
            # Vega (usually returned in decimals, multiply by 0.01 for $ change per 1% vol)
            vega = S * si.norm.pdf(d1) * np.sqrt(T) * 0.01 
            # Vanna: dDelta/dVol
            vanna = -si.norm.pdf(d1) * (d2 / sigma) 
        else:
            delta = -si.norm.cdf(-d1)
            gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * si.norm.pdf(d1) * np.sqrt(T) * 0.01
            vanna = -si.norm.pdf(d1) * (d2 / sigma) 

        return delta, gamma, vega, vanna

    def compute_dealer_gex(self):
        """
        Calculates Net GEX and Vanna from the option chain.
        Assumption: Dealers are Short Calls and Short Puts (Short Volatility).
        Standard Model:
        - Dealer sells Call to user -> User Long, Dealer Short Call -> Dealer Short Gamma.
        - Dealer sells Put to user -> User Long, Dealer Short Put -> Dealer Long Gamma (stabilizing).
        
        SqueezeMetrics Convention:
        - Call OI adds Negative GEX (Dealer must sell into strength)
        - Put OI adds Positive GEX (Dealer must buy into weakness)
        """
        if self._options.empty:
            return 0, 0, pd.DataFrame()

        df = self._options.copy()
        spot = self._ohlcv['Close'].iloc[-1]
        
        # Time to expiry in years
        today = pd.Timestamp.now()
        df['T'] = (df['expiration'] - today).dt.days / 365.0
        
        # Filter expired
        df = df[df['T'] > 0]

        # Calculate Greeks
        # Note: yfinance impliesVolatility is a percentage e.g. 0.20
        # If yfinance gives missing IV, fill with average
        df['impliedVolatility'] = df['impliedVolatility'].replace(0, np.nan)
        df['impliedVolatility'] = df['impliedVolatility'].fillna(df['impliedVolatility'].mean())

        # Vectorized Calculation
        greeks = df.apply(
            lambda x: self.black_scholes_greeks(
                S=spot, 
                K=x['strike'], 
                T=x['T'], 
                r=RISK_FREE_RATE, 
                sigma=x['impliedVolatility'], 
                opt_type=x['type']
            ), axis=1, result_type='expand'
        )
        
        df[['delta', 'gamma', 'vega', 'vanna']] = greeks

        # Calculate GEX ($ exposure per 1% move)
        # GEX = Gamma * Spot * Spot * 0.01 * OpenInterest * Contributor
        # Contributor: Call = -1 (Dealer Short), Put = +1 (Dealer Long)
        df['contributor'] = np.where(df['type'] == 'call', -1, 1)
        
        # Standard GEX formula (Dollar Gamma)
        # Gamma is dDelta/dS. Total Gamma Exposure = Sum(Gamma * OI * 100 * Spot * Spot * 0.01) ??
        # Simplified: GEX = Sum(Gamma * OI * 100 * Spot) -> Amount of shares dealer buys/sells per 1 pt move
        # We use the standard notation: Gamma * Spot * OI * 100 * Direction
        df['GEX'] = df['gamma'] * spot * 100 * df['openInterest'] * df['contributor']
        
        # Vanna Exposure
        df['VannaExp'] = df['vanna'] * 100 * df['openInterest'] * df['contributor']

        total_net_gex = df['GEX'].sum()
        total_vanna = df['VannaExp'].sum()

        return total_net_gex, total_vanna, df

    def compute_macro_regime(self):
        """
        Determines market regime based on Volatility and Price Trend.
        """
        df = self._ohlcv.copy()
        
        # Calculate Realized Volatility (20d)
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['RV'] = df['log_ret'].rolling(window=20).std() * np.sqrt(252)
        
        # Simple Regime Logic
        # 1. Bullish: Price > SMA50 & RV < Threshold
        # 2. Bearish: Price < SMA50
        # 3. Volatile: RV > High Threshold
        
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        conditions = [
            (df['Close'] > df['SMA50']) & (df['RV'] < 0.15),
            (df['Close'] < df['SMA50']),
            (df['RV'] >= 0.15)
        ]
        choices = ['Risk-On', 'Defensive', 'High-Vol']
        
        df['Regime'] = np.select(conditions, choices, default='Neutral')
        return df

# ==========================================
# CLASS 3: DASHBOARD RENDERER
# ==========================================
class DashboardRenderer:
    """
    Generates HTML report using Offline Plotly (No CDN).
    Handles JS injection for tab resizing.
    """
    def __init__(self, ticker, ohlcv_regime, net_gex, options_surface):
        self.ticker = ticker
        self.df = ohlcv_regime
        self.net_gex = net_gex
        self.surface_df = options_surface
        
        # Ensure we restrict the chart to the last 6 months for clarity
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
        self.df_render = self.df[self.df.index > cutoff].copy()

    def _get_js_header(self):
        """
        Injects Plotly JS directly and adds the resize event handler for Tabs.
        """
        plotly_js = get_plotlyjs()
        
        # JS to fix Plotly rendering inside hidden tabs
        resize_script = """
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                var tabs = document.querySelectorAll('.tab-button');
                tabs.forEach(function(tab) {
                    tab.addEventListener('click', function() {
                        setTimeout(function() { 
                            window.dispatchEvent(new Event('resize')); 
                        }, 100);
                    });
                });
            });
            
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
                window.dispatchEvent(new Event('resize'));
            }
        </script>
        """
        
        css = """
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; padding: 20px;}
            .tab { overflow: hidden; border-bottom: 1px solid #444; background-color: #2d2d2d; }
            .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #aaa; font-size: 16px;}
            .tab button:hover { background-color: #444; }
            .tab button.active { background-color: #007acc; color: white; }
            .tabcontent { display: none; padding: 6px 12px; border: 1px solid #444; border-top: none; animation: fadeEffect 0.5s; }
            @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
            h1 { color: #007acc; }
            .metric-box { display: inline-block; background: #333; padding: 10px; margin: 5px; border-radius: 5px; min-width: 150px; text-align: center; }
            .val { font-size: 1.2em; font-weight: bold; color: #fff; }
            .lbl { font-size: 0.8em; color: #888; text-transform: uppercase; }
        </style>
        """
        
        return f'<head>{css}<script>{plotly_js}</script>{resize_script}</head>'

    def _plot_dealer_panel(self):
        """
        Main Panel: Price (Candles) + Regime Shading (Background) + Net GEX (Secondary Axis)
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 1. Price Candles
        fig.add_trace(go.Candlestick(
            x=self.df_render.index,
            open=self.df_render['Open'], high=self.df_render['High'],
            low=self.df_render['Low'], close=self.df_render['Close'],
            name="OHLC"
        ), secondary_y=False)
        
        # 2. Regime Background Colors (Simulated via Shapes or Bar underlay)
        # Using a Bar chart spread across the background to represent regime
        # Map regime to colors
        colors = self.df_render['Regime'].map({
            'Risk-On': 'rgba(0, 255, 0, 0.1)',
            'Defensive': 'rgba(255, 0, 0, 0.1)',
            'High-Vol': 'rgba(255, 165, 0, 0.1)',
            'Neutral': 'rgba(128, 128, 128, 0.1)'
        })
        
        # We can't easily do variable background in Plotly without shapes, 
        # so we use a Scatter with fill or just leave it for now to keep performance high.
        # Alternative: Plot Close Price colored by Regime
        
        # 3. Net GEX (If history exists in df_render, otherwise just current point?)
        # Since we calculated Spot GEX only for *today* in this simplified script, 
        # we plot the stored history if available.
        if 'Net_GEX' in self.df_render.columns:
            fig.add_trace(go.Scatter(
                x=self.df_render.index, 
                y=self.df_render['Net_GEX'],
                name="Net GEX ($)",
                line=dict(color='cyan', width=1),
                opacity=0.6
            ), secondary_y=True)

        fig.update_layout(
            title=f"{self.ticker} Dealer Positioning & Price Action",
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _plot_vol_surface(self):
        """
        3D Surface or Term Structure.
        """
        if self.surface_df.empty:
            return "<div>No Option Data Available</div>"

        # Pivot for 3D Surface: Strike x Expiry -> IV
        try:
            # Filter near the money to make chart readable
            spot = self.df['Close'].iloc[-1]
            df_surf = self.surface_df[
                (self.surface_df['strike'] > spot * 0.8) & 
                (self.surface_df['strike'] < spot * 1.2)
            ].copy()
            
            # Create pivot table
            pivot = df_surf.pivot_table(index='strike', columns='expiration', values='impliedVolatility')
            
            fig = go.Figure(data=[go.Surface(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Viridis'
            )])
            
            fig.update_layout(
                title="Volatility Surface (IV)",
                scene=dict(
                    xaxis_title='Expiration',
                    yaxis_title='Strike',
                    zaxis_title='Implied Vol'
                ),
                template="plotly_dark",
                height=600
            )
            return fig.to_html(full_html=False, include_plotlyjs=False)
        except Exception as e:
            return f"<div>Error rendering surface: {e}</div>"

    def _plot_gamma_profile(self):
        """
        Line chart of Gamma Exposure by Strike for current expiry.
        """
        if self.surface_df.empty:
            return ""
            
        # Group by strike and sum GEX
        gex_profile = self.surface_df.groupby('strike')['GEX'].sum()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=gex_profile.index,
            y=gex_profile.values,
            marker_color=np.where(gex_profile.values < 0, 'red', 'green'),
            name="GEX"
        ))
        
        spot = self.df['Close'].iloc[-1]
        fig.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="Spot")
        
        fig.update_layout(
            title="Current GEX Profile by Strike",
            template="plotly_dark",
            xaxis_title="Strike",
            yaxis_title="Net Gamma Exposure ($)",
            height=600
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def generate_report(self):
        """
        Assembles the full HTML structure.
        """
        print("[DashboardRenderer] Generating HTML...")
        
        header = self._get_js_header()
        
        # Current Metrics
        last_regime = self.df_render['Regime'].iloc[-1]
        last_price = self.df_render['Close'].iloc[-1]
        
        metrics_html = f"""
        <div style='background-color:#252526; padding:15px; border-bottom:1px solid #333'>
            <h1>{self.ticker} Quantitative Dashboard</h1>
            <div class='metric-box'><div class='val'>{last_price:.2f}</div><div class='lbl'>Spot Price</div></div>
            <div class='metric-box'><div class='val'>{last_regime}</div><div class='lbl'>Macro Regime</div></div>
            <div class='metric-box'><div class='val'>${self.net_gex/1e9:.2f}B</div><div class='lbl'>Net GEX</div></div>
        </div>
        """
        
        # Tabs HTML
        tabs_nav = """
        <div class="tab">
          <button class="tablinks active" onclick="openTab(event, 'Dealer')">Dealer Positioning</button>
          <button class="tablinks" onclick="openTab(event, 'VolSurface')">Vol Surface</button>
          <button class="tablinks" onclick="openTab(event, 'GammaProfile')">Gamma Profile</button>
        </div>
        """
        
        # Content
        dealer_html = self._plot_dealer_panel()
        surface_html = self._plot_vol_surface()
        profile_html = self._plot_gamma_profile()
        
        body = f"""
        <body>
            {metrics_html}
            {tabs_nav}
            
            <div id="Dealer" class="tabcontent" style="display:block;">
                {dealer_html}
            </div>
            
            <div id="VolSurface" class="tabcontent">
                {surface_html}
            </div>
            
             <div id="GammaProfile" class="tabcontent">
                {profile_html}
            </div>
        </body>
        """
        
        full_html = f"<html>{header}{body}</html>"
        
        filename = "market_dashboard.html"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"[DashboardRenderer] Report saved to {filename}")
        return filename

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== Starting Standalone Market Dashboard ===")
    
    # 1. Ingest
    ingestor = DataIngestion(TICKER, CACHE_DIR)
    df_ohlcv = ingestor.get_ohlcv()
    df_opts = ingestor.get_option_chain()
    
    # 2. Analyze
    if not df_ohlcv.empty:
        analyzer = FinancialAnalysis(df_ohlcv, df_opts)
        
        # Regime
        df_analyzed = analyzer.compute_macro_regime()
        
        # Options Metrics
        net_gex, net_vanna, df_surface = analyzer.compute_dealer_gex()
        
        # Merge calculated GEX into history for persistence (Simple append logic for demo)
        # In a real db we would update the row for today
        if not df_ohlcv.empty:
            # Update the latest row in the dataframe passed to renderer
            # Note: This is transient. To persist, we would append to the CSV in DataIngestion
            df_analyzed.loc[df_analyzed.index[-1], 'Net_GEX'] = net_gex
            
            # Load historical context for the chart
            df_final = ingestor.load_historical_metrics(df_analyzed)
            
            # 3. Render
            renderer = DashboardRenderer(TICKER, df_final, net_gex, df_surface)
            renderer.generate_report()
            
            print("=== Process Complete ===")
    else:
        print("[Error] No OHLCV data available. Exiting.")
