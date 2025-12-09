import argparse
import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm, linregress
from scipy.interpolate import griddata, interp1d
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')  # Silence pandas chained assignment warnings

# -----------------------------------------------------------------------------
# 1. DataIngestion (IO & Sanitization)
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Sole responsibility: IO, data fetching (disk-first), and sanitization.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        The 'Universal Fixer' for yfinance inconsistencies.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Flatten MultiIndex Columns
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Join levels with underscore, skipping empty levels
                    new_cols.append("_".join([str(c) for c in col if str(c)]))
                else:
                    new_cols.append(str(col))
            df.columns = new_cols
        
        # 2. Swap Levels / Handle Specific yfinance quirks
        df.columns = [c.strip() for c in df.columns]

        # 3. Strict Datetime Index
        if data_type == 'price':
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    if 'Date' in df.columns:
                        df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index, utc=True)
                except Exception as e:
                    logger.warning(f"Could not coerce index to datetime: {e}")
            
            # Remove Timezone for easier filtering
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            df.sort_index(inplace=True)

        # 4. Numeric Coercion
        cols_to_numeric = []
        if data_type == 'options':
            cols_to_numeric = ['strike', 'lastPrice', 'bid', 'ask', 'change', 
                               'volume', 'openInterest', 'impliedVolatility']
        elif data_type == 'price':
            cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        for col in df.columns:
            if col in cols_to_numeric or (data_type == 'options' and col not in ['contractSymbol', 'type', 'currency', 'expiration']):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where critical numeric data is entirely missing
        if data_type == 'options':
            df = df.dropna(subset=['strike', 'impliedVolatility'])
            if 'openInterest' in df.columns:
                df['openInterest'] = df['openInterest'].fillna(0)
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)

        return df

    def get_price_history(self, ticker: str, lookback_years: float) -> pd.DataFrame:
        """Disk-first pipeline for price history."""
        file_path = os.path.join(self.output_dir, f"prices_{ticker}.csv")
        
        # 1. Check Disk
        if os.path.exists(file_path):
            logger.info(f"[{ticker}] Reading prices from disk.")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return self._sanitize_df(df, 'price')

        # 2. Shadow Backfill
        logger.info(f"[{ticker}] Downloading prices (Shadow Backfill)...")
        start_date = (datetime.now() - timedelta(days=int(lookback_years * 365))).strftime('%Y-%m-%d')
        try:
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            if df.empty:
                logger.warning(f"[{ticker}] Downloaded empty price dataframe.")
        except Exception as e:
            logger.error(f"[{ticker}] Price download failed: {e}")
            return pd.DataFrame()
        
        sanitized_df = self._sanitize_df(df, 'price')
        sanitized_df.to_csv(file_path)
        
        return self._sanitize_df(pd.read_csv(file_path, index_col=0, parse_dates=True), 'price')

    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        """Disk-first pipeline for full options chain."""
        file_path = os.path.join(self.output_dir, f"options_{ticker}.csv")

        # 1. Check Disk
        if os.path.exists(file_path):
            logger.info(f"[{ticker}] Reading options from disk.")
            df = pd.read_csv(file_path)
            return self._sanitize_df(df, 'options')

        # 2. Shadow Backfill
        logger.info(f"[{ticker}] Downloading full option chain (Shadow Backfill)...")
        tk = yf.Ticker(ticker)
        
        try:
            expirations = tk.options
        except Exception as e:
            logger.error(f"[{ticker}] Failed to fetch expirations: {e}")
            return pd.DataFrame()

        all_opts = []
        for exp in expirations:
            try:
                opt = tk.option_chain(exp)
                calls = opt.calls
                calls['type'] = 'C'
                puts = opt.puts
                puts['type'] = 'P'
                
                calls['expiration'] = exp
                puts['expiration'] = exp
                
                all_opts.append(calls)
                all_opts.append(puts)
            except Exception as e:
                # logger.warning(f"[{ticker}] Failed to fetch expiry {exp}: {e}")
                continue

        if not all_opts:
            logger.error(f"[{ticker}] No options data found.")
            return pd.DataFrame()

        raw_df = pd.concat(all_opts, ignore_index=True)
        sanitized_df = self._sanitize_df(raw_df, 'options')
        sanitized_df.to_csv(file_path, index=False)

        return self._sanitize_df(pd.read_csv(file_path), 'options')

# -----------------------------------------------------------------------------
# 2. FinancialAnalysis (Math & Logic)
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Sole responsibility: Quant math, Black-Scholes, Greeks, Surfaces.
    """
    def __init__(self, prices: pd.DataFrame, options: pd.DataFrame, risk_free_rate: float = 0.04):
        self.prices = prices
        self.options = options
        self.r = risk_free_rate
        self.q = 0.0
        self.spot = self._get_current_spot()
    
    def _get_current_spot(self) -> float:
        if self.prices.empty:
            return 0.0
        # Robust column selection for spot price
        if 'Adj Close' in self.prices.columns:
            col = 'Adj Close'
        elif 'Close' in self.prices.columns:
            col = 'Close'
        else:
             # Fallback: grab first float column
             possible = self.prices.select_dtypes(include=['float', 'int']).columns
             col = possible[0] if len(possible) > 0 else None
        
        if col:
            return float(self.prices.iloc[-1][col])
        return 0.0

    def compute_analytics(self) -> Dict[str, Any]:
        """Main driver for all calculations."""
        if self.options.empty or self.spot == 0:
            return {}

        df = self.options.copy()
        
        # 1. Time to Expiry (T)
        df['expiry_date'] = pd.to_datetime(df['expiration'])
        now = datetime.now()
        df['T'] = (df['expiry_date'] - now).dt.days / 365.0
        df = df[df['T'] > 0.001].copy()

        # 2. Forward Price & Log Moneyness
        df['F'] = self.spot * np.exp((self.r - self.q) * df['T'])
        df['k'] = np.log(df['strike'] / df['F'])

        # 3. Greeks Calculation
        df = df[df['impliedVolatility'] > 0.0001].copy()
        
        sigma = df['impliedVolatility']
        T_sqrt = np.sqrt(df['T'])
        d1 = (np.log(self.spot / df['strike']) + (self.r - self.q + 0.5 * sigma**2) * df['T']) / (sigma * T_sqrt)
        d2 = d1 - sigma * T_sqrt
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)

        # Delta
        df['delta'] = 0.0
        df.loc[df['type'] == 'C', 'delta'] = np.exp(-self.q * df['T']) * cdf_d1
        df.loc[df['type'] == 'P', 'delta'] = -np.exp(-self.q * df['T']) * cdf_neg_d1

        # Gamma
        df['gamma'] = (np.exp(-self.q * df['T']) * pdf_d1) / (self.spot * sigma * T_sqrt)

        # Vega
        df['vega'] = self.spot * np.exp(-self.q * df['T']) * pdf_d1 * T_sqrt

        # Vanna
        term2 = 1 - (d1 / (sigma * T_sqrt))
        df['vanna'] = np.exp(-self.q * df['T']) * T_sqrt * pdf_d1 * term2

        # 4. GEX & Dollar Gamma
        contract_size = 100
        dealer_sign = -1
        
        df['dollar_gamma'] = df['gamma'] * (self.spot ** 2) * contract_size
        df['GEX'] = df['dollar_gamma'] * df['openInterest'] * dealer_sign

        # 5. Aggregates
        gex_total = df['GEX'].sum()
        
        # 6. Term Structure & Skew Analysis
        term_structure = self._analyze_term_structure(df)
        
        # 7. Surface Grid (k, T)
        surface_grid = self._build_vol_surface(df)

        return {
            'spot': self.spot,
            'options_df': df,
            'gex_total': gex_total,
            'term_structure': term_structure,
            'surface_grid': surface_grid
        }

    def _analyze_term_structure(self, df: pd.DataFrame) -> Dict:
        results = []
        expiries = sorted(df['expiration'].unique())

        for exp in expiries:
            sub = df[df['expiration'] == exp]
            if sub.empty: continue
            
            T = sub['T'].iloc[0]
            
            # ATM IV: Interpolate IV at k=0
            sub = sub.sort_values('k')
            sub_dedup = sub.drop_duplicates(subset=['k'])
            if len(sub_dedup) < 3: continue

            try:
                # 1. ATM IV
                f_iv = interp1d(sub_dedup['k'], sub_dedup['impliedVolatility'], kind='linear', fill_value="extrapolate")
                atm_iv = float(f_iv(0))

                # 2. Vega-Weighted Skew
                reg_df = sub_dedup.dropna(subset=['vega', 'k', 'impliedVolatility'])
                if len(reg_df) > 5:
                    weights = reg_df['vega'].values
                    weights = weights / (weights.sum() + 1e-9)
                    slope, intercept = np.polyfit(reg_df['k'], reg_df['impliedVolatility'], 1, w=reg_df['vega'])
                    skew_weighted = slope
                else:
                    skew_weighted = 0.0

                # 3. 25 Delta Risk Reversal & Fly
                calls = sub_dedup[sub_dedup['type'] == 'C']
                puts = sub_dedup[sub_dedup['type'] == 'P']
                
                rr25, fly25 = np.nan, np.nan
                
                if len(calls) > 2 and len(puts) > 2:
                    f_iv_call = interp1d(calls['delta'], calls['impliedVolatility'], bounds_error=False, fill_value=np.nan)
                    f_iv_put = interp1d(puts['delta'], puts['impliedVolatility'], bounds_error=False, fill_value=np.nan)
                    
                    iv_25c = float(f_iv_call(0.25))
                    iv_25p = float(f_iv_put(-0.25))
                    
                    if not np.isnan(iv_25c) and not np.isnan(iv_25p):
                        rr25 = iv_25c - iv_25p
                        fly25 = 0.5 * (iv_25c + iv_25p) - atm_iv

                results.append({
                    'expiration': exp,
                    'T': T,
                    'atm_iv': atm_iv,
                    'skew_vega': skew_weighted,
                    'rr25': rr25,
                    'fly25': fly25
                })

            except Exception as e:
                # logger.warning(f"Error calculating term structure for {exp}: {e}")
                continue

        ts_df = pd.DataFrame(results)
        if not ts_df.empty:
            ts_df.sort_values('T', inplace=True)
            ts_df['term_slope'] = ts_df['atm_iv'].diff() / ts_df['T'].diff()
        
        return ts_df

    def _build_vol_surface(self, df: pd.DataFrame) -> Dict:
        points_df = df.dropna(subset=['k', 'T', 'impliedVolatility'])
        points_df = points_df[points_df['impliedVolatility'] < 5.0]
        
        if points_df.empty:
            return {}

        k_pts = points_df['k'].values
        t_pts = points_df['T'].values
        iv_pts = points_df['impliedVolatility'].values

        grid_k_dim = np.linspace(max(-1.0, k_pts.min()), min(1.0, k_pts.max()), 50)
        grid_t_dim = np.linspace(max(0.01, t_pts.min()), t_pts.max(), 50)
        
        grid_k, grid_t = np.meshgrid(grid_k_dim, grid_t_dim)
        
        try:
            grid_iv = griddata((k_pts, t_pts), iv_pts, (grid_k, grid_t), method='linear')
            mask = np.isnan(grid_iv)
            if np.any(mask):
                 grid_iv[mask] = griddata((k_pts, t_pts), iv_pts, (grid_k[mask], grid_t[mask]), method='nearest')
        except Exception as e:
            return {}

        return {
            'k': grid_k_dim,
            't': grid_t_dim,
            'iv': grid_iv
        }

# -----------------------------------------------------------------------------
# 3. DashboardRenderer (Visualization)
# -----------------------------------------------------------------------------
class DashboardRenderer:
    """
    Sole responsibility: Offline Plotly HTML generation.
    """
    def __init__(self, analytics_map: Dict[str, Any], context: Dict):
        self.analytics_map = analytics_map
        self.context = context
        self.html_buffer = []

    def _get_plotly_js(self):
        return py_offline.get_plotlyjs()

    def _generate_css(self):
        return """
        <style>
            body { font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; padding: 0; }
            .container { padding: 20px; }
            /* Tabs */
            .tab { overflow: hidden; border-bottom: 1px solid #444; background-color: #2d2d2d; }
            .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-size: 16px; }
            .tab button:hover { background-color: #444; }
            .tab button.active { background-color: #007acc; color: white; }
            
            .tabcontent { display: none; padding: 20px; border-top: none; animation: fadeEffect 0.5s; }
            @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
            
            /* Nested Tabs */
            .sub-tab { overflow: hidden; border-bottom: 1px solid #444; margin-top: 10px; background-color: #252525; }
            .sub-tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 10px 12px; font-size: 14px; color: #aaa; }
            .sub-tab button.active { border-bottom: 2px solid #007acc; color: #fff; }
            
            .chart-container { background-color: #252525; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            h1, h2, h3 { color: #f0f0f0; }
            .metric-box { display: inline-block; background: #333; padding: 10px 20px; border-radius: 4px; margin-right: 10px; border-left: 4px solid #007acc; }
            .metric-label { font-size: 0.8em; color: #aaa; display: block; }
            .metric-value { font-size: 1.2em; font-weight: bold; }
            .metric-stabilizing { border-left-color: #4caf50; }
            .metric-destabilizing { border-left-color: #f44336; }
        </style>
        """

    def _generate_js(self):
        return """
        <script>
            function openTicker(evt, tickerName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("main-tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("main-tablink");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tickerName).style.display = "block";
                evt.currentTarget.className += " active";
                // Trigger resize for Plotly
                window.dispatchEvent(new Event('resize'));
                
                // Open first subtab by default if none active
                var subTab = document.getElementById(tickerName).querySelector('.sub-tablink');
                if (subTab) subTab.click();
            }

            function openSubTab(evt, tabId) {
                var i, tabcontent, tablinks;
                // Get parent ticker container
                var parent = evt.currentTarget.parentElement.parentElement;
                
                var tabcontents = parent.getElementsByClassName("sub-tabcontent");
                for (i = 0; i < tabcontents.length; i++) {
                    tabcontents[i].style.display = "none";
                }
                
                var tablinks = parent.getElementsByClassName("sub-tablink");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                
                document.getElementById(tabId).style.display = "block";
                evt.currentTarget.className += " active";
                window.dispatchEvent(new Event('resize'));
            }
        </script>
        """

    def render_all(self, filename: str = "dashboard.html"):
        # Header
        self.html_buffer.append("<html><head>")
        self.html_buffer.append(f"<title>VolAnalytics Dashboard</title>")
        
        # --- FIX: Wrap Plotly JS in script tags ---
        self.html_buffer.append(f'<script type="text/javascript">{self._get_plotly_js()}</script>')
        # -------------------------------------------
        
        self.html_buffer.append(self._generate_css())
        self.html_buffer.append("</head><body>")
        self.html_buffer.append(f"<div class='container'><h1>Quant Volatility Engine</h1>")
        self.html_buffer.append(f"<p style='color:#888'>Generated: {datetime.now()} | Risk-Free: {self.context['risk_free_rate']}</p>")

        # Ticker Tabs (Top Level)
        tickers = list(self.analytics_map.keys())
        self.html_buffer.append('<div class="tab">')
        for i, ticker in enumerate(tickers):
            active_cls = " active" if i == 0 else ""
            self.html_buffer.append(f'<button class="main-tablink{active_cls}" onclick="openTicker(event, \'{ticker}\')">{ticker}</button>')
        self.html_buffer.append('</div>')

        # Content for each ticker
        for i, ticker in enumerate(tickers):
            data = self.analytics_map[ticker]
            display_style = "block" if i == 0 else "none"
            self.html_buffer.append(f'<div id="{ticker}" class="main-tabcontent" style="display:{display_style}">')
            self._render_ticker_content(ticker, data)
            self.html_buffer.append('</div>')

        self.html_buffer.append(self._generate_js())
        self.html_buffer.append("</div></body></html>")

        # Write to file
        full_path = os.path.join(self.context['output_dir'], filename)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("".join(self.html_buffer))
        logger.info(f"Dashboard generated at: {full_path}")

    def _render_ticker_content(self, ticker, data):
        # Sub Tabs
        tabs = ['Overview', 'Term Structure', 'Smile', 'GEX', 'Surface']
        self.html_buffer.append('<div class="sub-tab">')
        for i, tab in enumerate(tabs):
            tab_id = f"{ticker}_{tab.replace(' ', '')}"
            active_cls = " active" if i == 0 else ""
            self.html_buffer.append(f'<button class="sub-tablink{active_cls}" onclick="openSubTab(event, \'{tab_id}\')">{tab}</button>')
        self.html_buffer.append('</div>')

        self.html_buffer.append(f'<div id="{ticker}_Overview" class="sub-tabcontent" style="display:block">')
        self._render_overview(ticker, data)
        self.html_buffer.append('</div>')

        self.html_buffer.append(f'<div id="{ticker}_TermStructure" class="sub-tabcontent">')
        self._render_term_structure(ticker, data)
        self.html_buffer.append('</div>')
        
        self.html_buffer.append(f'<div id="{ticker}_Smile" class="sub-tabcontent">')
        self._render_smile(ticker, data)
        self.html_buffer.append('</div>')

        self.html_buffer.append(f'<div id="{ticker}_GEX" class="sub-tabcontent">')
        self._render_gex(ticker, data)
        self.html_buffer.append('</div>')

        self.html_buffer.append(f'<div id="{ticker}_Surface" class="sub-tabcontent">')
        self._render_surface(ticker, data)
        self.html_buffer.append('</div>')

    def _render_overview(self, ticker, data):
        spot = data['spot']
        gex = data['gex_total'] / 1e9 
        ts = data['term_structure']
        
        atm_str = "N/A"
        slope_str = "N/A"
        if not ts.empty:
            atm_str = f"{ts.iloc[0]['atm_iv']:.2%}" if len(ts) > 0 else "N/A"
            slope_str = f"{ts['term_slope'].iloc[0]:.4f}" if 'term_slope' in ts.columns and not np.isnan(ts['term_slope'].iloc[0]) else "N/A"

        gex_cls = "metric-stabilizing" if gex > 0 else "metric-destabilizing"
        gex_lbl = "Stabilizing" if gex > 0 else "Destabilizing"

        html = f"""
        <div class="chart-container">
            <div class="metric-box"><span class="metric-label">Spot Price</span><span class="metric-value">${spot:.2f}</span></div>
            <div class="metric-box"><span class="metric-label">Front ATM IV</span><span class="metric-value">{atm_str}</span></div>
            <div class="metric-box"><span class="metric-label">Term Slope</span><span class="metric-value">{slope_str}</span></div>
            <div class="metric-box {gex_cls}"><span class="metric-label">Total GEX ({gex_lbl})</span><span class="metric-value">${gex:.2f}B</span></div>
        </div>
        """
        self.html_buffer.append(html)

        prices = self.context['price_data'][ticker]
        
        # --- FIX: Robust Column Detection ---
        if 'Adj Close' in prices.columns:
            plot_col = 'Adj Close'
        elif 'Close' in prices.columns:
            plot_col = 'Close'
        else:
            plot_col = prices.select_dtypes(include=['float', 'int']).columns[0]
        # ------------------------------------

        fig = go.Figure(data=[go.Scatter(x=prices.index, y=prices[plot_col], mode='lines', name='Price')])
        fig.update_layout(title=f"{ticker} Price History", template="plotly_dark", height=400, margin=dict(l=40, r=40, t=40, b=40))
        self.html_buffer.append(py_offline.plot(fig, include_plotlyjs=False, output_type='div'))

    def _render_term_structure(self, ticker, data):
        ts = data['term_structure']
        if ts.empty:
            self.html_buffer.append("<p>No Term Structure Data Available</p>")
            return

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=ts['T'], y=ts['atm_iv'], name="ATM IV", line=dict(color='#007acc')), secondary_y=False)
        fig.add_trace(go.Scatter(x=ts['T'], y=ts['skew_vega'], name="Vega-Weighted Skew", line=dict(color='#f44336', dash='dot')), secondary_y=True)
        
        fig.update_layout(title="Term Structure & Skew", template="plotly_dark", height=500, xaxis_title="Time to Expiry (Years)")
        fig.update_yaxes(title_text="ATM Vol", secondary_y=False)
        fig.update_yaxes(title_text="Skew Slope", secondary_y=True)
        
        self.html_buffer.append(py_offline.plot(fig, include_plotlyjs=False, output_type='div'))

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=ts['expiration'], y=ts['rr25'], name="25d Risk Reversal"))
        fig2.add_trace(go.Bar(x=ts['expiration'], y=ts['fly25'], name="25d Butterfly"))
        fig2.update_layout(title="Skew (RR) & Convexity (Fly) Structure", template="plotly_dark", height=400, barmode='group')
        self.html_buffer.append(py_offline.plot(fig2, include_plotlyjs=False, output_type='div'))

    def _render_smile(self, ticker, data):
        df = data['options_df']
        expiries = sorted(df['expiration'].unique())[:4]
        
        fig = go.Figure()
        for exp in expiries:
            sub = df[df['expiration'] == exp]
            fig.add_trace(go.Scatter(x=sub['k'], y=sub['impliedVolatility'], mode='markers', name=f"{exp} (Obs)", marker=dict(size=3, opacity=0.6)))
            sub = sub.sort_values('k')
            fig.add_trace(go.Scatter(x=sub['k'], y=sub['impliedVolatility'], mode='lines', name=f"{exp} (Line)", line=dict(width=1), visible='legendonly'))

        fig.update_layout(title="Volatility Smile (Raw) - Top 4 Expiries", xaxis_title="Log Moneyness (k)", yaxis_title="Implied Vol", template="plotly_dark", height=500)
        self.html_buffer.append(py_offline.plot(fig, include_plotlyjs=False, output_type='div'))

    def _render_gex(self, ticker, data):
        df = data['options_df']
        if not df.empty:
            front_exp = sorted(df['expiration'].unique())[0]
            front_df = df[df['expiration'] == front_exp]
            gex_strike = front_df.groupby('strike')['GEX'].sum().reset_index()
            
            colors = ['#4caf50' if v > 0 else '#f44336' for v in gex_strike['GEX']]
            
            fig = go.Figure(go.Bar(x=gex_strike['strike'], y=gex_strike['GEX'], marker_color=colors))
            fig.add_vline(x=data['spot'], line_dash="dash", line_color="white", annotation_text="Spot")
            fig.update_layout(title=f"GEX by Strike ({front_exp})", template="plotly_dark", height=400, xaxis_title="Strike", yaxis_title="Dollar Gamma Exposure")
            self.html_buffer.append(py_offline.plot(fig, include_plotlyjs=False, output_type='div'))

        try:
            top_exps = sorted(df['expiration'].unique())[:10]
            sub = df[df['expiration'].isin(top_exps)].copy()
            sub['strike_bin'] = (sub['strike'] / 5).round() * 5
            pivot = sub.pivot_table(index='expiration', columns='strike_bin', values='vanna', aggfunc='sum').fillna(0)
            
            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdBu',
                zmid=0
            ))
            fig_hm.update_layout(title="Net Vanna Heatmap (Strike vs Expiry)", template="plotly_dark", height=500)
            self.html_buffer.append(py_offline.plot(fig_hm, include_plotlyjs=False, output_type='div'))
        except Exception as e:
            self.html_buffer.append(f"<p>Could not generate heatmap: {e}</p>")

    def _render_surface(self, ticker, data):
        grid = data['surface_grid']
        if not grid:
            self.html_buffer.append("<p>Not enough data for Vol Surface</p>")
            return
            
        fig = go.Figure(data=[go.Surface(z=grid['iv'], x=grid['k'], y=grid['t'], colorscale='Viridis')])
        fig.update_layout(title="3D Implied Volatility Surface", 
                          scene=dict(xaxis_title='Log Moneyness (k)', yaxis_title='Maturity (T)', zaxis_title='IV'),
                          template="plotly_dark", height=700, margin=dict(l=10, r=10, b=10, t=40))
        self.html_buffer.append(py_offline.plot(fig, include_plotlyjs=False, output_type='div'))


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Institutional Volatility Engine & Dashboard")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help="List of tickers")
    parser.add_argument('--output-dir', default='./market_data', help="Data storage directory")
    parser.add_argument('--lookback', type=float, default=1.0, help="Years of price history")
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help="Risk free rate (decimal)")
    
    args = parser.parse_args()
    
    logger.info("=== Starting Quant Volatility Engine ===")
    logger.info(f"Tickers: {args.tickers}")
    
    # 1. Instantiate Ingestion
    ingestion = DataIngestion(args.output_dir)
    
    analytics_results = {}
    price_context = {}
    
    # 2. Loop Tickers
    for i, ticker in enumerate(args.tickers):
        if i > 0: time.sleep(1) # Rate limit
        
        try:
            # Data Pipeline
            prices = ingestion.get_price_history(ticker, args.lookback)
            options = ingestion.get_options_chain(ticker)
            
            if prices.empty or options.empty:
                logger.error(f"Skipping {ticker} due to missing data.")
                continue

            price_context[ticker] = prices

            # Analysis
            engine = FinancialAnalysis(prices, options, args.risk_free_rate)
            results = engine.compute_analytics()
            
            if results:
                analytics_results[ticker] = results
                logger.info(f"[{ticker}] Analytics complete. GEX: {results['gex_total'] / 1e9:.2f}B")
        
        except Exception as e:
            logger.error(f"[{ticker}] Critical failure: {e}", exc_info=True)

    # 3. Render
    if analytics_results:
        logger.info("Rendering Dashboard...")
        renderer_context = {
            'output_dir': args.output_dir, 
            'risk_free_rate': args.risk_free_rate,
            'price_data': price_context
        }
        renderer = DashboardRenderer(analytics_results, renderer_context)
        renderer.render_all("options_dashboard.html")
    else:
        logger.warning("No analytics generated. Dashboard skipped.")

    logger.info("=== Done ===")

if __name__ == "__main__":
    main()
