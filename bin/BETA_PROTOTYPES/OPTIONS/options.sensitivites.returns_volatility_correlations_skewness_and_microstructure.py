# SCRIPTNAME: ok.options.sensitivites.returns_volatility_correlations_skewness_and_microstructure.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py_offline
from datetime import datetime, timedelta

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Phase 1: Architecture - Class Definitions
# --------------------------------------------------------------------------------

class DataIngestion:
    """
    Handles 'Disk-First' data loading, API interactions with rate limiting,
    and rigorous dataframe sanitization.
    """
    def __init__(self, output_dir, lookback_years, intraday, interval):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        self.intraday = intraday
        self.interval = interval
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_data(self, tickers):
        data = {}
        for ticker in tickers:
            try:
                df = self._get_ticker_data(ticker)
                if df is not None and not df.empty:
                    data[ticker] = df
            except Exception as e:
                logger.error(f"Failed to ingest {ticker}: {e}")
        return data

    def _get_ticker_data(self, ticker):
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        
        # 1. Check Disk
        if os.path.exists(file_path):
            logger.info(f"Loading {ticker} from disk...")
            try:
                # Read CSV - Dates parsed, index set later in sanitize
                df = pd.read_csv(file_path)
                return self._sanitize_df(df, ticker)
            except Exception as e:
                logger.warning(f"Corrupt CSV for {ticker}, forcing download. Error: {e}")

        # 2. Fallback to API
        logger.info(f"Downloading {ticker} via yfinance...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_years * 365)
        
        # Enforce rate limit sleep
        time.sleep(1) 
        
        try:
            df = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                interval=self.interval, 
                progress=False,
                threads=False # Disable threading to control rate limits manually
            )
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return None

            # 3. Save to Disk immediately
            # Reset index to ensure Date is a column in CSV
            df.to_csv(file_path)
            
            return self._sanitize_df(df, ticker)
            
        except Exception as e:
            logger.error(f"Download failed for {ticker}: {e}")
            return None

    def _sanitize_df(self, df, ticker_name):
        """
        Applies strict sanitization: MultiIndex flattening, Type coercion, TZ-naive.
        """
        # MultiIndex Fix: Swap levels if columns are MultiIndex (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            # If the columns look like (Price, Ticker), we want to flatten them
            # Usually yfinance returns (Price, Ticker). 
            # We want to keep it simple. If valid data, usually 'Close', 'Open', etc are level 0
            
            # Drop the ticker level if it exists to simplify standard access
            if df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(1)

        # Ensure Date is the index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
            
        # Type Safety: Coerce Index to TZ-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Flattening & Type Safety: Force numeric
        # Handle cases where 'Close' might be an object due to bad parsing
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj Close' in df.columns:
            cols_to_numeric.append('Adj Close')
            
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop fully NaN columns/rows
        df.dropna(how='all', axis=1, inplace=True)
        df.dropna(how='all', axis=0, inplace=True)
        
        return df


class FinancialAnalysis:
    """
    Stateless container for mathematical logic.
    Implements Volatility Proxies, Moments, and Finite Difference Greeks.
    """
    @staticmethod
    def process(df, risk_free_rate, compute_greeks=False):
        """
        Main pipeline for processing a single ticker dataframe.
        """
        df = df.copy()

        # 1. Volatility & Distribution
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Realized_Vol'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)

        df['Skewness'] = df['Log_Returns'].rolling(window=20).skew()
        df['Kurtosis'] = df['Log_Returns'].rolling(window=20).kurt()

        # 2. Volatility Proxies ("High-Low" Logic)
        df['IV_Proxy'] = np.log(df['High'] / df['Low'].replace(0, np.nan))
        df['Risk_Reversal_Proxy'] = df['High'] - df['Low']

        long_term_iv = df['IV_Proxy'].rolling(window=30).mean()
        short_term_iv = df['IV_Proxy'].rolling(window=7).mean()
        df['Term_Structure'] = long_term_iv - short_term_iv

        # ---------------------------------------------------------
        # 3. Microstructure Proxies (The Fix)
        # ---------------------------------------------------------
        required_micro = ['Bid', 'Ask', 'BidSize', 'AskSize']

        # Check if we ACTUALLY have Level 1 data (Rare for yfinance)
        if all(col in df.columns for col in required_micro):
            denom = df['BidSize'] + df['AskSize']
            denom = denom.replace(0, np.nan)
            df['Microprice'] = (df['Bid'] * df['AskSize'] + df['Ask'] * df['BidSize']) / denom
            df['Order_Imbalance'] = (df['BidSize'] - df['AskSize']) / denom

        else:
            # --- APPROXIMATION LOGIC ---

            # A. Order Imbalance Proxy (Money Flow Multiplier)
            # Range -1 (Selling) to +1 (Buying)
            high_low_range = (df['High'] - df['Low']).replace(0, np.nan)
            df['Order_Imbalance'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_range

            # B. Microprice Proxy
            # We estimate Microprice as the Midpoint weighted by the Imbalance.
            # If Imbalance is +1, Microprice pushes towards High.
            mid_price = (df['High'] + df['Low']) / 2

            # We assume the "spread" width is proportional to the range (e.g., 10% of range)
            # This is a heuristic to prevent the microprice from just equaling the Close.
            spread_proxy = high_low_range * 0.1
            df['Microprice'] = mid_price + (df['Order_Imbalance'] * spread_proxy)

        # ---------------------------------------------------------

        # 4. Numerical Greeks (Finite Difference)
        if compute_greeks:
            df = FinancialAnalysis._calculate_numerical_greeks(df, risk_free_rate)

        return df

    @staticmethod
    def _black_scholes_price(S, K, T, r, sigma, option_type='call'):
        """
        Helper pricer to obtain V for Finite Difference calculations.
        """
        if T <= 0 or sigma <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        return price

    @staticmethod
    def _calculate_numerical_greeks(df, r):
        """
        Calculates Greeks using Finite Difference Method.
        Assumes ATM option with 30 days to expiration for the time series.
        """
        # Parameters for perturbation
        h_S = 0.01 * df['Close'] # 1% Price step
        h_vol = 0.01             # 1% Vol step
        h_t = 1/365.0            # 1 Day Time step
        h_r = 0.001              # 10bps Rate step
        
        # Fixed Assumptions for the time-series Greeks
        T = 30/365.0 
        # Use Realized Vol as the input sigma, fallback to 20% if NaN
        sigmas = df['Realized_Vol'].fillna(0.20)
        
        deltas, gammas, vegas, thetas, rhos = [], [], [], [], []
        
        for i in range(len(df)):
            S = df['Close'].iloc[i]
            sigma = sigmas.iloc[i]
            K = S # ATM assumption
            
            if pd.isna(S) or S <= 0:
                deltas.append(np.nan); gammas.append(np.nan); vegas.append(np.nan); thetas.append(np.nan); rhos.append(np.nan)
                continue
                
            # Base Pricer Wrapper
            def V(S_in, sigma_in, T_in, r_in):
                return FinancialAnalysis._black_scholes_price(S_in, K, T_in, r_in, sigma_in)

            # Delta: (V(S+h) - V(S-h)) / 2h
            step_s = h_S.iloc[i]
            delta = (V(S + step_s, sigma, T, r) - V(S - step_s, sigma, T, r)) / (2 * step_s)
            
            # Gamma: (V(S+h) - 2V(S) + V(S-h)) / h^2
            gamma = (V(S + step_s, sigma, T, r) - 2*V(S, sigma, T, r) + V(S - step_s, sigma, T, r)) / (step_s ** 2)
            
            # Vega: (V(σ+h) - V(σ-h)) / 2h
            vega = (V(S, sigma + h_vol, T, r) - V(S, sigma - h_vol, T, r)) / (2 * h_vol)
            
            # Theta: (V(t) - V(t-h)) / h  (Note: usually Theta is negative decay)
            # We look backwards in time T vs T-h (decay)
            theta = (V(S, sigma, T, r) - V(S, sigma, T - h_t, r)) / h_t
            
            # Rho: (V(r+h) - V(r-h)) / 2h
            rho = (V(S, sigma, T, r + h_r) - V(S, sigma, T, r - h_r)) / (2 * h_r)
            
            deltas.append(delta)
            gammas.append(gamma)
            vegas.append(vega)
            thetas.append(theta)
            rhos.append(rho)

        df['Delta'] = deltas
        df['Gamma'] = gammas
        df['Vega'] = vegas
        df['Theta'] = thetas
        df['Rho'] = rhos
        
        return df


class DashboardRenderer:
    """
    Handles Offline Plotly generation with embedded JS and the specific
    resize hack for tab rendering.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_dashboard(self, ticker_data, correlations):
        """
        Generates the HTML file.
        ticker_data: Dict of {ticker: processed_df}
        correlations: DataFrame of correlation matrix
        """
        # Create Subplots with 6 Tabs logic simulated via visibility toggling or specific layout
        # Plotly doesn't support native "Tabs" in a single figure object easily. 
        # Standard approach: Create a single HTML with Bootstrap tabs or similar custom HTML.
        # However, to keep it a "single script" without external CSS/JS files, we will build
        # the HTML wrapper string manually and embed the plotly divs.
        
        html_content = self._build_html_structure(ticker_data, correlations)
        
        output_path = os.path.join(self.output_dir, "market_dashboard.html")
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {output_path}")

    def _build_html_structure(self, ticker_data, correlations):
        # 1. Generate Plotly Divs for each section
        # We will use the first ticker for detailed single-asset charts, 
        # or aggregate them. For this requirement, let's focus on the PRIMARY ticker (first in list)
        primary_ticker = list(ticker_data.keys())[0]
        df = ticker_data[primary_ticker]
        
        div_price_vol = self._plot_price_vol(df, primary_ticker)
        div_iv_skew = self._plot_iv_skew(df, primary_ticker)
        div_micro = self._plot_microstructure(df, primary_ticker)
        div_greeks = self._plot_greeks(df, primary_ticker) # Greeks heatmap
        div_corr = self._plot_correlations(correlations, ticker_data)
        div_summary = self._plot_summary(df, primary_ticker)

        # 2. Get Plotly JS Source
        plotly_js = py_offline.get_plotlyjs()

        # 3. HTML Template with Resize Hack
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Analytics Engine</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #1e1e1e; color: #e0e0e0; margin: 0; }}
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #2d2d2d; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #007acc; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; height: 90vh; }}
                .chart-container {{ height: 100%; width: 100%; }}
            </style>
        </head>
        <body>

        <div class="tab">
          <button class="tablinks" onclick="openTab(event, 'PriceVol')" id="defaultOpen">Price & Volatility</button>
          <button class="tablinks" onclick="openTab(event, 'IVSkew')">IV Skew & Term</button>
          <button class="tablinks" onclick="openTab(event, 'Micro')">Microstructure</button>
          <button class="tablinks" onclick="openTab(event, 'Greeks')">Numerical Greeks</button>
          <button class="tablinks" onclick="openTab(event, 'Correlations')">Correlations</button>
          <button class="tablinks" onclick="openTab(event, 'Summary')">Summary</button>
        </div>

        <div id="PriceVol" class="tabcontent">
            <div class="chart-container">{div_price_vol}</div>
        </div>

        <div id="IVSkew" class="tabcontent">
            <div class="chart-container">{div_iv_skew}</div>
        </div>

        <div id="Micro" class="tabcontent">
            <div class="chart-container">{div_micro}</div>
        </div>

        <div id="Greeks" class="tabcontent">
            <div class="chart-container">{div_greeks}</div>
        </div>
        
        <div id="Correlations" class="tabcontent">
            <div class="chart-container">{div_corr}</div>
        </div>
        
        <div id="Summary" class="tabcontent">
            <div class="chart-container">{div_summary}</div>
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

            // THE RESIZE EVENT HACK
            // Triggers a window resize event to force Plotly to redraw/adjust to new container size
            window.dispatchEvent(new Event('resize'));
        }}

        // Get the element with id="defaultOpen" and click on it
        document.getElementById("defaultOpen").click();
        </script>
        </body>
        </html>
        """
        return html

    def _plot_price_vol(self, df, ticker):
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=("Price", "Realized Volatility (20d)", "Return Histogram", "Skew/Kurtosis"),
                            vertical_spacing=0.1)
        
        # Price
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'), row=1, col=1)
        
        # Realized Vol
        fig.add_trace(go.Scatter(x=df.index, y=df['Realized_Vol'], name='Realized Vol', line=dict(color='orange')), row=1, col=2)
        
        # Histogram
        fig.add_trace(go.Histogram(x=df['Log_Returns'], name='Returns Dist', nbinsx=50), row=2, col=1)
        
        # Skew/Kurt
        fig.add_trace(go.Scatter(x=df.index, y=df['Skewness'], name='Skewness', line=dict(color='green')), row=2, col=2)
        fig.add_trace(go.Scatter(x=df.index, y=df['Kurtosis'], name='Kurtosis', line=dict(color='red', dash='dot')), row=2, col=2)
        
        fig.update_layout(template="plotly_dark", title_text=f"{ticker} - Price & Volatility")
        return py_offline.plot(fig, output_type='div', include_plotlyjs=False)

    def _plot_iv_skew(self, df, ticker):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Risk Reversal Proxy
        fig.add_trace(go.Scatter(x=df.index, y=df['Risk_Reversal_Proxy'], name='Risk Reversal (H-L)', fill='tozeroy'), row=1, col=1)
        
        # Term Structure Slope
        fig.add_trace(go.Scatter(x=df.index, y=df['Term_Structure'], name='Term Slope (30d-7d)', line=dict(color='cyan')), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", title_text=f"{ticker} - IV Proxies")
        return py_offline.plot(fig, output_type='div', include_plotlyjs=False)

    def _plot_microstructure(self, df, ticker):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Microprice vs Mid (Mid is approx (High+Low)/2 if Bid/Ask missing, but we plot what we have)
        if 'Microprice' in df.columns and not df['Microprice'].isna().all():
            fig.add_trace(go.Scatter(x=df.index, y=df['Microprice'], name='Microprice', line=dict(color='yellow')), row=1, col=1)
            # Proxy Midprice
            mid = (df['High'] + df['Low']) / 2
            fig.add_trace(go.Scatter(x=df.index, y=mid, name='Midprice Proxy', line=dict(dash='dot', color='gray')), row=1, col=1)
        else:
             fig.add_annotation(text="No Microstructure Data (Bid/Ask) Available", showarrow=False)

        # Imbalance
        if 'Order_Imbalance' in df.columns and not df['Order_Imbalance'].isna().all():
            fig.add_trace(go.Bar(x=df.index, y=df['Order_Imbalance'], name='Order Imbalance'), row=2, col=1)
            
        fig.update_layout(template="plotly_dark", title_text=f"{ticker} - Microstructure Analysis")
        return py_offline.plot(fig, output_type='div', include_plotlyjs=False)

    def _plot_greeks(self, df, ticker):
        # Heatmaps for Vega, Gamma, Delta (Strike vs Expiry simulation)
        # Since we have Time-Series data, we plot the Greeks over Time
        if 'Vega' not in df.columns:
             return "<div>Greeks calculation disabled</div>"
             
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Delta", "Gamma", "Vega", "Theta"))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Delta'], name='Delta', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Gamma'], name='Gamma', line=dict(color='purple')), row=1, col=2)
        fig.add_trace(go.Scatter(x=df.index, y=df['Vega'], name='Vega', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Theta'], name='Theta', line=dict(color='red')), row=2, col=2)

        fig.update_layout(template="plotly_dark", title_text=f"{ticker} - Numerical Greeks (ATM, 30D)")
        return py_offline.plot(fig, output_type='div', include_plotlyjs=False)

    def _plot_correlations(self, correlations, ticker_data):
        fig = make_subplots(rows=1, cols=2)
        
        # Heatmap
        fig.add_trace(go.Heatmap(
            z=correlations.values,
            x=correlations.columns,
            y=correlations.index,
            colorscale='Viridis'
        ), row=1, col=1)
        
        # Rolling Beta (First vs Second ticker if available)
        tickers = list(ticker_data.keys())
        if len(tickers) >= 2:
            t1, t2 = tickers[0], tickers[1]
            r1 = ticker_data[t1]['Log_Returns']
            r2 = ticker_data[t2]['Log_Returns']
            
            # Simple Rolling Beta
            cov = r1.rolling(60).cov(r2)
            var = r2.rolling(60).var()
            beta = cov / var
            
            fig.add_trace(go.Scatter(x=beta.index, y=beta, name=f'Beta {t1}/{t2}'), row=1, col=2)
            
        fig.update_layout(template="plotly_dark", title_text="Correlations & Beta")
        return py_offline.plot(fig, output_type='div', include_plotlyjs=False)

    def _plot_summary(self, df, ticker):
        fig = go.Figure()
        
        # Gauge for Current Vol Regime (Percentile)
        current_vol = df['Realized_Vol'].iloc[-1]
        percentile = stats.percentileofscore(df['Realized_Vol'].dropna(), current_vol)
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = percentile,
            title = {'text': "Vol Regime (Percentile)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"}},
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))
        
        fig.update_layout(template="plotly_dark", title_text=f"{ticker} - Summary Indicators")
        return py_offline.plot(fig, output_type='div', include_plotlyjs=False)


# --------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Performance Financial Analytics Engine")
    parser.add_argument("--tickers", nargs='+', default=['SPY', 'QQQ', 'IWM'], help="List of tickers")
    parser.add_argument("--output-dir", default="./market_data", help="Data storage path")
    parser.add_argument("--lookback", type=int, default=1, help="Years of history")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk free rate")
    parser.add_argument("--intraday", action="store_true", help="Enable intraday fetching")
    parser.add_argument("--interval", default="1d", help="Data interval (e.g. 1d, 1h)")
    parser.add_argument("--compute-greeks", action="store_true", help="Enable expensive Greek calcs")
    
    args = parser.parse_args()
    
    # 1. Ingestion
    print("--- Phase 1: Ingestion ---")
    ingestor = DataIngestion(args.output_dir, args.lookback, args.intraday, args.interval)
    raw_data = ingestor.fetch_data(args.tickers)
    
    if not raw_data:
        print("No data fetched. Exiting.")
        exit()

    # 2. Analysis
    print("--- Phase 2: Analysis ---")
    processed_data = {}
    returns_df = pd.DataFrame()
    
    for ticker, df in raw_data.items():
        print(f"Processing {ticker}...")
        try:
            analyzed_df = FinancialAnalysis.process(df, args.risk_free_rate, args.compute_greeks)
            processed_data[ticker] = analyzed_df
            returns_df[ticker] = analyzed_df['Log_Returns']
        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {e}")
            
    # Calculate Correlations
    correlations = returns_df.corr()

    # 3. Rendering
    print("--- Phase 3: Rendering ---")
    renderer = DashboardRenderer(args.output_dir)
    try:
        renderer.generate_dashboard(processed_data, correlations)
    except Exception as e:
        logger.error(f"Rendering failed: {e}")

    print("Complete.")
