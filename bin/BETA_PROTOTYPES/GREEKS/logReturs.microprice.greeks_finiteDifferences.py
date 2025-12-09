# SCRIPTNAME: ok.logReturs.microprice.greeks_finiteDifferences.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import sys
import os
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

# ============================================================
# CONFIGURATION & UTILS
# ============================================================
warnings.filterwarnings("ignore")

def print_status(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# ============================================================
# CLASS 1: DataIngestion
# ============================================================
class DataIngestion:
    """
    Responsibilities:
    - Download data via yfinance
    - Enforce disk-first caching
    - Enforce 1s sleep
    - Sanitize DataFrames (Swap level fix, flattening, numeric enforcement)
    """
    def __init__(self, output_dir, lookback_years, intraday, interval):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        self.intraday = intraday
        self.interval = interval
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_ticker_df(self, ticker):
        """
        Disk-first ingestion strategy with Shadow Backfill.
        """
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        
        # 1. Attempt Load from Disk
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                print_status(f"Loaded {ticker} from disk.")
                # Verify it's not empty
                if df.empty:
                    print_status(f"Cached file for {ticker} is empty. Triggering shadow backfill.")
                    return self._download_and_save(ticker, file_path)
                return df
            except Exception as e:
                print_status(f"Error reading {ticker} from disk: {e}. Re-downloading.")
                return self._download_and_save(ticker, file_path)
        else:
            return self._download_and_save(ticker, file_path)

    def get_options_chain(self, ticker):
        """
        Fetches the nearest expiry option chain for Greek calculation.
        """
        try:
            print_status(f"Fetching options chain for {ticker}...")
            time.sleep(1) # Enforce rate limit
            tk = yf.Ticker(ticker)
            exps = tk.options
            if not exps:
                return pd.DataFrame()
            
            # Get nearest expiry
            opt = tk.option_chain(exps[0])
            calls = opt.calls.copy()
            calls['type'] = 'call'
            puts = opt.puts.copy()
            puts['type'] = 'put'
            
            chain = pd.concat([calls, puts])
            chain['expiry'] = exps[0]
            
            # Safely get current price
            hist = tk.history(period='1d')
            if not hist.empty:
                chain['underlying_price'] = hist['Close'].iloc[-1]
            else:
                return pd.DataFrame()
                
            return chain
        except Exception as e:
            print_status(f"Failed to fetch options for {ticker}: {e}")
            return pd.DataFrame()

    def _download_and_save(self, ticker, file_path):
        print_status(f"Downloading {ticker} via yfinance...")
        time.sleep(1) # Enforce sleep
        
        period = f"{self.lookback_years}y"
        if self.intraday:
            period = "5d" # yfinance limit for 1m is 7d
        
        try:
            df = yf.download(
                ticker, 
                period=period, 
                interval=self.interval, 
                progress=False,
                auto_adjust=False
            )
            
            df = self._sanitize_df(df, ticker)
            
            if df.empty:
                print_status(f"Warning: Downloaded data for {ticker} is empty.")
            else:
                df.to_csv(file_path)
                
            return df
        except Exception as e:
            print_status(f"Download failed for {ticker}: {e}")
            return pd.DataFrame()

    def _sanitize_df(self, df, ticker_name):
        """
        MANDATORY SANITIZATION LOGIC
        """
        # 1. Swap-Level Fix for yfinance MultiIndex Bugs
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(1):
                df = df.swaplevel(0, 1, axis=1)
            
            # 2. Flatten Columns
            new_cols = []
            for col in df.columns:
                attr = col[0].replace(" ", "")
                tick = col[1] if len(col) > 1 else ticker_name
                if tick == "": tick = ticker_name
                new_cols.append(f"{attr}_{tick}")
            df.columns = new_cols

        else:
            df.columns = [f"{c.replace(' ', '')}_{ticker_name}" for c in df.columns]

        # 3. Strict Datetime Index Normalization
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df.sort_index()

        # 4. Numeric Enforcement
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how='all')

        # 5. No Missing-Data Pass-through
        if df.empty:
            return pd.DataFrame()
            
        return df

# ============================================================
# CLASS 2: FinancialAnalysis
# ============================================================
class FinancialAnalysis:
    """
    Responsibilities:
    - ALL Financial Mathematics
    - Returns, Volatility, Greeks, Microstructure
    - MUST NOT download data
    """
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    def calculate_basic_metrics(self, df):
        """
        Computes log returns, rolling vol, SMA.
        """
        processed = df.copy()
        
        close_cols = [c for c in df.columns if "Close" in c and "Adj" not in c]
        
        for col in close_cols:
            ticker = col.split('_')[-1]
            
            # Log Returns
            processed[f'LogRet_{ticker}'] = np.log(processed[col] / processed[col].shift(1))
            
            # Rolling Realized Vol (Annualized)
            processed[f'RealizedVol_{ticker}'] = processed[f'LogRet_{ticker}'].rolling(window=20).std() * np.sqrt(252)
            
            # VWAP Logic (Approximation using OHLC if Volume exists)
            vol_col = f"Volume_{ticker}"
            if vol_col in df.columns:
                high = processed[f"High_{ticker}"]
                low = processed[f"Low_{ticker}"]
                close = processed[col]
                volume = processed[vol_col]
                
                tp = (high + low + close) / 3
                processed[f'VWAP_{ticker}'] = (tp * volume).cumsum() / volume.cumsum()

        return processed

    def calculate_microstructure(self, df, ticker):
        """
        Approximates Microstructure features using OHLC data since 
        Level 2 data is unavailable in yfinance.
        """
        close = df[f"Close_{ticker}"]
        high = df[f"High_{ticker}"]
        low = df[f"Low_{ticker}"]
        
        # 1. Estimate "Effective Spread" based on daily range volatility
        # We assume the spread is tighter when volatility is low
        # Using bfill to handle the start of the series
        spread_proxy = (high - low).rolling(5).mean().fillna(method='bfill') * 0.1
        
        # 2. Synthetic Bid/Ask derivation
        # We model the Bid/Ask center around the Close
        # synthetic_bid = close - (spread_proxy / 2)
        # synthetic_ask = close + (spread_proxy / 2)
        
        # 3. Order Imbalance Proxy (Buying Pressure)
        # Formula: Where did we close relative to the range? 
        # (Close - Low) / (High - Low) -> Scaled to -1 to 1
        rng = high - low
        # Avoid division by zero
        rng = rng.replace(0, 0.01) 
        
        # This acts as our "Imbalance" proxy (-1 = full bear, +1 = full bull)
        # Range 0 to 1 -> *2 -1 -> -1 to 1
        imbalance = ((close - low) / rng) * 2 - 1
        
        # 4. Microprice Proxy
        # Adjust the midprice by the imbalance factor
        microprice = close + (imbalance * (spread_proxy / 2))
        
        return microprice, imbalance

    # --- GREEKS IMPLEMENTATION ---
    def _black_scholes(self, S, K, T, r, sigma, option_type='call'):
        # Safety for T=0
        if T <= 0: T = 0.0001
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        return price

    def compute_numerical_greeks(self, chain_df):
        """
        Computes Finite Difference Greeks for every strike in the chain.
        """
        if chain_df.empty: return chain_df
        
        results = []
        
        # Constants for FD
        h_S = 0.01 * chain_df['underlying_price'].mean() # 1% price shift
        h_v = 0.01 # 1% vol shift
        h_t = 1/365.0 # 1 day shift
        h_r = 0.0001 # 1 bp shift
        
        for _, row in chain_df.iterrows():
            S = row['underlying_price']
            K = row['strike']
            # Time to expiry in years.
            try:
                exp_date = pd.to_datetime(row['expiry'])
                T = (exp_date - datetime.datetime.now()).days / 365.0
            except:
                T = 30/365.0 # Fallback
            
            if T <= 0: T = 0.001
            
            sigma = row['impliedVolatility']
            if pd.isna(sigma) or sigma == 0: sigma = 0.2
            
            r = self.r
            opt_type = row['type']
            
            # Base Price
            V = self._black_scholes(S, K, T, r, sigma, opt_type)
            
            # DELTA & GAMMA (Centred Difference)
            V_up = self._black_scholes(S + h_S, K, T, r, sigma, opt_type)
            V_dn = self._black_scholes(S - h_S, K, T, r, sigma, opt_type)
            
            delta = (V_up - V_dn) / (2 * h_S)
            gamma = (V_up - 2*V + V_dn) / (h_S ** 2)
            
            # VEGA
            V_vol_up = self._black_scholes(S, K, T, r, sigma + h_v, opt_type)
            V_vol_dn = self._black_scholes(S, K, T, r, sigma - h_v, opt_type)
            vega = (V_vol_up - V_vol_dn) / (2 * h_v)
            
            # THETA
            V_time_decay = self._black_scholes(S, K, T - h_t, r, sigma, opt_type)
            theta = (V_time_decay - V) / h_t
            
            # RHO
            V_r_up = self._black_scholes(S, K, T, r + h_r, sigma, opt_type)
            V_r_dn = self._black_scholes(S, K, T, r - h_r, sigma, opt_type)
            rho = (V_r_up - V_r_dn) / (2 * h_r)
            
            row['delta_num'] = delta
            row['gamma_num'] = gamma
            row['vega_num'] = vega / 100
            row['theta_num'] = theta
            row['rho_num'] = rho
            results.append(row)
            
        return pd.DataFrame(results)

# ============================================================
# CLASS 3: DashboardRenderer
# ============================================================
class DashboardRenderer:
    """
    Responsibilities:
    - Build multi-tab Plotly HTML
    - Offline embedding
    - Resize fix
    """
    def __init__(self, market_data, greeks_data, analysis_results):
        self.market_data = market_data
        self.greeks_data = greeks_data
        self.analysis = analysis_results
        self.layout_template = "plotly_dark"
        
    def generate_html(self, output_path):
        print_status("Rendering Dashboard...")
        
        figs = self._create_plots()
        plotly_js = get_plotlyjs()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Quantitative Analysis Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #111; color: #eee; margin: 0; padding: 20px; }}
                .tab {{ overflow: hidden; border-bottom: 1px solid #444; margin-bottom: 20px; }}
                .tab button {{ background-color: #222; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888; font-weight: bold; }}
                .tab button:hover {{ background-color: #333; color: #fff; }}
                .tab button.active {{ background-color: #007bff; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #444; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
                .chart-container {{ height: 800px; width: 100%; }}
            </style>
        </head>
        <body>

        <h2>Hedge Fund Grade Analytics Dashboard</h2>
        <p>Generated: {datetime.datetime.now()}</p>

        <div class="tab">
            <button class="tablinks" onclick="openTab(event, 'Tab1')" id="defaultOpen">Price & Vol</button>
            <button class="tablinks" onclick="openTab(event, 'Tab2')">IV & Skew</button>
            <button class="tablinks" onclick="openTab(event, 'Tab3')">Microstructure</button>
            <button class="tablinks" onclick="openTab(event, 'Tab4')">Greeks Explorer</button>
            <button class="tablinks" onclick="openTab(event, 'Tab5')">Correlations</button>
            <button class="tablinks" onclick="openTab(event, 'Tab6')">Summary</button>
        </div>

        <div id="Tab1" class="tabcontent">
            <div id="plot1" class="chart-container"></div>
        </div>

        <div id="Tab2" class="tabcontent">
            <div id="plot2" class="chart-container"></div>
        </div>

        <div id="Tab3" class="tabcontent">
            <div id="plot3" class="chart-container"></div>
        </div>

        <div id="Tab4" class="tabcontent">
            <div id="plot4" class="chart-container"></div>
        </div>
        
        <div id="Tab5" class="tabcontent">
            <div id="plot5" class="chart-container"></div>
        </div>

        <div id="Tab6" class="tabcontent">
            <div style="padding: 20px;">
                <h3>Risk Structural Summary</h3>
                <pre style="font-size: 14px; color: #0f0;">{self._generate_summary_text()}</pre>
            </div>
        </div>

        <script>
            var plot1_data = {figs['tab1']};
            var plot2_data = {figs['tab2']};
            var plot3_data = {figs['tab3']};
            var plot4_data = {figs['tab4']};
            var plot5_data = {figs['tab5']};

            Plotly.newPlot('plot1', plot1_data.data, plot1_data.layout);
            Plotly.newPlot('plot2', plot2_data.data, plot2_data.layout);
            Plotly.newPlot('plot3', plot3_data.data, plot3_data.layout);
            Plotly.newPlot('plot4', plot4_data.data, plot4_data.layout);
            Plotly.newPlot('plot5', plot5_data.data, plot5_data.layout);

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
                window.dispatchEvent(new Event('resize'));
            }}
            
            document.getElementById("defaultOpen").click();
        </script>

        </body>
        </html>
        """
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print_status(f"Dashboard saved to {output_path}")

    def _create_plots(self):
        figs = {}
        
        # --- TAB 1: Price & Vol ---
        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        for col in self.market_data.columns:
            if "Close_" in col:
                ticker = col.split("_")[1]
                fig1.add_trace(go.Scatter(x=self.market_data.index, y=self.market_data[col], name=f"{ticker} Price"), row=1, col=1)
            if "RealizedVol_" in col:
                ticker = col.split("_")[1]
                fig1.add_trace(go.Scatter(x=self.market_data.index, y=self.market_data[col], name=f"{ticker} Vol"), row=2, col=1)
        fig1.update_layout(template=self.layout_template, title="Price Action & Volatility Regime")
        figs['tab1'] = pio.to_json(fig1)
        
        # --- TAB 2: IV Skew (Proxy) ---
        fig2 = make_subplots(rows=2, cols=1)
        for col in self.market_data.columns:
            if "Close_" in col:
                ticker = col.split("_")[1]
                if f"LogRet_{ticker}" in self.market_data:
                    roll_skew = self.market_data[f"LogRet_{ticker}"].rolling(30).skew()
                    fig2.add_trace(go.Scatter(x=self.market_data.index, y=roll_skew, name=f"{ticker} Rlzd Skew"), row=1, col=1)
                    fig2.add_trace(go.Histogram(x=self.market_data[f"LogRet_{ticker}"], name=f"{ticker} Dist"), row=2, col=1)
        fig2.update_layout(template=self.layout_template, title="Realized Skew & Return Distribution")
        figs['tab2'] = pio.to_json(fig2)

        # --- TAB 3: Microstructure (UPDATED LOGIC) ---
        fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True)
        try:
            # We assume "Close_<ticker>" exists. We pick the first one.
            ticker_list = [c.split("_")[1] for c in self.market_data.columns if "Close_" in c]
            if ticker_list:
                first_ticker = ticker_list[0]
                mp_col = f"Microprice_{first_ticker}"
                imb_col = f"Imbalance_{first_ticker}"
                close_col = f"Close_{first_ticker}"
                
                if mp_col in self.market_data.columns:
                    # Plot Price vs Microprice
                    fig3.add_trace(go.Scatter(x=self.market_data.index, y=self.market_data[close_col], name="Mid (Close)"), row=1, col=1)
                    fig3.add_trace(go.Scatter(x=self.market_data.index, y=self.market_data[mp_col], name="Microprice (Proxy)", line=dict(dash='dot', width=1)), row=1, col=1)
                    
                    # Plot Imbalance
                    fig3.add_trace(go.Scatter(x=self.market_data.index, y=self.market_data[imb_col], name="Order Imbalance", fill='tozeroy'), row=2, col=1)
                    
                    # Add zero line to imbalance
                    fig3.add_shape(type="line", x0=self.market_data.index[0], y0=0, x1=self.market_data.index[-1], y1=0, line=dict(color="white", width=1, dash="dash"), row=2, col=1)
        except Exception as e:
            print(f"Error plotting microstructure: {e}")
            
        fig3.update_layout(template=self.layout_template, title="Intraday Microstructure (Algorithmic Approximation)")
        figs['tab3'] = pio.to_json(fig3)

        # --- TAB 4: Greeks Explorer (UPDATED: Theoretical Fallback) ---
        if self.greeks_data.empty and not self.market_data.empty:
            # Theoretical Surface
            try:
                t_col = [c for c in self.market_data.columns if "Close_" in c][0]
                S = self.market_data[t_col].iloc[-1]
                
                strikes = np.linspace(S * 0.8, S * 1.2, 20)
                vols = np.linspace(0.1, 0.5, 20)
                X, Y = np.meshgrid(strikes, vols)
                T = 30/365.0
                r = 0.04
                
                d1 = (np.log(S / X) + (r + 0.5 * Y ** 2) * T) / (Y * np.sqrt(T))
                Z = stats.norm.cdf(d1)
                
                fig4 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
                fig4.update_layout(
                    template=self.layout_template, 
                    title=f"Theoretical Delta Surface (30D) - {t_col.split('_')[1]} [No Live Options Data]",
                    scene=dict(xaxis_title='Strike', yaxis_title='Volatility', zaxis_title='Delta')
                )
            except Exception as e:
                fig4 = go.Figure()
                fig4.add_annotation(text=f"Error generating theoretical surface: {e}", showarrow=False)
                
        elif not self.greeks_data.empty:
            df_g = self.greeks_data[self.greeks_data['type'] == 'call']
            fig4 = go.Figure(data=[go.Scatter3d(
                x=df_g['strike'],
                y=df_g['impliedVolatility'],
                z=df_g['delta_num'],
                mode='markers',
                marker=dict(size=4, color=df_g['delta_num'], colorscale='Viridis'),
            )])
            fig4.update_layout(
                template=self.layout_template, 
                title="Live Option Greeks Surface",
                scene=dict(xaxis_title='Strike', yaxis_title='Implied Vol', zaxis_title='Delta')
            )
        else:
            fig4 = go.Figure()
            fig4.add_annotation(text="No Options Data Available", showarrow=False)
            
        figs['tab4'] = pio.to_json(fig4)

        # --- TAB 5: Correlations (UPDATED: DropNa Fix) ---
        fig5 = go.Figure()
        tickers = list(set([c.split("_")[1] for c in self.market_data.columns if "Close_" in c]))
        
        if len(tickers) >= 2:
            t1, t2 = tickers[0], tickers[1]
            col1 = f"LogRet_{t1}"
            col2 = f"LogRet_{t2}"
            
            if col1 in self.market_data and col2 in self.market_data:
                # Force alignment and drop NaNs for correlation calc
                pair_df = self.market_data[[col1, col2]].dropna()
                
                if not pair_df.empty:
                    roll_corr = pair_df[col1].rolling(30).corr(pair_df[col2])
                    fig5.add_trace(go.Scatter(x=pair_df.index, y=roll_corr, name=f"Corr {t1}/{t2}"))
                    fig5.add_shape(type="line", x0=pair_df.index[0], y0=0, x1=pair_df.index[-1], y1=0, line=dict(color="gray", width=1, dash="dash"))
                    fig5.update_layout(template=self.layout_template, title=f"30-Period Rolling Correlation: {t1} vs {t2}")
                else:
                    fig5.add_annotation(text="Insufficient overlapping data after cleaning", showarrow=False)
        else:
             fig5.add_annotation(text="Need >1 ticker for correlation", showarrow=False)
             
        figs['tab5'] = pio.to_json(fig5)
        
        return figs

    def _generate_summary_text(self):
        txt = "MARKET REGIME SUMMARY:\n"
        txt += "=" * 30 + "\n"
        for col in self.market_data.columns:
            if "RealizedVol_" in col:
                last_val = self.market_data[col].iloc[-1]
                ticker = col.split("_")[1]
                txt += f"{ticker} Annualized Vol: {last_val:.2%}\n"
        
        txt += "\nSKEW ALERTS:\n"
        for col in self.market_data.columns:
            if "Close_" in col:
                ticker = col.split("_")[1]
                if f"LogRet_{ticker}" in self.market_data:
                    skew = self.market_data[f"LogRet_{ticker}"].skew()
                    txt += f"{ticker} Global Skew: {skew:.4f} ({'Bearish Tail' if skew < -0.5 else 'Neutral/Bullish'})\n"
        
        return txt

# ============================================================
# MAIN EXECUTION FLOW
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Quantitative Finance Dashboard")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"], help="List of tickers")
    parser.add_argument("--output-dir", default="./market_data", help="Data cache directory")
    parser.add_argument("--lookback", type=int, default=1, help="Years of history")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk free rate")
    parser.add_argument("--intraday", action="store_true", help="Fetch intraday data")
    parser.add_argument("--compute-greeks", action="store_true", help="Compute numerical Greeks")
    parser.add_argument("--interval", default="1d", help="Interval (1m, 5m, 1h, 1d)")

    args = parser.parse_args()
    
    if args.intraday and args.interval == "1d":
        args.interval = "5m" 
        print_status("Intraday flag set: Overriding interval to 5m")

    # 1. Initialize Ingestion
    ingestor = DataIngestion(args.output_dir, args.lookback, args.intraday, args.interval)
    
    combined_df = pd.DataFrame()
    
    # 2. Ingest Data
    for t in args.tickers:
        df = ingestor.get_ticker_df(t)
        if not df.empty:
            if combined_df.empty:
                combined_df = df
            else:
                # Outer join to preserve all timestamps
                combined_df = combined_df.join(df, how='outer')
    
    if combined_df.empty:
        print("No data available. Exiting.")
        sys.exit(1)

    # 3. Financial Analysis
    analyzer = FinancialAnalysis(risk_free_rate=args.risk_free_rate)
    
    # Basic Metrics
    analysis_df = analyzer.calculate_basic_metrics(combined_df)
    
    # Microstructure (Deterministically calculated for ALL tickers, even on Daily data)
    for t in args.tickers:
        if f"Close_{t}" in analysis_df.columns:
            mp, imb = analyzer.calculate_microstructure(analysis_df, t)
            analysis_df[f"Microprice_{t}"] = mp
            analysis_df[f"Imbalance_{t}"] = imb

    # Greeks (if requested)
    greeks_df = pd.DataFrame()
    if args.compute_greeks:
        target_ticker = args.tickers[0]
        chain = ingestor.get_options_chain(target_ticker)
        if not chain.empty:
            print_status(f"Computing Numerical Greeks for {target_ticker}...")
            greeks_df = analyzer.compute_numerical_greeks(chain)
        else:
            print_status("No options data found, Greeks tab will use theoretical fallback.")

    # 4. Rendering
    renderer = DashboardRenderer(analysis_df, greeks_df, {})
    ts = int(time.time())
    renderer.generate_html(f"dashboard_{ts}.html")
    
    print_status("Done.")

if __name__ == "__main__":
    main()
