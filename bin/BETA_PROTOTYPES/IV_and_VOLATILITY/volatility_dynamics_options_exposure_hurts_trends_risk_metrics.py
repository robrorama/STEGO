# SCRIPTNAME: ok.volatility_dynamics_options_exposure_hurts_trends_risk_metrics.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as si
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py_offline
from datetime import datetime, timedelta

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
DEFAULT_TICKERS = ['SPY', 'QQQ', 'IWM']
DEFAULT_RISK_FREE = 0.04
DEFAULT_LOOKBACK = 1

class DataIngestion:
    """
    Handles robust data fetching, disk caching, and sanitization.
    Strictly prohibits the Analysis layer from touching the API directly.
    """
    def __init__(self, output_dir, lookback_years=1, intraday=False, interval='5m'):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        self.intraday = intraday
        self.interval = interval
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        The 'Universal Fixer' for yfinance inconsistencies.
        """
        if df.empty:
            return df

        # 1. Swap Levels if (Price, Ticker) format
        if isinstance(df.columns, pd.MultiIndex):
            # Check if the first level contains the ticker (Standard yfinance behavior varies)
            if ticker in df.columns.get_level_values(0):
                # Format is likely (Ticker, Price) - do nothing or drop level
                df.columns = df.columns.droplevel(0)
            elif ticker in df.columns.get_level_values(1):
                # Format is likely (Price, Ticker) -> Swap
                df = df.swaplevel(0, 1, axis=1)
                df.columns = df.columns.droplevel(0)

        # 2. Flatten Columns & Rename
        # Ensure we have standard columns. 
        # Note: yfinance auto_adjust=False gives 'Adj Close' and 'Close'.
        # We standardize to simple names prefixed by ticker for safety.
        clean_cols = {}
        for col in df.columns:
            clean_cols[col] = f"{ticker}_{col}"
        df = df.rename(columns=clean_cols)

        # 3. Strict Typing
        # Timezone removal
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Numeric Coercion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        return df

    def get_data(self, ticker: str) -> pd.DataFrame:
        """
        Disk-first logic: Check CSV -> Read -> Return. 
        Else -> Download -> Sanitize -> Save -> Return.
        """
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")

        # A. Disk Check
        if os.path.exists(file_path):
            print(f"[{ticker}] Found local cache. Loading...")
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Re-sanitize ensures types are correct after CSV read
                # We assume CSV headers are already flattened, so we just check types
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
            except Exception as e:
                print(f"[{ticker}] Corrupt cache ({e}). Redownloading.")

        # B. API Download
        print(f"[{ticker}] Downloading from API...")
        
        # Calculate start date
        start_date = (datetime.now() - timedelta(days=self.lookback_years*365)).strftime('%Y-%m-%d')
        
        try:
            # Rate limit protection
            time.sleep(1.0) 
            
            df = yf.download(
                ticker, 
                start=start_date, 
                interval=self.interval if self.intraday else '1d',
                auto_adjust=False,
                progress=False,
                group_by='column' 
            )

            # Shadow Backfill
            if df.empty:
                print(f"[{ticker}] Initial download empty. Attempting 'max' period backfill.")
                time.sleep(1.0)
                df = yf.download(
                    ticker, 
                    period='max',
                    interval=self.interval if self.intraday else '1d',
                    auto_adjust=False,
                    progress=False,
                    group_by='column'
                )

            # Sanitize
            df = self._sanitize_df(df, ticker)
            
            # Save
            if not df.empty:
                df.to_csv(file_path)
            
            return df

        except Exception as e:
            print(f"[{ticker}] Download failed: {e}")
            return pd.DataFrame()

    def get_options_chain(self, ticker: str):
        """
        Fetches options chain for GEX analysis. 
        Note: Options data is real-time and rarely caches well for static analysis 
        unless complex, so we fetch live for the engine run.
        """
        try:
            tk = yf.Ticker(ticker)
            # Get nearest expiration
            if not tk.options:
                return None, None
            
            expiry = tk.options[0] # Nearest expiry
            chain = tk.option_chain(expiry)
            return chain.calls, chain.puts, expiry
        except Exception as e:
            print(f"[{ticker}] Options fetch failed: {e}")
            return None, None, None


class FinancialAnalysis:
    """
    Pure math engine. Stateless where possible.
    """
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    # --- BSM & Greeks ---
    def _d1_d2(self, S, K, T, r, sigma):
        # Avoid division by zero
        if T <= 0 or sigma <= 0:
            return 0, 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def bs_price(self, S, K, T, r, sigma, flag='c'):
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        if flag == 'c':
            return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

    def calculate_greeks(self, S, K, T, r, sigma, flag='c'):
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        delta = si.norm.cdf(d1) if flag == 'c' else si.norm.cdf(d1) - 1
        gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * si.norm.pdf(d1) * np.sqrt(T) / 100 # Standard convention
        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2 if flag == 'c' else -d2)) / 365
        rho = (K * T * np.exp(-r * T) * si.norm.cdf(d2 if flag == 'c' else -d2)) / 100

        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

    def implied_volatility(self, market_price, S, K, T, r, flag='c'):
        """Brent's method to find root of BSM - MarketPrice"""
        if T <= 0: return 0.0
        
        def obj_func(sigma):
            return self.bs_price(S, K, T, r, sigma, flag) - market_price
        
        try:
            # Search space for IV: 1% to 500%
            return brentq(obj_func, 0.01, 5.0)
        except:
            return np.nan

    def compute_gex(self, calls, puts, spot_price):
        """
        Calculate Gamma Exposure per strike.
        GEX = Gamma * OpenInterest * 100 * Spot * Direction
        Direction: Dealer is Long Calls (+), Short Puts (-) -> Wait.
        Standard Market GEX:
        Dealers sell calls to customers -> Dealer Short Gamma -> Price Up, Dealer Buys.
        Dealers buy puts from customers -> Dealer Long Gamma.
        
        However, the standard 'GEX' chart metric usually assumes:
        Call OI: Contributing + Gamma (Dealers Short Calls, hence they must hedge by buying as price rises? 
        Actually, the convention is: Call OI implies Dealers are Short Calls -> Negative Gamma.
        Put OI implies Dealers are Short Puts -> Positive Gamma.
        
        Let's use the SPOTGamma / SqueezeMetrics convention:
        Call GEX = Gamma * OI * 100 * Spot (Positive contribution to Index)
        Put GEX = Gamma * OI * 100 * Spot * -1 (Negative contribution to Index)
        """
        # Time to expiry (approx 1 month for nearest usually, simplifying to 30/365 for single chain calculation if T not provided)
        # In a real engine, T is calculated per option. Here we assume T=30days for the profile snapshot
        T = 30 / 365.0 
        
        gex_data = []

        # Process Calls
        for _, row in calls.iterrows():
            iv = row.get('impliedVolatility', 0)
            if iv == 0: continue
            
            gamma = self.calculate_greeks(spot_price, row['strike'], T, self.r, iv, 'c')['Gamma']
            # Call GEX: Dealers short calls -> Negative Gamma? 
            # Convention: Positive bar = Call Wall.
            val = gamma * row['openInterest'] * 100 * spot_price
            gex_data.append({'Strike': row['strike'], 'GEX': val, 'Type': 'Call'})

        # Process Puts
        for _, row in puts.iterrows():
            iv = row.get('impliedVolatility', 0)
            if iv == 0: continue
            
            gamma = self.calculate_greeks(spot_price, row['strike'], T, self.r, iv, 'p')['Gamma']
            # Put GEX: Negative bar
            val = gamma * row['openInterest'] * 100 * spot_price * -1
            gex_data.append({'Strike': row['strike'], 'GEX': val, 'Type': 'Put'})
            
        return pd.DataFrame(gex_data)

    # --- Time Series Analytics ---
    def process_timeseries(self, df: pd.DataFrame, ticker: str):
        p_col = f"{ticker}_Close"
        o_col = f"{ticker}_Open"
        h_col = f"{ticker}_High"
        l_col = f"{ticker}_Low"
        v_col = f"{ticker}_Volume"
        
        # 1. Returns
        df['Log_Ret'] = np.log(df[p_col] / df[p_col].shift(1))
        
        # 2. Realized Volatility (Annualized)
        for w in [5, 10, 21, 60]:
            df[f'RVol_{w}'] = df['Log_Ret'].rolling(window=w).std() * np.sqrt(252)

        # 3. Vol Regime (Z-Score of 21d Vol vs 1y avg)
        rolling_mean_vol = df['RVol_21'].rolling(window=252).mean()
        rolling_std_vol = df['RVol_21'].rolling(window=252).std()
        df['Vol_ZScore'] = (df['RVol_21'] - rolling_mean_vol) / rolling_std_vol
        
        # 4. Trends
        df['SMA_20'] = df[p_col].rolling(window=20).mean()
        df['SMA_50'] = df[p_col].rolling(window=50).mean()
        df['SMA_200'] = df[p_col].rolling(window=200).mean()
        
        # Market State
        df['State'] = 'Neutral'
        bull_mask = (df[p_col] > df['SMA_200']) & (df['SMA_50'] > df['SMA_200'])
        df.loc[bull_mask, 'State'] = 'Bull'

        # 5. Microstructure
        # Microprice
        df['Microprice'] = (df[h_col] + df[l_col] + df[p_col]) / 3
        
        # OFI Proxy
        df['Price_Delta'] = df[p_col] - df[o_col]
        df['OFI_Sign'] = np.where(df['Price_Delta'] > 0, 1, np.where(df['Price_Delta'] < 0, -1, 0))
        df['OFI_Flow'] = df['OFI_Sign'] * df[v_col]
        df['Cumulative_OFI'] = df['OFI_Flow'].cumsum()
        
        # Anchored VWAP (YTD)
        current_year = df.index[-1].year
        ytd_mask = df.index.year == current_year
        df.loc[ytd_mask, 'PV'] = df.loc[ytd_mask, p_col] * df.loc[ytd_mask, v_col]
        df.loc[ytd_mask, 'CumPV'] = df.loc[ytd_mask, 'PV'].cumsum()
        df.loc[ytd_mask, 'CumVol'] = df.loc[ytd_mask, v_col].cumsum()
        df['AVWAP_YTD'] = df['CumPV'] / df['CumVol']

        return df

    def calculate_hurst(self, series):
        """
        Calculate Hurst Exponent using R/S Analysis.
        H < 0.5: Mean Reverting
        H = 0.5: Random Walk
        H > 0.5: Trending
        """
        try:
            series = series.dropna()
            if len(series) < 100: return 0.5
            
            # Create a range of lag values
            lags = range(2, 20)
            tau = []
            
            for lag in lags:
                # Calculate the variance of the differences
                # Simplification of R/S for speed in rolling windows
                # Using standard deviation of differences method
                diff = np.subtract(series[lag:], series[:-lag])
                tau.append(np.sqrt(np.std(diff)))
            
            # Regress log(tau) vs log(lags)
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0] * 2 # Adjustment for this specific estimation method
            return hurst
        except:
            return 0.5

class DashboardRenderer:
    """
    Renders offline HTML dashboard with embedded JS.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_html(self, ticker_data_map, ticker_gex_map, correlations):
        
        tabs_html = ""
        content_html = ""
        
        # Get Plotly JS source
        plotly_js = py_offline.get_plotlyjs()

        for i, ticker in enumerate(ticker_data_map.keys()):
            df = ticker_data_map[ticker]
            gex_df = ticker_gex_map.get(ticker)
            
            # --- Generate Figures ---
            
            # 1. Price & Volatility Subplot
            fig_pv = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig_pv.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_Close"], name='Price', line=dict(color='black')), row=1, col=1)
            fig_pv.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
            fig_pv.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='blue', width=1)), row=1, col=1)
            fig_pv.add_trace(go.Scatter(x=df.index, y=df['RVol_21'], name='RVol (21d)', line=dict(color='red')), row=2, col=1)
            fig_pv.update_layout(title=f"{ticker} Price & Volatility Regime", height=600, template="plotly_white")
            
            # 2. Options Analysis (GEX)
            if gex_df is not None and not gex_df.empty:
                fig_gex = go.Figure()
                calls = gex_df[gex_df['Type'] == 'Call']
                puts = gex_df[gex_df['Type'] == 'Put']
                fig_gex.add_trace(go.Bar(x=calls['Strike'], y=calls['GEX'], name='Call GEX', marker_color='green'))
                fig_gex.add_trace(go.Bar(x=puts['Strike'], y=puts['GEX'], name='Put GEX', marker_color='red'))
                
                # Net GEX Gauge
                net_gex = gex_df['GEX'].sum()
                fig_gex.add_annotation(text=f"Net GEX: ${net_gex/1e9:.2f}B", x=0.5, y=1.1, xref='paper', yref='paper', showarrow=False, font=dict(size=20))
                fig_gex.update_layout(title=f"{ticker} Gamma Exposure Profile", barmode='relative', height=500, template="plotly_white")
            else:
                fig_gex = go.Figure().add_annotation(text="No Options Data Available", showarrow=False)

            # 3. Microstructure (OFI & Microprice)
            fig_micro = make_subplots(specs=[[{"secondary_y": True}]])
            fig_micro.add_trace(go.Scatter(x=df.index, y=df['Microprice'], name='Microprice'), secondary_y=False)
            fig_micro.add_trace(go.Scatter(x=df.index, y=df['Cumulative_OFI'], name='Cumul. OFI', fill='tozeroy', line=dict(color='purple', width=0)), secondary_y=True)
            fig_micro.update_layout(title=f"{ticker} Microstructure: Price vs Order Flow", height=500, template="plotly_white")

            # 4. Fractal Analysis (AVWAP + Hurst)
            latest_hurst = df['Hurst_Rolling'].iloc[-1] if 'Hurst_Rolling' in df.columns else 0.5
            fig_frac = go.Figure()
            fig_frac.add_trace(go.Scatter(x=df.index, y=df[f"{ticker}_Close"], name='Price'))
            fig_frac.add_trace(go.Scatter(x=df.index, y=df['AVWAP_YTD'], name='YTD AVWAP', line=dict(dash='dash', color='orange')))
            fig_frac.add_annotation(text=f"Hurst Exp: {latest_hurst:.2f}<br>{'Trending' if latest_hurst > 0.55 else 'Mean Reverting' if latest_hurst < 0.45 else 'Random'}", 
                                   x=0.05, y=0.95, xref='paper', yref='paper', showarrow=False, bgcolor="white", bordercolor="black")
            fig_frac.update_layout(title=f"{ticker} Fractal Analysis & Anchored VWAP", height=500, template="plotly_white")

            # Convert to Divs
            div_pv = py_offline.plot(fig_pv, output_type='div', include_plotlyjs=False)
            div_gex = py_offline.plot(fig_gex, output_type='div', include_plotlyjs=False)
            div_micro = py_offline.plot(fig_micro, output_type='div', include_plotlyjs=False)
            div_frac = py_offline.plot(fig_frac, output_type='div', include_plotlyjs=False)

            # Build Tab Content
            active = "active" if i == 0 else ""
            display = "block" if i == 0 else "none"
            
            tabs_html += f'<button class="tablinks {active}" onclick="openCity(event, \'{ticker}\')">{ticker}</button>'
            
            content_html += f"""
            <div id="{ticker}" class="tabcontent" style="display: {display};">
                <div class="row">{div_pv}</div>
                <div class="row">
                    <div class="half">{div_gex}</div>
                    <div class="half">{div_frac}</div>
                </div>
                <div class="row">{div_micro}</div>
            </div>
            """

        # Correlation Heatmap (Global)
        if correlations is not None and not correlations.empty:
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlations.values,
                x=correlations.columns,
                y=correlations.index,
                colorscale='Viridis'))
            fig_corr.update_layout(title="Rolling 60d Correlation Matrix", height=500)
            div_corr = py_offline.plot(fig_corr, output_type='div', include_plotlyjs=False)
            
            tabs_html += '<button class="tablinks" onclick="openCity(event, \'Correlations\')">Correlations</button>'
            content_html += f'<div id="Correlations" class="tabcontent" style="display: none;">{div_corr}</div>'

        # Full HTML Template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <title>Quant Engine Dashboard</title>
        <script>{plotly_js}</script>
        <style>
            body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f4f4f9; }}
            .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
            .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }}
            .tab button:hover {{ background-color: #ddd; }}
            .tab button.active {{ background-color: #ccc; }}
            .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; animation: fadeEffect 1s; }}
            @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            .row {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
            .half {{ width: 50%; }}
            @media screen and (max-width: 800px) {{ .half {{ width: 100%; }} }}
        </style>
        </head>
        <body>

        <h2>Hedge Fund Grade Market Engine</h2>
        <div class="tab">
            {tabs_html}
        </div>

        {content_html}

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
            
            // CRITICAL: Force Plotly resize on tab switch
            window.dispatchEvent(new Event('resize'));
        }}
        </script>
        </body>
        </html>
        """
        
        with open(os.path.join(self.output_dir, "dashboard.html"), "w", encoding='utf-8') as f:
            f.write(html_template)
        print(f"\n[Dashboard] Successfully generated at {os.path.join(self.output_dir, 'dashboard.html')}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Institutional Grade Market Engine")
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS, help='List of tickers')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Data/Chart storage path')
    parser.add_argument('--lookback', type=int, default=DEFAULT_LOOKBACK, help='Years of history')
    parser.add_argument('--risk-free-rate', type=float, default=DEFAULT_RISK_FREE, help='Risk free rate (decimal)')
    parser.add_argument('--intraday', action='store_true', help='Enable intraday fetch')
    parser.add_argument('--intraday-interval', type=str, default='5m', help='Intraday interval')
    
    args = parser.parse_args()
    
    print(f"--- Starting Market Engine ---")
    print(f"Tickers: {args.tickers}")
    print(f"Storage: {args.output_dir}")
    
    # 1. Initialize Modules
    ingestor = DataIngestion(args.output_dir, args.lookback, args.intraday, args.intraday_interval)
    analyst = FinancialAnalysis(args.risk_free_rate)
    renderer = DashboardRenderer(args.output_dir)
    
    ticker_data_map = {}
    ticker_gex_map = {}
    
    # 2. Pipeline Execution
    for ticker in args.tickers:
        print(f"\nProcessing {ticker}...")
        
        # A. Ingestion
        df = ingestor.get_data(ticker)
        if df.empty:
            print(f"Skipping {ticker} due to data failure.")
            continue
            
        # B. Analysis (Time Series)
        df_analyzed = analyst.process_timeseries(df, ticker)
        
        # Calculate Hurst (Rolling 100 day)
        col_name = f"{ticker}_Close"
        df_analyzed['Hurst_Rolling'] = df_analyzed[col_name].rolling(100).apply(analyst.calculate_hurst)
        
        ticker_data_map[ticker] = df_analyzed
        
        # C. Analysis (Derivatives/GEX)
        # Only perform if we have recent data
        if (datetime.now() - df_analyzed.index[-1]).days < 5:
            print(f"[{ticker}] Fetching Option Chain for GEX...")
            calls, puts, expiry = ingestor.get_options_chain(ticker)
            if calls is not None:
                spot_price = df_analyzed[f"{ticker}_Close"].iloc[-1]
                gex_df = analyst.compute_gex(calls, puts, spot_price)
                ticker_gex_map[ticker] = gex_df
                print(f"[{ticker}] GEX Calculated. Net GEX: {gex_df['GEX'].sum():,.0f}")
            else:
                print(f"[{ticker}] No options found.")
        else:
            print(f"[{ticker}] Data too old for Options analysis. Skipping GEX.")

    # D. Multi-Asset Correlation
    print("\nCalculating Correlations...")
    close_prices = pd.DataFrame({t: d[f"{t}_Close"] for t, d in ticker_data_map.items()})
    
    # FIX: Added fill_method=None to avoid FutureWarning
    corr_matrix = close_prices.pct_change(fill_method=None).corr()
    
    # 3. Render
    print("Rendering Dashboard...")
    renderer.generate_html(ticker_data_map, ticker_gex_map, corr_matrix)
    print("Done.")

if __name__ == "__main__":
    main()
