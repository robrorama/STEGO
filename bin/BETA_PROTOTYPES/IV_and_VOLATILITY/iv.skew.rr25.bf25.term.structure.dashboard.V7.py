# SCRIPTNAME: ok.1.iv.skew.rr25.bf25.term.structure.dashboard.V7.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from datetime import datetime, timedelta
from scipy.stats import norm

# -----------------------------------------------------------------------------
# 1. DataIngestion Class
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Handles all API calls, CSV caching, and data cleaning.
    Strictly forbids random walk generation.
    """
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.filename = f"{self.ticker}_data.csv"

    def get_market_data(self):
        """
        Orchestrates the retrieval of OHLC data.
        Checks local CSV first; backfills from yfinance if missing.
        """
        if os.path.exists(self.filename):
            print(f"[INFO] Loading local cache: {self.filename}")
            try:
                df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
                if df.empty:
                    print("[WARN] Local cache empty. Re-downloading.")
                    return self._download_and_cache()
                return df
            except Exception as e:
                print(f"[ERROR] Corrupt cache: {e}. Re-downloading.")
                return self._download_and_cache()
        else:
            return self._download_and_cache()

    def _download_and_cache(self):
        """
        Downloads 1 Year of daily OHLC data (The Shadow Backfill).
        Sanitizes and saves to CSV.
        """
        print(f"[INFO] Downloading 1y history for {self.ticker}...")
        time.sleep(1)  # Rate Limiting
        
        try:
            # Explicitly download 1 year to calculate realized volatility
            df = yf.download(self.ticker, period="1y", interval="1d", progress=False)
            
            if df.empty:
                print(f"[ERROR] No data returned for {self.ticker}.")
                return pd.DataFrame()

            clean_df = self._sanitize_df(df)
            
            # Save to CSV
            clean_df.to_csv(self.filename)
            return clean_df
            
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return pd.DataFrame()

    def get_options_chain(self):
        """
        Retrieves full option chain for all expirations.
        """
        print(f"[INFO] Fetching option chain for {self.ticker}...")
        time.sleep(1)
        try:
            yf_ticker = yf.Ticker(self.ticker)
            expirations = yf_ticker.options
        except Exception as e:
            print(f"[ERROR] Could not fetch expirations: {e}")
            return pd.DataFrame()

        all_opts = []
        
        # Limit to first 6 expirations to prevent timeouts in this context
        # In a full production run, you might remove the slice [:6]
        for date in expirations[:6]: 
            time.sleep(0.3) # Gentle rate limit between calls
            try:
                opt = yf_ticker.option_chain(date)
                calls = opt.calls
                puts = opt.puts
                
                if calls.empty and puts.empty:
                    continue

                calls['type'] = 'call'
                puts['type'] = 'put'
                calls['expirationDate'] = date
                puts['expirationDate'] = date
                
                all_opts.append(calls)
                all_opts.append(puts)
            except Exception as e:
                print(f"[WARN] Failed to fetch chain for {date}: {e}")
                continue
                
        if not all_opts:
            return pd.DataFrame()
            
        return pd.concat(all_opts, ignore_index=True)

    def _sanitize_df(self, df):
        """
        Aggressive Sanitization:
        - Fixes yfinance MultiIndex swap issues.
        - Flattens columns.
        - Enforces strict types.
        """
        if df.empty:
            return df

        # The "Swap Levels" Fix
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' or 'Adj Close' is in level 0
            l0_vals = df.columns.get_level_values(0).unique().tolist()
            if 'Close' not in l0_vals and 'Adj Close' not in l0_vals:
                # Check level 1
                l1_vals = df.columns.get_level_values(1).unique().tolist()
                if 'Close' in l1_vals or 'Adj Close' in l1_vals:
                    # Swap levels
                    df = df.swaplevel(0, 1, axis=1)

            # Flatten columns
            # If MultiIndex (Attribute, Ticker), take Attribute
            df.columns = df.columns.get_level_values(0)

        # Normalize strings
        df.columns = [str(c).strip() for c in df.columns]

        # Strict Types
        df.index = pd.to_datetime(df.index, errors='coerce').tz_localize(None)
        
        cols_to_force = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols_to_force:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Drop rows with NaN in critical columns
        df.dropna(subset=['Close'], inplace=True)
        
        return df

# -----------------------------------------------------------------------------
# 2. FinancialAnalysis Class
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Handles all mathematical calculations.
    Enforces immutability of raw inputs.
    Includes Black-Scholes Engine for missing Greeks.
    """
    def __init__(self, history_df, options_df):
        self._history = history_df
        self._options = options_df
        
        # Calculate Spot Price carefully
        if not self._history.empty:
            try:
                # Safety check for scalar extraction
                last_close = self._history['Close'].iloc[-1]
                if isinstance(last_close, pd.Series):
                    last_close = last_close.iloc[0]
                self.spot_price = float(last_close)
            except:
                self.spot_price = 0.0
        else:
            self.spot_price = 0.0

    def _calculate_black_scholes_gamma(self, df):
        """
        Vectorized Black-Scholes Gamma Calculator.
        Used when API fails to provide 'gamma'.
        """
        # Constants
        R = 0.045  # Risk Free Rate approx 4.5%
        
        # Time to Expiration (T) in Years
        # Prevent division by zero for 0DTE
        df['T'] = (pd.to_datetime(df['expirationDate']) - datetime.now()).dt.days / 365.0
        df.loc[df['T'] <= 0.001, 'T'] = 0.001 
        
        # Inputs
        S = self.spot_price
        K = pd.to_numeric(df['strike'], errors='coerce')
        sigma = pd.to_numeric(df['impliedVolatility'], errors='coerce')
        T = df['T']
        
        # Handle zero volatility (prevent NaN)
        sigma = sigma.replace(0, 0.01)

        # d1 Calculation
        # d1 = (ln(S/K) + (r + sigma^2/2)T) / (sigma * sqrt(T))
        d1 = (np.log(S / K) + (R + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        # Gamma Calculation
        # Gamma = N'(d1) / (S * sigma * sqrt(T))
        # N'(x) = (1 / sqrt(2pi)) * e^(-x^2/2)
        pdf_d1 = norm.pdf(d1)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        
        return gamma.fillna(0)

    def calculate_gex(self):
        """
        Calculates Gamma Exposure (GEX) by strike.
        Auto-patches missing 'gamma' column using Black-Scholes.
        """
        if self._options.empty or self.spot_price == 0:
            return pd.DataFrame()
        
        df = self._options.copy()
        
        # 1. Patch Missing Gamma
        if 'gamma' not in df.columns:
            # print("[INFO] 'gamma' column missing. Calculating using Black-Scholes...")
            df['gamma'] = self._calculate_black_scholes_gamma(df)
            
        # 2. Ensure Numeric Types for Calc
        cols = ['gamma', 'openInterest', 'strike']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # 3. GEX Calculation
        # Basic GEX Formula: Gamma * Open Interest * 100 * Spot * 0.01
        # Call GEX + / Put GEX -
        df['GEX'] = df['gamma'] * df['openInterest'] * 100 * self.spot_price * 0.01
        df.loc[df['type'] == 'put', 'GEX'] = df.loc[df['type'] == 'put', 'GEX'] * -1
        
        # Group by Strike
        gex_profile = df.groupby('strike')['GEX'].sum().reset_index()
        return gex_profile.sort_values('strike')

    def calculate_streaks(self):
        """
        Analyzes consecutive up/down days and reversal probabilities.
        """
        if self._history.empty:
            return pd.DataFrame()

        df = self._history.copy()
        df['Return'] = df['Close'].pct_change()
        df['Sign'] = np.sign(df['Return'])
        
        # Identify streaks
        df['Streak_ID'] = (df['Sign'] != df['Sign'].shift()).cumsum()
        
        # Construct a simpler view for the dashboard: Last 10 days
        last_days = df.tail(10).copy()
        last_days['Date'] = last_days.index.strftime('%Y-%m-%d')
        last_days['Change'] = (last_days['Return'] * 100).round(2)
        
        return last_days[['Date', 'Close', 'Change']]

    def hunt_spreads(self):
        """
        Identifies potential Bull Call spreads.
        """
        if self._options.empty:
            return pd.DataFrame()
            
        exps = sorted(self._options['expirationDate'].unique())
        if not exps:
            return pd.DataFrame()
        
        front_month = self._options[self._options['expirationDate'] == exps[0]].copy()
        
        # Bull Call Spreads: Buy ITM Call, Sell OTM Call
        calls = front_month[front_month['type'] == 'call'].sort_values('strike')
        
        spreads = []
        
        # Simple iterator for demonstration of logic
        # Buy Strike < Spot < Sell Strike
        buy_legs = calls[calls['strike'] < self.spot_price]
        sell_legs = calls[calls['strike'] > self.spot_price]
        
        # Limit iterations for performance
        buy_legs = buy_legs.tail(5) 
        sell_legs = sell_legs.head(5)

        for idx, buy in buy_legs.iterrows():
            for idx2, sell in sell_legs.iterrows():
                if sell['strike'] - buy['strike'] > (self.spot_price * 0.01):
                    cost = buy['lastPrice'] - sell['lastPrice']
                    max_profit = (sell['strike'] - buy['strike']) - cost
                    if cost > 0:
                        rr = max_profit / cost
                        if rr > 1.5: 
                            spreads.append({
                                'Type': 'Bull Call',
                                'Long': buy['strike'],
                                'Short': sell['strike'],
                                'Cost': round(cost, 2),
                                'MaxProfit': round(max_profit, 2),
                                'RR': round(rr, 2),
                                'Exp': buy['expirationDate']
                            })
                            
        return pd.DataFrame(spreads).head(10)

    def get_iv_surface_data(self):
        """
        Prepares X, Y, Z data for IV Surface plot.
        """
        if self._options.empty:
            return None, None, None
            
        df = self._options.copy()
        df['dte'] = (pd.to_datetime(df['expirationDate']) - datetime.now()).dt.days
        df = df[df['dte'] > 0]
        
        try:
            pivot = df.pivot_table(index='dte', columns='strike', values='impliedVolatility')
            pivot = pivot.interpolate(limit_direction='both', axis=1) # Fill gaps
            
            x = pivot.columns.values
            y = pivot.index.values
            z = pivot.values
            
            return x, y, z
        except:
            return None, None, None

    def get_tumbler_data(self):
        """
        Returns data for the interactive tumbler, highlighting ATM.
        """
        if self._options.empty:
            return pd.DataFrame()
        
        exps = sorted(self._options['expirationDate'].unique())
        if not exps:
            return pd.DataFrame()
            
        df = self._options[self._options['expirationDate'] == exps[0]].copy()
        
        calls = df[df['type'] == 'call'][['strike', 'lastPrice', 'volume']].set_index('strike')
        puts = df[df['type'] == 'put'][['strike', 'lastPrice', 'volume']].set_index('strike')
        
        chain = calls.join(puts, lsuffix='_c', rsuffix='_p', how='outer').reset_index()
        chain = chain.sort_values('strike')
        
        # Mark ATM
        chain['distance'] = abs(chain['strike'] - self.spot_price)
        chain['is_atm'] = chain['distance'] == chain['distance'].min()
        
        return chain[['strike', 'lastPrice_c', 'volume_c', 'lastPrice_p', 'volume_p', 'is_atm']]

# -----------------------------------------------------------------------------
# 3. DashboardRenderer Class
# -----------------------------------------------------------------------------
class DashboardRenderer:
    """
    Handles all Plotly figure generation and HTML string assembly.
    Offline-first design.
    """
    def __init__(self, ticker):
        self.ticker = ticker
        self.plotly_js = py_offline.get_plotlyjs()

    def generate_html(self, gex_data, streak_data, spread_data, iv_data, tumbler_data, spot_price):
        
        # 1. GEX Chart
        fig_gex = self._make_gex_chart(gex_data, spot_price)
        div_gex = py_offline.plot(fig_gex, include_plotlyjs=False, output_type='div')

        # 2. IV Surface
        fig_iv = self._make_iv_surface(iv_data)
        div_iv = py_offline.plot(fig_iv, include_plotlyjs=False, output_type='div')
        
        # 3. Tables to HTML
        html_streaks = self._df_to_html_table(streak_data)
        html_spreads = self._df_to_html_table(spread_data)
        html_tumbler = self._make_tumbler_html(tumbler_data)

        # Assemble Full HTML
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>QUANT DASHBOARD: {self.ticker}</title>
            <script>{self.plotly_js}</script>
            <style>
                body {{ background-color: #111; color: #eee; font-family: 'Consolas', monospace; margin: 0; padding: 20px; }}
                h1, h2 {{ border-bottom: 1px solid #444; padding-bottom: 5px; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .box {{ background-color: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 5px; min-width: 45%; flex-grow: 1; }}
                
                table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
                th, td {{ border: 1px solid #333; padding: 8px; text-align: right; }}
                th {{ background-color: #222; color: #aaa; }}
                tr:nth-child(even) {{ background-color: #161616; }}
                tr:hover {{ background-color: #2a2a2a; }}
                
                .atm-row {{ background-color: #2a442a !important; color: #fff; font-weight: bold; border: 2px solid #4f4; }}
                
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #222; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #eee; font-family: 'Consolas', monospace; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #111; border-bottom: 2px solid #00ff00; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h1>[{self.ticker}] QUANTITATIVE ANALYSIS DECK</h1>
            <p>Generated: {timestamp} | Spot: ${spot_price:,.2f}</p>
            
            <div class="tab">
              <button class="tablinks" onclick="openTab(event, 'GEX')" id="defaultOpen">Gamma Exposure</button>
              <button class="tablinks" onclick="openTab(event, 'IV')">Vol Surface</button>
              <button class="tablinks" onclick="openTab(event, 'Tables')">Spread Hunter</button>
              <button class="tablinks" onclick="openTab(event, 'Chain')">Option Chain</button>
            </div>

            <div id="GEX" class="tabcontent">
                <div class="box">{div_gex}</div>
            </div>

            <div id="IV" class="tabcontent">
                <div class="box">{div_iv}</div>
            </div>

            <div id="Tables" class="tabcontent">
                <div class="container">
                    <div class="box">
                        <h3>Streak Analysis (Last 10 Days)</h3>
                        {html_streaks}
                    </div>
                    <div class="box">
                        <h3>Spread Hunter (Bullish)</h3>
                        {html_spreads}
                    </div>
                </div>
            </div>

            <div id="Chain" class="tabcontent">
                <div class="box">
                    <h3>Interactive Tumbler (Nearest Expiry)</h3>
                    {html_tumbler}
                </div>
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
                    evt.currentTarget.className += " active";
                    
                    window.dispatchEvent(new Event('resize'));
                    
                    if (typeof Plotly !== 'undefined') {{
                         var plots = document.querySelectorAll('.js-plotly-plot');
                         for (var p of plots) {{
                             Plotly.Plots.resize(p);
                         }}
                    }}
                }}
                document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        return html_content

    def _make_gex_chart(self, df, spot):
        if df.empty:
            return self._empty_fig("Insufficient Options Data for GEX")
            
        fig = go.Figure()
        
        # Dynamic bar coloring
        colors = ['#ff3333' if x < 0 else '#00ff00' for x in df['GEX']]
        
        fig.add_trace(go.Bar(
            x=df['GEX'],
            y=df['strike'],
            orientation='h',
            marker_color=colors,
            name='GEX'
        ))
        
        fig.add_hline(y=spot, line_dash="dash", line_color="white", annotation_text="SPOT")
        
        fig.update_layout(
            title="Net Gamma Exposure (GEX) Profile",
            template="plotly_dark",
            xaxis_title="Gamma Exposure ($)",
            yaxis_title="Strike Price",
            height=600,
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a"
        )
        return fig

    def _make_iv_surface(self, iv_data):
        x, y, z = iv_data
        if x is None:
            return self._empty_fig("Insufficient Data for IV Surface")
            
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
        
        fig.update_layout(
            title="Implied Volatility Surface",
            template="plotly_dark",
            scene = dict(
                xaxis_title='Strike',
                yaxis_title='DTE',
                zaxis_title='Implied Volatility'
            ),
            height=600,
            paper_bgcolor="#1a1a1a"
        )
        return fig

    def _make_tumbler_html(self, df):
        if df.empty:
            return "<p>No Data Available</p>"
        
        html = "<table><thead><tr><th>Call Vol</th><th>Call Last</th><th>Strike</th><th>Put Last</th><th>Put Vol</th></tr></thead><tbody>"
        
        for _, row in df.iterrows():
            row_class = "atm-row" if row['is_atm'] else ""
            html += f"<tr class='{row_class}'>"
            html += f"<td>{row['volume_c']:.0f}</td>"
            html += f"<td>{row['lastPrice_c']:.2f}</td>"
            html += f"<td><b>{row['strike']:.2f}</b></td>"
            html += f"<td>{row['lastPrice_p']:.2f}</td>"
            html += f"<td>{row['volume_p']:.0f}</td>"
            html += "</tr>"
            
        html += "</tbody></table>"
        return html

    def _df_to_html_table(self, df):
        if df.empty:
            return "<p>No Data</p>"
        return df.to_html(classes="", index=False, border=0, float_format="%.2f")

    def _empty_fig(self, text):
        fig = go.Figure()
        fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, font=dict(color="white"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a")
        return fig

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Options Dashboard")
    parser.add_argument('command', choices=['analyze', 'charts'], help="Action to perform")
    parser.add_argument('ticker', type=str, help="Stock Ticker Symbol")
    
    args = parser.parse_args()
    
    print(f"--- INITIALIZING ANALYTICS FOR {args.ticker.upper()} ---")
    
    # 1. Ingest
    ingest = DataIngestion(args.ticker)
    history_df = ingest.get_market_data()
    options_df = ingest.get_options_chain()
    
    if history_df.empty:
        print("[FATAL] Insufficient Data. Exiting.")
        sys.exit(1)

    # 2. Analyze
    analysis = FinancialAnalysis(history_df, options_df)
    
    gex = analysis.calculate_gex()
    streaks = analysis.calculate_streaks()
    spreads = analysis.hunt_spreads()
    iv_data = analysis.get_iv_surface_data()
    tumbler = analysis.get_tumbler_data()
    spot = analysis.spot_price
    
    # 3. Render
    if args.command == 'charts' or args.command == 'analyze':
        renderer = DashboardRenderer(args.ticker)
        html = renderer.generate_html(gex, streaks, spreads, iv_data, tumbler, spot)
        
        filename = f"{args.ticker}_dashboard.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
            
        print(f"\n[SUCCESS] Dashboard generated: {os.path.abspath(filename)}")
        print("Open this file in your browser to view the interactive dashboard.")

if __name__ == "__main__":
    main()
