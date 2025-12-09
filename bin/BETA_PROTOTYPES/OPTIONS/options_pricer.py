# SCRIPTNAME: ok.05.options_pricer.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import time
import argparse
import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.offline as py_offline
from datetime import datetime, timedelta

# ==========================================
# 1. DATA INGESTION (Disk-First Architecture)
# ==========================================

class DataIngestion:
    """
    Handles all I/O, downloading, caching, and sanitization.
    Strictly no financial math allowed here.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df):
        """
        The Universal Fixer: Normalizes yfinance idiosyncrasies.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Swap Levels if 'Close' is in Level 1 (common yf bug)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if the column structure is inverted
            if df.columns.nlevels > 1:
                # Heuristic: if the top level has more unique values than the second
                # or if specific known headers are in level 1
                if 'Close' in df.columns.get_level_values(1):
                    df = df.swaplevel(0, 1, axis=1)
            
            # 2. Flatten MultiIndex columns
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
            # Clean up: If flattening resulted in "Close_SPY", map back to "Close" if it's the only ticker
            # For this script, we usually fetch single tickers at a time, so we rename standard columns
            clean_cols = {}
            for col in df.columns:
                if col.startswith('Close'): clean_cols[col] = 'Close'
                elif col.startswith('Open'): clean_cols[col] = 'Open'
                elif col.startswith('High'): clean_cols[col] = 'High'
                elif col.startswith('Low'): clean_cols[col] = 'Low'
                elif col.startswith('Volume'): clean_cols[col] = 'Volume'
                elif col.startswith('Adj Close'): clean_cols[col] = 'Adj Close'
            df = df.rename(columns=clean_cols)

        # 3. Strict Index cleaning
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 4. Coerce numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def fetch_data(self, ticker, lookback_years=1):
        """
        Disk-First logic to fetch spot data.
        """
        file_path = os.path.join(self.output_dir, f"{ticker}_spot.csv")
        
        # 1. Check Disk
        if os.path.exists(file_path):
            print(f"[DataIngestion] Loading {ticker} from disk...")
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                print(f"[DataIngestion] Error reading CSV: {e}. Re-downloading.")

        # 2. Download
        print(f"[DataIngestion] Downloading {ticker} via yfinance...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years*365)
        
        # Enforce rate limiting specifically for downloads
        time.sleep(1)

        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, group_by='column')
            
            # 3. Sanitize
            df = self._sanitize_df(df)
            
            # 4. Save
            df.to_csv(file_path)
            
            # 5. Reload to ensure consistency
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
            
        except Exception as e:
            print(f"[Error] Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def fetch_market_volatility(self):
        """
        Fetches VIX (Near term) and VIX3M (Medium term).
        """
        indices = {'^VIX': 'VIX', '^VIX3M': 'VIX3M'}
        results = {}
        
        for ticker, name in indices.items():
            # Rate limiting is now handled inside fetch_data's download block
            df = self.fetch_data(ticker)
            if not df.empty:
                # Get most recent Close
                results[name] = df['Close'].iloc[-1]
            else:
                results[name] = 0.0 # Shadow backfill
                
        return results

    def get_option_chain_details(self, ticker_symbol, expiry=None, strike=None, opt_type='call'):
        """
        Fetches option chain to get current Market Price and Implied Volatility.
        If strike/expiry not provided, finds ATM and Next Monthly.
        """
        tk = yf.Ticker(ticker_symbol)
        
        # Get Expiry
        if not expiry:
            try:
                # Default to 2nd expiry to ensure some time value (simplified 'next month' logic)
                expiry = tk.options[1] if len(tk.options) > 1 else tk.options[0]
            except IndexError:
                print("[Error] No options found for ticker.")
                return None
        
        print(f"[DataIngestion] Fetching chain for {expiry}...")
        try:
            chain = tk.option_chain(expiry)
            data = chain.calls if opt_type.lower() == 'call' else chain.puts
        except Exception:
            # Fallback if API fails
            return None

        # Get Underlying Price for ATM calculation
        # We need a fresh live price for ATM logic, strictly speaking, 
        # but we use the last close from fetch_data for consistency in the model.
        # However, for selecting the strike from the chain, we rely on the chain data.
        
        # If strike not provided, find closest to current spot (ATM)
        if not strike:
            # We assume the average of strikes is roughly around spot or use the spot from history
            # Better: use the middle of the chain dataframe
            # Or fetch live spot:
            spot_df = self.fetch_data(ticker_symbol)
            current_spot = spot_df['Close'].iloc[-1]
            
            # Find closest strike
            data['abs_diff'] = abs(data['strike'] - current_spot)
            row = data.sort_values('abs_diff').iloc[0]
        else:
            # Find specific strike
            row = data[data['strike'] == float(strike)]
            if row.empty:
                print(f"[Error] Strike {strike} not found in chain.")
                return None
            row = row.iloc[0]

        return {
            'strike': row['strike'],
            'lastPrice': row['lastPrice'],
            'impliedVolatility': row['impliedVolatility'],
            'expiry': expiry,
            'contractSymbol': row['contractSymbol']
        }


# ==========================================
# 2. FINANCIAL ANALYSIS (Math & Logic)
# ==========================================

class FinancialAnalysis:
    """
    Handles all math, Black-Scholes logic, and scenario generation.
    No I/O allowed here.
    """
    def __init__(self, spot, strike, time_to_expiry, risk_free_rate, iv, opt_type='call'):
        self.S = float(spot)
        self.K = float(strike)
        self.T = float(time_to_expiry) # in years
        self.r = float(risk_free_rate)
        self.sigma = float(iv)
        self.opt_type = opt_type.lower()

    def _d1_d2(self, S, sigma):
        # Prevent div by zero
        if sigma <= 0 or self.T <= 0:
            return 0, 0
        d1 = (math.log(S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * math.sqrt(self.T))
        d2 = d1 - sigma * math.sqrt(self.T)
        return d1, d2

    def price(self, S=None, sigma=None):
        S = S if S is not None else self.S
        sigma = sigma if sigma is not None else self.sigma
        
        d1, d2 = self._d1_d2(S, sigma)
        
        if self.opt_type == 'call':
            p = S * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            p = self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return p

    def calculate_greeks(self):
        d1, d2 = self._d1_d2(self.S, self.sigma)
        
        # Delta
        if self.opt_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            
        # Gamma (Same for Call and Put)
        gamma = norm.pdf(d1) / (self.S * self.sigma * math.sqrt(self.T))
        
        # Vega (Same for Call and Put) - Usually shown as 1% change
        vega = self.S * norm.pdf(d1) * math.sqrt(self.T) * 0.01
        
        # Theta (Annual)
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T))
        if self.opt_type == 'call':
            term2 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
            theta = term1 - term2
        else:
            term2 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)
            theta = term1 + term2
            
        # Convert Theta to daily for display
        theta_daily = theta / 365.0

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Vega': vega,
            'Theta': theta_daily,
            'Price': self.price()
        }

    def generate_shock_matrix(self):
        """
        3x3 Pricing Matrix.
        Spot Shocks: 99.5%, 100%, 100.5%
        IV Shocks: -1%, 0%, +1% (Absolute)
        """
        spot_shocks = [0.995, 1.0, 1.005]
        iv_shocks = [-0.01, 0.0, 0.01] # Absolute shocks
        
        matrix = []
        
        # X Axis: IV, Y Axis: Spot
        iv_axis = []
        spot_axis = []
        z_values = []

        for s_mult in spot_shocks:
            row_z = []
            test_spot = self.S * s_mult
            spot_axis.append(test_spot)
            
            for vol_add in iv_shocks:
                test_vol = max(0.001, self.sigma + vol_add)
                if s_mult == 0.995: # Only populate X axis once
                    iv_axis.append(test_vol)
                
                price = self.price(S=test_spot, sigma=test_vol)
                row_z.append(price)
            
            z_values.append(row_z)
            
        return {
            'x_iv': iv_axis,
            'y_spot': spot_axis,
            'z_price': z_values
        }

    def get_gtc_bands(self, matrix_data):
        """
        Derive Buy/Sell bands from Min/Max of the shock matrix.
        """
        # Flatten z_values
        all_prices = [p for row in matrix_data['z_price'] for p in row]
        return min(all_prices), max(all_prices)


# ==========================================
# 3. DASHBOARD RENDERER (Visualization)
# ==========================================

class DashboardRenderer:
    """
    Handles HTML generation and Plotly code.
    """
    def generate_html(self, analysis_data, filename="dashboard.html"):
        """
        Constructs a standalone HTML file with embedded JS.
        """
        # Unpack data
        greeks = analysis_data['greeks']
        matrix = analysis_data['matrix']
        gtc_min, gtc_max = analysis_data['gtc_bands']
        vol_struct = analysis_data['vol_structure']
        ticker_info = analysis_data['info']

        # 1. Indicator & Table
        fig_summary = go.Figure()
        fig_summary.add_trace(go.Indicator(
            mode = "number",
            value = greeks['Price'],
            title = {"text": f"Theoretical Price ({ticker_info['type'].upper()})"},
            domain = {'x': [0, 0.5], 'y': [0.5, 1]}
        ))
        
        fig_summary.add_trace(go.Table(
            header=dict(values=['Metric', 'Value'], fill_color='paleturquoise', align='left'),
            cells=dict(values=[
                ['Delta', 'Gamma', 'Vega (1%)', 'Theta (Daily)', 'Strike', 'Expiry', 'Underlying', 'Input IV'],
                [f"{greeks['Delta']:.3f}", f"{greeks['Gamma']:.4f}", f"{greeks['Vega']:.3f}", 
                 f"{greeks['Theta']:.3f}", f"{ticker_info['strike']}", ticker_info['expiry'], 
                 f"{ticker_info['spot']:.2f}", f"{ticker_info['iv']:.2%}"]
            ], fill_color='lavender', align='left'),
            domain = {'x': [0.5, 1], 'y': [0, 1]}
        ))
        fig_summary.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))

        # 2. Shock Matrix (Heatmap)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=matrix['z_price'],
            x=[f"{v:.1%}" for v in matrix['x_iv']],
            y=[f"{s:.2f}" for s in matrix['y_spot']],
            colorscale='Viridis',
            texttemplate="%{z:.2f}",
            textfont={"size": 12}
        ))
        fig_heatmap.update_layout(
            title="Shock Matrix: Price Sensitivity",
            xaxis_title="Implied Volatility",
            yaxis_title="Spot Price",
            height=500
        )

        # 3. Shock Surface (3D)
        fig_surface = go.Figure(data=[go.Surface(
            z=matrix['z_price'], 
            x=matrix['x_iv'], 
            y=matrix['y_spot'],
            colorscale='Viridis'
        )])
        fig_surface.update_layout(
            title="3D Pricing Surface",
            scene = dict(
                xaxis_title='IV',
                yaxis_title='Spot',
                zaxis_title='Price'
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # 4. GTC Bands (Gauge)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = greeks['Price'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "GTC Limit Bands (Shock Range)"},
            gauge = {
                'axis': {'range': [gtc_min*0.9, gtc_max*1.1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [gtc_min, gtc_max], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': greeks['Price']
                }
            }
        ))
        fig_gauge.update_layout(height=400)

        # 5. Vol Structure (Bar)
        vol_x = ['Option IV', 'VIX (Near)', 'VIX3M (Med)']
        vol_y = [ticker_info['iv']*100, vol_struct['VIX'], vol_struct['VIX3M']]
        fig_vol = go.Figure(data=[go.Bar(x=vol_x, y=vol_y, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])])
        fig_vol.update_layout(title="Volatility Term Structure (%)", height=400)


        # --- HTML ASSEMBLY ---
        
        # Get Plotly JS lib code
        plotly_js = py_offline.get_plotlyjs()
        
        # Get Div strings
        div_summary = py_offline.plot(fig_summary, output_type='div', include_plotlyjs=False)
        div_heatmap = py_offline.plot(fig_heatmap, output_type='div', include_plotlyjs=False)
        div_surface = py_offline.plot(fig_surface, output_type='div', include_plotlyjs=False)
        div_gauge = py_offline.plot(fig_gauge, output_type='div', include_plotlyjs=False)
        div_vol = py_offline.plot(fig_vol, output_type='div', include_plotlyjs=False)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Option Analysis: {ticker_info['ticker']}</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }}
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #ccc; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; background-color: white; }}
                h1 {{ color: #333; }}
                .card {{ background: white; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 5px; }}
            </style>
        </head>
        <body>

        <h1>Option Analysis: {ticker_info['ticker']} {ticker_info['strike']} {ticker_info['type'].upper()}</h1>
        <p>Expiry: {ticker_info['expiry']}</p>

        <div class="tab">
          <button class="tablinks" onclick="openTab(event, 'Summary')" id="defaultOpen">Baseline & Greeks</button>
          <button class="tablinks" onclick="openTab(event, 'ShockMatrix')">Shock Matrix (2D)</button>
          <button class="tablinks" onclick="openTab(event, 'Surface')">Shock Surface (3D)</button>
          <button class="tablinks" onclick="openTab(event, 'RiskLimits')">GTC Bands</button>
          <button class="tablinks" onclick="openTab(event, 'VolStruct')">Vol Structure</button>
        </div>

        <div id="Summary" class="tabcontent">
            <div class="card">{div_summary}</div>
        </div>

        <div id="ShockMatrix" class="tabcontent">
            <div class="card">{div_heatmap}</div>
        </div>

        <div id="Surface" class="tabcontent">
            <div class="card">{div_surface}</div>
        </div>
        
        <div id="RiskLimits" class="tabcontent">
            <div class="card">{div_gauge}</div>
        </div>

        <div id="VolStruct" class="tabcontent">
            <div class="card">{div_vol}</div>
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
            
            // CRITICAL: Force Plotly resize on tab switch
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
        
        print(f"[Dashboard] Saved to {os.path.abspath(filename)}")


# ==========================================
# 4. MAIN ORCHESTRATION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Quantitative Option Pricing Engine")
    parser.add_argument('--tickers', nargs='+', default=['SPY'], help='Underlying symbol(s)')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Data storage directory')
    parser.add_argument('--lookback', type=int, default=1, help='Years of historical data')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate (decimal)')
    parser.add_argument('--strike', type=float, help='Strike price (optional)')
    parser.add_argument('--expiry', type=str, help='Expiry YYYY-MM-DD (optional)')
    parser.add_argument('--option-type', type=str, default='call', choices=['call', 'put'], help='Option type')

    args = parser.parse_args()

    # Instantiate Data Pipeline
    data_engine = DataIngestion(args.output_dir)

    for ticker in args.tickers:
        print(f"\n--- Processing {ticker} ---")
        
        # 1. Fetch Underlying Spot (Ensures caching)
        spot_df = data_engine.fetch_data(ticker, args.lookback)
        if spot_df.empty:
            print(f"Skipping {ticker} due to data error.")
            continue
            
        current_spot = spot_df['Close'].iloc[-1]
        
        # 2. Fetch Market Vols
        vol_data = data_engine.fetch_market_volatility()
        
        # 3. Get Chain Details (Price/IV)
        opt_details = data_engine.get_option_chain_details(
            ticker, 
            expiry=args.expiry, 
            strike=args.strike, 
            opt_type=args.option_type
        )
        
        if not opt_details:
            print(f"Could not retrieve option chain for {ticker}.")
            continue
            
        # Calculate time to expiry in years
        expiry_date = pd.to_datetime(opt_details['expiry'])
        days_to_expiry = (expiry_date - datetime.now()).days
        # Prevent zero-time error
        time_to_expiry = max(days_to_expiry / 365.0, 0.001)

        print(f"Contract: {opt_details['contractSymbol']} | Spot: {current_spot:.2f} | Strike: {opt_details['strike']} | IV: {opt_details['impliedVolatility']:.2%}")

        # 4. Instantiate Financial Analysis
        fin_engine = FinancialAnalysis(
            spot=current_spot,
            strike=opt_details['strike'],
            time_to_expiry=time_to_expiry,
            risk_free_rate=args.risk_free_rate,
            iv=opt_details['impliedVolatility'],
            opt_type=args.option_type
        )
        
        # 5. Run Calcs
        greeks = fin_engine.calculate_greeks()
        shock_matrix = fin_engine.generate_shock_matrix()
        gtc_min, gtc_max = fin_engine.get_gtc_bands(shock_matrix)

        # 6. Render Dashboard
        dashboard = DashboardRenderer()
        
        analysis_packet = {
            'info': {
                'ticker': ticker,
                'spot': current_spot,
                'strike': opt_details['strike'],
                'expiry': opt_details['expiry'],
                'type': args.option_type,
                'iv': opt_details['impliedVolatility']
            },
            'greeks': greeks,
            'matrix': shock_matrix,
            'gtc_bands': (gtc_min, gtc_max),
            'vol_structure': vol_data
        }
        
        filename = f"{ticker}_{args.option_type}_{opt_details['strike']}_dashboard.html"
        dashboard.generate_html(analysis_packet, filename)

if __name__ == "__main__":
    main()
