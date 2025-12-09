import argparse
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as si
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from datetime import datetime, date

# ---------------------------------------------------------
# 1. DATA INGESTION (Disk-First, Staggered)
# ---------------------------------------------------------
class DataIngestion:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df):
        """
        The 'Universal Fixer': Flattens, Normalizes, Coerces.
        """
        # 1. Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # 2. Date Normalization
        # Remove timezones and set to midnight
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration']).dt.tz_localize(None).dt.normalize()

        # 3. Numeric Coercion
        numeric_cols = ['strike', 'lastPrice', 'bid', 'ask', 'openInterest', 'impliedVolatility']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df

    def _download_chain(self, ticker_symbol):
        """
        Downloads option chain with STRICT staggering to avoid rate limits.
        """
        print(f"   [API] Initializing download for {ticker_symbol}...")
        tk = yf.Ticker(ticker_symbol)
        
        try:
            expirations = tk.options
        except Exception as e:
            print(f"   [Error] Could not fetch expirations for {ticker_symbol}: {e}")
            return pd.DataFrame()

        all_opts = []
        
        print(f"   [API] Found {len(expirations)} expiration dates. Starting staggered download...")
        
        for exp_date in expirations:
            # CRITICAL: STAGGER THE DOWNLOADS
            time.sleep(1) 
            
            try:
                # Download specific expiration
                chain = tk.option_chain(exp_date)
                calls = chain.calls
                puts = chain.puts
                
                calls['type'] = 'call'
                puts['type'] = 'put'
                calls['expiration'] = exp_date
                puts['expiration'] = exp_date
                
                all_opts.append(calls)
                all_opts.append(puts)
                print(f"      -> Fetched {exp_date}")
            except Exception as e:
                print(f"      -> Failed {exp_date}: {e}")

        if not all_opts:
            return pd.DataFrame()

        return pd.concat(all_opts, ignore_index=True)

    def get_data(self, ticker):
        file_path = os.path.join(self.output_dir, f"{ticker}_options.csv")

        # 1. Check Disk
        if os.path.exists(file_path):
            print(f"[{ticker}] Found local cache at {file_path}. Loading from disk.")
            df = pd.read_csv(file_path)
            
            # Ensure dates are parsed correctly on reload
            if 'expiration' in df.columns:
                df['expiration'] = pd.to_datetime(df['expiration'])
            return df

        # 2. If Missing, Download
        print(f"[{ticker}] Cache missing. Hitting API.")
        df = self._download_chain(ticker)
        
        if df.empty:
            print(f"[{ticker}] No data found.")
            return df

        # 3. Sanitize
        df = self._sanitize_df(df)

        # 4. Save to Disk
        df.to_csv(file_path, index=False)
        print(f"[{ticker}] Data saved to {file_path}.")

        # 5. Read from Disk (Verify Write)
        return pd.read_csv(file_path)

# ---------------------------------------------------------
# 2. FINANCIAL ANALYSIS (Math & Greeks)
# ---------------------------------------------------------
class FinancialAnalysis:
    def __init__(self, risk_free_rate=0.04, div_yield=0.0):
        self.r = risk_free_rate
        self.q = div_yield

    def _black_scholes_greeks(self, S, K, T, sigma, opt_type):
        """
        Manual implementation of Black-Scholes for Gamma, Vanna, and Delta.
        """
        # Safety for very small T or sigma
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0, 0.0

        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # PDF and CDF
        pdf_d1 = si.norm.pdf(d1)
        cdf_d1 = si.norm.cdf(d1)
        
        # Gamma (Same for Call and Put)
        gamma = (np.exp(-self.q * T) * pdf_d1) / (S * sigma * np.sqrt(T))

        # Vanna (Sensitivity of Delta to Volatility)
        # Formula: -e^(-qT) * pdf(d1) * (d2 / sigma)
        vanna = -np.exp(-self.q * T) * pdf_d1 * (d2 / sigma)

        # Delta
        if opt_type == 'call':
            delta = np.exp(-self.q * T) * cdf_d1
        else:
            delta = -np.exp(-self.q * T) * si.norm.cdf(-d1)

        return gamma, vanna, delta

    def compute_metrics(self, df):
        if df.empty:
            return df

        # Ensure expiration is datetime
        df['expiration'] = pd.to_datetime(df['expiration'])
        today = pd.Timestamp.now().normalize()
        
        # Calculate Time to Expiration (T) in years
        df['T'] = (df['expiration'] - today).dt.days / 365.0
        # Filter expired options
        df = df[df['T'] > 0].copy()

        # -------------------------------------------------------
        # Spot Price Estimation (Method: "Option D")
        # -------------------------------------------------------
        # We process by Expiration Date
        enriched_groups = []
        
        for exp_date, group in df.groupby('expiration'):
            # Separate Calls and Puts
            calls = group[group['type'] == 'call'].set_index('strike')
            puts = group[group['type'] == 'put'].set_index('strike')
            
            # Find common strikes
            common_strikes = calls.index.intersection(puts.index)
            
            if common_strikes.empty:
                continue

            # 2. Mid Price Calculation
            c_mid = (calls.loc[common_strikes]['bid'] + calls.loc[common_strikes]['ask']) / 2
            p_mid = (puts.loc[common_strikes]['bid'] + puts.loc[common_strikes]['ask']) / 2
            
            # 3. Find Parity Strike (Min Abs Diff)
            # Concept: At parity, C - P = S*exp(-qT) - K*exp(-rT)
            # We use this to get an INITIAL spot estimate to calculate Deltas
            diff = (c_mid - p_mid).abs()
            parity_strike = diff.idxmin()
            
            # Initial Spot Estimate from Put-Call Parity at Parity Strike
            # S = (C - P + K * e^-rT) / e^-qT
            c_val = c_mid.loc[parity_strike]
            p_val = p_mid.loc[parity_strike]
            K_val = parity_strike
            T_val = group['T'].iloc[0]
            
            S_initial = (c_val - p_val + K_val * np.exp(-self.r * T_val)) / np.exp(-self.q * T_val)

            # 4. Calculate Deltas using S_initial
            # We need to map this back to the group dataframe
            # We'll calculate temporary deltas just to find the 0.5 candidates
            
            # 5. Identify 0.5 Delta Candidates
            # We iterate strictly through the group rows
            strikes_05 = []
            
            for idx, row in group.iterrows():
                sigma = row['impliedVolatility']
                K = row['strike']
                otype = row['type']
                
                _, _, delta = self._black_scholes_greeks(S_initial, K, T_val, sigma, otype)
                
                # Check if this is a candidate (Call near 0.5, Put near -0.5)
                if otype == 'call' and abs(delta - 0.5) < 0.1: # Threshold for "near"
                    strikes_05.append(K)
                elif otype == 'put' and abs(delta + 0.5) < 0.1:
                    strikes_05.append(K)
            
            # 6. Final Spot Estimate
            if strikes_05:
                S_final = np.mean(strikes_05)
            else:
                S_final = S_initial # Fallback
            
            # Apply S_final to the group and calculate real Greeks
            group = group.copy()
            group['estimated_spot'] = S_final
            
            greeks = group.apply(
                lambda row: self._black_scholes_greeks(
                    S_final, 
                    row['strike'], 
                    row['T'], 
                    row['impliedVolatility'], 
                    row['type']
                ), axis=1
            )
            
            group['Gamma'] = [x[0] for x in greeks]
            group['Vanna'] = [x[1] for x in greeks]
            # group['Delta'] = [x[2] for x in greeks] # Optional, not requested for display

            # Exposure Calculations
            # Gamma Exposure: Gamma * OI * 100 * Spot
            group['GammaExposure'] = group['Gamma'] * group['openInterest'] * 100 * S_final
            
            # Vanna Exposure: Vanna * OI * 100
            group['VannaExposure'] = group['Vanna'] * group['openInterest'] * 100

            enriched_groups.append(group)
            
        if not enriched_groups:
            return pd.DataFrame()
            
        return pd.concat(enriched_groups)

# ---------------------------------------------------------
# 3. DASHBOARD RENDERER (Plotly HTML)
# ---------------------------------------------------------
class DashboardRenderer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_dashboard(self, data_dict):
        # RESIZE FIX SCRIPT
        resize_script = """
        <script>
            window.addEventListener('load', function() {
                window.dispatchEvent(new Event('resize'));
            });
            setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 500);
        </script>
        """

        html_content = [f"<html><head><title>Hedge Fund Grade Greeks</title>{resize_script}</head><body>"]
        html_content.append(f"<h1>Market Gamma & Vanna Dashboard</h1><p>Generated: {datetime.now()}</p>")

        # Embed Plotly JS Offline
        html_content.append(f'<script type="text/javascript">{pyo.get_plotlyjs()}</script>')

        for ticker, df in data_dict.items():
            if df.empty or 'GammaExposure' not in df.columns:
                html_content.append(f"<h2>{ticker}: No Data Available</h2><hr>")
                continue
                
            html_content.append(f"<h2>Analysis: {ticker}</h2>")
            html_content.append(f"<p>Estimated Spot (Near-term): {df['estimated_spot'].iloc[0]:.2f}</p>")
            
            # Prepare Grid for Surfaces
            # We aggregate by Strike and Expiration for the Surface
            # Summing exposures if multiple options map to same grid point (rare but possible with data noise)
            pivot_gamma = df.pivot_table(index='expiration', columns='strike', values='GammaExposure', aggfunc='sum').fillna(0)
            pivot_vanna = df.pivot_table(index='expiration', columns='strike', values='VannaExposure', aggfunc='sum').fillna(0)
            
            # Extract axes
            x_strikes = pivot_gamma.columns
            y_dates = pivot_gamma.index
            z_gamma = pivot_gamma.values
            z_vanna = pivot_vanna.values

            # --- Chart 1: Gamma Surface ---
            fig_gamma_surf = go.Figure(data=[go.Surface(z=z_gamma, x=x_strikes, y=y_dates, colorscale='Viridis')])
            fig_gamma_surf.update_layout(title=f'{ticker} - 3D Gamma Exposure Surface', scene=dict(xaxis_title='Strike', yaxis_title='Expiration', zaxis_title='Gamma Exposure'))
            
            # --- Chart 2: Gamma Heatmap ---
            fig_gamma_heat = go.Figure(data=go.Heatmap(z=z_gamma, x=x_strikes, y=y_dates, colorscale='Viridis'))
            fig_gamma_heat.update_layout(title=f'{ticker} - Gamma Exposure Heatmap')

            # --- Chart 3: Vanna Surface ---
            fig_vanna_surf = go.Figure(data=[go.Surface(z=z_vanna, x=x_strikes, y=y_dates, colorscale='RdBu')])
            fig_vanna_surf.update_layout(title=f'{ticker} - 3D Vanna Exposure Surface', scene=dict(xaxis_title='Strike', yaxis_title='Expiration', zaxis_title='Vanna Exposure'))

            # --- Chart 4: Vanna Heatmap ---
            fig_vanna_heat = go.Figure(data=go.Heatmap(z=z_vanna, x=x_strikes, y=y_dates, colorscale='RdBu'))
            fig_vanna_heat.update_layout(title=f'{ticker} - Vanna Exposure Heatmap')

            # --- Chart 5: Term Structure (ATM Slice) ---
            # Find nearest strike to current estimated spot for the ATM slice
            current_spot = df['estimated_spot'].mean() # Average across term for simplicity in slice finding
            atm_strike = min(x_strikes, key=lambda x:abs(x-current_spot))
            
            term_slice = pivot_gamma[atm_strike]
            
            fig_term = go.Figure(data=go.Scatter(x=term_slice.index, y=term_slice.values, mode='lines+markers'))
            fig_term.update_layout(title=f'{ticker} - Term Structure (ATM Strike: {atm_strike})')

            # Render Charts to HTML div
            html_content.append(pyo.plot(fig_gamma_surf, include_plotlyjs=False, output_type='div'))
            html_content.append(pyo.plot(fig_gamma_heat, include_plotlyjs=False, output_type='div'))
            html_content.append(pyo.plot(fig_vanna_surf, include_plotlyjs=False, output_type='div'))
            html_content.append(pyo.plot(fig_vanna_heat, include_plotlyjs=False, output_type='div'))
            html_content.append(pyo.plot(fig_term, include_plotlyjs=False, output_type='div'))
            
            html_content.append("<hr>")

        html_content.append("</body></html>")
        
        output_path = os.path.join(self.output_dir, "market_dashboard.html")
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("".join(html_content))
        
        return output_path

# ---------------------------------------------------------
# 4. EXECUTION FLOW
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Options Analysis")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help="List of tickers")
    parser.add_argument('--output-dir', default='./market_data', help="Data storage directory")
    parser.add_argument('--lookback', type=int, default=1, help="Years of history (Context)")
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help="Risk Free Rate")
    parser.add_argument('--div-yield', type=float, default=0.0, help="Dividend Yield")

    args = parser.parse_args()

    print("--- Hedge Fund Options Analytics Initialized ---")
    
    # 1. Ingestion
    ingestion = DataIngestion(args.output_dir)
    raw_data = {}
    for t in args.tickers:
        raw_data[t] = ingestion.get_data(t)

    # 2. Analysis
    analyzer = FinancialAnalysis(args.risk_free_rate, args.div_yield)
    processed_data = {}
    for t, df in raw_data.items():
        print(f"[{t}] Running Financial Analysis (Greeks & Exposures)...")
        processed_data[t] = analyzer.compute_metrics(df)

    # 3. Rendering
    renderer = DashboardRenderer(args.output_dir)
    report_path = renderer.generate_dashboard(processed_data)

    print(f"\nSUCCESS: Dashboard generated at: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()
