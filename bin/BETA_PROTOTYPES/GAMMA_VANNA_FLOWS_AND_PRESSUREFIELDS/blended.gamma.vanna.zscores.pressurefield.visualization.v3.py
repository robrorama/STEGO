# SCRIPTNAME: ok.03.blended.gamma.vanna.zscores.pressurefield.visualization.v3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import time
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.offline as py_offline

# Suppress pandas chained assignment warnings for cleaner output
pd.options.mode.chained_assignment = None

# -----------------------------------------------------------------------------
# CLASS A: DataIngestion (IO & Persistence)
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Handles reliable data fetching, disk caching, and sanitization.
    Enforces a disk-first retrieval policy to minimize API dependency.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"[System] Created storage directory: {self.output_dir}")

    def get_data(self, ticker):
        """
        Orchestrates the fetch-or-load logic.
        1. Check Disk.
        2. If Missing -> Download -> Sanitize -> Save.
        3. Load from Disk.
        """
        file_path = os.path.join(self.output_dir, f"{ticker}_options.csv")

        if os.path.exists(file_path):
            print(f"[IO] Loading {ticker} from disk...")
            return pd.read_csv(file_path)
        
        print(f"[IO] {ticker} not found on disk. Downloading from source...")
        try:
            raw_df, spot_price = self._download_data(ticker)
            if raw_df.empty:
                print(f"[Error] No option data found for {ticker}")
                return None
            
            # Sanitize
            clean_df = self._sanitize_df(raw_df)
            
            # Add spot price as a column for context persistence
            clean_df['underlying_price'] = spot_price
            
            # Save
            clean_df.to_csv(file_path, index=False)
            print(f"[IO] Saved sanitized data to {file_path}")
            
            # Load back to ensure consistency
            return pd.read_csv(file_path)
            
        except Exception as e:
            print(f"[Error] Failed to ingest {ticker}: {e}")
            return None

    def _download_data(self, ticker_symbol):
        """
        Downloads all available option chains for the ticker.
        """
        tk = yf.Ticker(ticker_symbol)
        
        # Get Spot Price (needed for Greeks)
        try:
            hist = tk.history(period="1d")
            if hist.empty:
                raise ValueError("Could not retrieve spot price")
            spot = hist['Close'].iloc[-1]
        except Exception:
            print(f"[Warning] Could not fetch live price, using generic fallback.")
            spot = 100.0

        try:
            expirations = tk.options
        except Exception:
            return pd.DataFrame(), spot

        all_opts = []
        print(f"    - Fetching {len(expirations)} expiration dates...")
        
        for e in expirations:
            try:
                # Force specific download pattern
                chain = tk.option_chain(e)
                calls = chain.calls
                calls['type'] = 'call'
                calls['expiration'] = e
                
                puts = chain.puts
                puts['type'] = 'put'
                puts['expiration'] = e
                
                all_opts.append(calls)
                all_opts.append(puts)
            except Exception as err:
                continue

        if not all_opts:
            return pd.DataFrame(), spot

        return pd.concat(all_opts, axis=0), spot

    def _sanitize_df(self, df):
        """
        Universal Fixer: Flattens MultiIndexes, handles types, fixes N/A.
        """
        # 1. MultiIndex Handling
        if isinstance(df.columns, pd.MultiIndex):
            # 2. Swap Levels if 'Close' is in level 1
            if df.columns.nlevels > 1:
                # Simple heuristic: if the first level looks like tickers, swap
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # 3. Strict Typing & Cleaning
        # Normalize column names to lowercase for consistency
        df.columns = [c.lower() for c in df.columns]

        # Map typical yfinance columns to internal standard
        col_map = {
            'strike': 'strike',
            'lastprice': 'lastPrice',
            'impliedvolatility': 'impliedVolatility',
            'openinterest': 'openInterest',
            'expiration': 'expiration',
            'type': 'type'
        }
        
        # Rename if columns exist
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Coerce numerics
        numeric_cols = ['strike', 'impliedVolatility', 'openInterest']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Drop rows where IV is effectively zero (bad data usually)
        if 'impliedVolatility' in df.columns:
            df = df[df['impliedVolatility'] > 0.001]

        return df

# -----------------------------------------------------------------------------
# CLASS B: FinancialAnalysis (The Core Logic)
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Performs Black-Scholes Greek estimation and Pressure Surface calculation.
    Pure mathematical transformation layer.
    """
    def __init__(self, risk_free_rate=0.04, w_gamma=0.6, w_vanna=0.4):
        self.rf = risk_free_rate
        self.w_gamma = w_gamma
        self.w_vanna = w_vanna

    def _black_scholes_greeks(self, S, K, T, r, sigma, opt_type):
        """
        Vectorized Black-Scholes Greek calculator.
        Returns Tuple: (Gamma, Vanna)
        """
        # Avoid division by zero
        T = np.maximum(T, 1/365.0) 
        sigma = np.maximum(sigma, 0.01)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        pdf_d1 = norm.pdf(d1)
        
        # Gamma: N'(d1) / (S * sigma * sqrt(T))
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        
        # Vanna: -N'(d1) * d2 / sigma
        # Sensitivity of Delta to Volatility (or Vega to Spot)
        vanna = -pdf_d1 * d2 / sigma

        return gamma, vanna

    def process_ticker(self, df):
        """
        Main logic pipeline:
        1. Calculate Greeks (Gamma/Vanna).
        2. Calculate Exposure (Greek * OI * 100).
        3. Normalize (Z-Score).
        4. Construct Surface & Gradients.
        """
        if df is None or df.empty:
            return None, None

        # Prepare T (Time to Expiry)
        today = datetime.datetime.now()
        df['expiration_dt'] = pd.to_datetime(df['expiration'])
        df['days_to_expiry'] = (df['expiration_dt'] - today).dt.days
        # Filter expired
        df = df[df['days_to_expiry'] > 0]
        df['T'] = df['days_to_expiry'] / 365.0

        spot = df['underlying_price'].iloc[0]

        # 1. Compute Greeks
        # Helper wrapper for vectorize or apply. Using apply for clarity here, 
        # though vectorization is faster for massive datasets.
        # We need to distinguish Calls vs Puts? 
        # Gamma is generally positive for long calls and long puts.
        # Vanna sign flips based on type/moneyness. 
        # For this model, we stick to the provided formulas.
        
        greeks = df.apply(
            lambda row: self._black_scholes_greeks(
                spot, 
                row['strike'], 
                row['T'], 
                self.rf, 
                row['impliedVolatility'], 
                row['type']
            ), axis=1
        )
        
        df['gamma'] = [x[0] for x in greeks]
        df['vanna'] = [x[1] for x in greeks]

        # 2. Compute Exposure
        # exp = greek * OI * 100
        df['gamma_exp'] = df['gamma'] * df['openInterest'] * 100
        df['vanna_exp'] = df['vanna'] * df['openInterest'] * 100

        # Aggregation: Sum exposures across Calls and Puts for the same Strike/Expiry
        grouped = df.groupby(['expiration_dt', 'strike'])[['gamma_exp', 'vanna_exp']].sum().reset_index()

        # 3. Z-Score Normalization (Group by Expiry)
        def z_score(x):
            return (x - x.mean()) / (x.std() + 1e-6)

        grouped['gamma_z'] = grouped.groupby('expiration_dt')['gamma_exp'].transform(z_score)
        grouped['vanna_z'] = grouped.groupby('expiration_dt')['vanna_exp'].transform(z_score)

        # 4. Surface Construction (Pivot)
        # We pivot on Z-Scores to create the grid
        pivot_gamma = grouped.pivot(index='expiration_dt', columns='strike', values='gamma_z')
        pivot_vanna = grouped.pivot(index='expiration_dt', columns='strike', values='vanna_z')

        # Interpolation (Critical for surfaces)
        # Interpolate along strikes (axis 1), then fill remaining
        pivot_gamma = pivot_gamma.interpolate(method='linear', axis=1, limit_direction='both').fillna(0)
        pivot_vanna = pivot_vanna.interpolate(method='linear', axis=1, limit_direction='both').fillna(0)
        
        # Sort index to ensure time continuity for gradient
        pivot_gamma.sort_index(inplace=True)
        pivot_vanna.sort_index(inplace=True)

        # 5. Gradient Norm (Pressure)
        def calc_pressure(surface_df):
            values = surface_df.values
            # gradients returns [gradient_axis_0 (time), gradient_axis_1 (strike)]
            grads = np.gradient(values)
            # Magnitude: sqrt(dy^2 + dx^2)
            magnitude = np.sqrt(grads[0]**2 + grads[1]**2)
            return pd.DataFrame(magnitude, index=surface_df.index, columns=surface_df.columns)

        press_gamma = calc_pressure(pivot_gamma)
        press_vanna = calc_pressure(pivot_vanna)

        # 6. Blending
        total_pressure = (self.w_gamma * press_gamma) + (self.w_vanna * press_vanna)

        return total_pressure, grouped

# -----------------------------------------------------------------------------
# CLASS C: DashboardRenderer (Visualization)
# -----------------------------------------------------------------------------
class DashboardRenderer:
    """
    Generates high-fidelity, standalone HTML reports using Offline Plotly.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_report(self, ticker, pressure_surface, raw_data):
        if pressure_surface is None:
            return

        # Prepare data for plotting
        y_dates = pressure_surface.index.astype(str)
        x_strikes = pressure_surface.columns
        z_values = pressure_surface.values

        # 1. Main Heatmap
        heatmap = go.Heatmap(
            z=z_values,
            x=x_strikes,
            y=y_dates,
            colorscale='Jet',
            colorbar=dict(title='Pressure Intensity'),
            hovertemplate='Expiry: %{y}<br>Strike: %{x}<br>Pressure: %{z:.2f}<extra></extra>'
        )

        layout = go.Layout(
            title=f"Market Pressure Field: {ticker} (Gamma/Vanna Gradient)",
            xaxis=dict(title='Strike Price', tickmode='auto'),
            yaxis=dict(title='Expiration Date', type='category', automargin=True),
            template='plotly_dark',
            height=800
        )

        fig_surface = go.Figure(data=[heatmap], layout=layout)

        # 2. Slice Analysis (Next 3 Expiries)
        fig_slices = go.Figure()
        # Grab first 3 rows if available
        dates_to_plot = pressure_surface.index[:4]
        
        for dt in dates_to_plot:
            series = pressure_surface.loc[dt]
            fig_slices.add_trace(go.Scatter(
                x=series.index, 
                y=series.values, 
                mode='lines', 
                name=str(dt.date())
            ))
            
        fig_slices.update_layout(
            title=f"Pressure Slices (Near-Term Expiries): {ticker}",
            xaxis_title="Strike",
            yaxis_title="Pressure Magnitude",
            template="plotly_dark",
            height=400
        )

        # Combine into HTML
        html_content = self._build_html(ticker, fig_surface, fig_slices)
        
        filename = f"pressure_field_{ticker}.html"
        full_path = os.path.join(self.output_dir, filename)
        
        with open(full_path, "w", encoding='utf-8') as f:
            f.write(html_content)
            
        return full_path

    def _build_html(self, ticker, fig1, fig2):
        """
        Assembles the HTML with embedded JS.
        """
        # Get the JS library code
        plotly_js = py_offline.get_plotlyjs()
        
        # Get div strings
        div1 = py_offline.plot(fig1, include_plotlyjs=False, output_type='div')
        div2 = py_offline.plot(fig2, include_plotlyjs=False, output_type='div')

        html = f"""
        <html>
        <head>
            <title>{ticker} Pressure Field</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: sans-serif; background-color: #111; color: #ddd; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .chart-box {{ margin-bottom: 40px; border: 1px solid #333; padding: 10px; background: #1a1a1a; }}
                h1 {{ border-bottom: 2px solid #444; padding-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quant Pressure Analysis: {ticker}</h1>
                <p>Metrics: Blended Gradient Norm of Gamma & Vanna Exposure Z-Scores.</p>
                
                <div class="chart-box">
                    {div1}
                </div>
                
                <div class="chart-box">
                    {div2}
                </div>
            </div>
            <script>
                // Resize fix for hidden tabs/window resizing
                window.onresize = function() {{
                    window.dispatchEvent(new Event('resize'));
                }};
            </script>
        </body>
        </html>
        """
        return html

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Market Pressure Visualizer")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='List of tickers')
    parser.add_argument('--output-dir', default='./market_data', help='Data storage path')
    parser.add_argument('--lookback', type=int, default=1, help='Years of history (Not used for chain snapshot but kept for compat)')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate for BS Model')
    parser.add_argument('--w-gamma', type=float, default=0.6, help='Weight for Gamma Pressure')
    parser.add_argument('--w-vanna', type=float, default=0.4, help='Weight for Vanna Pressure')

    args = parser.parse_args()

    # 1. Initialize Components
    ingestor = DataIngestion(args.output_dir)
    analyst = FinancialAnalysis(
        risk_free_rate=args.risk_free_rate,
        w_gamma=args.w_gamma,
        w_vanna=args.w_vanna
    )
    renderer = DashboardRenderer(args.output_dir)

    print(f"=== Starting Market Pressure Analysis ===")
    print(f"Targets: {args.tickers}")
    print(f"Weights: Gamma={args.w_gamma}, Vanna={args.w_vanna}")
    print("-" * 40)

    # 2. Execution Loop
    for i, ticker in enumerate(args.tickers):
        print(f"\nProcessing [{ticker}] ({i+1}/{len(args.tickers)})...")
        
        # Rate limiting check (simple sleep if not first)
        if i > 0:
            time.sleep(1)

        # A. Ingestion
        df = ingestor.get_data(ticker)
        
        if df is None:
            continue

        # B. Analysis
        print(f"    - Calculating Greeks & Pressure Surface...")
        pressure_surface, raw_grouped = analyst.process_ticker(df)

        if pressure_surface is None:
            print(f"    - [Skipped] Insufficient data for calculations.")
            continue

        # C. Rendering
        print(f"    - Generating HTML Dashboard...")
        report_path = renderer.generate_report(ticker, pressure_surface, raw_grouped)
        
        print(f"    - [Success] Report saved: {report_path}")

    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
