# SCRIPTNAME: ok.vol_surface_dashboard.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import sys
import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# -----------------------------------------------------------------------------
# 1. QUANTITATIVE ANALYTICS ENGINE
# -----------------------------------------------------------------------------

class VolQuantUtils:
    """
    Helper class for financial math, specifically Black-Scholes Delta 
    and Volatility Surface metrics.
    """

    @staticmethod
    def calculate_delta(S, K, T, r, sigma, option_type):
        """
        Approximates the Delta of an option using Black-Scholes.
        Used when yfinance does not provide Greeks.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            return stats.norm.cdf(d1)
        else:
            return stats.norm.cdf(d1) - 1.0

    @staticmethod
    def curvature_2nd_deriv(x, y):
        """
        Calculates discrete 2nd derivative (convexity) of IV vs Strike/Delta.
        dy/dx = (y2-y1)/(x2-x1)
        curvature = d(dy/dx)/dx
        """
        if len(x) < 3:
            return np.zeros(len(x))
        
        # First derivative
        dy = np.diff(y)
        dx = np.diff(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = dy / dx
            
        # Second derivative (centered)
        d2 = np.zeros(len(y))
        # Simple finite difference approximation for internal points
        for i in range(1, len(y)-1):
            if dx[i-1] == 0 or dx[i] == 0: continue
            d2[i] = (d1[i] - d1[i-1]) / (0.5 * (dx[i] + dx[i-1]))
            
        return d2

# -----------------------------------------------------------------------------
# 2. DATA HANDLER
# -----------------------------------------------------------------------------

class VolatilitySurfaceBuilder:
    def __init__(self, tickers, max_expiries=12, risk_free_rate=0.0):
        self.tickers = tickers
        self.max_expiries = max_expiries
        self.r = risk_free_rate
        self.data_store = {} # {ticker: {summary_df, full_chain_df}}

    def run_analysis(self):
        for ticker in self.tickers:
            print(f"[*] Processing {ticker}...")
            try:
                self.process_ticker(ticker)
            except Exception as e:
                print(f"[!] Error processing {ticker}: {e}")

    def process_ticker(self, ticker_symbol):
        yf_ticker = yf.Ticker(ticker_symbol)
        
        # 1. Get Spot Price
        try:
            hist = yf_ticker.history(period="5d")
            if hist.empty:
                print(f"    Skipping {ticker_symbol}: No price history.")
                return
            spot = hist['Close'].iloc[-1]
        except Exception:
            print(f"    Skipping {ticker_symbol}: Failed to fetch spot.")
            return

        print(f"    Spot: {spot:.2f}")

        # 2. Get Expirations
        expirations = yf_ticker.options
        if not expirations:
            print("    No options found.")
            return

        valid_expirations = expirations[:self.max_expiries]
        all_options = []
        term_structure_metrics = []

        today = datetime.datetime.now()

        for exp_str in valid_expirations:
            try:
                # Sleep to prevent rate limiting
                time.sleep(0.3)
                
                # Retrieve Chain
                chain = yf_ticker.option_chain(exp_str)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                
                # Check for empty data
                if calls.empty and puts.empty:
                    continue

                calls['type'] = 'call'
                puts['type'] = 'put'

                # Merge
                df = pd.concat([calls, puts], ignore_index=True)
                
                # Time to Maturity (T)
                exp_date = pd.to_datetime(exp_str)
                days_to_exp = (exp_date - today).days
                if days_to_exp < 3: 
                    continue # Skip very near term noise
                T = days_to_exp / 365.0

                # Data Cleaning & Feature Engineering
                # Midprice
                df['mid'] = (df['bid'] + df['ask']) / 2
                df.loc[df['mid'] == 0, 'mid'] = df['lastPrice'] # Fallback
                
                # Clean IV
                df = df[df['impliedVolatility'] > 0.001].copy()
                df = df[df['impliedVolatility'] < 5.0].copy() # Remove garbage > 500% vol
                
                # Basic Greeks (Delta)
                df['T'] = T
                df['dte'] = days_to_exp  # Fix: Ensure dte is available in full chain
                df['S'] = spot
                df['delta'] = df.apply(lambda row: VolQuantUtils.calculate_delta(
                    spot, row['strike'], T, self.r, row['impliedVolatility'], row['type']
                ), axis=1)

                # --- METRICS CALCULATION ---

                # 1. ATM Identification (Closest Strike)
                df['dist_to_spot'] = abs(df['strike'] - spot)
                atm_row = df.loc[df['dist_to_spot'].idxmin()]
                atm_iv = df[df['strike'] == atm_row['strike']]['impliedVolatility'].mean()

                # 2. Skew Interpolation (25 Delta)
                # Split calls and puts
                call_df = df[df['type']=='call'].sort_values('delta')
                put_df = df[df['type']=='put'].sort_values('delta')

                # Interpolate IV at specific deltas
                # 25-Delta Call (Delta ~= 0.25)
                # 25-Delta Put (Delta ~= -0.25)
                
                iv_25c = np.interp(0.25, call_df['delta'], call_df['impliedVolatility']) if len(call_df) > 1 else np.nan
                iv_25p = np.interp(-0.25, put_df['delta'], put_df['impliedVolatility']) if len(put_df) > 1 else np.nan
                
                risk_reversal = iv_25c - iv_25p # Positive means Calls expensive (Bullish skew), Negative means Puts expensive (Bearish)

                # 3. Curvature (Convexity)
                # We calculate curvature of IV vs Strike for Calls (standard approach)
                call_df_strike_sort = df[df['type'] == 'call'].sort_values('strike')
                curvature = VolQuantUtils.curvature_2nd_deriv(call_df_strike_sort['strike'].values, call_df_strike_sort['impliedVolatility'].values)
                # Assign average curvature to the expiry
                avg_curvature = np.mean(np.abs(curvature)) * 1000 # Scale up

                # Store Metric Summary
                term_structure_metrics.append({
                    'expiry': exp_date,
                    'dte': days_to_exp,
                    'atm_iv': atm_iv,
                    'iv_25c': iv_25c,
                    'iv_25p': iv_25p,
                    'risk_reversal': risk_reversal,
                    'curvature': avg_curvature,
                    'volume_total': df['volume'].sum(),
                    'oi_total': df['openInterest'].sum()
                })

                # Store Full Chain for Surfaces
                df['expiry_date'] = exp_date
                df['curvature'] = 0.0 # Placeholder
                all_options.append(df)
                
                print(f"    -> Analyzed {exp_str} (ATM IV: {atm_iv:.1%})")

            except Exception as e:
                print(f"    [!] Failed {exp_str}: {e}")

        if not all_options:
            return

        # Consolidate
        full_chain_df = pd.concat(all_options, ignore_index=True)
        summary_df = pd.DataFrame(term_structure_metrics).sort_values('expiry')
        
        # Calc Slope (Forward Vol)
        summary_df['slope_iv'] = summary_df['atm_iv'].diff() # Simple slope
        
        self.data_store[ticker_symbol] = {
            'summary': summary_df,
            'chain': full_chain_df,
            'spot': spot
        }

# -----------------------------------------------------------------------------
# 3. DASHBOARD GENERATOR (PLOTLY HTML)
# -----------------------------------------------------------------------------

class DashboardRenderer:
    def __init__(self, data_store):
        self.data_store = data_store
        # We will use the first ticker in the list for the main display if multiple
        # But this script supports multi-ticker ingestion, we focus display on the first one
        # or generate a combined report.
        self.active_ticker = list(data_store.keys())[0] if data_store else None

    def create_html(self, output_file):
        if not self.active_ticker:
            print("No data to visualize.")
            return

        print(f"[*] Building Dashboard for {self.active_ticker}...")
        data = self.data_store[self.active_ticker]
        summary = data['summary']
        chain = data['chain']
        spot = data['spot']

        # --- TAB 1: ATM Term Structure ---
        # Trader View: Is vol cheap or expensive in the back months? 
        # Contango (upward slope) is normal. Backwardation (downward) implies near-term fear.
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=summary['expiry'], y=summary['atm_iv'], 
                                 mode='lines+markers', name='ATM IV',
                                 line=dict(color='#00e676', width=3)))
        # Trendline (Polyfit)
        z = np.polyfit(summary['dte'], summary['atm_iv'], 2)
        p = np.poly1d(z)
        fig1.add_trace(go.Scatter(x=summary['expiry'], y=p(summary['dte']), 
                                 mode='lines', name='Fit', line=dict(dash='dot', color='gray')))
        fig1.update_layout(title=f"ATM Term Structure ({self.active_ticker})", xaxis_title="Expiry", yaxis_title="Implied Volatility")

        # --- TAB 2: Full Delta Smile per Expiry ---
        # Trader View: How much are OTM options bid up? 
        # Slider allows scanning through time to see how the smile steepens.
        fig2 = go.Figure()
        
        # Add traces, but make them invisible initially
        for exp in summary['expiry'].unique():
            exp_chain = chain[chain['expiry_date'] == exp]
            calls = exp_chain[exp_chain['type'] == 'call']
            
            # Map Call Delta 0-1.
            fig2.add_trace(go.Scatter(
                x=calls['delta'], y=calls['impliedVolatility'],
                mode='markers+lines', name=str(exp.date()),
                visible=False
            ))

        if fig2.data:
            fig2.data[0].visible = True

        # Slider logic
        steps = []
        for i, exp in enumerate(summary['expiry'].unique()):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig2.data)},
                      {"title": f"Volatility Smile: {exp.date()}"}],
                label=str(exp.date())
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)

        fig2.update_layout(
            sliders=[dict(active=0, currentvalue={"prefix": "Expiry: "}, steps=steps)],
            title="Volatility Smile by Delta (Call Side)",
            xaxis_title="Delta (0.0 to 1.0)", yaxis_title="Implied Volatility"
        )

        # --- TAB 3: Risk Reversal Time Series ---
        # Trader View: Sentiment indicator. 
        # Positive = Calls > Puts (Bullish/Crash-up fear). Negative = Puts > Calls (Bearish/Crash-down fear).
        colors = ['#ff1744' if v < 0 else '#2979ff' for v in summary['risk_reversal']]
        fig3 = go.Figure(go.Bar(
            x=summary['expiry'], y=summary['risk_reversal'],
            marker_color=colors
        ))
        fig3.update_layout(title="25-Delta Risk Reversal (Skew)", yaxis_title="IV Call - IV Put")

        # --- TAB 4: Curvature Heatmap ---
        # Trader View: Convexity. High curvature means wings are very expensive relative to ATM.
        # Approximation: Using relative strike distance
        heatmap_data = []
        strikes_pct = np.linspace(0.8, 1.2, 20) # 80% to 120% of spot
        expiries_str = [str(d.date()) for d in summary['expiry']]
        
        z_grid = []
        for exp in summary['expiry']:
            # Fix: Use .copy() to prevent SettingWithCopyWarning
            exp_df = chain[(chain['expiry_date'] == exp) & (chain['type'] == 'call')].copy()
            # Interpolate IV on fixed moneyness grid
            exp_df['moneyness'] = exp_df['strike'] / spot
            iv_interp = np.interp(strikes_pct, exp_df['moneyness'], exp_df['impliedVolatility'])
            
            # Calculate 2nd derivative (curvature) on this grid
            curv = np.gradient(np.gradient(iv_interp)) * 100 # Scale
            z_grid.append(curv)

        fig4 = go.Figure(data=go.Heatmap(
            z=z_grid, x=strikes_pct*100, y=expiries_str,
            colorscale='Magma', colorbar=dict(title="Convexity")
        ))
        fig4.update_layout(title="Smile Curvature Heatmap", xaxis_title="% of Spot", yaxis_title="Expiry")

        # --- TAB 5: 2D Strike-IV Scatter + PolyFit ---
        # Trader View: 'Birds eye' view of all quotes to spot skew anomalies.
        fig5 = go.Figure()
        # Normalized Strike
        chain['pct_spot'] = chain['strike'] / spot
        
        fig5.add_trace(go.Scattergl(
            x=chain['pct_spot'], y=chain['impliedVolatility'],
            mode='markers', marker=dict(size=3, color=chain['dte'], colorscale='Viridis', showscale=True),
            text=chain['expiry_date'].dt.date, name='Quotes'
        ))
        
        # Global Polyfit (Visual Guide)
        clean_fit = chain.dropna(subset=['impliedVolatility'])
        if not clean_fit.empty:
            z_poly = np.polyfit(clean_fit['pct_spot'], clean_fit['impliedVolatility'], 3)
            p_poly = np.poly1d(z_poly)
            x_rng = np.linspace(clean_fit['pct_spot'].min(), clean_fit['pct_spot'].max(), 100)
            fig5.add_trace(go.Scatter(x=x_rng, y=p_poly(x_rng), mode='lines', name='Global Cubic Fit', line=dict(color='white', width=2)))

        fig5.update_layout(title="Global Skew Scatter (All Expiries)", xaxis_title="Strike (% of Spot)", yaxis_title="IV")

        # --- TAB 6: 3D Vol Surface ---
        # Trader View: The Holy Grail. Visualizing term structure and skew simultaneously.
        # Meshgrid required
        try:
            calls_only = chain[chain['type'] == 'call']
            # Create grid
            x_surf = np.linspace(0.8 * spot, 1.2 * spot, 30) # Strikes
            y_surf = summary['dte'].values # DTE
            X, Y = np.meshgrid(x_surf, y_surf)
            
            # Interpolate Z (IV)
            # Flatten source data
            points = calls_only[['strike', 'dte']].values
            values = calls_only['impliedVolatility'].values
            
            Z = griddata(points, values, (X, Y), method='linear')
            
            fig6 = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Electric')])
            fig6.update_layout(title="3D Volatility Surface", scene=dict(
                xaxis_title='Strike', yaxis_title='DTE', zaxis_title='IV'
            ))
        except Exception as e:
            fig6 = go.Figure().add_annotation(text=f"Surface Error: {e}")

        # --- TAB 7: Volume Heatmap ---
        # Trader View: Where are the big bets? Liquidity Shelves.
        # Aggregate Volume by Strike and Expiry
        pivot_vol = chain.pivot_table(index='expiry_date', columns='strike', values='volume', aggfunc='sum').fillna(0)
        # Filter sparse columns for cleaner plot
        pivot_vol = pivot_vol.loc[:, (pivot_vol.sum(axis=0) > 100)] 
        
        fig7 = go.Figure(data=go.Heatmap(
            z=pivot_vol.values,
            x=pivot_vol.columns,
            y=[str(d.date()) for d in pivot_vol.index],
            colorscale='Mint'
        ))
        fig7.update_layout(title="Volume Concentration Heatmap", xaxis_title="Strike", yaxis_title="Expiry")

        # --- TAB 8: Term Structure Slope ---
        # Trader View: Rate of change in IV across time.
        # Steep Contango = Normal. Flat/Inverted = Event Risk.
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=summary['expiry'], y=summary['slope_iv'],
            fill='tozeroy', mode='lines', line=dict(color='orange')
        ))
        fig8.update_layout(title="Term Structure Slope (dIV/dt)", yaxis_title="Slope (Change in IV)")

        # --- EXPORT TO HTML ---
        self._generate_html_file(output_file, [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8])

    def _generate_html_file(self, filename, figures):
        """
        Compiles distinct Plotly figures into a single tabbed HTML file.
        """
        # Convert figures to HTML divs
        divs = [pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False, config={'responsive': True}) 
                for i, fig in enumerate(figures)]

        # Tab Names
        tabs = [
            "1. ATM Term Structure", "2. Delta Smile", "3. Risk Reversal", 
            "4. Curvature", "5. Global Scatter", "6. 3D Surface", 
            "7. Volume Map", "8. TS Slope"
        ]

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vol Surface: {self.active_ticker}</title>
            <style>
                body {{ font-family: sans-serif; background-color: #111; color: #eee; margin: 0; }}
                .tab {{ overflow: hidden; border-bottom: 1px solid #333; background-color: #222; }}
                .tab button {{
                    background-color: inherit; float: left; border: none; outline: none;
                    cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888;
                }}
                .tab button:hover {{ background-color: #333; color: white; }}
                .tab button.active {{ background-color: #00e676; color: #000; font-weight: bold; }}
                .tabcontent {{ display: none; padding: 6px 12px; height: 90vh; }}
                h1 {{ padding: 10px; margin: 0; font-size: 18px; color: #00e676; }}
            </style>
        </head>
        <body>
            <div style="display:flex; justify-content:space-between; align-items:center; background:#1e1e1e;">
                <h1>Volatility Diagnostics: {self.active_ticker} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>
            </div>

            <div class="tab">
                {''.join([f'<button class="tablinks" onclick="openTab(event, \'Tab{i}\')">{name}</button>' for i, name in enumerate(tabs)])}
            </div>

            {''.join([f'<div id="Tab{i}" class="tabcontent">{div}</div>' for i, div in enumerate(divs)])}

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
                    window.dispatchEvent(new Event('resize')); // Fix Plotly resize bug
                }}
                // Open first tab default
                document.getElementsByClassName("tablinks")[0].click();
            </script>
        </body>
        </html>
        """

        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[SUCCESS] Dashboard saved to {filename}")

# -----------------------------------------------------------------------------
# 4. MAIN CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Professional Volatility Surface & Skew Dashboard")
    parser.add_argument("--tickers", nargs='+', default=['SPY'], help="List of tickers (e.g. SPY NVDA)")
    parser.add_argument("--output", default="vol_dashboard.html", help="Output HTML filename")
    parser.add_argument("--max-expiries", type=int, default=12, help="Max number of expirations to process")
    parser.add_argument("--riskfree", type=float, default=0.0, help="Risk free rate (decimal, e.g. 0.045)")

    args = parser.parse_args()

    # 1. Build Surface Data
    builder = VolatilitySurfaceBuilder(args.tickers, args.max_expiries, args.riskfree)
    builder.run_analysis()

    # 2. Render Dashboard
    if builder.data_store:
        renderer = DashboardRenderer(builder.data_store)
        renderer.create_html(args.output)
    else:
        print("[!] No data successfully processed.")

if __name__ == "__main__":
    main()
