# SCRIPTNAME: 04.dealer.gamma.PNL.curvature.bands.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import os
import time
import datetime
import warnings
import json
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from scipy.stats import norm
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & UTILS
# -----------------------------------------------------------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# -----------------------------------------------------------------------------
# CLASS 1: DataIngestion (The Fortress)
# -----------------------------------------------------------------------------
class DataIngestion:
    """
    Responsibilities:
    - All Data IO (Yfinance, Disk Read/Write).
    - Enforcing Disk-First Pipeline.
    - Sanitizing Data (Universal Fixer).
    """
    def __init__(self, output_dir, lookback_years, snapshot_options):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, 'data')
        self.options_dir = os.path.join(output_dir, 'options')
        self.lookback_years = lookback_years
        self.snapshot_options = snapshot_options
        
        ensure_dir(self.data_dir)
        ensure_dir(self.options_dir)

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Universal Fixer: Handles yfinance API quirks, MultiIndex swapping,
        flattening, and numeric coercion.
        """
        if df.empty:
            return df

        # 1. Swap levels if column structure is (Ticker, Field) instead of (Field, Ticker)
        if isinstance(df.columns, pd.MultiIndex):
            # If level 0 contains the ticker, swap.
            if ticker in df.columns.get_level_values(0) and 'Close' not in df.columns.get_level_values(0):
                 df = df.swaplevel(0, 1, axis=1)

        # 2. Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. Strict Index Handling
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        # 4. Numeric Coercion
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df

    def get_underlying_history(self, ticker: str) -> pd.DataFrame:
        """
        Disk-First pipeline for underlying price data.
        """
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")

        # Check Disk
        if os.path.exists(file_path):
            print(f"[DataIngestion] Loading {ticker} underlying from disk.")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return self._sanitize_df(df, ticker)

        # Download
        print(f"[DataIngestion] Downloading {ticker} underlying via yfinance...")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=self.lookback_years * 365)
        
        # Add sleep to respect rate limits
        time.sleep(1.0)
        
        try:
            df_raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            df_clean = self._sanitize_df(df_raw, ticker)
            
            # Save to Disk
            df_clean.to_csv(file_path)
            return df_clean
        except Exception as e:
            print(f"[DataIngestion] Error downloading {ticker}: {e}")
            return pd.DataFrame()

    def get_options_snapshot(self, ticker: str, max_expiries: int = 6) -> pd.DataFrame:
        """
        Disk-First pipeline for options chains.
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"options_{ticker}_{ts}.csv"
        file_path = os.path.join(self.options_dir, filename)

        print(f"[DataIngestion] Fetching options chain for {ticker}...")
        
        try:
            yf_ticker = yf.Ticker(ticker)
            expiries = yf_ticker.options
            
            if not expiries:
                print(f"[DataIngestion] No options data found for {ticker}")
                return pd.DataFrame()

            all_opts = []
            count = 0
            
            for exp in expiries:
                if count >= max_expiries:
                    break
                
                # Sleep briefly
                time.sleep(0.2)
                
                try:
                    chain = yf_ticker.option_chain(exp)
                    calls = chain.calls
                    puts = chain.puts
                    
                    calls['optionType'] = 'call'
                    puts['optionType'] = 'put'
                    
                    df_exp = pd.concat([calls, puts])
                    df_exp['expirationDate'] = exp
                    df_exp['downloadTimestamp'] = ts
                    
                    all_opts.append(df_exp)
                    count += 1
                except Exception as ex:
                    print(f"Failed to fetch expiry {exp} for {ticker}: {ex}")

            if not all_opts:
                return pd.DataFrame()

            full_chain = pd.concat(all_opts, ignore_index=True)
            
            # Save to Disk
            if self.snapshot_options:
                full_chain.to_csv(file_path, index=False)
            else:
                # Save as 'latest' overwriting
                latest_path = os.path.join(self.options_dir, f"options_{ticker}_latest.csv")
                full_chain.to_csv(latest_path, index=False)
                file_path = latest_path

            # Read back from Disk (Enforcing Pipeline)
            return pd.read_csv(file_path)

        except Exception as e:
            print(f"[DataIngestion] Critical error fetching options for {ticker}: {e}")
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# CLASS 2: FinancialAnalysis (The Math Engine)
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Responsibilities:
    - Pure numerical computation.
    - Black-Scholes, Greeks, Dealer Positioning.
    - Curvature Engine (Taylor Expansion P&L).
    """
    def __init__(self, risk_free_rate, scenario_spot_pct, scenario_vol_points):
        self.r = risk_free_rate
        self.scenario_spot_pct = scenario_spot_pct
        self.scenario_vol_points = scenario_vol_points

    def analyze_underlying(self, df: pd.DataFrame, ticker: str, output_dir: str):
        """Computes realized vol, log returns, SMAs."""
        if df.empty: return None

        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rv_20'] = df['log_ret'].rolling(window=20).std() * np.sqrt(252)
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()

        # Save Artifact
        save_path = os.path.join(output_dir, f"{ticker}_underlying_analytics.parquet")
        df.to_parquet(save_path)
        return df

    def compute_greeks_and_exposures(self, df_opts: pd.DataFrame, spot_price: float, ticker: str, output_dir: str, dealer_short: bool):
        """
        Core pricing engine. Computes BS Greeks and aggregates dealer exposures.
        """
        if df_opts.empty: return None, None, None

        # 1. Normalization
        col_map = {
            'strike': 'strike',
            'lastPrice': 'lastPrice',
            'impliedVolatility': 'iv',
            'openInterest': 'oi',
            'volume': 'vol',
            'expirationDate': 'expiry',
            'optionType': 'type'
        }
        df = df_opts.rename(columns={k:v for k,v in col_map.items() if k in df_opts.columns})
        
        # Fill NaN OI/Vol with 0
        df['oi'] = df['oi'].fillna(0)
        df['vol'] = df['vol'].fillna(0)
        
        # Filter bad data
        df = df[(df['iv'] > 0.001) & (df['iv'] < 5.0) & (df['strike'] > 0)].copy()

        # Time to expiry
        today = datetime.datetime.now()
        df['expiry_dt'] = pd.to_datetime(df['expiry'])
        df['dte_days'] = (df['expiry_dt'] - today).dt.days
        # If DTE <= 0, set to small epsilon (0.5 day)
        df['dte_days'] = df['dte_days'].apply(lambda x: max(x, 0.5))
        df['T'] = df['dte_days'] / 365.0

        # Moneyness
        df['moneyness'] = df['strike'] / spot_price

        # 2. Black-Scholes Greeks
        n_pdf = norm.pdf
        n_cdf = norm.cdf

        # Vectorized calcs
        S = spot_price
        K = df['strike'].values
        T = df['T'].values
        sigma = df['iv'].values
        r = self.r
        q = 0.0 

        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        df['delta'] = 0.0
        mask_call = df['type'] == 'call'
        mask_put = df['type'] == 'put'
        
        df.loc[mask_call, 'delta'] = np.exp(-q*T[mask_call]) * n_cdf(d1[mask_call])
        df.loc[mask_put, 'delta'] = -np.exp(-q*T[mask_put]) * n_cdf(-d1[mask_put])

        # Gamma
        df['gamma'] = (np.exp(-q*T) * n_pdf(d1)) / (S * sigma * np.sqrt(T))

        # Vega (per 1% vol change -> divide by 100)
        df['vega'] = (S * np.exp(-q*T) * n_pdf(d1) * np.sqrt(T)) / 100.0

        # Theta (per day -> divide by 365)
        term1 = -(S * sigma * np.exp(-q*T) * n_pdf(d1)) / (2 * np.sqrt(T))
        term2_call = -r * K * np.exp(-r*T) * n_cdf(d2) + q * S * np.exp(-q*T) * n_cdf(d1)
        term2_put = r * K * np.exp(-r*T) * n_cdf(-d2) - q * S * np.exp(-q*T) * n_cdf(-d1)
        
        df.loc[mask_call, 'theta'] = (term1[mask_call] + term2_call[mask_call]) / 365.0
        df.loc[mask_put, 'theta'] = (term1[mask_put] + term2_put[mask_put]) / 365.0

        # Vanna
        df['vanna'] = (-np.exp(-q*T) * n_pdf(d1) * (d2 / sigma)) / 100.0

        # Charm (Corrected for Broadcasting)
        # Calculates for ALL rows first
        term_charm = (2*(r-q)*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
        
        # Apply masks to T and term_charm during assignment
        df.loc[mask_call, 'charm'] = (q*np.exp(-q*T[mask_call])*n_cdf(d1[mask_call]) - np.exp(-q*T[mask_call])*n_pdf(d1[mask_call])*term_charm[mask_call]) / 365.0
        df.loc[mask_put, 'charm'] = (-q*np.exp(-q*T[mask_put])*n_cdf(-d1[mask_put]) - np.exp(-q*T[mask_put])*n_pdf(d1[mask_put])*term_charm[mask_put]) / 365.0

        # 3. Exposure Aggregation
        contract_mult = 100
        df['gex'] = df['gamma'] * (S**2) * df['oi'] * contract_mult
        df['vanna_ex'] = df['vanna'] * df['oi'] * contract_mult
        df['charm_ex'] = df['charm'] * df['oi'] * contract_mult
        df['vega_ex'] = df['vega'] * df['oi'] * contract_mult
        df['delta_ex'] = df['delta'] * S * df['oi'] * contract_mult 

        # Dealer Short Convention
        sign_mult = -1 if dealer_short else 1

        # Aggregates
        exposures = {
            'total_gex': df['gex'].sum() * sign_mult,
            'total_vega': df['vega_ex'].sum() * sign_mult,
            'total_vanna': df['vanna_ex'].sum() * sign_mult,
            'total_charm': df['charm_ex'].sum() * sign_mult,
            'total_delta': df['delta_ex'].sum() * sign_mult,
            'net_gamma': (df['gamma'] * df['oi'] * contract_mult).sum() * sign_mult,
            'net_vega': (df['vega'] * df['oi'] * contract_mult).sum() * sign_mult,
            'net_delta': (df['delta'] * df['oi'] * contract_mult).sum() * sign_mult,
            'net_vanna': (df['vanna'] * df['oi'] * contract_mult).sum() * sign_mult
        }

        # Save Options + Greeks
        df.to_parquet(os.path.join(output_dir, f"{ticker}_options_greeks.parquet"))

        return df, exposures, sign_mult

    def compute_curvature_grid(self, exposures: dict, spot_price: float, ticker: str, output_dir: str):
        """
        Generates the Dealer P&L Curvature Bands.
        """
        if not exposures: return None, "Neutral"

        max_pct = self.scenario_spot_pct
        max_vol = self.scenario_vol_points 
        
        pct_moves = np.linspace(-max_pct, max_pct, 21)
        vol_shocks = [-max_vol, -max_vol/2, 0, max_vol/2, max_vol]

        net_delta = exposures['net_delta']
        net_gamma = exposures['net_gamma']
        net_vega = exposures['net_vega'] 
        net_vanna = exposures['net_vanna'] 

        grid_data = []

        for v_shock in vol_shocks:
            for pct in pct_moves:
                dS = spot_price * pct
                
                # P&L Components
                pnl_delta = net_delta * dS
                pnl_gamma = 0.5 * net_gamma * (dS**2)
                pnl_vega = net_vega * v_shock
                pnl_vanna = net_vanna * dS * v_shock
                
                total_pnl = pnl_delta + pnl_gamma + pnl_vega + pnl_vanna
                
                grid_data.append({
                    'spot_pct': pct * 100, 
                    'vol_shock': v_shock,
                    'pnl': total_pnl,
                    'pnl_norm': total_pnl 
                })

        df_grid = pd.DataFrame(grid_data)
        
        # Regime Logic
        regime = "Neutral"
        if net_gamma < 0:
            regime = "Panic-Hedging (Short Gamma)"
        elif net_gamma > 0:
            regime = "Vol-Harvesting (Long Gamma)"

        save_path = os.path.join(output_dir, f"{ticker}_pnl_curvature.parquet")
        df_grid.to_parquet(save_path)
        
        return df_grid, regime

# -----------------------------------------------------------------------------
# CLASS 3: DashboardRenderer (The Artist)
# -----------------------------------------------------------------------------
class DashboardRenderer:
    """
    Responsibilities:
    - Plotly Offline Visualization.
    - HTML Generation.
    - JS Fixes (Tab Resize).
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.dashboard_dir = os.path.join(output_dir, 'dashboard')
        ensure_dir(self.dashboard_dir)

    def generate_html(self, data_map: dict):
        html_head = """
        <html>
        <head>
            <title>Hedge Fund Options Analytics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: 'Roboto', sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; padding: 20px; }
                .container { max-width: 1400px; margin: 0 auto; }
                .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444; padding-bottom: 10px; margin-bottom: 20px; }
                h1 { margin: 0; font-size: 24px; color: #4db8ff; }
                select { padding: 8px; background: #333; color: white; border: 1px solid #555; border-radius: 4px; font-size: 16px; }
                .tab { overflow: hidden; border: 1px solid #444; background-color: #2c2c2c; margin-top: 20px; }
                .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #aaa; font-size: 14px; font-weight: bold; }
                .tab button:hover { background-color: #444; color: white; }
                .tab button.active { background-color: #4db8ff; color: #000; }
                .tabcontent { display: none; padding: 20px; border: 1px solid #444; border-top: none; background-color: #252525; animation: fadeEffect 1s; }
                @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
                .metric-card { background: #333; padding: 15px; border-radius: 5px; margin-bottom: 10px; display: inline-block; width: 200px; margin-right: 10px; vertical-align: top;}
                .metric-val { font-size: 18px; font-weight: bold; color: #fff; }
                .metric-label { font-size: 12px; color: #aaa; }
                .regime-badge { padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 14px; }
                .long-gamma { background-color: rgba(0, 255, 0, 0.2); color: #00ff00; }
                .short-gamma { background-color: rgba(255, 0, 0, 0.2); color: #ff4444; }
                .ticker-section { display: none; }
                .ticker-section.active { display: block; }
            </style>
        </head>
        <body>
        <div class="container">
            <div class="header">
                <h1>Hedge Fund Quant Dashboard (Dealer Positioning)</h1>
                <div>
                    <label>Select Ticker: </label>
                    <select id="tickerSelect" onchange="showTicker(this.value)">
        """
        
        tickers = list(data_map.keys())
        for t in tickers:
            html_head += f'<option value="{t}">{t}</option>'
        
        html_head += """
                    </select>
                </div>
            </div>
        """

        html_body = ""
        
        for ticker, data in data_map.items():
            if data['options'] is None:
                html_body += f'<div id="{ticker}" class="ticker-section"><p>No Data Available for {ticker}</p></div>'
                continue

            # Generate Plots
            # A. Underlying Price
            fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig_price.add_trace(go.Candlestick(x=data['underlying'].index,
                                               open=data['underlying']['Open'], high=data['underlying']['High'],
                                               low=data['underlying']['Low'], close=data['underlying']['Close'], name='OHLC'), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=data['underlying'].index, y=data['underlying']['rv_20'], name='RV (20d)', line=dict(color='orange')), row=2, col=1)
            fig_price.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=30, b=20))
            div_price = py_offline.plot(fig_price, include_plotlyjs=False, output_type='div')

            # B. Exposures (GEX Bar)
            df_opts = data['options']
            gex_by_strike = df_opts.groupby('strike')['gex'].sum().sort_index()
            sign_mult = data['sign_mult']
            gex_by_strike = gex_by_strike * sign_mult
            
            fig_gex = go.Figure(go.Bar(x=gex_by_strike.index, y=gex_by_strike.values, marker_color=np.where(gex_by_strike.values>0, 'green', 'red')))
            fig_gex.update_layout(title=f"Total GEX by Strike ({'Dealer Short' if sign_mult == -1 else 'Raw'})", template="plotly_dark", height=500)
            div_gex = py_offline.plot(fig_gex, include_plotlyjs=False, output_type='div')

            # C. Curvature Bands
            df_grid = data['grid']
            fig_curve = go.Figure()
            if df_grid is not None:
                vol_shocks = sorted(df_grid['vol_shock'].unique())
                for v in vol_shocks:
                    subset = df_grid[df_grid['vol_shock'] == v]
                    width = 4 if v == 0 else 1
                    opacity = 1.0 if v == 0 else 0.5
                    name = f"Vol Shock {v}"
                    fig_curve.add_trace(go.Scatter(x=subset['spot_pct'], y=subset['pnl'], mode='lines', line=dict(width=width), opacity=opacity, name=name))
                
                fig_curve.update_layout(title="Dealer P&L Curvature vs Spot Move", xaxis_title="Spot Move (%)", yaxis_title="Dealer P&L ($)", template="plotly_dark", height=500)
            div_curve = py_offline.plot(fig_curve, include_plotlyjs=False, output_type='div')

            # D. Risk Radar
            exposures = data['exposures']
            keys = ['net_delta', 'net_gamma', 'net_vega', 'net_vanna']
            vals = [abs(exposures[k]) for k in keys]
            fig_radar = go.Figure(data=go.Scatterpolar(r=vals, theta=keys, fill='toself'))
            fig_radar.update_layout(title="Risk Magnitude Radar", template="plotly_dark", height=400)
            div_radar = py_offline.plot(fig_radar, include_plotlyjs=False, output_type='div')

            # Ticker Section
            regime_class = "long-gamma" if "Long" in data['regime'] else "short-gamma" if "Short" in data['regime'] else ""
            
            html_body += f"""
            <div id="{ticker}" class="ticker-section">
                <div class="metric-card">
                    <div class="metric-label">Regime</div>
                    <div class="metric-val"><span class="regime-badge {regime_class}">{data['regime']}</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Net Gamma ($)</div>
                    <div class="metric-val">{exposures['net_gamma']:,.0f}</div>
                </div>
                 <div class="metric-card">
                    <div class="metric-label">Net Vega ($/vol pt)</div>
                    <div class="metric-val">{exposures['net_vega']:,.0f}</div>
                </div>

                <div class="tab">
                  <button class="tablinks" onclick="openTab(event, '{ticker}_Price')">Underlying</button>
                  <button class="tablinks" onclick="openTab(event, '{ticker}_Exposures')">Exposures</button>
                  <button class="tablinks" onclick="openTab(event, '{ticker}_Curvature')">P&L Curvature</button>
                  <button class="tablinks" onclick="openTab(event, '{ticker}_Radar')">Risk Radar</button>
                  <button class="tablinks" onclick="openTab(event, '{ticker}_Method')">Methodology</button>
                </div>

                <div id="{ticker}_Price" class="tabcontent" style="display:block;">{div_price}</div>
                <div id="{ticker}_Exposures" class="tabcontent">{div_gex}</div>
                <div id="{ticker}_Curvature" class="tabcontent">
                    <h3>Dealer P&L Sensitivity</h3>
                    <p>Simulated P&L for Dealer Book based on Delta, Gamma, Vega, Vanna vectors.</p>
                    {div_curve}
                </div>
                <div id="{ticker}_Radar" class="tabcontent">{div_radar}</div>
                <div id="{ticker}_Method" class="tabcontent">
                    <h3>Methodology</h3>
                    <ul>
                        <li><b>Source:</b> yfinance (sanitized).</li>
                        <li><b>Model:</b> Black-Scholes. Risk-free rate: {data.get('r_rate', 'N/A')}. Div Yield: 0.</li>
                        <li><b>Dealer P&L:</b> Taylor expansion approximation: dP&L = Delta*dS + 0.5*Gamma*dS^2 + Vega*dVol + Vanna*dS*dVol.</li>
                        <li><b>Sign Convention:</b> { 'Dealer Short (flipped)' if sign_mult == -1 else 'Raw Market Total' }.</li>
                    </ul>
                </div>
            </div>
            """

        html_script = """
        <script>
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

            function showTicker(ticker) {
                var sections = document.getElementsByClassName("ticker-section");
                for (var i = 0; i < sections.length; i++) {
                    sections[i].classList.remove('active');
                }
                document.getElementById(ticker).classList.add('active');
                window.dispatchEvent(new Event('resize'));
            }

            document.addEventListener('DOMContentLoaded', function() {
                var selector = document.getElementById('tickerSelect');
                if (selector.options.length > 0) {
                    showTicker(selector.options[0].value);
                    var firstTab = document.getElementById(selector.options[0].value + "_Price");
                    if(firstTab) firstTab.style.display = "block";
                }
            });
        </script>
        </body>
        </html>
        """
        
        full_html = html_head + html_body + html_script
        
        with open(os.path.join(self.dashboard_dir, "options_dashboard.html"), "w", encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"[DashboardRenderer] Dashboard saved to {os.path.join(self.dashboard_dir, 'options_dashboard.html')}")


# -----------------------------------------------------------------------------
# MAIN CLI EXECUTION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Options Analytics")
    parser.add_argument("--tickers", nargs="+", default=['SPY', 'QQQ', 'IWM'])
    parser.add_argument("--output-dir", default="./market_data")
    parser.add_argument("--lookback", type=int, default=1)
    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--intraday", action="store_true")
    parser.add_argument("--max-expiries", type=int, default=6)
    parser.add_argument("--snapshot-options", action="store_true")
    parser.add_argument("--dealer-short-convention", action="store_true", help="Flip signs to view from Dealer Short perspective")
    parser.add_argument("--scenario-spot-pct", type=float, default=0.02)
    parser.add_argument("--scenario-vol-points", type=float, default=2.0)

    args = parser.parse_args()

    print("--- STARTING QUANT ENGINE ---")
    
    ingest = DataIngestion(args.output_dir, args.lookback, args.snapshot_options)
    fin = FinancialAnalysis(args.risk_free_rate, args.scenario_spot_pct, args.scenario_vol_points)
    dash = DashboardRenderer(args.output_dir)

    dashboard_data = {}

    for ticker in args.tickers:
        print(f"\nProcessing {ticker}...")
        
        df_underlying = ingest.get_underlying_history(ticker)
        df_opts = ingest.get_options_snapshot(ticker, args.max_expiries)
        
        if df_underlying.empty or df_opts.empty:
            print(f"Skipping {ticker} due to missing data.")
            dashboard_data[ticker] = {'options': None}
            continue

        df_underlying = fin.analyze_underlying(df_underlying, ticker, args.output_dir)
        spot_price = df_underlying['Close'].iloc[-1]
        print(f"  Spot Price: {spot_price:.2f}")

        df_greeks, exposures, sign_mult = fin.compute_greeks_and_exposures(df_opts, spot_price, ticker, args.output_dir, args.dealer_short_convention)
        df_grid, regime = fin.compute_curvature_grid(exposures, spot_price, ticker, args.output_dir)

        dashboard_data[ticker] = {
            'underlying': df_underlying,
            'options': df_greeks,
            'grid': df_grid,
            'exposures': exposures,
            'regime': regime,
            'sign_mult': sign_mult,
            'r_rate': args.risk_free_rate
        }

    dash.generate_html(dashboard_data)
    print("\n--- FINISHED ---")

if __name__ == "__main__":
    main()
