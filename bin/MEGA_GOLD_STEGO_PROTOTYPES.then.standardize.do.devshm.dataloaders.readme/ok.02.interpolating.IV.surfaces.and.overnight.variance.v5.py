"""
================================================================================
VOLATILITY SURFACE ENGINE - HEDGE FUND GRADE ARCHITECTURE
================================================================================
Role: Senior Quantitative Developer
Description: Modular, disk-first pipeline for Volatility Surface construction,
             pricing, Greeks, and offline visualization.
             
UPDATES:
- Fixed "Shard" artifact in Contour plot by upsampling grid with scipy.griddata
- Realistic Shadow Backfill
Dependencies: numpy, pandas, scipy, plotly, yfinance
Usage: python vol_surface_engine.py --tickers SPY --lookback 1
================================================================================
"""

import os
import sys
import time
import math
import logging
import argparse
import datetime
import warnings
from typing import List, Dict, Tuple, Optional, Any, Union

# Third-party imports
try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import scipy.stats as stats
    from scipy.interpolate import PchipInterpolator, griddata
    import plotly.graph_objects as go
    import plotly.offline as py_offline
    from plotly.subplots import make_subplots
except ImportError as e:
    sys.exit(f"CRITICAL: Missing dependency. {e}")

# Suppress minor pandas warnings for clean CLI output
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CLASS 1: DATA INGESTION (Disk-First Pipeline)
# ==============================================================================

class DataIngestion:
    """
    Responsible strictly for IO, downloading, sanitization, and persistence.
    No financial math is performed here.
    """
    def __init__(self, output_dir: str, lookback_years: int, options_suffix: str):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        self.options_suffix = options_suffix
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        The 'Universal Fixer' for yfinance quirks.
        """
        df = df.copy()

        # 1. Swap Levels if MultiIndex (Open, High, Low, Close on level 1)
        if isinstance(df.columns, pd.MultiIndex):
            # Heuristic: if 'Close' is not in level 0 but is in level 1
            if 'Close' not in df.columns.get_level_values(0) and 'Close' in df.columns.get_level_values(1):
                df = df.swaplevel(0, 1, axis=1)
            
            # 2. Flatten Columns
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # e.g., ('Close', 'SPY') -> 'Close_SPY'
                    c_name = f"{col[0]}_{col[1]}" if len(col) > 1 else col[0]
                else:
                    c_name = col
                # Normalize
                c_name = c_name.replace(" ", "").replace("^", "")
                new_cols.append(c_name)
            df.columns = new_cols
        
        # 3. Strict Index Handling
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df.sort_index()

        # 4. Numeric Coercion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def get_underlying_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        data_map = {}
        
        for ticker in tickers:
            file_path = os.path.join(self.output_dir, f"{ticker}.csv")
            
            # Disk-First Logic
            if os.path.exists(file_path):
                logger.info(f"Loading underlying from disk: {ticker}")
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = self._sanitize_df(df, ticker)
            else:
                logger.info(f"Downloading underlying: {ticker}")
                
                # Rate limit protection
                if len(tickers) > 1:
                    time.sleep(1)
                
                # Download
                raw_df = yf.download(ticker, period=f"{self.lookback_years}y", progress=False)
                
                # Sanitize immediately
                sanitized_df = self._sanitize_df(raw_df, ticker)
                
                # Write to disk
                sanitized_df.to_csv(file_path)
                
                # Immediate Read-Back (Round-trip enforcement)
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = self._sanitize_df(df, ticker) # Double tap for safety

            data_map[ticker] = df
            
        return data_map

    def get_options_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Loads options data from CSV. 
        Implements SHADOW BACKFILL if data is missing to allow Dashboard demo.
        """
        opts_map = {}
        
        for ticker in tickers:
            fname = f"{ticker}{self.options_suffix}"
            fpath = os.path.join(self.output_dir, fname)
            
            if os.path.exists(fpath):
                logger.info(f"Loading options quotes: {fname}")
                df = pd.read_csv(fpath)
                # Lightweight options sanitization
                df['expiry'] = pd.to_datetime(df['expiry'])
                # Ensure columns exist
                req_cols = ['strike', 'type', 'iv', 'expiry']
                if not all(c in df.columns for c in req_cols):
                    logger.warning(f"Options CSV {fname} missing required columns. Skipping.")
                    continue
                opts_map[ticker] = df
            else:
                logger.warning(f"Options CSV missing for {ticker}. Initiating SHADOW BACKFILL.")
                # Shadow Backfill: Generate synthetic data so the dashboard works
                opts_map[ticker] = self._generate_shadow_options(ticker)
        
        return opts_map

    def _generate_shadow_options(self, ticker: str) -> pd.DataFrame:
        """
        Creates a synthetic volatility surface for demonstration if data is missing.
        """
        today = datetime.datetime.now()
        # Create 4 specific expiries
        expiries = [today + datetime.timedelta(days=d) for d in [30, 60, 90, 180]]
        strikes_pct = np.linspace(0.8, 1.2, 20) # 80% to 120% moneyness
        
        rows = []
        for exp in expiries:
            t = (exp - today).days / 365.0
            if t < 1e-3: t = 1e-3
            
            # --- REALISTIC SHADOW LOGIC ---
            # 1. Term Structure (Inverted/Contango)
            base_vol = 0.18 + (0.07 * np.exp(-2 * t))
            
            # 2. Skew Flattening (1/sqrt(T) effect)
            skew_intensity = 0.15 / np.sqrt(t)
            convexity = 0.8 / np.sqrt(t)
            
            for k_pct in strikes_pct:
                x = np.log(k_pct) 
                
                # Quadratic Smile: Base - Skew*x + Curvature*x^2
                iv = base_vol - (skew_intensity * x) + (convexity * x**2)
                iv = max(iv, 0.01)

                # Assume spot ~ 400 for context
                spot_proxy = 400
                strike = spot_proxy * k_pct
                
                rows.append({
                    'expiry': exp,
                    'strike': strike,
                    'type': 'C',
                    'iv': iv,
                    'bid': 0, 'ask': 0
                })
                rows.append({
                    'expiry': exp,
                    'strike': strike,
                    'type': 'P',
                    'iv': iv,
                    'bid': 0, 'ask': 0
                })
        
        logger.warning(f"Generated {len(rows)} rows of SHADOW options data for {ticker} (With Term Structure).")
        return pd.DataFrame(rows)

# ==============================================================================
# CLASS 2: FINANCIAL ANALYSIS (The Engine)
# ==============================================================================

class FinancialAnalysis:
    """
    Core Logic: Surface construction, Total Variance, Overnight adjustment, Pricing, Greeks.
    """
    def __init__(self, risk_free_rate: float, day_count: float, overnight_var_annual: float):
        self.r = risk_free_rate
        self.day_count = day_count
        self.on_var_annual = overnight_var_annual
        
        # Norm CDF/PDF for Greeks
        self.N = stats.norm.cdf
        self.n = stats.norm.pdf

    def compute_time_to_expiry(self, expiry_date: pd.Timestamp) -> float:
        """Simple ACT/DayCount calculation."""
        now = datetime.datetime.now()
        diff = (expiry_date - now).days
        if diff < 0: return 0.0
        return max(diff / self.day_count, 1e-4) # Avoid div by zero

    def get_spot_price(self, df_underlying: pd.DataFrame, ticker: str) -> float:
        """Extracts latest close from sanitized underlying DF."""
        cols = [c for c in df_underlying.columns if "Close" in c]
        if not cols:
            raise ValueError(f"No Close price found for {ticker}")
        return float(df_underlying[cols[0]].iloc[-1])

    # --------------------------------------------------------------------------
    # Surface Construction
    # --------------------------------------------------------------------------
    
    def build_surface(self, df_opts: pd.DataFrame, spot: float) -> Dict[str, Any]:
        """
        Constructs the volatility surface object.
        Returns a dict of {expiry_date: PchipInterpolator_object} and metadata.
        """
        surface = {}
        unique_exps = sorted(df_opts['expiry'].unique())
        
        smiles = {} # Store raw data for viz
        
        for exp in unique_exps:
            T = self.compute_time_to_expiry(pd.Timestamp(exp))
            if T <= 0.002: continue # Skip expired or Today
            
            # Filter for this expiry
            df_slice = df_opts[df_opts['expiry'] == exp].copy()
            
            # Simple Forward: F = S * e^(rT)
            F = spot * np.exp(self.r * T)
            
            # Calculate Log Moneyness k = ln(K/F)
            df_slice['k'] = np.log(df_slice['strike'] / F)
            
            # Calculate Total Variance w = sigma^2 * T
            df_slice['w'] = (df_slice['iv'] ** 2) * T
            
            # Aggregate to unique k (average IV if duplicates exist)
            agg = df_slice.groupby('k')['w'].mean().sort_index()
            
            k_vals = agg.index.values
            w_vals = agg.values
            
            # Fit Monotone Spline (PCHIP)
            if len(k_vals) < 3:
                continue 
            
            spline = PchipInterpolator(k_vals, w_vals)
            
            surface[T] = {
                'spline': spline,
                'min_k': k_vals[0],
                'max_k': k_vals[-1],
                'raw_k': k_vals,
                'raw_w': w_vals
            }
            
            smiles[str(exp.date())] = {
                'T': T, 'k': k_vals, 'iv': np.sqrt(w_vals/T)
            }
            
        return {'model': surface, 'spot': spot, 'smiles_data': smiles}

    def get_interpolated_variance(self, surface_model: Dict, T_query: float, k_query: float) -> float:
        """
        1. Find bounding expiries T1 <= T_query <= T2.
        2. Interpolate w linearly in time.
        3. Add overnight variance adjustment.
        """
        model = surface_model['model']
        known_Ts = sorted(model.keys())
        
        if not known_Ts:
            return 0.04 # Fallback
            
        # Extrapolation / Bounding
        if T_query <= known_Ts[0]:
            T1, T2 = known_Ts[0], known_Ts[0]
            w1 = model[T1]['spline'](k_query)
            w2 = w1
            ratio = 0
        elif T_query >= known_Ts[-1]:
            T1, T2 = known_Ts[-1], known_Ts[-1]
            w1 = model[T1]['spline'](k_query)
            w2 = w1
            ratio = 0
        else:
            # Find bracket
            idx = np.searchsorted(known_Ts, T_query)
            T1 = known_Ts[idx-1]
            T2 = known_Ts[idx]
            
            w1 = float(model[T1]['spline'](k_query))
            w2 = float(model[T2]['spline'](k_query))
            
            ratio = (T_query - T1) / (T2 - T1)
        
        w_interp = (1 - ratio) * w1 + ratio * w2
        
        # --- Overnight Variance Adjustment ---
        n_overnights = int(T_query * 365)
        w_overnight_add = n_overnights * (self.on_var_annual / self.day_count)
        
        w_final = w_interp + w_overnight_add
        return max(w_final, 1e-6)

    # --------------------------------------------------------------------------
    # Pricing & Greeks
    # --------------------------------------------------------------------------

    def bs_pricer(self, S, K, T, r, sigma, opt_type='C'):
        if T <= 0: return max(S - K, 0) if opt_type == 'C' else max(K - S, 0)
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if opt_type == 'C':
            price = S * self.N(d1) - K * np.exp(-r*T) * self.N(d2)
        else:
            price = K * np.exp(-r*T) * self.N(-d2) - S * self.N(-d1)
        return price

    def calculate_greeks(self, S, K, T, r, sigma, opt_type='C'):
        if T <= 1e-4: return {k:0.0 for k in ['delta','gamma','vega','theta']}
        
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        nd1 = self.n(d1)
        Nd1 = self.N(d1)
        Nd2 = self.N(d2)
        N_d1 = self.N(-d1)
        N_d2 = self.N(-d2)

        if opt_type == 'C':
            delta = Nd1
            theta = (-S * nd1 * sigma / (2 * sqrt_T) - r * K * np.exp(-r*T) * Nd2) / self.day_count
        else:
            delta = Nd1 - 1
            theta = (-S * nd1 * sigma / (2 * sqrt_T) + r * K * np.exp(-r*T) * N_d2) / self.day_count

        gamma = nd1 / (S * sigma * sqrt_T)
        vega = S * nd1 * sqrt_T * 0.01

        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

    def run_analysis(self, ticker: str, spot: float, surface_data: Dict) -> pd.DataFrame:
        """
        Generates a standard grid of queries for the CSV output.
        """
        logger.info(f"Running Financial Analysis for {ticker} (Spot: {spot:.2f})")
        
        results = []
        
        # Grid: 5 expiries, Moneyness 0.8 to 1.2 (9 points)
        Ts = list(surface_data['model'].keys())[:5]
        Ks_pct = np.linspace(0.8, 1.2, 9)
        
        for T in Ts:
            F = spot * np.exp(self.r * T)
            for kp in Ks_pct:
                K = F * kp
                k_log = np.log(K/F)
                
                w = self.get_interpolated_variance(surface_data, T, k_log)
                sigma = np.sqrt(w / T)
                
                px_c = self.bs_pricer(spot, K, T, self.r, sigma, 'C')
                greeks_c = self.calculate_greeks(spot, K, T, self.r, sigma, 'C')
                
                px_p = self.bs_pricer(spot, K, T, self.r, sigma, 'P')
                greeks_p = self.calculate_greeks(spot, K, T, self.r, sigma, 'P')
                
                base_rec = {
                    'ticker': ticker, 'T': round(T, 4), 'K': round(K, 2),
                    'LogMoneyness': round(k_log, 4), 'ImpliedVol': round(sigma, 4)
                }
                
                rc = base_rec.copy()
                rc.update({'Type': 'Call', 'Price': round(px_c, 4), **greeks_c})
                results.append(rc)
                
                rp = base_rec.copy()
                rp.update({'Type': 'Put', 'Price': round(px_p, 4), **greeks_p})
                results.append(rp)
                
        return pd.DataFrame(results)

# ==============================================================================
# CLASS 3: DASHBOARD RENDERER (Visualization)
# ==============================================================================

class DashboardRenderer:
    """
    Generates the offline HTML dashboard using Plotly.
    Includes JS fixes for tabs and embedded data.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def generate_dashboard(self, analysis_results: Dict[str, Any]):
        """
        Constructs the HTML file.
        analysis_results structure: { ticker: { 'smiles': ..., 'df_out': ..., 'spot': ... } }
        """
        logger.info("Rendering Dashboard...")
        
        plots_html = ""
        tickers = list(analysis_results.keys())
        
        for ticker in tickers:
            data = analysis_results[ticker]
            smiles_data = data['smiles']
            spot = data['spot']
            df_res = data['df_out']
            
            # --- 1. Smile Sanity Check ---
            fig_smile = go.Figure()
            colors = ['#00F0FF', '#00FF99', '#FFFF00', '#FF6600', '#FF0000']
            for i, (exp_date, s_data) in enumerate(smiles_data.items()):
                if i >= 5: break
                color = colors[i % len(colors)]
                F = spot * np.exp(0.04 * s_data['T']) 
                strikes = F * np.exp(s_data['k'])
                
                fig_smile.add_trace(go.Scatter(
                    x=strikes, y=s_data['iv'],
                    mode='lines+markers',
                    name=f"{exp_date} (T={s_data['T']:.2f})",
                    line=dict(color=color, width=2)
                ))
            
            fig_smile.update_layout(
                title=f"<b>{ticker} Volatility Smile Structure</b>",
                xaxis_title="Strike Price",
                yaxis_title="Implied Volatility",
                template="plotly_dark",
                hovermode="x unified",
                height=600
            )
            
            # --- 2. 3D Volatility Surface ---
            if smiles_data:
                all_T, all_K, all_Vol = [], [], []
                for exp_date, s_data in smiles_data.items():
                    F = spot * np.exp(0.04 * s_data['T'])
                    strikes = F * np.exp(s_data['k'])
                    all_T.extend([s_data['T']] * len(strikes))
                    all_K.extend(strikes)
                    all_Vol.extend(s_data['iv'])
                
                fig_3d = go.Figure(data=[go.Mesh3d(
                    x=all_K, y=all_T, z=all_Vol,
                    opacity=0.8, color='cyan', intensity=all_Vol, colorscale='Viridis'
                )])
                fig_3d.update_layout(
                    title=f"<b>{ticker} 3D Volatility Surface</b>",
                    scene=dict(xaxis_title='Strike', yaxis_title='Time (Y)', zaxis_title='Implied Vol'),
                    template="plotly_dark", height=600
                )
            else:
                fig_3d = go.Figure().add_annotation(text="No Data for Surface")

            # --- 3. Greeks Profile (Standard Line) ---
            df_calls = df_res[df_res['Type'] == 'Call']
            fig_greeks = go.Figure()
            for T in sorted(df_calls['T'].unique())[:4]:
                sub = df_calls[df_calls['T'] == T]
                fig_greeks.add_trace(go.Scatter(
                    x=sub['K'], y=sub['delta'],
                    mode='lines', name=f"Delta T={T}"
                ))
            fig_greeks.update_layout(
                title=f"<b>{ticker} Call Delta (Line View)</b>",
                xaxis_title="Strike", yaxis_title="Delta",
                template="plotly_dark", height=600
            )

            # --- 4. NEW: Delta Contour Heatmap (The "Cone") ---
            # FIX: UPSAMPLE THE GRID TO REMOVE "SHARDS"
            
            # Filter for Calls
            df_c = df_calls[['K', 'T', 'delta']].copy()
            
            # Create a dense grid for smooth contouring
            # We want ~100x100 points
            if len(df_c) > 10:
                grid_x, grid_y = np.mgrid[
                    df_c['K'].min():df_c['K'].max():100j,
                    df_c['T'].min():df_c['T'].max():100j
                ]
                
                # Interpolate using cubic or linear
                points = df_c[['K', 'T']].values
                values = df_c['delta'].values
                
                try:
                    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
                    
                    # Contour Plot with Dense Grid
                    fig_contour = go.Figure(data=go.Contour(
                        z=grid_z.T,     # Transpose because griddata returns (x,y)
                        x=grid_x[:,0],  # Unique X values
                        y=grid_y[0,:],  # Unique Y values
                        colorscale='RdBu_r', 
                        contours=dict(
                            start=0, end=1, size=0.1,
                            showlabels=True, labelfont=dict(size=12, color='white')
                        ),
                        colorbar=dict(title='Call Delta')
                    ))
                except Exception as e:
                    logger.warning(f"Interpolation failed: {e}. Falling back to raw data.")
                    # Fallback to sparse plot if interpolation fails
                    fig_contour = go.Figure(data=go.Contour(
                        z=df_c['delta'], x=df_c['K'], y=df_c['T'],
                        colorscale='RdBu_r'
                    ))

                fig_contour.update_layout(
                    title=f"<b>{ticker} Delta Probability Cone (Smoothed)</b><br><span style='font-size:12px;color:#aaa'>Red=Deep ITM, Blue=Deep OTM, White=ATM</span>",
                    xaxis_title="Strike Price",
                    yaxis_title="Time to Expiry (Years)",
                    template="plotly_dark",
                    height=600
                )
            else:
                fig_contour = go.Figure().add_annotation(text="Insufficient Data for Contour")


            # Generate DIVs
            div_smile = py_offline.plot(fig_smile, include_plotlyjs=False, output_type='div')
            div_3d = py_offline.plot(fig_3d, include_plotlyjs=False, output_type='div')
            div_greeks = py_offline.plot(fig_greeks, include_plotlyjs=False, output_type='div')
            div_contour = py_offline.plot(fig_contour, include_plotlyjs=False, output_type='div')

            plots_html += f"""
            <div class="ticker-section">
                <h2 style="color: #00F0FF; border-bottom: 1px solid #444;">{ticker} Analysis</h2>
                <div class="tab">
                  <button class="tablinks" onclick="openCity(event, '{ticker}_Smile')">Smile Sanity</button>
                  <button class="tablinks" onclick="openCity(event, '{ticker}_Surface')">3D Surface</button>
                  <button class="tablinks" onclick="openCity(event, '{ticker}_Greeks')">Greeks (Line)</button>
                  <button class="tablinks" onclick="openCity(event, '{ticker}_Contour')">Delta Cone (New)</button>
                </div>
                <div id="{ticker}_Smile" class="tabcontent" style="display:block;">{div_smile}</div>
                <div id="{ticker}_Surface" class="tabcontent">{div_3d}</div>
                <div id="{ticker}_Greeks" class="tabcontent">{div_greeks}</div>
                <div id="{ticker}_Contour" class="tabcontent">{div_contour}</div>
            </div>
            <hr style="border-color: #333; margin: 40px 0;">
            """

        # --- FINAL HTML ASSEMBLY ---
        plotly_js = py_offline.get_plotlyjs()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hedge Fund Volatility Surface</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ background-color: #111; color: #ddd; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #222; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-weight: bold; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #00F0FF; color: #000; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h1 style="text-align: center; color: #fff;">Institutional Volatility Surface Dashboard</h1>
            <p style="text-align: center; color: #888;">Generated: {datetime.datetime.now()}</p>
            {plots_html}
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
                window.dispatchEvent(new Event('resize'));
            }}
            </script>
        </body>
        </html>
        """
        
        out_path = os.path.join(self.output_dir, "vol_surface_dashboard.html")
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard written to: {out_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Volatility Surface Engine")
    
    # Required Baseline
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='List of tickers')
    parser.add_argument('--output-dir', default='./market_data', help='Output directory')
    parser.add_argument('--lookback', type=int, default=1, help='Years of history')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate (decimal)')
    
    # Vol Specific
    parser.add_argument('--overnight-variance', type=float, default=0.0324, help='Annualized variance per overnight (0.18^2)')
    parser.add_argument('--day-count', type=float, default=365.0, help='Annualization factor')
    parser.add_argument('--options-csv-suffix', default='_options.csv', help='Suffix for options CSV files')
    
    args = parser.parse_args()

    logger.info("Initializing Volatility Engine...")
    
    # 1. Instantiate Classes
    ingestion = DataIngestion(args.output_dir, args.lookback, args.options_csv_suffix)
    financials = FinancialAnalysis(args.risk_free_rate, args.day_count, args.overnight_variance)
    renderer = DashboardRenderer(args.output_dir)

    # 2. Ingest Data
    underlying_map = ingestion.get_underlying_data(args.tickers)
    options_map = ingestion.get_options_data(args.tickers)

    dashboard_data = {}

    # 3. Process each ticker
    for ticker in args.tickers:
        try:
            logger.info(f"Processing {ticker}...")
            
            if ticker not in underlying_map or ticker not in options_map:
                logger.error(f"Missing data for {ticker}, skipping.")
                continue

            # Get Spot
            spot = financials.get_spot_price(underlying_map[ticker], ticker)
            
            # Build Surface
            surface_data = financials.build_surface(options_map[ticker], spot)
            
            if not surface_data['model']:
                logger.warning(f"Could not build valid surface for {ticker} (insufficient data).")
                continue
            
            # Run Analysis (Pricing/Greeks Grid)
            df_results = financials.run_analysis(ticker, spot, surface_data)
            
            # Save Results CSV
            res_path = os.path.join(args.output_dir, f"{ticker}_analytics.csv")
            df_results.to_csv(res_path, index=False)
            
            # Store for Dashboard
            dashboard_data[ticker] = {
                'smiles': surface_data['smiles_data'],
                'df_out': df_results,
                'spot': spot
            }
            
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}", exc_info=True)

    # 4. Render Dashboard
    if dashboard_data:
        renderer.generate_dashboard(dashboard_data)
        logger.info("Process Complete. Open the HTML dashboard to view results.")
    else:
        logger.error("No dashboard generated. No valid data processed.")

if __name__ == "__main__":
    main()
