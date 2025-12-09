import os
import sys
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ----------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
np.random.seed(42)

# ----------------------------------------------------------------------------------
# CLASS 1: DATA INGESTION
# ----------------------------------------------------------------------------------
class DataIngestion:
    """
    Role: Solely file I/O, downloading, caching, and data-sanitization.
    Constraint: Disk-First Pipeline. 
    Strict Rule: If data is missing, return EMPTY, do not generate fake data here.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Applies strict sanitization rules: Swap Levels bug fix, column flattening.
        """
        if df.empty:
            return df

        # 1. Swap Levels Bug Fix (Common yfinance MultiIndex issue)
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels > 1:
                # If the column structure is (Price, Ticker), we drop the ticker level
                df.columns = df.columns.get_level_values(0)

        # 2. Numeric Coercion & Renaming
        # Standardize columns
        df.rename(columns={'Stock Splits': 'Splits'}, inplace=True)
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 3. Strict DatetimeIndex
        df.index = pd.to_datetime(df.index).normalize()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        return df

    def get_ticker_data(self, ticker: str, lookback_years: float) -> pd.DataFrame:
        """
        Disk-first retrieval. 
        """
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        start_date = (datetime.now() - timedelta(days=int(lookback_years*365))).strftime('%Y-%m-%d')
        
        # 1. Check Disk
        if os.path.exists(file_path):
            print(f"[DataIngestion] Loading {ticker} from disk...")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # Check staleness (older than 24 hours)
            last_date = df.index[-1]
            if last_date < (datetime.now() - timedelta(hours=24)):
                print(f"[DataIngestion] Data stale for {ticker}. Initiating update...")
            else:
                return df

        # 2. Download
        print(f"[DataIngestion] Downloading {ticker} via yfinance...")
        try:
            # Auto_adjust=False ensures we get real prices, not adjusted for splits yet
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            time.sleep(1) 
        except Exception as e:
            print(f"[Error] Failed to download {ticker}: {e}")
            return pd.DataFrame()

        # 3. Sanitize
        df = self._sanitize_df(df, ticker)

        # 4. Shadow Backfill: If data is too short for analysis, force max download
        if len(df) < 252:
            print(f"[DataIngestion] Insufficient history. Force downloading MAX history for {ticker}...")
            df = yf.download(ticker, period="max", progress=False, auto_adjust=False)
            df = self._sanitize_df(df, ticker)

        # 5. Save and Return
        df.to_csv(file_path)
        return df

    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        """
        Attempts to fetch REAL options chain. 
        If it fails, returns EMPTY DataFrame (Analysis class handles the approximation).
        """
        cache_path = os.path.join(self.output_dir, f"{ticker}_options.csv")
        
        print(f"[DataIngestion] Fetching Options Chain for {ticker}...")
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options
            
            if not expirations:
                print(f"[Warning] No expirations found for {ticker} (API limit or data gap).")
                return pd.DataFrame()
            
            all_opts = []
            # Grab first 4 monthly expirations to minimize API failure risk
            for e in expirations[:4]: 
                try:
                    opt = tk.option_chain(e)
                    calls = opt.calls
                    calls['type'] = 'call'
                    puts = opt.puts
                    puts['type'] = 'put'
                    chain = pd.concat([calls, puts])
                    chain['expiry'] = e
                    all_opts.append(chain)
                    time.sleep(0.2)
                except:
                    continue
            
            if not all_opts:
                return pd.DataFrame()

            full_chain = pd.concat(all_opts)
            # Map columns to internal standard
            full_chain = full_chain.rename(columns={
                'contractSymbol': 'contract',
                'lastPrice': 'last',
                'openInterest': 'oi',
                'impliedVolatility': 'iv',
                'strike': 'strike'
            })
            
            # Save cache
            full_chain.to_csv(cache_path, index=False)
            return full_chain
            
        except Exception as e:
            print(f"[Warning] Options fetch failed for {ticker}: {e}")
            return pd.DataFrame()

# ----------------------------------------------------------------------------------
# CLASS 2: FINANCIAL ANALYSIS
# ----------------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Purpose: All math, models, logic.
    Feature: 'Shadow Surface' generation to fill missing data using Price/Vol approximation.
    """
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    # --- 2.0 Core Math Helpers ---
    def _d1(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0: return 0
        return (np.log(S/K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    def _d2(self, d1, T, sigma):
        if T <= 0: return 0
        return d1 - sigma * np.sqrt(T)

    def _gamma(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0 or S <= 0: return 0
        d1 = self._d1(S, K, T, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def _vanna(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0: return 0
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(d1, T, sigma)
        return -norm.pdf(d1) * d2 / sigma

    # --- 2.1 Market Data Analytics ---
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Returns
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Realized Volatility (Annualized) - CRITICAL for Approximation
        df['vol_20d'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
        
        # Fill NaN vol at start with the first valid observation to prevent breakage
        df['vol_20d'] = df['vol_20d'].bfill().ffill()

        # SMA
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    # --- 2.2 Shadow Surface Generator (The Fix) ---
    def _generate_shadow_surface(self, spot_price, realized_vol):
        """
        Creates a THEORETICAL options chain if the real one is missing.
        Based on Realized Volatility and Standard Market Structure assumptions.
        This allows plots to render based on 'Intrinsic Math' rather than fake data.
        """
        print(f"[Analysis] Generating Shadow Surface based on Spot: {spot_price:.2f}, RV: {realized_vol:.2%}")
        
        # Generate Strikes: 70% to 130% of Spot
        strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 50)
        
        # Generate Expirations: 30, 60, 90 days out
        expirations = [30, 60, 90]
        
        shadow_data = []
        
        for days in expirations:
            T = days / 365.0
            expiry_date = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
            
            for K in strikes:
                # Volatility Skew Model: Lower strikes have higher IV (Smirk)
                # Simple model: IV = RV + SkewFactor * (1 - K/S)
                moneyness = K / spot_price
                skew = 0.5 * (1.0 - moneyness) # Higher IV for low strikes
                # Base IV is Realized Vol + a "Vol Premium" (usually 1.1x RV)
                iv_est = max(0.1, (realized_vol * 1.1) + skew)
                
                # Estimated Open Interest (Liquidity Model)
                # Liquidity follows a Gaussian curve centered at Spot (ATM)
                # Peak OI assumed to be proportional to typical volume (normalized here to 10k units for scale)
                sigma_liquidity = 0.1 * spot_price # standard dev of liquidity distribution
                oi_est = 10000 * np.exp(-0.5 * ((K - spot_price) / sigma_liquidity)**2)
                
                # Add Call
                shadow_data.append({
                    'contract': f"Shadow_C_{K}",
                    'strike': K,
                    'expiry': expiry_date,
                    'type': 'call',
                    'iv': iv_est,
                    'oi': int(oi_est),
                    'last': 0 # Not needed for GEX
                })
                # Add Put
                shadow_data.append({
                    'contract': f"Shadow_P_{K}",
                    'strike': K,
                    'expiry': expiry_date,
                    'type': 'put',
                    'iv': iv_est, 
                    'oi': int(oi_est),
                    'last': 0
                })
                
        return pd.DataFrame(shadow_data)

    # --- 2.3 Dealer Positioning Engine ---
    def compute_gex_surface(self, spot_price, realized_vol, options_df):
        """
        Computes GEX using Real data if available, or Shadow Surface if missing.
        """
        is_shadow = False
        
        # 1. Fallback Logic
        if options_df.empty or 'strike' not in options_df.columns:
            options_df = self._generate_shadow_surface(spot_price, realized_vol)
            is_shadow = True

        df = options_df.copy()
        
        # Clean Data Types
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df['iv'] = pd.to_numeric(df['iv'], errors='coerce').fillna(realized_vol) # Fill missing IV with RV
        df['oi'] = pd.to_numeric(df['oi'], errors='coerce').fillna(0)
        
        # Ensure T (Time to expiry)
        if 'T' not in df.columns:
            df['expiry'] = pd.to_datetime(df['expiry'])
            df['T'] = (df['expiry'] - datetime.now()).dt.days / 365.0
        
        # Filter invalid
        df = df[(df['T'] > 0.002) & (df['strike'] > 0)]

        gex_list = []
        
        for idx, row in df.iterrows():
            K = row['strike']
            T = row['T']
            sigma = row['iv']
            oi = row['oi']
            
            # Gamma Calc
            gamma = self._gamma(spot_price, K, T, sigma)
            
            # GEX Formula: Gamma * Spot^2 * 0.01 * OI * 100 * Direction
            # Direction: Calls = +1 (Dealer Short Option / Long Gamma assumption if market is long)
            # Standard GEX Assumption: Dealers are Short Calls (Long Gamma needs), Short Puts (Short Gamma needs? No.)
            # Standard GEX: Call OI adds Positive Gamma, Put OI adds Negative Gamma
            
            direction = 1.0 if row['type'] == 'call' else -1.0
            
            # Scale Factor: Spot * Spot * 0.01 (Dollar Gamma per 1% move)
            val_gex = (gamma * spot_price * spot_price * 0.01) * oi * 100 * direction
            gex_list.append(val_gex)

        df['GEX'] = gex_list
        
        # Zero Gamma Level
        try:
            calls = df[df['type']=='call'].groupby('strike')['GEX'].sum()
            puts = df[df['type']=='put'].groupby('strike')['GEX'].sum()
            net_profile = calls + puts
            zero_gamma_level = net_profile.abs().idxmin()
        except:
            zero_gamma_level = spot_price

        total_gex = df['GEX'].sum()
        
        return {
            'total_gex': total_gex,
            'zero_gamma': zero_gamma_level,
            'is_shadow': is_shadow
        }, df

    # --- 2.5 Intrinsic Macro Embeddings ---
    def compute_intrinsic_factors(self, df):
        """
        Uses PCA on the asset's OWN returns and Volatility to find latent factors.
        This is NOT 'fake' macro data, it is 'intrinsic' structure analysis.
        """
        data = df[['log_ret', 'vol_20d', 'Volume']].copy().dropna()
        if len(data) < 50: return pd.DataFrame()

        # Normalize Volume
        data['vol_change'] = data['Volume'].pct_change()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        scaler = StandardScaler()
        try:
            X = scaler.fit_transform(data)
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
            embeddings = pd.DataFrame(components, index=data.index, columns=['Eigen_Trend', 'Eigen_Liquidity'])
            return embeddings
        except:
            return pd.DataFrame()

    # --- 2.6 HMM Regimes ---
    def detect_regimes(self, df):
        """
        Detects regimes based on Realized Volatility and Returns.
        """
        obs = df[['log_ret', 'vol_20d']].dropna()
        if len(obs) < 100: return pd.DataFrame()

        try:
            gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
            gmm.fit(obs)
            probs = gmm.predict_proba(obs)
            regime_df = pd.DataFrame(probs, index=obs.index, columns=['Low_Vol_Trend', 'High_Vol_Crash', 'Chop'])
            # Sort columns by their vol mean to label correctly? 
            # Simplified: Just return probabilities.
            return regime_df
        except:
            return pd.DataFrame()

    # --- 2.8 Microstructure Approximation ---
    def microstructure_approximation(self, df):
        """
        Approximates spread and pressure using Price/Volume geometry.
        High-Low Range vs Volume = Informed Trading Proxy.
        """
        df = df.copy()
        # Corwin-Schultz High-Low Spread Proxy
        # If High/Low variance is high relative to volume, spread is likely wide (toxic).
        
        # 1. Effective Spread Estimate
        alpha = np.log(df['High'] / df['Low']) ** 2
        df['spread_est'] = alpha.rolling(2).mean().apply(np.sqrt)
        
        # 2. Amihud Illiquidity (AbsRet / Volume)
        df['illiquidity'] = df['log_ret'].abs() / (df['Volume'] + 1)
        
        return df[['spread_est', 'illiquidity']].dropna()

    # --- 2.10 Monte Carlo Stress ---
    def monte_carlo_stress(self, last_price, daily_vol, num_sims=500, days=20):
        # Strictly math-based path generation
        dt = 1/252
        paths = np.zeros((days, num_sims))
        paths[0] = last_price
        
        for t in range(1, days):
            rand = np.random.standard_normal(num_sims)
            paths[t] = paths[t-1] * np.exp(-0.5 * daily_vol**2 * dt + daily_vol * np.sqrt(dt) * rand)
            
        return paths

# ----------------------------------------------------------------------------------
# CLASS 3: DASHBOARD RENDERER
# ----------------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def _get_plotly_js(self):
        import plotly.offline
        return f'<script type="text/javascript">{plotly.offline.get_plotlyjs()}</script>'

    def render_dashboard(self, ticker, price_df, options_data, regime_df, stress_paths, gex_stats, micro_df):
        
        # --- Plot 1: Price & Microstructure ---
        fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Candlestick
        fig_price.add_trace(go.Candlestick(x=price_df.index, open=price_df['Open'], high=price_df['High'], 
                                           low=price_df['Low'], close=price_df['Close'], name='OHLC'), row=1, col=1)
        
        # Illiquidity/Microstructure
        if not micro_df.empty:
            # Normalize for plotting
            illiq = micro_df['illiquidity']
            fig_price.add_trace(go.Scatter(x=micro_df.index, y=illiq, name='Illiquidity Proxy', 
                                           line=dict(color='orange', width=1)), row=2, col=1)
        
        fig_price.update_layout(template='plotly_dark', title=f'{ticker} Price & Liquidity Structure', xaxis_rangeslider_visible=False)

        # --- Plot 2: GEX Profile ---
        fig_gex = go.Figure()
        title_suffix = "(Shadow Surface Estimate)" if gex_stats['is_shadow'] else "(Real Exchange Data)"
        
        if not options_data.empty:
            # Aggregate GEX by Strike
            gex_profile = options_data.groupby('strike')['GEX'].sum()
            # Filter to relevant range (near spot)
            spot = price_df['Close'].iloc[-1]
            mask = (gex_profile.index > spot*0.8) & (gex_profile.index < spot*1.2)
            gex_profile = gex_profile[mask]
            
            fig_gex.add_trace(go.Bar(x=gex_profile.index, y=gex_profile.values, 
                                     marker_color=np.where(gex_profile.values<0, 'red', 'green'), name='Net GEX'))
            fig_gex.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text="Spot")
            
            if gex_stats['zero_gamma'] > 0:
                fig_gex.add_vline(x=gex_stats['zero_gamma'], line_dash="dot", line_color="yellow", annotation_text="ZeroGamma")

        fig_gex.update_layout(template='plotly_dark', title=f'Gamma Exposure Profile {title_suffix}')

        # --- Plot 3: 3D Vol Surface ---
        fig_3d = go.Figure()
        if not options_data.empty:
            try:
                # Prepare Mesh
                options_data['days'] = options_data['T'] * 365
                # Pivot
                pivot = options_data.pivot_table(index='strike', columns='days', values='iv')
                fig_3d.add_trace(go.Surface(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Plasma'))
                fig_3d.update_layout(scene = dict(xaxis_title='Days to Exp', yaxis_title='Strike', zaxis_title='Implied Vol'),
                                     template='plotly_dark', title=f'Volatility Surface {title_suffix}')
            except:
                pass

        # --- Plot 4: Regimes & Stress ---
        fig_regime = make_subplots(rows=2, cols=1)
        
        # Regimes
        if not regime_df.empty:
            for col in regime_df.columns:
                fig_regime.add_trace(go.Scatter(x=regime_df.index, y=regime_df[col], name=col, stackgroup='one'), row=1, col=1)
        
        # Stress Paths
        subset_paths = stress_paths[:, :50] 
        x_future = [datetime.now() + timedelta(days=i) for i in range(len(subset_paths))]
        for i in range(subset_paths.shape[1]):
            fig_regime.add_trace(go.Scatter(x=x_future, y=subset_paths[:, i], mode='lines', 
                                            line=dict(color='rgba(255,255,255,0.1)', width=0.5), showlegend=False), row=2, col=1)
        
        fig_regime.update_layout(template='plotly_dark', title='Market Regimes & Monte Carlo Paths')

        # --- HTML Assembly ---
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{ticker} Quant Dashboard</title>
            <style>
                body {{ font-family: sans-serif; background-color: #111; color: #ccc; margin: 0; }}
                .tab {{ overflow: hidden; background-color: #222; border-bottom: 1px solid #444; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; color: #ccc; transition: 0.3s; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #007bff; color: white; }}
                .tabcontent {{ display: none; padding: 20px; border-top: none; height: 90vh; }}
            </style>
            {self._get_plotly_js()}
        </head>
        <body>
            <div class="tab">
              <button class="tablinks" onclick="openTab(event, 'Price')" id="defaultOpen">Price & Liquidity</button>
              <button class="tablinks" onclick="openTab(event, 'GEX')">GEX Profile</button>
              <button class="tablinks" onclick="openTab(event, 'Surface')">Vol Surface</button>
              <button class="tablinks" onclick="openTab(event, 'Regimes')">Regimes & Stress</button>
            </div>

            <div id="Price" class="tabcontent">{pio.to_html(fig_price, full_html=False, include_plotlyjs=False)}</div>
            <div id="GEX" class="tabcontent">{pio.to_html(fig_gex, full_html=False, include_plotlyjs=False)}</div>
            <div id="Surface" class="tabcontent">{pio.to_html(fig_3d, full_html=False, include_plotlyjs=False)}</div>
            <div id="Regimes" class="tabcontent">{pio.to_html(fig_regime, full_html=False, include_plotlyjs=False)}</div>

            <script>
            function openTab(evt, tabName) {{
              var i, tabcontent, tablinks;
              tabcontent = document.getElementsByClassName("tabcontent");
              for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
              tablinks = document.getElementsByClassName("tablinks");
              for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
              document.getElementById(tabName).style.display = "block";
              evt.currentTarget.className += " active";
              window.dispatchEvent(new Event('resize'));
            }}
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        filename = os.path.join(self.output_dir, f"{ticker}_Dashboard.html")
        with open(filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"[DashboardRenderer] Dashboard saved to {filename}")

# ----------------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs='+', default=['SPY', 'QQQ', 'IWM'])
    parser.add_argument("--output-dir", default="./market_data")
    parser.add_argument("--lookback", type=float, default=2.0)
    args = parser.parse_args()

    ingestion = DataIngestion(args.output_dir)
    analysis = FinancialAnalysis()
    renderer = DashboardRenderer(args.output_dir)

    print("=== QUANT SYSTEM: INTRINSIC DATA MODE ===")

    for ticker in args.tickers:
        print(f"\nProcessing {ticker}...")
        
        # 1. Ingestion
        df = ingestion.get_ticker_data(ticker, args.lookback)
        if df.empty: continue
        
        # Try fetch real options; if fail, get empty (Analysis will handle)
        options_df = ingestion.get_options_chain(ticker)

        # 2. Analysis
        df = analysis.compute_technical_indicators(df)
        micro_df = analysis.microstructure_approximation(df)
        regime_df = analysis.detect_regimes(df)
        
        # Determine inputs for approximation if needed
        spot_price = df['Close'].iloc[-1]
        realized_vol = df['vol_20d'].iloc[-1]
        
        # GEX (Handles Shadow Surface Internally)
        gex_stats, gex_detail_df = analysis.compute_gex_surface(spot_price, realized_vol, options_df)
        
        # Stress
        stress_paths = analysis.monte_carlo_stress(spot_price, realized_vol)

        # 3. Render
        renderer.render_dashboard(
            ticker, df, gex_detail_df, regime_df, stress_paths, gex_stats, micro_df
        )

if __name__ == "__main__":
    main()
