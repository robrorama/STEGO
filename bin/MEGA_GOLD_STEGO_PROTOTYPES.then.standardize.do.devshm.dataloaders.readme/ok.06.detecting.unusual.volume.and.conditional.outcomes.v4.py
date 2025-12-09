import os
import time
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize, brentq
from scipy.interpolate import griddata
from datetime import datetime, timedelta

# Visualization
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots
import plotly.io as pio

# Data
import yfinance as yf

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# PHASE 1: CORE ENGINEERING & ARCHITECTURE
# ==============================================================================

class DataIngestion:
    """
    Handles Disk I/O, Downloading, Caching, and Sanitization.
    Strictly disk-first pipeline.
    """
    def __init__(self, output_dir='./market_data'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """ The Universal Fixer. """
        if isinstance(df.columns, pd.MultiIndex):
            # Checklist Item 1: Robust flattening
            df.columns = ['_'.join(str(c) for c in col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
        
        rename_map = {}
        for col in df.columns:
            if 'Open' in str(col): rename_map[col] = 'Open'
            elif 'High' in str(col): rename_map[col] = 'High'
            elif 'Low' in str(col): rename_map[col] = 'Low'
            elif 'Close' in str(col): rename_map[col] = 'Close'
            elif 'Volume' in str(col): rename_map[col] = 'Volume'
        df = df.rename(columns=rename_map)

        if pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = df.index.tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)

        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df.dropna(subset=['Close'], inplace=True)
        
        # Checklist Item 4 Fix: Replace deprecated fillna(method=...) with ffill/bfill
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        return df

    def get_data(self, ticker: str, lookback_years: float, intraday: bool = False) -> pd.DataFrame:
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        
        if os.path.exists(file_path):
            logger.info(f"[{ticker}] Found on disk. Loading...")
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Fix: Handle timezone if present in CSV load
                if hasattr(df.index, 'tz') and df.index.tz is not None: 
                    df.index = df.index.tz_localize(None)
                
                start_date = df.index[0]
                required_start = datetime.now() - timedelta(days=lookback_years*365)
                if start_date > required_start:
                    logger.info(f"[{ticker}] Insufficient history. Redownloading.")
                else:
                    return df
            except Exception as e:
                logger.warning(f"[{ticker}] Corrupt file. Redownloading. Error: {e}")

        logger.info(f"[{ticker}] Downloading via yfinance...")
        time.sleep(1.0)
        
        interval = "1h" if intraday else "1d"
        period = "2y" if intraday else f"{int(lookback_years)}y"
        if not intraday and lookback_years > 2: period = "max"

        raw_df = yf.download(ticker, period=period, interval=interval, progress=False, group_by='column')
        
        if raw_df.empty:
            return pd.DataFrame()

        clean_df = self._sanitize_df(raw_df, ticker)
        clean_df.to_csv(file_path)
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    def get_options_chain(self, ticker: str, num_expirations=3):
        """
        Fetches option chains for the nearest `num_expirations` to build surfaces.
        """
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            if not exps:
                return None
            
            all_chains = []
            # Fetch top N expirations
            for e in exps[:num_expirations]:
                try:
                    chain = tk.option_chain(e)
                    calls = chain.calls
                    puts = chain.puts
                    calls['type'] = 'call'
                    puts['type'] = 'put'
                    df = pd.concat([calls, puts])
                    df['expiration'] = e
                    all_chains.append(df)
                    time.sleep(0.5) # Rate limit nice-ness
                except Exception as e_inner:
                    logger.warning(f"Failed to fetch expiry {e}: {e_inner}")
            
            if not all_chains: return None
            
            full_df = pd.concat(all_chains)
            
            # Add spot price
            hist = tk.history(period="1d")
            if not hist.empty:
                full_df['spot'] = hist['Close'].iloc[-1]
            else:
                full_df['spot'] = 0.0
                
            return full_df
        except Exception as e:
            logger.warning(f"Could not fetch options for {ticker}: {e}")
            return None


class FinancialAnalysis:
    """
    Quantitative Logic, Models, Greeks, ML, Surfaces.
    """
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    # ------------------------------------------------------------------
    # A) OPTIONS & GREEKS MATH
    # ------------------------------------------------------------------
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """ Vectorized BSM Greeks """
        T = np.maximum(T, 1e-5) 
        sigma = np.maximum(sigma, 1e-3)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        
        if option_type == 'call':
            delta = cdf_d1
            charm = -pdf_d1 * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
        else:
            delta = cdf_d1 - 1
            charm = -pdf_d1 * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
            
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * pdf_d1 * np.sqrt(T) * 0.01
        vanna = -pdf_d1 * d2 / sigma
        
        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'vanna': vanna, 'charm': charm}

    def analyze_dealer_positioning(self, chain_df):
        """
        Estimates GEX, Vanna Exposure, and 'Gamma Flip' levels.
        """
        if chain_df is None or chain_df.empty:
            return pd.DataFrame(), pd.DataFrame(), 0.0

        S = chain_df['spot'].iloc[0]
        
        # Calculate DTE
        chain_df['dte'] = (pd.to_datetime(chain_df['expiration']) - datetime.now()).dt.days
        chain_df['T'] = chain_df['dte'] / 365.0
        chain_df['T'] = chain_df['T'].clip(lower=0.001)
        
        # Calculate Greeks
        deltas, gammas, vannas = [], [], []
        
        # Iterating for safety with mixed types, though vectorization preferred
        for idx, row in chain_df.iterrows():
            g = self.calculate_greeks(S, row['strike'], row['T'], self.r, row['impliedVolatility'], row['type'])
            deltas.append(g['delta'])
            gammas.append(g['gamma'])
            vannas.append(g['vanna'])
            
        chain_df['delta'] = deltas
        chain_df['gamma'] = gammas
        chain_df['vanna'] = vannas
        
        # GEX Calculation
        chain_df['GEX'] = chain_df['gamma'] * chain_df['openInterest'] * (S**2) * 0.01
        chain_df['signed_GEX'] = np.where(chain_df['type'] == 'call', chain_df['GEX'], -chain_df['GEX'])

        # Aggregate by Strike
        exposure_profile = chain_df.groupby('strike')[['signed_GEX']].sum()
        
        # Separate Call/Put contribution for "Professional" view
        call_gex = chain_df[chain_df['type']=='call'].groupby('strike')['GEX'].sum()
        put_gex = chain_df[chain_df['type']=='put'].groupby('strike')['GEX'].sum() * -1
        exposure_profile['call_gex'] = call_gex
        exposure_profile['put_gex'] = put_gex
        exposure_profile = exposure_profile.fillna(0)

        # Estimate Gamma Flip (Zero GEX Level)
        try:
            sorted_exp = exposure_profile.sort_index()
            signs = np.sign(sorted_exp['signed_GEX']).diff()
            flip_candidates = sorted_exp[signs != 0].index
            if len(flip_candidates) > 0:
                gamma_flip = flip_candidates[np.abs(flip_candidates - S).argmin()]
            else:
                gamma_flip = S
        except:
            gamma_flip = S
            
        return chain_df, exposure_profile, gamma_flip

    # ------------------------------------------------------------------
    # B) PRICE ACTION & VOL REGIMES
    # ------------------------------------------------------------------
    def calculate_volatility_metrics(self, df):
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility Cones Windows
        windows = [5, 10, 20, 60]
        for w in windows:
            df[f'rv_{w}'] = df['log_ret'].rolling(window=w).std() * np.sqrt(252)
            
        # Vol of Vol
        df['vol_of_vol'] = df['rv_20'].rolling(window=20).std()
        return df

    def detect_regimes_hmm(self, df):
        data = df[['log_ret', 'rv_20']].dropna()
        if len(data) < 50: return df
        
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
        
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(X)
        states = gmm.predict(X)
        
        # Order states by vol
        vol_means = []
        for i in range(3):
            vol_means.append(data.iloc[states == i]['rv_20'].mean())
        mapping = {old: new for new, old in enumerate(np.argsort(vol_means))}
        df.loc[data.index, 'regime'] = [mapping[s] for s in states]
        
        return df

    # ------------------------------------------------------------------
    # D) MICROSTRUCTURE (TAPE READING)
    # ------------------------------------------------------------------
    def calc_microstructure(self, df):
        # Order Imbalance
        df['range'] = df['High'] - df['Low']
        df['range'] = df['range'].replace(0, 0.01)
        df['imb_proxy'] = ((df['Close'] - df['Open']) / df['range']) * df['Volume']
        
        # CVD Proxy (Cumulative Volume Delta)
        # Refine: Use position within candle
        buy_p = ((df['Close'] - df['Low']) / df['range']) * df['Volume']
        sell_p = ((df['High'] - df['Close']) / df['range']) * df['Volume']
        df['net_flow'] = buy_p - sell_p
        df['CVD'] = df['net_flow'].cumsum()
        
        # Unusual Vol
        df['vol_ma'] = df['Volume'].rolling(20).mean()
        df['vol_std'] = df['Volume'].rolling(20).std()
        df['vol_z'] = (df['Volume'] - df['vol_ma']) / df['vol_std']
        
        # Entropy
        df['prob_break'] = df['log_ret'].rolling(20).apply(lambda x: stats.entropy(np.histogram(x, bins=5)[0]))
        return df

    # ------------------------------------------------------------------
    # E) FORECASTING & SCENARIOS
    # ------------------------------------------------------------------
    def run_lstm_forecast(self, df):
        try:
            data = df[['log_ret', 'rv_20', 'Volume']].dropna()
            if len(data) < 100: return np.zeros(10)
            
            for lag in [1, 2, 3, 5]:
                data[f'lag_{lag}'] = data['log_ret'].shift(lag)
            data = data.dropna()
            
            X = data.drop('log_ret', axis=1)
            y = data['log_ret']
            
            model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42)
            model.fit(X, y)
            
            # Recursive Forecast
            last_row = X.iloc[-1].values.reshape(1, -1)
            preds = []
            curr = last_row
            for _ in range(5):
                p = model.predict(curr)[0]
                preds.append(p)
                curr = np.roll(curr, 1)
                curr[0, 0] = p 
            return np.array(preds)
        except:
            return np.zeros(5)

    def monte_carlo_paths(self, df, n_sims=500, horizon=20):
        last_price = df['Close'].iloc[-1]
        vol = df['rv_20'].iloc[-1] / np.sqrt(252)
        
        paths = []
        for _ in range(n_sims):
            daily_returns = np.random.normal(0, vol, horizon)
            price_path = [last_price]
            for r in daily_returns:
                price_path.append(price_path[-1] * (1 + r))
            paths.append(price_path)
        return np.array(paths)

# ==============================================================================
# PHASE 3: VISUALIZATION (PROFESSIONAL GRADE)
# ==============================================================================

class DashboardRenderer:
    def __init__(self, output_file='hedge_fund_dashboard.html'):
        self.output_file = output_file
        self.figures = {}
        
    def add_section(self, title, fig):
        self.figures[title] = fig

    def generate_html(self):
        # Checklist Item 2 Fix: Inject offline Plotly JS instead of CDN link
        plotly_js = py_offline.get_plotlyjs()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quant System Terminal</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: 'Roboto Mono', monospace; background-color: #0e0e0e; color: #c0c0c0; margin: 0; }}
                .tab {{ overflow: hidden; border-bottom: 2px solid #333; background-color: #1a1a1a; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #666; font-weight: bold; font-size: 14px;}}
                .tab button:hover {{ background-color: #333; color: #fff; }}
                .tab button.active {{ background-color: #252525; color: #00e676; border-bottom: 3px solid #00e676; }}
                .tabcontent {{ display: none; padding: 10px; height: 95vh; animation: fadeEffect 0.5s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <div class="tab">
        """
        for i, title in enumerate(self.figures.keys()):
            active = "active" if i == 0 else ""
            html_content += f'<button class="tablinks {active}" onclick="openTab(event, \'{title}\')">{title}</button>\n'
        html_content += "</div>\n"
        
        for i, (title, fig) in enumerate(self.figures.items()):
            display = "block" if i == 0 else "none"
            plot_div = py_offline.plot(fig, output_type='div', include_plotlyjs=False)
            html_content += f'<div id="{title}" class="tabcontent" style="display:{display};">{plot_div}</div>\n'
            
        html_content += """
            <script>
            function openTab(evt, cityName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) tabcontent[i].style.display = "none";
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) tablinks[i].className = tablinks[i].className.replace(" active", "");
                document.getElementById(cityName).style.display = "block";
                evt.currentTarget.className += " active";
                window.dispatchEvent(new Event('resize'));
            }
            </script>
        </body></html>
        """
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Dashboard saved to {self.output_file}")


def create_professional_dashboard(df, chain_df, exposure_df, gamma_flip, mc_paths, preds, ticker):
    """
    Creates a high-density, professional hedge fund dashboard.
    """
    
    # Define Grid
    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.4, 0.3, 0.3],
        specs=[
            [{"secondary_y": True}, {"type": "surface"}],
            [{"secondary_y": True}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "domain"}]
        ],
        subplot_titles=(
            f"{ticker} Price Action & Tape (CVD)", "Implied Volatility Surface (Term Structure)",
            f"Dealer GEX Profile (Flip: ${gamma_flip:.2f})", "Volatility Cone (Realized)",
            "Monte Carlo Scenario Fan", "Market Fragility (Entropy)"
        ),
        vertical_spacing=0.08, horizontal_spacing=0.03
    )

    # -----------------------------------------------------
    # 1. Price Action (Candles) + CVD
    # -----------------------------------------------------
    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC', increasing_line_color='#00e676', decreasing_line_color='#ff1744'
    ), row=1, col=1)

    # CVD Line (Secondary Axis)
    if 'CVD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['CVD'], name='CVD (Tape)',
            line=dict(color='cyan', width=1), opacity=0.5
        ), row=1, col=1, secondary_y=True)

    # -----------------------------------------------------
    # 2. 3D Volatility Surface
    # -----------------------------------------------------
    if chain_df is not None and not chain_df.empty:
        # Prepare grid
        # X: Strike, Y: Days to Expiry, Z: IV
        X = chain_df['strike']
        Y = chain_df['dte']
        Z = chain_df['impliedVolatility']
        
        # Griddata interpolation for smooth surface
        xi = np.linspace(X.min(), X.max(), 50)
        yi = np.linspace(Y.min(), Y.max(), 50)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((X, Y), Z, (XI, YI), method='linear')
        
        fig.add_trace(go.Surface(
            x=XI, y=YI, z=ZI, colorscale='Viridis', 
            name='Vol Surface', showscale=False, opacity=0.9
        ), row=1, col=2)
    else:
        # Fix: Use Scatter3d for 3D subplot fallback
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], 
            text=["No Options Data<br>(Enable --dealer-greeks)"], 
            mode="text", showlegend=False
        ), row=1, col=2)

    # -----------------------------------------------------
    # 3. GEX Profile (Stacked)
    # -----------------------------------------------------
    if not exposure_df.empty:
        spot = chain_df['spot'].iloc[0]
        # Filter range
        mask = (exposure_df.index > spot * 0.85) & (exposure_df.index < spot * 1.15)
        sub_exp = exposure_df[mask]
        
        # Put GEX (Negative usually)
        fig.add_trace(go.Bar(
            x=sub_exp.index, y=sub_exp['put_gex'],
            name='Put GEX', marker_color='#ff1744'
        ), row=2, col=1)
        
        # Call GEX
        fig.add_trace(go.Bar(
            x=sub_exp.index, y=sub_exp['call_gex'],
            name='Call GEX', marker_color='#00e676'
        ), row=2, col=1)
        
        # Net GEX Line
        fig.add_trace(go.Scatter(
            x=sub_exp.index, y=sub_exp['signed_GEX'],
            name='Net GEX', line=dict(color='white', width=2, dash='dot')
        ), row=2, col=1)
        
        # Gamma Flip Annotation - FIXED: Replaced add_vline with add_shape to check data bounds
        # We must calculate the Y range manually because add_shape doesn't auto-scale to axes in mixed subplots well
        y_min = min(sub_exp['put_gex'].min(), sub_exp['call_gex'].min(), sub_exp['signed_GEX'].min())
        y_max = max(sub_exp['put_gex'].max(), sub_exp['call_gex'].max(), sub_exp['signed_GEX'].max())

        fig.add_shape(
            type="line", x0=gamma_flip, x1=gamma_flip, y0=y_min, y1=y_max,
            line=dict(color="yellow", width=1, dash="dash"),
            row=2, col=1
        )
        fig.add_shape(
            type="line", x0=spot, x1=spot, y0=y_min, y1=y_max,
            line=dict(color="white", width=1),
            row=2, col=1
        )

    # -----------------------------------------------------
    # 4. Volatility Cone
    # -----------------------------------------------------
    # Calculate stats for the windows
    windows = [5, 10, 20, 60]
    max_vols = [df[f'rv_{w}'].max() for w in windows]
    min_vols = [df[f'rv_{w}'].min() for w in windows]
    med_vols = [df[f'rv_{w}'].median() for w in windows]
    cur_vols = [df[f'rv_{w}'].iloc[-1] for w in windows]
    
    fig.add_trace(go.Scatter(x=windows, y=max_vols, mode='lines+markers', name='Max Vol', line=dict(color='#ff5252')), row=2, col=2)
    fig.add_trace(go.Scatter(x=windows, y=med_vols, mode='lines+markers', name='Median Vol', line=dict(color='#bdbdbd')), row=2, col=2)
    fig.add_trace(go.Scatter(x=windows, y=min_vols, mode='lines+markers', name='Min Vol', line=dict(color='#4caf50')), row=2, col=2)
    fig.add_trace(go.Scatter(x=windows, y=cur_vols, mode='lines+markers', name='Current Vol', line=dict(color='cyan', width=3)), row=2, col=2)

    # -----------------------------------------------------
    # 5. Monte Carlo Fan
    # -----------------------------------------------------
    if len(mc_paths) > 0:
        # Calculate percentiles
        p05 = np.percentile(mc_paths, 5, axis=0)
        p25 = np.percentile(mc_paths, 25, axis=0)
        p50 = np.percentile(mc_paths, 50, axis=0)
        p75 = np.percentile(mc_paths, 75, axis=0)
        p95 = np.percentile(mc_paths, 95, axis=0)
        x_axis = np.arange(len(p50))
        
        # Fan Area (using fill='tonexty')
        # 5-95
        fig.add_trace(go.Scatter(
            x=x_axis, y=p95, mode='lines', line=dict(width=0), showlegend=False
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=p05, mode='lines', fill='tonexty', fillcolor='rgba(100, 100, 100, 0.2)',
            line=dict(width=0), name='5-95% CI'
        ), row=3, col=1)
        
        # 25-75
        fig.add_trace(go.Scatter(
            x=x_axis, y=p75, mode='lines', line=dict(width=0), showlegend=False
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=p25, mode='lines', fill='tonexty', fillcolor='rgba(0, 230, 118, 0.2)',
            line=dict(width=0), name='25-75% CI'
        ), row=3, col=1)
        
        # Median
        fig.add_trace(go.Scatter(
            x=x_axis, y=p50, mode='lines', line=dict(color='white', width=2), name='Median Path'
        ), row=3, col=1)

    # -----------------------------------------------------
    # 6. Trend Entropy Gauge
    # -----------------------------------------------------
    last_prob = df['prob_break'].iloc[-1] if 'prob_break' in df.columns else 0.5
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=last_prob,
        title={'text': "Trend Fragility"},
        gauge={
            'axis': {'range': [None, 3]}, 
            'bar': {'color': "#ff4081"},
            'steps': [
                {'range': [0, 1.5], 'color': "#1a1a1a"},
                {'range': [1.5, 2.5], 'color': "#333"}
            ]
        }
    ), row=3, col=2)

    # -----------------------------------------------------
    # Styling
    # -----------------------------------------------------
    fig.update_layout(
        height=1400, 
        template='plotly_dark', 
        paper_bgcolor='#0e0e0e',
        plot_bgcolor='#141414',
        font=dict(family="Roboto Mono", size=10, color="#b0b0b0"),
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40),
        scene=dict(
            xaxis_title='Strike', yaxis_title='DTE', zaxis_title='Implied Vol',
            xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'), zaxis=dict(gridcolor='#333')
        )
    )
    
    # Grid lines tweak
    fig.update_xaxes(gridcolor='#333', zerolinecolor='#444')
    fig.update_yaxes(gridcolor='#333', zerolinecolor='#444')
    
    return fig

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Quant System")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'])
    parser.add_argument('--lookback', type=float, default=1.0)
    parser.add_argument('--intraday', action='store_true')
    parser.add_argument('--hmm-regimes', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--monte-carlo', action='store_true')
    parser.add_argument('--dealer-greeks', action='store_true')
    
    args = parser.parse_args()
    
    ingestion = DataIngestion()
    analysis = FinancialAnalysis()
    renderer = DashboardRenderer()
    
    logger.info("Initializing Professional Quant System...")
    
    for ticker in args.tickers:
        logger.info(f"Processing {ticker}...")
        
        df = ingestion.get_data(ticker, args.lookback, args.intraday)
        if df.empty: continue
        
        df = analysis.calculate_volatility_metrics(df)
        if args.hmm_regimes: df = analysis.detect_regimes_hmm(df)
        df = analysis.calc_microstructure(df)
        
        chain_df = None
        exposure_df = pd.DataFrame()
        gamma_flip = 0.0
        
        if args.dealer_greeks:
            logger.info("Fetching Options Chain (Front 3 Expiries)...")
            chain_df = ingestion.get_options_chain(ticker, num_expirations=3)
            if chain_df is not None:
                chain_df, exposure_df, gamma_flip = analysis.analyze_dealer_positioning(chain_df)
        
        preds = []
        if args.lstm: preds = analysis.run_lstm_forecast(df)
            
        mc_paths = []
        if args.monte_carlo: mc_paths = analysis.monte_carlo_paths(df)
            
        fig = create_professional_dashboard(df, chain_df, exposure_df, gamma_flip, mc_paths, preds, ticker)
        renderer.add_section(ticker, fig)
        
    renderer.generate_html()
    logger.info("Analysis Complete. Opening Terminal.")

if __name__ == "__main__":
    main()
