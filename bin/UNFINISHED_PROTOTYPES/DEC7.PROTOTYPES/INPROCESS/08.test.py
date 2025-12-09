# SCRIPTNAME: 08.test.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import time
import json
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from colorama import Fore, Style, init

# Initialize Colorama for CLI
init(autoreset=True)
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # --- File System ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    DASHBOARD_PATH = os.path.join(BASE_DIR, 'dashboard.html')

    # --- Tickers ---
    UNDERLYING_TICKER = "SPY"  # Change to SPX, QQQ, IWM as needed
    VOL_TICKERS = {
        "VIX": "^VIX",
        "VIX9D": "^VIX9D", 
        "VIX3M": "^VIX3M",
        "VVIX": "^VVIX"
    }

    # --- Quant Constants ---
    RISK_FREE_RATE = 0.045
    TRADING_DAYS = 252
    
    # --- Operational ---
    REFRESH_RATE_SECONDS = 30
    MAX_EXPIRIES_TO_FETCH = 6  # Limit to front months for speed
    
    # --- Alerts ---
    GEX_Z_SCORE_THRESHOLD = 2.0
    GAMMA_FLIP_PROXIMITY_PCT = 0.005 # 0.5% proximity warning

# ==========================================
# 2. DATA INGESTION ENGINE (The Fortress)
# ==========================================
class DataIngestion:
    def __init__(self):
        self._ensure_directories()
        self._setup_logging()

    def _ensure_directories(self):
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)

    def _setup_logging(self):
        logging.basicConfig(
            filename=os.path.join(Config.LOG_DIR, 'system.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def save_snapshot(self, data_dict, filename_prefix="snapshot"):
        """Disk-First Pattern: Save raw data before processing."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        path = os.path.join(Config.DATA_DIR, filename)
        
        serializable_data = {}
        for k, v in data_dict.items():
            if isinstance(v, pd.DataFrame):
                serializable_data[k] = v.to_json(date_format='iso', orient='split')
            else:
                serializable_data[k] = v
                
        with open(path, 'w') as f:
            json.dump(serializable_data, f)
        
        self._cleanup_old_files()

    def _cleanup_old_files(self):
        """Keep only the last 5 snapshots to prevent disk bloat."""
        files = sorted([os.path.join(Config.DATA_DIR, f) for f in os.listdir(Config.DATA_DIR) if f.endswith('.json')])
        if len(files) > 5:
            for f in files[:-5]:
                try:
                    os.remove(f)
                except OSError:
                    pass

    def fetch_market_data(self):
        print(f"{Fore.CYAN}[INGEST] Fetching Market Data for {Config.UNDERLYING_TICKER}...")
        try:
            # 1. Underlying Price (1m candles for microstructure)
            spy = yf.Ticker(Config.UNDERLYING_TICKER)
            hist = spy.history(period="1d", interval="1m")
            current_price = hist['Close'].iloc[-1] if not hist.empty else 0
            
            # 2. Volatility Surface
            vol_data = {}
            for name, ticker in Config.VOL_TICKERS.items():
                try:
                    v = yf.Ticker(ticker)
                    v_hist = v.history(period="5d")
                    vol_data[name] = v_hist['Close'].iloc[-1] if not v_hist.empty else 0
                except:
                    vol_data[name] = 0

            return {
                "price_history": hist,
                "current_price": current_price,
                "vol_surface": vol_data
            }
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            print(f"{Fore.RED}[ERROR] Market Data Failed: {e}")
            return None

    def fetch_options_chain(self):
        print(f"{Fore.CYAN}[INGEST] Fetching Options Chain (Depth: {Config.MAX_EXPIRIES_TO_FETCH} expiries)...")
        try:
            tk = yf.Ticker(Config.UNDERLYING_TICKER)
            expiries = tk.options
            
            if not expiries:
                print(f"{Fore.RED}[ERROR] No expiries found.")
                return pd.DataFrame()

            target_expiries = expiries[:Config.MAX_EXPIRIES_TO_FETCH]
            
            calls_list = []
            puts_list = []
            
            for expr in target_expiries:
                try:
                    opt = tk.option_chain(expr)
                    c = opt.calls
                    c['expiry'] = expr
                    c['type'] = 'call'
                    calls_list.append(c)
                    
                    p = opt.puts
                    p['expiry'] = expr
                    p['type'] = 'put'
                    puts_list.append(p)
                except Exception as e:
                    continue

            if not calls_list:
                return pd.DataFrame()

            full_chain = pd.concat(calls_list + puts_list, ignore_index=True)
            
            # Sanitization
            cols_to_numeric = ['strike', 'lastPrice', 'bid', 'ask', 'openInterest', 'volume', 'impliedVolatility']
            for col in cols_to_numeric:
                if col in full_chain.columns:
                    full_chain[col] = pd.to_numeric(full_chain[col], errors='coerce').fillna(0)
            
            # Approximation Logic: Use Volume if OI is lagged/zero
            full_chain['positioning_proxy'] = np.where(
                full_chain['openInterest'] > 0, 
                full_chain['openInterest'], 
                full_chain['volume']
            )
            
            return full_chain

        except Exception as e:
            logging.error(f"Error fetching options chain: {e}")
            print(f"{Fore.RED}[ERROR] Chain Fetch Failed: {e}")
            return pd.DataFrame()

# ==========================================
# 3. ANALYTICS ENGINE (The Math)
# ==========================================
class AnalyticsEngine:
    
    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, opt_type):
        """Vectorized Black-Scholes Greeks."""
        T = np.maximum(T, 1e-5) # Avoid div by zero
        sigma = np.maximum(sigma, 1e-5)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        # Gamma (Same for Call and Put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return gamma

    @staticmethod
    def calculate_gex_profile(df, current_price):
        if df.empty or current_price == 0:
            return None

        # Time to expiry in years
        today = datetime.datetime.now()
        df['dte_days'] = (pd.to_datetime(df['expiry']) - today).dt.days
        df['T'] = df['dte_days'] / 252.0
        
        # Filter expired and valid IV
        df = df[(df['T'] > 0) & (df['impliedVolatility'] > 0)].copy()

        # Vectorized Calculation
        S = current_price
        K = df['strike'].values
        T = df['T'].values
        r = Config.RISK_FREE_RATE
        sigma = df['impliedVolatility'].values
        
        # Calculate Gamma
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # GEX = Gamma * OI (or proxy) * 100 * Spot
        df['gex_notional'] = gamma * df['positioning_proxy'] * 100 * S
        
        # Direction: Dealers Long Calls (+), Short Puts (-)
        df['net_gex'] = np.where(df['type'] == 'call', df['gex_notional'], -df['gex_notional'])

        # Aggregate by Strike
        gex_by_strike = df.groupby('strike')['net_gex'].sum().sort_index()
        
        return {
            'full_chain': df,
            'gex_by_strike': gex_by_strike,
            'total_gex': df['net_gex'].sum(),
            'zero_flip': AnalyticsEngine.find_flip_level(gex_by_strike, current_price)
        }

    @staticmethod
    def find_flip_level(gex_series, current_price):
        """Finds the price where Net GEX flips from positive to negative."""
        try:
            signs = np.sign(gex_series).diff()
            flips = signs[signs != 0].dropna()
            
            if flips.empty:
                return 0
                
            flip_strikes = flips.index
            # Find flip closest to current price
            closest_flip = flip_strikes[np.argmin(np.abs(flip_strikes - current_price))]
            return closest_flip
        except:
            return 0

    @staticmethod
    def analyze_microstructure(hist_df):
        if hist_df is None or hist_df.empty:
            return {'realized_vol': 0, 'buy_pressure': 0}
            
        returns = hist_df['Close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(390) * 100 # Annualized intraday %
        
        # Pressure Approximation
        price_change = hist_df['Close'].diff()
        pressure_num = (hist_df['Close'] * hist_df['Volume'] * np.sign(price_change)).sum()
        pressure_denom = hist_df['Volume'].sum()
        buy_pressure = pressure_num / pressure_denom if pressure_denom != 0 else 0
        
        return {
            'realized_vol': realized_vol,
            'buy_pressure': buy_pressure
        }
    
    @staticmethod
    def check_alerts(metrics):
        alerts = []
        if not metrics: return alerts
        
        curr_price = metrics['market_data']['current_price']
        gex_data = metrics.get('gex_data')
        
        if gex_data:
            flip_level = gex_data['zero_flip']
            if flip_level > 0:
                dist_pct = abs(curr_price - flip_level) / curr_price
                if dist_pct < Config.GAMMA_FLIP_PROXIMITY_PCT:
                    alerts.append(f"CRITICAL: Price near Gamma Flip ({flip_level:.2f})")

        vol = metrics['market_data']['vol_surface']
        if vol.get('VIX', 0) > vol.get('VIX3M', 0) and vol.get('VIX', 0) > 0:
            alerts.append("REGIME: Term Structure Inversion (Backwardation)")
            
        return alerts

# ==========================================
# 4. DASHBOARD RENDERER (The Artist)
# ==========================================
class DashboardRenderer:
    @staticmethod
    def render(metrics, alerts):
        """Generates the Offline HTML Dashboard with JS Fixes."""
        
        gex_data = metrics.get('gex_data')
        mkt_data = metrics.get('market_data')
        micro = metrics.get('microstructure')
        
        if not gex_data or not mkt_data:
            return

        # Prepare Data
        gex_series = gex_data['gex_by_strike']
        # Filter strikes near money for cleaner chart
        curr_price = mkt_data['current_price']
        lower_bound = curr_price * 0.85
        upper_bound = curr_price * 1.15
        gex_series = gex_series[(gex_series.index >= lower_bound) & (gex_series.index <= upper_bound)]

        strikes = gex_series.index
        gex_vals = gex_series.values / 1e9 # Billions
        
        # Colors: Green for Long Gamma, Red for Short Gamma
        colors = ['#00ff00' if x >= 0 else '#ff0000' for x in gex_vals]

        # --- Create Figure ---
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.3, 0.7],
            row_heights=[0.3, 0.7],
            specs=[[{"type": "indicator"}, {"type": "xy"}],
                   [{"colspan": 2, "type": "xy"}, None]],
            subplot_titles=("Total Dealer GEX", "Intraday Realized Vol", "Gamma Exposure Profile ($Bn)")
        )

        # 1. Gauge: Total GEX
        total_gex_bn = gex_data['total_gex'] / 1e9
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=total_gex_bn,
            title={'text': "Net GEX ($Bn)"},
            gauge={
                'axis': {'range': [-5, 5]},
                'bar': {'color': "#00ff00" if total_gex_bn > 0 else "#ff0000"},
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 0}
            }
        ), row=1, col=1)

        # 2. Scatter: Microstructure
        # Placeholder for intraday trend
        hist = mkt_data['price_history']
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Close'],
                mode='lines', name='Price',
                line=dict(color='cyan')
            ), row=1, col=2)

        # 3. Bar: GEX Profile
        fig.add_trace(go.Bar(
            x=strikes, y=gex_vals,
            marker_color=colors,
            name='GEX'
        ), row=2, col=1)

        # Add Vertical Line for Current Price
        fig.add_vline(x=curr_price, line_width=2, line_dash="dash", line_color="yellow", annotation_text="Spot", row=2, col=1)
        
        # Add Vertical Line for Flip Level
        flip = gex_data['zero_flip']
        if flip > 0 and lower_bound < flip < upper_bound:
            fig.add_vline(x=flip, line_width=2, line_dash="dot", line_color="magenta", annotation_text="Flip", row=2, col=1)

        # --- Layout Styling ---
        fig.update_layout(
            template="plotly_dark",
            title_text=f"HF Dealer Dashboard | {Config.UNDERLYING_TICKER} | {datetime.datetime.now().strftime('%H:%M:%S')}",
            height=900,
            showlegend=False
        )

        # Alerts Annotation
        alert_text = "<br>".join(alerts) if alerts else "SYSTEM NORMAL"
        alert_color = "red" if alerts else "#00ff00"
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.98,
            text=f"<b>ALERTS:</b><br>{alert_text}",
            showarrow=False,
            font=dict(color=alert_color, size=14),
            align="left",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=alert_color
        )

        # Save HTML
        print(f"{Fore.YELLOW}[RENDER] Updating Dashboard: {Config.DASHBOARD_PATH}")
        fig.write_html(
            Config.DASHBOARD_PATH,
            include_plotlyjs='cdn',
            config={'responsive': True, 'displayModeBar': False}
        )

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    print(f"{Fore.GREEN}=== HEDGE FUND GEX DASHBOARD INITIALIZED ===")
    print(f"{Fore.GREEN}Tracking: {Config.UNDERLYING_TICKER}")
    print(f"{Fore.GREEN}Mode: Standalone Real-Time Loop")
    print("-" * 50)

    ingestion = DataIngestion()
    analytics = AnalyticsEngine()
    renderer = DashboardRenderer()

    while True:
        try:
            start_time = time.time()
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # 1. Fetch
            mkt_data = ingestion.fetch_market_data()
            if not mkt_data:
                print(f"{Fore.RED}Retrying in 10s...")
                time.sleep(10)
                continue
                
            chain_df = ingestion.fetch_options_chain()
            
            # 2. Save Raw (Disk First)
            snapshot = {
                'market': mkt_data['current_price'],
                'vol': mkt_data['vol_surface'],
                'chain_sample': chain_df.head().to_json() # Log partial for sanity
            }
            ingestion.save_snapshot(snapshot)

            # 3. Analyze
            print(f"{Fore.CYAN}[QUANT] Running GEX & Greek Models...")
            gex_results = analytics.calculate_gex_profile(chain_df, mkt_data['current_price'])
            micro_results = analytics.analyze_microstructure(mkt_data['price_history'])
            
            metrics = {
                'market_data': mkt_data,
                'gex_data': gex_results,
                'microstructure': micro_results
            }
            
            alerts = analytics.check_alerts(metrics)

            # 4. Render
            renderer.render(metrics, alerts)

            # 5. CLI Summary
            if gex_results:
                total_gex = gex_results['total_gex'] / 1e9
                flip = gex_results['zero_flip']
                print(f"\n{Fore.WHITE}--- {timestamp} SUMMARY ---")
                print(f"Price: {mkt_data['current_price']:.2f}")
                print(f"Net GEX: {Fore.GREEN if total_gex > 0 else Fore.RED}${total_gex:.2f} Bn")
                print(f"Flip Level: {Fore.MAGENTA}{flip:.2f}")
                print(f"VIX: {mkt_data['vol_surface'].get('VIX', 0):.2f}")
                if alerts:
                    print(f"{Fore.RED}ALERTS ACTIVE: {len(alerts)}")

            # Sleep Logic
            elapsed = time.time() - start_time
            sleep_time = max(0, Config.REFRESH_RATE_SECONDS - elapsed)
            print(f"{Fore.BLUE}[SYSTEM] Sleeping {sleep_time:.1f}s...\n")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}System Shutdown Requested.")
            break
        except Exception as e:
            logging.error(f"Critical Loop Error: {e}")
            print(f"{Fore.RED}[CRITICAL] Loop Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
