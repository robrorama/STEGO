# SCRIPTNAME: ok.03.d.merged.gamma.flow.alignment.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = 'SPY'
PERIOD = '5d'       
INTERVAL = '1m'     
RISK_FREE_RATE = 0.045
    
# ==========================================
# 1. QUANTITATIVE ENGINE (BSM & Greek Reconstruction)
# ==========================================

class DerivativesEngine:
    """
    Handles the physics of option pricing and Greek calculation.
    """
    @staticmethod
    def black_scholes_gamma(S, K, T, r, sigma):
        if T <= 0.001: T = 0.001 
        if sigma <= 0.01: sigma = 0.01
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    @staticmethod
    def reconstruct_book_from_microstructure(spot_price, history_df):
        returns = history_df['Close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252 * 390) 
        
        strikes = np.linspace(spot_price * 0.97, spot_price * 1.03, 40)
        T = 1.0 / 365.0 
        
        book = []
        for K in strikes:
            moneyness = np.log(K / spot_price)
            skew_impact = -0.5 * moneyness 
            sigma = max(0.05, realized_vol + skew_impact)
            
            dist_prob = norm.pdf(K, spot_price, spot_price*0.015)
            open_int = dist_prob * 10000 
            
            book.append({
                'strike': K, 'T': T, 'r': RISK_FREE_RATE, 'sigma': sigma,
                'oi': open_int, 'type': 'call' if K > spot_price else 'put' 
            })
            
        return pd.DataFrame(book)

# ==========================================
# 2. DATA PIPELINE
# ==========================================

class HedgeFundPipeline:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = None
        self.book = None

    def fetch_data(self):
        print(f"--- [1/4] Fetching Microstructure Data for {self.ticker} ---")
        try:
            raw = yf.download(self.ticker, period=PERIOD, interval=INTERVAL, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna()
            
            last_date = raw.index[-1].date()
            self.today_df = raw[raw.index.date == last_date].copy()
            self.history_df = raw
            
            if self.today_df.empty: raise ValueError("Market data empty.")
        except Exception as e:
            print(f"Error: {e}")
            return False
        return True

    def calculate_greeks_and_flow(self):
        print("--- [2/4] Reconstructing Dealer Books & Calculating Dynamic Gamma ---")
        open_price = self.today_df['Open'].iloc[0]
        self.book = DerivativesEngine.reconstruct_book_from_microstructure(open_price, self.history_df)
        
        gex_values = []
        strikes = self.book['strike'].values
        Ts = self.book['T'].values
        rs = self.book['r'].values
        sigmas = self.book['sigma'].values
        ois = self.book['oi'].values
        
        for current_spot in self.today_df['Close'].values:
            d1 = (np.log(current_spot / strikes) + (rs + 0.5 * sigmas ** 2) * Ts) / (sigmas * np.sqrt(Ts))
            gammas = norm.pdf(d1) / (current_spot * sigmas * np.sqrt(Ts))
            net_gex = np.sum(gammas * ois * 100 * current_spot * -1)
            gex_values.append(net_gex)
            
        self.today_df['GEX'] = gex_values
        
        print("--- [3/4] Aligning Spot Drift vs. Hedge Flow ---")
        self.today_df['dS'] = self.today_df['Close'].diff()
        self.today_df['HedgeFlow'] = self.today_df['GEX'] * self.today_df['dS'] * -1
        
        hf_diff = self.today_df['HedgeFlow'].diff()
        ds_diff = self.today_df['dS']
        
        hf_clean = hf_diff.clip(hf_diff.quantile(0.01), hf_diff.quantile(0.99))
        ds_clean = ds_diff.clip(ds_diff.quantile(0.01), ds_diff.quantile(0.99))
        
        # Rolling Correlation
        self.today_df['Rho'] = ds_clean.rolling(15).corr(hf_clean)
        self.today_df.dropna(inplace=True)

# ==========================================
# 3. VISUALIZATION (FIXED)
# ==========================================

def render_dashboard(pipeline):
    print("--- [4/4] Generating Interactive Dashboard ---")
    df = pipeline.today_df
    ticker = pipeline.ticker
    
    # Create Subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} Institutional Price Action", "Flow Alignment (Correlation)", "Net Dealer Gamma (GEX)")
    )

    # --- TRACE 1: SPOT PRICE (LINE) ---
    # Replaced Candlestick with Scatter Line for better visibility
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines', line=dict(color='white', width=1.5),
        name='Spot Price', visible=True
    ), row=1, col=1)

    # --- TRACE 2: MOMENTUM SIGNALS ---
    squeeze_mask = (df['GEX'] < 0) & (df['Rho'] > 0.4)
    squeeze_df = df[squeeze_mask]
    
    fig.add_trace(go.Scatter(
        x=squeeze_df.index, y=squeeze_df['Close'],
        mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ffcc', line=dict(width=1, color='black')),
        name='Momentum Accel', visible=True
    ), row=1, col=1)

    # --- TRACE 3: MEAN REVERSION SIGNALS ---
    damp_mask = (df['GEX'] > 0) & (df['Rho'] < -0.2)
    damp_df = df[damp_mask]
    
    fig.add_trace(go.Scatter(
        x=damp_df.index, y=damp_df['Close'],
        mode='markers', marker=dict(symbol='circle', size=8, color='#ff00ff', opacity=0.6),
        name='Vol Pinning', visible=True
    ), row=1, col=1)

    # --- TRACE 4: CORRELATION (FILLED AREA) ---
    # Gradient fill logic manually done by splitting traces or simple fill
    # Using simple fill to 0 for clarity
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Rho'],
        mode='lines', line=dict(color='#00d4ff', width=1),
        fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)',
        name='Flow Correlation', visible=True
    ), row=2, col=1)
    
    # Add Red Line for Threshold
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=0.4, y1=0.4, 
                  line=dict(color="red", width=1, dash="dot"), row=2, col=1)

    # --- TRACE 5: GAMMA EXPOSURE (BARS) ---
    colors = np.where(df['GEX'] < 0, '#ff3b30', '#4cd964') 
    fig.add_trace(go.Bar(
        x=df.index, y=df['GEX'],
        marker_color=colors,
        name='Net GEX', visible=True
    ), row=3, col=1)

    # ==========================================
    # LAYOUT & INTERACTIVITY
    # ==========================================
    
    fig.update_layout(
        template="plotly_dark",
        height=900,
        title_text=f"<b>QUANTITATIVE DEALER FLOW: {ticker}</b>",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_rangeslider_visible=False, # DISABLING THIS FIXES LAYOUT GLITCHES
        
        # TABS LOGIC
        # We have 5 Traces: [Price, MomSignal, RevSignal, Corr, GEX]
        updatemenus=[
            dict(
                type="buttons", direction="right", active=0, x=0.5, y=1.02, xanchor='center', yanchor='bottom',
                buttons=list([
                    dict(label="FULL DASHBOARD",
                         method="update",
                         args=[{"visible": [True, True, True, True, True]}]),
                    dict(label="PRICE FOCUS",
                         method="update",
                         args=[{"visible": [True, True, True, False, False]}]),
                    dict(label="STRUCTURE FOCUS",
                         method="update",
                         args=[{"visible": [False, False, False, True, True]}])
                ]),
            )
        ]
    )
    
    # Y-Axis formatting
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Corr", range=[-1, 1], row=2, col=1)
    fig.update_yaxes(title_text="GEX ($)", row=3, col=1)

    fig.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    strategy = HedgeFundPipeline(TICKER)
    if strategy.fetch_data():
        strategy.calculate_greeks_and_flow()
        render_dashboard(strategy)
        print("\nNote: Charts updated to Line Format for visibility.")
