import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# ==========================================
# 1. CONFIGURATION & SOPHISTICATED MATH
# ==========================================
# Default Tickers if none provided
DEFAULT_TICKERS = ['SPY', 'QQQ', 'NVDA', 'IWM'] 

# Model Parameters
RISK_FREE_RATE = 0.045
MIN_DTE = 5
MAX_DTE = 120
VOL_LOOKBACK = 30 

# "Sticky Delta" Sensitivity (Beta)
# How much IV usually expands when spot drops 1%?
# Indices ~ 0.15, Single Stocks ~ 0.05-0.10
SKEW_BETAS = {'SPY': 0.15, 'QQQ': 0.18, 'IWM': 0.16, 'DEFAULT': 0.10}

warnings.filterwarnings('ignore')

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def get_strike_by_delta(chain, target_delta, S, T, r, option_type='put'):
    """
    Inverse mapping: Find Strike K for a specific Delta using interpolation.
    """
    # Calculate delta for existing strikes
    chain = chain.copy()
    
    # Vectorized Delta Calc
    # d1 = (ln(S/K) + (r + 0.5v^2)T) / (v sqrt(T))
    # Delta Put = N(d1) - 1
    # Delta Call = N(d1)
    
    sigma = chain['impliedVolatility'].values
    K = chain['strike'].values
    
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        chain['delta'] = stats.norm.cdf(d1)
    else:
        chain['delta'] = stats.norm.cdf(d1) - 1
        
    # Sort for interpolation
    chain = chain.sort_values('delta')
    
    # Interpolate IV at target delta
    try:
        return np.interp(target_delta, chain['delta'], chain['impliedVolatility'])
    except:
        return np.nan

# ==========================================
# 2. DATA ENGINE (REAL + APPROXIMATION)
# ==========================================
class VolatilityScanner:
    def __init__(self, tickers=None):
        self.tickers = tickers if tickers else DEFAULT_TICKERS
        self.data = {}
        self.insights = []

    def fetch_and_process(self):
        """
        Main pipeline: Fetch Context -> Fetch Chain -> Approx History -> Calc Rotation
        """
        # 1. Fetch Context (Spot & Realized Vol)
        print("--- 📡 Establishing Uplink to Market Data (yfinance) ---")
        try:
            history = yf.download(self.tickers, period="40d", progress=False)['Close']
            # Handle single ticker case returning Series instead of DF
            if isinstance(history, pd.Series):
                history = history.to_frame(name=self.tickers[0])
        except Exception as e:
            print(f"CRITICAL: Failed to download price history. {e}")
            return

        for ticker in self.tickers:
            print(f"Processing {ticker}...")
            try:
                # SPOT DATA
                if ticker not in history.columns: continue
                prices = history[ticker].dropna()
                current_spot = prices.iloc[-1]
                spot_5d_ago = prices.iloc[-6] if len(prices) >= 6 else prices.iloc[0]
                spot_return = (current_spot / spot_5d_ago) - 1
                
                # REALIZED VOL (for scaling)
                log_rets = np.log(prices / prices.shift(1))
                rv_current = log_rets.tail(5).std() * np.sqrt(252)
                rv_past = log_rets.iloc[-10:-5].std() * np.sqrt(252)
                rv_ratio = rv_current / rv_past if rv_past > 0 else 1.0

                # OPTIONS CHAIN
                tk = yf.Ticker(ticker)
                exps = tk.options
                
                surface = []
                
                # Limit expirations to speed up (first 8 relevant ones)
                valid_exps = [e for e in exps if MIN_DTE < (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days < MAX_DTE][:8]
                
                for date_str in valid_exps:
                    expiry_date = datetime.strptime(date_str, '%Y-%m-%d')
                    dte = (expiry_date - datetime.now()).days
                    T = dte / 365.0
                    
                    # Fetch
                    chain = tk.option_chain(date_str)
                    puts = chain.puts
                    calls = chain.calls
                    
                    # 1. Find ATM IV (Avg of Put/Call near spot)
                    # Simple proxy: Average IV of strikes closest to spot
                    atm_strike_dist = np.abs(puts['strike'] - current_spot)
                    atm_row = puts.loc[atm_strike_dist.idxmin()]
                    atm_iv = atm_row['impliedVolatility']
                    
                    # 2. Find 25-Delta Put IV
                    iv_25d_put = get_strike_by_delta(puts, -0.25, current_spot, T, RISK_FREE_RATE, 'put')
                    
                    if np.isnan(iv_25d_put) or iv_25d_put == 0: continue
                    
                    # METRIC: Skew Premium (Downside Fear)
                    # 25D Put IV minus ATM IV
                    skew_premium = iv_25d_put - atm_iv
                    
                    # 3. BACKCASTING (Approximating T-5)
                    # Model: Skew_Past = Skew_Curr - Beta * (Spot_Return)
                    # If Spot dropped 2% (Ret = -0.02), Skew likely ROSE.
                    # So Past Skew was LOWER.
                    # Past = Curr - (-Beta * -0.02) -> Curr - (Pos) -> Lower. Correct.
                    
                    beta = SKEW_BETAS.get(ticker, SKEW_BETAS['DEFAULT'])
                    # Adjustment for Spot Move
                    delta_skew_spot = -beta * (spot_return * 100) / 100 # Scaling to vol points
                    
                    # Adjustment for Vol Regime (If vol crushed, skew likely crushed too)
                    approx_past_skew = (skew_premium - delta_skew_spot) / np.sqrt(rv_ratio)
                    
                    surface.append({
                        'dte': dte,
                        'current_skew': skew_premium * 100, # bps
                        'past_skew': approx_past_skew * 100, # bps
                        'change': (skew_premium - approx_past_skew) * 100,
                        'atm': atm_iv
                    })
                
                if not surface: continue
                
                df = pd.DataFrame(surface).sort_values('dte')
                self.data[ticker] = df
                
                # GENERATE INSIGHT
                self.analyze_rotation(ticker, df, spot_return)
                
            except Exception as e:
                print(f"Skipping {ticker}: {e}")
                continue

    def analyze_rotation(self, ticker, df, spot_ret):
        """
        Generate textual analysis based on the surface rotation.
        """
        front = df[df['dte'] < 45]['change'].mean()
        back = df[df['dte'] >= 45]['change'].mean()
        
        insight = {"ticker": ticker}
        
        # Logic Tree
        if front < -10: # Front Skew Crushing
            if spot_ret > 0:
                msg = "BULLISH STABILIZATION: Fear premium draining from front-end despite spot rally. Chase protection is being sold."
            else:
                msg = "BEARISH EXHAUSTION: Spot down but Skew flattening? Puts are being monetized. Possible bounce."
            color = "green"
        elif front > 10: # Front Skew Spiking
            if spot_ret < 0:
                msg = "PANIC ACCELERATION: Downside puts being bid aggressively. Hedging demand is high."
            else:
                msg = "SKEPTICAL RALLY: Spot up but Skew rising? Market doesn't trust this move. Collars active."
            color = "red"
        else:
            msg = "NEUTRAL / STRUCTURED: Normal term structure roll. No aggressive repositioning."
            color = "gray"
            
        insight['msg'] = msg
        insight['color'] = color
        insight['front_chg'] = front
        insight['back_chg'] = back
        self.insights.append(insight)

    # ==========================================
    # 3. VISUALIZATION ENGINE (PLOTLY)
    # ==========================================
    def visualize(self):
        if not self.data:
            print("No data available to plot.")
            return

        # Create Dashboard Layout
        # Row 1: Term Structure Skew Twist (per ticker)
        # Row 2: Cross-Sectional Heatmap
        
        rows = 2
        cols = max(2, len(self.data)) # Dynamic columns
        
        # We'll use a specific layout: 
        # Top Row: Individual Ticker Twists
        # Bottom Row: Summary Heatmap (Spanning all cols)
        
        specs = [[{"type": "scatter"} for _ in range(len(self.data))],
                 [{"type": "heatmap", "colspan": len(self.data)}, *[None]*(len(self.data)-1)]]
                 
        fig = make_subplots(
            rows=2, cols=len(self.data),
            row_heights=[0.7, 0.3],
            specs=specs,
            subplot_titles=[f"{t} Skew Structure" for t in self.data.keys()] + ["Cross-Asset Skew Rotation Heatmap"]
        )

        # -- PLOT 1: Individual Skew Twists --
        for i, (ticker, df) in enumerate(self.data.items()):
            col_idx = i + 1
            
            # 1. Current Skew Line
            fig.add_trace(go.Scatter(
                x=df['dte'], y=df['current_skew'],
                mode='lines+markers',
                name=f'{ticker} Now',
                line=dict(color='cyan', width=3),
                legendgroup=ticker,
                hovertemplate="DTE: %{x}<br>Skew: %{y:.1f} bps"
            ), row=1, col=col_idx)
            
            # 2. Past Skew Line (Approximated)
            fig.add_trace(go.Scatter(
                x=df['dte'], y=df['past_skew'],
                mode='lines',
                name=f'{ticker} 5d-Ago (Est)',
                line=dict(color='gray', width=2, dash='dot'),
                legendgroup=ticker,
                fill='tonexty', # Fill between this and the previous trace
                fillcolor='rgba(255, 0, 0, 0.1)' if df['change'].mean() > 0 else 'rgba(0, 255, 0, 0.1)',
                hovertemplate="Est Past: %{y:.1f} bps"
            ), row=1, col=col_idx)
            
            # Annotation: Insight Text
            insight = next((x for x in self.insights if x['ticker'] == ticker), None)
            if insight:
                # Add text annotation inside the plot
                fig.add_annotation(
                    dict(
                        x=0.5, y=1.1, xref=f"x{col_idx} domain", yref=f"y{col_idx} domain",
                        text=f"<b>{insight['msg']}</b>",
                        showarrow=False,
                        font=dict(size=10, color=insight['color']),
                        bgcolor="rgba(20,20,20,0.8)",
                        borderpad=4
                    )
                )

        # -- PLOT 2: Cross-Asset Heatmap --
        # Prepare Data
        # X-Axis: Expiry Buckets (Front, Back), Y-Axis: Tickers, Z: Change
        
        tickers_idx = []
        buckets_x = []
        values_z = []
        
        for ticker, df in self.data.items():
            # Front Bucket (0-45)
            chg_front = df[df['dte'] <= 45]['change'].mean()
            # Back Bucket (45+)
            chg_back = df[df['dte'] > 45]['change'].mean()
            
            tickers_idx.append(ticker)
            values_z.append([chg_front, chg_back])
        
        # Plotly Heatmap expects z as list of lists (rows)
        # We need to orient it correctly
        z_matrix = values_z 
        
        fig.add_trace(go.Heatmap(
            z=z_matrix,
            x=['Front Term (<45d)', 'Back Term (>45d)'],
            y=tickers_idx,
            colorscale='RdBu_r', # Red = Increasing Skew (Fear), Blue = Decreasing Skew (Calm)
            zmid=0,
            texttemplate="%{z:.1f} bps",
            showscale=True,
            colorbar=dict(title="Skew Chg (bps)", len=0.3, y=0.15)
        ), row=2, col=1)

        # -- STYLING --
        fig.update_layout(
            template="plotly_dark",
            title_text="<b>Institutional Volatility Rotation Dashboard</b><br><i>Visualizing the 'Twist' in Downside Skew (25d Put - ATM)</i>",
            height=900,
            hovermode="x unified",
            showlegend=True
        )
        
        # Update Axis Labels
        fig.update_yaxes(title_text="Skew Premium (bps)", row=1, col=1)
        fig.update_xaxes(title_text="Days to Expiry", row=1)

        fig.show()
        
        # Print Text Summary for Logging
        print("\n=== ANALYST SUMMARY ===")
        for ins in self.insights:
            print(f"[{ins['ticker']}] Change: {ins['front_chg']:+.1f} bps | {ins['msg']}")

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # You can pass specific tickers here, e.g., Scanner(['TSLA', 'AAPL'])
    # Leaving empty uses defaults [SPY, QQQ, NVDA, IWM]
    scanner = VolatilityScanner() 
    scanner.fetch_and_process()
    scanner.visualize()
