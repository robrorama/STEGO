# SCRIPTNAME: ok.06.l2.tradeability.dashboard.v1.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
SYMBOL = "SPY"  # Change to your target ticker
LOOKBACK_DAYS = 365
ROLLING_WINDOW = 20  # For baseline calculations
EXPIRY_INDEX = 0     # 0 = nearest expiry, 1 = next, etc.

# Tradeability Weights (from your spec)
W1, W2, W3, W4 = 0.35, 0.25, 0.25, 0.15 
ALPHA, BETA = 0.20, 0.10

# ==============================================================================
# DATA INGESTION ENGINE
# ==============================================================================
def fetch_market_data(ticker_symbol):
    """
    Fetches real OHLCV data and Options Chain.
    NO FAKE DATA: Returns standard yfinance objects.
    """
    print(f"--- Fetching Data for {ticker_symbol} ---")
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Get Historical Price Data
    df = ticker.history(period="1y", interval="1d")
    if df.empty:
        raise ValueError("No price data found. Check ticker or internet connection.")
    
    # 2. Get Options Chain (for Implied Volatility Proxy)
    # We use IV to approximate the "Spread" and "Market Maker Anxiety"
    try:
        exps = ticker.options
        if exps:
            opt_date = exps[min(EXPIRY_INDEX, len(exps)-1)]
            opt_chain = ticker.option_chain(opt_date)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Filter for near-the-money liquidity to get representative IV
            current_price = df['Close'].iloc[-1]
            calls['abs_diff'] = abs(calls['strike'] - current_price)
            # Average IV of the 5 closest strikes
            iv_proxy = calls.nsmallest(5, 'abs_diff')['impliedVolatility'].mean()
        else:
            iv_proxy = 0.0
    except Exception as e:
        print(f"Warning: Could not fetch options data ({e}). Defaulting IV to historical vol.")
        iv_proxy = None

    return df, iv_proxy

# ==============================================================================
# MICROSTRUCTURE APPROXIMATION ENGINE
# ==============================================================================
def process_signals(df, iv_current):
    """
    Approximates L2 signals using L1 data + Statistical proxies.
    """
    data = df.copy()
    
    # --- 1. Basic Calculation ---
    data['Mid_Proxy'] = (data['High'] + data['Low']) / 2
    # Typical Price (VWAP proxy)
    data['Microprice_m'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    # --- 2. Imbalance (I) Approximation ---
    # Logic: Where did we close relative to the range?
    # +1 = Closed at High (Buying Pressure), -1 = Closed at Low (Selling Pressure)
    # Handle division by zero
    range_hl = data['High'] - data['Low']
    data['Imbalance_I'] = np.where(
        range_hl == 0, 
        0, 
        (2 * data['Close'] - data['High'] - data['Low']) / range_hl
    )
    
    # --- 3. Sweep/Depth Proxy (Volume Z-Score) ---
    # We don't have ticks, but we have Volume. 
    # High Volume + High Range = Sweep. High Volume + Low Range = Absorbtion (Deep Book).
    data['Vol_Mean'] = data['Volume'].rolling(window=ROLLING_WINDOW).mean()
    data['Vol_Std'] = data['Volume'].rolling(window=ROLLING_WINDOW).std()
    data['Vol_Z'] = (data['Volume'] - data['Vol_Mean']) / data['Vol_Std']
    data['Vol_Z'] = data['Vol_Z'].fillna(0) # Handle NaN at start
    
    # Sweep Rate Proxy: If Volume is > 1.5 sigma and price moved significantly
    data['Sweep_Rate_Proxy'] = np.where(
        (data['Vol_Z'] > 1.5) & (abs(data['Close'] - data['Open']) > data['Close']*0.005), 
        1, 
        0
    )
    
    # --- 4. Spread & Penalty Approximation ---
    # Real MM spread is correlated with Volatility (ATR) and Inverse Volume
    data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()
    
    # If we have real IV, use it to scale the spread estimation
    # If IV is high, spreads are wider.
    vol_scalar = iv_current if iv_current else data['Close'].pct_change().std() * np.sqrt(252)
    
    # Synthetic Spread: Fraction of ATR scaled by Volatility
    # (High vol = wider spread)
    data['Spread_Est'] = (data['ATR'] * 0.05) * (1 + vol_scalar)
    data['Tick_Size'] = 0.01 # Assumption for equities
    
    # Distance Term D (Microprice pull)
    data['D'] = abs(data['Microprice_m'] - data['Mid_Proxy']) / data['Tick_Size']
    
    # Penalties
    data['P_spread'] = ALPHA * (data['Spread_Est'] / data['Tick_Size'])
    # Cancel Rate Proxy: High volatility usually equals high cancel rates
    data['P_cancel'] = BETA * (data['ATR'] / data['Close']) * 100 

    # --- 5. Composite Tradeability Score (S) ---
    # S = w1*D + w2*|I| + w3*sweep + w4*vol - penalties
    data['Raw_Score'] = (
        W1 * data['D'] + 
        W2 * abs(data['Imbalance_I']) + 
        W3 * data['Sweep_Rate_Proxy'] + 
        W4 * data['Vol_Z'] - 
        (data['P_spread'] + data['P_cancel'])
    )
    
    # Normalize to 0-100 (Robust Scaling based on last 60 days to adapt to regime)
    lookback_norm = 60
    roll_min = data['Raw_Score'].rolling(lookback_norm).min()
    roll_max = data['Raw_Score'].rolling(lookback_norm).max()
    
    data['T_Score'] = ((data['Raw_Score'] - roll_min) / (roll_max - roll_min)) * 100
    data['T_Score'] = data['T_Score'].clip(0, 100).fillna(50) # Default to 50 if data missing
    
    return data.tail(100) # Only return recent history for visualization

# ==============================================================================
# VISUALIZATION ENGINE (PLOTLY)
# ==============================================================================
def create_dashboard(df, ticker, iv):
    
    last_t = df['T_Score'].iloc[-1]
    
    # Determine Theme Colors based on Tradeability
    if last_t >= 60:
        gauge_color = "#00ffcc" # Cyan/Green (High Tradeability)
        status_msg = "LIQUID / AGGRESSIVE"
    elif last_t <= 30:
        gauge_color = "#ff0055" # Red (Low Tradeability)
        status_msg = "ILLIQUID / DEFENSIVE"
    else:
        gauge_color = "#ffcc00" # Amber
        status_msg = "NEUTRAL"

    # Layout: 
    # Row 1: Price Action with "Liquidity Cloud" (Approximated Depth)
    # Row 2: Microstructure Signals (Imbalance & Vol Z)
    # Row 3: Tradeability Score Heatmap
    
    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.5, 0.25, 0.25],
        specs=[
            [{"type": "xy"}, {"type": "indicator", "rowspan": 3}],
            [{"type": "xy"}, None],
            [{"type": "xy"}, None]
        ],
        vertical_spacing=0.05,
        subplot_titles=(
            f"{ticker} Price & Approx. Liquidity Depth", 
            "Tradeability Gauge", 
            "Market Imbalance & Volume Regimes", 
            "Composite Tradeability Score (T)"
        )
    )

    # --- 1. Main Chart: Price & Liquidity Bands ---
    # We visualize "Book Depth" as bands around the price. 
    # Wider bands = Thinner Book (High Vol). Narrow bands = Thicker Book.
    
    # Liquidity Band (Upper)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['High'] + df['Spread_Est']*2,
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    
    # Liquidity Band (Lower) - Fill to Upper
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Low'] - df['Spread_Est']*2,
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(255, 255, 255, 0.05)',
        name='Est. Market Depth',
        hoverinfo='skip'
    ), row=1, col=1)

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Microprice (VWAP Proxy) Overlay
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Microprice_m'],
        mode='lines', line=dict(color='#ff00ff', width=1, dash='dot'),
        name='Microprice (m) Proxy'
    ), row=1, col=1)

    # --- 2. Subchart: Imbalance & Volume ---
    # Bar chart for Imbalance, colored by Volume Z-Score
    colors = ['#ff0055' if x < 0 else '#00ffcc' for x in df['Imbalance_I']]
    
    fig.add_trace(go.Bar(
        x=df.index, y=df['Imbalance_I'],
        marker=dict(color=colors, line=dict(width=0)),
        name='Order Imbalance (I)',
        opacity=0.8
    ), row=2, col=1)
    
    # Add Threshold lines for Imbalance
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="gray", row=2, col=1)

    # --- 3. Subchart: Tradeability Score ---
    # Area chart for T-Score
    fig.add_trace(go.Scatter(
        x=df.index, y=df['T_Score'],
        mode='lines', fill='tozeroy',
        line=dict(color=gauge_color, width=2),
        name='Tradeability (T)'
    ), row=3, col=1)
    
    # --- 4. Sidebar: The Gauge & Metrics ---
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = last_t,
        title = {'text': f"<b>T-SCORE</b><br><span style='font-size:0.6em;color:gray'>{status_msg}</span>"},
        delta = {'reference': df['T_Score'].iloc[-2], 'increasing': {'color': "#00ffcc"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': gauge_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(255, 0, 85, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(255, 204, 0, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(0, 255, 204, 0.3)'}
            ],
        }
    ), row=1, col=2)

    # IV Text
    iv_text = f"{iv*100:.1f}%" if iv else "N/A"
    fig.add_annotation(
        text=f"<b>Implied Vol:</b> {iv_text}<br><b>Avg Vol Z:</b> {df['Vol_Z'].iloc[-1]:.2f}",
        xref="paper", yref="paper",
        x=1.15, y=0.3, showarrow=False,
        font=dict(color="white", size=12),
        align="left",
        bordercolor="gray", borderwidth=1, borderpad=10, bgcolor="#1e1e1e"
    )

    # Update Layout
    fig.update_layout(
        template="plotly_dark",
        height=900,
        title_text=f"<b>L2 TRADEABILITY DASHBOARD</b> | {datetime.now().strftime('%Y-%m-%d')}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Specific Axis Tweaks
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Imbalance", range=[-1.1, 1.1], row=2, col=1)
    fig.update_yaxes(title_text="T-Score", range=[0, 100], row=3, col=1)

    return fig

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. Fetch
        df, iv = fetch_market_data(SYMBOL)
        
        # 2. Approximate Microstructure
        processed_df = process_signals(df, iv)
        
        # 3. Render
        fig = create_dashboard(processed_df, SYMBOL, iv)
        
        # 4. Save
        filename = f"{SYMBOL}_Tradeability_Dashboard.html"
        fig.write_html(filename, include_plotlyjs='cdn')
        
        print(f"\n[SUCCESS] Dashboard generated: {filename}")
        print(f"Current Tradeability Score: {processed_df['T_Score'].iloc[-1]:.2f}/100")
        print("Note: L2 signals (sweeps, depth) are STATISTICALLY APPROXIMATED from Daily OHLCV & Options Volatility.")
        
    except Exception as e:
        print(f"[ERROR] Failed to run: {e}")
