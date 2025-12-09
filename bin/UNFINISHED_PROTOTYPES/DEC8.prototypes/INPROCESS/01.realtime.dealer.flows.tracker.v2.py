#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------
QUANTITATIVE DERIVATIVES MONITOR (SINGLE-SHOT EDITION)
-------------------------------------------------------------------------------
Description:
    Performs a ONE-TIME analysis of Dealer Flow, Vol Surface, and Macro signals.
    Designed to prevent API rate-limit abuse.
    
    Generates a complex interactive HTML dashboard for interpretation.

Usage:
    python quant_monitor_v2.py --ticker SPY --real_data

Requirements:
    pip install numpy pandas scipy scikit-learn plotly yfinance
-------------------------------------------------------------------------------
"""

import sys
import os
import json
import logging
import argparse
import datetime
import warnings
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, norm
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try importing yfinance for real data, handle gracefully if missing
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("QuantEngine")

OUTPUT_DIR = "quant_output"
RISK_FREE_RATE = 0.045

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class MarketSnapshot:
    timestamp: datetime.datetime
    ticker: str
    price: float
    # Option Chain: [expiry, strike, type, iv, oi, gamma(calc), vanna(calc)]
    chain_data: pd.DataFrame
    # Term Structure: {days: iv}
    term_structure: Dict[int, float]
    # Macro
    real_yield_10y: float
    vol_of_vol: float
    # Market Stats
    adv_20d: float

@dataclass
class ComputedMetrics:
    # Dealer Flow
    gex_total_notional: float
    gex_profile: pd.DataFrame  # GEX per strike
    zero_gex_level: float
    vanna_exposure: float
    # Vol Surface
    skew_slope: float
    term_slope: float  # Front - Back
    # Macro
    macro_regime_score: float # 0 to 100 (Risk On to Risk Off)

# ==============================================================================
# DATA INGESTION (REAL + SYNTHETIC)
# ==============================================================================

class DataAdapter:
    def fetch(self) -> MarketSnapshot:
        raise NotImplementedError

class YFinanceAdapter(DataAdapter):
    """Real-world data adapter using yfinance (Single Request)."""
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()

    def fetch(self) -> MarketSnapshot:
        if not HAS_YF:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        logger.info(f"Fetching LIVE data for {self.ticker} via yfinance...")
        stock = yf.Ticker(self.ticker)
        
        # 1. Price & Volume
        hist = stock.history(period="1mo")
        if hist.empty:
            raise ValueError(f"No history found for {self.ticker}")
        
        current_price = hist['Close'].iloc[-1]
        adv = (hist['Volume'] * hist['Close']).mean()
        
        # 2. Options Chain (Fetch nearest 2 expiries for speed)
        expirations = stock.options
        if not expirations:
            raise ValueError("No options chain found.")
        
        chain_frames = []
        # Limit to first 3 expiries to save time/requests
        target_expiries = expirations[:3] 
        
        for date_str in target_expiries:
            try:
                opt = stock.option_chain(date_str)
                calls = opt.calls
                puts = opt.puts
                
                # Parse Expiry
                exp_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                days_to_exp = (exp_date - datetime.datetime.now()).days
                if days_to_exp < 1: days_to_exp = 1
                
                # Normalize Columns
                calls['type'] = 'C'
                puts['type'] = 'P'
                
                df = pd.concat([calls, puts])
                df['expiry'] = days_to_exp
                df['strike'] = df['strike'].astype(float)
                df['iv'] = df['impliedVolatility']
                df['oi'] = df['openInterest'].fillna(0)
                
                chain_frames.append(df[['expiry', 'strike', 'type', 'iv', 'oi']])
            except Exception as e:
                logger.warning(f"Failed to fetch expiry {date_str}: {e}")
                
        full_chain = pd.concat(chain_frames)
        
        # 3. Term Structure (Approximate from chain)
        # Group by expiry, get ATM IV
        term_struct = {}
        for dte in full_chain['expiry'].unique():
            # Filter for strikes near spot
            near_atm = full_chain[
                (full_chain['expiry'] == dte) & 
                (full_chain['strike'] >= current_price * 0.95) & 
                (full_chain['strike'] <= current_price * 1.05)
            ]
            if not near_atm.empty:
                term_struct[int(dte)] = near_atm['iv'].mean()
        
        # 4. Macro (Proxy using TNX for yield, VIX for VoV)
        # Note: In a real script, you'd fetch ^TNX and ^VVIX separately
        # Here we use static proxies or fetch if requested to avoid too many calls
        real_yield = 2.0 # Fallback proxy
        vov = 85.0       # Fallback proxy
        
        return MarketSnapshot(
            timestamp=datetime.datetime.now(),
            ticker=self.ticker,
            price=current_price,
            chain_data=full_chain,
            term_structure=term_struct,
            real_yield_10y=real_yield,
            vol_of_vol=vov,
            adv_20d=adv
        )

class SyntheticAdapter(DataAdapter):
    """Generates a realistic Vol Surface and GEX landscape for testing visuals."""
    def __init__(self, ticker: str):
        self.ticker = ticker
        
    def fetch(self) -> MarketSnapshot:
        logger.info(f"Generating SYNTHETIC data for {self.ticker}...")
        spot = 4500.0
        
        # 1. Generate Chain
        strikes = np.linspace(4000, 5000, 50)
        expiries = [5, 30, 60, 90]
        data = []
        
        # "Smirk" generation
        for dte in expiries:
            t = dte / 365.0
            for k in strikes:
                moneyness = spot / k
                iv_base = 0.15 + (0.1 / np.sqrt(t)) * (np.log(k/spot)**2) # Parabolic skew
                iv_skew = 0.05 * (1 - moneyness) # Linear skew
                iv = max(0.05, iv_base + iv_skew + np.random.normal(0, 0.002))
                
                # OI Clustering (Gamma Walls at round numbers)
                is_round = (k % 100 == 0)
                oi_base = 1000 if is_round else 200
                oi = int(oi_base * np.exp(-0.5 * ((k-spot)/200)**2))
                
                data.append({'expiry': dte, 'strike': k, 'type': 'C', 'iv': iv, 'oi': oi})
                data.append({'expiry': dte, 'strike': k, 'type': 'P', 'iv': iv, 'oi': oi})
                
        df = pd.DataFrame(data)
        
        # 2. Term Structure
        ts = {5: 0.18, 30: 0.16, 60: 0.155, 90: 0.15}
        
        return MarketSnapshot(
            timestamp=datetime.datetime.now(),
            ticker=self.ticker,
            price=spot,
            chain_data=df,
            term_structure=ts,
            real_yield_10y=2.1 + np.random.normal(0, 0.1),
            vol_of_vol=90.0 + np.random.normal(0, 5),
            adv_20d=1_000_000_000
        )

# ==============================================================================
# CALCULATION ENGINE
# ==============================================================================

class QuantEngine:
    @staticmethod
    def calculate_greeks_and_metrics(snap: MarketSnapshot) -> ComputedMetrics:
        df = snap.chain_data.copy()
        S = snap.price
        r = RISK_FREE_RATE
        
        # 1. Black-Scholes Greeks Approximation
        # Need time in years
        df['t'] = df['expiry'] / 365.0
        # Avoid division by zero
        df['t'] = df['t'].replace(0, 0.001) 
        
        df['d1'] = (np.log(S / df['strike']) + (r + 0.5 * df['iv']**2) * df['t']) / (df['iv'] * np.sqrt(df['t']))
        df['nd1'] = norm.pdf(df['d1'])
        
        # Gamma: N'(d1) / (S * sigma * sqrt(t))
        df['gamma'] = df['nd1'] / (S * df['iv'] * np.sqrt(df['t']))
        
        # Vanna: -N'(d1) * d1 / sigma
        # (Simplified, often dVanna/dVol)
        df['vanna'] = -df['nd1'] * df['d1'] / df['iv']

        # 2. GEX Calculation (Dealer Direction)
        # Dealers: Long Call (clients sell), Short Put (clients buy) -> This is simplistic.
        # Standard GEX Model: Dealers Short Gamma when they sell options.
        # Assumption: Clients BUY Calls (Dealers Short Call), Clients BUY Puts (Dealers Short Put).
        # Actually standard "GEX" usually implies:
        # Dealer Gamma = Call OI * Gamma (Long) - Put OI * Gamma (Short)
        # (assuming dealers are long calls they sold?? No.)
        #
        # Standard Market Convention for GEX:
        # Dealers are Short Calls (Negative Gamma)
        # Dealers are Long Puts (Positive Gamma) -- WAIT.
        # Let's use the SpotGamma/SqueezeMetrics convention:
        # Call GEX contribution is positive (Dealers hedging acts to dampen vol? No, Dealers short calls -> Short Gamma -> Amplify).
        # Let's stick to the basic "Long Gamma" vs "Short Gamma" view.
        # If dealers are Long Gamma -> Buy low/Sell high -> Stabilizing.
        # If dealers are Short Gamma -> Sell low/Buy high -> Destabilizing.
        # Typically: Clients Buy Calls -> Dealer Short Call (Short Gamma).
        # Clients Buy Puts -> Dealer Short Put (Short Gamma).
        # But usually Puts are hedged by dealers being Short stock?
        
        # Let's use the standard "GEX Flip" formula:
        # GEX = Gamma * OI * Spot * 100
        # Call GEX is usually considered Positive (Long Gamma) in many models if Clients are selling calls (Overwriting).
        # However, for this visual script, we will calculate Net GEX per strike assuming:
        # Call OI = Dealer Short Gamma ( - )
        # Put OI = Dealer Long Gamma ( + )  <-- This varies by model.
        # Let's use the "Naive GEX": Call Gamma - Put Gamma.
        
        df['gex_contrib'] = 0.0
        mask_c = df['type'] == 'C'
        mask_p = df['type'] == 'P'
        
        # Model: Dealers are long calls (clients writing overwrites) and short puts (clients buying protection).
        # This is a debated assumption. Let's use:
        # Call GEX = + Gamma * OI (Dealers Long)
        # Put GEX = - Gamma * OI (Dealers Short)
        df.loc[mask_c, 'gex_contrib'] = df.loc[mask_c, 'gamma'] * df.loc[mask_c, 'oi'] * 100 * S
        df.loc[mask_p, 'gex_contrib'] = df.loc[mask_p, 'gamma'] * df.loc[mask_p, 'oi'] * 100 * S * -1
        
        total_gex = df['gex_contrib'].sum()
        
        # GEX Profile by Strike
        gex_profile = df.groupby('strike')['gex_contrib'].sum().reset_index()
        gex_profile = gex_profile.sort_values('strike')
        
        # Find Zero GEX Flip Level (Interpolation)
        zero_flip = S # Fallback
        # Simple scan for sign change
        signs = np.sign(gex_profile['gex_contrib'])
        diffs = np.diff(signs)
        crossings = np.where(diffs != 0)[0]
        if len(crossings) > 0:
            # Take the crossing nearest to price
            candidates = gex_profile.iloc[crossings]['strike']
            zero_flip = candidates.iloc[(candidates - S).abs().argsort()[:1]].values[0]

        # 3. Macro Scoring (Mock)
        # High Yield + High VoV = High Risk
        risk_score = min(100, (snap.real_yield_10y * 10) + (snap.vol_of_vol / 2))

        return ComputedMetrics(
            gex_total_notional=total_gex,
            gex_profile=gex_profile,
            zero_gex_level=zero_flip,
            vanna_exposure=df['vanna'].sum(),
            skew_slope=0.0, # Placeholder
            term_slope=0.0,
            macro_regime_score=risk_score
        )

# ==============================================================================
# VISUALIZATION ENGINE (COMPLEX)
# ==============================================================================

class DashboardGenerator:
    @staticmethod
    def render(snap: MarketSnapshot, metrics: ComputedMetrics, filename: str):
        logger.info("Generating Complex Visuals...")
        
        # Create a dashboard with specs for different chart types
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.5, 0.5],
            specs=[
                [{"type": "xy"}, {"type": "surface"}],  # Row 1: GEX Profile (2D) | Vol Surface (3D)
                [{"type": "xy"}, {"type": "indicator"}] # Row 2: Term Structure | Macro Score
            ],
            subplot_titles=(
                f"Dealer GEX Profile (Spot: {snap.price:.2f})", 
                "3D Volatility Surface", 
                "Term Structure Cone", 
                "Macro Risk Regime"
            )
        )

        # --- 1. GEX Profile (Bar Chart) ---
        # Color bars based on Positive/Negative Gamma
        colors = ['#00FF00' if x >= 0 else '#FF0000' for x in metrics.gex_profile['gex_contrib']]
        
        fig.add_trace(go.Bar(
            x=metrics.gex_profile['strike'],
            y=metrics.gex_profile['gex_contrib'],
            marker_color=colors,
            name="Net GEX",
            opacity=0.8
        ), row=1, col=1)
        
        # Add Spot Line
        fig.add_vline(x=snap.price, line_width=2, line_dash="dash", line_color="white", row=1, col=1)
        fig.add_annotation(x=snap.price, y=0, text="SPOT", showarrow=True, arrowhead=1, row=1, col=1)

        # --- 2. 3D Volatility Surface ---
        # Pivot data for mesh
        df = snap.chain_data
        # We need a grid. Filter for calls to map skew.
        calls = df[df['type'] == 'C']
        if len(calls) > 10:
            # Create grid
            pivot_iv = calls.pivot_table(index='expiry', columns='strike', values='iv').interpolate(method='linear', axis=1)
            
            fig.add_trace(go.Surface(
                z=pivot_iv.values,
                x=pivot_iv.columns, # Strikes
                y=pivot_iv.index,   # Expiry
                colorscale='Viridis',
                name="Vol Surface",
                colorbar=dict(len=0.5, y=0.8)
            ), row=1, col=2)

        # --- 3. Term Structure Cone ---
        ts_days = list(snap.term_structure.keys())
        ts_ivs = list(snap.term_structure.values())
        
        fig.add_trace(go.Scatter(
            x=ts_days, y=ts_ivs,
            mode='lines+markers',
            line=dict(color='cyan', width=3),
            marker=dict(size=8),
            name="Current Term Struct"
        ), row=2, col=1)
        
        # Add visual bands (mock cones for context)
        fig.add_trace(go.Scatter(
            x=ts_days, y=[x * 1.2 for x in ts_ivs],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=ts_days, y=[x * 0.8 for x in ts_ivs],
            mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,255,255,0.1)',
            name="Normal Range"
        ), row=2, col=1)

        # --- 4. Macro Gauge ---
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = metrics.macro_regime_score,
            title = {'text': "Macro Risk Index"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80}
            }
        ), row=2, col=2)

        # Styling
        fig.update_layout(
            template="plotly_dark",
            title_text=f"QUANT MONITOR: {snap.ticker} | {snap.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            height=900,
            showlegend=False
        )
        
        # Camera angle for 3D plot
        fig.update_scenes(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))

        path = os.path.join(OUTPUT_DIR, filename)
        fig.write_html(path)
        print(f"\n[SUCCESS] Dashboard generated at: {path}")

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Single-Shot Quant Monitor")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--real_data", action="store_true", help="Use yfinance (real) instead of synthetic")
    args = parser.parse_args()

    # Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Select Adapter
    if args.real_data:
        if not HAS_YF:
            print("Error: yfinance not installed. Please install or omit --real_data.")
            sys.exit(1)
        adapter = YFinanceAdapter(args.ticker)
    else:
        adapter = SyntheticAdapter(args.ticker)

    try:
        # 2. Execution Pipeline (One Pass)
        print("--- STARTING SINGLE-SHOT ANALYSIS ---")
        
        snapshot = adapter.fetch()
        print(f"Data Loaded: {snapshot.ticker} @ {snapshot.price:.2f}")
        
        metrics = QuantEngine.calculate_greeks_and_metrics(snapshot)
        print(f"Metrics Calculated: Net GEX {metrics.gex_total_notional/1e9:.2f}B | Zero Flip {metrics.zero_gex_level:.2f}")
        
        # 3. Visualization
        filename = f"dashboard_{args.ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        DashboardGenerator.render(snapshot, metrics, filename)
        
        print("--- ANALYSIS COMPLETE ---")

    except Exception as e:
        logger.error(f"Execution Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
