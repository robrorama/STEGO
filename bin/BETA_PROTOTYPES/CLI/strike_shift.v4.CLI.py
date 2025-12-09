# SCRIPTNAME: ok.strike_shift.v4.CLI.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- DEPENDENCY INJECTION ---
try:
    import options_data_retrieval as odr
    import data_retrieval as dr
except ImportError:
    print("CRITICAL: Libraries 'options_data_retrieval' or 'data_retrieval' missing.")
    sys.exit(1)

# --- CONFIGURATION ---
OUTPUT_DIR = "/dev/shm"
DEFAULT_IV_SPIKE = 0.0  # Default to 0 so it has no effect unless specified

# Ensure output path exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# CORE UTILITIES
# ==============================================================================

def get_realtime_spot(ticker):
    """Fetches real-time spot price via yfinance fast_info."""
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.last_price
        if price is None:
            hist = t.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
    except Exception as e:
        print(f"[!] Error fetching spot: {e}")
        return None
    return price

def find_closest_expiration(ticker, weeks_out):
    """Finds the expiration date closest to 'weeks_out' from today."""
    target_date = datetime.now() + timedelta(weeks=weeks_out)
    try:
        exps = odr.get_available_remote_expirations(ticker, source="yfinance")
        if not exps: return None
        # exps are timestamps, convert target to timestamp for comparison
        target_ts = pd.Timestamp(target_date)
        closest_exp = min(exps, key=lambda x: abs(x - target_ts))
        return closest_exp
    except Exception as e:
        print(f"[!] Error finding expiration: {e}")
        return None

def find_closest_strike_in_chain(chain_df, target_strike_val):
    """Snaps a theoretical strike price to the nearest actual strike in the chain."""
    available_strikes = chain_df['strike'].values
    idx = (np.abs(available_strikes - target_strike_val)).argmin()
    return available_strikes[idx]

def calculate_atm_iv(chain_df, spot_price):
    """Estimates ATM IV."""
    if 'impliedVolatility' not in chain_df.columns: return 20.0
    valid = chain_df[chain_df['impliedVolatility'] > 0].copy()
    if valid.empty: return 20.0
    valid['dist'] = abs(valid['strike'] - spot_price)
    return valid.nsmallest(4, 'dist')['impliedVolatility'].mean() * 100

def interpolate_price(chain_df, target_strike):
    """Get price for a strike (interpolated if exact match missing)."""
    # Prefer Midpoint
    if 'bid' in chain_df.columns and 'ask' in chain_df.columns:
        chain_df['mid'] = (chain_df['bid'] + chain_df['ask']) / 2
        chain_df['px'] = np.where(chain_df['mid'] > 0, chain_df['mid'], chain_df['lastPrice'])
    else:
        chain_df['px'] = chain_df['lastPrice']

    df_clean = chain_df.sort_values('strike').dropna(subset=['px'])
    if df_clean.empty: return 0.0
    
    # Exact match check
    exact = df_clean[df_clean['strike'] == target_strike]
    if not exact.empty: return exact.iloc[0]['px']

    return np.interp(target_strike, df_clean['strike'].values, df_clean['px'].values)

# ==============================================================================
# VISUALIZATION & REPORTING
# ==============================================================================

def generate_dashboard(ctx):
    """Prints ASCII Dashboard."""
    spot = ctx['spot']
    target_spot = ctx['target_spot']
    
    print("\n" + "█"*70)
    print(f"   STEGO FRAMEWORK :: {ctx['ticker']} :: STRIKE-SHIFT ESTIMATOR")
    print("█"*70)
    
    # Overview
    print(f"\n >> CONTEXT")
    print(f"    Current Spot:    ${spot:.2f}")
    print(f"    Target Spot:     ${target_spot:.2f} (Implied from Strike Offset)")
    print(f"    Target Date:     {ctx['expiration'].strftime('%Y-%m-%d')} (Approx {ctx['weeks_out']} wks out)")
    print(f"    Your Strike:     {ctx['user_strike']} (Offset: {ctx['strike_offset']:+.2f})")
    print(f"    ATM IV:          {ctx['current_iv']:.2f}%")
    
    # Logic
    print(f"\n >> LOGIC TRACE")
    print(f"    1. Future Moneyness: Your strike {ctx['user_strike']} at target ${target_spot:.2f} is ${abs(ctx['future_moneyness']):.2f} {ctx['status']}.")
    print(f"    2. Strike-Shift: We look for the strike currently ${abs(ctx['future_moneyness']):.2f} {ctx['status']}.")
    print(f"    3. Proxy Strike: {ctx['proxy_strike']:.2f}")

    # Estimates
    print(f"\n >> PRICING TARGETS (GTC LIMITS)")
    print(f"    [FLOOR]      Conservative (Flat IV):   ${ctx['base_price']:.2f}")
    print(f"    [REALISTIC]  Adjusted (+{ctx['iv_jump']}vol):      ${ctx['adj_price']:.2f}")
    print(f"    [AGGRESSIVE] Panic Premium (+20%):     ${ctx['agg_price']:.2f}")
    
    # Matrix
    print(f"\n >> SENSITIVITY MATRIX")
    print(ctx['matrix'].to_string(index=False))
    print("\n" + "█"*70 + "\n")

def generate_plot(ctx, chain_df):
    """Generates Plotly Visualization."""
    chain_df = chain_df.sort_values('strike').copy()
    if 'bid' in chain_df.columns:
        chain_df['px'] = (chain_df['bid'] + chain_df['ask']) / 2
        chain_df['px'] = np.where(chain_df['px'] > 0, chain_df['px'], chain_df['lastPrice'])
    else:
        chain_df['px'] = chain_df['lastPrice']

    fig = go.Figure()
    
    # Main Curve
    fig.add_trace(go.Scatter(
        x=chain_df['strike'], y=chain_df['px'],
        mode='lines+markers', name='Current Curve',
        line=dict(color='#636EFA', width=2)
    ))
    
    # User Strike (Current)
    user_px = interpolate_price(chain_df, ctx['user_strike'])
    fig.add_trace(go.Scatter(
        x=[ctx['user_strike']], y=[user_px],
        mode='markers', name=f"Your Strike ({ctx['user_strike']})",
        marker=dict(color='cyan', size=12)
    ))
    
    # Proxy Strike
    fig.add_trace(go.Scatter(
        x=[ctx['proxy_strike']], y=[ctx['base_price']],
        mode='markers', name=f"Proxy Strike ({ctx['proxy_strike']:.1f})",
        marker=dict(color='orange', size=12, symbol='diamond')
    ))
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=[ctx['proxy_strike']], y=[ctx['adj_price']],
        mode='markers', name='Target Price (IV Adj)',
        marker=dict(color='#00CC96', size=15, symbol='star')
    ))

    # Annotation
    fig.add_annotation(
        x=ctx['proxy_strike'], y=ctx['adj_price'],
        text=f"Sell Target: ${ctx['adj_price']:.2f}",
        showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color="#00CC96")
    )

    fig.update_layout(
        title=f"STEGO Strike-Shift: {ctx['ticker']} | Target ${ctx['target_spot']:.2f}",
        xaxis_title="Strike Price", yaxis_title="Option Premium",
        template="plotly_dark", height=600
    )
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(OUTPUT_DIR, f"{ctx['ticker']}_STEGO_{ts}.html")
    fig.write_html(fname)
    print(f" >> Visualization saved: {fname}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("--- STEGO FRAMEWORK INITIALIZED ---")
    
    # 1. Inputs
    ticker = input("Ticker (e.g. SPY): ").strip().upper()
    if not ticker: sys.exit(0)
    
    try:
        weeks_in = int(input("Weeks Out (e.g. 2): ").strip())
        
        # User calculates inputs relative to spot
        print("NOTE: Target Spot will be derived from Strike Offset (Spot + Offset)")
        strike_offset = float(input("Strike Offset (e.g. -10 for strike $10 below spot): ").strip())
        
        # Optional IV input
        vol_in = input(f"Exp. Vol Spike (Default {DEFAULT_IV_SPIKE}): ").strip()
        iv_jump = float(vol_in) if vol_in else DEFAULT_IV_SPIKE
        
    except ValueError:
        print("Invalid numerical input."); sys.exit(1)

    # 2. Data Fetching
    print(f"\n[1/3] Fetching Live Data for {ticker}...")
    spot = get_realtime_spot(ticker)
    if not spot: 
        print("Error: Could not fetch spot price."); sys.exit(1)
    
    print(f"      Current Spot: ${spot:.2f}")

    # Calculate absolute targets based on offsets
    # Use Strike Offset for Price Offset (Assumption: Crash to Strike)
    price_offset = strike_offset
    target_spot = spot + price_offset
    calculated_strike_target = spot + strike_offset
    
    # Resolve Expiration
    expiry = find_closest_expiration(ticker, weeks_in)
    if not expiry:
        print(f"No expiration found ~{weeks_in} weeks out."); sys.exit(1)
    
    print(f"      Matched Expiration: {expiry.date()}")
    
    # Load Chain
    try:
        chain = odr.load_or_download_option_chain(ticker, expiry, force_refresh=True)
    except Exception as e:
        print(f"Chain Error: {e}"); sys.exit(1)

    # Filter Puts
    puts = chain[chain['type'] == 'put'].copy()
    if puts.empty: print("No Puts found."); sys.exit(1)

    # Snap to nearest actual strike
    # e.g., Spot 100, Offset -2 -> Calc 98. Nearest might be 98, or 97.5, or 100.
    user_strike = find_closest_strike_in_chain(puts, calculated_strike_target)
    
    # 3. Calculation
    print("[2/3] Running Strike-Shift Model...")
    current_iv = calculate_atm_iv(puts, spot)
    
    def run_scenario(tgt_spot_local):
        # Moneyness Logic: Strike - Spot (For Put, + is ITM)
        future_moneyness = user_strike - tgt_spot_local
        
        # Proxy Strike Logic
        # Proxy = Current_Spot + Future_Moneyness
        proxy_k = spot + future_moneyness
        
        # Base Price
        base_p = interpolate_price(puts, proxy_k)
        
        # IV Adj: Price * (1 + (Jump/IV))^0.72
        # If Jump is 0, mult is 1.0 (No change)
        if current_iv > 0:
            mult = (1 + (iv_jump / current_iv)) ** 0.72
        else:
            mult = 1.0
            
        adj_p = base_p * mult
        
        return future_moneyness, proxy_k, base_p, adj_p

    fut_money, proxy_k, base_p, adj_p = run_scenario(target_spot)
    agg_p = adj_p * 1.20
    
    # Matrix
    matrix_data = []
    for off in [-5.0, 0.0, 5.0]:
        t_s = target_spot + off
        _, _, _, est = run_scenario(t_s)
        matrix_data.append({
            "Target Spot": f"${t_s:.2f}",
            "Offset": f"{price_offset + off:+.2f}",
            "Est. Put Price": f"${est:.2f}"
        })
    matrix_df = pd.DataFrame(matrix_data)
    
    # Context Dict
    ctx = {
        'ticker': ticker, 'expiration': expiry, 'weeks_out': weeks_in,
        'spot': spot, 'target_spot': target_spot, 'price_offset': price_offset,
        'user_strike': user_strike, 'strike_offset': strike_offset,
        'future_moneyness': fut_money, 'status': "ITM" if fut_money > 0 else "OTM",
        'proxy_strike': proxy_k, 'base_price': base_p,
        'adj_price': adj_p, 'agg_price': agg_p,
        'iv_jump': iv_jump, 'current_iv': current_iv,
        'matrix': matrix_df
    }

    # 4. Outputs
    print("[3/3] Generating Artifacts...")
    generate_dashboard(ctx)
    generate_plot(ctx, puts)
    
    # CSV Export
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_STEGO_EST_{ts}.csv")
    # Convert dataframe to CSV compatible format (ctx is dict, matrix is df)
    export_df = pd.DataFrame([{k:v for k,v in ctx.items() if k != 'matrix'}])
    export_df.to_csv(csv_path, index=False)
    print(f" >> Data exported: {csv_path}")

if __name__ == "__main__":
    main()
