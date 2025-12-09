#!/usr/bin/env python3
# SCRIPTNAME: ok.interactive_sunday_gtc_pricer.V2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
interactive_sunday_gtc_pricer.py

Fully standalone, interactive Sunday–GTC options pricer:

- Prompts the user for all parameters
- Defaults to:
    * nearest expiration (auto)
    * nearest ATM strike (auto)
    * IV from option-chain mid (auto)
- Provides full Plotly dashboards
- Zero dependencies on STEGO or prior libraries
"""

import yfinance as yf
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import webbrowser

# --------------------------------------------------------------------
# Black-Scholes utilities
# --------------------------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_d1_d2(S, K, T, r, q, sigma):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return d1, d2

def bs_price(S, K, T, r, q, sigma, typ):
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    if typ == "call":
        return S*math.exp(-q*T)*norm_cdf(d1) - K*math.exp(-r*T)*norm_cdf(d2)
    else:
        return K*math.exp(-r*T)*norm_cdf(-d2) - S*math.exp(-q*T)*norm_cdf(-d1)

def bs_greeks(S, K, T, r, q, sigma, typ):
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    pdf = (1/math.sqrt(2*math.pi))*math.exp(-0.5*d1*d1)
    delta = math.exp(-q*T)*norm_cdf(d1) if typ=="call" else math.exp(-q*T)*(norm_cdf(d1)-1)
    gamma = (math.exp(-q*T)*pdf)/(S*sigma*math.sqrt(T))
    vega  = S*math.exp(-q*T)*pdf*math.sqrt(T)
    theta = -(S*math.exp(-q*T)*pdf*sigma)/(2*math.sqrt(T))
    if typ=="call":
        theta += -r*K*math.exp(-r*T)*norm_cdf(d2) + q*S*math.exp(-q*T)*norm_cdf(d1)
    else:
        theta += r*K*math.exp(-r*T)*norm_cdf(-d2) - q*S*math.exp(-q*T)*norm_cdf(-d1)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def next_monday():
    today = date.today()
    days = (0 - today.weekday()) % 7
    return today + timedelta(days=days if days else 7)

def choose_default_exp_and_strike(ticker, spot, monday):
    tk = yf.Ticker(ticker)
    try:
        exps = tk.options
        if not exps:
            raise ValueError("No options found")
    except Exception:
        # Fallback if yfinance fails or no options
        default_exp = monday + timedelta(days=7)
        return default_exp, default_exp.isoformat(), round(spot, 0)

    exp_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in exps]
    after = [d for d in exp_dates if d > monday]
    exp = min(after) if after else min(exp_dates)
    exp_str = exp.isoformat()

    try:
        chain = tk.option_chain(exp_str)
        # Check if calls exist
        if chain.calls.empty:
            return exp, exp_str, round(spot, 0)
        
        strikes = chain.calls["strike"].values
        strike = float(strikes[np.argmin(abs(strikes - spot))])
    except:
        strike = round(spot, 0)
        
    return exp, exp_str, strike

def auto_iv_from_chain(ticker, expiry_str, strike, typ):
    tk = yf.Ticker(ticker)
    try:
        chain = tk.option_chain(expiry_str)
        df = chain.calls if typ=="call" else chain.puts
        row = df[df["strike"] == strike]
        
        if row.empty:
            return 0.20
            
        # FIX: Use .iloc[0] to access value before float conversion
        bid = float(row["bid"].iloc[0])
        ask = float(row["ask"].iloc[0])
        mid = (bid+ask)/2 if bid>0 and ask>0 else ask or bid
        return max(0.05, min(1.5, mid/strike))
    except:
        return 0.20


# --------------------------------------------------------------------
# Visualizations
# --------------------------------------------------------------------

def build_dashboard(summary):
    figs = []

    # Summary Table
    df = pd.DataFrame(list(summary.items()), columns=["Metric","Value"])
    figs.append(go.Figure(data=[go.Table(
        header=dict(values=["Metric","Value"]),
        cells=dict(values=[df["Metric"], df["Value"]])
    )]))

    # Price vs Spot
    S0 = summary["spot"]
    K  = summary["strike"]
    T  = summary["T"]
    r  = summary["r"]
    q  = summary["q"]
    iv = summary["iv"]
    iv_up = iv + summary["iv_shift"]
    typ = summary["type"]

    xs = np.linspace(0.7*S0, 1.3*S0, 200)
    p_base = [bs_price(x,K,T,r,q,iv,typ) for x in xs]
    p_up   = [bs_price(x,K,T,r,q,iv_up,typ) for x in xs]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=xs,y=p_base,name=f"IV {iv:.1%}"))
    fig1.add_trace(go.Scatter(x=xs,y=p_up, name=f"IV {iv_up:.1%}", line=dict(dash="dash")))
    fig1.update_layout(title="Projected Price vs Spot", xaxis_title="Spot Price", yaxis_title="Option Price")
    figs.append(fig1)

    # Greeks
    deltas=[]
    gammas=[]
    vegas=[]
    for s in xs:
        g=bs_greeks(s,K,T,r,q,iv,typ)
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        vegas.append(g["vega"])

    fig2 = make_subplots(rows=3,cols=1,shared_xaxes=True, subplot_titles=("Delta", "Gamma", "Vega"))
    fig2.add_trace(go.Scatter(x=xs,y=deltas,name="Delta"),row=1,col=1)
    fig2.add_trace(go.Scatter(x=xs,y=gammas,name="Gamma"),row=2,col=1)
    fig2.add_trace(go.Scatter(x=xs,y=vegas ,name="Vega" ),row=3,col=1)
    fig2.update_layout(title="Option Greeks vs Spot", height=800)
    figs.append(fig2)

    return figs


# --------------------------------------------------------------------
# MAIN INTERACTIVE ENTRY
# --------------------------------------------------------------------

def main():
    print("\n=== Sunday GTC Interactive Pricer ===\n")

    ticker = input("Ticker (e.g. SPY): ").upper().strip()
    while not ticker:
        ticker = input("Ticker: ").upper().strip()

    typ = input("Option Type [call/put] (default=call): ").lower().strip() or "call"

    # Spot
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            raise ValueError("No history found")
        spot = hist["Close"].iloc[-1]
    except:
        spot_in = input("Could not fetch spot. Enter Spot Price manually: ")
        spot = float(spot_in)

    monday = next_monday()

    # Expiry default
    exp_in = input("Expiry (YYYY-MM-DD, default=nearest): ").strip()
    if exp_in:
        expiry = datetime.strptime(exp_in, "%Y-%m-%d").date()
        expiry_str = exp_in
    else:
        expiry, expiry_str, _ = choose_default_exp_and_strike(ticker, spot, monday)

    # Strike default
    strike_in = input("Strike (default=ATM): ").strip()
    if strike_in:
        strike = float(strike_in)
    else:
        _, _, strike = choose_default_exp_and_strike(ticker, spot, monday)

    # IV default
    iv_in = input("Base IV (decimal, default from chain): ").strip()
    if iv_in:
        iv = float(iv_in)
    else:
        iv = auto_iv_from_chain(ticker, expiry_str, strike, typ)

    # IV shift
    shift_in = input("IV Shift for Monday (default=0.02): ").strip()
    iv_shift = float(shift_in) if shift_in else 0.02

    # Side
    side = input("Side [buy/sell] (default=buy): ").lower().strip() or "buy"

    # Buffer
    buf_in = input("Limit buffer (default=0.05): ").strip()
    buffer = float(buf_in) if buf_in else 0.05

    # Rates
    r = 0.04
    q = 0.0

    # Time to expiry
    T = (expiry - monday).days / 365
    if T <= 0: T = 0.001 # Prevent division by zero if exp is same day

    # Two-scenario price
    price_base = bs_price(spot,strike,T,r,q,iv,typ)
    price_up   = bs_price(spot,strike,T,r,q,iv+iv_shift,typ)
    mid = 0.5*(price_base + price_up)

    suggested = mid - buffer if side=="buy" else mid + buffer

    summary = {
        "ticker": ticker,
        "type": typ,
        "spot": spot,
        "strike": strike,
        "expiry": expiry_str,
        "monday": monday.isoformat(),
        "iv": iv,
        "iv_shift": iv_shift,
        "r": r,
        "q": q,
        "T": T,
        "price_base": price_base,
        "price_up": price_up,
        "modeled_mid": mid,
        "suggested_limit": suggested,
        "side": side
    }

    print("\n=== RESULTS ===")
    for k,v in summary.items():
        print(f"{k}: {v}")

    figs = build_dashboard(summary)

    html = f"sunday_gtc_{ticker}.html"
    
    # FIX: Write multiple figures to a single HTML file manually
    # ploty.io.write_html does NOT support a list of figures directly
    with open(html, 'w') as f:
        f.write("<html><head><title>Sunday GTC Dashboard</title></head><body>")
        f.write(f"<h1>Analysis for {ticker}</h1>")
        for i, fig in enumerate(figs):
            # include_plotlyjs='cdn' creates a smaller file, but only needed once per page
            # technically, though putting it on every div is safer if they are isolated
            f.write(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
        f.write("</body></html>")

    print(f"\nDashboard saved to {html}\n")
    
    # Auto open logic
    try:
        webbrowser.open('file://' + os.path.realpath(html))
    except Exception as e:
        print(f"Could not auto-open browser: {e}")


if __name__ == "__main__":
    main()
