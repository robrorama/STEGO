#!/usr/bin/env python3
"""
interactive_sunday_gtc_pricer.py

Single-figure (subplot) Sunday GTC pricer with interactive prompting,
auto-expiry detection, auto-ATM strike, auto-IV estimation, and one
giant Plotly dashboard.

No dependencies beyond: yfinance, numpy, pandas, plotly
"""

import yfinance as yf
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# --------------------------------------------------------------------
# Black–Scholes utilities
# --------------------------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_d1_d2(S, K, T, r, q, sigma):
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
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
    # Theta per YEAR
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
    expiries = tk.options
    exp_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in expiries]
    after = [d for d in exp_dates if d > monday]
    expiry = min(after) if after else min(exp_dates)
    expiry_str = expiry.isoformat()

    chain = tk.option_chain(expiry_str)
    strikes = chain.calls["strike"].values
    strike = float(strikes[np.argmin(abs(strikes - spot))])
    return expiry, expiry_str, strike

def auto_iv_from_chain(ticker, expiry_str, strike, typ):
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry_str)
    df = chain.calls if typ=="call" else chain.puts
    row = df[df["strike"] == strike]
    if row.empty:
        return 0.20
    bid = row["bid"].iloc[0]
    ask = row["ask"].iloc[0]
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    else:
        mid = max(bid, ask)
    if mid <= 0:
        return 0.20
    return max(0.05, min(1.5, mid/strike))


# --------------------------------------------------------------------
# Build SINGLE dashboard figure with subplots
# --------------------------------------------------------------------

def build_master_figure(summary):

    S0 = summary["spot"]
    K  = summary["strike"]
    T  = summary["T"]
    r  = summary["r"]
    q  = summary["q"]
    iv = summary["iv"]
    iv_up = iv + summary["iv_shift"]
    typ = summary["type"]
    days_total = summary["days_to_expiry"]

    xs = np.linspace(0.7*S0, 1.3*S0, 200)
    p_base = [bs_price(x, K, T, r, q, iv, typ) for x in xs]
    p_up   = [bs_price(x, K, T, r, q, iv_up, typ) for x in xs]

    deltas = []
    gammas = []
    vegas  = []
    for s in xs:
        g = bs_greeks(s, K, T, r, q, iv, typ)
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        vegas.append(g["vega"])

    # Theta decay curve
    days = np.linspace(0, days_total, 60)
    theta_prices = []
    for d in days:
        T_t = max(1e-6, (days_total-d)/365)
        theta_prices.append(bs_price(S0, K, T_t, r, q, iv, typ))

    # Create the master subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "table"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        subplot_titles=[
            "Summary Table", "Price vs Spot (Base + IV Shift)",
            "Delta vs Spot", "Gamma vs Spot",
            "Vega vs Spot", "Theta Decay Curve"
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.10
    )

    # Summary Table block
    df = pd.DataFrame([
        ["Ticker", summary["ticker"]],
        ["Type", summary["type"]],
        ["Side", summary["side"]],
        ["Spot", S0],
        ["Strike", K],
        ["Expiry", summary["expiry"]],
        ["Monday", summary["monday"]],
        ["Days→Exp", days_total],
        ["Base IV", iv],
        ["IV Up", iv_up],
        ["Price Base", summary["price_base"]],
        ["Price IV Up", summary["price_up"]],
        ["Modeled Mid", summary["modeled_mid"]],
        ["Suggested Limit", summary["suggested_limit"]],
    ], columns=["Metric","Value"])

    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"]),
            cells=dict(values=[df["Metric"], df["Value"]])
        ),
        row=1, col=1
    )

    # Price vs spot
    fig.add_trace(
        go.Scatter(x=xs, y=p_base, name=f"IV {iv:.1%}"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=xs, y=p_up, name=f"IV {iv_up:.1%}", line=dict(dash="dash")),
        row=1, col=2
    )

    # Delta
    fig.add_trace(go.Scatter(x=xs, y=deltas, name="Delta"), row=2, col=1)

    # Gamma
    fig.add_trace(go.Scatter(x=xs, y=gammas, name="Gamma"), row=2, col=2)

    # Vega
    fig.add_trace(go.Scatter(x=xs, y=vegas, name="Vega"), row=3, col=1)

    # Theta decay
    fig.add_trace(
        go.Scatter(x=days, y=theta_prices, name="Theta Price Path"),
        row=3, col=2
    )

    fig.update_layout(
        height=1600,
        width=2000,
        title_text="Sunday GTC Option Dashboard (Single Figure)",
        showlegend=False
    )

    return fig


# --------------------------------------------------------------------
# MAIN INTERACTIVE ENTRY
# --------------------------------------------------------------------

def main():
    print("\n=== Sunday GTC Interactive Pricer ===\n")

    ticker = input("Ticker (e.g. SPY): ").upper().strip()
    while not ticker:
        ticker = input("Ticker: ").upper().strip()

    typ = input("Option Type [call/put] (default=call): ").strip().lower() or "call"

    hist = yf.Ticker(ticker).history(period="1d")
    spot = float(hist["Close"].iloc[-1])

    monday = next_monday()

    exp_in = input("Expiry (YYYY-MM-DD, default=nearest): ").strip()
    if exp_in:
        expiry = datetime.strptime(exp_in, "%Y-%m-%d").date()
        expiry_str = exp_in
    else:
        expiry, expiry_str, _ = choose_default_exp_and_strike(ticker, spot, monday)

    strike_in = input("Strike (default=ATM): ").strip()
    if strike_in:
        strike = float(strike_in)
    else:
        _, _, strike = choose_default_exp_and_strike(ticker, spot, monday)

    iv_in = input("Base IV (decimal, default from chain): ").strip()
    if iv_in:
        iv = float(iv_in)
    else:
        iv = auto_iv_from_chain(ticker, expiry_str, strike, typ)

    shift_in = input("IV Shift for Monday (default=0.02): ").strip()
    iv_shift = float(shift_in) if shift_in else 0.02

    side = input("Side [buy/sell] (default=buy): ").strip().lower() or "buy"

    buf_in = input("Limit buffer (default=0.05): ").strip()
    buffer = float(buf_in) if buf_in else 0.05

    r = 0.04
    q = 0.0

    days_to_exp = (expiry - monday).days
    T = days_to_exp / 365

    price_base = bs_price(spot, strike, T, r, q, iv, typ)
    price_up   = bs_price(spot, strike, T, r, q, iv+iv_shift, typ)
    mid = 0.5*(price_base + price_up)

    suggested = mid - buffer if side=="buy" else mid + buffer

    summary = {
        "ticker": ticker,
        "type": typ,
        "side": side,
        "spot": spot,
        "strike": strike,
        "expiry": expiry_str,
        "monday": monday.isoformat(),
        "days_to_expiry": days_to_exp,
        "T": T,
        "iv": iv,
        "iv_shift": iv_shift,
        "r": r,
        "q": q,
        "price_base": price_base,
        "price_up": price_up,
        "modeled_mid": mid,
        "suggested_limit": suggested
    }

    print("\n=== RESULTS SUMMARY ===\n")
    for k,v in summary.items():
        print(f"{k}: {v}")

    fig = build_master_figure(summary)

    out = f"sunday_gtc_{ticker}.html"
    pio.write_html(fig, file=out, auto_open=True)
    print(f"\nDashboard saved to {out}\n")


if __name__ == "__main__":
    main()

