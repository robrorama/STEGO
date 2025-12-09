#!/usr/bin/env python3
# SCRIPTNAME: ok.vol_skew_rr25_bf25_vix_termstructure.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

# FIXES:
# - Replaced illegal f-string conditional in plot_price_and_vix hovertext
# - Added safe formatter fmt() to avoid ValueError: invalid format specifier
# - Everything else unchanged

import argparse
import logging
import os
import sys
import math
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

try:
    import data_retrieval as dr
except ImportError:
    print("ERROR: data_retrieval.py missing", file=sys.stderr)
    raise

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)

def BASE_OPTIONS_PATH() -> str:
    return _env("BASE_OPTIONS_PATH", "/dev/shm/options")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_d1(S,K,T,r,q,sigma):
    if S<=0 or K<=0 or T<=0 or sigma<=0: return 0.0
    try:
        return (math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    except:
        return 0.0

def call_delta(S,K,T,r,q,sigma):
    d1 = bs_d1(S,K,T,r,q,sigma)
    return math.exp(-q*T)*norm_cdf(d1)

def put_delta(S,K,T,r,q,sigma):
    d1 = bs_d1(S,K,T,r,q,sigma)
    return -math.exp(-q*T)*norm_cdf(-d1)

def _options_cache_path(t,e):
    base = BASE_OPTIONS_PATH()
    ensure_dir(os.path.join(base,t))
    return os.path.join(base,t,f"{e}.pkl")

def load_or_download_option_chain(ticker, expiry_str, refresh=False):
    cp = _options_cache_path(ticker, expiry_str)
    if (not refresh) and os.path.exists(cp):
        try:
            d = pd.read_pickle(cp)
            return d["calls"], d["puts"]
        except: pass

    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry_str)
    calls = chain.calls.copy()
    puts  = chain.puts.copy()
    for df in (calls,puts):
        if "impliedVolatility" not in df.columns:
            df["impliedVolatility"]=np.nan
    pd.to_pickle({"calls":calls,"puts":puts}, cp)
    return calls, puts

def compute_rr25_bf25_for_expiry(calls, puts, spot, exp_date, today, r=0.05, q=0.0):
    days = (exp_date - today).days
    if days<=0: return None
    T = days/365

    calls = calls.dropna(subset=["strike","impliedVolatility"])
    puts  = puts.dropna(subset=["strike","impliedVolatility"])
    calls = calls[calls["impliedVolatility"]>0]
    puts  = puts[puts["impliedVolatility"]>0]
    if calls.empty or puts.empty: return None

    calls = calls.copy()
    puts  = puts.copy()

    calls["delta"] = [call_delta(spot,float(K),T,r,q,float(iv))
                      for K,iv in zip(calls["strike"],calls["impliedVolatility"])]
    puts["delta"]  = [put_delta (spot,float(K),T,r,q,float(iv))
                      for K,iv in zip(puts["strike"], puts["impliedVolatility"])]

    calls["d25"] = (calls["delta"]-0.25).abs()
    puts ["d25"] = (puts ["delta"]+0.25).abs()
    call_25 = calls.sort_values("d25").iloc[0]
    put_25  = puts.sort_values ("d25").iloc[0]

    calls["datm"] = (calls["delta"]-0.5).abs()
    atm_call = calls.sort_values("datm").iloc[0]

    s25c = float(call_25["impliedVolatility"])
    s25p = float(put_25["impliedVolatility"])
    satm = float(atm_call["impliedVolatility"])

    rr25 = s25c - s25p
    bf25 = 0.5*(s25c+s25p) - satm

    return {
        "expiry":exp_date, "days_to_expiry":days, "T_years":T,
        "sigma_25c":s25c, "sigma_25p":s25p, "sigma_atm":satm,
        "RR25":rr25, "BF25":bf25,
        "strike_25c":float(call_25["strike"]),
        "strike_25p":float(put_25 ["strike"]),
        "strike_atm":float(atm_call["strike"]),
        "delta_25c":float(call_25["delta"]),
        "delta_25p":float(put_25 ["delta"]),
        "delta_atm":float(atm_call["delta"])
    }

def compute_rr25_bf25_term_structure(ticker, max_exp=8, r=0.05, q=0.0):
    spot_df = dr.load_or_download_ticker(ticker, period="5d")
    spot = float(spot_df["Close"].iloc[-1])
    today = spot_df.index[-1].date()

    tk = yf.Ticker(ticker)
    exps = tk.options[:max_exp]
    out=[]
    for e in exps:
        try: ed = datetime.strptime(e,"%Y-%m-%d").date()
        except: continue
        calls,puts = load_or_download_option_chain(ticker,e)
        row = compute_rr25_bf25_for_expiry(calls,puts,spot,ed,today,r,q)
        if row: out.append(row)

    if not out: return pd.DataFrame()
    df = pd.DataFrame(out).set_index(pd.to_datetime([r["expiry"] for r in out]))
    return df

def load_vix_family(period="1y"):
    v9  = dr.load_or_download_ticker("^VIX9D", period=period)
    vix = dr.load_or_download_ticker("^VIX",   period=period)
    v3  = dr.load_or_download_ticker("^VIX3M", period=period)

    df = pd.DataFrame()
    df = df.join(v9 ["Close"].rename("VIX9D"), how="outer")
    df = df.join(vix["Close"].rename("VIX"),   how="outer")
    df = df.join(v3 ["Close"].rename("VIX3M"), how="outer")
    df = df.sort_index()

    df["regime"] = np.where(df["VIX3M"]>df["VIX9D"],"contango",
                      np.where(df["VIX3M"]<df["VIX9D"],"backwardation","flat"))
    return df

# ---------------- FIX: safe formatter ---------------- #
def fmt(x):
    if isinstance(x,(float,int,np.floating,np.integer)):
        try: return f"{x:.2f}"
        except: return str(x)
    return ""

def plot_price_and_vix(ticker, price_df, vix_df, out_html, open_browser=True):

    vix_df = vix_df.loc[price_df.index.min():price_df.index.max()] if not vix_df.empty else vix_df

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7,0.3], vertical_spacing=0.03,
        specs=[[{"type":"candlestick"}],[{"type":"scatter"}]]
    )

    fig.add_trace(go.Candlestick(
        x=price_df.index, open=price_df["Open"], high=price_df["High"],
        low=price_df["Low"], close=price_df["Close"], name=f"{ticker} OHLC",
        showlegend=False
    ),1,1)

    for col,name in [("VIX9D","VIX9D"),("VIX","VIX"),("VIX3M","VIX3M")]:
        if col in vix_df.columns:
            hover=[f"Date: {d.date()}<br>{name}: {fmt(v)}<br>Regime: {r}"
                   for d,v,r in zip(vix_df.index, vix_df[col], vix_df["regime"])]
            fig.add_trace(go.Scatter(
                x=vix_df.index, y=vix_df[col], mode="lines",
                name=name, hovertext=hover, hoverinfo="text"
            ),2,1)

    fig.update_layout(
        title=f"{ticker} Price + VIX Curve",
        hovermode="x unified",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1)
    )

    ensure_dir(os.path.dirname(out_html))
    fig.write_html(out_html, auto_open=False)
    if open_browser:
        import webbrowser
        webbrowser.open("file://"+os.path.abspath(out_html))

def plot_rr25_bf25_term_structure(ticker,df,out_html,open_browser=True):
    if df.empty: return
    d=df.copy()
    d["expiry"]=d.index.date
    d["days_to_expiry"]=d["days_to_expiry"].astype(int)

    fig=make_subplots(rows=1,cols=1,specs=[[{"secondary_y":True}]])
    cd=np.stack([
        d["expiry"].astype(str), d["sigma_25c"], d["sigma_25p"],
        d["sigma_atm"], d["strike_25c"], d["strike_25p"],
        d["strike_atm"], d["delta_25c"], d["delta_25p"], d["delta_atm"]
    ],axis=-1)

    fig.add_trace(go.Scatter(
        x=d["days_to_expiry"], y=d["RR25"],
        mode="lines+markers", name="RR25",
        customdata=cd,
        hovertemplate="Days: %{x}<br>RR25: %{y:.4f}<br>"
                      "Expiry: %{customdata[0]}<br>"
                      "σ25C: %{customdata[1]:.4f}<br>"
                      "σ25P: %{customdata[2]:.4f}<br>"
                      "σATM: %{customdata[3]:.4f}<br>"
                      "K25C: %{customdata[4]:.2f}<br>"
                      "K25P: %{customdata[5]:.2f}<br>"
                      "KATM: %{customdata[6]:.2f}<br>"
                      "Δ25C: %{customdata[7]:.3f}<br>"
                      "Δ25P: %{customdata[8]:.3f}<br>"
                      "ΔATM: %{customdata[9]:.3f}<extra></extra>"
    ),secondary_y=False)

    cd2=np.stack([
        d["expiry"].astype(str), d["sigma_25c"], d["sigma_25p"], d["sigma_atm"]
    ],axis=-1)

    fig.add_trace(go.Scatter(
        x=d["days_to_expiry"], y=d["BF25"],
        mode="lines+markers", name="BF25",
        customdata=cd2,
        hovertemplate="Days: %{x}<br>BF25: %{y:.4f}<br>"
                      "Expiry: %{customdata[0]}<br>"
                      "σ25C: %{customdata[1]:.4f}<br>"
                      "σ25P: %{customdata[2]:.4f}<br>"
                      "σATM: %{customdata[3]:.4f}<br><extra></extra>"
    ),secondary_y=True)

    fig.update_layout(
        title=f"{ticker} RR25/BF25 Term Structure",
        xaxis_title="Days to Expiry",
        yaxis_title="RR25",
        yaxis2_title="BF25"
    )

    ensure_dir(os.path.dirname(out_html))
    fig.write_html(out_html,auto_open=False)
    if open_browser:
        import webbrowser; webbrowser.open("file://"+os.path.abspath(out_html))

def parse_args(a=None):
    p=argparse.ArgumentParser()
    p.add_argument("--ticker","-t",type=str,default="SPY")
    p.add_argument("--period",type=str,default="1y")
    p.add_argument("--max_expiries",type=int,default=8)
    p.add_argument("--outdir",type=str,default="/dev/shm/plots")
    p.add_argument("--no-open",action="store_true")
    p.add_argument("--risk-free",type=float,default=0.05)
    p.add_argument("--div-yield",type=float,default=0.0)
    p.add_argument("--log-level",type=str,default="INFO")
    return p.parse_args(a)

def main(a=None):
    args=parse_args(a)
    logging.basicConfig(level=args.log_level.upper(),format="%(asctime)s [%(levelname)s] %(message)s")

    t=args.ticker.upper()
    p=args.period
    openb=not args["no_open"] if isinstance(args,dict) else not args.no_open

    price = dr.load_or_download_ticker(t, period=p)
    vix   = load_vix_family(period=p)
    rrdf  = compute_rr25_bf25_term_structure(t, max_exp=args.max_expiries,
                                             r=args.risk_free, q=args.div_yield)

    ensure_dir(args.outdir)

    plot_price_and_vix(
        ticker=t, price_df=price, vix_df=vix,
        out_html=os.path.join(args.outdir,f"{t}_price_vix.html"),
        open_browser=openb
    )

    if not rrdf.empty:
        plot_rr25_bf25_term_structure(
            ticker=t, df=rrdf,
            out_html=os.path.join(args.outdir,f"{t}_rr25_bf25_term.html"),
            open_browser=openb
        )

if __name__=="__main__":
    main()

