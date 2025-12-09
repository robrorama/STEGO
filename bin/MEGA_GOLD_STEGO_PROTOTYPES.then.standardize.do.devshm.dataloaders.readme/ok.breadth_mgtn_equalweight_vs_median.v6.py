#!/usr/bin/env python3
# SCRIPTNAME: breadth_mgtn_equalweight_vs_median.v6.py
#
# BREADTH PACK A + B + C (FINAL v6)
#   Pack A: EQW, MED, MW, spreads
#   Pack B: EMA breadth (20/50/200), AD line, constituents
#   Pack C: McClellan Osc / Summation, NH/NL, Regime classifier,
#           BVIX breadth volatility, Breadth RSI suite
#
#   Fully patched universal loader supports ETFs like IBIT, BRRR, BTCO, HODL.
#
# USAGE:
#   python3 breadth_mgtn_equalweight_vs_median.v6.py
#   python3 breadth_mgtn_equalweight_vs_median.v6.py --tickers iwm
#   python3 breadth_mgtn_equalweight_vs_median.v6.py nvda msft
#   python3 breadth_mgtn_equalweight_vs_median.v6.py --tickers ibit
#
# OUTPUT:
#   /dev/shm/BREADTH_MGTN/YYYY-MM-DD/index.html + tab html files

import os
import sys
import argparse
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import webbrowser

OUTROOT = "/dev/shm/BREADTH_MGTN"
PRICE_FIELD = "Adj Close"
START = "2020-01-01"

DEFAULT_MGTN = [
    "NVDA","MSFT","AAPL","GOOGL","AMZN",
    "META","TSLA","AVGO","NFLX","AMD"
]

# -----------------------------------------------------
# UNIVERSAL BULLET-PROOF YFINANCE LOADER (v6)
# -----------------------------------------------------
def load_prices(tickers, start=START):
    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)

    if raw is None or raw.empty:
        raise RuntimeError("No data returned from yfinance")

    # -------- MULTI-INDEX CASE --------
    if isinstance(raw.columns, pd.MultiIndex):

        # Preferred price fields
        candidates = ["Adj Close", "Close", "Last Price", "Last", "Price"]

        lvl1 = raw.columns.get_level_values(1)

        # Try known price fields
        for c in candidates:
            if c in lvl1:
                df = raw.xs(c, level=1, axis=1)
                return df.ffill().dropna(how="all")

        # Fallback: per-ticker numeric column extraction
        fallback = {}
        tickers_lvl0 = np.unique(raw.columns.get_level_values(0))
        for t in tickers_lvl0:
            sub = raw[t]
            num_cols = [col for col in sub.columns if pd.api.types.is_numeric_dtype(sub[col])]
            if len(num_cols) > 0:
                fallback[t] = sub[num_cols[0]]
        if not fallback:
            raise RuntimeError("No numeric price-like column found for MultiIndex dataset")
        return pd.DataFrame(fallback).ffill().dropna(how="all")

    # -------- SINGLE-TICKER CASE --------
    else:
        for c in ["Adj Close", "Close", "Last Price", "Last", "Price"]:
            if c in raw.columns:
                px = raw[c]
                colname = tickers if isinstance(tickers, str) else tickers[0]
                return px.to_frame(colname).ffill().dropna(how="all")

        # fallback to first numeric column
        num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if not num_cols:
            raise RuntimeError("No numeric price-like column found for single ticker")
        px = raw[num_cols[0]]
        colname = tickers if isinstance(tickers, str) else tickers[0]
        return px.to_frame(colname).ffill().dropna(how="all")

# -----------------------------------------------------
# BREADTH MATH HELPERS
# -----------------------------------------------------
def normalize(df):
    base = df.ffill().bfill().iloc[0]
    return df.divide(base) * 100.0

def slope(series, lookback=10):
    if len(series) < lookback:
        return np.nan
    y = series.iloc[-lookback:].values
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])

def rolling_slope(series, window=20):
    def _sl(s):
        x = np.arange(len(s))
        return float(np.polyfit(x, s.values, 1)[0])
    return series.rolling(window, min_periods=window).apply(_sl, raw=False)

def fetch_market_caps(tickers):
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).fast_info.market_cap
        except:
            caps[t] = None
    return caps

def market_weight_index(normdf):
    caps = fetch_market_caps(normdf.columns)
    caps_clean = {k:(v if isinstance(v,(int,float)) and v and v>0 else 1) for k,v in caps.items()}
    w = np.array(list(caps_clean.values()), dtype=float)
    w = w / w.sum()
    return (normdf * w).sum(axis=1)

def pct_above_ema(px, span):
    ema = px.ewm(span=span).mean()
    pct = (px > ema).sum(axis=1) / px.shape[1] * 100.0
    return pct

def ad_line(px):
    r = px.pct_change().fillna(0)
    adv = (r > 0).sum(axis=1)
    dec = (r < 0).sum(axis=1)
    ad = adv - dec
    return ad, ad.cumsum()

def mcclellan(ad, n1=19, n2=39):
    ema1 = ad.ewm(span=n1).mean()
    ema2 = ad.ewm(span=n2).mean()
    mo = ema1 - ema2
    msi = mo.cumsum()
    return mo, msi

def new_highs_lows(px, lookback=252, min_periods=50):
    prev_max = px.shift(1).rolling(lookback, min_periods=min_periods).max()
    prev_min = px.shift(1).rolling(lookback, min_periods=min_periods).min()
    nh = (px > prev_max)
    nl = (px < prev_min)
    pct_nh = nh.sum(axis=1) / px.shape[1] * 100
    pct_nl = nl.sum(axis=1) / px.shape[1] * 100
    return pct_nh, pct_nl, pct_nh - pct_nl

def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

def breadth_vol_index(ad, pct20, eqw):
    ad_vol = ad.rolling(20).std()
    pct20_vol = pct20.rolling(20).std()
    eqw_vol = eqw.pct_change().rolling(20).std()
    return (zscore(ad_vol) + zscore(pct20_vol) + zscore(eqw_vol)) / 3

def rsi(s, period=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Breadth Packs A+B+C Dashboard v6")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("positional", nargs="*")
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    elif args.positional:
        tickers = args.positional
    else:
        tickers = DEFAULT_MGTN

    tickers = [t.upper() for t in tickers]

    asof = dt.date.today().isoformat()
    outdir = os.path.join(OUTROOT, asof)
    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Loading data for: {tickers}")
    px = load_prices(tickers, START)
    norm = normalize(px)

    # ---------- Pack A ----------
    EQW = norm.mean(axis=1)
    MED = norm.median(axis=1)
    MW = market_weight_index(norm)

    spread_med = EQW - MED
    spread_mw = EQW - MW

    # ---------- Pack B ----------
    pct20 = pct_above_ema(px, 20)
    pct50 = pct_above_ema(px, 50)
    pct200 = pct_above_ema(px, 200)

    ad, ad_cum = ad_line(px)

    # ---------- Pack C ----------
    mo, msi = mcclellan(ad)
    pct_nh, pct_nl, nh_minus_nl = new_highs_lows(px)

    eqw_sl = rolling_slope(EQW)
    med_sl = rolling_slope(MED)
    ad_sl = rolling_slope(ad_cum)

    bvix = breadth_vol_index(ad, pct20, EQW)

    rsi_eqw = rsi(EQW)
    rsi_mw = rsi(MW)
    rsi_ad = rsi(ad_cum)
    rsi_breadth50 = rsi(pct50)

    bullish = (EQW > MW) & (eqw_sl > 0) & (med_sl > 0) & (ad_sl > 0) & (pct50 > 50)
    bearish = (EQW < MW) & (eqw_sl < 0) & (med_sl < 0) & (ad_sl < 0) & (pct50 < 40)

    regime = pd.Series(1, index=EQW.index, dtype=float)
    regime[bullish] = 2
    regime[bearish] = 0

    # ---------- Save CSV ----------
    df = pd.DataFrame({
        "EQW": EQW,
        "MW": MW,
        "MED": MED,
        "SPREAD_EQW_MED": spread_med,
        "SPREAD_EQW_MW": spread_mw,
        "PCT>EMA20": pct20,
        "PCT>EMA50": pct50,
        "PCT>EMA200": pct200,
        "AD": ad,
        "AD_CUM": ad_cum,
        "MO": mo,
        "MSI": msi,
        "PCT_NH": pct_nh,
        "PCT_NL": pct_nl,
        "NH_MINUS_NL": nh_minus_nl,
        "BVIX": bvix,
        "RSI_EQW": rsi_eqw,
        "RSI_MW": rsi_mw,
        "RSI_AD": rsi_ad,
        "RSI_BREADTH50": rsi_breadth50,
        "REGIME": regime
    })

    df.to_csv(os.path.join(outdir, "breadth_v6_all.csv"))

    # -----------------------------------------------------
    # MULTI-TAB OUTPUT
    # -----------------------------------------------------
    tabs = []

    def add_tab(name, fig):
        tabs.append((name, fig))
        fig.write_html(
            os.path.join(outdir, f"{name}.html"),
            include_plotlyjs="cdn"
        )

    # TAB 1: EQW/MW/MED
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=EQW.index, y=EQW, name="EQW"))
    f1.add_trace(go.Scatter(x=MW.index, y=MW, name="MW"))
    f1.add_trace(go.Scatter(x=MED.index, y=MED, name="MED"))
    f1.update_layout(title="EQW vs MW vs Median")
    add_tab("EQW_MW_Median", f1)

    # TAB 2: Spreads
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=spread_med.index, y=spread_med, name="EQW-MED"))
    f2.add_trace(go.Scatter(x=spread_mw.index, y=spread_mw, name="EQW-MW"))
    f2.add_hline(y=0)
    f2.update_layout(title="Spreads")
    add_tab("Spreads", f2)

    # TAB 3: EMA Breadth
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=pct20.index, y=pct20, name="%>EMA20"))
    f3.add_trace(go.Scatter(x=pct50.index, y=pct50, name="%>EMA50"))
    f3.add_trace(go.Scatter(x=pct200.index, y=pct200, name="%>EMA200"))
    f3.update_layout(title="EMA Breadth (20/50/200)")
    add_tab("EMA_Breadth", f3)

    # TAB 4: AD Line
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=ad.index, y=ad, name="AD"))
    f4.add_trace(go.Scatter(x=ad_cum.index, y=ad_cum, name="AD CUM"))
    f4.update_layout(title="Advance-Decline")
    add_tab("AD_Line", f4)

    # TAB 5: Constituents normalized
    f5 = go.Figure()
    for c in norm.columns:
        f5.add_trace(go.Scatter(x=norm.index, y=norm[c], name=c, opacity=0.4))
    f5.update_layout(title="Constituent Normalized (100=base)")
    add_tab("Constituents", f5)

    # TAB 6: McClellan
    f6 = go.Figure()
    f6.add_trace(go.Scatter(x=mo.index, y=mo, name="MO"))
    f6.add_trace(go.Scatter(x=msi.index, y=msi, name="MSI"))
    f6.update_layout(title="McClellan Oscillator / Summation")
    add_tab("McClellan", f6)

    # TAB 7: New Highs / New Lows
    f7 = go.Figure()
    f7.add_trace(go.Scatter(x=pct_nh.index, y=pct_nh, name="New Highs"))
    f7.add_trace(go.Scatter(x=pct_nl.index, y=pct_nl, name="New Lows"))
    f7.add_trace(go.Scatter(x=nh_minus_nl.index, y=nh_minus_nl, name="NH-NL"))
    f7.update_layout(title="New Highs / New Lows")
    add_tab("NH_NL", f7)

    # TAB 8: Regime heatmap
    f8 = go.Figure(data=go.Heatmap(
        x=regime.index,
        y=["Regime"],
        z=np.expand_dims(regime.values, axis=0)
    ))
    f8.update_layout(title="Regime (0 Bear, 1 Neutral, 2 Bull)")
    add_tab("Regime", f8)

    # TAB 9: BVIX
    f9 = go.Figure()
    f9.add_trace(go.Scatter(x=bvix.index, y=bvix, name="BVIX"))
    f9.update_layout(title="Breadth Volatility Index (BVIX)")
    add_tab("BVIX", f9)

    # TAB 10: Breadth RSI
    f10 = go.Figure()
    f10.add_trace(go.Scatter(x=rsi_eqw.index, y=rsi_eqw, name="RSI EQW"))
    f10.add_trace(go.Scatter(x=rsi_mw.index, y=rsi_mw, name="RSI MW"))
    f10.add_trace(go.Scatter(x=rsi_ad.index, y=rsi_ad, name="RSI AD_CUM"))
    f10.add_trace(go.Scatter(x=rsi_breadth50.index, y=rsi_breadth50, name="RSI %>EMA50"))
    f10.update_layout(title="Breadth RSI Suite")
    add_tab("RSI", f10)

    # Index
    index_html = "<h1>Breadth Packs A+B+C Dashboard (v6)</h1><ul>"
    for name, _ in tabs:
        index_html += f"<li><a href='{name}.html' target='_blank'>{name}</a></li>"
    index_html += "</ul>"

    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(index_html)

    print(f"[INFO] Dashboard created: {os.path.join(outdir, 'index.html')}")
    try:
        webbrowser.open_new_tab("file://" + os.path.join(outdir, "index.html"))
    except:
        pass


if __name__ == "__main__":
    main()

