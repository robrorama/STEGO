#!/usr/bin/env python3
# SCRIPTNAME: options.gamma_vanna.zscore_pressure.v5.py
# AUTHOR: Michael Derby
# DATE:   November 24, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# PURPOSE:
#   v5 = v4 + FULL PRICE-INTEGRATED DEALER MAP
#
#   Builds EVERYTHING below in separate Plotly HTML tabs:
#
#   1) Core Surfaces (as before)
#      - Gamma Z-Score Heatmap (expiry x ln(K/S))
#      - Vanna Z-Score Heatmap (expiry x ln(K/S))
#      - Blended Pressure Heatmap (expiry x ln(K/S))
#
#   2) Price-Integrated Views (NEW)
#      A) Price History + Expiry Markers
#         - 90-day candlestick chart
#         - Vertical lines for each selected option expiration
#
#      B) ATM Pressure Term Structure
#         - x: expiry (calendar)
#         - y: blended pressure at near-ATM moneyness (|ln(K/S)| < 0.05)
#         - shows where dealer pressure is strongest vs time
#
#      C) 3D Pressure Surface
#         - x: moneyness ln(K/S)
#         - y: expiry
#         - z: blended pressure
#         - Color = pressure; 3D scatter surface
#
#      D) Strike vs Price Pressure Profile (key expiry)
#         - choose expiry near 30D (if available)
#         - x: strike
#         - y: blended pressure
#         - vertical line at spot price
#
#      E) Dealer Book Map
#         - Panel 1: Gamma exposure (expiry x strike)
#         - Panel 2: Vanna exposure (expiry x strike)
#         - Overlays horizontal line at spot (strike dimension)
#
#      F) Delta-Bucket Heatmaps (already present)
#         - expiry x delta_bucket for z_gamma, z_vanna, pressure
#
#   All outputs are stored under:
#       BASE_DATA_PATH/YYYY-MM-DD/MEGA_CORR_V14/OPTIONS_PRESSURE/TICKER/
#
#   And each figure is opened in its own browser tab.
#
#   NO IMAGINARY ARGS:
#       Uses ONLY:
#           options_data_retrieval.get_available_remote_expirations()
#           options_data_retrieval.load_or_download_option_chain()
#           data_retrieval.load_or_download_ticker()
#
#   Greeks (delta, gamma, vanna) are computed internally via Black–Scholes.

import os
import sys
import argparse
import logging
import webbrowser
from datetime import datetime
from math import log, sqrt, exp, pi

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Import user modules
# ------------------------------------------------------------
try:
    import data_retrieval
except Exception as e:
    print("ERROR importing data_retrieval:", e)
    sys.exit(1)

try:
    import options_data_retrieval
except Exception as e:
    print("ERROR importing options_data_retrieval:", e)
    sys.exit(1)

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def ensure_dir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def zscore_series(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isclose(std, 0):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std

def apply_liquidity_mask(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'openInterest' not in df.columns:
        df['openInterest'] = df.get('oi', 0.0)
    if 'volume' not in df.columns:
        df['volume'] = df.get('volume', 0.0)
    if 'bid' not in df.columns:
        df['bid'] = df.get('bid', np.nan)
    if 'ask' not in df.columns:
        df['ask'] = df.get('ask', np.nan)
    df['mask'] = True
    df.loc[df['openInterest'] <= 0, 'mask'] = False
    df.loc[df['volume'] <= 0, 'mask'] = False
    spread = df['ask'] - df['bid']
    spread_ratio = spread / df['ask'].replace(0, np.nan)
    bad_spread = (df['ask'] <= df['bid']) | (spread_ratio > 0.50)
    df.loc[bad_spread, 'mask'] = False
    return df

# ------------------------------------------------------------
# Normal/erf helpers for BS
# ------------------------------------------------------------
def norm_pdf(x: float) -> float:
    return (1.0 / sqrt(2 * pi)) * exp(-0.5 * x * x)

def erf(x: float) -> float:
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-x*x)
    return sign * y

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))

# ------------------------------------------------------------
# Black–Scholes Greeks (Delta, Gamma, Vanna)
# ------------------------------------------------------------
def compute_greeks(row, spot: float, r: float = 0.01):
    K = float(row['strike'])
    T = float(row['T'])
    iv = float(row.get('impliedVolatility', np.nan))

    if np.isnan(iv) or iv <= 0:
        iv = 0.20

    if K <= 0:
        return 0.0, 0.0, 0.0

    T = max(T, 1e-8)

    d1 = (log(spot / K) + (r + 0.5 * iv * iv) * T) / (iv * sqrt(T))
    d2 = d1 - iv * sqrt(T)
    pdf_d1 = norm_pdf(d1)

    opt_type = str(row.get("type", "")).lower()
    if "call" in opt_type:
        delta = norm_cdf(d1)
    else:
        delta = norm_cdf(d1) - 1.0

    gamma = pdf_d1 / (spot * iv * sqrt(T))
    vanna = gamma * (spot * iv * T)  # approximate vanna

    return delta, gamma, vanna

# ------------------------------------------------------------
# Chain → tidy format
# ------------------------------------------------------------
def chain_df_to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in df.iterrows():
        try:
            rt = str(row.get("type", "")).lower()
            if "call" in rt:
                opt_type = "C"
            elif "put" in rt:
                opt_type = "P"
            else:
                opt_type = "?"
            rec = dict(
                expiry=pd.to_datetime(row['expiration']).normalize(),
                strike=float(row['strike']),
                openInterest=float(row.get("openInterest", 0.0)),
                volume=float(row.get("volume", 0.0)),
                bid=float(row.get("bid", np.nan)),
                ask=float(row.get("ask", np.nan)),
                option_type=opt_type,
                gamma=float(row['gamma']),
                vanna=float(row['vanna']),
                delta=float(row['delta'])
            )
            out.append(rec)
        except Exception:
            continue
    return pd.DataFrame(out)

# ------------------------------------------------------------
# Moneyness
# ------------------------------------------------------------
def compute_moneyness(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    df = df.copy()
    df['moneyness'] = np.log(df['strike'] / spot)
    return df

# ------------------------------------------------------------
# Heatmap helpers
# ------------------------------------------------------------
def build_heatmap_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    mat = df.pivot_table(index="expiry", columns="moneyness", values=value_col, aggfunc='mean')
    return mat.sort_index()

def make_heatmap(mat: pd.DataFrame, title: str, outfile: str) -> str:
    fig = px.imshow(mat, aspect="auto", labels={"x":"Moneyness ln(K/S)", "y":"Expiry", "color":title}, title=title)
    pio.write_html(fig, outfile, auto_open=False)
    return outfile

# ------------------------------------------------------------
# Term structure slices (pressure vs expiry by moneyness bucket)
# ------------------------------------------------------------
def build_term_structure_slices(tidy: pd.DataFrame, ticker: str, outdir: str) -> str:
    df = tidy.copy()
    df['m_bucket'] = np.round(df['moneyness'] / 0.2) * 0.2
    df = df[(df['m_bucket'] >= -0.8) & (df['m_bucket'] <= 0.8)]
    if df.empty:
        logging.warning("No data for term-structure slices.")
        return ""
    g = df.groupby(['expiry','m_bucket'])['pressure'].mean().reset_index()
    fig = px.line(
        g,
        x='expiry',
        y='pressure',
        color='m_bucket',
        markers=True,
        title=f"{ticker} Blended Pressure Term Structure (by ln(K/S) bucket)",
        labels={'m_bucket': 'Moneyness Bucket ln(K/S)', 'pressure':'Blended Pressure'}
    )
    outfile = os.path.join(outdir, f"{ticker}_term_structure_pressure.html")
    pio.write_html(fig, outfile, auto_open=False)
    return outfile

# ------------------------------------------------------------
# Smile slice (pressure vs moneyness for ~30D expiry)
# ------------------------------------------------------------
def build_smile_slice(tidy: pd.DataFrame, ticker: str, outdir: str) -> str:
    df = tidy.copy()
    today = pd.Timestamp(datetime.now().date())
    df['dte'] = (df['expiry'] - today).dt.days
    df = df[df['dte'] >= 0]
    if df.empty:
        return ""
    exps = df[['expiry','dte']].drop_duplicates()
    exps['dist'] = (exps['dte'] - 30).abs()
    chosen_row = exps.sort_values(['dist','dte']).iloc[0]
    chosen_expiry = chosen_row['expiry']
    slice_df = df[df['expiry'] == chosen_expiry].copy()
    if slice_df.empty:
        return ""
    fig = px.scatter(
        slice_df,
        x='moneyness',
        y='pressure',
        color='option_type',
        title=f"{ticker} Blended Pressure Smile (Expiry {chosen_expiry.date()} ~{int(chosen_row['dte'])}D)",
        labels={'moneyness':'Moneyness ln(K/S)','pressure':'Blended Pressure'}
    )
    outfile = os.path.join(outdir, f"{ticker}_smile_slice_pressure.html")
    pio.write_html(fig, outfile, auto_open=False)
    return outfile

# ------------------------------------------------------------
# Delta-bucket heatmaps
# ------------------------------------------------------------
def build_delta_bucket_heatmaps(tidy: pd.DataFrame, ticker: str, outdir: str) -> list:
    htmls = []
    if 'delta' not in tidy.columns:
        return htmls
    df = tidy.copy()
    df['delta_bucket'] = np.round(df['delta'] * 10) / 10.0
    for col, label in [('z_gamma','Gamma Z-Score'), ('z_vanna','Vanna Z-Score'), ('pressure','Blended Pressure')]:
        mat = df.pivot_table(index="expiry", columns="delta_bucket", values=col, aggfunc='mean')
        mat = mat.sort_index().sort_index(axis=1)
        fig = px.imshow(
            mat,
            aspect="auto",
            labels={"x":"Delta Bucket","y":"Expiry","color":label},
            title=f"{ticker} {label} (Expiry x Delta Bucket)"
        )
        outfile = os.path.join(outdir, f"{ticker}_{col}_delta_buckets.html")
        pio.write_html(fig, outfile, auto_open=False)
        htmls.append(outfile)
    return htmls

# ------------------------------------------------------------
# A) Price history + expiries
# ------------------------------------------------------------
def build_price_history_with_expiries(ticker: str, spot_df: pd.DataFrame, expirations: list, outdir: str) -> str:
    df = spot_df.sort_index().copy()
    if len(df) > 90:
        df = df.iloc[-90:]
    df = df.reset_index().rename(columns={'index':'Date'})
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    for exp in expirations:
        fig.add_vline(x=exp, line=dict(dash="dash"), opacity=0.5)
    fig.update_layout(
        title=f"{ticker} Price History (90D) with Expiration Markers",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    outfile = os.path.join(outdir, f"{ticker}_price_history_expiries.html")
    pio.write_html(fig, outfile, auto_open=False)
    return outfile

# ------------------------------------------------------------
# B) ATM pressure term structure (near moneyness=0)
# ------------------------------------------------------------
def build_atm_pressure_term(tidy: pd.DataFrame, ticker: str, outdir: str) -> str:
    df = tidy.copy()
    df = df[df['moneyness'].abs() < 0.05]
    if df.empty:
        logging.warning("No near-ATM data for ATM pressure term structure.")
        return ""
    g = df.groupby('expiry')['pressure'].mean().reset_index()
    fig = px.line(
        g,
        x='expiry',
        y='pressure',
        markers=True,
        title=f"{ticker} ATM Blended Pressure Term Structure (|ln(K/S)|<0.05)",
        labels={'expiry':'Expiry','pressure':'ATM Blended Pressure'}
    )
    outfile = os.path.join(outdir, f"{ticker}_atm_pressure_term.html")
    pio.write_html(fig, outfile, auto_open=False)
    return outfile

# ------------------------------------------------------------
# C) 3D pressure surface (expiry, moneyness, pressure)
# ------------------------------------------------------------
def build_pressure_3d_surface(tidy: pd.DataFrame, ticker: str, outdir: str) -> str:
    df = tidy.copy()
    df['expiry_date'] = df['expiry'].dt.date
    fig = px.scatter_3d(
        df,
        x='moneyness',
        y='expiry_date',
        z='pressure',
        color='pressure',
        title=f"{ticker} 3D Blended Pressure Surface",
        labels={'moneyness':'ln(K/S)','expiry_date':'Expiry','pressure':'Pressure'}
    )
    outfile = os.path.join(outdir, f"{ticker}_pressure_3d_surface.html")
    pio.write_html(fig, outfile, auto_open=False)
    return outfile

# ------------------------------------------------------------
# D) Strike vs price pressure profile (for key expiry)
# ------------------------------------------------------------
def build_strike_pressure_profile(tidy: pd.DataFrame, ticker: str, spot: float, outdir: str) -> str:
    df = tidy.copy()
    today = pd.Timestamp(datetime.now().date())
    df['dte'] = (df['expiry'] - today).dt.days
    df = df[df['dte'] >= 0]
    if df.empty:
        return ""
    exps = df[['expiry','dte']].drop_duplicates()
    exps['dist'] = (exps['dte'] - 30).abs()
    chosen = exps.sort_values(['dist','dte']).iloc[0]['expiry']
    slice_df = df[df['expiry'] == chosen].copy()
    if slice_df.empty:
        return ""
    fig = px.scatter(
        slice_df,
        x='strike',
        y='pressure',
        color='option_type',
        title=f"{ticker} Strike vs Pressure Profile (Expiry {chosen.date()})",
        labels={'strike':'Strike','pressure':'Blended Pressure'}
    )
    fig.add_vline(x=spot, line=dict(color="black", dash="dash"), annotation_text="Spot", annotation_position="top")
    outfile = os.path.join(outdir, f"{ticker}_strike_pressure_profile.html")
    pio.write_html(fig, outfile, auto_open=False)
    return outfile

# ------------------------------------------------------------
# E) Dealer book map (gamma & vanna vs strike/expiry)
# ------------------------------------------------------------
def build_dealer_book_map(tidy: pd.DataFrame, ticker: str, spot: float, outdir: str) -> str:
    df = tidy.copy()
    df['expiry_date'] = df['expiry'].dt.date
    g_gamma = df.groupby(['expiry_date','strike'])['gamma'].sum().reset_index()
    g_vanna = df.groupby(['expiry_date','strike'])['vanna'].sum().reset_index()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, subplot_titles=("Gamma Map","Vanna Map"))
    fig.add_trace(
        go.Heatmap(
            x=g_gamma['strike'],
            y=g_gamma['expiry_date'],
            z=g_gamma['gamma'],
            colorbar_title="Gamma"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(
            x=g_vanna['strike'],
            y=g_vanna['expiry_date'],
            z=g_vanna['vanna'],
            colorbar_title="Vanna"
        ),
        row=2, col=1
    )
    fig.add_vline(x=spot, line=dict(color="white", dash="dash"), row=1, col=1)
    fig.add_vline(x=spot, line=dict(color="white", dash="dash"), row=2, col=1)
    fig.update_layout(
        title=f"{ticker} Dealer Book Map (Gamma & Vanna vs Strike/Expiry)",
        xaxis2_title="Strike"
    )
    outfile = os.path.join(outdir, f"{ticker}_dealer_book_map.html")
    pio.write_html(fig, outfile, auto_open=False)
    return outfile

# ------------------------------------------------------------
# Expiration selection (real API)
# ------------------------------------------------------------
def choose_expirations(ticker: str, max_expiries: int, source: str="yfinance") -> list:
    today = pd.Timestamp(datetime.now().date())
    try:
        exps = options_data_retrieval.get_available_remote_expirations(ticker, source)
    except Exception as e:
        logging.error(f"Failed to get remote expirations: {e}")
        return []
    exps = sorted(exps)
    fut = [e for e in exps if e >= today]
    if fut:
        exps = fut
    if max_expiries is None or max_expiries <= 0:
        return exps
    return exps[:max_expiries]

# ------------------------------------------------------------
# Core pipeline
# ------------------------------------------------------------
def process_chains_for_ticker(
    ticker: str,
    w_gamma: float,
    w_vanna: float,
    max_expiries: int,
    outdir: str,
    source: str="yfinance"
):
    html_files = []

    # Spot / OHLCV
    logging.info(f"Loading OHLCV for {ticker}...")
    spot_df = data_retrieval.load_or_download_ticker(ticker)
    spot_df = spot_df.sort_index()
    spot = float(spot_df['Close'].iloc[-1])
    logging.info(f"Spot = {spot:.4f}")

    # Expirations
    expirations = choose_expirations(ticker, max_expiries, source)
    if not expirations:
        logging.error("No expirations available for this ticker.")
        sys.exit(1)
    logging.info(f"Selected expirations: {[e.date().isoformat() for e in expirations]}")

    # Load chains and compute greeks
    frames = []
    today = pd.Timestamp(datetime.now().date())
    for exp in expirations:
        try:
            raw = options_data_retrieval.load_or_download_option_chain(ticker, exp, source)
        except Exception as e:
            logging.warning(f"Failed to load chain for {ticker} @ {exp.date()}: {e}")
            continue
        if raw is None or raw.empty:
            continue
        raw = raw.copy()
        raw['T'] = (exp - today).days / 365.0
        raw['T'] = raw['T'].clip(lower=1e-8)
        deltas, gammas, vannas = [], [], []
        for _, row in raw.iterrows():
            d, g, v = compute_greeks(row, spot)
            deltas.append(d)
            gammas.append(g)
            vannas.append(v)
        raw['delta'] = deltas
        raw['gamma'] = gammas
        raw['vanna'] = vannas
        frames.append(raw)

    if not frames:
        logging.error("No chain data loaded after processing expirations.")
        sys.exit(1)

    full_chain = pd.concat(frames, ignore_index=True)
    tidy = chain_df_to_tidy(full_chain)
    tidy = compute_moneyness(tidy, spot)
    tidy = apply_liquidity_mask(tidy)
    tidy = tidy[tidy['mask'] == True].copy()
    if tidy.empty:
        logging.error("All rows removed by liquidity filter.")
        sys.exit(1)

    # Z-scores per expiry
    tidy['z_gamma'] = tidy.groupby('expiry')['gamma'].transform(zscore_series)
    tidy['z_vanna'] = tidy.groupby('expiry')['vanna'].transform(zscore_series)
    tidy['z_gamma'] = tidy['z_gamma'].clip(-3,3)
    tidy['z_vanna'] = tidy['z_vanna'].clip(-3,3)
    tidy['pressure'] = w_gamma * tidy['z_gamma'] + w_vanna * tidy['z_vanna']

    # Core heatmaps
    mg = build_heatmap_matrix(tidy, 'z_gamma')
    mv = build_heatmap_matrix(tidy, 'z_vanna')
    mp = build_heatmap_matrix(tidy, 'pressure')

    f_gamma = os.path.join(outdir, f"{ticker}_gamma_zscore.html")
    f_vanna = os.path.join(outdir, f"{ticker}_vanna_zscore.html")
    f_press = os.path.join(outdir, f"{ticker}_pressure.html")

    make_heatmap(mg, f"{ticker} Gamma Z-Score Heatmap", f_gamma)
    make_heatmap(mv, f"{ticker} Vanna Z-Score Heatmap", f_vanna)
    make_heatmap(mp, f"{ticker} Blended Pressure Heatmap", f_press)

    html_files.extend([f_gamma, f_vanna, f_press])

    # Term structure slices
    ts_html = build_term_structure_slices(tidy, ticker, outdir)
    if ts_html: html_files.append(ts_html)

    # Smile slice
    smile_html = build_smile_slice(tidy, ticker, outdir)
    if smile_html: html_files.append(smile_html)

    # Delta-bucket heatmaps
    delta_htmls = build_delta_bucket_heatmaps(tidy, ticker, outdir)
    html_files.extend(delta_htmls)

    # Price history with expiries
    ph_html = build_price_history_with_expiries(ticker, spot_df, expirations, outdir)
    if ph_html: html_files.append(ph_html)

    # ATM pressure term structure
    atm_html = build_atm_pressure_term(tidy, ticker, outdir)
    if atm_html: html_files.append(atm_html)

    # 3D pressure surface
    p3d_html = build_pressure_3d_surface(tidy, ticker, outdir)
    if p3d_html: html_files.append(p3d_html)

    # Strike vs price pressure profile
    spp_html = build_strike_pressure_profile(tidy, ticker, spot, outdir)
    if spp_html: html_files.append(spp_html)

    # Dealer book map (gamma + vanna)
    dbm_html = build_dealer_book_map(tidy, ticker, spot, outdir)
    if dbm_html: html_files.append(dbm_html)

    return tidy, html_files

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gamma/Vanna Z-Score Pressure Visualizer v5 (Full Price-Integrated Dealer Map)")
    parser.add_argument("ticker", type=str, help="Underlying ticker (e.g. IWM, SPY, NVDA)")
    parser.add_argument("--weights", nargs=2, type=float, default=[1.0,1.0], help="Weights [w_gamma, w_vanna]")
    parser.add_argument("--max-expiries", type=int, default=8, help="Max number of nearest expiries to load")
    parser.add_argument("--source", type=str, default="yfinance", help="Options data source (default: yfinance)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    w_gamma, w_vanna = args.weights
    max_expiries = args.max_expiries
    source = args.source

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")

    try:
        base_path = data_retrieval.BASE_DATA_PATH()
    except Exception:
        base_path = "/dev/shm/data"

    today_str = datetime.now().strftime("%Y-%m-%d")
    outdir = os.path.join(base_path, today_str, "MEGA_CORR_V14", "OPTIONS_PRESSURE", ticker)
    ensure_dir(outdir)
    logging.info(f"Output directory: {outdir}")
    logging.info(f"Ticker={ticker}, w_gamma={w_gamma}, w_vanna={w_vanna}, max_expiries={max_expiries}")

    tidy, html_files = process_chains_for_ticker(ticker, w_gamma, w_vanna, max_expiries, outdir, source)

    logging.info("Opening Plotly dashboards in browser tabs...")
    for f in html_files:
        if f and os.path.exists(f):
            webbrowser.open(f"file://{os.path.abspath(f)}", new=2)

    logging.info("DONE.")

if __name__ == "__main__":
    main()

