#!/usr/bin/env python3
# SCRIPTNAME: options_rr25_skew_termstructure_dashboard.v1.py
# AUTHOR: Michael Derby
# FRAMEWORK: STEGO Financial Framework
#
# RR25 skew term-structure dashboard with dual methodologies:
#   A) Interpolated 25-delta vol from BS deltas (model-free-ish)
#   B) Nearest-Delta approximation (default)
#
# Requirements / Design:
#   - Uses data_retrieval.py and options_data_retrieval.py as-is.
#   - Writes all outputs into /dev/shm by default:
#         /dev/shm/OPTIONS_RR25_SKEW_TERMSTRUCTURE/{TICKER}/{YYYY-MM-DD}/
#   - Saves all key tables as CSVs for STEGO pipelines:
#         rr25_termstructure.csv
#         atm_iv_termstructure.csv
#         rr25_slopes.csv
#         rr25_tenor_mapping.csv
#   - Produces a high-fidelity HTML dashboard with Plotly tabs.
#
# Usage:
#   python3 options_rr25_skew_termstructure_dashboard.v1.py SPY
#   python3 options_rr25_skew_termstructure_dashboard.v1.py IWM --method A --max-expiries 12

import argparse
import os
import sys
import math
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.utils as putils
from plotly.subplots import make_subplots
import webbrowser

# Local modules (must be in PYTHONPATH or same directory)
try:
    import data_retrieval
    import options_data_retrieval
except ImportError:
    sys.exit("CRITICAL: specific data_retrieval libraries for STEGO framework not found.")

# --------------------------------------------------------------------------------------
# Configuration & Logging
# --------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] STEGO.RR25: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------------------------------------------------------------------------
# Black-Scholes helpers for delta
# --------------------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_delta(spot: float,
             strike: float,
             ttm_years: float,
             iv: float,
             option_type: str,
             r: float = 0.0,
             q: float = 0.0) -> float:
    """
    Black-Scholes delta for European call/put.
    Assumes r=0, q=0 for ranking purposes unless specified.
    """
    try:
        if spot <= 0 or strike <= 0 or ttm_years <= 0 or iv <= 0:
            return float("nan")
        option_type = option_type.lower()
        vol_sqrt_t = iv * math.sqrt(ttm_years)
        if vol_sqrt_t <= 0:
            return float("nan")
        d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * ttm_years) / vol_sqrt_t
        if option_type == "call":
            return math.exp(-q * ttm_years) * _norm_cdf(d1)
        elif option_type == "put":
            return -math.exp(-q * ttm_years) * _norm_cdf(-d1)
        else:
            return float("nan")
    except Exception:
        return float("nan")

# --------------------------------------------------------------------------------------
# RR25 computation for a single expiration
# --------------------------------------------------------------------------------------

def compute_ttm_years(expiration_ts: pd.Timestamp, asof_ts: pd.Timestamp) -> float:
    """Compute time-to-maturity in years (ACT/365.25)."""
    exp_date = expiration_ts.normalize()
    asof_date = asof_ts.normalize()
    days = (exp_date - asof_date).days
    if days <= 0:
        days = 1
    return days / 365.25

def _compute_deltas_for_chain(chain: pd.DataFrame,
                              spot: float,
                              ttm_years: float) -> pd.DataFrame:
    """Add Black-Scholes deltas to a chain DataFrame."""
    df = chain.copy()
    required = ['type', 'strike', 'impliedVolatility']
    if not all(col in df.columns for col in required):
        logging.warning("Option chain missing required columns.")
        df['delta'] = np.nan
        return df

    strikes = df['strike'].astype(float).values
    ivs = df['impliedVolatility'].astype(float).values
    types = df['type'].astype(str).values

    deltas = []
    # Vectorization possible, but loop is safe/clear for mixed types
    for s, k, v, ot in zip([spot] * len(df), strikes, ivs, types):
        d = bs_delta(s, k, ttm_years, v, ot)
        deltas.append(d)
    df['delta'] = deltas
    return df

def _select_25_delta_nearest(chain: pd.DataFrame,
                             target_delta: float,
                             option_side: str) -> float:
    """Method B: Nearest-delta approximation."""
    sub = chain.dropna(subset=['delta', 'impliedVolatility']).copy()
    if sub.empty: return float("nan")

    if option_side.lower() == "call":
        sub = sub[(sub['delta'] > 0) & (sub['delta'] < 1)]
    elif option_side.lower() == "put":
        sub = sub[(sub['delta'] < 0) & (sub['delta'] > -1)]

    if sub.empty: return float("nan")

    sub['delta_diff'] = (sub['delta'] - target_delta).abs()
    row = sub.loc[sub['delta_diff'].idxmin()]
    return float(row['impliedVolatility'])

def _interpolate_iv_for_delta(chain: pd.DataFrame,
                              target_delta: float) -> float:
    """Method A: Linear Interpolation in delta-space."""
    sub = chain.dropna(subset=['delta', 'impliedVolatility']).copy()
    if sub.empty or len(sub) < 2:
        return float("nan")

    sub = sub.sort_values('delta')
    deltas = sub['delta'].values
    ivs = sub['impliedVolatility'].values

    try:
        iv_target = float(np.interp(target_delta, deltas, ivs))
    except Exception:
        iv_target = float("nan")
    return iv_target

def compute_rr25_for_expiration(chain: pd.DataFrame,
                                spot: float,
                                expiration_ts: pd.Timestamp,
                                asof_ts: pd.Timestamp,
                                method: str = "B") -> dict:
    """Compute RR25 for a single expiration."""
    method = method.upper()
    ttm_years = compute_ttm_years(expiration_ts, asof_ts)

    # Base return object
    res = {
        "expiration": expiration_ts.normalize().date(),
        "ttm_years": ttm_years,
        "rr25": float("nan"),
        "iv_call_25": float("nan"),
        "iv_put_25": float("nan"),
        "atm_iv": float("nan"),
        "method": method,
    }

    df = chain.copy()
    if 'impliedVolatility' not in df.columns:
        return res

    df = df.dropna(subset=['impliedVolatility', 'strike'])
    df = df[(df['impliedVolatility'] > 0) & (df['strike'] > 0)]
    if df.empty: return res

    # Compute deltas
    df = _compute_deltas_for_chain(df, spot, ttm_years)

    calls = df[df['type'].str.lower() == "call"].copy()
    puts = df[df['type'].str.lower() == "put"].copy()

    # ATM IV: nearest strike to spot
    df['moneyness'] = (df['strike'] - spot).abs()
    if not df.empty:
        res["atm_iv"] = float(df.sort_values('moneyness').iloc[0]['impliedVolatility'])

    if method == "A":
        iv_call = _interpolate_iv_for_delta(calls, target_delta=0.25)
        iv_put = _interpolate_iv_for_delta(puts, target_delta=-0.25)
    else:
        iv_call = _select_25_delta_nearest(calls, target_delta=0.25, option_side="call")
        iv_put = _select_25_delta_nearest(puts, target_delta=-0.25, option_side="put")

    res["iv_call_25"] = iv_call
    res["iv_put_25"] = iv_put

    if math.isfinite(iv_call) and math.isfinite(iv_put):
        res["rr25"] = iv_call - iv_put

    return res

# --------------------------------------------------------------------------------------
# Tenor mapping & Slopes
# --------------------------------------------------------------------------------------

def approximate_tenor_indices(rr25_df: pd.DataFrame, asof_date: date) -> dict:
    if rr25_df.empty:
        return {"1w": None, "1m": None, "3m": None, "6m": None, "days": pd.Series(dtype=float)}

    exps = pd.to_datetime(rr25_df['expiration']).dt.date
    target_days = {"1w": 7, "1m": 30, "3m": 90, "6m": 180}

    days_to_exp = []
    for exp in exps:
        d = (exp - asof_date).days
        days_to_exp.append(d if d > 0 else 1)
    days_series = pd.Series(days_to_exp, index=rr25_df.index)

    tenor_indices = {"days": days_series}
    for label, target in target_days.items():
        if days_series.empty:
            tenor_indices[label] = None
        else:
            idx = (days_series - target).abs().idxmin()
            tenor_indices[label] = idx

    return tenor_indices

def compute_rr25_slopes(rr25_df: pd.DataFrame, tenor_indices: dict) -> pd.DataFrame:
    cols = ['rr25_1w', 'rr25_1m', 'rr25_3m', 'rr25_6m',
            'slope_1w_1m', 'slope_1w_3m', 'slope_1w_6m']
    out = {c: float("nan") for c in cols}

    if rr25_df.empty: return pd.DataFrame([out])
    
    rr25_s = rr25_df['rr25']
    try:
        t_idx = {k: tenor_indices.get(k) for k in ["1w","1m","3m","6m"]}
        rr_vals = {k: (rr25_s.loc[v] if v is not None else float("nan")) for k,v in t_idx.items()}

        out.update({f'rr25_{k}': rr_vals[k] for k in t_idx})
        
        # Calculate slopes
        if math.isfinite(rr_vals['1w']):
            if math.isfinite(rr_vals['1m']): out['slope_1w_1m'] = rr_vals['1w'] - rr_vals['1m']
            if math.isfinite(rr_vals['3m']): out['slope_1w_3m'] = rr_vals['1w'] - rr_vals['3m']
            if math.isfinite(rr_vals['6m']): out['slope_1w_6m'] = rr_vals['1w'] - rr_vals['6m']

    except Exception as e:
        logging.warning("Error computing slopes: %s", e)

    return pd.DataFrame([out])

# --------------------------------------------------------------------------------------
# VISUALIZATION: Enhanced Plotly Figures
# --------------------------------------------------------------------------------------

def _get_skew_color(val):
    """Green for positive skew (Bullish/Call Premium), Red for negative (Bearish/Put Premium)."""
    if np.isnan(val): return "grey"
    return "#00e676" if val >= 0 else "#ff5252"

def build_rr25_termstructure_figure(rr25_df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Dual-Axis Chart:
    LHS: RR25 Skew (Fill-to-Zero Area)
    RHS: Raw Call/Put IVs (Dashed Lines)
    """
    if rr25_df.empty: return go.Figure()

    # Create Dual Axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Skew Area (LHS)
    # We split into positive and negative parts to color them differently
    pos_mask = rr25_df['rr25'] >= 0
    neg_mask = rr25_df['rr25'] < 0

    # To draw continuous lines with changing colors, usually one trace is easiest, 
    # but for fill-to-zero with distinct colors, we can use a single trace with a gradient 
    # or just a solid line with a fill. For simplicity and impact, we use a single dynamic line.
    
    fig.add_trace(go.Scatter(
        x=rr25_df['expiration'], y=rr25_df['rr25'],
        name="RR25 (Call-Put)",
        mode='lines+markers',
        line=dict(width=3, color='#fdd835'), # Yellow/Gold core line
        fill='tozeroy',
        fillcolor='rgba(255, 255, 255, 0.1)' # Subtle fill
    ), secondary_y=False)

    # 2. Raw IVs (RHS)
    fig.add_trace(go.Scatter(
        x=rr25_df['expiration'], y=rr25_df['iv_call_25'],
        name="Call 25Δ IV",
        mode='lines',
        line=dict(width=1, dash='dot', color='#00e676'), # Green dashed
        opacity=0.7
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=rr25_df['expiration'], y=rr25_df['iv_put_25'],
        name="Put 25Δ IV",
        mode='lines',
        line=dict(width=1, dash='dot', color='#ff5252'), # Red dashed
        opacity=0.7
    ), secondary_y=True)

    # Layout
    fig.update_layout(
        title=dict(text=f"<b>{ticker} RR25 Skew Term Structure</b><br><sup>LHS: Skew (Call IV - Put IV) | RHS: Raw Implied Vols</sup>"),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Axis Formatting
    fig.update_yaxes(title_text="<b>RR25 Skew</b>", zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255,255,255,0.2)', secondary_y=False)
    fig.update_yaxes(title_text="Raw IV", showgrid=False, secondary_y=True)
    fig.update_xaxes(title_text="Expiration")

    return fig

def build_atm_iv_figure(rr25_df: pd.DataFrame, ticker: str) -> go.Figure:
    if rr25_df.empty: return go.Figure()

    fig = go.Figure()
    
    # Main ATM Line
    fig.add_trace(go.Scatter(
        x=rr25_df['expiration'],
        y=rr25_df['atm_iv'],
        mode="lines+markers+text",
        name="ATM IV",
        line=dict(color='#29b6f6', width=3), # Light Blue
        marker=dict(size=8, color='#0277bd', line=dict(width=2, color='white')),
        text=np.round(rr25_df['atm_iv'], 3),
        textposition="top center"
    ))

    fig.update_layout(
        title=f"<b>{ticker} ATM Implied Volatility Term Structure</b>",
        xaxis_title="Expiration",
        yaxis_title="ATM Implied Volatility",
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig

def build_rr25_table_figure(rr25_df: pd.DataFrame, ticker: str) -> go.Figure:
    if rr25_df.empty: return go.Figure()

    df = rr25_df.copy()
    # Format strings
    df['exp_str'] = pd.to_datetime(df['expiration']).dt.strftime("%Y-%m-%d")
    
    # Colors for the table cells (Heatmap style)
    # IVs: Darker blue for higher IV
    # Skew: Green for positive, Red for negative
    
    skew_vals = df['rr25'].fillna(0)
    skew_colors = ['#1b5e20' if v > 0.05 else '#2e7d32' if v > 0 else '#c62828' if v < -0.05 else '#d32f2f' for v in skew_vals]
    
    # Create Table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Expiration</b>", "<b>TTM (Y)</b>", "<b>RR25 Skew</b>", "<b>ATM IV</b>", "<b>Call 25Δ</b>", "<b>Put 25Δ</b>"],
            fill_color='#303030',
            font=dict(color='white', size=12),
            align='center',
            line_color='grey'
        ),
        cells=dict(
            values=[
                df['exp_str'],
                df['ttm_years'].map('{:.3f}'.format),
                df['rr25'].map('{:+.4f}'.format),
                df['atm_iv'].map('{:.4f}'.format),
                df['iv_call_25'].map('{:.4f}'.format),
                df['iv_put_25'].map('{:.4f}'.format),
            ],
            fill_color=[
                '#1e1e1e', # Date
                '#1e1e1e', # TTM
                skew_colors, # RR25 (Conditional)
                '#1e1e1e', # ATM
                '#1e1e1e', # Call
                '#1e1e1e', # Put
            ],
            font=dict(color='white', size=12),
            align='center',
            line_color='grey',
            height=30
        )
    )])
    
    fig.update_layout(
        title=f"<b>{ticker} Data Table</b>",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def build_rr25_slopes_figure(slopes_df: pd.DataFrame, ticker: str) -> go.Figure:
    if slopes_df is None or slopes_df.empty: return go.Figure()

    row = slopes_df.iloc[0]
    labels = ["1w - 1m", "1w - 3m", "1w - 6m"]
    values = [
        row.get('slope_1w_1m', np.nan),
        row.get('slope_1w_3m', np.nan),
        row.get('slope_1w_6m', np.nan),
    ]
    
    # Conditional colors
    colors = ['#00e676' if v >= 0 else '#ff5252' for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        name="Slope",
        marker_color=colors,
        text=np.round(values, 4),
        textposition='auto'
    ))

    fig.update_layout(
        title=f"<b>{ticker} Skew Term Structure Slopes</b><br><sup>(Short Term Skew minus Long Term Skew)</sup>",
        xaxis_title="Tenor Spread",
        yaxis_title="Slope (Difference in RR25)",
        template="plotly_dark",
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white')
    )
    return fig

# --------------------------------------------------------------------------------------
# HTML Dashboard Construction
# --------------------------------------------------------------------------------------

def build_tabbed_html(figures: dict, output_html_path: str, title: str):
    logging.info("Building Dark Mode Dashboard: %s", output_html_path)
    
    # Dark Mode CSS
    css = """
    body { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; }
    h2 { color: #fdd835; margin: 20px; border-bottom: 2px solid #333; padding-bottom: 10px; }
    .tab { overflow: hidden; border-bottom: 1px solid #333; background-color: #1e1e1e; padding-left: 20px; }
    .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 20px; transition: 0.3s; font-size: 15px; color: #aaa; }
    .tab button:hover { background-color: #333; color: #fff; }
    .tab button.active { background-color: #29b6f6; color: #000; font-weight: bold; }
    .tabcontent { display: none; padding: 20px; animation: fadeEffect 0.5s; }
    @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
    .footer { font-size: 10px; color: #555; text-align: center; margin-top: 50px; }
    """

    script_func = """
    function openTab(evt, tabName) {
      var i, tabcontent, tablinks;
      tabcontent = document.getElementsByClassName("tabcontent");
      for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
      tablinks = document.getElementsByClassName("tablinks");
      for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }
    // Open default tab
    document.addEventListener("DOMContentLoaded", function() { document.querySelector(".tablinks").click(); });
    """

    html_parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>",
        f"<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        f"<style>{css}</style></head><body>",
        f"<h2>{title} <span style='font-size:14px; color:#777;'>| STEGO Financial Framework</span></h2>",
        '<div class="tab">'
    ]

    # Buttons
    for i, (tid, (label, _)) in enumerate(figures.items()):
        html_parts.append(f'<button class="tablinks" onclick="openTab(event, \'{tid}\')">{label}</button>')
    html_parts.append('</div>')

    # Divs
    for tid, (_, _) in figures.items():
        html_parts.append(f'<div id="{tid}" class="tabcontent"><div id="plot_{tid}" style="width:100%; height:85vh;"></div></div>')

    # Plotly Scripts
    html_parts.append('<script>')
    for tid, (_, fig) in figures.items():
        j = putils.PlotlyJSONEncoder().encode(fig)
        html_parts.append(f"var f_{tid} = {j}; Plotly.newPlot('plot_{tid}', f_{tid}.data, f_{tid}.layout);")
    html_parts.append(script_func)
    html_parts.append('</script>')
    
    html_parts.append(f'<div class="footer">Generated by STEGO Framework | Author: Michael Derby | {datetime.now()}</div>')
    html_parts.append('</body></html>')

    with open(output_html_path, "w") as f:
        f.write("".join(html_parts))

# --------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="STEGO RR25 Skew Dashboard")
    parser.add_argument("ticker", type=str, help="Underlying Ticker (e.g., SPY)")
    parser.add_argument("--method", type=str, choices=["A", "B"], default="B", help="Interpolation Method")
    parser.add_argument("--max-expiries", type=int, default=10, help="Max expirations to fetch")
    parser.add_argument("--output-root", type=str, default="/dev/shm/OPTIONS_RR25_SKEW_TERMSTRUCTURE")
    parser.add_argument("--source", type=str, default="yfinance")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    
    # Paths
    today_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(args.output_root, ticker, today_str)
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Starting STEGO Analysis for %s", ticker)
    
    # 1. Spot Data
    try:
        ohlcv = data_retrieval.load_or_download_ticker(ticker, period="max")
        if ohlcv.empty or 'Close' not in ohlcv.columns: raise ValueError("No data")
        spot = float(ohlcv['Close'].iloc[-1])
        asof_ts = ohlcv.index[-1]
    except Exception as e:
        logging.error("Failed to load spot data: %s", e)
        sys.exit(1)

    # 2. Expirations
    try:
        remote_exps = options_data_retrieval.get_available_remote_expirations(ticker, source=args.source)
        selected_exps = sorted(remote_exps)[:args.max_expiries]
        options_data_retrieval.ensure_option_chains_cached(ticker, selected_exps, source=args.source)
    except Exception as e:
        logging.error("Failed to retrieve options data: %s", e)
        sys.exit(1)

    # 3. Compute Metrics
    rows = []
    for exp_ts in selected_exps:
        try:
            chain = options_data_retrieval.load_or_download_option_chain(ticker, exp_ts, source=args.source)
            res = compute_rr25_for_expiration(chain, spot, exp_ts, asof_ts, method=args.method)
            rows.append(res)
        except Exception:
            continue

    if not rows:
        logging.error("No valid options data found.")
        sys.exit(1)

    rr25_df = pd.DataFrame(rows).sort_values('expiration').reset_index(drop=True)

    # 4. Save CSVs (STEGO Pipeline Requirement)
    rr25_df.to_csv(os.path.join(output_dir, "rr25_termstructure.csv"), index=False)
    rr25_df[['expiration', 'ttm_years', 'atm_iv']].to_csv(os.path.join(output_dir, "atm_iv_termstructure.csv"), index=False)
    
    asof_date = asof_ts.normalize().date()
    tenor_indices = approximate_tenor_indices(rr25_df, asof_date)
    slopes_df = compute_rr25_slopes(rr25_df, tenor_indices)
    slopes_df.to_csv(os.path.join(output_dir, "rr25_slopes.csv"), index=False)
    
    # Save Tenor Map
    map_recs = []
    days_s = tenor_indices.get("days", pd.Series(dtype=float))
    for i, d in days_s.items():
        map_recs.append({"row_index": i, "expiration": rr25_df.loc[i, 'expiration'], "days": d})
    pd.DataFrame(map_recs).to_csv(os.path.join(output_dir, "rr25_tenor_mapping.csv"), index=False)

    # 5. Generate Figures
    figs = {
        "tab_rr25": ("RR25 Skew Structure", build_rr25_termstructure_figure(rr25_df, ticker)),
        "tab_atm": ("ATM Volatility", build_atm_iv_figure(rr25_df, ticker)),
        "tab_slopes": ("Skew Slopes", build_rr25_slopes_figure(slopes_df, ticker)),
        "tab_table": ("Data Grid", build_rr25_table_figure(rr25_df, ticker))
    }

    # 6. Render Dashboard
    html_path = os.path.join(output_dir, f"{ticker}_rr25_dashboard.html")
    build_tabbed_html(figs, html_path, f"{ticker} STEGO Skew Dashboard")

    logging.info("Dashboard generated at: %s", html_path)
    try:
        webbrowser.open("file://" + os.path.abspath(html_path))
    except:
        pass

if __name__ == "__main__":
    main()
