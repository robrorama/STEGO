#!/usr/bin/env python3
"""
chartlib_unified.py

Unified engine for options data access, analytics, and Plotly/HTML visuals.

Key capabilities (clean reimplementation):
- Uses options_data_retrieval.py for cache-first options access.
- Streak statistics from OHLCV (probabilities, next-day returns)
- Bull Call Debit and Bull Put Credit spread finders
- Plotly figures: 3D bubbles, aggregated 2D OI, 3D OI, distribution dashboard
- Additional charts: total OI by strike, unusual activity, OI heatmap
- Interactive Option Chain “tumbler” HTML (side-by-side calls/puts, heatmaps, strike sliding)

OHLCV source: data_retrieval.load_or_download_ticker.
"""

from __future__ import annotations

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

# CONSTRAINT: Import local data retrieval modules
try:
    from data_retrieval import load_or_download_ticker
    import options_data_retrieval as odr
except ImportError:
    print("Error: data_retrieval.py or options_data_retrieval.py not found.")
    sys.exit(1)

# Optional SciPy smoothing for IV surface
try:
    from scipy.interpolate import griddata  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------------------------------------------------------
# A. Core Data Functions (Delegated to options_data_retrieval)
# -----------------------------------------------------------------------------

def list_expiration_dates(ticker: str) -> List[str]:
    """
    Enumerate option expirations (YYYY-MM-DD) for a ticker.
    Uses options_data_retrieval to check cache first, then remote.
    """
    # odr returns Timestamps
    try:
        # Prefer remote check to get full list, odr handles caching internally if we ask for it
        # But to list what is available, we ask remote.
        timestamps = odr.get_available_remote_expirations(ticker)
        return [ts.strftime("%Y-%m-%d") for ts in timestamps]
    except Exception:
        return []


def get_options_data(ticker: str, expiration_date: str) -> pd.DataFrame:
    """
    Cache-first load of the FULL option chain for a given expiration via options_data_retrieval.
    """
    # odr expects string or Timestamp
    df = odr.load_or_download_option_chain(ticker, expiration_date)
    
    if df.empty:
        return pd.DataFrame()

    # Normalize columns to match chartlib expectations
    # odr returns 'type' ('call'/'put'), chartlib expects 'optionType'
    if 'type' in df.columns and 'optionType' not in df.columns:
        df['optionType'] = df['type']

    # Ensure numeric types
    for c in ("strike", "openInterest", "volume", "impliedVolatility", "lastPrice", "bid", "ask", "delta"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def get_all_options_data(ticker: str) -> pd.DataFrame:
    """
    Load (or download) and concatenate option chains for ALL expirations.
    """
    # Use odr's bulk loader which handles caching
    # We first need the list of expirations
    exps = list_expiration_dates(ticker)
    
    # odr.ensure_option_chains_cached(ticker, exps) # Optional pre-fetch
    
    # Load one by one to build the big DF (odr.load_all_cached_option_chains only loads what is on disk)
    # So we must ensure they are on disk first if we want "all"
    
    frames: List[pd.DataFrame] = []
    for exp in exps:
        df = get_options_data(ticker, exp)
        if df is None or df.empty:
            continue
        dfx = df.copy()
        dfx["expiration"] = exp
        if 'type' in dfx.columns:
            dfx['optionType'] = dfx['type']
            
        frames.append(dfx)
        
    if not frames:
        return pd.DataFrame()
        
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# B. Analysis Functions (Non-Plotting)
# -----------------------------------------------------------------------------

def calculate_streak_probabilities(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Next‑day up/down probabilities and returns conditioned on signed streak length.
    """
    if "Close" not in ohlc_df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")
    d = ohlc_df.sort_index().copy()
    d["Return"] = d["Close"].pct_change()
    d["Up"] = d["Return"] > 0
    d["Direction"] = d["Up"].apply(lambda x: 1 if x else -1)
    d["Dir_Change"] = d["Direction"] != d["Direction"].shift(1)
    d["Streak_ID"] = d["Dir_Change"].cumsum()
    d["Streak_Length"] = d.groupby("Streak_ID").cumcount() + 1
    d["Signed_Streak_Length"] = d["Streak_Length"] * d["Direction"]
    d["Next_Day_Up"] = d["Up"].shift(-1)
    d["Next_Day_Return"] = d["Return"].shift(-1)
    d = d.dropna(subset=["Next_Day_Up", "Next_Day_Return"])

    pt = d.groupby("Signed_Streak_Length")["Next_Day_Up"].value_counts().unstack(fill_value=0)
    for k in (True, False):
        if k not in pt.columns:
            pt[k] = 0
    pt["Next_Day_Up_Count"] = pt.get(True, 0)
    pt["Next_Day_Down_Count"] = pt.get(False, 0)
    pt["Total"] = pt["Next_Day_Up_Count"] + pt["Next_Day_Down_Count"]
    pt["Prob_Next_Day_Up"] = pt["Next_Day_Up_Count"] / pt["Total"].replace(0, np.nan)
    pt["Prob_Next_Day_Down"] = pt["Next_Day_Down_Count"] / pt["Total"].replace(0, np.nan)

    mean_next = d.groupby("Signed_Streak_Length")["Next_Day_Return"].mean()
    std_next = d.groupby("Signed_Streak_Length")["Next_Day_Return"].std()
    mean_curr = d.groupby("Signed_Streak_Length")["Return"].mean()

    out = (pt.merge(mean_next.rename("Avg_Next_Day_Return"), left_index=True, right_index=True)
             .merge(std_next.rename("Std_Next_Day_Return"), left_index=True, right_index=True)
             .merge(mean_curr.rename("Avg_Current_Streak_Return"), left_index=True, right_index=True))
    out = out[[
        "Next_Day_Up_Count", "Next_Day_Down_Count", "Total",
        "Prob_Next_Day_Up", "Prob_Next_Day_Down",
        "Avg_Next_Day_Return", "Std_Next_Day_Return",
        "Avg_Current_Streak_Return"
    ]].sort_index().reset_index()

    # Round float columns to two decimal places as requested
    for col in out.select_dtypes(include='float').columns:
        out[col] = out[col].round(2)

    return out


def calculate_bull_call_debit_spread(calls_df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    Bull Call Debit: Buy ATM call; Sell OTM ≈ +0.30 delta (fallback: nearest above ATM).
    """
    calls = calls_df.copy()
    calls["strike_diff"] = (calls["strike"] - current_price).abs()
    atm_call = calls.sort_values("strike_diff").iloc[0]
    otm = calls[calls["strike"] > float(atm_call["strike"])].copy()
    if otm.empty:
        raise ValueError("No suitable OTM call above ATM.")
    if "delta" in otm.columns and otm["delta"].notna().any():
        otm["delta_diff"] = (otm["delta"] - 0.30).abs()
        sell = otm.sort_values(["delta_diff", "strike"]).iloc[0]
    else:
        otm["above_diff"] = (otm["strike"] - float(atm_call["strike"])).abs()
        sell = otm.sort_values(["above_diff", "strike"]).iloc[0]
    net_debit = round(float(atm_call["lastPrice"]) - float(sell["lastPrice"]), 2)
    width = float(sell["strike"]) - float(atm_call["strike"])
    return {
        "strategy": "Bull Call Debit Spread",
        "buy_call_strike": round(float(atm_call["strike"]), 2),
        "buy_call_price": round(float(atm_call["lastPrice"]), 2),
        "sell_call_strike": round(float(sell["strike"]), 2),
        "sell_call_price": round(float(sell["lastPrice"]), 2),
        "net_debit": net_debit,
        "max_profit": round(width - net_debit, 2),
        "breakeven_price": round(float(atm_call["strike"]) + net_debit, 2),
    }


def calculate_bull_put_credit_spread(puts_df: pd.DataFrame, current_price: float,
                                     max_otm_percent: float = 5.0) -> Dict[str, Any]:
    """
    Bull Put Credit: Sell ATM put; Buy OTM put with strike ≥ spot*(1 - max_otm_percent/100).
    Prefer ≈ -0.30 delta (fallback: nearest below ATM).
    """
    puts = puts_df.copy()
    puts["strike_diff"] = (puts["strike"] - current_price).abs()
    atm_put = puts.sort_values("strike_diff").iloc[0]

    min_otm = current_price * (1.0 - max_otm_percent / 100.0)
    otm = puts[(puts["strike"] < float(atm_put["strike"])) & (puts["strike"] >= min_otm)].copy()
    if otm.empty:
        raise ValueError("No suitable OTM put within specified range.")
    if "delta" in otm.columns and otm["delta"].notna().any():
        otm["delta_diff"] = (otm["delta"] - (-0.30)).abs()
        buy = otm.sort_values(["delta_diff", "strike"], ascending=[True, False]).iloc[0]
    else:
        otm["below_diff"] = (float(atm_put["strike"]) - otm["strike"]).abs()
        buy = otm.sort_values(["below_diff", "strike"], ascending=[True, False]).iloc[0]
    net_credit = round(float(atm_put["lastPrice"]) - float(buy["lastPrice"]), 2)
    width = float(atm_put["strike"]) - float(buy["strike"])
    return {
        "strategy": "Bull Put Credit Spread",
        "sell_put_strike": round(float(atm_put["strike"]), 2),
        "sell_put_price": round(float(atm_put["lastPrice"]), 2),
        "buy_put_strike": round(float(buy["strike"]), 2),
        "buy_put_price": round(float(buy["lastPrice"]), 2),
        "net_credit": net_credit,
        "max_loss": round(width - net_credit, 2),
        "breakeven_price": round(float(atm_put["strike"]) - net_credit, 2),
    }


# -----------------------------------------------------------------------------
# C. Visualization Functions
# -----------------------------------------------------------------------------

def _today_norm() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _ensure_days_to_exp(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "expiration" not in d.columns:
        return d
    d["expiration_dt"] = pd.to_datetime(d["expiration"], errors="coerce")
    d["days_to_exp"] = (d["expiration_dt"] - _today_norm()).dt.days
    return d


# --- Interactive Chain Tumbler (full HTML string) ------------------------------
def plot_interactive_chain_tumbler(ticker: str, expiration_date: str) -> str:
    """
    Return a complete, self-contained HTML string that renders a side-by-side
    CALLS/PUTS chain with heatmaps and sliding logic.
    """
    t = ticker.upper()
    df = get_options_data(t, expiration_date)
    if df is None or df.empty:
        return "<html><body><h3>No options data available.</h3></body></html>"

    if "optionType" not in df.columns:
        return "<html><body><h3>Invalid options dataset (missing optionType).</h3></body></html>"

    calls = df[df["optionType"] == "call"].copy()
    puts  = df[df["optionType"] == "put"].copy()

    # Canonical rename maps
    call_map = {
        "lastPrice": "last_call", "bid": "bid_call", "ask": "ask_call",
        "volume": "volume_call", "openInterest": "oi_call",
        "impliedVolatility": "iv_call", "delta": "delta_call",
    }
    put_map = {
        "lastPrice": "last_put", "bid": "bid_put", "ask": "ask_put",
        "volume": "volume_put", "openInterest": "oi_put",
        "impliedVolatility": "iv_put", "delta": "delta_put",
    }
    calls.rename(columns={k: v for k, v in call_map.items() if k in calls.columns}, inplace=True)
    puts.rename(columns={k: v for k, v in put_map.items() if k in puts.columns}, inplace=True)

    for col in ["last_call", "bid_call", "ask_call", "volume_call", "oi_call", "iv_call", "delta_call"]:
        if col not in calls.columns:
            calls[col] = np.nan
    for col in ["last_put", "bid_put", "ask_put", "volume_put", "oi_put", "iv_put", "delta_put"]:
        if col not in puts.columns:
            puts[col] = np.nan

    calls = calls[["strike", "last_call", "bid_call", "ask_call", "volume_call", "oi_call", "iv_call", "delta_call"]]
    puts  = puts[ ["strike", "last_put",  "bid_put",  "ask_put",  "volume_put",  "oi_put",  "iv_put",  "delta_put" ]]

    merged = pd.merge(calls, puts, on="strike", how="outer").sort_values("strike").reset_index(drop=True)

    center_idxs = set()
    for col in ["volume_call", "oi_call", "volume_put", "oi_put"]:
        if col in merged and not merged[col].dropna().empty:
            vmax = merged[col].max()
            if pd.notna(vmax) and vmax > 0:
                center_idxs.update(merged.index[merged[col] == vmax].tolist())

    def _fmt(x, prec=2):
        return "" if pd.isna(x) else f"{x:.{prec}f}"

    window = 10
    if center_idxs:
        rows_idx = sorted({i for idx in center_idxs for i in range(max(0, idx - window), min(len(merged) - 1, idx + window) + 1)})
        table = merged.iloc[rows_idx]
    else:
        table = merged

    html_rows = []
    for _, r in table.iterrows():
        ivc = r["iv_call"]; ivp = r["iv_put"]
        html_rows.append(
            "    <tr>"
            f"<td><input type='number' class='price' step='0.01' value='{_fmt(r['last_call'])}'></td>"
            f"<td>{_fmt(r['bid_call'])}</td>"
            f"<td>{_fmt(r['ask_call'])}</td>"
            f"<td class='vol'>{_fmt(r['volume_call'],0)}</td>"
            f"<td class='oi'>{_fmt(r['oi_call'],0)}</td>"
            f"<td>{_fmt(ivc*100,2) if pd.notna(ivc) else ''}</td>"
            f"<td>{_fmt(r['delta_call'])}</td>"
            f"<td class='strike'>{_fmt(r['strike'])}</td>"
            f"<td><input type='number' class='price' step='0.01' value='{_fmt(r['last_put'])}'></td>"
            f"<td>{_fmt(r['bid_put'])}</td>"
            f"<td>{_fmt(r['ask_put'])}</td>"
            f"<td class='vol'>{_fmt(r['volume_put'],0)}</td>"
            f"<td class='oi'>{_fmt(r['oi_put'],0)}</td>"
            f"<td>{_fmt(ivp*100,2) if pd.notna(ivp) else ''}</td>"
            f"<td>{_fmt(r['delta_put'])}</td>"
            "</tr>"
        )
    skeleton = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Option Chain – {t} {expiration_date}</title>
<style>
  body {{ font-family: Arial,Helvetica,sans-serif; margin: 1.2em; }}
  h2   {{ margin-top: 0; }}
  table{{ border-collapse: collapse; width: 100%; font-size: 0.92em; }}
  th, td {{ border: 1px solid #777; padding: 4px; text-align: right; white-space: nowrap; }}
  th.strike, td.strike {{ background: #f8f8f8; font-weight: bold; text-align: center; }}
  tr:nth-child(even) {{ background: #fbfbfb; }}
  input.price {{ width: 5em; text-align: right; border: none; background: transparent; font: inherit; }}
  td.vol, td.oi, td.strike {{ color: #000; }}
</style>
</head>
<body>
<h2>{t} — exp {expiration_date}</h2>
<button id="slide-up">Slide Up</button>
<button id="slide-down">Slide Down</button>
<table id="optTable">
  <thead>
    <tr>
      <th colspan="7">CALLS</th>
      <th class="strike">Strike</th>
      <th colspan="7">PUTS</th>
    </tr>
    <tr>
      <th>Last</th><th>Bid</th><th>Ask</th><th>Vol</th><th>OI</th><th>IV%</th><th>Δ</th>
      <th class="strike">Strike</th>
      <th>Last</th><th>Bid</th><th>Ask</th><th>Vol</th><th>OI</th><th>IV%</th><th>Δ</th>
    </tr>
  </thead>
  <tbody>
{os.linesep.join(html_rows)}
  </tbody>
</table>
<script>
(function(){{
  const volCells = Array.from(document.querySelectorAll('td.vol'));
  const oiCells  = Array.from(document.querySelectorAll('td.oi'));
  const vol = volCells.map(c => +c.textContent || 0);
  const oi  = oiCells .map(c => +c.textContent || 0);
  const vMax = Math.max(...vol, 1);
  const oMax = Math.max(...oi, 1);
  volCells.forEach(c => {{
    const x = +c.textContent || 0;
    const a = 0.12 + 0.72*(x/vMax);
    c.style.backgroundColor = `rgba(0,128,255,${{a}})`;
  }});
  oiCells.forEach(c => {{
    const x = +c.textContent || 0;
    const a = 0.12 + 0.72*(x/oMax);
    c.style.backgroundColor = `rgba(255,128,0,${{a}})`;
  }});
}})();

(function(){{
  const rows = Array.from(document.querySelectorAll('#optTable tbody tr'));
  if (!rows.length) return;
  let center = -1, best = -1;
  rows.forEach((r, i) => {{
    const txt = r.children[4].textContent;
    const val = parseFloat(txt) || 0;
    if (val > best) {{ best = val; center = i; }}
  }});
  if (center < 0) return;
  rows.forEach((r, i) => {{
    const d = Math.abs(i - center);
    if (d <= 10) {{
      const light = 20 + (d/10)*60;
      r.children[7].style.backgroundColor = `hsl(120, 100%, ${{light}}%)`;
    }}
  }});
}})();

function slideStrikes(isUp){{
  const strikeCells = Array.from(document.querySelectorAll('td.strike'));
  const vals = strikeCells.map(c => parseFloat(c.textContent) || 0);
  if (vals.length < 2) return;
  let nv;
  if (isUp) {{
    nv = vals.slice(1);
    const step = vals[vals.length-1] - vals[vals.length-2];
    nv.push(vals[vals.length-1] + step);
  }} else {{
    nv = vals.slice(0, -1);
    const step = vals[1] - vals[0];
    nv.unshift(vals[0] - step);
  }}
  strikeCells.forEach((c,i)=> c.textContent = nv[i].toFixed(2));
}}
document.getElementById('slide-up').addEventListener('click', ()=>slideStrikes(true));
document.getElementById('slide-down').addEventListener('click', ()=>slideStrikes(false));
</script>
</body>
</html>"""
    return skeleton


# --- IV Surface (3D) with optional smoothing & robust colors -------------------
def plot_iv_surface(all_options_df: pd.DataFrame,
                    ticker: str,
                    smooth: bool = False,
                    dte_step: int = 1,
                    strike_step: Optional[float] = None,
                    smooth_kernel: int = 3,
                    iv_min: Optional[float] = None,
                    iv_max: Optional[float] = None,
                    iv_pct: bool = False,
                    colorscale: str = "Turbo") -> go.Figure:
    d = all_options_df.copy()
    if "expiration" not in d.columns or "impliedVolatility" not in d.columns or "strike" not in d.columns:
        return go.Figure()

    d["expiration_dt"] = pd.to_datetime(d["expiration"], errors="coerce")
    d["days_to_exp"] = (d["expiration_dt"] - _today_norm()).dt.days
    d["impliedVolatility"] = pd.to_numeric(d["impliedVolatility"], errors="coerce")
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d = d.dropna(subset=["impliedVolatility", "strike", "days_to_exp"])
    d = d[(d["days_to_exp"] >= 0) & (d["strike"] > 0) & (d["impliedVolatility"] > 0) & (d["impliedVolatility"] < 5)]
    if d.empty:
        return go.Figure()
    if iv_pct:
        d["impliedVolatility"] *= 100.0

    piv = d.pivot_table(values="impliedVolatility", index="strike", columns="days_to_exp", aggfunc="mean")
    piv = piv.dropna(how="all").dropna(axis=1, how="all").sort_index().sort_index(axis=1)
    if piv.empty:
        return go.Figure()

    if smooth and _HAS_SCIPY:
        dte_vals = piv.columns.to_numpy(dtype=int)
        strike_vals = piv.index.to_numpy(dtype=float)
        if strike_step is None or strike_step <= 0:
            diffs = np.diff(np.sort(strike_vals))
            diffs = diffs[diffs > 0]
            step = float(np.median(diffs)) if diffs.size else 1.0
            strike_step = round(step, 2) if step < 1 else round(step)
            if strike_step <= 0:
                strike_step = 1.0
        new_dte = np.arange(int(dte_vals.min()), int(dte_vals.max()) + max(1, int(dte_step)), max(1, int(dte_step)), dtype=int)
        new_strike = np.arange(float(strike_vals.min()), float(strike_vals.max()) + strike_step, strike_step, dtype=float)

        GX, GY = np.meshgrid(new_dte, new_strike)
        pts = np.column_stack([d["days_to_exp"].to_numpy(), d["strike"].to_numpy()])
        vals = d["impliedVolatility"].to_numpy()
        Z_lin = griddata(pts, vals, (GX, GY), method="linear")
        Z_nn = griddata(pts, vals, (GX, GY), method="nearest")
        Z = np.where(np.isnan(Z_lin), Z_nn, Z_lin)

        if smooth_kernel and smooth_kernel > 1:
            Z = pd.DataFrame(Z).rolling(smooth_kernel, min_periods=1).mean() \
                               .T.rolling(smooth_kernel, min_periods=1).mean().T.to_numpy()
        X, Y = new_dte, new_strike
    else:
        X = piv.columns.astype(int).to_list()
        Y = piv.index.astype(float).to_list()
        Z = [[(float(v) if np.isfinite(v) else None) for v in row] for row in piv.to_numpy()]

    Znum = np.asarray(Z, dtype=float)
    msk = np.isfinite(Znum)
    if not np.any(msk):
        return go.Figure()

    if iv_min is None or iv_max is None:
        qlo, qhi = np.quantile(Znum[msk], [0.02, 0.98])
        cmin = iv_min if iv_min is not None else float(qlo)
        cmax = iv_max if iv_max is not None else float(qhi)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            cmin, cmax = float(np.nanmin(Znum)), float(np.nanmax(Znum))
    else:
        cmin, cmax = float(iv_min), float(iv_max)

    ztitle = "Implied Volatility (%)" if iv_pct else "Implied Volatility"
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale=colorscale, cmin=cmin, cmax=cmax)])
    fig.update_layout(
        title=f"{ticker.upper()} — Implied Volatility Surface" + (" (smoothed)" if smooth else ""),
        scene=dict(xaxis_title="Days to Expiration", yaxis_title="Strike", zaxis_title=ztitle),
        template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig


# --- Four OI dashboard visuals -------------------------------------------------
def _prep_options_for_dashboard(all_options_df: pd.DataFrame) -> pd.DataFrame:
    d = all_options_df.copy()
    d["openInterest"] = pd.to_numeric(d.get("openInterest"), errors="coerce")
    d["strike"] = pd.to_numeric(d.get("strike"), errors="coerce")
    d["expiration_dt"] = pd.to_datetime(d.get("expiration"), errors="coerce")
    d["days_to_exp"] = (d["expiration_dt"] - _today_norm()).dt.days
    d["contractSymbol"] = d.get("contractSymbol", "")
    return d


def _fig_3d_bubbles(calls_data: pd.DataFrame, puts_data: pd.DataFrame,
                    ticker: str, current_price: Optional[float]) -> go.Figure:
    calls = calls_data.copy()
    puts = puts_data.copy()
    calls["expiration_dt"] = pd.to_datetime(calls["expiration"])
    puts["expiration_dt"] = pd.to_datetime(puts["expiration"])
    all_exps = sorted(pd.concat([calls["expiration_dt"], puts["expiration_dt"]]).unique())
    date_to_z = {dt: i for i, dt in enumerate(all_exps)}

    now = _today_norm()
    days = [(dt - now).days for dt in all_exps] if all_exps else [0]
    min_days, max_days = (min(days), max(days)) if days else (0, 1)

    c_m = calls["openInterest"].mean()
    c_s = calls["openInterest"].std()
    c_th = (c_m if pd.notna(c_m) else 0) + (c_s if pd.notna(c_s) else 0)
    p_m = puts["openInterest"].mean()
    p_s = puts["openInterest"].std()
    p_th = (p_m if pd.notna(p_m) else 0) + (p_s if pd.notna(p_s) else 0)

    cf = calls[calls["openInterest"] > c_th]
    pf = puts[puts["openInterest"] > p_th]

    fig = go.Figure()
    for dt in all_exps:
        part = pf[pf["expiration_dt"] == dt]
        if part.empty:
            continue
        z_val = date_to_z[dt]
        dte = (dt - now).days
        norm = 0.2 + 0.8 * ((dte - min_days) / max(1, (max_days - min_days)))
        color = pc.sample_colorscale("Reds", [norm])[0]
        label = dt.strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter3d(
            x=-part["openInterest"], y=part["strike"], z=[z_val] * len(part),
            mode="markers",
            marker=dict(size=part["openInterest"] / max(1, pf["openInterest"].max()) * 20,
                        color=color, opacity=0.8),
            name=f"Puts {label}",
            hovertemplate="Strike: %{y}<br>Open Interest: %{-x}<br>Expiration: " + label + "<extra></extra>"
        ))
    for dt in all_exps:
        part = cf[cf["expiration_dt"] == dt]
        if part.empty:
            continue
        z_val = date_to_z[dt]
        dte = (dt - now).days
        norm = 0.2 + 0.8 * ((dte - min_days) / max(1, (max_days - min_days)))
        color = pc.sample_colorscale("Greens", [norm])[0]
        label = dt.strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter3d(
            x=part["openInterest"], y=part["strike"], z=[z_val] * len(part),
            mode="markers",
            marker=dict(size=part["openInterest"] / max(1, cf["openInterest"].max()) * 20,
                        color=color, opacity=0.8),
            name=f"Calls {label}",
            hovertemplate="Strike: %{y}<br>Open Interest: %{x}<br>Expiration: " + label + "<extra></extra>"
        ))
    if current_price is not None:
        fig.add_trace(go.Scatter3d(
            x=[0] * len(all_exps), y=[current_price] * len(all_exps), z=list(range(len(all_exps))),
            mode="lines", line=dict(color="yellow", width=5), name=f"Current Price: ${current_price:.2f}"
        ))
    z_ticks = [dt.strftime("%Y-%m-%d") for dt in all_exps]
    title = f"{ticker} Options Open Interest (3D Bubbles)" + (f" — Spot ${current_price:.2f}" if current_price else "")
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Open Interest", yaxis_title="Strike Price", zaxis_title="Expiration Date",
            zaxis=dict(tickvals=list(range(len(all_exps))), ticktext=z_ticks, tickangle=45),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        template="plotly_dark", margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig


def _fig_aggregated_2d(calls_data: pd.DataFrame, puts_data: pd.DataFrame,
                       ticker: str, current_price: Optional[float]) -> go.Figure:
    calls = calls_data.copy()
    puts = puts_data.copy()
    c_m, c_s = calls["openInterest"].mean(), calls["openInterest"].std()
    p_m, p_s = puts["openInterest"].mean(), puts["openInterest"].std()
    c_th = (c_m if pd.notna(c_m) else 0) + (c_s if pd.notna(c_s) else 0)
    p_th = (p_m if pd.notna(p_m) else 0) + (p_s if pd.notna(p_s) else 0)
    cf = calls[calls["openInterest"] > c_th]
    pf = puts[puts["openInterest"] > p_th]
    all_days = pd.concat([cf["days_to_exp"], pf["days_to_exp"]])
    min_d, max_d = (all_days.min(), all_days.max()) if not all_days.empty else (0, 1)

    fig = go.Figure()
    for exp in sorted(pf["expiration"].unique()):
        part = pf[pf["expiration"] == exp]
        if part.empty:
            continue
        dte = part["days_to_exp"].iloc[0]
        norm = 0.2 + 0.8 * ((dte - min_d) / max(1, (max_d - min_d)))
        color = pc.sample_colorscale("Reds", [norm])[0]
        fig.add_trace(go.Bar(
            x=-part["openInterest"], y=part["strike"], orientation="h",
            marker=dict(color=color, opacity=0.7), name=f"Puts {exp}"
        ))
    for exp in sorted(cf["expiration"].unique()):
        part = cf[cf["expiration"] == exp]
        if part.empty:
            continue
        dte = part["days_to_exp"].iloc[0]
        norm = 0.2 + 0.8 * ((dte - min_d) / max(1, (max_d - min_d)))
        color = pc.sample_colorscale("Greens", [norm])[0]
        fig.add_trace(go.Bar(
            x=part["openInterest"], y=part["strike"], orientation="h",
            marker=dict(color=color, opacity=0.7), name=f"Calls {exp}"
        ))
    ttl = f"{ticker} OI (Aggregated 2D)" + (f" — Spot ${current_price:.2f}" if current_price else "")
    fig.update_layout(title=ttl, xaxis_title="Open Interest", yaxis_title="Strike", template="plotly_dark", barmode="overlay")
    
    if not cf.empty:
        y_min, y_max = cf["strike"].min(), cf["strike"].max()
        fig.add_shape(type="line", x0=c_th, x1=c_th, y0=y_min, y1=y_max, line=dict(color="yellow", dash="dash"))
    if not pf.empty:
        y_min, y_max = pf["strike"].min(), pf["strike"].max()
        fig.add_shape(type="line", x0=-p_th, x1=-p_th, y0=y_min, y1=y_max, line=dict(color="yellow", dash="dash"))
    return fig


def _fig_3d_oi(calls_data: pd.DataFrame, puts_data: pd.DataFrame,
               ticker: str, current_price: Optional[float]) -> go.Figure:
    calls = calls_data.copy().dropna(subset=["openInterest"])
    puts = puts_data.copy().dropna(subset=["openInterest"])

    if calls.empty and puts.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{ticker} — 3D OI (Z=OI) - No Data",
            template="plotly_dark", margin=dict(l=0, r=0, b=0, t=50)
        )
        return fig

    all_oi = pd.concat([
        calls["openInterest"] if not calls.empty else pd.Series(dtype=float),
        puts["openInterest"] if not puts.empty else pd.Series(dtype=float)
    ])
    max_oi = all_oi.max() if not all_oi.empty and pd.notna(all_oi.max()) else 0.0

    all_days = pd.concat([calls["days_to_exp"], puts["days_to_exp"]])
    min_d, max_d = (all_days.min(), all_days.max()) if not all_days.empty else (0, 1)
    fig = go.Figure()

    if not calls.empty:
        fig.add_trace(go.Scatter3d(
            x=calls["strike"], y=calls["days_to_exp"], z=calls["openInterest"], mode="markers",
            marker=dict(size=calls["openInterest"] / max(1.0, max_oi) * 20 + 2, color=calls["days_to_exp"],
                        colorscale="Greens", opacity=0.7, showscale=True),
            name="Calls", text=calls.get("contractSymbol", "")
        ))
    if not puts.empty:
        fig.add_trace(go.Scatter3d(
            x=puts["strike"], y=puts["days_to_exp"], z=puts["openInterest"], mode="markers",
            marker=dict(size=puts["openInterest"] / max(1.0, max_oi) * 20 + 2, color=puts["days_to_exp"],
                        colorscale="Reds", opacity=0.7, showscale=False),
            name="Puts", text=puts.get("contractSymbol", "")
        ))

    if current_price is not None:
        fig.add_trace(go.Scatter3d(
            x=[current_price, current_price], y=[min_d, max_d], z=[0, 0],
            mode="lines", line=dict(color="yellow", width=5), name=f"Spot ${current_price:.2f}"
        ))
    fig.update_layout(
        title=f"{ticker} — 3D OI (Z=OI)", template="plotly_dark",
        scene=dict(xaxis_title="Strike", yaxis_title="DTE", zaxis_title="Open Interest"),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig


def _fig_distribution_dashboard(calls: pd.DataFrame, puts: pd.DataFrame,
                               ticker: str, current_price: Optional[float]) -> go.Figure:
    gc = calls.groupby("expiration")
    gp = puts.groupby("expiration")
    expirations = sorted(set(gc.groups.keys()) | set(gp.groups.keys()))
    pcr, c_avg, p_avg, c_skew, p_skew = [], [], [], [], []
    for exp in expirations:
        c = gc.get_group(exp) if exp in gc.groups else pd.DataFrame(columns=["strike", "openInterest"])
        p = gp.get_group(exp) if exp in gp.groups else pd.DataFrame(columns=["strike", "openInterest"])
        c_tot, p_tot = c["openInterest"].sum(), p["openInterest"].sum()
        pcr.append(p_tot / c_tot if c_tot > 0 else np.nan)
        if c_tot > 0:
            w = np.average(c["strike"], weights=c["openInterest"]); c_avg.append(w)
            var = np.average((c["strike"] - w) ** 2, weights=c["openInterest"]); std = np.sqrt(var) if var > 0 else 1
            c_skew.append(np.average(((c["strike"] - w) / std) ** 3, weights=c["openInterest"]))
        else:
            c_avg.append(np.nan); c_skew.append(np.nan)
        if p_tot > 0:
            w = np.average(p["strike"], weights=p["openInterest"]); p_avg.append(w)
            var = np.average((p["strike"] - w) ** 2, weights=p["openInterest"]); std = np.sqrt(var) if var > 0 else 1
            p_skew.append(np.average(((p["strike"] - w) / std) ** 3, weights=p["openInterest"]))
        else:
            p_avg.append(np.nan); p_skew.append(np.nan)

    fig = make_subplots(
        rows=4, cols=1, specs=[[{}], [{}], [{}], [{"type": "table"}]],
        subplot_titles=("Put-Call Ratio per Expiration", "Weighted Average Strike",
                        "Weighted Skewness of Strikes", "Significant Strikes"),
        vertical_spacing=0.08
    )
    fig.add_trace(go.Bar(x=expirations, y=pcr, name="PCR", marker_color="yellow"), row=1, col=1)
    fig.add_trace(go.Scatter(x=expirations, y=c_avg, mode="lines+markers", name="Call Avg", line=dict(color="lightgreen")), row=2, col=1)
    fig.add_trace(go.Scatter(x=expirations, y=p_avg, mode="lines+markers", name="Put Avg", line=dict(color="#FF6347")), row=2, col=1)
    if current_price is not None:
        fig.add_trace(go.Scatter(x=expirations, y=[current_price] * len(expirations), mode="lines",
                                 name="Spot", line=dict(color="white", dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=expirations, y=c_skew, mode="lines+markers", name="Call Skew", line=dict(color="lightgreen")), row=3, col=1)
    fig.add_trace(go.Scatter(x=expirations, y=p_skew, mode="lines+markers", name="Put Skew", line=dict(color="#FF6347")), row=3, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="white", row=3, col=1)

    c_thr = calls["openInterest"].quantile(0.90) if not calls.empty else 0
    p_thr = puts["openInterest"].quantile(0.90) if not puts.empty else 0
    header = ["Expiration", "Calls (Strike / OI)", "Puts (Strike / OI)"]
    cells = [[], [], []]
    for exp in expirations:
        c = gc.get_group(exp) if exp in gc.groups else pd.DataFrame()
        p = gp.get_group(exp) if exp in gp.groups else pd.DataFrame()
        sig_c = c[c["openInterest"] > c_thr][["strike", "openInterest"]].sort_values("openInterest", ascending=False).head(5)
        sig_p = p[p["openInterest"] > p_thr][["strike", "openInterest"]].sort_values("openInterest", ascending=False).head(5)
        cells[0].append(exp)
        cells[1].append("<br>".join([f"{row['strike']:.2f} / {int(row['openInterest'])}" for _, row in sig_c.iterrows()]))
        cells[2].append("<br>".join([f"{row['strike']:.2f} / {int(row['openInterest'])}" for _, row in sig_p.iterrows()]))
    fig.add_trace(go.Table(header=dict(values=header, fill_color="grey", align="center",
                                       font=dict(color="white", size=12), height=40),
                           cells=dict(values=cells, fill_color="#1E1E1E", align="center",
                                      font=dict(color="white", size=11), height=28)), row=4, col=1)
    fig.update_layout(
        height=1400, title=f"{ticker} — Options Distribution & Skew Analysis",
        template="plotly_dark", showlegend=True, margin=dict(l=40, r=20, t=60, b=20)
    )
    return fig


def plot_oi_dashboard(all_options_df: pd.DataFrame, ticker: str,
                      current_price: Optional[float] = None) -> Dict[str, go.Figure]:
    """
    Return dict of four figures:
      - 'bubbles3d', 'aggregated2d', 'oi3d', 'distribution'
    """
    d = _prep_options_for_dashboard(all_options_df)
    calls = d[d.get("optionType") == "call"].copy()
    puts = d[d.get("optionType") == "put"].copy()
    figs = {
        "bubbles3d": _fig_3d_bubbles(calls, puts, ticker, current_price),
        "aggregated2d": _fig_aggregated_2d(calls, puts, ticker, current_price),
        "oi3d": _fig_3d_oi(calls, puts, ticker, current_price),
        "distribution": _fig_distribution_dashboard(calls, puts, ticker, current_price),
    }
    return figs


# --- Additional charts ----------------------------------------------------------
def plot_additional_charts(all_options_df: pd.DataFrame, ticker: str,
                           quantile: float = 0.95) -> Dict[str, go.Figure]:
    """
    Return dict with:
      - 'oi_by_strike' : total OI by strike (bar)
      - 'unusual_activity' : scatter (OI vs Strike), points above OI/Vol quantile
      - 'oi_heatmap' : strike × expiration heatmap of OI
    """
    d = all_options_df.copy()
    d["openInterest"] = pd.to_numeric(d.get("openInterest"), errors="coerce")
    d["volume"] = pd.to_numeric(d.get("volume"), errors="coerce")
    d["strike"] = pd.to_numeric(d.get("strike"), errors="coerce")

    # 1) OI by strike
    d1 = d.dropna(subset=["openInterest", "strike"])
    agg = d1.groupby("strike")["openInterest"].sum().reset_index().sort_values("strike")
    fig_oi_by_strike = go.Figure(go.Bar(x=agg["strike"], y=agg["openInterest"]))
    fig_oi_by_strike.update_layout(
        title=f"{ticker} — Total Open Interest by Strike", xaxis_title="Strike",
        yaxis_title="Open Interest", template="plotly_dark", margin=dict(l=50, r=20, t=60, b=60)
    )

    # 2) Unusual activity
    d2 = d.dropna(subset=["openInterest", "volume", "strike"])
    if not d2.empty:
        oi_th = d2["openInterest"].quantile(quantile)
        vol_th = d2["volume"].quantile(quantile)
        subset = d2[(d2["openInterest"] > oi_th) | (d2["volume"] > vol_th)]
    else:
        subset = d2
    fig_unusual = go.Figure()
    for exp in sorted(subset.get("expiration", pd.Series(dtype=str)).unique()):
        part = subset[subset["expiration"] == exp]
        if part.empty:
            continue
        fig_unusual.add_trace(go.Scatter(
            x=part["openInterest"], y=part["strike"], mode="markers", name=str(exp),
            text=part.get("contractSymbol", ""),
            hovertemplate="<b>Strike:</b> %{y}<br><b>OI:</b> %{x}<br><b>Exp:</b> " + str(exp) +
                          "<br><b>Contract:</b> %{text}<extra></extra>",
            opacity=0.85
        ))
    fig_unusual.update_layout(
        title=f"{ticker} — Unusual Activity (>{int(quantile*100)}th pct on OI or Vol)",
        xaxis_title="Open Interest", yaxis_title="Strike",
        template="plotly_dark", margin=dict(l=50, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 3) OI heatmap
    heat = d.dropna(subset=["openInterest", "strike"]).pivot_table(
        index="strike", columns="expiration", values="openInterest", aggfunc="sum", fill_value=0
    )
    heat = heat.sort_index().sort_index(axis=1)
    fig_heatmap = go.Figure(go.Heatmap(z=heat.values,
                                       x=list(map(str, heat.columns.tolist())),
                                       y=heat.index.values,
                                       coloraxis="coloraxis"))
    fig_heatmap.update_layout(
        title=f"{ticker} — OI Heatmap (Strike × Expiration)",
        xaxis_title="Expiration", yaxis_title="Strike",
        coloraxis=dict(colorscale="Turbo"),
        template="plotly_dark", margin=dict(l=50, r=20, t=60, b=60)
    )

    return {
        "oi_by_strike": fig_oi_by_strike,
        "unusual_activity": fig_unusual,
        "oi_heatmap": fig_heatmap,
    }
