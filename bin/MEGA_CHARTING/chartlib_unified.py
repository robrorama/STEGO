#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chartlib_unified.py
Unified plotting/analysis library (Matplotlib + Plotly + mplfinance + TA-Lib)
- Only imports data from data_retrieval.py
- Opens Plotly charts as HTML tabs
- Writes Matplotlib/mplfinance PNGs and per-image viewer HTMLs
- Builds a single mega PDF (images only) including the run log (rendered as an image)
"""

from __future__ import annotations
import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import io, time, math, webbrowser, pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from string import Template
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------- Headless Matplotlib ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional libs (graceful degradation)
try:
    import statsmodels.api as sm
except Exception:
    sm = None

try:
    import mplfinance as mpf
except Exception:
    mpf = None

try:
    import talib
except Exception:
    talib = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

try:
    from fpdf import FPDF
except Exception:
    FPDF = None

# Plotly stack
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

# FFT
from numpy.fft import fft

# Project data layer (ONLY allowed import)
try:
    import data_retrieval as dr  # required
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)


# ============================== Utilities ===============================

@dataclass
class RunResult:
    png_files: List[str]  = field(default_factory=list)
    html_files: List[str] = field(default_factory=list)
    gif_files: List[str]  = field(default_factory=list)
    pdf_files: List[str]  = field(default_factory=list)
    log_file: Optional[str] = None


@dataclass
class Paths:
    root: str
    png: str
    html: str
    html_viewers: str
    pdf: str
    gif: str
    logs: str


def _mk_dirs(root: str) -> Paths:
    png  = os.path.join(root, "PNGS");       os.makedirs(png, exist_ok=True)
    html = os.path.join(root, "HTML");       os.makedirs(html, exist_ok=True)
    view = os.path.join(html, "viewers");    os.makedirs(view, exist_ok=True)
    pdf  = os.path.join(root, "PDFS");       os.makedirs(pdf, exist_ok=True)
    gif  = os.path.join(root, "GIF");        os.makedirs(gif, exist_ok=True)
    logs = os.path.join(root, "LOGS");       os.makedirs(logs, exist_ok=True)
    return Paths(root=root, png=png, html=html, html_viewers=view, pdf=pdf, gif=gif, logs=logs)


def _now_ts() -> str:
    return time.strftime("%H:%M:%S")


# Progress logger (to console + buffer)
class _Logger:
    def __init__(self): self.lines: List[str] = []
    def log(self, msg: str):
        line = f"[{_now_ts()}] {msg}"
        print(line, flush=True)
        self.lines.append(line)

    def write_files(self, paths: Paths) -> Tuple[str, Optional[str]]:
        # text file
        txt_path = os.path.join(paths.logs, "run_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")

        # render a PNG image of the log for PDF (ASCII-safe)
        png_path = None
        if Image is not None:
            try:
                # sanitize non-ascii visually but keep information
                text = "\n".join(self.lines).replace("σ", "sigma").replace("—", "-").replace("–", "-")
                # crude wrapping
                lines = []
                for raw in text.split("\n"):
                    if len(raw) <= 110:
                        lines.append(raw)
                    else:
                        s = raw
                        while len(s) > 110:
                            lines.append(s[:110])
                            s = s[110:]
                        if s: lines.append(s)
                W, H = 2400, 3200
                img = Image.new("RGB", (W, H), (18, 18, 18))
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                x, y = 30, 30
                for ln in ["RUN REPORT", ""] + lines:
                    if y > H - 40: break
                    draw.text((x, y), ln, fill=(230,230,230), font=font, spacing=4)
                    y += 26
                png_path = os.path.join(paths.png, "run_report.png")
                img.save(png_path, format="PNG")
            except Exception:
                png_path = None

        return txt_path, png_path


def _latin1_safe(s: str) -> str:
    # For any last-resort text that *must* be passed to FPDF (we avoid it anyway)
    return s.replace("σ", "sigma").replace("—", "-").replace("–", "-")


def _save_mpl(fig: plt.Figure, out_path: str, res: RunResult):
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    res.png_files.append(out_path)


_VIEWER_TMPL = Template("""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
<meta http-equiv="Pragma" content="no-cache"/><meta http-equiv="Expires" content="0"/>
<title>${title}</title>
<style>html,body{height:100%;margin:0;background:#111}
#viewport{width:100%;height:100%;overflow:hidden;position:relative}
img{transform-origin:0 0;cursor:grab;user-select:none;-webkit-user-drag:none;position:absolute;left:0;top:0}
.info{position:absolute;left:8px;top:8px;color:#eee;background:rgba(0,0,0,.35);padding:4px 6px;border-radius:4px;font:12px monospace}
</style></head><body>
<div id="viewport"><div class="info">wheel: zoom · drag: pan · dblclick: reset</div><img id="img" src="${img_rel}"></div>
<script>
(function(){let s=1,ox=0,oy=0,d=false,lx=0,ly=0;const im=document.getElementById('img');
function A(){im.style.transform='translate('+ox+'px,'+oy+'px) scale('+s+')';}
document.addEventListener('wheel',function(e){e.preventDefault();const r=im.getBoundingClientRect();
const mx=e.clientX-r.left,my=e.clientY-r.top;const k=(e.deltaY<0)?1.1:0.9;ox=mx-(mx-ox)*k;oy=my-(my-oy)*k;s*=k;s=Math.max(0.1,Math.min(s,100));A();},{passive:false});
im.addEventListener('mousedown',e=>{d=true;lx=e.clientX;ly=e.clientY;im.style.cursor='grabbing';});
window.addEventListener('mouseup',()=>{d=false;im.style.cursor='grab';});
window.addEventListener('mousemove',e=>{if(!d)return;ox+=e.clientX-lx;oy+=e.clientY-ly;lx=e.clientX;ly=e.clientY;A();});
window.addEventListener('dblclick',()=>{s=1;ox=0;oy=0;A();});A();})();
</script></body></html>""")


def _write_viewer(img_path: str, title: str, paths: Paths) -> str:
    img_rel = os.path.relpath(img_path, start=paths.html_viewers)
    out = os.path.join(paths.html_viewers, f"view_{os.path.basename(img_path).replace('.png','')}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(_VIEWER_TMPL.substitute(title=title, img_rel=img_rel))
    return out


def open_tabs(urls: List[str], max_tabs: int = 256, tab_delay_ms: int = 60):
    urls = urls[:max_tabs]
    for i, u in enumerate(urls):
        p = pathlib.Path(u)
        url = p.as_uri() if p.exists() else u
        if i == 0:
            webbrowser.open_new(url)
        else:
            webbrowser.open_new_tab(url)
        time.sleep(max(0, tab_delay_ms) / 1000.0)


# ============================== Data helpers =============================

def _load_prices(ticker: str, start: Optional[str], end: Optional[str], period: str) -> pd.DataFrame:
    if start and end:
        df = dr.load_or_download_ticker(ticker, start=start, end=end)
    else:
        df = dr.load_or_download_ticker(ticker, period=period or "6mo")
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)
    return df


# ============================== Matplotlib set ===========================

def mpl_lrc_smas(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Matplotlib: LRC+SMAs")
    x = np.arange(len(df.index))
    y = df["Close"].values.astype(float)
    # Linear reg via polyfit
    sl, itcpt = np.polyfit(x, y, 1)
    yhat = sl * x + itcpt
    std = float(np.std(yhat)) if len(yhat) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.2, color="tab:blue")
    for col, c in [("SMA30","green"),("SMA50","orange"),("SMA100","blue"),("SMA200","purple"),("SMA300","red")]:
        if col in df:
            ax.plot(df.index, df[col], label=col, linewidth=1.0, color=c)
    ax.plot(df.index, yhat, label="LRC", color="black", linestyle="--", linewidth=1.2)
    for k in [0.25,0.5,0.75,1,1.25,1.5,1.75,2,3,4,5]:
        ax.plot(df.index, yhat + k*std, color="gray", linewidth=0.5, alpha=0.6)
        ax.plot(df.index, yhat - k*std, color="gray", linewidth=0.5, alpha=0.6)
    ax.set_title(f"{ticker} LRC + SMAs"); ax.grid(True, linestyle="--", alpha=0.4); ax.legend()
    out = os.path.join(paths.png, f"{ticker.lower()}_lrc_smas.png")
    _save_mpl(fig, out, res)
    res.html_files.append(_write_viewer(out, os.path.basename(out), paths))
    L.log(f"PNG saved: {out}")


def mpl_outliers_3sigma(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Matplotlib: Outliers 3σ")
    s = df[['Close']].copy()
    s['simple_rtn'] = s['Close'].pct_change()
    roll = s[['simple_rtn']].rolling(window=21).agg(['mean','std'])
    roll.columns = roll.columns.droplevel()
    s = s.join(roll)
    s['outlier'] = ((s['simple_rtn'] > s['mean'] + 3*s['std']) | (s['simple_rtn'] < s['mean'] - 3*s['std'])).astype(int)
    outliers = s[s['outlier'] == 1]
    L.log(f"3σ outliers: {len(outliers)}")
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(s.index, s['simple_rtn'], label="Return", linewidth=0.6)
    ax.scatter(outliers.index, outliers['simple_rtn'], color="red", s=36, label="Anomaly (3 sigma)")
    ax.set_title(f"{ticker} Returns & 3-sigma Outliers"); ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
    out = os.path.join(paths.png, f"{ticker.lower()}_outliers_3sigma.png")
    _save_mpl(fig, out, res)
    res.html_files.append(_write_viewer(out, os.path.basename(out), paths))
    L.log(f"PNG saved: {out}")


def mpl_streaks(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger) -> Dict[int, Tuple[int,int,int,float]]:
    L.log("Matplotlib: Streaks")
    d = df[['Close']].copy()
    d['r'] = d['Close'].pct_change()
    d.dropna(inplace=True)
    d['dir'] = np.where(d['r'] > 0, 1, np.where(d['r'] < 0, -1, 0))
    d['grp'] = (d['dir'] != d['dir'].shift()).cumsum()
    d['len'] = d.groupby('grp').cumcount()+1
    d['signed'] = d['len'] * d['dir']

    # frequency table for 1..9 days
    total = len(d)
    freqs: Dict[int, Tuple[int,int,int,float]] = {}
    for k in range(1, 10):
        up = int((d['signed'] ==  k).sum())
        dn = int((d['signed'] == -k).sum())
        allk = up + dn
        pct = (allk/total*100.0) if total else 0.0
        freqs[k] = (up, dn, allk, pct)
        L.log(f"Streak {k}d — Up:{up} Down:{dn} [{pct:.2f}%]")

    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(d.index, d['signed'], color="green", linewidth=1.2, label="Signed streak")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_title(f"{ticker} Consecutive Up/Down Streaks"); ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
    out = os.path.join(paths.png, f"{ticker.lower()}_streaks.png")
    _save_mpl(fig, out, res)
    res.html_files.append(_write_viewer(out, os.path.basename(out), paths))
    L.log(f"PNG saved: {out}")
    return freqs


def mpl_derivatives(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Matplotlib: Derivatives")
    s1 = df['Close'].diff()
    s2 = s1.diff()
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(df.index, s1, label="First derivative", linewidth=1.0, color="gold")
    ax.plot(df.index, s2, label="Second derivative", linewidth=1.0, color="brown")
    ax.set_title(f"{ticker} First & Second Derivatives"); ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
    out = os.path.join(paths.png, f"{ticker.lower()}_derivatives.png")
    _save_mpl(fig, out, res)
    res.html_files.append(_write_viewer(out, os.path.basename(out), paths))
    L.log(f"PNG saved: {out}")


def mpl_dist_qq(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Matplotlib: Dist + Q-Q")
    if sm is None:
        L.log("statsmodels not available; skipping Q-Q.")
        return
    s = df[['Close']].copy()
    s['logr'] = np.log(s['Close'] / s['Close'].shift(1))
    s.dropna(inplace=True)
    mu, sigma = float(s['logr'].mean()), float(s['logr'].std())
    x = np.linspace(s['logr'].min(), s['logr'].max(), 1000)
    norm_pdf = (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2) if sigma > 0 else np.zeros_like(x)
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    ax[0].hist(s['logr'].values, bins=60, density=True, color="tab:blue", alpha=0.7)
    if sigma > 0: ax[0].plot(x, norm_pdf, color="green", linewidth=2, label=f"N({mu:.3f},{sigma**2:.5f})"); ax[0].legend()
    sm.qqplot(s['logr'].values, line='s', ax=ax[1])
    ax[0].set_title("Log-return distribution"); ax[1].set_title("Q-Q plot")
    out = os.path.join(paths.png, f"{ticker.lower()}_dist_qq.png")
    _save_mpl(fig, out, res)
    res.html_files.append(_write_viewer(out, os.path.basename(out), paths))
    L.log(f"PNG saved: {out}")


def mpl_polyfit_fft(series: pd.Series, series_name: str, ticker: str, poly_level: int,
                    paths: Paths, res: RunResult, L: _Logger):
    # Robust version compatible with older Matplotlib (no 'use_line_collection')
    clean = series.dropna()
    if len(clean) < poly_level + 1:
        return
    L.log(f"Matplotlib: Polyfit+FFT on {series_name} (deg={poly_level})")
    x = np.arange(len(clean)); y = clean.values.astype(float)
    coeffs = np.polyfit(x, y, poly_level); model = np.poly1d(coeffs)
    xp = np.linspace(x.min(), x.max(), 500)
    # map to dates for plotting
    dates = clean.index
    if len(dates) > 1:
        date_line = pd.date_range(start=dates.min(), end=dates.max(), periods=len(xp))
    else:
        date_line = dates

    fig1 = plt.figure(figsize=(16,8))
    plt.scatter(dates, y, s=10, alpha=0.7, label="Data")
    plt.plot(date_line, model(xp), label=f"Polyfit({poly_level})", linewidth=1.5)
    plt.title(f"{ticker} — {series_name} polyfit {poly_level}")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    out1 = os.path.join(paths.png, f"{ticker.lower()}_polyfit_{series_name.replace(' ','_').lower()}_{poly_level}.png")
    _save_mpl(fig1, out1, res); res.html_files.append(_write_viewer(out1, os.path.basename(out1), paths)); L.log(f"PNG saved: {out1}")

    # FFT
    X = fft(y); N = len(X); n = np.arange(N); sampling_interval = 1.0
    freq = n / (N * sampling_interval); nyq = 0.5 / sampling_interval
    fig2 = plt.figure(figsize=(16,8))
    plt.title(f"{ticker} — FFT ({series_name})")
    plt.stem(freq[:N//2], np.abs(X[:N//2]))  # no deprecated kwargs
    plt.xlim(0, nyq); plt.xlabel("Frequency (cycles/day)"); plt.ylabel("|X(freq)|")
    out2 = os.path.join(paths.png, f"{ticker.lower()}_fft_{series_name.replace(' ','_').lower()}_{poly_level}.png")
    _save_mpl(fig2, out2, res); res.html_files.append(_write_viewer(out2, os.path.basename(out2), paths)); L.log(f"PNG saved: {out2}")


def mplfinance_lrc(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger, period: int = 144):
    if mpf is None:
        L.log("mplfinance not available; skipping.")
        return
    L.log("mplfinance: long-term LRC")
    if len(df) < period:
        L.log(f"Not enough data for {period} bars; skipping mplfinance LRC.")
        return
    sub = df.iloc[-period:].copy()
    # Use log-regression if all positive
    val = np.log(sub['Close']) if (sub['Close'] > 0).all() else sub['Close']
    x = np.arange(len(sub))
    sl, itcpt = np.polyfit(x, val.values, 1)
    reg = sl * np.arange(len(df)) + itcpt
    std = float((val - (sl*x + itcpt)).std())
    ap = [mpf.make_addplot(np.exp(reg) if (df['Close']>0).all() else reg, width=1.5)]
    for dev in [1,2,3]:
        for sgn in [-1,1]:
            band = reg + sgn*dev*std
            ap.append(mpf.make_addplot(np.exp(band) if (df['Close']>0).all() else band, linestyle='--', width=0.8))
    out = os.path.join(paths.png, f"{ticker.lower()}_mplf_lrc_{period}.png")
    mpf.plot(df, type='candle', style='charles', title=f"{ticker} LRC ({period}d)",
             addplot=ap, volume=True, figsize=(15,10), savefig=out)
    res.png_files.append(out); res.html_files.append(_write_viewer(out, os.path.basename(out), paths))
    L.log(f"mplfinance PNG saved: {out}")


# ============================== Plotly set ===============================

def _write_plotly(fig: go.Figure, base: str, ticker: str, paths: Paths,
                  res: RunResult, L: _Logger,
                  write_png_for_pdf: bool = True,
                  height: Optional[int] = None):
    html = os.path.join(paths.html, f"{ticker.lower()}_{base}.html")
    png  = os.path.join(paths.png,  f"{ticker.lower()}_{base}.png")
    if height:
        fig.update_layout(height=height)
    pio.write_html(fig, file=html, auto_open=False, include_plotlyjs=True, full_html=True)
    res.html_files.append(html)
    L.log(f"HTML saved: {html}")
    if write_png_for_pdf:
        try:
            fig.write_image(png, width=1600, height=900, scale=2.0)
            res.png_files.append(png)  # added to PDF later; no viewer page (avoid dup tabs)
            L.log(f"Plotly PNG saved (for PDF only): {png}")
        except Exception as e:
            L.log(f"[warn] Plotly PNG export failed: {e} (install kaleido)")

# LRC helpers
def _lrc_yhat_from_series(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y))
    sl, itcpt = np.polyfit(x, y, 1)
    return sl * x + itcpt

def pl_lrc_smas(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: LRC+SMAs")
    y = df["Close"].values.astype(float); yhat = _lrc_yhat_from_series(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    for col, c in [("SMA30","green"),("SMA50","orange"),("SMA100","blue"),("SMA200","purple"),("SMA300","red")]:
        if col in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    std = float(np.std(yhat)) if len(yhat) > 1 else 0.0
    fig.add_trace(go.Scatter(x=df.index, y=yhat, mode="lines", name="LRC", line=dict(width=1.4, dash="dash")))
    for k in [0.25,0.5,0.75,1,1.25,1.5,1.75,2,3,4,5]:
        fig.add_trace(go.Scatter(x=df.index, y=yhat + k*std, mode="lines", line=dict(width=0.6, color="rgba(200,200,200,0.5)"), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=yhat - k*std, mode="lines", line=dict(width=0.6, color="rgba(200,200,200,0.5)"), showlegend=False))
    fig.update_layout(template="plotly_dark", title=f"{ticker} LRC + SMAs", legend=dict(orientation="h", y=1.02))
    _write_plotly(fig, "pl_lrc_smas", ticker, paths, res, L)

def pl_outliers(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Outliers")
    s = df[['Close']].copy()
    s['simple_rtn'] = s['Close'].pct_change()
    roll = s[['simple_rtn']].rolling(21).agg(['mean','std'])
    roll.columns = roll.columns.droplevel()
    s = s.join(roll)
    s['outlier'] = ((s['simple_rtn'] > s['mean'] + 3*s['std']) | (s['simple_rtn'] < s['mean'] - 3*s['std'])).astype(int)
    outs = s[s['outlier']==1]
    L.log(f"Plotly 3σ outliers: {len(outs)}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s['simple_rtn'], mode="lines+markers", marker=dict(size=3), name="Return"))
    if not outs.empty:
        fig.add_trace(go.Scatter(x=outs.index, y=outs['simple_rtn'], mode="markers", marker=dict(size=7, color="red"), name="Anomaly (3 sigma)"))
    fig.update_layout(template="plotly_dark", title=f"{ticker} Returns & 3-sigma Outliers")
    _write_plotly(fig, "pl_outliers", ticker, paths, res, L)

def pl_streaks_line(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Streaks")
    s = df[['Close']].copy()
    s['r'] = s['Close'].pct_change()
    s.dropna(inplace=True)
    s['dir'] = np.where(s['r'] > 0, 1, np.where(s['r'] < 0, -1, 0))
    s['grp'] = (s['dir'] != s['dir'].shift()).cumsum()
    s['len'] = s.groupby('grp').cumcount()+1
    s['signed'] = s['len'] * s['dir']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s['signed'], mode="lines", name="Signed streak"))
    fig.add_hline(y=0, line_width=0.6, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_dark", title=f"{ticker} Consecutive Up/Down Streaks")
    _write_plotly(fig, "pl_streaks", ticker, paths, res, L)

def pl_streaks_bar_counts(freqs: Dict[int, Tuple[int,int,int,float]], ticker: str,
                          paths: Paths, res: RunResult, L: _Logger):
    # Up/Down counts for 1..9 days; emphasize 5d
    if not freqs:
        return
    L.log("Plotly: Streak counts (1..9d) with 5d highlight")
    ks = sorted(freqs.keys())
    ups = [freqs[k][0] for k in ks]
    dns = [freqs[k][1] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ks, y=ups, name="Up runs"))
    fig.add_trace(go.Bar(x=ks, y=dns, name="Down runs"))
    fig.update_layout(barmode="group", template="plotly_dark",
                      title=f"{ticker} Up/Down Run Counts (1..9 days)")
    # Emphasize 5-day bars
    if 5 in ks:
        i5 = ks.index(5)
        fig.add_vrect(x0=4.5, x1=5.5, fillcolor="rgba(255,255,255,0.08)", line_width=0)
        fig.add_annotation(x=5, y=max(ups[i5], dns[i5]) * 1.05,
                           text="5-day focus", showarrow=False)
    _write_plotly(fig, "pl_streaks_summary_1to9d", ticker, paths, res, L, height=500)

def pl_dist_qq(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Distribution + Q-Q")
    s = df[['Close']].copy()
    s['logr'] = np.log(s['Close'] / s['Close'].shift(1))
    s.dropna(inplace=True)
    if s.empty:
        return
    mu = float(s['logr'].mean()); sigma = float(s['logr'].std())
    x = np.linspace(s['logr'].min(), s['logr'].max(), 400)
    pdf = (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2) if sigma>0 else np.zeros_like(x)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution", "Q-Q"))
    fig.add_trace(go.Histogram(x=s['logr'], histnorm="probability density", nbinsx=60, name="logr"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines", name=f"N({mu:.4f},{sigma**2:.6f})"), row=1, col=1)
    # Q-Q
    if sm is not None:
        q = np.sort((s['logr'] - mu) / (sigma if sigma>0 else 1.0))
        th = np.sort(np.random.normal(size=len(q)))
        fig.add_trace(go.Scatter(x=th, y=q, mode="markers", name="Q-Q"), row=1, col=2)
    else:
        # fallback: simple order vs order
        q = np.sort(s['logr'].values)
        th = np.sort(q)
        fig.add_trace(go.Scatter(x=th, y=q, mode="markers", name="Q-Q"), row=1, col=2)
    fig.update_layout(template="plotly_dark", height=650)
    _write_plotly(fig, "pl_dist_qq", ticker, paths, res, L)

def pl_first_derivative(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: First derivative")
    s = df["Close"].diff(-1) * -1
    s = s.dropna()
    if len(s) < 2:
        L.log(f"Warning: Not enough data points ({len(s)}) to plot first derivative LRC.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="First Derivative"))
    yhat = _lrc_yhat_from_series(s.values.astype(float))
    std = float(np.std(yhat))
    fig.add_trace(go.Scatter(x=s.index, y=yhat, mode="lines", name="LRC", line=dict(width=1.2, dash="dash")))
    for k in [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5]:
        fig.add_trace(go.Scatter(x=s.index, y=yhat+k*std, mode="lines", line=dict(width=0.6, color="rgba(200,200,200,0.5)"), showlegend=False))
        fig.add_trace(go.Scatter(x=s.index, y=yhat-k*std, mode="lines", line=dict(width=0.6, color="rgba(200,200,200,0.5)"), showlegend=False))
    fig.update_layout(template="plotly_dark", title=f"{ticker} First Derivative")
    _write_plotly(fig, "pl_first_derivative", ticker, paths, res, L)

# ---------- Ichimoku + BB + DMAs (unchanged behavior) ----------
def _median_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2: return pd.Timedelta(days=1)
    diffs = np.diff(index.values).astype("timedelta64[ns]").astype(np.int64)
    med = np.median(diffs) if len(diffs) else 0
    return pd.Timedelta(int(max(med, 1)), unit="ns")

def _calc_ichimoku_bbands_dmas(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x['9DMA']  = x['Close'].rolling(9).mean()
    x['20DMA'] = x['Close'].rolling(20).mean()
    x['50DMA'] = x['Close'].rolling(50).mean()
    r20 = x['Close'].rolling(20); mid = r20.mean(); sd = r20.std(ddof=0)
    x['BB_Middle'] = mid; x['BB_Upper_1std'] = mid + sd; x['BB_Upper_2std'] = mid + 2*sd
    x['BB_Lower_1std'] = mid - sd; x['BB_Lower_2std'] = mid - 2*sd
    h9, l9 = x['High'].rolling(9).max(),  x['Low'].rolling(9).min()
    h26,l26= x['High'].rolling(26).max(), x['Low'].rolling(26).min()
    h52,l52= x['High'].rolling(52).max(), x['Low'].rolling(52).min()
    x['ICH_Tenkan']  = (h9 + l9)/2; x['ICH_Kijun'] = (h26 + l26)/2
    x['ICH_SenkouA'] = (x['ICH_Tenkan'] + x['ICH_Kijun'])/2; x['ICH_SenkouB'] = (h52 + l52)/2
    x['ICH_Chikou']  = x['Close']
    return x

def pl_ichi_bb_dma(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Ichimoku/BB/DMAs")
    x = _calc_ichimoku_bbands_dmas(df.copy())
    step = _median_bar_delta(x.index); lead = step*26; lag = step*26

    dup_open_idx   = x.index[x['Open'].duplicated(keep=False)]
    dup_close_idx  = x.index[x['Close'].duplicated(keep=False)]
    dup_prices = np.sort(pd.concat([x.loc[dup_open_idx,'Open'], x.loc[dup_close_idx,'Close']]).dropna().unique())

    traces = [go.Candlestick(x=x.index, open=x['Open'], high=x['High'], low=x['Low'], close=x['Close'], name='Candles')]
    for c in ['20DMA','50DMA','9DMA']:
        if c in x: traces.append(go.Scatter(x=x.index, y=x[c], name=c))
    # Bollinger
    traces += [
        go.Scatter(x=x.index, y=x['BB_Middle'], name='BB_Middle', line=dict(dash='dash')),
        go.Scatter(x=x.index, y=x['BB_Upper_1std'], name='BB_Upper_1σ', line=dict(dash='dot')),
        go.Scatter(x=x.index, y=x['BB_Upper_2std'], name='BB_Upper_2σ', line=dict(dash='dot')),
        go.Scatter(x=x.index, y=x['BB_Lower_1std'], name='BB_Lower_1σ', line=dict(dash='dot')),
        go.Scatter(x=x.index, y=x['BB_Lower_2std'], name='BB_Lower_2σ', line=dict(dash='dot')),
    ]
    # O/C/H/L/Midpoint dots
    traces += [
        go.Scatter(x=x.index, y=x['Open'],  mode='markers', name='Open',  marker=dict(size=2)),
        go.Scatter(x=x.index, y=x['Close'], mode='markers', name='Close', marker=dict(size=2)),
        go.Scatter(x=x.index, y=x['High'],  mode='markers', name='High',  marker=dict(size=2)),
        go.Scatter(x=x.index, y=x['Low'],   mode='markers', name='Low',   marker=dict(size=2)),
        go.Scatter(x=x.index, y=(x['High'] + x['Low'])/2, mode='markers', name='Midpoint', marker=dict(size=2)),
    ]
    # Duplicate price levels
    traces += [
        go.Scatter(x=dup_open_idx,  y=x.loc[dup_open_idx,'Open'],  mode='markers', name='Duplicate Values', marker=dict(size=10), legendgroup='dup'),
        go.Scatter(x=dup_close_idx, y=x.loc[dup_close_idx,'Close'], mode='markers', showlegend=False, marker=dict(size=10), legendgroup='dup'),
    ]
    first=True
    for price in dup_prices:
        traces.append(go.Scatter(x=[x.index[0], x.index[-1]], y=[price, price], mode='lines',
                                 line=dict(dash='dot', width=1), name='Duplicate Levels',
                                 legendgroup='dup', showlegend=first)); first=False
    # Extremes
    hi_ext = x['High'] > x['BB_Upper_2std']; lo_ext = x['Low'] < x['BB_Lower_2std']
    traces += [
        go.Scatter(x=x.index[hi_ext], y=x.loc[hi_ext,'High'], mode='markers', name='High > 2σ', marker=dict(size=8, line=dict(width=2))),
        go.Scatter(x=x.index[lo_ext], y=x.loc[lo_ext,'Low'],  mode='markers', name='Low < 2σ',  marker=dict(size=8, line=dict(width=2))),
    ]
    # Ichimoku
    xlead = x.index + lead
    traces += [go.Scatter(x=x.index, y=x['ICH_Tenkan'], name='Ich Tenkan (9)'),
               go.Scatter(x=x.index, y=x['ICH_Kijun'],  name='Ich Kijun (26)'),
               go.Scatter(x=xlead, y=x['ICH_SenkouA'],  name='Senkou A (+26)'),
               go.Scatter(x=xlead, y=x['ICH_SenkouB'],  name='Senkou B (+26)', fill='tonexty', fillcolor='rgba(100,100,255,0.2)'),
               go.Scatter(x=x.index - lag, y=x['ICH_Chikou'], name='Chikou (-26)')]

    # 20>50 green fill / 20<50 red fill
    if '20DMA' in x and '50DMA' in x:
        mask_up = (x['20DMA'] > x['50DMA'])
        y50_up = x['50DMA'].where(mask_up); y20_up = x['20DMA'].where(mask_up)
        traces += [go.Scatter(x=x.index, y=y50_up, name='DMA base (up)', line=dict(width=0), showlegend=False),
                   go.Scatter(x=x.index, y=y20_up, name='20>50 Fill', fill='tonexty', fillcolor='rgba(0,255,0,0.25)', line=dict(width=0), showlegend=False)]
        y50_dn = x['50DMA'].where(~mask_up); y20_dn = x['20DMA'].where(~mask_up)
        traces += [go.Scatter(x=x.index, y=y50_dn, name='DMA base (dn)', line=dict(width=0), showlegend=False),
                   go.Scatter(x=x.index, y=y20_dn, name='20<50 Fill', fill='tonexty', fillcolor='rgba(255,0,0,0.25)', line=dict(width=0), showlegend=False)]

    # 6M extremes
    try:
        six_months_ago = x.index[-1] - pd.DateOffset(months=6)
        last6 = x[x.index > six_months_ago]
        hi2 = last6['High'][last6['High'] > last6['BB_Upper_2std']].max()
        lo2 = last6['Low'][last6['Low']   < last6['BB_Lower_2std']].min()
        if pd.notna(hi2):
            traces.append(go.Scatter(x=[x.index[0], x.index[-1]], y=[hi2, hi2], mode='lines',
                                     name='Highest > 2σ (6M)', line=dict(color='lawngreen', dash='dash', width=1)))
        if pd.notna(lo2):
            traces.append(go.Scatter(x=[x.index[0], x.index[-1]], y=[lo2, lo2], mode='lines',
                                     name='Lowest < 2σ (6M)', line=dict(color='red', dash='dash', width=1)))
    except Exception:
        pass

    fig = go.Figure(data=traces)
    fig.update_layout(title=f'{ticker} — Ichimoku, Bollinger, DMAs', template='plotly_dark',
                      xaxis=dict(rangeslider=dict(visible=True)), height=900, width=1600)
    _write_plotly(fig, "pl_ichi_bb_dma", ticker, paths, res, L)

# ---------- Multi-term LRC with σ-bands, EMA overlay ----------
def pl_multi_term_lrc(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Multi-term LRC")
    d = df.copy()
    def _add_term(term: str, frac: float):
        if frac >= 1.0:
            sub = d
        else:
            span = max(int(len(d)*frac), 10)
            sub = d.iloc[-span:]
        x = np.arange(len(sub)); y = sub['Close'].values.astype(float)
        sl,itcpt = np.polyfit(x, y, 1)
        fit = sl * np.arange(len(sub)) + itcpt
        resid = y - (sl*x + itcpt); std = float(np.std(resid, ddof=1)) if len(resid)>1 else 0.0
        pref = term
        d.loc[sub.index, f'{pref}_LRC'] = fit
        for i, k in enumerate([0.5,1,1.25,1.5,1.75,2,2.25,3,4], 1):
            d.loc[sub.index, f'{pref}_Hi_{i}'] = fit + k*std
            d.loc[sub.index, f'{pref}_Lo_{i}'] = fit - k*std

    _add_term("Long", 1.0); _add_term("Mid", 0.5); _add_term("Short", 0.25)

    for p in [20,50,100,200]:
        d[f'EMA{p}'] = d['Close'].ewm(span=p, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d['Open'], high=d['High'], low=d['Low'], close=d['Close'], name='Price'))
    colors = {"Long":"#1f77b4","Mid":"#2ca02c","Short":"#d62728"}
    for term in ["Long","Mid","Short"]:
        base = f"{term}_LRC"
        if base in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[base], line=dict(color=colors[term], width=2), name=f"{term} LRC"))
            for i in range(1,10):
                hi, lo = f"{term}_Hi_{i}", f"{term}_Lo_{i}"
                if hi in d and lo in d:
                    fig.add_trace(go.Scatter(x=d.index, y=d[hi], line=dict(width=1, dash='dot', color=colors[term]), showlegend=False))
                    fig.add_trace(go.Scatter(x=d.index, y=d[lo], line=dict(width=1, dash='dot', color=colors[term]), fill='tonexty',
                                             fillcolor='rgba(31,119,180,0.08)' if term=="Long" else ('rgba(44,160,44,0.12)' if term=="Mid" else 'rgba(214,39,40,0.12)'),
                                             showlegend=False))
    for p in [20,50,100,200]:
        col = f'EMA{p}'
        if col in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], line=dict(width=1.6), name=col))
    fig.update_layout(template="plotly_dark", title=f"{ticker} — Multi-term LRC + EMAs", height=900, width=1600)
    _write_plotly(fig, "pl_multi_term_lrc", ticker, paths, res, L)

# ---------- Hyperbolic transforms ----------
def pl_hyperbolic(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Hyperbolic transforms")
    d = df.copy()
    d['Daily_Return'] = d['Close'].pct_change()*100; d.dropna(inplace=True)
    mu, sd = float(d['Daily_Return'].mean()), float(d['Daily_Return'].std(ddof=0) or 1.0)
    d['Tanh_Return'] = np.tanh((d['Daily_Return']-mu)/sd)
    d['Momentum_14'] = d['Close'].diff(14); d.dropna(inplace=True)
    alpha = 0.01
    d['Momentum_Tanh'] = np.tanh(alpha*d['Momentum_14'])
    d['Momentum_Sinh'] = np.sinh(alpha*d['Momentum_14'])
    d['Momentum_Cosh'] = np.cosh(alpha*d['Momentum_14'])
    sig = d[(d['Tanh_Return'] > 0.8) | (d['Tanh_Return'] < -0.8)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{ticker} – Daily Returns vs Tanh(Scaled Returns)",
                                        "Momentum(14) with Hyperbolic Transforms"])
    fig.add_trace(go.Scatter(x=d.index, y=d['Daily_Return'], mode='lines', name='Daily % Return'), row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['Tanh_Return'],   mode='lines', name='Tanh(Scaled Return)'), row=1,col=1)
    if not sig.empty:
        fig.add_trace(go.Scatter(x=sig.index, y=sig['Daily_Return'], mode='markers', name='Significant', marker=dict(size=8, symbol='circle-open')), row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['Momentum_14'],   mode='lines', name='Momentum(14)'), row=2,col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['Momentum_Tanh'], mode='lines', name='Momentum Tanh'), row=2,col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['Momentum_Sinh'], mode='lines', name='Momentum Sinh'), row=2,col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['Momentum_Cosh'], mode='lines', name='Momentum Cosh'), row=2,col=1)
    fig.update_layout(template='plotly_dark', title=f"{ticker} – Hyperbolic-Based Visualization", height=800)
    _write_plotly(fig, "pl_hyperbolic", ticker, paths, res, L)

# ---------- P/E regression bands ----------
def pl_pe_lrc(ticker: str, df: pd.DataFrame, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: P/E regression bands")
    try:
        import yfinance as yf
    except Exception:
        L.log("yfinance not available; skipping P/E.")
        return
    try:
        stock = yf.Ticker(ticker)
        income_stmt = getattr(stock, "quarterly_financials", None)
        if income_stmt is None or income_stmt.empty or "Net Income" not in income_stmt.index:
            L.log("[PE] No quarterly financials; skipping.")
            return
        net_income_ttm = income_stmt.loc["Net Income"].iloc[:4].sum()
        info = getattr(stock, "info", {}) or {}
        shares_outstanding = info.get("sharesOutstanding", 0)
        if not shares_outstanding:
            L.log("[PE] No sharesOutstanding; skipping.")
            return
        eps_ttm = float(net_income_ttm) / float(shares_outstanding)
        if eps_ttm == 0:
            L.log("[PE] EPS is zero; skipping.")
            return
    except Exception as e:
        L.log(f"[PE] fundamentals error: {e}")
        return

    d = df.copy()
    d["PE_Ratio"] = d["Close"] / eps_ttm
    d.dropna(subset=["PE_Ratio"], inplace=True)
    if d.empty:
        L.log("[PE] No P/E data.")
        return

    x = np.arange(len(d.index))
    sl,itcpt = np.polyfit(x, d['PE_Ratio'].values.astype(float), 1)
    d['PE_LRC'] = sl*x + itcpt
    resid_std = float((d['PE_Ratio'] - d['PE_LRC']).std())

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d['Open'], high=d['High'], low=d['Low'], close=d['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=d.index, y=d['PE_Ratio'], name='P/E Ratio'))
    fig.add_trace(go.Scatter(x=d.index, y=d['PE_LRC'], name='P/E LRC', line=dict(width=2)))
    for k in [0.5,1,1.25,1.5,1.75,2,2.25,3,4]:
        fig.add_trace(go.Scatter(x=d.index, y=d['PE_LRC'] + k*resid_std, line=dict(dash='dot', width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=d.index, y=d['PE_LRC'] - k*resid_std, line=dict(dash='dot', width=1), showlegend=False, fill='tonexty',
                                 fillcolor='rgba(200,200,200,0.06)'))
    fig.update_layout(template='plotly_dark', title=f"{ticker} – Candlesticks & P/E (Regression)")
    _write_plotly(fig, "pl_pe_lrc", ticker, paths, res, L)

# ---------- Support/Resistance + Patterns ----------
def _support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    from scipy.signal import argrelextrema
    out = df.copy()
    if len(out) < window: return out
    r_idx = argrelextrema(out['High'].values, np.greater_equal, order=window)[0]
    s_idx = argrelextrema(out['Low'].values,  np.less_equal,   order=window)[0]
    out['resistance'] = np.nan; out['support'] = np.nan
    if len(r_idx)>0: out.iloc[r_idx, out.columns.get_loc('resistance')] = out['High'].iloc[r_idx]
    if len(s_idx)>0: out.iloc[s_idx, out.columns.get_loc('support')]   = out['Low'].iloc[s_idx]
    return out

def _add_talib_patterns(df: pd.DataFrame, include_all: bool) -> pd.DataFrame:
    if talib is None:
        return df
    out = df.copy()
    if include_all:
        group = talib.get_function_groups().get('Pattern Recognition', [])
        for p in group:
            try: out[p] = getattr(talib, p)(out['Open'], out['High'], out['Low'], out['Close'])
            except Exception: pass
    else:
        pats = ['CDLHAMMER','CDLINVERTEDHAMMER','CDLENGULFING','CDLPIERCING','CDLDARKCLOUDCOVER','CDLDOJI','CDLMORNINGSTAR','CDLEVENINGSTAR','CDLSHOOTINGSTAR','CDLHARAMI']
        for p in pats:
            try: out[p] = getattr(talib, p)(out['Open'], out['High'], out['Low'], out['Close'])
            except Exception: pass
    return out

def pl_sr_patterns(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger, include_all: bool):
    L.log(f"Patterns + S/R (include_all={include_all})")
    x = _support_resistance(df.copy())
    x = _add_talib_patterns(x, include_all)
    fig = go.Figure(go.Candlestick(x=x.index, open=x['Open'], high=x['High'], low=x['Low'], close=x['Close'], name='Price'))
    if 'support' in x:    fig.add_trace(go.Scatter(x=x.index, y=x['support'].dropna(),    mode='markers', name='Support'))
    if 'resistance' in x: fig.add_trace(go.Scatter(x=x.index, y=x['resistance'].dropna(), mode='markers', name='Resistance'))
    # plot all non-zero pattern markers a little off price
    rng = (x['High'].max() - x['Low'].min()) * 0.02
    if talib is not None:
        pats = [c for c in x.columns if c.startswith("CDL")]
        for p in pats:
            d = x[x[p] != 0]
            if d.empty: continue
            y = np.where(d[p] > 0, d['High'] + rng, d['Low'] - rng)
            fig.add_trace(go.Scatter(x=d.index, y=y, mode='markers', marker=dict(size=6, opacity=0.7), name=p.replace('CDL','')))
    fig.update_layout(template='plotly_dark', title=f"{ticker} – S/R + {'ALL' if include_all else 'Selected'} Patterns")
    _write_plotly(fig, "pl_sr_patterns_all" if include_all else "pl_sr_patterns_selected", ticker, paths, res, L)

# ---------- EMA fanning + GIF ----------
def ema_fanning_and_gif(ticker: str, df: pd.DataFrame, paths: Paths, res: RunResult, L: _Logger):
    L.log("EMA fanning + GIF")
    d = df.copy()
    for s in (50,100,150,200,300):
        d[f'EMA{s}'] = d['Close'].ewm(span=s, adjust=False).mean()

    def _one(name: str, start_dt: pd.Timestamp):
        sub = d[d.index >= start_dt]
        if sub.empty or len(sub) < 2: return None, None
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=(f"{ticker} Price & EMAs", "EMA Spread Factor"))
        # top
        fig.add_trace(go.Scatter(x=sub.index, y=sub['Close'], mode='lines', name='Close'), row=1, col=1)
        for s in (50,100,150,200,300):
            col = f'EMA{s}'
            if col in sub:
                fig.add_trace(go.Scatter(x=sub.index, y=sub[col], mode='lines', name=col), row=1, col=1)
        # bottom spread factor
        cols = [f'EMA{a}-EMA{b}' for a,b in [(50,100),(100,150),(150,200),(200,300)]]
        tmp = sub.copy()
        tmp['EMA_Spread_Factor'] = 0.0
        for a,b in [(50,100),(100,150),(150,200),(200,300)]:
            ca, cb = f'EMA{a}', f'EMA{b}'
            if ca in tmp and cb in tmp:
                tmp['EMA_Spread_Factor'] += (tmp[ca]-tmp[cb])
        fig.add_trace(go.Scatter(x=tmp.index, y=tmp['EMA_Spread_Factor'], mode='lines', name='EMA Spread Factor'), row=2, col=1)

        html = os.path.join(paths.html, f"{ticker.lower()}_ema_fanning_{name}.html")
        png  = os.path.join(paths.png,  f"{ticker.lower()}_ema_fanning_{name}.png")
        pio.write_html(fig, file=html, auto_open=False, include_plotlyjs=True, full_html=True)
        res.html_files.append(html); L.log(f"HTML saved: {html}")
        try:
            fig.write_image(png, width=1600, height=900, scale=2.0); res.png_files.append(png)
            L.log(f"Plotly PNG saved (for PDF only): {png}")
        except Exception:
            pass
        return html, png

    now = d.index.max()
    date_ranges = [
        ("last_7d",   now - pd.Timedelta(days=7)),
        ("last_1m",   now - pd.Timedelta(days=30)),
        ("last_3m",   now - pd.Timedelta(days=90)),
        ("last_6m",   now - pd.Timedelta(days=180)),
        ("last_1y",   now - pd.Timedelta(days=365)),
        ("last_2y",   now - pd.Timedelta(days=730)),
        ("last_5y",   now - pd.Timedelta(days=1825)),
        ("max",       d.index.min())
    ]

    frames = []
    for name, start_dt in date_ranges:
        h, p = _one(name, start_dt)
        if p and os.path.exists(p):
            frames.append(Image.open(p) if Image is not None else p)
    # GIF
    gif_out = os.path.join(paths.gif, f"{ticker.lower()}_ema_fanning.gif")
    try:
        if imageio is not None and frames:
            imgs = [imageio.imread(fp) if isinstance(fp, str) else np.array(fp) for fp in frames]
            imageio.mimsave(gif_out, imgs, duration=1.0, loop=0)
            res.gif_files.append(gif_out)
            L.log(f"GIF written: {gif_out}")
    except Exception as e:
        L.log(f"[warn] GIF failed: {e}")


# ---------- Ratio suite ----------
def ratio_suite(ticker: str, ratio_with: Optional[str], period: str,
                start: Optional[str], end: Optional[str],
                paths: Paths, res: RunResult, L: _Logger):
    if not ratio_with:
        return
    L.log(f"Ratio suite: {ticker}/{ratio_with}")
    rdf = dr.get_ratio_dataframe(ticker, ratio_with, f"{start},{end}" if (start and end) else period)
    if rdf is None or rdf.empty:
        L.log("[ratio] No data.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rdf.index, y=rdf['Close'], mode="lines", name=f"{ticker}/{ratio_with}"))
    fig.update_layout(template="plotly_dark", title=f"{ticker}/{ratio_with} Close ratio")
    _write_plotly(fig, f"ratio_{ratio_with.lower()}", ticker, paths, res, L)


# ---------- Options OI (nearest target exp) ----------
def plot_options_oi(ticker: str, weeks_ahead: int, paths: Paths, res: RunResult, L: _Logger):
    L.log(f"Options OI (weeks ahead={weeks_ahead})")
    try:
        import yfinance as yf
        import plotly.express as px
    except Exception:
        L.log("yfinance/plotly.express not available; skipping options.")
        return
    try:
        t = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            L.log("[options] no expirations")
            return
        target = datetime.today() + timedelta(weeks=weeks_ahead)
        exp = min(dates, key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d") - target))
        opt = t.option_chain(exp); calls, puts = opt.calls.copy(), opt.puts.copy()
        calls['type']='call'; puts['type']='put'
        df = pd.concat([calls, puts], ignore_index=True).fillna(0.0)
        # current price line
        pxdf = dr.load_or_download_ticker(ticker, period="5d")
        cur = float(pxdf['Close'].dropna().iloc[-1]) if pxdf is not None and not pxdf.empty else None

        def scatter(data, typ):
            if data.empty: return
            fig = px.scatter(data, x="strike", y="openInterest", size="openInterest", color="openInterest",
                             hover_data=['lastPrice','volume','impliedVolatility'], opacity=0.85)
            if cur is not None: fig.add_vline(x=cur, line_dash='dash')
            fig.update_layout(template='plotly_dark', title=f"{ticker} {typ} Options OI ({exp}) — Current: {cur if cur is not None else 'N/A'}")
            base_filename = f"pl_options_oi_{typ.lower()}_{exp.replace('-','')}"
            _write_plotly(fig, base_filename, ticker, paths, res, L)

        scatter(df[df['type']=='call'], "Call")
        scatter(df[df['type']=='put'],  "Put")
    except Exception as e:
        L.log(f"[options] error: {e}")


# ---------- Percent Change LRC ----------
def pl_pct_change_lrc(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Percent Change LRC")
    d = df[['Close']].copy()
    # Use the pre-calculated 'ClosePC' if available, otherwise calculate it
    if 'ClosePC' in df.columns:
        d['ClosePC'] = df['ClosePC']
    else:
        d['ClosePC'] = d['Close'].pct_change()
    d.dropna(inplace=True)

    if len(d) < 2:
        L.log("[pct_change_lrc] Not enough data to plot.")
        return

    # Calculate LRC and residual std dev
    y = d['ClosePC'].values.astype(float)
    x = np.arange(len(y))
    sl, itcpt = np.polyfit(x, y, 1)
    yhat = sl * x + itcpt
    residuals = y - yhat
    resid_std = float(np.std(residuals))

    fig = go.Figure()

    # Plot the actual percentage change
    fig.add_trace(go.Scatter(
        x=d.index, y=d['ClosePC'], mode='lines',
        name='Daily % Change', line=dict(width=1.5)
    ))

    # Plot the LRC line
    fig.add_trace(go.Scatter(
        x=d.index, y=yhat, mode='lines',
        name='LRC of % Change', line=dict(width=2, dash='dash')
    ))

    # Plot the standard deviation bands
    for k in [1, 2, 3, 4, 5]:
        fig.add_trace(go.Scatter(
            x=d.index, y=yhat + k * resid_std, mode='lines',
            line=dict(width=0.8, color='rgba(220, 220, 220, 0.5)'), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=d.index, y=yhat - k * resid_std, mode='lines',
            line=dict(width=0.8, color='rgba(220, 220, 220, 0.5)'),
            fill='tonexty', fillcolor=f'rgba(200, 200, 200, {0.12 - k*0.02})',
            showlegend=False
        ))

    fig.update_layout(
        template='plotly_dark',
        title=f"{ticker} – Linear Regression Channel on Daily Percent Change",
        yaxis_title="Percent Change",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    _write_plotly(fig, "pl_pct_change_lrc", ticker, paths, res, L)


# ---------- SMA Percentage Change LRC ----------
def pl_sma_pct_change_lrc(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: SMA % Change LRC")
    base_pc = 'SMA30PC'
    if base_pc not in df.columns or df[base_pc].dropna().empty:
        L.log(f"[sma_pct_change_lrc] Base series '{base_pc}' not available. Skipping.")
        return

    d = df.copy()
    d.dropna(subset=[base_pc], inplace=True)
    if len(d) < 2:
        L.log("[sma_pct_change_lrc] Not enough data to plot.")
        return

    # Calculate LRC and residual std dev on the base series
    y = d[base_pc].values.astype(float)
    x = np.arange(len(y))
    sl, itcpt = np.polyfit(x, y, 1)
    yhat = sl * x + itcpt
    residuals = y - yhat
    resid_std = float(np.std(residuals))

    fig = go.Figure()

    # Plot the base percentage change
    fig.add_trace(go.Scatter(
        x=d.index, y=d[base_pc], mode='lines', name='SMA30 % Change',
        line=dict(width=1.5, color='rgba(144, 238, 144, 0.8)') # lightgreen
    ))

    # Plot the LRC line
    fig.add_trace(go.Scatter(
        x=d.index, y=yhat, mode='lines', name='LRC', line=dict(width=2, dash='dash')
    ))

    # Plot std dev bands
    for k in [1, 2, 3, 4]:
        fig.add_trace(go.Scatter(x=d.index, y=yhat + k * resid_std, mode='lines', line=dict(width=0.8, color='rgba(220, 220, 220, 0.5)'), showlegend=False))
        fig.add_trace(go.Scatter(x=d.index, y=yhat - k * resid_std, mode='lines', line=dict(width=0.8, color='rgba(220, 220, 220, 0.5)'), fill='tonexty', fillcolor=f'rgba(200, 200, 200, {0.12 - k*0.02})', showlegend=False))

    # Overlay other SMA PC lines
    for pc, color in [('SMA50PC', 'orange'), ('SMA100PC', 'blue'), ('SMA200PC', 'purple')]:
        if pc in d.columns and not d[pc].isna().all():
            fig.add_trace(go.Scatter(x=d.index, y=d[pc], mode='lines', name=pc.replace('PC', ' % Change'), line=dict(width=0.8, color=color)))

    fig.update_layout(template='plotly_dark', title=f"{ticker} – LRC on SMA % Change", yaxis_title="Percent Change")
    _write_plotly(fig, "pl_sma_pct_change_lrc", ticker, paths, res, L)


# ---------- Log Volume LRC ----------
def pl_log_volume_lrc(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Log Volume LRC")
    if 'VolLOG' not in df.columns or df['VolLOG'].dropna().empty:
        L.log("[log_volume_lrc] 'VolLOG' series not available or empty. Skipping.")
        return

    d = df[['VolLOG']].copy().dropna()
    if len(d) < 2:
        L.log("[log_volume_lrc] Not enough data to plot.")
        return

    y = d['VolLOG'].values.astype(float)
    x = np.arange(len(y))
    sl, itcpt = np.polyfit(x, y, 1)
    yhat = sl * x + itcpt
    residuals = y - yhat
    resid_std = float(np.std(residuals))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d['VolLOG'], mode='lines', name='Log2(Volume)', line=dict(color='gold', width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=yhat, mode='lines', name='LRC', line=dict(width=2, dash='dash')))

    for k in [1, 2, 3, 4, 5, 10]:
        # **FIXED**: Ensure the alpha value in fillcolor is never negative.
        alpha = max(0, 0.10 - k * 0.015)
        fig.add_trace(go.Scatter(x=d.index, y=yhat + k * resid_std, mode='lines', line=dict(width=0.8, color='rgba(220, 220, 220, 0.5)'), showlegend=False))
        fig.add_trace(go.Scatter(x=d.index, y=yhat - k * resid_std, mode='lines', line=dict(width=0.8, color='rgba(220, 220, 220, 0.5)'), showlegend=False, fill='tonexty', fillcolor=f'rgba(200, 200, 200, {alpha})'))

    fig.update_layout(template='plotly_dark', title=f"{ticker} – LRC on Log(Volume)", yaxis_title="Log2(Volume)")
    _write_plotly(fig, "pl_log_volume_lrc", ticker, paths, res, L)


# ---------- Velocity, Diffs, OHLC, Subplots ----------
def pl_ma_velocity(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger, spans=(50,100,200,300)):
    L.log("Plotly: MA Velocity")
    d = df.copy()
    fig_price = go.Figure()
    fig_vol = go.Figure()
    price_added, vol_added = False, False

    for s in spans:
        # Price velocity
        if 'Close' in d.columns:
            ma_col = f'SMA{s}'
            vel_col = f'{ma_col}_velocity'
            if ma_col in d:
                d[vel_col] = d[ma_col].diff(7) / 7.0
                fig_price.add_trace(go.Scatter(x=d.index, y=d[vel_col], name=f'{s}d Price MA Vel'))
                price_added = True
        # Volume velocity
        if 'Volume' in d.columns:
            ma_col = f'VolMA{s}'
            vel_col = f'{ma_col}_velocity'
            d[ma_col] = d['Volume'].rolling(s).mean()
            d[vel_col] = d[ma_col].diff(7) / 7.0
            fig_vol.add_trace(go.Scatter(x=d.index, y=d[vel_col], name=f'{s}d Vol MA Vel'))
            vol_added = True

    if price_added:
        fig_price.update_layout(title=f'{ticker} — Price MA Velocity (weekly avg Δ)', template='plotly_dark', width=1000, height=600, yaxis_title='Velocity')
        _write_plotly(fig_price, "pl_price_ma_velocity", ticker, paths, res, L)
    if vol_added:
        fig_vol.update_layout(title=f'{ticker} — Volume MA Velocity (weekly avg Δ)', template='plotly_dark', width=1000, height=600, yaxis_title='Velocity')
        _write_plotly(fig_vol, "pl_volume_ma_velocity", ticker, paths, res, L)

def pl_abs_ema_diffs(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: Absolute EMA Diffs")
    d = df.copy()
    d['ema50_close'] = d['Close'].ewm(span=50, adjust=False).mean()
    d['ema100_close'] = d['Close'].ewm(span=100, adjust=False).mean()
    d['ema50_volume'] = d['Volume'].ewm(span=50, adjust=False).mean()
    d['ema100_volume'] = d['Volume'].ewm(span=100, adjust=False).mean()
    d['abs_diff_price'] = (d['ema50_close'] - d['ema100_close']).abs()
    d['abs_diff_volume'] = (d['ema50_volume'] - d['ema100_volume']).abs()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('|EMA(50) - EMA(100)| — Price', '|EMA(50) - EMA(100)| — Volume'))
    fig.add_trace(go.Scatter(x=d.index, y=d['abs_diff_price'], name='|Δ EMA| Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['abs_diff_volume'], name='|Δ EMA| Volume'), row=2, col=1)
    fig.update_layout(title=f'{ticker} — Absolute EMA(50,100) Diffs', template='plotly_dark', height=700, width=1200)
    _write_plotly(fig, "pl_abs_ema_diffs", ticker, paths, res, L)

def pl_ohlc_ma_volume(df: pd.DataFrame, ticker: str, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: OHLC + MAs + Volume")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if 'SMA50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='MA50'), row=1, col=1)
    if 'SMA100' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA100'], name='MA100'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
    fig.update_layout(title=f'{ticker} — OHLC + MAs + Volume', template='plotly_dark', width=1200, height=900)
    _write_plotly(fig, "pl_ohlc_ma_volume", ticker, paths, res, L)

def pl_ema_diffs_subplots(df: pd.DataFrame, ticker: str, spans: list, paths: Paths, res: RunResult, L: _Logger):
    L.log("Plotly: EMA Diffs Subplots")
    d = df.copy()
    pairs = [(spans[i], spans[j]) for i in range(len(spans)) for j in range(i + 1, len(spans))]
    if not pairs: return
    
    # Pre-calculate diffs
    has_data = False
    for s1, s2 in pairs:
        col = f'Diff_EMA{s1}_EMA{s2}'
        if f'EMA{s1}' in d and f'EMA{s2}' in d:
            d[col] = (d[f'EMA{s2}'] - d[f'EMA{s1}']) / d[f'EMA{s1}'].replace(0, np.nan) * 100.0
            if d[col].notna().any():
                has_data = True

    if not has_data: return

    cols = 2; rows = (len(pairs) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'EMA{s1}-EMA{s2} % Diff' for s1, s2 in pairs])
    for idx, (s1, s2) in enumerate(pairs):
        col = f'Diff_EMA{s1}_EMA{s2}'
        r, c = (idx // cols) + 1, (idx % cols) + 1
        if col in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], mode='lines', name=col), row=r, col=c)

    fig.update_layout(title=f"{ticker} — EMA % Differences", template='plotly_dark', height=rows * 300, width=1200, showlegend=False)
    _write_plotly(fig, "pl_ema_diffs_subplots", ticker, paths, res, L)


# ============================== PDF assembly =============================

def _assemble_pdf_from_images(ticker: str, paths: Paths, res: RunResult, report_png: Optional[str], L: _Logger):
    L.log("[pdf] assembling …")
    if not (FPDF and Image):
        L.log("[pdf] FPDF or Pillow not available; skipping.")
        return

    imgs = []
    if report_png and os.path.exists(report_png):
        imgs.append(report_png)
    # De-dup and sort by name for stability
    seen = set()
    for p in sorted(res.png_files):
        if p in seen: continue
        seen.add(p); imgs.append(p)
    if not imgs:
        L.log("[pdf] no images to include.")
        return
    try:
        pdf = FPDF("L", unit="pt", format="letter")
        pdf.set_auto_page_break(auto=True, margin=15)
        for img_path in imgs:
            with Image.open(img_path) as im:
                w_px, h_px = im.size
            pdf.add_page()
            page_w = pdf.w - 2*pdf.l_margin
            page_h = pdf.h - 2*pdf.t_margin
            aspect = w_px / float(h_px)
            img_w = page_w; img_h = img_w / aspect
            if img_h > page_h:
                img_h = page_h
                img_w = img_h * aspect
            x = (pdf.w - img_w) / 2
            y = (pdf.h - img_h) / 2
            pdf.image(img_path, x=x, y=y, w=img_w, h=img_h)
        out = os.path.join(paths.pdf, f"{ticker.upper()}_ALL_CHARTS.pdf")
        pdf.set_title(_latin1_safe(f"{ticker} charts"))
        pdf.output(out, "F")
        res.pdf_files.append(out)
        L.log(f"[pdf] written: {out}")
    except Exception as e:
        L.log(f"[pdf] failed: {e}")


# ============================== Orchestrator =============================

def generate_all_for_ticker(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    period: str,
    out_root: Optional[str],
    ratio_with: Optional[str],
    include_all_patterns: bool,
    options_weeks_ahead: int = 4,
    also_build_pdf: bool = True
) -> RunResult:
    T = ticker.upper()
    L = _Logger()
    L.log(f"=== START run for {T} ===")
    root = out_root or dr.create_output_directory(T)
    paths = _mk_dirs(root)
    res = RunResult()

    # Load data
    df = _load_prices(T, start, end, period)
    nrows = len(df)
    L.log(f"Loaded {T} (period={period if (not start and not end) else (start+'..'+end)}); rows={nrows}")

    # MAs and other features used in plots
    ema_spans = [50, 100, 150, 200, 300]
    for w in (10,20,30,50,100,200,300):
        df[f"SMA{w}"] = df['Close'].rolling(w).mean()
    for s in ema_spans:
        df[f'EMA{s}'] = df['Close'].ewm(span=s, adjust=False).mean()
    df['ClosePC'] = df['Close'].pct_change()
    df['SMA30PC'] = df['SMA30'].pct_change()
    df['SMA50PC'] = df['SMA50'].pct_change()
    df['SMA100PC'] = df['SMA100'].pct_change()
    df['SMA200PC'] = df['SMA200'].pct_change()
    df['VolLOG'] = np.log2(df['Volume'].replace(0, np.nan))

    # ========== Matplotlib set ==========
    mpl_lrc_smas(df, T, paths, res, L)
    mpl_outliers_3sigma(df, T, paths, res, L)
    freqs = mpl_streaks(df, T, paths, res, L)
    mpl_derivatives(df, T, paths, res, L)
    mpl_dist_qq(df, T, paths, res, L)
    mpl_polyfit_fft(df['Close'], "Close", T, 3, paths, res, L)
    if 'ClosePC' in df.columns and not df['ClosePC'].isna().all():
        mpl_polyfit_fft(df['ClosePC'], "PercentChange", T, 3, paths, res, L)
    mplfinance_lrc(df, T, paths, res, L, period=144)

    # ========== Plotly set ==========
    pl_lrc_smas(df, T, paths, res, L)
    pl_outliers(df, T, paths, res, L)
    pl_streaks_line(df, T, paths, res, L)
    pl_streaks_bar_counts(freqs, T, paths, res, L)
    pl_dist_qq(df, T, paths, res, L)
    pl_first_derivative(df, T, paths, res, L)
    pl_pct_change_lrc(df, T, paths, res, L)
    pl_sma_pct_change_lrc(df, T, paths, res, L)
    pl_log_volume_lrc(df, T, paths, res, L)
    pl_ichi_bb_dma(df, T, paths, res, L)
    pl_multi_term_lrc(df, T, paths, res, L)
    pl_hyperbolic(df, T, paths, res, L)
    pl_pe_lrc(T, df, paths, res, L)
    pl_sr_patterns(df, T, paths, res, L, include_all=include_all_patterns)
    pl_ma_velocity(df, T, paths, res, L)
    pl_abs_ema_diffs(df, T, paths, res, L)
    pl_ohlc_ma_volume(df, T, paths, res, L)
    pl_ema_diffs_subplots(df, T, ema_spans, paths, res, L)

    # EMA fanning & GIF
    ema_fanning_and_gif(T, df, paths, res, L)

    # Options OI
    plot_options_oi(T, weeks_ahead=options_weeks_ahead, paths=paths, res=res, L=L)

    # Ratio suite
    ratio_suite(T, ratio_with, period, start, end, paths, res, L)

    # Run log files
    txt_log, png_log = L.write_files(paths)
    res.log_file = txt_log
    L.log("Run log written.")

    # PDF assembly
    if also_build_pdf:
        _assemble_pdf_from_images(T, paths, res, png_log, L)

    L.log("=== END run ===")
    return res
