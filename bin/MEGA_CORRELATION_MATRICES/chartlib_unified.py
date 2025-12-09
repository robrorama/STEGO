#!/usr/bin/env python3
"""
chartlib_unified.py

All processing, analytics, clustering, and visualization helpers.
Only external project dependency: data_retrieval.py (for data access).

- Restored: PCA, Clustering, and Grouping helpers.
- Updated: Plotly-only visualizations (Static replaced with Plotly Image Export).
"""

from __future__ import annotations
import sys
import os

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

from typing import List, Optional
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

# Data access
# CONSTRAINT: Ensure data_retrieval is available
try:
    from data_retrieval import load_or_download_ticker
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# Plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Clustering
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


# =========================
# 1) DATA PREPARATION
# =========================
def build_price_table(
    tickers: List[str],
    labels: List[str],
    price_col: str = "Close",
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "5y",
    join_policy: str = "inner",
    resample_rule: Optional[str] = "D",
    stagger_sec: float = 0.0,
    verbose: bool = True
) -> pd.DataFrame:
    if len(tickers) != len(labels):
        raise ValueError("tickers and labels must have the same length")

    ser = []
    total = len(tickers)
    
    for i, (lab, tkr) in enumerate(zip(labels, tickers), 1):
        if verbose:
            print(f"[{i}/{total}] Fetching {tkr} ({lab})...")

        if start and end:
            df = load_or_download_ticker(tkr, start=start, end=end)
        else:
            df = load_or_download_ticker(tkr, period=period or "5y")

        if df is None or df.empty or price_col not in df.columns:
            if verbose:
                print(f"   -> Warning: No data for {tkr}")
            continue

        s = pd.to_numeric(df[price_col], errors="coerce").dropna()
        if resample_rule and resample_rule.upper() != "D":
            s = s.resample(resample_rule.upper()).last().dropna()
        s.name = lab
        ser.append(s)
        
        if stagger_sec > 0 and i < total:
            time.sleep(stagger_sec)

    if not ser:
        return pd.DataFrame()

    out = pd.concat(ser, axis=1, join=join_policy).sort_index()
    out = out.dropna(axis=1, how="all")
    return out


def transform_prices(prices_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "prices":
        return prices_df.copy()
    if mode == "returns":
        return prices_df.pct_change()
    if mode == "logreturns":
        return np.log(prices_df).diff()
    raise ValueError("mode must be one of: 'returns', 'logreturns', 'prices'")


# ======================================
# 2) CORRELATION + ORDERING / CLUSTERING
# ======================================
def compute_correlation(
    data_df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1,
) -> pd.DataFrame:
    if data_df.empty:
        return pd.DataFrame()
    return data_df.corr(method=method, min_periods=min_periods)


def order_by_abs_mean(corr: pd.DataFrame) -> pd.DataFrame:
    if corr.shape[0] <= 2:
        return corr
    a = corr.copy()
    np.fill_diagonal(a.values, np.nan)
    order = a.abs().mean(axis=1).sort_values(ascending=False).index
    return corr.loc[order, order]


def order_by_pca(corr: pd.DataFrame) -> pd.DataFrame:
    """Reorder correlation matrix by the first principal component."""
    if corr.shape[0] <= 2:
        return corr
    c = corr.copy()
    np.fill_diagonal(c.values, 1.0)
    # Handle NaNs if any remain (shouldn't for PCA but safety first)
    c = c.fillna(0)
    
    w, v = np.linalg.eigh(c.values)
    idx = np.argsort(w)[::-1]
    vec1 = v[:, idx[0]]
    order = np.argsort(vec1)
    return corr.iloc[order, order]


def cluster_and_order(
    returns_df: pd.DataFrame,
    *,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Hierarchical clustering order on pairwise correlations."""
    if returns_df.empty or returns_df.shape[1] < 2:
        return pd.DataFrame()

    corr = returns_df.corr(min_periods=min_periods)
    keep_cols = list(corr.columns)
    C = corr.loc[keep_cols, keep_cols]
    
    # Greedy prune to largest complete submatrix
    while C.isna().any().any() and len(keep_cols) > 1:
        drop_col = C.isna().sum().idxmax()
        keep_cols.remove(drop_col)
        C = corr.loc[keep_cols, keep_cols]

    if len(keep_cols) < 2:
        return pd.DataFrame()

    corr_complete = C
    dist = 1.0 - corr_complete
    np.fill_diagonal(dist.values, 0.0)

    condensed = squareform(dist.values, checks=False)
    Z = hierarchy.linkage(condensed, method="average", optimal_ordering=True)
    order = hierarchy.leaves_list(Z)
    return corr_complete.iloc[order, order]


# =========================
# 3) VISUALIZATIONS (Plotly)
# =========================
def plot_heatmap_plotly(corr: pd.DataFrame, title: str, zmin=-1, zmax=1) -> go.Figure:
    labels = corr.index.tolist()
    z = corr.values
    hover = [[f"{ri} × {cj}<br>ρ = {z[i, j]:+.3f}" for j, cj in enumerate(labels)]
             for i, ri in enumerate(labels)]
    
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=labels, y=labels,
            zmin=zmin, zmax=zmax, colorscale="RdBu",
            hoverinfo="text", text=hover, colorbar=dict(title="ρ")
        )
    )
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=40, t=80, b=80),
        width=1000, height=1000
    )
    return fig


def plot_heatmap_lower_triangle_plotly(corr: pd.DataFrame, title: str) -> go.Figure:
    """Lower-triangle correlation heatmap (Plotly)."""
    mask = np.triu(np.ones_like(corr, dtype=bool))
    m = corr.where(~mask)
    
    fig = px.imshow(
        m, text_auto=".2f", color_continuous_scale="RdBu",
        x=corr.columns, y=corr.index, labels=dict(color="Correlation"),
        zmin=-1, zmax=1, aspect="auto"
    )
    fig.update_layout(
        title=title, width=1200, height=1200, template="plotly_white",
        xaxis=dict(side="bottom")
    )
    return fig


# =========================
# 4) DASHBOARDS
# =========================
_COLOR_PALETTE = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF', '#FF304F'
] * 5

def _percent_from_first(prices_outer: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prices_outer.index)
    for col in prices_outer.columns:
        s = prices_outer[col].ffill().dropna()
        if s.empty: continue
        out[col] = (s / s.iloc[0] - 1.0) * 100.0
    return out

def create_timeseries_dashboard(prices_df: pd.DataFrame, *, price_mode: bool) -> go.Figure:
    if price_mode:
        plot_df = prices_df.copy()
        metric = plot_df.apply(lambda s: s.max() - s.min()) 
    else:
        plot_df = _percent_from_first(prices_df)
        vol = prices_df.pct_change(fill_method=None)
        metric = vol.std(skipna=True)

    assets_sorted = metric.sort_values(ascending=False).index.tolist()
    if not assets_sorted: return go.Figure()

    n = len(assets_sorted)
    gsize = max(1, n // 3)
    groups = {
        ("High Range" if price_mode else "High Volatility"): assets_sorted[:gsize],
        ("Medium Range" if price_mode else "Medium Volatility"): assets_sorted[gsize:2 * gsize],
        ("Low Range" if price_mode else "Low Volatility"): assets_sorted[2 * gsize:],
    }

    fig = make_subplots(
        rows=3, cols=1, vertical_spacing=0.05, shared_xaxes=True,
        subplot_titles=list(groups.keys()), specs=[[{"secondary_y": True}] for _ in range(3)]
    )

    cmap = {name: _COLOR_PALETTE[i] for i, name in enumerate(plot_df.columns)}
    
    for r, (gname, aset) in enumerate(groups.items(), start=1):
        for a in aset:
            if a not in plot_df.columns: continue
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df[a], name=a,
                    mode="lines", line=dict(width=1.5, color=cmap.get(a)),
                    hovertemplate=f"<b>{a}</b><br>%{{y:.2f}}<extra></extra>"
                ), row=r, col=1
            )
        if not price_mode:
            fig.add_hline(y=0, line=dict(color="black", width=0.5, dash="dot"), row=r, col=1)

    fig.update_layout(
        title=f"Asset {'Prices' if price_mode else 'Performance'}",
        height=1000, template="plotly_white", hovermode="x unified"
    )
    return fig


# =========================
# 5) OUTPUT / UTILS
# =========================
def save_plotly_fig(fig: go.Figure, html_path: Path, png_path: Optional[Path] = None) -> None:
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    if png_path:
        try:
            png_path = Path(png_path)
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(png_path), scale=2, width=1600, height=1200)
        except Exception as e:
            print(f"[info] PNG export skipped (kaleido missing?): {e}")

def open_in_browser(local_path: Path) -> None:
    try:
        webbrowser.open(f"file://{Path(local_path).resolve()}", new=2)
    except Exception:
        pass
