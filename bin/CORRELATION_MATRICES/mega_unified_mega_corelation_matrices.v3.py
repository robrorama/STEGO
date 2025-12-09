#!/usr/bin/env python3
# SCRIPTNAME: mega_unified_mega_corelation_matrices.v3.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
"""
mega_unified_run.py

Single CLI entrypoint that imports:
  - chartlib_unified.py  (all logic)
  - data_retrieval.py    (data access only)

Corrections:
  - Opens ALL major charts in browser (not just the first one).
  - Fixed 'all' universe to actually combine Core + Categories (resulting in >28 tickers).
  - Default universe is now 'all' (was 'categories').
"""

from __future__ import annotations

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Library
import chartlib_unified as CL

# Data access
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ---------- Universes ----------
CORE_TICKERS = [
    'SPY', 'TLT', 'GLD', 'TIP', 'XLY', 'XLI', 'ITB',
    'UUP', 'DBC', 'VIXY', 'IBIT', 'BITB', 'BTC-USD',
    'XLK', 'XLE', 'XLF', 'EEM', 'HYG', 'VNQ'
]

ASSETS: Dict[str, Dict[str, str]] = {
    "Equities": {
        "S&P 500": "^GSPC", "Nasdaq": "^NDX", "Russell 2000": "^RUT",
        "Nikkei 225": "^N225", "DAX": "^GDAXI", "FTSE 100": "^FTSE"
    },
    "Commodities": {
        "Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", 
        "Nat Gas": "NG=F", "Copper": "HG=F", "Corn": "ZC=F"
    },
    "Crypto": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"
    },
    "Fixed Income": {
        "10Y Treasury": "IEF", "10Y Yield": "^TNX", "HY Bonds": "HYG", "LQD": "LQD"
    },
    "Sectors": {
        "Tech": "XLK", "Energy": "XLE", "Financials": "XLF", "Utilities": "XLU", "Healthcare": "XLV"
    },
    "Forex": {
        "USD Index": "DX-Y.NYB", "Euro": "EURUSD=X", "Yen": "JPY=X", "Pound": "GBPUSD=X"
    }
}

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified CLI for correlations & dashboards.")

    # Data Window
    p.add_argument("--period", default="5y", help="e.g. 1y, 2y, 5y, max.")
    p.add_argument("--dates", default=None, help="YYYY-MM-DD,YYYY-MM-DD.")

    # Data Processing
    p.add_argument("--freq", default="D", help="Resample rule (D/W/M).")
    p.add_argument("--price-col", default="Close", help="Column for price.")
    p.add_argument("--stagger", type=float, default=0.5, help="Seconds to wait between downloads.")
    p.add_argument("--min-obs", type=int, default=60, help="Min pairwise observations.")

    # Alignment
    p.add_argument("--align", choices=["inner", "outer"], default="inner")

    # Universe
    # CHANGED DEFAULT TO 'all' so you get more than 28 tickers by default
    p.add_argument("--universe", choices=["core", "categories", "all"], default="all")
    p.add_argument("--assets", default=None, help="Comma-separated tickers/labels to restrict to.")
    p.add_argument("--groups", default="all", help="Comma-separated groups to run (or 'all').")

    # Analysis Modes
    p.add_argument("--split-correlations", action="store_true", help="Generate separate Pos/Neg/Neutral heatmaps.")
    p.add_argument("--time-slices", action="store_true", help="Generate correlations per year.")

    # Output Control
    p.add_argument("--out-root", default=None)
    p.add_argument("--no-open", action="store_true", help="Prevent browser tabs from opening.")
    p.add_argument("--generate-pdf", action="store_true", help="Compile PNGs into a PDF report.")
    
    # Feature Skips
    p.add_argument("--skip-combined", action="store_true")
    p.add_argument("--skip-per-group", action="store_true")
    p.add_argument("--skip-pca", action="store_true")
    p.add_argument("--skip-clustered", action="store_true")
    p.add_argument("--skip-ts", action="store_true")

    return p.parse_args()


def _effective_window(dates_arg: Optional[str], period_arg: str):
    if dates_arg:
        st, en = [d.strip() for d in dates_arg.split(",", 1)]
        return st, en, None
    return None, None, period_arg


def _get_universe(universe_mode: str) -> Tuple[List[str], List[str]]:
    # 1. Just Core
    if universe_mode == "core":
        return CORE_TICKERS[:], CORE_TICKERS[:]
    
    # 2. Categories Only
    if universe_mode == "categories":
        labels, tickers = [], []
        for cat in ASSETS.values():
            labels.extend(cat.keys())
            tickers.extend(cat.values())
        return labels, tickers
    
    # 3. All (Core + Categories, deduplicated)
    if universe_mode == "all":
        seen = set()
        labels, tickers = [], []
        
        # Add Core first
        for t in CORE_TICKERS:
            labels.append(t) # Label is ticker for core
            tickers.append(t)
            seen.add(t)
            
        # Add Categories
        for cat in ASSETS.values():
            for l, t in cat.items():
                if t not in seen:
                    labels.append(l)
                    tickers.append(t)
                    seen.add(t)
        return labels, tickers

    return [], []


def _restrict_universe(labels, tickers, assets_arg):
    if not assets_arg: return labels, tickers
    want = set(x.strip().upper() for x in assets_arg.split(","))
    L, T = [], []
    for lab, tkr in zip(labels, tickers):
        if lab.upper() in want or tkr.upper() in want:
            L.append(lab)
            T.append(tkr)
    return L, T


def _compile_pdf_report(pages: List[Tuple[str, Path]], out_pdf: Path) -> None:
    try:
        from fpdf import FPDF
    except ImportError:
        print("[info] PDF skipped (fpdf not installed). pip install fpdf2")
        return

    pdf = FPDF(unit="pt", format="Letter")
    for title, png in pages:
        if not png.exists(): continue
        pdf.add_page()
        pdf.set_font("Helvetica", size=14)
        pdf.multi_cell(0, 20, title)
        try:
            pdf.image(str(png), x=36, y=60, w=540, h=0)
        except Exception as e:
            print(f"Error adding image {png} to PDF: {e}")
    
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_pdf))
    print(f"[OK] PDF saved: {out_pdf}")


def main():
    args = parse_args()
    date_tag = datetime.now().strftime("%Y-%m-%d")
    
    # Output Setup
    # CONSTRAINT: Default to /dev/shm location
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = Path(dr.BASE_DATA_PATH()) / "MEGA_CORR" / date_tag

    subdirs = ["COMBINED", "GROUPS", "PCA", "CLUSTERED", "SPLIT", "YEARLY", "TS"]
    for sub in subdirs:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # Universe
    labels, tickers = _get_universe(args.universe)
    labels, tickers = _restrict_universe(labels, tickers, args.assets)
    
    print(f"--- Universe Selected: {args.universe.upper()} ({len(tickers)} assets) ---")

    # Data Loading (Staggered)
    st, en, per = _effective_window(args.dates, args.period)
    print(f"--- Starting Data Download (Stagger: {args.stagger}s) ---")
    
    # 1. Correlation Data (Inner Join)
    df_corr = CL.build_price_table(
        tickers, labels, args.price_col,
        start=st, end=en, period=per,
        join_policy=args.align, resample_rule=args.freq,
        stagger_sec=args.stagger, verbose=True
    )

    if df_corr.empty:
        sys.exit("No data retrieved.")

    pdf_pages = []
    returns = CL.transform_prices(df_corr, "returns")
    # Drop columns with no variance
    returns = returns.loc[:, returns.std() > 0]

    # --- 1. Combined Matrix ---
    if not args.skip_combined:
        print("--- Generating Combined Correlation ---")
        corr = CL.compute_correlation(returns, min_periods=args.min_obs)
        corr_ord = CL.order_by_abs_mean(corr)
        
        t = "Correlation Matrix (Combined)"
        fig = CL.plot_heatmap_plotly(corr_ord, t)
        p_html = out_root / "COMBINED" / "combined.html"
        p_png = out_root / "COMBINED" / "combined.png"
        CL.save_plotly_fig(fig, p_html, p_png)
        pdf_pages.append((t, p_png))
        
        # OPEN IN BROWSER
        if not args.no_open: CL.open_in_browser(p_html)

        # Split Correlations (Pos/Neg/Neutral)
        if args.split_correlations:
            print("    > Generating Split Correlations...")
            out_split = out_root / "SPLIT"
            
            # Positive
            mask_pos = corr_ord.where(corr_ord > 0.5)
            if not mask_pos.dropna(how='all').empty:
                fig_p = CL.plot_heatmap_plotly(mask_pos, "High Positive (>0.5)", zmin=0.5, zmax=1)
                path_p = out_split / "pos.html"
                CL.save_plotly_fig(fig_p, path_p, out_split / "pos.png")
                pdf_pages.append(("High Positive (>0.5)", out_split / "pos.png"))
                # OPEN IN BROWSER
                if not args.no_open: CL.open_in_browser(path_p)
            
            # Negative
            mask_neg = corr_ord.where(corr_ord < -0.5)
            if not mask_neg.dropna(how='all').empty:
                fig_n = CL.plot_heatmap_plotly(mask_neg, "High Negative (<-0.5)", zmin=-1, zmax=-0.5)
                path_n = out_split / "neg.html"
                CL.save_plotly_fig(fig_n, path_n, out_split / "neg.png")
                pdf_pages.append(("High Negative (<-0.5)", out_split / "neg.png"))
                # OPEN IN BROWSER
                if not args.no_open: CL.open_in_browser(path_n)

    # --- 2. Per-Group Matrices ---
    if not args.skip_per_group:
        print("--- Generating Group Correlations ---")
        groups_req = set(args.groups.split(",")) if args.groups != "all" else set(ASSETS.keys())
        
        for gname, mapping in ASSETS.items():
            if args.groups != "all" and gname not in groups_req: continue
            
            # Filter universe to this group
            g_labs, g_tkrs = [], []
            for l, t in mapping.items():
                if l in labels: # Only if present in current data
                    g_labs.append(l)
                    g_tkrs.append(t)
            
            if len(g_labs) < 2: continue
            
            # Subset data
            g_df = df_corr[g_labs]
            g_ret = CL.transform_prices(g_df, "returns")
            g_corr = CL.compute_correlation(g_ret, min_periods=args.min_obs)
            
            t = f"Correlation - {gname}"
            fig = CL.plot_heatmap_plotly(g_corr, t)
            base = gname.replace(" ", "_")
            p_html = out_root / "GROUPS" / f"{base}.html"
            p_png = out_root / "GROUPS" / f"{base}.png"
            CL.save_plotly_fig(fig, p_html, p_png)
            pdf_pages.append((t, p_png))
            # Group charts NOT auto-opened to avoid spamming 10+ tabs

    # --- 3. PCA Ordered ---
    if not args.skip_pca:
        print("--- Generating PCA Ordered Matrix ---")
        corr_pca = CL.order_by_pca(CL.compute_correlation(returns))
        t = "Correlation (PCA Ordered)"
        fig = CL.plot_heatmap_plotly(corr_pca, t)
        p_html = out_root / "PCA" / "pca.html"
        p_png = out_root / "PCA" / "pca.png"
        CL.save_plotly_fig(fig, p_html, p_png)
        pdf_pages.append((t, p_png))
        
        # OPEN IN BROWSER
        if not args.no_open: CL.open_in_browser(p_html)

    # --- 4. Clustered (Hierarchical) ---
    if not args.skip_clustered:
        print("--- Generating Clustered Matrix ---")
        corr_clust = CL.cluster_and_order(returns, min_periods=args.min_obs)
        t = "Correlation (Clustered)"
        fig = CL.plot_heatmap_lower_triangle_plotly(corr_clust, t)
        p_html = out_root / "CLUSTERED" / "clustered.html"
        p_png = out_root / "CLUSTERED" / "clustered.png"
        CL.save_plotly_fig(fig, p_html, p_png)
        pdf_pages.append((t, p_png))
        
        # OPEN IN BROWSER
        if not args.no_open: CL.open_in_browser(p_html)

    # --- 5. Yearly Slices ---
    if args.time_slices:
        print("--- Generating Yearly Slices ---")
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
            
        years = sorted(returns.index.year.unique())
        for y in years:
            slice_ret = returns[returns.index.year == y]
            if slice_ret.shape[0] < 30: continue
                
            corr_slice = CL.compute_correlation(slice_ret)
            corr_slice = CL.order_by_abs_mean(corr_slice)
            
            t = f"Correlation {y}"
            fig = CL.plot_heatmap_plotly(corr_slice, t)
            path_html = out_root / "YEARLY" / f"{y}.html"
            path_png = out_root / "YEARLY" / f"{y}.png"
            CL.save_plotly_fig(fig, path_html, path_png)
            pdf_pages.append((t, path_png))
            # NOT auto-opened

    # --- 6. Time Series Dashboards ---
    if not args.skip_ts:
        print("--- Generating Dashboards ---")
        # Fetch Outer Join data for dashboards
        df_outer = CL.build_price_table(
            tickers, labels, args.price_col,
            start=st, end=en, period=per,
            join_policy="outer", resample_rule=None,
            stagger_sec=0.0, verbose=False 
        )
        
        # Percent Dashboard
        fig_pct = CL.create_timeseries_dashboard(df_outer, price_mode=False)
        p_pct_html = out_root / "TS" / "perf_pct.html"
        CL.save_plotly_fig(fig_pct, p_pct_html, out_root / "TS" / "perf_pct.png")
        pdf_pages.append(("Performance (%)", out_root / "TS" / "perf_pct.png"))
        # OPEN IN BROWSER
        if not args.no_open: CL.open_in_browser(p_pct_html)

        # Price Dashboard
        fig_prc = CL.create_timeseries_dashboard(df_outer, price_mode=True)
        p_prc_html = out_root / "TS" / "perf_price.html"
        CL.save_plotly_fig(fig_prc, p_prc_html, out_root / "TS" / "perf_price.png")
        pdf_pages.append(("Performance (Price)", out_root / "TS" / "perf_price.png"))
        # OPEN IN BROWSER
        if not args.no_open: CL.open_in_browser(p_prc_html)

    # --- 7. PDF Report ---
    if args.generate_pdf:
        print("--- Compiling PDF ---")
        _compile_pdf_report(pdf_pages, out_root / "mega_report.pdf")
    else:
        print("[info] PDF skipped (enable with --generate-pdf)")

    print(f"\n[DONE] Output saved to: {out_root}")

if __name__ == "__main__":
    main()
