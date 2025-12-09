#!/usr/bin/env python3
# SCRIPTNAME: revamp.heatmap.tickers.by.volume.downloads.first.v3.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   - Implements a two-phase process for generating a Market Heatmap.
#   1. Verification Phase: Checks for local data and downloads only missing tickers.
#   2. Processing Phase: Builds the heatmap using only local data.
#   - Uses the robust data_retrieval.py for all I/O.

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import json
import math
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ---------------- Config ----------------
# CONSTRAINT: Ensure cache/manifests are written to /dev/shm
CACHE_DIR = os.path.join(dr.BASE_CACHE_PATH(), "heatmap_metadata")
DEFAULT_CSV = os.path.join(CACHE_DIR, "combined_tickers.csv")
MANIFEST_DIR = os.path.join(CACHE_DIR, "manifests")

# Only create directories if we are actually running, but safe to do here since it's /dev/shm
os.makedirs(MANIFEST_DIR, exist_ok=True)

RETRIES_PER_TICKER = 3
SLEEP_BETWEEN_RETRIES_SEC = 1.0

# ---------------- Helpers ----------------
def fix_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

def read_universe(csv_path: str) -> list[str]:
    if not os.path.exists(csv_path):
        # Fallback/Hint if default file missing
        if csv_path == DEFAULT_CSV:
            print(f"Default ticker CSV not found at {csv_path}.")
            print("Please provide a path to a CSV file containing a 'Symbol' column.")
            sys.exit(1)
        raise SystemExit(f"Ticker CSV not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    if "Symbol" not in df.columns:
        for c in df.columns:
            if str(c).lower().startswith("symbol"):
                df = df.rename(columns={c: "Symbol"})
                break
    if "Symbol" not in df.columns:
        raise SystemExit("Ticker CSV must contain a 'Symbol' column.")
    seen, uniq = set(), []
    for s in df["Symbol"].dropna().tolist():
        s = fix_symbol(s)
        if s and s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

def pct_change_last_two(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Close" not in df.columns: return np.nan
    c = df["Close"].dropna()
    if len(c) < 2: return np.nan
    prev_c, last_c = float(c.iloc[-2]), float(c.iloc[-1])
    if prev_c == 0 or not math.isfinite(prev_c) or not math.isfinite(last_c): return np.nan
    return (last_c / prev_c - 1.0) * 100.0

def size_proxy_30d_dollar_vol(df: pd.DataFrame) -> float:
    if df is None or df.empty or not {"Close","Volume"}.issubset(df.columns): return np.nan
    dv = (df["Close"] * df["Volume"]).dropna()
    if dv.empty: return np.nan
    return float(dv.tail(30).mean())

def good_enough(df_short: pd.DataFrame, df_long: pd.DataFrame) -> bool:
    ok_short = (df_short is not None and not df_short.empty and "Close" in df_short.columns and df_short["Close"].dropna().shape[0] >= 2)
    ok_long  = (df_long  is not None and not df_long.empty and {"Close","Volume"}.issubset(df_long.columns) and df_long.dropna(subset=["Close","Volume"]).shape[0] >= 5)
    return ok_short and ok_long

# --- PHASE 1: Verify local data and download if missing ---
def ensure_data_is_present(tickers: list[str], delay_sec: float):
    print("\n--- Phase 1: Verifying Local Data / Downloading Missing Tickers ---")
    success, failed = [], []
    n = len(tickers)
    start_time = time.time()

    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{n}] Verifying data for {t}...")
        is_ok = False
        was_downloaded_in_session = False
        for attempt in range(1, RETRIES_PER_TICKER + 1):
            df_short, downloaded1 = dr.load_or_download_ticker(t, period="5d")
            df_long, downloaded2 = dr.load_or_download_ticker(t, period="6mo")
            was_downloaded_in_session = was_downloaded_in_session or downloaded1 or downloaded2
            
            if good_enough(df_short, df_long):
                is_ok = True
                break
            
            if attempt < RETRIES_PER_TICKER:
                time.sleep(SLEEP_BETWEEN_RETRIES_SEC)
        
        if is_ok:
            success.append(t)
        else:
            print(f" -> Failed to retrieve valid data for {t} after {RETRIES_PER_TICKER} attempts.", file=sys.stderr)
            failed.append(t)

        if was_downloaded_in_session and delay_sec > 0:
            print(f"  -> Download occurred, pausing for {delay_sec}s...")
            time.sleep(delay_sec)

    elapsed = time.time() - start_time
    print(f"\n[*] Data verification phase complete in {elapsed:.1f}s.")
    print(f"    Data present for: {len(success)}")
    print(f"    Failed to get data for: {len(failed)}")
    return success, failed

# --- PHASE 2: Process data from disk and build the heatmap ---
def build_heatmap_from_disk(tickers: list[str], out_dir: str | None):
    print("\n--- Phase 2: Processing Data From Disk ---")
    rows = []
    n = len(tickers)
    start_time = time.time()

    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{n}] Processing {t} from disk...")
        try:
            # Uses dr logic which checks cache first
            df_short, _ = dr.load_or_download_ticker(t, period="5d")
            df_long, _ = dr.load_or_download_ticker(t, period="6mo")
            
            pct = pct_change_last_two(df_short)
            size = size_proxy_30d_dollar_vol(df_long)
            
            if not (math.isfinite(pct) and math.isfinite(size)):
                raise RuntimeError("non-finite pct/size")
                
            rows.append({"ticker": t, "pct_change": pct, "size_value": size})
        except Exception as e:
            print(f" -> Error processing {t}: {e}", file=sys.stderr)
    
    elapsed = time.time() - start_time
    print(f"\n[*] Processing phase complete in {elapsed:.1f}s.")
    
    if not rows:
        raise SystemExit("No valid data could be processed to build a heatmap.")

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["pct_change","size_value"])
    min_val = df["size_value"].min()
    if min_val <= 0:
        df["size_value"] += abs(min_val) + 1.0
    df = df.sort_values("size_value", ascending=False).reset_index(drop=True)

    out_html = plot_heatmap(df, out_dir)
    print(f"\nSUCCESS: {len(rows)} tickers included in heatmap.")
    print(f"Heat-map: {out_html}")

def plot_heatmap(df: pd.DataFrame, out_dir: str | None) -> str:
    fig = px.treemap(
        df,
        path=[px.Constant("S&P 500"), "ticker"], values="size_value",
        color="pct_change", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0,
        hover_data={"ticker": True, "pct_change": ":.2f", "size_value": False},
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>% change: %{customdata[0]:.2f}%<extra></extra>",
        customdata=np.stack([df["pct_change"].values], axis=-1),
    )
    fig.update_layout(margin=dict(t=40,l=10,r=10,b=10),
                      title="S&P 500 Heat Map — % Change (area ≈ 30D Avg Dollar Vol)")

    if out_dir is None:
        out_base = dr.create_output_directory("SP500")
    else:
        out_base = out_dir
        os.makedirs(out_base, exist_ok=True)

    out_html = os.path.join(out_base, f"sp500_heatmap_{datetime.now().strftime('%Y%m%d')}.html")
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    return out_html

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Generate an S&P 500 heatmap.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV, help=f"Path to the ticker CSV file.\n(Default: {DEFAULT_CSV})")
    parser.add_argument("out_dir", nargs="?", default=None, help="Directory to save the output HTML.\n(Default: a dated folder)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds ONLY AFTER an actual download.\n(Default: 1.0)")
    args = parser.parse_args()

    tickers = read_universe(args.csv_path)
    print(f"[*] Universe size: {len(tickers)} tickers from {os.path.basename(args.csv_path)}")
    print(f"[*] Conditional delay between tickers: {args.delay}s")

    successful_tickers, _ = ensure_data_is_present(tickers, args.delay)
    if not successful_tickers:
        print("\nNo ticker data could be retrieved. Cannot build heatmap.", file=sys.stderr)
        sys.exit(1)

    build_heatmap_from_disk(successful_tickers, args.out_dir)

if __name__ == "__main__":
    main()
