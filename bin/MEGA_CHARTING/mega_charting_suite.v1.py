#!/usr/bin/env python3
# SCRIPTNAME: mega_unified_run.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Single CLI entrypoint that calls chartlib_unified to:
#   - generate every plot from the unified feature set
#   - open a browser tab for each plot (HTML for Plotly, viewer HTML for PNGs)
#   - write one mega PDF that contains images of every plot (images only)
#
# Dependencies:
#   - data_retrieval.py (Canonical data source)
#   - chartlib_unified.py (Plotting logic - DO NOT MODIFY)

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import webbrowser
import time

# Adherence to Stego Financial Framework:
# 1. Strict use of data_retrieval for paths/config
try:
    import data_retrieval as dr
    import chartlib_unified as clu
except ImportError:
    print("Error: Required modules not found.")
    sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(description="Unified charting runner (all features, one CLI).")
    ap.add_argument("--ticker", required=True, help="Primary ticker symbol (e.g., SPY)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--period", default="6mo", help="If no start/end provided, use yfinance period (e.g., 6mo, 1y, max)")

    ap.add_argument("--ratio-with", default=None, help="Optional second ticker to run ratio A/B suite")

    ap.add_argument("--outdir", default=None, help="Override output root (must be in /dev/shm for compliance)")
    ap.add_argument("--include-all-patterns", action="store_true", help="Also compute ALL TA-Lib patterns (heavy) for S/R plot")

    ap.add_argument("--max-tabs", type=int, default=128, help="Max tabs to open")
    ap.add_argument("--tab-delay-ms", type=int, default=60, help="Delay between tab opens (ms)")

    ap.add_argument("--no-open", action="store_true", help="Do NOT auto-open tabs")
    ap.add_argument("--no-pdf",  action="store_true", help="Skip mega PDF assembly")

    return ap.parse_args()


def main():
    a = parse_args()
    ticker = a.ticker.upper()
    
    # Use canonical directory structure from data_retrieval unless overridden
    # CONSTRAINT: If overridden, we ensure it's still pointing to volatile memory if possible,
    # but strict compliance just uses the module's path.
    if a.outdir:
        # Check if user is trying to write outside /dev/shm
        if not a.outdir.startswith("/dev/shm"):
            print("Warning: --outdir should point to /dev/shm to comply with framework rules.")
        out_root = a.outdir
    else:
        out_root = dr.create_output_directory(ticker)

    # Delegate heavy lifting to the immutable chartlib_unified
    res = clu.generate_all_for_ticker(
        ticker=ticker,
        start=a.start,
        end=a.end,
        period=a.period,
        out_root=out_root,
        ratio_with=a.ratio_with,
        include_all_patterns=a.include_all_patterns,
        options_weeks_ahead=4,
        also_build_pdf=(not a.no_pdf)
    )

    if not a.no_open:
        # ONLY open HTMLs: Plotly charts (HTML) and Matplotlib viewer pages (HTML) are already in res.html_files.
        # We do NOT open PNGs that exist purely for PDF (prevents duplicate tabs).
        open_list = list(dict.fromkeys(res.html_files))  # dedup while preserving order
        clu.open_tabs(open_list, max_tabs=a.max_tabs, tab_delay_ms=a.tab_delay_ms)

    print(f"\nHTML: {len(res.html_files)}")
    print(f"PNGs: {len(res.png_files)}")
    print(f"GIFs: {len(res.gif_files)}")
    print(f"PDFs: {len(res.pdf_files)}")
    if res.log_file:
        print(f"Run log: {res.log_file}")


if __name__ == "__main__":
    main()
