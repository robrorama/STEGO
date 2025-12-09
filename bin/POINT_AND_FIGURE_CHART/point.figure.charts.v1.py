#!/usr/bin/env python3
# SCRIPTNAME: point.figure.charts.v1.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Point & Figure chart generator using data_retrieval.py for IO.
#   - Downloads via data_retrieval, then reloads from the on-disk cache before use.
#   - Supports close-only or hi-lo construction.
#   - Box size: fixed value (e.g., 1.0) or percent of last close (e.g., 1.0%).
#   - Reversal in boxes (default 3).
#   - Outputs: PNG (chart) and optional CSV (box grid).

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval
except ImportError:
    print("Error: 'data_retrieval.py' not found in the current directory.")
    sys.exit(1)

# ---------- Helpers for cache-first-then-load ----------
def _cache_key_for_range(start: str, end: str) -> str:
    def norm(d):
        return d.replace('-', '')
    return f"{norm(start)}_{norm(end)}"

def load_df_from_disk_cache(ticker: str, period: str = None, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Ensures data is cached using data_retrieval, then reloads from CSV on disk.
    Satisfies: 'download any data first and then load it off disk before using it'.
    """
    # Trigger download/write-to-cache if needed
    if start and end:
        _ = data_retrieval.load_or_download_ticker(ticker, start=start, end=end)
        cache_key = _cache_key_for_range(start, end)
        cache_path = data_retrieval.get_local_cache_path(ticker, cache_key)
    else:
        period = period or "1y"
        _ = data_retrieval.load_or_download_ticker(ticker, period=period)
        cache_path = data_retrieval.get_local_cache_path(ticker, period)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file expected but not found: {cache_path}")

    # Reload from disk
    df = pd.read_csv(cache_path, parse_dates=True, index_col=0)
    df = data_retrieval.fix_yfinance_dataframe(df)
    df.sort_index(inplace=True)
    return df

# ---------- P&F construction ----------
def _quantize_to_grid(price: float, box: float, mode: str):
    """
    Quantize price to nearest box boundary depending on direction/mode.
    mode in {"floor","ceil","round"}.
    """
    if box <= 0:
        raise ValueError("Box size must be > 0")
    units = price / box
    if mode == "floor":
        q = math.floor(units) * box
    elif mode == "ceil":
        q = math.ceil(units) * box
    else:
        q = round(units) * box
    return q

def compute_box_size(df: pd.DataFrame, box_size: float = None, box_pct: float = None) -> float:
    """
    Decide a constant box size for the entire P&F chart.
    Priority: explicit box_size > box_pct of last close > default 1% of last close.
    """
    last_close = float(df["Close"].iloc[-1])
    if box_size and box_size > 0:
        return float(box_size)
    pct = box_pct if (box_pct and box_pct > 0) else 1.0
    return max(1e-6, (pct / 100.0) * last_close)

def build_pnf_close(df: pd.DataFrame, box: float, reversal: int):
    """
    Close-only P&F.
    Returns list of columns; each column: dict(type 'X'/'O', boxes [prices], start_date, end_date)
    """
    closes = df["Close"].to_numpy(dtype=float)
    dates = df.index.to_list()
    if len(closes) == 0:
        return []

    # Seed with first box level
    first = _quantize_to_grid(closes[0], box, "round")
    cols = []
    curr_type = None
    curr_boxes = [first]
    top = bottom = first

    for i in range(1, len(closes)):
        p = closes[i]
        d = dates[i]

        # If no direction yet, decide when price moves by >= reversal boxes
        if curr_type is None:
            up_thresh = first + reversal * box
            down_thresh = first - reversal * box
            if p >= up_thresh:
                curr_type = 'X'
                # extend up to ceil
                new_top = _quantize_to_grid(p, box, "ceil")
                levels = np.arange(top + box, new_top + 1e-12, box)
                curr_boxes = [first] + list(levels) if len(levels) else [first]
                top = curr_boxes[-1]
                bottom = curr_boxes[0]
            elif p <= down_thresh:
                curr_type = 'O'
                new_bottom = _quantize_to_grid(p, box, "floor")
                levels = np.arange(bottom - box, new_bottom - 1e-12, -box)
                curr_boxes = [first] + list(levels) if len(levels) else [first]
                bottom = curr_boxes[-1]
                top = curr_boxes[0]
            # else remain undecided
            continue

        if curr_type == 'X':
            # Try extend up
            new_top_candidate = _quantize_to_grid(p, box, "ceil")
            if new_top_candidate >= top + box:
                add = np.arange(top + box, new_top_candidate + 1e-12, box)
                curr_boxes.extend(list(add))
                top = curr_boxes[-1]
            else:
                # Check reversal
                if p <= top - reversal * box:
                    # finalize current col
                    cols.append(dict(type='X', boxes=curr_boxes, start_date=dates[i-1], end_date=dates[i-1]))
                    # start 'O' column with reversal boxes down from top - box to level containing p
                    new_bottom = _quantize_to_grid(p, box, "floor")
                    start = top - box  # first O is one box below the X top
                    new_col = [start]
                    while new_col[-1] - box >= new_bottom - 1e-12:
                        new_col.append(new_col[-1] - box)
                    curr_type = 'O'
                    curr_boxes = new_col
                    bottom = curr_boxes[-1]
                    top = curr_boxes[0]
        else:  # 'O'
            # Try extend down
            new_bot_candidate = _quantize_to_grid(p, box, "floor")
            if new_bot_candidate <= bottom - box:
                add = np.arange(bottom - box, new_bot_candidate - 1e-12, -box)
                curr_boxes.extend(list(add))
                bottom = curr_boxes[-1]
            else:
                # Check reversal
                if p >= bottom + reversal * box:
                    cols.append(dict(type='O', boxes=curr_boxes, start_date=dates[i-1], end_date=dates[i-1]))
                    # start 'X' column one box above bottom
                    new_top = _quantize_to_grid(p, box, "ceil")
                    start = bottom + box
                    new_col = [start]
                    while new_col[-1] + box <= new_top + 1e-12:
                        new_col.append(new_col[-1] + box)
                    curr_type = 'X'
                    curr_boxes = new_col
                    top = curr_boxes[-1]
                    bottom = curr_boxes[0]

    # Close out last column
    if curr_boxes:
        cols.append(dict(type=curr_type if curr_type else 'X', boxes=curr_boxes,
                         start_date=dates[-1], end_date=dates[-1]))
    return cols

def build_pnf_hilo(df: pd.DataFrame, box: float, reversal: int):
    """
    Hi-Lo method per day:
    - In X column, attempt to extend upward using High; if not, test for reversal using Low.
    - In O column, attempt to extend downward using Low; if not, test for reversal using High.
    """
    highs = df["High"].to_numpy(dtype=float)
    lows  = df["Low"].to_numpy(dtype=float)
    dates = df.index.to_list()
    if len(highs) == 0:
        return []

    # Initialize around first close grid
    first = _quantize_to_grid(float(df["Close"].iloc[0]), box, "round")
    cols = []
    curr_type = None
    curr_boxes = [first]
    top = bottom = first

    for i in range(len(df)):
        hi, lo, d = highs[i], lows[i], dates[i]

        if curr_type is None:
            # seed direction when range breaks
            if hi >= first + reversal * box:
                curr_type = 'X'
                new_top = _quantize_to_grid(hi, box, "ceil")
                levels = np.arange(top + box, new_top + 1e-12, box)
                curr_boxes = [first] + list(levels) if len(levels) else [first]
                top = curr_boxes[-1]; bottom = curr_boxes[0]
            elif lo <= first - reversal * box:
                curr_type = 'O'
                new_bottom = _quantize_to_grid(lo, box, "floor")
                levels = np.arange(bottom - box, new_bottom - 1e-12, -box)
                curr_boxes = [first] + list(levels) if len(levels) else [first]
                bottom = curr_boxes[-1]; top = curr_boxes[0]
            continue

        if curr_type == 'X':
            # try extend up with High
            new_top_candidate = _quantize_to_grid(hi, box, "ceil")
            if new_top_candidate >= top + box:
                add = np.arange(top + box, new_top_candidate + 1e-12, box)
                curr_boxes.extend(list(add)); top = curr_boxes[-1]
            else:
                # reversal check with Low
                if lo <= top - reversal * box:
                    cols.append(dict(type='X', boxes=curr_boxes, start_date=d, end_date=d))
                    new_bottom = _quantize_to_grid(lo, box, "floor")
                    start = top - box
                    new_col = [start]
                    while new_col[-1] - box >= new_bottom - 1e-12:
                        new_col.append(new_col[-1] - box)
                    curr_type = 'O'; curr_boxes = new_col
                    bottom = curr_boxes[-1]; top = curr_boxes[0]
        else:  # 'O'
            # try extend down with Low
            new_bot_candidate = _quantize_to_grid(lo, box, "floor")
            if new_bot_candidate <= bottom - box:
                add = np.arange(bottom - box, new_bot_candidate - 1e-12, -box)
                curr_boxes.extend(list(add)); bottom = curr_boxes[-1]
            else:
                # reversal check with High
                if hi >= bottom + reversal * box:
                    cols.append(dict(type='O', boxes=curr_boxes, start_date=d, end_date=d))
                    new_top = _quantize_to_grid(hi, box, "ceil")
                    start = bottom + box
                    new_col = [start]
                    while new_col[-1] + box <= new_top + 1e-12:
                        new_col.append(new_col[-1] + box)
                    curr_type = 'X'; curr_boxes = new_col
                    top = curr_boxes[-1]; bottom = curr_boxes[0]

    if curr_boxes:
        cols.append(dict(type=curr_type if curr_type else 'X', boxes=curr_boxes,
                         start_date=dates[-1], end_date=dates[-1]))
    return cols

# ---------- Plotting ----------
def plot_pnf(columns, box: float, ticker: str, title_note: str = ""):
    """
    Render P&F using matplotlib by placing 'X'/'O' glyphs on a box grid.
    X = green (rising), O = red (falling).
    """
    if not columns:
        raise ValueError("No P&F columns produced.")

    # Determine vertical extent
    all_levels = [lvl for col in columns for lvl in col['boxes']]
    ymin, ymax = min(all_levels), max(all_levels)

    # Build grid
    ncols = len(columns)
    fig, ax = plt.subplots(figsize=(max(10, ncols*0.3), 8))
    ax.set_title(f"{ticker} — Point & Figure {title_note}".strip())

    # y grid lines (box levels)
    levels = np.arange(_quantize_to_grid(ymin, box, "floor") - box,
                       _quantize_to_grid(ymax, box, "ceil") + box, box)
    for y in levels:
        ax.hlines(y, -0.5, ncols - 0.5, linewidth=0.3, alpha=0.4)

    # draw columns with colors
    for ci, col in enumerate(columns):
        glyph = 'X' if col['type'] == 'X' else 'O'
        color = 'green' if glyph == 'X' else 'red'
        for lvl in col['boxes']:
            ax.text(ci, lvl, glyph, ha='center', va='center', fontsize=10, color=color)

        # column boundary
        ax.vlines(ci - 0.5, ymin - box, ymax + box, linewidth=0.5, alpha=0.2)

    # legend (minimal)
    ax.plot([], [], linestyle='None', marker='$X$', color='green', label='X (rising)')
    ax.plot([], [], linestyle='None', marker='$O$', color='red',   label='O (falling)')
    ax.legend(loc='upper left')

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(ymin - box, ymax + box)
    ax.set_xlabel("Columns (time →)")
    ax.set_ylabel("Price")
    ax.grid(False)
    # y ticks every N boxes (keep ~12 ticks)
    step_boxes = max(1, int(len(levels) / 12))
    ax.set_yticks(levels[::step_boxes])
    ax.set_yticklabels([f"{y:.2f}" for y in levels[::step_boxes]])

    fig.tight_layout()
    return fig

# ---------- CSV export ----------
def columns_to_dataframe(columns):
    rows = []
    for idx, col in enumerate(columns):
        for lvl in col["boxes"]:
            rows.append({
                "column_index": idx,
                "type": col["type"],
                "price_level": lvl,
                "start_date": col.get("start_date"),
                "end_date": col.get("end_date"),
            })
    return pd.DataFrame(rows)

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Point & Figure chart using cached data via data_retrieval.py")
    p.add_argument("ticker", help="Ticker symbol (e.g., AAPL)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--period", default="1y", help="yfinance period (default: 1y)")
    g.add_argument("--date-range", metavar="START,END", help="YYYY-MM-DD,YYYY-MM-DD")
    p.add_argument("--method", choices=["close", "hilo"], default="hilo",
                   help="P&F construction method: close-only or hi-lo (default: hilo)")
    p.add_argument("--box-size", type=float, default=None, help="Fixed box size in price units (e.g., 1.0)")
    p.add_argument("--box-pct", type=float, default=None, help="Box size as %% of last close (e.g., 1.0)")
    p.add_argument("--reversal", type=int, default=3, help="Reversal in boxes (default: 3)")
    p.add_argument("--save-png", action="store_true", help="Save chart PNG under dated output dir")
    p.add_argument("--save-csv", action="store_true", help="Save CSV with columns/boxes")
    p.add_argument("--no-show", action="store_true", help="Do not display chart window")
    args = p.parse_args()

    # Load data: download first, then reload from disk cache
    if args.date_range:
        try:
            start, end = [s.strip() for s in args.date_range.split(",")]
        except Exception:
            sys.exit("Invalid --date-range. Use START,END with YYYY-MM-DD,YYYY-MM-DD.")
        df = load_df_from_disk_cache(args.ticker, start=start, end=end)
        title_note = f"(Hi-Lo {args.reversal}-box, {start}→{end})" if args.method == "hilo" else f"(Close {args.reversal}-box, {start}→{end})"
    else:
        df = load_df_from_disk_cache(args.ticker, period=args.period)
        title_note = f"(Hi-Lo {args.reversal}-box, period={args.period})" if args.method == "hilo" else f"(Close {args.reversal}-box, period={args.period})"

    if df.empty:
        sys.exit("No data after loading from cache.")

    # Compute box size
    box = compute_box_size(df, box_size=args.box_size, box_pct=args.box_pct)

    # Build P&F
    if args.method == "close":
        columns = build_pnf_close(df, box=box, reversal=args.reversal)
    else:
        columns = build_pnf_hilo(df, box=box, reversal=args.reversal)

    if not columns:
        sys.exit("P&F construction produced no columns. Try a larger period or smaller box size.")

    # Plot
    fig = plot_pnf(columns, box=box, ticker=args.ticker, title_note=title_note)

    # Save outputs to /dev/shm via data_retrieval
    out_dir = data_retrieval.create_output_directory(args.ticker)
    base = os.path.join(out_dir, f"{args.ticker}_pnf_{args.method}_rev{args.reversal}_box{box:.6f}")
    if args.save_png:
        png_path = base + ".png"
        fig.savefig(png_path, dpi=150)
        print(f"[+] Saved PNG: {png_path}")
    if args.save_csv:
        df_cols = columns_to_dataframe(columns)
        csv_path = base + ".csv"
        df_cols.to_csv(csv_path, index=False)
        print(f"[+] Saved CSV: {csv_path}")

    if not args.no_show:
        # Use default MPL viewer
        plt.show()

if __name__ == "__main__":
    main()
