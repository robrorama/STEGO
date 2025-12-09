#!/usr/bin/env python3
# SCRIPTNAME: summary.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import pandas as pd
import os

def generate_summary_output(ticker, buy, sell, stars, wick, fib_wick, ma, out_dir):
    data = []
    for d, p in buy: data.append({"Date": d, "Signal": "Buy", "Price": p})
    for d, p in sell: data.append({"Date": d, "Signal": "Sell", "Price": p})
    for d, p, s, c in stars: data.append({"Date": d, "Signal": f"{c.capitalize()} Star", "Price": p, "Size": s})
    for d, (l, p) in wick: data.append({"Date": d, "Signal": "Wick Touch", "Level": l, "Price": p})
    for d, (l, p) in fib_wick: data.append({"Date": d, "Signal": "Fib Wick Touch", "Level": l, "Price": p})
    for d, (m, p) in ma: data.append({"Date": d, "Signal": "MA Touch", "MA": m, "Price": p})

    # CONSTRAINT: Output directory is handled by caller (micra.py) which uses data_retrieval
    pd.DataFrame(data).to_csv(os.path.join(out_dir, f"{ticker}_detailed_signal_summary.csv"), index=False)
    print(f"Summary saved to {out_dir}")
