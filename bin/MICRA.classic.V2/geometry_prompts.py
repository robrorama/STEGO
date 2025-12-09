#!/usr/bin/env python3
# SCRIPTNAME: geometry_prompts.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import pandas as pd
import numpy as np
from geometry import find_two_high_peaks_in_period, find_two_low_troughs_in_period, plot_projection_line

def interval_to_days(interval):
    mapping = {'1m': 30, '3m': 90, '6m': 180, '1y': 365, '3y': 1095, '5y': 1825, '10y': 3650}
    return mapping.get(interval.lower(), 90)

def get_loc_nearest(dtindex, date_to_find):
    target = np.datetime64(date_to_find)
    pos = dtindex.searchsorted(target)
    if pos == 0: return 0
    if pos == len(dtindex): return len(dtindex) - 1
    before, after = pos - 1, pos
    return after if (dtindex[after] - target) < (target - dtindex[before]) else before

def manual_peaks_or_troughs(df, is_peaks=True):
    label = 'High' if is_peaks else 'Low'
    d1 = pd.to_datetime(input(f"Enter date for first {label} (YYYY-MM-DD): ").strip())
    d2 = pd.to_datetime(input(f"Enter date for second {label} (YYYY-MM-DD): ").strip())
    
    i1 = get_loc_nearest(df.index, d1)
    i2 = get_loc_nearest(df.index, d2)
    
    return pd.Series([df[label].iloc[i1], df[label].iloc[i2]], index=[df.index[i1], df.index[i2]])
