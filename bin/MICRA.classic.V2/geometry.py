#!/usr/bin/env python3
# SCRIPTNAME: geometry.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta
from scipy.signal import argrelextrema

def find_two_high_peaks_in_period(df, days=90):
    cutoff = df.index[-1] - timedelta(days=days)
    recent_data = df[df.index >= cutoff]
    if len(recent_data) < 2 or 'High' not in recent_data.columns:
        raise ValueError("Not enough data or missing High column.")

    indices = argrelextrema(recent_data['High'].values, np.greater_equal, order=5)[0]
    if len(indices) >= 2:
        peaks = recent_data.iloc[indices].nlargest(2, 'High')
    else:
        peaks = recent_data.nlargest(2, 'High')

    if len(peaks) < 2:
        raise ValueError("Could not identify two distinct high peaks.")
    return peaks

def find_two_low_troughs_in_period(df, days=90):
    cutoff = df.index[-1] - timedelta(days=days)
    recent_data = df[df.index >= cutoff]
    if len(recent_data) < 2 or 'Low' not in recent_data.columns:
        raise ValueError("Not enough data or missing Low column.")

    indices = argrelextrema(recent_data['Low'].values, np.less_equal, order=5)[0]
    if len(indices) >= 2:
        troughs = recent_data.iloc[indices].nsmallest(2, 'Low')
    else:
        troughs = recent_data.nsmallest(2, 'Low')
        
    if len(troughs) < 2:
        raise ValueError("Could not identify two distinct low troughs.")
    return troughs

def calculate_intersection(slope1, intercept1, slope2, intercept2):
    if slope1 == slope2:
        raise ValueError("Lines are parallel.")
    if any(val is None for val in [slope1, intercept1, slope2, intercept2]):
        raise ValueError("Undefined line parameters.")

    x_intersect_ord = (intercept2 - intercept1) / (slope1 - slope2)
    
    try:
        date_intersect = pd.Timestamp.fromordinal(int(round(x_intersect_ord)))
    except ValueError as e:
        raise ValueError(f"Date conversion error: {e}")
        
    y_intersect_price = slope1 * x_intersect_ord + intercept1
    return date_intersect, y_intersect_price

def plot_projection_line(df, fig, points_series, color='black', line_name='Projection Line', project_until_date=None):
    if not isinstance(points_series, pd.Series) or len(points_series) < 2:
        return None, None
    points_series = points_series.sort_index()
    
    d1, d2 = points_series.index[0], points_series.index[1]
    p1, p2 = points_series.iloc[0], points_series.iloc[1]

    days_diff = (d2.toordinal() - d1.toordinal())
    if days_diff == 0: return None, None

    slope = (p2 - p1) / days_diff
    intercept = p1 - slope * (d1.toordinal())

    # Initial segment
    fig.add_trace(go.Scatter(x=[d1, d2], y=[p1, p2], mode='lines', line=dict(color=color, width=2), name=f'{line_name} (Segment)'))

    # Projection
    end_target = project_until_date if project_until_date else (df.index[-1] + timedelta(days=30))
    if end_target > d2:
        x_proj = pd.date_range(start=d2 + timedelta(days=1), end=end_target, freq='B')
        if len(x_proj) > 0:
            ords = np.array([d.toordinal() for d in x_proj])
            y_proj = slope * ords + intercept
            fig.add_trace(go.Scatter(x=[d2] + list(x_proj), y=[p2] + list(y_proj), mode='lines', line=dict(color=color, width=2, dash='dash'), name=f'{line_name} (Projection)'))

    fig.add_trace(go.Scatter(x=[d1, d2], y=[p1, p2], mode='markers', marker=dict(symbol='circle', size=10, color=color, line=dict(width=1, color='DarkSlateGrey')), name=f'{line_name} Points'))
    return slope, intercept

def calculate_linear_regression_and_deviations(df, length, price_column='Close'):
    if len(df) < length: return None, None, None, {}
    
    slice_df = df.iloc[-length:]
    y = slice_df[price_column].values
    valid = ~pd.isna(y)
    
    if np.sum(valid) < 2: return None, None, None, {}
    
    x_ord = np.array([d.toordinal() for d in slice_df.index])
    x_clean = x_ord[valid]
    y_clean = y[valid]

    A = np.vstack([x_clean, np.ones(len(x_clean))]).T
    slope, intercept = np.linalg.lstsq(A, y_clean, rcond=None)[0]
    
    resid = y_clean - (slope * x_clean + intercept)
    std = np.std(resid)
    
    # Calculate bands for full length
    base = slope * x_ord + intercept
    deviations = {}
    for m in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        deviations[f'upper_{m}'] = base + m * std
        deviations[f'lower_{m}'] = base - m * std
        
    return slope, intercept, std, deviations
