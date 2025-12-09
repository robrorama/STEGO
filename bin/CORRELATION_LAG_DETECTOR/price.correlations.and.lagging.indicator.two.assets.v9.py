#!/usr/bin/env python3
# SCRIPTNAME: price.correlations.and.lagging.indicator.two.assets.v9.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Computes rolling cross-correlations and lead-lag effects between two assets.
#   Uses sliding windows to determine how the correlation evolves and detects
#   optimal time shifts (lags) where correlation is maximized.

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go

# Centralized Data Retrieval
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# Define known commodities for help dialog
KNOWN_COMMODITIES = {
    # Energy
    'CL=F': 'WTI Crude Oil', 'BZ=F': 'Brent Crude Oil', 'NG=F': 'Natural Gas', 'HO=F': 'Heating Oil', 'RB=F': 'RBOB Gasoline', 'EH=F': 'Ethanol',
    # Precious Metals
    'GC=F': 'Gold', 'SI=F': 'Silver', 'PL=F': 'Platinum', 'PA=F': 'Palladium',
    # Industrial Metals
    'HG=F': 'Copper', 'ALI=F': 'Aluminum', 'LME-LEAD=F': 'Lead', 'LME-NICKEL=F': 'Nickel', 'LME-ZINC=F': 'Zinc', 'LME-TIN=F': 'Tin',
    # Grains and Oilseeds
    'ZC=F': 'Corn', 'ZS=F': 'Soybeans', 'ZM=F': 'Soybean Meal', 'ZL=F': 'Soybean Oil', 'ZW=F': 'Wheat', 'ZO=F': 'Oats', 'ZR=F': 'Rough Rice', 'RS=F': 'Canola',
    # Livestock
    'LE=F': 'Live Cattle', 'GF=F': 'Feeder Cattle', 'HE=F': 'Lean Hogs',
    # Soft Commodities
    'KC=F': 'Coffee', 'CC=F': 'Cocoa', 'CT=F': 'Cotton', 'SB=F': 'Sugar #11', 'LBS=F': 'Lumber', 'OJ=F': 'Orange Juice',
}

def print_help_info():
    print("Usage: python price.correlations.and.lagging.indicator.two.assets.v9.py --ticker1 TICKER1 --ticker2 TICKER2")
    print("\nThis script computes correlations and lead-lag effects between the daily returns of two commodities.")
    print("Data is fetched from Yahoo Finance via the data_retrieval module.")
    print("\nKnown commodity symbols (futures):")
    for sym, name in sorted(KNOWN_COMMODITIES.items()):
        print(f"  {sym:<12} : {name}")

def main():
    parser = argparse.ArgumentParser(description='Compute rolling cross-correlations between two asset price returns.')
    parser.add_argument('--ticker1', type=str, help='First ticker symbol (e.g., GC=F)')
    parser.add_argument('--ticker2', type=str, help='Second ticker symbol (e.g., SI=F)')
    parser.add_argument('--no-show', action='store_true', help='Do not open browser tabs, only save files')
    args = parser.parse_args()

    if args.ticker1 is None or args.ticker2 is None:
        print_help_info()
        sys.exit(0)

    # Load historical data using data_retrieval (caches/loads from disk)
    print(f"Loading data for {args.ticker1} and {args.ticker2}...")
    asset1_df = dr.load_or_download_ticker(args.ticker1, period='max')
    asset2_df = dr.load_or_download_ticker(args.ticker2, period='max')
    
    if asset1_df.empty or asset2_df.empty:
        print("Error: One or both tickers returned no data.")
        sys.exit(1)

    asset1 = asset1_df['Close']
    asset2 = asset2_df['Close']
    
    # Merge on index
    df = pd.DataFrame({args.ticker1: asset1, args.ticker2: asset2}).dropna()

    # Use returns for correlation (prices are non-stationary)
    df['ret1'] = df[args.ticker1].pct_change()
    df['ret2'] = df[args.ticker2].pct_change()
    df.dropna(inplace=True)

    # Parameters
    window_size = 252
    step = 63
    lags = range(-30, 31)

    # List of windows
    windows = []
    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        windows.append((start, end))

    if not windows:
        print("Not enough overlapping data to compute sliding windows (Need > 252 days).")
        sys.exit(1)

    # For each window, compute corr at each lag
    results = []
    ccfs = []
    
    print(f"Processing {len(windows)} rolling windows...")
    for start, end in windows:
        ret1 = df['ret1'].iloc[start:end]
        ret2 = df['ret2'].iloc[start:end]
        corrs = []
        for l in lags:
            if l > 0: 
                shifted_ret1 = ret1.iloc[l:]
                shifted_ret2 = ret2.iloc[:-l]
            elif l < 0: 
                shifted_ret1 = ret1.iloc[:l]
                shifted_ret2 = ret2.iloc[-l:]
            else: 
                shifted_ret1 = ret1
                shifted_ret2 = ret2
            
            if len(shifted_ret1) > 1:
                corr, _ = pearsonr(shifted_ret1, shifted_ret2)
            else:
                corr = np.nan
            corrs.append(corr)
            
        ccfs.append(corrs)
        max_corr_val = np.nanmax(corrs)
        max_lag_val = lags[np.nanargmax(corrs)]
        start_date, end_date = df.index[start], df.index[end-1]
        results.append((start_date, end_date, max_corr_val, max_lag_val))

    # Table of results
    table = pd.DataFrame(results, columns=['Start', 'End', 'Max Corr', 'Lag'])
    top = table.sort_values('Max Corr', ascending=False).head(5)
    least = table.sort_values('Max Corr', ascending=True).head(5)
    
    print(f"\nTop most correlated time frames between {args.ticker1} and {args.ticker2}:")
    print(top)
    print(f"\nLeast correlated time frames between {args.ticker1} and {args.ticker2}:")
    print(least)

    # Prepare output directory in /dev/shm
    out_dir = dr.create_output_directory(f"{args.ticker1}_{args.ticker2}")

    # Plot 1: Correlation curves
    fig = go.Figure()
    # Safely find max index handling NaNs
    max_vals = [np.nanmax(c) if not np.all(np.isnan(c)) else -1 for c in ccfs]
    max_idx = np.argmax(max_vals)
    
    for i, corr in enumerate(ccfs):
        fig.add_trace(go.Scatter(x=list(lags), y=corr, mode='lines', opacity=0.2, showlegend=False, line=dict(color='grey')))
    fig.add_trace(go.Scatter(x=list(lags), y=ccfs[max_idx], mode='lines', name='Most Correlated Window', line=dict(color='red', width=3)))
    fig.update_layout(title='Cross-Correlations vs. Lag for All Time Frames', xaxis_title='Lag (days)', yaxis_title='Correlation')
    
    f1_path = os.path.join(out_dir, "1_correlation_curves.html")
    fig.write_html(f1_path)
    if not args.no_show: fig.show()

    # Plot 2: Prices with highlighted time frames
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df[args.ticker1], mode='lines', name=f'{args.ticker1} Price', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=df.index, y=df[args.ticker2], mode='lines', name=f'{args.ticker2} Price', line=dict(color='green'), yaxis='y2'))
    fig2.update_layout(
        yaxis=dict(title=f'{args.ticker1} Price'), 
        yaxis2=dict(title=f'{args.ticker2} Price', overlaying='y', side='right'), 
        title=f'{args.ticker1} and {args.ticker2} Prices with Highlighted Time Frames', xaxis_title='Date'
    )
    for _, row in top.iterrows(): 
        fig2.add_vrect(x0=row['Start'], x1=row['End'], fillcolor='green', opacity=0.15, line_width=0)
    for _, row in least.iterrows(): 
        fig2.add_vrect(x0=row['Start'], x1=row['End'], fillcolor='red', opacity=0.15, line_width=0)
    
    f2_path = os.path.join(out_dir, "2_prices_highlighted.html")
    fig2.write_html(f2_path)
    if not args.no_show: fig2.show()

    # Plot 3: Overlay of rebased prices with annotation
    fig3 = go.Figure()
    best_start, best_end = windows[max_idx]
    for start, end in windows:
        # Rebase to 1.0 at start of window
        rebased1 = df[args.ticker1].iloc[start:end] / df[args.ticker1].iloc[start]
        rebased2 = df[args.ticker2].iloc[start:end] / df[args.ticker2].iloc[start]
        fig3.add_trace(go.Scatter(x=list(range(len(rebased1))), y=rebased1, mode='lines', opacity=0.2, showlegend=False, line=dict(color='lightblue')))
        fig3.add_trace(go.Scatter(x=list(range(len(rebased2))), y=rebased2, mode='lines', opacity=0.2, showlegend=False, line=dict(color='lightgreen')))
    
    best_rebased1 = df[args.ticker1].iloc[best_start:best_end] / df[args.ticker1].iloc[best_start]
    best_rebased2 = df[args.ticker2].iloc[best_start:best_end] / df[args.ticker2].iloc[best_start]
    fig3.add_trace(go.Scatter(x=list(range(len(best_rebased1))), y=best_rebased1, mode='lines', name=f'{args.ticker1} (Most Correlated)', line=dict(color='blue', width=3)))
    fig3.add_trace(go.Scatter(x=list(range(len(best_rebased2))), y=best_rebased2, mode='lines', name=f'{args.ticker2} (Most Correlated)', line=dict(color='green', width=3)))
    
    top_row = top.iloc[0]
    annotation_text = f"<b>Most Correlated Window</b><br>From: {top_row['Start'].strftime('%Y-%m-%d')}<br>To: {top_row['End'].strftime('%Y-%m-%d')}<br>Correlation: {top_row['Max Corr']:.4f}<br>Optimal Lag: {top_row['Lag']} days"
    fig3.add_annotation(text=annotation_text, align='left', showarrow=False, xref='paper', yref='paper', x=0.02, y=0.98, bordercolor="black", borderwidth=1, bgcolor="rgba(255, 255, 255, 0.7)")
    fig3.update_layout(title='Overlay of Rebased Price Series for All Time Windows', xaxis_title=f'Days into {window_size}-day Window', yaxis_title='Price (Rebased to 1.0 at Start)')
    
    f3_path = os.path.join(out_dir, "3_rebased_overlay.html")
    fig3.write_html(f3_path)
    if not args.no_show: fig3.show()

    # PLOT 4: Interactive Lag Slider
    fig4 = go.Figure()
    best_window1 = df[args.ticker1].iloc[best_start:best_end]
    best_window2 = df[args.ticker2].iloc[best_start:best_end]
    
    # Generate traces for every lag
    for lag in lags:
        if lag > 0: shifted1, shifted2 = best_window1.iloc[lag:], best_window2.iloc[:-lag]
        elif lag < 0: shifted1, shifted2 = best_window1.iloc[:lag], best_window2.iloc[-lag:]
        else: shifted1, shifted2 = best_window1, best_window2
        
        fig4.add_trace(go.Scatter(x=shifted1.reset_index(drop=True).index, y=shifted1.reset_index(drop=True), name=args.ticker1, visible=False, line=dict(color='blue')))
        fig4.add_trace(go.Scatter(x=shifted2.reset_index(drop=True).index, y=shifted2.reset_index(drop=True), name=args.ticker2, visible=False, line=dict(color='green'), yaxis='y2'))
    
    initial_lag_index = list(lags).index(0)
    fig4.data[2*initial_lag_index].visible = True
    fig4.data[2*initial_lag_index+1].visible = True
    fig4.data[2*initial_lag_index].name = f'{args.ticker1} (Lag: 0)'
    
    steps = []
    for i, lag in enumerate(lags):
        # Calculate corr for this specific shift
        if lag > 0: s1, s2 = best_window1.iloc[lag:], best_window2.iloc[:-lag]
        elif lag < 0: s1, s2 = best_window1.iloc[:lag], best_window2.iloc[-lag:]
        else: s1, s2 = best_window1, best_window2
        
        if len(s1) > 1:
            current_corr, _ = pearsonr(s1, s2)
        else:
            current_corr = 0
            
        visibility = [False] * len(fig4.data)
        visibility[2*i] = visibility[2*i+1] = True
        names = [args.ticker1 if j % 2 == 0 else args.ticker2 for j in range(len(fig4.data))]
        names[2*i] = f'{args.ticker1} (Lag: {lag})'
        
        step = dict(method="update", 
                    args=[{"visible": visibility, "name": names}, 
                          {"title": f"Interactive Lag Analysis<br>Lag = {lag} days, Correlation = {current_corr:.4f}"}], 
                    label=str(lag))
        steps.append(step)
        
    sliders = [dict(active=initial_lag_index, currentvalue={"prefix": "Lag: "}, pad={"t": 50}, steps=steps)]
    start_date_str, end_date_str = top_row['Start'].strftime('%Y-%m-%d'), top_row['End'].strftime('%Y-%m-%d')
    initial_corr, _ = pearsonr(best_window1, best_window2)
    fig4.update_layout(sliders=sliders, title=f"Interactive Lag Analysis ({start_date_str} to {end_date_str})<br>Lag = 0 days, Correlation = {initial_corr:.4f}", xaxis_title='Trading Days in Window', yaxis=dict(title=f'{args.ticker1} Price'), yaxis2=dict(title=f'{args.ticker2} Price', overlaying='y', side='right'))
    
    f4_path = os.path.join(out_dir, "4_interactive_lag.html")
    fig4.write_html(f4_path)
    if not args.no_show: fig4.show()

    # PLOT 5: Smoothed Data with Buttons and Slider
    fig5 = go.Figure()
    smoothing_periods = [1, 5, 9, 20, 50, 100, 200, 300] # 1 = Unsmoothed
    all_sliders = []
    
    for period in smoothing_periods:
        if period > 1:
            smoothed1 = best_window1.rolling(window=period).mean().dropna()
            smoothed2 = best_window2.rolling(window=period).mean().dropna()
        else:
            smoothed1, smoothed2 = best_window1, best_window2
            
        for lag in lags:
            if lag > 0: s1, s2 = smoothed1.iloc[lag:], smoothed2.iloc[:-lag]
            elif lag < 0: s1, s2 = smoothed1.iloc[:lag], smoothed2.iloc[-lag:]
            else: s1, s2 = smoothed1, smoothed2
            
            fig5.add_trace(go.Scatter(x=s1.reset_index(drop=True).index, y=s1.reset_index(drop=True), name=f'{args.ticker1}', visible=False, line=dict(color='blue')))
            fig5.add_trace(go.Scatter(x=s2.reset_index(drop=True).index, y=s2.reset_index(drop=True), name=f'{args.ticker2}', visible=False, line=dict(color='green'), yaxis='y2'))

    # Create sliders
    num_lags = len(lags)
    for i, period in enumerate(smoothing_periods):
        steps = []
        for j, lag in enumerate(lags):
            base_idx = (i * num_lags * 2) + (j * 2)
            visibility = [False] * len(fig5.data)
            visibility[base_idx] = visibility[base_idx + 1] = True
            
            # Recalc corr for title
            if period > 1: 
                s1_base = best_window1.rolling(window=period).mean().dropna()
                s2_base = best_window2.rolling(window=period).mean().dropna()
            else: 
                s1_base, s2_base = best_window1, best_window2
                
            if lag > 0: cs1, cs2 = s1_base.iloc[lag:], s2_base.iloc[:-lag]
            elif lag < 0: cs1, cs2 = s1_base.iloc[:lag], s2_base.iloc[-lag:]
            else: cs1, cs2 = s1_base, s2_base
            
            if len(cs1) > 1:
                current_corr, _ = pearsonr(cs1, cs2)
            else:
                current_corr = 0
                
            step = dict(method="update", args=[{"visible": visibility}, {"title": f"Smoothed Lag Analysis ({period}-Day SMA)<br>Lag = {lag} days, Correlation = {current_corr:.4f}"}], label=str(lag))
            steps.append(step)
        slider = dict(active=initial_lag_index, currentvalue={"prefix": "Lag: "}, pad={"t": 50}, steps=steps)
        all_sliders.append(slider)

    buttons = []
    for i, period in enumerate(smoothing_periods):
        base_idx = (i * num_lags * 2) + (initial_lag_index * 2)
        visibility = [False] * len(fig5.data)
        visibility[base_idx] = visibility[base_idx + 1] = True
        label = f"{period}-Day SMA" if period > 1 else "Unsmoothed"
        button = dict(label=label, method="update", args=[{"visible": visibility, "sliders": [all_sliders[i]]}])
        buttons.append(button)

    fig5.data[initial_lag_index*2].visible = True
    fig5.data[initial_lag_index*2+1].visible = True

    fig5.update_layout(
        updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center", y=1.15, yanchor="top", buttons=buttons)],
        sliders=[all_sliders[0]],
        title=f"Smoothed Lag Analysis (Unsmoothed)<br>Lag = 0 days, Correlation = {initial_corr:.4f}",
        xaxis_title='Trading Days in Window',
        yaxis=dict(title=f'{args.ticker1} Price'),
        yaxis2=dict(title=f'{args.ticker2} Price', overlaying='y', side='right'),
    )
    
    f5_path = os.path.join(out_dir, "5_smoothed_lag.html")
    fig5.write_html(f5_path)
    if not args.no_show: fig5.show()

    print(f"\nAnalysis Complete. Charts saved to: {out_dir}")

    # Stats
    print("\nNote on lag interpretation: Negative lag means asset2 lags asset1 (asset1 leads); Positive lag means asset1 lags asset2 (asset2 leads).")
    print(f"Here, asset1 is {args.ticker1}, asset2 is {args.ticker2}.")
    print("\nSummary of lags across all windows:")
    lags_all = table['Lag']
    print(f"Min lag: {lags_all.min()} days")
    print(f"Max lag: {lags_all.max()} days")
    print(f"Mean lag: {lags_all.mean():.2f} days")
    print(f"Median lag: {lags_all.median()} days")
    if not lags_all.mode().empty:
        print(f"Mode lag: {lags_all.mode()[0]} days")

if __name__ == "__main__":
    main()
