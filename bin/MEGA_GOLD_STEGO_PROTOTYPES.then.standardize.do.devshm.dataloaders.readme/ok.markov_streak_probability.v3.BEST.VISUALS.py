#!/usr/bin/env python3
"""
SCRIPTNAME: markov_dashboard_v3.py
AUTHOR: Stego Financial Framework
DATE: November 25, 2025

DESCRIPTION:
    V3 of the Empirical Markov Chain Dashboard.
    - Generates TWO Dashboard Tabs:
      1. Multi-Ticker Comparison (2x4 Grid)
      2. Advanced Intelligence (5 Creative Visuals: Heatmap, Sankey, 3D, Phase, Sunburst)
    - Automatically opens both in browser.

USAGE:
    python markov_dashboard_v3.py --tickers "NVDA,AMD" --period 2y
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

# CONSTRAINT: Prevent __pycache__ creation
sys.dont_write_bytecode = True

# CONSTRAINT: Use provided data_retrieval module strictly
try:
    import data_retrieval as dr
except ImportError:
    print("CRITICAL: data_retrieval.py not found.")
    sys.exit(1)

# --- Configuration ---
MAX_STREAK_DISPLAY = 10
MIN_SAMPLE_SIZE = 5
TICKER_COLORS = ['#00FF00', '#FF4500', '#00BFFF', '#FFD700', '#DA70D6']

# ==============================================================================
#  CORE LOGIC (From V2)
# ==============================================================================

def calculate_continuation_prob(df: pd.DataFrame, condition_mask=None) -> pd.DataFrame:
    target = df[condition_mask].copy() if condition_mask is not None else df.copy()
    if target.empty: return pd.DataFrame()
    stats = target.groupby('streak_len')['is_continuation'].agg(['count', 'sum'])
    stats.rename(columns={'sum': 'continued'}, inplace=True)
    stats['prob'] = (stats['continued'] / stats['count']) * 100.0
    stats = stats[stats['count'] >= MIN_SAMPLE_SIZE]
    return stats[stats.index <= MAX_STREAK_DISPLAY]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['daily_ret'] = d['Close'].pct_change()
    d['direction'] = np.sign(d['daily_ret']).replace(0, method='ffill')
    d['streak_id'] = (d['direction'] != d['direction'].shift()).cumsum()
    d['streak_len'] = d.groupby('streak_id').cumcount() + 1
    d['is_continuation'] = (d['streak_id'] == d['streak_id'].shift(-1)).astype(int)
    
    # Volatility & Levels
    d['roll_std'] = d['daily_ret'].rolling(20).std()
    d['z_ret'] = d['daily_ret'].abs() / d['roll_std']
    
    # Filters
    d['cond_high_mag'] = d['z_ret'] > 1.5
    d['cond_low_mag']  = d['z_ret'] < 0.5
    d['high_252'] = d['High'].rolling(252).max()
    d['support_50'] = d['Low'].rolling(50).min()
    d['cond_structural'] = (d['Close'] >= d['high_252']*0.98) | (d['Close'] <= d['support_50']*1.02)
    d['cond_open_air'] = ~d['cond_structural']
    
    # Derivatives
    d['velocity'] = d['Close'].diff()
    d['acceleration'] = d['velocity'].diff()
    d['is_accel'] = ((d['direction']==1) & (d['acceleration']>0)) | ((d['direction']==-1) & (d['acceleration']<0))
    d['cond_accelerating'] = d['is_accel']
    d['cond_decelerating'] = ~d['is_accel']
    
    # Trend
    d['sma50'] = d['Close'].rolling(50).mean()
    d['dist_sma'] = (d['Close'] - d['sma50']) / d['roll_std']
    d['cond_extended'] = d['dist_sma'].abs() > 2.0
    d['cond_compressed'] = d['dist_sma'].abs() < 1.0
    
    # Bollinger
    bb_mid = d['Close'].rolling(20).mean()
    bb_std = d['Close'].rolling(20).std()
    d['cond_piercing'] = (d['Close'] > (bb_mid+2*bb_std)) | (d['Close'] < (bb_mid-2*bb_std))
    d['cond_inside'] = ~d['cond_piercing']
    
    # Volume
    v_fast = d['Volume'].rolling(3).mean()
    v_slow = d['Volume'].rolling(10).mean()
    d['cond_vol_rising'] = v_fast > v_slow
    d['cond_vol_falling'] = v_fast < v_slow
    
    # RSI
    d['rsi'] = calculate_rsi(d['Close'], 14)
    d['cond_rsi_bull'] = d['rsi'] > 60
    d['cond_rsi_bear'] = d['rsi'] < 40

    return d.dropna()

# ==============================================================================
#  ADVANCED VISUALIZATION PREP
# ==============================================================================

def prep_survival_matrix(df):
    """Generates Z-Matrix for Heatmap: [Conditions] x [Streak Day]"""
    conditions = {
        'Base': None,
        'High Volatility': df['cond_high_mag'],
        'Low Volatility': df['cond_low_mag'],
        'Accelerating': df['cond_accelerating'],
        'Decelerating': df['cond_decelerating'],
        'Extended': df['cond_extended'],
        'Compressed': df['cond_compressed'],
        'RSI Bull': df['cond_rsi_bull'],
        'RSI Bear': df['cond_rsi_bear'],
        'Piercing BB': df['cond_piercing']
    }
    
    matrix_data = []
    y_labels = []
    
    for name, mask in conditions.items():
        stats = calculate_continuation_prob(df, mask)
        # Create row for days 1..10
        row = []
        for day in range(1, 11):
            if day in stats.index:
                row.append(stats.loc[day, 'prob'])
            else:
                row.append(None) # NaN
        matrix_data.append(row)
        y_labels.append(name)
        
    return matrix_data, y_labels

def prep_sankey_data(df):
    """Filters for UP streaks and maps the flow 1->2->3..."""
    # Filter for Positive Streaks only
    up_streaks = df[df['direction'] == 1].copy()
    
    # Nodes: Start, Day1_Up, Day1_End, Day2_Up, Day2_End...
    # Simplified Logic: 
    # Levels: Day 1, Day 2, Day 3, Day 4, Day 5
    # For each level, we have "Survivors" and "Casualties"
    
    labels = ["Start"]
    sources = []
    targets = []
    values = []
    colors = []
    
    # Map node indices
    # 0: Start
    # 1: Day 1 Continue (Up)
    # 2: Day 1 End (Revert)
    # 3: Day 2 Continue...
    
    # Let's limit depth to 5 days for the River
    depth = 6
    
    # Initial Volume: Total Day 1 Up streaks
    day1_stats = up_streaks[up_streaks['streak_len'] == 1]
    total_start = len(day1_stats)
    
    # Start -> Day 1 Logic is slightly redundant as 'Streak 1' implies it exists.
    # Let's map Day K -> Day K+1
    
    # Create Labels
    for i in range(1, depth):
        labels.append(f"Day {i} Up")       # Index 2*i - 1
        labels.append(f"Day {i} Revert")   # Index 2*i
        
    # Build Links
    # Logic: From "Day K Up", how many go to "Day K+1 Up" vs "Day K Revert"
    # Note: "Day K Revert" happens AT day K+1 check.
    
    # Inject Start Node (0) -> Day 1 Up (1)
    sources.append(0); targets.append(1); values.append(total_start); colors.append("rgba(0, 255, 0, 0.5)")
    
    current_up_node_idx = 1
    
    for day in range(1, depth-1):
        # Identify streaks that reached this day
        survivors = up_streaks[up_streaks['streak_len'] == day]
        count_continue = survivors['is_continuation'].sum()
        count_die = survivors['is_continuation'].count() - count_continue
        
        next_up_node = current_up_node_idx + 2
        next_die_node = current_up_node_idx + 3 # Actually die node is usually terminal, but we visualize it as a branch
        
        # Link: Day K Up -> Day K+1 Up
        if count_continue > 0:
            sources.append(current_up_node_idx)
            targets.append(next_up_node)
            values.append(count_continue)
            colors.append("rgba(0, 200, 0, 0.6)") # Green flow
            
        # Link: Day K Up -> Revert
        if count_die > 0:
            sources.append(current_up_node_idx)
            targets.append(next_up_node + 1) # The Revert node for the NEXT level
            values.append(count_die)
            colors.append("rgba(200, 0, 0, 0.4)") # Red flow
            
        current_up_node_idx = next_up_node

    return labels, sources, targets, values, colors

def prep_3d_terrain(df):
    """Buckets streaks by Length (X) and Magnitude (Y) to find Prob (Z)"""
    # Binning Magnitude (Z-Ret)
    df['mag_bin'] = pd.qcut(df['z_ret'], 5, labels=False, duplicates='drop')
    
    z_matrix = np.zeros((5, 10))
    x_days = list(range(1, 11))
    y_bins = ["Very Low", "Low", "Med", "High", "Very High"]
    
    for bin_idx in range(5):
        mask = df['mag_bin'] == bin_idx
        stats = calculate_continuation_prob(df, mask)
        for day in range(1, 11):
            if day in stats.index:
                z_matrix[bin_idx, day-1] = stats.loc[day, 'prob']
            else:
                z_matrix[bin_idx, day-1] = 0 # or NaN
                
    return x_days, y_bins, z_matrix

def prep_phase_space(df):
    """Velocity vs Acceleration colored by Continuation Probability"""
    # Use normalized velocity (daily_ret) and acceleration (diff of daily_ret)
    # for better cross-comparison
    df['vel_norm'] = df['daily_ret'] * 100
    df['accel_norm'] = df['daily_ret'].diff() * 100
    df_clean = df.dropna()
    
    return df_clean['vel_norm'], df_clean['accel_norm'], df_clean['is_continuation']

def prep_sunburst(df):
    """Builds hierarchical paths of Up/Down sequences (Depth 4)"""
    # We need to reconstruct sequences.
    # Sliding window of 4 days.
    # Convert direction 1/-1 to "Up"/"Down"
    
    dirs = df['direction'].map({1: 'Up', -1: 'Down'}).values
    
    paths = []
    # Collect all 4-day sequences
    for i in range(len(dirs) - 3):
        seq = dirs[i:i+4]
        paths.append(list(seq))
        
    # Build Plotly Sunburst format (ids, labels, parents, values)
    # Root -> L1 -> L2 -> L3
    
    # Helper to aggregate
    # We can use pandas to groupby all levels
    pdf = pd.DataFrame(paths, columns=['L1', 'L2', 'L3', 'L4'])
    
    # Sunburst specs
    ids = []
    labels = []
    parents = []
    values = []
    
    # Root
    ids.append("Start")
    labels.append("Start")
    parents.append("")
    values.append(len(pdf))
    
    # Level 1
    for L1, group in pdf.groupby('L1'):
        id_1 = f"Start - {L1}"
        ids.append(id_1); labels.append(L1); parents.append("Start"); values.append(len(group))
        
        # Level 2
        for L2, group2 in group.groupby('L2'):
            id_2 = f"{id_1} - {L2}"
            ids.append(id_2); labels.append(L2); parents.append(id_1); values.append(len(group2))
            
            # Level 3
            for L3, group3 in group2.groupby('L3'):
                id_3 = f"{id_2} - {L3}"
                ids.append(id_3); labels.append(L3); parents.append(id_2); values.append(len(group3))
    
    return ids, labels, parents, values

# ==============================================================================
#  PLOTTING
# ==============================================================================

def plot_multi_ticker_grid(ticker_data_map, output_dir):
    """(V2 Logic) Generates the 2x4 Comparison Grid"""
    ticker_str = " vs ".join(ticker_data_map.keys())
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=("1. Base", "2. Magnitude", "3. Structure", "4. Velocity", 
                        "5. Trend Dev", "6. Bollinger", "7. Volume", "8. RSI Regime"),
        shared_xaxes=True
    )

    panels = [
        (1, 2, 'cond_high_mag', 'High Mag', 'cond_low_mag', 'Low Mag'),
        (1, 3, 'cond_structural', 'Structural', 'cond_open_air', 'Open Air'),
        (1, 4, 'cond_accelerating', 'Accel', 'cond_decelerating', 'Decel'),
        (2, 1, 'cond_extended', 'Extended', 'cond_compressed', 'Compressed'),
        (2, 2, 'cond_piercing', 'Piercing', 'cond_inside', 'Inside BB'),
        (2, 3, 'cond_vol_rising', 'Vol Rising', 'cond_vol_falling', 'Vol Falling'),
        (2, 4, 'cond_rsi_bull', 'RSI Bull', 'cond_rsi_bear', 'RSI Bear')
    ]

    for i, (ticker, df) in enumerate(ticker_data_map.items()):
        color = TICKER_COLORS[i % len(TICKER_COLORS)]
        
        # Pane 1 Base
        stats = calculate_continuation_prob(df)
        if not stats.empty:
            fig.add_trace(go.Scatter(x=stats.index, y=stats['prob'], mode='lines+markers', 
                                     name=f"{ticker} Base", line=dict(color=color, width=3), 
                                     marker=dict(size=6), legendgroup=ticker), row=1, col=1)
        
        # Panes 2-8
        for (r, c, col1, name1, col2, name2) in panels:
            s1 = calculate_continuation_prob(df, df[col1])
            s2 = calculate_continuation_prob(df, df[col2])
            if not s1.empty:
                fig.add_trace(go.Scatter(x=s1.index, y=s1['prob'], mode='lines', 
                                         line=dict(color=color, width=2, dash='solid'), showlegend=False), row=r, col=c)
            if not s2.empty:
                fig.add_trace(go.Scatter(x=s2.index, y=s2['prob'], mode='lines', 
                                         line=dict(color=color, width=2, dash='dot'), showlegend=False), row=r, col=c)

    # Reference Lines
    for r in [1, 2]:
        for c in [1, 2, 3, 4]:
            fig.add_shape(type="line", x0=1, x1=MAX_STREAK_DISPLAY, y0=50, y1=50, 
                          line=dict(color="gray", width=1, dash="dash"), row=r, col=c)

    fig.update_layout(title=f"Markov Dashboard: {ticker_str}", template="plotly_dark", height=900, width=1600)
    out_path = os.path.join(output_dir, f"markov_grid_{ticker_str.replace(' ', '_')}.html")
    fig.write_html(out_path)
    return out_path

def plot_advanced_visuals(ticker, df, output_dir):
    """Generates the 5 Advanced Visuals in a Layout"""
    
    # Prep Data
    hm_z, hm_y = prep_survival_matrix(df)
    sk_lbl, sk_src, sk_tgt, sk_val, sk_col = prep_sankey_data(df)
    tr_x, tr_y, tr_z = prep_3d_terrain(df)
    ph_x, ph_y, ph_c = prep_phase_space(df)
    sb_id, sb_lbl, sb_par, sb_val = prep_sunburst(df)

    # Create Complex Subplot Layout
    # Row 1: Heatmap (Span all)
    # Row 2: Sankey (Domain) | Sunburst (Domain)
    # Row 3: 3D Surface (Scene) | Phase Space (XY)
    
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2, "type": "xy"}, None],
            [{"type": "domain"}, {"type": "domain"}],
            [{"type": "scene"}, {"type": "xy"}]
        ],
        subplot_titles=(
            "1. Survival Matrix (Conditional Heatmap)",
            "2. River of Returns (Up-Streak Sankey)", "3. Sequence Probability (Sunburst)",
            "4. Probability Terrain (Magnitude vs Streak)", "5. Phase Space (Vel vs Accel)"
        ),
        vertical_spacing=0.1
    )
    
    # 1. Heatmap
    fig.add_trace(go.Heatmap(
        z=hm_z, x=list(range(1, 11)), y=hm_y,
        colorscale='RdYlGn', zmin=30, zmax=70,
        hoverongaps=False, showscale=True, colorbar=dict(title="Prob %", x=1.02, y=0.9)
    ), row=1, col=1)
    
    # 2. Sankey
    fig.add_trace(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=sk_lbl, color="blue"),
        link=dict(source=sk_src, target=sk_tgt, value=sk_val, color=sk_col)
    ), row=2, col=1)
    
    # 3. Sunburst
    fig.add_trace(go.Sunburst(
        ids=sb_id, labels=sb_lbl, parents=sb_par, values=sb_val,
        branchvalues="total", marker=dict(colorscale='RdYlGn')
    ), row=2, col=2)
    
    # 4. 3D Terrain
    fig.add_trace(go.Surface(
        z=tr_z, x=tr_x, y=tr_y,
        colorscale='RdYlGn', showscale=False, opacity=0.9
    ), row=3, col=1)
    
    # 5. Phase Space (Binning for Heatmap effect)
    fig.add_trace(go.Histogram2dContour(
        x=ph_x, y=ph_y, z=ph_c, histfunc='avg',
        colorscale='RdYlGn', nbinsx=30, nbinsy=30,
        contours=dict(coloring='heatmap'),
        colorbar=dict(title="Avg Continuation Prob", x=1.02, y=0.1)
    ), row=3, col=2)

    fig.update_layout(
        title=f"Advanced Intelligence Dashboard: {ticker}",
        template="plotly_dark",
        height=1400, width=1600,
        scene=dict(
            xaxis_title="Streak Length",
            yaxis_title="Magnitude Bin",
            zaxis_title="Probability %",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
    )
    
    out_path = os.path.join(output_dir, f"markov_advanced_{ticker}.html")
    fig.write_html(out_path)
    return out_path

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, type=str, help="Comma-separated (e.g., NVDA,AMD)")
    parser.add_argument("--period", default="2y", type=str)
    args = parser.parse_args()

    ticker_list = [t.strip().upper() for t in args.tickers.split(',')]
    data_map = {}
    
    # 1. Fetch Data
    for t in ticker_list:
        print(f"Processing {t}...")
        raw_df = dr.get_stock_data(t, period=args.period)
        if not raw_df.empty:
            data_map[t] = prepare_streak_features(raw_df)
        else:
            print(f"Warning: No data for {t}")

    if not data_map: sys.exit(1)

    # 2. Generate Grid Dashboard (Multi-Ticker)
    out_dir = dr.create_output_directory(ticker_list[0])
    print("Generating Multi-Ticker Grid...")
    grid_path = plot_multi_ticker_grid(data_map, out_dir)
    
    # 3. Generate Advanced Dashboard (Primary Ticker Only)
    primary_ticker = ticker_list[0]
    print(f"Generating Advanced Visuals for {primary_ticker}...")
    adv_path = plot_advanced_visuals(primary_ticker, data_map[primary_ticker], out_dir)
    
    # 4. Open Tabs
    print(f"Opening Tab 1: {grid_path}")
    webbrowser.open(f"file://{os.path.abspath(grid_path)}")
    
    print(f"Opening Tab 2: {adv_path}")
    webbrowser.open(f"file://{os.path.abspath(adv_path)}")

if __name__ == "__main__":
    main()
