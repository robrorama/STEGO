#!/usr/bin/env python3
# SCRIPTNAME: ok.equity_anomalies_cli.V3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
equity_anomalies_cli.py

A standalone CLI tool to analyze equity market anomalies (Momentum, Reversal, TOM, PEAD).
Adheres to strict constraints: sequential processing, CSV round-trips, and specific robustness fixes.

Usage:
    python3 equity_anomalies_cli.py TICKERS... [--mode {A,B,C}] [--out OUTDIR]

Dependencies:
    numpy, pandas, yfinance, plotly, scipy
"""

import sys
import os
import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# Suppress pandas chained assignment warnings for cleaner CLI output
pd.options.mode.chained_assignment = None

# ==========================================
# 1. Helper Functions & Mandatory Fixes
# ==========================================

def get_output_dir(user_out, ticker):
    """
    Determines output directory based on priority:
    1. User provided --out
    2. /dev/shm/ANOM (if writable, for speed)
    3. ./out (fallback)
    
    Structure: <ROOT>/<TICKER>/<YYYY-MM-DD>/
    """
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    
    if user_out:
        base = Path(user_out)
    else:
        # Try /dev/shm for speed on Linux
        shm_path = Path('/dev/shm')
        if shm_path.exists() and os.access(str(shm_path), os.W_OK):
            base = shm_path / 'ANOM'
        else:
            base = Path('./out')
            
    target_dir = base / ticker / today_str
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def force_naive_index(df):
    """
    MANDATORY FIX: Timezone Fix.
    Force standard index and remove timezone info.
    """
    if df is None or df.empty:
        return df
        
    # Force standard index
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def robust_close_series(df):
    """
    MANDATORY FIX: yfinance Fix.
    Robust column extraction for price.
    """
    if isinstance(df, pd.DataFrame):
        if 'Close' in df.columns:
            s = df['Close']
        elif 'Adj Close' in df.columns:
            s = df['Adj Close']
        else:
            s = df.iloc[:, 0]  # Fallback: first column
    else:
        s = df
    
    # Return numeric (coerce errors)
    return pd.to_numeric(s, errors='coerce')

def intersect_common_index(df_assets, df_proxies=None, df_fx=None):
    """
    MANDATORY FIX: Alignment Fix.
    Find intersection of ALL loaded data first.
    """
    if df_proxies is None: df_proxies = pd.DataFrame()
    if df_fx is None: df_fx = pd.DataFrame()

    # Find intersection of ALL loaded data first
    common = df_assets.index
    if not df_proxies.empty: common = common.intersection(df_proxies.index)
    if not df_fx.empty: common = common.intersection(df_fx.index)

    # Hard-filter everything to this common index
    df_assets = df_assets.loc[common]
    
    # Only filter if not empty (avoids KeyError on empty frames)
    if not df_proxies.empty:
        df_proxies = df_proxies.loc[common]
    if not df_fx.empty:
        df_fx = df_fx.loc[common]
    
    return df_assets, df_proxies, df_fx

def strict_csv_roundtrip(df, filepath, index_label='Date'):
    """
    Writes DataFrame to CSV, reads it back, applies Timezone Fix.
    This ensures strict persistence logic is followed.
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    # Write
    df.to_csv(filepath, index_label=index_label)
    
    # Read back
    df_read = pd.read_csv(filepath, index_col=0)
    
    # Apply Mandatory Timezone Fix immediately after read
    df_read = force_naive_index(df_read)
    
    return df_read

# ==========================================
# 2. Data Loading
# ==========================================

def load_prices_one_ticker(ticker, output_dir):
    """
    Fetches history from yfinance, saves raw CSV, round-trips.
    """
    print(f"[{ticker}] Downloading price history...")
    try:
        # Constraints: Period=max, no auto_adjust, no actions
        t = yf.Ticker(ticker)
        df = t.history(period="max", auto_adjust=False, actions=False)
        
        if df.empty:
            print(f"[{ticker}] Warning: No data found.")
            return pd.DataFrame()

        # Apply TZ fix immediately after fetch (before save)
        df = force_naive_index(df)

        # Save and Round-trip
        path = output_dir / f"{ticker}_raw.csv"
        df_round = strict_csv_roundtrip(df, path)
        
        return df_round
    except Exception as e:
        print(f"[{ticker}] Error downloading: {e}")
        return pd.DataFrame()

def load_earnings_dates(ticker, output_dir):
    """
    Fetches earnings dates, saves raw CSV, round-trips.
    """
    print(f"[{ticker}] Downloading earnings dates...")
    try:
        t = yf.Ticker(ticker)
        # Constraint: limit=24
        df = t.get_earnings_dates(limit=24)
        
        if df is None or df.empty:
            return pd.DataFrame()

        # Apply TZ fix immediately
        df = force_naive_index(df)

        path = output_dir / f"{ticker}_earnings_raw.csv"
        df_round = strict_csv_roundtrip(df, path)
        return df_round
    except Exception as e:
        print(f"[{ticker}] Warning: Could not fetch earnings ({e})")
        return pd.DataFrame()

# ==========================================
# 3. Computations
# ==========================================

def compute_basics_and_momentum(df_raw, output_dir):
    """
    Builds price, log returns, and momentum columns.
    """
    # 1. Extract Price (Robust Fix)
    s_price = robust_close_series(df_raw)
    
    # 2. Alignment Fix (Trivial here as we only have assets, but must implement logic)
    df_assets = pd.DataFrame({'price': s_price})
    df_assets, _, _ = intersect_common_index(df_assets)
    
    if df_assets.empty:
        return pd.DataFrame()

    # 3. Compute Basic Returns
    df_assets['ret'] = np.log(df_assets['price'] / df_assets['price'].shift(1))
    
    # 4. Compute Momentum (3m=63, 6m=126, 12m=252)
    # Formula: price / price.shift(N) - 1
    df_assets['mom_3m'] = df_assets['price'] / df_assets['price'].shift(63) - 1
    df_assets['mom_6m'] = df_assets['price'] / df_assets['price'].shift(126) - 1
    df_assets['mom_12m'] = df_assets['price'] / df_assets['price'].shift(252) - 1
    
    # Round-trip
    path = output_dir / "momentum.csv"
    return strict_csv_roundtrip(df_assets, path)

def compute_weekly_reversal(df_mom, output_dir):
    """
    Calculates Past 5d return vs Next 5d return.
    """
    if df_mom.empty: return pd.DataFrame(), pd.DataFrame()

    df = df_mom[['ret']].copy()
    
    # past_5d = rolling(5).sum(ret)
    df['past_5d'] = df['ret'].rolling(5).sum()
    
    # next_5d = shift(-5).rolling(5).sum(ret)
    # Note: rolling is backward looking, so shift(-5) brings future data to current row, 
    # then rolling sums that future block.
    # Actually, specific requirement: next_5d = shift(-5).rolling(5).sum(ret)
    # This implies we look ahead 5 days.
    # To get sum of t+1 to t+5:
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    df['next_5d'] = df['ret'].rolling(window=indexer).sum().shift(-1) 
    # Alternatively, strictly following the prompt text: "shift(-5).rolling(5).sum(ret)"
    # If we shift(-5), row T gets row T+5. 
    # Let's interpret strictly as: The return over the next 5 days.
    # Implementation: Sum of ret[t+1]...ret[t+5].
    # Using shift(-5) directly on ret moves t+5 to t. 
    # Let's use the explicit math: (Price[t+5]/Price[t]) - 1 approx sum log ret
    # Prompt says: "shift(-5).rolling(5).sum(ret)"
    # Let's do exactly what is requested literally, though standard pandas rolling is looking back.
    # If we do df['ret'].shift(-5).rolling(5).sum(), at time T:
    # We have data from T+5. The rolling(5) sums T+5, T+4, T+3, T+2, T+1. 
    # This correctly sums the next 5 days returns.
    df['next_5d'] = df['ret'].shift(-5).rolling(5).sum()

    df = df.dropna()

    # Label: "Up" if past_5d > 0 else "Down"
    df['label'] = np.where(df['past_5d'] > 0, 'Up', 'Down')
    
    # Save intermediate
    strict_csv_roundtrip(df, output_dir / "reversal_intermediate.csv")
    
    # Summary Table
    # mean_next_5d, count, win_rate (share of next_5d > 0)
    summary = df.groupby('label')['next_5d'].agg(['mean', 'count'])
    summary['win_rate'] = df.groupby('label')['next_5d'].apply(lambda x: (x > 0).mean())
    
    # Round-trip summary
    path_summ = output_dir / "reversal_summary.csv"
    summary.to_csv(path_summ) # Index is Label (Up/Down), not Date, so just standard write
    # Read back standard
    summary_rt = pd.read_csv(path_summ, index_col=0)
    
    return df, summary_rt

def compute_tom(df_mom, output_dir):
    """
    Turn-of-Month: Last 5 trading days of month vs Rest.
    """
    if df_mom.empty: return pd.DataFrame(), pd.DataFrame()

    df = df_mom[['ret']].copy()
    
    # Identify last 5 trading days of each month
    # Group by Year-Month
    # We can use to_period('M')
    df['ym'] = df.index.to_period('M')
    
    # Mask for TOM days
    # For each group, get indices of last 5
    tom_indices = set()
    for _, group in df.groupby('ym'):
        if len(group) >= 5:
            tom_indices.update(group.index[-5:])
        else:
            tom_indices.update(group.index) # If month short, all are TOM
            
    df['is_tom'] = df.index.isin(tom_indices)
    
    # Save labels
    strict_csv_roundtrip(df, output_dir / "tom_labels.csv")
    
    # Summary: Mean(ret) for TOM vs Rest
    # 2-row summary table
    summary = df.groupby('is_tom')['ret'].mean().reset_index()
    summary['period'] = np.where(summary['is_tom'], 'TOM (Last 5 Days)', 'Rest of Month')
    summary = summary.set_index('period')[['ret']]
    summary.columns = ['mean_daily_ret']
    
    path_summ = output_dir / "tom_summary.csv"
    summary.to_csv(path_summ)
    summary_rt = pd.read_csv(path_summ, index_col=0)
    
    return df, summary_rt

def compute_pead(df_mom, df_earnings, output_dir):
    """
    PEAD: 20-day drift after earnings.
    """
    if df_mom.empty or df_earnings.empty:
        return pd.DataFrame()
        
    # Columns needed: Price
    # From df_mom which has 'price'
    prices = df_mom['price']
    
    results = []
    
    # Process earnings dates
    # Constraints: Force timezone-naive date; align to next available trading day
    # df_earnings index is typically the earnings date/time
    
    # Ensure earnings index is naive (already done by loader, but good to be safe)
    earnings_dates = df_earnings.index
    
    # Clean up column names for Surprise (yfinance varies)
    surprise_col = None
    for c in df_earnings.columns:
        if 'Surprise' in c:
            surprise_col = c
            break
            
    for edate in earnings_dates:
        # Align to next available trading day in prices
        # searchsorted finds the insertion point to maintain order
        # if edate is in prices, idx points to it. If not, points to next.
        idx_loc = prices.index.searchsorted(edate)
        
        if idx_loc >= len(prices):
            continue
            
        aligned_date = prices.index[idx_loc]
        
        # We need T+20 (20 trading days later)
        if idx_loc + 20 >= len(prices):
            continue
            
        price_t = prices.iloc[idx_loc]
        price_t20 = prices.iloc[idx_loc + 20]
        
        drift = (price_t20 / price_t) - 1
        
        surprise_val = 0.0
        if surprise_col:
            val = df_earnings.loc[edate, surprise_col]
            # Handle potential Series if duplicate index
            if isinstance(val, pd.Series): val = val.iloc[0]
            surprise_val = val if pd.notnull(val) else 0.0
            
        row = {
            'Date': aligned_date,
            'Surprise': surprise_val,
            'Type': 'Positive' if surprise_val > 0 else 'Negative',
            'Drift_20d': drift
        }
        results.append(row)
        
    if not results:
        return pd.DataFrame()
        
    pead_df = pd.DataFrame(results)
    pead_df = pead_df.set_index('Date') # Set index for CSV roundtrip compliance
    
    # Round-trip
    return strict_csv_roundtrip(pead_df, output_dir / "pead_events.csv")

def compute_stats_mode_b(tom_df, rev_df, pead_df, output_dir):
    """
    Mode B Stats using SciPy.
    """
    stats_res = []

    # 1. TOM T-test
    if not tom_df.empty and 'is_tom' in tom_df.columns:
        tom_ret = tom_df[tom_df['is_tom']]['ret'].dropna()
        rest_ret = tom_df[~tom_df['is_tom']]['ret'].dropna()
        if len(tom_ret) > 1 and len(rest_ret) > 1:
            t_stat, p_val = stats.ttest_ind(tom_ret, rest_ret, equal_var=False)
            stats_res.append({'Test': 'TOM vs Rest (Daily Ret)', 'Stat': t_stat, 'p-value': p_val})

    # 2. Reversal T-test (Up vs Down groups for next_5d)
    if not rev_df.empty:
        up_g = rev_df[rev_df['label'] == 'Up']['next_5d'].dropna()
        down_g = rev_df[rev_df['label'] == 'Down']['next_5d'].dropna()
        if len(up_g) > 1 and len(down_g) > 1:
            t_stat, p_val = stats.ttest_ind(up_g, down_g, equal_var=False)
            stats_res.append({'Test': 'Reversal (Up vs Down)', 'Stat': t_stat, 'p-value': p_val})

    # 3. PEAD Correlation
    if not pead_df.empty:
        # Surprise vs Drift
        # Clean data
        sub = pead_df[['Surprise', 'Drift_20d']].dropna()
        if len(sub) > 2:
            r, p_val = stats.pearsonr(sub['Surprise'], sub['Drift_20d'])
            stats_res.append({'Test': 'PEAD (Surprise vs Drift)', 'Stat': r, 'p-value': p_val})
            
    df_stats = pd.DataFrame(stats_res)
    if not df_stats.empty:
        df_stats.to_csv(output_dir / "stats_summary.csv", index=False)
        # Read back
        df_stats = pd.read_csv(output_dir / "stats_summary.csv")
        
    return df_stats

# ==========================================
# 4. Visualization
# ==========================================

def generate_dashboard(ticker, output_dir, df_mom, df_rev_summ, df_tom_summ, df_pead, df_stats, mode):
    """
    Generates Plotly HTML dashboard.
    """
    # Determine layout rows
    # Row 1: Price + Mom
    # Row 2: Reversal + TOM (2 columns)
    # Row 3: PEAD (if exists)
    # Row 4: Stats (if Mode B)
    
    rows = 2
    if not df_pead.empty: rows += 1
    if mode == 'B' and not df_stats.empty: rows += 1
    
    # Corrected specs: Ensure Row 1 has 2 elements (matching cols=2)
    # We span the first chart across both columns
    specs = [[{"secondary_y": True, "colspan": 2}, None]] # Row 1
    
    specs.append([{"type": "bar"}, {"type": "bar"}]) # Row 2
    if not df_pead.empty: specs.append([{"colspan": 2, "type": "scatter"}, None])
    if mode == 'B' and not df_stats.empty: specs.append([{"colspan": 2, "type": "table"}, None])
    
    fig = make_subplots(
        rows=rows, cols=2,
        specs=specs,
        subplot_titles=(
            f"{ticker} Price & Momentum", 
            "Weekly Reversal (Next 5d Return)", 
            "Turn-of-Month (Daily Return)",
            "PEAD: Surprise vs 20d Drift" if not df_pead.empty else None,
            "Statistical Tests" if (mode == 'B' and not df_stats.empty) else None
        ),
        vertical_spacing=0.1
    )
    
    # --- Figure 1: Price & Momentum ---
    if not df_mom.empty:
        # Price (Secondary Y)
        fig.add_trace(
            go.Scatter(x=df_mom.index, y=df_mom['price'], name="Price", line=dict(color='white', width=1)),
            row=1, col=1, secondary_y=True
        )
        # Momentum (Primary Y)
        fig.add_trace(
            go.Scatter(x=df_mom.index, y=df_mom['mom_3m'], name="Mom 3M", line=dict(color='cyan', width=1)),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df_mom.index, y=df_mom['mom_12m'], name="Mom 12M", line=dict(color='magenta', width=1)),
            row=1, col=1, secondary_y=False
        )

    # --- Figure 2a: Weekly Reversal ---
    if not df_rev_summ.empty:
        colors = ['#00FF00' if v > 0 else '#FF0000' for v in df_rev_summ['mean']]
        fig.add_trace(
            go.Bar(
                x=df_rev_summ.index, 
                y=df_rev_summ['mean'], 
                name="Reversal Next 5d",
                marker_color=colors
            ),
            row=2, col=1
        )

    # --- Figure 2b: TOM ---
    if not df_tom_summ.empty:
        colors = ['#00FF00' if v > 0 else '#FF0000' for v in df_tom_summ['mean_daily_ret']]
        fig.add_trace(
            go.Bar(
                x=df_tom_summ.index, 
                y=df_tom_summ['mean_daily_ret'], 
                name="TOM Mean Ret",
                marker_color=colors
            ),
            row=2, col=2
        )

    curr_row = 3
    
    # --- Figure 3: PEAD ---
    if not df_pead.empty:
        # Positive Surprise
        pos = df_pead[df_pead['Type'] == 'Positive']
        neg = df_pead[df_pead['Type'] == 'Negative']
        
        if not pos.empty:
            fig.add_trace(
                go.Scatter(
                    x=pos.index, y=pos['Drift_20d'], 
                    mode='markers', name='Pos Surprise',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=curr_row, col=1
            )
        if not neg.empty:
            fig.add_trace(
                go.Scatter(
                    x=neg.index, y=neg['Drift_20d'], 
                    mode='markers', name='Neg Surprise',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=curr_row, col=1
            )
        
        # Zero line
        fig.add_hline(y=0, row=curr_row, col=1, line_dash="dash", line_color="gray")
        curr_row += 1

    # --- Figure 4: Stats Table (Mode B) ---
    if mode == 'B' and not df_stats.empty:
        fig.add_trace(
            go.Table(
                header=dict(values=list(df_stats.columns), fill_color='grey', align='left'),
                cells=dict(values=[df_stats[k].tolist() for k in df_stats.columns], fill_color='darkgrey', align='left')
            ),
            row=curr_row, col=1
        )
        
    # Layout Update
    fig.update_layout(
        template='plotly_dark',
        height=300 * rows,
        title_text=f"Anomalies Dashboard: {ticker}",
        showlegend=True
    )
    
    out_path = output_dir / f"{ticker}_dashboard.html"
    fig.write_html(str(out_path))
    return out_path

# ==========================================
# 5. Main Logic
# ==========================================

def process_ticker(ticker, mode, user_out):
    # Setup Output
    out_dir = get_output_dir(user_out, ticker)
    print(f"\nProcessing {ticker} -> {out_dir}")
    
    # 1. Download Price
    df_raw = load_prices_one_ticker(ticker, out_dir)
    if df_raw.empty:
        return
        
    # 2. Compute Momentum (includes Alignment Fix)
    df_mom = compute_basics_and_momentum(df_raw, out_dir)
    if df_mom.empty:
        print("Error in momentum computation (empty result).")
        return

    # 3. Weekly Reversal
    df_rev_inter, df_rev_summ = compute_weekly_reversal(df_mom, out_dir)
    
    # 4. Turn of Month
    df_tom_labels, df_tom_summ = compute_tom(df_mom, out_dir)
    
    # 5. PEAD (Optional but requested)
    df_pead = pd.DataFrame()
    try:
        df_earnings = load_earnings_dates(ticker, out_dir)
        if not df_earnings.empty:
            df_pead = compute_pead(df_mom, df_earnings, out_dir)
    except Exception as e:
        print(f"PEAD Processing failed: {e}")

    # 6. Stats (Mode B)
    df_stats = pd.DataFrame()
    if mode == 'B':
        df_stats = compute_stats_mode_b(df_tom_labels, df_rev_inter, df_pead, out_dir)
        
    # 7. Visualize
    try:
        html_path = generate_dashboard(ticker, out_dir, df_mom, df_rev_summ, df_tom_summ, df_pead, df_stats, mode)
        print(f"Dashboard generated: {html_path}")
    except Exception as e:
        print(f"Visualization failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Equity Anomalies CLI")
    parser.add_argument('tickers', nargs='+', help='List of tickers')
    parser.add_argument('--mode', choices=['A', 'B', 'C'], default='A', help='Analysis mode')
    parser.add_argument('--out', help='Output directory root')
    
    args = parser.parse_args()
    
    # Expand comma-separated tickers if user did "AAPL,MSFT" instead of "AAPL MSFT"
    tickers = []
    for t in args.tickers:
        tickers.extend(t.split(','))
    
    for ticker in tickers:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker: continue
        try:
            process_ticker(clean_ticker, args.mode, args.out)
        except Exception as e:
            print(f"CRITICAL ERROR processing {clean_ticker}: {e}")
            # Continue to next ticker, do not crash full script
            
    sys.exit(0)

if __name__ == '__main__':
    main()
