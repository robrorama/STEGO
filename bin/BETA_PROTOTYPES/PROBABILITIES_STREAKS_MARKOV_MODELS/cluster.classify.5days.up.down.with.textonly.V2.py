#!/usr/bin/env python3
# SCRIPTNAME: ok.cluster.classify.5days.up.down.with.textonly.V2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

# classify_5days_up_down_with_download.py

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import glob
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import get_stock_data, create_output_directory
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# Configuration
TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B']
PERIOD = '2y'  # Download 2 years of data

# CONSTRAINT: Use /dev/shm via data_retrieval
# We create one shared directory for this clustering run
OUTPUT_DIR = create_output_directory("CLUSTER_5DAY_DIRECTION")

def compute_5day_directions(df):
    """
    Compute 5-day price change directions.
    Returns a series of direction strings based on 5-day returns.
    """
    if 'Close' not in df.columns:
        print("Warning: 'Close' column not found")
        return pd.Series(dtype=object)
    
    # Calculate 5-day returns
    returns_5d = df['Close'].pct_change(5)
    
    # Define direction bins (you can adjust these thresholds)
    # For example: strong down, down, flat, up, strong up
    bins = [-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf]
    labels = ['-2', '-1', '0', '1', '2']  # Direction labels
    
    directions = pd.cut(returns_5d, bins=bins, labels=labels)
    return directions

def compute_transition_matrix(directions):
    """
    Compute transition matrix from direction series.
    Returns a DataFrame representing transition probabilities.
    """
    # Get unique states
    states = sorted(directions.dropna().unique())
    
    # Initialize transition matrix
    transition_matrix = pd.DataFrame(0, index=states, columns=states)
    
    # Count transitions
    for i in range(len(directions) - 1):
        if pd.notna(directions.iloc[i]) and pd.notna(directions.iloc[i + 1]):
            from_state = directions.iloc[i]
            to_state = directions.iloc[i + 1]
            transition_matrix.loc[from_state, to_state] += 1
    
    # Convert to probabilities
    for state in states:
        row_sum = transition_matrix.loc[state].sum()
        if row_sum > 0:
            transition_matrix.loc[state] = transition_matrix.loc[state] / row_sum
    
    return transition_matrix

def download_and_compute_matrices(out_dir):
    """Download data and compute direction-based transition matrices for all tickers."""
    matrices = {}
    
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        
        # Download data using data_retrieval.py (checks cache/downloads)
        df = get_stock_data(ticker, period=PERIOD)
        
        if df is None or df.empty:
            print(f"No data available for {ticker}")
            continue
        
        # Compute 5-day directions
        directions = compute_5day_directions(df)
        
        if directions.empty:
            print(f"Could not compute directions for {ticker}")
            continue
        
        # Compute transition matrix
        transition_matrix = compute_transition_matrix(directions)
        
        # Save to CSV in /dev/shm
        output_path = os.path.join(out_dir, f"{ticker}_direction_transition_matrix.csv")
        transition_matrix.to_csv(output_path)
        print(f"Saved transition matrix for {ticker} to {output_path}")
        
        matrices[ticker] = transition_matrix
    
    return matrices

def load_transition_matrices(folder):
    """Load all CSV direction-based transition matrices into {ticker: DataFrame}."""
    matrices = {}
    # Only look for files we just created in the specific output folder
    pattern = os.path.join(folder, "*_direction_transition_matrix.csv")
    for file in glob.glob(pattern):
        ticker = os.path.basename(file).split('_direction_transition_matrix.csv')[0]
        df = pd.read_csv(file, index_col=0)
        matrices[ticker] = df
    return matrices

def unify_matrix_shapes(matrices):
    """Ensure all matrices share the same row/column indices by taking their union."""
    all_states = set()
    for df in matrices.values():
        all_states.update(df.index.astype(str))
        all_states.update(df.columns.astype(str))

    # Sorted for consistent ordering
    # Handle cases where labels might be read as numbers or strings
    try:
        all_states = sorted(list(all_states), key=lambda x: float(x))
    except ValueError:
        all_states = sorted(list(all_states))

    unified = {}
    for ticker, df in matrices.items():
        # Reindex with fill_value=0
        # Ensure index/cols are strings to match all_states if mixed types occurred
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        
        df_unified = df.reindex(index=all_states, columns=all_states, fill_value=0.0)
        unified[ticker] = df_unified
    return unified

def flatten_matrix(df):
    """Flatten transition matrix to 1D array."""
    return df.values.flatten()

def cluster_matrices(matrices, n_clusters=3):
    """Cluster direction-based transition matrices into n_clusters."""
    # 1) unify shapes
    unified = unify_matrix_shapes(matrices)

    # 2) Flatten
    flattened = [flatten_matrix(m) for m in unified.values()]
    X = np.array(flattened)

    # 3) KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # 4) Map ticker -> cluster
    return dict(zip(unified.keys(), labels))

if __name__ == "__main__":
    print(f"Output Directory: {OUTPUT_DIR}")

    # Step 1: Download data and compute transition matrices
    print("Step 1: Downloading data and computing transition matrices...")
    download_and_compute_matrices(OUTPUT_DIR)
    
    # Step 2: Load direction-based transition matrices from disk
    print("\nStep 2: Loading transition matrices from disk...")
    matrices = load_transition_matrices(OUTPUT_DIR)
    
    if not matrices:
        print("No transition matrices found!")
        exit(1)
    
    # Step 3: Cluster them
    print(f"\nStep 3: Clustering {len(matrices)} matrices...")
    assignments = cluster_matrices(matrices, n_clusters=3)

    # Print results
    print("\nTicker direction-based classification:")
    for ticker, label in sorted(assignments.items()):
        print(f"  {ticker} -> Cluster {label}")
