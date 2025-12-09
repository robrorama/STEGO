#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SCRIPTNAME: ok.markhov.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
states_unified.py

Integrates functionality from first.py and second.py while routing all data I/O
through data_retrieval.py.

Usage:
  python3 states_unified.py <TICKER> [PERIOD]

Args:
  <TICKER>  e.g., NVDA
  [PERIOD]  yfinance-style period string for history (default: 'max'), e.g.:
            '1y', '2y', '5y', '10y', 'max'

Outputs:
  - <output_dir>/<TICKER>_price_indicators.html
  - <output_dir>/<TICKER>_state_heatmaps.html
  Both are opened in separate browser tabs.
"""

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import webbrowser
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot as offline_plot

# --- strict data layer: use ONLY data_retrieval.py ---
# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr  # get_stock_data(), create_output_directory()
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)


# ----------------------------- Indicators & States -----------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 200-SMA and distance
    df['200_SMA'] = df['Close'].rolling(window=200).mean()
    df['Distance_200_SMA'] = df['Close'] - df['200_SMA']

    # 4/50-SMA and their slopes
    df['4_SMA'] = df['Close'].rolling(window=4).mean()
    df['50_SMA'] = df['Close'].rolling(window=50).mean()
    df['4_SMA_Slope'] = df['4_SMA'].diff()
    df['50_SMA_Slope'] = df['50_SMA'].diff()

    # Bollinger(20, ±2σ) and distances
    df['20_SMA'] = df['Close'].rolling(window=20).mean()
    df['20_STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Bollinger'] = df['20_SMA'] + 2 * df['20_STD']
    df['Lower_Bollinger'] = df['20_SMA'] - 2 * df['20_STD']
    df['Distance_Upper_Bollinger'] = df['Close'] - df['Upper_Bollinger']
    df['Distance_Lower_Bollinger'] = df['Close'] - df['Lower_Bollinger']

    # Returns, signs, contiguous up/down counter
    df['Daily_Return'] = df['Close'].pct_change()
    df['Up_Down'] = np.sign(df['Daily_Return'])
    grp = (df['Up_Down'] != df['Up_Down'].shift()).cumsum()
    df['Contiguous_Up_Down'] = df['Up_Down'].groupby(grp).cumsum()

    # Derivatives
    df['Derivative_1'] = df['Close'].diff()
    df['Derivative_2'] = df['Derivative_1'].diff()
    df['Derivative_3'] = df['Derivative_2'].diff()

    return df.dropna()


def categorize_states(df: pd.DataFrame) -> pd.DataFrame:
    df['State_200_SMA'] = np.where(df['Distance_200_SMA'] > 0, 'Above_200', 'Below_200')

    df['State_Consecutive_Up_Down'] = np.where(
        df['Contiguous_Up_Down'] > 0,
        np.where(df['Contiguous_Up_Down'] == 1, 'Consecutive_Up_1', 'Consecutive_Up_2+'),
        np.where(df['Contiguous_Up_Down'] == -1, 'Consecutive_Down_1', 'Consecutive_Down_2+')
    )

    df['State_4_SMA_Slope']  = np.where(df['4_SMA_Slope']  > 0, '4_SMA_Up',  '4_SMA_Down')
    df['State_50_SMA_Slope'] = np.where(df['50_SMA_Slope'] > 0, '50_SMA_Up', '50_SMA_Down')

    df['State_Bollinger'] = np.where(
        df['Distance_Upper_Bollinger'] > 0, 'Above_Upper',
        np.where(df['Distance_Lower_Bollinger'] < 0, 'Below_Lower', 'Inside_Bollinger')
    )

    df['State_Derivative_1'] = np.where(df['Derivative_1'] > 0, 'Positive_Deriv_1', 'Negative_Deriv_1')
    df['State_Derivative_2'] = np.where(df['Derivative_2'] > 0, 'Positive_Deriv_2', 'Negative_Deriv_2')
    df['State_Derivative_3'] = np.where(df['Derivative_3'] > 0, 'Positive_Deriv_3', 'Negative_Deriv_3')

    return df


def calculate_transition_probabilities(df: pd.DataFrame, state_columns):
    """
    Returns dict: {state_col: DataFrame probs (prev_state -> next_state)} and prints each.
    """
    out = {}
    for col in state_columns:
        ctab = pd.crosstab(df[col].shift(), df[col], normalize='index').fillna(0.0)
        out[col] = ctab
        print(f"\nTransition Probabilities for {col}:\n{ctab}")
    return out


def make_prediction(df: pd.DataFrame, transition_probs: dict, current_index: int):
    """
    Example (from second.py): considers probability of moving to 'Below_200'.
    """
    if current_index < 1:
        return "Not enough data for prediction."

    curr_state = df['State_200_SMA'].iloc[current_index - 1]
    tbl = transition_probs.get('State_200_SMA')
    if tbl is None or curr_state not in tbl.index:
        return "Prediction unavailable."

    prob_below = float(tbl.loc[curr_state].get('Below_200', 0.0))
    return "Likely to move below 200 SMA." if prob_below > 0.6 else "Likely to stay above 200 SMA."


# ----------------------------------- Plots -------------------------------------

def create_price_indicators_figure(ticker: str, df: pd.DataFrame, out_dir: str) -> str:
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=(f"{ticker} Price and Technical Indicators",))

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'), row=1, col=1)
    for w in [200, 50, 20, 4]:
        col = f'{w}_SMA'
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'{w} SMA'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Bollinger'], mode='lines', name='Upper Bollinger'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Bollinger'], mode='lines', name='Lower Bollinger'), row=1, col=1)

    fig.update_layout(title=f"{ticker} Price and Technical Indicators",
                      height=700, width=1200,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      xaxis_rangeslider_visible=False, template='plotly_dark')
    fig.update_yaxes(title_text="Price", row=1, col=1)

    out_path = os.path.join(out_dir, f"{ticker}_price_indicators.html")
    offline_plot(fig, filename=out_path, auto_open=False)
    return out_path


def create_transition_heatmaps_figure(ticker: str,
                                      transition_probs: dict,
                                      out_dir: str) -> str:
    keys = list(transition_probs.keys())
    n = len(keys)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=tuple(keys),
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    r = c = 1
    for k in keys:
        tbl = transition_probs[k].copy()  # rows: prev, cols: next
        # Ensure square and consistent category order
        cats = sorted(set(tbl.index.tolist()) | set(tbl.columns.tolist()))
        tbl = tbl.reindex(index=cats, columns=cats, fill_value=0.0)

        fig.add_trace(
            go.Heatmap(z=tbl.values, x=tbl.columns, y=tbl.index,
                       zmin=0.0, zmax=1.0, colorbar=dict(title="P")),
            row=r, col=c
        )

        c += 1
        if c > ncols:
            c = 1
            r += 1

    fig.update_layout(title=f"{ticker} Markov State Transition Probabilities",
                      height=400 * nrows, width=1200,
                      template='plotly_dark')

    out_path = os.path.join(out_dir, f"{ticker}_state_heatmaps.html")
    offline_plot(fig, filename=out_path, auto_open=False)
    return out_path


# ----------------------------------- Main --------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 states_unified.py <TICKER> [PERIOD]")
        sys.exit(1)

    ticker = sys.argv[1].strip().upper()
    period = sys.argv[2] if len(sys.argv) > 2 else 'max'

    # CONSTRAINT: Output to /dev/shm via data_retrieval logic
    out_dir = dr.create_output_directory(ticker)
    # CONSTRAINT: Use data_retrieval logic
    df = dr.get_stock_data(ticker, period=period)

    required = {'Open','High','Low','Close'}
    if df.empty or not required.issubset(df.columns):
        print(f"Data not available or missing OHLC columns for {ticker}.")
        sys.exit(1)

    df = add_technical_indicators(df)
    df = categorize_states(df)

    state_columns = [
        'State_200_SMA',
        'State_Consecutive_Up_Down',
        'State_4_SMA_Slope',
        'State_50_SMA_Slope',
        'State_Bollinger',
        'State_Derivative_1',
        'State_Derivative_2',
        'State_Derivative_3'
    ]

    transition_probs = calculate_transition_probabilities(df, state_columns)

    # Example prediction (as in second.py)
    current_index = len(df) - 1
    prediction = make_prediction(df, transition_probs, current_index)
    print(f"\nPrediction: {prediction}")

    # Plots -> separate tabs
    price_html = create_price_indicators_figure(ticker, df, out_dir)
    heat_html  = create_transition_heatmaps_figure(ticker, transition_probs, out_dir)

    webbrowser.open_new_tab(f"file://{price_html}")
    webbrowser.open_new_tab(f"file://{heat_html}")

    print(f"\nSaved:")
    print(f"  {price_html}")
    print(f"  {heat_html}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
