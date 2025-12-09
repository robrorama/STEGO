#!/usr/bin/env python3
"""
Visualize stock gaps and downstream probabilities using data saved+reloaded from daily CSV via data_retrieval.py.
This script loads daily price data that has been saved and re‑loaded through
`data_retrieval.py`, detects overnight price gaps, and computes the probability distribution
of subsequent price movements for each gap size. It then produces visualizations (heat‑maps,
bar charts, etc.) that show how different gap magnitudes relate to downstream return
probabilities.
"""

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# CONSTRAINT: Import local data retrieval module
try:
    from data_retrieval import (
        load_or_download_ticker,
        create_output_directory,
        fix_yfinance_dataframe,
    )
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

def retrieve_to_csv_and_reload(ticker: str, period: str = 'max') -> pd.DataFrame:
    df = load_or_download_ticker(ticker, period=period)
    if df is None or df.empty:
        return pd.DataFrame()

    df = fix_yfinance_dataframe(df)
    if 'Date' in df.columns:
        to_save = df.copy()
    else:
        idx_name = df.index.name or 'Date'
        to_save = df.reset_index().rename(columns={idx_name: 'Date'})
    
    # CONSTRAINT: Output to /dev/shm via data_retrieval logic
    out_dir = create_output_directory(ticker)
    out_csv = os.path.join(out_dir, f"{ticker}.csv")
    to_save.to_csv(out_csv, index=False)
    
    re = pd.read_csv(out_csv, parse_dates=['Date'])
    re.sort_values('Date', inplace=True)
    re.set_index('Date', inplace=True)
    return re

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    if not {'Open', 'Close'}.issubset(df.columns):
        return pd.DataFrame()
    df = df.sort_index()
    df['Pct_Change'] = df['Close'].pct_change()
    prev_close = df['Close'].shift(1)
    df['Gap_Direction'] = np.where(df['Open'] > prev_close, 1, np.where(df['Open'] < prev_close, -1, 0))
    df.dropna(subset=['Pct_Change'], inplace=True)
    return df

def probability_after_gap(df: pd.DataFrame, days_ahead: int = 5) -> dict:
    gap_up_higher = gap_up_lower = gap_down_higher = gap_down_lower = 0
    total_gap_up = total_gap_down = 0
    limit = len(df) - days_ahead
    for idx in range(limit):
        current_gap = df['Gap_Direction'].iloc[idx]
        future_return = df['Pct_Change'].iloc[idx + days_ahead]
        if current_gap == 1:
            total_gap_up += 1
            if future_return > 0: gap_up_higher += 1
            elif future_return < 0: gap_up_lower += 1
        elif current_gap == -1:
            total_gap_down += 1
            if future_return > 0: gap_down_higher += 1
            elif future_return < 0: gap_down_lower += 1
    return {
        'gap_up_higher_prob': (gap_up_higher / total_gap_up) if total_gap_up else 0.0,
        'gap_up_lower_prob': (gap_up_lower / total_gap_up) if total_gap_up else 0.0,
        'gap_down_higher_prob': (gap_down_higher / total_gap_down) if total_gap_down else 0.0,
        'gap_down_lower_prob': (gap_down_lower / total_gap_down) if total_gap_down else 0.0,
    }

def visualize_ticker(ticker: str, df: pd.DataFrame, days_ahead: int, out_dir: str) -> None:
    # 1. Candlestick
    fig0 = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC")])
    fig0.update_layout(title=f'{ticker}: Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    fig0.write_html(os.path.join(out_dir, f"{ticker}_candlestick.html"))
    fig0.show()

    # 2. Gaps Scatter
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    gap_up = df[df['Gap_Direction'] == 1]
    gap_down = df[df['Gap_Direction'] == -1]
    fig1.add_trace(go.Scatter(x=gap_up.index, y=gap_up['Close'], mode='markers', name='Gap Up', marker=dict(symbol='triangle-up', size=10)))
    fig1.add_trace(go.Scatter(x=gap_down.index, y=gap_down['Close'], mode='markers', name='Gap Down', marker=dict(symbol='triangle-down', size=10)))
    fig1.update_layout(title=f'{ticker}: Close Price with Gap Events Highlighted', xaxis_title='Date', yaxis_title='Close ($)', hovermode='x unified', template='plotly_white', legend=dict(x=0, y=1.0))
    fig1.write_html(os.path.join(out_dir, f"{ticker}_gaps_scatter.html"))
    fig1.show()

    # 3. Distribution
    fig2 = px.histogram(df, x='Pct_Change', nbins=50, title=f'{ticker}: Distribution of Daily % Changes', labels={'Pct_Change': 'Daily % Change'}, opacity=0.75, marginal="box", hover_data=["Pct_Change"])
    fig2.update_layout(xaxis_title='Daily % Change', yaxis_title='Frequency', template='plotly_white')
    fig2.write_html(os.path.join(out_dir, f"{ticker}_dist_pct_change.html"))
    fig2.show()

    # 4. Probabilities
    results = probability_after_gap(df, days_ahead=days_ahead)
    prob_df = pd.DataFrame({
        'Condition': ['Gap Up Higher', 'Gap Up Lower', 'Gap Down Higher', 'Gap Down Lower'],
        'Probability': [results['gap_up_higher_prob'], results['gap_up_lower_prob'], results['gap_down_higher_prob'], results['gap_down_lower_prob']]
    })
    fig3 = px.bar(prob_df, x='Condition', y='Probability', title=f'{ticker}: Probability {days_ahead} Days After Gap', labels={'Probability': 'Probability'}, text=prob_df['Probability'].apply(lambda x: f"{x:.2f}"))
    fig3.update_traces(textposition='outside', marker=dict(line=dict(width=0.5)))
    fig3.update_layout(yaxis=dict(range=[0, 1]), xaxis_title='Condition', yaxis_title='Probability', uniformtext_minsize=8, uniformtext_mode='hide', template='plotly_white')
    fig3.write_html(os.path.join(out_dir, f"{ticker}_gap_probabilities.html"))
    fig3.show()
    
    print(f"Charts saved to: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize stock data with gap events using daily saved+reloaded CSV.")
    parser.add_argument('ticker', type=str, help='Ticker symbol (e.g., SPY)')
    parser.add_argument('--period', type=str, default='max', help='yfinance period (e.g., 1y, 5y, max).')
    parser.add_argument('--days_ahead', type=int, default=5, help='Number of days ahead to analyze after a gap event.')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    df_raw = retrieve_to_csv_and_reload(ticker, period=args.period)
    if df_raw is None or df_raw.empty:
        print(f"Error: No data returned for '{ticker}'.")
        sys.exit(1)

    df = preprocess_data(df_raw)
    if df.empty:
        print(f"Error: No valid data available for ticker '{ticker}' after preprocessing.")
        sys.exit(1)

    # Get canonical output dir to save plots
    out_dir = create_output_directory(ticker)
    visualize_ticker(ticker, df, days_ahead=args.days_ahead, out_dir=out_dir)

if __name__ == "__main__":
    main()
