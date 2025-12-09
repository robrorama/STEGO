#!/usr/bin/env python3
# SCRIPTNAME: projectiles.v5.with.inflection.point.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.stats import linregress
from datetime import datetime

# Centralized Data Retrieval
try:
    import data_retrieval as dr
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# --- Technical Analysis and Physics-Based Functions ---

def add_linear_regression_bands(df):
    """Adds linear regression channel bands to the DataFrame."""
    df_numeric_index = np.arange(len(df.index))
    slope, intercept, _, _, _ = linregress(df_numeric_index, df['Close'])
    
    df['Linear_Reg'] = intercept + slope * df_numeric_index
    df['Residuals'] = df['Close'] - df['Linear_Reg']
    residuals_std = df['Residuals'].std()

    desired_values = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    for i, num_std in enumerate(desired_values):
        df[f'Reg_High_{i+1}std'] = df['Linear_Reg'] + residuals_std * num_std
        df[f'Reg_Low_{i+1}std'] = df['Linear_Reg'] - residuals_std * num_std
    return df

def add_ema(df, time_periods):
    """Adds Exponential Moving Averages to the DataFrame."""
    for time_period in time_periods:
        df[f'EMA_{time_period}'] = df['Close'].ewm(span=time_period, adjust=False).mean()
    return df

def analyze_projectile_trajectory(df, lookback_days=90):
    """
    Analyzes the recent price trend to find a launch point, trajectory, and peak.
    Model: y = ax^2 + bx + c
    """
    if len(df) < lookback_days: return None, None, None

    recent_df = df.iloc[-lookback_days:]
    launch_date = recent_df['Low'].idxmin()
    launch_price = recent_df.loc[launch_date, 'Low']
    launch_info = {'date': launch_date, 'price': launch_price}
    
    trajectory_df = df.loc[launch_date:]
    
    if len(trajectory_df) < 3: return None, None, None

    x_numeric = np.arange(len(trajectory_df))
    y_prices = trajectory_df['High'].values
    coeffs = np.polyfit(x_numeric, y_prices, 2)
    a, b, c = coeffs

    # Physics check: 'a' must be negative for a downward-curving parabola (projectile gravity)
    if a >= 0:
        print("Trajectory is not parabolic (upward-opening or linear). No peak forecasted.")
        return None, None, None

    peak_x_numeric = -b / (2 * a) 
    peak_y_price = a * (peak_x_numeric**2) + b * peak_x_numeric + c

    # Calculate estimated date of peak
    time_delta = df.index.to_series().diff().mean()
    if pd.isna(time_delta): time_delta = pd.Timedelta(days=1)
        
    peak_date = launch_date + (time_delta * peak_x_numeric)
    
    # Generate curve points for plotting
    plot_x_numeric = np.arange(int(peak_x_numeric) + 20) # Extend plot slightly further
    parabola_y = a * (plot_x_numeric**2) + b * plot_x_numeric + c
    parabola_dates = [launch_date + (time_delta * i) for i in plot_x_numeric]

    trajectory_curve = pd.Series(parabola_y, index=parabola_dates)
    peak_info = {'date': peak_date, 'price': peak_y_price}
    
    print(f"Projectile Analysis Found Inflection: Date={launch_date.strftime('%Y-%m-%d')}, Price={launch_price:.2f}")
    print(f"Projectile Analysis Found Peak: Date={peak_date.strftime('%Y-%m-%d')}, Price={peak_y_price:.2f}")

    return trajectory_curve, peak_info, launch_info

# --- Plotting Function ---

def plot_data(df, ticker, lookback, show_plot=True):
    """Plots all data including candlesticks, LRC, EMAs, and trajectory."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Market Data'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Reg'], line=dict(color='blue', width=2), name='Linear Regression'))
    
    colors = ['grey', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']
    desired_values = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    
    for i, num_std in enumerate(desired_values):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_High_{i+1}std'], line=dict(color=colors[i], width=1, dash='dot'), name=f'Reg High {num_std} std'))
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_Low_{i+1}std'], line=dict(color=colors[i], width=1, dash='dot'), name=f'Reg Low {num_std} std'))

    ema_colors = ['purple', 'orange', 'green', 'red', 'blue']
    ema_periods = [20, 50, 100, 200, 300]
    for i, time_period in enumerate(ema_periods):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{time_period}'], line=dict(color=ema_colors[i], width=2), name=f'{time_period}-day EMA'))

    trajectory_curve, peak_info, launch_info = analyze_projectile_trajectory(df, lookback_days=lookback)
    
    if trajectory_curve is not None:
        fig.add_trace(go.Scatter(x=trajectory_curve.index, y=trajectory_curve.values, mode='lines', name='Price Trajectory', line=dict(color='magenta', width=3, dash='dash')))
        
        # Annotation for the Predicted Apex
        fig.add_annotation(x=peak_info['date'], y=peak_info['price'], text=f"Predicted Apex<br>${peak_info['price']:.2f}", showarrow=True, arrowhead=4, arrowwidth=2, ax=0, ay=-60, font=dict(size=12, color="white"), align="center", bgcolor="rgba(138, 43, 226, 0.7)")
        
        # Annotation for the Inflection Point
        fig.add_annotation(x=launch_info['date'], y=launch_info['price'], text=f"Inflection Point<br>${launch_info['price']:.2f}", showarrow=True, arrowhead=4, arrowwidth=2, ax=0, ay=60, font=dict(size=12, color="white"), align="center", bgcolor="rgba(0, 128, 0, 0.7)")

    today_str = datetime.today().strftime("%Y-%m-%d")
    fig.update_layout(
        title=f"{ticker} Analysis ({today_str}): LRC, EMAs, and Projectile Trajectory", 
        xaxis_title='Date', yaxis_title='Price', 
        height=800, width=1500, 
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    # Use canonical output directory via data_retrieval
    out_dir = dr.create_output_directory(ticker)
    image_file = os.path.join(out_dir, f'{ticker}_projectile_analysis.png')
    
    try:
        fig.write_image(image_file, width=1920, height=1080, scale=2)
        print(f"Plot saved to {image_file}")
    except Exception as e:
        print(f"Warning: Could not save static image (requires kaleido). Error: {e}")
        
    html_file = os.path.join(out_dir, f'{ticker}_projectile_analysis.html')
    fig.write_html(html_file)
    print(f"Interactive chart saved to {html_file}")

    if show_plot:
        fig.show()

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Projectile Trajectory Price Analyzer")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. NVDA)")
    parser.add_argument("--period", default="2y", help="Data period (e.g. 1y, 2y, max). Default: 2y")
    parser.add_argument("--lookback", type=int, default=90, help="Lookback days to find the 'launch' low point. Default: 90")
    parser.add_argument("--no-show", action="store_true", help="Do not open browser tab")
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    # CONSTRAINT: Use data_retrieval to load data
    df = dr.load_or_download_ticker(ticker, period=args.period)
    
    if df is None or df.empty:
        print(f"Error: No data found for {ticker}")
        sys.exit(1)
        
    df = add_linear_regression_bands(df)
    df = add_ema(df, [20, 50, 100, 200, 300])
    
    plot_data(df, ticker, args.lookback, not args.no_show)

if __name__ == '__main__':
    main()
