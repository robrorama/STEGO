#!/usr/bin/env python3
# SCRIPTNAME: signals.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

def calculate_bollinger_bands(df, window=20):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['Upper Band'] = df['SMA'] + (2 * df['Close'].rolling(window=window).std())
    df['Lower Band'] = df['SMA'] - (2 * df['Close'].rolling(window=window).std())
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper_1std'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 1)
    df['BB_Upper_2std'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower_1std'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 1)
    df['BB_Lower_2std'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 2)
    
    return df

def add_bollinger_band_markers(fig, df):
    upper_pierce_x = []
    upper_pierce_y = []
    lower_pierce_x = []
    lower_pierce_y = []

    for i in range(len(df)):
        if (pd.notna(df['BB_Upper_2std'].iloc[i]) 
            and df['High'].iloc[i] > df['BB_Upper_2std'].iloc[i]):
            upper_pierce_x.append(df.index[i])
            upper_pierce_y.append(df['High'].iloc[i])
        elif (pd.notna(df['BB_Lower_2std'].iloc[i]) 
              and df['Low'].iloc[i] < df['BB_Lower_2std'].iloc[i]):
            lower_pierce_x.append(df.index[i])
            lower_pierce_y.append(df['Low'].iloc[i])

    if upper_pierce_x:
        fig.add_trace(go.Scatter(
            x=upper_pierce_x,
            y=upper_pierce_y,
            mode='markers',
            marker=dict(symbol='square-open', size=10, color='black'),
            name='Pierces Upper Band'
        ))

    if lower_pierce_x:
        fig.add_trace(go.Scatter(
            x=lower_pierce_x,
            y=lower_pierce_y,
            mode='markers',
            marker=dict(symbol='square-open', size=10, color='black'),
            name='Pierces Lower Band'
        ))

    return fig

def detect_volume_price_spikes(df, volume_std_threshold=1.5, price_std_threshold=2, rolling_window=20):
    df['VolumeChange'] = df['Volume'].diff()
    df['PriceChange'] = df['High'] - df['Low']

    df['VolumeMean'] = df['VolumeChange'].rolling(window=rolling_window).mean()
    df['VolumeStd'] = df['VolumeChange'].rolling(window=rolling_window).std()
    df['PriceMean'] = df['PriceChange'].rolling(window=rolling_window).mean()
    df['PriceStd'] = df['PriceChange'].rolling(window=rolling_window).std()

    spike_days = []
    for i in range(rolling_window, len(df)):
        if (abs(df['VolumeChange'].iloc[i] - df['VolumeMean'].iloc[i]) 
            > volume_std_threshold * df['VolumeStd'].iloc[i] 
            and abs(df['PriceChange'].iloc[i] - df['PriceMean'].iloc[i]) 
            > price_std_threshold * df['PriceStd'].iloc[i]):
            spike_days.append((df.index[i], df['Close'].iloc[i]))

    return spike_days

def detect_signals(df):
    buy_signals = []
    sell_signals = []
    
    for i in range(1, len(df) - 1):
        if df['Close'].iloc[i] > df['Open'].iloc[i] and df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            buy_signals.append((df.index[i], df['Close'].iloc[i]))
        elif df['Close'].iloc[i] < df['Open'].iloc[i] and df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            sell_signals.append((df.index[i], df['Close'].iloc[i]))

    return buy_signals, sell_signals

def calculate_fibonacci_levels(df):
    high_price = df['High'].max()
    low_price = df['Low'].min()
    diff = high_price - low_price
    levels = [
        high_price - diff * ratio 
        for ratio in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    ]
    fib_levels = pd.DataFrame(data=levels, columns=['Fibonacci Levels'])
    return fib_levels, high_price, low_price

def detect_wick_touches(df, deviations, len_regression):
    wick_touches = []
    touched_devs = set()

    if len(df) < 2 or len_regression is None or len_regression <= 1:
        return wick_touches, touched_devs

    L = min(int(len_regression), len(df))

    for i in range(-L + 1, 0):
        high_touched = False
        low_touched = False

        for dev, prices in deviations.items():
            if not isinstance(prices, (list, tuple, np.ndarray)): continue
            if len(prices) != len_regression: continue

            price = prices[i + len_regression - 1]

            try:
                if (not high_touched and df['High'].iloc[i] >= price and df['High'].iloc[i - 1] < price):
                    wick_touches.append((df.index[i], (dev, price)))
                    touched_devs.add(dev)
                    high_touched = True
            except IndexError:
                pass

            try:
                if (not low_touched and df['Low'].iloc[i] <= price and df['Low'].iloc[i - 1] > price):
                    wick_touches.append((df.index[i], (dev, price)))
                    touched_devs.add(dev)
                    low_touched = True
            except IndexError:
                pass

            if high_touched and low_touched:
                break

    return wick_touches, touched_devs

def detect_fib_wick_touches(df, fib_levels):
    fib_wick_touches = []
    touched_fibs = set()
    
    for i in range(len(df)):
        for level in fib_levels['Fibonacci Levels']:
            if df['High'].iloc[i] >= level and df['Low'].iloc[i] <= level:
                fib_wick_touches.append((df.index[i], (round(level, 3), round(level, 2))))
                touched_fibs.add(level)

    return fib_wick_touches, touched_fibs

def detect_body_ma_touches(df):
    ma_touches = []
    touched_mas = set()
    
    for i in range(len(df)):
        for ma in ['SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200']:
            if ma in df.columns:
                price = df[ma].iloc[i]
                if (df['Open'].iloc[i] <= price <= df['Close'].iloc[i] 
                    or df['Close'].iloc[i] <= price <= df['Open'].iloc[i]):
                    ma_touches.append((df.index[i], (ma, price)))
                    touched_mas.add(ma)

    return ma_touches, touched_mas

def detect_consecutive_days(df):
    sequence_stars = []
    current_sequence = 1
    up_down = 'neutral'

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Open'].iloc[i]:
            if up_down == 'up':
                current_sequence += 1
            else:
                up_down = 'up'
                current_sequence = 1
        elif df['Close'].iloc[i] < df['Open'].iloc[i]:
            if up_down == 'down':
                current_sequence += 1
            else:
                up_down = 'down'
                current_sequence = 1
        else:
            up_down = 'neutral'
            current_sequence = 1

        if current_sequence >= 2:
            size = 8 + (current_sequence - 1) * 5
            color = 'green' if up_down == 'up' else 'red'
            sequence_stars.append((df.index[i], df['Close'].iloc[i], size, color))

    return sequence_stars
