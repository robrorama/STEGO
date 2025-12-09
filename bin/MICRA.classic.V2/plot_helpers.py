#!/usr/bin/env python3
# SCRIPTNAME: plot_helpers.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_candlestick_chart(df):
    fig = go.Figure()
    # Add OHLC dots
    for price_type, color, size in [
        ('Open', 'cyan', 2), ('Close', 'white', 2), ('High', 'green', 2), ('Low', 'yellow', 2), ('Midpoint', 'orange', 2)
    ]:
        y_values = (df['High'] + df['Low']) / 2 if price_type == 'Midpoint' else df[price_type]
        fig.add_trace(go.Scatter(x=df.index, y=y_values, mode='markers', name=price_type, marker=dict(color=color, size=size)))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlesticks'
    ))

    # Fill between 20DMA and 50DMA
    for i in range(1, len(df)):
        if pd.notna(df['20DMA'].iloc[i]) and pd.notna(df['50DMA'].iloc[i]):
            color = 'rgba(0, 255, 0, 0.5)' if df['20DMA'].iloc[i] > df['50DMA'].iloc[i] else 'rgba(255, 0, 0, 0.5)'
            fig.add_trace(go.Scatter(
                x=[df.index[i-1], df.index[i], df.index[i], df.index[i-1]],
                y=[df['50DMA'].iloc[i-1], df['50DMA'].iloc[i], df['20DMA'].iloc[i], df['20DMA'].iloc[i-1]],
                fill='toself', fillcolor=color, mode='none', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))

    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker=dict(color=df['Close'] > df['Open'], colorscale=[[0, 'red'], [1, 'green']], opacity=0.2),
        yaxis='y2'
    ))
    fig.update_layout(yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False), barmode='overlay')
    return fig

def add_moving_averages(fig, df):
    for ma in ['SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200']:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode='lines', name=ma))
    return fig

def add_buy_signals(fig, buy_signals):
    dates, prices = zip(*buy_signals) if buy_signals else ([], [])
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signals'))
    return fig

def add_sell_signals(fig, sell_signals):
    dates, prices = zip(*sell_signals) if sell_signals else ([], [])
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signals'))
    return fig

def add_fibonacci_levels(fig, fib_levels, df):
    pcts = [23.6, 38.2, 50, 61.8, 78.6]
    colors = ['purple', 'green', 'blue', 'red', 'orange']
    x = [df.index[0], df.index[-1]]
    for i, (level, pct) in enumerate(zip(fib_levels['Fibonacci Levels'], pcts)):
        fig.add_trace(go.Scatter(x=x, y=[level, level], mode="lines", line=dict(dash="dash", color=colors[i % len(colors)]), name=f"{pct}% @ {round(level)}"))
    return fig

def add_linear_regression(fig, slope, intercept, df):
    x0, x1 = df.index[0].toordinal(), df.index[-1].toordinal()
    y0, y1 = slope * x0 + intercept, slope * x1 + intercept
    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[y0, y1], mode='lines', line=dict(color='black', dash='solid', width=2), name='Regression Line'))
    return fig

def add_regression_bands(fig, slope, intercept, std_dev, df):
    x0, x1 = df.index[0].toordinal(), df.index[-1].toordinal()
    for dev in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        y0u, y1u = slope*x0 + intercept + (dev*std_dev), slope*x1 + intercept + (dev*std_dev)
        y0l, y1l = slope*x0 + intercept - (dev*std_dev), slope*x1 + intercept - (dev*std_dev)
        fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[y0u, y1u], mode='lines', line=dict(color='blue', dash='dot', width=1), name=f'Upper {dev}σ'))
        fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[y0l, y1l], mode='lines', line=dict(color='orange', dash='dot', width=1), name=f'Lower {dev}σ'))
    return fig

def add_deviation_bands(fig, deviations, df, touched_devs):
    for dev_name, dev_prices in deviations.items():
        if dev_name not in touched_devs: continue
        color = 'blue' if 'upper' in dev_name else 'orange'
        dev_value = float(dev_name.split('_')[1])
        line_name = f"{'Upper' if 'upper' in dev_name else 'Lower'} {dev_value}σ"
        fig.add_trace(go.Scatter(x=[df.index[-len(dev_prices)], df.index[-1]], y=dev_prices, mode='lines', line=dict(color=color, dash='dot', width=1), name=line_name))
    return fig

def add_wick_touches(fig, wick_touches):
    x, y = ([date for date, _ in wick_touches], [price for _, (_, price) in wick_touches])
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(symbol='x', color='blue', size=8), name='Wick Touches'))
    return fig

def add_fib_wick_touches(fig, fib_wick_touches):
    x, y = ([date for date, _ in fib_wick_touches], [price for _, (_, price) in fib_wick_touches])
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(symbol='circle', color='purple', size=6), name='Fib Wick Touches'))
    return fig

def add_ma_touches(fig, ma_touches):
    x, y = ([date for date, _ in ma_touches], [price for _, (_, price) in ma_touches])
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(symbol='cross', color='black', size=8), name='MA Touches'))
    return fig

def add_open_shape_indicator(fig, spike_days):
    dates, prices = zip(*spike_days) if spike_days else ([], [])
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='markers', marker=dict(symbol='circle-open', size=15, color='black', line=dict(width=2)), name='Volume & Price Spike'))
    return fig

def add_sequence_stars(fig, sequence_stars):
    for color in set([star[3] for star in sequence_stars]):
        star_data = [(date, price, size) for date, price, size, c in sequence_stars if c == color]
        if not star_data: continue
        x, y, sizes = zip(*star_data)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(symbol='star', size=sizes, color=color), name=f'{color.capitalize()} Stars'))
    return fig

def add_anchored_volume_profile(fig, df, anchor_price, period=20):
    return fig # Placeholder

def plot_intersection_marker(fig, date_intersect, y_intersect):
    for i, radius in enumerate([10, 15, 20], start=1):
        fig.add_trace(go.Scatter(x=[date_intersect], y=[y_intersect], mode="markers", visible="legendonly", marker=dict(symbol="circle-open", size=radius, color="red"), name=f"Intersection Circle {i}"))

def finalize_layout(fig, ticker):
    fig.update_layout(
        title=f'{ticker} Candlestick Chart with Interactive Signal Toggles',
        xaxis_title='Date', yaxis_title='Price',
        legend=dict(itemsizing='constant'), plot_bgcolor='white', paper_bgcolor='white',
        hovermode='x unified', spikedistance=1000,
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True, spikethickness=1, spikedash='solid', spikecolor='black'),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True, spikethickness=1, spikedash='solid', spikecolor='black', side='right'),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family="Courier New")
    )
    return fig

def plot_signals_with_candlestick_refactored(
    df, buy_signals, sell_signals, fib_levels, wick_touches, 
    fib_wick_touches, ma_touches, sequence_stars, slope, 
    intercept, std_dev, ticker, deviations, touched_devs, spike_days
):
    fig = create_candlestick_chart(df)
    add_moving_averages(fig, df)
    add_buy_signals(fig, buy_signals)
    add_sell_signals(fig, sell_signals)
    add_fibonacci_levels(fig, fib_levels, df)
    add_sequence_stars(fig, sequence_stars)
    add_linear_regression(fig, slope, intercept, df)
    add_regression_bands(fig, slope, intercept, std_dev, df)
    add_open_shape_indicator(fig, spike_days)
    add_wick_touches(fig, wick_touches)
    add_fib_wick_touches(fig, fib_wick_touches)
    add_ma_touches(fig, ma_touches)
    finalize_layout(fig, ticker)
    return fig
