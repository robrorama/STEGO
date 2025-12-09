#!/usr/bin/env python3
# SCRIPTNAME: micra.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import math
import plotly.io as pio
import pandas as pd
import numpy as np

# Local imports (must exist in same dir or path)
try:
    import geometry_prompts
    from data_retrieval import (
        get_stock_data,
        load_or_download_ticker,
        add_moving_averages,
        create_output_directory,
    )
    from plot_helpers import (
        add_open_shape_indicator,
        plot_signals_with_candlestick_refactored,
        plot_intersection_marker,
        add_anchored_volume_profile,
    )
    from signals import (
        detect_signals,
        calculate_fibonacci_levels,
        detect_wick_touches,
        detect_fib_wick_touches,
        detect_body_ma_touches,
        detect_consecutive_days,
        calculate_bollinger_bands,
        add_bollinger_band_markers,
        detect_volume_price_spikes,
    )
    from geometry import (
        find_two_high_peaks_in_period,
        find_two_low_troughs_in_period,
        calculate_intersection,
        plot_projection_line,
        calculate_linear_regression_and_deviations,
    )
    from summary import generate_summary_output
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)


def analyze_and_plot_for_df(df, ticker, date_range_label, output_directory, geometry_mode="auto"):
    """
    Run the existing pipeline for a given df and show + save plot.
    """
    # --- Indicators / prep ---
    df = add_moving_averages(df)
    df = calculate_bollinger_bands(df)

    # Simple MAs for annotations
    df['9DMA'] = df['Close'].rolling(window=9).mean()
    df['20DMA'] = df['Close'].rolling(window=20).mean()
    df['50DMA'] = df['Close'].rolling(window=50).mean()

    # Signals
    latest_date   = df.index[-1].strftime("%Y-%m-%d")
    current_price = df['Close'].iloc[-1]
    buy_signals, sell_signals = detect_signals(df)
    fib_levels, high_price, low_price = calculate_fibonacci_levels(df)

    # ---- Regression & deviation bands ----
    # Window fit to available data so slope/intercept are never None.
    N = len(df)
    len_reg = min(144, max(30, N))  # 30-floor to avoid tiny windows; 144 default cap
    slope, intercept, std_dev, deviations = calculate_linear_regression_and_deviations(df, len_reg)

    wick_touches, touched_devs     = detect_wick_touches(df, deviations, len_reg)
    fib_wick_touches, touched_fibs = detect_fib_wick_touches(df, fib_levels)
    ma_touches, touched_mas        = detect_body_ma_touches(df)
    sequence_stars                 = detect_consecutive_days(df)
    spike_days                     = detect_volume_price_spikes(df)

    # Summary CSV
    generate_summary_output(
        ticker,
        buy_signals,
        sell_signals,
        sequence_stars,
        wick_touches,
        fib_wick_touches,
        ma_touches,
        output_directory,
    )

    # Main Plot (existing function)
    fig = plot_signals_with_candlestick_refactored(
        df,
        buy_signals,
        sell_signals,
        fib_levels,
        wick_touches,
        fib_wick_touches,
        ma_touches,
        sequence_stars,
        slope,
        intercept,
        std_dev,
        ticker,
        deviations,
        touched_devs,
        spike_days,
    )

    # Bollinger overlays (unchanged)
    for band, color, style in [
        ('BB_Middle',    'gray',       'dash'),
        ('BB_Upper_1std','green',      'dot'),
        ('BB_Upper_2std','lightgreen', 'dot'),
        ('BB_Lower_1std','orange',     'dot'),
        ('BB_Lower_2std','pink',       'dot'),
    ]:
        fig.add_trace(
            dict(
                x=df.index,
                y=df[band],
                mode='lines',
                name=band,
                line=dict(color=color, dash=style),
            )
        )

    # Markers beyond 2σ (unchanged)
    hi = df[df['High'] > df['BB_Upper_2std']]
    lo = df[df['Low']  < df['BB_Lower_2std']]
    if not hi.empty:
        fig.add_trace(dict(x=hi.index, y=hi['High'], mode='markers',
                           name='Pierces +2σ', marker=dict(symbol='square', size=10)))
    if not lo.empty:
        fig.add_trace(dict(x=lo.index, y=lo['Low'], mode='markers',
                           name='Pierces -2σ', marker=dict(symbol='square', size=10)))

    # --- Geometry logic (unchanged defaults) ---
    slope_h = int_h = slope_l = int_l = None
    if geometry_mode == 'prompt':
        print("--- Manual Geometry Input ---")
        if input("Manually specify peaks? (y/n): ").lower() == 'y':
            manual_peaks = geometry_prompts.manual_peaks_or_troughs(df, is_peaks=True)
            slope_h, int_h = plot_projection_line(df, fig, manual_peaks, color='green', line_name='High Peak Line')
        if input("Manually specify troughs? (y/n): ").lower() == 'y':
            manual_troughs = geometry_prompts.manual_peaks_or_troughs(df, is_peaks=False)
            slope_l, int_l = plot_projection_line(df, fig, manual_troughs, color='red', line_name='Low Trough Line')
    else:
        high_peaks = find_two_high_peaks_in_period(df)
        low_troughs = find_two_low_troughs_in_period(df)
        slope_h, int_h = plot_projection_line(df, fig, high_peaks['High'], color='green', line_name='High Peak Line')
        slope_l, int_l = plot_projection_line(df, fig, low_troughs['Low'],  color='red',   line_name='Low Trough Line')

    angle_h_str = f" | High Angle: {round(math.degrees(math.atan(slope_h)), 2)}°" if slope_h is not None else ""
    angle_l_str = f" | Low Angle: {round(math.degrees(math.atan(slope_l)), 2)}°"  if slope_l is not None else ""

    inter_str = ""
    if all(v is not None for v in [slope_h, int_h, slope_l, int_l]):
        try:
            date_i, price_i = calculate_intersection(slope_h, int_h, slope_l, int_l)
            plot_intersection_marker(fig, date_i, price_i)
            inter_str = f" | Intersection: {date_i.strftime('%Y-%m-%d')} @ {price_i:.2f}"
        except ValueError:
            inter_str = ""

    fig.update_layout(title=f"{ticker} Analysis [{date_range_label}]{angle_h_str}{angle_l_str}{inter_str}")

    # Watermark (unchanged)
    fig.add_annotation(
        text=f"{ticker.upper()} \n- {latest_date}\n- ${current_price:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=100, color="green"),
        opacity=0.1,
    )

    # Render + save
    fig.show()
    dev_fig = fig.full_figure_for_development(warn=False)
    dev_fig.update_layout(showlegend=False, xaxis_rangeslider_visible=False)
    plot_fn = os.path.join(output_directory, f"{ticker}_{date_range_label.replace(',', '_')}_plot.png")
    pio.write_image(dev_fig, plot_fn, format="png")
    print(f"Plot saved as {plot_fn}")


def main():
    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    parser.add_argument("ticker", help="A single stock ticker symbol (e.g., SPY)")
    parser.add_argument("--date-range", default="1y", help="YF period (e.g., 1y) or date range 'YYYY-MM-DD,YYYY-MM-DD'")
    parser.add_argument("--geometry-mode", default="auto", choices=["auto", "prompt"], help="Geometry line detection mode.")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    date_range_input = args.date_range
    
    # CONSTRAINT: Ensure output directory is in /dev/shm
    output_directory = create_output_directory(ticker)

    # If explicit date range, preserve original single-render behavior
    if ',' in date_range_input:
        start_date, end_date = [d.strip() for d in date_range_input.split(',')]
        df = load_or_download_ticker(ticker, start=start_date, end=end_date)
        analyze_and_plot_for_df(df, ticker, f"{start_date}_{end_date}", output_directory, geometry_mode=args.geometry_mode)
        return

    # Period mode: add extra 6mo and 5y views
    primary_period = date_range_input.strip().lower()
    # Unique ordered list: primary, then extras
    ordered_periods = list(dict.fromkeys([primary_period, '6mo', '5y']))

    for period in ordered_periods:
        df = get_stock_data(ticker, period=period)
        analyze_and_plot_for_df(df, ticker, period, output_directory, geometry_mode=args.geometry_mode)


if __name__ == "__main__":
    main()
