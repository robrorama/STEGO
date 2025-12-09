NAME
    ichimoku.v7.py - Advanced Interactive Stock Charting with Ichimoku-Style Overlays

SYNOPSIS
    python ichimoku.v7.py TICKER [OPTIONS]

DESCRIPTION
    This command-line script generates a highly detailed, interactive financial 
    chart for a specified stock ticker. It is designed for deep technical 
    analysis, featuring a "save-and-reload" architecture that ensures data 
    integrity by persisting downloaded data to CSV before visualization.

    The visualization includes:
    * **Candlestick Chart**: Standard OHLC representation.
    * **Moving Averages**: 9-day, 20-day, and 50-day Simple Moving Averages (SMAs).
    * **Bollinger Bands**: A complete set including the middle band (20 SMA) 
        and upper/lower bands at both 1 and 2 Standard Deviations.
    * **Trend Cloud**: Dynamic shading between the 20-day and 50-day SMAs 
        (green for bullish, red for bearish) to visualize trend strength.
    * **Volatility Markers**: Special indicators highlighting bars where 
        Highs or Lows breach the 2-Standard-Deviation Bollinger Bands.
    * **Earnings Dates**: Gold markers along the bottom axis indicating 
        past or upcoming earnings reports (cached locally to minimize API calls).
    * **6-Month Extremes**: Dashed horizontal lines marking the highest high 
        and lowest low achieved outside the Bollinger Bands over the last 6 months.

    The resulting chart is rendered as an interactive Plotly figure and 
    automatically opened in the default web browser.

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., META, AMZN). The script 
        is case-insensitive.

OPTIONS
    --start_date YYYY-MM-DD
        The start date for the historical data retrieval.
        Default: 2023-01-01

    --end_date YYYY-MM-DD
        The end date for the historical data retrieval.
        Default: 2023-12-31

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    1. Data Persistence:
       * Price Data: output/TICKER/YYYY-MM-DD/{TICKER}.csv
       * Earnings Cache: cache/{TICKER}_earnings_dates.csv

    2. Visualization:
       * The script generates a temporary interactive view in the web browser.

EXAMPLES
    1. Standard Analysis (Default Dates)
       Analyze Tesla (TSLA) using the default date range (2023-01-01 to 
       2023-12-31).
       
       $ python ichimoku.v7.py TSLA

    2. Custom Date Range
       Analyze NVIDIA (NVDA) for the first half of 2024.
       
       $ python ichimoku.v7.py NVDA --start_date 2024-01-01 --end_date 2024-06-30

    3. Long-Term Analysis
       Analyze Apple (AAPL) over a two-year period to visualize long-term 
       trend clouds and volatility extremes.
       
       $ python ichimoku.v7.py AAPL --start_date 2022-01-01 --end_date 2023-12-31

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
