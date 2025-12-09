NAME
    point.figure.charts.v1.py - Point & Figure Chart Generator

SYNOPSIS
    python point.figure.charts.v1.py TICKER [OPTIONS]

DESCRIPTION
    This tool generates classic Point & Figure (P&F) charts, a timeless 
    method for visualizing price action that filters out time and noise 
    to focus purely on supply and demand.

    Unlike candlestick charts, P&F charts only move when price moves by a 
    set "box size." They create a column of X's (rising price) or O's 
    (falling price). A new column is only started when price reverses 
    direction by a set amount (the "reversal amount").

    Features:
    * Supports both "Close-Only" and "High-Low" construction methods.
    * Configurable Box Size (fixed dollar amount or percentage of price).
    * Configurable Reversal Amount (typically 3 boxes).
    * Renders using Matplotlib for clean, publication-quality grids.

    Dependencies:
    * data_retrieval.py (Canonical data source)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., SPY, NVDA).

OPTIONS
    --period PERIOD
        Specifies the historical data window to load.
        Valid values: 1y, 2y, 5y, max.
        Default: 1y.

    --date-range "START,END"
        Overrides the period with a specific date range.
        Format: "YYYY-MM-DD,YYYY-MM-DD".

    --method {close, hilo}
        The construction method for the P&F chart.
        * close: Uses only the closing price. Good for long-term trends.
        * hilo: Uses the High and Low of the day. Captures more intraday 
          volatility and typically creates more columns.
        Default: hilo.

    --box-size FLOAT
        Sets a fixed box size in price units (e.g., 1.0 = $1 box).
        If not set, defaults to --box-pct or 1% of last price.

    --box-pct FLOAT
        Sets the box size as a percentage of the asset's last closing price.
        Example: 1.0 means the box size is 1% of the current price.
        Ignored if --box-size is provided.

    --reversal INT
        The number of boxes required to trigger a column reversal.
        Standard P&F charts use a 3-box reversal.
        Default: 3.

    --save-png
        Saves the resulting chart as a PNG image in the output directory.
        Default: Disabled.

    --save-csv
        Saves the raw P&F grid data (columns, levels, dates) to a CSV file.
        Useful for quantitative analysis of the chart structure.
        Default: Disabled.

    --no-show
        Prevents the Matplotlib window from opening. Use this for batch 
        generation.
        Default: Disabled.

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/TICKER/YYYY-MM-DD/

    Generated Files (if save flags are used):
    * {TICKER}_pnf_{METHOD}_rev{REV}_box{SIZE}.png
    * {TICKER}_pnf_{METHOD}_rev{REV}_box{SIZE}.csv

EXAMPLES
    1. Standard 3-Box Reversal
       Generate a High-Low P&F chart for Apple (AAPL) over the last year 
       using a 1% box size (default).
       
       $ python point.figure.charts.v1.py AAPL

    2. Fixed Box Size (Forex/Commodities)
       Analyze Gold (GC=F) using a fixed $10 box size and 3-box reversal.
       
       $ python point.figure.charts.v1.py GC=F --box-size 10 --reversal 3

    3. Long-Term Close-Only
       Analyze the S&P 500 (SPY) over 5 years using only closing prices to 
       filter out daily noise.
       
       $ python point.figure.charts.v1.py SPY --period 5y --method close

    4. Batch Generation
       Generate and save a chart for Tesla (TSLA) without viewing it.
       
       $ python point.figure.charts.v1.py TSLA --save-png --no-show

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
