NAME
    volume.visualizer.V2.py - Stock Volume & Price Visualizer

SYNOPSIS
    python volume.visualizer.V2.py TICKER [PERIOD | START_DATE END_DATE] [OPTIONS]

DESCRIPTION
    This command-line script generates a comprehensive suite of financial charts
    for a specified stock ticker. Its primary function is to create 
    detailed price and volume analysis plots across multiple time windows 
    (e.g., 1 month, 1 year, max).

    The charts include key technical indicators such as Bollinger Bands, MACD, 
    RSI, and Stochastic Oscillators. A key feature is the automatic 
    detection and visualization of high-volume trading days, which are marked 
    with bubbles sized by volume. The script also identifies and 
    highlights recurring seasonal high-volume patterns with gold stars.

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., AAPL, GOOG). This 
        argument is required.

    DATE_ARGS
        Specifies the time frame for the analysis. This can be 
        provided in one of two formats:
        
        1. A single period string: 
           1d, 5d, 1m, 1mo, 3m, 3mo, 6m, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        
        2. A specific start and end date range:
           YYYY-MM-DD YYYY-MM-DD.

OPTIONS
    --zscore FLOAT
        Sets the Z-score threshold for identifying a high-volume day. A day's 
        volume Z-score must exceed this value to be flagged.
        Default: 2.0.

    --hv-mode {zonly, strict}
        Defines the criteria for flagging a high-volume day.
        
        zonly:  Volume Z-score is greater than the threshold.
        strict: Volume Z-score is greater than the threshold AND the daily 
                volume is greater than its 20-day moving average plus two 
                standard deviations.
        
        Default: zonly.

    --rec-threshold INT
        The minimum number of times a specific month-day (e.g., "10-28") must 
        appear in the high-volume data to be marked as a recurring seasonal 
        event.
        Default: 3.

    --vol-color {up, down}
        Sets the rule for coloring the volume bars.
        
        up:   Bars are green if Close >= Open.
        down: Bars are green if Open >= Close.
        
        Default: up.

    --no-tabs
        A flag to prevent the script from automatically opening the generated 
        charts in a web browser.
        Default: Disabled (tabs open automatically).

    --no-gif
        A flag to disable the creation of the animated GIF that cycles through 
        the price charts of different time windows.
        Default: Disabled (GIF is created automatically).

    --extra-tabs
        A flag to generate and display two additional charts: a combined 
        Price + Volume chart and a consolidated Oscillators chart.
        Default: Disabled.

OUTPUT
    The script creates a dated output directory structured as:
    output/TICKER/YYYY-MM-DD/.
    
    The directory contains:
    * HTML Files: Interactive Plotly charts (.html) for each plot.
    * PNG Files: Static, high-resolution images (.png) of charts.
    * CSV Snapshot: A .csv file containing the full raw data series.
    * Animated GIF: An animated macd.gif file located in the images/TICKER/ 
      directory that cycles through price charts.

EXAMPLES
    1. Basic Analysis for the Last Year
       Runs a standard analysis for NVDA over the past year using default 
       settings.
       
       $ python volume.visualizer.V2.py NVDA 1y

    2. Analysis for a Specific Date Range
       Analyzes TSLA data specifically for the calendar year 2024.
       
       $ python volume.visualizer.V2.py TSLA 2024-01-01 2024-12-31

    3. Strict High-Volume Analysis
       Analyzes MSFT for the last 6 months, utilizing a stricter criterion 
       for high-volume days and a higher Z-score threshold.
       
       $ python volume.visualizer.V2.py MSFT 6m --hv-mode strict --zscore 2.5

    4. Automated Run with No Browser Output
       Runs a full analysis for AAPL over the last two years. It generates 
       extra charts but suppresses browser tabs, making it ideal for 
       automation.
       
       $ python volume.visualizer.V2.py AAPL 2y --no-tabs --extra-tabs

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
