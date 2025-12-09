NAME
    revamp.heatmap.tickers.by.volume.downloads.first.v3.py - S&P 500 Heatmap Generator

SYNOPSIS
    python revamp.heatmap.tickers.by.volume.downloads.first.v3.py [CSV_PATH] [OUT_DIR] [OPTIONS]

DESCRIPTION
    This script generates a visual "Heatmap" (Treemap) of a stock universe, 
    typically the S&P 500. 
    
    It follows a robust Two-Phase Process:
    1. Verification Phase: Iterates through the list of tickers, checking 
       if valid cached data exists. If not, it downloads the missing data via 
       the centralized `data_retrieval` module.
    2. Processing Phase: Reads strictly from the local disk cache to build 
       the visualization.
    
    The resulting Treemap visualizes:
    * Box Size: Proportional to the 30-Day Average Dollar Volume (Liquidity).
    * Box Color: Based on the % Change (Green = Up, Red = Down).

    Dependencies:
    * data_retrieval.py (Canonical data source)
    * pandas, plotly, numpy

POSITIONAL ARGUMENTS
    CSV_PATH
        The path to a CSV file containing the list of tickers to analyze.
        The CSV must contain a column named "Symbol" or "Ticker".
        Default: cache/combined_tickers.csv

    OUT_DIR
        The directory where the final HTML heatmap file will be saved.
        Default: output/SP500/YYYY-MM-DD/ (Managed by data_retrieval)

OPTIONS
    --delay SECONDS
        The pause duration (in seconds) inserted *only* after a fresh 
        download occurs. This helps prevent rate-limiting from the data 
        provider when fetching large batches of missing data.
        Default: 1.0

OUTPUT
    The script generates a single interactive HTML file:
    * sp500_heatmap_YYYYMMDD.html

    This file contains a Plotly Treemap that can be opened in any web browser.

EXAMPLES
    1. Standard Run (Default Config)
       Generate a heatmap using the default ticker list in 'cache/combined_tickers.csv'.
       
       $ python revamp.heatmap.tickers.by.volume.downloads.first.v3.py

    2. Custom Universe
       Generate a heatmap for a custom list of stocks (e.g., a Nasdaq 100 list).
       
       $ python revamp.heatmap.tickers.by.volume.downloads.first.v3.py my_nasdaq_tickers.csv

    3. Aggressive Download (Fast)
       Run the script with a shorter delay (0.5s) to speed up the download 
       phase if many tickers are missing.
       
       $ python revamp.heatmap.tickers.by.volume.downloads.first.v3.py --delay 0.5

    4. Custom Output Location
       Save the resulting HTML report to a specific "reports" folder.
       
       $ python revamp.heatmap.tickers.by.volume.downloads.first.v3.py cache/combined_tickers.csv ./reports/

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
