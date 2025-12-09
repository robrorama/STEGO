NAME
    unified.pie.charts.v2.py - Analyst Recommendations Visualizer

SYNOPSIS
    python unified.pie.charts.v2.py TICKER [OPTIONS]

DESCRIPTION
    This command-line script fetches and visualizes financial analyst 
    recommendations for a specific stock ticker. It aggregates data to 
    produce a breakdown of "Buy", "Hold", and "Sell" ratings.

    The script generates a high-quality, interactive Pie Chart using Plotly. 
    It prioritizes summary data (e.g., Strong Buy, Hold) but employs a 
    fallback mechanism to parse detailed time-series recommendation logs 
    if summary data is unavailable.

    All data and output images are stored in the centralized data directories 
    defined by the Stego Financial Framework's retrieval modules.

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., NVDA, AMD). The script 
        is case-insensitive and will automatically normalize the input 
        to uppercase.

OPTIONS
    --no-browser
        By default, the script automatically opens the generated interactive 
        HTML chart in a new tab of your default web browser. Use this flag 
        to suppress this behavior (useful for batch processing or remote 
        environments).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    1. Data Storage:
       csv files are saved to: output/TICKER/YYYY-MM-DD/
       * {TICKER}_recommendations_summary.csv
       * {TICKER}_recommendations.csv

    2. Visualizations:
       * Interactive HTML: output/TICKER/YYYY-MM-DD/{TICKER}_analyst_recommendations.html
       * Static PNG: images/{TICKER}/{TICKER}_analyst_recommendations.png
       * Archive PNG: PNGS/{TICKER}_{DATE}_analyst_recommendations.png

EXAMPLES
    1. Standard Analysis
       Fetch recommendations for Nvidia and open the chart in the browser.
       
       $ python unified.pie.charts.v2.py NVDA

    2. Batch/Silent Mode
       Fetch recommendations for Ford Motor Company without opening the 
       browser.
       
       $ python unified.pie.charts.v2.py F --no-browser

    3. ETF Analysis
       Analyze recommendations for an ETF (e.g., QQQ), relying on the 
       fallback time-series parser if summary data is missing.
       
       $ python unified.pie.charts.v2.py QQQ

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
