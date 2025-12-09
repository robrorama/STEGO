NAME
    darvas.v3.py - Darvas Box & Multi-Set SMA Visualizer

SYNOPSIS
    python darvas.v3.py TICKER [OPTIONS]

DESCRIPTION
    This command-line script generates advanced technical analysis charts 
    focusing on the "Darvas Box" trading theory. 
    
    It overlays identifying boxes (based on pivot highs and subsequent breakout levels) onto a 
    candlestick chart.

    To provide a comprehensive trend context, the script generates TWO 
    distinct visualization files for every run:
    
    1. Compact Set (V1): A standard set of Simple Moving Averages (9, 15, 
       20, 50, 100, 200) for general trend analysis.
    2. Extended Set (V2): A dense array of Moving Averages (4, 20, 50, 75, 
       100, 125, 150, 175, 200, 225, 250) for identifying complex support/
       resistance ribbons.

    All data is sourced using the centralized 'data_retrieval.py' module, 
    ensuring consistent caching and file organization within the Stego 
    Financial Framework.

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., NVDA, TSLA). The script 
        is case-insensitive.

OPTIONS
    --period PERIOD
        Specifies the historical data lookback period. 
        Valid values include: 1y, 2y, 5y, 10y, ytd, max.
        Default: max (to ensure long-term box formation context).

    --no-browser
        Flag to prevent the script from automatically opening the generated 
        HTML charts in your default web browser.
        Default: Disabled (charts open automatically).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/TICKER/YYYY-MM-DD/

    Generated Files:
    * {TICKER}_darvas_compact_MAs_V1.html
    * {TICKER}_darvas_many_MAs_V2.html

EXAMPLES
    1. Standard Full History Analysis
       Generate Darvas charts for Amazon (AMZN) using the maximum available 
       data history.
       
       $ python darvas.v3.py AMZN

    2. Recent History Analysis
       Analyze AMD over the last 2 years to focus on recent price structure.
       
       $ python darvas.v3.py AMD --period 2y

    3. Batch Mode
       Generate charts for Apple (AAPL) without interrupting your workflow 
       by opening browser tabs.
       
       $ python darvas.v3.py AAPL --no-browser

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
