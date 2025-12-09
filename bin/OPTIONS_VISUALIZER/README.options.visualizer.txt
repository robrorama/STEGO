NAME
    options_visualizer.v1.py - Unified Options & Technical Analysis CLI

SYNOPSIS
    python options_visualizer.v1.py COMMAND SUBCOMMAND [OPTIONS]

DESCRIPTION
    This is the flagship orchestrator script for the Stego Financial Framework. 
    It combines high-level options analytics (spreads, Greeks, IV surfaces) 
    with quantitative technical analysis (streak probabilities) into a single 
    command-line interface.

    It leverages 'chartlib_unified.py' for heavy lifting and 'data_retrieval.py' 
    for canonical data access.

COMMANDS
    analyze
        Perform numerical analysis without generating heavy charts.
        
        Subcommands:
        * spreads: Calculate Bull Call Debit & Bull Put Credit spreads.
        * streaks: Calculate win/loss streak probabilities based on historical data.

    charts
        Generate comprehensive visual reports.
        
        Subcommands:
        * all: Generate all available Plotly charts (IV Surface, OI Dashboard, 
          Chain Tumbler) and compile them into a single PDF report.

OPTIONS (Global)
    --help, -h
        Show help message and exit.

SUBCOMMAND OPTIONS

    analyze spreads TICKER WEEKS_TO_EXPIRY [OPTIONS]
        TICKER (str)
            Stock symbol (e.g., AAPL).
        WEEKS_TO_EXPIRY (int)
            Index of the expiration date to analyze (0 = nearest, 1 = next, etc.).
        
        --offset FLOAT
            Simulate a price shock. E.g., --offset -5.0 calculates spreads 
            assuming the stock price drops by $5.
            Default: 0.0
        
        --max-otm-percent FLOAT
            Maximum percentage out-of-the-money for the long leg of a credit spread.
            Default: 5.0
        
        --save-csv
            Save the resulting spread table to 'output/analysis/'.

    analyze streaks TICKER [OPTIONS]
        TICKER (str)
            Stock symbol.
        
        --period STR
            Lookback period for historical data (e.g., 1y, 5y, max).
            Default: max
        
        --save-csv
            Save the streak statistics table to 'output/analysis/'.

    charts all TICKER [OPTIONS]
        TICKER (str)
            Stock symbol.
        
        --quantile FLOAT
            Threshold for flagging "Unusual Activity" (Volume/OI). 
            0.95 = Top 5% percentile.
            Default: 0.95
        
        --smooth-iv
            Apply SciPy griddata smoothing to the Implied Volatility Surface 3D plot.
            Default: Disabled (Raw data).
        
        --iv-min FLOAT
        --iv-max FLOAT
            Manually override the color scale min/max for the IV Surface. 
            Useful for standardizing visualizations across tickers.

OUTPUT
    The script organizes outputs into the canonical structure:
    
    * output/html/{TICKER}/
      Interactive .html files for every chart (Chain Tumbler, IV Surface, etc.).
    
    * output/images/{TICKER}/
      Static .png snapshots of every chart (requires 'kaleido').
    
    * output/reports/
      {TICKER}_mega_report.pdf: A multi-page PDF compiling all static images.
    
    * output/analysis/
      CSV files for spread calculations and streak statistics.

EXAMPLES
    1. Quick Spread Check
       Check spreads for AAPL for the nearest expiration (index 0), assuming 
       a $2 price drop.
       
       $ python options_visualizer.v1.py analyze spreads AAPL 0 --offset -2.0

    2. Streak Analysis
       Calculate the probability of a green day after 3 consecutive red days 
       for SPY over its entire history.
       
       $ python options_visualizer.v1.py analyze streaks SPY --period max

    3. Full Report Generation
       Generate all charts for NVDA, smooth the IV surface, and compile a PDF.
       
       $ python options_visualizer.v1.py charts all NVDA --smooth-iv

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
