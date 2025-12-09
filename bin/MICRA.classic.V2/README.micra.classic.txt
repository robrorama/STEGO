NAME
micra.py - The "Micra" Stock Analysis & Visualization Toolkit

SYNOPSIS
python micra.py TICKER [OPTIONS]

DESCRIPTION
The Micra toolkit is a sophisticated technical analysis engine designed
to visualize complex market geometry and signal detection.

It combines traditional indicators (Bollinger Bands, Moving Averages) with 
advanced geometric projections (Peak/Trough trendlines, Convergence points) 
and algorithmic signal detection (Wick touches, Consecutive streaks, 
Volume spikes).

Unlike standard chart tools, Micra is designed to find "intersections" 
in price and time, projecting where major high and low trendlines 
converge in the future.

Dependencies:
* data_retrieval.py (Canonical data source)
* geometry.py (Peak/Trough detection logic)
* geometry_prompts.py (Interactive user input handling)
* plot_helpers.py (Plotly rendering utilities)
* signals.py (Algorithmic detection logic)
* summary.py (CSV report generation)


POSITIONAL ARGUMENTS
TICKER
(Required) The stock ticker symbol to analyze (e.g., SPY, NVDA).
The script automatically converts this to uppercase.

OPTIONS
--date-range RANGE
Specifies the historical data window.

    Formats:
    1. Period String: "1y", "6mo", "5y", "max".
       If a period string is provided, Micra will generate THREE charts:
       - The requested period.
       - A 6-month short-term view.
       - A 5-year long-term view.
    
    2. Explicit Dates: "YYYY-MM-DD,YYYY-MM-DD".
       If explicit dates are provided, Micra generates exactly ONE chart 
       for that specific window.

    Default: "1y"

--geometry-mode MODE
    Controls how the Peak and Trough projection lines are calculated.
    
    Choices:
    * auto: (Default) The script algorithmically identifies the two 
      most significant peaks and the two most significant troughs 
      in the lookback period and draws projection lines automatically.
    
    * prompt: The script enters an interactive mode, pausing execution 
      to ask the user to manually input the specific dates for the 
      peaks and troughs they wish to use for projection.


OUTPUT
The script creates a directory: output/{TICKER}/YYYY-MM-DD/

Inside, it generates:
1. Interactive Charts: {TICKER}_{PERIOD}_plot.png (High-res snapshot)
   (The interactive HTML version opens in your browser).
2. Signal Summary: {TICKER}_detailed_signal_summary.csv
   Contains a log of every Buy/Sell signal, Wick touch, and MA touch 
   detected during the period.


EXAMPLES
1. Standard Tri-View Analysis
Generate 1-year, 6-month, and 5-year charts for SPY with automatic
geometry detection.

   $ python micra.py SPY

2. Focused Timeframe
   Analyze Nvidia (NVDA) for the last 3 months only.
   
   $ python micra.py NVDA --date-range 3mo

3. Specific Event Analysis
   Analyze Tesla (TSLA) during the 2022 calendar year.
   
   $ python micra.py TSLA --date-range "2022-01-01,2022-12-31"

4. Manual Geometry Override
   Analyze Microsoft (MSFT), but manually specify the exact tops and 
   bottoms to draw trendlines (useful for testing specific theories).
   
   $ python micra.py MSFT --geometry-mode prompt

   (User is then prompted:)
   > Manually specify peaks? (y/n): y
   > Enter date for first High (YYYY-MM-DD): 2023-07-18
   > Enter date for second High (YYYY-MM-DD): 2023-11-20


AUTHOR
Michael Derby
November 20, 2025
( STEGO FINANCIAL FRAMEWORK )

