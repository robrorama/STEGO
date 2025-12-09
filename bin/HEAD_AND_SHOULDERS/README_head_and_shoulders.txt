NAME
    head.shoulders.v4.py - Advanced Chart Pattern Recognition Tool

SYNOPSIS
    python head.shoulders.v4.py TICKER [OPTIONS]

DESCRIPTION
    This powerful pattern recognition script identifies and visualizes 
    classic technical chart patterns alongside advanced trend analysis.

    Key Functionality:
    1. Automated Pattern Detection: Scans historical price data to identify:
       * Head and Shoulders Top (Bearish Reversal)
       * Inverse Head and Shoulders Bottom (Bullish Reversal)
       * Triple Top (Bearish Reversal)
       * Triple Bottom (Bullish Reversal)
              
    2. Trend Analysis (Figure 1):
       * Displays candlestick charts with all detected Peaks (Green) and 
         Troughs (Red) highlighted.
       * Draws "Touch Trendlines" that attempt to connect the maximum number 
         of pivot points.

    3. Multiscale Regression (Figure 2):
       * Applies Linear Regression Channels to the identified peaks and 
         troughs across multiple timeframes (Short/Medium/Long) to show the 
         slope of the trend structure.

    4. Pattern Visualization:
       * Every detected pattern is rendered on its own dedicated chart.
       * The script draws the specific "Neckline" (Support/Resistance) and 
         highlights the constituent peaks/troughs (Shoulders, Head, etc.).

    Dependencies:
    * data_retrieval.py (Canonical data source)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., AMD, INTC).

OPTIONS
    --period PERIOD
        Specifies the historical data window to scan for patterns.
        Valid values: 1y, 2y, 5y, max.
        Default: 1y.
        (Ignored if --date-range is used).

    --date-range "START,END"
        Overrides the period with a specific date range.
        Format: "YYYY-MM-DD,YYYY-MM-DD".

    --order INT
        The "Pivot Order" sensitivity. Defines how many bars on either side 
        of a high/low are required to confirm a local peak or trough.
        Higher values = fewer, more significant patterns.
        Lower values = more sensitive, potentially noisy patterns.
        Default: 5.

    --short-days INT
        Lookback window for Short-Term regression trendlines.
        Default: 60.

    --medium-days INT
        Lookback window for Medium-Term regression trendlines.
        Default: 120.

    --long-days INT
        Lookback window for Long-Term regression trendlines.
        Default: 252.

    --save-html
        Saves interactive HTML charts for every detected pattern and the 
        main analysis figures.
        Default: Disabled.

    --save-png
        Saves static PNG images for every chart.
        Default: Disabled.

    --no-show
        Prevents the script from opening browser tabs.
        Default: Disabled (All charts open automatically).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/TICKER/YYYY-MM-DD/

    Generated Files:
    * {TICKER}_peaks_troughs_pivot_touch.html
    * {TICKER}_peaks_troughs_multiscale.html
    * {TICKER}_hs_top_{N}.html (If detected)
    * {TICKER}_hs_bottom_{N}.html (If detected)
    * {TICKER}_triple_top_{N}.html (If detected)
    * {TICKER}_triple_bottom_{N}.html (If detected)

EXAMPLES
    1. Standard Scan
       Scan Apple (AAPL) for patterns over the last year.
       
       $ python head.shoulders.v4.py AAPL

    2. Deep History Scan
       Analyze Microsoft (MSFT) over 5 years with a higher pivot order (10) 
       to find only the most significant, major chart patterns.
       
       $ python head.shoulders.v4.py MSFT --period 5y --order 10

    3. Batch Report Generation
       Generate HTML reports for all patterns detected in Tesla (TSLA) 
       without opening browser windows.
       
       $ python head.shoulders.v4.py TSLA --save-html --no-show

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )

