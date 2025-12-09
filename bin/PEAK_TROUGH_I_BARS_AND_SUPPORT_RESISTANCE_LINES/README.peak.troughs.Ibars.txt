NAME
    peak_TROUGH_I_Bars.v1.py - Drawdown & Recovery Analyzer

SYNOPSIS
    python peak_TROUGH_I_Bars.v1.py TICKER [OPTIONS]

DESCRIPTION
    This specialized technical analysis tool focuses on visualizing market 
    cycles through the lens of "Peak-to-Trough" drawdowns and subsequent 
    recoveries.

    It generates two key visualizations:
    
    1. I-Bar Cycle Chart: A comprehensive 4-panel plot containing:
       * Candlesticks marked with significant Peaks (Green) and Troughs (Red).
       * "I-Bars": Blue vertical lines connecting a Peak to the next Trough, 
         annotated with the exact percentage drawdown.
       * "Recovery Boxes": Gold shaded areas connecting a Trough to the next 
         Peak, annotated with the percentage recovery gain.
       * Supporting panels for Volume, MACD, and RSI.

    2. Support & Resistance Chart: A focused view highlighting the simple 
       min/max price levels over the last N sessions to establish immediate 
       trading boundaries.

    Dependencies:
    * data_retrieval.py (Canonical data source)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., SPY, QQQ).

OPTIONS
    --period PERIOD
        Specifies the historical data lookback period.
        Valid values: 1y, 2y, 5y, max.
        Default: 1y.
        (Ignored if --date-range is used).

    --date-range "START,END"
        Overrides the period with a specific date range.
        Format: "YYYY-MM-DD,YYYY-MM-DD".

    --order INT
        The window half-size for peak/trough detection. 
        Higher values = identify only major, macro-level pivots.
        Lower values = identify minor, short-term pivots.
        Default: 5.

    --sr-last-n INT
        The lookback window (in days) for calculating recent Support & 
        Resistance levels on the secondary chart.
        Default: 30.

    --save-html
        Saves the interactive charts as HTML files in the output directory.
        Default: Disabled.

    --save-png
        Saves high-resolution static images of the charts.
        Requires the 'kaleido' library.
        Default: Disabled.

    --no-show
        Prevents the script from automatically opening the generated charts 
        in the web browser.
        Default: Disabled (Charts open automatically).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/TICKER/YYYY-MM-DD/

    Generated Files (if save flags are used):
    * {TICKER}_ibars_indicators.html / .png
    * {TICKER}_support_resistance.html / .png

EXAMPLES
    1. Standard Cycle Analysis
       Analyze the last year of SPY to see drawdown depths and recovery times.
       
       $ python peak_TROUGH_I_Bars.v1.py SPY

    2. Bear Market Study
       Analyze Bitcoin (BTC-USD) during the 2022 crash to visualize the 
       magnitude of drawdowns.
       
       $ python peak_TROUGH_I_Bars.v1.py BTC-USD --date-range "2022-01-01,2022-12-31"

    3. Macro Trend View
       Analyze NVIDIA (NVDA) over 5 years with a high pivot order (20) to 
       see only the massive, structural market cycles.
       
       $ python peak_TROUGH_I_Bars.v1.py NVDA --period 5y --order 20

    4. Daily Report Generation
       Generate and save HTML reports for Tesla (TSLA) without opening 
       browser tabs.
       
       $ python peak_TROUGH_I_Bars.v1.py TSLA --save-html --no-show

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
