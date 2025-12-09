NAME
    peak.troughs.v21.with.macd.rsi.works.just.ignores.most.recent.py - 
    Advanced Multiscale Divergence Analyzer

SYNOPSIS
    python peak.troughs.v21.with.macd.rsi.works.just.ignores.most.recent.py TICKER [OPTIONS]

DESCRIPTION
    This sophisticated analysis tool is designed to detect price/momentum 
    divergences across multiple timeframes simultaneously.

    It generates two distinct, interactive visualizations:
    
    1. Price & Volume Chart:
       * OHLC Candlesticks overlaid with automatically detected Peak (Green) 
         and Trough (Red) pivot points.
       * Multiscale Linear Regression trendlines drawn from these pivots 
         for Short (60d), Medium (120d), and Long (252d) terms.
       * Color-coded Volume bars (Green for up-days, Red for down-days).

    2. Momentum Indicator Chart:
       * MACD (12, 26, 9) and RSI (14) panes.
       * Crucially, this script applies the SAME peak/trough detection and 
         multiscale regression logic to the INDICATORS themselves.
       * This allows for the immediate visual identification of "Indicator 
         Trendlines" and divergences (e.g., Price making higher highs while 
         RSI regression line slopes downward).

    Dependencies:
    * data_retrieval.py (Canonical data source)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., SPY, NVDA).

OPTIONS
    --period PERIOD
        Specifies the historical data lookback window.
        Valid values: 1y, 2y, 5y, max.
        Default: 1y.
        (Ignored if --date-range is used).

    --date-range "START,END"
        Overrides the period with a specific date range.
        Format: "YYYY-MM-DD,YYYY-MM-DD".

    --order INT
        The window half-size for the local extrema detection algorithm.
        A value of 5 means a point must be the highest/lowest in a 
        10-bar window (5 before, 5 after) to be marked as a pivot.
        Default: 5.

    --short-days INT
        Lookback window for the Short-Term trendlines.
        Default: 60.

    --medium-days INT
        Lookback window for the Medium-Term trendlines.
        Default: 120.

    --long-days INT
        Lookback window for the Long-Term trendlines.
        Default: 252.

    --save-html
        Saves the interactive charts as .html files in the output directory.
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
    * {TICKER}_price_volume_multiscale.html / .png
    * {TICKER}_indicators_macd_rsi_multiscale.html / .png

EXAMPLES
    1. Standard Divergence Check
       Analyze Apple (AAPL) over the last year to spot potential RSI/Price 
       divergences.
       
       $ python peak.troughs.v21.with.macd.rsi.works.just.ignores.most.recent.py AAPL

    2. Long-Term Trend Structure
       Analyze the S&P 500 (SPY) over 5 years with wider pivot detection 
       (order=10) to filter out noise and find major structural turns.
       
       $ python peak.troughs.v21.with.macd.rsi.works.just.ignores.most.recent.py SPY --period 5y --order 10

    3. Custom Trend Windows
       Analyze Tesla (TSLA) with custom windows for Quarter (63d), Half-Year 
       (126d), and Full Year (252d) trendlines.
       
       $ python peak.troughs.v21.with.macd.rsi.works.just.ignores.most.recent.py TSLA --short-days 63 --medium-days 126 --long-days 252

    4. Automated Reporting
       Generate and save HTML reports for NVIDIA (NVDA) without opening the 
       browser.
       
       $ python peak.troughs.v21.with.macd.rsi.works.just.ignores.most.recent.py NVDA --save-html --no-show

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
