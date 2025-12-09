NAME
    peakTrough.circles.v3.py - Algorithmic Trend & Pivot Analyzer

SYNOPSIS
    python peakTrough.circles.v3.py TICKER [OPTIONS]

DESCRIPTION
    This tool performs a geometric analysis of price action by identifying
    key pivot points (Peaks and Troughs) and drawing automated trendlines.

    Unlike simple moving averages, this script uses a windowed local extrema
    algorithm to find "swing highs" and "swing lows". It then calculates 
    Linear Regression lines specifically fitting *only* those pivot points 
    across multiple timeframes (Short, Medium, Long).

    This helps identify the "Slope of the Structure" rather than just the 
    average price, making it highly effective for identifying trend channel 
    breakouts or breakdowns.

    Dependencies:
    * data_retrieval.py (Canonical data source)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., SPY, QQQ).

OPTIONS
    --period PERIOD
        Specifies the historical lookback period for the chart.
        Valid values: 1mo, 6mo, 1y, 2y, 5y, max.
        Default: 1y.

    --dates "START,END"
        Overrides the period with a specific date range.
        Format: "YYYY-MM-DD,YYYY-MM-DD".

    --order INT
        The "Pivot Order" or window size for peak detection.
        A value of 5 means a high must be higher than the 5 bars before it 
        and the 5 bars after it to be considered a Peak.
        Higher values = fewer, more significant pivots.
        Lower values = more pivots, more noise.
        Default: 5.

    --short-days INT
        Lookback window (in days) for the Short-Term Pivot Trendline.
        Default: 60 (approx. 1 quarter).

    --medium-days INT
        Lookback window (in days) for the Medium-Term Pivot Trendline.
        Default: 120 (approx. 2 quarters).

    --long-days INT
        Lookback window (in days) for the Long-Term Pivot Trendline.
        Default: 252 (approx. 1 trading year).

    --no-show
        Prevents the script from automatically opening the generated HTML 
        chart in the web browser.
        Default: Disabled (Chart opens automatically).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/TICKER/YYYY-MM-DD/

    Generated File:
    * {TICKER}_peaks_troughs.html

VISUALIZATION DETAILS
    * Green Circles: Identified Peaks (Swing Highs).
    * Red Circles: Identified Troughs (Swing Lows).
    * Green Dotted Line: Short-term Regression of Peaks.
    * Green Solid Line: Short-term Regression of Troughs.
    * Blue Lines: Medium-term Regression of Peaks/Troughs.
    * Purple Lines: Long-term Regression of Peaks/Troughs.

EXAMPLES
    1. Standard Trend Analysis
       Analyze SPY over the last year with default pivot settings.
       
       $ python peakTrough.circles.v3.py SPY

    2. Macro Trend Structure
       Analyze Bitcoin (BTC-USD) over 3 years, using a wider pivot order (10) 
       to filter out crypto volatility and find major swings only.
       
       $ python peakTrough.circles.v3.py BTC-USD --period 3y --order 10

    3. Custom Trendline Windows
       Analyze Apple (AAPL) focusing on very short-term trends (20, 40, 60 days).
       
       $ python peakTrough.circles.v3.py AAPL --short-days 20 --medium-days 40 --long-days 60

    4. Batch Processing
       Generate the analysis for META without opening the browser window.
       
       $ python peakTrough.circles.v3.py META --no-show

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
