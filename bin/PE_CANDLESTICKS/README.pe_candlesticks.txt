NAME
    PE_candlesticks.V3.py - Fundamental & Technical Fusion Analyzer

SYNOPSIS
    python PE_candlesticks.V3.py TICKER [OPTIONS]

DESCRIPTION
    This unique analysis tool bridges the gap between Technical Analysis (Price)
    and Fundamental Analysis (Valuation).

    It generates an interactive Plotly chart that overlays the stock's Price 
    Candlesticks with its Trailing P/E Ratio (Price-to-Earnings) over time.

    Crucially, it applies Linear Regression Channels directly to the P/E Ratio
    itself. This allows analysts to determine if a stock is statistically 
    "overvalued" or "undervalued" relative to its own historical valuation 
    trend, rather than just its nominal price trend.

    Dependencies:
    * data_retrieval.py (Canonical price source)
    * yfinance (Canonical fundamental source)
    * scipy.stats (Linear regression logic)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., AAPL, MSFT).

OPTIONS
    --period PERIOD
        Specifies the historical data window.
        
        Valid formats:
        * Period String: "1y", "2y", "5y", "10y", "max".
        * Date Range: "YYYY-MM-DD,YYYY-MM-DD".
        
        Default: "max" (To capture long-term valuation cycles).

    --no-show
        Prevents the script from automatically opening the generated HTML 
        chart in the web browser. Useful for batch generation.
        Default: Disabled (Chart opens automatically).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/TICKER/YYYY-MM-DD/

    Generated File:
    * {TICKER}_pe_regression.html

VISUALIZATION DETAILS
    * Candlesticks: Standard OHLC price action.
    * Black Line: The Trailing P/E Ratio over the selected period.
    * Blue Line: The Linear Regression trendline of the P/E Ratio.
    * Dotted Bands: Standard Deviation bands (0.5 to 4.0 sigma) deviating 
      from the P/E trend. Use these to spot extreme valuation outliers.

EXAMPLES
    1. Standard Valuation Check
       Analyze Apple (AAPL) over its entire available history to see where 
       current valuation sits relative to its long-term mean.
       
       $ python PE_candlesticks.V3.py AAPL

    2. Recent Trend Analysis
       Analyze Microsoft (MSFT) over the last 2 years.
       
       $ python PE_candlesticks.V3.py MSFT --period 2y

    3. Specific Cycle Analysis
       Analyze NVIDIA (NVDA) during the 2020-2022 period.
       
       $ python PE_candlesticks.V3.py NVDA --period "2020-01-01,2022-12-31"

    4. Batch Generation
       Generate the chart for Tesla (TSLA) but do not open the browser.
       
       $ python PE_candlesticks.V3.py TSLA --no-show

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
