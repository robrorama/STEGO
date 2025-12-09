NAME
    VolumeByPrice.V1.py - Horizontal Volume Profile Visualizer

SYNOPSIS
    python VolumeByPrice.V1.py TICKER [OPTIONS]

DESCRIPTION
    This tool generates a specialized "Volume by Price" (VBP) analysis, 
    often referred to as a Volume Profile. 

    Unlike standard volume charts that show volume over *time* (vertical bars 
    at the bottom), this script calculates volume distribution over *price levels* (horizontal bars on the right). This helps identify significant price zones 
    of high liquidity (Point of Control) and low liquidity (Volume Gaps) 
    which often act as Support and Resistance.

    The output includes:
    1. Main Candlestick Chart: Standard OHLC price action.
    2. Volume Chart: Standard volume over time.
    3. VBP Histogram: A horizontal bar chart aligned with the price axis, 
       showing how much volume was transacted at specific price ranges.

    Dependencies:
    * data_retrieval.py (Canonical data source)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., AMD, INTU).

OPTIONS
    --period PERIOD
        Specifies the historical lookback period.
        Valid values: 1mo, 3mo, 6mo, 1y, 2y, 5y, max.
        Default: 1y.

    --start YYYY-MM-DD
    --end YYYY-MM-DD
        Specify an explicit date range for the analysis. Both flags must be 
        provided together. Overrides --period.

    --bins INT
        The resolution of the Volume Profile. This determines how many 
        horizontal price "buckets" the price range is divided into.
        Higher values = finer detail (more bars).
        Lower values = broader aggregation (fewer bars).
        Default: 24.

    --no-show
        Prevents the script from automatically opening the generated HTML 
        chart in the web browser.
        Default: Disabled (Chart opens automatically).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/TICKER/YYYY-MM-DD/

    Generated File:
    * {TICKER}_VolumeByPrice.html

EXAMPLES
    1. Standard Profile
       View the Volume Profile for Microsoft (MSFT) over the last year.
       
       $ python VolumeByPrice.V1.py MSFT

    2. High Resolution Analysis
       Analyze Amazon (AMZN) with 50 price bins to see very specific support 
       levels.
       
       $ python VolumeByPrice.V1.py AMZN --bins 50

    3. Long-Term Liquidity
       Analyze the SPY ETF over 5 years to find major historical value zones.
       
       $ python VolumeByPrice.V1.py SPY --period 5y

    4. Specific Event Window
       Analyze Tesla (TSLA) during a specific month to see where volume 
       accumulated.
       
       $ python VolumeByPrice.V1.py TSLA --start 2023-01-01 --end 2023-01-31

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
