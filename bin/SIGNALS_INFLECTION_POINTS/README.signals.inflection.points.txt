NAME
    identify.signal.inflection.points.V3.plotly.py - Volatility Signal Detector

SYNOPSIS
    python identify.signal.inflection.points.V3.plotly.py TICKER [OPTIONS]

DESCRIPTION
    This tool is designed to identify potential "inflection points" in price 
    action where volatility extremes coincide with trend exhaustion.

    It generates an interactive Plotly chart containing three primary 
    analytical layers:
    1. Bollinger Bands: To visualize standard volatility expansion/contraction.
       Specific markers (red/green triangles) are generated when price 
       pierces the upper or lower bands.
    2. Linear Regression Channel: A moving-window channel that provides a 
       mean-reversion baseline distinct from simple moving averages.
    3. Simple Moving Averages (SMA): Customizable trend baselines.

    Dependencies:
    * data_retrieval.py (Canonical data source)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., TSLA, AMD).

OPTIONS
    --period PERIOD
        Specifies the historical lookback period for the chart.
        Examples: 1mo, 3mo, 6mo, 1y, 2y, 5y, max.
        Default: 6mo.
        (Ignored if --start and --end are used).

    --start YYYY-MM-DD
    --end YYYY-MM-DD
        Specify an explicit date range for the analysis. Both flags must be 
        provided together.

    --ma INT [INT ...]
        A space-separated list of integers specifying the Simple Moving 
        Average periods to calculate and display.
        Default: 20 50 200.

    --bollinger-period INT
        The lookback window for the moving average used in Bollinger Band 
        calculations.
        Default: 20.

    --std-dev FLOAT
        The number of standard deviations for the Bollinger Band width. 
        Increasing this reduces signal sensitivity (fewer but potentially 
        more significant signals).
        Default: 2.0.

    --regression-period INT
        The rolling window size used to calculate the Linear Regression 
        Channel midpoints and deviations.
        Default: 21.

OUTPUT
    The script generates a temporary interactive HTML visualization in the 
    default web browser. It does not persistently save files to disk by 
    default, focusing on immediate interactive analysis.

EXAMPLES
    1. Standard Analysis
       Analyze Apple (AAPL) over the last 6 months with default settings.
       
       $ python identify.signal.inflection.points.V3.plotly.py AAPL

    2. Long-Term Trend Check
       Analyze SPY over 2 years, showing the 50, 100, and 200-day SMAs.
       
       $ python identify.signal.inflection.points.V3.plotly.py SPY --period 2y --ma 50 100 200

    3. High-Volatility Sensitivity
       Analyze Tesla (TSLA) using tighter Bollinger Bands (1.5 std dev) to 
       catch more minor volatility breakouts.
       
       $ python identify.signal.inflection.points.V3.plotly.py TSLA --std-dev 1.5

    4. Custom Date & Regression
       Analyze NVIDIA (NVDA) for Q1 2024, using a faster 14-day linear 
       regression channel.
       
       $ python identify.signal.inflection.points.V3.plotly.py NVDA --start 2024-01-01 --end 2024-03-31 --regression-period 14

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
