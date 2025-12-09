NAME
    projectiles.v5.with.inflection.point.py - Physics-Based Trend Analyzer

SYNOPSIS
    python projectiles.v5.with.inflection.point.py TICKER [OPTIONS]

DESCRIPTION
    This experimental tool applies kinematic projectile physics to financial 
    charts to forecast trend exhaustion.

    The script identifies a recent "Inflection Point" (the lowest low within 
    a specified lookback window) and attempts to fit a parabolic trajectory 
    curve to the subsequent price highs. 

    If the price action resembles a launched projectile (concave down 
    parabola), the script projects the curve forward to identify the 
    theoretical "Apex" (Peak Price and Date) where momentum reaches zero 
    before gravity (mean reversion) takes over.

    Dependencies:
    * data_retrieval.py (Canonical data source)
    * scipy (Polynomial fitting)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., NVDA, SMCI).

OPTIONS
    --period PERIOD
        Specifies the total historical data window loaded for context.
        Valid values: 1y, 2y, 5y, max.
        Default: "2y".

    --lookback INT
        The number of days to scan backwards to find the "Launch Point" 
        (the significant low from which the current trend began).
        Default: 90 (Approx. 1 quarter).

    --no-show
        Prevents the script from automatically opening the generated HTML 
        chart in the web browser.
        Default: Disabled (Chart opens automatically).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/TICKER/YYYY-MM-DD/

    Generated Files:
    * {TICKER}_projectile_analysis.png
    * {TICKER}_projectile_analysis.html

VISUALIZATION DETAILS
    * Magenta Dashed Line: The projected parabolic trajectory.
    * Green Annotation: The detected "Inflection Point" (Launch).
    * Purple Annotation: The forecasted "Predicted Apex" (Peak).
    * Blue Line: Linear Regression baseline.
    * Dotted Bands: Standard Deviation bands (0.5 to 4.0 sigma).

EXAMPLES
    1. Standard Trajectory Check
       Analyze NVIDIA (NVDA) using the default 90-day launch window.
       
       $ python projectiles.v5.with.inflection.point.py NVDA

    2. Long-Term Parabola
       Analyze Bitcoin (BTC-USD) looking back 180 days for the launch point, 
       loaded with 2 years of history.
       
       $ python projectiles.v5.with.inflection.point.py BTC-USD --period 2y --lookback 180

    3. Batch Generation
       Generate the analysis for Super Micro (SMCI) but do not open the 
       browser.
       
       $ python projectiles.v5.with.inflection.point.py SMCI --no-show

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
