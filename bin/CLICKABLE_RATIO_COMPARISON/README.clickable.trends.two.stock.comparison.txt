NAME
    clickable_trends_compare_two.py - Comparative Stock Analysis & Ratio Charting

SYNOPSIS
    python clickable_trends_compare_two.py TICKER1 TICKER2 [OPTIONS]

DESCRIPTION
    This command-line tool generates a unified comparative analysis of two 
    financial assets. It focuses on the relative strength/weakness between 
    two tickers (e.g., Gold vs. Dollar, Tech vs. Energy).

    The script provides three distinct visualization modes:
    
    1. Normalized Overlay (Z-Score): 
       Plots both tickers on a standardized scale (Z-score) to visualize 
       correlation and divergence over time, regardless of their absolute 
       price differences.

    2. Ratio Analysis (TICKER1 / TICKER2):
       Calculates the ratio between the two assets and plots it with:
       * Linear Regression Channels (Standard Deviation Bands)
       * Exponential Moving Averages (EMAs: 20, 50, 100, 200, 300)
       * Automatic Support & Resistance Levels (last 30 periods)
       * Automated Trendlines connecting major Peaks and Troughs

    3. Interactive "Clickable" Mode (Matplotlib):
       Launches a specialized Matplotlib window allowing the user to 
       manually draw custom trendlines on the Ratio chart by clicking 
       any two points.

    Dependencies:
    * data_retrieval.py (Canonical data source)

POSITIONAL ARGUMENTS
    TICKER1
        The primary asset symbol (numerator of the ratio).
        Example: SPY

    TICKER2
        The secondary asset symbol (denominator of the ratio).
        Example: QQQ

OPTIONS
    --period PERIOD
        Specifies the historical data window.
        Formats:
        * Period String: "1mo", "3mo", "6mo", "1y", "2y", "5y", "max".
        * Date Range: "YYYY-MM-DD,YYYY-MM-DD".
        Default: "1y".

    --clickable
        Enables the interactive Matplotlib window. This allows you to click 
        two points on the chart to draw a custom line, useful for identifying 
        non-standard trend geometries.
        Default: Disabled.

    --all-timeframes
        Automatically generates and opens Ratio Analysis charts for a 
        predefined set of timeframes (defined by --timeframes). Useful for 
        seeing the fractal nature of the ratio trend.
        Default: Disabled.

    --timeframes LIST
        A comma-separated list of periods to use when --all-timeframes is 
        active.
        Default: "1mo,3mo,6mo,1y,max".

    --no-normalized
        Skip the generation of the "Normalized Overlay" (Z-score) chart.
        Default: Disabled (Chart is generated).

    --no-ratio
        Skip the generation of the standard "Ratio Analysis" chart.
        Default: Disabled (Chart is generated).

    --no-show
        Prevent any charts (Plotly or Matplotlib) from opening on the screen. 
        Useful for background batch processing.
        Default: Disabled.

    --no-save
        Prevent the script from saving static PNG images to the output folder.
        Default: Disabled (Images are saved).

    --renderer RENDERER
        Specifies the Plotly renderer to use for browser-based charts.
        Default: "browser".

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/{TICKER1}_{TICKER2}/images/

    Generated Files:
    * {T1}_{T2}_normalized_{PERIOD}.png
    * {T1}_{T2}_ratio_{PERIOD}.png
    * (Interactive HTML charts are displayed in the browser but not 
       persisted as files by default in this script).

EXAMPLES
    1. Standard Comparison (Tech vs. S&P 500)
       Compare QQQ against SPY over the last year. Opens normalized and 
       ratio charts in the browser.
       
       $ python clickable_trends_compare_two.py QQQ SPY

    2. Interactive Trend Analysis (Gold vs. Dollar)
       Compare Gold (GLD) vs the Dollar Index (UUP) for the max available 
       history, launching the clickable window to draw long-term trends.
       
       $ python clickable_trends_compare_two.py GLD UUP --period max --clickable

    3. Multi-Timeframe Deep Dive (Oil vs. Gas)
       Generate ratio charts for CL=F (Crude) and NG=F (Natural Gas) across 
       1-month, 6-month, and 1-year views simultaneously.
       
       $ python clickable_trends_compare_two.py CL=F NG=F --all-timeframes --timeframes "1mo,6mo,1y"

    4. Specific Event Window (2022 Bear Market)
       Compare Bitcoin (BTC-USD) vs Ethereum (ETH-USD) during the 2022 
       calendar year.
       
       $ python clickable_trends_compare_two.py BTC-USD ETH-USD --period "2022-01-01,2022-12-31"

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
