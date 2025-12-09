NAME
    mega_charting_suite.v1.py - The Stego "Kitchen Sink" Financial Analysis Suite

SYNOPSIS
    python 
    mega_charting_suite.v1.py --ticker TICKER [OPTIONS]

DESCRIPTION
    This is the flagship orchestrator script for the Stego Financial Framework. 
    It executes a massive, unified analysis of a single stock ticker by 
    triggering the generation of dozens of distinct financial charts and 
    indicators.

    Unlike single-purpose scripts, this tool runs EVERYTHING:
    1. Matplotlib Static Charts: LRC, Outliers, Streaks, Derivatives, FFT, etc.
    2. Plotly Interactive Charts: Ichimoku, Bollinger, Moving Averages, 
       Multi-term Regression, Hyperbolic Transforms, etc.
    3. PDF Report Assembly: Automatically compiles all generated static charts 
       into a single, easy-to-share PDF document.
    4. Browser Automation: Automatically opens all interactive charts in your 
       web browser for immediate review.

    Dependencies:
    * data_retrieval.py (Canonical data source)
    * chartlib_unified.py (Core plotting library)

POSITIONAL ARGUMENTS
    None. All inputs are handled via flags.

OPTIONS
    --ticker TICKER
        (Required) The primary stock ticker symbol to analyze (e.g., SPY, NVDA).

    --period PERIOD
        The lookback period for historical data.
        Valid values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        Default: 6mo.
        (Ignored if --start and --end are provided).

    --start YYYY-MM-DD
        Explicit start date for the analysis window.

    --end YYYY-MM-DD
        Explicit end date for the analysis window.

    --ratio-with TICKER_B
        Triggers a "Ratio Suite" analysis, generating additional charts that 
        compare the performance of the primary TICKER against TICKER_B 
        (e.g., comparing AAPL performance relative to SPY).

    --include-all-patterns
        By default, the script scans for a curated list of major candlestick 
        patterns. Use this flag to force the underlying TA-Lib engine to 
        compute and visualize ALL available candlestick patterns (computationally 
        intensive).

    --outdir PATH
        Override the canonical output directory. By default, files are saved 
        to 'output/TICKER/YYYY-MM-DD/'. Use this flag to force a custom path.

    --no-open
        Suppress the automatic opening of browser tabs. Essential for automated 
        server-side scripts or batch processing.

    --no-pdf
        Skip the assembly of the final PDF report. Useful if you only want 
        interactive HTML charts and want to save processing time.

    --max-tabs INT
        Limit the maximum number of browser tabs the script will attempt to 
        open.
        Default: 128.

    --tab-delay-ms INT
        The delay (in milliseconds) between opening each browser tab. Increase 
        this if your browser freezes when opening many charts at once.
        Default: 60.

OUTPUT
    The script generates a massive array of files in the target directory:
    
    * PDF Report: {TICKER}_ALL_CHARTS.pdf (Aggregated static views)
    * Interactive HTML: dozens of .html files for Plotly charts.
    * Static Images: dozens of .png files used for the PDF.
    * Run Log: run_report.txt and run_report.png.

EXAMPLES
    1. The Standard "Daily Driver"
       Analyze SPY for the last 6 months, open all charts, and build a PDF.
       
       $ python 
    mega_charting_suite.v1.py --ticker SPY

    2. The "Deep Dive"
       Analyze NVDA vs AMD (Ratio) over the last 2 years, checking every 
       single candlestick pattern.
       
       $ python 
    mega_charting_suite.v1.py --ticker NVDA --period 2y --ratio-with AMD --include-all-patterns

    3. The "Bot Run"
       Analyze Tesla for the last quarter, save the PDF, but do NOT open any 
       browser windows (headless mode).
       
       $ python 
    mega_charting_suite.v1.py --ticker TSLA --period 3mo --no-open

    4. The "Specific Event"
       Analyze Microsoft during the 2023 calendar year.
       
       $ python 
    mega_charting_suite.v1.py --ticker MSFT --start 2023-01-01 --end 2023-12-31

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
