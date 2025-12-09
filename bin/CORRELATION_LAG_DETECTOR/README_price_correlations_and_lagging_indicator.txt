NAME
    price.correlations.and.lagging.indicator.two.assets.v9.py - Correlation & Lead-Lag Analyzer

SYNOPSIS
    python price.correlations.and.lagging.indicator.two.assets.v9.py --ticker1 T1 --ticker2 T2

DESCRIPTION
    This script performs a deep statistical analysis of the relationship 
    between two financial assets. It is specifically designed to uncover 
    "Lead-Lag" relationships where one asset's price movement predicts 
    the other's with a time delay.

    The tool computes rolling cross-correlations over a sliding window 
    (default 252 trading days) to identify periods of high correlation. 
    It then determines the optimal "Lag" (in days) that maximizes this 
    correlation, effectively telling you if Asset A leads Asset B and by 
    how many days.

    Features:
    1. Rolling Correlation Matrix: Identifies the specific time windows 
       where the two assets were most strongly linked.
    2. Optimal Lag Detection: Calculates the shift in days (-30 to +30) 
       that produces the highest Pearson correlation coefficient.
    3. Interactive Lag Slider: A Plotly chart allowing you to manually 
       shift one asset's price curve against the other to visually confirm 
       the lag.
    4. Smoothed Analysis: Applies various SMA filters (5d to 300d) to 
       remove noise and reveal long-term macro correlations.

    Dependencies:
    * data_retrieval.py (Canonical data source)
    * scipy.stats (Pearson correlation)

OPTIONS
    --ticker1 TICKER
        (Required) The first asset symbol (e.g., GC=F for Gold).
        Acts as the "Reference" asset.

    --ticker2 TICKER
        (Required) The second asset symbol (e.g., SI=F for Silver).
        Acts as the "Comparison" asset.

OUTPUT
    The script opens 5 interactive Plotly charts in your web browser:
    
    1. Cross-Correlation vs. Lag: A curve showing how correlation improves 
       as you shift the time difference.
    2. Price History with Highlights: The full price history of both assets, 
       with Green highlights for highly correlated periods and Red for 
       divergent periods.
    3. Rebased Overlay: A normalized comparison of price performance during 
       the most correlated window.
    4. Interactive Lag Slider: A tool to manually shift time and observe 
       alignment.
    5. Smoothed Analysis: Similar to #4, but with selectable Moving Average 
       smoothing to filter daily volatility.

    It also prints a statistical summary table to the console, detailing 
    the Top 5 and Bottom 5 correlated time windows.

EXAMPLES
    1. Precious Metals Lead-Lag
       Determine if Gold leads Silver or vice versa.
       
       $ python price.correlations.and.lagging.indicator.two.assets.v9.py --ticker1 GC=F --ticker2 SI=F

    2. Energy Correlation
       Analyze the relationship between Crude Oil and Heating Oil.
       
       $ python price.correlations.and.lagging.indicator.two.assets.v9.py --ticker1 CL=F --ticker2 HO=F

    3. Equity vs. Yield
       Check for lead-lag effects between the S&P 500 and 10-Year Treasury Yields.
       
       $ python price.correlations.and.lagging.indicator.two.assets.v9.py --ticker1 SPY --ticker2 ^TNX

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
