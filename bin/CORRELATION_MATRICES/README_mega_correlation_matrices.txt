NAME
    mega_unified_mega_corelation_matrices.v3.py - Market-Wide Correlation Engine

SYNOPSIS
    python mega_unified_mega_corelation_matrices.v3.py [OPTIONS]

DESCRIPTION
    This powerful analytics engine calculates, clusters, and visualizes correlation 
    matrices for massive sets of financial assets. 
    
    Unlike simple one-to-one tools, this script manages entire asset "Universes" 
    (Equities, Crypto, Commodities, Forex, Fixed Income) to reveal systemic 
    relationships, cluster behaviors, and time-varying correlations.

    Key Features:
    1. Multi-Mode Correlation:
       * Standard Heatmap: The classic view of all assets against all assets.
       * PCA-Ordered: Reorders the matrix based on the first Principal Component 
         to reveal the dominant market factor.
       * Hierarchical Clustering: Uses dendrogram logic to group similar assets 
         together (e.g., grouping all Energy stocks).
    
    2. Time-Slicing:
       * Can generate separate correlation matrices for every calendar year 
         in the dataset to show how relationships evolve (e.g., 2022 vs 2023).
    
    3. Performance Dashboards:
       * Generates time-series plots comparing the relative performance (%) of 
         all assets, grouped by volatility.

    Dependencies:
    * data_retrieval.py (Canonical data source)
    * chartlib_unified.py (Math & Visualization engine)

OPTIONS (Data Scope)
    --period PERIOD
        Lookback window for the correlation analysis.
        Default: "5y".

    --dates "START,END"
        Explicit date range. Overrides --period.

    --freq {D, W, M}
        Resampling frequency. D=Daily, W=Weekly, M=Monthly.
        Using Monthly data often reveals stronger macro correlations.
        Default: "D".

    --universe {core, categories, all}
        Predefined lists of assets to analyze.
        * core: ~20 major ETFs and Indices (SPY, GLD, TLT, etc.).
        * categories: A broad set of sector ETFs and commodities.
        * all: Everything combined (~50+ tickers).
        Default: "all".

    --assets "TICKER1,TICKER2..."
        Manually specify a list of tickers to analyze, overriding the universe.

OPTIONS (Analysis & Output)
    --split-correlations
        Generates additional heatmaps isolating only "High Positive" (>0.5) 
        and "High Negative" (<-0.5) correlations.
        Default: Disabled.

    --time-slices
        Generates a separate correlation heatmap for every year in the dataset.
        Default: Disabled.

    --generate-pdf
        Compiles all generated charts into a single "mega_report.pdf".
        Requires the 'fpdf' library.
        Default: Disabled.

    --no-open
        Prevents the script from opening browser tabs. 
        Default: Disabled.

    --out-root PATH
        Directory to save all outputs. 
        Default: ./MEGA_CORR/{YYYY-MM-DD}/

EXAMPLES
    1. The Macro View (Default)
       Analyze the entire "All" universe over 5 years using Daily returns.
       
       $ python mega_unified_mega_corelation_matrices.v3.py

    2. Crypto vs. Tech
       Analyze specific assets to see if Bitcoin correlates with Nvidia.
       
       $ python mega_unified_mega_corelation_matrices.v3.py --assets "BTC-USD,ETH-USD,NVDA,QQQ,COIN" --period 1y

    3. Long-Term Macro Study
       Analyze Monthly correlations over 10 years, splitting by year to see trends.
       
       $ python mega_unified_mega_corelation_matrices.v3.py --period 10y --freq M --time-slices

    4. PDF Report Generation
       Run the full analysis on the Core universe and compile a PDF report.
       
       $ python mega_unified_mega_corelation_matrices.v3.py --universe core --generate-pdf --no-open

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
