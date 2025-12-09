NAME
    gamma.charmm.visualizer.v4.py - Advanced Option Greek Exposure Analyzer

SYNOPSIS
    python gamma.charmm.visualizer.v4.py TICKER [OPTIONS]

DESCRIPTION
    This specialized derivatives tool calculates and visualizes the aggregate 
    Gamma and Vanna exposure across the entire option chain for a given ticker.

    Market Makers (dealers) must hedge their exposure. Understanding their 
    aggregate Gamma position allows traders to predict whether dealers will 
    act as liquidity providers (suppressing volatility in positive gamma) or 
    liquidity takers (accelerating volatility in negative gamma).

    Core Calculations:
    1. Spot Price Estimation: Uses a robust "Option D" method (nearest 0.5 delta 
       options) to infer the underlying price per expiration date, even for 
       illiquid chains.
    2. Greek Calculation: Computes Black-Scholes Gamma and Vanna for every 
       single option contract.
    3. Exposure Aggregation: Multiplies the Greeks by Open Interest and Contract 
       Size to determine the total dollar-gamma exposure at each strike.

    Dependencies:
    * options_data_retrieval.py (Canonical option chain source)
    * matplotlib (Static 3D surface plots)

POSITIONAL ARGUMENTS
    TICKER
        The stock ticker symbol to analyze (e.g., SPY, QQQ).

OPTIONS
    --ensure_remote
        Forces the script to check for and download the latest option chains 
        from the remote source before processing. 
        Default: Disabled (Uses local cache only).

    --risk_free FLOAT
        Sets the risk-free interest rate (r) for the Black-Scholes model.
        Default: 0.05 (5%).

    --div_yield FLOAT
        Sets the dividend yield (q) for the Black-Scholes model.
        Default: 0.0 (0%).

    --contract_size INT
        The multiplier for the option contracts.
        Default: 100 (Standard US Equities).

    --max_expirations INT
        Limits the analysis to the first N upcoming expiration dates. 
        Useful for focusing on near-term gamma risk (OpEx).
        Default: None (All available expirations).

    --strike_slice PRICE
        Generates a specific 2D "Term Structure" chart for Gamma/Vanna 
        exposure at the specified strike price across all expiration dates.

    --animate
        Creates an MP4 video animation ("gamma_sweep.mp4") sweeping through 
        all expiration dates to show how the Gamma Exposure profile evolves 
        over time.

    --outdir PATH
        Overrides the default output directory.

OUTPUT
    The script creates a dated output directory: 
    /dev/shm/data/YYYY-MM-DD/{TICKER}_OPTIONS_GAMMA_VANNA/

    Generated Files:
    * gamma_surface.png: 3D Surface plot of Gamma Exposure (Strike vs. Expiry).
    * gamma_heatmap.png: 2D Heatmap of Gamma Exposure.
    * vanna_surface.png: 3D Surface plot of Vanna Exposure.
    * vanna_heatmap.png: 2D Heatmap of Vanna Exposure.
    * gamma_term_{PRICE}.png: (Optional) Term structure slice.
    * gamma_sweep.mp4: (Optional) Video animation.

EXAMPLES
    1. Standard Gamma Check
       Analyze SPY using existing cached data.
       
       $ python gamma.charmm.visualizer.v4.py SPY

    2. Fresh Data Download
       Analyze NVDA, ensuring the latest option chains are downloaded first.
       
       $ python gamma.charmm.visualizer.v4.py NVDA --ensure_remote

    3. Near-Term Risk
       Analyze QQQ but only look at the next 4 expiration dates.
       
       $ python gamma.charmm.visualizer.v4.py QQQ --max_expirations 4

    4. Deep Dive
       Analyze Tesla (TSLA) with a custom interest rate (4.5%), generate a 
       term structure slice at the $250 strike, and create an animation.
       
       $ python gamma.charmm.visualizer.v4.py TSLA --risk_free 0.045 --strike_slice 250 --animate

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
