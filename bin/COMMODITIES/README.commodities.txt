NAME
    commodities.v3.py - Unified Global Commodities Analysis Toolkit

SYNOPSIS
    python commodities.v3.py [OPTIONS]

DESCRIPTION
    This script generates a comprehensive suite of interactive visualizations 
    for tracking the global commodities market. It automatically fetches data 
    for a wide basket of futures contracts (Energy, Metals, Agriculture, 
    Livestock) and presents them in three distinct analytical views.

    The tool uses a centralized data retrieval system to ensure efficient 
    caching and standardized output locations.

    Visualization Modes:
    1. Grouped Sector Grid (2x2): Organizes commodities into Energy, Metals, 
       Agriculture, and Livestock quadrants. All prices are Z-score 
       normalized, allowing for direct correlation comparison between assets 
       with vastly different nominal prices (e.g., Gold vs. Natural Gas).
    2. Technical Grid (LRC): A massive grid of individual charts for every 
       commodity, featuring 50/200 EMAs and Log-Linear Regression Channels 
       (50-day & 144-day periods) to identify trend deviations.
    3. Combined Overlay: A single, dense chart overlaying every commodity 
       (normalized) to identify macro-level divergences and outliers.

    Dependencies:
    * data_retrieval.py (Canonical data source)

OPTIONS
    --mode {plotly-grouped, plotly-lrc, plotly-combined, all}
        Selects which chart type(s) to generate.
        * plotly-grouped: The 2x2 sector comparison grid.
        * plotly-lrc: The technical analysis grid with channels.
        * plotly-combined: The "spaghetti chart" of all assets overlaid.
        * all: (Default) Generates and opens all three charts.

    --period PERIOD
        Specifies the historical lookback window.
        Valid values: 1mo, 6mo, 1y, 2y, 5y, 10y, max.
        Default: 1y.

    --dates "START,END"
        Overrides the period with a specific date range.
        Format: "YYYY-MM-DD,YYYY-MM-DD".

    --no-open
        Prevents the script from automatically opening the generated HTML 
        charts in the web browser. Useful for batch reports.

    --log-channels
        Forces the Linear Regression Channels (LRC) to be calculated on 
        log-transformed price data. This is generally better for long-term 
        trends or volatile assets like Crypto/Commodities.
        Default: Enabled (True).

OUTPUT
    The script utilizes the canonical directory structure managed by 
    data_retrieval.py.

    Output Directory: output/COMMODITIES/YYYY-MM-DD/

    Generated Files:
    * plotly_grouped_{TAG}.html
    * plotly_lrc_{TAG}.html
    * plotly_combined_{TAG}.html

ASSETS TRACKED
    Energy:      Crude Oil (CL=F), Brent (BZ=F), Nat Gas (NG=F), 
                 Heating Oil (HO=F), RBOB Gas (RB=F)
    Metals:      Gold (GC=F), Silver (SI=F), Copper (HG=F), 
                 Platinum (PL=F), Palladium (PA=F)
    Agriculture: Corn (ZC=F), Wheat (ZW=F), Soybeans (ZS=F), 
                 Soybean Oil (ZL=F), Sugar (SB=F), Coffee (KC=F), 
                 Cocoa (CC=F), Cotton (CT=F), Orange Juice (OJ=F)
    Livestock:   Live Cattle (LE=F), Lean Hogs (HE=F), Feeder Cattle (GF=F)

EXAMPLES
    1. The "Macro View" (Default)
       Generate all three dashboard styles for the last year.
       
       $ python commodities.v3.py

    2. Long-Term Trend Analysis
       Generate only the technical grid (LRC) for a 5-year lookback to see 
       long-term regression channels.
       
       $ python commodities.v3.py --mode plotly-lrc --period 5y

    3. Inflation Shock Study
       Analyze all commodities during the 2022 inflation spike window.
       
       $ python commodities.v3.py --dates "2022-01-01,2022-12-31"

    4. Sector Correlation Check
       Generate the grouped sector view to see how Energy correlates with 
       Metals over the last 6 months.
       
       $ python commodities.v3.py --mode plotly-grouped --period 6mo

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
