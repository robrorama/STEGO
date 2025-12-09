NAME
    mas.lrc.vwap.multifan.peak.troughs.ath.v34.py - Adaptive VWAP & Technical Suite

SYNOPSIS
    python mas.lrc.vwap.multifan.peak.troughs.ath.v34.py --ticker TICKER [OPTIONS]

DESCRIPTION
    This script is a powerhouse of technical analysis, combining a proprietary
    "Adaptive Anchored VWAP" engine with a suite of other advanced indicators.

    Core Logic:
    1. Adaptive AVWAP: Unlike standard VWAP, this tool algorithmically 
       detects significant "Anchor Points" based on Market Structure (Peaks/Troughs),
       Momentum Inflections, and Volatility Regimes. It then draws a "Fan" of 
       VWAP lines originating from these critical turning points.

    2. Multi-Layer Overlays: The script can optionally layer on:
       * Linear Regression Channels (LRC)
       * All-Time High (ATH) Detection
       * Price Frequency Distribution (Horizontal Support/Resistance)
       * Multiple Moving Averages (SMA, EMA, SMMA, WMA, VWMA)
       * Dual-Mode Bollinger Bands (Long-term "Fan" vs Short-term "Bands")
       * Ichimoku Cloud
       * OHLC Price Dots (Classic detailed price view)

    Dependencies:
    * data_retrieval.py (Canonical data source)
    * options_data_retrieval.py (Optional cache warmer)

POSITIONAL ARGUMENTS
    None. All inputs are flags.

OPTIONS (General)
    --ticker TICKER
        (Required) Stock symbol to analyze.

    --period PERIOD
        Historical lookback window.
        Default: "2y".

    --start YYYY-MM-DD
    --end YYYY-MM-DD
        Specific date range (overrides --period).

    --plot
        (Required to see anything) Generates and opens the interactive HTML chart.

    --no-save
        Prevents saving the analysis CSV to disk.

OPTIONS (Indicator Toggles)
    --show-ohlc-dots
        [NEW] Overlays individual colored dots for Open, High, Low, Close, 
        and Midpoint on every candle. Useful for precise price action study.

    --show-lrc
        Enables Linear Regression Channel overlay.

    --lrc-length INT
        Lookback length for the LRC calculation. Default: 252.

    --show-ath
        Enables All-Time High detection line.

    --show-freq
        Enables Price Frequency (Mode) lines to spot high-volume price nodes.

    --show-ma
        Enables Multiple Moving Average overlay.

    --ma-type {SMA, EMA, SMMA, WMA, VWMA}
        Selects the calculation method for the moving averages.
        Default: EMA.

    --show-long-bbs (or --show-bb)
        Enables the "Long Term" Bollinger Band Fan (Length 50).

    --show-short-bbs
        Enables the "Short Term" Bollinger Bands (Lengths 5 & 20).

    --show-ichimoku
        Enables the Ichimoku Cloud overlay.

OPTIONS (Algorithm Tuning)
    --peaks-only
        If set, the AVWAP engine will ONLY anchor to major price Peaks and 
        Troughs, ignoring volatility or momentum triggers.

    --peak-order INT
        Sensitivity of peak detection. Higher = fewer, more significant peaks.
        Default: 10.

OUTPUT
    Output Directory: output/TICKER/YYYY-MM-DD/
    
    Generated Files:
    * {TICKER}_analysis.html (The interactive chart)
    * {TICKER}_full_analysis.csv (Underlying data)

EXAMPLES
    1. The "Full Stack" (Max Info)
       Generate a chart for NVIDIA with VWAP, LRC, and Ichimoku Cloud.
       
       $ python mas.lrc.vwap.multifan.peak.troughs.ath.v34.py --ticker NVDA --plot --show-lrc --show-ichimoku

    2. Pure Price Action Study
       Analyze SPY using only the Adaptive VWAP and the new OHLC Dots for 
       precision.
       
       $ python mas.lrc.vwap.multifan.peak.troughs.ath.v34.py --ticker SPY --plot --show-ohlc-dots --peaks-only

    3. Trend Following Setup
       Analyze Tesla using a 200-day Linear Regression Channel and EMA ribbon.
       
       $ python mas.lrc.vwap.multifan.peak.troughs.ath.v34.py --ticker TSLA --plot --show-lrc --lrc-length 200 --show-ma --ma-type EMA

AUTHOR
    Michael Derby
    November 20, 2025
    ( STEGO FINANCIAL FRAMEWORK )
