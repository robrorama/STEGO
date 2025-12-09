# SCRIPTNAME: ok.options_vol_dealer_regime.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from scipy.stats import norm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
NEUTRAL_VOL_WINDOW = 252  # 1 year roughly
RISK_FREE_RATE = 0.045     # Approx 4.5%

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# CLASS 1: DataIngestion
# --------------------------------------------------------------------------------
class DataIngestion:
    """
    Solely responsible for:
    - Downloading data via yfinance.
    - Saving/loading local CSVs (caching).
    - Running all data cleaning and sanitization.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()

    def get_daily_ohlc(self, days: int = 365) -> pd.DataFrame:
        """Get daily history. Tries cache first, then API."""
        filename = os.path.join(DATA_DIR, f"{self.ticker}_daily.csv")
        
        # 1. Try Cache
        if os.path.exists(filename):
            logger.info(f"Loading daily data from cache: {filename}")
            try:
                df = pd.read_csv(filename, index_col=0)
                sanitized_df = self._sanitize_df(df)
                if not sanitized_df.empty:
                    # Check if cache is stale (older than 1 day)
                    last_date = sanitized_df.index[-1]
                    if (datetime.now() - last_date).days < 2:
                        return sanitized_df
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Downloading fresh.")

        # 2. Download Fresh
        logger.info(f"Downloading daily data for {self.ticker}...")
        time.sleep(1) # Rate limit protection
        
        # Calculate start date
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        try:
            df = yf.download(self.ticker, start=start_date, group_by='column', progress=False)
            sanitized_df = self._sanitize_df(df)
            
            # Cold Start / Shadow Backfill
            if sanitized_df.empty:
                logger.warning("Downloaded data empty. Attempting shadow backfill.")
                sanitized_df = self._backfill_shadow_history(self.ticker)
            
            # Save to Cache
            sanitized_df.to_csv(filename)
            return sanitized_df
            
        except Exception as e:
            logger.error(f"Failed to download daily data: {e}")
            return pd.DataFrame()

    def get_intraday_ohlc(self, interval: str = '5m') -> pd.DataFrame:
        """Get intraday history. Tries cache first, then API."""
        filename = os.path.join(DATA_DIR, f"{self.ticker}_intraday_{interval}.csv")
        
        # 1. Try Cache
        if os.path.exists(filename):
            logger.info(f"Loading intraday data from cache: {filename}")
            try:
                df = pd.read_csv(filename, index_col=0)
                sanitized_df = self._sanitize_df(df)
                # Intraday cache expires faster (e.g., 1 hour), but for simplicity we use it if present
                if not sanitized_df.empty:
                     # Simple check: if last data point is from today, use it.
                    if sanitized_df.index[-1].date() == datetime.now().date():
                        return sanitized_df
            except Exception:
                pass

        # 2. Download Fresh
        logger.info(f"Downloading intraday {interval} data for {self.ticker}...")
        time.sleep(1)
        
        try:
            # yfinance limits intraday history (usually 60d for 5m, but realistically last 5-7 days for small intervals)
            df = yf.download(self.ticker, period="5d", interval=interval, group_by='column', progress=False)
            sanitized_df = self._sanitize_df(df)
            
            if not sanitized_df.empty:
                sanitized_df.to_csv(filename)
            return sanitized_df
            
        except Exception as e:
            logger.error(f"Failed to download intraday data: {e}")
            return pd.DataFrame()

    def get_options_chain(self) -> Tuple[pd.DataFrame, float]:
        """
        Fetch current options chain for nearest expiries.
        Returns (options_df, spot_price).
        Does NOT cache options chain to disk heavily to ensure freshness, 
        but respects rate limits.
        """
        logger.info(f"Fetching options chain for {self.ticker}...")
        time.sleep(1)
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            
            # Get current price robustly
            history = ticker_obj.history(period="1d")
            if history.empty:
                logger.warning("Could not get current spot price.")
                return pd.DataFrame(), 0.0
            spot_price = history['Close'].iloc[-1]
            
            expirations = ticker_obj.options
            if not expirations:
                logger.warning("No options expirations found.")
                return pd.DataFrame(), spot_price

            all_opts = []
            # Grab first 4-5 expiries
            target_expiries = expirations[:5]
            
            for date in target_expiries:
                try:
                    time.sleep(0.5) # Gentle on the API
                    opt = ticker_obj.option_chain(date)
                    calls = opt.calls
                    puts = opt.puts
                    
                    calls['type'] = 'call'
                    puts['type'] = 'put'
                    
                    # Merge
                    chain = pd.concat([calls, puts], ignore_index=True)
                    chain['expiry'] = date
                    all_opts.append(chain)
                except Exception as e:
                    logger.warning(f"Failed to fetch expiry {date}: {e}")
                    continue

            if not all_opts:
                return pd.DataFrame(), spot_price

            full_chain = pd.concat(all_opts, ignore_index=True)
            return full_chain, spot_price

        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return pd.DataFrame(), 0.0

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """The Universal Fixer for yfinance data."""
        if df.empty:
            return df

        df = df.copy()

        # 1. Handle MultiIndex Columns (The "Swap Levels" Fix)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' is in level 0
            if 'Close' not in df.columns.get_level_values(0):
                if 'Close' in df.columns.get_level_values(1):
                    # Swap levels so attributes are on top
                    df = df.swaplevel(0, 1, axis=1)
            
            # Flatten columns: ('Close', 'SPY') -> 'Close'
            # We prefer simple names 'Open', 'High', 'Low', 'Close', 'Volume'
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # If the level 0 is the attribute (Close, Open, etc)
                    if col[0] in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                        new_cols.append(col[0])
                    else:
                        new_cols.append(f"{col[0]}_{col[1]}")
                else:
                    new_cols.append(col)
            df.columns = new_cols

        # 2. Strict Datetime Index
        # Reset index if it looks like integers (0, 1, 2...)
        if pd.api.types.is_integer_dtype(df.index):
            for col_name in ['Date', 'Datetime', 'Timestamp']:
                if col_name in df.columns:
                    df = df.set_index(col_name)
                    break
        
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notnull()] # Drop NaT

        # 3. Strip Timezones
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 4. Numeric Coercion
        # Ensure core columns exist and are numeric
        core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in core_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['Close']) # Minimal viability
        return df

    def _backfill_shadow_history(self, ticker: str) -> pd.DataFrame:
        """Create a synthetic history if standard download fails (Cold Start)."""
        logger.info("Generating Shadow Backfill history...")
        
        # Create dates
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=252, freq='B') # Business days
        
        # Create synthetic price walk
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, len(dates)) # 1% daily vol approx
        price_path = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame(index=dates)
        df['Open'] = price_path
        df['High'] = price_path * 1.01
        df['Low'] = price_path * 0.99
        df['Close'] = price_path
        df['Volume'] = 1000000
        
        # Ensure index is clean
        df.index.name = 'Date'
        return df


# --------------------------------------------------------------------------------
# CLASS 2: FinancialAnalysis
# --------------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Solely responsible for calculations, metrics, and domain logic.
    Never modifies raw data in place.
    """

    def __init__(self, daily_df: pd.DataFrame, intraday_df: pd.DataFrame, 
                 options_df: pd.DataFrame, spot_price: float):
        self._daily_ohlc = daily_df
        self._intraday_ohlc = intraday_df
        self._options_chain = options_df
        self._spot_price = spot_price

    def get_realized_vol_term_structure(self) -> pd.DataFrame:
        """Compute rolling realized volatility for 5d, 10d, 21d, 63d."""
        if self._daily_ohlc.empty:
            return pd.DataFrame()
        
        df = self._daily_ohlc.copy()
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        windows = [5, 10, 21, 63]
        results = pd.DataFrame(index=df.index)
        
        for w in windows:
            # Annualize: std * sqrt(252)
            col_name = f'RV_{w}d'
            results[col_name] = df['log_ret'].rolling(window=w).std() * np.sqrt(252) * 100
        
        results['Close'] = df['Close'] # Keep price for plotting
        return results.dropna()

    def get_shadow_gex_proxy(self) -> pd.DataFrame:
        """
        Compute Shadow GEX proxy.
        Formula: (Neutral_Vol - Realized_Vol_21d) * (Close * Volume_Rolling_Avg_Constant)
        """
        if self._daily_ohlc.empty:
            return pd.DataFrame()

        df = self._daily_ohlc.copy()
        
        # 1. Calc Realized Vol (21d)
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rv_21'] = df['log_ret'].rolling(window=21).std() * np.sqrt(252)
        
        # 2. Define Neutral Vol (Long term median or fixed 16% ie. vol 16)
        # Using a dynamic neutral: median of last year
        neutral_vol = df['rv_21'].median()
        if pd.isna(neutral_vol) or neutral_vol == 0:
            neutral_vol = 0.16 # Fallback
            
        # 3. Notional Proxy (scaled down for readability)
        # We smooth volume to avoid daily spikes distorting the GEX signal too much
        df['vol_smooth'] = df['Volume'].rolling(window=5).mean()
        notional_proxy = df['Close'] * df['vol_smooth'] / 1e9  # In Billions roughly
        
        # 4. Shadow GEX
        # If Realized Vol < Neutral, Dealer is Long Gamma (Supportive) -> Positive GEX
        # If Realized Vol > Neutral, Dealer is Short Gamma (Destabilizing) -> Negative GEX
        # Note: This is a heuristics-based proxy, not actual OI analysis.
        df['shadow_gex'] = (neutral_vol - df['rv_21']) * notional_proxy
        
        # 5. Z-Score for coloring
        df['gex_z'] = (df['shadow_gex'] - df['shadow_gex'].rolling(window=63).mean()) / df['shadow_gex'].rolling(window=63).std()
        
        return df[['Close', 'shadow_gex', 'gex_z']].dropna()

    def get_options_term_structure_and_skew(self) -> pd.DataFrame:
        """
        Analyze ATM IV and Skew for available expiries.
        """
        if self._options_chain.empty or self._spot_price <= 0:
            return pd.DataFrame()
            
        df = self._options_chain.copy()
        spot = self._spot_price
        
        # Ensure we have impliedVolatility column
        if 'impliedVolatility' not in df.columns:
            return pd.DataFrame()

        results = []
        
        for expiry, group in df.groupby('expiry'):
            # Days to expiry
            expiry_date = pd.to_datetime(expiry)
            dte = (expiry_date - datetime.now()).days
            if dte < 1: 
                continue # Skip expired or expiring today
                
            # Filter bad data
            group = group[group['impliedVolatility'] > 0]
            
            # --- ATM IV ---
            # Find strike closest to spot
            group['dist_to_spot'] = abs(group['strike'] - spot)
            atm_strike = group.loc[group['dist_to_spot'].idxmin(), 'strike']
            
            # Average IV of call and put at ATM strike
            atm_row = group[group['strike'] == atm_strike]
            if atm_row.empty:
                continue
            atm_iv = atm_row['impliedVolatility'].mean() * 100 # In percentage
            
            # --- Skew (25 Delta Proxy) ---
            # Rough proxy: 95% Moneyness (Put) vs 105% Moneyness (Call)
            # This approximates 25-30 delta often enough for a dashboard.
            put_strike_target = spot * 0.95
            call_strike_target = spot * 1.05
            
            # Find nearest put to target
            puts = group[group['type'] == 'put']
            calls = group[group['type'] == 'call']
            
            iv_put_wing = np.nan
            iv_call_wing = np.nan
            
            if not puts.empty:
                nearest_put_idx = (puts['strike'] - put_strike_target).abs().idxmin()
                iv_put_wing = puts.loc[nearest_put_idx, 'impliedVolatility'] * 100
                
            if not calls.empty:
                nearest_call_idx = (calls['strike'] - call_strike_target).abs().idxmin()
                iv_call_wing = calls.loc[nearest_call_idx, 'impliedVolatility'] * 100
            
            skew = np.nan
            if not np.isnan(iv_put_wing) and not np.isnan(iv_call_wing):
                # Skew = Put IV - Call IV (Put Skew) or vice versa. 
                # Standard Skew definition: OTM Put IV - OTM Call IV (Risk Reversal style)
                # If positive, Puts are more expensive (Bearish fear).
                skew = iv_put_wing - iv_call_wing

            results.append({
                'expiry': expiry,
                'dte': dte,
                'atm_iv': atm_iv,
                'skew_risk_reversal': skew
            })
            
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values('dte')
        return res_df

    def get_intraday_stats(self) -> Tuple[pd.DataFrame, float]:
        """
        Process intraday data for plotting and calc realized vol.
        """
        if self._intraday_ohlc.empty:
            return pd.DataFrame(), 0.0
            
        df = self._intraday_ohlc.copy()
        
        # Calculate intraday realized vol (annualized)
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Assuming 5m bars -> 78 bars per day approx (6.5 hours * 12)
        # Scale factor for annualizing from 5m frequency
        # But simpler: just std of returns * sqrt(number of 5m bars in a year)
        # 252 trading days * 78 bars = 19656
        bars_per_year = 252 * 78
        
        # Rolling window of last ~4 hours (48 bars)
        df['intraday_rv'] = df['log_ret'].rolling(window=48).std() * np.sqrt(bars_per_year) * 100
        
        current_intraday_rv = 0.0
        if not df['intraday_rv'].dropna().empty:
            current_intraday_rv = df['intraday_rv'].iloc[-1]
            
        return df, current_intraday_rv


# --------------------------------------------------------------------------------
# CLASS 3: DashboardRenderer
# --------------------------------------------------------------------------------
class DashboardRenderer:
    """
    Solely responsible for creating Plotly figures and the offline HTML page.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.figures = {} # Store figures by tab name

    def create_price_vol_charts(self, rv_df: pd.DataFrame):
        """Tab 1: Price History & Realized Vol Term Structure"""
        if rv_df.empty:
            return

        # Figure 1: Price + MAs
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=rv_df.index, open=rv_df['Open'] if 'Open' in rv_df else rv_df['Close'],
            high=rv_df['High'] if 'High' in rv_df else rv_df['Close'],
            low=rv_df['Low'] if 'Low' in rv_df else rv_df['Close'],
            close=rv_df['Close'], name='OHLC'
        ))
        # Add MAs
        for ma in [21, 63]:
            ma_series = rv_df['Close'].rolling(window=ma).mean()
            fig_price.add_trace(go.Scatter(x=rv_df.index, y=ma_series, mode='lines', name=f'{ma}d MA', line=dict(width=1)))

        fig_price.update_layout(title=f"{self.ticker} Price History", height=400, template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))

        # Figure 2: Realized Vol
        fig_rv = go.Figure()
        cols = ['RV_5d', 'RV_10d', 'RV_21d', 'RV_63d']
        for c in cols:
            if c in rv_df.columns:
                fig_rv.add_trace(go.Scatter(x=rv_df.index, y=rv_df[c], mode='lines', name=c))
        
        fig_rv.update_layout(title="Realized Volatility Term Structure", yaxis_title="Vol %", height=350, template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))

        self.figures['tab1'] = [fig_price, fig_rv]

    def create_term_structure_charts(self, ts_df: pd.DataFrame):
        """Tab 2: Options Term Structure & Skew"""
        if ts_df.empty:
            # Create empty placeholders with text
            fig_empty = go.Figure()
            fig_empty.add_annotation(text="No Options Data Available", showarrow=False)
            self.figures['tab2'] = [fig_empty, fig_empty]
            return

        # Figure 1: ATM IV Term Structure
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(x=ts_df['dte'], y=ts_df['atm_iv'], mode='lines+markers', name='ATM IV', line=dict(color='cyan', width=3)))
        fig_iv.update_layout(title="ATM Implied Volatility Term Structure", xaxis_title="Days to Expiry", yaxis_title="IV %", height=400, template="plotly_dark")

        # Figure 2: Skew
        fig_skew = go.Figure()
        # Color positive skew (Fear) red, negative skew (Greed/Call buying) green
        colors = ['red' if x > 0 else 'lime' for x in ts_df['skew_risk_reversal']]
        
        fig_skew.add_trace(go.Bar(
            x=ts_df['dte'], 
            y=ts_df['skew_risk_reversal'],
            marker_color=colors,
            name='Skew (Put - Call)'
        ))
        fig_skew.update_layout(title="Skew (OTM Put IV - OTM Call IV)", xaxis_title="Days to Expiry", yaxis_title="Skew Points", height=400, template="plotly_dark")

        self.figures['tab2'] = [fig_iv, fig_skew]

    def create_shadow_gex_charts(self, gex_df: pd.DataFrame):
        """Tab 3: Shadow GEX Regime"""
        if gex_df.empty:
            return

        # Figure 1: GEX Time Series
        fig_gex = go.Figure()
        fig_gex.add_trace(go.Bar(x=gex_df.index, y=gex_df['shadow_gex'], name='Shadow GEX Proxy', marker_color='orange'))
        fig_gex.update_layout(title="Shadow GEX Proxy (Dealer Gamma Exposure Est)", height=400, template="plotly_dark")

        # Figure 2: Regime Scatter (Price vs Z-Score)
        fig_regime = go.Figure()
        
        # Color points by Z-score
        fig_regime.add_trace(go.Scatter(
            x=gex_df.index, 
            y=gex_df['Close'],
            mode='markers',
            marker=dict(
                size=6,
                color=gex_df['gex_z'],
                colorscale='RdYlGn', # Red (Negative/Short Gamma) to Green (Positive/Long Gamma)
                showscale=True,
                colorbar=dict(title="GEX Z-Score")
            ),
            name='Regime'
        ))
        fig_regime.update_layout(title="Price Colored by Dealer Regime (Green=Stable/Long Gamma, Red=Volatile/Short Gamma)", height=400, template="plotly_dark")

        self.figures['tab3'] = [fig_gex, fig_regime]

    def create_intraday_charts(self, intra_df: pd.DataFrame, current_rv: float):
        """Tab 4: Intraday Analysis"""
        if intra_df.empty:
            return

        fig_intra = go.Figure()
        fig_intra.add_trace(go.Scatter(x=intra_df.index, y=intra_df['Close'], mode='lines', name='Price', line=dict(color='white')))
        
        # Add secondary axis for RV if available
        if 'intraday_rv' in intra_df.columns:
            fig_intra.add_trace(go.Scatter(x=intra_df.index, y=intra_df['intraday_rv'], mode='lines', name='Rolling RV', line=dict(color='yellow', width=1), yaxis='y2'))
        
        fig_intra.update_layout(
            title=f"Intraday Price Action (Current Rolling RV: {current_rv:.2f}%)",
            height=600,
            template="plotly_dark",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Intraday RV %", overlaying='y', side='right')
        )

        self.figures['tab4'] = [fig_intra]

    def generate_html(self, output_path: str):
        """Generate the fully self-contained HTML file."""
        import plotly.utils  # <--- Add this import here

        # 1. Get Plotly JS
        plotly_js = py_offline.get_plotlyjs()

        # 2. Serialize Figures to JSON
        # Structure: { 'tab1': [json_fig1, json_fig2], ... }
        plot_data = {}
        for tab, figs in self.figures.items():
            # FIXED: Use plotly.utils.PlotlyJSONEncoder instead of pd.io.json.PlotlyJSONEncoder
            plot_data[tab] = [json.loads(json.dumps(f, cls=plotly.utils.PlotlyJSONEncoder)) for f in figs]
        
        plot_data_json = json.dumps(plot_data)

        # 3. HTML Template (rest of the method remains the same...)
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.ticker} Volatility Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #111; color: #ddd; margin: 0; padding: 20px; }}
        .header {{ margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 10px; }}
        .header h1 {{ margin: 0; font-weight: 300; }}
        .tabs {{ overflow: hidden; border-bottom: 1px solid #444; margin-bottom: 20px; }}
        .tab-btn {{ background-color: #222; float: left; border: none; outline: none; cursor: pointer; padding: 14px 20px; transition: 0.3s; color: #888; font-size: 16px; }}
        .tab-btn:hover {{ background-color: #333; color: white; }}
        .tab-btn.active {{ background-color: #444; color: #00d2ff; border-bottom: 2px solid #00d2ff; }}
        .tab-content {{ display: none; animation: fadeEffect 0.5s; }}
        @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
        .chart-container {{ margin-bottom: 20px; border: 1px solid #333; padding: 10px; background: #1a1a1a; }}
    </style>
    <script>
        {plotly_js}
    </script>
</head>
<body>

<div class="header">
    <h1>{self.ticker} Options & Volatility Dashboard</h1>
    <small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
</div>

<div class="tabs">
    <button class="tab-btn" onclick="openTab(event, 'tab1')" id="defaultOpen">Price & Realized Vol</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">Term Structure & Skew</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">Shadow GEX Regime</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">Intraday</button>
</div>

<div id="tab1" class="tab-content">
    <div id="t1_c1" class="chart-container"></div>
    <div id="t1_c2" class="chart-container"></div>
</div>

<div id="tab2" class="tab-content">
    <div id="t2_c1" class="chart-container"></div>
    <div id="t2_c2" class="chart-container"></div>
</div>

<div id="tab3" class="tab-content">
    <div id="t3_c1" class="chart-container"></div>
    <div id="t3_c2" class="chart-container"></div>
</div>

<div id="tab4" class="tab-content">
    <div id="t4_c1" class="chart-container"></div>
</div>

<script>
    // Embedded Plot Data
    var plotData = {plot_data_json};

    function renderCharts() {{
        // Tab 1
        if (plotData.tab1 && plotData.tab1[0]) Plotly.newPlot('t1_c1', plotData.tab1[0].data, plotData.tab1[0].layout);
        if (plotData.tab1 && plotData.tab1[1]) Plotly.newPlot('t1_c2', plotData.tab1[1].data, plotData.tab1[1].layout);
        
        // Tab 2
        if (plotData.tab2 && plotData.tab2[0]) Plotly.newPlot('t2_c1', plotData.tab2[0].data, plotData.tab2[0].layout);
        if (plotData.tab2 && plotData.tab2[1]) Plotly.newPlot('t2_c2', plotData.tab2[1].data, plotData.tab2[1].layout);
        
        // Tab 3
        if (plotData.tab3 && plotData.tab3[0]) Plotly.newPlot('t3_c1', plotData.tab3[0].data, plotData.tab3[0].layout);
        if (plotData.tab3 && plotData.tab3[1]) Plotly.newPlot('t3_c2', plotData.tab3[1].data, plotData.tab3[1].layout);
        
        // Tab 4
        if (plotData.tab4 && plotData.tab4[0]) Plotly.newPlot('t4_c1', plotData.tab4[0].data, plotData.tab4[0].layout);
    }}

    function openTab(evt, tabName) {{
        var i, tabcontent, tablinks;
        
        // Hide all tab content
        tabcontent = document.getElementsByClassName("tab-content");
        for (i = 0; i < tabcontent.length; i++) {{
            tabcontent[i].style.display = "none";
        }}
        
        // Deactivate all buttons
        tablinks = document.getElementsByClassName("tab-btn");
        for (i = 0; i < tablinks.length; i++) {{
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }}
        
        // Show current tab
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
        
        // Resize Fix: Force Plotly to resize when tab becomes visible
        window.dispatchEvent(new Event('resize'));
        
        // Specific Plotly resize for visible charts in this tab
        var containers = document.getElementById(tabName).getElementsByClassName("chart-container");
        for(var j=0; j<containers.length; j++) {{
            if (containers[j].id) {{
                Plotly.Plots.resize(document.getElementById(containers[j].id));
            }}
        }}
    }}

    // Initial Render
    renderCharts();
    document.getElementById("defaultOpen").click();

</script>

</body>
</html>
        """
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Dashboard successfully generated: {output_path}")
            print(f"SUCCESS: Open {output_path} in your browser.")
        except Exception as e:
            logger.error(f"Failed to write HTML file: {e}")

    def generate_htmlOLD(self, output_path: str):

        """Generate the fully self-contained HTML file."""
        
        # 1. Get Plotly JS
        plotly_js = py_offline.get_plotlyjs()

        # 2. Serialize Figures to JSON
        # Structure: { 'tab1': [json_fig1, json_fig2], ... }
        plot_data = {}
        for tab, figs in self.figures.items():
            plot_data[tab] = [json.loads(json.dumps(f, cls=pd.io.json.PlotlyJSONEncoder)) for f in figs]
        
        plot_data_json = json.dumps(plot_data)

        # 3. HTML Template
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.ticker} Volatility Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #111; color: #ddd; margin: 0; padding: 20px; }}
        .header {{ margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 10px; }}
        .header h1 {{ margin: 0; font-weight: 300; }}
        .tabs {{ overflow: hidden; border-bottom: 1px solid #444; margin-bottom: 20px; }}
        .tab-btn {{ background-color: #222; float: left; border: none; outline: none; cursor: pointer; padding: 14px 20px; transition: 0.3s; color: #888; font-size: 16px; }}
        .tab-btn:hover {{ background-color: #333; color: white; }}
        .tab-btn.active {{ background-color: #444; color: #00d2ff; border-bottom: 2px solid #00d2ff; }}
        .tab-content {{ display: none; animation: fadeEffect 0.5s; }}
        @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
        .chart-container {{ margin-bottom: 20px; border: 1px solid #333; padding: 10px; background: #1a1a1a; }}
    </style>
    <script>
        {plotly_js}
    </script>
</head>
<body>

<div class="header">
    <h1>{self.ticker} Options & Volatility Dashboard</h1>
    <small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
</div>

<div class="tabs">
    <button class="tab-btn" onclick="openTab(event, 'tab1')" id="defaultOpen">Price & Realized Vol</button>
    <button class="tab-btn" onclick="openTab(event, 'tab2')">Term Structure & Skew</button>
    <button class="tab-btn" onclick="openTab(event, 'tab3')">Shadow GEX Regime</button>
    <button class="tab-btn" onclick="openTab(event, 'tab4')">Intraday</button>
</div>

<div id="tab1" class="tab-content">
    <div id="t1_c1" class="chart-container"></div>
    <div id="t1_c2" class="chart-container"></div>
</div>

<div id="tab2" class="tab-content">
    <div id="t2_c1" class="chart-container"></div>
    <div id="t2_c2" class="chart-container"></div>
</div>

<div id="tab3" class="tab-content">
    <div id="t3_c1" class="chart-container"></div>
    <div id="t3_c2" class="chart-container"></div>
</div>

<div id="tab4" class="tab-content">
    <div id="t4_c1" class="chart-container"></div>
</div>

<script>
    // Embedded Plot Data
    var plotData = {plot_data_json};

    function renderCharts() {{
        // Tab 1
        if (plotData.tab1 && plotData.tab1[0]) Plotly.newPlot('t1_c1', plotData.tab1[0].data, plotData.tab1[0].layout);
        if (plotData.tab1 && plotData.tab1[1]) Plotly.newPlot('t1_c2', plotData.tab1[1].data, plotData.tab1[1].layout);
        
        // Tab 2
        if (plotData.tab2 && plotData.tab2[0]) Plotly.newPlot('t2_c1', plotData.tab2[0].data, plotData.tab2[0].layout);
        if (plotData.tab2 && plotData.tab2[1]) Plotly.newPlot('t2_c2', plotData.tab2[1].data, plotData.tab2[1].layout);
        
        // Tab 3
        if (plotData.tab3 && plotData.tab3[0]) Plotly.newPlot('t3_c1', plotData.tab3[0].data, plotData.tab3[0].layout);
        if (plotData.tab3 && plotData.tab3[1]) Plotly.newPlot('t3_c2', plotData.tab3[1].data, plotData.tab3[1].layout);
        
        // Tab 4
        if (plotData.tab4 && plotData.tab4[0]) Plotly.newPlot('t4_c1', plotData.tab4[0].data, plotData.tab4[0].layout);
    }}

    function openTab(evt, tabName) {{
        var i, tabcontent, tablinks;
        
        // Hide all tab content
        tabcontent = document.getElementsByClassName("tab-content");
        for (i = 0; i < tabcontent.length; i++) {{
            tabcontent[i].style.display = "none";
        }}
        
        // Deactivate all buttons
        tablinks = document.getElementsByClassName("tab-btn");
        for (i = 0; i < tablinks.length; i++) {{
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }}
        
        // Show current tab
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
        
        // Resize Fix: Force Plotly to resize when tab becomes visible
        window.dispatchEvent(new Event('resize'));
        
        // Specific Plotly resize for visible charts in this tab
        var containers = document.getElementById(tabName).getElementsByClassName("chart-container");
        for(var j=0; j<containers.length; j++) {{
            if (containers[j].id) {{
                Plotly.Plots.resize(document.getElementById(containers[j].id));
            }}
        }}
    }}

    // Initial Render
    renderCharts();
    document.getElementById("defaultOpen").click();

</script>

</body>
</html>
        """
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Dashboard successfully generated: {output_path}")
            print(f"SUCCESS: Open {output_path} in your browser.")
        except Exception as e:
            logger.error(f"Failed to write HTML file: {e}")

# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Professional Options Volatility Dashboard")
    parser.add_argument("--ticker", required=True, type=str, help="Underlying ticker (e.g., SPY, NVDA)")
    parser.add_argument("--days", default=365, type=int, help="Days of daily history")
    parser.add_argument("--intraday", default="5m", type=str, help="Intraday interval (1m, 5m, 15m)")
    parser.add_argument("--output", type=str, default="", help="Custom output filename")
    
    args = parser.parse_args()
    
    ticker = args.ticker
    logger.info(f"Starting analysis for {ticker}...")
    
    # 1. Ingestion
    ingest = DataIngestion(ticker)
    daily_df = ingest.get_daily_ohlc(days=args.days)
    intraday_df = ingest.get_intraday_ohlc(interval=args.intraday)
    options_df, spot_price = ingest.get_options_chain()
    
    if daily_df.empty:
        logger.error("No daily data available. Exiting.")
        sys.exit(1)

    # 2. Analysis
    analyst = FinancialAnalysis(daily_df, intraday_df, options_df, spot_price)
    
    rv_df = analyst.get_realized_vol_term_structure()
    gex_df = analyst.get_shadow_gex_proxy()
    ts_df = analyst.get_options_term_structure_and_skew()
    intra_stats_df, current_intra_rv = analyst.get_intraday_stats()
    
    # 3. Rendering
    renderer = DashboardRenderer(ticker)
    renderer.create_price_vol_charts(rv_df)
    renderer.create_term_structure_charts(ts_df)
    renderer.create_shadow_gex_charts(gex_df)
    renderer.create_intraday_charts(intra_stats_df, current_intra_rv)
    
    # 4. Output
    if not args.output:
        date_str = datetime.now().strftime("%Y%m%d")
        output_file = f"dashboard_{ticker}_{date_str}.html"
    else:
        output_file = args.output
        
    renderer.generate_html(output_file)

if __name__ == "__main__":
    main()
