"""
Limit-Order Hit-Rate & Confidence Interval Dashboard
----------------------------------------------------
A standalone tool for hedge fund professionals to calibrate GTC limit-order pricing 
and analyze micro-alpha persistence.

Features:
- Robust Data Ingestion: Aggressive sanitization of yfinance data (MultiIndex fixing, timezone stripping).
- Local Caching: Prevents rate-limiting and enables offline re-runs.
- Financial Logic: Simulates limit orders with configurable slippage, offset, and hold horizons.
- Statistical Rigor: Computes Normal & Wilson Score Confidence Intervals and Required Sample Size (N).
- Offline Dashboard: Generates a single HTML file with embedded Plotly JS (no CDNs) and resize-aware tabs.
- Shadow Backfill: Generates synthetic 'Shadow GEX' history if primary data fails, preventing blank dashboards.

Usage:
    python3 limit_order_hit_rate_ci_dashboard.py SPY --interval 5m --period 5d
    python3 limit_order_hit_rate_ci_dashboard.py NVDA --side short --limit-offset-bps 10
"""

import argparse
import os
import time
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. DataIngestion Class
# -----------------------------------------------------------------------------

class DataIngestion:
    """
    Handles data acquisition, local caching, and aggressive sanitization.
    Strictly decouples I/O from financial logic.
    """
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_price_history(self, ticker, interval, start=None, end=None, period=None):
        """
        Retrieves historical price data with local caching and sanitization.
        """
        # Construct deterministic filename
        if start and end:
            fname = f"{ticker}_{interval}_{start}_{end}.csv"
        else:
            fname = f"{ticker}_{interval}_{period}.csv"
        
        # Sanitize filename for OS
        fname = fname.replace(":", "").replace("/", "-")
        cache_path = self.cache_dir / fname

        # 1. Try Cache
        if cache_path.exists() and cache_path.stat().st_size > 0:
            print(f"[INFO] Loading cached data from {cache_path}")
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                # Re-sanitize post-load to ensure types/index are perfect
                return self._sanitize_df(df)
            except Exception as e:
                print(f"[WARN] Cache read failed ({e}). Re-downloading.")

        # 2. Download via yfinance
        print(f"[INFO] Downloading {ticker} (Interval: {interval}, Period: {period or (start, end)})...")
        time.sleep(1.1)  # Rate limit protection

        try:
            if start and end:
                df = yf.download(ticker, start=start, end=end, interval=interval, group_by="column", progress=False)
            else:
                df = yf.download(ticker, period=period, interval=interval, group_by="column", progress=False)
        except Exception as e:
            print(f"[ERROR] yfinance download failed: {e}")
            return pd.DataFrame()

        # 3. Sanitize
        df_clean = self._sanitize_df(df)

        # 4. Cache
        if not df_clean.empty:
            df_clean.to_csv(cache_path)
            print(f"[INFO] Cached {len(df_clean)} rows to {cache_path}")
        
        return df_clean

    def _sanitize_df(self, df):
        """
        Aggressive Data Sanitization / Universal Fixer.
        Handles MultiIndex swapping, timezone stripping, and numeric coercion.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # --- A. MultiIndex Handling & Column Normalization ---
        # yfinance often returns MultiIndex columns: (Attribute, Ticker) or (Ticker, Attribute)
        if isinstance(df.columns, pd.MultiIndex):
            # Detect which level contains standard OHLCV names
            level_0_vals = set(df.columns.get_level_values(0))
            level_1_vals = set(df.columns.get_level_values(1))
            
            target_cols = {'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close'}
            
            # Case 1: Attributes in Level 1 (Ticker in Level 0) -> standard yfinance new behavior
            if not target_cols.intersection(level_0_vals) and target_cols.intersection(level_1_vals):
                # Swap levels so Attributes are at Level 0
                df = df.swaplevel(0, 1, axis=1)
            
            # Now flatten: Select only the relevant columns if multiple tickers exist (though we assume 1)
            # We just want the attribute level.
            # If multiple tickers were somehow passed, this might mix them, but we assume 1 ticker per CLI.
            df.columns = df.columns.get_level_values(0)

        # Handle Series vs DataFrame (rare edge case)
        if isinstance(df, pd.Series):
            df = df.to_frame(name="Close")

        # --- B. Index Handling ---
        # Reset index if it's not the datetime index (e.g. integer index with 'Date' column)
        if not isinstance(df.index, pd.DatetimeIndex):
            # Look for date-like columns
            for col in df.columns:
                if col.lower() in ['date', 'datetime', 'timestamp']:
                    df = df.set_index(col)
                    break
        
        # Enforce DatetimeIndex
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]

        # --- C. Timezone Stripping ---
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # --- D. Numeric Coercion ---
        # Ensure standard columns exist
        needed = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in needed:
            if col not in df.columns and 'Adj Close' in df.columns and col == 'Close':
                 df['Close'] = df['Adj Close'] # Fallback
            elif col not in df.columns:
                df[col] = np.nan # Create missing col as NaN

        # Force numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where all prices are NaN
        price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
        df = df.dropna(subset=price_cols, how='all')

        # --- E. Final Sorting ---
        df = df.sort_index()
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]

        return df

    def _backfill_shadow_history(self, ticker):
        """
        Cold Start Prevention. Downloads 1Y daily data to generate 'Shadow GEX' proxies.
        Used if primary intraday data is missing.
        """
        print(f"[INFO] Attempting Shadow Backfill for {ticker}...")
        try:
            time.sleep(1.0)
            df = yf.download(ticker, period="1y", interval="1d", group_by="column", progress=False)
            df = self._sanitize_df(df)
            
            if df.empty:
                return pd.DataFrame()

            # Compute Shadow Metrics
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            # Realized Vol (20-day rolling std, annualized)
            df['Realized_Vol'] = df['Log_Ret'].rolling(20).std() * sqrt(252)
            
            # Synthetic "Shadow GEX" Proxy
            # Logic: (Neutral Vol - Realized Vol) * Price. 
            # If RV is low, dealers might be short gamma (stable). If RV is high, long gamma. 
            # *This is purely illustrative for the dashboard visualization*
            neutral_vol = 0.15 
            df['Shadow_GEX'] = (neutral_vol - df['Realized_Vol']) * df['Close'] * 1000 

            cache_path = self.cache_dir / f"{ticker}_SHADOW_1d_1y.csv"
            df.to_csv(cache_path)
            return df
        except Exception as e:
            print(f"[WARN] Shadow backfill failed: {e}")
            return pd.DataFrame()


# -----------------------------------------------------------------------------
# 2. FinancialAnalysis Class
# -----------------------------------------------------------------------------

class FinancialAnalysis:
    """
    Handles all financial logic, trade simulation, and statistical computations.
    Immutable pattern: does not modify input raw_data in place.
    """
    def __init__(self, raw_data):
        self._raw_data = raw_data.copy()

    def compute_signals_and_trades(self, 
                                   dma_length=20, 
                                   limit_offset_bps=0.0, 
                                   slippage_bps=2.0, 
                                   hold_bars=20, 
                                   side='long',
                                   session_mode='all',
                                   session_start_str=None,
                                   session_end_str=None):
        """
        Generates signals based on DMA crossover and simulates limit orders.
        """
        df = self._raw_data.copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 1. Indicators
        df['SMA'] = df['Close'].rolling(window=dma_length).mean()
        df['Prev_Close'] = df['Close'].shift(1)
        
        # 2. Signal Generation (Crossover)
        # Long: Price drops below MA (mean reversion style or dip buy)? 
        # Prompt: "Price crosses below a rolling moving average" -> Place limit.
        if side == 'long':
            # Previous Close >= SMA AND Current Close < SMA
            df['Signal'] = (df['Prev_Close'] >= df['SMA']) & (df['Close'] < df['SMA'])
        else:
            # Previous Close <= SMA AND Current Close > SMA
            df['Signal'] = (df['Prev_Close'] <= df['SMA']) & (df['Close'] > df['SMA'])

        # 3. Session Filtering
        if session_mode != 'all':
            # Filter based on time
            # Assume naive local times (from DataIngestion)
            # Create a time index
            times = df.index.time
            
            if session_mode == 'rth':
                start_t = datetime.strptime("09:30", "%H:%M").time()
                end_t = datetime.strptime("16:00", "%H:%M").time()
            elif session_mode == 'preopen':
                start_t = datetime.strptime("04:00", "%H:%M").time()
                end_t = datetime.strptime("09:30", "%H:%M").time()
            elif session_mode == 'custom' and session_start_str and session_end_str:
                start_t = datetime.strptime(session_start_str, "%H:%M").time()
                end_t = datetime.strptime(session_end_str, "%H:%M").time()
            else:
                start_t = datetime.strptime("00:00", "%H:%M").time()
                end_t = datetime.strptime("23:59", "%H:%M").time()

            mask = (times >= start_t) & (times <= end_t)
            df['Signal'] = df['Signal'] & mask

        # Extract Signal Rows
        signals = df[df['Signal']].copy()
        signals['Signal_Time'] = signals.index
        
        trades = []
        
        # 4. Trade Simulation
        # Iterate through signals (vectorization is harder for path-dependent look-ahead)
        for idx, row in signals.iterrows():
            signal_time = idx
            
            # Locate integer location of signal
            try:
                sig_iloc = df.index.get_loc(signal_time)
            except KeyError:
                continue
            
            # Define Limit Price
            ref_price = row['Close'] # Using Close of signal bar as reference
            offset_mult = limit_offset_bps / 10000.0
            
            if side == 'long':
                limit_price = ref_price * (1 - offset_mult)
                # Slippage logic: We need price <= limit - slippage? 
                # No, Fill condition: Market Low <= Limit Price.
                # Fill Price: Limit Price (or worse if we gap through, but limit orders guarantee price or better).
                # To be conservative/realistic:
                # If Low <= Limit, we get filled. 
                # Effective Entry Price with slippage (cost) = Limit Price + (Limit * fill_slippage_bps)
                fill_thresh = limit_price
                entry_cost_adjustment = limit_price * (slippage_bps / 10000.0)
                exec_price = limit_price + entry_cost_adjustment # Long: buy higher due to slippage/fees
            else:
                limit_price = ref_price * (1 + offset_mult)
                fill_thresh = limit_price
                entry_cost_adjustment = limit_price * (slippage_bps / 10000.0)
                exec_price = limit_price - entry_cost_adjustment # Short: sell lower

            # Look ahead 'hold_bars'
            # slice the dataframe for the future window
            future_window = df.iloc[sig_iloc+1 : sig_iloc+1+hold_bars]
            
            filled = False
            fill_time = None
            exit_price = None
            pnl = 0.0
            
            if future_window.empty:
                # End of data, cannot evaluate
                continue

            for fw_idx, fw_row in future_window.iterrows():
                # Check for fill
                if side == 'long':
                    if fw_row['Low'] <= fill_thresh:
                        filled = True
                        fill_time = fw_idx
                        break
                else:
                    if fw_row['High'] >= fill_thresh:
                        filled = True
                        fill_time = fw_idx
                        break
            
            if filled:
                # Exit Logic: Simple time-based exit at end of hold_bars (or end of data)
                # If filled at index K, we hold for the remainder of the window or strictly hold_bars?
                # Simplification: Exit at Close of the last bar in the window relative to signal.
                # Or exit 'hold_bars' after FILL? The prompt says "hold bars after fill".
                
                # Find fill integer loc
                fill_iloc = df.index.get_loc(fill_time)
                exit_iloc = fill_iloc + hold_bars
                if exit_iloc >= len(df):
                    exit_iloc = len(df) - 1
                
                exit_row = df.iloc[exit_iloc]
                exit_price = exit_row['Close']
                exit_time = exit_row.name
                
                if side == 'long':
                    pnl = exit_price - exec_price
                else:
                    pnl = exec_price - exit_price
            else:
                # Unfilled
                exit_price = np.nan
                exit_time = np.nan
                pnl = 0.0

            trades.append({
                'signal_time': signal_time,
                'fill_time': fill_time,
                'limit_price': limit_price,
                'fill_price': exec_price if filled else np.nan,
                'exit_price': exit_price,
                'exit_time': exit_time if filled else np.nan,
                'pnl': pnl,
                'filled': filled,
                'profitable': pnl > 0,
                'side': side,
                'regime_vol': 0, # Placeholder
                'regime_hour': signal_time.hour,
                'regime_dow': signal_time.dayofweek
            })

        return pd.DataFrame(trades)

    def compute_stats(self, trades_df, target_moe=0.03):
        """
        Computes Hit Rate, Normal CIs, Wilson Score CIs, and Required N.
        """
        if trades_df.empty or not trades_df['filled'].any():
            return {
                'total_signals': len(trades_df),
                'filled_trades': 0,
                'profitable_trades': 0,
                'hit_rate': 0.0,
                'ci_lower_norm': 0.0, 'ci_upper_norm': 0.0,
                'ci_lower_wilson': 0.0, 'ci_upper_wilson': 0.0,
                'required_n': 0,
                'current_moe': 0.0
            }

        filled_df = trades_df[trades_df['filled']]
        n = len(filled_df)
        k = filled_df['profitable'].sum()
        p_hat = k / n if n > 0 else 0.0

        # Normal Approx CI (95%)
        z = 1.96
        if n > 0:
            se = sqrt(p_hat * (1 - p_hat) / n)
            ci_lower_norm = p_hat - z * se
            ci_upper_norm = p_hat + z * se
        else:
            ci_lower_norm, ci_upper_norm = 0.0, 0.0

        # Wilson Score Interval (95%)
        # Better for small N or extreme p (close to 0 or 1)
        if n > 0:
            denom = 1 + z**2/n
            center = (p_hat + z**2 / (2*n)) / denom
            diff = z * sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denom
            ci_lower_wilson = center - diff
            ci_upper_wilson = center + diff
        else:
            ci_lower_wilson, ci_upper_wilson = 0.0, 0.0

        # Required Sample Size for Target MoE (assuming p=0.5 for worst case)
        # MoE = z * sqrt(0.5*0.5/N) -> N = (z*0.5/MoE)^2
        try:
            req_n = int(np.ceil((z * 0.5 / target_moe)**2))
        except ZeroDivisionError:
            req_n = 0

        # Current MoE
        current_moe = z * sqrt(0.25/n) if n > 0 else 1.0

        return {
            'total_signals': len(trades_df),
            'filled_trades': n,
            'profitable_trades': k,
            'hit_rate': p_hat,
            'ci_lower_norm': ci_lower_norm,
            'ci_upper_norm': ci_upper_norm,
            'ci_lower_wilson': ci_lower_wilson,
            'ci_upper_wilson': ci_upper_wilson,
            'required_n': req_n,
            'current_moe': current_moe,
            'pnl_mean': filled_df['pnl'].mean(),
            'pnl_median': filled_df['pnl'].median(),
            'pnl_std': filled_df['pnl'].std(),
            'pnl_skew': filled_df['pnl'].skew()
        }

    def get_regime_stats(self, trades_df):
        """
        Breakdown hit rates by Time of Day and Day of Week.
        """
        if trades_df.empty or not trades_df['filled'].any():
            return pd.DataFrame(), pd.DataFrame()

        filled = trades_df[trades_df['filled']].copy()
        
        # Hour Group
        by_hour = filled.groupby('regime_hour')['profitable'].agg(['count', 'mean']).rename(columns={'count':'N', 'mean':'HitRate'})
        
        # DOW Group
        by_dow = filled.groupby('regime_dow')['profitable'].agg(['count', 'mean']).rename(columns={'count':'N', 'mean':'HitRate'})
        
        return by_hour, by_dow

# -----------------------------------------------------------------------------
# 3. DashboardRenderer Class
# -----------------------------------------------------------------------------

class DashboardRenderer:
    """
    Generates a multi-tab Plotly HTML dashboard with offline JS embedding.
    """
    def __init__(self, ticker):
        self.ticker = ticker

    def _get_offline_plotly_js(self):
        """Returns the full Plotly JS library string for offline embedding."""
        return py_offline.get_plotlyjs()

    def generate_html(self, 
                      df_prices, 
                      trades_df, 
                      stats_dict, 
                      shadow_df, 
                      output_path, 
                      auto_open=True):
        
        # --- Pre-computation for Plots ---
        
        # 1. Price & Signals
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df_prices.index, y=df_prices['Close'], mode='lines', name='Price', line=dict(color='#2c3e50')))
        
        if not trades_df.empty:
            # Signal markers
            fig_price.add_trace(go.Scatter(
                x=trades_df['signal_time'], 
                y=trades_df['limit_price'], 
                mode='markers', 
                name='Limit Set',
                marker=dict(symbol='circle-open', color='orange', size=6)
            ))
            
            # Fill markers (Profitable vs Loss)
            filled = trades_df[trades_df['filled']]
            prof = filled[filled['profitable']]
            loss = filled[~filled['profitable']]
            
            fig_price.add_trace(go.Scatter(
                x=prof['fill_time'], y=prof['fill_price'],
                mode='markers', name='Fill (Profit)',
                marker=dict(symbol='triangle-up', color='green', size=10)
            ))
            fig_price.add_trace(go.Scatter(
                x=loss['fill_time'], y=loss['fill_price'],
                mode='markers', name='Fill (Loss)',
                marker=dict(symbol='triangle-down', color='red', size=10)
            ))

        fig_price.update_layout(title=f"{self.ticker} Price & Signals", template="plotly_white", height=600)

        # 2. Hit-Rate Evolution
        fig_evol = go.Figure()
        if not trades_df.empty and trades_df['filled'].any():
            filled = trades_df[trades_df['filled']].copy().sort_values('fill_time')
            filled['cum_profit'] = filled['profitable'].cumsum()
            filled['cum_n'] = np.arange(1, len(filled) + 1)
            filled['cum_rate'] = filled['cum_profit'] / filled['cum_n']
            
            # Calc Wilson intervals dynamically (vectorized approx)
            z = 1.96
            n = filled['cum_n']
            p = filled['cum_rate']
            denom = 1 + z**2/n
            center = (p + z**2/(2*n)) / denom
            diff = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
            
            fig_evol.add_trace(go.Scatter(x=filled['fill_time'], y=center+diff, mode='lines', line=dict(width=0), showlegend=False, name='Upper'))
            fig_evol.add_trace(go.Scatter(x=filled['fill_time'], y=center-diff, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', name='95% Wilson CI'))
            fig_evol.add_trace(go.Scatter(x=filled['fill_time'], y=filled['cum_rate'], mode='lines', name='Cumulative Hit Rate', line=dict(color='blue')))
            
        fig_evol.update_layout(title="Hit-Rate Stability & Convergence", yaxis=dict(range=[0, 1]), template="plotly_white", height=500)

        # 3. PnL Distribution
        fig_pnl = go.Figure()
        if not trades_df.empty and trades_df['filled'].any():
            filled_pnl = trades_df[trades_df['filled']]['pnl']
            fig_pnl.add_trace(go.Histogram(x=filled_pnl, nbinsx=30, name='PnL', marker_color='#34495e'))
        fig_pnl.update_layout(title="Trade PnL Distribution", template="plotly_white", height=500)

        # 4. Sampling Error Curve
        fig_sample = go.Figure()
        N_range = np.arange(10, stats_dict['required_n'] * 1.5 + 100, 10)
        # Margin of Error = 1.96 * sqrt(0.5*0.5 / N)
        moe_curve = 1.96 * np.sqrt(0.25 / N_range)
        

        fig_sample.add_trace(go.Scatter(x=N_range, y=moe_curve, mode='lines', name='Theoretical MoE (95%)'))
        # Add Current N
        curr_n = stats_dict['filled_trades']
        if curr_n > 0:
            curr_moe = stats_dict['current_moe']
            fig_sample.add_trace(go.Scatter(x=[curr_n], y=[curr_moe], mode='markers+text', text=["Current N"], textposition="top right", marker=dict(size=12, color='red'), name='Current'))
        
        # Add Required N
        req_n = stats_dict['required_n']
        fig_sample.add_vline(x=req_n, line_dash="dash", line_color="green", annotation_text="Target MoE")
        #fig_sample.add_trace(go.vline(x=req_n, line_dash="dash", line_color="green", annotation_text="Target MoE"))
        
        fig_sample.update_layout(title="Margin of Error vs Sample Size", template="plotly_white", height=500, xaxis_title="N (Trades)", yaxis_title="Margin of Error (+/-)")

        # 5. Shadow History
        fig_shadow = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Price History (1Y)", "Synthetic Shadow GEX Proxy"))
        if not shadow_df.empty:
            fig_shadow.add_trace(go.Scatter(x=shadow_df.index, y=shadow_df['Close'], name='Close'), row=1, col=1)
            fig_shadow.add_trace(go.Scatter(x=shadow_df.index, y=shadow_df['Shadow_GEX'], name='Shadow GEX', line=dict(color='purple')), row=2, col=1)
        fig_shadow.update_layout(height=600, title="Shadow Backfill Context (Cold Start)", template="plotly_white")

        # --- HTML Construction ---
        
        plotly_js = self._get_offline_plotly_js()
        
        # Convert figures to JSON divs
        div_price = py_offline.plot(fig_price, include_plotlyjs=False, output_type='div')
        div_evol = py_offline.plot(fig_evol, include_plotlyjs=False, output_type='div')
        div_pnl = py_offline.plot(fig_pnl, include_plotlyjs=False, output_type='div')
        div_sample = py_offline.plot(fig_sample, include_plotlyjs=False, output_type='div')
        div_shadow = py_offline.plot(fig_shadow, include_plotlyjs=False, output_type='div')

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Limit Order Hit-Rate Dashboard: {self.ticker}</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }}
                .container {{ max_width: 1400px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .kpi-panel {{ display: flex; justify-content: space-around; background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .kpi-box {{ text-align: center; }}
                .kpi-val {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
                .kpi-label {{ font-size: 14px; color: #7f8c8d; }}
                
                /* Tab Styles */
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 5px 5px 0 0; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 16px; font-weight: 600; color: #555; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: white; border-bottom: 3px solid #3498db; color: #2c3e50; }}
                .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; animation: fadeEffect 0.5s; background: white; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                    
                    // Critical: Trigger resize for Plotly
                    window.dispatchEvent(new Event('resize'));
                    
                    // Specific Plotly resize call for extra safety
                    var plots = document.querySelectorAll('.js-plotly-plot');
                    plots.forEach(function(p) {{ Plotly.Plots.resize(p); }});
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h1>{self.ticker} Limit Order Strategy Dashboard</h1>
                
                <div class="kpi-panel">
                    <div class="kpi-box">
                        <div class="kpi-val">{stats_dict['hit_rate']:.1%}</div>
                        <div class="kpi-label">Hit Rate</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-val">[{stats_dict['ci_lower_wilson']:.1%}, {stats_dict['ci_upper_wilson']:.1%}]</div>
                        <div class="kpi-label">95% Wilson CI</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-val">{stats_dict['filled_trades']} / {stats_dict['total_signals']}</div>
                        <div class="kpi-label">Filled / Signals</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-val">{stats_dict['required_n']}</div>
                        <div class="kpi-label">Required N (MoE)</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-val">{stats_dict['pnl_mean']:.2f}</div>
                        <div class="kpi-label">Avg PnL</div>
                    </div>
                </div>

                <div class="tab">
                    <button class="tablinks active" onclick="openTab(event, 'Summary')">Summary</button>
                    <button class="tablinks" onclick="openTab(event, 'Price')">Price & Signals</button>
                    <button class="tablinks" onclick="openTab(event, 'Evolution')">Hit-Rate Evolution</button>
                    <button class="tablinks" onclick="openTab(event, 'PnL')">PnL Distribution</button>
                    <button class="tablinks" onclick="openTab(event, 'Sampling')">Sampling Error</button>
                    <button class="tablinks" onclick="openTab(event, 'Shadow')">Shadow History</button>
                </div>

                <div id="Summary" class="tabcontent" style="display: block;">
                    <h3>Strategy Configuration</h3>
                    <ul>
                        <li><strong>Period:</strong> {len(df_prices)} bars analyzed.</li>
                        <li><strong>Signals:</strong> {stats_dict['total_signals']} generated.</li>
                        <li><strong>Fills:</strong> {stats_dict['filled_trades']} executed.</li>
                        <li><strong>Hit Rate:</strong> {stats_dict['hit_rate']:.2%} (Profitable Trades / Filled Trades).</li>
                    </ul>
                    <p><em>Use the tabs above to explore detailed visualizations.</em></p>
                </div>

                <div id="Price" class="tabcontent">
                    {div_price}
                </div>

                <div id="Evolution" class="tabcontent">
                    {div_evol}
                </div>

                <div id="PnL" class="tabcontent">
                    {div_pnl}
                </div>

                <div id="Sampling" class="tabcontent">
                    {div_sample}
                </div>

                <div id="Shadow" class="tabcontent">
                    <p><strong>Note:</strong> This data is a 1-year daily backfill used for context ("Cold Start") when intraday data is sparse.</p>
                    {div_shadow}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[SUCCESS] Dashboard written to: {output_path}")
        if auto_open:
            webbrowser.open('file://' + os.path.realpath(output_path))


# -----------------------------------------------------------------------------
# 4. Main Execution & CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Limit-Order Hit-Rate & Confidence Interval Dashboard")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., SPY)")
    parser.add_argument("--interval", type=str, default="5m", help="1m, 2m, 5m, 1h, 1d")
    parser.add_argument("--period", type=str, default="5d", help="5d, 1mo, 3mo")
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    
    parser.add_argument("--session-mode", type=str, default="all", choices=['all', 'preopen', 'rth', 'custom'])
    parser.add_argument("--session-start", type=str, help="HH:MM for custom session")
    parser.add_argument("--session-end", type=str, help="HH:MM for custom session")
    
    parser.add_argument("--dma-length", type=int, default=20)
    parser.add_argument("--limit-offset-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--hold-bars", type=int, default=20)
    parser.add_argument("--side", type=str, default="long", choices=['long', 'short'])
    parser.add_argument("--target-moe", type=float, default=0.03)
    
    parser.add_argument("--output-html", type=str, default="")
    parser.add_argument("--no-open-html", action="store_true")
    
    args = parser.parse_args()

    # Setup
    data_mgr = DataIngestion()
    
    # 1. Fetch Data
    df = data_mgr.get_price_history(
        args.ticker, 
        args.interval, 
        start=args.start, 
        end=args.end, 
        period=None if (args.start and args.end) else args.period
    )
    
    # Shadow Backfill (always fetch for context/backup)
    shadow_df = data_mgr._backfill_shadow_history(args.ticker)

    if df.empty:
        print("[ERROR] Primary data is empty. Using Shadow Data only for visualization check.")
        # We can't run the strategy, but we can render the dashboard with shadow data
        stats = {k:0 for k in ['total_signals', 'filled_trades', 'profitable_trades', 'hit_rate', 
                               'ci_lower_norm', 'ci_upper_norm', 'ci_lower_wilson', 'ci_upper_wilson', 
                               'required_n', 'current_moe', 'pnl_mean', 'pnl_median', 'pnl_std', 'pnl_skew']}
        trades_df = pd.DataFrame()
    else:
        # 2. Run Financial Analysis
        analyzer = FinancialAnalysis(df)
        trades_df = analyzer.compute_signals_and_trades(
            dma_length=args.dma_length,
            limit_offset_bps=args.limit_offset_bps,
            slippage_bps=args.slippage_bps,
            hold_bars=args.hold_bars,
            side=args.side,
            session_mode=args.session_mode,
            session_start_str=args.session_start,
            session_end_str=args.session_end
        )
        
        # 3. Compute Stats
        stats = analyzer.compute_stats(trades_df, target_moe=args.target_moe)

    # 4. Render Dashboard
    renderer = DashboardRenderer(args.ticker)
    output_path = args.output_html if args.output_html else f"hit_rate_dashboard_{args.ticker}.html"
    
    renderer.generate_html(
        df_prices=df if not df.empty else shadow_df, # Fallback to shadow for price chart if primary empty
        trades_df=trades_df,
        stats_dict=stats,
        shadow_df=shadow_df,
        output_path=output_path,
        auto_open=not args.no_open_html
    )

if __name__ == "__main__":
    main()
