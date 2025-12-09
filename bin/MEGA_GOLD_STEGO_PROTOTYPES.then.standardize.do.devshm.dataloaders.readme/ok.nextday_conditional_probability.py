#!/usr/bin/env python3
"""
Next-Day Directional Conditional Probability Dashboard
Role: Senior Quantitative Developer & Hedge Fund Macro/Options Quant

Purpose:
    Analyzes whether prior-day returns (bucketed by magnitude) provide a statistically 
    significant edge for next-day direction (Up/Down).
    Produces an offline Plotly HTML dashboard and CSV statistics.

Usage:
    python3 nextday_conditional_direction_dashboard.py --ticker SPY
    python3 nextday_conditional_direction_dashboard.py --ticker QQQ --open-html

Dependencies:
    pip install numpy pandas scipy plotly yfinance
"""

import os
import sys
import time
import argparse
import logging
import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.stats
import plotly.graph_objects as go
import plotly.offline
import yfinance as yf

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. DataIngestion Class
# ==============================================================================

class DataIngestion:
    """
    Responsibilities:
    - Download OHLCV data via yfinance.
    - robust local CSV caching.
    - Aggressive sanitization of MultiIndex/Timezones.
    - Shadow backfill for cold starts.
    """

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_price_history(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Orchestrates the fetch process: check cache -> load -> sanitize. 
        If missing, download -> sanitize -> backfill -> save.
        """
        clean_ticker = ticker.upper().replace("^", "")
        filename = os.path.join(self.output_dir, f"{clean_ticker}_prices.csv")
        
        # 1. Try Load from Cache
        if os.path.exists(filename) and os.path.getsize(filename) > 100:
            logger.info(f"Loading cached data for {ticker} from {filename}...")
            try:
                # Load with low_memory=False to avoid dtype warnings, parse dates manually later
                df = pd.read_csv(filename, index_col=0)
                sanitized_df = self._sanitize_df(df)
                if not sanitized_df.empty:
                    return sanitized_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-downloading.")

        # 2. Download from yfinance
        logger.info(f"Downloading data for {ticker} (yfinance)...")
        time.sleep(1)  # Rate limit courtesy
        
        try:
            # yfinance download
            # Using group_by='column' to help control structure
            # auto_adjust=True is generally preferred for quant analysis (splits/divs)
            raw_df = yf.download(
                ticker, 
                start=start, 
                end=end, 
                group_by='column', 
                auto_adjust=True, 
                progress=False
            )
            
            if raw_df.empty:
                logger.error(f"yfinance returned empty data for {ticker}.")
                return pd.DataFrame()

            # 3. Sanitize
            clean_df = self._sanitize_df(raw_df)
            
            # 4. Shadow Backfill (Cold Start Prevention)
            if clean_df.empty or len(clean_df) < 252:
                logger.warning("Data insufficient. Attempting shadow backfill logic.")
                clean_df = self._backfill_shadow_history(clean_df, ticker)

            # 5. Save to CSV
            clean_df.to_csv(filename)
            logger.info(f"Saved {len(clean_df)} rows to {filename}")
            
            return clean_df

        except Exception as e:
            logger.error(f"Critical error downloading {ticker}: {e}")
            return pd.DataFrame()

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressive sanitization:
        - Handle MultiIndex columns (flatten/swap).
        - Enforce DatetimeIndex (timezone naive).
        - Numeric coercion.
        """
        df = df.copy()

        # 1. Column Normalization (MultiIndex Handling)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if levels need swapping. We want 'Close' or 'Adj Close' to be accessible.
            # Typical yf structure: (Price, Ticker) or (Ticker, Price)
            
            # Heuristic: if 'Close' is in level 1, swap.
            # If the index has levels, we check the values in the levels.
            lev0_vals = df.columns.get_level_values(0).unique()
            lev1_vals = df.columns.get_level_values(1).unique()

            if 'Close' not in lev0_vals and 'Close' in lev1_vals:
                df = df.swaplevel(0, 1, axis=1)

            # Flatten to single level: 'Close' or 'Ticker_Close'
            # If multiple tickers were downloaded (unlikely here but robust), 
            # we just join. If single ticker, yf often leaves one level empty or just the metric.
            new_cols = []
            for col in df.columns:
                # col is a tuple
                c_str = [str(c) for c in col if str(c) != '']
                new_cols.append('_'.join(c_str))
            df.columns = new_cols

        # 2. Strict Datetime Index
        # If index is currently int/range, look for a "Date" column
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            else:
                # Attempt to convert existing index
                df.index = pd.to_datetime(df.index, errors='coerce')

        # Ensure index is datetime objects
        df.index = pd.to_datetime(df.index, errors='coerce')
        
        # Drop rows with NaT index
        df = df[~df.index.isna()]

        # 3. Strip Timezones
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 4. Numeric Coercion
        # Force all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. Filter for essential columns
        # We need at least a Close column. 
        # Normalize column names to Title Case for easier lookup (Open, High, Low, Close)
        df.columns = [c.title() for c in df.columns]
        
        return df.sort_index()

    def _backfill_shadow_history(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Cold Start Prevention:
        If data is extremely short or empty, we download a longer history 
        (ignoring the user's specific start date constraint for the sake of functionality)
        or generate a synthetic structure if API fails completely (fallback).
        
        Here, we attempt a forced 2-year download as the shadow backfill.
        """
        logger.info("Executing Shadow Backfill (forced 2y download)...")
        try:
            # Force download 5 years to be safe
            raw = yf.download(ticker, period="5y", group_by='column', auto_adjust=True, progress=False)
            sanitized = self._sanitize_df(raw)
            if not sanitized.empty:
                return sanitized
        except Exception:
            pass
        
        return df


# ==============================================================================
# 2. FinancialAnalysis Class
# ==============================================================================

class FinancialAnalysis:
    """
    Responsibilities:
    - Immutable data storage.
    - Compute daily returns, buckets, conditional probabilities.
    - Clopper-Pearson Intervals & Exact Binomial Tests.
    """

    # Defined buckets: (-inf, -4%], (-4, -3], ... [4, inf)
    BUCKET_BINS = [-float('inf'), -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, float('inf')]
    BUCKET_LABELS = [
        "<= -4%", "(-4%, -3%]", "(-3%, -2%]", "(-2%, -1%]", "(-1%, 0%]",
        "(0%, 1%]", "(1%, 2%]", "(2%, 3%]", "(3%, 4%]", ">= 4%"
    ]

    def __init__(self, df: pd.DataFrame):
        self._raw_data = df.copy()
        self._returns = self._compute_daily_returns(self._raw_data)

    def _compute_daily_returns(self, df: pd.DataFrame) -> pd.Series:
        """Robustly extract Close and compute % returns."""
        # Prefer 'Adj Close', then 'Close'
        if 'Adj Close' in df.columns:
            close_series = df['Adj Close']
        elif 'Close' in df.columns:
            close_series = df['Close']
        else:
            # Fallback to first column
            close_series = df.iloc[:, 0]

        # Simple returns
        rets = close_series.pct_change()
        rets = rets.dropna()
        return rets

    def analyze_windows(self, windows: List[int], thresholds: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main entry point for analysis.
        Returns:
            1. stats_df: The core conditional prob table.
            2. thresholds_df: Aggregated stats for >= X% moves.
            3. plotting_data: Helper derived series for histograms/heatmaps.
        """
        all_stats = []
        
        # We also want to compute thresholds logic (e.g. >= 2%)
        # But first, the standard bucket analysis
        
        for w in windows:
            # 1. Slice trailing window
            if len(self._returns) < w:
                logger.warning(f"Window {w} exceeds data length {len(self._returns)}. Skipping.")
                continue
                
            window_rets = self._returns.iloc[-w:].copy()
            
            # 2. Prepare Prior/Next pairs
            # prior_ret[t] corresponds to returns at day t
            # next_ret[t] corresponds to returns at day t+1
            # We align them: index i has prior=rets[i], next=rets[i+1]
            
            # Create a DataFrame for easy shifting
            # shift(-1) moves t+1 to t.
            # We drop the last row because it has no 'next day'
            analysis_df = pd.DataFrame({
                'prior': window_rets,
                'next': window_rets.shift(-1)
            }).dropna()

            # 3. Discretize Prior Returns into Buckets
            # pd.cut handles the intervals. 
            # right=True means (a, b]. 
            # For (-1, 0], 0 is included. 
            analysis_df['bucket'] = pd.cut(
                analysis_df['prior'], 
                bins=self.BUCKET_BINS, 
                labels=self.BUCKET_LABELS,
                right=True
            )

            # Unconditional (Base) Probability
            # Pr(Next > 0)
            total_n = len(analysis_df)
            total_up = (analysis_df['next'] > 0).sum()
            p_base = total_up / total_n if total_n > 0 else 0.5

            # 4. Compute Conditional Stats per Bucket
            # Group by bucket
            grouped = analysis_df.groupby('bucket', observed=False)

            for bucket_label, group in grouped:
                n = len(group)
                k = (group['next'] > 0).sum()
                
                # Conditional Prob
                p_cond = k / n if n > 0 else np.nan
                
                # Clopper-Pearson 95% CI
                if n > 0:
                    ci_low = scipy.stats.beta.ppf(0.025, k, n - k + 1)
                    ci_high = scipy.stats.beta.ppf(0.975, k + 1, n - k)
                    # Handle edge cases where k=0 or k=n exactly
                    if k == 0: ci_low = 0.0
                    if k == n: ci_high = 1.0
                else:
                    ci_low, ci_high = np.nan, np.nan

                # Exact Binomial Test (Two-Sided)
                # H0: p_cond == p_base
                p_value = np.nan
                if n > 0:
                    res = scipy.stats.binomtest(k, n, p_base, alternative='two-sided')
                    p_value = res.pvalue

                # Reliability Flag
                if n < 30:
                    flag = "Unreliable"
                elif n < 100:
                    flag = "Caution"
                elif n < 1000:
                    flag = "Useable"
                else:
                    flag = "Robust"

                # Next Day Vol (Annualized) - strictly inside this bucket
                # std dev of next day returns * sqrt(252)
                if n > 1:
                    next_vol = group['next'].std() * np.sqrt(252)
                else:
                    next_vol = 0.0

                all_stats.append({
                    'window': w,
                    'bucket': bucket_label,
                    'n': n,
                    'k_up': k,
                    'p_cond': p_cond,
                    'p_base': p_base,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'p_value': p_value,
                    'reliability': flag,
                    'next_vol': next_vol
                })

        stats_df = pd.DataFrame(all_stats)
        
        # --- Threshold Analysis (e.g. >= 2% or <= -2%) ---
        # Options traders often care about "Big Moves" generically
        threshold_stats = []
        
        for w in windows:
            if len(self._returns) < w: continue
            window_rets = self._returns.iloc[-w:].copy()
            analysis_df = pd.DataFrame({'prior': window_rets, 'next': window_rets.shift(-1)}).dropna()
            
            total_n = len(analysis_df)
            total_up = (analysis_df['next'] > 0).sum()
            p_base = total_up / total_n if total_n > 0 else 0.5

            for t_pct in thresholds:
                t_dec = t_pct / 100.0
                
                # Directional subsets
                # "Up Big": prior >= t
                # "Down Big": prior <= -t
                
                for direction, mask in [
                    (f"Drop >= {t_pct}%", analysis_df['prior'] <= -t_dec),
                    (f"Rally >= {t_pct}%", analysis_df['prior'] >= t_dec)
                ]:
                    subset = analysis_df[mask]
                    n = len(subset)
                    k = (subset['next'] > 0).sum()
                    
                    p_cond = k / n if n > 0 else np.nan
                    
                    if n > 0:
                        ci_low = scipy.stats.beta.ppf(0.025, k, n - k + 1)
                        ci_high = scipy.stats.beta.ppf(0.975, k + 1, n - k)
                        if k == 0: ci_low = 0.0
                        if k == n: ci_high = 1.0
                        
                        res = scipy.stats.binomtest(k, n, p_base, alternative='two-sided')
                        p_val = res.pvalue
                        next_vol = subset['next'].std() * np.sqrt(252) if n > 1 else 0.0
                    else:
                        ci_low, ci_high, p_val, next_vol = np.nan, np.nan, np.nan, np.nan
                        
                    threshold_stats.append({
                        'window': w,
                        'condition': direction,
                        'threshold_pct': t_pct,
                        'n': n,
                        'p_cond': p_cond,
                        'p_base': p_base,
                        'p_value': p_val,
                        'next_vol': next_vol,
                        'ci_low': ci_low,
                        'ci_high': ci_high
                    })

        thresh_df = pd.DataFrame(threshold_stats)
        
        return stats_df, thresh_df, self._returns


# ==============================================================================
# 3. DashboardRenderer Class
# ==============================================================================

class DashboardRenderer:
    """
    Responsibilities:
    - Build Offline Plotly HTML.
    - Embed Plotly JS.
    - Implement Tab resizing logic.
    """

    def __init__(self, stats_df: pd.DataFrame, thresh_df: pd.DataFrame, returns_series: pd.Series, ticker: str):
        self.stats = stats_df
        self.thresh = thresh_df
        self.returns = returns_series
        self.ticker = ticker

    def render_html(self, output_path: str, open_browser: bool = False):
        if self.stats.empty:
            logger.error("No stats available to render.")
            return

        # 1. Generate Figures
        fig_ladder = self._make_bucket_ladder()
        fig_heatmap = self._make_heatmap()
        fig_thresh = self._make_threshold_view()
        fig_dist = self._make_distribution_view()

        # 2. Get Plotly JS string (offline)
        plotly_js = plotly.offline.get_plotlyjs()

        # 3. Convert figures to HTML <div> strings (without full html wrapper)
        div_ladder = plotly.offline.plot(fig_ladder, include_plotlyjs=False, output_type='div')
        div_heatmap = plotly.offline.plot(fig_heatmap, include_plotlyjs=False, output_type='div')
        div_thresh = plotly.offline.plot(fig_thresh, include_plotlyjs=False, output_type='div')
        div_dist = plotly.offline.plot(fig_dist, include_plotlyjs=False, output_type='div')

        # 4. Assemble HTML Template
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{self.ticker} Directional Dashboard</title>
            <script type="text/javascript">
                {plotly_js}
            </script>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background: #f4f4f4; }}
                h1 {{ color: #333; }}
                .meta {{ font-size: 0.9em; color: #666; margin-bottom: 20px; }}
                
                /* Tab Styling */
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 5px 5px 0 0; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; font-weight: 600; color: #555; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #fff; color: #000; border-bottom: 2px solid #007bff; }}
                
                .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; background: #fff; border-radius: 0 0 5px 5px; animation: fadeEffect 0.5s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>

            <h1>{self.ticker}: Conditional Next-Day Direction Dashboard</h1>
            <div class="meta">Generated: {timestamp} | Local Analysis</div>

            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Ladder')" id="defaultOpen">1. Bucket Ladder</button>
                <button class="tablinks" onclick="openTab(event, 'Heatmap')">2. Stability Heatmap</button>
                <button class="tablinks" onclick="openTab(event, 'Thresholds')">3. Option Thresholds</button>
                <button class="tablinks" onclick="openTab(event, 'Dist')">4. Distribution & Context</button>
            </div>

            <div id="Ladder" class="tabcontent">
                <h3>Conditional Probability of "Up" Day by Prior Return Bucket</h3>
                <p>Bars show Prob(Next > 0). Error bars are 95% Clopper-Pearson CI. Horizontal line is unconditional baseline.</p>
                {div_ladder}
            </div>

            <div id="Heatmap" class="tabcontent">
                <h3>Edge Stability (Prob_Cond - Prob_Base)</h3>
                <p>Color represents the excess probability of an UP day relative to the baseline. Red = Bearish Edge, Green = Bullish Edge.</p>
                {div_heatmap}
            </div>

            <div id="Thresholds" class="tabcontent">
                <h3>Large Move Analysis</h3>
                <p>Statistics for moves exceeding specific thresholds (e.g. >= 2% or <= -2%). Includes Next-Day Volatility.</p>
                {div_thresh}
            </div>
            
            <div id="Dist" class="tabcontent">
                <h3>Historical Distribution & Today's Context</h3>
                <p>Where does the most recent close fit in history?</p>
                {div_dist}
            </div>

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
                    
                    // Specific Plotly resize call for safety
                    var contentDiv = document.getElementById(tabName);
                    var plots = contentDiv.getElementsByClassName('plotly-graph-div');
                    for (var j=0; j<plots.length; j++) {{
                        Plotly.Plots.resize(plots[j]);
                    }}
                }}

                // Open default tab
                document.getElementById("defaultOpen").click();
            </script>

        </body>
        </html>
        """

        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to: {output_path}")

        if open_browser:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_path))

    def _make_bucket_ladder(self):
        """Tab 1: Bar chart with Error Bars for the longest window."""
        max_win = self.stats['window'].max()
        df_sub = self.stats[self.stats['window'] == max_win].copy()
        
        # Color based on reliability
        colors = []
        for r in df_sub['reliability']:
            if r == 'Unreliable': colors.append('lightgray')
            elif r == 'Caution': colors.append('gold')
            elif r == 'Useable': colors.append('cornflowerblue')
            else: colors.append('darkblue') # Robust

        # Error bars
        error_y = dict(
            type='data',
            symmetric=False,
            array=df_sub['ci_high'] - df_sub['p_cond'],
            arrayminus=df_sub['p_cond'] - df_sub['ci_low'],
            color='black'
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_sub['bucket'],
            y=df_sub['p_cond'],
            error_y=error_y,
            marker_color=colors,
            name='Cond Prob (Up)',
            text=df_sub['n'].apply(lambda x: f"n={x}"),
            textposition='auto'
        ))

        # Baseline line
        base_p = df_sub['p_base'].iloc[0] if not df_sub.empty else 0.5
        fig.add_hline(y=base_p, line_dash="dash", line_color="red", annotation_text=f"Base: {base_p:.1%}")
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray")

        fig.update_layout(
            title=f"Conditional Up Probability (Window: {max_win} days)",
            yaxis=dict(title="Prob(Next > 0)", range=[0, 1]),
            xaxis=dict(title="Prior Day Return Bucket"),
            template="plotly_white",
            height=600
        )
        return fig

    def _make_heatmap(self):
        """Tab 2: Heatmap of (P_cond - P_base) across windows."""
        # Pivot: Index=Window, Columns=Bucket, Values=Excess
        df = self.stats.copy()
        df['excess'] = df['p_cond'] - df['p_base']
        
        pivot_exc = df.pivot(index='window', columns='bucket', values='excess')
        pivot_n = df.pivot(index='window', columns='bucket', values='n')
        
        # Ensure column order matches standard buckets
        valid_cols = [c for c in FinancialAnalysis.BUCKET_LABELS if c in pivot_exc.columns]
        pivot_exc = pivot_exc[valid_cols]
        pivot_n = pivot_n[valid_cols]

        # Annotations (show N)
        annotations = []
        for y_val in pivot_exc.index:
            for x_val in pivot_exc.columns:
                val = pivot_exc.loc[y_val, x_val]
                n_val = pivot_n.loc[y_val, x_val]
                if not np.isnan(val):
                    annotations.append(dict(
                        x=x_val, y=y_val,
                        text=f"{val:+.1%}<br>(n={int(n_val)})",
                        showarrow=False,
                        font=dict(color="black" if abs(val) < 0.15 else "white", size=10)
                    ))

        fig = go.Figure(data=go.Heatmap(
            z=pivot_exc.values,
            x=pivot_exc.columns,
            y=pivot_exc.index.astype(str),
            colorscale="RdBu",
            zmid=0,
            text=pivot_n.values,
            hovertemplate="Excess: %{z:.2%}<br>N: %{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Stability of Edge Across Time Windows",
            yaxis_title="Lookback Window (Days)",
            annotations=annotations,
            height=500
        )
        return fig

    def _make_threshold_view(self):
        """Tab 3: Side-by-side Prob and Volatility for Thresholds."""
        if self.thresh.empty:
            return go.Figure().add_annotation(text="No threshold data")

        # Filter to max window for clarity
        max_win = self.thresh['window'].max()
        df = self.thresh[self.thresh['window'] == max_win].copy()

        # Create subplot
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Win Prob vs Baseline", "Next Day Volatility"))

        # 1. Prob Chart
        fig.add_trace(go.Bar(
            x=df['condition'],
            y=df['p_cond'],
            name='Cond Win Rate',
            marker_color='royalblue',
            error_y=dict(type='data', array=df['ci_high']-df['p_cond'], arrayminus=df['p_cond']-df['ci_low'])
        ), row=1, col=1)
        
        # Add baseline markers
        fig.add_trace(go.Scatter(
            x=df['condition'],
            y=df['p_base'],
            mode='markers',
            marker=dict(symbol='line-ew', color='red', size=30, line=dict(width=3)),
            name='Baseline'
        ), row=1, col=1)

        # 2. Volatility Chart
        fig.add_trace(go.Bar(
            x=df['condition'],
            y=df['next_vol'],
            name='Next Day Vol (Ann)',
            marker_color='orange'
        ), row=1, col=2)

        fig.update_layout(title=f"Threshold Analysis (Window: {max_win})", height=500, showlegend=True)
        return fig

    def _make_distribution_view(self):
        """Tab 4: Histogram of returns + Line for yesterday."""
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=self.returns,
            nbinsx=100,
            name='Hist Dist',
            marker_color='lightgray',
            opacity=0.7
        ))

        # Current context
        if not self.returns.empty:
            last_ret = self.returns.iloc[-1]
            last_date = self.returns.index[-1].strftime('%Y-%m-%d')
            
            fig.add_vline(x=last_ret, line_width=3, line_color="blue", annotation_text=f"Last ({last_date}): {last_ret:.2%}")
            
            # Highlight the buckets
            for b in FinancialAnalysis.BUCKET_BINS[1:-1]:
                fig.add_vline(x=b, line_width=1, line_dash="dot", line_color="#ccc")

        fig.update_layout(
            title="Historical Return Distribution & Current Context",
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
            template="plotly_white"
        )
        return fig


# ==============================================================================
# 4. Main & CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Next-Day Conditional Direction Dashboard")
    parser.add_argument("--ticker", required=True, type=str, help="Ticker symbol (e.g. SPY)")
    parser.add_argument("--windows", nargs="+", type=int, default=[252, 756, 1260], help="Lookback windows (days)")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[1, 2, 3], help="Threshold percents (e.g. 1 2 3)")
    parser.add_argument("--start", type=str, help="Start Date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End Date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--open-html", action="store_true", help="Open HTML in browser after generation")

    args = parser.parse_args()

    # 1. Ingest
    ingestion = DataIngestion(output_dir=args.output_dir)
    price_df = ingestion.get_price_history(args.ticker, args.start, args.end)
    
    if price_df.empty:
        logger.error("No data found. Exiting.")
        sys.exit(1)

    # 2. Analyze
    analyzer = FinancialAnalysis(price_df)
    stats_df, thresh_df, returns_series = analyzer.analyze_windows(args.windows, args.thresholds)
    
    # Save CSV
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"{args.ticker}_stats_{ts_str}.csv")
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"Statistics saved to {csv_path}")

    # 3. Render
    renderer = DashboardRenderer(stats_df, thresh_df, returns_series, args.ticker)
    html_path = os.path.join(args.output_dir, f"{args.ticker}_dashboard_{ts_str}.html")
    renderer.render_html(html_path, open_browser=args.open_html)

if __name__ == "__main__":
    main()
