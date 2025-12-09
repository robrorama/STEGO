#!/usr/bin/env python3
# SCRIPTNAME: ok.streaks_dashboard.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Streak Continuation & Reversion Dashboard
-----------------------------------------
Target User: Professional Options Trader / Quant
Purpose:
  1. Ingest daily OHLCV data for multiple tickers (e.g., SPY, QQQ, NVDA).
  2. Detect "Streaks": N consecutive days up/down or N days > X% threshold.
  3. Analyze "Next-Day" performance (Continuation vs. Reversion).
  4. Compute conditional statistics: Win Rate, Expectancy, Kelly Criterion, Skew.
  5. Render a fully offline, multi-tab Plotly dashboard comparing tickers.

Usage:
  python streak_continuation_dashboard.py SPY,QQQ,NVDA --start 2010-01-01 --thresholds 1.0,2.0 --open-html
"""

import argparse
import datetime
import logging
import os
import sys
import time
import webbrowser
import warnings

# --- Third Party Imports ---
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import scipy.stats
    import plotly.graph_objects as go
    import plotly.offline as py_offline
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"[CRITICAL] Missing dependency: {e}")
    print("Please install: pip install yfinance pandas numpy plotly scipy")
    sys.exit(1)

# --- Configuration & Constants ---
pd.options.mode.chained_assignment = None  # Suppress copy warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning) # Suppress minor runtime warnings if caught

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Data Ingestion Layer
# =============================================================================
class DataIngestion:
    """
    Handles downloading, caching, and sanitizing daily equity data.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_daily_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads daily data from local cache if available and covers range; 
        otherwise downloads from yfinance.
        """
        # File cache naming
        filename = f"{ticker}_OHLC_daily.csv"
        filepath = os.path.join(self.output_dir, filename)

        # 1. Try Local Load
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0)
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
                
                # Check if cache covers requested range (simple check)
                # If cache ends before requested end_date, re-download
                req_end = pd.to_datetime(end_date)
                if df.index[-1] >= req_end - datetime.timedelta(days=3): # Allow weekend gap
                    logger.info(f"Loading cached {ticker}: {filepath}")
                    return self._sanitize_df(df, ticker)
            except Exception as e:
                logger.warning(f"Cache load failed for {ticker}: {e}")

        # 2. Download
        logger.info(f"Downloading {ticker} from {start_date} to {end_date}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, group_by='column', progress=False)
            time.sleep(1) # Courtesy sleep
        except Exception as e:
            logger.error(f"yfinance download failed for {ticker}: {e}")
            return pd.DataFrame()

        # 3. Sanitize
        df_clean = self._sanitize_df(df, ticker)

        # 4. Cache
        if not df_clean.empty:
            df_clean.to_csv(filepath)
            logger.info(f"Cached {ticker} to {filepath}")

        return df_clean

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Normalizes columns from yfinance MultiIndex to flat format.
        """
        if df.empty: return df

        # Handle MultiIndex Columns
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns.values:
                if isinstance(col, tuple):
                    # Filter out empty strings and ticker name
                    clean_parts = [str(x) for x in col if str(x) and str(x).upper() != ticker.upper()]
                    if not clean_parts: clean_parts = [str(x) for x in col]
                    
                    # Keep standard names
                    standard_cols = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
                    found_std = [x for x in col if x in standard_cols]
                    
                    if found_std: new_cols.append(found_std[0])
                    else: new_cols.append("_".join(map(str, col)))
                else:
                    new_cols.append(str(col))
            df.columns = new_cols

        # Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            potential_dates = [c for c in df.columns if 'date' in c.lower()]
            if potential_dates: df = df.set_index(potential_dates[0])
        
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

        # Numeric Coercion
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        df.dropna(subset=['Close'], inplace=True)
        return df


# =============================================================================
# 2. Financial Analysis Layer
# =============================================================================
class FinancialAnalysis:
    """
    Computes returns, identifies streaks, and calculates conditional stats.
    """
    def __init__(self, data_map: dict):
        self.data_map = data_map # {ticker: df}

    def compute_returns(self):
        """Adds LogRet and PctRet to all dataframes."""
        for ticker, df in self.data_map.items():
            if df.empty: continue
            df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
            df['PctRet'] = df['Close'].pct_change()
            df['AbsRet'] = df['PctRet'].abs()
            self.data_map[ticker] = df

    def analyze_streaks(self, ticker: str, threshold_pct: float = 0.0):
        """
        Identifies streaks where returns are consistently Positive or Negative.
        Optionally requires magnitude > threshold_pct.
        
        Returns a DataFrame of streaks: 
        [StartDate, EndDate, Length, Type (Up/Down), TotalMove, NextDayRet]
        """
        df = self.data_map.get(ticker)
        if df is None or df.empty: return pd.DataFrame()

        # Threshold filter (0.0 means any up/down day)
        threshold = threshold_pct / 100.0
        
        # 1. Define Direction: 1 (Up > thresh), -1 (Down < -thresh), 0 (Noise)
        conditions = [
            (df['PctRet'] > threshold),
            (df['PctRet'] < -threshold)
        ]
        choices = [1, -1]
        df['Direction'] = np.select(conditions, choices, default=0)

        # 2. Group consecutive days
        # We create a 'group_id' that increments whenever Direction changes
        df['streak_id'] = (df['Direction'] != df['Direction'].shift(1)).cumsum()
        
        # 3. Aggregate Streaks
        # We filter out Direction == 0 (days that didn't meet threshold)
        valid_streaks = df[df['Direction'] != 0]
        
        if valid_streaks.empty:
            return pd.DataFrame()

        # Group by streak_id
        grouped = valid_streaks.groupby('streak_id')
        
        streak_list = []
        for sid, group in grouped:
            length = len(group)
            if length < 2: continue # Ignore single days, looking for streaks >= 2
            
            start_date = group.index[0]
            end_date = group.index[-1]
            direction = "UP" if group['Direction'].iloc[0] == 1 else "DOWN"
            
            # Total move during the streak (compounded)
            # (1+r1)*(1+r2)... - 1
            total_ret = (1 + group['PctRet']).prod() - 1
            
            # Find Next Day Return (Outcome)
            # We need the index location of the last day of streak
            last_idx_loc = df.index.get_loc(end_date)
            
            if last_idx_loc + 1 < len(df):
                next_ret = df['PctRet'].iloc[last_idx_loc + 1]
            else:
                next_ret = np.nan # Streak is current (still alive) or data ends

            streak_list.append({
                'Ticker': ticker,
                'Start': start_date,
                'End': end_date,
                'Length': length,
                'Type': direction,
                'Threshold': threshold_pct,
                'TotalMove': total_ret,
                'NextDayRet': next_ret
            })
            
        return pd.DataFrame(streak_list)

    def compute_conditional_stats(self, streaks_df):
        """
        Aggregates streak data into stats: Win Rate, Mean Reversion, Expectancy.
        """
        if streaks_df.empty: return pd.DataFrame()

        stats = []
        
        # Group by Type (UP/DOWN) and Length
        groups = streaks_df.groupby(['Ticker', 'Type', 'Length', 'Threshold'])
        
        for (tkr, stype, length, thresh), grp in groups:
            # Drop NaN next returns (active streaks)
            valid_outcomes = grp.dropna(subset=['NextDayRet'])
            if valid_outcomes.empty: continue
            
            next_rets_arr = valid_outcomes['NextDayRet'].values
            
            # Logic: 
            # If Streak was UP, "Continuation" means Next Day > 0.
            # If Streak was DOWN, "Continuation" means Next Day < 0.
            
            if stype == "UP":
                continuations = next_rets_arr[next_rets_arr > 0]
                reversions = next_rets_arr[next_rets_arr <= 0] # Reversion = Down
            else: # DOWN
                continuations = next_rets_arr[next_rets_arr < 0]
                reversions = next_rets_arr[next_rets_arr >= 0] # Reversion = Up

            n = len(next_rets_arr)
            win_rate = len(continuations) / n # "Win" = Trend Continues
            
            avg_win = np.mean(np.abs(continuations)) if len(continuations) > 0 else 0
            avg_loss = np.mean(np.abs(reversions)) if len(reversions) > 0 else 0
            
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Robust Skew Calculation (Fix for RuntimeWarning)
            # Check for sufficient length and non-zero variance
            if len(next_rets_arr) > 2 and np.std(next_rets_arr) > 1e-8:
                skew_val = scipy.stats.skew(next_rets_arr)
            else:
                skew_val = 0.0

            stats.append({
                'Ticker': tkr,
                'Type': stype,
                'Length': length,
                'Threshold%': thresh,
                'Count': n,
                'ContinuationRate': win_rate,
                'ReversionRate': 1.0 - win_rate,
                'AvgContinuation': avg_win,
                'AvgReversion': avg_loss,
                'Expectancy': expectancy,
                'Skew': skew_val
            })
            
        return pd.DataFrame(stats)


# =============================================================================
# 3. Dashboard Renderer Layer
# =============================================================================
class DashboardRenderer:
    """
    Renders offline Plotly HTML with subplots and tabbed analysis.
    """
    def __init__(self, tickers):
        self.tickers = tickers

    def render(self, stats_df, raw_streaks_df, output_path):
        
        # --- 1. Streak Probability Decay (Continuation Rate vs Length) ---
        fig_prob = make_subplots(rows=1, cols=2, subplot_titles=("UP Streaks: Continuation Probability", "DOWN Streaks: Continuation Probability"))
        
        colors = {'SPY': 'blue', 'QQQ': 'orange', 'NVDA': 'green', 'IWM': 'grey', 'TSLA': 'red'}
        
        # UP Streaks
        for tkr in self.tickers:
            subset = stats_df[(stats_df['Ticker'] == tkr) & (stats_df['Type'] == 'UP')]
            if subset.empty: continue
            # Sort by length
            subset = subset.sort_values('Length')
            
            col_val = colors.get(tkr, None) 
            
            fig_prob.add_trace(go.Scatter(
                x=subset['Length'], y=subset['ContinuationRate'],
                mode='lines+markers', name=f"{tkr} UP",
                line=dict(color=col_val),
                hovertemplate="Length: %{x}<br>Continuation: %{y:.1%}<br>Count: %{text}",
                text=subset['Count']
            ), row=1, col=1)

        # DOWN Streaks
        for tkr in self.tickers:
            subset = stats_df[(stats_df['Ticker'] == tkr) & (stats_df['Type'] == 'DOWN')]
            if subset.empty: continue
            subset = subset.sort_values('Length')
            
            col_val = colors.get(tkr, None) 
            
            fig_prob.add_trace(go.Scatter(
                x=subset['Length'], y=subset['ContinuationRate'],
                mode='lines+markers', name=f"{tkr} DOWN",
                line=dict(color=col_val, dash='dot'),
                hovertemplate="Length: %{x}<br>Continuation: %{y:.1%}<br>Count: %{text}",
                text=subset['Count']
            ), row=1, col=2)

        fig_prob.update_layout(title="Streak Continuation Probability (Will the trend persist tomorrow?)", height=500, yaxis_tickformat='.0%')
        fig_prob.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random (50%)")

        # --- 2. Expectancy & Edge (Bar Chart) ---
        fig_exp = go.Figure()
        
        # Filter for significant sample sizes (e.g., Length 3, 4, 5)
        # We want to see if betting on streak continuation has +EV
        sig_lengths = [3, 4, 5, 6]
        
        for tkr in self.tickers:
            tkr_stats = stats_df[(stats_df['Ticker'] == tkr) & (stats_df['Length'].isin(sig_lengths))]
            if tkr_stats.empty: continue
            
            # Combine UP/DOWN for specific lengths to show EV
            for _, row in tkr_stats.iterrows():
                label = f"{tkr} {row['Type']} {row['Length']}d"
                color = 'green' if row['Expectancy'] > 0 else 'red'
                
                fig_exp.add_trace(go.Bar(
                    x=[label], y=[row['Expectancy'] * 100], # In bps/percent
                    name=label, marker_color=color,
                    showlegend=False
                ))

        fig_exp.update_layout(title="Expectancy of Betting Continuation (Next Day Return %)", 
                              yaxis_title="Expectancy (%)", height=500)

        # --- 3. Skewness Analysis (Tail Risk) ---
        fig_skew = go.Figure()
        for tkr in self.tickers:
            skew_data = stats_df[stats_df['Ticker'] == tkr]
            if skew_data.empty: continue
            
            fig_skew.add_trace(go.Scatter(
                x=skew_data['Length'].astype(str) + " " + skew_data['Type'],
                y=skew_data['Skew'],
                mode='markers', marker=dict(size=12),
                name=tkr
            ))
        
        fig_skew.update_layout(title="Next-Day Return Skewness (Tail Risk after Streak)", 
                               yaxis_title="Skew", xaxis_title="Streak Type")


        # --- HTML Assembly ---
        div_prob = py_offline.plot(fig_prob, include_plotlyjs=False, output_type='div')
        div_exp = py_offline.plot(fig_exp, include_plotlyjs=False, output_type='div')
        div_skew = py_offline.plot(fig_skew, include_plotlyjs=False, output_type='div')
        
        plotly_js = py_offline.get_plotlyjs()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Streak Analytics</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 20px; background: #f4f4f4; }}
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; font-size: 16px; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #ccc; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; background: #fff; }}
            </style>
        </head>
        <body>
            <h2>Streak Continuation Dashboard: {", ".join(self.tickers)}</h2>
            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Prob')" id="defaultOpen">Continuation Probabilities</button>
                <button class="tablinks" onclick="openTab(event, 'Exp')">Edge & Expectancy</button>
                <button class="tablinks" onclick="openTab(event, 'Skew')">Tail Risk (Skew)</button>
            </div>

            <div id="Prob" class="tabcontent">{div_prob}</div>
            <div id="Exp" class="tabcontent">{div_exp}</div>
            <div id="Skew" class="tabcontent">{div_skew}</div>

            <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
                window.dispatchEvent(new Event('resize'));
            }}
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """

        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


# =============================================================================
# 4. Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Streak Continuation Analytics")
    parser.add_argument("tickers", type=str, help="Comma-separated tickers (e.g. SPY,QQQ)")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (defaults to today)")
    parser.add_argument("--thresholds", default="0.0,1.0", help="Comma-sep thresholds % (e.g. 0.0,1.0,2.0)")
    parser.add_argument("--output-dir", default="streak_dashboard_output", help="Output directory")
    parser.add_argument("--open-html", action="store_true", help="Auto-open HTML")

    args = parser.parse_args()
    
    ticker_list = [t.strip().upper() for t in args.tickers.split(',')]
    threshold_list = [float(x) for x in args.thresholds.split(',')]
    
    end_date = args.end if args.end else datetime.datetime.now().strftime("%Y-%m-%d")

    print("="*60)
    print(f"Starting Streak Analytics for: {ticker_list}")
    print(f"Time Range: {args.start} to {end_date}")
    print("="*60)

    # 1. Data Ingestion
    ingest = DataIngestion(args.output_dir)
    data_map = {}
    
    for tkr in ticker_list:
        print(f"\nProcessing {tkr}...")
        df = ingest.get_daily_data(tkr, args.start, end_date)
        if not df.empty:
            data_map[tkr] = df
        else:
            logger.warning(f"Skipping {tkr} due to missing data.")

    # 2. Analysis
    analyzer = FinancialAnalysis(data_map)
    analyzer.compute_returns()

    all_streaks = []
    all_stats = []

    print("\n[Analysis Phase Started]")
    
    for tkr in ticker_list:
        if tkr not in data_map: continue
        
        print(f"--> Analyzing Ticker: {tkr}")
        for thresh in threshold_list:
            # FEEDBACK: Print what is happening so user doesn't think it froze
            print(f"    [..] Calculating Streaks (Threshold: {thresh}%)...", end='', flush=True)
            
            # Heavy Calculation
            streaks = analyzer.analyze_streaks(tkr, threshold_pct=thresh)
            
            if not streaks.empty:
                all_streaks.append(streaks)
                stats = analyzer.compute_conditional_stats(streaks)
                if not stats.empty:
                    all_stats.append(stats)
            
            # FEEDBACK: Done message
            print(f"\r    [OK] Calculated Streaks (Threshold: {thresh}%)   ")

    if not all_stats:
        print("\n[WARN] No streaks found matching criteria.")
        sys.exit(0)

    final_stats = pd.concat(all_stats, ignore_index=True)
    final_streaks = pd.concat(all_streaks, ignore_index=True)
    
    # Save CSVs
    final_stats.to_csv(os.path.join(args.output_dir, "streak_stats_summary.csv"))
    final_streaks.to_csv(os.path.join(args.output_dir, "streak_log_detailed.csv"))
    
    print(f"\nAnalysis Complete. Stats saved to {args.output_dir}/streak_stats_summary.csv")

    # 3. Render
    renderer = DashboardRenderer(ticker_list)
    html_path = os.path.join(args.output_dir, "streak_dashboard.html")
    saved_file = renderer.render(final_stats, final_streaks, html_path)
    
    print(f"Dashboard saved to: {saved_file}")
    
    if args.open_html:
        webbrowser.open(f"file://{os.path.abspath(saved_file)}")

if __name__ == "__main__":
    main()
