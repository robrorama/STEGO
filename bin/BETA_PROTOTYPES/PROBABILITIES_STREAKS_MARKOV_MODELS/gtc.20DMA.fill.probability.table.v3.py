# SCRIPTNAME: ok.03.gtc.20DMA.fill.probability.table.v3.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import os
import sys
import time
import json
import datetime
import warnings
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
import scipy.stats as stats

# Suppress minor pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 1. Data Ingestion Layer (Disk-First, Sanitization)
# ==============================================================================
class DataIngestion:
    """
    Handles all I/O, yfinance downloads, disk caching, and data sanitization.
    Strictly follows the 'Disk-First' architecture.
    """
    def __init__(self, output_dir: str, lookback_years: float):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Orchestrates the fetch logic: Check Disk -> Check Freshness -> Download if needed -> Save -> Read from Disk.
        Returns a clean DataFrame with canonical columns.
        """
        csv_path = os.path.join(self.output_dir, f"{ticker}.csv")
        today = datetime.datetime.now()
        start_date = today - datetime.timedelta(days=int(self.lookback_years * 365))
        
        needs_download = True
        
        # 1. Check if exists
        if os.path.exists(csv_path):
            try:
                # Force UTC then localize to None to handle timezone mess
                df_temp = pd.read_csv(csv_path, index_col=0)
                df_temp.index = pd.to_datetime(df_temp.index, utc=True).tz_localize(None)

                if not df_temp.empty:
                    last_date = df_temp.index[-1]
                    first_date = df_temp.index[0]
                    
                    # Shadow Backfill Logic:
                    # If data is stale (> 2 days old) OR doesn't go back far enough
                    is_stale = (today - last_date).days > 2
                    is_short = first_date > (start_date + datetime.timedelta(days=10)) # Buffer
                    
                    if not is_stale and not is_short:
                        needs_download = False
            except Exception as e:
                print(f"[{ticker}] Local cache invalid ({e}), re-downloading...")
                needs_download = True

        # 2. Download if needed
        if needs_download:
            print(f"[{ticker}] Downloading/Updating history (Lookback: {self.lookback_years}y)...")
            time.sleep(1.0) 
            
            try:
                raw_df = yf.download(
                    ticker, 
                    start=start_date.strftime('%Y-%m-%d'), 
                    end=today.strftime('%Y-%m-%d'), 
                    progress=False,
                    group_by='column', 
                    auto_adjust=False 
                )
                
                if raw_df.empty:
                    print(f"[{ticker}] Warning: yfinance returned empty data.")
                    return pd.DataFrame()

                clean_df = self._sanitize_df(raw_df, ticker)
                clean_df.to_csv(csv_path)
                
            except Exception as e:
                print(f"[{ticker}] Download failed: {e}")
                return pd.DataFrame()

        # 3. Strict Re-read from Disk (The "Disk-First" Mandate)
        if os.path.exists(csv_path):
            try:
                final_df = pd.read_csv(csv_path, index_col=0)
                final_df.index = pd.to_datetime(final_df.index, utc=True).tz_localize(None)
                return final_df.sort_index()
            except Exception as e:
                print(f"[{ticker}] Final disk read failed: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        The 'Universal Fixer'. Normalizes yfinance multi-index mess into a 
        clean, flat OHLCV DataFrame.
        """
        # 1. Aggressive Column Flattening
        if isinstance(df.columns, pd.MultiIndex):
            target_level = None
            for i in range(df.columns.nlevels):
                level_values = df.columns.get_level_values(i)
                if 'Close' in level_values:
                    target_level = i
                    break
            
            if target_level is not None:
                df.columns = df.columns.get_level_values(target_level)
            else:
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)

        # 2. Rename & Filter
        rename_map = {
            'Adj Close': 'Adj Close',
            'adj close': 'Adj Close',
            'Volume': 'Volume',
            'volume': 'Volume'
        }
        df = df.rename(columns=rename_map)
        
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        final_cols = [c for c in expected_cols if c in df.columns]
        
        df = df[final_cols].copy()

        # 3. Strict Datetime Index
        df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index(ascending=True)

        # 4. Numeric Coercion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Close' in df.columns and 'Adj Close' in df.columns:
            df = df.dropna(subset=['Close', 'Adj Close'])
        else:
            raise ValueError(f"Critical columns missing. Found: {df.columns.tolist()}")

        return df


# ==============================================================================
# 2. Financial Analysis Layer (Logic & Statistics)
# ==============================================================================
class FinancialAnalysis:
    """
    Core quantitative logic. 
    Calculates DMAs, detects events, computes fill probabilities.
    """
    def __init__(self, dma_window: int, fill_horizons: List[int], 
                 risk_free_rate: float, min_sample_size: int, min_prob: float, ci_alpha: float):
        self.dma_window = dma_window
        self.fill_horizons = fill_horizons
        self.rfr = risk_free_rate
        self.min_sample_size = min_sample_size
        self.min_prob = min_prob  # NEW: Configurable probability threshold
        self.alpha = ci_alpha

    def analyze_ticker(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Runs the full analysis pipeline for a single ticker."""
        if df.empty or len(df) < self.dma_window:
            return None

        # 1. Compute DMA
        df['DMA'] = df['Adj Close'].rolling(window=self.dma_window).mean()
        
        # 2. Define Events
        # Touch
        touch_condition = (df['Low'] <= df['DMA']) & (df['High'] >= df['DMA'])
        touch_prev = touch_condition.shift(1).fillna(False)
        df['event_touch'] = touch_condition & (~touch_prev)

        # Cross Up
        prev_close = df['Adj Close'].shift(1)
        prev_dma = df['DMA'].shift(1)
        df['event_cross_up'] = (prev_close < prev_dma) & (df['Adj Close'] >= df['DMA'])

        # Cross Down
        df['event_cross_dn'] = (prev_close > prev_dma) & (df['Adj Close'] <= df['DMA'])

        results = []
        gtc_heuristics = []
        event_types = ['event_touch', 'event_cross_up', 'event_cross_dn']

        # 3. Fill Probability & Stats
        for etype in event_types:
            event_dates = df.index[df[etype]].tolist()
            
            for horizon in self.fill_horizons:
                successes = 0
                trials = 0
                time_to_fills = []

                for date in event_dates:
                    try:
                        idx_loc = df.index.get_loc(date)
                    except KeyError:
                        continue
                        
                    start_idx = idx_loc + 1
                    end_idx = min(idx_loc + horizon, len(df) - 1)
                    
                    if start_idx >= len(df):
                        continue

                    # The forward window
                    window = df.iloc[start_idx : end_idx + 1]
                    if window.empty:
                        continue
                        
                    trials += 1
                    
                    # Fill check
                    fill_mask = (window['Low'] <= window['DMA']) & (window['High'] >= window['DMA'])
                    
                    if fill_mask.any():
                        successes += 1
                        fill_idx = np.where(fill_mask)[0][0]
                        time_to_fills.append(fill_idx + 1)

                p_hat, ci_low, ci_high = self._clopper_pearson(successes, trials, self.alpha)
                
                mean_ttf = np.mean(time_to_fills) if time_to_fills else np.nan
                
                record = {
                    "ticker": ticker,
                    "event_type": etype.replace("event_", ""),
                    "fill_horizon": horizon,
                    "trials": trials,
                    "successes": successes,
                    "p_hat": p_hat,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "mean_time_to_fill": mean_ttf
                }
                results.append(record)

                # 4. GTC Heuristics Generation - RELAXED LOGIC
                # Use self.min_prob instead of hardcoded 0.70
                if trials >= self.min_sample_size and p_hat >= self.min_prob:
                    self._generate_heuristic(df, ticker, etype, horizon, record, gtc_heuristics)

        return {
            "stats": results,
            "heuristics": gtc_heuristics,
            "df_processed": df
        }

    def _clopper_pearson(self, k, n, alpha) -> Tuple[float, float, float]:
        if n == 0:
            return np.nan, np.nan, np.nan
        
        p_hat = k / n
        if k == 0:
            ci_low = 0.0
        else:
            ci_low = stats.beta.ppf(alpha / 2, k, n - k + 1)
        if k == n:
            ci_high = 1.0
        else:
            ci_high = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
        return p_hat, ci_low, ci_high

    def _generate_heuristic(self, df, ticker, etype, horizon, stats_rec, out_list):
        current_dma = df['DMA'].iloc[-1]
        if np.isnan(current_dma):
             # Safety check if script runs before market open or on bad data
             current_dma = 0.0

        clean_etype = etype.replace("event_", "")
        action = "none"
        offset = 0.0
        
        note = ""

        if clean_etype == "cross_dn":
            action = "buy_limit"
            offset = -0.001 
            target = current_dma * (1 + offset)
            note = f"Place GTC BUY LMT @ {target:.2f} (DMA -0.1%). Exp: {horizon} days."
            
        elif clean_etype == "cross_up":
            action = "sell_limit"
            offset = 0.001 
            target = current_dma * (1 + offset)
            note = f"Place GTC SELL LMT @ {target:.2f} (DMA +0.1%). Exp: {horizon} days."
        
        elif clean_etype == "touch":
             action = "bracket"
             note = f"High prob of sticking to DMA. Range trade DMA +/- 0.5%."

        heuristic = {
            "ticker": ticker,
            "event_type": clean_etype,
            "horizon": horizon,
            "p_hat": stats_rec['p_hat'],
            "ci_low": stats_rec['ci_low'],
            "ci_high": stats_rec['ci_high'],
            "trials": stats_rec['trials'],
            "suggested_action": action,
            "heuristic_text": note
        }
        out_list.append(heuristic)


# ==============================================================================
# 3. Dashboard Renderer Layer (Visualization & HTML)
# ==============================================================================
class DashboardRenderer:
    def __init__(self, all_stats: List[Dict], all_heuristics: List[Dict], 
                 ticker_dfs: Dict[str, pd.DataFrame], output_dir: str, filename: str,
                 min_sample: int, min_prob: float):
        self.stats_df = pd.DataFrame(all_stats)
        self.heuristics_df = pd.DataFrame(all_heuristics)
        self.ticker_dfs = ticker_dfs
        self.output_path = os.path.join(output_dir, filename)
        self.min_sample = min_sample
        self.min_prob = min_prob

    def render(self):
        print(f"Rendering dashboard to {self.output_path}...")
        
        fig_table = self._build_summary_table()
        div_table = py_offline.plot(fig_table, include_plotlyjs=False, output_type='div')

        fig_heatmap = self._build_heatmaps()
        div_heatmap = py_offline.plot(fig_heatmap, include_plotlyjs=False, output_type='div')

        fig_timeline = self._build_timelines()
        div_timeline = py_offline.plot(fig_timeline, include_plotlyjs=False, output_type='div')

        fig_gtc = self._build_gtc_table()
        div_gtc = py_offline.plot(fig_gtc, include_plotlyjs=False, output_type='div')

        self._write_html(div_table, div_heatmap, div_timeline, div_gtc)

    def _build_summary_table(self):
        if self.stats_df.empty:
            return go.Figure()
        
        df = self.stats_df.copy().sort_values(['ticker', 'event_type', 'fill_horizon'])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Ticker', 'Event', 'Horizon (Days)', 'Trials', 'Successes', 'Prob (P_hat)', 'CI Low', 'CI High', 'Avg TimeToFill'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[df.ticker, df.event_type, df.fill_horizon, df.trials, df.successes, 
                        df.p_hat.map('{:.1%}'.format), 
                        df.ci_low.map('{:.1%}'.format), 
                        df.ci_high.map('{:.1%}'.format),
                        df.mean_time_to_fill.map('{:.1f}'.format)],
                fill_color='lavender',
                align='left'
            )
        )])
        fig.update_layout(title="Full Backtest Summary", margin=dict(l=20, r=20, t=40, b=20))
        return fig

    def _build_heatmaps(self):
        if self.stats_df.empty:
            return go.Figure()
        
        df = self.stats_df.copy()
        df['y_label'] = df['ticker'] + " - " + df['event_type']
        
        pivot = df.pivot(index='y_label', columns='fill_horizon', values='p_hat')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            zmin=0, zmax=1,
            text=np.around(pivot.values, 2),
            texttemplate="%{text}",
            showscale=True
        ))
        
        fig.update_layout(
            title="Fill Probability Heatmap (X=Horizon, Y=Event)",
            xaxis_title="Days Horizon",
            height=max(400, len(pivot)*30)
        )
        return fig

    def _build_timelines(self):
        if not self.ticker_dfs:
            return go.Figure()
        
        # Grab first ticker
        ticker = list(self.ticker_dfs.keys())[0]
        df = self.ticker_dfs[ticker]
        df_recent = df.tail(252).copy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Adj Close'], mode='lines', name='Price', line=dict(color='white', width=1)))
        fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['DMA'], mode='lines', name='20-DMA', line=dict(color='orange', width=2)))
        
        cross_up = df_recent[df_recent['event_cross_up']]
        cross_dn = df_recent[df_recent['event_cross_dn']]
        
        fig.add_trace(go.Scatter(x=cross_up.index, y=cross_up['DMA'], mode='markers', name='Cross Up', marker=dict(color='green', symbol='triangle-up', size=10)))
        fig.add_trace(go.Scatter(x=cross_dn.index, y=cross_dn['DMA'], mode='markers', name='Cross Down', marker=dict(color='red', symbol='triangle-down', size=10)))
        
        fig.update_layout(
            title=f"Event Timeline: {ticker} (Last 1 Year)",
            template="plotly_dark",
            height=600
        )
        return fig

    def _build_gtc_table(self):
        if self.heuristics_df.empty:
            return go.Figure(layout=dict(title=f"No configurations found with Trials >= {self.min_sample} and Prob >= {self.min_prob:.0%}"))
            
        df = self.heuristics_df.copy()
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Ticker', 'Event', 'Horizon', 'Prob', 'Action', 'Instruction'], fill_color='darkred', font=dict(color='white')),
            cells=dict(values=[
                df.ticker, df.event_type, df.horizon, df.p_hat.map('{:.1%}'.format),
                df.suggested_action, df.heuristic_text
            ], fill_color='grey', font=dict(color='white'))
        )])
        fig.update_layout(title="GTC Limit Heuristics Cheatsheet", margin=dict(l=20, r=20, t=40, b=20))
        return fig

    def _write_html(self, table_div, heatmap_div, timeline_div, gtc_div):
        js_resize_fix = """
        <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
            window.dispatchEvent(new Event('resize'));
        }
        document.addEventListener("DOMContentLoaded", function() {
           document.getElementById("defaultOpen").click();
        });
        </script>
        """

        css = """
        <style>
        body {font-family: Arial, sans-serif; background-color: #1e1e1e; color: #ddd;}
        .tab { overflow: hidden; border: 1px solid #ccc; background-color: #333; }
        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: white; font-size: 17px;}
        .tab button:hover { background-color: #ddd; color: black; }
        .tab button.active { background-color: #ccc; color: black; }
        .tabcontent { display: none; padding: 6px 12px; border-top: none; }
        </style>
        """

        plotly_js = py_offline.get_plotlyjs()

        html_content = f"""
        <html>
        <head>
            <title>20-DMA Fill Analysis</title>
            <script type="text/javascript">{plotly_js}</script>
            {css}
        </head>
        <body>
            <h2>20-DMA Fill Probability & GTC Dashboard</h2>
            <div class="tab">
              <button class="tablinks" onclick="openTab(event, 'Summary')" id="defaultOpen">Summary Table</button>
              <button class="tablinks" onclick="openTab(event, 'Heatmap')">Probability Heatmaps</button>
              <button class="tablinks" onclick="openTab(event, 'Timeline')">Event Timeline</button>
              <button class="tablinks" onclick="openTab(event, 'GTC')">GTC Cheatsheet</button>
            </div>

            <div id="Summary" class="tabcontent">
              {table_div}
            </div>

            <div id="Heatmap" class="tabcontent">
              {heatmap_div}
            </div>

            <div id="Timeline" class="tabcontent">
              {timeline_div}
            </div>
            
            <div id="GTC" class="tabcontent">
              <p>Current Filters: Trials >= {self.min_sample}, Probability >= {self.min_prob:.0%}.</p>
              {gtc_div}
            </div>

            {js_resize_fix}
        </body>
        </html>
        """
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Dashboard successfully written to {self.output_path}")


# ==============================================================================
# 4. Main Controller
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="20-DMA Fill Probability Dashboard")
    
    # Required/Core
    parser.add_argument("--tickers", type=str, default="SPY,QQQ,IWM", help="Comma-separated tickers")
    parser.add_argument("--output-dir", type=str, default="./market_data", help="Root directory for data")
    parser.add_argument("--lookback", type=float, default=1, help="Years of history")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk free rate (0.04 = 4%)")
    
    # Tuning - UPDATED DEFAULTS TO BE LESS RIGID
    parser.add_argument("--dma-window", type=int, default=20)
    parser.add_argument("--fill-horizons", type=str, default="1,2,3,5,10,20")
    parser.add_argument("--min-sample-size", type=int, default=10, help="Lowered default from 50 to 10")
    parser.add_argument("--min-prob", type=float, default=0.55, help="Minimum probability for GTC list (default 0.55)")
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument("--html-filename", type=str, default="dma_fill_dashboard.html")

    args = parser.parse_args()
    
    ticker_list = [t.strip().upper() for t in args.tickers.split(",")]
    horizon_list = [int(h.strip()) for h in args.fill_horizons.split(",")]

    ingestor = DataIngestion(args.output_dir, args.lookback)
    
    analyzer = FinancialAnalysis(
        dma_window=args.dma_window,
        fill_horizons=horizon_list,
        risk_free_rate=args.risk_free_rate,
        min_sample_size=args.min_sample_size,
        min_prob=args.min_prob,
        ci_alpha=args.ci_alpha
    )

    all_stats = []
    all_heuristics = []
    processed_dfs = {}

    for ticker in ticker_list:
        print(f"Processing {ticker}...")
        df = ingestor.get_ticker_data(ticker)
        
        if df.empty:
            print(f"Skipping {ticker} due to missing data.")
            continue
            
        analysis_output = analyzer.analyze_ticker(df, ticker)
        
        if analysis_output:
            all_stats.extend(analysis_output['stats'])
            all_heuristics.extend(analysis_output['heuristics'])
            processed_dfs[ticker] = analysis_output['df_processed']

    if all_stats:
        summary_csv = os.path.join(args.output_dir, "dma_fill_summary.csv")
        pd.DataFrame(all_stats).to_csv(summary_csv, index=False)
        print(f"Summary CSV saved to {summary_csv}")
        
        summary_json = os.path.join(args.output_dir, "dma_fill_summary.json")
        with open(summary_json, 'w') as f:
            json.dump({"meta": vars(args), "rows": all_stats}, f, indent=2, default=str)
            
        gtc_json = os.path.join(args.output_dir, "dma_fill_gtc_heuristics.json")
        with open(gtc_json, 'w') as f:
            json.dump({"rules": all_heuristics}, f, indent=2, default=str)

    renderer = DashboardRenderer(all_stats, all_heuristics, processed_dfs, 
                                 args.output_dir, args.html_filename, 
                                 args.min_sample_size, args.min_prob)
    renderer.render()

if __name__ == "__main__":
    main()
