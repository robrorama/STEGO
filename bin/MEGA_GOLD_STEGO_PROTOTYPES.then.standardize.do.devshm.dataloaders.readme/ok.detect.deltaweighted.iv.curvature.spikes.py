"""
IV Curvature Monitor & Event-Aware Dashboard (v2)
Role: Senior Quantitative Developer
Author: Michael Derby (STEGO Financial Framework)
Description: 
    - Downloads/Caches underlying & options data via yfinance (Disk-First).
    - Computes Delta-by-Expiry IV Curvature (IV_10d + IV_40d - 2*IV_25d).
    - Detects extreme regimes (99th percentile) and persistent skew.
    - Generates a standalone, offline Plotly dashboard.
    - AUTO-OPENS in default browser.
"""

import os
import sys
import time
import argparse
import datetime
import warnings
import webbrowser  # <--- Added for auto-opening
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots

# Suppress minor pandas warnings for cleaner CLI output
warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------------------------------------------------------
# 1. Data Ingestion Class
# -----------------------------------------------------------------------------
class DataIngestion:
    def __init__(self, output_dir, lookback_years):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df):
        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            level_0 = df.columns.get_level_values(0)
            level_1 = df.columns.get_level_values(1)
            
            if 'Adj Close' in level_1 and 'Adj Close' not in level_0:
                df = df.swaplevel(0, 1, axis=1)

            new_cols = []
            for col in df.columns:
                suffix = f"_{col[1]}" if col[1] else ""
                new_cols.append(f"{col[0]}{suffix}")
            df.columns = new_cols

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df.sort_index()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(how='all', inplace=True)
        return df

    def get_underlying_history(self, ticker):
        file_path = os.path.join(self.output_dir, f"{ticker}_ohlcv.csv")
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = today - datetime.timedelta(days=self.lookback_years * 365)
        
        df = pd.DataFrame()
        needs_download = True

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = self._sanitize_df(df)
                
                if not df.empty:
                    first_date = df.index.min()
                    last_date = df.index.max()
                    is_stale = (today - last_date).days > 1
                    has_enough_history = first_date <= start_date

                    if not is_stale and has_enough_history:
                        needs_download = False
                        print(f"[{ticker}] Loaded cached underlying data.")
                    else:
                        print(f"[{ticker}] Cache exists but requires backfill/update.")
            except Exception as e:
                print(f"[{ticker}] Error reading cache: {e}. Re-downloading.")
                needs_download = True

        if needs_download:
            print(f"[{ticker}] Downloading underlying history...")
            try:
                new_data = yf.download(
                    ticker, 
                    start=start_date.strftime('%Y-%m-%d'), 
                    group_by='column', 
                    progress=False,
                    auto_adjust=False 
                )
                new_data = self._sanitize_df(new_data)
                
                if not df.empty:
                    df = pd.concat([df, new_data])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
                else:
                    df = new_data
                
                df.to_csv(file_path)
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df = self._sanitize_df(df)
            except Exception as e:
                print(f"[{ticker}] Download failed: {e}")

        return df

    def get_options_chain(self, ticker):
        print(f"[{ticker}] Fetching options chain...")
        tk = yf.Ticker(ticker)
        try:
            expiries = tk.options
        except Exception:
            print(f"[{ticker}] No options found.")
            return {}

        chain_data = {}
        snapshot_ts = datetime.datetime.now()
        snapshot_str = snapshot_ts.strftime('%Y%m%d%H%M%S')

        for expiry in expiries:
            time.sleep(1) 
            try:
                opt = tk.option_chain(expiry)
                calls = opt.calls
                puts = opt.puts
                
                calls['side'] = 'call'
                puts['side'] = 'put'
                
                full_chain = pd.concat([calls, puts], ignore_index=True)
                full_chain['snapshot_timestamp'] = snapshot_ts
                full_chain['expiry'] = expiry
                
                filename = f"{ticker}_options_{expiry.replace('-','')}_{snapshot_str}.csv"
                save_path = os.path.join(self.output_dir, filename)
                full_chain.to_csv(save_path, index=False)
                
                df_disk = pd.read_csv(save_path)
                df_disk['snapshot_timestamp'] = pd.to_datetime(df_disk['snapshot_timestamp'])
                
                chain_data[expiry] = df_disk
                
            except Exception as e:
                print(f"[{ticker}] Failed to fetch expiry {expiry}: {e}")
                continue
                
        return chain_data

    def load_events_csv(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return pd.DataFrame()
        try:
            df = pd.read_csv(filepath)
            df.columns = [c.lower() for c in df.columns]
            if 'event_datetime' in df.columns:
                df['event_datetime'] = pd.to_datetime(df['event_datetime'])
            return df
        except Exception as e:
            print(f"Error loading events CSV: {e}")
            return pd.DataFrame()

    def manage_curvature_history(self, ticker, new_rows_df=None):
        hist_path = os.path.join(self.output_dir, f"{ticker}_curvature_history.csv")
        history_df = pd.DataFrame()
        if os.path.exists(hist_path):
            history_df = pd.read_csv(hist_path)
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df['snapshot_timestamp'] = pd.to_datetime(history_df['snapshot_timestamp'])

        if new_rows_df is not None and not new_rows_df.empty:
            if not history_df.empty:
                history_df = pd.concat([history_df, new_rows_df], ignore_index=True)
                history_df.drop_duplicates(subset=['date', 'ticker', 'expiry'], keep='last', inplace=True)
            else:
                history_df = new_rows_df

            history_df.to_csv(hist_path, index=False)
            history_df = pd.read_csv(hist_path)
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df['snapshot_timestamp'] = pd.to_datetime(history_df['snapshot_timestamp'])

        return history_df

# -----------------------------------------------------------------------------
# 2. Financial Analysis Class
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    def __init__(self, risk_free_rate, min_history_days):
        self.r = risk_free_rate
        self.min_history_days = min_history_days
        # FIXED: Widened bucket from 0.02 to 0.05 to capture more data
        self.bucket_width = 0.05 

    def _black_scholes_delta_put(self, S, K, T, sigma, r):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return np.nan
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = norm.cdf(d1) - 1.0
        return abs(delta)

    def compute_curvature_snapshot(self, ticker, underlying_df, chain_data):
        results = []
        if underlying_df.empty or not chain_data:
            return pd.DataFrame()

        current_price = underlying_df['Adj Close'].iloc[-1] if 'Adj Close' in underlying_df else underlying_df.iloc[-1, 0]
        current_date = pd.Timestamp.now().normalize() 

        for expiry, chain in chain_data.items():
            puts = chain[chain['side'] == 'put'].copy()
            if puts.empty: continue

            snapshot_ts = puts['snapshot_timestamp'].iloc[0]
            expiry_dt = pd.Timestamp(expiry)
            
            T = (expiry_dt - snapshot_ts).total_seconds() / (365.0 * 24 * 3600)
            if T <= 0: continue

            if 'impliedVolatility' not in puts.columns: continue

            puts['delta_abs'] = puts.apply(
                lambda row: self._black_scholes_delta_put(
                    current_price, row['strike'], T, row['impliedVolatility'], self.r
                ), axis=1
            )
            
            iv_buckets = {}
            for target_delta in [0.10, 0.25, 0.40]:
                lower = target_delta - self.bucket_width
                upper = target_delta + self.bucket_width
                
                subset = puts[(puts['delta_abs'] >= lower) & (puts['delta_abs'] <= upper)]
                
                if subset.empty:
                    iv_buckets[target_delta] = np.nan
                else:
                    if 'openInterest' in subset.columns and subset['openInterest'].sum() > 0:
                        w_avg = np.average(subset['impliedVolatility'], weights=subset['openInterest'])
                        iv_buckets[target_delta] = w_avg
                    else:
                        iv_buckets[target_delta] = subset['impliedVolatility'].mean()

            iv10 = iv_buckets[0.10]
            iv25 = iv_buckets[0.25]
            iv40 = iv_buckets[0.40]

            if np.isnan(iv10) or np.isnan(iv25) or np.isnan(iv40):
                curvature = np.nan
            else:
                curvature = iv10 + iv40 - (2 * iv25)

            results.append({
                'date': current_date,
                'snapshot_timestamp': snapshot_ts,
                'ticker': ticker,
                'expiry': expiry,
                'days_to_expiry': T * 365.0,
                'iv_10d': iv10,
                'iv_25d': iv25,
                'iv_40d': iv40,
                'curvature': curvature,
                'underlying_price': current_price
            })

        return pd.DataFrame(results)

    def analyze_history_and_detect_regimes(self, history_df):
        if history_df.empty:
            return history_df

        history_df = history_df.sort_values(['expiry', 'date'])
        
        results = []
        for expiry, group in history_df.groupby('expiry'):
            group = group.sort_values('date').copy()
            
            mean = group['curvature'].expanding(min_periods=self.min_history_days).mean()
            std = group['curvature'].expanding(min_periods=self.min_history_days).std()
            group['curvature_zscore'] = (group['curvature'] - mean) / std.replace(0, np.nan)
            
            group['curvature_percentile'] = group['curvature'].expanding(min_periods=self.min_history_days).rank(pct=True) * 100.0
            
            group['curvature_change_vs_prev'] = group['curvature'].diff()
            
            high_mask = group['curvature_percentile'] >= 95.0
            group['persistent_positive'] = high_mask.rolling(window=3).sum() >= 3

            group['extreme_flag'] = group['curvature_percentile'] >= 99.0
            group['high_flag'] = (group['curvature_percentile'] >= 95.0) & (group['curvature_percentile'] < 99.0)
            
            results.append(group)
            
        if not results:
            return history_df
            
        analyzed_df = pd.concat(results)
        counts = history_df.groupby('expiry')['date'].transform('count')
        analyzed_df['limited_history'] = counts < self.min_history_days

        return analyzed_df

    def cross_reference_events(self, alerts_df, events_df):
        if alerts_df.empty: return alerts_df
            
        alerts_df = alerts_df.copy()
        alerts_df['event_type'] = np.nan
        alerts_df['event_description'] = np.nan
        alerts_df['event_datetime'] = np.nan
        
        if events_df.empty: return alerts_df

        events_df['event_datetime'] = pd.to_datetime(events_df['event_datetime'])
        alerts_df['date'] = pd.to_datetime(alerts_df['date'])
        
        for idx, row in alerts_df.iterrows():
            obs_date = row['date']
            ticker = row['ticker']
            
            mask = (events_df['event_datetime'] >= obs_date - datetime.timedelta(days=1)) & \
                   (events_df['event_datetime'] <= obs_date + datetime.timedelta(days=1))
            ticker_mask = (events_df['ticker'] == ticker) | (events_df['ticker'].isna())
            
            matched = events_df[mask & ticker_mask]
            
            if not matched.empty:
                desc = "; ".join(matched['description'].astype(str).tolist())
                types = "; ".join(matched['event_type'].astype(str).unique().tolist())
                alerts_df.at[idx, 'event_type'] = types
                alerts_df.at[idx, 'event_description'] = desc
                alerts_df.at[idx, 'event_datetime'] = matched['event_datetime'].iloc[0]
            
        return alerts_df

# -----------------------------------------------------------------------------
# 3. Dashboard Renderer Class
# -----------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self, output_dir, filename):
        self.output_dir = output_dir
        self.filename = filename

    def _get_plotly_js(self):
        return py_offline.get_plotlyjs()

    def generate_dashboard(self, data_map):
        html_content = []
        html_content.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IV Curvature Monitor</title>
            <script>{self._get_plotly_js()}</script>
            <style>
                body {{ font-family: "Segoe UI", sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; padding: 20px; }}
                .container {{ max-width: 1600px; margin: auto; }}
                h1, h2, h3 {{ color: #ffffff; }}
                .tab {{ overflow: hidden; border-bottom: 1px solid #444; margin-bottom: 20px; }}
                .tab button {{ background-color: #333; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-weight: bold; }}
                .tab button:hover {{ background-color: #555; }}
                .tab button.active {{ background-color: #007acc; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
                .card {{ background-color: #2d2d2d; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
                th {{ background-color: #333; }}
                .regime-extreme {{ color: #ff4d4d; font-weight: bold; }}
                .regime-high {{ color: #ffae42; font-weight: bold; }}
                .regime-normal {{ color: #4caf50; }}
            </style>
        </head>
        <body>
        <div class="container">
            <h1>Quant IV Curvature Monitor</h1>
            <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """)

        tickers = list(data_map.keys())
        html_content.append('<div class="tab">')
        html_content.append(f'<button class="tablinks active" onclick="openTab(event, \'Overview\')">Overview & Alerts</button>')
        for ticker in tickers:
            html_content.append(f'<button class="tablinks" onclick="openTab(event, \'{ticker}\')">{ticker}</button>')
        html_content.append('</div>')

        html_content.append('<div id="Overview" class="tabcontent" style="display:block;">')
        html_content.append(self._generate_alerts_html(data_map))
        html_content.append('</div>')

        for ticker in tickers:
            df = data_map[ticker]
            if df.empty:
                html_content.append(f'<div id="{ticker}" class="tabcontent"><p>No data available.</p></div>')
                continue

            fig_bars = self._make_curvature_bar_chart(df)
            div_bars = py_offline.plot(fig_bars, include_plotlyjs=False, output_type='div')

            fig_ts = self._make_time_series_chart(df)
            div_ts = py_offline.plot(fig_ts, include_plotlyjs=False, output_type='div')

            fig_hm = self._make_heatmap(df)
            div_hm = py_offline.plot(fig_hm, include_plotlyjs=False, output_type='div')

            html_content.append(f"""
            <div id="{ticker}" class="tabcontent">
                <div class="card">
                    <h3>Current Curvature Structure (Snapshots)</h3>
                    {div_bars}
                </div>
                <div class="card">
                    <h3>Historical Curvature & Percentiles (Front Expiries)</h3>
                    {div_ts}
                </div>
                <div class="card">
                    <h3>Regime Heatmap (Expiry vs Date)</h3>
                    {div_hm}
                </div>
            </div>
            """)

        html_content.append("""
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
        </script>
        </body>
        </html>
        """)

        out_path = os.path.join(self.output_dir, self.filename)
        with open(out_path, "w", encoding='utf-8') as f:
            f.write("\n".join(html_content))
        print(f"Dashboard saved to: {out_path}")
        
        # FIXED: Auto-open browser
        try:
            webbrowser.open('file://' + os.path.abspath(out_path))
        except Exception as e:
            print(f"Could not auto-open browser: {e}")

    def _generate_alerts_html(self, data_map):
        all_alerts = []
        for ticker, df in data_map.items():
            if df.empty: continue
            latest_date = df['date'].max()
            latest = df[df['date'] == latest_date].copy()
            
            # FIXED: If flags exist, use them. If not (Day 1), show top curvature rows.
            alerts = latest[(latest['high_flag']) | (latest['extreme_flag'])].copy()
            if alerts.empty:
                # Fallback: Show top 10 highest raw curvature for context
                alerts = latest.sort_values('curvature', ascending=False).head(10).copy()
                alerts['note'] = "Top Raw (No History Flags)"
            else:
                alerts['note'] = "Flagged"
                
            if not alerts.empty:
                all_alerts.append(alerts)
        
        if not all_alerts:
            return "<div class='card'><p>No data detected in current snapshot.</p></div>"
        
        combined = pd.concat(all_alerts)
        combined.sort_values('days_to_expiry', inplace=True)
        
        html = ["<div class='card'><h3>Curvature Monitor (Snapshot)</h3><table>"]
        html.append("<thead><tr><th>Ticker</th><th>Expiry</th><th>DTE</th><th>Curvature</th><th>Percentile</th><th>Z-Score</th><th>Event</th></tr></thead><tbody>")
        
        for _, row in combined.iterrows():
            regime_class = "regime-normal"
            if row.get('extreme_flag'): regime_class = "regime-extreme"
            elif row.get('high_flag'): regime_class = "regime-high"
            
            event_type = row.get('event_type')
            event_desc = row.get('event_description')
            if pd.notnull(event_type):
                event_txt = f"{event_type}: {event_desc}"
            else:
                event_txt = "-"
            if len(str(event_txt)) > 50: event_txt = str(event_txt)[:47] + "..."

            # Handle NaNs in percentile gracefully for display
            pct_disp = f"{row['curvature_percentile']:.1f}%" if pd.notnull(row['curvature_percentile']) else "N/A"
            z_disp = f"{row['curvature_zscore']:.2f}" if pd.notnull(row['curvature_zscore']) else "N/A"

            html.append(f"""
            <tr>
                <td><b>{row['ticker']}</b></td>
                <td>{row['expiry']}</td>
                <td>{row['days_to_expiry']:.1f}</td>
                <td>{row['curvature']:.4f}</td>
                <td class="{regime_class}">{pct_disp}</td>
                <td>{z_disp}</td>
                <td>{event_txt}</td>
            </tr>
            """)
        html.append("</tbody></table></div>")
        return "".join(html)

    def _make_curvature_bar_chart(self, df):
        latest_date = df['date'].max()
        data = df[df['date'] == latest_date].sort_values('days_to_expiry')
        
        colors = []
        for p in data['curvature_percentile']:
            if pd.isnull(p): colors.append('#888') # Grey for N/A
            elif p >= 99: colors.append('#ff4d4d')
            elif p >= 95: colors.append('#ffae42')
            else: colors.append('#007acc')

        fig = go.Figure(go.Bar(
            x=data['expiry'],
            y=data['curvature'],
            marker_color=colors,
            name='Curvature'
        ))
        
        fig.update_layout(
            title=f"Term Structure Curvature (Snapshot: {latest_date.date()})",
            xaxis_title="Expiry",
            yaxis_title="Curvature",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig
    def _make_time_series_chart(self, df):
        latest_date = df['date'].max()
        snapshot = df[df['date'] == latest_date].sort_values('days_to_expiry')
        target_expiries = snapshot['expiry'].head(3).tolist()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for i, expiry in enumerate(target_expiries):
            subset = df[df['expiry'] == expiry].sort_values('date')
            if subset.empty: continue
            
            # FIX: Changed mode to 'lines+markers' so single points are visible
            fig.add_trace(go.Scatter(
                x=subset['date'], y=subset['curvature'],
                mode='lines+markers',  # <--- CHANGED THIS
                name=f'{expiry} Curv',
                line=dict(width=2), opacity=0.8,
                marker=dict(size=6)    # <--- Added marker size
            ), secondary_y=False)

            if 'event_type' in subset.columns:
                events = subset[pd.notnull(subset['event_type'])]
                if not events.empty:
                    fig.add_trace(go.Scatter(
                        x=events['date'], y=events['curvature'],
                        mode='markers', name='Event',
                        marker=dict(symbol='star', size=12, color='white'),
                        text=events['event_type'],
                        hoverinfo='text+x'
                    ), secondary_y=False)

        fig.update_layout(
            title="Curvature Time Series (Front Expiries)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            height=500
        )
        return fig

    def _make_time_series_chartOLD(self, df):
        latest_date = df['date'].max()
        snapshot = df[df['date'] == latest_date].sort_values('days_to_expiry')
        target_expiries = snapshot['expiry'].head(3).tolist()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for i, expiry in enumerate(target_expiries):
            subset = df[df['expiry'] == expiry].sort_values('date')
            if subset.empty: continue
            
            fig.add_trace(go.Scatter(
                x=subset['date'], y=subset['curvature'],
                mode='lines', name=f'{expiry} Curv',
                line=dict(width=2), opacity=0.8
            ), secondary_y=False)

            if 'event_type' in subset.columns:
                events = subset[pd.notnull(subset['event_type'])]
                if not events.empty:
                    fig.add_trace(go.Scatter(
                        x=events['date'], y=events['curvature'],
                        mode='markers', name='Event',
                        marker=dict(symbol='star', size=12, color='white'),
                        text=events['event_type'],
                        hoverinfo='text+x'
                    ), secondary_y=False)

        fig.update_layout(
            title="Curvature Time Series (Front Expiries)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            height=500
        )
        return fig

    def _make_heatmap(self, df):
        pivot = df.pivot_table(index='date', columns='expiry', values='curvature_percentile')
        pivot = pivot[sorted(pivot.columns)]
        
        # Handle empty/NaN pivot (Day 1)
        z_vals = pivot.values
        if np.all(np.isnan(z_vals)):
            # If all NaNs, plot raw curvature instead for Day 1 visibility
            pivot = df.pivot_table(index='date', columns='expiry', values='curvature')
            pivot = pivot[sorted(pivot.columns)]
            z_vals = pivot.values
            title_txt = "Curvature (Raw) Map [History Insufficient for %]"
        else:
            title_txt = "Curvature Percentile Regime Map"

        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdBu_r',
            colorbar=dict(title='Value')
        ))
        
        fig.update_layout(
            title=title_txt,
            xaxis_title="Expiry",
            yaxis_title="Observation Date",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig

def main():
    parser = argparse.ArgumentParser(description="Delta-by-Expiry IV Curvature Monitor")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='Tickers to process')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Data directory')
    parser.add_argument('--lookback', type=float, default=1.0, help='Lookback years for history')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate (e.g., 0.04)')
    parser.add_argument('--events-csv', type=str, default=None, help='Path to events CSV')
    parser.add_argument('--html-filename', type=str, default='iv_curvature_dashboard.html')
    parser.add_argument('--min-history-days', type=int, default=10, help='Min days for reliable Z-score')

    args = parser.parse_args()

    ingestion = DataIngestion(args.output_dir, args.lookback)
    analysis = FinancialAnalysis(args.risk_free_rate, args.min_history_days)
    renderer = DashboardRenderer(args.output_dir, args.html_filename)

    events_df = pd.DataFrame()
    if args.events_csv:
        events_df = ingestion.load_events_csv(args.events_csv)
        print(f"Loaded {len(events_df)} events.")

    processed_data = {}

    for ticker in args.tickers:
        print(f"\n--- Processing {ticker} ---")
        underlying_df = ingestion.get_underlying_history(ticker)
        chain_data = ingestion.get_options_chain(ticker)
        snapshot_df = analysis.compute_curvature_snapshot(ticker, underlying_df, chain_data)
        
        if not snapshot_df.empty:
            print(f"[{ticker}] Computed curvature for {len(snapshot_df)} expiries.")
            history_df = ingestion.manage_curvature_history(ticker, snapshot_df)
            analyzed_df = analysis.analyze_history_and_detect_regimes(history_df)
            final_df = analysis.cross_reference_events(analyzed_df, events_df)
            processed_data[ticker] = final_df
        else:
            print(f"[{ticker}] No valid curvature data derived.")
            processed_data[ticker] = pd.DataFrame()

    renderer.generate_dashboard(processed_data)
    print("\nJob Complete.")

if __name__ == "__main__":
    main()
