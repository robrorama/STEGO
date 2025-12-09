import argparse
import os
import glob
import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots

# Try importing sklearn for Robust Regression, handle if missing
try:
    from sklearn.linear_model import HuberRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==========================================
# 1. DATA INGESTION (Disk-First Pipeline)
# ==========================================
class DataIngestion:
    """
    Handles all I/O operations. strictly follows the Disk-First philosophy.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df.set_index('Date', inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def get_spot_data(self, ticker: str, lookback_years: int = 1) -> pd.DataFrame:
        file_path = os.path.join(self.output_dir, f"{ticker}_spot.csv")
        if os.path.exists(file_path):
            print(f"[{ticker}] Loading Spot data from disk...")
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            return df
        
        print(f"[{ticker}] Downloading Spot data via yfinance...")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
        try:
            raw_df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True, group_by='column')
            time.sleep(1)
            clean_df = self._sanitize_df(raw_df)
            clean_df.to_csv(file_path)
            return clean_df
        except Exception as e:
            print(f"Error downloading spot for {ticker}: {e}")
            return pd.DataFrame()

    def get_option_data(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
        """
        Returns DataFrames for the two most recent dates.
        If only 1 file exists, returns (df_recent, EmptyDataFrame, date_recent, "Realized Volatility Proxy")
        """
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        today_file = os.path.join(self.output_dir, f"{ticker}_options_{today_str}.csv")

        # 1. Ensure Today's Data Exists
        if not os.path.exists(today_file):
            print(f"[{ticker}] Fetching Option Chain for {today_str}...")
            yf_ticker = yf.Ticker(ticker)
            all_opts = []
            try:
                expiries = yf_ticker.options
                if expiries:
                    for exp in expiries:
                        try:
                            chain = yf_ticker.option_chain(exp)
                            calls = chain.calls.copy()
                            calls['Type'] = 'Call'
                            calls['Expiration'] = exp
                            puts = chain.puts.copy()
                            puts['Type'] = 'Put'
                            puts['Expiration'] = exp
                            all_opts.append(pd.concat([calls, puts]))
                        except: pass
                    if all_opts:
                        final_df = pd.concat(all_opts, ignore_index=True)
                        final_df['ExtractionDate'] = today_str
                        final_df.to_csv(today_file, index=False)
                        print(f"[{ticker}] Saved {len(final_df)} contracts.")
            except Exception as e:
                print(f"Failed to fetch options for {ticker}: {e}")

        # 2. Load History Strategy
        pattern = os.path.join(self.output_dir, f"{ticker}_options_*.csv")
        files = sorted(glob.glob(pattern))
        
        if not files:
            return pd.DataFrame(), pd.DataFrame(), "", ""
        
        # If only 1 file, return it as T1, and empty T0 (Triggers Snapshot Mode)
        if len(files) == 1:
            t1_file = files[0]
            d1 = t1_file.split('_')[-1].replace('.csv','')
            print(f"[{ticker}] Single snapshot found ({d1}). Running Snapshot Mode vs Realized Volatility.")
            return pd.read_csv(t1_file), pd.DataFrame(), d1, "HV_Proxy"
        
        # Comparison Mode
        t1_file = files[-1]
        t0_file = files[-2]
        d1 = t1_file.split('_')[-1].replace('.csv','')
        d0 = t0_file.split('_')[-1].replace('.csv','')
        print(f"[{ticker}] Comparing Dates: {d0} (t0) vs {d1} (t1)")
        
        return pd.read_csv(t1_file), pd.read_csv(t0_file), d1, d0


# ==========================================
# 2. FINANCIAL ANALYSIS (Logic Layer)
# ==========================================
class FinancialAnalysis:
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    def calculate_realized_volatility(self, spot_df: pd.DataFrame, window: int = 20) -> float:
        """
        Calculates Annualized Realized Volatility (HV) from spot history.
        Proxy for 'Baseline' when option history is missing.
        """
        if len(spot_df) < window + 1:
            return 0.0
        
        # Log returns: ln(Pt / Pt-1)
        spot_df['LogReturn'] = np.log(spot_df['Close'] / spot_df['Close'].shift(1))
        
        # Rolling Std Dev * sqrt(252)
        rolling_vol = spot_df['LogReturn'].rolling(window=window).std() * np.sqrt(252)
        
        return rolling_vol.iloc[-1]

    def prepare_data(self, opt_df: pd.DataFrame, spot_df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        if opt_df.empty: return pd.DataFrame()
        
        # Get Spot Price
        try:
            target_date = pd.to_datetime(date_str)
            idx = spot_df.index.get_indexer([target_date], method='nearest')[0]
            spot_price = spot_df.iloc[idx]['Close']
        except:
            if not spot_df.empty: spot_price = spot_df.iloc[-1]['Close']
            else: return pd.DataFrame()

        df = opt_df.copy()
        df['Spot'] = spot_price
        
        df['ExpirationDate'] = pd.to_datetime(df['Expiration'])
        df['ExtractionDate'] = pd.to_datetime(df['ExtractionDate'])
        df['DTE'] = (df['ExpirationDate'] - df['ExtractionDate']).dt.days
        df['T'] = df['DTE'] / 365.0
        
        df = df[df['DTE'] > 2].copy()
        df['LogMoneyness'] = np.log(df['strike'] / df['Spot'])
        
        if 'impliedVolatility' in df.columns:
            df = df[(df['impliedVolatility'] > 0.0005) & (df['impliedVolatility'] < 5.0)]
        
        # Filter for relevant moneyness
        df = df[(df['LogMoneyness'] > -0.4) & (df['LogMoneyness'] < 0.4)]
        
        return df

    def _calculate_robust_slope(self, group):
        X = group['LogMoneyness'].values.reshape(-1, 1)
        y = group['impliedVolatility'].values
        if len(y) < 5: return np.nan
        
        if SKLEARN_AVAILABLE:
            try:
                huber = HuberRegressor().fit(X, y)
                return huber.coef_[0]
            except: pass
        
        z = np.polyfit(group['LogMoneyness'], y, 1)
        return z[0]

    def compute_skew_structure(self, df: pd.DataFrame):
        if df.empty: return pd.DataFrame()
        
        results = []
        for exp, group in df.groupby('Expiration'):
            slope = self._calculate_robust_slope(group)
            avg_iv = group['impliedVolatility'].mean()
            dte = group['DTE'].iloc[0]
            
            results.append({
                'Expiration': exp,
                'DTE': dte,
                'Slope': slope,
                'AvgIV': avg_iv,
                'Contracts': len(group)
            })
            
        return pd.DataFrame(results).sort_values('DTE')

    def generate_heatmap_data(self, df_t0, df_t1):
        """
        Only runs if BOTH dataframes are populated.
        """
        if df_t0.empty or df_t1.empty: return None

        bins = np.arange(-0.25, 0.26, 0.05)
        labels = [f"{round(b,2)}" for b in bins[:-1]]
        
        def process_grid(df):
            df['Bucket'] = pd.cut(df['LogMoneyness'], bins=bins, labels=labels)
            grid_res = []
            for (exp, bucket), group in df.groupby(['Expiration', 'Bucket'], observed=True):
                slope = self._calculate_robust_slope(group)
                grid_res.append({
                    'Expiration': exp,
                    'Bucket': bucket,
                    'Slope': slope
                })
            return pd.DataFrame(grid_res)

        g0 = process_grid(df_t0)
        g1 = process_grid(df_t1)
        
        if g0.empty or g1.empty: return None
        
        merged = pd.merge(g1, g0, on=['Expiration', 'Bucket'], suffixes=('_t1', '_t0'))
        merged['SlopeDelta'] = merged['Slope_t1'] - merged['Slope_t0']
        
        return merged.pivot(index='Expiration', columns='Bucket', values='SlopeDelta')


# ==========================================
# 3. DASHBOARD RENDERER (Visualization)
# ==========================================
class DashboardRenderer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def _build_smile_slider(self, df_t0, df_t1, d0, d1, snapshot_mode=False):
        fig = go.Figure()
        expirations = sorted(df_t1['Expiration'].unique())
        if not expirations: return fig

        for i, exp in enumerate(expirations):
            mask_t1 = df_t1['Expiration'] == exp
            sub_t1 = df_t1[mask_t1].sort_values('LogMoneyness')
            visible = (i == 0)
            
            # T1 Trace (Current)
            fig.add_trace(go.Scatter(
                x=sub_t1['LogMoneyness'], y=sub_t1['impliedVolatility'],
                mode='markers+lines', name=f'{d1} (Current)',
                marker=dict(size=6, color='blue'),
                line=dict(width=2, color='blue'),
                visible=visible
            ))
            
            # T0 Trace (Only if not snapshot)
            if not snapshot_mode and not df_t0.empty:
                mask_t0 = df_t0['Expiration'] == exp
                sub_t0 = df_t0[mask_t0].sort_values('LogMoneyness')
                fig.add_trace(go.Scatter(
                    x=sub_t0['LogMoneyness'], y=sub_t0['impliedVolatility'],
                    mode='markers+lines', name=f'{d0} (Prev)',
                    marker=dict(size=6, color='orange', symbol='x'),
                    line=dict(width=1, dash='dash', color='orange'),
                    visible=visible
                ))

        # Slider Steps
        steps = []
        trace_count = 1 if snapshot_mode else 2
        
        for i, exp in enumerate(expirations):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}, {"title": f"Volatility Smile: Expiry {exp}"}],
                label=str(exp)
            )
            for j in range(trace_count):
                step["args"][0]["visible"][trace_count*i + j] = True
            steps.append(step)

        sliders = [dict(active=0, currentvalue={"prefix": "Expiration: "}, pad={"t": 50}, steps=steps)]
        fig.update_layout(sliders=sliders, title=f"Volatility Smile Evolution", 
                          xaxis_title="Log Moneyness", yaxis_title="Implied Volatility", height=500)
        return fig

    def render_html(self, ticker, d0, d1, skew_t0, skew_t1, heatmap_df, df_t0, df_t1, realized_vol):
        """
        Renders dashboard. Adapts layout if d0 is "HV_Proxy" (Snapshot Mode).
        """
        snapshot_mode = (d0 == "HV_Proxy")
        
        # 1. Term Structure: Skew + IV vs HV
        fig_term = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar: Skew Slope
        fig_term.add_trace(go.Bar(x=skew_t1['Expiration'], y=skew_t1['Slope'], name='Current Skew Slope', marker_color='blue'), secondary_y=False)
        
        # Line: Avg IV vs Realized Vol
        fig_term.add_trace(go.Scatter(x=skew_t1['Expiration'], y=skew_t1['AvgIV'], name='Avg IV', mode='lines+markers', line=dict(color='purple')), secondary_y=True)
        
        # Line: Realized Volatility Proxy
        fig_term.add_trace(go.Scatter(x=skew_t1['Expiration'], y=[realized_vol]*len(skew_t1), name=f'Realized Vol (20d): {realized_vol:.1%}', 
                                      mode='lines', line=dict(color='green', dash='dot')), secondary_y=True)

        fig_term.update_layout(title=f'Term Structure: Skew Slope & Premium over HV', xaxis_title='Expiration')
        fig_term.update_yaxes(title_text="Skew Slope", secondary_y=False)
        fig_term.update_yaxes(title_text="Volatility (IV vs HV)", secondary_y=True)

        # 2. Interactive Smile
        fig_smile = self._build_smile_slider(df_t0, df_t1, d0, d1, snapshot_mode)
        
        # 3. Comparison specific charts (Only if not snapshot)
        comparison_html = ""
        if not snapshot_mode and heatmap_df is not None:
            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_df.values, x=heatmap_df.columns, y=heatmap_df.index,
                colorscale='RdBu_r', zmid=0, colorbar=dict(title="Delta Slope")
            ))
            fig_heat.update_layout(title=f'Skew Change Heatmap ({d1} - {d0})')
            comparison_html += f'<div class="chart-card"><h2>Skew Change Matrix</h2>{fig_heat.to_html(full_html=False, include_plotlyjs=False)}</div>'

        # Generate HTML
        raw_html = f"""
        <html>
        <head>
            <title>{ticker} Volatility Analysis</title>
            <script>{py_offline.get_plotlyjs()}</script>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #eef2f5; padding: 20px; }}
                .container {{ max_width: 1200px; margin: 0 auto; }}
                .chart-card {{ background: white; padding: 25px; margin-bottom: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
                .alert {{ padding: 15px; background: #fff3cd; color: #856404; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{ticker} Volatility Analytics</h1>
                <div class="meta">Analysis Date: <b>{d1}</b></div>
                
                {'<div class="alert"><b>Snapshot Mode:</b> Only today\'s data available. Comparing IV against <b>Realized Volatility (HV)</b> proxy.</div>' if snapshot_mode else ''}

                <div class="chart-card">
                    <h2>1. Term Structure & Premium Analysis</h2>
                    <p>Bars = Skew Slope. Lines = Implied Volatility (Purple) vs Realized Volatility (Green). If Purple > Green, options are expensive relative to actual movement.</p>
                    {fig_term.to_html(full_html=False, include_plotlyjs=False)}
                </div>

                <div class="chart-card">
                    <h2>2. Volatility Smile (Interactive)</h2>
                    {fig_smile.to_html(full_html=False, include_plotlyjs=False)}
                </div>
                
                {comparison_html}
            </div>
             <script>window.dispatchEvent(new Event('resize'));</script>
        </body>
        </html>
        """
        
        filename = os.path.join(self.output_dir, f"{ticker}_report.html")
        with open(filename, "w", encoding='utf-8') as f:
            f.write(raw_html)
        print(f"[{ticker}] Report generated: {filename}")


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="IV Surface Analyzer")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='List of tickers')
    parser.add_argument('--output-dir', default='./market_data', help='Data storage directory')
    parser.add_argument('--lookback', type=int, default=1, help='Years of spot history')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate')
    
    args = parser.parse_args()
    
    ingestor = DataIngestion(args.output_dir)
    analyzer = FinancialAnalysis(risk_free_rate=args.risk_free_rate)
    renderer = DashboardRenderer(args.output_dir)
    
    for ticker in args.tickers:
        print(f"\n--- Processing {ticker} ---")
        
        spot_df = ingestor.get_spot_data(ticker, args.lookback)
        if spot_df.empty: continue
            
        # Get Options (Comparison OR Snapshot)
        df_raw_t1, df_raw_t0, d1, d0 = ingestor.get_option_data(ticker)
        
        if df_raw_t1.empty: continue

        # Calculate Realized Volatility Proxy (Annualized, 20-day window)
        hv_proxy = analyzer.calculate_realized_volatility(spot_df, window=20)
        
        # Prepare Data
        df_t1 = analyzer.prepare_data(df_raw_t1, spot_df, d1)
        df_t0 = analyzer.prepare_data(df_raw_t0, spot_df, d0)
        
        skew_t1 = analyzer.compute_skew_structure(df_t1)
        skew_t0 = analyzer.compute_skew_structure(df_t0)
        
        heatmap_df = analyzer.generate_heatmap_data(df_t0, df_t1)
        
        renderer.render_html(ticker, d0, d1, skew_t0, skew_t1, heatmap_df, df_t0, df_t1, hv_proxy)

if __name__ == "__main__":
    main()
