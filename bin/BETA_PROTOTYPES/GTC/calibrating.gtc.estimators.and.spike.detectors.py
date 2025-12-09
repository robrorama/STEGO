# SCRIPTNAME: ok.03.calibrating.gtc.estimators.and.spike.detectors.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import time
import argparse
import datetime
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.offline as py_offline

# suppress pandas fragmentation warnings for high-column counts
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ==============================================================================
# 1. DATA INGESTION (Disk-First, Shadow Backfill, Universal Fixer)
# ==============================================================================

class DataIngestion:
    """
    Handles all IO and external data access.
    Enforces Disk-First contract and Shadow Backfill logic.
    """
    def __init__(self, tickers: List[str], output_dir: str, lookback_years: float):
        self.tickers = tickers
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        self.shadow_backfill_log = []

    def ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"[DataIngestion] Created output directory: {self.output_dir}")

    def get_price_data(self) -> Dict[str, pd.DataFrame]:
        self.ensure_output_dir()
        data_map = {}
        
        # Calculate start date based on lookback
        start_date = (datetime.datetime.now() - datetime.timedelta(days=int(self.lookback_years * 365))).strftime('%Y-%m-%d')

        for ticker in self.tickers:
            file_path = os.path.join(self.output_dir, f"{ticker}.csv")
            df = pd.DataFrame()
            needs_download = False
            reason = ""

            # 1. Attempt Read from Disk
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    df = self._sanitize_df(df, ticker)
                    
                    # Shadow Backfill Check: Is data empty or significantly truncated?
                    # A rough heuristic: if we requested 1 year (252 days) and have < 20 days, force reload.
                    required_days = int(self.lookback_years * 252 * 0.5) 
                    if df.empty or len(df) < required_days:
                        needs_download = True
                        reason = f"Insufficient history on disk (rows={len(df)} vs req approx {required_days})"
                except Exception as e:
                    needs_download = True
                    reason = f"Corrupt CSV: {str(e)}"
            else:
                needs_download = True
                reason = "File not found"

            # 2. Download if needed (Shadow Backfill logic)
            if needs_download:
                print(f"[DataIngestion] Downloading {ticker}... ({reason})")
                if "Insufficient" in reason or "Corrupt" in reason:
                    self.shadow_backfill_log.append(f"{ticker}: {reason}")

                try:
                    # yf.download often returns MultiIndex columns. We handle this in _sanitize_df.
                    raw_df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
                    time.sleep(1) # Rate limit protection

                    if not raw_df.empty:
                        # Sanitize BEFORE saving to ensure clean CSV
                        clean_df = self._sanitize_df(raw_df, ticker)
                        clean_df.to_csv(file_path)
                        
                        # Re-read from disk to strictly satisfy "Analysis consumes disk-materialized data"
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        df = self._sanitize_df(df, ticker)
                    else:
                        print(f"[DataIngestion] Warning: Download returned empty data for {ticker}")
                        continue
                except Exception as e:
                    print(f"[DataIngestion] Error downloading {ticker}: {e}")
                    continue

            if not df.empty:
                data_map[ticker] = df

        return data_map

    def _sanitize_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Universal Fixer: Repairs yfinance quirks, flattens MultiIndexes, ensures numeric types.
        """
        if df.empty:
            return df

        # 1. Swap Levels & Flatten MultiIndex
        # yfinance often returns columns like (Price, Ticker). We need Price.
        if isinstance(df.columns, pd.MultiIndex):
            # Check if Ticker is level 0 or level 1. If 'Close' is in level 1, swap.
            # Usually structure is ('Adj Close', 'SPY'). If flipped, swap.
            # We assume standard yfinance 'group_by="column"' default often creates (Attribute, Ticker)
            
            # If the columns look like ('SPY', 'Close'), swap to ('Close', 'SPY')
            if ticker in df.columns.get_level_values(0) and 'Close' in df.columns.get_level_values(1):
                 df = df.swaplevel(0, 1, axis=1)

            # Flatten columns to Price_Ticker or just Price
            new_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Filter out the ticker name if it's the only ticker, 
                    # but easiest is to just grab the attribute part if it matches standard OHLC
                    attr, tick = col[0], col[1]
                    if tick == ticker:
                        new_cols.append(attr)
                    else:
                        new_cols.append(f"{attr}_{tick}")
                else:
                    new_cols.append(col)
            df.columns = new_cols

        # 2. Strict Date Index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        
        # Remove Timezone info to avoid Plotly/Pandas mismatches
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df.sort_index().drop_duplicates()

        # 3. Numeric Coercion & Schema Check
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Map 'Adj Close' to 'Adj Close' if exists
        
        for col in list(df.columns):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows that are entirely NaN in critical columns
        valid_cols = [c for c in required_cols if c in df.columns]
        df.dropna(subset=valid_cols, how='all', inplace=True)
        
        return df

    def get_backfill_log(self):
        return self.shadow_backfill_log


# ==============================================================================
# 2. FINANCIAL ANALYSIS (Math Logic Only)
# ==============================================================================

class FinancialAnalysis:
    """
    Pure transformation logic. No API calls.
    """
    def __init__(self, data_map: Dict[str, pd.DataFrame], risk_free_rate: float):
        self.data_map = data_map
        self.rf_annual = risk_free_rate
        self.rf_daily = (1 + risk_free_rate)**(1/252) - 1
        self.benchmark_ticker = list(data_map.keys())[0] if data_map else None

    def run_analysis(self):
        analyzed_data = {}
        summary_stats = []
        
        # Cross-sectional alignment for correlations
        combined_returns = pd.DataFrame()

        for ticker, df in self.data_map.items():
            # Create a copy to avoid mutating the original cache
            df = df.copy()
            
            # --- 1. Returns ---
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df['ret'] = df[price_col].pct_change().fillna(0)
            df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1)).fillna(0)
            df['cum_ret'] = (1 + df['ret']).cumprod() - 1
            
            # Keep for correlation matrix
            combined_returns[ticker] = df['ret']

            # --- 2. Rolling Volatility (Annualized) ---
            for w in [21, 63, 252]:
                df[f'realized_vol_{w}'] = df['log_ret'].rolling(w).std() * np.sqrt(252)

            # --- 3. Sharpe & Excess Returns ---
            df['excess_ret'] = df['ret'] - self.rf_daily
            for w in [63, 252]:
                # Rolling Annualized Sharpe
                roll_mu = df['excess_ret'].rolling(w).mean() * 252
                roll_std = df['ret'].rolling(w).std() * np.sqrt(252)
                # Avoid div by zero
                df[f'sharpe_{w}'] = roll_mu / (roll_std + 1e-9)

            # Full Sample metrics
            full_std = df['ret'].std() * np.sqrt(252)
            full_ret = df['ret'].mean() * 252
            full_sharpe = (full_ret - self.rf_annual) / (full_std + 1e-9)

            # --- 4. Drawdowns ---
            cum_equity = (1 + df['ret']).cumprod()
            running_max = cum_equity.cummax()
            df['drawdown'] = (cum_equity / running_max) - 1
            max_dd = df['drawdown'].min()

            # --- 5. Trend & Technicals ---
            close = df['Close']
            df['SMA_20'] = close.rolling(20).mean()
            df['SMA_50'] = close.rolling(50).mean()
            df['SMA_200'] = close.rolling(200).mean()
            
            # Bollinger Bands (20, 2)
            std_20 = close.rolling(20).std()
            df['BB_Upper'] = df['SMA_20'] + (2 * std_20)
            df['BB_Lower'] = df['SMA_20'] - (2 * std_20)
            
            # Z-Score vs SMA 50
            df['z_price_50'] = (close - df['SMA_50']) / (close.rolling(50).std() + 1e-9)
            
            # RSI 14
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))

            analyzed_data[ticker] = df
            
            # Summary Row
            latest_price = df['Close'].iloc[-1]
            ytd_ret = 0.0 # Placeholder logic for brevity
            one_year_ret = df['ret'].tail(252).sum() if len(df) >= 252 else df['ret'].sum()

            summary_stats.append({
                'Ticker': ticker,
                'Latest Close': latest_price,
                'Full Sharpe': full_sharpe,
                'Ann Vol (252)': df[f'realized_vol_252'].iloc[-1] if len(df) > 252 else 0,
                'Max DD': max_dd
            })

        # --- 6. Correlations ---
        corr_matrix = combined_returns.corr()
        
        rolling_corrs = {}
        if self.benchmark_ticker and len(combined_returns.columns) > 1:
            bench_ret = combined_returns[self.benchmark_ticker]
            for col in combined_returns.columns:
                if col != self.benchmark_ticker:
                    rolling_corrs[col] = combined_returns[col].rolling(63).corr(bench_ret)

        # --- 7. Portfolio (Equal Weight) ---
        # Align dates
        combined_returns.dropna(inplace=True)
        portfolio_ret = combined_returns.mean(axis=1)
        portfolio_cum = (1 + portfolio_ret).cumprod() - 1
        portfolio_dd = ((1 + portfolio_ret).cumprod() / (1 + portfolio_ret).cumprod().cummax()) - 1
        
        port_metrics = {
            'cum_ret': portfolio_cum,
            'drawdown': portfolio_dd,
            'full_sharpe': (portfolio_ret.mean()*252 - self.rf_annual) / (portfolio_ret.std()*np.sqrt(252) + 1e-9)
        }

        return {
            'tickers': analyzed_data,
            'summary': pd.DataFrame(summary_stats),
            'corr_matrix': corr_matrix,
            'rolling_corrs': rolling_corrs,
            'portfolio': port_metrics,
            'benchmark': self.benchmark_ticker
        }


# ==============================================================================
# 3. DASHBOARD RENDERER (Visualization Only)
# ==============================================================================

class DashboardRenderer:
    """
    Generates offline HTML dashboard. No calculation, just plotting.
    """
    def __init__(self, analysis_output: dict, metadata: dict, output_dir: str):
        self.data = analysis_output
        self.metadata = metadata
        self.output_dir = output_dir

    def generate_dashboard(self):
        # 1. Create Figures
        fig_overview = self._create_overview_tab()
        fig_risk = self._create_risk_tab()
        fig_corr = self._create_corr_tab()
        fig_quality = self._create_quality_tab()

        # 2. Convert to HTML divs (excluding plotly.js)
        # We define a helper to configure the plot for dark mode and responsiveness
        config = {'responsive': True, 'displayModeBar': True}
        
        div_overview = py_offline.plot(fig_overview, include_plotlyjs=False, output_type='div', config=config)
        div_risk = py_offline.plot(fig_risk, include_plotlyjs=False, output_type='div', config=config)
        div_corr = py_offline.plot(fig_corr, include_plotlyjs=False, output_type='div', config=config)
        div_quality = py_offline.plot(fig_quality, include_plotlyjs=False, output_type='div', config=config)

        # 3. Get Offline JS
        plotly_js = py_offline.get_plotlyjs()

        # 4. Assemble HTML with Tabs and Resize Fix
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Risk Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; }}
                .tab {{ overflow: hidden; border-bottom: 1px solid #333; background-color: #252526; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-size: 14px; font-weight: 600; }}
                .tab button:hover {{ background-color: #3e3e42; }}
                .tab button.active {{ background-color: #007acc; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border-top: none; height: 90vh; }}
                h3 {{ color: #007acc; }}
                .meta {{ font-size: 0.8em; color: #888; margin-bottom: 10px; }}
            </style>
        </head>
        <body>

        <div class="tab">
          <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Overview & Trend</button>
          <button class="tablinks" onclick="openTab(event, 'Risk')">Volatility & Drawdowns</button>
          <button class="tablinks" onclick="openTab(event, 'Correlations')">Correlations & Regimes</button>
          <button class="tablinks" onclick="openTab(event, 'Quality')">Data Quality</button>
        </div>

        <div id="Overview" class="tabcontent">
          {div_overview}
        </div>

        <div id="Risk" class="tabcontent">
          {div_risk}
        </div>

        <div id="Correlations" class="tabcontent">
          {div_corr}
        </div>
        
        <div id="Quality" class="tabcontent">
          {div_quality}
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
            
            // CRITICAL RESIZE FIX
            window.dispatchEvent(new Event('resize'));
        }}
        // Get the element with id="defaultOpen" and click on it
        document.getElementById("defaultOpen").click();
        </script>
        </body>
        </html>
        """

        out_path = os.path.join(self.output_dir, "market_dashboard.html")
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[DashboardRenderer] Dashboard saved to: {out_path}")

    # --- Plotly Builders ---

    def _create_overview_tab(self):
        # Create a subplot for each ticker (max 4 for readability, else loop)
        tickers = list(self.data['tickers'].keys())
        rows = len(tickers)
        fig = make_subplots(rows=rows, cols=2, shared_xaxes=False, 
                            vertical_spacing=0.05, 
                            column_width=[0.7, 0.3],
                            subplot_titles=[f"{t} Price/Bands" if i%2==0 else f"{t} Technicals" 
                                            for t in tickers for i in range(2)])

        for i, ticker in enumerate(tickers):
            df = self.data['tickers'][ticker]
            row = i + 1
            
            # Col 1: Price + Bands
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name=f"{ticker} OHLC"),
                          row=row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'),
                                     name='BB Upper', showlegend=False), row=row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'),
                                     name='BB Lower', showlegend=False, fill='tonexty'), row=row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1),
                                     name='SMA 50'), row=row, col=1)

            # Col 2: RSI + ZScore
            # Using secondary y-axis logic manually via stacking 
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='cyan', width=1.5),
                                     name=f"{ticker} RSI"), row=row, col=2)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=2)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=2)

        fig.update_layout(height=400*rows, template="plotly_dark", title_text="Price Action & Mean Reversion Structure")
        fig.update_xaxes(rangeslider_visible=False)
        return fig

    def _create_risk_tab(self):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Realized Volatility Term Structure (21d, 63d, 252d)", 
                                            "Drawdown Depth", "Portfolio vs Benchmark"))
        
        tickers = list(self.data['tickers'].keys())
        
        # 1. Volatility Cone
        for ticker in tickers:
            df = self.data['tickers'][ticker]
            # Plot only 63d vol for clarity if many tickers, or all for few
            fig.add_trace(go.Scatter(x=df.index, y=df['realized_vol_63'], 
                                     name=f"{ticker} Vol 63d"), row=1, col=1)
            
        # 2. Drawdowns
        for ticker in tickers:
            df = self.data['tickers'][ticker]
            fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], 
                                     fill='tozeroy', name=f"{ticker} DD"), row=2, col=1)

        # 3. Portfolio
        port = self.data['portfolio']
        fig.add_trace(go.Scatter(x=port['cum_ret'].index, y=port['cum_ret'], 
                                 line=dict(color='gold', width=3), name="Equal Weight Portfolio"), row=3, col=1)
        
        fig.update_layout(height=900, template="plotly_dark", hovermode="x unified")
        return fig

    def _create_corr_tab(self):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
                            subplot_titles=("Static Correlation Matrix (Returns)", 
                                            f"Rolling 63d Correlation vs {self.data['benchmark']}"))

        # 1. Heatmap
        corr = self.data['corr_matrix']
        fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, 
                                 colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr.values, 2),
                                 texttemplate="%{text}"), row=1, col=1)

        # 2. Rolling
        for ticker, series in self.data['rolling_corrs'].items():
            fig.add_trace(go.Scatter(x=series.index, y=series, name=f"{ticker} vs Bench"), row=2, col=1)
            
        fig.update_layout(height=900, template="plotly_dark")
        return fig

    def _create_quality_tab(self):
        # Text based summary
        shadow_log = self.metadata['backfill_log']
        log_text = "<br>".join(shadow_log) if shadow_log else "No shadow backfills triggered. Data clean."
        
        summ = self.data['summary']
        
        # Create Table
        fig = make_subplots(rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "table"}]],
                            subplot_titles=("Metrics Summary", "Data & Run Log"))
        
        fig.add_trace(go.Table(
            header=dict(values=list(summ.columns), fill_color='paleturquoise', align='left', font=dict(color='black')),
            cells=dict(values=[summ[k].tolist() for k in summ.columns], fill_color='lavender', align='left', font=dict(color='black'))
        ), row=1, col=1)
        
        # Meta table
        meta_data = [
            ['Risk Free Rate', 'Lookback', 'Run Time', 'Shadow Backfill Log'],
            [self.metadata['rf'], self.metadata['lookback'], str(datetime.datetime.now()), log_text]
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=["Parameter", "Value"], fill_color='darkred', align='left'),
            cells=dict(values=meta_data, fill_color='black', font=dict(color='white'), align='left')
        ), row=2, col=1)

        fig.update_layout(height=800, template="plotly_dark")
        return fig


# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hedge-Fund Grade Market Dashboard")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='List of tickers')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Output directory')
    parser.add_argument('--lookback', type=float, default=1, help='Lookback in years')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Annualized Risk Free Rate')
    
    args = parser.parse_args()
    
    print("--- Starting Market Dashboard Pipeline ---")
    
    # 1. Ingestion
    ingestor = DataIngestion(args.tickers, args.output_dir, args.lookback)
    price_data = ingestor.get_price_data()
    backfill_log = ingestor.get_backfill_log()
    
    if not price_data:
        print("Critical Error: No data available for any ticker.")
        sys.exit(1)

    # 2. Analysis
    analyzer = FinancialAnalysis(price_data, args.risk_free_rate)
    results = analyzer.run_analysis()

    # 3. Rendering
    meta = {
        'rf': args.risk_free_rate, 
        'lookback': args.lookback, 
        'backfill_log': backfill_log
    }
    
    renderer = DashboardRenderer(results, meta, args.output_dir)
    renderer.generate_dashboard()
    
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
