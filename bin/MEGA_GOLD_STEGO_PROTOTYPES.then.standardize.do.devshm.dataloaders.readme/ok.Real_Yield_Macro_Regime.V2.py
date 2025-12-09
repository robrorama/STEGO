import os
import time
import argparse
import logging
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs, plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. Data Ingestion (Disk-First Pipeline)
# -----------------------------------------------------------------------------

class DataIngestion:
    """
    Handles all data fetching, caching, and sanitization.
    Strict disk-first policy: checks CSV, sanitizes, backfills if needed.
    """
    def __init__(self, tickers: List[str], equity_proxy: str, output_dir: str, lookback_years: float):
        self.tickers = tickers
        self.equity_proxy = equity_proxy
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        
        # Core Macro Tickers Mapping
        self.macro_tickers = {
            'REAL_YIELD': 'DFII10',     # 10-Yr Real Yield (TIPS)
            'NOMINAL_YIELD': '^TNX',    # 10-Yr Treasury Yield
            'USD_INDEX': 'DX-Y.NYB',    # US Dollar Index
            'SKEW': '^SKEW',            # CBOE Skew Index
            'OIL': 'BZ=F',              # Brent Crude
            'GOLD': 'GC=F'              # Gold Futures
        }
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal fixer for yfinance data quirks.
        Handles MultiIndex columns, timezone stripping, and numeric coercion.
        """
        if df.empty:
            return df

        # 1. Index Normalization
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        # Drop duplicate indices, keeping the last
        df = df[~df.index.duplicated(keep='last')]

        # 2. MultiIndex Handling & Flattening
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' is in level 1 instead of 0
            if 'Close' in df.columns.get_level_values(1) and 'Close' not in df.columns.get_level_values(0):
                 df = df.swaplevel(0, 1, axis=1)
            
            # Flatten columns to "{Attribute}_{Ticker}"
            new_cols = []
            for col in df.columns:
                attr, ticker = col[0], col[1]
                new_cols.append(f"{attr}_{ticker}")
            df.columns = new_cols
        
        # 3. Numeric Coercion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop completely empty columns
        df.dropna(how='all', axis=1, inplace=True)

        return df

    def _fetch_or_load(self, ticker: str, name_tag: str) -> pd.DataFrame:
        """
        Disk-first logic:
        1. Check CSV.
        2. If exists -> load -> sanitize -> check date coverage.
        3. If missing/stale -> download -> sanitize -> save -> reload.
        """
        safe_ticker = ticker.replace('^', '').replace('=', '')
        file_path = os.path.join(self.output_dir, f"{name_tag}_{safe_ticker}.csv")
        
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=self.lookback_years)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        needs_download = True
        df = pd.DataFrame()

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0)
                df = self._sanitize_df(df)
                
                if not df.empty:
                    last_date = df.index[-1]
                    first_date = df.index[0]
                    
                    # Check staleness (5 days) and history length
                    days_lag = (end_date - last_date).days
                    history_ok = first_date <= (start_date + pd.Timedelta(days=30))
                    
                    if days_lag < 5 and history_ok:
                        needs_download = False
                        logger.info(f"Loaded {ticker} from cache.")
                    else:
                        logger.info(f"Cache for {ticker} stale or insufficient. Redownloading.")
            except Exception as e:
                logger.warning(f"Failed to read cache for {ticker}: {e}")

        if needs_download:
            logger.info(f"Downloading {ticker}...")
            try:
                time.sleep(1.0) # Rate limit hygiene
                # 'auto_adjust=True' is often default now, but explicit is better if supported
                # Using group_by='column' to ensure consistent structure
                raw_df = yf.download(ticker, start=start_date_str, progress=False, group_by='column')
                
                if raw_df.empty:
                    logger.warning(f"Download returned empty data for {ticker}")
                    return pd.DataFrame()

                clean_df = self._sanitize_df(raw_df)
                clean_df.to_csv(file_path)
                
                # Reload to ensure consistency
                df = pd.read_csv(file_path, index_col=0)
                df = self._sanitize_df(df)
                
            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")
                return pd.DataFrame()

        return df

    def get_equity_data(self) -> Dict[str, pd.DataFrame]:
        results = {}
        for t in self.tickers:
            df = self._fetch_or_load(t, "equity")
            results[t] = df
        return results

    def get_macro_series(self) -> Dict[str, pd.DataFrame]:
        results = {}
        
        # 1. Nominal Yield (Needed for context AND as potential proxy)
        df_nom = self._fetch_or_load(self.macro_tickers['NOMINAL_YIELD'], "macro")
        results['NOMINAL_YIELD'] = df_nom

        # 2. Real Yield
        df_real = self._fetch_or_load(self.macro_tickers['REAL_YIELD'], "macro")
        
        if df_real.empty:
             logger.warning("Primary Real Yield (DFII10) failed. Swapping in Nominal Yield (^TNX) as proxy.")
             # CRITICAL FIX: Explicitly assign the backup dataframe
             df_real = df_nom.copy()
             
        results['REAL_YIELD'] = df_real

        # 3. USD Index
        df_usd = self._fetch_or_load(self.macro_tickers['USD_INDEX'], "macro")
        if df_usd.empty:
            logger.info("Primary USD (DX-Y.NYB) failed, trying DXY fallback.")
            df_usd = self._fetch_or_load("DXY", "macro")
            if df_usd.empty:
                 df_usd = self._fetch_or_load("UUP", "macro")
        results['USD_INDEX'] = df_usd

        # 4. Skew
        results['SKEW'] = self._fetch_or_load(self.macro_tickers['SKEW'], "macro")

        # 5. Oil
        df_oil = self._fetch_or_load(self.macro_tickers['OIL'], "macro")
        if df_oil.empty:
            df_oil = self._fetch_or_load("CL=F", "macro")
        results['OIL'] = df_oil

        # 6. Gold (Optional)
        results['GOLD'] = self._fetch_or_load(self.macro_tickers['GOLD'], "macro")

        return results


# -----------------------------------------------------------------------------
# 2. Financial Analysis (Math & Regimes)
# -----------------------------------------------------------------------------

class FinancialAnalysis:
    """
    Core math logic: Correlations, Two-Stage Regression, Skew metrics.
    """
    def __init__(self, equity_data: Dict[str, pd.DataFrame], 
                 macro_data: Dict[str, pd.DataFrame], 
                 equity_proxy: str, 
                 risk_free_rate: float, 
                 window_days: int):
        self.equity_data = equity_data
        self.macro_data = macro_data
        self.equity_proxy = equity_proxy
        self.rfr = risk_free_rate
        self.window_days = window_days

    def _get_close_series(self, df: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Helper to extract Close or Adj Close.
        Smart-match: If specific ticker column is missing (due to proxy swap),
        grab the first available 'Close' column.
        """
        if df.empty:
            return pd.Series(dtype=float)
        
        # 1. Try exact match
        cols = [c for c in df.columns if f"Close_{ticker}" in c or f"Adj Close_{ticker}" in c]
        
        # 2. If no exact match (likely a proxy swap), try any Close column
        if not cols:
            cols = [c for c in df.columns if "Close" in c]
        
        # Prefer Adj Close if available, else Close
        adj = [c for c in cols if "Adj Close" in c]
        target = adj[0] if adj else (cols[0] if cols else None)
        
        if target:
            return df[target]
        return pd.Series(dtype=float)

    def build_windowed_panel(self) -> pd.DataFrame:
        """
        Aligns SPX, Real Yield, and USD on a common business day index
        for the tactical window.
        """
        # Extract Series
        spx_df = self.equity_data.get(self.equity_proxy)
        spx = self._get_close_series(spx_df, self.equity_proxy)
        
        ry_df = self.macro_data.get('REAL_YIELD')
        # Note: We ask for 'DFII10' but _get_close_series will handle if it's actually ^TNX
        ry = self._get_close_series(ry_df, 'DFII10') 
        
        usd_df = self.macro_data.get('USD_INDEX')
        # Pass a generic guess, helper will find actual column
        usd = self._get_close_series(usd_df, 'DX-Y.NYB')

        if spx.empty or ry.empty or usd.empty:
            logger.error("Missing core series (SPX, Yield, or USD) after ingestion. Cannot build panel.")
            return pd.DataFrame()

        # Merge
        df = pd.DataFrame({'SPX': spx, 'RealYield': ry, 'USD': usd})
        df = df.dropna(how='all')
        
        # Forward fill strictly for macro data gaps (rates often miss days)
        df = df.ffill().dropna()
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter to window
        end_date = df.index[-1]
        start_date = end_date - pd.Timedelta(days=self.window_days)
        window_df = df.loc[start_date:end_date].copy()
        
        return window_df

    def compute_equity_rates_usd_stats(self, window_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes returns and raw correlations.
        """
        if window_df.empty:
            return {}

        df = window_df.copy()
        df['SPX_Ret'] = np.log(df['SPX'] / df['SPX'].shift(1))
        df['USD_Ret'] = np.log(df['USD'] / df['USD'].shift(1))
        df['RealYield_Chg'] = df['RealYield'].diff() * 100 # bps
        
        df.dropna(inplace=True)
        
        if df.empty:
            return {}

        # Raw Correlations
        corr_spx_ry = df['SPX_Ret'].corr(df['RealYield_Chg'])
        corr_spx_usd = df['SPX_Ret'].corr(df['USD_Ret'])
        
        return {
            'data': df,
            'corr_spx_ry': corr_spx_ry,
            'corr_spx_usd': corr_spx_usd
        }

    def compute_usd_conditioned_correlation(self, stats_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Regresses SPX and RealYields against USD to find residual correlation.
        """
        if stats_data.empty or len(stats_data) < 10:
            return {}

        df = stats_data.copy()
        Y_spx = df['SPX_Ret']
        Y_ry = df['RealYield_Chg']
        X = df['USD_Ret']
        X = sm.add_constant(X)

        try:
            # Regression A: SPX ~ USD
            model_a = sm.OLS(Y_spx, X).fit()
            resid_spx = model_a.resid

            # Regression B: RealYield ~ USD
            model_b = sm.OLS(Y_ry, X).fit()
            resid_ry = model_b.resid

            # Conditional Correlation
            cond_corr = resid_spx.corr(resid_ry)

            # Rolling Correlations (Raw)
            rolling_raw = df['SPX_Ret'].rolling(20).corr(df['RealYield_Chg'])
            
            return {
                'cond_corr': cond_corr,
                'resid_spx': resid_spx,
                'resid_ry': resid_ry,
                'rolling_raw': rolling_raw,
            }
        except Exception as e:
            logger.error(f"Regression failed: {e}")
            return {}

    def compute_scatter_data(self, stats_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data for Scatter Plot (colored by USD direction).
        """
        if stats_data.empty:
            return pd.DataFrame()

        df = stats_data.copy()
        
        def label_usd(ret):
            if ret > 0.0005: return 'USD Up'
            if ret < -0.0005: return 'USD Down'
            return 'Flat'
            
        df['USD_Dir'] = df['USD_Ret'].apply(label_usd)
        return df

    def compute_skew_and_tail_risk_metrics(self) -> pd.DataFrame:
        """
        Analyzes SKEW index for tail risk regimes.
        """
        skew_df = self.macro_data.get('SKEW')
        skew_series = self._get_close_series(skew_df, '^SKEW')
        
        if skew_series.empty:
            return pd.DataFrame()
            
        # Align to window
        end_date = skew_series.index[-1]
        start_date = end_date - pd.Timedelta(days=self.window_days + 60) # buffer
        df = skew_series.loc[start_date:end_date].to_frame(name='Skew')
        
        # Metrics
        df['Z_Score'] = (df['Skew'] - df['Skew'].rolling(252).mean()) / df['Skew'].rolling(252).std()
        if df['Z_Score'].isnull().all():
             df['Z_Score'] = (df['Skew'] - df['Skew'].rolling(20).mean()) / df['Skew'].rolling(20).std()

        df['Elevated'] = df['Z_Score'] > 1.0
        df['Extreme'] = df['Z_Score'] > 2.0
        
        # Trim to exact window
        eff_start = end_date - pd.Timedelta(days=self.window_days)
        return df.loc[eff_start:]

    def compute_macro_context_series(self) -> Dict[str, pd.Series]:
        """
        Returns full lookback series for context charts.
        """
        def get_s(d, k, t):
            data = d.get(k)
            return self._get_close_series(data, t)
            
        return {
            'SPX': get_s(self.equity_data, self.equity_proxy, self.equity_proxy),
            'Oil': get_s(self.macro_data, 'OIL', 'BZ=F'),
            'USD': get_s(self.macro_data, 'USD_INDEX', 'DX-Y.NYB'),
            'RealYield': get_s(self.macro_data, 'REAL_YIELD', 'DFII10')
        }

# -----------------------------------------------------------------------------
# 3. Dashboard Renderer (Visualization)
# -----------------------------------------------------------------------------

class DashboardRenderer:
    """
    Generates Offline Plotly Charts, PNGs, and HTML Dashboard.
    """
    def __init__(self, output_dir, window_df, corr_stats, cond_stats, scatter_df, skew_df, macro_context):
        self.output_dir = output_dir
        self.window_df = window_df
        self.corr_stats = corr_stats
        self.cond_stats = cond_stats
        self.scatter_df = scatter_df
        self.skew_df = skew_df
        self.macro_context = macro_context
        
        self.png_dir = os.path.join(output_dir, 'PNGS')
        if not os.path.exists(self.png_dir):
            os.makedirs(self.png_dir)

    def _save_viz(self, fig, name):
        """Save PNG and return HTML div"""
        try:
            # Requires kaleido installed for static export
            fig.write_image(os.path.join(self.png_dir, f"{name}.png"), width=1200, height=700, scale=2)
        except Exception as e:
            logger.warning(f"Could not save PNG for {name} (kaleido missing or error): {e}")
            
        return plot(fig, output_type='div', include_plotlyjs=False)

    def build_regime_panel(self):
        df = self.window_df
        if df.empty: return "<div>No Data for Regime Panel</div>"
        
        fig = go.Figure()
        
        # Normalized SPX
        spx_base = df['SPX'].iloc[0]
        fig.add_trace(go.Scatter(x=df.index, y=df['SPX']/spx_base*100, name='SPX (Norm)', line=dict(color='blue', width=2)))
        
        # Real Yield (Right Axis)
        fig.add_trace(go.Scatter(x=df.index, y=df['RealYield'], name='Yield (Real/Nom Proxy)', line=dict(color='orange', width=2), yaxis='y2'))
        
        # USD (Right Axis - Normalized)
        usd_base = df['USD'].iloc[0]
        fig.add_trace(go.Scatter(x=df.index, y=df['USD']/usd_base*100, name='USD (Norm)', line=dict(color='green', dash='dot'), yaxis='y2'))

        fig.update_layout(
            title='Tactical Regime: SPX vs Yields vs USD',
            yaxis=dict(title='Normalized Level (100)'),
            yaxis2=dict(title='Yield (%) / USD', overlaying='y', side='right'),
            template='plotly_white',
            hovermode='x unified'
        )
        return self._save_viz(fig, 'regime_panel_spx_real_yield_usd')

    def build_correlation_panel(self):
        if not self.cond_stats: return "<div>No Data for Correlations</div>"
        
        raw = self.corr_stats.get('corr_spx_ry', 0)
        cond = self.cond_stats.get('cond_corr', 0)
        rolling_raw = self.cond_stats.get('rolling_raw')
        
        fig = go.Figure()
        
        # Rolling Raw
        if rolling_raw is not None:
            fig.add_trace(go.Scatter(x=rolling_raw.index, y=rolling_raw, name='Rolling 20D Raw Corr', line=dict(color='gray')))
            
        # Annotations
        fig.add_trace(go.Bar(
            x=['Raw Correlation', 'USD-Conditioned'],
            y=[raw, cond],
            marker_color=['indianred', 'royalblue'],
            name='Window Stat'
        ))
        
        fig.update_layout(
            title=f'Correlations: Raw ({raw:.2f}) vs USD-Conditioned ({cond:.2f})',
            yaxis=dict(range=[-1, 1], title='Correlation'),
            template='plotly_white'
        )
        return self._save_viz(fig, 'correlation_raw_vs_usd_conditioned')

    def build_scatter_panel(self):
        df = self.scatter_df
        if df.empty: return "<div>No Data for Scatter</div>"
        
        fig = go.Figure()
        colors = {'USD Up': 'green', 'USD Down': 'red', 'Flat': 'gray'}
        
        for direction, col in colors.items():
            sub = df[df['USD_Dir'] == direction]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub['RealYield_Chg'],
                    y=sub['SPX_Ret']*100,
                    mode='markers',
                    marker=dict(color=col, size=8, opacity=0.7),
                    name=direction,
                    text=sub.index.strftime('%Y-%m-%d'),
                    hovertemplate='<b>%{text}</b><br>Yield Chg: %{x} bps<br>SPX Ret: %{y:.2f}%'
                ))

        if len(df) > 1:
            try:
                m, b = np.polyfit(df['RealYield_Chg'], df['SPX_Ret']*100, 1)
                x_line = np.linspace(df['RealYield_Chg'].min(), df['RealYield_Chg'].max(), 100)
                y_line = m*x_line + b
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Trend', line=dict(color='black', dash='dash')))
            except:
                pass

        fig.update_layout(
            title='Scatter: SPX Returns vs Yield Changes (Conditioned on USD)',
            xaxis=dict(title='Yield Change (bps)'),
            yaxis=dict(title='SPX Daily Return (%)'),
            template='plotly_white'
        )
        return self._save_viz(fig, 'scatter_spx_vs_real_yield')

    def build_skew_panel(self):
        df = self.skew_df
        if df.empty: return "<div>No Data for Skew</div>"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Skew'], name='Skew Index', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Z_Score'], name='Z-Score', line=dict(color='gray', width=1), yaxis='y2'))
        
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=1, y1=1, yref='y2', line=dict(color="orange", dash="dot"))
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=2, y1=2, yref='y2', line=dict(color="red", dash="dot"))
        
        extremes = df[df['Extreme']]
        if not extremes.empty:
            fig.add_trace(go.Scatter(x=extremes.index, y=extremes['Skew'], mode='markers', marker=dict(color='red', symbol='x', size=10), name='Extreme Risk'))

        fig.update_layout(
            title='Tail Risk: SPX Skew Index & Z-Score',
            yaxis=dict(title='Skew Index'),
            yaxis2=dict(title='Z-Score', overlaying='y', side='right'),
            template='plotly_white'
        )
        return self._save_viz(fig, 'skew_tail_risk_panel')

    def build_macro_context(self):
        ctx = self.macro_context
        # Check if we have any valid data
        has_data = any(not s.empty for s in ctx.values())
        if not has_data: return "<div>No Data for Context</div>"
        
        fig = go.Figure()
        
        for k, s in ctx.items():
            if s.empty: continue
            valid = s.dropna()
            if valid.empty: continue
            base = valid.iloc[0]
            # Avoid division by zero
            if base == 0: base = 1.0
            fig.add_trace(go.Scatter(x=valid.index, y=valid/base*100, name=k))

        fig.update_layout(
            title='Macro Context (Lookback Horizon)',
            yaxis=dict(title='Normalized Performance (Base=100)'),
            template='plotly_white'
        )
        return self._save_viz(fig, 'macro_context_spx_usd_oil')

    def build_all_dashboards(self):
        logger.info("Generating plots...")
        
        div_regime = self.build_regime_panel()
        div_corr = self.build_correlation_panel()
        div_scatter = self.build_scatter_panel()
        div_skew = self.build_skew_panel()
        div_macro = self.build_macro_context()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Macro Regime Dashboard</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; background: #f4f4f4; }}
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #ccc; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; background: white; }}
            </style>
            <script type="text/javascript">
                {get_plotlyjs()}
            </script>
        </head>
        <body>
            <h2>Quantitative Regime Dashboard</h2>
            
            <div class="tab">
              <button class="tablinks" onclick="openTab(event, 'Regime')" id="defaultOpen">Regime Panel</button>
              <button class="tablinks" onclick="openTab(event, 'Correlation')">Corr & Residuals</button>
              <button class="tablinks" onclick="openTab(event, 'Scatter')">Yield Scatter</button>
              <button class="tablinks" onclick="openTab(event, 'Skew')">Skew / Tail Risk</button>
              <button class="tablinks" onclick="openTab(event, 'Macro')">Macro Context</button>
            </div>

            <div id="Regime" class="tabcontent">{div_regime}</div>
            <div id="Correlation" class="tabcontent">{div_corr}</div>
            <div id="Scatter" class="tabcontent">{div_scatter}</div>
            <div id="Skew" class="tabcontent">{div_skew}</div>
            <div id="Macro" class="tabcontent">{div_macro}</div>

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
                window.dispatchEvent(new Event('resize'));
            }}
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        out_path = os.path.join(self.output_dir, "dashboard.html")
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {out_path}")


# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Yields / USD / Equity Skew Regime Dashboard")
    
    # Required Arguments
    parser.add_argument('--tickers', nargs='+', default=['^GSPC'], help='List of equity tickers')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Output directory')
    parser.add_argument('--lookback', type=float, default=1, help='Lookback in years')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Annualized RFR')
    
    # Project Specific
    parser.add_argument('--window-days', type=int, default=90, help='Tactical window days')
    parser.add_argument('--equity-proxy', type=str, default='^GSPC', help='Primary equity proxy')

    args = parser.parse_args()

    # 1. Ingest
    ingestor = DataIngestion(
        tickers=args.tickers,
        equity_proxy=args.equity_proxy,
        output_dir=args.output_dir,
        lookback_years=args.lookback
    )
    
    logger.info("Starting Data Ingestion...")
    equity_data = ingestor.get_equity_data()
    macro_data = ingestor.get_macro_series()

    # 2. Analyze
    logger.info("Starting Financial Analysis...")
    analyzer = FinancialAnalysis(
        equity_data=equity_data,
        macro_data=macro_data,
        equity_proxy=args.equity_proxy,
        risk_free_rate=args.risk_free_rate,
        window_days=args.window_days
    )
    
    window_df = analyzer.build_windowed_panel()
    
    if not window_df.empty:
        stats_raw = analyzer.compute_equity_rates_usd_stats(window_df)
        if 'data' in stats_raw:
            stats_cond = analyzer.compute_usd_conditioned_correlation(stats_raw['data'])
            scatter_df = analyzer.compute_scatter_data(stats_raw['data'])
        else:
            stats_cond = {}
            scatter_df = pd.DataFrame()
    else:
        logger.error("Windowed DataFrame is empty. Skipping main analysis.")
        stats_raw = {}
        stats_cond = {}
        scatter_df = pd.DataFrame()
        
    skew_metrics = analyzer.compute_skew_and_tail_risk_metrics()
    macro_ctx = analyzer.compute_macro_context_series()

    # 3. Render
    logger.info("Rendering Dashboards...")
    renderer = DashboardRenderer(
        output_dir=args.output_dir,
        window_df=window_df,
        corr_stats=stats_raw,
        cond_stats=stats_cond,
        scatter_df=scatter_df,
        skew_df=skew_metrics,
        macro_context=macro_ctx
    )
    
    renderer.build_all_dashboards()
    logger.info("Done.")
