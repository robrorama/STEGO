# SCRIPTNAME: 02.novel.quant.visuals.v4.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import argparse
import time
import datetime
import warnings
import logging
import json
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf

import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

# =============================================================================
# LOGGING & CONFIG
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantDashboard")
warnings.filterwarnings("ignore")

# =============================================================================
# CLASS 1: DATA INGESTION (ROBUST & FLATTENED)
# =============================================================================

class DataIngestion:
    def __init__(self, output_dir: str, lookback_years: float):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _flatten_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggressively flattens MultiIndex columns from yfinance."""
        if isinstance(df.columns, pd.MultiIndex):
            # If we have (Price, Ticker), drop the Ticker level
            df.columns = df.columns.get_level_values(0)
        
        # Ensure we have standard names
        # Map common variations to standard
        col_map = {
            'Stock Splits': 'Split', 'Dividends': 'Div',
            'Adj Close': 'Close' # Prefer Adj Close if available, or handle logic
        }
        df.rename(columns=col_map, inplace=True)
        return df

    def get_underlying_data(self, ticker: str) -> pd.DataFrame:
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        
        # 1. Try Download
        logger.info(f"[{ticker}] Downloading Price Data...")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=int(self.lookback_years*365) + 30)).strftime('%Y-%m-%d')
        
        try:
            # auto_adjust=True gives OHLC adjusted for splits/divs (Trading view style)
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"[{ticker}] Download returned empty dataframe.")
            else:
                df = self._flatten_cols(df)
                df.to_csv(file_path) # Cache it
                return df
        except Exception as e:
            logger.error(f"[{ticker}] Download failed: {e}")

        # 2. Fallback to Disk
        if os.path.exists(file_path):
            logger.info(f"[{ticker}] Reading from disk fallback.")
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return df
            except:
                return pd.DataFrame()
        
        return pd.DataFrame()

    def get_options_data(self, ticker: str) -> pd.DataFrame:
        """Attempts to fetch options. Returns empty DF if failed (to trigger fallback)."""
        try:
            # Check for fresh file (less than 12 hours old)
            files = [f for f in os.listdir(self.output_dir) if f.startswith(f"options_{ticker}_")]
            files.sort(reverse=True)
            if files:
                # Assuming YYYYMMDD_HHMMSS format
                ts_str = files[0].split('_')[2].replace('.csv', '')
                file_ts = datetime.datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                if (datetime.datetime.now() - file_ts).total_seconds() < 43200: # 12 hours
                    logger.info(f"[{ticker}] Using cached options data.")
                    return pd.read_csv(os.path.join(self.output_dir, files[0]))

            # Fetch Live
            logger.info(f"[{ticker}] Fetching Options Chain...")
            yf_ticker = yf.Ticker(ticker)
            try:
                exps = yf_ticker.options
            except:
                logger.warning(f"[{ticker}] No options chain available (or API error).")
                return pd.DataFrame()

            if not exps: return pd.DataFrame()

            # Get nearest 2 expiries only (for speed & relevance)
            all_opts = []
            for e in exps[:2]:
                try:
                    opt = yf_ticker.option_chain(e)
                    c = opt.calls
                    c['type'] = 'call'
                    p = opt.puts
                    p['type'] = 'put'
                    combined = pd.concat([c, p])
                    combined['expirationDate'] = e
                    all_opts.append(combined)
                except:
                    continue
            
            if not all_opts: return pd.DataFrame()
            
            df = pd.concat(all_opts)
            # Normalize
            rename_map = {'contractSymbol':'contract', 'lastPrice':'last', 'openInterest':'open_interest', 'impliedVolatility':'iv'}
            df.rename(columns=rename_map, inplace=True)
            
            # Save
            ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            df.to_csv(os.path.join(self.output_dir, f"options_{ticker}_{ts}.csv"), index=False)
            return df
            
        except Exception as e:
            logger.warning(f"[{ticker}] Options fetch failed ({e}). Enabling Approx Mode.")
            return pd.DataFrame()

# =============================================================================
# CLASS 2: FINANCIAL ANALYSIS (WITH APPROXIMATION LOGIC)
# =============================================================================

class FinancialAnalysis:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def calc_structural_metrics(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Price/Volume metrics.
        CRITICAL: This runs purely on OHLCV, so it always works.
        """
        if df.empty: return df
        df = df.copy()
        
        # 1. Garman-Klass Volatility (The "Proxy" for Implied Vol)
        # 0.5 * ln(H/L)^2 - (2ln2 - 1) * ln(C/O)^2
        # Handle zeros/negatives safely
        df['High'] = df[['High', 'Close']].max(axis=1) # Safety
        df['Low'] = df[['Low', 'Close']].min(axis=1)
        
        log_hl = np.log(df['High'] / df['Low'])
        log_co = np.log(df['Close'] / df['Open'])
        
        gk_var = 0.5 * (log_hl**2) - (2*np.log(2)-1) * (log_co**2)
        df['vol_gk'] = np.sqrt(gk_var)
        df['vol_gk_z'] = (df['vol_gk'] - df['vol_gk'].rolling(20).mean()) / df['vol_gk'].rolling(20).std()

        # 2. VWAP (Volume Weighted Average Price) - Rolling 20D
        df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['pv'] = df['tp'] * df['Volume']
        df['vwap'] = df['pv'].rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # 3. Volume Profile approximation (High Volume Nodes)
        # We calculate this dynamically in the renderer, but let's prep a flag
        df['is_hvn'] = False # Placeholder
        
        # Save
        df.to_csv(os.path.join(self.output_dir, f"analytics_{ticker}_struct.csv"))
        return df

    def get_levels_approximation(self, ticker: str, df_und: pd.DataFrame, df_opt: pd.DataFrame) -> Dict:
        """
        Returns Key Levels.
        If Options Data Exists -> Returns GEX Walls.
        If Options Data Missing -> Returns Volume Profile Nodes (Approximation).
        """
        levels = {
            'source': 'NONE',
            'support': None,
            'resistance': None,
            'pivot': None
        }

        current_price = df_und['Close'].iloc[-1]

        # STRATEGY A: OPTIONS DATA (The "Sniper" Method)
        if not df_opt.empty:
            try:
                # Ensure numeric
                df_opt['open_interest'] = pd.to_numeric(df_opt['open_interest'], errors='coerce').fillna(0)
                df_opt['strike'] = pd.to_numeric(df_opt['strike'], errors='coerce')
                
                # Filter for nearest expiry
                exps = sorted(df_opt['expirationDate'].unique())
                if exps:
                    near = df_opt[df_opt['expirationDate'] == exps[0]]
                    
                    # Call Wall (Resistance) = Max Call OI
                    cw = near[near['type'] == 'call'].sort_values('open_interest', ascending=False).iloc[0]['strike']
                    
                    # Put Wall (Support) = Max Put OI
                    pw = near[near['type'] == 'put'].sort_values('open_interest', ascending=False).iloc[0]['strike']
                    
                    levels['source'] = 'OPTIONS_GEX'
                    levels['resistance'] = cw
                    levels['support'] = pw
                    levels['pivot'] = (cw + pw) / 2
                    return levels
            except Exception as e:
                logger.warning(f"Options level calc failed: {e}")
        
        # STRATEGY B: VOLUME DATA (The "Approximation" Method)
        # If we are here, Options failed. use Volume Nodes.
        logger.info(f"[{ticker}] Approximating levels using Volume Profile...")
        
        # Take last 60 days
        subset = df_und.tail(60)
        
        # Create Histogram of Volume by Price
        price_bins = np.linspace(subset['Low'].min(), subset['High'].max(), 30)
        hist, bin_edges = np.histogram(subset['Close'], bins=price_bins, weights=subset['Volume'])
        
        # Find peaks
        # Simple approach: Find the bin with max volume (POC - Point of Control)
        max_idx = np.argmax(hist)
        poc = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
        
        # Find secondary peaks for Support/Res
        # We split data into above and below POC
        
        levels['source'] = 'VOLUME_APPROX'
        levels['pivot'] = poc # The highest volume price is the magnet
        
        # Resistance: Highest volume node ABOVE current price
        upper_hist = hist[max_idx+1:]
        upper_edges = bin_edges[max_idx+1:]
        if len(upper_hist) > 0:
            res_idx = np.argmax(upper_hist)
            levels['resistance'] = (upper_edges[res_idx] + upper_edges[res_idx+1]) / 2
        else:
            levels['resistance'] = subset['High'].max()
            
        # Support: Highest volume node BELOW current price
        lower_hist = hist[:max_idx]
        lower_edges = bin_edges[:max_idx]
        if len(lower_hist) > 0:
            sup_idx = np.argmax(lower_hist)
            levels['support'] = (lower_edges[sup_idx] + lower_edges[sup_idx+1]) / 2
        else:
            levels['support'] = subset['Low'].min()
            
        return levels

# =============================================================================
# CLASS 3: DASHBOARD RENDERER (The "Always On" Display)
# =============================================================================

class DashboardRenderer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def generate_html(self, tickers: List[str]):
        # CSS & JS
        css = """
        <style>
            body { background-color: #0e1117; color: #fafafa; font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; }
            .grid-container { display: grid; grid-template-columns: 200px 1fr; gap: 20px; }
            .sidebar { background: #161b22; padding: 20px; border-radius: 8px; height: 90vh; }
            .content { background: #0e1117; }
            .tab-btn { display: block; width: 100%; padding: 12px; margin-bottom: 5px; background: #21262d; color: #c9d1d9; border: 1px solid #30363d; cursor: pointer; text-align: left; border-radius: 6px; }
            .tab-btn:hover { background: #30363d; }
            .tab-btn.active { background: #1f6feb; color: white; border-color: #1f6feb; }
            .plot-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 20px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
            .badge-opt { background: #238636; color: white; }
            .badge-vol { background: #9e6a03; color: white; }
        </style>
        """
        
        js = """
        <script>
            function openTab(evt, ticker) {
                var i, content, links;
                content = document.getElementsByClassName("tab-content");
                for (i = 0; i < content.length; i++) { content[i].style.display = "none"; }
                links = document.getElementsByClassName("tab-btn");
                for (i = 0; i < links.length; i++) { links[i].className = links[i].className.replace(" active", ""); }
                document.getElementById(ticker).style.display = "block";
                evt.currentTarget.className += " active";
                window.dispatchEvent(new Event('resize'));
            }
            // Auto click first
            document.addEventListener("DOMContentLoaded", function() {
                var btns = document.getElementsByClassName("tab-btn");
                if(btns.length > 0) btns[0].click();
            });
        </script>
        """
        
        sidebar = '<div class="sidebar"><h3>Tickers</h3>'
        content = '<div class="content">'
        
        for t in tickers:
            sidebar += f'<button class="tab-btn" onclick="openTab(event, \'{t}\')">{t}</button>'
            plot_html = self._render_ticker(t)
            content += f'<div id="{t}" class="tab-content" style="display:none;">{plot_html}</div>'
            
        sidebar += '</div>'
        content += '</div>'
        
        html = f"""
        <!DOCTYPE html>
        <html><head><title>Hedge Fund Desk</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        {css}</head>
        <body>
        <div class="grid-container">
            {sidebar}
            {content}
        </div>
        {js}
        </body></html>
        """
        
        with open(os.path.join(self.output_dir, "dashboard.html"), "w", encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Report Ready: {os.path.join(self.output_dir, 'dashboard.html')}")

    def _render_ticker(self, ticker: str) -> str:
        # Load Data
        struct_path = os.path.join(self.output_dir, f"analytics_{ticker}_struct.csv")
        
        if not os.path.exists(struct_path):
            return "<div class='plot-card'><h3>No Data Found</h3><p>Check internet connection or ticker spelling.</p></div>"
            
        df = pd.read_csv(struct_path, index_col=0, parse_dates=True)
        
        # Load Levels (Recalc on fly to handle fallback)
        fa = FinancialAnalysis(self.output_dir)
        # Try to find options file
        opts_files = [f for f in os.listdir(self.output_dir) if f.startswith(f"options_{ticker}_")]
        opts_df = pd.DataFrame()
        if opts_files:
            opts_files.sort(reverse=True)
            opts_df = pd.read_csv(os.path.join(self.output_dir, opts_files[0]))
            
        levels = fa.get_levels_approximation(ticker, df, opts_df)
        
        # ---------------------------
        # PLOT 1: THE "WAR ROOM" CHART
        # ---------------------------
        # Candles + VWAP + Levels (GEX or Volume)
        
        df_sub = df.tail(120)
        
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.01)
        
        # Candles
        fig.add_trace(go.Candlestick(x=df_sub.index, open=df_sub['Open'], high=df_sub['High'], 
                                     low=df_sub['Low'], close=df_sub['Close'], name='Price'), row=1, col=1)
        
        # VWAP
        if 'vwap' in df_sub.columns:
            fig.add_trace(go.Scatter(x=df_sub.index, y=df_sub['vwap'], line=dict(color='cyan', width=1), name='VWAP'), row=1, col=1)

        # Levels
        source_label = "GEX WALLS" if levels['source'] == 'OPTIONS_GEX' else "VOL NODES (APPROX)"
        color_line = "rgba(0, 255, 0, 0.7)" if levels['source'] == 'OPTIONS_GEX' else "rgba(255, 165, 0, 0.7)"
        
        if levels['resistance']:
            fig.add_hline(y=levels['resistance'], line_dash="dash", line_color="red", 
                          annotation_text=f"Res: {levels['resistance']:.2f}", row=1, col=1)
        if levels['support']:
            fig.add_hline(y=levels['support'], line_dash="dash", line_color="green", 
                          annotation_text=f"Sup: {levels['support']:.2f}", row=1, col=1)
        if levels['pivot']:
            fig.add_hline(y=levels['pivot'], line_dash="dot", line_color="white", 
                          annotation_text="Pivot", row=1, col=1)

        # Volume Profile (Right Sidebar)
        # Calc profile on the subset
        price_bins = np.linspace(df_sub['Low'].min(), df_sub['High'].max(), 30)
        hist, bin_edges = np.histogram(df_sub['Close'], bins=price_bins, weights=df_sub['Volume'])
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig.add_trace(go.Bar(x=hist, y=centers, orientation='h', marker_color='rgba(255,255,255,0.2)', showlegend=False), row=1, col=2)
        
        badge_class = "badge-opt" if levels['source'] == 'OPTIONS_GEX' else "badge-vol"
        title_html = f"{ticker} Structure <span class='badge {badge_class}'>{source_label}</span>"
        
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, xaxis_rangeslider_visible=False)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        plot1 = f"<div class='plot-card'><h3>{title_html}</h3>{pyo.plot(fig, output_type='div', include_plotlyjs=False)}</div>"

        # ---------------------------
        # PLOT 2: REGIME SCATTER (Vol vs Trend)
        # ---------------------------
        # Uses Garman-Klass Vol (Approx) if IV missing
        
        # Data Prep
        df['ret_5d'] = df['Close'].pct_change(5) * 100
        # Check if we have GK Vol, if not calc standard
        vol_col = 'vol_gk_z' if 'vol_gk_z' in df.columns else 'Close' # Fallback safety
        
        if vol_col != 'Close':
            curr = df.iloc[-1]
            trail = df.tail(20)
            
            fig2 = go.Figure()
            
            # Quadrants
            fig2.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor="green", opacity=0.1, layer="below", line_width=0)
            fig2.add_shape(type="rect", x0=-100, y0=0, x1=0, y1=100, fillcolor="red", opacity=0.1, layer="below", line_width=0)
            
            fig2.add_trace(go.Scatter(x=trail['ret_5d'], y=trail[vol_col], mode='lines+markers', 
                                      marker=dict(color=np.arange(20), colorscale='Bluered'), name='Trail'))
            fig2.add_trace(go.Scatter(x=[curr['ret_5d']], y=[curr[vol_col]], mode='markers', 
                                      marker=dict(size=15, color='yellow', line=dict(width=2, color='white')), name='Current'))
            
            fig2.update_layout(template="plotly_dark", height=400, title="Regime: 5D Return vs Volatility (Z-Score)",
                               xaxis_title="5D Return %", yaxis_title="Vol Z-Score",
                               xaxis_range=[-10, 10], yaxis_range=[-3, 4])
            
            plot2 = f"<div class='plot-card'><h3>Market Regime</h3>{pyo.plot(fig2, output_type='div', include_plotlyjs=False)}</div>"
        else:
            plot2 = ""

        return plot1 + plot2

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'NVDA', 'IWM'])
    parser.add_argument('--output-dir', default='./market_data')
    args = parser.parse_args()
    
    # 1. Init
    di = DataIngestion(args.output_dir, lookback_years=1.0)
    fa = FinancialAnalysis(args.output_dir)
    dr = DashboardRenderer(args.output_dir)
    
    # 2. Process
    for t in args.tickers:
        logger.info(f"--- {t} ---")
        
        # A. Get Price (Robust)
        df_u = di.get_underlying_data(t)
        
        # B. Get Options (May fail, returns empty)
        di.get_options_data(t)
        
        # C. Analyze (With Fallbacks)
        if not df_u.empty:
            fa.calc_structural_metrics(t, df_u)
            
    # 3. Render
    dr.generate_html(args.tickers)

if __name__ == "__main__":
    main()
