# SCRIPTNAME: ok.intraday_term_and_skew_dashboard.V5.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import sys
import os
import time
import random
import argparse
import datetime
import warnings
import webbrowser
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from scipy.stats import norm
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

CACHE_DIR = "market_data_cache"
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "META", "AMZN", "GOOGL", "TSLA", "MSFT"]
RISK_FREE_RATE_DEFAULT = 0.045  # 4.5%

# -----------------------------------------------------------------------------
# MATH UTILITIES (Black-Scholes Delta)
# -----------------------------------------------------------------------------
def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """
    Computes Black-Scholes delta.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return np.nan
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0

# -----------------------------------------------------------------------------
# 1. DATA INGESTION LAYER
# -----------------------------------------------------------------------------
class DataIngestion:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_path(self, ticker, data_type, suffix=""):
        safe_ticker = ticker.replace("^", "").replace("/", "-")
        return os.path.join(self.cache_dir, f"{data_type}_{safe_ticker}{suffix}.csv")

    def _sanitize_df(self, df):
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Fix MultiIndex Columns (common in yfinance v0.2+)
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels == 2:
                df.columns = df.columns.get_level_values(0)
        
        # 2. Date Index Handling
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ['Date', 'Datetime', 'date', 'datetime']:
                if col in df.columns:
                    df = df.set_index(col)
                    break
            df.index = pd.to_datetime(df.index, errors='coerce')

        # 3. Timezone Stripping
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 4. Numeric Coercion
        cols_to_convert = [c for c in df.columns if c not in ['contractSymbol', 'lastTradeDate', 'currency']]
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_price_history(self, ticker):
        path = self._get_cache_path(ticker, "prices", "_daily")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df = self._sanitize_df(df)
                if not df.empty and (datetime.datetime.now() - df.index[-1]).days < 1:
                    return df
            except: pass

        print(f"   [DataIngestion] Downloading history for {ticker}...")
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False, group_by='column', auto_adjust=True)
            df = self._sanitize_df(df)
            if not df.empty:
                df.to_csv(path)
            time.sleep(1.0) 
            return df
        except Exception as e:
            print(f"   [Error] Failed history download for {ticker}: {e}")
            return pd.DataFrame()

    def get_options_expiries(self, ticker):
        try:
            tk = yf.Ticker(ticker)
            return tk.options
        except Exception as e:
            print(f"   [Error] Could not fetch expiries for {ticker}: {e}")
            return []

    def get_option_chain(self, ticker, expiry):
        path = self._get_cache_path(ticker, "options", f"_{expiry}")
        if os.path.exists(path):
            last_mod = datetime.datetime.fromtimestamp(os.path.getmtime(path))
            if (datetime.datetime.now() - last_mod).days < 1:
                try:
                    df = pd.read_csv(path)
                    return self._sanitize_df(df)
                except: pass

        print(f"   [DataIngestion] Downloading chain {ticker} @ {expiry}...")
        try:
            tk = yf.Ticker(ticker)
            opts = tk.option_chain(expiry)
            chain = pd.concat([opts.calls.assign(type='call'), opts.puts.assign(type='put')], ignore_index=True)
            chain = self._sanitize_df(chain)
            if not chain.empty:
                chain.to_csv(path, index=False)
            return chain
        except Exception as e:
            print(f"   [Error] Failed chain download {ticker}/{expiry}: {e}")
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. FINANCIAL ANALYSIS LAYER
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    def __init__(self, risk_free_rate=0.04):
        self.r = risk_free_rate

    def calculate_realized_vol(self, price_df, windows=[10, 20, 60]):
        if price_df.empty: return pd.DataFrame()
        df = price_df.copy()
        
        close_col = 'Close' if 'Close' in df.columns else 'Adj Close'
        if close_col not in df.columns:
            if len(df.columns) > 0: close_col = df.columns[0]
            else: return pd.DataFrame()

        df['log_ret'] = np.log(df[close_col] / df[close_col].shift(1))
        
        rv_data = {}
        for w in windows:
            rv_data[f'RV_{w}d'] = df['log_ret'].rolling(window=w).std() * np.sqrt(252) * 100
            
        return pd.concat([df[[close_col]], pd.DataFrame(rv_data, index=df.index)], axis=1)

    def analyze_ticker(self, ticker, price_df, expiries, ingest_engine):
        if price_df.empty: return None

        close_col = 'Close' if 'Close' in price_df.columns else price_df.columns[0]
        try: current_spot = float(price_df[close_col].iloc[-1])
        except: return None

        today = pd.Timestamp.now()
        term_structure = []
        
        print(f"   [Analysis] Processing {len(expiries)} expiries for {ticker}...")
        
        for exp in expiries:
            # Staggered download
            time.sleep(random.uniform(0.5, 1.5))
            
            chain = ingest_engine.get_option_chain(ticker, exp)
            if chain.empty: continue

            exp_date = pd.to_datetime(exp)
            days_to_exp = (exp_date - today).days
            if days_to_exp < 1: days_to_exp = 1
            T = days_to_exp / 365.0

            valid_chain = chain[chain['impliedVolatility'] > 0.0].copy()
            if valid_chain.empty: continue

            # Vectorized Delta
            def get_delta_vec(row):
                return calculate_delta(current_spot, row['strike'], T, self.r, row['impliedVolatility'], row['type'])

            valid_chain['delta'] = valid_chain.apply(get_delta_vec, axis=1)
            valid_chain = valid_chain.dropna(subset=['delta'])

            # ATM IV
            valid_chain['dist_to_spot'] = abs(valid_chain['strike'] - current_spot)
            atm_row = valid_chain.loc[valid_chain['dist_to_spot'].idxmin()]
            atm_iv = atm_row['impliedVolatility'] * 100

            # 25 Delta Call
            calls = valid_chain[valid_chain['type'] == 'call']
            c25_iv = np.nan
            if not calls.empty:
                c25_iv = calls.loc[abs(calls['delta'] - 0.25).idxmin()]['impliedVolatility'] * 100

            # 25 Delta Put
            puts = valid_chain[valid_chain['type'] == 'put']
            p25_iv = np.nan
            if not puts.empty:
                p25_iv = puts.loc[abs(puts['delta'] - (-0.25)).idxmin()]['impliedVolatility'] * 100

            rr25 = c25_iv - p25_iv if (not np.isnan(c25_iv) and not np.isnan(p25_iv)) else np.nan

            term_structure.append({
                'expiry': exp_date,
                'days': days_to_exp,
                'atm_iv': atm_iv,
                'c25_iv': c25_iv,
                'p25_iv': p25_iv,
                'rr25': rr25
            })

        if not term_structure: return None
        return pd.DataFrame(term_structure).sort_values('days')

# -----------------------------------------------------------------------------
# 3. DASHBOARD RENDERER (CREATIVE OVERHAUL)
# -----------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self, output_file="options_dashboard.html"):
        self.output_file = output_file
        self.figures = {} 

    def generate_html(self, analysis_results, price_histories):
        # 1. Build Plots
        self._build_overview_tab_creative(analysis_results, price_histories)
        for ticker, ts_df in analysis_results.items():
            price_df = price_histories.get(ticker)
            self._build_ticker_tab(ticker, ts_df, price_df)

        # 2. Get Plotly JS (Offline)
        plotly_js = py_offline.get_plotlyjs()

        # 3. HTML Template
        tabs_html = ""
        content_html = ""
        
        script_js = """
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
            document.getElementsByClassName("tablinks")[0].click();
        });
        </script>
        """

        style_css = """
        <style>
        body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; margin: 0;}
        .tab { overflow: hidden; border-bottom: 2px solid #34495e; background-color: #2c3e50; }
        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ecf0f1; font-weight: 600; font-size: 14px;}
        .tab button:hover { background-color: #34495e; }
        .tab button.active { background-color: #1abc9c; color: white; border-bottom: 4px solid #16a085; }
        .tabcontent { display: none; padding: 20px; animation: fadeEffect 0.5s; }
        @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
        .chart-container { background: white; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-radius: 8px; border: 1px solid #e1e4e8; }
        .dashboard-header { padding: 20px; background: #34495e; color: white; display: flex; justify-content: space-between; align-items: center; }
        .dashboard-header h1 { margin: 0; font-size: 24px; font-weight: 300; }
        .grid-row { display: flex; gap: 20px; margin-bottom: 20px; }
        .grid-col { flex: 1; min-width: 0; }
        </style>
        """

        tab_names = list(self.figures.keys())
        tabs_html += '<div class="tab">'
        for name in tab_names:
            safe_name = name.replace(" ", "_").replace(":", "")
            tabs_html += f'<button class="tablinks" onclick="openTab(event, \'{safe_name}\')">{name}</button>'
        tabs_html += '</div>'

        for name in tab_names:
            safe_name = name.replace(" ", "_").replace(":", "")
            figs = self.figures[name]
            
            content_html += f'<div id="{safe_name}" class="tabcontent">'
            for fig in figs:
                div_str = py_offline.plot(fig, include_plotlyjs=False, output_type='div')
                content_html += f'<div class="chart-container">{div_str}</div>'
            content_html += '</div>'

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Options Analytics</title>
            <script>{plotly_js}</script>
            {style_css}
            {script_js}
        </head>
        <body>
            <div class="dashboard-header">
                <h1>Hedge Fund Volatility & Skew Dashboard</h1>
                <span>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
            </div>
            {tabs_html}
            {content_html}
        </body>
        </html>
        """

        with open(self.output_file, "w", encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"\n[Dashboard] Successfully generated: {os.path.abspath(self.output_file)}")

    def _build_overview_tab_creative(self, results, price_histories):
        """
        Builds a multi-panel, data-rich Overview tab.
        1. Skew Heatmap (Global)
        2. Implied vs Realized Volatility Scatter (Rich/Cheap Analysis)
        3. Global Volatility Term Structure Overlay
        """
        overview_figs = []
        
        # --- DATA PREP ---
        skew_data = []
        rich_cheap_data = []
        
        for ticker, df in results.items():
            if df is None or df.empty: continue
            
            # 1. Skew Data (Heatmap)
            valid_skew = df.dropna(subset=['rr25', 'days', 'expiry'])
            for _, row in valid_skew.iterrows():
                skew_data.append({
                    'Ticker': ticker,
                    'Days': int(row['days']),
                    'Expiry': row['expiry'].strftime('%Y-%m-%d'),
                    'RR25': row['rr25']
                })
            
            # 2. Rich/Cheap Data (Front Month ATM vs 20d RV)
            try:
                # Find row closest to 30 days
                idx_30d = (df['days'] - 30).abs().idxmin()
                front_row = df.loc[idx_30d]
                atm_iv = front_row['atm_iv']
                
                # Get RV
                ph = price_histories.get(ticker)
                rv_20d = np.nan
                if ph is not None and 'RV_20d' in ph.columns:
                    rv_20d = ph['RV_20d'].iloc[-1]
                
                if not np.isnan(atm_iv) and not np.isnan(rv_20d):
                    rich_cheap_data.append({
                        'Ticker': ticker,
                        'ATM_IV': atm_iv,
                        'RV_20d': rv_20d,
                        'Ratio': atm_iv / rv_20d,
                        'Diff': atm_iv - rv_20d
                    })
            except Exception as e:
                continue

        # --- FIG 1: GLOBAL SKEW HEATMAP ---
        if skew_data:
            sdf = pd.DataFrame(skew_data)
            sdf = sdf[sdf['Days'] < 400] # Limit to 1 year for readability
            
            heatmap_pivot = sdf.pivot_table(index='Ticker', columns='Days', values='RR25', aggfunc='mean')
            
            fig1 = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='RdBu_r', 
                zmid=0,
                colorbar=dict(title="RR25 (Call-Put)"),
                hovertemplate="Ticker: %{y}<br>Days: %{x}<br>RR25: %{z:.2f}<extra></extra>"
            ))
            fig1.update_layout(
                title="<b>Global Skew Map:</b> 25-Delta Risk Reversal (Blue = Call Skew, Red = Put Skew)",
                xaxis_title="Days to Expiry",
                yaxis_title=None,
                height=500,
                xaxis=dict(tickmode='linear', tick0=0, dtick=30, range=[0, 365]),
                template="plotly_white"
            )
            overview_figs.append(fig1)

        # --- FIG 2: RICH/CHEAP SCATTER (IV vs RV) ---
        if rich_cheap_data:
            rc_df = pd.DataFrame(rich_cheap_data)
            
            max_val = max(rc_df['ATM_IV'].max(), rc_df['RV_20d'].max()) * 1.1
            
            fig2 = go.Figure()
            # The Scatter
            fig2.add_trace(go.Scatter(
                x=rc_df['RV_20d'],
                y=rc_df['ATM_IV'],
                mode='markers+text',
                text=rc_df['Ticker'],
                textposition='top center',
                marker=dict(
                    size=15,
                    color=rc_df['Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="IV/RV Ratio"),
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertemplate="<b>%{text}</b><br>RV (20d): %{x:.1f}%<br>IV (30d): %{y:.1f}%<br>Ratio: %{marker.color:.2f}<extra></extra>"
            ))
            
            # The 1:1 Line
            fig2.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="Red", width=2, dash="dash"),
                layer="below")
            
            fig2.add_annotation(text="Expensive (IV > RV)", x=max_val*0.1, y=max_val*0.9, showarrow=False, font=dict(color="gray"))
            fig2.add_annotation(text="Cheap (IV < RV)", x=max_val*0.9, y=max_val*0.1, showarrow=False, font=dict(color="gray"))

            fig2.update_layout(
                title="<b>Rich/Cheap Analysis:</b> Implied Vol (30d) vs Realized Vol (20d)",
                xaxis_title="20-Day Realized Volatility (%)",
                yaxis_title="30-Day ATM Implied Volatility (%)",
                height=600,
                template="plotly_white"
            )
            overview_figs.append(fig2)

        # --- FIG 3: GLOBAL TERM STRUCTURE OVERLAY ---
        fig3 = go.Figure()
        for ticker, df in results.items():
            if df is None or df.empty: continue
            # Sort by days to ensure clean lines
            df = df.sort_values('days')
            fig3.add_trace(go.Scatter(
                x=df['days'], 
                y=df['atm_iv'], 
                mode='lines+markers', 
                name=ticker,
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig3.update_layout(
            title="<b>Global Volatility Term Structure Overlay</b> (ATM IV)",
            xaxis_title="Days to Expiry",
            yaxis_title="ATM Implied Volatility (%)",
            height=500,
            template="plotly_white",
            xaxis=dict(range=[0, 365]), # Zoom to 1 year
            hovermode="x unified"
        )
        overview_figs.append(fig3)

        self.figures["Overview"] = overview_figs

    def _build_ticker_tab(self, ticker, ts_df, price_df):
        if ts_df is None or price_df is None: return
        figs = []
        
        # 1. Price & RV
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        close_col = price_df.columns[0]
        fig1.add_trace(go.Scatter(x=price_df.index, y=price_df[close_col], name="Price", line=dict(color='#2c3e50')), secondary_y=False)
        rv_cols = [c for c in price_df.columns if 'RV_' in c]
        colors = ['#e67e22', '#e74c3c', '#9b59b6']
        for i, col in enumerate(rv_cols):
            fig1.add_trace(go.Scatter(x=price_df.index, y=price_df[col], name=col, line=dict(width=1, color=colors[i%3])), secondary_y=True)
        fig1.update_layout(title=f"<b>{ticker}:</b> Spot Price vs Realized Volatility", height=500, template="plotly_white")
        figs.append(fig1)

        # 2. Term Structure
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts_df['expiry'], y=ts_df['atm_iv'], name="ATM IV", mode='lines+markers', line=dict(width=3, color='#34495e')))
        fig2.add_trace(go.Scatter(x=ts_df['expiry'], y=ts_df['c25_iv'], name="25d Call IV", mode='lines', line=dict(dash='dot', color='#2ecc71')))
        fig2.add_trace(go.Scatter(x=ts_df['expiry'], y=ts_df['p25_iv'], name="25d Put IV", mode='lines', line=dict(dash='dot', color='#e74c3c')))
        fig2.update_layout(title=f"<b>{ticker}:</b> Implied Volatility Term Structure", height=500, template="plotly_white", hovermode="x unified")
        figs.append(fig2)

        # 3. Skew Structure
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=ts_df['expiry'], y=ts_df['rr25'], name="RR25",
            marker_color=ts_df['rr25'], marker_colorscale="RdBu_r", marker_cmid=0
        ))
        fig3.update_layout(title=f"<b>{ticker}:</b> Skew Term Structure (RR25)", height=400, template="plotly_white")
        figs.append(fig3)

        self.figures[f"Ticker: {ticker}"] = figs

# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION FLOW
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Options Skew Dashboard")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="List of tickers")
    parser.add_argument("--riskfree", type=float, default=RISK_FREE_RATE_DEFAULT, help="Risk free rate (decimal)")
    parser.add_argument("--output", type=str, default="options_dashboard_v5.html", help="Output HTML filename")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Starting Options Analysis Engine (v5 - Creative Suite)")
    print(f"Tickers: {args.tickers}")
    print("="*60)

    ingestion = DataIngestion()
    engine = FinancialAnalysis(risk_free_rate=args.riskfree)
    renderer = DashboardRenderer(output_file=args.output)
    
    analysis_results = {}
    price_histories = {}

    for ticker in args.tickers:
        print(f"\nProcessing {ticker}...")
        
        # 1. History
        price_df = ingestion.get_price_history(ticker)
        if price_df.empty:
            print(f"   [Skip] No price data for {ticker}")
            continue
        
        rv_df = engine.calculate_realized_vol(price_df)
        price_histories[ticker] = rv_df
        
        # 2. Options
        expiries = ingestion.get_options_expiries(ticker)
        if not expiries:
            print(f"   [Skip] No options found for {ticker}")
            continue
            
        target_expiries = expiries[:12] 
        ts_df = engine.analyze_ticker(ticker, price_df, target_expiries, ingestion)
        if ts_df is not None:
            analysis_results[ticker] = ts_df
        else:
            print(f"   [Warning] No valid term structure derived for {ticker}")

    print("\nGenerating Dashboard...")
    renderer.generate_html(analysis_results, price_histories)
    
    if not args.no_open:
        webbrowser.open('file://' + os.path.abspath(args.output))

if __name__ == "__main__":
    main()
