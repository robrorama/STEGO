# SCRIPTNAME: ok.01.new.screener.v4.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import os
import time
import json
import math
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from datetime import datetime

# ==========================================
# 1. DATA INGESTION (Disk-First Pipeline)
# ==========================================
class DataIngestion:
    """
    Solely responsible for downloading data via yfinance, saving to disk,
    and reading from disk. Enforces idempotency and rate limiting.
    """
    def __init__(self, tickers, output_dir):
        self.tickers = tickers
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_data(self):
        """
        Orchestrates the disk-check -> download -> save -> read pipeline.
        Returns a list of dictionaries (one per ticker).
        """
        data_payload = []
        print(f"[DataIngestion] Processing {len(self.tickers)} tickers...")

        for ticker in self.tickers:
            file_path = os.path.join(self.output_dir, f"{ticker}_fundamentals.json")
            
            # 1. Check Existence
            if os.path.exists(file_path):
                print(f"  -> Found cached data for {ticker}. Reading from disk.")
                with open(file_path, 'r') as f:
                    ticker_data = json.load(f)
            else:
                # 2. If Missing: Download
                print(f"  -> Fetching {ticker} from API...")
                try:
                    t = yf.Ticker(ticker)
                    raw_info = t.info
                    
                    # 3. CRITICAL: Rate Limit
                    time.sleep(1) 
                    
                    # 4. Save to Disk
                    with open(file_path, 'w') as f:
                        json.dump(raw_info, f)
                    
                    # 5. Read back from disk
                    with open(file_path, 'r') as f:
                        ticker_data = json.load(f)
                        
                except Exception as e:
                    print(f"  [!] Error fetching {ticker}: {e}")
                    continue

            if 'symbol' not in ticker_data:
                ticker_data['symbol'] = ticker
                
            data_payload.append(ticker_data)
            
        return data_payload


# ==========================================
# 2. FINANCIAL ANALYSIS (Sanitization & Logic)
# ==========================================
class FinancialAnalysis:
    """
    Solely responsible for cleaning, type-casting, and structuring data.
    """
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.metrics = [
            'trailingPE', 'forwardPE', 'pegRatio', 'priceToBook', 
            'beta', 'revenueGrowth', 'grossMargins', 'marketCap',
            'enterpriseToEbitda', 'dividendYield'
        ]

    def _sanitize_df(self, df):
        # 1. Numeric Coercion
        for col in self.metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = float('nan')

        # 2. Type Detection (Equity vs ETF)
        if 'quoteType' not in df.columns:
            df['quoteType'] = 'EQUITY'
            
        def categorize_type(qt):
            qt = str(qt).upper()
            if 'ETF' in qt:
                return 'ETF'
            return 'Equity'

        df['AssetClass'] = df['quoteType'].apply(categorize_type)

        # 3. Sector Normalization
        # ETFs often don't have a 'sector', so we label them explicitly
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'
        
        # Fill NaN sectors with 'ETF' if it's an ETF, else 'Other'
        df['sector'] = df.apply(
            lambda x: 'Exchange Traded Fund' if x['AssetClass'] == 'ETF' else (x['sector'] if pd.notnull(x['sector']) else 'Other'), 
            axis=1
        )

        # 4. Market Cap Size Normalization (Logarithmic)
        # We create a specific column for visual sizing
        # Formula: Log-Scale normalization mapped to min_size(15) and max_size(65)
        mcap = df['marketCap'].fillna(0)
        # Avoid log(0)
        mcap = mcap.replace(0, mcap.mean()) 
        
        log_mcap = np.log10(mcap)
        min_log = log_mcap.min()
        max_log = log_mcap.max()
        
        # Avoid division by zero if only 1 ticker
        if max_log == min_log:
            df['viz_size'] = 25
        else:
            # Scale 0.0 to 1.0
            scaler = (log_mcap - min_log) / (max_log - min_log)
            # Map to pixel range 10px to 60px
            df['viz_size'] = 10 + (scaler * 50)

        keep_cols = ['symbol', 'shortName', 'AssetClass', 'sector', 'viz_size'] + self.metrics
        final_cols = [c for c in keep_cols if c in df.columns]
        
        return df[final_cols]

    def process(self):
        df = pd.DataFrame(self.raw_data)
        clean_df = self._sanitize_df(df)
        print(f"[FinancialAnalysis] Processed {len(clean_df)} records.")
        return clean_df


# ==========================================
# 3. DASHBOARD RENDERER (Visualization)
# ==========================================
class DashboardRenderer:
    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        # Exclude metadata from axis selectors
        self.metrics = [c for c in df.columns if c not in ['symbol', 'shortName', 'AssetClass', 'sector', 'viz_size']]

    def generate_dashboard(self):
        print("[DashboardRenderer] Generating interactive HTML...")

        # 1. Prepare Data Grouping
        # We need a trace for every unique sector to allow for automatic coloring in the Legend
        sectors = sorted(self.df['sector'].unique())
        
        # Color palette generator (using Plotly's default cycle logic indirectly)
        # We will iterate sectors and create a trace for each.
        
        traces = []
        
        # Store indices for the "Filter" dropdown logic
        # We need to know which traces are Equities vs ETFs
        equity_indices = []
        etf_indices = []
        
        default_x = 'marketCap'
        default_y = 'trailingPE'

        for i, sec in enumerate(sectors):
            sec_df = self.df[self.df['sector'] == sec]
            
            # Determine if this sector is technically the ETF group
            is_etf_group = (sec == 'Exchange Traded Fund')
            if is_etf_group:
                etf_indices.append(i)
            else:
                equity_indices.append(i)

            trace = go.Scatter(
                x=sec_df[default_x],
                y=sec_df[default_y],
                mode='markers',
                name=sec, # This puts the Sector in the legend
                marker=dict(
                    size=sec_df['viz_size'], # The normalized log-scale size
                    opacity=0.7,
                    line=dict(width=1, color='white'),
                    # Auto-color by sector is handled effectively by Plotly automatically assigning defaults per trace
                ),
                text=sec_df['symbol'] + "<br>" + sec_df['shortName'].astype(str),
                hovertemplate="<b>%{text}</b><br>Cap: $%{marker.size:.0f} (Scaled)<br>X: %{x}<br>Y: %{y}<extra></extra>"
            )
            traces.append(trace)

        fig = go.Figure(data=traces)

        # 2. Build Axis Dropdowns (Must update ALL traces)
        # The 'args' must provide a list of arrays, one for each trace in order
        
        def build_axis_args(axis_col):
            # Returns the list of data arrays for every sector trace
            update_data = []
            for sec in sectors:
                sec_df = self.df[self.df['sector'] == sec]
                update_data.append(sec_df[axis_col])
            return update_data

        x_buttons = []
        for m in self.metrics:
            x_buttons.append(dict(
                method='update',
                label=m,
                args=[
                    {'x': build_axis_args(m)}, 
                    {'xaxis': {'title': m}}
                ]
            ))

        y_buttons = []
        for m in self.metrics:
            y_buttons.append(dict(
                method='update',
                label=m,
                args=[
                    {'y': build_axis_args(m)},
                    {'yaxis': {'title': m}}
                ]
            ))

        # 3. Build Filter Dropdown (Equities vs ETFs)
        # The 'visible' property takes a list of booleans matching the trace count
        
        total_traces = len(traces)
        
        # All Visible
        vis_all = [True] * total_traces
        
        # Equities Only (True if index in equity_indices)
        vis_eq = [True if x in equity_indices else False for x in range(total_traces)]
        
        # ETFs Only (True if index in etf_indices)
        vis_etf = [True if x in etf_indices else False for x in range(total_traces)]

        filter_buttons = [
            dict(label="All Assets", method="restyle", args=["visible", vis_all]),
            dict(label="Equities Only", method="restyle", args=["visible", vis_eq]),
            dict(label="ETFs Only", method="restyle", args=["visible", vis_etf])
        ]

        # 4. Layout
        fig.update_layout(
            title=f"Market Screener: Sector & Cap Analysis ({datetime.now().strftime('%Y-%m-%d')})",
            template="plotly_dark",
            height=850,
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333'),
            legend=dict(title="Sector"),
            updatemenus=[
                dict(
                    buttons=x_buttons, direction="down", showactive=True,
                    x=0.05, xanchor="left", y=1.12, yanchor="top"
                ),
                dict(
                    buttons=y_buttons, direction="down", showactive=True,
                    x=0.25, xanchor="left", y=1.12, yanchor="top"
                ),
                dict(
                    buttons=filter_buttons, direction="down", showactive=True,
                    x=0.45, xanchor="left", y=1.12, yanchor="top"
                ),
            ],
            annotations=[
                dict(text="X-Axis", x=0.05, y=1.15, xref="paper", yref="paper", showarrow=False, align="left", font=dict(color="gray")),
                dict(text="Y-Axis", x=0.25, y=1.15, xref="paper", yref="paper", showarrow=False, align="left", font=dict(color="gray")),
                dict(text="Filter", x=0.45, y=1.15, xref="paper", yref="paper", showarrow=False, align="left", font=dict(color="gray"))
            ]
        )

        # 5. Save Output
        output_file = os.path.join(self.output_dir, "screener_dashboard.html")
        plot_div = py_offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        
        html_content = f"""
        <html>
        <head>
            <title>Advanced Market Screener</title>
            <style>
                body{{margin:0; padding:0; background-color:#111; color:#ddd; font-family:'Segoe UI', sans-serif;}}
                .controls-info {{ padding: 10px 40px; font-size: 0.9em; color: #888; border-bottom: 1px solid #333; }}
            </style>
        </head>
        <body>
            <div class="controls-info">
                <strong>Config:</strong> Size = Log(MarketCap) &bull; Color = Sector &bull; Hover for Details.
            </div>
            {plot_div}
            <script>window.dispatchEvent(new Event('resize'));</script>
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(html_content)
        
        print(f"[Success] Dashboard saved to: {output_file}")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Senior Quant Market Screener")
    
    # Updated default list to show off the sector coloring better
    defaults = [
        'SPY', 'QQQ', 'IWM', # ETFs
        'AAPL', 'MSFT', 'NVDA', 'ORCL', 'ADBE', # Tech
        'JPM', 'BAC', 'GS', 'MS', # Financials
        'JNJ', 'PFE', 'LLY', 'UNH', # Healthcare
        'XOM', 'CVX', 'COP', # Energy
        'AMT', 'PLD', # Real Estate
        'PG', 'KO', 'PEP' # Staples
    ]
    
    parser.add_argument('--tickers', nargs='+', default=defaults, help='List of ticker symbols')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Directory for data and HTML')
    parser.add_argument('--lookback', type=int, default=1, help='Years of history')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate')
    
    args = parser.parse_args()

    ingestor = DataIngestion(tickers=args.tickers, output_dir=args.output_dir)
    raw_data = ingestor.fetch_data()

    if not raw_data:
        print("No data found. Exiting.")
        return

    analyst = FinancialAnalysis(raw_data)
    clean_df = analyst.process()

    renderer = DashboardRenderer(clean_df, args.output_dir)
    renderer.generate_dashboard()

if __name__ == "__main__":
    main()
