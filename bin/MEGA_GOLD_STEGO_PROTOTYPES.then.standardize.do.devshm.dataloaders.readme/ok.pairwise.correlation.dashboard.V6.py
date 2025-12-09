import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List, Optional, Dict
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = "./raw_data"
ROLLING_WINDOWS = [30, 90]  # Days for rolling windows

class MarketDataGateway:
    """
    I/O Layer. Responsible for fetching, caching, and sanitizing market data.
    Strictly forbids statistical analysis logic.
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_price_series(self, ticker: str) -> pd.Series:
        """
        Public entry point. Implements Read-Through Caching.
        """
        clean_ticker = ticker.upper().strip()
        file_path = os.path.join(self.cache_dir, f"{clean_ticker}.csv")

        # 1. Try Cache
        if os.path.exists(file_path):
            logger.info(f"Cache hit: {clean_ticker}")
            try:
                df = pd.read_csv(file_path)
                return self._sanitize_series(df, clean_ticker)
            except Exception as e:
                logger.warning(f"Corrupt cache for {clean_ticker}, redownloading. Error: {e}")

        # 2. Download (if cache missing or corrupt)
        return self._download_and_cache(clean_ticker, file_path)

    def _download_and_cache(self, ticker: str, file_path: str) -> pd.Series:
        """
        Downloads from yfinance and writes to disk immediately.
        """
        logger.info(f"Downloading: {ticker}")
        try:
            # Download full history
            df = yf.download(ticker, progress=False, multi_level_index=False)
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")

            # Save raw data to disk (persistence)
            df.to_csv(file_path)
            
            return self._sanitize_series(df, ticker)
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            raise

    def _sanitize_series(self, data: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Mandatory Data Sanitization Pipeline.
        Enforces index, checks for 'fake' data, and ensures numeric types.
        """
        # Create a deep copy to ensure immutability of source
        df = data.copy()

        # 1. Date Index Enforcement
        if not isinstance(df.index, pd.DatetimeIndex):
            # Attempt to find date column
            candidates = [c for c in df.columns if c.lower() in ['date', 'datetime', 'timestamp']]
            if candidates:
                df[candidates[0]] = pd.to_datetime(df[candidates[0]])
                df.set_index(candidates[0], inplace=True)
            else:
                raise ValueError(f"CRITICAL: Could not identify Datetime Index for {ticker}")

        # 2. Missing 'Close' Fallback
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                raise ValueError(f"CRITICAL: No Close/Adj Close column for {ticker}")

        # 3. Type Coercion
        series = pd.to_numeric(df['Close'], errors='coerce')
        
        # Drop NaNs created by coercion or missing data
        series = series.dropna()

        if series.empty:
            raise ValueError(f"Series is empty after sanitization for {ticker}")

        # 4. "Straight Line" / Counter Detection
        # Check if price is just 0, 1, 2, 3... or constant
        vals = series.values
        if len(vals) > 10:
            # Check for exact row counter match
            row_counter = np.arange(len(vals))
            # Check correlation with row counter or if std dev is 0
            if np.array_equal(vals, row_counter) or np.std(vals) == 0:
                raise ValueError(f"CRITICAL: Data detected as synthetic straight line/counter for {ticker}")

        series.name = ticker
        return series


class PairwiseAnalyzer:
    """
    Logic Layer. Pure statistical logic.
    Stateless: Receives data, returns new metrics.
    """
    
    def __init__(self):
        pass

    def calculate_log_returns(self, series: pd.Series) -> pd.Series:
        """ Calculates ln(Pt / Pt-1) """
        # Copy on write
        s = series.copy()
        return np.log(s / s.shift(1)).dropna()

    def align_series(self, s1: pd.Series, s2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """ 
        Strict Alignment.
        Uses inner join to ensure timestamps match exactly.
        """
        df = pd.concat([s1, s2], axis=1, join='inner').dropna()
        return df.iloc[:, 0], df.iloc[:, 1]

    def calculate_metrics(self, asset_a: pd.Series, asset_b: pd.Series) -> Dict:
        """
        Orchestrates the calculation of Rolling Correlation and Beta.
        Asset A is the primary, Asset B is the benchmark (for Beta).
        """
        # 1. Calculate Returns
        ret_a = self.calculate_log_returns(asset_a)
        ret_b = self.calculate_log_returns(asset_b)

        # 2. Re-Align returns (dropping the first NaN from shift)
        ret_a, ret_b = self.align_series(ret_a, ret_b)
        
        # 3. Align raw prices for performance chart (relative to common start date)
        price_a, price_b = self.align_series(asset_a, asset_b)
        
        # Normalize prices to 100
        norm_a = (price_a / price_a.iloc[0]) * 100
        norm_b = (price_b / price_b.iloc[0]) * 100

        results = {
            'prices': {'A': norm_a, 'B': norm_b},
            'rolling_stats': {}
        }

        # 4. Rolling Calculations
        for window in ROLLING_WINDOWS:
            # Pearson Correlation
            corr = ret_a.rolling(window=window).corr(ret_b)
            
            # Rolling Beta: Cov(A,B) / Var(B)
            cov = ret_a.rolling(window=window).cov(ret_b)
            var = ret_b.rolling(window=window).var()
            beta = cov / var

            results['rolling_stats'][window] = {
                'correlation': corr,
                'beta': beta
            }

        return results


class DashboardRenderer:
    """
    Visuals Layer. Generates HTML artifacts.
    """
    def generate_html_dashboard(self, results_map: Dict, output_file: str):
        """
        Creates a Single File HTML with custom JS tabs.
        results_map format: { "TICKER_A-TICKER_B": analysis_results_dict }
        """
        
        tabs_html = []
        content_html = []
        
        first = True
        
        for pair_name, data in results_map.items():
            fig = self._create_figure(pair_name, data)
            div_id = f"chart_{pair_name.replace('-', '_')}"
            
            # Generate Plotly DIV
            plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id=div_id)
            
            # Create Tab Button
            active_cls = "active" if first else ""
            tabs_html.append(
                f'<button class="tab-link {active_cls}" onclick="openPair(event, \'{div_id}\')">{pair_name}</button>'
            )
            
            # Create Tab Content
            display_style = "block" if first else "none"
            content_html.append(
                f'<div id="{div_id}" class="tab-content" style="display: {display_style};">{plot_div}</div>'
            )
            first = False

        # Assemble Full HTML
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pairwise Quant Dashboard</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f4f4f9; }}
                h1 {{ color: #333; }}
                .tab-container {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 5px 5px 0 0; }}
                .tab-link {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 15px; font-weight: 600; }}
                .tab-link:hover {{ background-color: #ddd; }}
                .tab-link.active {{ background-color: #007bff; color: white; }}
                .tab-content {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; background: white; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h1>Rolling Correlation & Beta Dashboard</h1>
            <div class="tab-container">
                {''.join(tabs_html)}
            </div>
            {''.join(content_html)}

            <script>
            function openPair(evt, pairName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tab-link");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(pairName).style.display = "block";
                evt.currentTarget.className += " active";
                
                // Trigger Plotly resize to fix rendering when unhiding
                window.dispatchEvent(new Event('resize'));
            }}
            </script>
        </body>
        </html>
        """

        with open(output_file, "w", encoding='utf-8') as f:
            f.write(full_html)
        logger.info(f"Dashboard saved successfully to {output_file}")

    def _create_figure(self, pair_name: str, data: Dict) -> go.Figure:
        ticker_a, ticker_b = pair_name.split('-')
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"Relative Performance (Rebased to 100)", 
                "Rolling Correlation", 
                f"Rolling Beta ({ticker_a} vs {ticker_b})"
            )
        )

        # Panel 1: Relative Performance
        prices = data['prices']
        fig.add_trace(go.Scatter(x=prices['A'].index, y=prices['A'], name=ticker_a, line=dict(width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=prices['B'].index, y=prices['B'], name=ticker_b, line=dict(width=2)), row=1, col=1)

        # Panel 2 & 3: Rolling Stats
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, window in enumerate(ROLLING_WINDOWS):
            stats = data['rolling_stats'][window]
            color = colors[i % len(colors)]
            
            # Correlation
            fig.add_trace(go.Scatter(
                x=stats['correlation'].index, 
                y=stats['correlation'], 
                name=f'{window}d Corr',
                line=dict(width=1.5, color=color),
                legendgroup=f'g{window}'
            ), row=2, col=1)

            # Beta
            fig.add_trace(go.Scatter(
                x=stats['beta'].index, 
                y=stats['beta'], 
                name=f'{window}d Beta',
                line=dict(width=1.5, color=color, dash='solid'),
                legendgroup=f'g{window}',
                showlegend=False
            ), row=3, col=1)

        # Decorations
        # Panel 2: Correlation Zones
        fig.add_hrect(y0=0.7, y1=1.0, fillcolor="green", opacity=0.1, line_width=0, row=2, col=1)
        fig.add_hrect(y0=-1.0, y1=-0.7, fillcolor="red", opacity=0.1, line_width=0, row=2, col=1)
        
        # Panel 3: Beta Benchmark Line
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(height=900, title_text=f"Pair Analysis: {pair_name}")
        return fig


def main():
    parser = argparse.ArgumentParser(description="Pairwise Rolling Correlation & Beta Dashboard")
    parser.add_argument("--pairs-csv", type=str, required=True, help="Path to CSV file with columns TickerA, TickerB")
    parser.add_argument("--output", type=str, default="dashboard.html", help="Output HTML filename")
    
    args = parser.parse_args()

    # 1. Initialize Components
    gateway = MarketDataGateway()
    analyzer = PairwiseAnalyzer()
    renderer = DashboardRenderer()
    
    results_map = {}

    # 2. Read Pairs Input
    try:
        # Assuming headerless CSV: TickerA, TickerB or with headers
        # We try to detect logic roughly
        input_df = pd.read_csv(args.pairs_csv, header=None)
        # If user provided headers, input_df might look wrong, simple normalization:
        if len(input_df.columns) < 2:
            logger.error("CSV must have at least two columns.")
            sys.exit(1)
            
        pairs = input_df.values.tolist()
    except Exception as e:
        logger.error(f"Failed to read pairs CSV: {e}")
        sys.exit(1)

    # 3. Processing Loop
    for pair in pairs:
        t_a, t_b = str(pair[0]).strip(), str(pair[1]).strip()
        pair_key = f"{t_a}-{t_b}"
        
        logger.info(f"Processing Pair: {pair_key}")
        
        try:
            # A. Fetch Data (Gateway)
            series_a = gateway.get_price_series(t_a)
            series_b = gateway.get_price_series(t_b)
            
            # B. Analyze (Analyzer)
            metrics = analyzer.calculate_metrics(series_a, series_b)
            
            # Store results
            results_map[pair_key] = metrics
            
        except ValueError as ve:
            logger.warning(f"Skipping pair {pair_key} due to data validation error: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error processing {pair_key}: {e}")

    # 4. Generate Output
    if not results_map:
        logger.error("No valid pairs processed. Dashboard generation aborted.")
        sys.exit(1)

    renderer.generate_html_dashboard(results_map, args.output)

if __name__ == "__main__":
    main()
