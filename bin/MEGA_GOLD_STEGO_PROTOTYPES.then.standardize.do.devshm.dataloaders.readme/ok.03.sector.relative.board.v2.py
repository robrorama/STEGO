import os
import time
import argparse
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot, get_plotlyjs
from datetime import datetime, timedelta
from tqdm import tqdm  # NEW: For progress bar

# -----------------------------------------------------------------------------
# DEFAULT CONFIGURATION
# -----------------------------------------------------------------------------
DEFAULT_UNIVERSE = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Energy': 'XLE',
    'Consumer Discret': 'XLY',
    'Consumer Staples': 'XLP',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
    'Industrials': 'XLI',
    'Comms Services': 'XLC',
    'Semi-Conductors': 'SMH',
    'Biotech': 'IBB',
    'Homebuilders': 'XHB'
}

# -----------------------------------------------------------------------------
# CLASS 1: DataIngestion
# Solely responsible for I/O, downloading, and sanitization.
# -----------------------------------------------------------------------------
class DataIngestion:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df):
        """
        The 'Universal Fixer' for yfinance formatting issues.
        """
        # 1. Handle MultiIndex Columns
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' not in df.columns.get_level_values(0) and 'Close' in df.columns.get_level_values(1):
                df = df.swaplevel(0, 1, axis=1)
            
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            
        # 2. Strict Indexing
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone info
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 3. Coerce to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def get_ticker_data(self, ticker, lookback_years=1):
        """
        Disk-First Pipeline: Check Disk -> Download if Missing -> Read Disk
        Uses tqdm.write to print logs without breaking the progress bar.
        """
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        
        # 1. Check Disk
        if os.path.exists(file_path):
            tqdm.write(f"[DISK] Loading {ticker} from {file_path}...")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df

        # 2. IF MISSING: Download
        tqdm.write(f"[API] Downloading {ticker} (Staggered)...")
        start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
        
        try:
            # Enforce sleep to prevent rate limiting
            # This ensures <1 request per second strictly
            time.sleep(1) 
            
            df = yf.download(ticker, start=start_date, progress=False)
            
            if df.empty:
                tqdm.write(f"[WARN] No data found for {ticker}")
                return pd.DataFrame()

            # 3. Sanitize
            df = self._sanitize_df(df)
            
            # 4. Save to Disk
            df.to_csv(file_path)
            
            # 5. Read Back (Strict Disk-First Rule)
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
            
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to download {ticker}: {e}")
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# CLASS 2: FinancialAnalysis
# Solely responsible for mathematical normalization.
# -----------------------------------------------------------------------------
class FinancialAnalysis:
    @staticmethod
    def calculate_relative_return(df):
        """
        Normalizes price history to percentage return relative to start date.
        Formula: Rt = (Pt / P0 - 1) * 100
        """
        if df.empty or 'Close' not in df.columns:
            return pd.Series(dtype=float)
        
        # Locate P0 (first valid Close price)
        series = df['Close'].dropna()
        if series.empty:
            return pd.Series(dtype=float)
            
        p0 = series.iloc[0]
        
        # Calculate Relative Return
        relative_return = ((series / p0) - 1) * 100
        
        return relative_return

# -----------------------------------------------------------------------------
# CLASS 3: DashboardRenderer
# Solely responsible for generating the Plotly HTML output.
# -----------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def render_dashboard(self, results_map):
        """
        Generates the HTML dashboard with Left (Line) and Right (Bar) panels.
        """
        if not results_map:
            tqdm.write("No data to render.")
            return

        # Prepare data for Bar Chart (Ranking)
        final_returns = []
        for ticker, series in results_map.items():
            if not series.empty:
                final_returns.append({
                    'ticker': ticker,
                    'final_val': series.iloc[-1],
                    'series': series
                })
        
        # Sort by final return (Best at top)
        final_returns.sort(key=lambda x: x['final_val'], reverse=False)

        # Define Layout
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=("Relative Performance (%)", "Current Ranking"),
            horizontal_spacing=0.05
        )

        # Generate Colors (Spectral palette)
        import plotly.colors as pcolors
        colors = pcolors.n_colors('rgb(5, 10, 172)', 'rgb(40, 190, 24)', len(final_returns), colortype='rgb')

        # Add Traces
        for i, item in enumerate(final_returns):
            ticker = item['ticker']
            series = item['series']
            color = colors[i]
            
            # Left Panel: Line Chart
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    name=ticker,
                    line=dict(color=color, width=2),
                    legendgroup=ticker,
                    showlegend=True
                ),
                row=1, col=1
            )

            # Right Panel: Bar Chart
            fig.add_trace(
                go.Bar(
                    x=[item['final_val']],
                    y=[ticker],
                    orientation='h',
                    name=ticker,
                    marker=dict(color=color),
                    text=[f"{item['final_val']:.2f}%"],
                    textposition='auto',
                    legendgroup=ticker,
                    showlegend=False
                ),
                row=1, col=2
            )

        # Update Layout
        fig.update_layout(
            template="plotly_white",
            height=800,
            title_text="Sector Relative Performance Dashboard",
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Zero line for Left Panel
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

        # Generate HTML
        plotly_js = get_plotlyjs()
        plot_div = plot(fig, output_type='div', include_plotlyjs=False)

        # Resize Fix Script
        resize_script = """
        <script>
            window.addEventListener('resize', function() {
                window.dispatchEvent(new Event('resize'));
            });
        </script>
        """

        html_content = f"""
        <html>
        <head>
            <title>Market Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
        </head>
        <body>
            {plot_div}
            {resize_script}
        </body>
        </html>
        """

        output_path = os.path.join(self.output_dir, "dashboard.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        tqdm.write(f"[SUCCESS] Dashboard generated at: {output_path}")

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sector Relative Performance Dashboard")
    parser.add_argument('--tickers', nargs='+', help='List of tickers')
    parser.add_argument('--output-dir', default='./market_data', help='Directory to store data and html')
    parser.add_argument('--lookback', type=int, default=1, help='Years of history')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate (unused in this view)')
    
    args = parser.parse_args()

    # Determine Universe
    tickers = args.tickers if args.tickers else list(DEFAULT_UNIVERSE.values())
    
    # Instantiate Modules
    ingestor = DataIngestion(args.output_dir)
    analyzer = FinancialAnalysis()
    renderer = DashboardRenderer(args.output_dir)
    
    results_map = {}

    # Execution Loop with Progress Bar
    # "unit='ticker'" adds context to the bar stats (e.g., "1.2s/ticker")
    with tqdm(tickers, unit="ticker") as pbar:
        for ticker in pbar:
            # Update the progress bar description to show what is currently happening
            pbar.set_description(f"Processing {ticker}")
            
            # 1. Ingest (Disk -> API -> Disk)
            df = ingestor.get_ticker_data(ticker, args.lookback)
            
            # 2. Analyze
            if not df.empty:
                relative_series = analyzer.calculate_relative_return(df)
                if not relative_series.empty:
                    results_map[ticker] = relative_series
    
    # 3. Render
    renderer.render_dashboard(results_map)

if __name__ == "__main__":
    main()
