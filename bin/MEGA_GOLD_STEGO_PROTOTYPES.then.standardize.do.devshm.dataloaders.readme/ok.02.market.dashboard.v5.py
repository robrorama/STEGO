import os
import time
import argparse
import sys
import webbrowser
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

# ==========================================
# 1. DataIngestion (The Gatekeeper)
# ==========================================
class DataIngestion:
    """
    Handles all I/O, downloading, caching, and sanitization.
    Protocol: Disk-First, Network-Second.
    """
    def __init__(self, output_dir, lookback_years):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _sanitize_df(self, df):
        """
        Sanitizes the dataframe: flattens MultiIndex, strips timezones, 
        forces floats.
        """
        # 1. Swap Levels / Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # If the columns are MultiIndex (e.g., Price, Ticker), drop the Ticker level
            df.columns = df.columns.droplevel(1)
        
        # Ensure columns are clean strings
        df.columns = [str(c).capitalize() for c in df.columns]

        # 2. Timezones
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 3. Coercion
        # Ensure standard OHLCV columns exist and are float
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def get_data(self, ticker):
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        
        # 1. Check Local
        if os.path.exists(file_path):
            print(f"[CACHE HIT] Loading {ticker} from disk...")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # Re-sanitize on load to be safe
            return self._sanitize_df(df)
        
        # 2. Network Download
        print(f"[NETWORK] Downloading {ticker}...")
        
        # Shadow Backfill: Request X + 1 years
        start_date = datetime.now() - timedelta(days=365 * (self.lookback_years + 1))
        
        # CRITICAL RATE LIMITING
        time.sleep(1) 
        
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            
            if df.empty:
                print(f"Error: No data found for {ticker}")
                return None

            # 3. Sanitize
            df = self._sanitize_df(df)

            # 4. Save to Disk
            df.to_csv(file_path)
            
            # 5. Return (Reload from disk logic implied by returning the sanitized object)
            return df

        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            return None


# ==========================================
# 2. FinancialAnalysis (The Engine)
# ==========================================
class FinancialAnalysis:
    """
    Pure mathematical logic. No I/O.
    """
    def __init__(self):
        self.MA_PERIOD = 30
        self.T3_VFACTOR = 0.7
        self.T3_PERIOD = 5
    
    def apply_indicators(self, df):
        # Work on a copy to avoid SettingWithCopy warnings
        d = df.copy()
        
        # Pre-calculate common inputs
        O = d['Open'].values
        H = d['High'].values
        L = d['Low'].values
        C = d['Close'].values
        V = d['Volume'].values.astype(float) # talib requires float volume

        # --- Overlap Studies ---
        d['BB_UP'], d['BB_MID'], d['BB_LOW'] = talib.BBANDS(C, timeperiod=self.MA_PERIOD)
        d['SMA'] = talib.SMA(C, timeperiod=self.MA_PERIOD)
        d['EMA'] = talib.EMA(C, timeperiod=self.MA_PERIOD)
        d['WMA'] = talib.WMA(C, timeperiod=self.MA_PERIOD)
        d['TRIMA'] = talib.TRIMA(C, timeperiod=self.MA_PERIOD)
        d['DEMA'] = talib.DEMA(C, timeperiod=self.MA_PERIOD)
        d['TEMA'] = talib.TEMA(C, timeperiod=self.MA_PERIOD)
        d['T3'] = talib.T3(C, timeperiod=self.T3_PERIOD, vfactor=self.T3_VFACTOR)
        d['KAMA'] = talib.KAMA(C, timeperiod=self.MA_PERIOD)
        d['HT_TRENDLINE'] = talib.HT_TRENDLINE(C)
        d['MAMA'], d['FAMA'] = talib.MAMA(C)
        d['SAR'] = talib.SAR(H, L)
        d['SAREXT'] = talib.SAREXT(H, L)

        # Ichimoku (Manual Calculation as TA-Lib doesn't have a direct single function)
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
        nine_period_high = d['High'].rolling(window=9).max()
        nine_period_low = d['Low'].rolling(window=9).min()
        d['ICH_TENKAN'] = (nine_period_high + nine_period_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        period26_high = d['High'].rolling(window=26).max()
        period26_low = d['Low'].rolling(window=26).min()
        d['ICH_KIJUN'] = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
        d['ICH_SPAN_A'] = ((d['ICH_TENKAN'] + d['ICH_KIJUN']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
        period52_high = d['High'].rolling(window=52).max()
        period52_low = d['Low'].rolling(window=52).min()
        d['ICH_SPAN_B'] = ((period52_high + period52_low) / 2).shift(26)

        # Chikou Span (Lagging Span): Close plotted 26 days in the past
        d['ICH_CHIKOU'] = d['Close'].shift(-26)

        # --- Momentum Indicators ---
        d['ADX'] = talib.ADX(H, L, C, timeperiod=14)
        d['APO'] = talib.APO(C)
        d['AROON_DOWN'], d['AROON_UP'] = talib.AROON(H, L)
        d['CCI'] = talib.CCI(H, L, C)
        d['CMO'] = talib.CMO(C)
        d['MACD'], d['MACD_SIG'], d['MACD_HIST'] = talib.MACD(C)
        d['MACDEXT'], d['MACDEXT_SIG'], d['MACDEXT_HIST'] = talib.MACDEXT(C)
        d['MACDFIX'], d['MACDFIX_SIG'], d['MACDFIX_HIST'] = talib.MACDFIX(C)
        d['MFI'] = talib.MFI(H, L, C, V)
        d['MOM'] = talib.MOM(C)
        d['PPO'] = talib.PPO(C)
        d['ROC'] = talib.ROC(C)
        d['RSI'] = talib.RSI(C)
        d['STOCH_K'], d['STOCH_D'] = talib.STOCH(H, L, C)
        d['STOCHF_K'], d['STOCHF_D'] = talib.STOCHF(H, L, C)
        d['STOCHRSI_K'], d['STOCHRSI_D'] = talib.STOCHRSI(C)
        d['TRIX'] = talib.TRIX(C)
        d['ULTOSC'] = talib.ULTOSC(H, L, C)
        d['WILLR'] = talib.WILLR(H, L, C)
        d['DX'] = talib.DX(H, L, C) # Added specifically for Favorite Signals

        # --- Volume Indicators ---
        d['AD'] = talib.AD(H, L, C, V)
        d['ADOSC'] = talib.ADOSC(H, L, C, V)
        d['OBV'] = talib.OBV(C, V)

        # --- Price Transform ---
        d['AVGPRICE'] = talib.AVGPRICE(O, H, L, C)
        d['MEDPRICE'] = talib.MEDPRICE(H, L)
        d['TYPPRICE'] = talib.TYPPRICE(H, L, C)
        d['WCLPRICE'] = talib.WCLPRICE(H, L, C)

        # --- Cycle Indicators ---
        d['HT_DCPERIOD'] = talib.HT_DCPERIOD(C)
        d['HT_PHASOR_INPHASE'], d['HT_PHASOR_QUAD'] = talib.HT_PHASOR(C)
        d['HT_SINE'], d['HT_LEADSINE'] = talib.HT_SINE(C)

        # --- Statistic Functions ---
        d['BETA'] = talib.BETA(H, L)
        d['CORREL'] = talib.CORREL(H, L)
        d['LINEARREG'] = talib.LINEARREG(C)
        d['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(C)
        d['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(C)
        d['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(C)
        d['STDDEV'] = talib.STDDEV(C)
        d['TSF'] = talib.TSF(C)
        d['VAR'] = talib.VAR(C)
        
        # Extra required for "ALL_SIGNALS"
        d['ATR'] = talib.ATR(H, L, C)

        return d

# ==========================================
# 3. DashboardRenderer (The Presenter)
# ==========================================
class DashboardRenderer:
    """
    Generates HTML reports strictly offline.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def _inject_offline_assets(self, fig_html):
        """
        Injects local Plotly JS and the Tab Resize Fix.
        """
        resize_fix = "<script>window.dispatchEvent(new Event('resize'));</script>"
        return fig_html.replace('</head>', f'{resize_fix}</head>')

    def _save_plot(self, fig, ticker, filename):
        ticker_html_dir = os.path.join(self.output_dir, ticker, 'html')
        # Ensure directory exists
        if not os.path.exists(ticker_html_dir):
            os.makedirs(ticker_html_dir)
        
        # Generate HTML
        raw_html = fig.to_html(include_plotlyjs=True, full_html=True)
        final_html = self._inject_offline_assets(raw_html)
        
        output_path = os.path.join(ticker_html_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"   -> Generated: {output_path}")
        return output_path

    def render_dashboards(self, df, ticker):
        print(f"Rendering dashboards for {ticker}...")
        generated_files = {} # Changed to dict to store category keys for the index
        
        # Helper to add standard candlestick
        def add_candlestick(fig, row, col):
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                         low=df['Low'], close=df['Close'], name='Price'), row=row, col=col)

        # 1. ALL_SIGNALS
        fig = make_subplots(rows=17, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        add_candlestick(fig, 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], name='BB Upper', line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOW'], name='BB Lower', line=dict(width=1)), row=1, col=1)
        
        indicators = [
            ('ADX', 'ADX'), ('STDDEV', 'STDDEV'), ('ATR', 'ATR'), ('CMO', 'CMO'), 
            ('ROC', 'ROC'), ('WILLR', 'WILLR'), ('STOCH_K', 'Stoch'), 
            ('LINEARREG_SLOPE', 'LR Slope'), ('RSI', 'RSI'), ('MACD', 'MACD'), 
            ('OBV', 'OBV'), ('SAR', 'SAR'), ('ULTOSC', 'UltOsc'), 
            ('MFI', 'MFI'), ('CCI', 'CCI'), ('ICH_TENKAN', 'Ichimoku')
        ]
        
        for i, (col_name, label) in enumerate(indicators):
            row_idx = i + 2
            if col_name == 'Ichimoku':
                 fig.add_trace(go.Scatter(x=df.index, y=df['ICH_TENKAN'], name='Tenkan'), row=row_idx, col=1)
                 fig.add_trace(go.Scatter(x=df.index, y=df['ICH_KIJUN'], name='Kijun'), row=row_idx, col=1)
            elif col_name == 'MACD':
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='MACD Hist'), row=row_idx, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=row_idx, col=1)
            else:
                fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=label), row=row_idx, col=1)

        fig.update_layout(title=f"{ticker} - ALL SIGNALS", height=3000, template='plotly_dark')
        generated_files['ALL SIGNALS'] = self._save_plot(fig, ticker, "ALL_SIGNALS.html")

        # 2. FAVORITE_SIGNALS
        fig = make_subplots(rows=9, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['TSF'], name='TSF'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['WCLPRICE'], name='WCLPrice'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['VAR'], name='VAR'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['DX'], name='DX'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HT_PHASOR_INPHASE'], name='InPhase'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HT_PHASOR_QUAD'], name='Quad'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HT_SINE'], name='Sine'), row=5, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HT_LEADSINE'], name='LeadSine'), row=5, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['LINEARREG_INTERCEPT'], name='Intercept'), row=6, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['LINEARREG_ANGLE'], name='Angle'), row=7, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['LINEARREG_SLOPE'], name='Slope'), row=8, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCHF_K'], name='StochF K'), row=9, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['STOCHRSI_K'], name='StochRSI K'), row=9, col=1)
        fig.update_layout(title=f"{ticker} - FAVORITES", height=1800, template='plotly_dark')
        generated_files['FAVORITES'] = self._save_plot(fig, ticker, "FAVORITE_SIGNALS.html")

        # 3. OTHER_LINES
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACDEXT'], name='MACD Ext'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACDFIX'], name='MACD Fix'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['FAMA'], name='FAMA'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MAMA'], name='MAMA'), row=3, col=1)
        fig.update_layout(title=f"{ticker} - OTHER LINES", height=800, template='plotly_dark')
        generated_files['OTHER LINES'] = self._save_plot(fig, ticker, "OTHER_LINES.html")

        # 4. COMPREHENSIVE
        fig = make_subplots(rows=1, cols=1) 
        add_candlestick(fig, 1, 1) 
        for col in df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, opacity=0.5), row=1, col=1)
        fig.update_layout(title=f"{ticker} - COMPREHENSIVE", height=1200, template='plotly_dark')
        generated_files['COMPREHENSIVE'] = self._save_plot(fig, ticker, "COMPREHENSIVE.html")

        # 5. TA-Lib Categories
        categories = {
            "Overlap": ['SMA', 'EMA', 'WMA', 'TRIMA', 'DEMA', 'TEMA', 'T3', 'KAMA', 'HT_TRENDLINE', 'SAR', 'SAREXT', 'BB_UP', 'MAMA'],
            "Momentum": ['ADX', 'APO', 'AROON_UP', 'CCI', 'CMO', 'MACD', 'MFI', 'MOM', 'PPO', 'ROC', 'RSI', 'STOCH_K', 'TRIX', 'ULTOSC', 'WILLR'],
            "Volume": ['AD', 'ADOSC', 'OBV'],
            "Volatility": ['ATR'],
            "Price_Transform": ['AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'],
            "Cycle": ['HT_DCPERIOD', 'HT_PHASOR_INPHASE', 'HT_SINE'],
            "Statistics": ['BETA', 'CORREL', 'LINEARREG', 'STDDEV', 'VAR', 'TSF']
        }
        
        for cat_name, cols in categories.items():
            fig = make_subplots(rows=len(cols)+1, cols=1, shared_xaxes=True)
            add_candlestick(fig, 1, 1)
            for i, col in enumerate(cols):
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col), row=i+2, col=1)
            
            fig.update_layout(title=f"{ticker} - {cat_name.upper()}", height=300 * (len(cols)+1), template='plotly_dark')
            generated_files[f'TALIB {cat_name}'] = self._save_plot(fig, ticker, f"TALIB_{cat_name}.html")
        
        return generated_files

    def create_index_page(self, ticker_map):
        """
        Creates a simple HTML index page linking to all generated dashboards.
        ticker_map: dict {ticker: {report_name: file_path}}
        """
        index_path = os.path.join(self.output_dir, "index.html")
        
        html_content = """
        <html>
        <head>
            <title>Market Analysis Dashboard</title>
            <style>
                body { font-family: sans-serif; background-color: #1e1e1e; color: #ddd; padding: 20px; }
                h1 { border-bottom: 2px solid #444; padding-bottom: 10px; }
                .ticker-block { background: #2e2e2e; margin: 20px 0; padding: 15px; border-radius: 8px; }
                .ticker-header { font-size: 1.5em; font-weight: bold; color: #4CAF50; margin-bottom: 10px; }
                .link-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
                a { display: block; padding: 10px; background: #3e3e3e; color: #fff; text-decoration: none; border-radius: 4px; text-align: center; }
                a:hover { background: #555; }
            </style>
        </head>
        <body>
            <h1>Market Analysis Engine - Index</h1>
        """
        
        for ticker, files in ticker_map.items():
            html_content += f"""
            <div class="ticker-block">
                <div class="ticker-header">{ticker}</div>
                <div class="link-grid">
            """
            for report_name, file_path in files.items():
                # Create relative path for the link
                rel_path = os.path.relpath(file_path, self.output_dir)
                html_content += f'<a href="{rel_path}" target="_blank">{report_name}</a>'
            
            html_content += "</div></div>"
        
        html_content += "</body></html>"
        
        with open(index_path, "w") as f:
            f.write(html_content)
        
        return index_path


# ==========================================
# Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Hedge Fund Grade Technical Analysis Engine")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM'], help='List of ticker symbols')
    parser.add_argument('--output-dir', default='./market_data', help='Output directory')
    parser.add_argument('--lookback', type=int, default=1, help='Years of history')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate')
    
    args = parser.parse_args()
    
    # Instantiate Classes
    ingestion = DataIngestion(args.output_dir, args.lookback)
    analysis = FinancialAnalysis()
    renderer = DashboardRenderer(args.output_dir)
    
    print(f"Starting analysis for: {args.tickers}")
    print(f"Storage: {args.output_dir}")
    
    # Master dictionary to store all generated file paths
    # Structure: { 'SPY': {'ALL_SIGNALS': '/path/to/file', ...}, 'QQQ': ... }
    master_file_map = {}

    for ticker in args.tickers:
        print(f"\n--- Processing {ticker} ---")
        
        # 1. Ingest
        df = ingestion.get_data(ticker)
        if df is None:
            continue
            
        # 2. Analyze
        print("Calculating indicators...")
        df_analyzed = analysis.apply_indicators(df)
        
        # 3. Render
        # renderer now returns a Dict: {'ReportName': 'path'}
        ticker_files = renderer.render_dashboards(df_analyzed, ticker)
        master_file_map[ticker] = ticker_files
        
    # 4. Generate Index Page
    print("\nGenerating Index Page...")
    index_page = renderer.create_index_page(master_file_map)
    
    print(f"Job Complete.")
    print(f"Opening Index: {index_page}")
    
    # Only open the ONE index file
    webbrowser.open(f'file://{os.path.abspath(index_page)}')

if __name__ == "__main__":
    main()
