import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import webbrowser

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
TICKER_MAP = {
    '10Y_YIELD': '^TNX',
    '3M_YIELD': '^IRX',
    'DXY': 'DX-Y.NYB',
    'BRENT': 'BZ=F',
    'GOLD': 'GC=F',
    'COPPER': 'HG=F',
    'SPY': 'SPY'
}

NEON_PALETTE = {
    'CYAN': '#00FFFF',
    'MAGENTA': '#FF00FF',
    'LIME': '#39FF14',
    'ORANGE': '#FF6700',
    'PURPLE': '#BF00FF',
    'WHITE': '#FFFFFF',
    'RED_GLOW': 'rgba(255, 0, 50, 0.6)',
    'GREEN_GLOW': 'rgba(57, 255, 20, 0.6)',
    'BG': '#111111'
}

DATA_FILE = 'macro_data.csv'
OUTPUT_FILE = 'macro_dashboard.html'


# ==========================================
# 1. CLASS: DATA INGESTION
# ==========================================
class DataIngestion:
    """
    Handles reliable data fetching, caching, and sanitization.
    Fixes yfinance MultiIndex instability and enforces strict typing.
    """
    def __init__(self, ticker_map: Dict[str, str], csv_path: str):
        self.tickers = list(ticker_map.values())
        self.rev_map = {v: k for k, v in ticker_map.items()}
        self.csv_path = Path(csv_path)

    def get_data(self) -> pd.DataFrame:
        """
        Orchestrates the check-local vs download-remote logic.
        """
        if self.csv_path.exists():
            print(f"[INFO] Loading cached data from {self.csv_path}...")
            try:
                df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
                # Quick validation to ensure CSV isn't corrupted
                if df.empty or len(df.columns) < len(self.tickers):
                    print("[WARN] Cache corrupted. Re-downloading...")
                    return self._download_and_save()
                return df
            except Exception as e:
                print(f"[ERROR] Failed to read CSV: {e}. Downloading fresh data.")
                return self._download_and_save()
        else:
            print("[INFO] No local cache found. Initiating cold start download...")
            return self._download_and_save()

    def _download_and_save(self) -> pd.DataFrame:
        """
        Downloads data via yfinance, sanitizes it, and saves to disk.
        """
        print(f"[INFO] Downloading tickers: {self.tickers}")
        try:
            # group_by='column' is essential for recent yfinance versions
            raw_df = yf.download(
                self.tickers, 
                period="5y", 
                group_by='column', 
                auto_adjust=False, 
                threads=True
            )
            
            clean_df = self._sanitize_df(raw_df)
            
            # Save to CSV
            clean_df.to_csv(self.csv_path)
            print(f"[SUCCESS] Data saved to {self.csv_path}")
            return clean_df
            
        except Exception as e:
            print(f"[CRITICAL] Download failed: {e}")
            sys.exit(1)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        THE UNIVERSAL FIXER: Handles yfinance MultiIndex weirdness.
        """
        # 1. Handle MultiIndex Columns
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' or 'Adj Close' is in Level 0 (wrong) or Level 1 (standard)
            # We prioritize 'Adj Close', fallback to 'Close'
            
            target_col = 'Adj Close' if 'Adj Close' in df.columns.get_level_values(0) or 'Adj Close' in df.columns.get_level_values(1) else 'Close'
            
            # Detect level of the price type
            if target_col in df.columns.get_level_values(0):
                # If Price Type is Level 0, we don't need to swap, just filter
                df = df[target_col]
            elif target_col in df.columns.get_level_values(1):
                # If Price Type is Level 1, Swap and filter
                df = df.swaplevel(0, 1, axis=1)
                df = df[target_col]
            else:
                # Fallback: Try to grab 'Close' if Adj Close failed
                try:
                    df = df.xs('Close', level=1, axis=1)
                except:
                    df = df.xs('Close', level=0, axis=1)

        # 2. Timezone Removal
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 3. Rename columns using the internal map for clarity
        # Only rename columns that exist in our map
        df.rename(columns=self.rev_map, inplace=True)

        # 4. Coercion and Scalar Safety
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 5. Drop rows where everything is NaN (market holidays)
        df.dropna(how='all', inplace=True)
        
        # 6. Forward fill strictly (hedge fund standard for missing ticks)
        df.ffill(inplace=True)

        return df


# ==========================================
# 2. CLASS: FINANCIAL ANALYSIS
# ==========================================
class FinancialAnalysis:
    """
    Pure math and financial logic engine.
    Implements immutability by working on copies.
    """
    def __init__(self, data: pd.DataFrame):
        self._raw_data = data.copy()

    def get_term_premium_data(self) -> pd.DataFrame:
        """
        Calculates Synthetic Term Premium.
        Logic: (10Y Yield / 10) - (3M Yield / 10)
        """
        df = self._raw_data[['10Y_YIELD', '3M_YIELD']].copy()
        
        # Normalize yields (yfinance returns 40.0 for 4.0%)
        df['10Y_Norm'] = df['10Y_YIELD'] / 10.0
        df['3M_Norm'] = df['3M_YIELD'] / 10.0
        
        df['Term_Premium'] = df['10Y_Norm'] - df['3M_Norm']
        df['Risk_Neutral_Rate'] = df['10Y_Norm'] - df['Term_Premium'] # Mathematical identity, used for visualization
        
        return df.dropna()

    def get_asset_correlations(self) -> Dict[str, pd.DataFrame]:
        """
        Calculates rolling correlations: DXY vs [Gold, Brent, Copper].
        Windows: 21, 63, 252.
        """
        windows = [21, 63, 252]
        assets = ['GOLD', 'BRENT', 'COPPER']
        results = {}

        # Ensure we have DXY
        if 'DXY' not in self._raw_data.columns:
            return {}

        dxy_ret = self._raw_data['DXY'].pct_change()

        for window in windows:
            corr_df = pd.DataFrame(index=self._raw_data.index)
            for asset in assets:
                if asset in self._raw_data.columns:
                    asset_ret = self._raw_data[asset].pct_change()
                    corr_df[asset] = asset_ret.rolling(window=window).corr(dxy_ret)
            results[f'{window}D'] = corr_df.dropna()
            
        return results

    def get_regime_pain_gauge(self) -> pd.DataFrame:
        """
        Regime: Rolling 63-day correlation between SPY Returns and 10Y Yield CHANGES.
        """
        df = self._raw_data[['SPY', '10Y_YIELD']].copy()
        
        df['SPY_Ret'] = df['SPY'].pct_change()
        # Yield change in basis points style (absolute change)
        df['Yield_Chg'] = df['10Y_YIELD'].diff() 
        
        df['Stock_Bond_Corr'] = df['SPY_Ret'].rolling(window=63).corr(df['Yield_Chg'])
        
        return df[['Stock_Bond_Corr']].dropna()

    def get_normalized_performance(self) -> pd.DataFrame:
        """
        Normalize all assets to 100 at start of series.
        """
        cols = ['DXY', 'GOLD', 'BRENT', 'COPPER']
        df = self._raw_data[cols].copy().dropna()
        # Normalize to 100
        return (df / df.iloc[0]) * 100


# ==========================================
# 3. CLASS: DASHBOARD RENDERER
# ==========================================
class DashboardRenderer:
    """
    Generates the offline HTML file with embedded JS and Plotly charts.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.layout_template = "plotly_dark"

    def generate_dashboard(self, 
                           perf_df: pd.DataFrame, 
                           corr_dict: Dict[str, pd.DataFrame],
                           tp_df: pd.DataFrame,
                           regime_df: pd.DataFrame):
        
        # 1. Generate Figures
        fig_perf = self._plot_performance(perf_df)
        fig_corr = self._plot_correlations(corr_dict)
        fig_tp = self._plot_term_premium(tp_df)
        fig_regime = self._plot_pain_gauge(regime_df)

        # 2. Get Raw HTML divs
        div_perf = fig_perf.to_html(full_html=False, include_plotlyjs=False)
        div_corr = fig_corr.to_html(full_html=False, include_plotlyjs=False)
        div_tp = fig_tp.to_html(full_html=False, include_plotlyjs=False)
        div_regime = fig_regime.to_html(full_html=False, include_plotlyjs=False)

        # 3. Assemble Full HTML
        self._write_html(div_perf, div_corr, div_tp, div_regime)

    def _plot_performance(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        colors = [NEON_PALETTE['CYAN'], NEON_PALETTE['MAGENTA'], NEON_PALETTE['ORANGE'], NEON_PALETTE['LIME']]
        
        for col, color in zip(df.columns, colors):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=2)))
            
        fig.update_layout(
            title="Relative Performance (Rebased to 100)",
            template=self.layout_template,
            height=600,
            hovermode="x unified"
        )
        return fig

    def _plot_correlations(self, corr_dict: Dict[str, pd.DataFrame]) -> go.Figure:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("21D Corr vs DXY", "63D Corr vs DXY", "252D Corr vs DXY"))
        
        row_map = {'21D': 1, '63D': 2, '252D': 3}
        
        for key, df in corr_dict.items():
            row = row_map[key]
            for col in df.columns:
                # Add Lines
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=f"{col} ({key})",
                    line=dict(width=1), showlegend=(row==1)
                ), row=row, col=1)
                
            # Add Horizontal Zero Line
            fig.add_hline(y=0, line_dash="dash", line_color="white", row=row, col=1)

            # Area Fills logic (Global Risk On/Off proxy using Mean Correlation)
            # Just visualizing the Average correlation of the group to show regime
            avg_corr = df.mean(axis=1)
            
            # Positive Fill (Green - Risk Off/Sync)
            fig.add_trace(go.Scatter(
                x=avg_corr.index, y=avg_corr.clip(lower=0),
                fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

            # Negative Fill (Red - Normal/Inverse)
            fig.add_trace(go.Scatter(
                x=avg_corr.index, y=avg_corr.clip(upper=0),
                fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

        fig.update_layout(template=self.layout_template, height=800, title="Correlation Structure (Assets vs DXY)")
        return fig

    def _plot_term_premium(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        # 10Y Yield (White)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['10Y_Norm'], name="10Y Yield",
            line=dict(color='white', width=2)
        ))
        
        # Risk Neutral (Implied)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['3M_Norm'], name="3M Yield (Proxy Risk Neutral)",
            line=dict(color=NEON_PALETTE['CYAN'], width=1, dash='dot')
        ))

        # Fill representing Term Premium
        # We cheat slightly for visual stack: Plot 10Y, then fill to 3M
        fig.add_trace(go.Scatter(
            x=df.index, y=df['10Y_Norm'],
            fill=None, mode='lines', line=dict(width=0), showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['3M_Norm'],
            fill='tonexty', # Fills to previous trace
            fillcolor='rgba(191, 0, 255, 0.3)', # Neon Purple Transparent
            mode='lines', line=dict(width=0), name="Term Premium Capture"
        ))

        fig.update_layout(title="Term Premium Decomposition (Purple Area = Premium)", template=self.layout_template, height=600)
        return fig

    def _plot_pain_gauge(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        data = df['Stock_Bond_Corr']
        
        # Plot the Line
        fig.add_trace(go.Scatter(
            x=data.index, y=data, name="Stock/Rate Corr (63D)",
            line=dict(color='white', width=2)
        ))
        
        # Positive Correlation Fill (RED - Bonds Fail)
        fig.add_trace(go.Scatter(
            x=data.index, y=data.clip(lower=0),
            fill='tozeroy',
            fillcolor=NEON_PALETTE['RED_GLOW'],
            line=dict(width=0), name="Inflation Risk (Pos Corr)"
        ))
        
        # Negative Correlation Fill (GREEN - Bonds Hedge)
        fig.add_trace(go.Scatter(
            x=data.index, y=data.clip(upper=0),
            fill='tozeroy',
            fillcolor=NEON_PALETTE['GREEN_GLOW'],
            line=dict(width=0), name="Growth Risk (Neg Corr)"
        ))

        fig.add_hline(y=0, line_color='white', line_width=1)
        
        fig.update_layout(
            title="The Pain Gauge: Equity vs Rates Correlation",
            template=self.layout_template, 
            height=600,
            yaxis=dict(title="Correlation")
        )
        return fig

    def _write_html(self, div_1, div_2, div_3, div_4):
        # Retrieve strict offline JS library (~3MB)
        plotly_js = get_plotlyjs()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Macro FX Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ background-color: #111; color: #eee; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
                h1 {{ color: {NEON_PALETTE['CYAN']}; text-transform: uppercase; letter-spacing: 2px; border-bottom: 2px solid {NEON_PALETTE['MAGENTA']}; display: inline-block; }}
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #222; margin-top: 20px; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888; font-weight: bold; }}
                .tab button:hover {{ background-color: #333; color: {NEON_PALETTE['LIME']}; }}
                .tab button.active {{ background-color: #333; color: {NEON_PALETTE['CYAN']}; border-bottom: 2px solid {NEON_PALETTE['CYAN']}; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h1>Macro Term Premium Dashboard</h1>
            
            <div class="tab">
              <button class="tablinks" onclick="openTab(event, 'Perf')" id="defaultOpen">Relative Perf</button>
              <button class="tablinks" onclick="openTab(event, 'Corr')">Correlations</button>
              <button class="tablinks" onclick="openTab(event, 'TP')">Term Premium</button>
              <button class="tablinks" onclick="openTab(event, 'Pain')">Pain Gauge</button>
            </div>

            <div id="Perf" class="tabcontent">{div_1}</div>
            <div id="Corr" class="tabcontent">{div_2}</div>
            <div id="TP" class="tabcontent">{div_3}</div>
            <div id="Pain" class="tabcontent">{div_4}</div>

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
                
                // CRITICAL FIX: Force Plotly resize on tab switch
                window.dispatchEvent(new Event('resize'));
            }}
            
            // Open default tab
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        with open(self.filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[SUCCESS] Dashboard generated at: {os.path.abspath(self.filename)}")


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print("Initializing Hedge Fund Grade Dashboard Engine...")
    
    # 1. Ingest
    ingestor = DataIngestion(TICKER_MAP, DATA_FILE)
    df = ingestor.get_data()
    
    # 2. Analyze
    analyzer = FinancialAnalysis(df)
    
    perf_df = analyzer.get_normalized_performance()
    corr_dict = analyzer.get_asset_correlations()
    tp_df = analyzer.get_term_premium_data()
    regime_df = analyzer.get_regime_pain_gauge()
    
    # 3. Render
    renderer = DashboardRenderer(OUTPUT_FILE)
    renderer.generate_dashboard(perf_df, corr_dict, tp_df, regime_df)
    
    # 4. Launch
    try:
        webbrowser.open('file://' + os.path.realpath(OUTPUT_FILE))
    except:
        print("Could not auto-open browser. Please open macro_dashboard.html manually.")

if __name__ == "__main__":
    main()
