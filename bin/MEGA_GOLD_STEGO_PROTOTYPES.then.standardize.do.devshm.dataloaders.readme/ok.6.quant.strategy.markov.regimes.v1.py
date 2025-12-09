import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ==========================================
# CLASS A: DataIngestion (The Data Layer)
# ==========================================
class DataIngestion:
    def __init__(self, tickers: List[str], data_dir: str = 'data'):
        self.tickers = tickers
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _generate_synthetic_data(self, ticker: str) -> pd.DataFrame:
        """Generates 252 days of geometric brownian motion for failover."""
        print(f"[{ticker}] API Failed. Generating synthetic shadow data.")
        np.random.seed(42)
        days = 252
        dt = 1/252
        mu = 0.05
        sigma = 0.15
        
        # S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        returns = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt, 
                                 scale=sigma * np.sqrt(dt), 
                                 size=days)
        price_path = 100 * np.cumprod(np.exp(returns))
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        df = pd.DataFrame(index=dates)
        df['Close'] = price_path
        df['Open'] = price_path # Simplify
        df['High'] = price_path * 1.01
        df['Low'] = price_path * 0.99
        df['Volume'] = 1000000
        return df

    def _fetch_single_ticker(self, ticker: str) -> pd.DataFrame:
        """Fetches data with local caching and specific multi-index handling."""
        csv_path = os.path.join(self.data_dir, f"{ticker}.csv")
        
        # 1. Check Cache
        if os.path.exists(csv_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path))
            if datetime.now() - mod_time < timedelta(hours=24):
                try:
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    print(f"[{ticker}] Loaded from cache.")
                    return df
                except Exception:
                    pass # Fall through to download if cache corrupt

        # 2. Download via yfinance
        try:
            # group_by='column' usually returns (Ticker, Attribute) or just Attribute if single
            # We force it to return consistent structure usually
            df = yf.download(ticker, period="2y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
            
            # 3. The "Swap Levels" Fix
            # If we get a MultiIndex with Ticker at Level 0, we want to normalize it
            # However, yf.download with group_by='ticker' for a SINGLE ticker often returns simple columns.
            # Let's handle the case where it returns a MultiIndex (common in recent versions or list downloads)
            if isinstance(df.columns, pd.MultiIndex):
                # If Ticker is level 0, we drop it to flatten
                df.columns = df.columns.droplevel(0)
            
            if df.empty:
                raise ValueError("Empty Dataframe")

            # 4. Sanitization
            df.index = pd.to_datetime(df.index).tz_convert(None)
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Intersection to handle cases where 'Adj Close' might be present/absent
            available_cols = [c for c in cols if c in df.columns]
            df = df[available_cols]
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(inplace=True)
            
            # Save to cache
            df.to_csv(csv_path)
            print(f"[{ticker}] Downloaded and cached.")
            return df

        except Exception as e:
            print(f"[{ticker}] Error: {e}")
            return self._generate_synthetic_data(ticker)

    def get_data_map(self) -> Dict[str, pd.DataFrame]:
        """Main entry point returning dict of ticker -> dataframe."""
        data_map = {}
        for t in self.tickers:
            data_map[t] = self._fetch_single_ticker(t)
        return data_map


# ==========================================
# CLASS B: QuantEngine (The Math Layer)
# ==========================================
class QuantEngine:
    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        self._data_map = data_map

    def _calculate_streaks(self, returns: pd.Series) -> pd.Series:
        """
        Calculates directional count.
        +1, +2, +3 for consecutive up days.
        -1, -2, -3 for consecutive down days.
        0 for neutral.
        """
        # Create a sign series (-1, 0, 1)
        signs = np.sign(returns)
        
        # Identify where sign changes (True if changed, False if same)
        # We assume 0 is a break in streak for simplicity
        change = (signs != signs.shift(1)) | (signs == 0)
        
        # Group cumulative sum identifies distinct "runs"
        run_ids = change.cumsum()
        
        # Count cumulative sequence per group
        # cumcount starts at 0, so we add 1
        streaks = returns.groupby(run_ids).cumcount() + 1
        
        # Apply the sign back to the count
        streaks = streaks * signs
        return streaks.fillna(0).astype(int)

    def _calculate_magnitudes(self, returns: pd.Series) -> pd.Series:
        """
        Calculates magnitude state.
        State = int(abs(return) * 100)
        e.g., -1.5% -> 1
        """
        return (returns.abs() * 100).astype(int)

    def _get_transition_matrix(self, series: pd.Series, name: str) -> pd.DataFrame:
        """
        Computes Markov Transition Matrix P(Xt+1 | Xt).
        """
        df = pd.DataFrame({'Current': series, 'Next': series.shift(-1)})
        df.dropna(inplace=True)
        
        # Crosstab counts transitions
        ct = pd.crosstab(df['Current'], df['Next'])
        
        # Normalize by row (probabilities)
        # Add slight epsilon to avoid div by zero if a state has no exit (rare)
        tm = ct.div(ct.sum(axis=1), axis=0)
        return tm

    def run_analysis(self) -> Dict:
        """
        Computes stats for all tickers.
        """
        results = {}
        
        for ticker, df in self._data_map.items():
            if df.empty or len(df) < 10:
                continue
                
            # Copy to avoid inplace mods
            d = df.copy()
            d['Return'] = d['Close'].pct_change()
            d.dropna(inplace=True)
            
            # Feature 1: Streaks
            d['Streak'] = self._calculate_streaks(d['Return'])
            
            # Feature 2: Magnitudes
            d['Magnitude'] = self._calculate_magnitudes(d['Return'])
            d['PrevReturn'] = d['Return'].shift(1)
            
            # Transition Matrices
            streak_tm = self._get_transition_matrix(d['Streak'], 'Streak')
            mag_tm = self._get_transition_matrix(d['Magnitude'], 'Magnitude')
            
            results[ticker] = {
                'data': d,
                'streak_tm': streak_tm,
                'mag_tm': mag_tm
            }
            
        return results


# ==========================================
# CLASS C: DashboardRenderer (The Viz Layer)
# ==========================================
class DashboardRenderer:
    def __init__(self, results: Dict):
        self.results = results
        # Use offline method to get JS
        self.plotly_js = py_offline.get_plotlyjs()

    def _create_streak_heatmap(self, tm: pd.DataFrame, ticker: str):
        # Sort index and columns for logical flow (-3, -2, -1, 1, 2, 3)
        idx = sorted(tm.index)
        cols = sorted(tm.columns)
        tm = tm.reindex(index=idx, columns=cols)
        
        fig = px.imshow(
            tm,
            labels=dict(x="Next Day Streak", y="Current Streak", color="Probability"),
            x=cols,
            y=idx,
            color_continuous_scale="Viridis",
            title=f"{ticker} Directional Streak Transition Probability"
        )
        return fig

    def _create_mag_heatmap(self, tm: pd.DataFrame, ticker: str):
        # Sort for logic
        idx = sorted(tm.index)
        cols = sorted(tm.columns)
        tm = tm.reindex(index=idx, columns=cols)
        
        fig = px.imshow(
            tm,
            labels=dict(x="Next Day Magnitude", y="Current Magnitude", color="Probability"),
            x=cols,
            y=idx,
            color_continuous_scale="Magma",
            title=f"{ticker} Volatility Regime Transition Probability"
        )
        return fig

    def _create_scatter(self, df: pd.DataFrame, ticker: str):
        # Remove NaN for plot
        plot_df = df.dropna(subset=['Return', 'PrevReturn', 'Streak'])
        
        fig = px.scatter(
            plot_df,
            x='PrevReturn',
            y='Return',
            color='Streak',
            color_continuous_scale="RdBu", # Red for neg streaks, Blue for pos
            title=f"{ticker} Return Autocorrelation by Streak Regime",
            hover_data=['Magnitude', 'Close']
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        return fig

    def generate_html(self, filename: str = "market_regime_dashboard.html"):
        
        html_content = [f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hedge Fund Grade Market Dashboard</title>
            <script type="text/javascript">{self.plotly_js}</script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; }}
                .container {{ padding: 20px; }}
                h1 {{ color: #00d4ff; text-align: center; }}
                
                /* Tab Styles */
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #2d2d2d; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-weight: bold; }}
                .tab button:hover {{ background-color: #444; }}
                .tab button.active {{ background-color: #00d4ff; color: #121212; }}
                
                /* Tab Content */
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
                
                .chart-row {{ display: flex; flex-wrap: wrap; justify-content: space-between; margin-bottom: 20px; }}
                .chart-container {{ width: 32%; min-width: 400px; background: #252526; padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); border-radius: 5px; }}
                
                @media screen and (max-width: 1200px) {{
                    .chart-container {{ width: 100%; margin-bottom: 20px; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quant Strategy: Markov Regimes & Streaks</h1>
                
                <div class="tab">
        """]

        tickers = list(self.results.keys())
        
        # 1. Generate Tabs Header
        for i, ticker in enumerate(tickers):
            active = "active" if i == 0 else ""
            html_content.append(f'<button class="tablinks {active}" onclick="openCity(event, \'{ticker}\')">{ticker}</button>')
        
        html_content.append("</div>") # End tab header

        # 2. Generate Tab Content
        for i, ticker in enumerate(tickers):
            display = "block" if i == 0 else "none"
            res = self.results[ticker]
            
            # Generate Plots
            fig1 = self._create_streak_heatmap(res['streak_tm'], ticker)
            fig2 = self._create_mag_heatmap(res['mag_tm'], ticker)
            fig3 = self._create_scatter(res['data'], ticker)
            
            # Update Layouts for dark theme
            for fig in [fig1, fig2, fig3]:
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=50, b=40)
                )

            # Get Divs (remove JS include since we have it in head)
            div1 = py_offline.plot(fig1, include_plotlyjs=False, output_type='div')
            div2 = py_offline.plot(fig2, include_plotlyjs=False, output_type='div')
            div3 = py_offline.plot(fig3, include_plotlyjs=False, output_type='div')

            html_content.append(f"""
            <div id="{ticker}" class="tabcontent" style="display: {display};">
                <h2 style="border-bottom: 1px solid #444; padding-bottom: 10px;">{ticker} Analysis</h2>
                <div class="chart-row">
                    <div class="chart-container">{div1}</div>
                    <div class="chart-container">{div2}</div>
                    <div class="chart-container">{div3}</div>
                </div>
            </div>
            """)

        # 3. Footer Script (Tab Logic + Resize Fix)
        html_content.append("""
            <script>
            function openCity(evt, cityName) {
                var i, tabcontent, tablinks;
                
                // Hide all tabs
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                
                // Deactivate buttons
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                
                // Show current tab
                document.getElementById(cityName).style.display = "block";
                evt.currentTarget.className += " active";
                
                // TRIGGER RESIZE: Critical for Plotly in hidden tabs
                window.dispatchEvent(new Event('resize'));
                
                // Explicitly find plotly graphs in this tab and resize
                var plotlyDivs = document.getElementById(cityName).querySelectorAll('.plotly-graph-div');
                for(var j=0; j<plotlyDivs.length; j++) {
                    Plotly.Plots.resize(plotlyDivs[j]);
                }
            }
            </script>
        </body>
        </html>
        """)
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write("".join(html_content))
        
        print(f"Dashboard generated successfully: {os.path.abspath(filename)}")


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    TICKERS = ['GLD', 'SPY', 'QQQ', 'IWM', 'TLT', 'USO', 'DIA']
    
    print("--- Starting Hedge Fund Dashboard Engine ---")
    
    # 1. Ingest
    ingestor = DataIngestion(TICKERS)
    data_map = ingestor.get_data_map()
    
    # 2. Compute
    engine = QuantEngine(data_map)
    results = engine.run_analysis()
    
    # 3. Render
    renderer = DashboardRenderer(results)
    renderer.generate_html()
    
    print("--- Process Complete ---")
