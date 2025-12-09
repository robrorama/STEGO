import sys
import argparse
import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots
from scipy.stats import norm

# ------------------------------------------------------------------------------
# 1. DataIngestion Class
# Responsibility: Downloading, cleaning, caching, and the "Swap Level" fix.
# ------------------------------------------------------------------------------
class DataIngestion:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.filename = f"gex_history_{self.ticker}.csv"

    def fetch_data(self) -> pd.DataFrame:
        """
        Main entry point.
        Offline First: Checks for local CSV. If missing/stale, downloads new data.
        """
        if os.path.exists(self.filename):
            print(f"[DataIngestion] Local cache found: {self.filename}")
            try:
                df = pd.read_csv(self.filename, index_col=0, parse_dates=True)
                # Basic staleness check (if last date is not today/yesterday)
                if df.index[-1].date() < (datetime.date.today() - datetime.timedelta(days=2)):
                    print("[DataIngestion] Cache stale. Refreshing...")
                    return self._refresh_data()
                return df
            except Exception as e:
                print(f"[DataIngestion] Error reading cache: {e}. Redownloading.")
                return self._refresh_data()
        else:
            print("[DataIngestion] No local cache. Downloading fresh data.")
            return self._refresh_data()

    def _refresh_data(self) -> pd.DataFrame:
        df = self._download_data()
        df = self._sanitize_df(df)
        df = self._backfill_shadow_history(df)
        
        # Save to CSV for offline usage
        df.to_csv(self.filename)
        return df

    def _download_data(self) -> pd.DataFrame:
        """
        Forces group_by='column' to mitigate multi-index ambiguity.
        Downloads 1 year of data.
        """
        print(f"[DataIngestion] Calling yfinance for {self.ticker}...")
        try:
            # force group_by='column' per requirements
            df = yf.download(self.ticker, period="1y", group_by='column', auto_adjust=False, progress=False)
            if df.empty:
                print(f"[Error] yfinance returned empty data for {self.ticker}.")
                sys.exit(1)
            return df
        except Exception as e:
            print(f"[Error] yfinance failed: {e}")
            sys.exit(1)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        HARD CONSTRAINT: The 'Swap Levels' Fix & Duplicate Column Prevention.
        """
        # 1. Handle MultiIndex (The Swap Fix)
        if isinstance(df.columns, pd.MultiIndex):
            # Check levels to see if Ticker is in Level 0 (unexpected) or Level 1 (expected)
            level_0_vals = df.columns.get_level_values(0)
            
            # If 'Close' is NOT in Level 0, it means Level 0 is likely the Ticker.
            # We need to swap them so Attributes (Close, Open) are on top.
            if not any(x in level_0_vals for x in ['Close', 'Adj Close', 'Open']):
                 # Double check Level 1
                level_1_vals = df.columns.get_level_values(1)
                if any(x in level_1_vals for x in ['Close', 'Adj Close', 'Open']):
                    print("[DataIngestion] Detected Ticker in Level 0. Applying SWAPLEVEL fix.")
                    df = df.swaplevel(0, 1, axis=1)
            
            # Flatten columns: We just want 'Close', 'Open', etc.
            df.columns = df.columns.get_level_values(0)
            
        # 2. Standardize column names (Title Case)
        df.columns = [c.strip().title() for c in df.columns] 
        
        # 3. CRITICAL FIX: Handle 'Adj Close' vs 'Close' collision
        # yfinance(auto_adjust=False) returns BOTH. We prefer Adj Close.
        # We must DROP the original 'Close' before renaming to avoid duplicates.
        if 'Adj Close' in df.columns:
            if 'Close' in df.columns:
                df.drop(columns=['Close'], inplace=True) # Drop raw close to prevent collision
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            
        # Fallback for underscore variation
        if 'Adj_Close' in df.columns:
            if 'Close' in df.columns:
                df.drop(columns=['Close'], inplace=True)
            df.rename(columns={'Adj_Close': 'Close'}, inplace=True)

        return df

    def _backfill_shadow_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        HARD CONSTRAINT: Cold Start / Shadow GEX.
        Formula Proxy: (Neutral_Vol - Realized_Vol) * Notional_Liquidity
        """
        df = df.copy()
        
        # Ensure numeric
        # This crashed previously because df['Close'] was a DataFrame (duplicate columns).
        # The _sanitize_df fix ensures it is now a Series.
        cols = ['Close', 'Volume']
        for c in cols:
            if c not in df.columns:
                print(f"[Warning] Column {c} missing. Shadow GEX may be inaccurate.")
                continue
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df.dropna(subset=['Close', 'Volume'], inplace=True)

        # 1. Realized Volatility (Short term, e.g., 5 days)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Realized_Vol'] = df['Log_Ret'].rolling(window=5).std() * np.sqrt(252)
        
        # 2. Neutral Volatility (Long term baseline, e.g., 20 days)
        df['Neutral_Vol'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
        
        # 3. Notional Liquidity
        df['Notional'] = df['Close'] * df['Volume']
        
        # 4. Shadow GEX Calculation
        df['Shadow_GEX'] = (df['Neutral_Vol'] - df['Realized_Vol']) * df['Notional']
        
        # Smoothing for visualization
        df['Shadow_GEX_EMA'] = df['Shadow_GEX'].ewm(span=3).mean()
        
        # Drop NaNs created by rolling windows
        df.dropna(inplace=True)
        
        print(f"[DataIngestion] Generated Shadow GEX history. Rows: {len(df)}")
        return df

# ------------------------------------------------------------------------------
# 2. FinancialAnalysis Class
# Responsibility: Pure math, Black-Scholes, Immutability.
# ------------------------------------------------------------------------------
class FinancialAnalysis:
    def __init__(self, raw_df: pd.DataFrame, risk_free_rate: float = 0.04):
        # IMMUTABILITY: Copy-on-Write pattern
        self._raw_data = raw_df.copy()
        self.r = risk_free_rate

    def get_price_history(self):
        return self._raw_data.copy()

    def calculate_live_gex_profile(self, ticker_symbol: str):
        """
        Fetches CURRENT option chain to build a strike-based GEX profile.
        This provides the 'Zero Gamma' level.
        """
        print(f"[FinancialAnalysis] Fetching live option chain for {ticker_symbol}...")
        tk = yf.Ticker(ticker_symbol)
        
        try:
            exps = tk.options
            if not exps:
                print("[FinancialAnalysis] No expirations found.")
                return None, None
            
            # Fetch nearest monthly expiration (heuristic for liquidity)
            # For robustness, we just take the next 2 expirations to save time
            target_exps = exps[:2] 
            
            calls_list = []
            puts_list = []
            
            if 'Close' not in self._raw_data.columns:
                 return None, None

            current_price = self._raw_data['Close'].iloc[-1]
            
            for e in target_exps:
                opt = tk.option_chain(e)
                
                # Process Calls
                c = opt.calls.copy()
                c['type'] = 'call'
                c['expiration'] = e
                calls_list.append(c)
                
                # Process Puts
                p = opt.puts.copy()
                p['type'] = 'put'
                p['expiration'] = e
                puts_list.append(p)
            
            if not calls_list and not puts_list:
                return None, None

            # Replace append with concat (Stability Requirement)
            options_df = pd.concat(calls_list + puts_list, ignore_index=True)
            
            # Clean data
            options_df['impliedVolatility'] = pd.to_numeric(options_df['impliedVolatility'], errors='coerce')
            options_df['strike'] = pd.to_numeric(options_df['strike'], errors='coerce')
            options_df['openInterest'] = pd.to_numeric(options_df['openInterest'], errors='coerce').fillna(0)
            
            # Calculate Gamma per strike
            # Time to expiry in years
            today = datetime.datetime.now()
            options_df['dte'] = (pd.to_datetime(options_df['expiration']) - today).dt.days / 365.0
            options_df['dte'] = options_df['dte'].clip(lower=0.001) # Avoid div by zero
            
            # Vectorized Black-Scholes Gamma
            S = current_price
            K = options_df['strike']
            sigma = options_df['impliedVolatility']
            T = options_df['dte']
            r = self.r
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            pdf_d1 = norm.pdf(d1)
            options_df['gamma'] = pdf_d1 / (S * sigma * np.sqrt(T))
            
            # GEX Calculation
            options_df['gex'] = options_df['gamma'] * options_df['openInterest'] * 100 * S 
            options_df.loc[options_df['type'] == 'put', 'gex'] *= -1
            
            # Group by Strike
            gex_by_strike = options_df.groupby('strike')['gex'].sum().sort_index()
            
            # Find crossover (Zero Gamma)
            zero_gamma_level = current_price
            try:
                pos_gex = gex_by_strike[gex_by_strike > 0]
                neg_gex = gex_by_strike[gex_by_strike < 0]
                if not pos_gex.empty and not neg_gex.empty:
                    # Find the gap
                    zero_gamma_level = (pos_gex.index[0] + neg_gex.index[-1]) / 2
            except:
                pass

            return gex_by_strike, zero_gamma_level
            
        except Exception as e:
            print(f"[FinancialAnalysis] Live chain fetch failed: {e}")
            return None, None

    def calculate_velocity_stats(self):
        """
        Prepares data for GEX vs Velocity scatter plot.
        """
        df = self._raw_data.copy()
        
        # Calculate Next Day Abs Return
        df['Next_Ret'] = df['Close'].shift(-1).pct_change()
        df['Next_Abs_Move'] = (df['Close'].shift(-1) - df['Close']).abs() / df['Close']
        
        clean_df = df[['Shadow_GEX_EMA', 'Next_Abs_Move']].dropna()
        return clean_df

# ------------------------------------------------------------------------------
# 3. DashboardRenderer Class
# Responsibility: HTML/Plotly generation, JS Injection, Dark Mode.
# ------------------------------------------------------------------------------
class DashboardRenderer:
    def __init__(self, ticker):
        self.ticker = ticker
    
    def render(self, history_df, gex_profile_series, zero_gamma_level, velocity_df):
        print("[DashboardRenderer] Generating standalone HTML...")
        
        # 1. HARD CONSTRAINT: Offline Plotly JS
        plotly_js = py_offline.get_plotlyjs()
        
        # ---------------------------
        # Chart 1: GEX Regime (Time Series)
        # ---------------------------
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Price Line
        fig_ts.add_trace(
            go.Scatter(x=history_df.index, y=history_df['Close'], name='Price', line=dict(color='white', width=1)),
            secondary_y=False
        )
        
        # GEX Shading/Area
        gex_vals = history_df['Shadow_GEX_EMA'].values
        fig_ts.add_trace(
            go.Scatter(
                x=history_df.index, 
                y=gex_vals, 
                name='Est. GEX',
                fill='tozeroy',
                line=dict(width=0),
                marker=dict(color=np.where(gex_vals >= 0, 'rgba(0, 255, 0, 0.2)', 'rgba(255, 0, 0, 0.2)'))
            ),
            secondary_y=True
        )

        fig_ts.update_layout(
            title=f"{self.ticker} Price vs Gamma Exposure Regime",
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2_title="Est. GEX ($)",
            hovermode="x unified"
        )

        # ---------------------------
        # Chart 2: GEX vs Velocity
        # ---------------------------
        fig_scat = go.Figure()
        
        colors = ['#00ff00' if x > 0 else '#ff0000' for x in velocity_df['Shadow_GEX_EMA']]
        
        fig_scat.add_trace(go.Scatter(
            x=velocity_df['Shadow_GEX_EMA'],
            y=velocity_df['Next_Abs_Move'],
            mode='markers',
            marker=dict(color=colors, size=8, opacity=0.7),
            text=[f"GEX: {x:.2f}<br>Move: {y:.2%}" for x, y in zip(velocity_df['Shadow_GEX_EMA'], velocity_df['Next_Abs_Move'])]
        ))
        
        fig_scat.update_layout(
            title="Volatility Thesis Validation: GEX vs Next Day Move",
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            xaxis_title="Gamma Exposure (Shadow)",
            yaxis_title="Next Day Absolute Price Change (%)"
        )

        # ---------------------------
        # Chart 3: Live Strike Profile
        # ---------------------------
        fig_profile = go.Figure()
        if gex_profile_series is not None:
            colors_prof = ['#00ff00' if x > 0 else '#ff0000' for x in gex_profile_series.values]
            fig_profile.add_trace(go.Bar(
                x=gex_profile_series.index,
                y=gex_profile_series.values,
                marker_color=colors_prof,
                name="Gamma"
            ))
            
            if zero_gamma_level:
                fig_profile.add_vline(x=zero_gamma_level, line_width=2, line_dash="dash", line_color="white", annotation_text="Flip Lvl")
            
            fig_profile.update_layout(
                title=f"Live Strike-Level Gamma Profile (Flip: {zero_gamma_level:.2f})",
                template="plotly_dark",
                paper_bgcolor="#1e1e1e",
                plot_bgcolor="#1e1e1e",
                xaxis_title="Strike Price",
                yaxis_title="Gamma Notional"
            )
        else:
            fig_profile.add_annotation(text="Live Option Data Unavailable or Market Closed", showarrow=False)

        # ---------------------------
        # HTML Generation
        # ---------------------------
        div_ts = py_offline.plot(fig_ts, include_plotlyjs=False, output_type='div')
        div_scat = py_offline.plot(fig_scat, include_plotlyjs=False, output_type='div')
        div_prof = py_offline.plot(fig_profile, include_plotlyjs=False, output_type='div')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.ticker} GEX Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
                h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #1e1e1e; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #888; font-weight: bold; }}
                .tab button:hover {{ background-color: #333; color: white; }}
                .tab button.active {{ background-color: #0055ff; color: white; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <h1>GEX Dashboard: {self.ticker}</h1>
            
            <div class="tab">
                <button class="tablinks" onclick="openCity(event, 'Regime')" id="defaultOpen">Regime Time-Series</button>
                <button class="tablinks" onclick="openCity(event, 'Profile')">Live Strike Profile</button>
                <button class="tablinks" onclick="openCity(event, 'Velocity')">Thesis Validation</button>
            </div>

            <div id="Regime" class="tabcontent">
                {div_ts}
            </div>

            <div id="Profile" class="tabcontent">
                {div_prof}
            </div>

            <div id="Velocity" class="tabcontent">
                {div_scat}
            </div>

            <script>
            function openCity(evt, cityName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(cityName).style.display = "block";
                evt.currentTarget.className += " active";
                window.dispatchEvent(new Event('resize'));
            }}
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        filename = f"{self.ticker}_dashboard.html"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[DashboardRenderer] Dashboard saved to: {os.path.abspath(filename)}")

# ------------------------------------------------------------------------------
# Main Execution Flow
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Standalone GEX Dashboard Generator")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk free rate (default: 0.04)")
    args = parser.parse_args()

    # 1. Ingest
    ingestor = DataIngestion(args.ticker)
    df_raw = ingestor.fetch_data()

    # 2. Analyze
    analyzer = FinancialAnalysis(df_raw, args.risk_free_rate)
    
    # Calculate Shadow GEX History
    history_df = analyzer.get_price_history()
    
    # Calculate Live Stats
    gex_profile, zero_gamma = analyzer.calculate_live_gex_profile(ingestor.ticker)
    
    # Calculate Velocity Stats
    velocity_df = analyzer.calculate_velocity_stats()

    # 3. Render
    renderer = DashboardRenderer(ingestor.ticker)
    renderer.render(history_df, gex_profile, zero_gamma, velocity_df)

if __name__ == "__main__":
    main()
