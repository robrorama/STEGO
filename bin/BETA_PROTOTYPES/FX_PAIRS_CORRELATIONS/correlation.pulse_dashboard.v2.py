# SCRIPTNAME: ok.01.correlation.pulse_dashboard.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import linregress
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==========================================
# 1. UTILITIES & DATA INGESTION
# ==========================================
class DataUtils:
    @staticmethod
    def get_data(tickers, start_date='2020-01-01', end_date=None):
        """
        Robust yfinance data pull with retry logic.
        """
        print(f"Fetching data for: {tickers}")
        try:
            # yfinance download
            df = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
            
            # Handle MultiIndex columns (yfinance > 0.2.x structure)
            if isinstance(df.columns, pd.MultiIndex):
                data = pd.DataFrame()
                for t in tickers:
                    try:
                        # Try Adj Close first, fallback to Close
                        if 'Adj Close' in df[t]:
                            data[t] = df[t]['Adj Close']
                        else:
                            data[t] = df[t]['Close']
                    except KeyError:
                        pass
            else:
                # Fallback for single ticker or flat structure
                data = df['Adj Close'] if 'Adj Close' in df else df['Close']
                
            data = data.dropna()
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    @staticmethod
    def calculate_log_returns(series):
        return np.log(series / series.shift(1))

    @staticmethod
    def calculate_real_yield_proxy_change(tip_series):
        """
        TIP ETF price moves inversely to Real Yields.
        Real Yield Change ~ -1 * TIP Log Returns.
        """
        return -1 * DataUtils.calculate_log_returns(tip_series)


# ==========================================
# 2. CORRELATION ENGINE
# ==========================================
class CorrelationEngine:
    def __init__(self, equity_series, yield_change_series):
        self.equity = equity_series
        self.yield_chg = yield_change_series
        self.windows = [30, 60, 90, 180]
        self.results = pd.DataFrame(index=self.equity.index)

    def run(self):
        # 1. Rolling Correlations
        for w in self.windows:
            col_name = f'corr_{w}d'
            self.results[col_name] = self.equity.rolling(window=w).corr(self.yield_chg)
            
            # 2. Z-Score Normalization (Expanding window to simulate real-time)
            mean_roll = self.results[col_name].expanding().mean()
            std_roll = self.results[col_name].expanding().std()
            
            self.results[f'{col_name}_zscore'] = (self.results[col_name] - mean_roll) / std_roll

        # 3. Correlation Pulse (Weighted Average)
        # Weights: Higher weight to faster windows to capture turns
        weights = {30: 0.4, 60: 0.3, 90: 0.2, 180: 0.1}
        self.results['pulse_raw'] = 0.0
        for w, weight in weights.items():
            self.results['pulse_raw'] += self.results[f'corr_{w}d'].fillna(0) * weight
            
        # Z-Score Pulse
        p_mean = self.results['pulse_raw'].expanding().mean()
        p_std = self.results['pulse_raw'].expanding().std()
        self.results['pulse_z'] = (self.results['pulse_raw'] - p_mean) / p_std
        
        return self.results


# ==========================================
# 3. BREAKEVEN ENGINE
# ==========================================
class BreakevenEngine:
    def __init__(self, nominal_yield_series, real_yield_etf_series):
        """
        nominal_yield_series: ^TNX (Index value, e.g. 4.00)
        real_yield_etf_series: TIP (Price, e.g. 107.00)
        """
        self.nominal = nominal_yield_series
        self.tip = real_yield_etf_series
        self.df = pd.DataFrame(index=self.nominal.index)

    def calculate_slope(self, series, window):
        # Rolling linear regression slope
        slopes = [np.nan] * len(series)
        y = series.values
        x = np.arange(window)
        
        for i in range(window, len(y)):
            y_window = y[i-window:i]
            # Simple slope estimate
            slope, _, _, _, _ = linregress(x, y_window)
            slopes[i] = slope
        return pd.Series(slopes, index=series.index)

    def run(self):
        # 1. Construct Synthetic Real Yield Level Proxy
        # Normalize both to start at 100 for slope comparison
        nom_norm = self.nominal / self.nominal.iloc[0] * 100
        # Inverse TIP Price to proxy Real Yield Level
        tip_inv_norm = (1 / self.tip) / (1 / self.tip.iloc[0]) * 100 
        
        # Synthetic Breakeven Spread Proxy (Nominal - Real Proxy)
        self.df['breakeven_proxy'] = nom_norm - tip_inv_norm 
        
        # 2. Slopes (1M = 21 days, 3M = 63 days)
        self.df['be_slope_1m'] = self.calculate_slope(self.df['breakeven_proxy'], 21)
        self.df['be_slope_3m'] = self.calculate_slope(self.df['breakeven_proxy'], 63)
        
        # Pass through raw values for plotting
        self.df['nominal_yield'] = self.nominal
        self.df['tip_price'] = self.tip
        
        return self.df


# ==========================================
# 4. REGIME DETECTOR
# ==========================================
class RegimeDetector:
    def __init__(self, corr_df, be_df, prices):
        self.df = pd.concat([corr_df, be_df], axis=1)
        self.prices = prices # SPY prices for Vol calc

    def run(self):
        # 1. Macro Regime Compression Warning Logic
        # Condition: Negative Corr (stocks down when yields up) + Rising Inflation (BE slope > 0) + Extreme Stress (Z < -0.5)
        
        c1 = self.df['corr_90d'] < 0
        c2 = self.df['be_slope_1m'] > 0
        c3 = self.df['corr_90d_zscore'] < -0.5
        
        self.df['warning_signal'] = (c1 & c2 & c3).astype(int)
        
        # 2. Forward Volatility Analysis
        # Calculate Realized Vol (21d)
        log_ret = np.log(self.prices / self.prices.shift(1))
        rv_21 = log_ret.rolling(21).std() * np.sqrt(252) * 100
        
        self.df['rv_21d'] = rv_21
        self.df['fwd_rv_21d'] = rv_21.shift(-21) # Look ahead 21 days
        
        # Calculate Delta in Vol (Did vol expand after the signal?)
        self.df['vol_expansion'] = self.df['fwd_rv_21d'] - self.df['rv_21d']
        
        # 3. Hit Rate Logic
        self.df['hit'] = np.where((self.df['warning_signal'] == 1) & (self.df['vol_expansion'] > 0), 1, 0)
        
        return self.df


# ==========================================
# 5. PLOT ENGINE
# ==========================================
class PlotEngine:
    def __init__(self, full_df, spy_rets, yield_chgs):
        self.df = full_df.dropna()
        self.spy_rets = spy_rets
        self.yield_chgs = yield_chgs

    def chart_rolling_corr(self):
        fig = go.Figure()
        windows = [30, 60, 90, 180]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for w, c in zip(windows, colors):
            fig.add_trace(go.Scatter(
                x=self.df.index, y=self.df[f'corr_{w}d'],
                mode='lines', name=f'{w}D Corr',
                line=dict(color=c, width=1.5)
            ))
            
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        
        fig.update_layout(
            title="Rolling Equity-Real Yield Correlations",
            template="plotly_dark",
            height=500,
            yaxis_title="Correlation",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def chart_pulse_zscore(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['pulse_z'],
            fill='tozeroy', mode='lines', name='Pulse Z-Score',
            line=dict(color='#00d4ff')
        ))
        
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", annotation_text="Stress Threshold")
        fig.update_layout(title="Correlation Pulse (Z-Score)", template="plotly_dark", height=400)
        return fig

    def chart_breakeven_overlay(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Secondary Y: Correlation
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['corr_90d'],
            name='90D Corr', line=dict(color='yellow', width=1),
            opacity=0.6
        ), secondary_y=True)
        
        # Primary Y: Breakeven Proxy
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['breakeven_proxy'],
            name='Breakeven Proxy Trend', line=dict(color='#9467bd', width=2)
        ), secondary_y=False)

        # Highlight Warning Zones
        warnings = self.df[self.df['warning_signal'] == 1]
        fig.add_trace(go.Scatter(
            x=warnings.index, y=warnings['breakeven_proxy'],
            mode='markers', name='Regime Flip Signal',
            marker=dict(color='red', size=8, symbol='x')
        ))

        fig.update_layout(title="Breakevens vs Correlation Overlay", template="plotly_dark", height=500)
        return fig

    def chart_scatter(self):
        # Color by regime (Pulse Z > 0 vs < 0)
        regime = np.where(self.df['pulse_z'] > 0, 'Positive Regime', 'Negative Regime')
        
        # Align data
        common_idx = self.df.index.intersection(self.spy_rets.index).intersection(self.yield_chgs.index)
        plot_df = pd.DataFrame({
            'SPX_Ret': self.spy_rets.loc[common_idx],
            'Yield_Chg': self.yield_chgs.loc[common_idx],
            'Regime': regime[-len(common_idx):]
        })
        
        fig = px.scatter(
            plot_df, x='Yield_Chg', y='SPX_Ret', color='Regime',
            title="SPX Returns vs Real Yield Changes",
            trendline="ols",
            color_discrete_map={'Positive Regime': '#00cc96', 'Negative Regime': '#ef553b'},
            template="plotly_dark"
        )
        return fig

    def chart_heatmap(self):
        # Resample to monthly average correlation
        monthly = self.df[['corr_30d', 'corr_60d', 'corr_90d', 'corr_180d']].resample('M').mean()
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly.T.values,
            x=monthly.index,
            y=monthly.columns,
            colorscale='RdYlGn',
            zmid=0
        ))
        fig.update_layout(title="Correlation Regime Heatmap", template="plotly_dark", height=400)
        return fig

    def chart_vol_predictor(self):
        signals = self.df[self.df['warning_signal'] == 1].copy()
        
        if signals.empty:
            fig = go.Figure()
            fig.update_layout(title="No Signals Detected", template="plotly_dark")
            return fig
            
        # FIX: Explicitly define specs for Indicator trace type
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Forward Vol Expansion", "Signal Hit Rate"),
            specs=[[{"type": "xy"}, {"type": "domain"}]]
        )
        
        # 1. Bar of Vol Change
        fig.add_trace(go.Bar(
            x=signals.index, y=signals['vol_expansion'],
            name='Vol Expansion (21d)', marker_color=np.where(signals['vol_expansion']>0, 'green', 'red')
        ), row=1, col=1)
        
        # 2. Hit Rate (Gauge/Indicator)
        hit_rate = signals['hit'].mean()
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = hit_rate * 100,
            title = {'text': "Hit Rate %"},
            gauge = {'axis': {'range': [None, 100]}}
        ), row=1, col=2)
        
        fig.update_layout(title="Volatility Expansion Predictor", template="plotly_dark", height=400)
        return fig

    def chart_term_structure(self):
        latest = self.df.iloc[-1]
        windows = ['30d', '60d', '90d', '180d']
        vals = [latest[f'corr_{w}'] for w in windows]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=vals,
            theta=windows,
            fill='toself',
            name='Current Term Structure'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
            title="Current Correlation Term Structure",
            template="plotly_dark"
        )
        return fig

    def get_latest_stats(self):
        latest = self.df.iloc[-1]
        return {
            'Date': latest.name.strftime('%Y-%m-%d'),
            'Pulse_Z': round(latest['pulse_z'], 2),
            'Corr_90d': round(latest['corr_90d'], 2),
            'Warning': "ACTIVE" if latest['warning_signal'] == 1 else "None"
        }


# ==========================================
# 6. DASHBOARD GENERATOR
# ==========================================
class DashboardGenerator:
    def __init__(self, charts_dict, stats):
        self.charts = charts_dict
        self.stats = stats

    def generate_html(self, filename="correlation_pulse_dashboard.html"):
        divs = {k: pio.to_html(v, full_html=False, include_plotlyjs='cdn') for k, v in self.charts.items()}
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quant Correlation Pulse</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background-color: #111; color: #eee; margin: 0; }}
                .container {{ width: 95%; margin: auto; padding: 20px; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 10px; }}
                .stat-box {{ background: #222; padding: 10px 20px; border-radius: 5px; text-align: center; }}
                .stat-val {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
                .stat-label {{ font-size: 12px; color: #888; }}
                
                /* Tabs */
                .tab {{ overflow: hidden; border-bottom: 1px solid #444; margin-top: 20px; }}
                .tab button {{ background-color: #222; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; }}
                .tab button:hover {{ background-color: #333; }}
                .tab button.active {{ background-color: #00d4ff; color: #000; font-weight: bold; }}
                .tabcontent {{ display: none; padding: 20px; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Yield-Equity Correlation Pulse Engine</h1>
                    <div style="display:flex; gap:15px;">
                        <div class="stat-box">
                            <div class="stat-val">{self.stats['Date']}</div>
                            <div class="stat-label">Latest Date</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val">{self.stats['Pulse_Z']}</div>
                            <div class="stat-label">Pulse Z-Score</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-val">{self.stats['Corr_90d']}</div>
                            <div class="stat-label">90D Corr</div>
                        </div>
                         <div class="stat-box">
                            <div class="stat-val" style="color: {'red' if self.stats['Warning'] == 'ACTIVE' else 'green'}">{self.stats['Warning']}</div>
                            <div class="stat-label">Warning Signal</div>
                        </div>
                    </div>
                </div>

                <div class="tab">
                  <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Overview</button>
                  <button class="tablinks" onclick="openTab(event, 'Pulse')">Pulse Z-Score</button>
                  <button class="tablinks" onclick="openTab(event, 'Breakevens')">Breakevens</button>
                  <button class="tablinks" onclick="openTab(event, 'Scatter')">Scatter Analysis</button>
                  <button class="tablinks" onclick="openTab(event, 'Heatmap')">Heatmap</button>
                  <button class="tablinks" onclick="openTab(event, 'Vol')">Vol Predictor</button>
                  <button class="tablinks" onclick="openTab(event, 'Term')">Term Structure</button>
                </div>

                <div id="Overview" class="tabcontent">{divs['rolling_corr']}</div>
                <div id="Pulse" class="tabcontent">{divs['pulse_z']}</div>
                <div id="Breakevens" class="tabcontent">{divs['breakeven']}</div>
                <div id="Scatter" class="tabcontent">{divs['scatter']}</div>
                <div id="Heatmap" class="tabcontent">{divs['heatmap']}</div>
                <div id="Vol" class="tabcontent">{divs['vol']}</div>
                <div id="Term" class="tabcontent">{divs['term']}</div>
            </div>

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
                
                // FORCE RESIZE FOR PLOTLY
                window.dispatchEvent(new Event('resize'));
            }}
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Dashboard generated: {filename}")


# ==========================================
# 7. MAIN ORCHESTRATOR
# ==========================================
def main():
    print(">>> Starting Rolling Yield-Equity Correlation Pulse Engine...")
    
    # 1. Data Ingestion
    tickers = ['SPY', '^TNX', 'TIP']
    raw_data = DataUtils.get_data(tickers, start_date='2010-01-01')
    
    if raw_data.empty:
        print("Failed to download data.")
        return

    # 2. Preprocessing
    spy_rets = DataUtils.calculate_log_returns(raw_data['SPY'])
    # Real Yield Change Proxy: Inverse TIP log returns
    real_yield_chg = DataUtils.calculate_real_yield_proxy_change(raw_data['TIP'])
    
    # Align Data (Inner Join)
    common_index = spy_rets.index.intersection(real_yield_chg.index).intersection(raw_data['^TNX'].index)
    spy_rets = spy_rets.loc[common_index]
    real_yield_chg = real_yield_chg.loc[common_index]
    nominal_yield_level = raw_data['^TNX'].loc[common_index]
    tip_price = raw_data['TIP'].loc[common_index]

    # 3. Analytics Engines
    print(">>> Running Correlation Engine...")
    corr_engine = CorrelationEngine(spy_rets, real_yield_chg)
    corr_results = corr_engine.run()
    
    print(">>> Running Breakeven Engine...")
    be_engine = BreakevenEngine(nominal_yield_level, tip_price)
    be_results = be_engine.run()
    
    print(">>> Detecting Macro Regimes...")
    regime_engine = RegimeDetector(corr_results, be_results, raw_data['SPY'].loc[common_index])
    final_df = regime_engine.run()
    
    # 4. Visualization
    print(">>> Generating Visualizations...")
    plotter = PlotEngine(final_df, spy_rets, real_yield_chg)
    
    charts = {
        'rolling_corr': plotter.chart_rolling_corr(),
        'pulse_z': plotter.chart_pulse_zscore(),
        'breakeven': plotter.chart_breakeven_overlay(),
        'scatter': plotter.chart_scatter(),
        'heatmap': plotter.chart_heatmap(),
        'vol': plotter.chart_vol_predictor(),
        'term': plotter.chart_term_structure()
    }
    
    # 5. Dashboard Assembly
    stats = plotter.get_latest_stats()
    dash = DashboardGenerator(charts, stats)
    dash.generate_html()
    
    print(">>> Process Complete.")

if __name__ == "__main__":
    main()
