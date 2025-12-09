# SCRIPTNAME: 07.visulize.l2.dept.and.tradeability.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.special import expit  # Logistic function
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
CONFIG = {
    'dirs': {
        'l2': 'project/data/l2',
        'prints': 'project/data/prints',
        'quotes': 'project/data/quotes',
        'analytics': 'project/analytics/metrics',
        'events': 'project/analytics/events',
        'output': 'project/output'
    },
    'params': {
        'depth_levels': 5,
        'slippage_window': 20,
        'sweep_time_ms': 100,
        'sweep_level_cross': 3,
        'weights': { # Tradeability weights
            'obi': 1.0, 'spread': 1.0, 'slippage': 1.0, 
            'micro': 1.0, 'sweep': 1.0, 'iceberg': 1.0
        }
    }
}

# ==========================================
# CLASS 1: DATA INGESTION (Disk-First)
# ==========================================
class DataIngestion:
    def __init__(self, config):
        self.config = config
        self._initialize_directories()

    def _initialize_directories(self):
        """Creates the required directory structure if missing."""
        for path in self.config['dirs'].values():
            os.makedirs(path, exist_ok=True)

    def get_data(self):
        """
        Disk-first loading strategy:
        1. Check disk for Parquet.
        2. If missing, generate synthetic hedge-fund grade data.
        3. Save to disk.
        4. Return DataFrame.
        """
        l2_path = os.path.join(self.config['dirs']['l2'], 'l2_snapshot.parquet')
        trades_path = os.path.join(self.config['dirs']['prints'], 'trades.parquet')

        if not os.path.exists(l2_path) or not os.path.exists(trades_path):
            print("(!) Data not found on disk. Generating synthetic microstructure data...")
            self._generate_synthetic_data(l2_path, trades_path)
        
        print(f"Loading data from disk...")
        df_l2 = pd.read_parquet(l2_path)
        df_trades = pd.read_parquet(trades_path)
        
        # Merge streams via simple forward fill for this MVP (Timestamp alignment)
        # In prod, utilize pd.merge_asof for high-precision alignment
        df = pd.merge_asof(
            df_l2.sort_values('timestamp'),
            df_trades.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        return df, df_trades

    def _generate_synthetic_data(self, l2_path, trades_path):
        """Generates realistic L2 and Trade data for testing."""
        periods = 5000
        base_price = 4500.0
        
        # Time index
        times = [datetime.now() + timedelta(milliseconds=x*100) for x in range(periods)]
        
        # Random Walk for Midprice
        np.random.seed(42)
        noise = np.random.normal(0, 0.5, periods).cumsum()
        mid = base_price + noise

        data = {'timestamp': times}
        
        # Generate L2 Data (5 Levels)
        for i in range(5):
            spread_widener = (i + 1) * 0.25
            data[f'bid_px_{i}'] = mid - spread_widener - np.random.rand(periods)*0.1
            data[f'ask_px_{i}'] = mid + spread_widener + np.random.rand(periods)*0.1
            data[f'bid_sz_{i}'] = np.random.randint(1, 100, periods)
            data[f'ask_sz_{i}'] = np.random.randint(1, 100, periods)

        df_l2 = pd.DataFrame(data)
        df_l2.to_parquet(l2_path)

        # Generate Trades
        trade_times = [datetime.now() + timedelta(milliseconds=x*300) for x in range(int(periods/3))]
        df_trades = pd.DataFrame({
            'timestamp': trade_times,
            'price': mid[::3] + np.random.normal(0, 0.1, len(trade_times)),
            'size': np.random.randint(1, 50, len(trade_times)),
            'aggressor_side': np.random.choice(['buy', 'sell'], len(trade_times))
        })
        df_trades.to_parquet(trades_path)

# ==========================================
# CLASS 2: MICROSTRUCTURE ENGINE (The Math)
# ==========================================
class MicrostructureEngine:
    def __init__(self, config, df):
        self.config = config
        self.df = df.copy()

    def run_pipeline(self):
        print("Running Analytics Pipeline...")
        self.calc_depth()
        self.calc_obi()
        self.calc_microprice()
        self.calc_slippage()
        self.detect_sweeps()
        self.detect_icebergs()
        self.calc_tradeability_score()
        return self.df

    def calc_depth(self):
        # 4.1 Depth Aggregation
        bid_cols = [c for c in self.df.columns if 'bid_sz' in c]
        ask_cols = [c for c in self.df.columns if 'ask_sz' in c]
        
        self.df['total_bid_depth'] = self.df[bid_cols].sum(axis=1)
        self.df['total_ask_depth'] = self.df[ask_cols].sum(axis=1)

    def calc_obi(self):
        # 4.2 Order Book Imbalance
        # Formula: (Bid - Ask) / (Bid + Ask)
        total = self.df['total_bid_depth'] + self.df['total_ask_depth']
        self.df['obi'] = (self.df['total_bid_depth'] - self.df['total_ask_depth']) / total
        self.df['obi'] = self.df['obi'].fillna(0)

    def calc_microprice(self):
        # 4.3 Microprice
        # Formula: (Ask0 * AskSize0 + Bid0 * BidSize0) / (AskSize0 + BidSize0)
        # Note: This formula weights towards the larger size (liquidity gravity)
        num = (self.df['ask_px_0'] * self.df['ask_sz_0']) + (self.df['bid_px_0'] * self.df['bid_sz_0'])
        denom = self.df['ask_sz_0'] + self.df['bid_sz_0']
        self.df['microprice'] = num / denom
        
        # Midprice for reference
        self.df['midprice'] = (self.df['ask_px_0'] + self.df['bid_px_0']) / 2
        self.df['micro_delta'] = self.df['microprice'] - self.df['midprice']

    def calc_slippage(self):
        # 4.5 Slippage Proxy (Rolling VWAP vs Micro)
        # Simplified execution for dataframe: Rolling window over rows
        N = self.config['params']['slippage_window']
        
        # Calculate Rolling VWAP
        v_price = self.df['price'] * self.df['size']
        roll_vp = v_price.rolling(N).sum()
        roll_v = self.df['size'].rolling(N).sum()
        self.df['rolling_vwap'] = roll_vp / roll_v
        
        # Slippage = Sign(aggressor) * (VWAP - Micro)
        # Map aggressor to sign
        sign = self.df['aggressor_side'].map({'buy': 1, 'sell': -1}).fillna(0)
        self.df['slippage'] = sign * (self.df['rolling_vwap'] - self.df['microprice'])

    def detect_sweeps(self):
        # 4.7 Sweep Detection (Simplified Vectorized Approach)
        # Detect where trade price changes rapidly against same timestamp diff
        self.df['sweep_event'] = 0
        self.df['sweep_pressure'] = 0.0 # For Score
        
        # If price moves > X levels in < 100ms
        dt = self.df['timestamp'].diff().dt.total_seconds() * 1000
        px_chg = self.df['price'].diff().abs()
        
        # Arbitrary threshold for synthetic data: large move in short time
        is_sweep = (dt < 100) & (px_chg > 0.5) 
        self.df.loc[is_sweep, 'sweep_event'] = 1
        self.df.loc[is_sweep, 'sweep_pressure'] = 1.0 # Normalized mock

    def detect_icebergs(self):
        # 4.6 Iceberg Detection (Heuristic)
        # High volume trade, price unchanged
        self.df['iceberg_detected'] = 0
        self.df['iceberg_pressure'] = 0.0
        
        high_vol = self.df['size'] > self.df['size'].quantile(0.90)
        price_stable = self.df['price'].diff() == 0
        
        is_ice = high_vol & price_stable
        self.df.loc[is_ice, 'iceberg_detected'] = 1
        self.df.loc[is_ice, 'iceberg_pressure'] = 1.0

    def calc_tradeability_score(self):
        # 4.9 Composite Tradeability Score
        # Using Logistic Function on weighted z-scores
        
        # 1. Normalize components (Simple MinMax for robustness in this demo)
        def normalize(series):
            return (series - series.mean()) / (series.std() + 1e-9)

        w = self.config['params']['weights']
        
        # Components
        z_obi = normalize(self.df['obi']) * w['obi']
        z_slip = normalize(self.df['slippage'].fillna(0)) * w['slippage']
        z_micro = normalize(self.df['micro_delta']) * w['micro']
        z_sweep = normalize(self.df['sweep_pressure']) * w['sweep']
        
        # Logit Input
        logit_input = z_obi - z_slip + z_micro + z_sweep
        
        # Logistic Transform to 0-1, then scale to 0-100
        self.df['tradeability_score'] = expit(logit_input) * 100

# ==========================================
# CLASS 3: DASHBOARD RENDERER (Plotly)
# ==========================================
class DashboardRenderer:
    def __init__(self, df, config):
        self.df = df
        self.config = config

    def generate_dashboard(self):
        print("Rendering Dashboard...")
        
        # Layout: Grid with Specs
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.05,
            specs=[
                [{"colspan": 2}, None],             # Row 1: Depth Heatmap (Wide)
                [{"type": "xy"}, {"type": "xy"}],   # Row 2: OBI | Microprice
                [{"type": "xy"}, {"type": "xy"}],   # Row 3: Sweeps | Icebergs
                [{"colspan": 2, "type": "indicator"}, None] # Row 4: Score Gauge
            ],
            subplot_titles=("L2 Depth Heatmap", "Order Book Imbalance (OBI)", 
                            "Microprice Divergence", "Sweep Events", "Iceberg Detection", 
                            "Composite Tradeability Score")
        )

        # 1. Depth Heatmap (Simplified contour for MVP)
        # Constructing a matrix for heatmapping
        # In a real heavy app, this would be downsampled
        fig.add_trace(go.Heatmap(
            x=self.df['timestamp'],
            y=[f'Level {i}' for i in range(5)],
            z=self.df[[f'bid_sz_{i}' for i in range(5)]].T.values,
            colorscale='Viridis',
            showscale=False
        ), row=1, col=1)

        # 2. OBI
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'], y=self.df['obi'],
            mode='lines', name='OBI',
            line=dict(color='#00d4ff', width=1)
        ), row=2, col=1)

        # 3. Microprice vs Mid
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'], y=self.df['microprice'],
            mode='lines', name='Microprice',
            line=dict(color='#ff006e')
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'], y=self.df['midprice'],
            mode='lines', name='Mid',
            line=dict(color='gray', dash='dot')
        ), row=2, col=2)

        # 4. Sweeps
        sweeps = self.df[self.df['sweep_event'] == 1]
        fig.add_trace(go.Scatter(
            x=sweeps['timestamp'], y=sweeps['price'],
            mode='markers', name='Sweeps',
            marker=dict(symbol='star', size=10, color='orange')
        ), row=3, col=1)

        # 5. Icebergs
        ice = self.df[self.df['iceberg_detected'] == 1]
        fig.add_trace(go.Scatter(
            x=ice['timestamp'], y=ice['price'],
            mode='markers', name='Icebergs',
            marker=dict(symbol='diamond', size=8, color='cyan')
        ), row=3, col=2)

        # 6. Tradeability Gauge (Last Value)
        last_score = self.df['tradeability_score'].iloc[-1]
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=last_score,
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': last_score}
            },
            title={'text': "Live Tradeability"}
        ), row=4, col=1)

        # Styling
        fig.update_layout(
            template="plotly_dark",
            height=1200,
            title_text="L2 Quant Analytics Engine",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        # 6.1 Output with Resize Fix
        output_file = os.path.join(self.config['dirs']['output'], f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        
        # Inject JavaScript for Tab Resizing Bug in Plotly
        post_script = """
        <script>
            window.onresize = function() {
                var graph = document.getElementsByClassName('plotly-graph-div')[0];
                Plotly.Plots.resize(graph);
            };
        </script>
        """
        
        # Write HTML
        pio.write_html(fig, file=output_file, include_plotlyjs='cdn', full_html=True)
        
        # Append Hack (Manual file edit to insert script at end of body)
        with open(output_file, "a") as f:
            f.write(post_script)
            
        print(f"Dashboard generated: {output_file}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Ingest
    ingestion = DataIngestion(CONFIG)
    df_merged, df_trades = ingestion.get_data()
    
    # 2. Compute
    engine = MicrostructureEngine(CONFIG, df_merged)
    df_analytics = engine.run_pipeline()
    
    # 3. Export Intermediate Data
    analytics_path = os.path.join(CONFIG['dirs']['analytics'], 'metrics.csv')
    df_analytics.to_csv(analytics_path)
    print(f"Metrics exported to {analytics_path}")
    
    # 4. Visualize
    renderer = DashboardRenderer(df_analytics, CONFIG)
    renderer.generate_dashboard()
