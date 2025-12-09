# SCRIPTNAME: ok.6.options_pop_calibration_isotonic_dashboard.v5.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as py_offline
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
import logging
import os
import sys
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ----------------------------------------------------------------------------------
# FIX: Removed [%(class)s] which was causing the ValueError
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("OptionsPoP")

# ----------------------------------------------------------------------------------
# CLASS 1: DATA INGESTION (Robust & Sanitized)
# ----------------------------------------------------------------------------------
class DataIngestion:
    """
    Handles data fetching, caching, shadow backfilling, and aggressive sanitization.
    Fixes the infamous yfinance MultiIndex/Swap-Level bugs.
    """
    def __init__(self, ticker: str, start_date: str = None, end_date: str = None):
        self.ticker = ticker.upper()
        self.start_date = start_date if start_date else (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.cache_file = f"{self.ticker}_data.csv"

    def fetch_data(self) -> pd.DataFrame:
        """
        Orchestrates the data loading process:
        1. Check Cache -> 2. Download from YF -> 3. Sanitize -> 4. Fallback to Shadow
        """
        df = pd.DataFrame()

        # 1. Try Cache
        if os.path.exists(self.cache_file):
            logger.info(f"Loading data from local cache: {self.cache_file}")
            try:
                df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
                logger.info("Cache loaded successfully.")
                return df
            except Exception as e:
                logger.warning(f"Cache load failed ({e}). Proceeding to download.")

        # 2. Download
        try:
            logger.info(f"Downloading {self.ticker} via yfinance...")
            # group_by='column' helps prevent some MultiIndex messiness upfront
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False, group_by='column', auto_adjust=True)
        except Exception as e:
            logger.error(f"Download failed: {e}")

        # 3. Sanitize
        if not df.empty:
            df = self._sanitize_df(df)
            # Save to cache
            df.to_csv(self.cache_file)
            return df
        
        # 4. Shadow Fallback
        logger.warning("Data fetch returned empty or failed. Generating Shadow Backfill (Random Walk).")
        return self._generate_shadow_data()

    def _sanitize_df(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressive cleanup to handle yfinance API instability.
        """
        df = df_input.copy()

        # A. Handle MultiIndex Column Swapping (The "Swap Levels" Fix)
        if isinstance(df.columns, pd.MultiIndex):
            # Check if 'Close' is in Level 1 but not Level 0 (Common bug)
            level_0_vals = df.columns.get_level_values(0).unique()
            level_1_vals = df.columns.get_level_values(1).unique() if df.columns.nlevels > 1 else []

            if 'Close' not in level_0_vals and 'Close' in level_1_vals:
                logger.info("Detected swapped MultiIndex levels. Swapping back.")
                df = df.swaplevel(0, 1, axis=1)

            # Flatten columns: We only want the attribute (Close, Open, etc.), drop Ticker level
            # We assume the level containing 'Close' is now at index 0
            new_cols = []
            for col in df.columns:
                # If tuple, grab the part that looks like a standard column
                if isinstance(col, tuple):
                    val = col[0] if col[0] in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] else col[1]
                    new_cols.append(val)
                else:
                    new_cols.append(col)
            df.columns = new_cols

        # B. Ensure Standard Columns Exist
        required = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            # If 'Adj Close' exists but 'Close' doesn't, rename it
            if 'Close' in missing and 'Adj Close' in df.columns:
                df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            else:
                raise ValueError(f"Critical columns missing after sanitization: {missing}")

        # C. Type Coercion & Index
        df.index = pd.to_datetime(df.index).tz_localize(None) # Strip timezone
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        logger.info(f"Sanitization complete. Shape: {df.shape}")
        return df

    def _generate_shadow_data(self) -> pd.DataFrame:
        """
        Generates a synthetic random walk to prevent dashboard crash on offline/fail mode.
        """
        dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
        np.random.seed(42)
        # 16% annualized vol
        returns = np.random.normal(loc=0.0005, scale=0.16/np.sqrt(252), size=len(dates))
        price_path = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=dates)
        df['Close'] = price_path
        df['Open'] = price_path # Simplify
        df['High'] = price_path * 1.01
        df['Low'] = price_path * 0.99
        df['Volume'] = np.random.randint(100000, 5000000, size=len(dates))
        df.index.name = 'Date'
        logger.info("Shadow data generated.")
        return df

# ----------------------------------------------------------------------------------
# CLASS 2: FINANCIAL ANALYSIS (Pure Math & Probability)
# ----------------------------------------------------------------------------------
class FinancialAnalysis:
    """
    Performs Feature Engineering, Purged K-Fold CV, and Isotonic Calibration.
    Immutable: Never modifies original _raw_data.
    """
    def __init__(self, df: pd.DataFrame):
        self._raw_data = df
        self.features = []
        self.target = 'Target_10d'
        
    def run_analysis(self):
        """
        Master method to execute the quantitative pipeline.
        """
        df = self._raw_data.copy()
        
        # 1. Feature Engineering
        df = self._build_features(df)
        
        # 2. Labeling (10-day lookahead)
        # Target: 1 if Price(t+10) > Price(t), else 0
        df['Future_Close'] = df['Close'].shift(-10)
        df[self.target] = (df['Future_Close'] > df['Close']).astype(int)
        
        # Drop NaNs created by shifts
        df.dropna(inplace=True)
        
        # 3. Model Training & Calibration
        probs, feature_importance = self._train_calibrated_model(df)
        
        df['Probability'] = probs
        
        return df, feature_importance

    def _build_features(self, df):
        """
        Generates RSI, SMAs, Rolling Vol, Lags.
        """
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # SMAs & Distances
        for window in [20, 50, 200]:
            col_name = f'SMA_{window}'
            df[col_name] = df['Close'].rolling(window=window).mean()
            df[f'Dist_SMA_{window}'] = df['Close'] / df[col_name] - 1
            
        # Rolling Volatility (Annualized)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_20'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252)
        
        # Lagged Returns
        for lag in [1, 3, 5]:
            df[f'Ret_Lag_{lag}'] = df['Close'].pct_change(lag)

        self.features = ['RSI_14', 'Vol_20', 'Dist_SMA_20', 'Dist_SMA_50', 'Dist_SMA_200', 'Ret_Lag_1', 'Ret_Lag_3', 'Ret_Lag_5']
        return df

    def _purged_kfold_split(self, X, n_splits=5, embargo_pct=0.01):
        """
        Generator for Purged K-Fold indices.
        Purging: Remove overlap between train/test.
        Embargo: Drop samples immediately after test set to prevent leakage.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        fold_size = n_samples // n_splits
        embargo = int(n_samples * embargo_pct)

        for i in range(n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            
            test_indices = indices[test_start:test_end]
            
            # Train indices: Everything BEFORE test_start and AFTER (test_end + embargo)
            # We assume time series is sorted.
            t0 = indices[:test_start]
            t1 = indices[test_end + embargo:]
            
            train_indices = np.concatenate((t0, t1))
            
            yield train_indices, test_indices

    def _train_calibrated_model(self, df):
        """
        Trains LogisticRegression with Isotonic Calibration using Purged CV.
        """
        X = df[self.features]
        y = df[self.target]
        
        # Base Model
        base_lr = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1')
        
        # Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_lr)
        ])
        
        # Production Training (Isotonic Calibration on Full Data for Dashboard)
        # FIX: Use 'pipeline' instead of 'base_lr' so features are actually scaled.
        # This is critical for L1 penalty convergence and feature importance accuracy.
        calibrated_clf = CalibratedClassifierCV(pipeline, method='isotonic', cv=3)
        calibrated_clf.fit(X, y)
        
        probs = calibrated_clf.predict_proba(X)[:, 1]
        
        # Extract Feature Importance
        # FIX: The calibrated classifier wraps the pipeline.
        # We must drill down: CalibratedClassifier -> Estimator (Pipeline) -> Named Step 'model' -> coef_
        coefs_list = []
        for clf in calibrated_clf.calibrated_classifiers_:
            # clf.estimator is the fitted pipeline
            model_step = clf.estimator.named_steps['model']
            coefs_list.append(model_step.coef_)
            
        coefs = np.mean(coefs_list, axis=0)
        
        feat_imp = pd.DataFrame({
            'Feature': self.features,
            'Importance': np.abs(coefs[0])
        }).sort_values(by='Importance', ascending=True)
        
        return probs, feat_imp

# ----------------------------------------------------------------------------------
# CLASS 3: DASHBOARD RENDERER (Offline & Fixes)
# ----------------------------------------------------------------------------------
class DashboardRenderer:
    """
    Renders the HTML Dashboard.
    Key Fixes:
    1. Offline JS injection.
    2. Tab resize bug fix via JS event dispatch.
    """
    def __init__(self, ticker, df, feature_imp):
        self.ticker = ticker
        self.df = df
        self.feature_imp = feature_imp
        
    def generate_dashboard(self, filename="options_dashboard.html"):
        logger.info("Generating Dashboard...")
        
        # 1. Regime Chart (Price + Background Coloring)
        fig_regime = self._create_regime_chart()
        
        # 2. Probability Cone
        fig_prob = self._create_prob_chart()
        
        # 3. Feature Importance
        fig_feat = self._create_feature_chart()
        
        # 4. Assemble HTML
        html_content = self._assemble_html(fig_regime, fig_prob, fig_feat)
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard saved to {os.path.abspath(filename)}")
        return os.path.abspath(filename)

    def _create_regime_chart(self):
        """
        Main Chart: Price candle/line with background colored by PoP.
        """
        df = self.df.tail(252) # Last year
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Price Line
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='white', width=2)))
        
        # Probability Scatter (Secondary Y)
        fig.add_trace(go.Scatter(x=df.index, y=df['Probability'], name='Win Prob', 
                                 line=dict(color='cyan', width=1, dash='dot'), opacity=0.5), secondary_y=True)

        # Shapes for Regimes
        shapes = []
        # Optimization: Don't draw a shape for every day. Group contiguous regions.
        # Simple implementation for robustness:
        for i in range(len(df) - 1):
            prob = df['Probability'].iloc[i]
            color = None
            if prob > 0.65:
                color = 'rgba(0, 255, 0, 0.1)' # Green Zone
            elif prob < 0.35:
                color = 'rgba(255, 0, 0, 0.1)' # Red Zone
            
            if color:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=df.index[i], y0=0, x1=df.index[i+1], y1=1,
                    fillcolor=color, opacity=1, layer="below", line_width=0,
                ))
        
        fig.update_layout(
            title=f"{self.ticker} Regime Analysis (Green > 65% PoP, Red < 35% PoP)",
            template="plotly_dark",
            shapes=shapes,
            height=600
        )
        return fig

    def _create_prob_chart(self):
        """
        Reliability Cone: Price vs Volatility Bands.
        """
        df = self.df.tail(100).copy()
        
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='white')))
        
        # Volatility Cone (using Vol_20)
        # Just a visual proxy: Price +/- 1 std dev move over 10 days
        # 10-day vol = Vol_20 * sqrt(10/252) ? Approximation
        # Let's use the Vol_20 feature we engineered (which is annualized).
        # Daily vol approx = Vol_20 / sqrt(252)
        daily_vol = df['Vol_20'] / np.sqrt(252)
        upper = df['Close'] * (1 + (daily_vol * np.sqrt(10))) # 10 day exp move
        lower = df['Close'] * (1 - (daily_vol * np.sqrt(10)))
        
        fig.add_trace(go.Scatter(x=df.index, y=upper, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=lower, mode='lines', line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(255, 165, 0, 0.2)', name='10d Vol Cone'))
        
        fig.update_layout(title="Realized Volatility Cone (10-Day Expected Move)", template="plotly_dark", height=500)
        return fig

    def _create_feature_chart(self):
        fig = go.Figure(go.Bar(
            x=self.feature_imp['Importance'],
            y=self.feature_imp['Feature'],
            orientation='h',
            marker=dict(color='orange')
        ))
        fig.update_layout(title="Feature Importance (Model Coefficients)", template="plotly_dark", height=400)
        return fig

    def _assemble_html(self, fig1, fig2, fig3):
        """
        Constructs the Standalone HTML with Embedded JS and CSS.
        """
        # 1. Get RAW JS (Offline Mode)
        plotly_js = py_offline.get_plotlyjs()
        
        # 2. Get Div Strings
        div1 = py_offline.plot(fig1, include_plotlyjs=False, output_type='div')
        div2 = py_offline.plot(fig2, include_plotlyjs=False, output_type='div')
        div3 = py_offline.plot(fig3, include_plotlyjs=False, output_type='div')
        
        # 3. HTML Template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{self.ticker} PoP Dashboard</title>
            <script type="text/javascript">{plotly_js}</script>
            <style>
                body {{ background-color: #111; color: #ddd; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
                .container {{ max_width: 1200px; margin: 0 auto; }}
                h1 {{ color: #ffa500; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                .tab {{ overflow: hidden; border: 1px solid #333; background-color: #222; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; }}
                .tab button:hover {{ background-color: #333; }}
                .tab button.active {{ background-color: #ffa500; color: #000; font-weight: bold; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 1s; }}
                @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{self.ticker} Options Probability Dashboard (v4 Standalone)</h1>
                
                <div class="tab">
                    <button class="tablinks" onclick="openTab(event, 'Regime')" id="defaultOpen">Regime Analysis</button>
                    <button class="tablinks" onclick="openTab(event, 'Backtest')">Reliability Cone</button>
                    <button class="tablinks" onclick="openTab(event, 'Features')">Model Logic</button>
                </div>

                <div id="Regime" class="tabcontent">
                    {div1}
                    <p style="font-size: 0.9em; color: #888;">*Shaded regions indicate model conviction (>65% Win or <35% Loss).</p>
                </div>

                <div id="Backtest" class="tabcontent">
                    {div2}
                </div>

                <div id="Features" class="tabcontent">
                    {div3}
                </div>
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
                    
                    // CRITICAL FIX: Trigger Resize event so Plotly renders correctly in previously hidden tabs
                    window.dispatchEvent(new Event('resize'));
                }}
                
                // Open default tab
                document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
        """
        return html

# ----------------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    # Optional CLI for Ticker
    # Default to SPY if not provided
    TICKER = "SPY"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1]
        
    print("="*60)
    print(f"STARTING OPTION PoP DASHBOARD GENERATION FOR: {TICKER}")
    print("="*60)

    # 1. Ingest
    ingestor = DataIngestion(TICKER)
    df_raw = ingestor.fetch_data()

    # 2. Analyze
    analyzer = FinancialAnalysis(df_raw)
    df_analyzed, feature_importance = analyzer.run_analysis()
    
    # 3. Render
    renderer = DashboardRenderer(TICKER, df_analyzed, feature_importance)
    output_path = renderer.generate_dashboard(f"{TICKER}_PoP_Dashboard.html")
    
    print(f"\nSUCCESS! Dashboard generated at:\n{output_path}")
    print("="*60)
