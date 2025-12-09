# SCRIPTNAME: ok.05.bayesian.probabilities.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Bayesian Next-Day Move Probability Model (Trader Visuals Edition)
-----------------------------------------------------------------
Role: Senior Quantitative Developer
Context: Hedge Fund Infrastructure
Description: 
    Estimates next-day up-move probabilities conditioned on market streaks 
    and absolute magnitude regimes. Features "Trader-First" visuals 
    using traffic-light coloring for rapid decision making.

Usage:
    python3 bayes_momentum.py --tickers SPY QQQ --lookback 20 --include-dashboard
"""

import os
import sys
import time
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=FutureWarning)

# JS Snippet to fix Plotly tab rendering issues in offline HTML
RESIZE_JS = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    var config = {responsive: true};
    var plots = document.getElementsByClassName("plotly-graph-div");
    
    // Observer to resize plots when tabs are switched (visibility changes)
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                for (var i = 0; i < plots.length; i++) {
                    Plotly.Plots.resize(plots[i]);
                }
            }
        });
    });

    window.onresize = function() {
        for (var i = 0; i < plots.length; i++) {
            Plotly.Plots.resize(plots[i]);
        }
    };
});
</script>
"""

# ==========================================
# CORE CLASSES
# ==========================================

class DataIngestion:
    """Handles disk-first data loading and sanitization."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_data(self, ticker: str, lookback_years: int) -> pd.DataFrame:
        file_path = self.output_dir / f"{ticker}.csv"
        
        # 1. Check Disk
        if file_path.exists():
            logger.info(f"[{ticker}] Found local data. Loading...")
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if 'Close' not in df.columns: raise ValueError("Malformed CSV")
                return self._sanitize(df, lookback_years)
            except Exception as e:
                logger.warning(f"[{ticker}] Local load failed ({e}). Redownloading.")
        
        # 2. Download
        logger.info(f"[{ticker}] Downloading via yfinance...")
        try:
            start_date = (datetime.now() - pd.DateOffset(years=lookback_years + 1)).strftime('%Y-%m-%d')
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            time.sleep(1) 
            
            if df.empty: raise ValueError("No data returned")
            
            # Flatten if multi-index (yfinance update fix)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = self._sanitize(df, lookback_years)
            df.to_csv(file_path)
            return df
            
        except Exception as e:
            logger.error(f"[{ticker}] Download failed: {e}")
            sys.exit(1)

    def _sanitize(self, df: pd.DataFrame, lookback_years: int) -> pd.DataFrame:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df.dropna(subset=['Close'], inplace=True)
        cutoff = datetime.now() - pd.DateOffset(years=lookback_years)
        return df[df.index >= cutoff]


class FeatureEngineering:
    """Calculates streaks, buckets, and targets."""
    
    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data['pct'] = data['Close'].pct_change() * 100
        data['sign'] = np.sign(data['pct']).replace(0, 1) # Default 0 to 1
        data['target_up'] = (data['sign'].shift(-1) > 0).astype(int)
        
        # Streak Calculation
        data['sign_change'] = data['sign'].ne(data['sign'].shift())
        data['streak_id'] = data['sign_change'].cumsum()
        data['streak_len_raw'] = data.groupby('streak_id').cumcount() + 1
        data['streak_len'] = data['streak_len_raw'].clip(upper=5)
        
        # Magnitude Buckets (Today's volatility)
        data['abs_pct'] = data['pct'].abs()
        # Adjusted bins for more granularity
        bins = [0, 0.5, 1.0, 2.0, 3.0, np.inf]
        labels = ['Flat (<0.5%)', 'Normal (0.5-1%)', 'Vol (1-2%)', 'High Vol (2-3%)', 'Extreme (>3%)']
        data['abs_bucket'] = pd.cut(data['abs_pct'], bins=bins, labels=labels, right=False)
        
        data['streak_sign_val'] = data['sign']
        data.dropna(inplace=True)
        return data


class BayesianModel:
    """Beta-Binomial Inference Engine."""
    
    def __init__(self, ci_level=0.80):
        self.ci_level = ci_level
        self.alpha_lower = (1 - ci_level) / 2
        self.alpha_upper = 1 - self.alpha_lower

    def fit_posterior(self, successes: int, trials: int) -> Dict:
        # Jeffreys Prior
        alpha_post = successes + 0.5
        beta_post = trials - successes + 0.5
        
        return {
            'mean': alpha_post / (alpha_post + beta_post),
            'ci_low': stats.beta.ppf(self.alpha_lower, alpha_post, beta_post),
            'ci_high': stats.beta.ppf(self.alpha_upper, alpha_post, beta_post),
            'n': trials,
            'k': successes
        }

    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        groups = df.groupby(['streak_sign_val', 'streak_len', 'abs_bucket'])
        results = []
        for name, group in groups:
            k = group['target_up'].sum()
            n = len(group)
            if n == 0: continue
            
            stats_dict = self.fit_posterior(k, n)
            results.append({
                'streak_sign': int(name[0]),
                'streak_len': int(name[1]),
                'abs_bucket': name[2],
                **stats_dict
            })
        return pd.DataFrame(results)


class CrossValidator:
    """LOYO Cross Validation."""
    
    def __init__(self, model: BayesianModel):
        self.model = model

    def run_loyo(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        years = df.index.year.unique()
        predictions = []
        
        for year in years:
            train = df[df.index.year != year]
            test = df[df.index.year == year]
            if test.empty or train.empty: continue
            
            train_stats = self.model.analyze_dataset(train)
            if train_stats.empty: continue
            
            # Create fast lookup map
            lookup = train_stats.set_index(['streak_sign', 'streak_len', 'abs_bucket'])['mean'].to_dict()
            global_mean = train['target_up'].mean()
            
            def get_prob(row):
                return lookup.get((row['streak_sign_val'], row['streak_len'], row['abs_bucket']), global_mean)
            
            test_preds = test.copy()
            test_preds['prob'] = test.apply(get_prob, axis=1)
            
            predictions.append(test_preds[['target_up', 'prob']])
            
        if not predictions:
            return pd.DataFrame(), {'brier': 0, 'ece': 0}
            
        full_preds = pd.concat(predictions)
        full_preds.rename(columns={'target_up': 'actual'}, inplace=True)
        
        # Metrics
        brier = ((full_preds['prob'] - full_preds['actual']) ** 2).mean()
        
        # ECE
        full_preds['bin'] = pd.cut(full_preds['prob'], bins=np.linspace(0, 1, 11), labels=False)
        ece = 0
        for b in range(10):
            bin_data = full_preds[full_preds['bin'] == b]
            if len(bin_data) > 0:
                ece += (len(bin_data) / len(full_preds)) * abs(bin_data['actual'].mean() - bin_data['prob'].mean())
                
        return full_preds, {'brier': brier, 'ece': ece}


class DashboardRenderer:
    """Generates Trader-First Plotly Dashboard."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate(self, ticker: str, 
                 bayes_df: pd.DataFrame, 
                 loyo_preds: pd.DataFrame, 
                 metrics: Dict,
                 raw_df: pd.DataFrame):
        
        logger.info(f"[{ticker}] Generating Trader Dashboard...")
        figs = []

        # =========================================================
        # VISUAL 1: The "Reversion Watch" (Down Streaks)
        # =========================================================
        # Context: Market is dropping. Do we catch the knife (Call) or short more (Put)?
        # Color Logic: Red (Bearish Continuation) -> Yellow (Coin Flip) -> Green (Bullish Bounce)
        
        down_data = bayes_df[bayes_df['streak_sign'] == -1].copy()
        if not down_data.empty:
            pivot_mean = down_data.pivot(index='streak_len', columns='abs_bucket', values='mean')
            pivot_n = down_data.pivot(index='streak_len', columns='abs_bucket', values='n').fillna(0)
            
            # Annotations: Prob% + Count
            # Use HTML styling in text for better readability
            text_vals = pivot_mean.applymap(lambda x: f"<b>{x:.1%}</b>") + "<br><span style='font-size:10px'>N=" + pivot_n.astype(int).astype(str) + "</span>"
            
            fig_down = go.Figure(data=go.Heatmap(
                z=pivot_mean.values,
                x=pivot_mean.columns,
                y=pivot_mean.index,
                text=text_vals.values,
                texttemplate="%{text}",
                # Red-Yellow-Green Diverging Scale centered at 0.5
                colorscale='RdYlGn', 
                zmid=0.5, 
                zmin=0.3, zmax=0.7, # Clamp contrast to highlighting edge
                showscale=True,
                colorbar=dict(title="Prob Next Day UP")
            ))
            fig_down.update_layout(
                title=f'<b>OVERSOLD WATCH:</b> Probability of GREEN Day after {ticker} drops',
                xaxis_title="Volatility (Absolute Drop Size)",
                yaxis_title="Consecutive Down Days",
                template="plotly_white",
                height=500
            )
            figs.append(fig_down)

        # =========================================================
        # VISUAL 2: The "Momentum Watch" (Up Streaks)
        # =========================================================
        up_data = bayes_df[bayes_df['streak_sign'] == 1].copy()
        if not up_data.empty:
            pivot_mean = up_data.pivot(index='streak_len', columns='abs_bucket', values='mean')
            pivot_n = up_data.pivot(index='streak_len', columns='abs_bucket', values='n').fillna(0)
            
            text_vals = pivot_mean.applymap(lambda x: f"<b>{x:.1%}</b>") + "<br><span style='font-size:10px'>N=" + pivot_n.astype(int).astype(str) + "</span>"
            
            fig_up = go.Figure(data=go.Heatmap(
                z=pivot_mean.values,
                x=pivot_mean.columns,
                y=pivot_mean.index,
                text=text_vals.values,
                texttemplate="%{text}",
                colorscale='RdYlGn',
                zmid=0.5, zmin=0.3, zmax=0.7,
                showscale=True,
                colorbar=dict(title="Prob Next Day UP")
            ))
            fig_up.update_layout(
                title=f'<b>MOMENTUM WATCH:</b> Probability of GREEN Day after {ticker} rises',
                xaxis_title="Volatility (Absolute Rise Size)",
                yaxis_title="Consecutive Up Days",
                template="plotly_white",
                height=500
            )
            figs.append(fig_up)

        # =========================================================
        # VISUAL 3: Calibration (Truth Check) - Fixed ZeroDiv
        # =========================================================
        if not loyo_preds.empty:
            bins = np.linspace(0, 1, 11)
            hist, edges = np.histogram(loyo_preds['prob'], bins=bins)
            prob_true, _ = np.histogram(loyo_preds[loyo_preds['actual']==1]['prob'], bins=bins)
            
            # Safe Division Mask
            mask = hist > 0
            centers = (edges[:-1] + edges[1:]) / 2
            
            valid_hist = hist[mask]
            valid_centers = centers[mask]
            frac_pos = prob_true[mask] / valid_hist
            
            # Wilson Score Interval (80%)
            z = 1.28 
            denom = 1 + z**2/valid_hist
            center_adj = (frac_pos + z**2/(2*valid_hist)) / denom
            
            # Error bars requires 'array' for symmetric or 'arrayminus'
            # Simpler approx for chart:
            err = z * np.sqrt(frac_pos*(1-frac_pos)/valid_hist + z**2/(4*valid_hist**2)) / denom
            
            fig_cal = go.Figure()
            # Identity Line
            fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', 
                                         line=dict(dash='dash', color='gray'), name='Ideal'))
            
            # Calibration Points
            fig_cal.add_trace(go.Scatter(
                x=valid_centers, y=frac_pos,
                mode='markers+lines',
                error_y=dict(type='data', array=err, visible=True, color='gray'),
                marker=dict(size=12, color='royalblue', line=dict(width=2, color='DarkSlateGrey')),
                name='Model Reality'
            ))
            
            fig_cal.add_annotation(
                text=f"<b>Brier Score:</b> {metrics['brier']:.3f} (Lower is better)<br><b>ECE:</b> {metrics['ece']:.3f}",
                xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1
            )
            fig_cal.update_layout(title=f'Model Reliability (LOYO Cross-Validation) - {ticker}',
                                  xaxis_title='Predicted Probability', yaxis_title='Actual Win Rate',
                                  template="plotly_white")
            figs.append(fig_cal)

        # =========================================================
        # VISUAL 4: Regime Stability Fan Chart (Panic Buy Scenario)
        # =========================================================
        # Scenario: "Buy the dip" (Down Streak 2 or 3, Normal Vol)
        # We want to see if this edge is fading.
        
        target_len = 2
        target_sign = -1
        target_bucket = 'Normal (0.5-1%)' # Adjust based on availability
        
        mask = (raw_df['streak_sign_val'] == target_sign) & \
               (raw_df['streak_len'] == target_len) & \
               (raw_df['abs_bucket'] == target_bucket)
        
        fan_df = raw_df[mask].copy()
        
        if len(fan_df) > 10:
            means, uppers, lowers, dates = [], [], [], []
            alpha_0, beta_0 = 0.5, 0.5
            cum_k, cum_n = 0, 0
            
            for date, row in fan_df.iterrows():
                cum_n += 1
                cum_k += row['target_up']
                a, b = cum_k + alpha_0, (cum_n - cum_k) + beta_0
                
                means.append(a / (a+b))
                lowers.append(stats.beta.ppf(0.10, a, b))
                uppers.append(stats.beta.ppf(0.90, a, b))
                dates.append(date)
            
            fig_fan = go.Figure()
            # CI Band
            fig_fan.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=uppers + lowers[::-1],
                fill='toself',
                fillcolor='rgba(0,176,246,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='80% Confidence'
            ))
            # Mean Line
            fig_fan.add_trace(go.Scatter(
                x=dates, y=means, 
                line=dict(color='rgb(0,176,246)', width=3),
                name='Probability Estimate'
            ))
            # 50% Line
            fig_fan.add_hline(y=0.5, line_dash="dot", line_color="gray")
            
            fig_fan.update_layout(
                title=f'<b>Edge Evolution:</b> Stability of "Dip Buying" (Down {target_len}, {target_bucket})',
                yaxis_title="Prob Next Day Green",
                yaxis_range=[0, 1],
                template="plotly_white"
            )
            figs.append(fig_fan)

        # =========================================================
        # HTML GENERATION
        # =========================================================
        html_content = f"""
        <html>
        <head>
            <title>{ticker} Option Analytics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 20px; }}
                .container {{ max_width: 1200px; margin: 0 auto; }}
                .header {{ background: #1a1a1a; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .metric-box {{ display: inline-block; padding: 10px 20px; background: #333; border-radius: 4px; color: #fff; margin-right: 10px; }}
                h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin:0">{ticker} Bayesian Options Signal</h1>
                    <p style="opacity:0.8">Lookback: {raw_df.index.min().date()} to {raw_df.index.max().date()} | Samples: {len(raw_df)}</p>
                </div>
        """
        
        for i, fig in enumerate(figs):
            div_str = pio.to_html(fig, full_html=False, include_plotlyjs=(i==0)) 
            html_content += f'<div class="card">{div_str}</div>'
            
        html_content += RESIZE_JS + "</div></body></html>"
        
        with open(self.output_dir / ticker / "dashboard.html", "w", encoding="utf-8") as f:
            f.write(html_content)


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_analysis(args):
    out_dir = Path(args.output_dir)
    
    ingestion = DataIngestion(out_dir)
    feat_eng = FeatureEngineering()
    bayes = BayesianModel(ci_level=args.ci_level)
    validator = CrossValidator(bayes)
    renderer = DashboardRenderer(out_dir)

    for ticker in args.tickers:
        print(f"\n{'='*40}")
        print(f"PROCESSING: {ticker}")
        print(f"{'='*40}")
        
        # 1. Ingestion
        df = ingestion.get_data(ticker, args.lookback)
        
        # 2. Features
        df_feats = feat_eng.process(df)
        if df_feats.empty:
            logger.warning(f"[{ticker}] Not enough data.")
            continue
            
        # 3. Bayesian Fit
        bayes_stats = bayes.analyze_dataset(df_feats)
        
        # 4. LOYO Validation
        loyo_preds, metrics = validator.run_loyo(df_feats)
        logger.info(f"[{ticker}] LOYO Brier: {metrics.get('brier', 'N/A'):.4f} (Low=Good)")
        
        # 5. Output
        ticker_dir = out_dir / ticker
        ticker_dir.mkdir(exist_ok=True)
        (ticker_dir / "tables").mkdir(exist_ok=True)
        
        if not bayes_stats.empty:
            bayes_stats.to_csv(ticker_dir / "tables" / "posterior_stats.csv", index=False)

        if args.include_dashboard:
            renderer.generate(ticker, bayes_stats, loyo_preds, metrics, df_feats)
            
        logger.info(f"[{ticker}] Dashboard: {(ticker_dir / 'dashboard.html').resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Momentum Analytics")
    parser.add_argument("--tickers", nargs="+", default=["SPY"], help="List of tickers")
    parser.add_argument("--output-dir", default="./bayes_prob", help="Output directory")
    parser.add_argument("--lookback", type=int, default=20, help="Years of history")
    parser.add_argument("--ci-level", type=float, default=0.80, help="Confidence Interval")
    parser.add_argument("--include-dashboard", action="store_true", default=True)
    
    args = parser.parse_args()
    
    try:
        run_analysis(args)
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        logger.exception("Fatal error")
