"""
GTC Limit Scenario Mapper (V2 - Memory Optimized)
-------------------------------------------------
A standalone tool for professional options traders to map IV and spread shocks 
to qualitative fill-probability changes and quantitative GTC limit-price multipliers.

V2 Update: Implements streaming file writes to prevent MemoryError on large datasets.

Architecture:
1. DataIngestion: Disk-first IO, yfinance downloads, sanitization.
2. FinancialAnalysis: Black-Scholes IV inversion, scenario logic, multiplier heuristics.
3. DashboardRenderer: Streaming HTML generation to handle massive Plotly outputs.

Author: Michael Derby
Framework: STEGO Financial Framework
"""

import os
import sys
import argparse
import time
import datetime
import math
import logging
import warnings
import gc
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as py_offline
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. DataIngestion Class
# -----------------------------------------------------------------------------

class DataIngestion:
    """
    Responsible for all IO and data retrieval.
    Enforces a strict "Disk-First" pipeline for OHLC data.
    """

    def __init__(self, output_dir: str, lookback_years: float):
        self.output_dir = output_dir
        self.lookback_years = lookback_years
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # 1. Swap Levels if necessary (Handle yfinance MultiIndex variations)
        if isinstance(df.columns, pd.MultiIndex):
            sample_level_1 = df.columns.get_level_values(1)
            if 'Close' in sample_level_1 or 'Adj Close' in sample_level_1:
                 if 'Close' not in df.columns.get_level_values(0):
                     df = df.swaplevel(0, 1, axis=1)

        # 2. Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col_tuple in df.columns:
                parts = [str(c) for c in col_tuple if str(c) != '']
                new_cols.append("_".join(parts))
            df.columns = new_cols

        # 3. Strict DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df[df.index.notnull()]

        # 4. Coerce numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(how='all', inplace=True)
        df.sort_index(inplace=True)

        return df

    def get_ohlc_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        results = {}
        for ticker in tickers:
            csv_path = os.path.join(self.output_dir, f"{ticker}_ohlc.csv")
            df = None

            # Step 1: Check disk
            if os.path.exists(csv_path):
                logger.info(f"Loading {ticker} from disk: {csv_path}")
                try:
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    df = self._sanitize_df(df)
                except Exception as e:
                    logger.error(f"Failed to read CSV for {ticker}: {e}")
                    df = None

            # Step 2: Download if missing
            if df is None or df.empty:
                logger.info(f"Downloading {ticker} via yfinance...")
                try:
                    days = int(self.lookback_years * 365)
                    period_str = f"{days}d" if days < 60 else f"{int(self.lookback_years)}y"
                    if self.lookback_years < 1.0 and days > 5:
                        period_str = "1y" 

                    raw_df = yf.download(
                        tickers=ticker, 
                        period=period_str, 
                        group_by='column', 
                        progress=False,
                        threads=False
                    )
                    
                    clean_df = self._sanitize_df(raw_df)
                    clean_df.to_csv(csv_path)
                    
                    # Round-trip verification
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    df = self._sanitize_df(df)
                    time.sleep(0.5) 

                except Exception as e:
                    logger.error(f"Failed to download {ticker}: {e}")
                    continue

            results[ticker] = df
        return results

    def get_options_snapshots(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        snapshots = {}
        for ticker in tickers:
            logger.info(f"Fetching options chain for {ticker}...")
            try:
                yf_ticker = yf.Ticker(ticker)
                expirations = yf_ticker.options
                if len(expirations) < 2:
                    continue

                front_expiry = expirations[0]
                back_expiry = expirations[1]

                time.sleep(0.5)
                chain_front = yf_ticker.option_chain(front_expiry)
                time.sleep(0.5)
                chain_back = yf_ticker.option_chain(back_expiry)

                snapshots[ticker] = {
                    "front_expiry": front_expiry,
                    "back_expiry": back_expiry,
                    "calls_front": chain_front.calls,
                    "puts_front": chain_front.puts,
                    "calls_back": chain_back.calls,
                    "puts_back": chain_back.puts
                }
            except Exception as e:
                logger.error(f"Error fetching options for {ticker}: {e}")
                continue
        return snapshots


# -----------------------------------------------------------------------------
# 2. FinancialAnalysis Class
# -----------------------------------------------------------------------------

class FinancialAnalysis:
    def __init__(self, 
                 underlying_data: Dict[str, pd.DataFrame], 
                 options_data: Dict[str, Dict[str, Any]],
                 risk_free_rate: float,
                 order_type: str,
                 base_limit_price: Optional[float] = None):
        
        self.underlying_data = underlying_data
        self.options_data = options_data
        self.r = risk_free_rate
        self.order_type = order_type.lower()
        self.base_limit_price = base_limit_price

    def _black_scholes_call_price(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(0, S - K)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    def _implied_volatility(self, market_price, S, K, T, r):
        sigma = 0.5
        for i in range(20):
            price = self._black_scholes_call_price(S, K, T, r, sigma)
            diff = market_price - price
            if abs(diff) < 1e-5: return sigma
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            vega = S * norm.cdf(d1) * math.sqrt(T)
            if vega == 0: break
            sigma = sigma + diff / vega
        return sigma

    def compute_iv_snapshots(self) -> pd.DataFrame:
        rows = []
        for ticker, data in self.options_data.items():
            ohlc = self.underlying_data.get(ticker)
            if ohlc is None or ohlc.empty: continue
            
            close_col = f"Close_{ticker}"
            if close_col not in ohlc.columns:
                candidates = [c for c in ohlc.columns if 'Close' in c]
                if not candidates: continue
                close_col = candidates[0]

            spot = ohlc[close_col].iloc[-1]
            
            def get_iv(calls_df, expiry_str):
                try:
                    exp_date = pd.to_datetime(expiry_str)
                    T = (exp_date - pd.Timestamp.now()).days / 365.0
                    if T < 1/365.0: T = 1/365.0
                except: return np.nan

                calls_df['dist'] = abs(calls_df['strike'] - spot)
                atm_row = calls_df.sort_values('dist').iloc[0]
                K = atm_row['strike']
                
                bid = atm_row.get('bid', 0)
                ask = atm_row.get('ask', 0)
                if pd.isna(bid) or bid == 0: bid = atm_row.get('lastPrice', 0)
                if pd.isna(ask) or ask == 0: ask = atm_row.get('lastPrice', 0)
                
                mid_price = 0.5 * (bid + ask)
                if mid_price == 0: mid_price = atm_row.get('lastPrice', 0)

                return self._implied_volatility(mid_price, spot, K, T, self.r)

            front_iv = get_iv(data['calls_front'], data['front_expiry'])
            back_iv = get_iv(data['calls_back'], data['back_expiry'])

            rows.append({
                "ticker": ticker,
                "spot": spot,
                "front_expiry": data['front_expiry'],
                "back_expiry": data['back_expiry'],
                "front_iv": front_iv,
                "back_iv": back_iv
            })

        return pd.DataFrame(rows)

    def generate_scenarios(self) -> pd.DataFrame:
        base_scenarios = [
            {"id": "S1", "f_iv": 2,  "b_iv": 0,  "spread": 0,  "desc": "Front IV +2 (Pop)"},
            {"id": "S2", "f_iv": -2, "b_iv": 0,  "spread": 0,  "desc": "Front IV -2 (Drop)"},
            {"id": "S3", "f_iv": 0,  "b_iv": 2,  "spread": 0,  "desc": "Back IV +2 (Pop)"},
            {"id": "S4", "f_iv": 0,  "b_iv": -2, "spread": 0,  "desc": "Back IV -2 (Drop)"},
            {"id": "S5", "f_iv": 5,  "b_iv": 0,  "spread": 0,  "desc": "Front IV +5 (Spike)"},
            {"id": "S6", "f_iv": -5, "b_iv": 0,  "spread": 0,  "desc": "Front IV -5 (Crush)"},
            {"id": "S7", "f_iv": 0,  "b_iv": 0,  "spread": 35, "desc": "Liquidity Stress (+35bps)"},
            {"id": "S8", "f_iv": 2,  "b_iv": 2,  "spread": 15, "desc": "Parallel Bump + Stress"},
        ]

        results = []
        tickers = list(self.underlying_data.keys())

        for ticker in tickers:
            for scen in base_scenarios:
                f_bump = scen['f_iv']
                b_bump = scen['b_iv']
                spread_bps = scen['spread']
                
                mult_base = 1.0
                delta_mult = 0.0
                
                if self.order_type == 'credit':
                    # Credit: +IV = Tighten (lower limit/higher credit), +Spread = Relax (higher limit)
                    # We model "Multiplier" as price aggressiveness.
                    f_coeff = -0.005 
                    p_fill = "≈"
                    if f_bump > 0: p_fill = "↑"
                    elif f_bump < 0: p_fill = "↓"
                    elif spread_bps > 0: p_fill = "↓"
                    
                    delta_mult += (f_bump * -0.005)
                    if abs(f_bump) > 3:
                         excess = abs(f_bump) - 3
                         sign = 1 if f_bump > 0 else -1
                         delta_mult += (excess * sign * -0.001)
                    delta_mult += (b_bump * -0.0005)

                else: # Debit
                    p_fill = "≈"
                    if f_bump > 0: p_fill = "↓"
                    elif f_bump < 0: p_fill = "↑"
                    elif spread_bps > 0: p_fill = "↓"
                    
                    delta_mult += (f_bump * 0.005)
                    delta_mult += (b_bump * 0.0005)

                # Spread Logic: +35bps ~= +3-4% relaxation
                delta_mult += (spread_bps * 0.0009)

                final_mult = mult_base + delta_mult
                final_mult = max(0.90, min(1.10, final_mult))
                
                new_price = None
                if self.base_limit_price is not None:
                    new_price = self.base_limit_price * final_mult

                results.append({
                    "ticker": ticker,
                    "scenario_id": scen['id'],
                    "description": scen['desc'],
                    "front_IV_bump": f_bump,
                    "back_IV_bump": b_bump,
                    "spread_widen_bps": spread_bps,
                    "expected_Pfill": p_fill,
                    "gtc_multiplier": round(final_mult, 4),
                    "new_limit_price": round(new_price, 2) if new_price else None
                })
        
        return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 3. DashboardRenderer Class (MEMORY OPTIMIZED)
# -----------------------------------------------------------------------------

class DashboardRenderer:
    """
    Generates the offline HTML dashboard using Plotly.
    Uses STREAMING WRITES to avoid MemoryError.
    """

    def __init__(self, 
                 iv_snapshots: pd.DataFrame, 
                 scenario_table: pd.DataFrame, 
                 config: Dict[str, Any]):
        self.iv_snapshots = iv_snapshots
        self.scenario_table = scenario_table
        self.config = config

    def generate_dashboard(self, output_path: str):
        """
        Stream writes the dashboard directly to disk to minimize RAM usage.
        """
        logger.info(f"Starting stream write to {output_path}...")
        
        # 1. Fetch Plotly JS (Large string, fetch once)
        plotly_js = py_offline.get_plotlyjs()

        # 2. Header Parts
        header_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GTC Scenario Dashboard</title>
    <script type="text/javascript">{plotly_js}</script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f4f4f4; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 5px; }}
        h1 {{ color: #333; }}
        .meta-info {{ background: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 0.9em; }}
        .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 5px 5px 0 0; }}
        .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 16px; font-weight: bold; color: #555; }}
        .tab button:hover {{ background-color: #ddd; }}
        .tab button.active {{ background-color: #fff; border-bottom: 2px solid #007bff; color: #007bff; }}
        .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; background: #fff; min-height: 500px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .footer {{ margin-top: 30px; font-size: 0.8em; color: #777; text-align: center; }}
    </style>
</head>
<body>
<div class="container">
    <h1>GTC Limit Scenario Mapper</h1>
    <div class="meta-info">
        <strong>Order Type:</strong> {self.config['order_type'].upper()} | 
        <strong>Base Limit:</strong> {self.config['base_limit_price'] if self.config['base_limit_price'] else 'N/A'} |
        <strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    <div class="tab">
        <button class="tablinks active" onclick="openTab(event, 'Overview')">Overview</button>
        <button class="tablinks" onclick="openTab(event, 'Waterfalls')">Scenario Waterfalls</button>
        <button class="tablinks" onclick="openTab(event, 'Heatmap')">Heatmap</button>
    </div>
"""

        # 3. Open File and Write Stream
        try:
            with open(output_path, "w", encoding='utf-8') as f:
                # -- WRITE HEADER --
                f.write(header_html)
                
                # -- TAB 1: OVERVIEW --
                f.write('<div id="Overview" class="tabcontent" style="display: block;"><h3>Current IV Snapshots</h3><div id="table-container">')
                if not self.iv_snapshots.empty:
                    disp_df = self.iv_snapshots.copy()
                    if 'front_iv' in disp_df.columns: disp_df['front_iv'] = disp_df['front_iv'].apply(lambda x: f"{x:.2%}")
                    if 'back_iv' in disp_df.columns: disp_df['back_iv'] = disp_df['back_iv'].apply(lambda x: f"{x:.2%}")
                    if 'spot' in disp_df.columns: disp_df['spot'] = disp_df['spot'].apply(lambda x: f"{x:.2f}")
                    f.write(disp_df.to_html(classes="display", index=False))
                else:
                    f.write("<p>No data available.</p>")
                f.write('</div></div>')

                # -- TAB 2: WATERFALLS (Iterative Write) --
                f.write('<div id="Waterfalls" class="tabcontent"><h3>Multiplier Impact by Scenario</h3><div id="waterfall-charts">')
                
                unique_tickers = self.scenario_table['ticker'].unique()
                for ticker in unique_tickers:
                    # Filter data
                    t_data = self.scenario_table[self.scenario_table['ticker'] == ticker].copy()
                    t_data['delta'] = t_data['gtc_multiplier'] - 1.0
                    
                    # Create Single Figure
                    fig = go.Figure(go.Waterfall(
                        name=ticker, 
                        orientation="v",
                        measure=["relative"] * len(t_data),
                        x=t_data['scenario_id'],
                        textposition="outside",
                        text=[f"{x:+.2%}" for x in t_data['delta']],
                        y=t_data['delta'],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    fig.update_layout(
                        title=f"{ticker} - GTC Multiplier Deltas ({self.config['order_type']})",
                        showlegend=False,
                        height=400,
                        yaxis=dict(title="Multiplier Delta", tickformat=".1%"),
                        xaxis=dict(title="Scenario")
                    )
                    
                    # Write DIV immediately to file
                    # include_plotlyjs=False is critical here
                    f.write(f"<div style='margin-bottom: 50px;'>")
                    f.write(py_offline.plot(fig, include_plotlyjs=False, output_type='div'))
                    f.write("</div><hr>")
                    
                    # Clean up memory
                    del fig
                    del t_data
                    gc.collect()

                f.write('</div></div>')

                # -- TAB 3: HEATMAP --
                f.write('<div id="Heatmap" class="tabcontent"><h3>Global Scenario Heatmap (Multipliers)</h3><div id="heatmap-container">')
                
                if not self.scenario_table.empty:
                    pivot_df = self.scenario_table.pivot(index="scenario_id", columns="ticker", values="gtc_multiplier")
                    pfill_pivot = self.scenario_table.pivot(index="scenario_id", columns="ticker", values="expected_Pfill")
                    desc_pivot = self.scenario_table.pivot(index="scenario_id", columns="ticker", values="description")
                    
                    hover_text = []
                    for i, row in pivot_df.iterrows():
                        row_txt = []
                        for col in pivot_df.columns:
                            val = row[col]
                            pf = pfill_pivot.loc[i, col]
                            desc = desc_pivot.loc[i, col]
                            txt = f"Scenario: {i}<br>Desc: {desc}<br>Ticker: {col}<br>Mult: {val:.4f}<br>Fill: {pf}"
                            row_txt.append(txt)
                        hover_text.append(row_txt)

                    fig_hm = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        text=hover_text,
                        hoverinfo='text',
                        colorscale='RdBu_r' if self.config['order_type'] == 'debit' else 'RdBu',
                        zmid=1.0
                    ))
                    fig_hm.update_layout(title="Multiplier Heatmap", height=600)
                    
                    f.write(py_offline.plot(fig_hm, include_plotlyjs=False, output_type='div'))
                    
                f.write('</div></div>')

                # -- FOOTER & SCRIPT --
                f.write("""
    <div class="footer">Generated by STEGO Financial Framework | GTC Scenario Mapper</div>
</div>
<script>
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
        window.dispatchEvent(new Event('resize'));
    }
</script>
</body>
</html>
""")
        except IOError as e:
            logger.error(f"Failed to write dashboard file: {e}")
            raise

        logger.info(f"Dashboard successfully generated at: {output_path}")


# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GTC Limit Scenario Mapper")
    
    parser.add_argument('--tickers', type=str, default='SPY,QQQ,IWM', help='Comma-separated tickers')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Output directory')
    parser.add_argument('--lookback', type=float, default=1.0, help='Lookback in years')
    parser.add_argument('--risk-free-rate', type=float, default=0.04, help='Risk free rate (decimal)')
    parser.add_argument('--order-type', type=str, required=True, choices=['credit', 'debit'], help='Order type')
    parser.add_argument('--base-limit-price', type=float, default=None, help='Base limit price for absolute calcs')
    parser.add_argument('--html-filename', type=str, default='gtc_scenario_dashboard.html', help='HTML filename')

    args = parser.parse_args()
    ticker_list = [t.strip() for t in args.tickers.split(',')]
    
    # 1. Ingest
    ingestor = DataIngestion(output_dir=args.output_dir, lookback_years=args.lookback)
    ohlc_data = ingestor.get_ohlc_data(ticker_list)
    options_data = ingestor.get_options_snapshots(ticker_list)

    if not options_data:
        logger.error("No options data available. Exiting.")
        sys.exit(1)

    # 2. Analyze
    analyzer = FinancialAnalysis(
        underlying_data=ohlc_data,
        options_data=options_data,
        risk_free_rate=args.risk_free_rate,
        order_type=args.order_type,
        base_limit_price=args.base_limit_price
    )
    
    logger.info("Computing IV Snapshots...")
    iv_table = analyzer.compute_iv_snapshots()
    logger.info("Generating Scenarios...")
    scenario_table = analyzer.generate_scenarios()

    # 3. Render (Memory Optimized)
    config = {"order_type": args.order_type, "base_limit_price": args.base_limit_price}
    renderer = DashboardRenderer(iv_table, scenario_table, config)
    output_path = os.path.join(args.output_dir, args.html_filename)
    
    renderer.generate_dashboard(output_path)

if __name__ == "__main__":
    main()
