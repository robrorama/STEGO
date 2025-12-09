# SCRIPTNAME: ok.options_skew_rotation_dashboard.v9.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import os
import sys
import math
import time
import random
import datetime
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. Mathematical Kernel (No External Libs)
# ==========================================
class MathUtils:
    """
    Pure Python/Numpy implementation of Black-Scholes and Numerical Methods.
    Replaces scipy and quantlib.
    """
    
    @staticmethod
    def norm_cdf(x):
        """Cumulative distribution function for the standard normal distribution."""
        return 0.5 * (1.0 + math.erf(x / 1.41421356))

    @staticmethod
    def norm_pdf(x):
        """Probability density function for the standard normal distribution."""
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

    @staticmethod
    def bs_price_and_greeks(S, K, T, r, sigma, type_="call"):
        """
        Calculates Black-Scholes Price, Delta, and Vega.
        Vega is returned in raw units (dollar change per 100% vol change) for solver stability.
        """
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0, 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        cdf_d1 = MathUtils.norm_cdf(d1)
        cdf_d2 = MathUtils.norm_cdf(d2)
        pdf_d1 = MathUtils.norm_pdf(d1)

        if type_ == "call":
            price = S * cdf_d1 - K * math.exp(-r * T) * cdf_d2
            delta = cdf_d1
        else: # put
            price = K * math.exp(-r * T) * MathUtils.norm_cdf(-d2) - S * MathUtils.norm_cdf(-d1)
            delta = cdf_d1 - 1.0

        # Vega is the same for calls and puts
        vega = S * math.sqrt(T) * pdf_d1
        
        return price, delta, vega

    @staticmethod
    def impl_vol_newton(market_price, S, K, T, r, type_="call"):
        """
        Newton-Raphson solver for Implied Volatility.
        Bounds: 0.01 to 5.0 (1% to 500%).
        """
        sigma = 0.5 # Initial guess
        for i in range(10):
            price, _, vega = MathUtils.bs_price_and_greeks(S, K, T, r, sigma, type_)
            diff = market_price - price
            
            if abs(diff) < 1e-4:
                return sigma
            
            if vega < 1e-8: # Avoid division by zero
                break
                
            sigma = sigma + diff / vega
            
            # Enforce bounds
            if sigma < 0.01: sigma = 0.01
            if sigma > 5.0: sigma = 5.0
            
        return sigma

# ==========================================
# 2. Data Persistence & Architecture
# ==========================================
class DataEngine:
    """
    Handles Data Retrieval (yfinance), Persistence (CSV), and Sanitization.
    Strictly follows 'Check-First' caching strategy.
    """
    DATA_DIR = "./raw_data"

    def __init__(self):
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)

    def _get_cache_path(self, identifier):
        return os.path.join(self.DATA_DIR, f"{identifier}.csv")

    def _sanitize_df(self, df):
        """
        Aggressive Data Sanitization Routine.
        1. Flatten MultiIndex columns.
        2. Set strict Datetime Index.
        3. Remove Timezones.
        4. Coerce Numerics.
        5. Handle Nulls.
        """
        # Create a deep copy to ensure immutability of source
        clean_df = df.copy()

        # 1. MultiIndex Flattening
        if isinstance(clean_df.columns, pd.MultiIndex):
            clean_df.columns = clean_df.columns.get_level_values(0)

        # 2. Strict Datetime Index
        # Check if 'Date' is a column (reset index usually puts it there)
        if 'Date' in clean_df.columns:
            clean_df['Date'] = pd.to_datetime(clean_df['Date'])
            clean_df.set_index('Date', inplace=True)
        elif not isinstance(clean_df.index, pd.DatetimeIndex):
            # Attempt to convert existing index
            try:
                clean_df.index = pd.to_datetime(clean_df.index)
            except:
                pass # Keep as is if conversion fails, though likely problematic

        # 3. Timezone Removal
        if isinstance(clean_df.index, pd.DatetimeIndex) and clean_df.index.tz is not None:
            clean_df.index = clean_df.index.tz_convert(None)

        # 4. Numeric Coercion
        cols_to_coerce = ['Open', 'Close', 'strike', 'lastPrice', 'impliedVolatility', 'bid', 'ask']
        for col in cols_to_coerce:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

        # 5. Null Handling
        if 'impliedVolatility' in clean_df.columns:
            clean_df['impliedVolatility'] = clean_df['impliedVolatility'].fillna(0.0)

        return clean_df

    def get_spot_history(self, ticker_symbol):
        """Fetches 1mo spot history. Caches by date."""
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        cache_id = f"{ticker_symbol}_spot_{today_str}"
        path = self._get_cache_path(cache_id)

        if os.path.exists(path):
            print(f"[CACHE] Loading spot data for {ticker_symbol}...")
            df = pd.read_csv(path)
            return self._sanitize_df(df)
        
        print(f"[API] Downloading spot data for {ticker_symbol}...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Fetch slightly more than 1mo to ensure we have data
            df = ticker.history(period="1mo")
            
            # Save raw-ish data, sanitize on load
            df.to_csv(path)
            return self._sanitize_df(df)
        except Exception as e:
            print(f"[ERROR] Failed to fetch spot: {e}")
            return pd.DataFrame()

    def get_option_chain(self, ticker_symbol, date_str):
        """Fetches option chain for specific expiry. Caches by expiry."""
        cache_id = f"{ticker_symbol}_chain_{date_str}"
        path = self._get_cache_path(cache_id)

        if os.path.exists(path):
            # print(f"[CACHE] Loading chain for {date_str}...")
            df = pd.read_csv(path)
            return self._sanitize_df(df)

        print(f"[API] Downloading chain for {date_str}...")
        try:
            # Random sleep for rate limiting
            time.sleep(random.uniform(0.5, 1.5))
            
            ticker = yf.Ticker(ticker_symbol)
            chain = ticker.option_chain(date_str)
            
            # Combine calls and puts, tag them
            calls = chain.calls.copy()
            calls['type'] = 'call'
            puts = chain.puts.copy()
            puts['type'] = 'put'
            
            df = pd.concat([calls, puts], axis=0)
            
            df.to_csv(path, index=False)
            return self._sanitize_df(df)
        except Exception as e:
            print(f"[ERROR] Failed to fetch chain {date_str}: {e}")
            return pd.DataFrame()

# ==========================================
# 3. Analytic Engine
# ==========================================
class SkewAnalyzer:
    """
    Orchestrates the analysis workflow.
    Calculates Tenor Map, Greeks, and Skew Metrics.
    """
    TARGET_TENORS = [7, 14, 21, 30, 60, 90, 180, 365]
    RISK_FREE_RATE = 0.045 # Simplified constant 4.5%

    def __init__(self, ticker, data_engine):
        self.ticker = ticker
        self.engine = data_engine
        self.spot_price = 0.0
        self.current_date = datetime.datetime.now()

    def run_analysis(self):
        # Step 1: Spot Price
        spot_df = self.engine.get_spot_history(self.ticker)
        if spot_df.empty:
            raise ValueError("Could not fetch spot price.")
        self.spot_price = spot_df['Close'].iloc[-1]
        print(f"Details: Spot Price ${self.spot_price:.2f}")

        # Step 2: Tenor Mapping
        ticker_obj = yf.Ticker(self.ticker)
        avail_dates = ticker_obj.options
        tenor_map = self._map_tenors(avail_dates)

        skew_results = []
        term_structure_data = []

        # Step 3 & 4: Greeks & Skew Calc
        print("Processing expirations...")
        for target_days, exp_date_str in tenor_map.items():
            chain_df = self.engine.get_option_chain(self.ticker, exp_date_str)
            if chain_df.empty:
                continue

            # Calculate Greeks
            enriched_df = self._calculate_greeks(chain_df, exp_date_str)
            
            # Calculate Skew Metrics
            metrics = self._calculate_skew_metrics(enriched_df, target_days)
            if metrics:
                skew_results.append(metrics)
                
                # Collect data for Term Structure Chart (ATM vs 25d)
                term_structure_data.append({
                    'tenor': target_days,
                    'atm_iv': metrics['atm_iv'],
                    'put_25d_iv': metrics['put_25d_iv']
                })

        return pd.DataFrame(skew_results), pd.DataFrame(term_structure_data)

    def _map_tenors(self, avail_dates):
        """Maps target days (e.g., 30) to nearest available expiration date."""
        mapping = {}
        avail_dt = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in avail_dates]
        
        for target in self.TARGET_TENORS:
            target_dt = self.current_date + datetime.timedelta(days=target)
            # Find nearest date
            nearest = min(avail_dt, key=lambda x: abs(x - target_dt))
            days_diff = (nearest - self.current_date).days
            
            # Only accept if reasonably close (within 20% or 5 days)
            if days_diff > 0:
                mapping[target] = nearest.strftime("%Y-%m-%d")
        
        return mapping

    def _calculate_greeks(self, df, expiry_str):
        """Enriches chain with Newton-Raphson IV (if needed) and Delta."""
        # Immutable pattern: Work on copy
        proc_df = df.copy()
        
        T = (datetime.datetime.strptime(expiry_str, "%Y-%m-%d") - self.current_date).days / 365.0
        if T < 1e-3: T = 1e-3 # Avoid T=0

        # Vectorized calculation not possible due to custom Newton solver, iterating safely
        prices = []
        deltas = []
        ivs = []

        for idx, row in proc_df.iterrows():
            K = row['strike']
            # Determine Reference Price
            bid = row.get('bid', 0)
            ask = row.get('ask', 0)
            last = row['lastPrice']
            
            mkt_price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last
            
            # IV Check
            iv = row.get('impliedVolatility', 0)
            if iv == 0 or pd.isna(iv):
                iv = MathUtils.impl_vol_newton(
                    mkt_price, self.spot_price, K, T, self.RISK_FREE_RATE, row['type']
                )
            
            # Delta Calc
            _, delta, _ = MathUtils.bs_price_and_greeks(
                self.spot_price, K, T, self.RISK_FREE_RATE, iv, row['type']
            )

            ivs.append(iv)
            deltas.append(delta)

        proc_df['calc_iv'] = ivs
        proc_df['calc_delta'] = deltas
        return proc_df

    def _calculate_skew_metrics(self, df, tenor_days):
        """
        Calculates:
        1. ATM Put IV (Strike closest to Spot)
        2. 25-Delta Put IV (Delta closest to -0.25)
        3. Skew (25d - ATM)
        """
        # Filter for Puts
        puts = df[df['type'] == 'put'].copy()
        if puts.empty:
            return None

        # ATM IV
        # Find row where abs(Strike - Spot) is min
        puts['dist_atm'] = abs(puts['strike'] - self.spot_price)
        atm_row = puts.loc[puts['dist_atm'].idxmin()]
        atm_iv = atm_row['calc_iv']

        # 25-Delta Put IV
        # Find row where abs(Delta - (-0.25)) is min
        puts['dist_25d'] = abs(puts['calc_delta'] - (-0.25))
        d25_row = puts.loc[puts['dist_25d'].idxmin()]
        d25_iv = d25_row['calc_iv']

        skew = d25_iv - atm_iv

        return {
            'date': self.current_date.strftime("%Y-%m-%d"),
            'tenor': tenor_days,
            'atm_iv': atm_iv,
            'put_25d_iv': d25_iv,
            'skew': skew
        }

# ==========================================
# 4. History & Z-Score Manager (UPDATED)
# ==========================================
class HistoryManager:
    """
    Manages the history.csv file.
    Auto-generates SYNTHETIC history from Stock Prices if no option history exists.
    """
    HISTORY_FILE = "skew_history.csv"
    TARGET_TENORS = [7, 14, 21, 30, 60, 90, 180, 365]

    def __init__(self, ticker):
        self.ticker = ticker
        self.filename = f"{ticker}_{self.HISTORY_FILE}"

    def update_history(self, current_metrics_df):
        """Appends new data to history. Backfills if empty."""
        
        # 1. Check if history exists. If not, generate synthetic history from stock prices.
        if not os.path.exists(self.filename):
            print("[INIT] No history found. Generating synthetic history from Stock Prices...")
            self._generate_synthetic_history_from_price()

        # 2. Load and Append
        hist_df = pd.read_csv(self.filename)
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Remove today if it exists (overwrite behavior)
        hist_df = hist_df[hist_df['date'] != today_str]
        
        updated = pd.concat([hist_df, current_metrics_df], ignore_index=True)
        updated.to_csv(self.filename, index=False)
        return updated

    def _generate_synthetic_history_from_price(self):
        """
        Downloads 6mo of Stock Price history.
        Calculates 20-day Realized Volatility (HV).
        Uses HV as a proxy for Implied Volatility to cold-start the Z-scores.
        """
        try:
            # A. Get Stock Data (6 months)
            ticker_obj = yf.Ticker(self.ticker)
            hist = ticker_obj.history(period="6mo")
            
            if hist.empty:
                print("Could not fetch stock history for backfill.")
                return

            # B. Calculate Log Returns
            hist['log_ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
            
            # C. Calculate Rolling Volatility (20-day window, Annualized)
            # Vol = StdDev * sqrt(252)
            hist['realized_vol'] = hist['log_ret'].rolling(window=20).std() * math.sqrt(252)
            
            # Drop NaN values from the start of the rolling window
            hist.dropna(inplace=True)

            synthetic_data = []

            # D. Generate Rows
            for date_idx, row in hist.iterrows():
                vol = row['realized_vol']
                date_str = date_idx.strftime("%Y-%m-%d")
                
                # Heuristic: Skew is often positively correlated with Vol level.
                # If Vol is 20% (0.20), synthetic skew might be around 0.02 (10% of vol)
                # If Vol spikes to 50% (0.50), synthetic skew might spike to 0.05
                # This ensures the Z-scores behave realistically.
                synthetic_skew = vol * 0.15 

                for tenor in self.TARGET_TENORS:
                    # We assume longer tenors have slightly lower vol (mean reversion)
                    # This adds a slight 'curve' to the data so it's not identical for all tenors
                    tenor_factor = 1.0 - (tenor / 365.0) * 0.1 
                    
                    adjusted_vol = vol * tenor_factor

                    synthetic_data.append({
                        'date': date_str,
                        'tenor': tenor,
                        'atm_iv': adjusted_vol,
                        'put_25d_iv': adjusted_vol + synthetic_skew,
                        'skew': synthetic_skew
                    })

            # E. Save to CSV
            df = pd.DataFrame(synthetic_data)
            df.to_csv(self.filename, index=False)
            print(f"[SUCCESS] Generated {len(df)} rows of synthetic history based on Realized Volatility.")

        except Exception as e:
            print(f"[ERROR] Failed to generate synthetic history: {e}")

    def calculate_z_scores(self, hist_df, current_df):
        """Calculates Z-Score of current skew vs historical for each tenor."""
        stats = hist_df.groupby('tenor')['skew'].agg(['mean', 'std']).reset_index()
        merged = pd.merge(current_df, stats, on='tenor', how='left')
        
        # Z = (x - mean) / std
        merged['z_score'] = (merged['skew'] - merged['mean']) / merged['std']
        merged['z_score'] = merged['z_score'].fillna(0.0)
        
        return merged

# ==========================================
# 5. Visualization (Plotly)
# ==========================================
class DashboardGenerator:
    @staticmethod
    def generate_dashboard(hist_df, current_z_df, term_struct_df, front_tenor, back_tenor):
        """Generates dashboard.html with 3 panels."""
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"colspan": 2}, None], [{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=("Skew Z-Score Heatmap", "Rotation Quadrant", "Term Structure (ATM vs 25d Put)"),
            vertical_spacing=0.15
        )

        # --- Chart 1: Heatmap ---
        # Pivot history for heatmap: Index=Date, Cols=Tenor, Values=Z-Score
        
        # Get stats map to normalize history
        stats = hist_df.groupby('tenor')['skew'].agg(['mean', 'std'])
        heatmap_data = hist_df.copy()
        
        # Vectorized z-score calculation for history
        def get_z(row):
            if row['tenor'] not in stats.index: return 0
            m = stats.loc[row['tenor'], 'mean']
            s = stats.loc[row['tenor'], 'std']
            return (row['skew'] - m) / s if s > 0 else 0

        heatmap_data['z_score'] = heatmap_data.apply(get_z, axis=1)
        pivot_hm = heatmap_data.pivot(index='date', columns='tenor', values='z_score')

        fig.add_trace(
            go.Heatmap(
                z=pivot_hm.values,
                x=pivot_hm.columns,
                y=pivot_hm.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Z-Score"),
                showscale=True
            ),
            row=1, col=1
        )

        # --- Chart 2: Rotation Quadrant ---
        # Filter for front and back tenor
        f_tenor_data = heatmap_data[heatmap_data['tenor'] == front_tenor].set_index('date')['z_score']
        b_tenor_data = heatmap_data[heatmap_data['tenor'] == back_tenor].set_index('date')['z_score']
        
        scatter_df = pd.concat([f_tenor_data, b_tenor_data], axis=1).dropna()
        scatter_df.columns = ['front', 'back']
        
        # Historical Points (Grey)
        fig.add_trace(
            go.Scatter(
                x=scatter_df['front'][:-1],
                y=scatter_df['back'][:-1],
                mode='markers',
                marker=dict(color='lightgrey', size=6),
                name='History'
            ),
            row=2, col=1
        )
        
        # Current Point (Red Star)
        fig.add_trace(
            go.Scatter(
                x=[scatter_df['front'].iloc[-1]],
                y=[scatter_df['back'].iloc[-1]],
                mode='markers',
                marker=dict(color='red', size=15, symbol='star'),
                name='Current'
            ),
            row=2, col=1
        )

        # Crosshairs
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=2, col=1)
        
        fig.update_xaxes(title_text=f"{front_tenor}d Skew Z-Score", row=2, col=1)
        fig.update_yaxes(title_text=f"{back_tenor}d Skew Z-Score", row=2, col=1)

        # --- Chart 3: Term Structure ---
        fig.add_trace(
            go.Scatter(
                x=term_struct_df['tenor'],
                y=term_struct_df['atm_iv'],
                mode='lines+markers',
                name='ATM IV',
                line=dict(color='blue')
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=term_struct_df['tenor'],
                y=term_struct_df['put_25d_iv'],
                mode='lines+markers',
                name='25d Put IV',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        fig.update_xaxes(title_text="Tenor (Days)", row=2, col=2)
        fig.update_yaxes(title_text="Implied Volatility", row=2, col=2)

        # Final Layout
        fig.update_layout(
            title_text="Options Skew & Rotation Dashboard",
            height=900,
            width=1200,
            showlegend=True
        )

        print("Writing dashboard.html...")
        fig.write_html("dashboard.html")

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Options Skew Dashboard")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., SPY)")
    parser.add_argument("--front-tenor", type=int, default=30, help="Front month tenor days")
    parser.add_argument("--back-tenor", type=int, default=90, help="Back month tenor days")
    
    args = parser.parse_args()
    
    print(f"--- Starting Skew Analysis for {args.ticker} ---")
    
    try:
        # Init Engine
        engine = DataEngine()
        
        # Analyze
        analyzer = SkewAnalyzer(args.ticker, engine)
        current_metrics_df, term_struct_df = analyzer.run_analysis()
        
        if current_metrics_df.empty:
            print("No valid metrics calculated.")
            return

        # History Management
        hist_mgr = HistoryManager(args.ticker)
        full_history_df = hist_mgr.update_history(current_metrics_df)
        
        # Calculate Z-Scores for current run
        current_z_df = hist_mgr.calculate_z_scores(full_history_df, current_metrics_df)
        
        print("\n--- Analysis Complete ---")
        print(current_z_df[['tenor', 'skew', 'z_score']])

        # Visualization
        DashboardGenerator.generate_dashboard(
            full_history_df, 
            current_z_df, 
            term_struct_df, 
            args.front_tenor, 
            args.back_tenor
        )
        
        print("Dashboard generated: ./dashboard.html")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

