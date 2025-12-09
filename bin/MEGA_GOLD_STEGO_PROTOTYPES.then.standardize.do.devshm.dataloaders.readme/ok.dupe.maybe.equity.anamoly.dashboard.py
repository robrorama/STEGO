import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, List

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ==========================================
# 0. Configuration & Logger
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = "raw_data"
COLORS = {
    "cyan": "#00E5FF",
    "pink": "#FF4081",
    "lime": "#CCFF00",
    "bg": "#111111"
}

# ==========================================
# 1. Quantitative Library (Math Helpers)
# ==========================================
class QuantitativeLib:
    """
    Stateless library for Options Greeks and generic math.
    """
    
    @staticmethod
    def norm_cdf(x: float) -> float:
        return norm.cdf(x)

    @staticmethod
    def norm_pdf(x: float) -> float:
        return norm.pdf(x)

    @staticmethod
    def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = 'call') -> float:
        """Black-Scholes Price."""
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K) if kind == 'call' else max(0.0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if kind == 'call':
            return S * QuantitativeLib.norm_cdf(d1) - K * np.exp(-r * T) * QuantitativeLib.norm_cdf(d2)
        else:
            return K * np.exp(-r * T) * QuantitativeLib.norm_cdf(-d2) - S * QuantitativeLib.norm_cdf(-d1)

    @staticmethod
    def impl_vol_newton(market_price: float, S: float, K: float, T: float, r: float, kind: str = 'call') -> float:
        """Calculate Implied Volatility using Newton-Raphson."""
        sigma = 0.5  # Initial guess
        for i in range(100):
            price = QuantitativeLib.bs_price(S, K, T, r, sigma, kind)
            diff = market_price - price
            
            if abs(diff) < 1e-5:
                return sigma
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = S * QuantitativeLib.norm_pdf(d1) * np.sqrt(T)
            
            if vega == 0:
                break
                
            sigma = sigma + diff / vega
            
        return np.nan  # Failed to converge

# ==========================================
# 2. Sanitization Layer
# ==========================================
class DataSanitizer:
    """
    Aggressively cleanses data to ensure type safety and index consistency.
    """
    @staticmethod
    def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        
        df_clean = df.copy()

        # 1. MultiIndex Flattening
        if isinstance(df_clean.columns, pd.MultiIndex):
            # If columns are like (Price, Ticker), drop the Ticker level
            try:
                # Assuming standard yfinance multi-index: Level 0 = Price Type, Level 1 = Ticker
                if df_clean.columns.nlevels > 1:
                    df_clean.columns = df_clean.columns.get_level_values(0)
            except Exception as e:
                logger.warning(f"Sanitization Warning (MultiIndex): {e}")

        # 2. Index Normalization
        # If it's a RangeIndex (0,1,2), try to find Date column
        if isinstance(df_clean.index, pd.RangeIndex):
            if 'Date' in df_clean.columns:
                df_clean.set_index('Date', inplace=True)
            elif 'date' in df_clean.columns:
                df_clean.set_index('date', inplace=True)

        # Coerce Index to Datetime
        try:
            df_clean.index = pd.to_datetime(df_clean.index, utc=True)
        except Exception as e:
            logger.warning(f"Index coercion failed: {e}")

        # Drop NaT indices
        df_clean = df_clean[df_clean.index.notna()]

        # 3. Timezone Stripping
        if df_clean.index.tz is not None:
            df_clean.index = df_clean.index.tz_localize(None)

        # 4. Numeric Coercion
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Drop rows where critical data is NaN
        if 'Close' in df_clean.columns:
            df_clean.dropna(subset=['Close'], inplace=True)

        return df_clean

# ==========================================
# 3. Persistence Layer
# ==========================================
class MarketDataManager:
    """
    Manages data acquisition from YFinance and local CSV caching.
    """
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self._ensure_directory()
        self._prices: Optional[pd.DataFrame] = None
        self._options: Optional[pd.DataFrame] = None
        self._earnings: Optional[pd.DataFrame] = None
        self.yf_ticker_obj = yf.Ticker(self.ticker)

    def _ensure_directory(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def load_data(self):
        """Orchestrates loading for Prices, Options, and Earnings."""
        self._load_prices()
        self._load_options()
        self._load_earnings()

    def _load_prices(self):
        file_path = os.path.join(CACHE_DIR, f"{self.ticker}_price.csv")
        
        # Check Disk
        if os.path.exists(file_path):
            logger.info(f"[{self.ticker}] Loading prices from cache.")
            raw_df = pd.read_csv(file_path)
            # Immediate Sanitization
            self._prices = DataSanitizer._sanitize_df(raw_df)
            return

        # Download on Miss
        logger.info(f"[{self.ticker}] Downloading prices from API.")
        try:
            # Download 2 years of data for 12m momentum calculations
            raw_df = yf.download(self.ticker, period="2y", interval="1d", progress=False)
            
            # Persistence
            if not raw_df.empty:
                # We save raw first to preserve structure, but for yfinance v0.2+ 
                # saving directly might save the MultiIndex structure which read_csv needs to handle.
                # To be safe, we reset index to ensure Date is a column in CSV.
                raw_df.to_csv(file_path)
                self._prices = DataSanitizer._sanitize_df(raw_df)
            else:
                logger.error(f"[{self.ticker}] No price data found.")
        except Exception as e:
            logger.error(f"[{self.ticker}] Price download failed: {e}")

    def _load_options(self):
        """Rate-limited options loading for next expiration."""
        try:
            exps = self.yf_ticker_obj.options
            if not exps:
                logger.warning(f"[{self.ticker}] No options expirations found.")
                return

            # Filter for next available future expiration
            today_str = datetime.now().strftime('%Y-%m-%d')
            future_exps = [e for e in exps if e > today_str]
            if not future_exps:
                return
            
            target_exp = future_exps[0]
            file_path = os.path.join(CACHE_DIR, f"{self.ticker}_options_{target_exp}.csv")

            if os.path.exists(file_path):
                logger.info(f"[{self.ticker}] Loading options ({target_exp}) from cache.")
                raw_df = pd.read_csv(file_path)
                self._options = raw_df # Options don't use standard sanitizer usually
            else:
                logger.info(f"[{self.ticker}] Downloading options for {target_exp}.")
                time.sleep(1) # Rate limiting
                opt_chain = self.yf_ticker_obj.option_chain(target_exp)
                
                # Combine calls and puts
                calls = opt_chain.calls.copy()
                calls['type'] = 'call'
                puts = opt_chain.puts.copy()
                puts['type'] = 'put'
                
                raw_df = pd.concat([calls, puts])
                raw_df['expirationDate'] = target_exp
                
                # Save
                raw_df.to_csv(file_path, index=False)
                self._options = raw_df

            # Type Casting for critical columns
            if self._options is not None:
                cols_to_num = ['impliedVolatility', 'openInterest', 'strike', 'lastPrice', 'bid', 'ask']
                for c in cols_to_num:
                    if c in self._options.columns:
                        self._options[c] = pd.to_numeric(self._options[c], errors='coerce')

        except Exception as e:
            logger.error(f"[{self.ticker}] Options load failed: {e}")

    def _load_earnings(self):
        """Attempt to load earnings dates. Not cached as deeply due to infrequency."""
        try:
            self._earnings = self.yf_ticker_obj.earnings_dates
        except Exception as e:
            logger.warning(f"[{self.ticker}] Earnings download failed: {e}")

    # Getters (Immutability Enforced)
    def get_prices(self) -> pd.DataFrame:
        return self._prices.copy() if self._prices is not None else pd.DataFrame()

    def get_options(self) -> pd.DataFrame:
        return self._options.copy() if self._options is not None else pd.DataFrame()

    def get_earnings(self) -> pd.DataFrame:
        return self._earnings.copy() if self._earnings is not None else pd.DataFrame()

# ==========================================
# 4. Analysis Logic
# ==========================================
class StegoAnalyzer:
    """
    Stateless analysis logic. Never modifies inputs.
    """
    
    @staticmethod
    def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """Returns last row of momentum factors."""
        if df.empty or 'Close' not in df.columns:
            return pd.DataFrame()
        
        # Work on a copy
        res = pd.DataFrame(index=df.index)
        close = df['Close']
        
        # 3M (63d), 6M (126d), 12M (252d)
        res['Mom_3M'] = close.pct_change(63)
        res['Mom_6M'] = close.pct_change(126)
        res['Mom_12M'] = close.pct_change(252)
        res['Close'] = close
        
        return res.dropna().tail(252) # Return last year of data for plotting

    @staticmethod
    def analyze_weekly_reversal(df: pd.DataFrame) -> Tuple[float, float]:
        """
        Weekly Reversal: Past 5d vs Next 5d.
        Returns (Mean Return Up Group, Mean Return Down Group).
        """
        if df.empty or len(df) < 15:
            return (0.0, 0.0)

        data = df[['Close']].copy()
        # Log returns
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Rolling 5d sum of log returns
        data['past_5d'] = data['log_ret'].rolling(window=5).sum()
        # Next 5d is past_5d shifted back by 5
        data['next_5d'] = data['past_5d'].shift(-5)
        
        data.dropna(inplace=True)
        
        # Classify
        up_mask = data['past_5d'] > 0
        down_mask = data['past_5d'] <= 0
        
        mean_next_ret_up = data.loc[up_mask, 'next_5d'].mean()
        mean_next_ret_down = data.loc[down_mask, 'next_5d'].mean()
        
        # Convert back to simple returns for display
        return (np.exp(mean_next_ret_up)-1, np.exp(mean_next_ret_down)-1)

    @staticmethod
    def analyze_tom(df: pd.DataFrame) -> Tuple[float, float]:
        """
        Turn of Month Analysis.
        Last 5 trading days vs Rest of Month.
        """
        if df.empty:
            return (0.0, 0.0)

        data = df[['Close']].copy()
        data['pct_ret'] = data['Close'].pct_change()
        data.dropna(inplace=True)

        # Identify TOM: Last 5 days of each month
        # Logic: Group by Y-M, select last 5 indices
        tom_indices = []
        
        # Group by Year and Month
        grouped = data.groupby([data.index.year, data.index.month])
        
        for _, group in grouped:
            if len(group) >= 5:
                tom_indices.extend(group.index[-5:])
            else:
                tom_indices.extend(group.index) # If month is short, take all
        
        tom_mask = data.index.isin(tom_indices)
        
        tom_mean = data.loc[tom_mask, 'pct_ret'].mean()
        rom_mean = data.loc[~tom_mask, 'pct_ret'].mean() # Rest of month
        
        # Annualize roughly (x252) for display impact or keep daily mean
        # Keeping daily mean for comparison
        return (tom_mean, rom_mean)

    @staticmethod
    def analyze_pead(prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
        """
        Post Earnings Announcement Drift.
        (Price T+20 / Price T0) - 1
        """
        if prices.empty or earnings.empty:
            return pd.DataFrame(columns=['Date', 'Surprise', 'Drift'])

        # Sanitize earnings index
        if earnings.index.tz is not None:
            earnings.index = earnings.index.tz_localize(None)

        # Filter earnings in range of prices
        min_date = prices.index.min()
        max_date = prices.index.max()
        valid_earnings = earnings[(earnings.index >= min_date) & (earnings.index <= max_date)].copy()
        
        if 'Surprise(%)' not in valid_earnings.columns:
             # Try to recover or create dummy if missing (yfinance format varies)
             valid_earnings['Surprise(%)'] = 0.0

        results = []

        for date, row in valid_earnings.iterrows():
            # Find closest trading day to earnings date (T0)
            # Use searchsorted to find insertion point
            idx_loc = prices.index.searchsorted(date)
            
            if idx_loc < len(prices):
                t0_date = prices.index[idx_loc]
                
                # We need T+20 trading days
                if idx_loc + 20 < len(prices):
                    t20_date = prices.index[idx_loc + 20]
                    
                    p0 = prices.loc[t0_date, 'Close']
                    p20 = prices.loc[t20_date, 'Close']
                    
                    drift = (p20 / p0) - 1
                    surprise = row['Surprise(%)'] if pd.notna(row['Surprise(%)']) else 0.0
                    
                    results.append({
                        'Date': date,
                        'Surprise': surprise,
                        'Drift': drift
                    })

        return pd.DataFrame(results)

# ==========================================
# 5. Dashboard Renderer
# ==========================================
class DashboardRenderer:
    def __init__(self, ticker: str):
        self.ticker = ticker

    def generate_dashboard(self, 
                           mom_df: pd.DataFrame,
                           reversal_res: Tuple[float, float],
                           tom_res: Tuple[float, float],
                           pead_df: pd.DataFrame,
                           options_df: pd.DataFrame,
                           spot_price: float):
        
        fig = make_subplots(
            rows=4, cols=2,
            specs=[
                [{"colspan": 2, "secondary_y": True}, None], # Row 1: Momentum
                [{}, {}],                                    # Row 2: Reversal | TOM
                [{"colspan": 2}, None],                      # Row 3: PEAD
                [{"colspan": 2, "type": "table"}, None]      # Row 4: Options
            ],
            subplot_titles=(
                f"{self.ticker} Momentum (Price vs Factors)", 
                "Weekly Reversal (Next 5d Return)", 
                "Turn of Month (Daily Mean)", 
                "PEAD Analysis (Drift T+20)", 
                "ATM Options Chain"
            ),
            vertical_spacing=0.08
        )

        # ----------------------
        # Row 1: Momentum
        # ----------------------
        if not mom_df.empty:
            # Price (Secondary Y)
            fig.add_trace(go.Scatter(
                x=mom_df.index, y=mom_df['Close'], name='Price',
                line=dict(color='white', width=1, dash='dot'), opacity=0.5
            ), row=1, col=1, secondary_y=True)

            # Factors (Primary Y)
            fig.add_trace(go.Scatter(x=mom_df.index, y=mom_df['Mom_3M'], name='3M Mom', line=dict(color=COLORS['cyan'])), row=1, col=1)
            fig.add_trace(go.Scatter(x=mom_df.index, y=mom_df['Mom_6M'], name='6M Mom', line=dict(color=COLORS['pink'])), row=1, col=1)
            fig.add_trace(go.Scatter(x=mom_df.index, y=mom_df['Mom_12M'], name='12M Mom', line=dict(color=COLORS['lime'])), row=1, col=1)

        # ----------------------
        # Row 2: Reversal & TOM
        # ----------------------
        # Reversal
        fig.add_trace(go.Bar(
            x=['After Up 5d', 'After Down 5d'], 
            y=[reversal_res[0], reversal_res[1]],
            marker_color=[COLORS['pink'], COLORS['lime']],
            name='Wk Reversal'
        ), row=2, col=1)

        # TOM
        fig.add_trace(go.Bar(
            x=['TOM Days', 'Rest of Month'], 
            y=[tom_res[0], tom_res[1]],
            marker_color=[COLORS['lime'], COLORS['cyan']],
            name='TOM Effect'
        ), row=2, col=2)

        # ----------------------
        # Row 3: PEAD
        # ----------------------
        if not pead_df.empty:
            colors = pead_df['Surprise'].apply(lambda x: COLORS['lime'] if x > 0 else COLORS['pink'])
            fig.add_trace(go.Scatter(
                x=pead_df['Date'], 
                y=pead_df['Drift'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color=colors),
                text=pead_df['Surprise'],
                name='PEAD Drift'
            ), row=3, col=1)
            
            # Add Zero Line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        # ----------------------
        # Row 4: ATM Options Snapshot
        # ----------------------
        if not options_df.empty and spot_price > 0:
            # Filter for near ATM (within 5%)
            lower_b = spot_price * 0.95
            upper_b = spot_price * 1.05
            atm_df = options_df[(options_df['strike'] >= lower_b) & (options_df['strike'] <= upper_b)].copy()
            
            # Sort by strike and type
            atm_df.sort_values(by=['strike', 'type'], inplace=True)
            
            # Recalculate IV if 0 using QuantitativeLib (Demonstration)
            # Assuming r=0.05, T=30/365 (Approx for next month)
            r = 0.05
            T = 30/365.0 
            
            # Vectorized calc for display table
            vals = []
            for _, row in atm_df.iterrows():
                iv = row['impliedVolatility']
                if pd.isna(iv) or iv < 0.001:
                    # Manual Calc
                    iv = QuantitativeLib.impl_vol_newton(
                        row['lastPrice'], spot_price, row['strike'], T, r, row['type']
                    )
                vals.append(round(iv, 4) if pd.notna(iv) else 0.0)
            
            atm_df['Manual_IV'] = vals

            fig.add_trace(go.Table(
                header=dict(values=['Type', 'Strike', 'Last Price', 'Vol', 'Open Int', 'YF IV', 'Calc IV'],
                            fill_color=COLORS['bg'], font=dict(color='white')),
                cells=dict(values=[
                    atm_df['type'], atm_df['strike'], atm_df['lastPrice'], 
                    atm_df['volume'], atm_df['openInterest'], 
                    atm_df['impliedVolatility'].round(4), atm_df['Manual_IV']
                ],
                fill_color='#222222', font=dict(color='white'))
            ), row=4, col=1)

        # ----------------------
        # Layout & Styling
        # ----------------------
        fig.update_layout(
            template="plotly_dark",
            title_text=f"STEGO ANOMALY DASHBOARD: {self.ticker}",
            height=1200,
            showlegend=False,
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg']
        )
        
        filename = f"{self.ticker}_anomaly_dashboard.html"
        fig.write_html(filename)
        logger.info(f"Dashboard saved to {filename}")
        
        # Auto open
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(filename))

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Stego Anomalies Dashboard Generator")
    parser.add_argument("tickers", nargs='+', help="List of ticker symbols (e.g., AAPL MSFT)")
    args = parser.parse_args()

    print("=== STEGO ANOMALY DETECTOR INITIALIZED ===")
    
    for ticker in args.tickers:
        try:
            print(f"\nProcessing {ticker}...")
            
            # 1. Initialize Manager & Load Data
            manager = MarketDataManager(ticker)
            manager.load_data()
            
            prices = manager.get_prices()
            options = manager.get_options()
            earnings = manager.get_earnings()
            
            if prices.empty:
                logger.error(f"Skipping {ticker}: No price data available.")
                continue

            # 2. Analyze
            analyzer = StegoAnalyzer()
            
            # Metrics
            mom_df = analyzer.calculate_momentum(prices)
            wk_rev = analyzer.analyze_weekly_reversal(prices)
            tom_res = analyzer.analyze_tom(prices)
            pead_df = analyzer.analyze_pead(prices, earnings)
            
            current_spot = prices['Close'].iloc[-1]
            
            # 3. Render
            renderer = DashboardRenderer(ticker)
            renderer.generate_dashboard(
                mom_df, wk_rev, tom_res, pead_df, options, current_spot
            )
            
        except Exception as e:
            logger.error(f"Critical failure processing {ticker}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
