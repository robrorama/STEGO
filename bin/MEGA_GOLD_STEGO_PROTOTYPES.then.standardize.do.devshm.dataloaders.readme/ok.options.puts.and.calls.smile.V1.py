import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles all Data I/O operations including downloading, caching, and sanitizing.
    Strictly decouples data acquisition from analysis.
    """
    
    DATA_DIR = 'raw_data'

    def __init__(self):
        """Initialize the ingestion engine and verify storage existence."""
        self._ensure_directory()

    def _ensure_directory(self):
        """Verifies existence of local raw_data directory."""
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
            logger.info(f"Created local data directory: {self.DATA_DIR}/")

    def get_options_chain(self, ticker: str, expiry: str) -> pd.DataFrame:
        """
        Orchestrates the Check-Disk-First strategy.
        
        Args:
            ticker (str): The asset symbol (e.g., 'SPY').
            expiry (str): The expiration date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Sanitized options data.
        """
        file_path = os.path.join(self.DATA_DIR, f"{ticker}_{expiry}.csv")

        # 1. Check Disk
        if os.path.exists(file_path):
            logger.info(f"Cache Hit: Loading {ticker} ({expiry}) from disk.")
            df = pd.read_csv(file_path)
            # Re-sanitize after loading from CSV to ensure types/index are restored correctly
            return self._sanitize_df(df)

        # 2. Download from Network if not found
        logger.info(f"Cache Miss: Downloading {ticker} ({expiry}) from API.")
        df = self._download_data(ticker, expiry)
        
        # 3. Save to Disk
        df.to_csv(file_path) # Save with default index, will be set in sanitization
        logger.info(f"Persisted data to {file_path}")

        # 4. Sanitize and Return
        return self._sanitize_df(df)

    def _download_data(self, ticker_symbol: str, expiry: str) -> pd.DataFrame:
        """
        Connects to yfinance to fetch call and put chains.
        Includes rate limiting to prevent HTTP 429.
        """
        try:
            # Rate Limiting
            time.sleep(1) 
            
            tk = yf.Ticker(ticker_symbol)
            
            # Fetch the chain
            chain = tk.option_chain(expiry)
            calls = chain.calls
            puts = chain.puts
            
            # Tag them before merging
            calls['type'] = 'Call'
            puts['type'] = 'Put'
            
            # Combine
            df = pd.concat([calls, puts], axis=0)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download data for {ticker_symbol}: {e}")
            raise

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressive Data Sanitization.
        Normalizes yfinance quirks and ensures type safety.
        """
        # Create a working copy to avoid SettingWithCopy warnings on inputs
        clean_df = df.copy()

        # 1. MultiIndex Flattening
        if isinstance(clean_df.columns, pd.MultiIndex):
            clean_df.columns = ['_'.join(col).strip() for col in clean_df.columns.values]
            logger.debug("Flattened MultiIndex columns.")

        # 2. Strict Datetime Index
        # yfinance options data usually has 'lastTradeDate'. We will use that.
        target_date_col = 'lastTradeDate'
        
        if target_date_col in clean_df.columns:
            # Convert to datetime
            clean_df[target_date_col] = pd.to_datetime(clean_df[target_date_col])
            
            # 3. Timezone Stripping
            if clean_df[target_date_col].dt.tz is not None:
                clean_df[target_date_col] = clean_df[target_date_col].dt.tz_convert(None)
            
            # Set as index if currently RangeIndex
            if isinstance(clean_df.index, pd.RangeIndex):
                clean_df.set_index(target_date_col, inplace=True)
                clean_df.sort_index(inplace=True)
        
        # 4. Numeric Coercion
        numeric_cols = [
            'strike', 'lastPrice', 'bid', 'ask', 
            'volume', 'openInterest', 'impliedVolatility'
        ]
        
        for col in numeric_cols:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0)

        # Ensure type safety for contract symbol
        if 'contractSymbol' in clean_df.columns:
            clean_df['contractSymbol'] = clean_df['contractSymbol'].astype(str)

        return clean_df


class OptionsAnalysis:
    """
    Handles mathematical logic and visualization.
    Enforces Immutability of the source dataframe.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): The sanitized source of truth.
        """
        self.df = df

    def get_summary_stats(self):
        """Returns basic stats about the chain."""
        local_df = self.df.copy()
        return local_df.groupby('type')[['volume', 'openInterest', 'impliedVolatility']].mean()

    def plot_volatility_smile(self, ticker: str, expiry: str):
        """
        Generates a Volatility Smile plot (IV vs Strike).
        
        Requirements:
        - Immutable data usage.
        - Plotly Dark Template.
        - Distinct colors for Calls/Puts.
        """
        # Strict Rule: Deep copy to prevent state pollution
        local_df = self.df.copy()

        # Filter mechanism (though our DF is likely already filtered by ingestion, good practice)
        calls = local_df[local_df['type'] == 'Call']
        puts = local_df[local_df['type'] == 'Put']

        # Sort by strike for clean lines
        calls = calls.sort_values('strike')
        puts = puts.sort_values('strike')

        fig = go.Figure()

        # Add Calls Trace (Cyan)
        fig.add_trace(go.Scatter(
            x=calls['strike'],
            y=calls['impliedVolatility'],
            mode='lines+markers',
            name='Calls',
            line=dict(color='cyan', width=2),
            marker=dict(size=6, opacity=0.8)
        ))

        # Add Puts Trace (Magenta)
        fig.add_trace(go.Scatter(
            x=puts['strike'],
            y=puts['impliedVolatility'],
            mode='lines+markers',
            name='Puts',
            line=dict(color='magenta', width=2),
            marker=dict(size=6, opacity=0.8)
        ))

        # Styling
        fig.update_layout(
            title=f"Volatility Smile: {ticker} (Exp: {expiry})",
            xaxis_title="Strike Price ($)",
            yaxis_title="Implied Volatility",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # In a real script we might show or save. 
        # For this task, we will just return the figure object or show it.
        fig.show()


def get_next_valid_expiry(ticker: str):
    """Helper to get a valid expiry date to prevent 404s."""
    tk = yf.Ticker(ticker)
    if not tk.options:
        raise ValueError(f"No options data found for {ticker}")
    # Return the 3rd available expiry to ensure some liquidity/time value
    idx = min(2, len(tk.options) - 1)
    return tk.options[idx]

if __name__ == "__main__":
    # --- Configuration ---
    TICKER = "SPY"
    
    try:
        # 1. Pipeline Setup
        print(f"--- Starting Quantitative Analysis for {TICKER} ---")
        
        # Helper to get a real expiry date dynamically
        EXPIRY_DATE = get_next_valid_expiry(TICKER)
        print(f"Target Expiry: {EXPIRY_DATE}")

        # 2. Instantiate Ingestion Engine
        ingestion = DataIngestion()

        # 3. Acquire Data (Check Disk -> Download -> Save -> Sanitize)
        clean_data = ingestion.get_options_chain(TICKER, EXPIRY_DATE)
        
        print(f"Data Loaded. Rows: {len(clean_data)}")
        print(f"Index Type: {type(clean_data.index)}")
        
        # 4. Instantiate Analysis Engine
        # Pass the clean dataframe. Analysis engine treats it as Read-Only.
        analyzer = OptionsAnalysis(clean_data)

        # 5. Generate Insights
        stats = analyzer.get_summary_stats()
        print("\nSummary Statistics:")
        print(stats)

        # 6. Visualization
        print("\nGenerating Volatility Smile Plot...")
        analyzer.plot_volatility_smile(TICKER, EXPIRY_DATE)
        
        print("--- Analysis Complete ---")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
