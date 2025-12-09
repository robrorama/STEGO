# -----------------------------------------------------------------------------
# SCRIPT: stego_streak_dashboard.py
# AUTHOR: Michael Derby
# FRAMEWORK: STEGO Financial Framework
# DATE: 2025-11-29
# PURPOSE: Interactive Streak Statistical Dashboard & Dashboard Generator
# -----------------------------------------------------------------------------

import os
import sys
import argparse
import webbrowser
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
from datetime import datetime, timedelta
from scipy.stats import zscore

# -----------------------------------------------------------------------------
# CONFIGURATION & STYLE
# -----------------------------------------------------------------------------
pd.options.mode.chained_assignment = None
pio.templates.default = "plotly_dark"

COLOR_PALETTE = {
    'bg': '#1e1e1e',
    'text': '#e0e0e0',
    'up': '#00cc96',
    'down': '#ef553b',
    'accent': '#636efa',
    'grid': '#333333'
}

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def flatten_yfinance_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Handle MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the top level (Ticker) if it exists
        df.columns = df.columns.droplevel(1)
    
    # Standardize names
    rename_map = {
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 
        'Close': 'Close', 'Adj Close': 'Adj Close', 'Volume': 'Volume'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def download_data(ticker: str, start_date: str) -> pd.DataFrame:
    print(f"[STEGO] Downloading {ticker} from {start_date}...")
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
    df = flatten_yfinance_cols(df)
    
    # Fallback to Close if Adj Close is missing (rare)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
        
    return df

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def calculate_regimes(df: pd.DataFrame, spy_df: pd.DataFrame = None, vix_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes Trend, Volatility, Volume, Breadth, and Risk regimes without lookahead bias.
    """
    df = df.copy()
    
    # 1. Trend Regime (EMA Hierarchy)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    conditions = [
        (df['Close'] > df['EMA20']) & (df['EMA20'] > df['EMA50']) & (df['EMA50'] > df['EMA200']),
        (df['Close'] < df['EMA20']) & (df['EMA20'] < df['EMA50']) & (df['EMA50'] < df['EMA200'])
    ]
    choices = ['Bullish', 'Bearish']
    df['Trend_Regime'] = np.select(conditions, choices, default='Neutral')

    # 2. Volatility Regime (ATR%)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['ATR_Pct'] = (atr / df['Close']) * 100
    
    # Quantile buckets (Tertiles) using expanding window to prevent leakage or simple rolling
    # For simplicity and speed in this dashboard, we use a long rolling window for distribution context
    rank_window = 252
    df['ATR_Rank'] = df['ATR_Pct'].rolling(rank_window).rank(pct=True)
    
    vol_conds = [df['ATR_Rank'] < 0.33, df['ATR_Rank'] > 0.66]
    vol_choices = ['Low Vol', 'High Vol']
    df['Vol_Regime'] = np.select(vol_conds, vol_choices, default='Normal Vol')

    # 3. Volume Regime (Z-Score)
    df['Vol_Z'] = (df['Volume'] - df['Volume'].rolling(60).mean()) / df['Volume'].rolling(60).std()
    
    v_conds = [df['Vol_Z'] < -1.0, df['Vol_Z'] > 1.0]
    v_choices = ['Low Volm', 'High Volm']
    df['Volume_Regime'] = np.select(v_conds, v_choices, default='Normal Volm')

    # 4. Breadth Proxy (SPY)
    if spy_df is not None:
        # Align dates
        spy_df = spy_df.reindex(df.index).ffill()
        spy_ema50 = spy_df['Close'].ewm(span=50, adjust=False).mean()
        spy_uptrend = (spy_df['Close'] > spy_ema50).astype(int)
        # Percent of last 20 days SPY was in uptrend
        breadth_val = spy_uptrend.rolling(20).mean()
        
        b_conds = [breadth_val > 0.7, breadth_val < 0.3]
        b_choices = ['High Breadth', 'Low Breadth']
        df['Breadth_Regime'] = np.select(b_conds, b_choices, default='Neut Breadth')
    else:
        df['Breadth_Regime'] = 'N/A'

    # 5. Risk Proxy (VIX)
    if vix_df is not None:
        vix_df = vix_df.reindex(df.index).ffill()
        vix_sma20 = vix_df['Close'].rolling(20).mean()
        
        r_conds = [vix_df['Close'] < vix_sma20, vix_df['Close'] > vix_sma20]
        r_choices = ['Risk-On', 'Risk-Off']
        df['Risk_Regime'] = np.select(r_conds, r_choices, default='Neutral')
    else:
        df['Risk_Regime'] = 'N/A'

    return df

def detect_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the current 'Up Streak' length for every day.
    Streak = Number of consecutive days where Close > Close[t-1].
    """
    df = df.copy()
    
    # Boolean Series for Up Day
    df['Is_Up'] = df['Close'] > df['Close'].shift(1)
    
    # Vectorized Streak Calculation
    # Compare current Is_Up with previous. Identify blocks.
    # We use a cumsum of the 'reset' condition (False) to identify groups.
    s = df['Is_Up'].astype(int)
    df['Streak_Len'] = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
    
    # If today is NOT up, streak is 0
    df.loc[~df['Is_Up'], 'Streak_Len'] = 0
    
    return df

# -----------------------------------------------------------------------------
# STATISTICAL ENGINE
# -----------------------------------------------------------------------------
def analyze_streaks(df: pd.DataFrame, streak_lengths=[3, 4, 5, 6, 7]):
    """
    Analyzes forward returns and continuation probabilities for specific streak lengths.
    Returns a collected stats summary and the event-level data.
    """
    
    # Labels for Next Day
    df['Next_Close'] = df['Close'].shift(-1)
    df['Next_Return'] = (df['Next_Close'] / df['Close']) - 1.0
    df['Next_Up'] = (df['Next_Close'] > df['Close']).astype(int)
    
    # Convert returns to basis points
    df['Next_Return_Bps'] = df['Next_Return'] * 10000
    
    results = []
    
    # Filters to iterate
    filter_cols = ['Trend_Regime', 'Vol_Regime', 'Volume_Regime', 'Risk_Regime']
    
    # 1. Overall Stats per K
    for k in streak_lengths:
        # subset: days where the streak length reached exactly k
        # Note: We look at the moment the streak hits k. 
        # (It might go to k+1 tomorrow, which counts as "Next_Up" = 1)
        events = df[df['Streak_Len'] == k].copy()
        events = events.dropna(subset=['Next_Return']) # Drop last row if valid
        
        if events.empty:
            continue
            
        # Bootstrap CI function
        def bootstrap_ci(data, n_boot=1000, block_size=5):
            if len(data) < 5:
                return (np.nan, np.nan)
            # Simple resampling for speed in this script
            means = []
            for _ in range(n_boot):
                sample = np.random.choice(data, size=len(data), replace=True)
                means.append(np.mean(sample))
            return np.percentile(means, [2.5, 97.5])

        # Overall K stats
        mean_ret = events['Next_Return_Bps'].mean()
        prob_cont = events['Next_Up'].mean()
        ci_lower, ci_upper = bootstrap_ci(events['Next_Return_Bps'].values)
        
        results.append({
            'K': k,
            'Filter_Type': 'Overall',
            'Filter_Value': 'All',
            'N': len(events),
            'Mean_Next_Ret_Bps': mean_ret,
            'Prob_Continue': prob_cont,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper
        })
        
        # 2. Stats by Regime
        for f_col in filter_cols:
            if f_col not in df.columns: continue
            
            groups = events.groupby(f_col)
            for name, group in groups:
                if len(group) < 3: continue # Noise filter
                
                m_ret = group['Next_Return_Bps'].mean()
                p_cont = group['Next_Up'].mean()
                
                results.append({
                    'K': k,
                    'Filter_Type': f_col,
                    'Filter_Value': name,
                    'N': len(group),
                    'Mean_Next_Ret_Bps': m_ret,
                    'Prob_Continue': p_cont,
                    'CI_Lower': np.nan, # Skip CI for subgroups to save time/space
                    'CI_Upper': np.nan
                })

    return pd.DataFrame(results), df

# -----------------------------------------------------------------------------
# PLOTLY DASHBOARD GENERATION
# -----------------------------------------------------------------------------
def generate_dashboard(ticker: str, df: pd.DataFrame, stats: pd.DataFrame, output_dir: str):
    
    # --- PREPARE DATA FOR PLOTS ---
    # 1. Overview Data
    overall = stats[stats['Filter_Type'] == 'Overall']
    
    # 2. Matrix Data (Regimes)
    # Pivot for Heatmaps
    def get_pivot(filter_type, value_col):
        subset = stats[stats['Filter_Type'] == filter_type]
        if subset.empty: return pd.DataFrame()
        return subset.pivot(index='Filter_Value', columns='K', values=value_col)

    # --- HTML TEMPLATE CONSTRUCTION ---
    
    # We will generate individual HTML divs for Plotly figures and inject them
    
    # FIG 1: Overview Bar Chart (Prob Continue vs K)
    fig_overview = go.Figure()
    fig_overview.add_trace(go.Bar(
        x=overall['K'], 
        y=overall['Prob_Continue'],
        text=overall['Prob_Continue'].apply(lambda x: f"{x:.1%}"),
        textposition='auto',
        marker_color=COLOR_PALETTE['accent'],
        name='Win Rate'
    ))
    fig_overview.add_trace(go.Scatter(
        x=overall['K'],
        y=overall['Mean_Next_Ret_Bps'],
        yaxis='y2',
        mode='lines+markers',
        name='Mean Return (bps)',
        line=dict(color=COLOR_PALETTE['up'], width=3)
    ))
    fig_overview.update_layout(
        title=f"Streak Continuation Probability & Return ({ticker})",
        template="plotly_dark",
        xaxis_title="Streak Length (Days)",
        yaxis=dict(title="Probability of Continuation", tickformat=".0%"),
        yaxis2=dict(title="Next Day Return (bps)", overlaying='y', side='right'),
        barmode='group'
    )
    
    # FIG 2: Heatmap - Trend Regime vs K (Prob Continue)
    trend_pivot = get_pivot('Trend_Regime', 'Prob_Continue')
    fig_trend = go.Figure(data=go.Heatmap(
        z=trend_pivot.values,
        x=trend_pivot.columns,
        y=trend_pivot.index,
        colorscale='Viridis',
        text=np.round(trend_pivot.values*100, 1),
        texttemplate="%{text}%"
    ))
    fig_trend.update_layout(title="Trend Regime vs Streak Length (Prob Continue)", template="plotly_dark")

    # FIG 3: Heatmap - Volatility Regime vs K (Return Bps)
    vol_pivot = get_pivot('Vol_Regime', 'Mean_Next_Ret_Bps')
    fig_vol = go.Figure(data=go.Heatmap(
        z=vol_pivot.values,
        x=vol_pivot.columns,
        y=vol_pivot.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(vol_pivot.values, 0),
        texttemplate="%{text} bps"
    ))
    fig_vol.update_layout(title="Volatility Regime vs Streak Length (Mean Return)", template="plotly_dark")

    # FIG 4: Timeline with Events
    # Filter recent history for performance
    subset_df = df.tail(252).copy()
    
    fig_timeline = go.Figure()
    
    # Candlestick
    fig_timeline.add_trace(go.Candlestick(
        x=subset_df.index,
        open=subset_df['Open'], high=subset_df['High'],
        low=subset_df['Low'], close=subset_df['Close'],
        name='Price'
    ))
    
    # Streak Markers
    # Find points where Streak >= 3
    streak_events = subset_df[subset_df['Streak_Len'] >= 3]
    if not streak_events.empty:
        fig_timeline.add_trace(go.Scatter(
            x=streak_events.index,
            y=streak_events['High'] * 1.02,
            mode='text',
            text=streak_events['Streak_Len'].astype(str),
            textposition="top center",
            name='Streak Count',
            textfont=dict(color=COLOR_PALETTE['accent'], size=14, weight='bold')
        ))
        # Add triangles
        fig_timeline.add_trace(go.Scatter(
            x=streak_events.index,
            y=streak_events['High'] * 1.01,
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color=COLOR_PALETTE['accent']),
            name='Streak Marker',
            showlegend=False
        ))

    fig_timeline.update_layout(
        title=f"{ticker} Recent Price Action & Streaks",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600
    )

    # --- SAVE HTML COMPONENTS ---
    div_overview = pio.to_html(fig_overview, full_html=False, include_plotlyjs='cdn')
    div_trend = pio.to_html(fig_trend, full_html=False, include_plotlyjs=False)
    div_vol = pio.to_html(fig_vol, full_html=False, include_plotlyjs=False)
    div_timeline = pio.to_html(fig_timeline, full_html=False, include_plotlyjs=False)

    # --- BUILD FINAL HTML DASHBOARD ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>STEGO Streak Dashboard: {ticker}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
            .header {{ padding: 20px; border-bottom: 2px solid #333; margin-bottom: 20px; }}
            h1 {{ margin: 0; color: #636efa; }}
            h3 {{ color: #a2a9b1; }}
            .tab {{ overflow: hidden; border: 1px solid #333; background-color: #1e1e1e; }}
            .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; color: #ccc; font-size: 16px; }}
            .tab button:hover {{ background-color: #333; }}
            .tab button.active {{ background-color: #636efa; color: white; }}
            .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #333; border-top: none; animation: fadeEffect 1s; }}
            @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
            .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .card {{ background-color: #1e1e1e; padding: 15px; border-radius: 5px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); }}
        </style>
    </head>
    <body>

    <div class="header">
        <h1>STEGO Financial Framework</h1>
        <h3>Streak Statistical Dashboard | Ticker: {ticker} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</h3>
    </div>

    <div class="tab">
      <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Overview</button>
      <button class="tablinks" onclick="openTab(event, 'Regimes')">Regime Heatmaps</button>
      <button class="tablinks" onclick="openTab(event, 'Timeline')">Event Timeline</button>
      <button class="tablinks" onclick="openTab(event, 'Data')">Statistical Summary</button>
    </div>

    <div id="Overview" class="tabcontent">
      <div class="card">
        {div_overview}
      </div>
      <div style="margin-top:20px; padding:10px; background:#1e1e1e;">
        <p><strong>Interpretation:</strong> This chart shows the probability that an up-streak will continue for one more day (Bars) and the average return of that next day (Line).</p>
      </div>
    </div>

    <div id="Regimes" class="tabcontent">
      <div class="grid-container">
        <div class="card">{div_trend}</div>
        <div class="card">{div_vol}</div>
      </div>
    </div>

    <div id="Timeline" class="tabcontent">
      <div class="card">
        {div_timeline}
      </div>
    </div>

    <div id="Data" class="tabcontent">
      <div class="card">
        <h3>Summary Table</h3>
        {overall.to_html(classes='table', float_format="%.2f", index=False, border=0, justify='left')}
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
    }}
    // Get the element with id="defaultOpen" and click on it
    document.getElementById("defaultOpen").click();
    
    // Style the pandas table via JS simple injection
    var tables = document.getElementsByTagName('table');
    for (var i=0; i<tables.length;i++){{
        tables[i].style.width = "100%";
        tables[i].style.color = "#e0e0e0";
        tables[i].style.borderCollapse = "collapse";
    }}
    </script>
    </body>
    </html>
    """
    
    # Write to file
    filename = f"{ticker}_streak_dashboard.html"
    full_path = os.path.join(output_dir, filename)
    with open(full_path, "w", encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[STEGO] Dashboard saved to: {full_path}")
    return full_path

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="STEGO Streak Dashboard Generator")
    parser.add_argument("--ticker", type=str, required=True, help="Stock Ticker (e.g., NVDA)")
    parser.add_argument("--years", type=int, default=5, help="Years of history")
    parser.add_argument("--output-root", type=str, default="/dev/shm/STREAK_DASHBOARD", help="Output directory")
    
    args = parser.parse_args()
    
    # 1. Setup Paths
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(args.output_root, args.ticker, today_str)
    ensure_dir(output_dir)
    
    # 2. Date Math
    start_date = (datetime.now() - timedelta(days=args.years*365)).strftime('%Y-%m-%d')
    
    # 3. Data Retrieval
    try:
        main_df = download_data(args.ticker, start_date)
        if main_df.empty:
            print(f"Error: No data found for {args.ticker}")
            return
            
        spy_df = download_data("SPY", start_date)
        vix_df = download_data("^VIX", start_date)
    except Exception as e:
        print(f"Data download failed: {e}")
        return

    # 4. Processing
    print("[STEGO] calculating regimes and identifying streaks...")
    df_processed = calculate_regimes(main_df, spy_df, vix_df)
    df_processed = detect_streaks(df_processed)
    
    # 5. Statistical Analysis
    print("[STEGO] Running statistical bootstrap analysis...")
    stats_df, event_df = analyze_streaks(df_processed)
    
    if stats_df.empty:
        print("No streak events found in the specified period.")
        return

    # 6. Save Raw Data
    csv_path = os.path.join(output_dir, f"{args.ticker}_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"[STEGO] Stats saved to {csv_path}")

    # 7. Generate Dashboard
    print("[STEGO] Building Plotly Dashboard...")
    dash_path = generate_dashboard(args.ticker, df_processed, stats_df, output_dir)
    
    # 8. Launch
    print(f"[STEGO] Opening {dash_path} ...")
    webbrowser.open(f"file://{dash_path}")

if __name__ == "__main__":
    main()
