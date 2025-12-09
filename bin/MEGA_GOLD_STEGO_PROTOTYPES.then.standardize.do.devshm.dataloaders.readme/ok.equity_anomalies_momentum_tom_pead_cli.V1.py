import argparse
import os
import sys
import math
import datetime
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots

# --- SciPy Optional Import ---
try:
    from scipy import stats
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

# --- Helpers ---

def _parse_list_arg(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.replace(',', ' ').split() if x.strip()]

def force_tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 1: Timezone Fix — force timezone-naive, sorted Date index"""
    if df is None or df.empty:
        return df
    
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    return df

def normalize_yf_df(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Normalization helper: MultiIndex handling + Rule 1"""
    if df.empty:
        return df

    # Handling MultiIndex Columns (Ticker, Field) or (Field, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # If the ticker level exists, try to select just that ticker
        # yfinance often returns columns like ('Adj Close', 'AAPL')
        try:
            # Check if ticker is in the columns
            if ticker and ticker in df.columns.get_level_values(1):
                df = df.xs(ticker, axis=1, level=1)
            elif ticker and ticker in df.columns.get_level_values(0):
                df = df.xs(ticker, axis=1, level=0)
            else:
                 # Last ditch: if we just have one ticker in multiindex, drop levels
                df.columns = df.columns.droplevel(1) 
        except Exception:
            # Flattening approach if selection fails
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Standardize columns
    # Drop 'Adj Close' if 'Close' exists, else rename 'Adj Close' to 'Close' is handled in robust_close_series logic mostly,
    # but here we just ensure numeric types.
    
    # Ensure numeric
    cols = df.columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return force_tz_naive_index(df)

def robust_close_series(df: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    """Rule 2: yfinance Fix — robust 'Close' extraction"""
    if isinstance(df, pd.DataFrame):
        if 'Close' in df.columns:
            s = df['Close']
        elif 'Adj Close' in df.columns:
            s = df['Adj Close']
        else:
            s = df.iloc[:, 0]  # Fallback: first column
    else:
        s = df
    return s

def download_history_single(sym: str, period: str = "3y", start: str = None, end: str = None) -> pd.DataFrame:
    print(f"[INFO] Downloading {sym}...")
    try:
        t = yf.Ticker(sym)
        if start and end:
            df = t.history(start=start, end=end)
        else:
            df = t.history(period=period)
        
        if df.empty:
             # Fallback
             df = yf.download([sym], period=period, start=start, end=end, threads=False, progress=False, group_by="column")
        
        df = normalize_yf_df(df, ticker=sym)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to download {sym}: {e}")
        return pd.DataFrame()

def build_close_panel(symbols: List[str], period: str, start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    panel_data = {}
    full_dfs = {}
    
    for sym in symbols:
        df = download_history_single(sym, period, start, end)
        if not df.empty:
            s = robust_close_series(df)
            # Ensure series name is the ticker
            s.name = sym
            panel_data[sym] = s
            full_dfs[sym] = df
            
    if not panel_data:
        return pd.DataFrame(), {}
        
    panel = pd.concat(panel_data.values(), axis=1)
    panel = force_tz_naive_index(panel)
    return panel, full_dfs

def align_by_intersection(df_assets: pd.DataFrame, df_proxies: pd.DataFrame, df_fx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rule 3: Alignment Fix — strict index intersection BEFORE any math"""
    if df_assets.empty:
        return df_assets, df_proxies, df_fx
        
    common = df_assets.index
    if not df_proxies.empty:
        common = common.intersection(df_proxies.index)
    if not df_fx.empty:
        common = common.intersection(df_fx.index)
        
    # Sort just in case intersection scrambled order (it shouldn't, but safe)
    common = common.sort_values()
    
    df_assets_out = df_assets.loc[common]
    df_proxies_out = df_proxies.loc[common] if not df_proxies.empty else df_proxies
    df_fx_out = df_fx.loc[common] if not df_fx.empty else df_fx
    
    return df_assets_out, df_proxies_out, df_fx_out

def compute_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pct_change().fillna(0.0)

def compute_momentum(panel: pd.DataFrame, long_w: int, short_w: int) -> pd.DataFrame:
    # mom_12m_ex_1m: 252-day return minus 21-day return
    # mom_1m: 21-day return
    
    # Return over N days = (P_t / P_{t-N}) - 1
    
    ret_long = panel.pct_change(periods=long_w)
    ret_short = panel.pct_change(periods=short_w)
    
    # 12-1 momentum often defined as return from t-12m to t-1m.
    # Logic: R_long - R_short is a simplified proxy, or strictly (P_{t-21} / P_{t-252}) - 1
    # We will use the prompt definition: "252-day return minus 21-day return"
    # Note: If these are simple returns, strict subtraction is an approximation of log returns, but requested.
    
    mom_12_1 = ret_long - ret_short
    mom_1 = ret_short
    
    # We want the latest values for the summary
    latest_12_1 = mom_12_1.iloc[-1]
    latest_1 = mom_1.iloc[-1]
    
    summary = pd.DataFrame({
        'mom_12m_ex_1m': latest_12_1,
        'mom_1m': latest_1
    })
    
    return summary

def compute_tom_stats(returns: pd.DataFrame, tom_pre: int, tom_post: int) -> pd.DataFrame:
    """
    TOM Logic:
    Identify last trading day of month (T).
    Window is [T - tom_pre, T + tom_post] relative to month boundary.
    Note: T is offset 0? Usually TOM is defined as Last Day (-1) to First 3 Days (+3).
    Let's map: 
    - End of Month Day: Day 0
    - First Day of Next Month: Day 1
    
    Algorithm:
    1. Iterate through index.
    2. Detect month changes.
    3. Collect offsets.
    """
    
    # Create a Series to hold offset tags (NaN if not in window)
    # We need a unified calendar. returns is aligned.
    dates = returns.index
    if len(dates) < 2:
        return pd.DataFrame()
    
    # Identify month ends
    # A day is a month end if the next day has a different month
    is_month_end = np.array([False] * len(dates))
    current_months = dates.month
    # Vectorized check: shift -1
    next_months = np.roll(current_months, -1)
    # last item is always false/undefined in roll, but practically last day of data might be month end
    # simple loop is safer for trading calendars
    
    month_ends_indices = []
    for i in range(len(dates) - 1):
        if dates[i].month != dates[i+1].month:
            month_ends_indices.append(i)
    # Check last date? If we don't know next day, we can't be sure, but usually ignored for historical stats.

    # Dictionary mapping relative_day -> list of returns
    # We need to support multiple columns (tickers). We will aggregate across all provided tickers or per ticker?
    # "Aggregate mean and t-stats across all months per offset day" implies aggregation over time.
    # Usually TOM is market-wide, but can be per asset. Let's compute per asset then average? 
    # Or pool all returns? "Aggregate mean... per offset day". 
    # Let's produce stats PER TICKER first, or AGGREGATE?
    # Prompt implies a single "TOM mean return by offset" chart (Figure 3).
    # We will average across the provided assets (equal weight portfolio logic) for the stats.
    
    # Create an equal-weight index of the assets for TOM stats
    ew_ret = returns.mean(axis=1)
    
    stats_collector = {i: [] for i in range(-tom_pre, tom_post + 1)}
    
    # Pre-calculate indices to avoid bounds checking inside loop repeatedly
    n = len(dates)
    
    for idx in month_ends_indices:
        # idx is the last day of the month (offset 0 in some nomenclatures, let's call it -1 if 1st day is 1? 
        # Prompt: "last day of month and first 3 days". 
        # Standard: Last day is Turn-1, First is Turn+1.
        # Let's map: Last Day = Offset 0. First Day = Offset 1.
        # tom_pre=1 (include last day). tom_post=3 (include first 3).
        # range: -tom_pre to +tom_post. 
        # If tom_pre=1, we want 1 day before the boundary (the last day).
        # Let's define Month End Day as offset 0.
        # Days before: -1, -2.
        # Days after: 1, 2.
        # Prompt says "last day of month and first 3 days".
        # If tom_pre=1, does it mean include offset 0?
        # Let's assume Offset 0 is Last Day of Month. Offset 1 is First Day of Next Month.
        # Range: 1 day before month end? No, "tom-pre number of trading days before month end".
        # If pre=1, we take the last day. 
        # Let's treat indices relative to the "Turn".
        # Let boundary be between idx and idx+1.
        
        # Offsets <= 0: go back from idx
        for i in range(tom_pre): # 0 to tom_pre-1
            # 0 -> idx (last day)
            # 1 -> idx-1
            lookback_idx = idx - i
            offset_label = -i # 0, -1, -2...
            if lookback_idx >= 0:
                stats_collector[offset_label].append(ew_ret.iloc[lookback_idx])

        # Offsets > 0: go forward from idx+1
        for i in range(tom_post): # 0 to tom_post-1
            # 0 -> idx+1 (first day)
            # 1 -> idx+2
            lookfwd_idx = idx + 1 + i
            offset_label = i + 1 # 1, 2, 3...
            if lookfwd_idx < n:
                stats_collector[offset_label].append(ew_ret.iloc[lookfwd_idx])

    # Compile results
    results = []
    # Sort offsets
    sorted_offsets = sorted(stats_collector.keys())
    
    for off in sorted_offsets:
        vals = np.array(stats_collector[off])
        if len(vals) < 2:
            results.append({'offset': off, 'mean': np.nan, 'std': np.nan, 'tstat': np.nan, 'p': np.nan, 'count': 0})
            continue
            
        mu = np.mean(vals)
        sigma = np.std(vals, ddof=1)
        count = len(vals)
        
        if SCIPY_AVAILABLE:
            tstat, pval = stats.ttest_1samp(vals, 0)
        else:
            # Manual t-stat
            se = sigma / math.sqrt(count)
            tstat = mu / se if se > 1e-12 else 0.0
            # Approx p-value from normal cdf
            # Two-tailed
            pval = 2 * (1 - norm_cdf(abs(tstat)))
            
        results.append({
            'offset': off, 
            'mean': mu, 
            'std': sigma, 
            'tstat': tstat, 
            'p': pval,
            'count': count
        })
        
    return pd.DataFrame(results).set_index('offset')

def get_earnings_dates(sym: str) -> pd.DatetimeIndex:
    try:
        t = yf.Ticker(sym)
        # Attempt specific methods
        dates = None
        
        # Method A: get_earnings_dates
        try:
            ed = t.get_earnings_dates(limit=50)
            if ed is not None and not ed.empty:
                dates = ed.index
        except:
            pass
            
        # Method B: calendar
        if dates is None:
            try:
                cal = t.calendar
                if cal is not None and not cal.empty:
                    # 'Earnings Date' or similar row
                    if 'Earnings Date' in cal.index:
                        dates = pd.to_datetime(cal.loc['Earnings Date'])
                    elif 'Earnings High' in cal.index: 
                         # Sometimes calendar is transposed or different structure, try best effort
                         pass
            except:
                pass
                
        if dates is not None:
            # Ensure timezone naive
            dates = pd.to_datetime(dates)
            if dates.tz is not None:
                dates = dates.tz_localize(None)
            return dates.sort_values()
            
    except Exception as e:
        print(f"[WARN] Could not fetch earnings for {sym}: {e}")
    
    return pd.DatetimeIndex([])

def compute_pead_caar(returns: pd.Series, proxy_returns: pd.Series, dates: pd.DatetimeIndex, pre: int, post: int) -> pd.DataFrame:
    """
    Event study: 
    AR = R_asset - R_proxy (if proxy exists) else R_asset
    Window: [date - pre, date + post] (trading days)
    """
    if dates.empty:
        return pd.DataFrame()
    
    # Calculate AR series first
    if not proxy_returns.empty:
        # Align series just to be safe
        c_idx = returns.index.intersection(proxy_returns.index)
        ar_series = returns.loc[c_idx] - proxy_returns.loc[c_idx]
    else:
        ar_series = returns
        
    # We need integer locations for windows
    # Map dates to indices in ar_series
    
    # Filter earnings dates to those within the returns range
    valid_dates = dates[(dates >= ar_series.index[0]) & (dates <= ar_series.index[-1])]
    
    if valid_dates.empty:
        return pd.DataFrame()

    # Get integer indices of the events
    # We use searchsorted. ar_series.index is sorted.
    # searchsorted returns the index where the date would be inserted
    target_indices = ar_series.index.searchsorted(valid_dates)
    
    # Collect windows
    windows = []
    
    for idx in target_indices:
        # Check bounds
        # idx points to the date >= event_date.
        # If event_date is a trading day, idx is it.
        # If event_date is Saturday, idx is Monday.
        # We usually snap to next trading day if non-trading.
        
        if idx >= len(ar_series): continue
        
        start_idx = idx + pre
        end_idx = idx + post + 1 # Slice exclusive
        
        if start_idx < 0 or end_idx > len(ar_series):
            continue
            
        window_ar = ar_series.iloc[start_idx:end_idx].values
        
        # Verify length
        expected_len = (post - pre) + 1
        if len(window_ar) == expected_len:
            windows.append(window_ar)
            
    if not windows:
        return pd.DataFrame()
        
    # Average across events (AAR)
    windows_arr = np.array(windows) # Shape (N_events, Window_Size)
    aar = np.nanmean(windows_arr, axis=0)
    
    # CAAR
    caar = np.cumsum(aar)
    
    # Create DataFrame
    offsets = list(range(pre, post + 1))
    return pd.DataFrame({'CAAR': caar}, index=offsets)

def generate_dashboard(
    returns_panel: pd.DataFrame, 
    momentum_df: pd.DataFrame, 
    tom_df: pd.DataFrame, 
    pead_results: Dict[str, pd.DataFrame], 
    last_ticker: str, 
    price_history: pd.DataFrame,
    earnings_dates: pd.DatetimeIndex,
    tom_pre: int, tom_post: int
) -> str:
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            f"Price History: {last_ticker} (Markers: TOM/Earnings)", 
            "Momentum (12m-1m vs 1m)", 
            "TOM Seasonality (Aggregated Mean Return)", 
            "PEAD CAAR (Earnings Event Study)"
        )
    )
    
    # --- Figure 1: Price Panel ---
    # Line
    s_price = robust_close_series(price_history)
    fig.add_trace(go.Scatter(x=s_price.index, y=s_price.values, mode='lines', name=f'{last_ticker} Price'), row=1, col=1)
    
    # Markers for Earnings
    if earnings_dates is not None and not earnings_dates.empty:
        # Align
        ed_in_range = earnings_dates[earnings_dates.isin(s_price.index)]
        if not ed_in_range.empty:
            prices_at_e = s_price.loc[ed_in_range]
            fig.add_trace(go.Scatter(
                x=prices_at_e.index, y=prices_at_e.values, 
                mode='markers', marker=dict(color='red', size=8, symbol='x'),
                name='Earnings'
            ), row=1, col=1)
            
    # --- Figure 2: Momentum ---
    if not momentum_df.empty:
        fig.add_trace(go.Bar(
            x=momentum_df.index, y=momentum_df['mom_12m_ex_1m'], 
            name='Mom 12-1', marker_color='blue'
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=momentum_df.index, y=momentum_df['mom_1m'], 
            name='Mom 1m', marker_color='orange'
        ), row=1, col=2)

    # --- Figure 3: TOM ---
    if not tom_df.empty:
        fig.add_trace(go.Bar(
            x=tom_df.index, y=tom_df['mean'],
            error_y=dict(type='data', array=tom_df['std']),
            name='Mean Return', marker_color='green'
        ), row=2, col=1)
        fig.update_xaxes(title_text="Offset (Days from Month End)", row=2, col=1)

    # --- Figure 4: PEAD ---
    for tkr, df_caar in pead_results.items():
        fig.add_trace(go.Scatter(
            x=df_caar.index, y=df_caar['CAAR'],
            mode='lines+markers', name=f'PEAD {tkr}'
        ), row=2, col=2)
    fig.update_xaxes(title_text="Event Window (Days)", row=2, col=2)
        
    fig.update_layout(height=800, title_text="Equity Anomalies Dashboard", showlegend=True)
    return pio.to_html(fig, include_plotlyjs='cdn', full_html=True)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Equity Anomalies CLI")
    parser.add_argument("tickers", type=str, help="Comma separated tickers")
    parser.add_argument("--period", type=str, default="3y")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--proxies", type=str, default="SPY")
    parser.add_argument("--fx", type=str, default="")
    parser.add_argument("--out", type=str, default="./out")
    
    # Params
    parser.add_argument("--mom-long", type=int, default=252)
    parser.add_argument("--mom-short", type=int, default=21)
    parser.add_argument("--tom-pre", type=int, default=1)
    parser.add_argument("--tom-post", type=int, default=3)
    parser.add_argument("--pead-pre", type=int, default=-1)
    parser.add_argument("--pead-post", type=int, default=5)
    
    args = parser.parse_args()
    
    # Setup
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    tickers = _parse_list_arg(args.tickers)
    proxies = _parse_list_arg(args.proxies)
    fx = _parse_list_arg(args.fx)
    
    print(f"[INFO] Tickers: {tickers}")
    print(f"[INFO] Proxies: {proxies}")
    
    # 1. Load Data
    print("[INFO] Building Asset Panel...")
    df_assets, raw_assets_map = build_close_panel(tickers, args.period, args.start, args.end)
    if df_assets.empty:
        print("[ERROR] No asset data downloaded. Exiting.")
        sys.exit(1)
        
    # Save per-ticker CSV
    for tkr, df in raw_assets_map.items():
        t_dir = os.path.join(args.out, tkr)
        if not os.path.exists(t_dir):
            os.makedirs(t_dir)
        df.to_csv(os.path.join(t_dir, "price_history.csv"))
        
    print("[INFO] Building Proxy/FX Panels...")
    df_proxies_panel, _ = build_close_panel(proxies, args.period, args.start, args.end)
    df_fx_panel, _ = build_close_panel(fx, args.period, args.start, args.end)
    
    # 2. Alignment
    print("[INFO] Aligning data...")
    df_assets, df_proxies_panel, df_fx_panel = align_by_intersection(df_assets, df_proxies_panel, df_fx_panel)
    
    if df_assets.empty:
        print("[ERROR] Data alignment resulted in empty set. Check date overlaps.")
        sys.exit(1)
        
    # 3. Metrics
    # Returns
    returns_panel = compute_returns(df_assets)
    returns_panel.to_csv(os.path.join(args.out, "returns_panel.csv"))
    
    # Proxy Returns (for PEAD)
    proxy_returns = pd.Series(dtype=float)
    if not df_proxies_panel.empty:
        # Use first proxy as the market benchmark
        proxy_returns = df_proxies_panel.iloc[:, 0].pct_change().fillna(0.0)
    
    # Momentum
    print("[INFO] Computing Momentum...")
    mom_df = compute_momentum(df_assets, args.mom_long, args.mom_short)
    mom_df.to_csv(os.path.join(args.out, "momentum_summary.csv"))
    
    # TOM
    print("[INFO] Computing TOM Stats...")
    tom_df = compute_tom_stats(returns_panel, args.tom_pre, args.tom_post)
    tom_df.to_csv(os.path.join(args.out, "tom_stats.csv"))
    
    # PEAD
    print("[INFO] Computing PEAD...")
    pead_results = {}
    last_ticker = tickers[-1]
    last_ticker_earnings = None
    
    for tkr in tickers:
        print(f"  ... {tkr}")
        dates = get_earnings_dates(tkr)
        if tkr == last_ticker:
            last_ticker_earnings = dates
            
        if not dates.empty:
            # Ticker specific return
            r_series = returns_panel[tkr]
            
            caar_df = compute_pead_caar(r_series, proxy_returns, dates, args.pead_pre, args.pead_post)
            
            if not caar_df.empty:
                pead_results[tkr] = caar_df
                caar_df.to_csv(os.path.join(args.out, f"pead_caar_{tkr}.csv"))
    
    # 4. Dashboard
    print("[INFO] Generating Dashboard...")
    # Get last ticker history for plot (raw)
    last_hist = raw_assets_map.get(last_ticker, pd.DataFrame())
    
    html = generate_dashboard(
        returns_panel, mom_df, tom_df, pead_results, 
        last_ticker, last_hist, last_ticker_earnings,
        args.tom_pre, args.tom_post
    )
    
    with open(os.path.join(args.out, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"[SUCCESS] Done. Output in {args.out}")

if __name__ == "__main__":
    main()

# requirements.txt
# yfinance
# pandas
# numpy
# plotly
# scipy  # optional
