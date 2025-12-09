# SCRIPTNAME: ok.market_dashboard.revamp.v1.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import sys
import os
import webbrowser
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
# NEW: Import offline to embed the JS library
import plotly.offline as py_offline

# -----------------------------------------------------------------------------
# 1. Argument Parsing & Setup
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Financial Dispersion & Correlation Dashboard")
    
    # Allow --tickers or --ticker
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tickers', type=str, help="Comma-separated list of tickers (e.g., AAPL,MSFT,GOOG)")
    group.add_argument('--ticker', type=str, help="Single ticker (e.g., SPY)")
    
    parser.add_argument('--period', type=str, default='2y', help="Data period to download (default: 2y)")
    parser.add_argument('--roll-window', type=int, default=60, help="Rolling window size (default: 60)")
    parser.add_argument('--min-periods', type=int, default=30, help="Minimum periods for rolling calcs (default: 30)")
    parser.add_argument('--out', type=str, default='dispersion_dashboard.html', help="Output HTML filename")
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# 2. Data Retrieval & Preprocessing
# -----------------------------------------------------------------------------
def get_data(ticker_list, period):
    print(f"[*] Downloading data for: {', '.join(ticker_list)} over {period}...")
    
    try:
        # Download data
        # group_by='column' is generally safer for parsing
        df = yf.download(
            ticker_list, 
            period=period, 
            interval='1d', 
            progress=False, 
            auto_adjust=False,
            group_by='column' 
        )
        
        if df.empty:
            raise ValueError("Downloaded data is empty. Check tickers or period.")

        # --- Proven Robust Column Handling ---
        data = pd.DataFrame()

        # Case 1: MultiIndex columns (Typical for >1 ticker, and new yfinance single tickers)
        if isinstance(df.columns, pd.MultiIndex):
            # Check levels for 'Adj Close' or 'Close'
            # yfinance v0.2.x sometimes puts Ticker in level 0, sometimes Attribute in level 0
            
            # Check Level 0 for Attribute
            if "Adj Close" in df.columns.get_level_values(0):
                data = df["Adj Close"].copy()
            elif "Close" in df.columns.get_level_values(0):
                data = df["Close"].copy()
            else:
                # Try swapping levels (Level 1 might be Attribute)
                swapped = df.swaplevel(0, 1, axis=1)
                if "Adj Close" in swapped.columns.get_level_values(0):
                    data = swapped["Adj Close"].copy()
                elif "Close" in swapped.columns.get_level_values(0):
                    data = swapped["Close"].copy()

        # Case 2: Single Index (Old yfinance or specific single ticker result)
        else:
            if "Adj Close" in df.columns:
                data = df[["Adj Close"]].copy()
            elif "Close" in df.columns:
                data = df[["Close"]].copy()
            
            # If we requested 1 ticker and got 1 column, rename it to the ticker
            if data.shape[1] == 1 and len(ticker_list) == 1:
                data.columns = ticker_list

        if data.empty:
            raise ValueError(f"Could not locate 'Adj Close' or 'Close' in data columns: {df.columns}")

        # Ensure we have all requested tickers (some might have failed silently)
        # Note: yfinance might change case, so we upper() everything for comparison
        data.columns = [str(c).upper() for c in data.columns]
        found_tickers = [t for t in ticker_list if t in data.columns]
        
        if not found_tickers:
            # Fallback: if we only have 1 column and 1 requested ticker, assume it's a match
            if data.shape[1] == 1 and len(ticker_list) == 1:
                data.columns = ticker_list
            else:
                raise ValueError("None of the requested tickers were found in the response.")

        # Filter to just the valid ones found
        data = data[found_tickers]
        
        # Cleaning
        data = data.ffill().bfill()
        
        # Calculate Log Returns
        log_returns = np.log(data / data.shift(1))
        # Drop the first NaN row created by returns
        log_returns = log_returns.dropna(how='all')
        
        return data, log_returns

    except Exception as e:
        print(f"[!] Error processing data: {e}")
        try: print(f"    Debug: Columns found -> {df.columns}")
        except: pass
        sys.exit(1)

# -----------------------------------------------------------------------------
# 3. Calculation Logic
# -----------------------------------------------------------------------------
def calculate_metrics(log_returns, window, min_periods, is_single_ticker):
    print("[*] Computing rolling metrics...")
    
    # 1. Rolling Z-Scores per ticker
    rolling_mean = log_returns.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = log_returns.rolling(window=window, min_periods=min_periods).std()
    z_scores = (log_returns - rolling_mean) / rolling_std
    
    metrics = {
        'z_scores': z_scores,
        'dispersion': None,
        'avg_correlation': None,
        'corr_matrix': None,
        'metric_labels': {}
    }

    if not is_single_ticker:
        # --- Multi-Ticker Logic ---
        metrics['metric_labels'] = {
            'disp': 'Cross-Sectional Dispersion (Std Dev)',
            'corr': 'Cross-Sectional Avg Correlation'
        }

        # A. Dispersion
        metrics['dispersion'] = log_returns.std(axis=1)
        
        # B. Average Correlation
        roll_corr_avg = pd.Series(index=log_returns.index, dtype=float)
        
        print("[*] Computing rolling correlations (this may take a moment)...")
        # Optimized rolling correlation loop
        # Calculate rolling correlation matrices
        rolling_corr = log_returns.rolling(window=window, min_periods=min_periods).corr()
        
        # We need to average off-diagonal elements for each date
        # Strategy: Get values, reshape, mask diagonal
        
        dates = log_returns.index
        avg_corrs = []
        
        for d in dates:
            try:
                # Slice safely using index
                mat = rolling_corr.loc[d]
                if mat.isna().all().all():
                    avg_corrs.append(np.nan)
                    continue
                
                # Convert to numpy to extract off-diagonals
                vals = mat.values
                # Create mask for off-diagonals
                mask = ~np.eye(vals.shape[0], dtype=bool)
                off_diag = vals[mask]
                
                if len(off_diag) > 0:
                    avg_corrs.append(np.nanmean(off_diag))
                else:
                    avg_corrs.append(np.nan)
            except KeyError:
                avg_corrs.append(np.nan)
                
        metrics['avg_correlation'] = pd.Series(avg_corrs, index=dates)
        
        # C. Correlation Matrix (Latest Window)
        metrics['corr_matrix'] = log_returns.tail(window).corr()

    else:
        # --- Single-Ticker Logic ---
        metrics['metric_labels'] = {
            'disp': 'Realized Volatility',
            'corr': 'Rolling Lag-1 Auto-Correlation'
        }
        
        # A. Dispersion (Realized Volatility)
        metrics['dispersion'] = log_returns.iloc[:, 0].rolling(window=window, min_periods=min_periods).std()
        
        # B. Average Correlation (Auto-Correlation)
        s = log_returns.iloc[:, 0]
        metrics['avg_correlation'] = s.rolling(window=window, min_periods=min_periods).corr(s.shift(1))
        
        # C. Lag-Lag Matrix
        lags_df = pd.DataFrame()
        ticker_name = log_returns.columns[0]
        max_lags = min(10, window // 2)
        base_series = log_returns[ticker_name].dropna()
        
        for i in range(max_lags + 1):
            lags_df[f'Lag_{i}'] = base_series.shift(i)
            
        metrics['corr_matrix'] = lags_df.tail(window).corr()

    return metrics

# -----------------------------------------------------------------------------
# 4. Visualization (Plotly)
# -----------------------------------------------------------------------------
def generate_plots(log_returns, metrics, is_single_ticker):
    print("[*] Generating visualizations...")
    
    plots = {}
    dates = log_returns.index
    disp = metrics['dispersion']
    corr = metrics['avg_correlation']
    z_scores = metrics['z_scores']
    
    layout_args = dict(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )

    # 1. Z-Score Heatmap
    # Handle NaNs in Z-score for cleaner heatmap (replace with 0 or drop)
    z_clean = z_scores.fillna(0)
    
    fig1 = go.Figure(data=go.Heatmap(
        z=z_clean.T.values,
        x=dates,
        y=z_scores.columns,
        colorscale='RdBu',
        zmin=-3,
        zmax=3,
        colorbar=dict(title="Z-Score")
    ))
    fig1.update_layout(title="Rolling Z-Scores of Returns", **layout_args)
    plots['heatmap'] = fig1

    # 2. Dispersion Line Chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=disp, mode='lines', name='Dispersion', line=dict(color='#00CC96')))
    fig2.update_layout(title=metrics['metric_labels']['disp'], **layout_args)
    plots['dispersion'] = fig2

    # 3. Average Correlation Line Chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dates, y=corr, mode='lines', name='Correlation', line=dict(color='#AB63FA')))
    fig3.update_layout(title=metrics['metric_labels']['corr'], **layout_args)
    plots['avg_corr'] = fig3

    # 4. Dual-Axis Chart
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(x=dates, y=disp, name="Dispersion", line=dict(color='#00CC96')), secondary_y=False)
    fig4.add_trace(go.Scatter(x=dates, y=corr, name="Correlation", line=dict(color='#AB63FA')), secondary_y=True)
    
    fig4.update_layout(title="Dispersion vs. Correlation (Dual Axis)", **layout_args)
    fig4.update_yaxes(title_text="Dispersion", secondary_y=False, title_font=dict(color='#00CC96'))
    fig4.update_yaxes(title_text="Correlation", secondary_y=True, title_font=dict(color='#AB63FA'))
    plots['dual_axis'] = fig4

    # 5. Scatter
    # Ensure dimensions match for scatter
    common_idx = disp.index.intersection(corr.index)
    d_scatter = disp.loc[common_idx]
    c_scatter = corr.loc[common_idx]
    dates_scatter = common_idx
    
    time_colors = np.linspace(0, 1, len(common_idx))
    
    fig5 = go.Figure(data=go.Scatter(
        x=d_scatter,
        y=c_scatter,
        mode='markers',
        text=dates_scatter.strftime('%Y-%m-%d'),
        marker=dict(
            size=6,
            color=time_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time Progression")
        )
    ))
    fig5.update_layout(
        title="Dispersion vs. Correlation Scatter (Colored by Time)",
        xaxis_title=metrics['metric_labels']['disp'],
        yaxis_title=metrics['metric_labels']['corr'],
        hovermode='closest',
        template="plotly_dark"
    )
    plots['scatter'] = fig5

    # 6. Matrix
    mat = metrics['corr_matrix']
    if mat is not None:
        fig6 = go.Figure(data=go.Heatmap(
            z=mat.values,
            x=mat.columns,
            y=mat.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        title_mat = "Correlation Matrix (Latest Window)" if not is_single_ticker else "Lag-Lag Auto-Correlation Matrix (Latest Window)"
        fig6.update_layout(title=title_mat, height=700, **layout_args)
        plots['matrix'] = fig6
    else:
        plots['matrix'] = go.Figure()

    # 7. Regime Map
    disp_z = ((disp - disp.mean()) / disp.std())
    corr_z = ((corr - corr.mean()) / corr.std())
    
    # Align indices again
    common_idx_z = disp_z.index.intersection(corr_z.index)
    dz = disp_z.loc[common_idx_z]
    cz = corr_z.loc[common_idx_z]
    
    time_colors_z = np.linspace(0, 1, len(common_idx_z))
    
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=dz,
        y=cz,
        mode='markers+lines',
        text=common_idx_z.strftime('%Y-%m-%d'),
        line=dict(color='rgba(200,200,200,0.3)', width=1),
        marker=dict(
            size=6,
            color=time_colors_z,
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="Time")
        )
    ))
    fig7.add_hline(y=0, line_dash="dash", line_color="gray")
    fig7.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Add quadrant labels if we have data bounds
    if not dz.empty:
        max_d = dz.max()
        min_d = dz.min()
        max_c = cz.max()
        min_c = cz.min()
        
        # Scale offsets slightly
        offset_x = (max_d - min_d) * 0.1
        offset_y = (max_c - min_c) * 0.1
        
        fig7.add_annotation(x=max_d-offset_x, y=max_c-offset_y, text="High Vol / High Corr", showarrow=False, font=dict(color="red"))
        fig7.add_annotation(x=min_d+offset_x, y=max_c-offset_y, text="Low Vol / High Corr", showarrow=False, font=dict(color="orange"))
        fig7.add_annotation(x=max_d-offset_x, y=min_c+offset_y, text="High Vol / Low Corr", showarrow=False, font=dict(color="orange"))
        fig7.add_annotation(x=min_d+offset_x, y=min_c+offset_y, text="Low Vol / Low Corr", showarrow=False, font=dict(color="green"))
    
    fig7.update_layout(
        title="Regime Map (Z-Score Space)",
        xaxis_title="Dispersion Z-Score",
        yaxis_title="Correlation Z-Score",
        hovermode='closest',
        template="plotly_dark"
    )
    plots['regime'] = fig7

    return plots

# -----------------------------------------------------------------------------
# 5. HTML Generation
# -----------------------------------------------------------------------------
def build_html_dashboard(plots, output_file):
    print(f"[*] Building HTML dashboard: {output_file}")
    
    # FIX: Get Plotly JS source code (so it works offline/without CDN)
    print("    Embedding Plotly JS library (approx 3MB)...")
    plotly_js = py_offline.get_plotlyjs()

    # Generate divs without script tags (we embed the script once in head)
    divs = {}
    for key, fig in plots.items():
        divs[key] = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Financial Dispersion Dashboard</title>
    <script type="text/javascript">{plotly_js}</script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }}
        h1 {{ text-align: center; margin-bottom: 20px; color: #ffffff; }}
        
        .tab {{
            overflow: hidden;
            border: 1px solid #333;
            background-color: #1e1e1e;
            border-radius: 5px 5px 0 0;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 20px;
            transition: 0.3s;
            font-size: 16px;
            color: #bbb;
        }}
        
        .tab button:hover {{ background-color: #333; color: white; }}
        .tab button.active {{ background-color: #007bff; color: white; }}
        
        .tabcontent {{
            display: none;
            padding: 20px;
            border: 1px solid #333;
            border-top: none;
            background-color: #1e1e1e;
            animation: fadeEffect 0.5s;
            border-radius: 0 0 5px 5px;
        }}
        
        @keyframes fadeEffect {{
            from {{opacity: 0;}}
            to {{opacity: 1;}}
        }}
        
        .plot-container {{ width: 100%; height: 80vh; }}
    </style>
</head>
<body>

    <h1>Market Dispersion & Correlation Analysis</h1>

    <div class="tab">
        <button class="tablinks" onclick="openTab(event, 'Tab1')" id="defaultOpen">Z-Score Heatmap</button>
        <button class="tablinks" onclick="openTab(event, 'Tab2')">Dispersion</button>
        <button class="tablinks" onclick="openTab(event, 'Tab3')">Avg Correlation</button>
        <button class="tablinks" onclick="openTab(event, 'Tab4')">Dual Axis</button>
        <button class="tablinks" onclick="openTab(event, 'Tab5')">Scatter Analysis</button>
        <button class="tablinks" onclick="openTab(event, 'Tab6')">Correlation Matrix</button>
        <button class="tablinks" onclick="openTab(event, 'Tab7')">Regime Map</button>
    </div>

    <div id="Tab1" class="tabcontent">
        <div class="plot-container">{divs['heatmap']}</div>
    </div>

    <div id="Tab2" class="tabcontent">
        <div class="plot-container">{divs['dispersion']}</div>
    </div>

    <div id="Tab3" class="tabcontent">
        <div class="plot-container">{divs['avg_corr']}</div>
    </div>

    <div id="Tab4" class="tabcontent">
        <div class="plot-container">{divs['dual_axis']}</div>
    </div>
    
    <div id="Tab5" class="tabcontent">
        <div class="plot-container">{divs['scatter']}</div>
    </div>

    <div id="Tab6" class="tabcontent">
        <div class="plot-container">{divs['matrix']}</div>
    </div>
    
    <div id="Tab7" class="tabcontent">
        <div class="plot-container">{divs['regime']}</div>
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
        if (evt) {{
            evt.currentTarget.className += " active";
        }}
        
        // --- FIX FOR BLANK PLOTS IN TABS ---
        // 1. Dispatch global resize
        window.dispatchEvent(new Event('resize'));
        
        // 2. Explicitly tell Plotly to resize the visible container
        setTimeout(function() {{
            var activeTab = document.getElementById(tabName);
            var plots = activeTab.querySelectorAll('.plotly-graph-div');
            for (var j = 0; j < plots.length; j++) {{
                Plotly.Plots.resize(plots[j]);
            }}
        }}, 50);
    }}
    
    document.getElementById("defaultOpen").click();
    </script>

</body>
</html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

# -----------------------------------------------------------------------------
# 6. Main Execution Block
# -----------------------------------------------------------------------------
def main():
    args = parse_arguments()
    
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = [args.ticker.strip().upper()]
        
    is_single = (len(tickers) == 1)
    
    prices, log_rets = get_data(tickers, args.period)
    
    metrics = calculate_metrics(log_rets, args.roll_window, args.min_periods, is_single)
    
    plots = generate_plots(log_rets, metrics, is_single)
    
    build_html_dashboard(plots, args.out)
    
    abs_path = os.path.abspath(args.out)
    print(f"[*] Dashboard ready: {abs_path}")
    webbrowser.open(f'file://{abs_path}')

if __name__ == "__main__":
    main()
