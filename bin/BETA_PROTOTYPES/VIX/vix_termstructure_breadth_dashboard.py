# SCRIPTNAME: ok.vix_termstructure_breadth_dashboard.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
SCRIPT: vix_termstructure_breadth_dashboard.py
AUTHOR: Michael Derby
PROJECT: STEGO Financial Framework
DESCRIPTION: 
    A standalone volatility regime analysis dashboard.
    - Analyzes Volatility Term Structure (VIX vs VIX3M vs VIX6M).
    - Synthesizes Market Breadth (SPX % > 50SMA).
    - Identifies "Quiet but Brittle" risk regimes.
    - Outputs a multi-tab interactive Plotly HTML dashboard.

DEPENDENCIES:
    - data_retrieval (local module)
    - yfinance
    - pandas
    - numpy
    - plotly

OUTPUT:
    /dev/shm/VIX_TERMSTRUCTURE_DASHBOARD/YYYY-MM-DD/
"""

import os
import sys
import datetime
import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Ensure local data_retrieval module can be imported
sys.path.append(os.getcwd())

try:
    import data_retrieval as dr
except ImportError:
    print("CRITICAL ERROR: 'data_retrieval.py' not found in the script directory.")
    print("Please ensure your custom data retrieval library is present.")
    sys.exit(1)

# --- CONFIGURATION ---
TODAY_STR = datetime.datetime.now().strftime('%Y-%m-%d')
BASE_OUTPUT_DIR = f"/dev/shm/VIX_TERMSTRUCTURE_DASHBOARD/{TODAY_STR}"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Styling Constants
COLOR_VIX = '#00F0FF'    # Cyan
COLOR_VIX3M = '#FF00E6'  # Magenta
COLOR_VIX6M = '#FFE600'  # Yellow
COLOR_RATIO = '#FFFFFF'  # White
COLOR_BULL = '#00FF9D'   # Spring Green
COLOR_BEAR = '#FF4D4D'   # Red
COLOR_WARN = '#FF9F1C'   # Orange
BG_COLOR = '#0e1117'     # Dark Background
GRID_COLOR = '#2d3436'

# --- 1. DATA LOADING & SYNTHESIS ---

def get_spx_constituents():
    """Fetches current S&P 500 tickers from Wikipedia."""
    print(" [STEGO] Fetching S&P 500 constituents from Wikipedia...")
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Clean tickers (replace dots with dashes for yfinance, e.g., BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        print(f" [STEGO] Error fetching constituents: {e}. Using fallback top holdings.")
        # Fallback list of top weighting stocks to approximate breadth if Wiki fails
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "AVGO", "JPM", "V"]

def synthesize_breadth_spxa50r(lookback_days=365):
    """
    Synthesizes % of SPX stocks above 50-day SMA.
    Downloads history for all constituents, computes SMA50, aggregates.
    """
    print(" [STEGO] Synthesizing SPXA50R Breadth (this may take time sequentially)...")
    
    tickers = get_spx_constituents()
    # Limit lookback to speed up calculation
    start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days + 100)).strftime('%Y-%m-%d')
    
    above_count = pd.Series(0, index=pd.date_range(start=start_date, end=TODAY_STR, freq='B'))
    total_count = pd.Series(0, index=pd.date_range(start=start_date, end=TODAY_STR, freq='B'))
    
    # We will use dr.get_stock_data which caches locally
    processed = 0
    for t in tickers:
        try:
            df = dr.get_stock_data(t, period="2y") # 2y to ensure enough for SMA
            if df.empty: continue
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            # Filter to relevant range
            df = df.sort_index()
            
            # Compute SMA50 if not present
            if 'SMA_50' not in df.columns:
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Reindex to master timeline to align dates
            df_reindexed = df[['Close', 'SMA_50']].reindex(total_count.index, method='ffill')
            
            # Check condition
            is_above = (df_reindexed['Close'] > df_reindexed['SMA_50']).astype(int)
            not_nan = df_reindexed['Close'].notna().astype(int)
            
            above_count = above_count.add(is_above, fill_value=0)
            total_count = total_count.add(not_nan, fill_value=0)
            
            processed += 1
            if processed % 50 == 0:
                print(f"   ...processed {processed}/{len(tickers)} tickers")
                
        except Exception as e:
            continue

    # Compute Percentage
    breadth_series = (above_count / total_count) * 100
    breadth_series = breadth_series.dropna()
    
    # Save raw breadth data
    breadth_df = pd.DataFrame({'SPXA50R_Synth': breadth_series})
    breadth_df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'breadth_source_data.csv'))
    
    return breadth_df

def load_data():
    """Main data orchestration function."""
    print(" [STEGO] Loading VIX Complex Data...")
    
    # 1. Volatility Data
    vix = dr.get_stock_data("^VIX", period="2y")
    vix3m = dr.get_stock_data("^VIX3M", period="2y")
    vix6m = dr.get_stock_data("^VIX6M", period="2y")
    
    # 2. SPX Data (for reference)
    spx = dr.get_stock_data("^GSPC", period="2y")
    
    # 3. Breadth Data
    # Try fetching official ticker first, if fail (likely), synthesize
    breadth = dr.get_stock_data("SPXA50R", period="2y") # Usually fails on free YF
    if breadth.empty or len(breadth) < 10:
        print(" [STEGO] Direct SPXA50R download failed. Initiating synthesis protocol.")
        breadth = synthesize_breadth_spxa50r()
    else:
        breadth = breadth.rename(columns={'Close': 'SPXA50R_Synth'})

    # Consolidate into Master DataFrame
    # Use VIX index as master
    master = pd.DataFrame(index=vix.index)
    master['VIX'] = vix['Close']
    master['VIX3M'] = vix3m['Close']
    master['VIX6M'] = vix6m['Close']
    master['SPX'] = spx['Close']
    
    # Merge Breadth (align dates)
    master = master.join(breadth['SPXA50R_Synth'], how='left')
    master['SPXA50R'] = master['SPXA50R_Synth'].interpolate(method='linear') # Fill gaps
    
    # Save Raw Data
    vix.to_csv(os.path.join(BASE_OUTPUT_DIR, 'raw_vix.csv'))
    vix3m.to_csv(os.path.join(BASE_OUTPUT_DIR, 'raw_vix3m.csv'))
    vix6m.to_csv(os.path.join(BASE_OUTPUT_DIR, 'raw_vix6m.csv'))
    master.to_csv(os.path.join(BASE_OUTPUT_DIR, 'master_data_raw.csv'))
    
    return master.dropna(subset=['VIX', 'VIX3M'])

# --- 2. BACKEND COMPUTATIONS ---

def compute_indicators(df):
    """Computes term structure ratios, slopes, and risk flags."""
    df = df.copy()
    
    # Ratios
    df['Ratio_VIX_VIX3M'] = df['VIX'] / df['VIX3M']
    
    # Slopes (Points)
    df['Slope_Short'] = df['VIX3M'] - df['VIX']
    df['Slope_Long'] = df['VIX6M'] - df['VIX3M']
    
    # Regimes
    # Contango: VIX < VIX3M (Ratio < 1). Normal market.
    # Backwardation: VIX > VIX3M (Ratio > 1). Stress.
    df['Contango_Flag'] = np.where(df['Ratio_VIX_VIX3M'] < 1.0, 1, 0)
    
    # "Quiet but Brittle" Risk Lens
    # Criteria: Contango (Ratio < 1) BUT Breadth is weak (< 40%) or falling
    # We use < 40% as the brittle threshold here
    df['Risk_Brittle'] = np.where(
        (df['Ratio_VIX_VIX3M'] < 1.0) & (df['SPXA50R'] < 40.0), 
        1, 0
    )
    
    # Save Indicators
    df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'master_indicators.csv'))
    return df

# --- 3. VISUALIZATION BUILDING BLOCKS ---

def create_tab_term_structure(df):
    """TAB 1: Volatility Term Structure Analysis."""
    
    # Layout: 2x2 Grid
    # Row 1, Col 1: VIX Levels
    # Row 1, Col 2: Ratio
    # Row 2, Col 1: Heatmap (Slope)
    # Row 2, Col 2: Radar (Current Shape)
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "polar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        subplot_titles=("VIX Complex Levels", "VIX/VIX3M Ratio (Signal)", "Term Structure Slope Heatmap", "Current Term Structure Shape")
    )

    # 1. Levels
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name='VIX', line=dict(color=COLOR_VIX, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX3M'], name='VIX3M', line=dict(color=COLOR_VIX3M, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX6M'], name='VIX6M', line=dict(color=COLOR_VIX6M, width=1.5)), row=1, col=1)

    # 2. Ratio
    fig.add_trace(go.Scatter(x=df.index, y=df['Ratio_VIX_VIX3M'], name='Ratio (Front/3M)', line=dict(color=COLOR_RATIO, width=1)), row=1, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=1, col=2, annotation_text="Contango/Backw. Flip")

    # 3. Heatmap (Slope)
    # We construct a heatmap where X=Date, Y=Tenor Segment, Z=Slope Value
    # Segment 0: Spot-3M (Slope Short), Segment 1: 3M-6M (Slope Long)
    slope_matrix = np.array([df['Slope_Short'].values, df['Slope_Long'].values])
    fig.add_trace(go.Heatmap(
        z=slope_matrix,
        x=df.index,
        y=['1M-3M Slope', '3M-6M Slope'],
        colorscale='RdBu', # Red = Negative (Backwardation/Stress), Blue = Positive (Contango/Calm)
        zmid=0,
        showscale=True,
        colorbar=dict(len=0.4, y=0.2)
    ), row=2, col=1)

    # 4. Radar (Last Data Point)
    last_row = df.iloc[-1]
    # Normalize for Radar: Simple Raw Values
    radar_vals = [last_row['VIX'], last_row['VIX3M'], last_row['VIX6M'], last_row['VIX']]
    radar_theta = ['VIX (1M)', 'VIX3M', 'VIX6M', 'VIX (1M)']
    
    fig.add_trace(go.Scatterpolar(
        r=radar_vals,
        theta=radar_theta,
        fill='toself',
        name='Current Shape',
        line=dict(color=COLOR_VIX)
    ), row=2, col=2)

    fig.update_layout(height=800, template='plotly_dark', title_text="Tab 1: Volatility Term Structure Dynamics")
    return fig

def create_tab_breadth(df):
    """TAB 2: Breadth Context (SPXA50R)."""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("SPX % Above 50SMA", "Rolling 30D Correlation (VIX vs Breadth)", "Scatter: Breadth vs VIX")
    )

    # 1. Breadth Line with Zones
    fig.add_trace(go.Scatter(x=df.index, y=df['SPXA50R'], name='SPXA50R', line=dict(color='#00ff00', width=2)), row=1, col=1)
    # Add colored bands
    fig.add_hrect(y0=50, y1=100, fillcolor="green", opacity=0.1, line_width=0, row=1, col=1)
    fig.add_hrect(y0=30, y1=50, fillcolor="yellow", opacity=0.1, line_width=0, row=1, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1, line_width=0, row=1, col=1)

    # 2. Rolling Correlation
    rolling_corr = df['VIX'].rolling(30).corr(df['SPXA50R'])
    fig.add_trace(go.Scatter(x=df.index, y=rolling_corr, name='30D Corr', fill='tozeroy', line=dict(color='orange')), row=2, col=1)

    # 3. Scatter (Separate X-axis logic needed, so we actually make this a separate figure logic usually, 
    # but for subplots shared_x implies time. We will disable shared_x for row 3 manually or just plot time series here)
    # Actually, let's do Breadth vs VIX over time for consistency in this pane, or insert a scatter separately.
    # To strictly follow the "Scatter" request in a subplot grid:
    fig.add_trace(go.Scatter(
        x=df['SPXA50R'], 
        y=df['VIX'], 
        mode='markers', 
        marker=dict(color=df.index.astype('int64'), colorscale='Viridis', showscale=False),
        name='Breadth vs VIX'
    ), row=3, col=1)
    
    # Update axes for Scatter (Row 3) to not be time-based if shared_x is True? 
    # Plotly handles mixed axes types in subplots if specified.
    # We will just not share X for the last row.
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Breadth %", row=3, col=1)
    fig.update_yaxes(title_text="VIX Level", row=3, col=1)

    fig.update_layout(height=900, template='plotly_dark', title_text="Tab 2: Market Breadth Analysis")
    return fig

def create_tab_risk_lens(df):
    """TAB 3: 'Quiet or Brittle?' Risk Lens."""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Left Axis: Ratio
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Ratio_VIX_VIX3M'], 
        name='VIX/VIX3M Ratio', 
        line=dict(color=COLOR_RATIO, width=1)
    ), secondary_y=False)
    
    # Right Axis: Breadth
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SPXA50R'], 
        name='Breadth (SPXA50R)', 
        line=dict(color=COLOR_BULL, width=1, dash='dot')
    ), secondary_y=True)
    
    # Highlight Brittle Regions
    brittle_dates = df[df['Risk_Brittle'] == 1].index
    brittle_vals = df[df['Risk_Brittle'] == 1]['Ratio_VIX_VIX3M']
    
    fig.add_trace(go.Scatter(
        x=brittle_dates, y=brittle_vals,
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='red'),
        name='Brittle Warning'
    ), secondary_y=False)
    
    fig.update_layout(
        height=600, 
        template='plotly_dark', 
        title_text="Tab 3: Risk Lens - Quiet but Brittle?",
        xaxis_title="Date",
        yaxis_title="VIX Ratio",
        yaxis2_title="Breadth %"
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", secondary_y=False)
    
    return fig

def create_tab_3d_surface(df):
    """TAB 4: 3D Volatility Surface."""
    
    # Prepare Mesh
    # X = Dates (convert to numerical for surface, then map back ticks ideally, but Plotly handles dates in 3D okay)
    # Y = Tenor [1, 3, 6]
    # Z = Vol
    
    # Resample to ensure smooth grid if needed, but we have daily.
    # Pivot for Surface format: Rows=Tenor, Cols=Date
    
    # Construct Grid
    x_data = df.index
    y_data = np.array([1, 3, 6]) # Months
    
    # Z matrix needs to be (len(y), len(x))
    z_data = np.vstack([
        df['VIX'].values,
        df['VIX3M'].values,
        df['VIX6M'].values
    ])
    
    fig = go.Figure(data=[go.Surface(
        z=z_data,
        x=x_data,
        y=y_data,
        colorscale='Viridis',
        opacity=0.9
    )])
    
    fig.update_layout(
        title='Tab 4: Implied Volatility Term Structure Surface',
        scene=dict(
            xaxis_title='Date',
            yaxis_title='Tenor (Months)',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ),
        template='plotly_dark',
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

def create_summary_html(df):
    """TAB 5: Text Summary Generation."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    state = "CONTANGO (Normal)" if last['Contango_Flag'] == 1 else "BACKWARDATION (Stress)"
    state_color = COLOR_BULL if last['Contango_Flag'] == 1 else COLOR_BEAR
    
    brittle_status = "YES - CAUTION" if last['Risk_Brittle'] == 1 else "NO"
    brittle_color = "red" if last['Risk_Brittle'] == 1 else "gray"
    
    html = f"""
    <div style="font-family: monospace; padding: 20px; color: #e0e0e0; background-color: {BG_COLOR};">
        <h2>STEGO Volatility Dashboard Summary</h2>
        <p><strong>Date:</strong> {last.name.strftime('%Y-%m-%d')}</p>
        <hr style="border-color: #444;">
        
        <h3 style="color: {state_color};">MARKET REGIME: {state}</h3>
        
        <table style="width: 50%; text-align: left; color: #ccc;">
            <tr><th>Metric</th><th>Value</th><th>Change</th></tr>
            <tr><td>VIX (Spot)</td><td>{last['VIX']:.2f}</td><td>{last['VIX'] - prev['VIX']:.2f}</td></tr>
            <tr><td>VIX3M</td><td>{last['VIX3M']:.2f}</td><td>{last['VIX3M'] - prev['VIX3M']:.2f}</td></tr>
            <tr><td>VIX6M</td><td>{last['VIX6M']:.2f}</td><td>{last['VIX6M'] - prev['VIX6M']:.2f}</td></tr>
            <tr><td>Ratio (Front/3M)</td><td>{last['Ratio_VIX_VIX3M']:.3f}</td><td>{last['Ratio_VIX_VIX3M'] - prev['Ratio_VIX_VIX3M']:.3f}</td></tr>
        </table>
        
        <br>
        <h3>Breadth & Risk</h3>
        <p><strong>SPXA50R (% > 50SMA):</strong> <span style="color: {COLOR_VIX};">{last['SPXA50R']:.1f}%</span></p>
        <p><strong>Quiet but Brittle?</strong> <span style="color: {brittle_color}; font-weight: bold;">{brittle_status}</span></p>
        <p><em>Condition: Ratio < 1.0 AND Breadth < 40%</em></p>
    </div>
    """
    return html

# --- 4. DASHBOARD ASSEMBLY ---

def build_html_dashboard(figs, summary_html):
    """Assembles all figures into a single HTML file with Tabs."""
    
    # Convert figs to HTML div strings
    divs = {}
    for name, fig in figs.items():
        divs[name] = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>STEGO VIX Term Structure Dashboard</title>
        <style>
            body {{ font-family: sans-serif; background-color: #0e1117; color: white; margin: 0; }}
            .tab {{ overflow: hidden; border-bottom: 1px solid #ccc; background-color: #1e1e1e; }}
            .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; color: #888; }}
            .tab button:hover {{ background-color: #333; color: white; }}
            .tab button.active {{ background-color: #00F0FF; color: black; font-weight: bold; }}
            .tabcontent {{ display: none; padding: 6px 12px; border-top: none; animation: fadeEffect 1s; }}
            @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
        </style>
    </head>
    <body>

    <div class="tab">
      <button class="tablinks" onclick="openTab(event, 'Tab1')" id="defaultOpen">Term Structure</button>
      <button class="tablinks" onclick="openTab(event, 'Tab2')">Breadth Context</button>
      <button class="tablinks" onclick="openTab(event, 'Tab3')">Risk Lens</button>
      <button class="tablinks" onclick="openTab(event, 'Tab4')">3D Surface</button>
      <button class="tablinks" onclick="openTab(event, 'Tab5')">Summary</button>
    </div>

    <div id="Tab1" class="tabcontent">{divs['ts']}</div>
    <div id="Tab2" class="tabcontent">{divs['breadth']}</div>
    <div id="Tab3" class="tabcontent">{divs['risk']}</div>
    <div id="Tab4" class="tabcontent">{divs['surface']}</div>
    <div id="Tab5" class="tabcontent">{summary_html}</div>

    <script>
    function openTab(evt, tabName) {{
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
    }}
    document.getElementById("defaultOpen").click();
    </script>
    
    </body>
    </html>
    """
    
    output_path = os.path.join(BASE_OUTPUT_DIR, 'dashboard.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path

# --- MAIN EXECUTION ---

def main():
    print(f" [STEGO] Starting VIX Term Structure Dashboard build for {TODAY_STR}")
    print(f" [STEGO] Output directory: {BASE_OUTPUT_DIR}")
    
    # 1. Load Data
    df = load_data()
    
    # 2. Compute
    df = compute_indicators(df)
    
    # 3. Build Figures
    print(" [STEGO] Generating Visualization Objects...")
    figs = {
        'ts': create_tab_term_structure(df),
        'breadth': create_tab_breadth(df),
        'risk': create_tab_risk_lens(df),
        'surface': create_tab_3d_surface(df)
    }
    
    # 4. Generate Summary
    summary = create_summary_html(df)
    
    # 5. Compile Dashboard
    print(" [STEGO] Compiling HTML Dashboard...")
    dashboard_path = build_html_dashboard(figs, summary)
    
    print(f" [STEGO] SUCCESS. Dashboard saved to: {dashboard_path}")
    
    # 6. Auto-open
    try:
        webbrowser.open(f'file://{os.path.abspath(dashboard_path)}')
    except Exception:
        print(" [STEGO] Could not auto-open browser. Please open file manually.")

if __name__ == "__main__":
    main()
