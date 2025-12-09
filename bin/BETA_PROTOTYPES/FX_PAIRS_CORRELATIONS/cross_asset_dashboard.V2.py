# SCRIPTNAME: ok.cross_asset_dashboard.V2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import datetime
import time

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS & CONFIG
# -----------------------------------------------------------------------------

TICKERS = {
    'SPY': 'Equities Proxy',
    'TLT': 'Long Bonds Proxy',
    'UUP': 'US Dollar Proxy',
    'GC=F': 'Gold Futures',
    'CL=F': 'WTI Crude Oil',
    '^VIX': 'VIX Index' # Using ^VIX for better reliability than VIX ticker
}

COLORS = {
    'SPY': '#1f77b4', 'TLT': '#2ca02c', 'UUP': '#9467bd',
    'GC=F': '#ff7f0e', 'CL=F': '#d62728', '^VIX': '#7f7f7f'
}

def sanitize_data(df):
    """Ensure timezone naive index and numeric columns."""
    if df.empty:
        return df
    # Fix MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Timezone naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df

def calculate_zscore(series, window=252):
    """Calculate rolling z-score over a 1-year window."""
    r_mean = series.rolling(window=window).mean()
    r_std = series.rolling(window=window).std()
    z = (series - r_mean) / r_std
    return z

# -----------------------------------------------------------------------------
# 2. DATA DOWNLOAD
# -----------------------------------------------------------------------------

def download_data():
    print("📥 Downloading data...")
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=730) # 2 years
    
    data = {}
    for ticker in TICKERS.keys():
        print(f"   Fetching {ticker}...")
        try:
            # Download individual ticker to ensure clean structure
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            df = sanitize_data(df)
            
            # Prefer 'Adj Close', fallback to 'Close'
            if 'Adj Close' in df.columns:
                data[ticker] = df['Adj Close']
            elif 'Close' in df.columns:
                data[ticker] = df['Close']
            else:
                print(f"⚠️ Warning: No close data for {ticker}")
            
            # Sleep to be polite to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error downloading {ticker}: {e}")
            
    # Combine into single DataFrame
    prices = pd.DataFrame(data)
    prices = prices.ffill().dropna()
    print(f"✅ Data ready: {prices.shape[0]} rows, {prices.shape[1]} columns.")
    return prices

# -----------------------------------------------------------------------------
# 3. METRIC COMPUTATION
# -----------------------------------------------------------------------------

def compute_metrics(prices):
    print("🧮 Computing financial metrics...")
    
    # Returns
    log_rets = np.log(prices / prices.shift(1))
    
    # 1. Rolling Correlation (60-day)
    # We compute the full correlation matrix for every day
    # Rolling correlation is computationally expensive if done via pandas rolling.corr() on full DF
    # We will do it efficiently
    rolling_corr = log_rets.rolling(window=60).corr()
    
    # 2. Volatility Metrics
    rv_21 = log_rets.rolling(window=21).std() * np.sqrt(252) * 100
    rv_63 = log_rets.rolling(window=63).std() * np.sqrt(252) * 100
    
    # 3. Macro Regime (Returns over 60 days)
    rolling_ret_60d = prices.pct_change(60)
    
    # 4. Dollar-Commodity Correlation
    # UUP vs Gold
    uup_gold_corr = log_rets['UUP'].rolling(60).corr(log_rets['GC=F'])
    # UUP vs Oil
    uup_oil_corr = log_rets['UUP'].rolling(60).corr(log_rets['CL=F'])
    
    # 5. Trend (EMAs)
    ema_50 = prices.ewm(span=50, adjust=False).mean()
    ema_200 = prices.ewm(span=200, adjust=False).mean()
    
    # MACD
    exp12 = prices.ewm(span=12, adjust=False).mean()
    exp26 = prices.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # VIX Term Structure Proxy (Just using VIX level relative to history here as we don't have VIX3M easily in yf)
    # We will normalize VIX
    vix_z = calculate_zscore(prices['^VIX'], window=252)
    
    return {
        'prices': prices,
        'log_rets': log_rets,
        'rolling_corr': rolling_corr,
        'rv_21': rv_21,
        'rv_63': rv_63,
        'rolling_ret_60d': rolling_ret_60d,
        'uup_gold_corr': uup_gold_corr,
        'uup_oil_corr': uup_oil_corr,
        'ema_50': ema_50,
        'ema_200': ema_200,
        'macd': macd,
        'signal': signal,
        'vix_z': vix_z
    }

# -----------------------------------------------------------------------------
# 4. VISUALIZATION GENERATORS
# -----------------------------------------------------------------------------

def create_heatmap(metrics):
    """A: Rolling Correlation Heatmap (Current Snapshot)"""
    rc = metrics['rolling_corr']
    # Get last valid correlation matrix
    last_date = rc.index.get_level_values(0)[-1]
    curr_corr = rc.loc[last_date]
    
    # Detect high Z-scores (Comparing current corr to its own 1y history)
    # This requires unpacking the multi-index rolling corr series which is complex
    # Simplified approach: Mark high correlations > 0.8 or < -0.8
    
    z_text = np.round(curr_corr, 2).astype(str)
    
    fig = go.Figure(data=go.Heatmap(
        z=curr_corr.values,
        x=curr_corr.columns,
        y=curr_corr.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=z_text,
        texttemplate="%{text}",
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'60-Day Rolling Correlation Matrix ({last_date.date()})',
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    return fig

def create_volatility_radar(metrics):
    """C: Volatility Radar"""
    rv21 = metrics['rv_21'].iloc[-1]
    rv63 = metrics['rv_63'].iloc[-1]
    vix_level = metrics['prices']['^VIX'].iloc[-1]
    
    # Normalize values for radar (0-100 scale rough approx for visualization)
    # We plot raw annualized vol
    
    categories = ['SPY RV21', 'SPY RV63', 'TLT RV21', 'Gold RV21', 'Oil RV21', 'VIX']
    values = [
        rv21['SPY'], rv63['SPY'], rv21['TLT'], 
        rv21['GC=F'], rv21['CL=F'], vix_level
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Vol Structure',
        line_color='#ff7f0e'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values)*1.2]
            )
        ),
        title="Cross-Asset Volatility Radar (Annualized %)",
        height=500
    )
    return fig

def create_macro_regime_map(metrics):
    """3: Macro Regime Map"""
    spy_ret = metrics['rolling_ret_60d']['SPY'] * 100
    tlt_ret = metrics['rolling_ret_60d']['TLT'] * 100
    vix_vals = metrics['prices']['^VIX']
    dates = spy_ret.index
    
    # Filter last 2 years
    
    fig = go.Figure(data=go.Scatter(
        x=spy_ret,
        y=tlt_ret,
        mode='markers',
        marker=dict(
            size=8,
            color=vix_vals, # Color by VIX
            colorscale='Turbo',
            colorbar=dict(title="VIX Level"),
            showscale=True
        ),
        text=dates.astype(str),
        hovertemplate="<b>Date: %{text}</b><br>SPY 60d: %{x:.1f}%<br>TLT 60d: %{y:.1f}%<br>VIX: %{marker.color:.1f}<extra></extra>"
    ))
    
    # Add quadrants
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Annotations
    fig.add_annotation(x=15, y=15, text="RISK ON / GOLDILOCKS", showarrow=False, font=dict(color="green"))
    fig.add_annotation(x=-15, y=-15, text="CRISIS / DELEVERAGING", showarrow=False, font=dict(color="red"))
    fig.add_annotation(x=-15, y=15, text="STAGFLATION / FLIGHT TO QUALITY", showarrow=False, font=dict(color="orange"))
    fig.add_annotation(x=15, y=-15, text="RATES SHOCK / REFLATION", showarrow=False, font=dict(color="purple"))

    fig.update_layout(
        title="Macro Regime Map (60-Day Returns)",
        xaxis_title="SPY 60-Day Return (%)",
        yaxis_title="TLT 60-Day Return (%)",
        height=600,
        hovermode='closest'
    )
    return fig

def create_dollar_commodity_gauge(metrics):
    """4: Dollar-Commodity Gauge"""
    uup = metrics['prices']['UUP']
    gold = metrics['prices']['GC=F']
    oil = metrics['prices']['CL=F']
    
    uup_gold_corr = metrics['uup_gold_corr']
    uup_oil_corr = metrics['uup_oil_corr']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Plot Correlations
    fig.add_trace(go.Scatter(x=uup_gold_corr.index, y=uup_gold_corr, name="UUP-Gold Corr (60d)", line=dict(color='gold')), secondary_y=False)
    fig.add_trace(go.Scatter(x=uup_oil_corr.index, y=uup_oil_corr, name="UUP-Oil Corr (60d)", line=dict(color='black')), secondary_y=False)
    
    # Plot UUP Price
    fig.add_trace(go.Scatter(x=uup.index, y=uup, name="UUP Price", line=dict(color='purple', dash='dot')), secondary_y=True)
    
    fig.update_layout(
        title="Dollar Strength vs. Commodity Correlations",
        xaxis_title="Date",
        height=500,
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Rolling Correlation", secondary_y=False, range=[-1, 1])
    fig.update_yaxes(title_text="UUP Price", secondary_y=True)
    
    # Add zero line for corr
    fig.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=False)
    
    return fig

def create_trend_dashboard(metrics):
    """5: Trend Dashboard (SPY Focus)"""
    prices = metrics['prices']
    ema50 = metrics['ema_50']
    ema200 = metrics['ema_200']
    macd = metrics['macd']
    signal = metrics['signal']
    
    # Create subplots for each asset? That's too big. Let's do a dropdown or just stack major ones.
    # We will do SPY and TLT stacked
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        subplot_titles=('SPY Price & EMAs', 'SPY MACD', 'TLT Price & EMAs', 'TLT MACD'))
    
    # SPY
    fig.add_trace(go.Scatter(x=prices.index, y=prices['SPY'], name='SPY', line=dict(color=COLORS['SPY'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=ema50.index, y=ema50['SPY'], name='SPY EMA50', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ema200.index, y=ema200['SPY'], name='SPY EMA200', line=dict(color='blue', width=1)), row=1, col=1)
    
    fig.add_trace(go.Bar(x=macd.index, y=macd['SPY']-signal['SPY'], name='SPY Hist', marker_color='grey'), row=2, col=1)
    fig.add_trace(go.Scatter(x=macd.index, y=macd['SPY'], name='SPY MACD', line=dict(color='black', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=signal.index, y=signal['SPY'], name='SPY Signal', line=dict(color='red', width=1)), row=2, col=1)

    # TLT
    fig.add_trace(go.Scatter(x=prices.index, y=prices['TLT'], name='TLT', line=dict(color=COLORS['TLT'])), row=3, col=1)
    fig.add_trace(go.Scatter(x=ema50.index, y=ema50['TLT'], name='TLT EMA50', line=dict(color='orange', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=ema200.index, y=ema200['TLT'], name='TLT EMA200', line=dict(color='blue', width=1)), row=3, col=1)
    
    fig.add_trace(go.Bar(x=macd.index, y=macd['TLT']-signal['TLT'], name='TLT Hist', marker_color='grey'), row=4, col=1)
    fig.add_trace(go.Scatter(x=macd.index, y=macd['TLT'], name='TLT MACD', line=dict(color='black', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=signal.index, y=signal['TLT'], name='TLT Signal', line=dict(color='red', width=1)), row=4, col=1)

    fig.update_layout(height=1000, title="Trend Strength Dashboard (SPY & TLT)")
    return fig

# -----------------------------------------------------------------------------
# 5. HTML ASSEMBLY
# -----------------------------------------------------------------------------

def generate_summary_card(metrics):
    """Generate HTML text for the top summary card."""
    prices = metrics['prices']
    ret_1d = prices.pct_change().iloc[-1] * 100
    rv_21 = metrics['rv_21'].iloc[-1]
    
    best_asset = ret_1d.idxmax()
    worst_asset = ret_1d.idxmin()
    high_vol = rv_21.idxmax()
    
    # Interpretation logic
    uup_oil = metrics['uup_oil_corr'].iloc[-1]
    
    # Safe extraction of latest correlation matrix snapshot for SPY/TLT
    # The rolling_corr df has MultiIndex (Date, Ticker) and Columns (Ticker)
    rc = metrics['rolling_corr']
    last_date = rc.index.get_level_values(0)[-1]
    current_corr_matrix = rc.loc[last_date] # Index is now just Ticker
    spy_tlt = current_corr_matrix.loc['SPY', 'TLT']
    
    macro_text = "Markets are stable."
    if spy_tlt > 0.5:
        macro_text = "⚠️ <b>High Correlation Warning:</b> Stocks and Bonds moving together (Risk Parity pain)."
    elif spy_tlt < -0.6:
        macro_text = "ℹ️ <b>Flight to Quality:</b> Strong inverse correlation suggests effective hedging."
    
    if uup_oil > 0.5:
        macro_text += " <br>⚠️ <b>Tightening Signal:</b> USD and Oil rising together."

    html = f"""
    <div class="card">
        <h3>🚀 Trading Desk Summary ({datetime.datetime.now().strftime('%Y-%m-%d')})</h3>
        <div class="metrics-grid">
            <div class="metric-box">
                <span class="label">Best Momentum (1D)</span>
                <span class="value green">{best_asset} ({ret_1d[best_asset]:.2f}%)</span>
            </div>
            <div class="metric-box">
                <span class="label">Worst Momentum (1D)</span>
                <span class="value red">{worst_asset} ({ret_1d[worst_asset]:.2f}%)</span>
            </div>
            <div class="metric-box">
                <span class="label">Highest Vol (21d)</span>
                <span class="value">{high_vol} ({rv_21[high_vol]:.1f}%)</span>
            </div>
            <div class="metric-box">
                <span class="label">SPY-TLT Corr</span>
                <span class="value">{spy_tlt:.2f}</span>
            </div>
        </div>
        <p class="macro-note">{macro_text}</p>
    </div>
    """
    return html

def save_dashboard(figures, summary_html):
    print("💾 Assembling HTML Dashboard...")
    
    # Convert figures to HTML divs
    fig_divs = {k: pio.to_html(v, full_html=False, include_plotlyjs='cdn') for k, v in figures.items()}
    
    # CSS
    style = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h3 { margin-top: 0; color: #333; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px; }
        .metric-box { background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; border-radius: 4px; }
        .label { display: block; font-size: 0.85em; color: #666; margin-bottom: 5px; }
        .value { font-size: 1.2em; font-weight: bold; color: #333; }
        .value.green { color: #28a745; }
        .value.red { color: #dc3545; }
        .macro-note { font-style: italic; color: #555; border-top: 1px solid #eee; padding-top: 10px; }
        
        /* Tabs */
        .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 8px 8px 0 0; }
        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }
        .tab button:hover { background-color: #ddd; }
        .tab button.active { background-color: white; border-bottom: 3px solid #007bff; font-weight: bold; }
        .tabcontent { display: none; padding: 20px; border: 1px solid #ccc; border-top: none; background: white; border-radius: 0 0 8px 8px; animation: fadeEffect 1s; }
        @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
    </style>
    """
    
    # JS for Tabs
    script = """
    <script>
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
        
        // Trigger resize for Plotly
        window.dispatchEvent(new Event('resize'));
    }
    document.addEventListener("DOMContentLoaded", function() {
        document.getElementById("defaultOpen").click();
    });
    </script>
    """
    
    # HTML Structure
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pro Options Trader Dashboard</title>
        {style}
    </head>
    <body>
        <div class="container">
            {summary_html}
            
            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'MacroRegime')" id="defaultOpen">Macro Regime</button>
                <button class="tablinks" onclick="openTab(event, 'Correlations')">Correlations</button>
                <button class="tablinks" onclick="openTab(event, 'Volatility')">Volatility</button>
                <button class="tablinks" onclick="openTab(event, 'Trends')">Trends</button>
                <button class="tablinks" onclick="openTab(event, 'Dollar')">Dollar/Commodities</button>
            </div>

            <div id="MacroRegime" class="tabcontent">
                {fig_divs['regime']}
            </div>

            <div id="Correlations" class="tabcontent">
                {fig_divs['heatmap']}
            </div>

            <div id="Volatility" class="tabcontent">
                {fig_divs['radar']}
            </div>
            
            <div id="Trends" class="tabcontent">
                {fig_divs['trends']}
            </div>
            
            <div id="Dollar" class="tabcontent">
                {fig_divs['dollar']}
            </div>
        </div>
        {script}
    </body>
    </html>
    """
    
    with open('cross_asset_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✅ Dashboard saved to 'cross_asset_dashboard.html'")

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    # 1. Download
    df_prices = download_data()
    if df_prices.empty:
        print("❌ No data downloaded. Exiting.")
        return

    # 2. Compute
    metrics = compute_metrics(df_prices)
    
    # 3. Build Figures
    print("📊 Generating visualizations...")
    figs = {}
    figs['heatmap'] = create_heatmap(metrics)
    figs['radar'] = create_volatility_radar(metrics)
    figs['regime'] = create_macro_regime_map(metrics)
    figs['dollar'] = create_dollar_commodity_gauge(metrics)
    figs['trends'] = create_trend_dashboard(metrics)
    
    # 4. Generate Summary & Save
    summary_html = generate_summary_card(metrics)
    save_dashboard(figs, summary_html)

if __name__ == "__main__":
    main()
