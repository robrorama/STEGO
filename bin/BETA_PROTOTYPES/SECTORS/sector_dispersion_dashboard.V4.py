#!/usr/bin/env python3
# SCRIPTNAME: ok.sector_dispersion_dashboard.V4.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
SCRIPT NAME: sector_dispersion_dashboard.py

Description
-----------
Standalone Plotly HTML dashboard to visualize "stock pickers' market" conditions.
FIXES APPLIED:
1. Embeds Plotly.js directly (offline mode) to fix blank plots caused by CDN blocks.
2. Adds render-on-click logic for tabs to ensure hidden plots size correctly.
3. Adds robustness for recent yfinance column formatting changes.
"""

import argparse
import math
import webbrowser
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as py_offline  # Added for offline JS embedding
import yfinance as yf


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
def parse_ticker_list(ticker_str: str) -> List[str]:
    return [t.strip().upper() for t in ticker_str.split(",") if t.strip()]


def download_prices(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted close prices for tickers using yfinance.
    Returns a DataFrame indexed by date with columns=tickers.
    """
    print(f"[DEBUG] Downloading: {tickers}")
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by='column' # Forces standard (Attribute, Ticker) structure if possible
    )

    if data.empty:
        raise RuntimeError("No data returned from yfinance. Check tickers/period/interval.")

    # -------------------------------------------------------
    # ROBUST COLUMN HANDLING (Fixes yfinance version differences)
    # -------------------------------------------------------
    px = pd.DataFrame()

    # Case 1: MultiIndex columns (Typical for >1 ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Check if level 0 is attributes (Price, Open, Close) or Tickers
        level0 = data.columns.get_level_values(0)
        
        # Standard yfinance: ('Adj Close', 'AAPL')
        if "Adj Close" in level0:
            px = data["Adj Close"].copy()
        elif "Close" in level0:
            px = data["Close"].copy()
        # Inverted yfinance: ('AAPL', 'Adj Close') - happens in some versions
        else:
            # Try swapping levels and checking again
            swapped = data.swaplevel(0, 1, axis=1)
            if "Adj Close" in swapped.columns.get_level_values(0):
                px = swapped["Adj Close"].copy()
            elif "Close" in swapped.columns.get_level_values(0):
                px = swapped["Close"].copy()

    # Case 2: Single Index columns (Typical for 1 ticker)
    else:
        if "Adj Close" in data.columns:
            px = data[["Adj Close"]].copy()
        elif "Close" in data.columns:
            px = data[["Close"]].copy()
        
        # If we downloaded 1 ticker, name the column properly
        if px.shape[1] == 1 and len(tickers) == 1:
            px.columns = tickers

    if px.empty:
        raise RuntimeError(f"Could not locate 'Adj Close' or 'Close' in data columns: {data.columns}")

    # Ensure all requested tickers are present
    # Note: yfinance might drop delisted stocks silently
    found_tickers = [t for t in tickers if t in px.columns]
    if not found_tickers:
        raise RuntimeError("None of the requested tickers were found in the response.")
    
    px = px[found_tickers]

    # Clean NaNs: sort, forward-fill, back-fill
    px = px.sort_index()
    px = px.ffill().bfill()
    return px


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from adjusted prices.
    """
    rets = np.log(prices / prices.shift(1))
    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all")
    return rets


def rolling_zscores(
    rets: pd.DataFrame,
    window: int = 60,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Rolling z-score of returns per ticker.
    z_t = (r_t - mean_{t-window+1..t}) / std_{t-window+1..t}
    """
    rolling_mean = rets.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = rets.rolling(window=window, min_periods=min_periods).std()
    z = (rets - rolling_mean) / rolling_std
    return z


def compute_dispersion_multi(rets: pd.DataFrame) -> pd.Series:
    """
    Multi-ticker dispersion: cross-sectional std across tickers each day.
    """
    return rets.std(axis=1)


def rolling_avg_correlation_multi(
    rets: pd.DataFrame, window: int = 60, min_periods: int = 30
) -> pd.Series:
    """
    Rolling average correlation (multi-ticker):
    For each day t, compute correlation matrix for the last `window` days
    of returns and average the off-diagonal entries.
    """
    dates = rets.index
    avg_corr_vals = []

    for i in range(len(dates)):
        end_idx = i + 1
        start_idx = max(0, end_idx - window)
        window_rets = rets.iloc[start_idx:end_idx]

        if len(window_rets) < min_periods or window_rets.shape[1] < 2:
            avg_corr_vals.append(np.nan)
            continue

        # Correlation calculation
        corr = window_rets.corr()
        n = corr.shape[0]
        # Mask diagonal
        mask_offdiag = ~np.eye(n, dtype=bool)
        vals = corr.values[mask_offdiag]
        
        if vals.size == 0:
            avg_corr_vals.append(np.nan)
        else:
            avg_corr_vals.append(np.nanmean(vals))

    return pd.Series(avg_corr_vals, index=dates, name="avg_corr")


def latest_corr_matrix_multi(
    rets: pd.DataFrame, window: int = 60, min_periods: int = 30
) -> Tuple[pd.Index, pd.DataFrame]:
    """
    Latest ticker-by-ticker correlation matrix over recent window (multi-ticker).
    """
    if len(rets) < min_periods:
        corr = rets.corr()
        return corr.index, corr

    window_rets = rets.iloc[-window:]
    if len(window_rets) < min_periods:
        window_rets = rets

    corr = window_rets.corr()
    return corr.index, corr


def compute_dispersion_single(
    series: pd.Series, window: int = 60, min_periods: int = 30
) -> pd.Series:
    """
    Single-ticker "dispersion": rolling realized volatility (std of returns).
    """
    disp = series.rolling(window=window, min_periods=min_periods).std()
    disp.name = "realized_vol"
    return disp


def rolling_auto_corr_single(
    series: pd.Series, window: int = 60, min_periods: int = 30
) -> pd.Series:
    """
    Single-ticker "average correlation": rolling lag-1 auto-correlation of returns.
    """
    idx = series.index
    vals = series.values
    ac_vals = []

    for i in range(len(idx)):
        end_idx = i + 1
        start_idx = max(0, end_idx - window)
        w = vals[start_idx:end_idx]
        w = w[~np.isnan(w)]
        if len(w) < max(min_periods, 2):
            ac_vals.append(np.nan)
            continue

        x = w[:-1]
        y = w[1:]
        if np.std(x) == 0 or np.std(y) == 0:
            ac_vals.append(np.nan)
        else:
            # simple lag-1 correlation
            corr = np.corrcoef(x, y)[0, 1]
            ac_vals.append(corr)

    return pd.Series(ac_vals, index=idx, name="avg_corr_auto")


def latest_corr_matrix_single(
    series: pd.Series, window: int = 60, min_periods: int = 30, max_lag: int = 10
) -> Tuple[pd.Index, pd.DataFrame]:
    """
    Single-ticker "correlation matrix": lag-lag auto-correlation matrix.
    """
    s = series.dropna()
    if s.empty:
        corr = pd.DataFrame([[1.0]], index=["lag_0"], columns=["lag_0"])
        return corr.index, corr

    w = s.iloc[-window:]
    if len(w) < min_periods:
        w = s

    if len(w) < 2:
        corr = pd.DataFrame([[1.0]], index=["lag_0"], columns=["lag_0"])
        return corr.index, corr

    L = min(max_lag, len(w) - 1)
    df_lags = {}
    for lag in range(L + 1):
        df_lags[f"lag_{lag}"] = w.shift(lag)

    df_lags = pd.DataFrame(df_lags).dropna()
    if df_lags.shape[0] < 2:
        corr = pd.DataFrame([[1.0]], index=["lag_0"], columns=["lag_0"])
        return corr.index, corr

    corr = df_lags.corr()
    return corr.index, corr


def standardize_series(s: pd.Series) -> pd.Series:
    """
    Simple z-score for a 1D series.
    """
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or math.isnan(sigma):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sigma


# ---------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------
def fig_zscore_heatmap(zscores: pd.DataFrame) -> go.Figure:
    z_clipped = zscores.clip(-3, 3)
    fig = go.Figure(
        data=go.Heatmap(
            z=z_clipped.values,
            x=zscores.columns,
            y=zscores.index,
            colorbar=dict(title="Return Z-Score"),
            hovertemplate="<b>%{y|%Y-%m-%d}</b><br>Ticker: %{x}<br>Z: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Return Z-Score Heatmap (Rolling, per Ticker)",
        xaxis_title="Ticker",
        yaxis_title="Date",
    )
    return fig


def fig_dispersion_line(dispersion: pd.Series, title_suffix: str = "") -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(
            x=dispersion.index,
            y=dispersion.values,
            mode="lines",
            name="Dispersion",
        )
    )
    fig.update_layout(
        title=f"Dispersion Over Time{title_suffix}",
        xaxis_title="Date",
        yaxis_title="Dispersion",
    )
    return fig


def fig_avgcorr_line(avg_corr: pd.Series, title_suffix: str = "") -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(
            x=avg_corr.index,
            y=avg_corr.values,
            mode="lines",
            name="Avg Correlation",
        )
    )
    fig.update_layout(
        title=f"Average Correlation Over Time{title_suffix}",
        xaxis_title="Date",
        yaxis_title="Average Correlation",
        yaxis=dict(range=[-1, 1]),
    )
    return fig


def fig_dual_axis(dispersion: pd.Series, avg_corr: pd.Series) -> go.Figure:
    common_idx = dispersion.index.intersection(avg_corr.index)
    d = dispersion.loc[common_idx]
    c = avg_corr.loc[common_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d.values,
            name="Dispersion",
            mode="lines",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=c.index,
            y=c.values,
            name="Avg Correlation",
            mode="lines",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Dispersion vs Average Correlation",
        xaxis=dict(domain=[0.05, 0.95], title="Date"),
        yaxis=dict(
            title="Dispersion",
            anchor="x",
        ),
        yaxis2=dict(
            title="Average Correlation",
            overlaying="y",
            side="right",
            range=[-1, 1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fig_dispersion_vs_corr_scatter(
    dispersion: pd.Series, avg_corr: pd.Series
) -> go.Figure:
    common_idx = dispersion.index.intersection(avg_corr.index)
    d = dispersion.loc[common_idx]
    c = avg_corr.loc[common_idx]

    t_numeric = np.linspace(0, 1, len(common_idx)) if len(common_idx) > 0 else []

    fig = go.Figure(
        data=go.Scatter(
            x=d.values,
            y=c.values,
            mode="markers",
            marker=dict(
                size=8,
                color=t_numeric,
                showscale=True,
                colorbar=dict(title="Time Progress"),
            ),
            text=[dt.strftime("%Y-%m-%d") for dt in common_idx],
            hovertemplate="Date: %{text}<br>Dispersion: %{x:.4f}<br>Avg Corr: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Dispersion vs Average Correlation (Color = Progress Through Time)",
        xaxis_title="Dispersion",
        yaxis_title="Average Correlation",
        yaxis=dict(range=[-1, 1]),
    )
    return fig


def fig_corr_matrix(corr_idx: pd.Index, corr: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr_idx,
            y=corr_idx,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Corr: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Axis",
        yaxis_title="Axis",
    )
    return fig


def fig_regime_map(
    dispersion: pd.Series,
    avg_corr: pd.Series,
) -> go.Figure:
    """
    Regime scatter:
    - X-axis: Dispersion Z-score
    - Y-axis: Avg Corr Z-score
    """
    common_idx = dispersion.index.intersection(avg_corr.index)
    d = dispersion.loc[common_idx]
    c = avg_corr.loc[common_idx]

    d_z = standardize_series(d)
    c_z = standardize_series(c)

    t_numeric = np.linspace(0, 1, len(common_idx)) if len(common_idx) > 0 else []

    fig = go.Figure(
        data=go.Scatter(
            x=d_z.values,
            y=c_z.values,
            mode="markers",
            marker=dict(
                size=8,
                color=t_numeric,
                showscale=True,
                colorbar=dict(title="Time Progress"),
            ),
            text=[dt.strftime("%Y-%m-%d") for dt in common_idx],
            hovertemplate="Date: %{text}<br>Disp Z: %{x:.2f}<br>AvgCorr Z: %{y:.2f}<extra></extra>",
        )
    )

    if len(common_idx) > 0:
        fig.add_shape(
            type="line",
            x0=0,
            y0=c_z.min() - 0.5,
            x1=0,
            y1=c_z.max() + 0.5,
            line=dict(dash="dash", width=1),
        )
        fig.add_shape(
            type="line",
            x0=d_z.min() - 0.5,
            y0=0,
            x1=d_z.max() + 0.5,
            y1=0,
            line=dict(dash="dash", width=1),
        )

        x_max = max(abs(d_z.min()), abs(d_z.max())) + 0.2
        y_max = max(abs(c_z.min()), abs(c_z.max())) + 0.2

        fig.add_annotation(
            x=x_max * 0.7,
            y=-y_max * 0.7,
            text="High Dispersion<br>Low Correlation<br><b>Stock Pickers' Market</b>",
            showarrow=False,
            align="center",
        )
        fig.add_annotation(
            x=-x_max * 0.7,
            y=y_max * 0.7,
            text="Low Dispersion<br>High Correlation<br><b>Indexer Market</b>",
            showarrow=False,
            align="center",
        )

    fig.update_layout(
        title="Regime Map: Dispersion Z vs Avg Correlation Z",
        xaxis_title="Dispersion Z-Score",
        yaxis_title="Average Correlation Z-Score",
    )
    return fig


# ---------------------------------------------------------------------
# HTML dashboard builder (Embeds JS for offline support)
# ---------------------------------------------------------------------
def build_dashboard_html(figs: dict, title: str) -> str:
    """
    Build a simple multi-tab HTML page containing all figures.
    Each figure gets its own tab at the top.
    """
    
    # 1. Get Plotly JS source code (so it works offline/without CDN)
    # This increases file size to ~3MB but guarantees no blank plots due to network blocks.
    print("[INFO] Embedding Plotly JS library (this may take a moment)...")
    plotly_js = py_offline.get_plotlyjs()
    
    tabs_html = []
    content_html = []

    for i, (name, fig) in enumerate(figs.items()):
        div_id = f"fig_{i}"
        
        # We generate the DIV but exclude the JS lib (we add it globally once in <head>)
        # include_plotlyjs=False prevents injecting the script tag here
        # full_html=False creates just the <div> and the script to call Plotly.newPlot
        fig_html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        
        active_class = "active" if i == 0 else ""
        display_style = "block" if i == 0 else "none"

        tabs_html.append(
            '<button class="tablink {active}" onclick="openTab(\'{div_id}\', this)">{name}</button>'.format(
                active=active_class, div_id=div_id, name=name
            )
        )
        content_html.append(
            '<div id="{div_id}_wrapper" class="tabcontent" style="display:{display_style}">'.format(
                div_id=div_id, display_style=display_style
            )
            + fig_html
            + "</div>"
        )

    html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '<meta charset="utf-8" />\n'
        "<title>" + title + "</title>\n"
        # EMBEDDED JS LIBRARY
        f'<script type="text/javascript">{plotly_js}</script>\n'
        "<style>\n"
        "body {\n"
        "    font-family: Arial, sans-serif;\n"
        "    margin: 0;\n"
        "    padding: 0;\n"
        "    background-color: #111;\n"
        "    color: #eee;\n"
        "}\n"
        "h1 {\n"
        "    margin: 0;\n"
        "    padding: 16px 24px;\n"
        "    background-color: #222;\n"
        "    border-bottom: 1px solid #333;\n"
        "}\n"
        ".tabbar {\n"
        "    display: flex;\n"
        "    flex-wrap: wrap;\n"
        "    background-color: #181818;\n"
        "    border-bottom: 1px solid #333;\n"
        "}\n"
        ".tablink {\n"
        "    background-color: #181818;\n"
        "    border: none;\n"
        "    color: #bbb;\n"
        "    padding: 10px 16px;\n"
        "    cursor: pointer;\n"
        "    font-size: 14px;\n"
        "    transition: background-color 0.2s, color 0.2s;\n"
        "}\n"
        ".tablink:hover {\n"
        "    background-color: #333;\n"
        "    color: #fff;\n"
        "}\n"
        ".tablink.active {\n"
        "    background-color: #444;\n"
        "    color: #fff;\n"
        "}\n"
        ".tabcontent {\n"
        "    padding: 8px 16px 24px 16px;\n"
        "}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>" + title + "</h1>\n"
        '<div class="tabbar">\n'
        + "".join(tabs_html)
        + "\n</div>\n"
        + "".join(content_html)
        +
        """
<script>
function openTab(figId, btn) {
    var i, contents, tabs;
    contents = document.getElementsByClassName("tabcontent");
    for (i = 0; i < contents.length; i++) {
        contents[i].style.display = "none";
    }
    tabs = document.getElementsByClassName("tablink");
    for (i = 0; i < tabs.length; i++) {
        tabs[i].classList.remove("active");
    }
    
    // Show the target tab
    var target = document.getElementById(figId + "_wrapper");
    target.style.display = "block";
    btn.classList.add("active");
    
    // CRITICAL FIX: Trigger a window resize event so Plotly redraws correct width
    // (Plotly often renders at 0 width inside a hidden div)
    window.dispatchEvent(new Event('resize'));
}
</script>
</body>
</html>
"""
    )
    return html


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Dispersion & Correlation Dashboard (Standalone Plotly HTML)"
    )
    # Support both --tickers and --ticker, mapping to same dest
    parser.add_argument(
        "--tickers",
        "--ticker",
        dest="tickers",
        type=str,
        default="XLE,XLF,XLK,XLY,XLP,XLI,XLV,XLU,XLB,XLRE,XLC",
        help="Comma-separated list of tickers (default: SPDR sectors).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="History period for yfinance (e.g. 1y, 2y, 5y, max). Default: 2y.",
    )
    parser.add_argument(
        "--roll-window",
        type=int,
        default=60,
        help="Rolling window length for z-scores and correlations (default: 60).",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=30,
        help="Minimum periods for rolling calculations (default: 30).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="sector_dispersion_dashboard.html",
        help="Output HTML file name.",
    )

    args = parser.parse_args()

    tickers = parse_ticker_list(args.tickers)
    print(f"[INFO] Using tickers: {tickers}")
    print(f"[INFO] Downloading {args.period} of daily data via yfinance...")

    try:
        prices = download_prices(tickers, period=args.period, interval="1d")
    except Exception as e:
        print(f"[ERROR] Data download failed: {e}")
        return

    print(f"[INFO] Downloaded prices with shape: {prices.shape}")

    rets = compute_returns(prices)
    print(f"[INFO] Computed returns with shape: {rets.shape}")

    if rets.empty:
        raise RuntimeError("Return series is empty after processing. No plots to show.")

    zscores = rolling_zscores(
        rets, window=args.roll_window, min_periods=args.min_periods
    )

    n_assets = rets.shape[1]
    single_mode = n_assets == 1

    if single_mode:
        print("[INFO] Single-ticker mode: using realized vol and auto-correlation.")
        series = rets.iloc[:, 0]
        dispersion = compute_dispersion_single(
            series, window=args.roll_window, min_periods=args.min_periods
        )
        avg_corr = rolling_auto_corr_single(
            series, window=args.roll_window, min_periods=args.min_periods
        )
        corr_idx, corr_matrix = latest_corr_matrix_single(
            series,
            window=args.roll_window,
            min_periods=args.min_periods,
            max_lag=10,
        )
        corr_title = "Lag-Lag Auto-Correlation Matrix (Single Ticker)"
        disp_title_suffix = " (Realized Volatility)"
        avgc_title_suffix = " (Lag-1 Auto-Correlation)"
    else:
        print("[INFO] Multi-ticker mode: cross-sectional dispersion and correlation.")
        dispersion = compute_dispersion_multi(rets)
        avg_corr = rolling_avg_correlation_multi(
            rets, window=args.roll_window, min_periods=args.min_periods
        )
        corr_idx, corr_matrix = latest_corr_matrix_multi(
            rets, window=args.roll_window, min_periods=args.min_periods
        )
        corr_title = "Latest Cross-Sectional Correlation Matrix (Tickers)"
        disp_title_suffix = ""
        avgc_title_suffix = ""

    # Build figures
    print("[INFO] Building Plotly figures...")
    figs = {
        "Z-Score Heatmap": fig_zscore_heatmap(zscores),
        "Dispersion Line": fig_dispersion_line(dispersion, disp_title_suffix),
        "Avg Corr Line": fig_avgcorr_line(avg_corr, avgc_title_suffix),
        "Dispersion vs Avg Corr (Dual Axis)": fig_dual_axis(dispersion, avg_corr),
        "Dispersion vs Avg Corr (Scatter)": fig_dispersion_vs_corr_scatter(
            dispersion, avg_corr
        ),
        "Correlation Matrix": fig_corr_matrix(corr_idx, corr_matrix, corr_title),
        "Regime Map (Z vs Z)": fig_regime_map(dispersion, avg_corr),
    }

    title = f"Dispersion & Correlation Dashboard ({', '.join(tickers)})"

    print(f"[INFO] Assembling HTML dashboard -> {args.out}")
    html = build_dashboard_html(figs, title=title)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    try:
        webbrowser.open_new_tab(args.out)
        print(f"[INFO] Opened in browser: {args.out}")
    except Exception as e:
        print(f"[WARN] Could not auto-open browser: {e}")
        print(f"[INFO] You can manually open: {args.out}")


if __name__ == "__main__":
    main()
