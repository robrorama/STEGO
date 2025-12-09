#!/usr/bin/env python3
# SCRIPTNAME: ok.macros.proxy.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
Fed / Macro Proxies & Signal Lines Dashboard (Plotly HTML, no Dash, no Jupyter)

- Uses ONLY data_retrieval.py to download data (no interval argument).
- Uses equities/ETFs as proxies for Fed expectations and macro risk.
- Produces a single multi-tab HTML file that opens in your browser (e.g., Firefox).
- Tabs:
    1) Proxies Overview (indexed performance of key ETFs)
    2) Signal Lines (ratios & composite indicators derived from those proxies)

Run example:
    python3 macros.proxy.v1.py --start 2020-01-01 --end 2025-11-23 --output macros.proxy.v1.html
"""

import argparse
import logging
import os
import sys
import webbrowser
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    print("ERROR: plotly is required. Install with: pip install plotly", file=sys.stderr)
    sys.exit(1)

# --- IMPORT YOUR CANONICAL DATA LAYER (UNMODIFIED) ---
try:
    import data_retrieval as dr
except ImportError:
    print("ERROR: Could not import data_retrieval.py. Make sure it is on PYTHONPATH.", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

PROXY_TICKERS = [
    # Core equity / index proxies
    "SPY",   # US broad market
    "QQQ",   # growth / duration
    "IWM",   # small caps / liquidity

    # Rate / duration proxies
    "TLT",   # long duration Treasuries
    "IEF",   # 7-10y Treasuries
    "SHY",   # 1-3y Treasuries
    "IEI",   # 3-7y Treasuries

    # Sector / rate-sensitive
    "XLF",   # financials
    "XLRE",  # real estate

    # Credit spread proxies
    "HYG",   # high yield
    "JNK",   # high yield

    # USD / FX proxy
    "UUP",   # US Dollar Bullish ETF

    # High-duration growth
    "ARKK",  # innovation ETF (optional; may fail if not available)
]


# ----------------------------------------------------------------------
# DATA LOADING WRAPPERS
# ----------------------------------------------------------------------

def load_prices_for_tickers(
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Use your canonical data_retrieval.py module to fetch OHLCV data for each ticker.
    Returns a dict: {ticker: DataFrame}
    """
    data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            logging.info(f"Loading data for {t}")
            # IMPORTANT: no interval argument – matches your canonical signature
            df = dr.load_or_download_ticker(t, start=start, end=end)
            if df is None or df.empty:
                logging.warning(f"No data for {t}, skipping.")
                continue

            # Ensure Date is index (if your module already does this, this is harmless)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df = df.set_index("Date")
                else:
                    # Try to coerce the index to datetime
                    df.index = pd.to_datetime(df.index)

            # We expect standard yfinance columns, including "Close"
            if "Close" not in df.columns:
                logging.warning(f"{t}: DataFrame has no 'Close' column, skipping.")
                continue

            data[t] = df
        except TypeError as e:
            # This will catch unexpected keyword args etc.
            logging.error(f"Error loading {t}: {e}", exc_info=False)
        except Exception as e:
            logging.error(f"Unexpected error loading {t}: {e}", exc_info=False)
    return data


# ----------------------------------------------------------------------
# SIGNAL CONSTRUCTION
# ----------------------------------------------------------------------

def compute_indexed_close(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """
    Return price series indexed to 1.0 at the first valid value.
    """
    if col not in df.columns:
        return pd.Series(dtype=float)

    s = df[col].astype(float).copy()
    s = s[s.notna()]
    if s.empty:
        return s
    base = s.iloc[0]
    if base == 0:
        return s * np.nan
    return s / base


def compute_ratio_series(
    num_df: pd.DataFrame,
    den_df: pd.DataFrame,
    col: str = "Close"
) -> pd.Series:
    """
    Compute ratio of num_df[col] / den_df[col] on common dates.
    """
    if col not in num_df.columns or col not in den_df.columns:
        return pd.Series(dtype=float)

    joined = pd.concat(
        [num_df[col].rename("num"), den_df[col].rename("den")],
        axis=1,
        join="inner"
    ).dropna()
    if joined.empty:
        return joined["num"] * np.nan
    return (joined["num"] / joined["den"]).rename(f"{col}_ratio")


def zscore(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series * np.nan
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or np.isnan(sigma):
        return series * np.nan
    out = (series - mu) / sigma
    # reindex to original index to preserve alignment
    return out.reindex(series.index)


def build_signal_lines(
    data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.Series]:
    """
    Build a set of derived signal lines using ONLY equity/ETF proxies.
    All outputs are aligned on their own intersection dates.
    """
    signals: Dict[str, pd.Series] = {}

    # Duration vs financials: QQQ / XLF
    if "QQQ" in data and "XLF" in data:
        signals["Duration_QQQ_over_XLF"] = compute_ratio_series(data["QQQ"], data["XLF"])

    # High-duration vs broad market: ARKK / SPY (if ARKK available)
    if "ARKK" in data and "SPY" in data:
        signals["Duration_ARKK_over_SPY"] = compute_ratio_series(data["ARKK"], data["SPY"])

    # Credit risk proxy: HYG / IEF
    if "HYG" in data and "IEF" in data:
        signals["Credit_HYG_over_IEF"] = compute_ratio_series(data["HYG"], data["IEF"])

    # Alternate credit proxy: JNK / IEF
    if "JNK" in data and "IEF" in data:
        signals["Credit_JNK_over_IEF"] = compute_ratio_series(data["JNK"], data["IEF"])

    # Small vs large: IWM / SPY
    if "IWM" in data and "SPY" in data:
        signals["Small_vs_Large_IWM_over_SPY"] = compute_ratio_series(data["IWM"], data["SPY"])

    # USD vs equities: UUP / SPY
    if "UUP" in data and "SPY" in data:
        signals["USD_vs_Equity_UUP_over_SPY"] = compute_ratio_series(data["UUP"], data["SPY"])

    # Simple "Dovishness Index":
    # High when duration outperforms + credit tightens + USD is weaker vs equities.
    components: List[pd.Series] = []
    if "Duration_QQQ_over_XLF" in signals:
        components.append(zscore(signals["Duration_QQQ_over_XLF"]).rename("Z_Duration_QQQ_over_XLF"))
    if "Duration_ARKK_over_SPY" in signals:
        components.append(zscore(signals["Duration_ARKK_over_SPY"]).rename("Z_Duration_ARKK_over_SPY"))
    if "Credit_HYG_over_IEF" in signals:
        components.append(zscore(signals["Credit_HYG_over_IEF"]).rename("Z_Credit_HYG_over_IEF"))
    if "USD_vs_Equity_UUP_over_SPY" in signals:
        # For dovishness, we want WEAKER USD to be positive.
        # So we flip sign of the z-score of UUP/SPY.
        components.append((-1.0 * zscore(signals["USD_vs_Equity_UUP_over_SPY"])).rename("Z_Neg_USD_vs_Equity"))

    if components:
        # Align all components on intersection of dates
        comp_aligned = pd.concat(components, axis=1, join="inner").dropna()
        if not comp_aligned.empty:
            signals["Dovishness_Index"] = comp_aligned.mean(axis=1)

    return signals


# ----------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------------------------

def make_proxies_indexed_figure(
    data: Dict[str, pd.DataFrame],
    title: str = "Fed-Sensitive Proxies – Indexed Performance"
) -> go.Figure:
    """
    Single figure with all proxies indexed to 1.0 at start.
    """
    fig = go.Figure()
    for t, df in data.items():
        idx_series = compute_indexed_close(df, "Close")
        if idx_series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=idx_series.index,
                y=idx_series.values,
                mode="lines",
                name=t
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Indexed Price (Start = 1)",
        hovermode="x unified"
    )
    return fig


def make_sector_duration_figure(
    data: Dict[str, pd.DataFrame]
) -> go.Figure:
    """
    Compare duration (TLT, IEF, QQQ) vs rate-sensitive sectors (XLF, XLRE).
    """
    fig = go.Figure()

    for t in ["TLT", "IEF", "QQQ", "XLF", "XLRE"]:
        if t not in data:
            continue
        idx_series = compute_indexed_close(data[t])
        if idx_series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=idx_series.index,
                y=idx_series.values,
                mode="lines",
                name=t
            )
        )

    fig.update_layout(
        title="Duration vs Rate-Sensitive Sectors – Indexed",
        xaxis_title="Date",
        yaxis_title="Indexed Price (Start = 1)",
        hovermode="x unified"
    )
    return fig


def make_credit_vs_treasury_figure(
    data: Dict[str, pd.DataFrame]
) -> go.Figure:
    """
    Credit vs Treasuries: HYG, JNK vs IEF/TLT.
    """
    fig = go.Figure()

    for t in ["HYG", "JNK", "IEF", "TLT"]:
        if t not in data:
            continue
        idx_series = compute_indexed_close(data[t])
        if idx_series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=idx_series.index,
                y=idx_series.values,
                mode="lines",
                name=t
            )
        )

    fig.update_layout(
        title="Credit vs Treasuries – Indexed",
        xaxis_title="Date",
        yaxis_title="Indexed Price (Start = 1)",
        hovermode="x unified"
    )
    return fig


def make_signal_figure(
    signals: Dict[str, pd.Series],
    keys: List[str],
    title: str
) -> go.Figure:
    fig = go.Figure()
    for k in keys:
        if k not in signals:
            continue
        s = signals[k].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=k
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified"
    )
    return fig


def make_dovishness_figure(
    signals: Dict[str, pd.Series]
) -> go.Figure:
    fig = go.Figure()
    if "Dovishness_Index" in signals:
        s = signals["Dovishness_Index"].dropna()
        if not s.empty:
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    name="Dovishness_Index"
                )
            )
    fig.update_layout(
        title="Composite Dovishness Index (Z-Score Blend of Proxies)",
        xaxis_title="Date",
        yaxis_title="Z-Score (Higher = More Dovish Pricing)",
        hovermode="x unified"
    )
    return fig


# ----------------------------------------------------------------------
# HTML DASHBOARD (NO DASH SERVER)
# ----------------------------------------------------------------------

def figures_to_tabbed_html(
    tab_specs: List[Tuple[str, List[go.Figure]]],
    output_path: str,
    page_title: str = "Fed / Macro Proxies & Signal Lines Dashboard"
) -> None:
    """
    Build a single HTML file with simple CSS/JS tabs.
    tab_specs: list of (tab_name, [fig1, fig2, ...])
    """
    logging.info(f"Building HTML dashboard: {output_path}")

    # Generate HTML snippets for each figure
    tab_divs: List[str] = []
    tab_buttons: List[str] = []

    # We'll only include plotly.js once (in the first figure)
    first_fig_done = False
    tab_index = 0

    for tab_name, figs in tab_specs:
        tab_id = f"tab_{tab_index}"
        tab_buttons.append(
            f'<button class="tablinks" onclick="openTab(event, \'{tab_id}\')" id="btn_{tab_id}">{tab_name}</button>'
        )

        fig_html_parts: List[str] = []
        for fi, fig in enumerate(figs):
            include_js = False
            if not first_fig_done:
                include_js = True
                first_fig_done = True

            div_id = f"{tab_id}_fig_{fi}"
            html_snip = pio.to_html(
                fig,
                include_plotlyjs="cdn" if include_js else False,
                full_html=False,
                div_id=div_id
            )
            fig_html_parts.append(html_snip)

        # Wrap figs for this tab
        tab_divs.append(
            f'<div id="{tab_id}" class="tabcontent">\n' +
            "\n".join(fig_html_parts) +
            "\n</div>"
        )

        tab_index += 1

    tabs_html = "\n".join(tab_divs)
    buttons_html = "\n".join(tab_buttons)

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{page_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        .tab {{
            overflow: hidden;
            border-bottom: 1px solid #ccc;
            background-color: #f1f1f1;
        }}
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 10px 16px;
            transition: 0.3s;
            font-size: 14px;
        }}
        .tab button:hover {{
            background-color: #ddd;
        }}
        .tab button.active {{
            background-color: #ccc;
        }}
        .tabcontent {{
            display: none;
            padding: 10px 10px;
        }}
    </style>
</head>
<body>
    <h2 style="margin-left:10px;">{page_title}</h2>
    <div class="tab">
        {buttons_html}
    </div>
    {tabs_html}
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

        // Open first tab by default
        document.addEventListener("DOMContentLoaded", function() {{
            var firstBtn = document.getElementsByClassName("tablinks")[0];
            if (firstBtn) {{
                firstBtn.click();
            }}
        }});
    </script>
</body>
</html>
"""

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    logging.info(f"Dashboard written to: {output_path}")


# ----------------------------------------------------------------------
# CLI / MAIN
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fed / Macro Proxies & Signal Lines Dashboard (Plotly HTML, no Dash, no Jupyter)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="macros.proxy.v1.html",
        help="Output HTML file path"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("=== Fed / Macro Proxies & Signal Lines Dashboard ===")
    logging.info(f"Date range: {args.start} -> {args.end}")

    # Load data
    data = load_prices_for_tickers(PROXY_TICKERS, start=args.start, end=args.end)
    if not data:
        logging.error("No data loaded for any ticker. Exiting.")
        sys.exit(1)

    # Build signals
    signals = build_signal_lines(data)

    # Build proxy figures (Tab 1)
    fig_proxies_all = make_proxies_indexed_figure(data)
    fig_sector_dur = make_sector_duration_figure(data)
    fig_credit_treas = make_credit_vs_treasury_figure(data)

    # Build signal figures (Tab 2)
    duration_signal_keys = [
        "Duration_QQQ_over_XLF",
        "Duration_ARKK_over_SPY"
    ]
    credit_signal_keys = [
        "Credit_HYG_over_IEF",
        "Credit_JNK_over_IEF"
    ]
    structural_signal_keys = [
        "Small_vs_Large_IWM_over_SPY",
        "USD_vs_Equity_UUP_over_SPY"
    ]

    fig_duration_signals = make_signal_figure(
        signals,
        duration_signal_keys,
        "Duration Signals (Ratios)"
    )
    fig_credit_signals = make_signal_figure(
        signals,
        credit_signal_keys,
        "Credit Signals (Ratios)"
    )
    fig_structural_signals = make_signal_figure(
        signals,
        structural_signal_keys,
        "Structural Signals (Small vs Large, USD vs Equities)"
    )
    fig_dovishness = make_dovishness_figure(signals)

    # Assemble tabs
    tab_specs = [
        ("Proxies Overview", [fig_proxies_all, fig_sector_dur, fig_credit_treas]),
        ("Signal Lines", [fig_duration_signals, fig_credit_signals, fig_structural_signals, fig_dovishness]),
    ]

    output_path = os.path.abspath(args.output)
    figures_to_tabbed_html(tab_specs, output_path=output_path)

    # Open in browser (Firefox / default)
    logging.info("Opening dashboard in default browser...")
    webbrowser.open("file://" + output_path)


if __name__ == "__main__":
    main()

