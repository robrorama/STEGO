#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dealer_gex_engine.py

Standalone "Dealer Gamma Exposure" (GEX) engine with a strict disk-first data
pipeline and offline Plotly dashboard generation.

Architecture:
    - DataIngestion: yfinance I/O + CSV caching ("disk-first" for prices/options/GEX).
    - FinancialAnalysis: Black-Scholes greeks, dealer GEX logic, shadow backfill model.
    - DashboardRenderer: Offline HTML dashboard with History / Strike Profile tabs.

Dependencies:
    - numpy
    - pandas
    - yfinance
    - plotly
    - scipy
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

from scipy.stats import norm

import json


class DataIngestion:
    """
    Handles all external I/O and enforces disk-first behavior.

    Responsibilities:
        - Download and cache underlying price history as CSV.
        - Download and cache options chains for the next N expiries as CSV.
        - Load / save GEX history as CSV.
    """

    def __init__(self, output_dir: str, lookback_years: float = 1.0) -> None:
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.lookback_years = lookback_years

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        """
        Disk-first loader for underlying price history.

        Workflow:
            1. If [ticker]_history.csv exists, load and sanitize.
            2. Else, download ~1.5 years of history via yfinance, sanitize,
               save to CSV, then reload from disk.
        """
        path = os.path.join(self.output_dir, f"{ticker}_history.csv")

        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = self._sanitize_df(df)
            return df

        # Download ~1.5 years of history
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(1.5 * 365))

        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            group_by="column",
            auto_adjust=False,
            progress=False,
        )
        time.sleep(1.0)  # rate limiting safeguard

        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no price data for {ticker}")

        df = self._sanitize_df(df)
        df.to_csv(path)
        # Disk-first: re-read from CSV
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = self._sanitize_df(df)
        return df

    def get_options_data(self, ticker: str, num_expiries: int = 3) -> pd.DataFrame:
        """
        Disk-first loader for options chain snapshot for the next num_expiries expirations.

        Data is cached per (ticker, run-date) as:
            [ticker]_options_YYYYMMDD.csv
        """
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = os.path.join(self.output_dir, f"{ticker}_options_{today_str}.csv")

        if os.path.exists(path):
            df = pd.read_csv(path)
            if "expiration" in df.columns:
                df["expiration"] = pd.to_datetime(df["expiration"])
            return df

        ticker_obj = yf.Ticker(ticker)
        # accessing .options triggers a request
        expirations = list(ticker_obj.options or [])
        time.sleep(1.0)

        if not expirations:
            # No listed options; return empty frame
            return pd.DataFrame()

        expirations = expirations[: max(1, num_expiries)]
        option_frames = []

        for exp in expirations:
            try:
                chain = ticker_obj.option_chain(exp)
                time.sleep(1.0)
            except Exception as exc:
                print(f"Warning: failed to fetch option chain for {ticker} @ {exp}: {exc}", file=sys.stderr)
                continue

            calls = getattr(chain, "calls", None)
            puts = getattr(chain, "puts", None)

            if calls is not None and not calls.empty:
                calls = calls.copy()
                calls["option_type"] = "call"
                calls["expiration"] = pd.to_datetime(exp)
                option_frames.append(calls)

            if puts is not None and not puts.empty:
                puts = puts.copy()
                puts["option_type"] = "put"
                puts["expiration"] = pd.to_datetime(exp)
                option_frames.append(puts)

        if not option_frames:
            return pd.DataFrame()

        df = pd.concat(option_frames, ignore_index=True)

        # Retain only the core columns we care about
        keep_cols = [
            "expiration",
            "option_type",
            "strike",
            "lastPrice",
            "openInterest",
            "impliedVolatility",
            "volume",
            "inTheMoney",
        ]
        df = df[[c for c in keep_cols if c in df.columns]].copy()
        df.to_csv(path, index=False)

        df = pd.read_csv(path)
        if "expiration" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"])
        return df

    def load_gex_history(self, ticker: str) -> pd.DataFrame:
        """
        Load pre-computed GEX history from disk if available.
        """
        path = os.path.join(self.output_dir, f"{ticker}_gex_history.csv")
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    def save_gex_history(self, ticker: str, gex_df: pd.DataFrame) -> None:
        """
        Persist GEX history to disk.
        """
        path = os.path.join(self.output_dir, f"{ticker}_gex_history.csv")
        gex_df.to_csv(path)

    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal fixer for yfinance price history frames.

        - Handles MultiIndex column swaps (Close/Adj Close in level 1).
        - Flattens MultiIndex columns to simple strings (keeps attribute level).
        - Forces DatetimeIndex, strips timezone.
        - Coerces numeric columns to float.
        """
        if isinstance(df.columns, pd.MultiIndex):
            level0 = df.columns.get_level_values(0)
            level1 = df.columns.get_level_values(1)
            target_attrs = {"Adj Close", "Close"}

            has_target_lvl0 = any(attr in set(level0) for attr in target_attrs)
            has_target_lvl1 = any(attr in set(level1) for attr in target_attrs)

            # If the interesting attributes are only in level 1, swap levels
            if has_target_lvl1 and not has_target_lvl0:
                df = df.swaplevel(0, 1, axis=1)

            # Flatten columns: keep the attribute level (index 0)
            flat_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    flat_cols.append(str(col[0]))
                else:
                    flat_cols.append(str(col))
            df.columns = flat_cols
        else:
            df.columns = [str(c) for c in df.columns]

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()

        # Coerce numeric columns to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


class FinancialAnalysis:
    """
    Implements Black-Scholes greeks, dealer GEX computations, and
    the shadow backfill model for GEX history.
    """

    def __init__(self, risk_free_rate: float = 0.04, lookback_years: float = 1.0) -> None:
        self.risk_free_rate = float(risk_free_rate)
        self.lookback_years = float(lookback_years)

    def _black_scholes_greeks(
        self,
        spot: float,
        strike: float,
        ttm: float,
        r: float,
        sigma: float,
        option_type: str,
    ) -> Tuple[float, float]:
        """
        Standard Black-Scholes greeks for European options.

        Returns:
            (delta, gamma)
        """
        if spot <= 0 or strike <= 0 or ttm <= 0 or sigma <= 0:
            return 0.0, 0.0

        sqrt_t = np.sqrt(ttm)
        d1 = (np.log(spot / strike) + (r + 0.5 * sigma * sigma) * ttm) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        opt_type = str(option_type).lower()
        if opt_type == "call":
            delta = float(norm.cdf(d1))
        else:
            # put delta
            delta = float(norm.cdf(d1) - 1.0)

        gamma = float(norm.pdf(d1) / (spot * sigma * sqrt_t))
        return delta, gamma

    def compute_dealer_gex_profile(
        self,
        options_df: pd.DataFrame,
        spot_price: float,
        as_of: Optional[datetime] = None,
    ) -> tuple[pd.DataFrame, float]:
        """
        Compute per-strike dealer GEX profile and aggregate net GEX.

        Dealer Gamma Exposure per line:
            If Call: -1 * Gamma * Spot^2 * 100 * Open Interest
            If Put:  +1 * Gamma * Spot^2 * 100 * Open Interest

        Aggregation:
            - Sum across all strikes & expirations to get net GEX.
            - Aggregate by strike (across expiries) for profile.
        """
        if options_df is None or options_df.empty or not np.isfinite(spot_price):
            empty = pd.DataFrame(columns=["strike", "dealer_gex", "color"])
            return empty, 0.0

        df = options_df.copy()

        # Basic cleaning
        for col in ("strike", "openInterest", "impliedVolatility"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["strike", "openInterest"])
        df = df[df["openInterest"] > 0]

        if df.empty:
            empty = pd.DataFrame(columns=["strike", "dealer_gex", "color"])
            return empty, 0.0

        # IV fallback
        if "impliedVolatility" not in df.columns:
            df["impliedVolatility"] = 0.20
        df["impliedVolatility"] = df["impliedVolatility"].replace(0.0, np.nan).fillna(0.20)

        # Expiration & TTM
        if "expiration" not in df.columns:
            raise ValueError("Options DataFrame must contain 'expiration' column")
        df["expiration"] = pd.to_datetime(df["expiration"])

        if as_of is None:
            as_of = datetime.now(timezone.utc)
        as_of_date = as_of.date()

        days_to_expiry = (df["expiration"].dt.date - as_of_date).apply(lambda d: max(d.days, 1))
        df["ttm_years"] = days_to_expiry / 365.0

        # Compute dealer GEX per row
        def _row_gex(row: pd.Series) -> float:
            opt_type = str(row.get("option_type", "call")).lower()
            strike = float(row["strike"])
            oi = float(row["openInterest"])
            sigma = float(row["impliedVolatility"])
            ttm = float(row["ttm_years"])

            # IV safeguard
            if not np.isfinite(sigma) or sigma <= 0.0:
                sigma = 0.20

            _, gamma = self._black_scholes_greeks(
                spot=spot_price,
                strike=strike,
                ttm=ttm,
                r=self.risk_free_rate,
                sigma=sigma,
                option_type=opt_type,
            )

            if gamma == 0.0 or oi <= 0.0:
                return 0.0

            if opt_type == "call":
                dealer_gamma = -1.0 * gamma * (spot_price ** 2) * 100.0 * oi
            else:
                dealer_gamma = +1.0 * gamma * (spot_price ** 2) * 100.0 * oi

            return float(dealer_gamma)

        df["dealer_gex"] = df.apply(_row_gex, axis=1)

        # Aggregate by strike
        profile = (
            df.groupby("strike", as_index=False)["dealer_gex"]
            .sum()
            .sort_values("strike")
            .reset_index(drop=True)
        )

        # Filter to +/- 20% around spot
        lower = 0.8 * spot_price
        upper = 1.2 * spot_price
        profile = profile[(profile["strike"] >= lower) & (profile["strike"] <= upper)]

        profile["color"] = np.where(profile["dealer_gex"] >= 0.0, "green", "red")

        net_gex = float(df["dealer_gex"].sum())
        return profile, net_gex

    def build_gex_history(
        self,
        price_df: pd.DataFrame,
        net_gex_today: float,
        existing_history: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build a GEX history time series.

        Shadow backfill model (per-ticker normalization):
            - Compute 20D RV and 20D average notional (price * volume).
            - Let rv_anchor = median(20D RV) over the lookback window.
            - Proxy GEX = (rv_anchor - RV_20d) * Avg_Notional * 0.2
              so that "low vs high vol" is defined relative to each ticker's
              own typical RV, making shapes comparable across tickers.

        Cold Start:
            - Use the shadow backfill model as the full history,
              then overwrite the last date with today's options-based net GEX.

        Warm Start (existing history on disk):
            - Combine existing history with the proxy series, preferring
              realized history when present.
        """
        if price_df is None or price_df.empty:
            return pd.DataFrame(columns=["net_gex", "gex_percentile"])

        # Restrict to requested lookback
        price_df = price_df.copy()
        if self.lookback_years > 0:
            max_ts = price_df.index.max()
            cutoff = max_ts - timedelta(days=int(self.lookback_years * 365))
            price_df = price_df[price_df.index >= cutoff]

        if price_df.empty:
            return pd.DataFrame(columns=["net_gex", "gex_percentile"])

        # Choose price column
        price_col = "Adj Close" if "Adj Close" in price_df.columns else "Close"
        if price_col not in price_df.columns:
            raise ValueError("Price DataFrame must contain 'Adj Close' or 'Close'")

        close = price_df[price_col].astype(float).copy()
        close = close.replace([np.inf, -np.inf], np.nan).dropna()
        if close.empty:
            return pd.DataFrame(columns=["net_gex", "gex_percentile"])

        # Align to close index
        price_df = price_df.loc[close.index]

        # Rolling realized volatility (20D)
        log_returns = np.log(close / close.shift(1))
        rv_20d = log_returns.rolling(window=20).std() * np.sqrt(252.0)

        # Rolling notional (20D average of price * volume)
        if "Volume" in price_df.columns:
            volume = price_df["Volume"].astype(float).fillna(0.0)
        else:
            volume = pd.Series(1.0, index=price_df.index)

        notional = close * volume
        avg_notional = notional.rolling(window=20).mean()

        # Per-ticker RV anchor so SPY/QQQ/IWM all have comparable "shape"
        rv_anchor = rv_20d.median(skipna=True)
        if not np.isfinite(rv_anchor) or rv_anchor <= 0:
            rv_anchor = 0.16  # hard fallback if something goes wrong

        # Shadow GEX proxy with per-ticker normalization
        gex_proxy = (rv_anchor - rv_20d) * avg_notional * 0.2
        gex_proxy = gex_proxy.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        proxy_series = gex_proxy.rename("net_gex")

        # Incorporate existing history if provided
        if existing_history is not None and not existing_history.empty:
            hist = existing_history.copy()
            if "net_gex" not in hist.columns:
                col = "net_gex" if "net_gex" in hist.columns else hist.columns[0]
                hist = hist.rename(columns={col: "net_gex"})
            hist_series = hist["net_gex"].astype(float)
            combined = proxy_series.combine_first(hist_series)
        else:
            combined = proxy_series

        combined = combined.sort_index()

        # Overwrite last date with today's net GEX
        if not combined.empty:
            combined.iloc[-1] = float(net_gex_today)

        gex_df = pd.DataFrame({"net_gex": combined})

        # Percentile rank (0-100)
        if len(gex_df) > 1:
            ranks = gex_df["net_gex"].rank(method="average")
            gex_df["gex_percentile"] = 100.0 * (ranks - 1.0) / (len(gex_df) - 1.0)
        else:
            gex_df["gex_percentile"] = 50.0

        return gex_df


class DashboardRenderer:
    """
    Renders an offline HTML dashboard with Plotly.

    Requirements:
        - No external CDNs; embed plotly.js via plotly.offline.get_plotlyjs().
        - Two tabs: "History" (time series) and "Profile" (strike profile).
        - Inject JS that dispatches window.resize on tab click to fix blank charts.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def render_dashboard(
        self,
        ticker: str,
        gex_history: pd.DataFrame,
        strike_profile: pd.DataFrame,
        spot_price: float,
    ) -> str:
        """
        Build the HTML dashboard for a single ticker.

        Returns:
            Path to the generated HTML file.
        """
        if gex_history is None or gex_history.empty:
            # Ensure non-empty frame for plotting
            gex_history = pd.DataFrame(
                {
                    "net_gex": [0.0],
                    "gex_percentile": [50.0],
                },
                index=[datetime.now(timezone.utc)],
            )

        # HISTORY FIGURE
        history_fig = make_subplots(specs=[[{"secondary_y": True}]])
        history_fig.add_trace(
            go.Scatter(
                x=gex_history.index,
                y=gex_history["net_gex"],
                mode="lines",
                name="Net Dealer GEX ($)",
            ),
            secondary_y=False,
        )
        history_fig.add_trace(
            go.Scatter(
                x=gex_history.index,
                y=gex_history["gex_percentile"],
                mode="lines",
                name="GEX Percentile Rank",
            ),
            secondary_y=True,
        )
        history_fig.update_xaxes(title_text="Date")
        history_fig.update_yaxes(title_text="Net Dealer GEX ($)", secondary_y=False)
        history_fig.update_yaxes(title_text="GEX Percentile (%)", range=[0, 100], secondary_y=True)
        history_fig.update_layout(
            title=f"{ticker} Dealer Gamma Exposure History",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        # STRIKE PROFILE FIGURE
        if strike_profile is None or strike_profile.empty:
            strike_profile = pd.DataFrame(
                {"strike": [spot_price], "dealer_gex": [0.0], "color": ["green"]}
            )

        bar = go.Bar(
            x=strike_profile["strike"],
            y=strike_profile["dealer_gex"],
            marker=dict(color=strike_profile["color"]),
            name="Dealer GEX by Strike",
        )
        profile_fig = go.Figure(data=[bar])
        profile_fig.update_layout(
            title=f"{ticker} Dealer GEX Strike Profile (Spot {spot_price:.2f})",
            xaxis_title="Strike Price",
            yaxis_title="Dealer GEX ($)",
            margin=dict(l=40, r=40, t=60, b=40),
        )

        # Serialize figures
        history_json = json.dumps(history_fig, cls=PlotlyJSONEncoder)
        profile_json = json.dumps(profile_fig, cls=PlotlyJSONEncoder)

        # Inline plotly.js
        plotly_js = get_plotlyjs()

        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{ticker} Dealer GEX Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        .tab-container {{ margin: 10px; }}
        .tab-button {{
            padding: 8px 12px;
            cursor: pointer;
            border: 1px solid #ccc;
            border-bottom: none;
            background: #f0f0f0;
            margin-right: 4px;
        }}
        .tab-button.active {{ background: #ffffff; font-weight: bold; }}
        .tab-content {{
            display: none;
            padding: 10px;
            border: 1px solid #ccc;
        }}
        #history_chart, #profile_chart {{
            width: 100%;
            height: 600px;
        }}
    </style>
</head>
<body>
    <div class="tab-container">
        <button id="tab-history" class="tab-button" onclick="showTab(event, 'history')">History</button>
        <button id="tab-profile" class="tab-button" onclick="showTab(event, 'profile')">Strike Profile</button>
    </div>
    <div id="history" class="tab-content">
        <div id="history_chart"></div>
    </div>
    <div id="profile" class="tab-content">
        <div id="profile_chart"></div>
    </div>
    <script type="text/javascript">
{plotly_js}
    </script>
    <script type="text/javascript">
        var historyFig = {history_json};
        var profileFig = {profile_json};

        function renderCharts() {{
            Plotly.newPlot('history_chart', historyFig.data, historyFig.layout, {{responsive: true}});
            Plotly.newPlot('profile_chart', profileFig.data, profileFig.layout, {{responsive: true}});
        }}

        function showTab(evt, tabId) {{
            var contents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < contents.length; i++) {{
                contents[i].style.display = 'none';
            }}
            var buttons = document.getElementsByClassName('tab-button');
            for (var j = 0; j < buttons.length; j++) {{
                buttons[j].classList.remove('active');
            }}
            var elem = document.getElementById(tabId);
            if (elem) {{
                elem.style.display = 'block';
            }}
            if (evt && evt.currentTarget) {{
                evt.currentTarget.classList.add('active');
            }}
            // Tab resize fix
            window.dispatchEvent(new Event('resize'));
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            // Default to history tab
            var historyTab = document.getElementById('tab-history');
            if (historyTab) {{
                historyTab.classList.add('active');
            }}
            var historyContent = document.getElementById('history');
            if (historyContent) {{
                historyContent.style.display = 'block';
            }}
            renderCharts();
        }});
    </script>
</body>
</html>
""".strip(
            "\n"
        )

        html = html_template.format(
            ticker=ticker,
            plotly_js=plotly_js,
            history_json=history_json,
            profile_json=profile_json,
        )

        out_path = os.path.join(self.output_dir, f"{ticker}_dealer_gex_dashboard.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dealer Gamma Exposure (GEX) Engine")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "IWM"],
        help="List of tickers to process (default: SPY QQQ IWM)",
    )
    parser.add_argument(
        "--output-dir",
        default="./market_data",
        help="Root directory for cached CSVs and dashboards",
    )
    parser.add_argument(
        "--lookback",
        type=float,
        default=1.0,
        help="Years of history for the backfill model (default: 1)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Risk-free interest rate for Black-Scholes (default: 0.04)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    tickers = args.tickers
    output_dir = args.output_dir
    lookback_years = args.lookback
    risk_free_rate = args.risk_free_rate

    data_ingestion = DataIngestion(output_dir=output_dir, lookback_years=lookback_years)
    analysis = FinancialAnalysis(risk_free_rate=risk_free_rate, lookback_years=lookback_years)
    renderer = DashboardRenderer(output_dir=output_dir)

    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue

        print(f"Processing {ticker}...", file=sys.stderr)

        # Price history (disk-first)
        try:
            price_df = data_ingestion.get_price_history(ticker)
        except Exception as exc:
            print(f"ERROR: Failed to load price history for {ticker}: {exc}", file=sys.stderr)
            continue

        # Spot price from last available close
        price_col = "Adj Close" if "Adj Close" in price_df.columns else "Close"
        if price_col not in price_df.columns:
            print(f"ERROR: No price column ('Adj Close'/'Close') for {ticker}", file=sys.stderr)
            continue

        spot_series = price_df[price_col].dropna()
        if spot_series.empty:
            print(f"ERROR: No valid spot prices for {ticker}", file=sys.stderr)
            continue

        spot_price = float(spot_series.iloc[-1])

        # Options data (disk-first)
        try:
            options_df = data_ingestion.get_options_data(ticker, num_expiries=3)
        except Exception as exc:
            print(f"WARNING: Failed to load options data for {ticker}: {exc}", file=sys.stderr)
            options_df = pd.DataFrame()

        # Dealer GEX (profile + net)
        strike_profile, net_gex_today = analysis.compute_dealer_gex_profile(
            options_df=options_df,
            spot_price=spot_price,
        )

        # GEX history (shadow model + disk history)
        existing_gex_history = data_ingestion.load_gex_history(ticker)
        gex_history = analysis.build_gex_history(
            price_df=price_df,
            net_gex_today=net_gex_today,
            existing_history=existing_gex_history,
        )
        data_ingestion.save_gex_history(ticker, gex_history)

        # Render dashboard
        html_path = renderer.render_dashboard(
            ticker=ticker,
            gex_history=gex_history,
            strike_profile=strike_profile,
            spot_price=spot_price,
        )

        print(f"Dashboard for {ticker} written to: {html_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

