#!/usr/bin/env python3
"""
SCRIPTNAME: bad_day_put_coverage_dashboard.py

Description:
    Standalone "bad-day" put-coverage scenario engine with creative Plotly visualizations.

    What it does:
      - Downloads underlying price + option chain from Yahoo via yfinance.
      - Selects a small but useful set of put hedges:
          * 3 expiries: short / medium / longer dated (when available)
          * 3 strikes per expiry: ATM, ~-5%, ~-10% OTM puts
      - Builds a scenario grid across:
          * Spot shocks at the open: -1%, -2%, -3%, -5%
          * IV paths: "Crush" (0.7x), "Base" (1.0x), "Spike" (1.3x), "Panic" (1.5x)
      - Prices each option under each scenario using Black–Scholes (no external libs).
      - Computes Greeks (Delta, Gamma, Vega, Theta) under each scenario.
      - Computes P&L vs current mid for a +1 long put.

    Visualizations (all embedded in ONE HTML dashboard):
      1) Option selection table (baseline metrics for each chosen put).
      2) Scenario P&L heatmap for the focus option (ATM, mid expiry).
      3) 3D P&L surface: spot shock vs IV multiplier.
      4) P&L vs spot shock lines for each IV path.
      5) Delta vs spot shock lines for each IV path.

    Usage:
        python bad_day_put_coverage_dashboard.py --ticker SPY
        python bad_day_put_coverage_dashboard.py --ticker QQQ --output my_dashboard.html

    Requirements:
        pip install yfinance plotly pandas numpy
"""

import argparse
import math
import os
import webbrowser
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import yfinance as yf


# -----------------------------
# Black–Scholes Put + Greeks
# -----------------------------

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (no SciPy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_put_price_and_greeks(S: float,
                            K: float,
                            T: float,
                            r: float,
                            sigma: float):
    """
    Black–Scholes European put price and Greeks.

    Returns:
        price, delta, gamma, vega, theta (theta per YEAR, not per day)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Expired or nonsensical; approximate as intrinsic.
        intrinsic = max(K - S, 0.0)
        return intrinsic, -1.0 if S < K else 0.0, 0.0, 0.0, 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    Nmd1 = _norm_cdf(-d1)
    Nmd2 = _norm_cdf(-d2)
    nd1 = _norm_pdf(d1)

    # Price
    price = K * math.exp(-r * T) * Nmd2 - S * Nmd1

    # Greeks
    delta = Nd1 - 1.0  # put delta
    gamma = nd1 / (S * sigma * sqrtT)
    vega = S * nd1 * sqrtT  # per 1.00 change in vol (i.e., 100 vol points)
    # Theta (annualized, Black–Scholes)
    # Use convention: theta is dPrice/dt, with t in YEARS -> usually negative for long options
    first_term = - (S * nd1 * sigma) / (2.0 * sqrtT)
    second_term = r * K * math.exp(-r * T) * Nmd2
    theta = first_term + second_term  # per YEAR

    return price, delta, gamma, vega, theta


# -----------------------------
# Data + Scenario Engine
# -----------------------------


def pick_expiries(expiries, min_days_list=(7, 30, 90)):
    """
    Choose up to 3 expiries: nearest >= each min_days in min_days_list.
    Returns a list of chosen expiry strings (sorted by date).
    """
    today = date.today()
    chosen = []
    # Convert strings to dates once
    exp_dates = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            exp_dates.append((e, d))
        except Exception:
            continue

    for min_days in min_days_list:
        target_date = today.toordinal() + min_days
        best = None
        best_dt = None
        for e, d in exp_dates:
            days = (d.toordinal() - today.toordinal())
            if days >= min_days:
                if best is None or d < best_dt:
                    best = e
                    best_dt = d
        if best is not None and best not in chosen:
            chosen.append(best)

    # As a fallback, if we got fewer than 1, just take first 1-3 expiries
    if not chosen and exp_dates:
        chosen = [e for e, _ in exp_dates[: min(3, len(exp_dates))]]

    # Sort by calendar date
    chosen_sorted = sorted(
        chosen,
        key=lambda e: datetime.strptime(e, "%Y-%m-%d").date()
    )
    return chosen_sorted


def pick_strikes_for_exp(puts_df: pd.DataFrame, spot: float):
    """
    Pick ATM, ~-5%, ~-10% OTM strikes from a puts DataFrame.
    Returns a subset DataFrame.
    """
    if puts_df.empty:
        return puts_df

    strikes = puts_df['strike'].values
    targets = [spot, spot * 0.95, spot * 0.90]
    chosen_rows = []

    for tgt in targets:
        idx = (np.abs(strikes - tgt)).argmin()
        chosen = puts_df.iloc[idx]
        chosen_rows.append(chosen)

    result = pd.DataFrame(chosen_rows).drop_duplicates(subset=['strike'])
    return result


def estimate_mid(row):
    """Estimate mid price from bid/ask/last."""
    bid = row.get('bid', np.nan)
    ask = row.get('ask', np.nan)
    last = row.get('lastPrice', np.nan)

    if not np.isnan(bid) and not np.isnan(ask) and ask > 0:
        return 0.5 * (bid + ask)
    if not np.isnan(last) and last > 0:
        return last
    if not np.isnan(bid) and bid > 0:
        return bid
    if not np.isnan(ask) and ask > 0:
        return ask
    return np.nan


def build_option_universe(ticker: str, r: float = 0.02):
    """
    Download underlying + options and build a compact put universe.

    Returns:
        spot (float),
        universe_df (DataFrame) with columns:
            ['ticker','expiry','days_to_exp','strike','mid','iv',
             'baseline_price','baseline_delta','baseline_gamma',
             'baseline_vega','baseline_theta']
    """
    tk = yf.Ticker(ticker)

    # Spot
    hist = tk.history(period="30d")
    if hist.empty:
        raise RuntimeError("No price history for ticker.")
    spot = float(hist['Close'].iloc[-1])

    # Expiries
    expiries = tk.options
    if not expiries:
        raise RuntimeError("No options listed for ticker.")

    chosen_expiries = pick_expiries(expiries)
    if not chosen_expiries:
        raise RuntimeError("Could not pick expiries from options list.")

    today = date.today()
    rows = []

    for exp in chosen_expiries:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        puts = chain.puts.copy()
        if puts.empty:
            continue

        subset = pick_strikes_for_exp(puts, spot)
        if subset.empty:
            continue

        exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
        days_to_exp = max((exp_dt.toordinal() - today.toordinal()), 0)
        T = days_to_exp / 365.0

        for _, row in subset.iterrows():
            strike = float(row['strike'])
            iv = float(row.get('impliedVolatility', np.nan))
            if not (iv > 0 and math.isfinite(iv)):
                # Fallback IV if missing: 25%
                iv = 0.25

            mid = estimate_mid(row)
            if not (mid > 0 and math.isfinite(mid)):
                mid = 0.0

            price, delta, gamma, vega, theta = bs_put_price_and_greeks(
                S=spot, K=strike, T=T, r=r, sigma=iv
            )

            rows.append({
                'ticker': ticker,
                'expiry': exp,
                'days_to_exp': days_to_exp,
                'strike': strike,
                'mid': mid,          # observed (approx) from market
                'iv': iv,
                'model_price': price,
                'baseline_delta': delta,
                'baseline_gamma': gamma,
                'baseline_vega': vega,
                'baseline_theta': theta,
            })

    if not rows:
        raise RuntimeError("No suitable put options found.")

    universe_df = pd.DataFrame(rows)
    # Use model price as baseline if mid is missing or 0
    universe_df['baseline_price'] = np.where(
        (universe_df['mid'] > 0) & np.isfinite(universe_df['mid']),
        universe_df['mid'],
        universe_df['model_price']
    )
    return spot, universe_df


def build_scenarios(spot: float):
    """
    Define spot shocks and IV multipliers for scenarios.

    Returns:
        spot_shocks (list of dict)
        iv_paths (list of dict)
    """
    spot_shocks = [
        {'name': '-1%', 'shock_pct': -0.01},
        {'name': '-2%', 'shock_pct': -0.02},
        {'name': '-3%', 'shock_pct': -0.03},
        {'name': '-5%', 'shock_pct': -0.05},
    ]
    iv_paths = [
        {'name': 'Crush (0.7x)', 'multiplier': 0.7},
        {'name': 'Base (1.0x)', 'multiplier': 1.0},
        {'name': 'Spike (1.3x)', 'multiplier': 1.3},
        {'name': 'Panic (1.5x)', 'multiplier': 1.5},
    ]
    return spot_shocks, iv_paths


def build_scenario_grid(spot: float,
                        universe_df: pd.DataFrame,
                        r: float = 0.02):
    """
    Build full scenario grid for all options in universe.

    Returns:
        scenario_df: DataFrame with columns:
            ['ticker','expiry','strike','days_to_exp','base_iv',
             'scenario_spot_name','scenario_spot_pct',
             'scenario_iv_name','iv_mult',
             'scenario_spot','scenario_iv',
             'scenario_price','scenario_delta','scenario_gamma',
             'scenario_vega','scenario_theta',
             'baseline_price','pnl']
    """
    spot_shocks, iv_paths = build_scenarios(spot)
    today = date.today()

    records = []
    for _, row in universe_df.iterrows():
        K = float(row['strike'])
        exp = row['expiry']
        base_iv = float(row['iv'])
        baseline_price = float(row['baseline_price'])

        exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
        days_to_exp = max((exp_dt.toordinal() - today.toordinal()), 0)
        T = days_to_exp / 365.0

        for ss in spot_shocks:
            scen_spot = spot * (1.0 + ss['shock_pct'])
            for ivp in iv_paths:
                scen_iv = base_iv * ivp['multiplier']

                price, delta, gamma, vega, theta = bs_put_price_and_greeks(
                    S=scen_spot, K=K, T=T, r=r, sigma=scen_iv
                )

                pnl = price - baseline_price  # long +1 put

                records.append({
                    'ticker': row['ticker'],
                    'expiry': exp,
                    'strike': K,
                    'days_to_exp': days_to_exp,
                    'base_iv': base_iv,
                    'baseline_price': baseline_price,
                    'scenario_spot_name': ss['name'],
                    'scenario_spot_pct': ss['shock_pct'],
                    'scenario_iv_name': ivp['name'],
                    'iv_mult': ivp['multiplier'],
                    'scenario_spot': scen_spot,
                    'scenario_iv': scen_iv,
                    'scenario_price': price,
                    'scenario_delta': delta,
                    'scenario_gamma': gamma,
                    'scenario_vega': vega,
                    'scenario_theta': theta,
                    'pnl': pnl,
                })

    scenario_df = pd.DataFrame(records)
    return scenario_df


# -----------------------------
# Visualization Builders
# -----------------------------


def make_option_table_fig(universe_df: pd.DataFrame, spot: float):
    """Table of selected puts and baseline metrics."""
    df = universe_df.copy().sort_values(['days_to_exp', 'strike'])

    display_df = pd.DataFrame({
        'Expiry': df['expiry'],
        'DTE': df['days_to_exp'].astype(int),
        'Strike': df['strike'].round(2),
        'Moneyness (% of Spot)': (df['strike'] / spot * 100.0).round(1),
        'Baseline Price': df['baseline_price'].round(2),
        'IV (base)': (df['iv'] * 100.0).round(1),
        'Delta': df['baseline_delta'].round(3),
        'Gamma': df['baseline_gamma'].round(4),
        'Vega': df['baseline_vega'].round(2),
        'Theta/yr': df['baseline_theta'].round(2),
    })

    header_values = list(display_df.columns)
    cell_values = [display_df[col].tolist() for col in display_df.columns]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    align="center",
                    font=dict(size=12, color="white"),
                    fill_color="#1f77b4",
                ),
                cells=dict(
                    values=cell_values,
                    align="right",
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Selected Put Hedge Universe (Spot = {spot:.2f})",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def choose_focus_option(universe_df: pd.DataFrame, spot: float):
    """Pick a 'focus' option: closest to ATM with medium DTE."""
    df = universe_df.copy()
    if df.empty:
        return None

    # Target: near 30 days to expiry, ATM
    df['atm_diff'] = (df['strike'] - spot).abs()
    df['dte_diff'] = (df['days_to_exp'] - 30).abs()
    df['rank'] = df['atm_diff'] + 0.1 * df['dte_diff']
    focus_row = df.sort_values('rank').iloc[0]
    return focus_row


def make_heatmap_fig(scenario_df: pd.DataFrame, focus_row, spot: float):
    """P&L heatmap for focus option across spot shocks and IV paths."""
    mask = (
        (scenario_df['expiry'] == focus_row['expiry'])
        & (scenario_df['strike'] == focus_row['strike'])
    )
    sub = scenario_df[mask].copy()
    if sub.empty:
        return go.Figure()

    # Pivot: rows = spot scenario, cols = IV path
    pivot = sub.pivot_table(
        index='scenario_spot_name',
        columns='scenario_iv_name',
        values='pnl',
    ).sort_index()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            coloraxis="coloraxis",
            hovertemplate="Spot: %{y}<br>IV Path: %{x}<br>P&L: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"P&L Heatmap - Focus Put: Exp {focus_row['expiry']}, "
            f"Strike {focus_row['strike']:.2f} (Spot {spot:.2f})"
        ),
        xaxis_title="IV Path",
        yaxis_title="Spot Shock Scenario",
        coloraxis=dict(
            colorbar_title="P&L",
        ),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def make_surface_fig(scenario_df: pd.DataFrame, focus_row, spot: float):
    """3D P&L surface: spot shock vs IV multiplier."""
    mask = (
        (scenario_df['expiry'] == focus_row['expiry'])
        & (scenario_df['strike'] == focus_row['strike'])
    )
    sub = scenario_df[mask].copy()
    if sub.empty:
        return go.Figure()

    # Create numeric grids: X = spot_pct, Y = iv_mult, Z = pnl
    pivot = sub.pivot_table(
        index='scenario_spot_pct',
        columns='iv_mult',
        values='pnl',
    ).sort_index(axis=0).sort_index(axis=1)

    X = pivot.index.values
    Y = pivot.columns.values
    Z = pivot.values

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                coloraxis="coloraxis",
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "project_z": True,
                    }
                },
            )
        ]
    )
    fig.update_layout(
        title=(
            f"3D P&L Surface - Focus Put: Exp {focus_row['expiry']}, "
            f"Strike {focus_row['strike']:.2f}"
        ),
        scene=dict(
            xaxis_title="Spot Shock (fraction)",
            yaxis_title="IV Multiplier",
            zaxis_title="P&L",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        coloraxis=dict(
            colorbar_title="P&L",
        ),
    )
    return fig


def make_pnl_lines_fig(scenario_df: pd.DataFrame, focus_row, spot: float):
    """P&L vs spot shock lines, one line per IV path."""
    mask = (
        (scenario_df['expiry'] == focus_row['expiry'])
        & (scenario_df['strike'] == focus_row['strike'])
    )
    sub = scenario_df[mask].copy()
    if sub.empty:
        return go.Figure()

    fig = go.Figure()
    for iv_name, grp in sub.groupby('scenario_iv_name'):
        grp_sorted = grp.sort_values('scenario_spot_pct')
        x = grp_sorted['scenario_spot_pct'] * 100.0  # %
        y = grp_sorted['pnl']
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=iv_name,
                hovertemplate="Shock: %{x:.1f}%<br>P&L: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=(
            f"P&L vs Spot Shock - Focus Put: Exp {focus_row['expiry']}, "
            f"Strike {focus_row['strike']:.2f} (Spot {spot:.2f})"
        ),
        xaxis_title="Spot Shock (%)",
        yaxis_title="P&L (Long 1 Put)",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def make_delta_lines_fig(scenario_df: pd.DataFrame, focus_row, spot: float):
    """Delta vs spot shock lines, one line per IV path."""
    mask = (
        (scenario_df['expiry'] == focus_row['expiry'])
        & (scenario_df['strike'] == focus_row['strike'])
    )
    sub = scenario_df[mask].copy()
    if sub.empty:
        return go.Figure()

    fig = go.Figure()
    for iv_name, grp in sub.groupby('scenario_iv_name'):
        grp_sorted = grp.sort_values('scenario_spot_pct')
        x = grp_sorted['scenario_spot_pct'] * 100.0  # %
        y = grp_sorted['scenario_delta']
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=iv_name,
                hovertemplate="Shock: %{x:.1f}%<br>Delta: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=(
            f"Delta vs Spot Shock - Focus Put: Exp {focus_row['expiry']}, "
            f"Strike {focus_row['strike']:.2f}"
        ),
        xaxis_title="Spot Shock (%)",
        yaxis_title="Delta",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def build_dashboard_html(figs, output_file: str, title: str):
    """
    Combine multiple Plotly figures into one HTML dashboard with stacked sections.
    """
    # First figure will include plotly.js; others will not.
    divs = []
    for i, fig in enumerate(figs):
        include_js = (i == 0)
        div = plot(
            fig,
            include_plotlyjs=include_js,
            output_type="div",
            show_link=False,
        )
        divs.append(div)

    body = "\n<hr style='margin:40px 0;'>\n".join(divs)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 20px;
            background: #fafafa;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #555;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="subtitle">
        Bad-Day Put Coverage Scenario Engine - Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
    {body}
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Bad-Day Put Coverage Dashboard (standalone)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Underlying ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.02,
        help="Risk-free rate (annualized, e.g., 0.02 for 2%%). Default: 0.02",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML filename (default: bad_day_put_coverage_<TICKER>.html)",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    r = float(args.rate)
    output_file = (
        args.output
        if args.output is not None
        else f"bad_day_put_coverage_{ticker}.html"
    )

    print(f"[INFO] Building bad-day put coverage dashboard for {ticker} ...")

    try:
        spot, universe_df = build_option_universe(ticker, r=r)
    except Exception as e:
        print(f"[ERROR] Failed to build option universe: {e}")
        return

    print(f"[INFO] Spot for {ticker}: {spot:.4f}")
    print(f"[INFO] Selected {len(universe_df)} put options for scenario analysis.")

    scenario_df = build_scenario_grid(spot, universe_df, r=r)

    focus_row = choose_focus_option(universe_df, spot)
    if focus_row is None:
        print("[ERROR] Could not choose a focus option.")
        return

    print(
        "[INFO] Focus option: "
        f"Expiry {focus_row['expiry']}, Strike {focus_row['strike']:.2f}, "
        f"DTE {int(focus_row['days_to_exp'])}"
    )

    # Build figures
    fig_table = make_option_table_fig(universe_df, spot)
    fig_heatmap = make_heatmap_fig(scenario_df, focus_row, spot)
    fig_surface = make_surface_fig(scenario_df, focus_row, spot)
    fig_pnl_lines = make_pnl_lines_fig(scenario_df, focus_row, spot)
    fig_delta_lines = make_delta_lines_fig(scenario_df, focus_row, spot)

    figs = [fig_table, fig_heatmap, fig_surface, fig_pnl_lines, fig_delta_lines]

    # Build combined HTML
    title = f"Bad-Day Put Coverage Dashboard - {ticker}"
    build_dashboard_html(figs, output_file, title=title)

    abs_path = os.path.abspath(output_file)
    print(f"[INFO] Dashboard written to: {abs_path}")
    webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()

