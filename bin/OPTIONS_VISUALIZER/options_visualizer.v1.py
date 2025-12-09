#!/usr/bin/env python3
# SCRIPTNAME: mega_unified_run.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
#
# Description:
#   Single CLI entry-point using chartlib_unified.py.
#   Orchestrates Spread Analysis, Streak Analysis, and Full Chart Reports.
#
# Commands:
#   analyze:
#     spreads <TICKER> <WEEKS_TO_EXPIRY> [--offset] [--max-otm-percent] [--save-csv]
#     streaks <TICKER> [--period] [--save-csv]
#   charts:
#     all <TICKER> [--quantile] [--smooth-iv] [--iv-min] [--iv-max]

from __future__ import annotations

import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
from typing import List, Tuple
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import chartlib_unified as clu

# Allowed OHLCV source
try:
    import data_retrieval as dr
    from data_retrieval import load_or_download_ticker
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)


# ------------------------------- helpers ---------------------------------------

def _spot_from_cache(ticker: str, period: str = "5d") -> float:
    df = load_or_download_ticker(ticker, period=period)
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError(f"Could not get spot for {ticker}")
    return float(df["Close"].dropna().iloc[-1])


def _ensure_dirs(ticker: str) -> Tuple[str, str, str]:
    # CONSTRAINT: Use /dev/shm via data_retrieval
    base_dir = dr.create_output_directory(ticker)
    
    html_dir = os.path.join(base_dir, "html")
    img_dir = os.path.join(base_dir, "images")
    rep_dir = os.path.join(base_dir, "reports")
    an_dir = os.path.join(base_dir, "analysis")
    
    for d in (html_dir, img_dir, rep_dir, an_dir):
        os.makedirs(d, exist_ok=True)
    return html_dir, img_dir, rep_dir


def _write_fig(fig, html_path: str, img_path: str) -> Tuple[str, str]:
    fig.write_html(html_path)
    # to PNG (requires kaleido)
    try:
        fig.write_image(img_path, format="png", scale=2)
    except Exception as e:
        print(f"[warn] Static image export failed for {os.path.basename(html_path)}: {e}\n"
              "       Install 'kaleido' (pip install -U kaleido) to enable PNG export.")
    return html_path, img_path


def _open_html(paths: List[str]) -> None:
    for p in paths:
        try:
            webbrowser.open_new_tab("file://" + os.path.abspath(p))
        except Exception:
            pass

def _dataframe_to_figure(df: pd.DataFrame, title: str) -> go.Figure:
    """Converts a DataFrame to a styled Plotly Table figure."""
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='grey',
                    align='left',
                    font=dict(color='white', size=12)),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='#1E1E1E',
                   align='left',
                   font=dict(color='white', size=11),
                   height=28)
    )])
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def _make_pdf(image_paths: List[str], pdf_path: str, titles: List[str]) -> None:
    """
    Compose a multi-page PDF from a list of PNGs. Requires Pillow.
    Each page gets a small text banner via Pillow (if available).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:
        print(f"[warn] Report PDF not generated: {e}\n"
              "       Install 'Pillow' (pip install -U pillow) to enable PDF reports.")
        return

    pages = []
    for path, title in zip(image_paths, titles):
        if not os.path.isfile(path):
            continue
        img = Image.open(path).convert("RGB")
        # add title banner at top
        W, H = img.size
        banner_h = max(40, int(0.05 * H))
        banner = Image.new("RGB", (W, banner_h), (30, 30, 30))
        draw = ImageDraw.Draw(banner)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((10, 10), title, fill=(255, 255, 255), font=font)
        combined = Image.new("RGB", (W, H + banner_h), (0, 0, 0))
        combined.paste(banner, (0, 0))
        combined.paste(img, (0, banner_h))
        pages.append(combined)

    if not pages:
        print("[warn] No images available to assemble report.")
        return

    first, rest = pages[0], pages[1:]
    first.save(pdf_path, save_all=True, append_images=rest)
    print(f"[ok] Wrote {pdf_path}")


# ------------------------------- analyze ---------------------------------------

def cmd_analyze_spreads(args):
    t = args.ticker.upper()
    spot = _spot_from_cache(t, period="5d")
    exps = clu.list_expiration_dates(t)
    if not exps:
        raise SystemExit(f"No option expirations found for {t}")
    if args.weeks_to_expiry < 0 or args.weeks_to_expiry >= len(exps):
        raise SystemExit(f"Invalid expiration index {args.weeks_to_expiry}. Available 0..{len(exps)-1}")
    exp = exps[args.weeks_to_expiry]
    df = clu.get_options_data(t, exp)
    if df is None or df.empty:
        raise SystemExit(f"No options data for {t} on {exp}")

    calls = df[df.get("optionType", "") == "call"].copy()
    puts = df[df.get("optionType", "") == "put"].copy()

    print(f"Ticker: {t} | Expiration: {exp} | Spot: {spot:.2f}")

    # Base strategies
    try:
        bcds = clu.calculate_bull_call_debit_spread(calls, spot)
        bpcs = clu.calculate_bull_put_credit_spread(puts, spot, max_otm_percent=args.max_otm_percent)
        print("\nBull Call Debit Spread")
        for k, v in bcds.items(): print(f"  {k}: {v}")
        print("\nBull Put Credit Spread")
        for k, v in bpcs.items(): print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nCould not calculate base spreads: {e}")
        bcds, bpcs = {}, {}

    # Offset stress
    if args.offset and args.offset != 0.0:
        adj = round(spot + args.offset, 2)
        print(f"\n[Offset] {args.offset:+} -> Adjusted Spot: {adj:.2f}")
        try:
            bcds2 = clu.calculate_bull_call_debit_spread(calls, adj)
            bpcs2 = clu.calculate_bull_put_credit_spread(puts, adj, max_otm_percent=args.max_otm_percent)
            print("\nBull Call Debit Spread (offset)")
            for k, v in bcds2.items(): print(f"  {k}: {v}")
            print("\nBull Put Credit Spread (offset)")
            for k, v in bpcs2.items(): print(f"  {k}: {v}")
        except Exception as e:
            print(f"\nCould not calculate offset spreads: {e}")

    if args.save_csv:
        if bcds and bpcs:
            df_bcds = pd.DataFrame(list(bcds.items()), columns=['Parameter', 'Value'])
            df_bpcs = pd.DataFrame(list(bpcs.items()), columns=['Parameter', 'Value'])
            df_report = pd.concat([df_bcds, df_bpcs], ignore_index=True)
            
            # CONSTRAINT: Use /dev/shm
            out_dir = os.path.join(dr.create_output_directory(t), "analysis")
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"{t}_spreads_{exp}.csv")
            df_report.to_csv(out_csv, index=False)
            print(f"[ok] Wrote spread analysis to {out_csv}")
        else:
            print("[warn] No spread data to save.")


def cmd_analyze_streaks(args):
    t = args.ticker.upper()
    df = load_or_download_ticker(t, period=args.period)
    if df is None or df.empty:
        raise SystemExit(f"No OHLCV for {t}")
    table = clu.calculate_streak_probabilities(df)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(table)

    if args.save_csv:
        # CONSTRAINT: Use /dev/shm
        out_dir = os.path.join(dr.create_output_directory(t), "analysis")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{t}_streak_statistics.csv")
        table.to_csv(out_csv, index=False)
        print(f"[ok] Wrote {out_csv}")


# ------------------------------- charts ----------------------------------------

def _save_chain_tumbler_assets(ticker: str, expiration: str,
                               html_dir: str, img_dir: str) -> Tuple[str, str]:
    t = ticker.upper()
    html_str = clu.plot_interactive_chain_tumbler(t, expiration)
    html_path = os.path.join(html_dir, f"{t}_chain_tumbler_{expiration.replace('-', '')}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    # Static summary figure for report:
    chain = clu.get_options_data(t, expiration)
    calls = chain[chain.get("optionType", "") == "call"].copy()
    puts = chain[chain.get("optionType", "") == "put"].copy()
    calls["openInterest"] = pd.to_numeric(calls.get("openInterest"), errors="coerce")
    puts["openInterest"] = pd.to_numeric(puts.get("openInterest"), errors="coerce")
    agg = (calls[["strike", "openInterest"]].rename(columns={"openInterest": "call_oi"})
             .merge(puts[["strike", "openInterest"]].rename(columns={"openInterest": "put_oi"}),
                    on="strike", how="outer").fillna(0).sort_values("strike"))
    
    fig = go.Figure()
    fig.add_bar(x=agg["strike"], y=agg["call_oi"], name="Calls OI")
    fig.add_bar(x=agg["strike"], y=agg["put_oi"], name="Puts OI")
    fig.update_layout(title=f"{t} — Chain Snapshot (OI by Strike) @ {expiration}",
                      xaxis_title="Strike", yaxis_title="Open Interest",
                      template="plotly_dark", barmode="overlay")
    img_path = os.path.join(img_dir, f"{t}_chain_tumbler_{expiration.replace('-', '')}.png")
    try:
        fig.write_image(img_path, format="png", scale=2)
    except Exception as e:
        print(f"[warn] PNG export for chain snapshot failed: {e}")
    return html_path, img_path


def cmd_charts_all(args):
    t = args.ticker.upper()
    html_dir, img_dir, rep_dir = _ensure_dirs(t)
    spot = _spot_from_cache(t, period="5d")

    all_opts = clu.get_all_options_data(t)
    if all_opts is None or all_opts.empty:
        raise SystemExit(f"No options data available for {t}")

    exps = clu.list_expiration_dates(t)
    if not exps:
        raise SystemExit(f"No expirations for {t}")
    near_exp = exps[0]
    
    # Perform streak analysis for the report
    streak_html, streak_png = None, None
    try:
        ohlc_df = load_or_download_ticker(t, period="max")
        streak_table = clu.calculate_streak_probabilities(ohlc_df)
        
        # CONSTRAINT: /dev/shm
        an_dir = os.path.join(dr.create_output_directory(t), "analysis")
        os.makedirs(an_dir, exist_ok=True)
        streak_csv_path = os.path.join(an_dir, f"{t}_streak_statistics.csv")
        streak_table.to_csv(streak_csv_path, index=False)
        print(f"[ok] Wrote streak analysis to {streak_csv_path}")
        
        fig_streaks = _dataframe_to_figure(streak_table.head(20), f"{t} — Streak Probabilities (Top 20)")
        streak_html = os.path.join(html_dir, f"{t}_streak_analysis.html")
        streak_png = os.path.join(img_dir, f"{t}_streak_analysis.png")
        _write_fig(fig_streaks, streak_html, streak_png)
    except Exception as e:
        print(f"[warn] Could not generate streak analysis table: {e}")

    ## MODIFIED: Perform spread analysis for the first 4 expirations ##
    spread_assets = []
    for i in range(min(4, len(exps))):
        exp_date = exps[i]
        try:
            chain = clu.get_options_data(t, exp_date)
            calls = chain[chain.get("optionType") == "call"].copy()
            puts = chain[chain.get("optionType") == "put"].copy()

            bcds = clu.calculate_bull_call_debit_spread(calls, spot)
            bpcs = clu.calculate_bull_put_credit_spread(puts, spot)

            df_bcds = pd.DataFrame(list(bcds.items()), columns=['Parameter', 'Value'])
            df_bpcs = pd.DataFrame(list(bpcs.items()), columns=['Parameter', 'Value'])
            df_spreads = pd.concat([df_bcds, df_bpcs], ignore_index=True)

            fig_spreads = _dataframe_to_figure(df_spreads, f"{t} Spread Analysis @ {exp_date} (Spot: ${spot:.2f})")
            html_path = os.path.join(html_dir, f"{t}_spread_analysis_{exp_date}.html")
            png_path = os.path.join(img_dir, f"{t}_spread_analysis_{exp_date}.png")
            _write_fig(fig_spreads, html_path, png_path)
            spread_assets.append({'html': html_path, 'png': png_path, 'title': f"{t} — Spread Analysis @ {exp_date}"})
        except Exception as e:
            print(f"[warn] Could not generate spread analysis for {exp_date}: {e}")

    # 1) Chain tumbler HTML (+ static summary PNG for report)
    tumbler_html, tumbler_png = _save_chain_tumbler_assets(t, near_exp, html_dir, img_dir)

    # 2) IV surface
    fig_iv = clu.plot_iv_surface(all_opts, t,
                                 smooth=args.smooth_iv,
                                 dte_step=1, strike_step=None,
                                 smooth_kernel=3,
                                 iv_min=args.iv_min, iv_max=args.iv_max,
                                 iv_pct=False, colorscale="Turbo")
    iv_html = os.path.join(html_dir, f"{t}_iv_surface.html")
    iv_png  = os.path.join(img_dir,  f"{t}_iv_surface.png")
    _write_fig(fig_iv, iv_html, iv_png)

    # 3) OI dashboard (4 figs)
    figs = clu.plot_oi_dashboard(all_opts, t, current_price=spot)
    dash_paths = []
    for name, fig in figs.items():
        h = os.path.join(html_dir, f"{t}_{name}.html")
        p = os.path.join(img_dir,  f"{t}_{name}.png")
        _write_fig(fig, h, p)
        dash_paths.append((h, p))

    # 4) Additional charts (3 figs)
    extras = clu.plot_additional_charts(all_opts, t, quantile=args.quantile)
    extra_paths = []
    for name, fig in extras.items():
        h = os.path.join(html_dir, f"{t}_{name}.html")
        p = os.path.join(img_dir,  f"{t}_{name}.png")
        _write_fig(fig, h, p)
        extra_paths.append((h, p))

    # Auto-open all HTMLs
    htmls = [h for (h, _) in dash_paths] + [h for (h, _) in extra_paths]
    if streak_html: htmls.insert(0, streak_html)
    for asset in reversed(spread_assets): htmls.insert(0, asset['html'])
    htmls.insert(0, iv_html)
    htmls.insert(0, tumbler_html)
    _open_html(htmls)

    # Assemble PDF report
    imgs = [p for (_, p) in dash_paths] + [p for (_, p) in extra_paths]
    titles = [
        f"{t} — OI 3D Bubbles", f"{t} — OI Aggregated 2D", f"{t} — OI 3D (Z=OI)",
        f"{t} — Distribution Dashboard", f"{t} — OI by Strike", f"{t} — Unusual Activity", f"{t} — OI Heatmap",
    ]
    if streak_png:
        imgs.insert(0, streak_png)
        titles.insert(0, f"{t} — Streak Analysis")
    for asset in reversed(spread_assets):
        imgs.insert(0, asset['png'])
        titles.insert(0, asset['title'])
    imgs.insert(0, iv_png)
    titles.insert(0, f"{t} — IV Surface")
    imgs.insert(0, tumbler_png)
    titles.insert(0, f"{t} — Chain Tumbler Snapshot @ {near_exp}")

    pdf_path = os.path.join(rep_dir, f"{t}_mega_report.pdf")
    _make_pdf(imgs, pdf_path, titles)
    print(f"[ok] Charts complete. HTML/PNG under {html_dir}, report compiled.")


# ------------------------------- parser ----------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mega_unified_run.py", description="Unified financial charting & analysis CLI")
    sub = p.add_subparsers(dest="command")

    # analyze
    pa = sub.add_parser("analyze", help="Run non-plotting analysis")
    sub_an = pa.add_subparsers(dest="subcommand")

    p_sp = sub_an.add_parser("spreads", help="Compute Bull Call Debit / Bull Put Credit spreads")
    p_sp.add_argument("ticker", type=str)
    p_sp.add_argument("weeks_to_expiry", type=int)
    p_sp.add_argument("--offset", type=float, default=0.0, help="Price offset for stress (+/-)")
    p_sp.add_argument("--max-otm-percent", type=float, default=5.0, help="Max %% below spot for OTM put buy")
    p_sp.add_argument("--save-csv", action="store_true", help="Save results to a CSV file")
    p_sp.set_defaults(func=cmd_analyze_spreads)

    p_st = sub_an.add_parser("streaks", help="Compute streak probabilities")
    p_st.add_argument("ticker", type=str)
    p_st.add_argument("--period", type=str, default="max")
    p_st.add_argument("--save-csv", action="store_true")
    p_st.set_defaults(func=cmd_analyze_streaks)

    # charts
    pc = sub.add_parser("charts", help="Generate visuals")
    sub_ch = pc.add_subparsers(dest="subcommand")

    p_all = sub_ch.add_parser("all", help="Generate every plot & compile a PDF report")
    p_all.add_argument("ticker", type=str)
    p_all.add_argument("--quantile", type=float, default=0.95)
    p_all.add_argument("--smooth-iv", action="store_true")
    p_all.add_argument("--iv-min", type=float, default=None)
    p_all.add_argument("--iv-max", type=float, default=None)
    p_all.set_defaults(func=cmd_charts_all)

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    if not getattr(args, "subcommand", None):
        # if user typed only 'analyze' or 'charts'
        parser.parse_args([args.command, "--help"])
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
