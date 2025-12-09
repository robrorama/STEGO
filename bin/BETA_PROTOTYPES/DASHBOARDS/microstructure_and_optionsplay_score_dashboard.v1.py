#!/usr/bin/env python3
# SCRIPTNAME: ok.microstructure_and_optionsplay_score_dashboard.v1.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

#
# Description:
#   - Uses your existing data_retrieval.py and options_data_retrieval.py (unchanged) to:
#       * Load underlying OHLCV data
#       * Optionally load current option-chain snapshot(s)
#   - Computes:
#       * Daily microstructure-style features (volume z-scores, ranges, returns)
#       * Simple "alert" flags for unusually large volume/returns
#       * A transparent OptionsPlay-style composite score with decomposed components:
#           - Trend
#           - Volatility
#           - Relative Strength
#           - Flow (options-based, if available)
#           - Seasonality
#           - Risk/Reward shape
#   - Writes EVERYTHING to /dev/shm:
#       /dev/shm/MICROSTRUCTURE_AND_SCORE/<TICKER>/<YYYY-MM-DD>/*.csv, *.html
#   - Visualizes with Plotly (no external servers, pure HTML files)
#
# Usage:
#   python3 microstructure_and_optionsplay_score_dashboard.v1.py TICKER
#
# Notes:
#   - Only requires a ticker argument; everything else uses sane defaults.
#   - Gracefully degrades if options data is unavailable (Flow component ~ neutral).
#   - CSVs are designed to be convenient for STEGO-style downstream pipelines.

import os
import sys
import argparse
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Import user-provided loaders (DO NOT MODIFY THESE FILES)
# ---------------------------------------------------------------------------
try:
    import data_retrieval
except ImportError as e:
    print("FATAL: Could not import data_retrieval. Make sure it is in PYTHONPATH.")
    raise

try:
    import options_data_retrieval
except ImportError as e:
    print("WARNING: Could not import options_data_retrieval. Options-based features will be disabled.")
    options_data_retrieval = None  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_ROOT = "/dev/shm/MICROSTRUCTURE_AND_SCORE"


def ensure_output_dir(ticker: str) -> str:
    """Create dated output directory for this script."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(OUTPUT_ROOT, ticker.upper(), today_str)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Simple stdout logger with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} [INFO] {msg}")


def safe_pct_change(series: pd.Series) -> pd.Series:
    return series.astype(float).pct_change().replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of a series."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z


def last_valid(series: pd.Series) -> Optional[float]:
    """Return last non-NaN value or None."""
    if series is None or series.empty:
        return None
    val = series.dropna()
    if val.empty:
        return None
    return float(val.iloc[-1])


# ---------------------------------------------------------------------------
# Microstructure-style daily features & alerts
# ---------------------------------------------------------------------------

def compute_daily_microstructure(df: pd.DataFrame, lookback_vol: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily microstructure-like features from OHLCV.

    Returns:
        features_df: per-day features (volume_z, range_pct, return_z, etc.)
        alerts_df  : subset of rows where flags are triggered
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Basic returns
    df["return"] = safe_pct_change(df["Close"])

    # Intraday range relative to close
    df["range"] = df["High"] - df["Low"]
    df["range_pct"] = df["range"] / df["Close"].replace(0, np.nan)

    # Volume z-score
    if "Volume" in df.columns:
        df["volume_z"] = rolling_zscore(df["Volume"].astype(float), lookback_vol)
    else:
        df["volume_z"] = np.nan

    # Return z-score
    df["return_z"] = rolling_zscore(df["return"], lookback_vol)

    # Range z-score
    df["range_z"] = rolling_zscore(df["range_pct"], lookback_vol)

    # Simple "microstructure" proxies
    # Spread proxy: normalized range
    df["spread_proxy"] = df["range_pct"]

    # Realized volatility (20d)
    df["rv_20"] = df["return"].rolling(20).std() * np.sqrt(252.0)

    # Basic alerts
    volume_spike_thr = 3.0
    return_spike_thr = 3.0
    range_spike_thr = 3.0

    df["alert_volume_spike"] = (df["volume_z"].abs() >= volume_spike_thr).astype(int)
    df["alert_return_spike"] = (df["return_z"].abs() >= return_spike_thr).astype(int)
    df["alert_range_spike"] = (df["range_z"].abs() >= range_spike_thr).astype(int)

    alerts_mask = (df["alert_volume_spike"] == 1) | (df["alert_return_spike"] == 1) | (df["alert_range_spike"] == 1)
    alerts_df = df.loc[alerts_mask].copy()

    features_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "return", "range", "range_pct",
        "volume_z", "return_z", "range_z",
        "spread_proxy", "rv_20",
        "alert_volume_spike", "alert_return_spike", "alert_range_spike",
    ]
    features_cols = [c for c in features_cols if c in df.columns]
    features_df = df[features_cols].copy()

    return features_df, alerts_df


# ---------------------------------------------------------------------------
# Seasonality Profiles (month-of-year, day-of-week) for volume/returns
# ---------------------------------------------------------------------------

def compute_seasonality_profiles(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute basic seasonality profiles:
        - By month of year (avg return, avg volume)
        - By day of week

    Returns dict of DataFrames.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["return"] = safe_pct_change(df["Close"])

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    df["month"] = df.index.month
    df["dow"] = df.index.dayofweek  # 0=Monday

    # By month
    month_prof = (
        df.groupby("month")
        .agg(
            avg_return=("return", "mean"),
            std_return=("return", "std"),
            avg_volume=("Volume", "mean"),
            count_days=("return", "count"),
        )
        .reset_index()
        .sort_values("month")
    )

    # By day-of-week
    dow_prof = (
        df.groupby("dow")
        .agg(
            avg_return=("return", "mean"),
            std_return=("return", "std"),
            avg_volume=("Volume", "mean"),
            count_days=("return", "count"),
        )
        .reset_index()
        .sort_values("dow")
    )

    return {
        "seasonality_by_month": month_prof,
        "seasonality_by_dow": dow_prof,
    }


# ---------------------------------------------------------------------------
# Options helpers
# ---------------------------------------------------------------------------

def load_current_option_snapshot(
    ticker: str,
    asof: Optional[date] = None,
    max_expiries: int = 6,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Load (or download) up to max_expiries option chains for ticker from options_data_retrieval.

    Returns:
        options_df: concatenated chain for selected expirations (may be empty)
        spot      : last close spot if available, else None
    """
    if options_data_retrieval is None:
        return pd.DataFrame(), None

    # Load underlying spot from data_retrieval
    log(f"Loading underlying OHLCV for {ticker} via data_retrieval.load_or_download_ticker(period='max')...")
    df_price = data_retrieval.load_or_download_ticker(ticker, period="max")
    if df_price.empty:
        log(f"WARNING: No price data found for {ticker}; cannot compute spot.")
        spot = None
    else:
        spot = float(df_price["Close"].dropna().iloc[-1])

    if asof is None:
        asof = datetime.now().date()

    # List remote expirations
    try:
        log(f"Querying available remote expirations via options_data_retrieval.get_available_remote_expirations for {ticker}...")
        exps = options_data_retrieval.get_available_remote_expirations(ticker)
    except Exception as e:
        log(f"WARNING: Failed to get remote expirations for {ticker}: {e}")
        return pd.DataFrame(), spot

    if not exps:
        log("WARNING: No remote expirations reported; options snapshot will be empty.")
        return pd.DataFrame(), spot

    # Choose expirations: prefer ~7-45 days; otherwise first max_expiries
    days = np.array([(exp.date() - asof).days for exp in exps])
    idx_valid = np.where((days >= 7) & (days <= 45))[0]

    if len(idx_valid) == 0:
        # Fallback: just first max_expiries
        selected_exps = exps[:max_expiries]
        log(f"No expirations in [7,45] days; using first {len(selected_exps)} expirations: {selected_exps}")
    else:
        exps_valid = [exps[i] for i in idx_valid]
        exps_valid = sorted(exps_valid, key=lambda x: abs((x.date() - asof).days - 30))
        selected_exps = exps_valid[:max_expiries]
        log(f"Using up to {len(selected_exps)} near-term expirations: {selected_exps}")

    try:
        options_data_retrieval.ensure_option_chains_cached(
            ticker=ticker,
            expirations=selected_exps,
            source="yfinance",
            force_refresh=False,
        )
    except Exception as e:
        log(f"WARNING: ensure_option_chains_cached failed: {e}")

    try:
        chains = options_data_retrieval.load_all_cached_option_chains(
            ticker=ticker,
            source="yfinance",
            expirations=selected_exps,
        )
    except Exception as e:
        log(f"WARNING: load_all_cached_option_chains failed: {e}")
        chains = pd.DataFrame()

    if chains is None:
        chains = pd.DataFrame()

    if chains.empty:
        log("WARNING: Options chains loaded but DataFrame is empty.")
        return chains, spot

    # Normalize basic fields
    # Expect columns like: 'type', 'expiration', 'strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'openInterest', 'volume', ...
    for col in ["bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume", "strike"]:
        if col in chains.columns:
            chains[col] = pd.to_numeric(chains[col], errors="coerce")

    return chains, spot


def compute_options_features(options_df: pd.DataFrame, spot: Optional[float], asof: date) -> Dict[str, Any]:
    """
    Compute cross-sectional options features from a snapshot (no history).

    Returns a dict of summary statistics suitable for building a Flow / Volatility component.
    """
    features: Dict[str, Any] = {
        "has_options": False,
        "asof": asof.isoformat(),
        "atm_iv_30d": np.nan,
        "atm_iv_30d_percentile_in_surface": np.nan,
        "total_call_oi": np.nan,
        "total_put_oi": np.nan,
        "put_call_oi_ratio": np.nan,
        "total_call_volume": np.nan,
        "total_put_volume": np.nan,
        "put_call_volume_ratio": np.nan,
    }

    if options_df is None or options_df.empty:
        return features

    if spot is None or spot <= 0:
        return features

    df = options_df.copy()

    # Ensure type + expiration
    if "type" not in df.columns or "expiration" not in df.columns:
        return features

    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    df["days_to_expiry"] = (df["expiration"] - asof).apply(lambda x: getattr(x, "days", np.nan))

    # Clean up
    for col in ["strike", "impliedVolatility"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["moneyness_log"] = np.log(df["strike"] / spot)

    # ATM 30d IV approx
    df_valid = df.dropna(subset=["impliedVolatility", "strike", "days_to_expiry"])
    if not df_valid.empty:
        df_30 = df_valid.loc[df_valid["days_to_expiry"].between(10, 60)]
        if df_30.empty:
            df_30 = df_valid

        # Close to 30 days
        df_30 = df_30.iloc[
            (df_30["days_to_expiry"] - 30).abs().sort_values().index
        ].copy()

        # Keep some subset near 30 days
        df_30 = df_30.head(200)

        # ATM band
        atm_band = df_30.loc[df_30["moneyness_log"].abs() <= 0.10]
        if not atm_band.empty:
            atm_iv = float(atm_band["impliedVolatility"].mean())
            features["atm_iv_30d"] = atm_iv

            # Percentile in full surface
            all_iv = df_valid["impliedVolatility"].dropna()
            if not all_iv.empty:
                pct = float((all_iv < atm_iv).mean() * 100.0)
                features["atm_iv_30d_percentile_in_surface"] = pct

    # Flow: OI and volume
    if "openInterest" in df.columns:
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Total OI
    if "openInterest" in df.columns:
        calls_oi = df.loc[df["type"] == "call", "openInterest"].sum(skipna=True)
        puts_oi = df.loc[df["type"] == "put", "openInterest"].sum(skipna=True)
        features["total_call_oi"] = float(calls_oi)
        features["total_put_oi"] = float(puts_oi)
        denom_oi = calls_oi + puts_oi
        if denom_oi > 0:
            features["put_call_oi_ratio"] = float(puts_oi / denom_oi)

    # Total volume
    if "volume" in df.columns:
        calls_vol = df.loc[df["type"] == "call", "volume"].sum(skipna=True)
        puts_vol = df.loc[df["type"] == "put", "volume"].sum(skipna=True)
        features["total_call_volume"] = float(calls_vol)
        features["total_put_volume"] = float(puts_vol)
        denom_vol = calls_vol + puts_vol
        if denom_vol > 0:
            features["put_call_volume_ratio"] = float(puts_vol / denom_vol)

    features["has_options"] = True
    return features


# ---------------------------------------------------------------------------
# Composite Score Components (OptionsPlay-style)
# ---------------------------------------------------------------------------

def compute_trend_component(price_df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    Trend component based on slopes of log price over multiple windows and
    price vs moving averages.

    Returns (score_0_100, detail_dict).
    """
    detail: Dict[str, Any] = {}
    df = price_df.copy()
    df = df.dropna(subset=["Close"])
    if df.empty or len(df) < 50:
        detail["note"] = "Insufficient data for robust trend component."
        return 50.0, detail

    df["log_close"] = np.log(df["Close"])

    def window_slope(series: pd.Series, window: int) -> Optional[float]:
        if len(series) < window:
            return None
        y = series.iloc[-window:]
        x = np.arange(len(y), dtype=float)
        # polyfit degree 1
        m, b = np.polyfit(x, y.values, 1)
        return float(m)

    slopes = {}
    for w in [20, 50, 200]:
        s = window_slope(df["log_close"], w)
        slopes[f"slope_{w}"] = s

    detail["slopes"] = slopes

    # Normalize slopes by an arbitrary scale (per-day slope)
    valid_slopes = [v for v in slopes.values() if v is not None]
    if not valid_slopes:
        detail["note"] = "No valid slopes; neutral trend."
        return 50.0, detail

    # Scale slopes to a bounded 0-100 using hyperbolic function
    # You can tune this scale; smaller means more sensitivity.
    scale = 0.001
    sub_scores = []
    for k, s in slopes.items():
        if s is None:
            continue
        # Map slope to (-1,1) via tanh then to 0-100
        s_norm = np.tanh(s / scale)
        score = (s_norm + 1.0) * 50.0
        sub_scores.append(score)

    # Price vs moving averages
    for w in [20, 50, 200]:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
    last_close = df["Close"].iloc[-1]
    ma_scores = []
    for w in [20, 50, 200]:
        sma = df[f"sma_{w}"].iloc[-1]
        if np.isnan(sma) or sma == 0:
            continue
        diff = (last_close - sma) / sma
        # reward being above MA by up to ~20%
        diff_clipped = np.clip(diff, -0.2, 0.2)
        score = (diff_clipped / 0.2 + 1.0) * 50.0
        ma_scores.append(score)

    detail["ma_scores"] = {"sma_scores": ma_scores, "last_close": float(last_close)}

    all_subscores = sub_scores + ma_scores
    if not all_subscores:
        overall = 50.0
    else:
        overall = float(np.mean(all_subscores))

    detail["overall_trend_score"] = overall
    return overall, detail


def compute_vol_component(price_df: pd.DataFrame, options_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Volatility component: realized vs implied (if available).
    """
    detail: Dict[str, Any] = {}
    df = price_df.copy()
    df = df.dropna(subset=["Close"])
    if df.empty or len(df) < 40:
        detail["note"] = "Insufficient data for volatility component."
        return 50.0, detail

    df["return"] = safe_pct_change(df["Close"])
    rv_20 = df["return"].rolling(20).std() * np.sqrt(252.0)
    rv_last = last_valid(rv_20)
    detail["rv_20_last"] = rv_last

    iv_atm_30d = options_features.get("atm_iv_30d", np.nan)
    iv_pctile_surface = options_features.get("atm_iv_30d_percentile_in_surface", np.nan)
    detail["iv_atm_30d"] = iv_atm_30d
    detail["iv_atm_30d_percentile_in_surface"] = iv_pctile_surface

    # If no options, base only on realized vol (higher vol -> lower score)
    if np.isnan(iv_atm_30d) or rv_last is None or np.isnan(rv_last):
        # Use realized only: treat moderate vol as best
        rv = rv_last if rv_last is not None else 0.2
        rv_clipped = np.clip(rv, 0.05, 0.60)  # 5% to 60%
        # Map 5% -> 80, 20% -> 60, 40% -> 40, 60% -> 20 roughly
        # Use a simple linear descending mapping
        score = 80.0 - (rv_clipped - 0.05) * (60.0 / (0.60 - 0.05))
        score = float(np.clip(score, 0.0, 100.0))
        detail["note"] = "No options IV; volatility score based on realized vol only."
        detail["score_realized_only"] = score
        return score, detail

    # Use ratio RV/IV: if RV << IV, potential long-vol edge; if RV >> IV, potential mean-revert/short-vol
    ratio = rv_last / iv_atm_30d if iv_atm_30d > 0 else np.nan
    detail["rv_iv_ratio"] = ratio

    if np.isnan(ratio):
        score = 50.0
    else:
        # Ideal region near ratio ~ 0.6-0.8 (IV priced rich vs realized)
        # Map ratio to 0-100 with some shape
        # If ratio ~0.7 => ~80 score; ratio ~1.5 => ~40; ratio <<0.3 or >>2 => lower.
        ratio_clipped = np.clip(ratio, 0.1, 3.0)
        # Use a simple quadratic-like penalty
        score = 100.0 - 60.0 * (ratio_clipped - 0.7) ** 2
        score = float(np.clip(score, 0.0, 100.0))

    # Adjust by IV percentile: low percentile -> +, high -> -
    if not np.isnan(iv_pctile_surface):
        # If IV is in low percentile (cheap vs surface), add some points
        # Map 0% -> +10, 50% -> 0, 100% -> -10
        adj = (50.0 - iv_pctile_surface) / 50.0 * 10.0
        score = float(np.clip(score + adj, 0.0, 100.0))
        detail["iv_pctile_adjustment"] = adj

    detail["overall_vol_score"] = score
    return score, detail


def compute_relative_strength_component(
    ticker_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback_days: int = 60,
) -> Tuple[float, Dict[str, Any]]:
    """
    Relative strength vs benchmark (e.g., SPY) over last N days.
    """
    detail: Dict[str, Any] = {}
    df_t = ticker_df.dropna(subset=["Close"]).copy()
    df_b = benchmark_df.dropna(subset=["Close"]).copy()

    if df_t.empty or df_b.empty:
        detail["note"] = "Missing data for RS; neutral."
        return 50.0, detail

    # Align by date
    df_t, df_b = df_t.align(df_b, join="inner", axis=0)
    if df_t.empty or df_b.empty:
        detail["note"] = "No overlapping dates for RS; neutral."
        return 50.0, detail

    df_t["ret"] = safe_pct_change(df_t["Close"])
    df_b["ret"] = safe_pct_change(df_b["Close"])

    df_t["cum_ret"] = (1.0 + df_t["ret"]).cumprod()
    df_b["cum_ret"] = (1.0 + df_b["ret"]).cumprod()

    if len(df_t) < lookback_days:
        lookback_days = len(df_t)

    df_t_lb = df_t.iloc[-lookback_days:]
    df_b_lb = df_b.iloc[-lookback_days:]

    if df_t_lb.empty or df_b_lb.empty:
        detail["note"] = "Too few days for RS; neutral."
        return 50.0, detail

    ticker_start = df_t_lb["cum_ret"].iloc[0]
    ticker_end = df_t_lb["cum_ret"].iloc[-1]
    bench_start = df_b_lb["cum_ret"].iloc[0]
    bench_end = df_b_lb["cum_ret"].iloc[-1]

    rs = (ticker_end / ticker_start) / (bench_end / bench_start + 1e-12)
    detail["rs_ratio"] = float(rs)

    # Map RS>1 to >50, RS<1 to <50, saturate extremes
    rs_clipped = np.clip(rs, 0.5, 1.5)
    score = (rs_clipped - 0.5) / (1.5 - 0.5) * 100.0  # 0.5->0, 1.5->100
    score = float(np.clip(score, 0.0, 100.0))
    detail["overall_rs_score"] = score
    return score, detail


def compute_flow_component(options_features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Flow component from options Features:
        - put_call_oi_ratio
        - put_call_volume_ratio
        - total_call/put notionals (notional proxies)
    """
    detail: Dict[str, Any] = {}
    if not options_features.get("has_options", False):
        detail["note"] = "No options features available; flow component neutral."
        return 50.0, detail

    pcoi = options_features.get("put_call_oi_ratio", np.nan)
    pcvol = options_features.get("put_call_volume_ratio", np.nan)
    tcoi = options_features.get("total_call_oi", np.nan)
    tpoi = options_features.get("total_put_oi", np.nan)
    tcvol = options_features.get("total_call_volume", np.nan)
    tpvol = options_features.get("total_put_volume", np.nan)

    detail["put_call_oi_ratio"] = pcoi
    detail["put_call_volume_ratio"] = pcvol
    detail["total_call_oi"] = tcoi
    detail["total_put_oi"] = tpoi
    detail["total_call_volume"] = tcvol
    detail["total_put_volume"] = tpvol

    subscores = []

    # Interpret OI ratio: slight call > put (ratio<0.5) -> more bullish; heavy put > call (ratio>0.7) -> more bearish
    if not np.isnan(pcoi):
        # Map ratio [0.1, 0.9] => score [80, 20] roughly
        r = float(np.clip(pcoi, 0.1, 0.9))
        score_oi = 80.0 - (r - 0.1) * (60.0 / (0.9 - 0.1))
        score_oi = float(np.clip(score_oi, 0.0, 100.0))
        subscores.append(score_oi)
        detail["score_oi_ratio"] = score_oi

    # Volume ratio: near-term flow; similar mapping but lower weight
    if not np.isnan(pcvol):
        r = float(np.clip(pcvol, 0.1, 0.9))
        score_vol = 80.0 - (r - 0.1) * (60.0 / (0.9 - 0.1))
        score_vol = float(np.clip(score_vol, 0.0, 100.0))
        subscores.append(score_vol)
        detail["score_volume_ratio"] = score_vol

    # Magnitude: large overall OI & volume indicates strong positioning / focus. Use as a mild boost.
    magnitude_boost = 0.0
    if not np.isnan(tcoi) and not np.isnan(tpoi):
        total_oi = tcoi + tpoi
        # Map [0, 100k] to [0, 10] approx; clip
        total_oi_clipped = np.clip(total_oi, 0.0, 100000.0)
        magnitude_boost += float(total_oi_clipped / 100000.0 * 10.0)

    if not np.isnan(tcvol) and not np.isnan(tpvol):
        total_vol = tcvol + tpvol
        total_vol_clipped = np.clip(total_vol, 0.0, 50000.0)
        magnitude_boost += float(total_vol_clipped / 50000.0 * 10.0)

    detail["magnitude_boost"] = magnitude_boost

    if not subscores:
        base_score = 50.0
    else:
        base_score = float(np.mean(subscores))

    overall = float(np.clip(base_score + magnitude_boost, 0.0, 100.0))
    detail["overall_flow_score"] = overall
    return overall, detail


def compute_seasonality_component(
    price_df: pd.DataFrame,
    seasonality_profiles: Dict[str, pd.DataFrame],
) -> Tuple[float, Dict[str, Any]]:
    """
    Seasonality component based on month-of-year mean returns vs all months.
    """
    detail: Dict[str, Any] = {}
    df = price_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.empty:
        detail["note"] = "No data for seasonality; neutral."
        return 50.0, detail

    df["return"] = safe_pct_change(df["Close"])
    df["month"] = df.index.month

    month_prof = seasonality_profiles.get("seasonality_by_month")
    if month_prof is None or month_prof.empty:
        detail["note"] = "No month profile; neutral."
        return 50.0, detail

    current_month = int(df.index[-1].month)
    mp = month_prof.set_index("month")
    if current_month not in mp.index:
        detail["note"] = "Current month missing from profile; neutral."
        return 50.0, detail

    avg_return_all = float(mp["avg_return"].mean())
    avg_return_month = float(mp.loc[current_month, "avg_return"])
    detail["avg_return_all_months"] = avg_return_all
    detail["avg_return_current_month"] = avg_return_month
    detail["current_month"] = current_month

    # If current-month mean return is > overall, positive factor; else negative.
    diff = avg_return_month - avg_return_all
    # Clip diff to [-1%, +1%]
    diff_clipped = np.clip(diff, -0.01, 0.01)
    score = 50.0 + diff_clipped / 0.01 * 20.0  # -0.01 -> 30, +0.01 -> 70
    score = float(np.clip(score, 0.0, 100.0))
    detail["overall_seasonality_score"] = score
    return score, detail


def compute_risk_reward_component(price_df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    Risk/reward component based on ATR and distance to recent high/low.
    """
    detail: Dict[str, Any] = {}
    df = price_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.empty or len(df) < 30:
        detail["note"] = "Insufficient data for risk/reward; neutral."
        return 50.0, detail

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df["ATR_14"] = atr_14

    # Recent high/low windows
    win_high = 20
    win_low = 20
    rolling_high = close.rolling(win_high).max()
    rolling_low = close.rolling(win_low).min()

    last_close = float(close.iloc[-1])
    last_atr = last_valid(atr_14)
    last_high = last_valid(rolling_high)
    last_low = last_valid(rolling_low)

    detail["last_close"] = last_close
    detail["last_atr_14"] = last_atr
    detail["rolling_high_20"] = last_high
    detail["rolling_low_20"] = last_low

    if last_atr is None or np.isnan(last_atr) or last_atr == 0 or last_high is None or last_low is None:
        detail["note"] = "Missing ATR or highs/lows; neutral."
        return 50.0, detail

    # Reward: distance to recent high in ATRs (potential upside)
    reward_atr = (last_high - last_close) / last_atr
    # Risk: distance to recent low in ATRs (downside)
    risk_atr = (last_close - last_low) / last_atr

    detail["reward_atr"] = float(reward_atr)
    detail["risk_atr"] = float(risk_atr)

    # Risk-reward ratio: reward_atr / risk_atr (clip)
    if risk_atr <= 0:
        rr_ratio = np.nan
    else:
        rr_ratio = reward_atr / risk_atr

    detail["rr_ratio"] = float(rr_ratio) if not np.isnan(rr_ratio) else np.nan

    # Score heuristics:
    #   - prefer rr_ratio >= 1.5
    #   - prefer reward_atr between 1 and 5 ATR
    #   - prefer risk_atr <= 2 ATR
    subscores = []

    if not np.isnan(rr_ratio):
        rr_clipped = np.clip(rr_ratio, 0.25, 4.0)
        score_rr = (rr_clipped - 0.25) / (4.0 - 0.25) * 100.0
        score_rr = float(np.clip(score_rr, 0.0, 100.0))
        subscores.append(score_rr)
        detail["score_rr_ratio"] = score_rr

    reward_clipped = np.clip(reward_atr, 0.0, 6.0)
    # Reward 1-4 ATR best; map 0->20, 1->60, 4->80, 6->60
    if reward_clipped <= 1.0:
        score_reward = 20.0 + reward_clipped * 40.0
    elif reward_clipped <= 4.0:
        score_reward = 60.0 + (reward_clipped - 1.0) * (20.0 / 3.0)
    else:
        score_reward = 80.0 - (reward_clipped - 4.0) * 10.0
    score_reward = float(np.clip(score_reward, 0.0, 100.0))
    subscores.append(score_reward)
    detail["score_reward_atr"] = score_reward

    risk_clipped = np.clip(risk_atr, 0.0, 4.0)
    # Lower risk is better: 0->90, 1->70, 2->50, 4->30
    score_risk = 90.0 - risk_clipped * 20.0
    score_risk = float(np.clip(score_risk, 0.0, 100.0))
    subscores.append(score_risk)
    detail["score_risk_atr"] = score_risk

    if not subscores:
        overall = 50.0
    else:
        overall = float(np.mean(subscores))

    detail["overall_risk_reward_score"] = overall
    return overall, detail


def compute_composite_score(
    ticker: str,
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    options_features: Dict[str, Any],
    seasonality_profiles: Dict[str, pd.DataFrame],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the full composite score and all component details.

    Weights:
        Trend          0.25
        Volatility     0.15
        Relative Str   0.20
        Flow           0.25
        Seasonality    0.05
        Risk/Reward    0.10
    """
    components: Dict[str, Any] = {}

    trend_score, trend_detail = compute_trend_component(price_df)
    vol_score, vol_detail = compute_vol_component(price_df, options_features)
    rs_score, rs_detail = compute_relative_strength_component(price_df, benchmark_df)
    flow_score, flow_detail = compute_flow_component(options_features)
    seas_score, seas_detail = compute_seasonality_component(price_df, seasonality_profiles)
    rr_score, rr_detail = compute_risk_reward_component(price_df)

    components["trend"] = {"score": trend_score, "detail": trend_detail}
    components["volatility"] = {"score": vol_score, "detail": vol_detail}
    components["relative_strength"] = {"score": rs_score, "detail": rs_detail}
    components["flow"] = {"score": flow_score, "detail": flow_detail}
    components["seasonality"] = {"score": seas_score, "detail": seas_detail}
    components["risk_reward"] = {"score": rr_score, "detail": rr_detail}

    weights = {
        "trend": 0.25,
        "volatility": 0.15,
        "relative_strength": 0.20,
        "flow": 0.25,
        "seasonality": 0.05,
        "risk_reward": 0.10,
    }

    comp_score = 0.0
    for name, comp in components.items():
        w = weights.get(name, 0.0)
        comp_score += w * comp["score"]

    composite_score = float(np.clip(comp_score, 0.0, 100.0))

    summary = {
        "ticker": ticker.upper(),
        "asof": price_df.index[-1].strftime("%Y-%m-%d") if not price_df.empty else "",
        "composite_score": composite_score,
        "weights": weights,
        "components": components,
    }

    return composite_score, summary


# ---------------------------------------------------------------------------
# Plotly Visualizations
# ---------------------------------------------------------------------------

def build_microstructure_figure(
    features_df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Candles + volume + volume_z overlay for recent 180 bars."""
    if features_df.empty:
        return go.Figure()

    # Use last 180 observations
    df = features_df.copy().iloc[-180:].copy()

    fig = go.Figure()

    # Candlestick
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=f"{ticker.upper()} OHLC",
                yaxis="y1",
            )
        )

    # Volume
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.4,
            )
        )

    # Volume z-score
    if "volume_z" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["volume_z"],
                name="Volume Z",
                yaxis="y3",
                mode="lines",
            )
        )

    fig.update_layout(
        title=f"{ticker.upper()} Daily Microstructure (OHLC, Volume, Volume Z)",
        xaxis=dict(domain=[0.0, 1.0]),
        yaxis=dict(
            title="Price",
            anchor="x",
            domain=[0.35, 1.0],
        ),
        yaxis2=dict(
            title="Volume",
            anchor="x",
            domain=[0.20, 0.35],
        ),
        yaxis3=dict(
            title="Volume Z",
            anchor="x",
            domain=[0.0, 0.20],
        ),
        showlegend=True,
        margin=dict(l=60, r=20, t=60, b=40),
    )

    return fig


def build_score_gauge_figure(
    ticker: str,
    composite_score: float,
) -> go.Figure:
    """Simple gauge indicator for composite score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=composite_score,
            title={"text": f"{ticker.upper()} Composite Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
            },
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def build_component_bar_figure(
    summary: Dict[str, Any],
) -> go.Figure:
    """Bar chart of component scores."""
    comps = summary["components"]
    names = []
    scores = []
    for k, v in comps.items():
        names.append(k)
        scores.append(v["score"])

    fig = go.Figure(
        go.Bar(
            x=names,
            y=scores,
            text=[f"{s:.1f}" for s in scores],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Component Scores (0-100)",
        xaxis_title="Component",
        yaxis_title="Score",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# CSV Writers
# ---------------------------------------------------------------------------

def write_csvs(
    out_dir: str,
    ticker: str,
    micro_features: pd.DataFrame,
    micro_alerts: pd.DataFrame,
    seasonality_profiles: Dict[str, pd.DataFrame],
    options_features: Dict[str, Any],
    score_summary: Dict[str, Any],
) -> None:
    """Write all key tables to CSV for STEGO pipelines."""
    # Microstructure features
    if not micro_features.empty:
        path = os.path.join(out_dir, f"{ticker.upper()}_microstructure_features.csv")
        micro_features.to_csv(path)
        log(f"Wrote microstructure features CSV: {path}")

    # Alerts
    if not micro_alerts.empty:
        path = os.path.join(out_dir, f"{ticker.upper()}_microstructure_alerts.csv")
        micro_alerts.to_csv(path)
        log(f"Wrote microstructure alerts CSV: {path}")

    # Seasonality profiles
    for key, df in seasonality_profiles.items():
        if df is not None and not df.empty:
            path = os.path.join(out_dir, f"{ticker.upper()}_{key}.csv")
            df.to_csv(path, index=False)
            log(f"Wrote {key} CSV: {path}")

    # Options features snapshot
    if options_features:
        opt_df = pd.DataFrame([options_features])
        path = os.path.join(out_dir, f"{ticker.upper()}_options_features_snapshot.csv")
        opt_df.to_csv(path, index=False)
        log(f"Wrote options features snapshot CSV: {path}")

    # Score summary + components
    comps = score_summary["components"]
    weights = score_summary["weights"]
    asof = score_summary["asof"]
    rows = []
    for name, comp in comps.items():
        rows.append(
            {
                "ticker": ticker.upper(),
                "asof": asof,
                "component": name,
                "weight": weights.get(name, 0.0),
                "score": comp["score"],
            }
        )
    comp_df = pd.DataFrame(rows)
    comp_path = os.path.join(out_dir, f"{ticker.upper()}_composite_score_components.csv")
    comp_df.to_csv(comp_path, index=False)
    log(f"Wrote composite score components CSV: {comp_path}")

    # Headline composite score row
    head_df = pd.DataFrame(
        [
            {
                "ticker": ticker.upper(),
                "asof": asof,
                "composite_score": score_summary["composite_score"],
            }
        ]
    )
    head_path = os.path.join(out_dir, f"{ticker.upper()}_composite_score_headline.csv")
    head_df.to_csv(head_path, index=False)
    log(f"Wrote composite score headline CSV: {head_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily Microstructure & OptionsPlay-style Composite Score Dashboard"
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Underlying ticker symbol (e.g., SPY, IWM, NVDA)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    ticker = args.ticker.upper()

    out_dir = ensure_output_dir(ticker)
    log(f"Output directory: {out_dir}")

    # Load underlying OHLCV
    log(f"Loading OHLCV for {ticker} via data_retrieval.load_or_download_ticker(period='max')...")
    df_price = data_retrieval.load_or_download_ticker(ticker, period="max")
    if df_price.empty:
        log(f"FATAL: No OHLCV data for {ticker}. Exiting.")
        sys.exit(1)

    if not isinstance(df_price.index, pd.DatetimeIndex):
        df_price.index = pd.to_datetime(df_price.index)

    # Microstructure features & alerts
    log("Computing daily microstructure features & alerts...")
    micro_features, micro_alerts = compute_daily_microstructure(df_price)

    # Seasonality profiles
    log("Computing seasonality profiles (month-of-year, day-of-week)...")
    seasonality_profiles = compute_seasonality_profiles(df_price)

    # Options snapshot & features
    asof = df_price.index[-1].date()
    if options_data_retrieval is not None:
        log("Loading options snapshot and computing options features...")
        opt_df, spot = load_current_option_snapshot(ticker, asof=asof, max_expiries=6)
        options_features = compute_options_features(opt_df, spot, asof=asof)
    else:
        log("Options data retrieval module not available; building empty options_features.")
        options_features = {
            "has_options": False,
            "asof": asof.isoformat(),
        }

    # Benchmark data (SPY)
    log("Loading SPY as benchmark for relative strength...")
    spy_df = data_retrieval.load_or_download_ticker("SPY", period="max")
    if spy_df.empty:
        log("WARNING: Could not load SPY; relative strength will be neutral.")
        spy_df = df_price.copy()

    # Composite score
    log("Computing composite score & components...")
    composite_score, score_summary = compute_composite_score(
        ticker=ticker,
        price_df=df_price,
        benchmark_df=spy_df,
        options_features=options_features,
        seasonality_profiles=seasonality_profiles,
    )
    log(f"Composite score for {ticker}: {composite_score:.2f}")

    # CSV outputs
    log("Writing CSV outputs for STEGO pipelines...")
    write_csvs(
        out_dir=out_dir,
        ticker=ticker,
        micro_features=micro_features,
        micro_alerts=micro_alerts,
        seasonality_profiles=seasonality_profiles,
        options_features=options_features,
        score_summary=score_summary,
    )

    # Plotly figures
    log("Building Plotly figures...")
    fig_micro = build_microstructure_figure(micro_features, ticker)
    fig_gauge = build_score_gauge_figure(ticker, composite_score)
    fig_components = build_component_bar_figure(score_summary)

    # Save HTML figures
    micro_html_path = os.path.join(out_dir, f"{ticker.upper()}_microstructure_dashboard.html")
    gauge_html_path = os.path.join(out_dir, f"{ticker.upper()}_composite_score_gauge.html")
    comps_html_path = os.path.join(out_dir, f"{ticker.upper()}_composite_score_components.html")

    pio.write_html(fig_micro, file=micro_html_path, auto_open=False, include_plotlyjs="cdn")
    log(f"Wrote microstructure dashboard HTML: {micro_html_path}")

    pio.write_html(fig_gauge, file=gauge_html_path, auto_open=False, include_plotlyjs="cdn")
    log(f"Wrote composite score gauge HTML: {gauge_html_path}")

    pio.write_html(fig_components, file=comps_html_path, auto_open=False, include_plotlyjs="cdn")
    log(f"Wrote composite score components HTML: {comps_html_path}")

    log("Done.")


if __name__ == "__main__":
    main()

