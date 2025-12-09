#!/usr/bin/env python3
# SCRIPTNAME: ok.derby_score_model.v1.PROTOTYPE.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
SCRIPTNAME: derby_score_model.v1.py
AUTHOR:     Michael Derby (concept), ChatGPT (implementation)
FRAMEWORK:  STEGO FINANCIAL FRAMEWORK
DATE:       2025-11-26

PURPOSE
-------
Implements an "OptionsPlay-style" scoring engine with BOTH single-ticker
and multi-ticker modes, wired into your existing data_retrieval.py and
options_data_retrieval.py without modifying them.

FEATURES
--------
- Uses YOUR data_retrieval.load_or_download_ticker() for OHLCV.
- Uses YOUR options_data_retrieval.load_or_download_option_chain() for options.
- Computes a rich feature set:
  * Trend: EMA20/EMA50, slopes, cross regime.
  * Momentum: RSI(14), MACD histogram & 3-day change.
  * Relative volume: 20-day RVOL.
  * Volatility: ATR(20), HV(20).
  * Risk–reward proxy using ATR vs upside range.
- Builds a RULE-BASED 0–10 SCORE per the earlier design.
- Optionally computes IV-based factors (IV carry, implied move) from current chain
  (safe-guarded so script still works if chain structure is different).
- Trains a (time-respecting split) logistic regression model on 10D forward returns:
  * Label = 1 if 10D forward return > threshold (default 2%).
  * Uses price/volume-based factors only for robustness.
  * Calibrates probabilities if sklearn is available.
- Blends RULE SCORE and ML PROB into a final score:
  FinalScore ≈ 0.5 * RuleScore + 5 * ProbSuccess
- Writes ALL outputs to /dev/shm/OPTIONS_PLAY_SCORE/YYYY-MM-DD/<TICKER>:
  * features_and_labels.csv
  * today_score_summary.csv
  * model_metadata.txt (if ML is available)
- Generates Plotly dashboards per ticker and opens them in browser tabs:
  * Price + EMA20/EMA50
  * Indicators (RSI & RVOL)
  * RuleScore vs ML probability scatter

USAGE
-----
Single ticker:
  python3 derby_score_model.v1.py IWM

Multiple tickers:
  python3 derby_score_model.v1.py IWM SPY QQQ AAPL

Optional arguments:
  --start-date YYYY-MM-DD   (default: auto from data)
  --end-date   YYYY-MM-DD   (default: last available)
  --label-horizon N         (default: 10, in trading days)
  --label-threshold R       (default: 0.02, i.e. +2% forward return)

NOTES
-----
- This script NEVER modifies your data_retrieval or options_data_retrieval modules.
- It is designed to be robust: if sklearn is missing, it gracefully falls back
  to rule-based scores only (still writing all CSVs and charts).
"""

import argparse
import datetime
import logging
import os
import sys
import textwrap
import webbrowser

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Try to import sklearn; if not available, we degrade gracefully.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import your data loaders (DO NOT MODIFY THESE MODULES)
try:
    import data_retrieval
except ImportError as e:
    print("ERROR: Could not import data_retrieval.py. Ensure it is on PYTHONPATH.", file=sys.stderr)
    raise

try:
    import options_data_retrieval
except ImportError:
    # Options may not always be needed; keep going but log warning.
    options_data_retrieval = None


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def ensure_output_dir(root: str, ticker: str) -> str:
    """Create and return the output directory for a ticker."""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    out_dir = os.path.join(root, today_str, ticker.upper())
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def safe_logger_setup():
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


# ----------------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------------

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder RSI."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)

    avg_gain = gain_series.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=1.0 / window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average True Range (simple rolling mean)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute core technical and volatility features.
    Expects df to have columns: 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]

    # Trend
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema20_slope"] = df["ema20"].diff()
    df["ema50_slope"] = df["ema50"].diff()

    # Momentum
    df["rsi_14"] = compute_rsi(close, window=14)
    macd_line, signal_line, macd_hist = compute_macd(close, 12, 26, 9)
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_hist
    df["macd_hist_change_3"] = df["macd_hist"] - df["macd_hist"].shift(3)

    # Relative Volume
    df["rvol_20"] = volume / volume.rolling(window=20, min_periods=1).mean()

    # Volatility
    df["atr_20"] = compute_atr(df, window=20)

    log_ret = np.log(close / close.shift(1))
    df["hv20"] = log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252.0)

    # Risk–reward proxy: upside over ATR
    # Use rolling 20-day max as upside reference
    rolling_max_20 = close.rolling(window=20, min_periods=1).max()
    df["rr_atr"] = (rolling_max_20 - close) / (df["atr_20"] + 1e-12)

    return df


# ----------------------------------------------------------------------
# Rule-Based Scoring
# ----------------------------------------------------------------------

def compute_rule_score_row(row: pd.Series) -> float:
    """
    Compute rule-based score (0–10) using:
    - Trend (EMA structure + slope)
    - Momentum (RSI, MACD hist change)
    - RVOL
    - IV carry (if available)
    - Risk–reward (ATR based)
    """
    score = 0.0

    ema20 = row.get("ema20", np.nan)
    ema50 = row.get("ema50", np.nan)
    ema20_slope = row.get("ema20_slope", np.nan)
    ema50_slope = row.get("ema50_slope", np.nan)
    rsi = row.get("rsi_14", np.nan)
    macd_hist_chg = row.get("macd_hist_change_3", np.nan)
    rvol = row.get("rvol_20", np.nan)
    iv_carry = row.get("iv_carry", np.nan)
    rr_atr = row.get("rr_atr", np.nan)

    # Trend
    if not np.isnan(ema20) and not np.isnan(ema50):
        if ema20 > ema50 and (ema20_slope > 0) and (ema50_slope > 0):
            score += 2.0
        elif ema20 < ema50 and (ema20_slope < 0) and (ema50_slope < 0):
            score -= 1.0

    # Momentum
    if not np.isnan(rsi):
        if rsi > 50 + 10:
            score += 1.0
        if rsi > 80 or rsi < 20:
            score -= 1.0

    if not np.isnan(macd_hist_chg):
        if macd_hist_chg > 0:
            score += 1.0

    # RVOL
    if not np.isnan(rvol):
        if rvol >= 1.5:
            score += 1.0
        elif rvol < 0.8:
            score -= 1.0

    # IV carry (if we were able to compute it)
    if not np.isnan(iv_carry):
        if iv_carry > 0.3:
            score -= 1.0
        elif iv_carry < 0.0:
            score += 1.0

    # Risk–reward
    if not np.isnan(rr_atr):
        if rr_atr >= 2.0:
            score += 2.0
        elif rr_atr >= 1.5:
            score += 1.0

    # Floor at 0, cap at 10
    score = max(0.0, min(10.0, score))
    return score


def append_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized wrapper to compute rule_score for each row."""
    df = df.copy()
    df["rule_score"] = df.apply(compute_rule_score_row, axis=1)
    return df


# ----------------------------------------------------------------------
# Forward Returns and Labels
# ----------------------------------------------------------------------

def compute_forward_returns_and_labels(df: pd.DataFrame,
                                       horizon: int = 10,
                                       threshold: float = 0.02) -> pd.DataFrame:
    """
    Compute forward returns and binary labels:
      forward_ret_h = Close.shift(-h)/Close - 1
      label = 1 if forward_ret_h > threshold else 0
    """
    df = df.copy()
    close = df["Close"]

    fwd_ret_col = f"fwd_ret_{horizon}d"
    df[fwd_ret_col] = close.shift(-horizon) / close - 1.0

    label_col = f"label_{horizon}d_gt_{threshold:.2f}"
    df[label_col] = (df[fwd_ret_col] > threshold).astype(float)

    # The last horizon rows will have NaN labels; that's ok.
    return df


# ----------------------------------------------------------------------
# Options / IV Factor Helpers (Best-Effort, Robust)
# ----------------------------------------------------------------------

def safe_load_option_chain(ticker: str):
    """
    Best-effort wrapper to load option chain for a ticker, if options_data_retrieval is available.
    We intentionally AVOID assuming extra keyword args since past scripts hit errors.
    """
    if options_data_retrieval is None:
        logging.warning("options_data_retrieval not available; skipping option chain usage for %s.", ticker)
        return None

    try:
        # IMPORTANT: only pass ticker to avoid unexpected kwargs issues.
        chain = options_data_retrieval.load_or_download_option_chain(ticker)
        return chain
    except TypeError as e:
        logging.error("Failed to load option chain for %s due to TypeError: %s", ticker, str(e))
        return None
    except Exception as e:
        logging.error("Failed to load option chain for %s: %s", ticker, str(e))
        return None


def infer_iv30_from_chain(chain: pd.DataFrame,
                          spot: float) -> Optional[float]:
    """
    Heuristic: pick an expiration ~30 calendar days out (or nearest),
    then choose strike closest to spot for calls and take its IV.
    Chain schema is unknown, so we work in a defensive, best-effort way.
    """
    if chain is None or len(chain) == 0:
        return None

    df = chain.copy()

    # Try to standardize columns
    cols = {c.lower(): c for c in df.columns}

    # Expiration column
    exp_col = None
    for candidate in ["expiration", "expiry", "expDate", "expdate"]:
        lc = candidate.lower()
        if lc in cols:
            exp_col = cols[lc]
            break
    if exp_col is None:
        logging.warning("No recognizable expiration column in options chain.")
        return None

    # Strike column
    strike_col = None
    for candidate in ["strike", "strik"]:
        lc = candidate.lower()
        if lc in cols:
            strike_col = cols[lc]
            break
    if strike_col is None:
        logging.warning("No recognizable strike column in options chain.")
        return None

    # IV column
    iv_col = None
    for candidate in ["impliedvolatility", "iv", "implied_volatility"]:
        lc = candidate.lower()
        if lc in cols:
            iv_col = cols[lc]
            break
    if iv_col is None:
        logging.warning("No recognizable implied volatility column in options chain.")
        return None

    # Option type / right column
    right_col = None
    for candidate in ["right", "type", "optiontype"]:
        lc = candidate.lower()
        if lc in cols:
            right_col = cols[lc]
            break

    # Parse expiration into datetime
    try:
        df["_exp_dt"] = pd.to_datetime(df[exp_col]).dt.date
    except Exception:
        logging.warning("Failed to parse expiration dates in options chain.")
        return None

    today = datetime.date.today()
    df["_days_to_exp"] = (df["_exp_dt"] - today).apply(lambda d: d.days)

    # Keep expiries with positive days (out in the future)
    df = df[df["_days_to_exp"] > 0]
    if df.empty:
        logging.warning("No future expirations in chain.")
        return None

    # Choose expiry nearest to 30 days; fallback to nearest positive
    target_days = 30
    df_exp = (
        df.groupby("_exp_dt")["_days_to_exp"]
        .mean()
        .reset_index()
        .sort_values("_days_to_exp")
    )
    df_exp["dist_30"] = (df_exp["_days_to_exp"] - target_days).abs()
    best_exp = df_exp.sort_values("dist_30").iloc[0]["_exp_dt"]

    df_sel = df[df["_exp_dt"] == best_exp].copy()
    if df_sel.empty:
        logging.warning("No rows for selected expiration in chain.")
        return None

    # Prefer calls if we can detect them
    if right_col is not None:
        right_vals = df_sel[right_col].astype(str).str.upper()
        is_call = right_vals.isin(["C", "CALL", "CALLS"])
        if is_call.any():
            df_sel = df_sel[is_call]

    # Pick strike closest to spot
    try:
        df_sel["_dist_strike"] = (df_sel[strike_col] - spot).abs()
        atm_row = df_sel.sort_values("_dist_strike").iloc[0]
        iv30 = float(atm_row[iv_col])
        if iv30 <= 0 or iv30 > 5.0:  # sanity (<=500%)
            logging.warning("IV30=%s looks implausible; discarding.", iv30)
            return None
        return iv30
    except Exception as e:
        logging.warning("Failed to infer ATM IV from chain: %s", str(e))
        return None


def append_iv_factors(df: pd.DataFrame,
                      ticker: str) -> pd.DataFrame:
    """
    Append IV-based factors to the LAST ROW ONLY, using current options chain.
    - iv30: approx 30D ATM IV
    - iv_carry: (iv30 - hv20) / hv20
    - implied_move_1d: spot * iv30 / sqrt(252)
    For historical backtest we leave these NaN except the last row.
    """
    df = df.copy()
    df["iv30"] = np.nan
    df["iv_carry"] = np.nan
    df["implied_move_1d"] = np.nan

    # Load chain
    chain = safe_load_option_chain(ticker)
    if chain is None or len(chain) == 0:
        return df

    # Use last available close as spot proxy
    if "Close" not in df.columns:
        return df
    spot = float(df["Close"].iloc[-1])

    iv30 = infer_iv30_from_chain(chain, spot)
    if iv30 is None:
        return df

    idx_last = df.index[-1]

    df.loc[idx_last, "iv30"] = iv30

    hv20_last = df.loc[idx_last, "hv20"]
    if not np.isnan(hv20_last) and hv20_last != 0:
        df.loc[idx_last, "iv_carry"] = (iv30 - hv20_last) / hv20_last
    else:
        df.loc[idx_last, "iv_carry"] = np.nan

    df.loc[idx_last, "implied_move_1d"] = spot * iv30 / np.sqrt(252.0)

    return df


# ----------------------------------------------------------------------
# Model Training
# ----------------------------------------------------------------------

def train_logistic_model(df: pd.DataFrame,
                         feature_cols: List[str],
                         label_col: str) -> Tuple[Optional[object], pd.DataFrame]:
    """
    Train a logistic regression model (with optional calibration).
    Returns (model_or_None, df_with_pred_columns).
    If sklearn is not available or data is insufficient, returns (None, df).
    """
    if not SKLEARN_AVAILABLE:
        logging.warning("sklearn not available; skipping ML model training.")
        return None, df

    df = df.copy()

    # Drop rows with NaNs in features or label
    mask_valid = df[feature_cols + [label_col]].notnull().all(axis=1)
    df_train_all = df[mask_valid].copy()

    if df_train_all.empty or df_train_all[label_col].nunique() < 2:
        logging.warning("Not enough valid data or label variance for ML training.")
        return None, df

    n_total = len(df_train_all)
    if n_total < 200:
        logging.warning("Data length (%d) is quite small for robust ML.", n_total)

    # Time-based split: first 70% train, last 30% test
    split_idx = int(0.7 * n_total)
    df_train = df_train_all.iloc[:split_idx]
    df_test = df_train_all.iloc[split_idx:]

    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values
    X_test = df_test[feature_cols].values

    # Pipeline
    base_model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=200,
        n_jobs=1,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", base_model),
    ])

    try:
        pipe.fit(X_train, y_train)
    except Exception as e:
        logging.error("Error fitting logistic regression: %s", str(e))
        return None, df

    # Calibrate
    try:
        calibrated = CalibratedClassifierCV(pipe, cv=3, method="isotonic")
        calibrated.fit(X_train, y_train)
        model = calibrated
    except Exception as e:
        logging.warning("Calibration failed (%s); using uncalibrated model.", str(e))
        model = pipe

    # Predict probabilities for all valid rows
    probs = model.predict_proba(df_train_all[feature_cols].values)[:, 1]
    df_train_all["ml_prob"] = probs

    # Merge back to full df
    df["ml_prob"] = np.nan
    df.loc[df_train_all.index, "ml_prob"] = df_train_all["ml_prob"]

    return model, df


# ----------------------------------------------------------------------
# Plotly Dashboards
# ----------------------------------------------------------------------

def plot_price_ema(df: pd.DataFrame,
                   ticker: str,
                   out_dir: str) -> str:
    """Price & EMAs time series."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close",
    ))
    if "ema20" in df:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["ema20"],
            mode="lines",
            name="EMA20",
        ))
    if "ema50" in df:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["ema50"],
            mode="lines",
            name="EMA50",
        ))

    fig.update_layout(
        title=f"{ticker} - Price & EMAs",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h"),
    )

    html_path = os.path.join(out_dir, f"{ticker}_price_ema.html")
    fig.write_html(html_path)
    webbrowser.open_new_tab(f"file://{html_path}")
    return html_path


def plot_indicators(df: pd.DataFrame,
                    ticker: str,
                    out_dir: str) -> str:
    """RSI and RVOL panel."""
    fig = go.Figure()

    if "rsi_14" in df:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["rsi_14"],
            mode="lines",
            name="RSI(14)",
        ))
    if "rvol_20" in df:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["rvol_20"],
            mode="lines",
            name="RVOL(20)",
            yaxis="y2",
        ))

    fig.update_layout(
        title=f"{ticker} - RSI & RVOL",
        xaxis=dict(title="Date"),
        yaxis=dict(title="RSI", side="left"),
        yaxis2=dict(
            title="RVOL",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h"),
    )

    html_path = os.path.join(out_dir, f"{ticker}_indicators.html")
    fig.write_html(html_path)
    webbrowser.open_new_tab(f"file://{html_path}")
    return html_path


def plot_score_vs_prob(df: pd.DataFrame,
                       ticker: str,
                       out_dir: str,
                       label_col: str) -> str:
    """Scatter of rule_score vs ml_prob, colored by label."""
    if "rule_score" not in df or "ml_prob" not in df:
        return ""

    mask = df[["rule_score", "ml_prob", label_col]].notnull().all(axis=1)
    if not mask.any():
        return ""

    dfp = df[mask].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfp["rule_score"],
        y=dfp["ml_prob"],
        mode="markers",
        name="Obs",
        text=dfp.index.astype(str),
        marker=dict(
            size=8,
            color=dfp[label_col],
            colorbar=dict(title="Label"),
        ),
    ))

    fig.update_layout(
        title=f"{ticker} - Rule Score vs ML Probability",
        xaxis_title="Rule Score (0-10)",
        yaxis_title="ML Probability (Success)",
    )

    html_path = os.path.join(out_dir, f"{ticker}_score_vs_prob.html")
    fig.write_html(html_path)
    webbrowser.open_new_tab(f"file://{html_path}")
    return html_path


# ----------------------------------------------------------------------
# Per-Ticker Pipeline
# ----------------------------------------------------------------------

def process_ticker(ticker: str,
                   start_date: Optional[str],
                   end_date: Optional[str],
                   label_horizon: int,
                   label_threshold: float,
                   output_root: str) -> Optional[Dict]:
    """
    Full pipeline for a single ticker:
      1) Load OHLCV via data_retrieval
      2) Compute features
      3) Append IV factors (current)
      4) Compute rule score
      5) Compute forward returns and labels
      6) Train ML model (if available)
      7) Blend final score for latest row
      8) Save CSVs & generate Plotly charts
    Returns a dict summary for the latest row (for today_score_summary).
    """

    tkr = ticker.upper()
    out_dir = ensure_output_dir(output_root, tkr)
    logging.info("Processing %s, output -> %s", tkr, out_dir)

    # 1) Load OHLCV
    try:
        # Use period='max' to leverage your caching; then slice by date.
        ohlcv = data_retrieval.load_or_download_ticker(tkr, period="max")
    except TypeError:
        # In case your loader uses a different signature: retry with only ticker.
        ohlcv = data_retrieval.load_or_download_ticker(tkr)
    except Exception as e:
        logging.error("Failed to load OHLCV for %s: %s", tkr, str(e))
        return None

    if ohlcv is None or ohlcv.empty:
        logging.error("No OHLCV data for %s", tkr)
        return None

    # Standardize index to datetime
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        ohlcv.index = pd.to_datetime(ohlcv.index)

    # Slice by dates if provided
    if start_date:
        ohlcv = ohlcv[ohlcv.index >= pd.to_datetime(start_date)]
    if end_date:
        ohlcv = ohlcv[ohlcv.index <= pd.to_datetime(end_date)]

    if len(ohlcv) < 60:
        logging.warning("Too few rows (%d) for robust feature calc for %s.", len(ohlcv), tkr)

    # 2) Compute base features
    df = compute_base_features(ohlcv)

    # 3) Options IV factors (only affect last row)
    df = append_iv_factors(df, tkr)

    # 4) Rule-based score
    df = append_rule_scores(df)

    # 5) Forward returns & labels
    df = compute_forward_returns_and_labels(df, horizon=label_horizon, threshold=label_threshold)
    label_col = f"label_{label_horizon}d_gt_{label_threshold:.2f}"

    # 6) Train ML model
    feature_cols = [
        "ema20_slope",
        "ema50_slope",
        "rsi_14",
        "macd_hist",
        "macd_hist_change_3",
        "rvol_20",
        "atr_20",
        "hv20",
        "rr_atr",
    ]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        logging.error("Missing required feature columns %s for %s; skipping ML.", missing_features, tkr)
        model = None
    else:
        model, df = train_logistic_model(df, feature_cols, label_col)

    # 7) Final score for latest row
    idx_last = df.index[-1]
    rule_score_last = float(df.loc[idx_last, "rule_score"])
    ml_prob_last = df.loc[idx_last, "ml_prob"] if "ml_prob" in df.columns else np.nan

    if not np.isnan(ml_prob_last):
        final_score_last = 0.5 * rule_score_last + 5.0 * ml_prob_last
    else:
        final_score_last = rule_score_last

    df.loc[idx_last, "final_score"] = final_score_last

    # 8) Save outputs
    features_csv = os.path.join(out_dir, f"{tkr}_features_and_labels.csv")
    df.to_csv(features_csv)
    logging.info("Saved features & labels CSV for %s -> %s", tkr, features_csv)

    # Append today summary row CSV (one-row file for this run)
    today_summary_path = os.path.join(out_dir, f"{tkr}_today_score_summary.csv")
    today_row = df.loc[[idx_last]].copy()
    today_row.to_csv(today_summary_path)
    logging.info("Saved today summary CSV for %s -> %s", tkr, today_summary_path)

    # Model metadata
    meta_path = os.path.join(out_dir, f"{tkr}_model_metadata.txt")
    with open(meta_path, "w") as f:
        f.write("TICKER: {}\n".format(tkr))
        f.write("LABEL_HORIZON_DAYS: {}\n".format(label_horizon))
        f.write("LABEL_THRESHOLD: {:.4f}\n".format(label_threshold))
        f.write("SKLEARN_AVAILABLE: {}\n".format(SKLEARN_AVAILABLE))
        f.write("FEATURE_COLUMNS: {}\n".format(", ".join(feature_cols)))
        f.write("DATA_ROWS_TOTAL: {}\n".format(len(df)))
    logging.info("Saved model metadata for %s -> %s", tkr, meta_path)

    # Plotly dashboards
    plot_price_ema(df, tkr, out_dir)
    plot_indicators(df, tkr, out_dir)
    plot_score_vs_prob(df, tkr, out_dir, label_col)

    # Build summary dict
    summary = {
        "ticker": tkr,
        "asof": idx_last.strftime("%Y-%m-%d"),
        "close": float(df.loc[idx_last, "Close"]),
        "rule_score": float(rule_score_last),
        "ml_prob": float(ml_prob_last) if not np.isnan(ml_prob_last) else np.nan,
        "final_score": float(final_score_last),
        "iv30": float(df.loc[idx_last, "iv30"]) if "iv30" in df.columns and not np.isnan(df.loc[idx_last, "iv30"]) else np.nan,
        "iv_carry": float(df.loc[idx_last, "iv_carry"]) if "iv_carry" in df.columns and not np.isnan(df.loc[idx_last, "iv_carry"]) else np.nan,
        "implied_move_1d": float(df.loc[idx_last, "implied_move_1d"]) if "implied_move_1d" in df.columns and not np.isnan(df.loc[idx_last, "implied_move_1d"]) else np.nan,
    }
    return summary


# ----------------------------------------------------------------------
# Main / CLI
# ----------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OptionsPlay-style scoring engine (single & multi-ticker) using STEGO data loaders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbol(s), e.g. IWM or IWM SPY QQQ",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD) for backtest window.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD) for backtest window.",
    )

    parser.add_argument(
        "--label-horizon",
        type=int,
        default=10,
        help="Forward return horizon (trading days) for labels.",
    )

    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.02,
        help="Forward return threshold for success label (e.g. 0.02 = +2%%).",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="/dev/shm/OPTIONS_PLAY_SCORE",
        help="Root directory for ALL output (CSVs, HTML, metadata).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    safe_logger_setup()
    args = parse_args(argv)

    tickers = args.tickers
    start_date = args.start_date
    end_date = args.end_date
    label_horizon = args.label_horizon
    label_threshold = args.label_threshold
    output_root = args.output_root

    logging.info("Tickers: %s", ", ".join(tickers))
    logging.info("Date range: %s -> %s", start_date, end_date)
    logging.info("Label horizon: %d days, threshold: %.4f", label_horizon, label_threshold)
    logging.info("Output root: %s", output_root)

    summaries = []
    for t in tickers:
        try:
            summary = process_ticker(
                ticker=t,
                start_date=start_date,
                end_date=end_date,
                label_horizon=label_horizon,
                label_threshold=label_threshold,
                output_root=output_root,
            )
            if summary is not None:
                summaries.append(summary)
        except Exception as e:
            logging.exception("Unexpected error processing %s: %s", t, str(e))

    # If multiple tickers, write a combined summary CSV in root/date dir
    if summaries:
        df_sum = pd.DataFrame(summaries)
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        multi_dir = os.path.join(output_root, today_str)
        os.makedirs(multi_dir, exist_ok=True)
        multi_csv = os.path.join(multi_dir, "multi_ticker_today_summary.csv")
        df_sum.to_csv(multi_csv, index=False)
        logging.info("Saved multi-ticker today summary -> %s", multi_csv)
    else:
        logging.warning("No successful ticker processing; no multi-ticker summary written.")


if __name__ == "__main__":
    main()

