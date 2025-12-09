#!/usr/bin/env python3
# SCRIPTNAME: options_directional_score.calibrated.v1.py
# AUTHOR: Michael Derby
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
# DATE: November 27, 2025
#
# PURPOSE
# -------
# This is a STANDALONE script that does BOTH:
#
#   (B) Builds a simple, fully self-contained directional model that predicts
#       the probability that a stock will be UP over the next N days.
#
#   (C) Wraps that calibrated directional probability into an
#       "OptionsPlay-style" composite score (0-100) that blends:
#         - Calibrated probability of an UP move
#         - Trend / regime information (50d vs 200d MA)
#         - Short-term momentum (10-day return)
#
# It:
#   - Uses YOUR data_retrieval module (unchanged) to load OHLCV.
#   - Requires ONLY a --ticker (no trades, no CSVs).
#   - Computes features, trains a logistic regression model, calibrates via
#     isotonic regression, evaluates with Brier scores and reliability diagrams.
#   - Computes TODAY's calibrated PoP and composite score.
#   - Writes all outputs to:
#
#         /dev/shm/OPTIONS_DIRECTIONAL_SCORE/YYYY-MM-DD/TICKER/
#
#     (or derived from data_retrieval.BASE_DATA_PATH() for consistency).
#   - Creates Plotly dashboards (HTML) and auto-opens them in your browser.
#
# EXAMPLE USAGE
# -------------
#   python3 options_directional_score.calibrated.v1.py --ticker NVDA
#
#   python3 options_directional_score.calibrated.v1.py \
#       --ticker SPY \
#       --horizon-days 5 \
#       --test-fraction 0.25
#
# DEPENDENCIES
# ------------
#   pip install numpy pandas plotly scikit-learn
#
# NOTES
# -----
# - This script does NOT depend on option trades or options chains.
#   It purely learns/calibrates the probability of a positive price move.
# - The "OptionsPlay-style" score here is for the UNDERLYING ONLY, but the
#   calibrated probability and score can be fed into your options engines.

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot

# -------------------------------------------------------------------------
# Try to import sklearn (for model + isotonic calibration)
# -------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# -------------------------------------------------------------------------
# Import your canonical data_retrieval module (do NOT modify it)
# -------------------------------------------------------------------------
try:
    import data_retrieval  # type: ignore
    _HAS_DATA_RETRIEVAL = True
except Exception:
    _HAS_DATA_RETRIEVAL = False


# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


# -------------------------------------------------------------------------
# /dev/shm-style paths (respect BASE_DATA_PATH if present)
# -------------------------------------------------------------------------
def _shm_root() -> str:
    """
    Infer the /dev/shm-style root using data_retrieval if available.
    Fallback: /dev/shm.
    """
    if _HAS_DATA_RETRIEVAL:
        try:
            base_data = data_retrieval.BASE_DATA_PATH()
            # e.g. /dev/shm/data -> parent /dev/shm
            shm_root = os.path.dirname(base_data.rstrip(os.sep))
            if shm_root:
                return shm_root
        except Exception:
            pass
    return "/dev/shm"


def default_output_root() -> str:
    """
    Default root under /dev/shm for directional score outputs.
    """
    shm_root = _shm_root()
    return os.path.join(shm_root, "OPTIONS_DIRECTIONAL_SCORE")


def build_output_dir(output_root: str, ticker: str) -> str:
    """
    Build an output directory of the form:
        {output_root}/YYYY-MM-DD/{TICKER}
    """
    today = datetime.now().strftime("%Y-%m-%d")
    tkr = ticker.upper()
    outdir = os.path.join(output_root, today, tkr)
    os.makedirs(outdir, exist_ok=True)
    return outdir


# -------------------------------------------------------------------------
# Basic technical indicators (no external TA libs)
# -------------------------------------------------------------------------
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Simple RSI implementation.
    """
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / window, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _rolling_vol(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling standard deviation of returns (annualized).
    """
    daily_ret = series.pct_change()
    vol = daily_ret.rolling(window).std() * np.sqrt(252.0)
    return vol


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD line and signal line.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


# -------------------------------------------------------------------------
# Feature engineering & label construction
# -------------------------------------------------------------------------
def build_feature_matrix(
    px: pd.DataFrame,
    horizon_days: int = 5
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Given an OHLCV DataFrame with 'Close' or 'Adj Close', build:

      - X: feature matrix
      - y: binary target (1 if future return > 0 over horizon, else 0)
      - future_ret: raw future return values

    horizon_days: number of days ahead to evaluate price change.
    """
    if "Adj Close" in px.columns:
        close = px["Adj Close"].copy()
    elif "Close" in px.columns:
        close = px["Close"].copy()
    else:
        raise KeyError("Price DataFrame must contain 'Adj Close' or 'Close'.")

    # Basic returns
    ret_1d = close.pct_change(1)
    ret_5d = close.pct_change(5)
    ret_10d = close.pct_change(10)

    # Rolling volatility
    vol_20 = _rolling_vol(close, window=20)
    vol_60 = _rolling_vol(close, window=60)

    # Moving averages
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()

    # Price vs MA ratios
    px_ma20 = close / ma_20 - 1.0
    px_ma50 = close / ma_50 - 1.0
    px_ma200 = close / ma_200 - 1.0

    # MACD
    macd, macd_signal = _macd(close)

    # RSI
    rsi_14 = _rsi(close, window=14)

    # Future returns & binary target
    future_close = close.shift(-horizon_days)
    future_ret = future_close / close - 1.0
    y = (future_ret > 0.0).astype(int)

    # Build feature DataFrame
    features = pd.DataFrame(
        {
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "vol_20": vol_20,
            "vol_60": vol_60,
            "px_ma20": px_ma20,
            "px_ma50": px_ma50,
            "px_ma200": px_ma200,
            "ma20_slope": ma_20.pct_change(5),
            "ma50_slope": ma_50.pct_change(5),
            "ma200_slope": ma_200.pct_change(5),
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd - macd_signal,
            "rsi_14": rsi_14,
        },
        index=close.index,
    )

    # Drop rows where we don't have full features or labels
    df = pd.concat([features, y.rename("y"), future_ret.rename("future_ret")], axis=1)
    df = df.dropna()

    X = df[features.columns].copy()
    y_clean = df["y"].astype(int)
    future_ret_clean = df["future_ret"]

    return X, y_clean, future_ret_clean


# -------------------------------------------------------------------------
# Calibration helpers
# -------------------------------------------------------------------------
def build_calibration_bins(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build a calibration table via quantile-based binning.

    Returns a DataFrame with columns:
        - bin_index
        - mean_pred
        - empirical
        - count
    """
    df = pd.DataFrame({"y": y_true, "p": p_pred})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "p"])
    df = df[(df["p"] >= 0.0) & (df["p"] <= 1.0)]

    if df.empty:
        raise ValueError("No valid rows after cleaning for calibration bins.")

    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except Exception:
        df["bin"] = pd.cut(df["p"], bins=n_bins)

    grouped = df.groupby("bin", observed=True)

    rows = []
    for idx, (interval, g) in enumerate(grouped, start=1):
        if g.empty:
            continue
        mean_pred = float(g["p"].mean())
        empirical = float(g["y"].mean())
        count = int(len(g))
        rows.append(
            {
                "bin_index": idx,
                "mean_pred": mean_pred,
                "empirical": empirical,
                "count": count,
            }
        )

    calib_df = pd.DataFrame(rows)
    calib_df.sort_values("mean_pred", inplace=True)
    calib_df.reset_index(drop=True, inplace=True)
    return calib_df


def fit_logistic_and_isotonic(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[LogisticRegression, IsotonicRegression]:
    """
    Fit a logistic regression model and an isotonic calibration on its
    predicted probabilities.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        )

    # Logistic model for raw probabilities
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        n_jobs=1,
    )
    clf.fit(X_train, y_train)

    p_train_raw = clf.predict_proba(X_train)[:, 1]

    # Isotonic regression for calibration
    iso = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    )
    iso.fit(p_train_raw, y_train)

    return clf, iso


# -------------------------------------------------------------------------
# OptionsPlay-style score from calibrated probability + trend/momentum
# -------------------------------------------------------------------------
def compute_trend_component(
    ma_50: float,
    ma_200: float,
    max_ratio: float = 0.10,
) -> float:
    """
    Trend component in [0, 100].

    - Positive when MA50 > MA200.
    - 0 when MA50 is much lower than MA200.
    - 100 when MA50 is much higher than MA200.
    """
    if ma_50 is None or ma_200 is None or ma_200 == 0:
        return 50.0

    ratio = (ma_50 / ma_200) - 1.0  # e.g. +0.05 = 5% above
    # Clip ratio into [-max_ratio, +max_ratio]
    ratio = max(-max_ratio, min(max_ratio, ratio))
    # Map [-max_ratio, +max_ratio] -> [0, 100]
    score = (ratio + max_ratio) / (2 * max_ratio) * 100.0
    return float(score)


def compute_momentum_component(
    ret_10d: float,
    max_ret: float = 0.15,
) -> float:
    """
    Momentum component in [0, 100] from 10-day return.
    """
    if ret_10d is None:
        return 50.0

    r = max(-max_ret, min(max_ret, ret_10d))
    score = (r + max_ret) / (2 * max_ret) * 100.0
    return float(score)


def compute_composite_score(
    p_calibrated: float,
    trend_score: float,
    momentum_score: float,
    w_prob: float = 0.6,
    w_trend: float = 0.25,
    w_mom: float = 0.15,
) -> float:
    """
    Composite OptionsPlay-style score in [0, 100].

    p_calibrated: probability of UP move (0-1).
    trend_score: 0-100.
    momentum_score: 0-100.
    """
    prob_score = 100.0 * p_calibrated
    score = (
        w_prob * prob_score
        + w_trend * trend_score
        + w_mom * momentum_score
    )
    return float(max(0.0, min(100.0, score)))


# -------------------------------------------------------------------------
# Plotly dashboards
# -------------------------------------------------------------------------
def plot_reliability_diagram(
    calib_raw: pd.DataFrame,
    calib_iso: pd.DataFrame,
    outdir: str,
    title_suffix: str = "",
) -> str:
    """
    Reliability diagram comparing raw vs calibrated predictions.
    """
    perfect_x = np.linspace(0.0, 1.0, 101)
    perfect_y = perfect_x

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=perfect_x,
            y=perfect_y,
            mode="lines",
            name="Perfect calibration",
            line=dict(dash="dash"),
        )
    )

    # Raw calibration bins
    fig.add_trace(
        go.Scatter(
            x=calib_raw["mean_pred"],
            y=calib_raw["empirical"],
            mode="markers+lines",
            name="Raw model",
            text=[f"n={c}" for c in calib_raw["count"]],
            hovertemplate="Mean p: %{x:.3f}<br>Empirical: %{y:.3f}<br>%{text}<extra></extra>",
        )
    )

    # Calibrated bins
    fig.add_trace(
        go.Scatter(
            x=calib_iso["mean_pred"],
            y=calib_iso["empirical"],
            mode="markers+lines",
            name="Isotonic-calibrated",
            text=[f"n={c}" for c in calib_iso["count"]],
            hovertemplate="Mean p_cal: %{x:.3f}<br>Empirical: %{y:.3f}<br>%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Reliability Diagram {title_suffix}",
        xaxis_title="Predicted probability",
        yaxis_title="Empirical frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    html_path = os.path.join(outdir, "reliability_diagram.html")
    plotly_plot(fig, filename=html_path, auto_open=True)
    return html_path


def plot_probability_histograms(
    p_raw_test: np.ndarray,
    p_cal_test: np.ndarray,
    outdir: str,
    title_suffix: str = "",
) -> str:
    """
    Histograms of raw vs calibrated probabilities on the test set.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=p_raw_test,
            name="Raw PoP (test)",
            opacity=0.6,
            nbinsx=20,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=p_cal_test,
            name="Calibrated PoP (test)",
            opacity=0.6,
            nbinsx=20,
        )
    )

    fig.update_layout(
        title=f"Distribution of Raw vs Calibrated Probabilities (Test) {title_suffix}",
        xaxis_title="Probability",
        yaxis_title="Count",
        barmode="overlay",
    )

    html_path = os.path.join(outdir, "probability_histograms.html")
    plotly_plot(fig, filename=html_path, auto_open=True)
    return html_path


def plot_edge_timeseries(
    dates: np.ndarray,
    p_cal: np.ndarray,
    horizon_days: int,
    outdir: str,
    title_suffix: str = "",
) -> str:
    """
    Plot the calibrated edge (p_cal - 0.5) over time.
    """
    edge = p_cal - 0.5
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=edge,
            mode="lines",
            name="Directional edge (p_cal - 0.5)",
        )
    )

    fig.update_layout(
        title=(
            f"Calibrated Directional Edge over Time (horizon={horizon_days}d) "
            f"{title_suffix}"
        ),
        xaxis_title="Date",
        yaxis_title="Edge (prob - 0.5)",
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=0,
                y1=0,
                line=dict(dash="dash"),
            )
        ],
    )

    html_path = os.path.join(outdir, "edge_timeseries.html")
    plotly_plot(fig, filename=html_path, auto_open=True)
    return html_path


# -------------------------------------------------------------------------
# High-level workflow
# -------------------------------------------------------------------------
def run_directional_score_workflow(
    ticker: str,
    horizon_days: int,
    test_fraction: float,
    output_root: str,
) -> None:
    """
    Main pipeline:
      - Load OHLCV via data_retrieval.
      - Build features & labels.
      - Train logistic model + isotonic calibration.
      - Evaluate on test set.
      - Compute today's PoP and OptionsPlay-style composite score.
      - Save CSVs + Plotly HTML dashboards under /dev/shm.
    """
    tkr_u = ticker.upper()
    logging.info(f"Loading OHLCV for {tkr_u} via data_retrieval...")

    if not _HAS_DATA_RETRIEVAL:
        raise ImportError(
            "data_retrieval module is not available. "
            "This script must use your existing data_retrieval.py."
        )

    # Use your canonical loader without changing it
    px = data_retrieval.load_or_download_ticker(
        tkr_u,
        period="max"
    )

    if px is None or px.empty:
        raise ValueError(f"No price data loaded for {tkr_u}.")

    # Build feature matrix and labels
    logging.info(
        f"Building features and labels for horizon={horizon_days} days..."
    )
    X, y, future_ret = build_feature_matrix(px, horizon_days=horizon_days)

    if len(X) < 250:
        logging.warning(
            f"Only {len(X)} usable rows of data; model may be unstable."
        )

    # Train/test split (time-based)
    n = len(X)
    split_idx = int(np.floor((1.0 - test_fraction) * n))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(
            f"Invalid split index {split_idx} for n={n} with test_fraction={test_fraction}."
        )

    X_train = X.iloc[:split_idx].values
    y_train = y.iloc[:split_idx].values
    X_test = X.iloc[split_idx:].values
    y_test = y.iloc[split_idx:].values

    dates_train = X.index[:split_idx]
    dates_test = X.index[split_idx:]

    logging.info(
        f"Train size: {len(X_train)}, Test size: {len(X_test)} "
        f"(split at {dates_test[0].date()})"
    )

    # Fit logistic model + isotonic calibration
    logging.info("Fitting logistic regression + isotonic calibration...")
    clf, iso = fit_logistic_and_isotonic(X_train, y_train)

    # Raw & calibrated probabilities on train/test
    p_train_raw = clf.predict_proba(X_train)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]

    p_train_cal = iso.predict(p_train_raw)
    p_test_cal = iso.predict(p_test_raw)

    # Brier scores
    brier_train_raw = brier_score_loss(y_train, p_train_raw)
    brier_train_cal = brier_score_loss(y_train, p_train_cal)
    brier_test_raw = brier_score_loss(y_test, p_test_raw)
    brier_test_cal = brier_score_loss(y_test, p_test_cal)

    logging.info(
        f"Brier (train) raw={brier_train_raw:.6f}, cal={brier_train_cal:.6f}"
    )
    logging.info(
        f"Brier (test)  raw={brier_test_raw:.6f}, cal={brier_test_cal:.6f}"
    )

    # Calibration tables (test set)
    calib_raw_test = build_calibration_bins(
        y_true=y_test,
        p_pred=p_test_raw,
        n_bins=10,
    )
    calib_iso_test = build_calibration_bins(
        y_true=y_test,
        p_pred=p_test_cal,
        n_bins=10,
    )

    # Output directory
    outdir = build_output_dir(output_root, tkr_u)
    logging.info(f"Output directory: {outdir}")

    # Save numeric outputs
    df_pred_test = pd.DataFrame(
        {
            "date": dates_test,
            "y_test": y_test,
            "p_raw": p_test_raw,
            "p_cal": p_test_cal,
        }
    ).set_index("date")

    df_pred_train = pd.DataFrame(
        {
            "date": dates_train,
            "y_train": y_train,
            "p_raw": p_train_raw,
            "p_cal": p_train_cal,
        }
    ).set_index("date")

    df_pred_train.to_csv(os.path.join(outdir, "train_predictions.csv"))
    df_pred_test.to_csv(os.path.join(outdir, "test_predictions.csv"))
    calib_raw_test.to_csv(
        os.path.join(outdir, "calibration_bins_test_raw.csv"), index=False
    )
    calib_iso_test.to_csv(
        os.path.join(outdir, "calibration_bins_test_isotonic.csv"), index=False
    )

    summary_df = pd.DataFrame(
        [
            {
                "dataset": "train",
                "metric": "brier_raw",
                "value": brier_train_raw,
            },
            {
                "dataset": "train",
                "metric": "brier_calibrated",
                "value": brier_train_cal,
            },
            {
                "dataset": "test",
                "metric": "brier_raw",
                "value": brier_test_raw,
            },
            {
                "dataset": "test",
                "metric": "brier_calibrated",
                "value": brier_test_cal,
            },
        ]
    )
    summary_df.to_csv(os.path.join(outdir, "calibration_summary.csv"), index=False)

    # Plotly dashboards
    title_suffix = f"(ticker={tkr_u}, horizon={horizon_days}d)"

    logging.info("Building Plotly reliability diagram (test set)...")
    plot_reliability_diagram(
        calib_raw=calib_raw_test,
        calib_iso=calib_iso_test,
        outdir=outdir,
        title_suffix=title_suffix,
    )

    logging.info("Building Plotly probability histograms (test set)...")
    plot_probability_histograms(
        p_raw_test=p_test_raw,
        p_cal_test=p_test_cal,
        outdir=outdir,
        title_suffix=title_suffix,
    )

    logging.info("Building Plotly edge timeseries (all data, calibrated)...")
    # Use all calibrated probabilities (concatenate train+test)
    p_all_cal = np.concatenate([p_train_cal, p_test_cal])
    dates_all = np.concatenate([dates_train.values, dates_test.values])
    plot_edge_timeseries(
        dates=dates_all,
        p_cal=p_all_cal,
        horizon_days=horizon_days,
        outdir=outdir,
        title_suffix=title_suffix,
    )

    # Compute today's PoP & composite score
    logging.info("Computing today's calibrated PoP and composite score...")

    # Last available feature row is the last usable date
    last_date = X.index[-1]
    x_today = X.iloc[-1].values.reshape(1, -1)

    p_today_raw = float(clf.predict_proba(x_today)[0, 1])
    p_today_cal = float(iso.predict([p_today_raw])[0])

    # Extract components for today's feature row
    last_row = X.iloc[-1]
    # Need 50/200-day MAs and 10-day return; recompute from px for clarity
    if "Adj Close" in px.columns:
        close = px["Adj Close"]
    else:
        close = px["Close"]

    ma_50 = close.rolling(50).mean().iloc[-1]
    ma_200 = close.rolling(200).mean().iloc[-1]
    ret_10d = close.pct_change(10).iloc[-1]

    trend_score = compute_trend_component(ma_50=ma_50, ma_200=ma_200)
    momentum_score = compute_momentum_component(ret_10d=ret_10d)
    composite_score = compute_composite_score(
        p_calibrated=p_today_cal,
        trend_score=trend_score,
        momentum_score=momentum_score,
    )

    today_summary = pd.DataFrame(
        [
            {
                "date": last_date,
                "ticker": tkr_u,
                "p_raw": p_today_raw,
                "p_calibrated": p_today_cal,
                "trend_score": trend_score,
                "momentum_score": momentum_score,
                "composite_score": composite_score,
                "horizon_days": horizon_days,
            }
        ]
    )
    today_summary_path = os.path.join(outdir, "today_directional_score.csv")
    today_summary.to_csv(today_summary_path, index=False)

    logging.info(
        f"TODAY {last_date.date()} {tkr_u}:\n"
        f"  Raw PoP (UP in {horizon_days}d)        = {p_today_raw:.4f}\n"
        f"  Calibrated PoP (UP in {horizon_days}d) = {p_today_cal:.4f}\n"
        f"  Trend score (0-100)                    = {trend_score:.1f}\n"
        f"  Momentum score (0-100)                 = {momentum_score:.1f}\n"
        f"  Composite score (0-100)                = {composite_score:.1f}\n"
        f"Saved to: {today_summary_path}"
    )

    logging.info("Directional score workflow complete.")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Directional probability + calibrated OptionsPlay-style score for a stock.\n"
            "Uses price history only (no trades), trains a simple model, calibrates\n"
            "via isotonic regression, and outputs dashboards + CSVs under /dev/shm."
        )
    )

    parser.add_argument(
        "--ticker",
        required=True,
        help="Underlying ticker symbol (e.g. SPY, QQQ, NVDA).",
    )

    parser.add_argument(
        "--horizon-days",
        type=int,
        default=5,
        help="Prediction horizon in trading days (default: 5).",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test set at the END of the history (default: 0.2).",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root directory for outputs (default: derived from /dev/shm using "
            "data_retrieval.BASE_DATA_PATH() if available, else /dev/shm/OPTIONS_DIRECTIONAL_SCORE)."
        ),
    )

    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    if not _HAS_SKLEARN:
        logging.error(
            "scikit-learn is not installed. Install it with:\n\n"
            "    pip install scikit-learn\n"
        )
        return

    output_root = args.output_root if args.output_root is not None else default_output_root()
    logging.info(f"Using output root: {output_root}")

    try:
        run_directional_score_workflow(
            ticker=args.ticker,
            horizon_days=args.horizon_days,
            test_fraction=args.test_fraction,
            output_root=output_root,
        )
    except Exception as e:
        logging.exception(f"Directional score workflow failed: {e}")


if __name__ == "__main__":
    main()

