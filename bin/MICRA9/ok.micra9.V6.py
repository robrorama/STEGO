#!/usr/bin/env python3
"""
Geometry-aware, regression-rich technical analysis dashboard.

Single-file script implementing:
- DataIngestion (disk-first, sanitized yfinance pipeline, ratio support)
- FinancialAnalysis (indicators, regressions, geometry, events, touch stats)
- DashboardRenderer (offline Plotly HTML with tabs, resize-safe)
- Option A: Legacy features restored (Smart Legends, Body Touches, Tolerance).
- Option B: Candle Physics (Magnitude, Streaks, Regimes, Gaps).
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import warnings  # for silencing pandas PerformanceWarning

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from scipy import signal, stats
import yfinance as yf
from plotly import offline
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Silence noisy fragmentation warnings from pandas; they are performance hints.
warnings.filterwarnings("ignore", category=PerformanceWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_tickers_string(tickers_str: str) -> List[str]:
    """Parse --tickers string into a list of instruments (tickers or ratios)."""
    if not tickers_str:
        return []
    parts = [p.strip().upper() for p in tickers_str.replace(" ", "").split(",") if p.strip()]
    return parts


def parse_reg_timeframes(spec: str) -> Dict[str, int]:
    """
    Parse reg-timeframes like "Long:144,Mid:60,Short:20" into a dict.
    """
    result: Dict[str, int] = {}
    if not spec:
        return result
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        name, days = item.split(":", 1)
        name = name.strip()
        try:
            result[name] = int(days.strip())
        except ValueError:
            continue
    return result


def parse_extrema_windows(spec: str) -> List[int]:
    """Parse extrema windows like '30,90,180' into a list of ints."""
    if not spec:
        return []
    out: List[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def parse_geometry_intervals(spec: str) -> List[str]:
    """Return raw intervals like ['6m','3m'] (interpretation done later)."""
    if not spec:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def interval_str_to_days(interval: str) -> int:
    """
    Convert geometry interval strings like '6m', '3m', '1y', '30d' into approx trading days.
    """
    if not interval:
        return 60
    s = interval.strip().lower()
    try:
        if s.endswith("d"):
            return max(1, int(s[:-1]))
        if s.endswith("m"):
            return max(1, int(s[:-1]) * 21)
        if s.endswith("y"):
            return max(1, int(s[:-1]) * 252)
        # plain number = days
        return max(1, int(s))
    except Exception:
        return 60


def parse_date_range_arg(date_range: Optional[str]) -> Tuple[Optional[date], Optional[date], Optional[str]]:
    """
    Interpret --date-range.
    - If None -> (None, None, None)
    - If "YYYY-MM-DD,YYYY-MM-DD" -> (start, end, None)
    - Else treat as yfinance period string -> (None, None, period)
    """
    if not date_range:
        return None, None, None

    s = date_range.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                start = datetime.strptime(parts[0], "%Y-%m-%d").date()
                end = datetime.strptime(parts[1], "%Y-%m-%d").date()
                return start, end, None
            except ValueError:
                pass
    # fall back to treating as period code
    return None, None, s


def default_period_for_lookback(lookback_years: float) -> str:
    """Choose a yfinance period code for a given lookback horizon."""
    if lookback_years <= 1:
        return "1y"
    if lookback_years <= 2:
        return "2y"
    if lookback_years <= 3:
        return "3y"
    if lookback_years <= 5:
        return "5y"
    if lookback_years <= 10:
        return "10y"
    return "max"


def years_to_period_for_backfill(lookback_years: float) -> str:
    """Map lookback_years into a longer yfinance period for shadow backfill."""
    years = max(int(math.ceil(lookback_years)), 5)
    if years <= 5:
        return "5y"
    if years <= 10:
        return "10y"
    return "max"


# ---------------------------------------------------------------------------
# DataIngestion
# ---------------------------------------------------------------------------

class DataIngestion:
    """
    Handles:
    - Disk-first pipeline with shadow backfill.
    - yfinance downloads.
    - Ratio series construction.
    - Universal sanitizer for yfinance outputs.
    """

    def __init__(
        self,
        output_dir: Path,
        lookback_years: float,
        date_start: Optional[date],
        date_end: Optional[date],
        period: Optional[str],
        logger: logging.Logger,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_years = float(lookback_years)
        self.date_start = date_start
        self.date_end = date_end
        self.period = period
        self.logger = logger

    # ----------------- public API -----------------

    def get_instrument_df(self, instrument: str) -> pd.DataFrame:
        """
        Return sanitized OHLCV DataFrame for a ticker or ratio instrument "A/B".
        """
        if "/" in instrument:
            lhs, rhs = instrument.split("/", 1)
            lhs = lhs.strip().upper()
            rhs = rhs.strip().upper()
            lhs_df = self._get_single_ticker_df(lhs)
            rhs_df = self._get_single_ticker_df(rhs)

            if lhs_df.empty or rhs_df.empty:
                self.logger.warning("Empty data for ratio %s (%s or %s empty).", instrument, lhs, rhs)
                return pd.DataFrame()

            combined = lhs_df.join(
                rhs_df,
                how="inner",
                lsuffix="_LHS",
                rsuffix="_RHS",
            )
            if combined.empty:
                self.logger.warning("No overlapping dates for ratio %s.", instrument)
                return pd.DataFrame()

            ratio = pd.DataFrame(index=combined.index)
            for field in ["Open", "High", "Low", "Close", "Volume"]:
                lcol = f"{field}_LHS"
                rcol = f"{field}_RHS"
                if lcol not in combined.columns or rcol not in combined.columns:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio[field] = combined[lcol] / combined[rcol]
            ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
            ratio.dropna(how="all", inplace=True)
            return ratio

        # single ticker
        return self._get_single_ticker_df(instrument.strip().upper())

    # ----------------- internals -----------------

    def _compute_required_start_date(self) -> date:
        """Compute minimum required start date for shadow backfill."""
        today = date.today()
        base = today - timedelta(days=int(self.lookback_years * 365))
        if self.date_start:
            return min(base, self.date_start)
        return base

    def _get_single_ticker_df(self, ticker: str) -> pd.DataFrame:
        ticker_dir = self.output_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        csv_path = ticker_dir / f"{ticker}_daily.csv"

        required_start = self._compute_required_start_date()

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, index_col=0)
                df.index = pd.to_datetime(df.index)
                df = self._sanitize_df(df, ticker=ticker)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Failed to read %s: %s. Re-downloading.", csv_path, exc)
                df = pd.DataFrame()

            need_backfill = False
            if df.empty or len(df) < 60:
                need_backfill = True
            elif df.index.min().date() > required_start:
                need_backfill = True

            if need_backfill:
                self.logger.info("Shadow backfill for %s.", ticker)
                df = self._download_and_persist(ticker, csv_path)
        else:
            df = self._download_and_persist(ticker, csv_path)

        return df

    def _download_and_persist(self, ticker: str, csv_path: Path) -> pd.DataFrame:
        df_raw = self.download_price_history(
            ticker,
            start_date=self.date_start,
            end_date=self.date_end,
            period=self.period or years_to_period_for_backfill(self.lookback_years),
        )
        df_clean = self._sanitize_df(df_raw, ticker=ticker)
        if df_clean is None or df_clean.empty:
            self.logger.warning("No data returned for %s from yfinance.", ticker)
            return pd.DataFrame()
        try:
            df_clean.to_csv(csv_path)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to write %s: %s", csv_path, exc)
        # Re-read and sanitize again
        try:
            df_disk = pd.read_csv(csv_path, index_col=0)
            df_disk.index = pd.to_datetime(df_disk.index)
            df_disk = self._sanitize_df(df_disk, ticker=ticker)
            return df_disk
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to re-read %s: %s", csv_path, exc)
            return df_clean

    def download_price_history(
        self,
        ticker: str,
        start_date: Optional[date],
        end_date: Optional[date],
        period: Optional[str],
    ) -> pd.DataFrame:
        """Download via yfinance with group_by='column' and basic rate limiting."""
        params: Dict[str, Any] = {
            "tickers": ticker,
            "group_by": "column",
            "auto_adjust": False,
            "actions": False,
        }

        if start_date and end_date and not period:
            params["start"] = start_date
            params["end"] = end_date
        else:
            params["period"] = period or "5y"

        self.logger.info("Downloading %s with params %s", ticker, params)
        try:
            df = yf.download(**params)
        except Exception as exc:  # pragma: no cover
            self.logger.error("yfinance download failed for %s: %s", ticker, exc)
            df = pd.DataFrame()

        # light rate limiting
        time.sleep(1.0)
        if isinstance(df, pd.Series):
            df = df.to_frame(name="Close")
        return df

    # ----------------- sanitizer -----------------

    def _sanitize_df(self, df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Universal fixer:
        - Normalizes MultiIndex columns (swap levels when needed).
        - Selects single ticker slice.
        - Flattens columns to canonical OHLCV names.
        - Enforces datetime index, tz-naive, sorted, no duplicates.
        - Coerces numeric data, drops fully-NaN OHLC rows.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # Ensure DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame("Close")

        df = df.copy()

        # Handle MultiIndex columns (ticker/field or field/ticker)
        if isinstance(df.columns, pd.MultiIndex):
            fields = {"open", "high", "low", "close", "adj close", "adj_close", "volume"}
            level0 = {str(c).lower() for c in df.columns.get_level_values(0)}
            level1 = {str(c).lower() for c in df.columns.get_level_values(1)}

            c0 = len(fields & level0)
            c1 = len(fields & level1)
            fields_level = 0 if c0 >= c1 else 1
            ticker_level = 1 - fields_level

            if fields_level != 0:
                df = df.swaplevel(0, 1, axis=1)

            # After possible swap, level0 should be fields
            df = df.sort_index(axis=1)
            try:
                all_tickers = list(df.columns.get_level_values(1).unique())
            except Exception:
                all_tickers = []

            chosen_ticker = None
            if ticker and ticker in all_tickers:
                chosen_ticker = ticker
            elif all_tickers:
                chosen_ticker = all_tickers[0]

            if chosen_ticker is not None:
                df = df.xs(chosen_ticker, axis=1, level=1)
            else:
                # Drop ticker level; keep whatever fields exist
                df.columns = df.columns.get_level_values(0)

        # Flatten column names and normalize; keep Close vs Adj Close distinct
        col_map: Dict[str, str] = {}
        for col in df.columns:
            key = str(col).strip()
            lkey = key.lower()
            if lkey == "open":
                col_map[col] = "Open"
            elif lkey == "high":
                col_map[col] = "High"
            elif lkey == "low":
                col_map[col] = "Low"
            elif lkey == "close":
                col_map[col] = "Close"
            elif lkey in ("adj close", "adj_close"):
                col_map[col] = "Adj Close"
            elif lkey == "volume":
                col_map[col] = "Volume"
            else:
                col_map[col] = key

        df = df.rename(columns=col_map)

        # Index normalization
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Coerce numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure OHLCV existence
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.logger.warning("Missing OHLCV columns %s in sanitized data for %s", missing, ticker or "")

        # Drop rows where all OHLC are NaN
        ohlc_cols = [c for c in required_cols if c in df.columns]
        if ohlc_cols:
            mask = df[ohlc_cols].isna().all(axis=1)
            df = df[~mask]

        return df


# ---------------------------------------------------------------------------
# FinancialAnalysis
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    name: str
    df: pd.DataFrame
    fib_levels: Dict[str, float]
    events: List[Dict[str, Any]]
    geometry: Dict[str, Any]
    performance: Dict[str, float]
    touch_stats: Optional[pd.DataFrame] = None
    touched_indicators: Optional[Set[str]] = None


@dataclass
class InstrumentResult:
    instrument: str
    df: pd.DataFrame
    windows: Dict[str, WindowResult]
    events: List[Dict[str, Any]]
    touch_stats: pd.DataFrame
    touched_indicators: Set[str]


class FinancialAnalysis:
    """
    Encapsulates:
    - Indicators (MAs, Bollinger, RSI, MACD)
    - Regressions (Legacy, Multi-timeframe, Log-price)
    - Rolling LR curves
    - Geometry
    - Events / signals
    - Indicator touch statistics
    - Performance metrics
    - Candle Physics (Z-Score, Wick Ratio, Streaks)
    """

    def __init__(
        self,
        risk_free_rate: float,
        reg_timeframes: Dict[str, int],
        legacy_lrc_days: int,
        geometry_mode: str,
        geometry_intervals: List[str],
        extrema_windows: List[int],
        extrema_n: int,
        extrema_order: int,
        extrema_project_days: int,
        feature_flags: Dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.risk_free_rate = float(risk_free_rate)
        self.reg_timeframes = reg_timeframes
        self.legacy_lrc_days = int(legacy_lrc_days)
        self.geometry_mode = geometry_mode
        self.geometry_intervals = geometry_intervals
        self.extrema_windows = extrema_windows
        self.extrema_n = extrema_n
        self.extrema_order = extrema_order
        self.extrema_project_days = extrema_project_days
        self.feature_flags = feature_flags
        self.logger = logger
        
        # Flags
        self.touch_tolerance = float(feature_flags.get("touch_tolerance", 0.0))
        self.detect_body_touches = feature_flags.get("detect_body_touches", False)
        self.enable_physics = not feature_flags.get("no_physics", False)
        self.enable_streaks = not feature_flags.get("no_streaks", False)
        self.enable_patterns = not feature_flags.get("no_patterns", False)
        self.enable_gaps = not feature_flags.get("no_gaps", False)

    # ----------------- public API -----------------

    def run_for_instrument(self, instrument: str, df: pd.DataFrame) -> InstrumentResult:
        """
        Run full analysis pipeline for a sanitized OHLCV DataFrame.
        """
        if df is None or df.empty:
            return InstrumentResult(
                instrument=instrument,
                df=pd.DataFrame(),
                windows={},
                events=[],
                touch_stats=pd.DataFrame(),
                touched_indicators=set(),
            )

        df = df.copy()

        # Indicators
        self._compute_mas(df)
        self._compute_bollinger(df)
        self._compute_rsi(df)
        self._compute_macd(df)

        # Regressions
        legacy_info = self._compute_legacy_lrc(df) if self.legacy_lrc_days > 0 else {"events": []}
        self._compute_numeric_regression(df)
        self._compute_multitime_regressions(df)
        self._compute_log_channels(df)
        self._compute_rolling_lr_curves(df)

        # Geometry
        geometry_info = self._compute_geometry(df)

        # Windows
        windows = self._build_windows(df)

        # Events & touch stats on full data
        events_all: List[Dict[str, Any]] = []
        events_all.extend(self._detect_buy_sell_and_stars(df))
        events_all.extend(self._detect_spikes(df))
        events_all.extend(self._detect_ma_touches(df))
        events_all.extend(legacy_info.get("events", []))
        
        # Legacy Option: Detect Regression Touches
        reg_events = self._detect_regression_touches(df)
        events_all.extend(reg_events)

        # OPTION B: Candle Physics & Advanced Patterns
        if self.enable_physics or self.enable_streaks or self.enable_patterns or self.enable_gaps:
             events_all.extend(self._detect_advanced_patterns(df))

        # Collect set of all touched indicators for Smart Legends
        touched_set = {e["label"] for e in events_all if "label" in e}

        # Touch statistics on full window
        touch_stats_df = self._compute_indicator_touch_stats(df)

        # Build per-window results
        window_results: Dict[str, WindowResult] = {}
        for wname, wdf in windows.items():
            fib = self._compute_fib_levels(wdf)
            perf = self._compute_performance_metrics(wdf)
            # filter events by date in window
            event_subset = [e for e in events_all if e.get("date") in wdf.index]
            
            # Recalculate touched set for this window
            win_touched = {e["label"] for e in event_subset if "label" in e}

            wr = WindowResult(
                name=wname,
                df=wdf,
                fib_levels=fib,
                events=event_subset,
                geometry=geometry_info,
                performance=perf,
                touch_stats=touch_stats_df if wname == "full" else None,
                touched_indicators=win_touched
            )
            window_results[wname] = wr

            # Fibonacci touches per window
            fib_touch_events = self._detect_fib_touches(wdf, fib)
            events_all.extend(fib_touch_events)
            wr.events.extend(fib_touch_events)
            
            # Add fib labels to touched set
            for e in fib_touch_events:
                win_touched.add(e["label"])
                touched_set.add(e["label"])

        return InstrumentResult(
            instrument=instrument,
            df=df,
            windows=window_results,
            events=events_all,
            touch_stats=touch_stats_df,
            touched_indicators=touched_set
        )

    # ----------------- indicators -----------------

    def _compute_mas(self, df: pd.DataFrame) -> None:
        if "Close" not in df.columns:
            return
        for p in [9, 10, 20, 50, 100, 200, 300]:
            df[f"SMA_{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
        for p in [9, 12, 26, 50, 100, 200]:
            df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()

    def _compute_bollinger(self, df: pd.DataFrame) -> None:
        if "Close" not in df.columns:
            return
        window = 20
        sma = df["Close"].rolling(window=window, min_periods=window).mean()
        std = df["Close"].rolling(window=window, min_periods=window).std()
        df["BB_Middle"] = sma
        df["BB_Upper_1std"] = sma + std
        df["BB_Upper_2std"] = sma + 2 * std
        df["BB_Lower_1std"] = sma - std
        df["BB_Lower_2std"] = sma - 2 * std

    def _compute_rsi(self, df: pd.DataFrame, period: int = 14) -> None:
        if "Close" not in df.columns:
            return
        delta = df["Close"].diff()
        gain = delta.where(delta > 0.0, 0.0)
        loss = -delta.where(delta < 0.0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df["RSI_14"] = rsi

    def _compute_macd(self, df: pd.DataFrame) -> None:
        if "Close" not in df.columns:
            return
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal_line
        df["MACD"] = macd
        df["MACD_Signal"] = signal_line
        df["MACD_Hist"] = hist

    # ----------------- regr / channels -----------------

    def _compute_legacy_lrc(self, df: pd.DataFrame) -> Dict[str, Any]:
        info: Dict[str, Any] = {"events": []}
        if "Close" not in df.columns:
            return info
        days = self.legacy_lrc_days 
        if days <= 1 or len(df) < days:
            return info

        tail = df.tail(days)
        x = np.arange(len(tail), dtype=float)
        y = tail["Close"].values.astype(float)

        try:
            slope, intercept, _r, _p, _stderr = stats.linregress(x, y)
        except Exception:
            return info

        base = slope * x + intercept
        resid = y - base
        std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

        multipliers = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        center_series = pd.Series(np.nan, index=df.index)
        center_series.iloc[-days:] = base
        df["LegacyLRC_Center"] = center_series

        for m in multipliers:
            up_series = pd.Series(np.nan, index=df.index)
            lo_series = pd.Series(np.nan, index=df.index)
            up_series.iloc[-days:] = base + m * std
            lo_series.iloc[-days:] = base - m * std
            df[f"LegacyLRC_Up_{m:.2f}"] = up_series
            df[f"LegacyLRC_Lo_{m:.2f}"] = lo_series

        # Wick-touch events
        events: List[Dict[str, Any]] = []
        for i, (idx, row) in enumerate(tail.iterrows()):
            low = row.get("Low")
            high = row.get("High")
            if pd.isna(low) or pd.isna(high):
                continue
            base_val = base[i]
            for m in multipliers:
                up_price = base_val + m * std
                lo_price = base_val - m * std
                if low <= up_price <= high:
                    events.append(
                        {
                            "date": idx,
                            "event_type": "legacy_lrc_touch",
                            "price": float(up_price),
                            "label": f"LegacyLRC_Up_{m:.2f}",
                            "extra": "",
                        }
                    )
                if low <= lo_price <= high:
                    events.append(
                        {
                            "date": idx,
                            "event_type": "legacy_lrc_touch",
                            "price": float(lo_price),
                            "label": f"LegacyLRC_Lo_{m:.2f}",
                            "extra": "",
                        }
                    )

        info.update({"slope": slope, "intercept": intercept, "std": std, "events": events})
        return info

    def _compute_numeric_regression(self, df: pd.DataFrame) -> None:
        if "Close" not in df.columns or len(df) < 2:
            return
        n = len(df)
        x = np.arange(n, dtype=float)
        y = df["Close"].values.astype(float)
        try:
            slope, intercept, _r, _p, _stderr = stats.linregress(x, y)
        except Exception:
            return

        base = slope * x + intercept
        df["Reg_Num_Center"] = base
        resid = y - base
        std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        multipliers = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
        for m in multipliers:
            df[f"Reg_Num_High_{m:.2f}"] = base + m * std
            df[f"Reg_Num_Low_{m:.2f}"] = base - m * std

    def _compute_multitime_regressions(self, df: pd.DataFrame) -> None:
        if "Close" not in df.columns or not self.reg_timeframes:
            return
        n = len(df)
        for name, days in self.reg_timeframes.items():
            if days <= 1 or n < days:
                continue
            tail = df["Close"].tail(days)
            x = np.arange(len(tail), dtype=float)
            try:
                slope, intercept, _r, _p, _stderr = stats.linregress(x, tail.values.astype(float))
            except Exception:
                continue
            base = slope * x + intercept
            resid = tail.values - base
            std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
            center = pd.Series(np.nan, index=df.index)
            center.iloc[-days:] = base
            df[f"{name}_Linear_Reg"] = center
            for m in [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]:
                up = pd.Series(np.nan, index=df.index)
                lo = pd.Series(np.nan, index=df.index)
                up.iloc[-days:] = base + m * std
                lo.iloc[-days:] = base - m * std
                df[f"{name}_Reg_High_{m:.2f}"] = up
                df[f"{name}_Reg_Low_{m:.2f}"] = lo

    def _compute_log_channels(self, df: pd.DataFrame) -> None:
        if "Close" not in df.columns:
            return
        for period in [50, 144]:
            if len(df) < period:
                continue
            tail = df["Close"].tail(period).dropna()
            if len(tail) < 2:
                continue
            log_price = np.log(tail.values.astype(float))
            x = np.arange(len(log_price), dtype=float)
            try:
                slope, intercept, _r, _p, _stderr = stats.linregress(x, log_price)
            except Exception:
                continue
            base = slope * x + intercept
            resid = log_price - base
            std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

            center = pd.Series(np.nan, index=df.index)
            up1 = pd.Series(np.nan, index=df.index)
            lo1 = pd.Series(np.nan, index=df.index)
            up2 = pd.Series(np.nan, index=df.index)
            lo2 = pd.Series(np.nan, index=df.index)

            exp_center = np.exp(base)
            exp_up1 = np.exp(base + std)
            exp_lo1 = np.exp(base - std)
            exp_up2 = np.exp(base + 2 * std)
            exp_lo2 = np.exp(base - 2 * std)

            center.iloc[-period:] = exp_center
            up1.iloc[-period:] = exp_up1
            lo1.iloc[-period:] = exp_lo1
            up2.iloc[-period:] = exp_up2
            lo2.iloc[-period:] = exp_lo2

            prefix = f"LogReg{period}"
            df[f"{prefix}_Center"] = center
            df[f"{prefix}_Up1"] = up1
            df[f"{prefix}_Lo1"] = lo1
            df[f"{prefix}_Up2"] = up2
            df[f"{prefix}_Lo2"] = lo2

    def _compute_rolling_lr_curves(self, df: pd.DataFrame) -> None:
        if "Close" not in df.columns:
            return

        def rolling_lr(series: pd.Series, window: int) -> pd.Series:
            values = series.values.astype(float)
            n = len(values)
            out = np.full(n, np.nan, dtype=float)
            if window <= 1 or n < window:
                return pd.Series(out, index=series.index)
            for i in range(window - 1, n):
                y = values[i - window + 1 : i + 1]
                x = np.arange(window, dtype=float)
                try:
                    slope, intercept, _r, _p, _stderr = stats.linregress(x, y)
                except Exception:
                    continue
                out[i] = slope * (window - 1) + intercept
            return pd.Series(out, index=series.index)

        for span in [12, 20, 50, 100, 150]:
            df[f"LRcurve_{span}"] = rolling_lr(df["Close"], span)

    # ----------------- geometry -----------------

    def _compute_geometry(self, df: pd.DataFrame) -> Dict[str, Any]:
        geometry: Dict[str, Any] = {"lines": [], "intersections": []}
        if df.empty or "High" not in df.columns or "Low" not in df.columns:
            return geometry

        for interval_str in self.geometry_intervals:
            days = interval_str_to_days(interval_str)
            subset = df.tail(days) if len(df) > days else df
            if subset.empty:
                continue

            if self.geometry_mode == "prompt":
                peaks_pts, troughs_pts = self._prompt_geometry_points(subset, interval_str)
            else:
                peaks_pts, troughs_pts = self._auto_geometry_points(subset)

            if len(peaks_pts) >= 2:
                line = self._fit_geometry_line(df, peaks_pts, f"Geom_Peak_{interval_str}")
                geometry["lines"].append(line)
            if len(troughs_pts) >= 2:
                line = self._fit_geometry_line(df, troughs_pts, f"Geom_Trough_{interval_str}")
                geometry["lines"].append(line)

        # intersections
        lines = geometry["lines"]
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                l1 = lines[i]
                l2 = lines[j]
                m1, b1 = l1["slope"], l1["intercept"]
                m2, b2 = l2["slope"], l2["intercept"]
                if m1 == m2:
                    continue
                x_int = (b2 - b1) / (m1 - m2)
                x_round = int(round(x_int))
                try:
                    dt = date.fromordinal(x_round)
                except Exception:
                    continue
                price = m1 * x_int + b1
                geometry["intersections"].append(
                    {
                        "date": dt,
                        "price": float(price),
                        "lines": (l1["name"], l2["name"]),
                    }
                )

        return geometry

    def _auto_geometry_points(
        self, subset: pd.DataFrame
    ) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
        peaks_pts: List[Tuple[pd.Timestamp, float]] = []
        troughs_pts: List[Tuple[pd.Timestamp, float]] = []
        highs = subset["High"].values.astype(float)
        lows = subset["Low"].values.astype(float)
        idx_array = subset.index.to_numpy()

        try:
            peak_idx = signal.argrelextrema(highs, np.greater_equal, order=self.extrema_order)[0]
            trough_idx = signal.argrelextrema(lows, np.less_equal, order=self.extrema_order)[0]
        except Exception:
            peak_idx = np.array([], dtype=int)
            trough_idx = np.array([], dtype=int)

        for i in peak_idx:
            peaks_pts.append((pd.Timestamp(idx_array[i]), float(highs[i])))
        for i in trough_idx:
            troughs_pts.append((pd.Timestamp(idx_array[i]), float(lows[i])))

        # fallback: take highest highs / lowest lows if not enough extrema
        if len(peaks_pts) < 2:
            sorted_highs = sorted(
                ((pd.Timestamp(idx_array[i]), float(highs[i])) for i in range(len(highs))),
                key=lambda x: x[1],
                reverse=True,
            )
            peaks_pts = sorted_highs[: min(2, len(sorted_highs))]
        if len(troughs_pts) < 2:
            sorted_lows = sorted(
                ((pd.Timestamp(idx_array[i]), float(lows[i])) for i in range(len(lows))),
                key=lambda x: x[1],
            )
            troughs_pts = sorted_lows[: min(2, len(sorted_lows))]

        return peaks_pts, troughs_pts

    def _prompt_geometry_points(
        self, subset: pd.DataFrame, interval_str: str
    ) -> Tuple[List[Tuple[pd.Timestamp, float]], List[Tuple[pd.Timestamp, float]]]:
        """
        Prompt user in geometry-mode=prompt for peak/trough anchor dates.
        If input is blank or invalid, fall back to auto detection.
        """
        self.logger.info("Prompting for geometry anchors for interval %s", interval_str)
        try:
            peaks_input = input(
                f"Enter two peak dates for geometry interval {interval_str} "
                "(YYYY-MM-DD,YYYY-MM-DD) or empty for auto: "
            ).strip()
            troughs_input = input(
                f"Enter two trough dates for geometry interval {interval_str} "
                "(YYYY-MM-DD,YYYY-MM-DD) or empty for auto: "
            ).strip()
        except EOFError:
            return self._auto_geometry_points(subset)

        def parse_dates(s: str) -> List[pd.Timestamp]:
            out: List[pd.Timestamp] = []
            if not s:
                return out
            parts = [p.strip() for p in s.split(",") if p.strip()]
            for part in parts:
                try:
                    dt = datetime.strptime(part, "%Y-%m-%d")
                    out.append(pd.Timestamp(dt))
                except ValueError:
                    continue
            return out

        peaks_dates = parse_dates(peaks_input)
        troughs_dates = parse_dates(troughs_input)

        def snap_points(dates_list: List[pd.Timestamp], use_high: bool) -> List[Tuple[pd.Timestamp, float]]:
            if not dates_list:
                return []
            pts: List[Tuple[pd.Timestamp, float]] = []
            for dt in dates_list:
                # snap to nearest index
                idx = subset.index.get_indexer([dt], method="nearest")
                if len(idx) == 0 or idx[0] < 0:
                    continue
                i = idx[0]
                ts = subset.index[i]
                price = float(subset["High"].iloc[i] if use_high else subset["Low"].iloc[i])
                pts.append((ts, price))
            return pts

        peaks_pts = snap_points(peaks_dates, use_high=True)
        troughs_pts = snap_points(troughs_dates, use_high=False)

        if len(peaks_pts) < 2 or len(troughs_pts) < 2:
            return self._auto_geometry_points(subset)

        return peaks_pts, troughs_pts

    def _fit_geometry_line(
        self,
        full_df: pd.DataFrame,
        points: List[Tuple[pd.Timestamp, float]],
        name: str,
    ) -> Dict[str, Any]:
        pts = sorted(points, key=lambda x: x[0])
        p1, p2 = pts[0], pts[-1]
        x1 = p1[0].to_pydatetime().toordinal()
        x2 = p2[0].to_pydatetime().toordinal()
        y1 = p1[1]
        y2 = p2[1]
        if x1 == x2:
            slope = 0.0
        else:
            slope = (y2 - y1) / float(x2 - x1)
        intercept = y1 - slope * x1
        angle_deg = math.degrees(math.atan(slope))  # simple orientation

        x_all = np.array([dt.to_pydatetime().toordinal() for dt in full_df.index], dtype=float)
        line_vals = slope * x_all + intercept
        series = pd.Series(line_vals, index=full_df.index)
        full_df[name] = series

        return {
            "name": name,
            "slope": slope,
            "intercept": intercept,
            "angle_deg": angle_deg,
            "points": [(p[0].to_pydatetime().date(), float(p[1])) for p in pts],
        }

    # ----------------- windows / fib / perf -----------------

    def _build_windows(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        windows: Dict[str, pd.DataFrame] = {"full": df}
        if df.empty:
            return windows
        last_dt = df.index.max()
        one_year_ago = last_dt - pd.Timedelta(days=252)
        six_months_ago = last_dt - pd.Timedelta(days=126)
        five_years_ago = last_dt - pd.Timedelta(days=252 * 5)

        windows["1y"] = df[df.index >= one_year_ago]
        windows["6mo"] = df[df.index >= six_months_ago]
        windows["5y"] = df[df.index >= five_years_ago]
        return windows

    def _compute_fib_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty or "High" not in df.columns or "Low" not in df.columns:
            return {}
        high_price = float(df["High"].max())
        low_price = float(df["Low"].min())
        diff = high_price - low_price
        if diff <= 0:
            return {}
        ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        levels: Dict[str, float] = {}
        for r in ratios:
            level = high_price - diff * r
            label = f"{int(round(r * 100))}%"
            levels[label] = float(level)
        return levels

    def _compute_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        result: Dict[str, float] = {
            "mu_ann": float("nan"),
            "sigma_ann": float("nan"),
            "sharpe": float("nan"),
            "rf_annual": self.risk_free_rate,
        }
        if df is None or df.empty or "Close" not in df.columns:
            return result
        prices = df["Close"].dropna()
        if len(prices) < 2:
            return result
        r = np.log(prices / prices.shift(1)).dropna()
        if r.empty:
            return result
        mu_daily = float(r.mean())
        sigma_daily = float(r.std(ddof=1))
        mu_ann = mu_daily * 252.0
        sigma_ann = sigma_daily * math.sqrt(252.0)
        rf_ann = self.risk_free_rate
        mu_excess = mu_ann - rf_ann
        sharpe = mu_excess / sigma_ann if sigma_ann > 0 else float("nan")
        result["mu_ann"] = mu_ann
        result["sigma_ann"] = sigma_ann
        result["sharpe"] = sharpe
        return result

    # ----------------- events & touch stats -----------------

    def _detect_buy_sell_and_stars(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if "Open" not in df.columns or "Close" not in df.columns:
            return events
        opens = df["Open"]
        closes = df["Close"]

        # Simple buy/sell
        for i in range(1, max(1, len(df) - 1)):
            idx = df.index[i]
            c0 = closes.iloc[i]
            o0 = opens.iloc[i]
            c_prev = closes.iloc[i - 1]
            if pd.isna(c0) or pd.isna(o0) or pd.isna(c_prev):
                continue
            if c0 > o0 and c0 > c_prev:
                events.append(
                    {
                        "date": idx,
                        "event_type": "buy",
                        "price": float(c0),
                        "label": "SimpleBuy",
                        "extra": "",
                    }
                )
            elif c0 < o0 and c0 < c_prev:
                events.append(
                    {
                        "date": idx,
                        "event_type": "sell",
                        "price": float(c0),
                        "label": "SimpleSell",
                        "extra": "",
                    }
                )

        # Stars (consecutive up/down sequences)
        k = 0
        direction: Optional[str] = None
        for i in range(len(df)):
            idx = df.index[i]
            o = opens.iloc[i]
            c = closes.iloc[i]
            if pd.isna(o) or pd.isna(c):
                direction = None
                k = 0
                continue
            if c > o:
                new_dir = "up"
            elif c < o:
                new_dir = "down"
            else:
                new_dir = None

            if new_dir is None:
                direction = None
                k = 0
                continue

            if direction == new_dir:
                k += 1
            else:
                direction = new_dir
                k = 1

            if k >= 2:
                size = 8 + (k - 2) * 4
                color = "green" if new_dir == "up" else "red"
                events.append(
                    {
                        "date": idx,
                        "event_type": "star",
                        "price": float(c),
                        "label": f"{new_dir}_seq_{k}",
                        "extra": json.dumps({"size": size, "color": color}),
                    }
                )

        return events

    def _detect_spikes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if "Volume" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
            return events
        vol = df["Volume"]
        price_range = df["High"] - df["Low"]
        vol_change = vol.diff()

        window = 20
        vol_mean = vol_change.rolling(window, min_periods=window).mean()
        vol_std = vol_change.rolling(window, min_periods=window).std()
        price_mean = price_range.rolling(window, min_periods=window).mean()
        price_std = price_range.rolling(window, min_periods=window).std()

        volume_threshold = 1.5
        price_threshold = 2.0

        cond = (
            (vol_std > 0)
            & (price_std > 0)
            & ((vol_change - vol_mean).abs() > volume_threshold * vol_std)
            & ((price_range - price_mean).abs() > price_threshold * price_std)
        )
        idxs = df.index[cond.fillna(False)]

        for ts in idxs:
            c = df.loc[ts, "Close"]
            events.append(
                {
                    "date": ts,
                    "event_type": "volume_price_spike",
                    "price": float(c),
                    "label": "Spike",
                    "extra": "",
                }
            )
        return events

    def _detect_ma_touches(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not {"Open", "Close", "Low", "High"}.issubset(df.columns):
            return events
        ma_periods = [9, 10, 20, 50, 100, 200, 300]

        for i in range(len(df)):
            row = df.iloc[i]
            idx = row.name
            o = row["Open"]
            c = row["Close"]
            low = row["Low"]
            high = row["High"]
            if pd.isna(o) or pd.isna(c) or pd.isna(low) or pd.isna(high):
                continue
            body_low = min(o, c)
            body_high = max(o, c)
            for p in ma_periods:
                col = f"SMA_{p}"
                if col not in df.columns:
                    continue
                ma = row[col]
                if pd.isna(ma):
                    continue
                if body_low <= ma <= body_high:
                    events.append(
                        {
                            "date": idx,
                            "event_type": "ma_body_touch",
                            "price": float(ma),
                            "label": col,
                            "extra": "",
                        }
                    )
        return events

    def _detect_regression_touches(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Includes Legacy Options:
        - touch_tolerance: if > 0, expands the hit box.
        - detect_body_touches: if True, adds a separate event type for body crosses.
        """
        events: List[Dict[str, Any]] = []
        if not {"Low", "High", "Open", "Close"}.issubset(df.columns):
            return events
        
        band_cols: List[str] = []
        for col in df.columns:
            name = str(col)
            if name.startswith("LegacyLRC_") or name.startswith("Reg_Num_"):
                band_cols.append(name)
            for prefix in self.reg_timeframes.keys():
                if name.startswith(f"{prefix}_Reg_") or name.startswith(f"{prefix}_Linear_"):
                    band_cols.append(name)
            if name.startswith("LogReg50_") or name.startswith("LogReg144_"):
                band_cols.append(name)

        band_cols = sorted(set(band_cols))
        if not band_cols:
            return events

        tol = self.touch_tolerance

        for i in range(len(df)):
            row = df.iloc[i]
            idx = row.name
            low = row["Low"]
            high = row["High"]
            open_ = row["Open"]
            close = row["Close"]
            
            if pd.isna(low) or pd.isna(high):
                continue

            for col in band_cols:
                val = row.get(col)
                if pd.isna(val):
                    continue
                
                # Tolerance adjustment
                if tol > 0:
                    delta = abs(val * tol)
                    check_low = val - delta
                    check_high = val + delta
                else:
                    check_low = val
                    check_high = val

                # 1. Wick/Range Touch
                if not (high < check_low or low > check_high):
                    events.append(
                        {
                            "date": idx,
                            "event_type": "regression_touch",
                            "price": float(val),
                            "label": col,
                            "extra": "",
                        }
                    )

                # 2. Body Touch (Legacy Option)
                if self.detect_body_touches:
                    body_min = min(open_, close)
                    body_max = max(open_, close)
                    if not (body_max < check_low or body_min > check_high):
                         events.append(
                        {
                            "date": idx,
                            "event_type": "regression_body_touch",
                            "price": float(val),
                            "label": col,
                            "extra": "",
                        }
                    )

        return events

    def _detect_fib_touches(self, df: pd.DataFrame, fib_levels: Dict[str, float]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not fib_levels or not {"Low", "High"}.issubset(df.columns):
            return events
        for i in range(len(df)):
            row = df.iloc[i]
            idx = row.name
            low = row["Low"]
            high = row["High"]
            if pd.isna(low) or pd.isna(high):
                continue
            for label, level in fib_levels.items():
                if low <= level <= high:
                    events.append(
                        {
                            "date": idx,
                            "event_type": "fib_touch",
                            "price": float(level),
                            "label": label,
                            "extra": "",
                        }
                    )
        return events

    # ----------------- Option B: Candle Physics & Patterns -----------------
    def _detect_advanced_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if len(df) < 20 or not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            return events

        # 1. Candle Physics: Elephant Bars (Z-Score of Range)
        if self.enable_physics:
            ranges = df["High"] - df["Low"]
            rolling_mean = ranges.rolling(20).mean()
            rolling_std = ranges.rolling(20).std()
            # Avoid div by zero
            z_scores = (ranges - rolling_mean) / rolling_std.replace(0, 1)
            
            # Wicks for Pinbar
            body_top = df[["Open", "Close"]].max(axis=1)
            body_bottom = df[["Open", "Close"]].min(axis=1)
            upper_wick = df["High"] - body_top
            lower_wick = body_bottom - df["Low"]
            total_wick = upper_wick + lower_wick
            wick_ratio = total_wick / ranges.replace(0, 1)

            for i in range(20, len(df)):
                idx = df.index[i]
                c = df["Close"].iloc[i]
                
                # Elephant
                z = z_scores.iloc[i]
                if z > 3.0:
                    events.append({
                        "date": idx,
                        "event_type": "physics_elephant",
                        "price": float(c),
                        "label": f"Elephant_{z:.1f}x",
                        "extra": ""
                    })
                
                # Pinbar (Rejection > 66% of range)
                wr = wick_ratio.iloc[i]
                if wr > 0.66:
                    events.append({
                        "date": idx,
                        "event_type": "physics_pinbar",
                        "price": float(c),
                        "label": f"Pinbar_{wr:.2f}",
                        "extra": ""
                    })

        # 2. Sequential Streaks (9-Count)
        if self.enable_streaks:
            # We need to iterate
            closes = df["Close"].values
            dates = df.index
            setup_up = 0
            setup_down = 0
            
            for i in range(4, len(closes)):
                if closes[i] > closes[i-4]:
                    setup_up += 1
                    setup_down = 0
                elif closes[i] < closes[i-4]:
                    setup_down += 1
                    setup_up = 0
                else:
                    setup_up = 0
                    setup_down = 0
                
                if setup_up == 9:
                     events.append({
                        "date": dates[i],
                        "event_type": "streak_9_up",
                        "price": float(closes[i]),
                        "label": "9_Up",
                        "extra": ""
                    })
                if setup_down == 9:
                    events.append({
                        "date": dates[i],
                        "event_type": "streak_9_down",
                        "price": float(closes[i]),
                        "label": "9_Down",
                        "extra": ""
                    })

        # 3. Market Regimes (NR7, Inside, Outside)
        if self.enable_patterns:
            ranges = df["High"] - df["Low"]
            highs = df["High"].values
            lows = df["Low"].values
            dates = df.index
            
            for i in range(7, len(df)):
                # NR7
                current_range = ranges.iloc[i]
                prev_6 = ranges.iloc[i-6:i]
                if current_range < prev_6.min():
                    events.append({
                        "date": dates[i],
                        "event_type": "pattern_nr7",
                        "price": float(df["Close"].iloc[i]),
                        "label": "NR7",
                        "extra": ""
                    })
                
                # Inside/Outside (requires i-1)
                h0, l0 = highs[i], lows[i]
                h1, l1 = highs[i-1], lows[i-1]
                
                if h0 < h1 and l0 > l1:
                     events.append({
                        "date": dates[i],
                        "event_type": "pattern_inside",
                        "price": float(df["Close"].iloc[i]),
                        "label": "Inside",
                        "extra": ""
                    })
                elif h0 > h1 and l0 < l1:
                     events.append({
                        "date": dates[i],
                        "event_type": "pattern_outside",
                        "price": float(df["Close"].iloc[i]),
                        "label": "Outside",
                        "extra": ""
                    })

        # 4. Gaps
        if self.enable_gaps:
             highs = df["High"].values
             lows = df["Low"].values
             dates = df.index
             
             for i in range(1, len(df)):
                 # Bull Gap: Low[i] > High[i-1]
                 if lows[i] > highs[i-1]:
                     events.append({
                        "date": dates[i],
                        "event_type": "gap_bull",
                        "price": float(lows[i]),
                        "label": "Gap Up",
                        "extra": ""
                    })
                 # Bear Gap: High[i] < Low[i-1]
                 elif highs[i] < lows[i-1]:
                     events.append({
                        "date": dates[i],
                        "event_type": "gap_bear",
                        "price": float(highs[i]),
                        "label": "Gap Down",
                        "extra": ""
                    })

        return events


    def _compute_indicator_touch_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute indicator touch statistics for all indicator lines:
        - SMA_*
        - BB_* (Bollinger bands)
        - Reg_Num_* (numeric regression)
        - <tf>_* (multi-timeframe regressions)
        - LogReg* (log-price channels)
        - LegacyLRC_* (legacy regression)
        """
        if df.empty or not {"Open", "Close", "Low", "High"}.issubset(df.columns):
            return pd.DataFrame()

        # Collect all indicators we want to track
        indicator_cols: Dict[str, pd.Series] = {}
        for col in df.columns:
            name = str(col)
            if name.startswith("SMA_"):
                indicator_cols[name] = df[col]
            elif name.startswith("BB_"):
                indicator_cols[name] = df[col]
            elif name.startswith("Reg_Num_"):
                indicator_cols[name] = df[col]
            elif any(name.startswith(f"{prefix}_") for prefix in self.reg_timeframes.keys()):
                indicator_cols[name] = df[col]
            elif name.startswith("LogReg"):
                indicator_cols[name] = df[col]
            elif name.startswith("LegacyLRC_"):
                indicator_cols[name] = df[col]

        if not indicator_cols:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []
        total_candles = len(df)
        if total_candles <= 0:
            return pd.DataFrame()

        for name, series in indicator_cols.items():
            high_wick = 0
            low_wick = 0
            open_touch = 0
            close_touch = 0
            body_cross = 0
            touches = 0

            for i in range(total_candles):
                row = df.iloc[i]
                ind_val = series.iloc[i]
                if pd.isna(ind_val):
                    continue

                o = row["Open"]
                c = row["Close"]
                low = row["Low"]
                high = row["High"]
                if pd.isna(o) or pd.isna(c) or pd.isna(low) or pd.isna(high):
                    continue

                # Must intersect the candle range at all
                if not (low <= ind_val <= high):
                    continue

                touches += 1
                body_low = min(o, c)
                body_high = max(o, c)
                tol = 1e-8

                if body_low <= ind_val <= body_high:
                    body_cross += 1
                    if abs(ind_val - o) <= tol:
                        open_touch += 1
                    if abs(ind_val - c) <= tol:
                        close_touch += 1
                else:
                    if ind_val > body_high:
                        high_wick += 1
                    elif ind_val < body_low:
                        low_wick += 1

            total_candles_f = float(total_candles)
            rec = {
                "Indicator": name,
                "Total Candles": total_candles,
                "Total Touches": touches,
                "Total Touches %": (touches / total_candles_f) * 100.0,
                "High Wick %": (high_wick / total_candles_f) * 100.0,
                "Low Wick %": (low_wick / total_candles_f) * 100.0,
                "Open %": (open_touch / total_candles_f) * 100.0,
                "Close %": (close_touch / total_candles_f) * 100.0,
                "Body Cross %": (body_cross / total_candles_f) * 100.0,
            }
            records.append(rec)

        if not records:
            return pd.DataFrame()

        df_stats = pd.DataFrame(records)
        df_stats = df_stats.sort_values("Total Touches %", ascending=False)
        return df_stats.reset_index(drop=True)
 



# ---------------------------------------------------------------------------
# DashboardRenderer
# ---------------------------------------------------------------------------

class DashboardRenderer:
    """
    Responsible for:
    - Building per-view Plotly figures for each instrument.
    - Building offline HTML dashboard with tabs.
    - Writing event summary & touch stats CSVs.
    """

    def __init__(
        self,
        output_dir: Path,
        views: str,
        feature_flags: Dict[str, Any],
        clean_plot: bool,
        logger: logging.Logger,
    ) -> None:
        self.output_dir = output_dir
        self.views = self._parse_views(views)
        self.feature_flags = feature_flags
        self.clean_plot = clean_plot
        self.logger = logger
        self.instruments: Dict[str, InstrumentResult] = {}
        
        self.smart_legends = feature_flags.get("smart_legends", False)

    def _parse_views(self, views_str: str) -> List[str]:
        all_views = [
            "overview",
            "touch_stats",
            "log_channels",
            "regression",
            "last_year",
            "emas_extrema",
            "rolling_lr",
        ]
        s = (views_str or "").strip()
        if s.lower() == "all":
            return all_views
        views = [v.strip() for v in s.split(",") if v.strip()]
        return [v for v in views if v in all_views] or all_views

    def register_instrument(self, instrument: str, result: InstrumentResult) -> None:
        self.instruments[instrument] = result

    # ----------------- CSV writing -----------------

    def _write_csvs(self) -> None:
        for inst, res in self.instruments.items():
            # FIX: Sanitize the instrument name to remove slashes (e.g. SPY/IBIT -> SPY_IBIT)
            safe_inst = inst.replace("/", "_").replace("\\", "_")
            
            ticker_dir = self.output_dir / safe_inst
            ticker_dir.mkdir(parents=True, exist_ok=True)
            
            # Event summary
            if res.events:
                df_events = pd.DataFrame(res.events)
                if "date" in df_events.columns:
                    df_events = df_events.sort_values("date")
                # Use safe_inst for the filename
                df_events.to_csv(ticker_dir / f"{safe_inst}_detailed_signal_summary.csv", index=False)
            
            # Touch stats
            if res.touch_stats is not None and not res.touch_stats.empty:
                # Use safe_inst for the filename
                res.touch_stats.to_csv(ticker_dir / f"{safe_inst}_touch_statistics.csv", index=False)

    # ----------------- HTML rendering -----------------

    def render_all_dashboards(self) -> Path:
        self._write_csvs()
        if not self.instruments:
            main_path = self.output_dir / "dashboard_all.html"
            main_path.write_text("<html><body><h1>No data</h1></body></html>", encoding="utf-8")
            return main_path

        plotly_js = offline.get_plotlyjs()
        tab_buttons: List[str] = []
        contents: List[str] = []
        first = True

        for inst, res in self.instruments.items():
            for view_id in self.views:
                tab_id = f"{inst}_{view_id}".replace("/", "_").replace("\\", "_")
                active_class = " active" if first else ""
                tab_buttons.append(
                    f'<button class="tab-button{active_class}" data-target="{tab_id}">'
                    f"{inst} - {view_id}</button>"
                )
                div_html = self._build_view_div(inst, res, view_id, tab_id, active=first)
                contents.append(div_html)
                if first:
                    first = False

        tabs_html = "\n".join(tab_buttons)
        contents_html = "\n".join(contents)

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Technical Geometry Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin:0; padding:0; }}
.tab-bar {{ display:flex; flex-wrap:wrap; border-bottom:1px solid #ccc; background:#f5f5f5; }}
.tab-button {{ border:none; padding:8px 14px; cursor:pointer; background:#f5f5f5; font-size:13px; }}
.tab-button.active {{ background:#ffffff; border-bottom:2px solid #007bff; }}
.tab-content {{ display:none; padding:8px; }}
.tab-content.active {{ display:block; }}
</style>
<script type="text/javascript">
{plotly_js}
function activateTab(id, btn) {{
  var tabs = document.getElementsByClassName('tab-content');
  for (var i=0; i<tabs.length; i++) {{
    tabs[i].classList.remove('active');
  }}
  var buttons = document.getElementsByClassName('tab-button');
  for (var j=0; j<buttons.length; j++) {{
    buttons[j].classList.remove('active');
  }}
  var el = document.getElementById(id);
  if (el) {{
    el.classList.add('active');
  }}
  if (btn) {{
    btn.classList.add('active');
  }}
  window.setTimeout(function() {{
    window.dispatchEvent(new Event('resize'));
  }}, 0);
}}
window.addEventListener('load', function() {{
  var buttons = document.getElementsByClassName('tab-button');
  for (var i=0; i<buttons.length; i++) {{
    buttons[i].addEventListener('click', function(e) {{
      var target = this.getAttribute('data-target');
      activateTab(target, this);
    }});
  }}
}});
</script>
</head>
<body>
<div class="tab-bar">
{tabs_html}
</div>
{contents_html}
</body>
</html>
"""

        main_path = self.output_dir / "dashboard_all.html"
        main_path.write_text(html, encoding="utf-8")
        self.logger.info("Wrote dashboard: %s", main_path)
        return main_path

    def _choose_window_for_view(self, res: InstrumentResult, view_id: str) -> WindowResult:
        # Basic mapping of views to windows
        if view_id == "last_year":
            wname = "1y"
        elif view_id in ("overview",):
            wname = "5y" if "5y" in res.windows else "full"
        elif view_id in ("regression", "log_channels", "emas_extrema", "rolling_lr", "touch_stats"):
            wname = "1y" if "1y" in res.windows else "full"
        else:
            wname = "full"
        return res.windows.get(wname, next(iter(res.windows.values())))

    def _build_view_div(
        self,
        instrument: str,
        res: InstrumentResult,
        view_id: str,
        tab_id: str,
        active: bool,
    ) -> str:
        win = self._choose_window_for_view(res, view_id)
        df = win.df
        events = win.events
        geometry = win.geometry
        fib_levels = win.fib_levels
        touched = win.touched_indicators

        fig = self._build_figure_for_view(view_id, df, events, geometry, fib_levels, res, touched)
        fig_div = offline.plot(fig, include_plotlyjs=False, output_type="div")

        extra_html = ""
        if view_id == "touch_stats" and res.touch_stats is not None and not res.touch_stats.empty:
            table_html = res.touch_stats.to_html(
                index=False,
                float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x),
            )
            extra_html = f'<div style="max-height:300px; overflow:auto; margin-top:6px;">{table_html}</div>'

        active_class = "tab-content active" if active else "tab-content"
        return f'<div id="{tab_id}" class="{active_class}">\n{fig_div}\n{extra_html}\n</div>'

    def _build_figure_for_view(
        self,
        view_id: str,
        df: pd.DataFrame,
        events: List[Dict[str, Any]],
        geometry: Dict[str, Any],
        fib_levels: Dict[str, float],
        res: InstrumentResult,
        touched_indicators: Optional[Set[str]]
    ) -> go.Figure:
        show_touch_focus = view_id == "touch_stats"
        only_log_channels = view_id == "log_channels"
        only_rolling_lr = view_id == "rolling_lr"
        highlight_extrema = view_id == "emas_extrema"

        fig = self._build_price_figure(
            df,
            events,
            geometry,
            fib_levels,
            show_touch_focus=show_touch_focus,
            only_log_channels=only_log_channels,
            only_rolling_lr=only_rolling_lr,
            highlight_extrema=highlight_extrema,
            touched_indicators=touched_indicators
        )
        # Basic title annotation
        if hasattr(res, "windows") and "full" in res.windows:
            full_perf = res.windows["full"].performance
        else:
            full_perf = {}

        title_parts = [res.instrument, view_id]
        if full_perf:
            mu = full_perf.get("mu_ann")
            sig = full_perf.get("sigma_ann")
            sh = full_perf.get("sharpe")
            if mu is not None and not pd.isna(mu):
                title_parts.append(f"AnnRet {mu:.2%}")
            if sig is not None and not pd.isna(sig):
                title_parts.append(f"AnnVol {sig:.2%}")
            if sh is not None and not pd.isna(sh):
                title_parts.append(f"Sharpe {sh:.2f}")
        fig.update_layout(
            title=" | ".join(title_parts),
            hovermode="x unified",
        )
        return fig

    def _build_price_figure(
        self,
        df: pd.DataFrame,
        events: List[Dict[str, Any]],
        geometry: Dict[str, Any],
        fib_levels: Dict[str, float],
        show_touch_focus: bool,
        only_log_channels: bool,
        only_rolling_lr: bool,
        highlight_extrema: bool,
        touched_indicators: Optional[Set[str]]
    ) -> go.Figure:
        has_volume = (
            "Volume" in df.columns
            and not self.feature_flags.get("no_volume_bars", False)
            and not self.clean_plot
        )
        has_rsi = (
            "RSI_14" in df.columns
            and not self.feature_flags.get("no_rsi_subplot", False)
            and not self.clean_plot
        )
        has_macd = (
            {"MACD", "MACD_Signal", "MACD_Hist"}.issubset(df.columns)
            and not self.feature_flags.get("no_macd_subplot", False)
            and not self.clean_plot
        )

        n_rows = 1 + int(has_volume) + int(has_rsi) + int(has_macd)
        row_heights = [1.0] * n_rows
        if n_rows >= 2:
            row_heights[0] = 0.6
            for i in range(1, n_rows):
                row_heights[i] = 0.4 / (n_rows - 1)

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
        )
        row_price = 1
        x = df.index
        
        # Helper for Smart Legend visibility
        def get_vis(name):
            if not self.smart_legends or touched_indicators is None:
                return True
            return True if name in touched_indicators else 'legendonly'

        # Price
        if not self.feature_flags.get("no_candlesticks", False) and {"Open", "High", "Low", "Close"}.issubset(
            df.columns
        ):
            fig.add_trace(
                go.Candlestick(
                    x=x,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Price",
                    showlegend=True,
                ),
                row=row_price,
                col=1,
            )
        elif "Close" in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df["Close"], name="Close", mode="lines"),
                row=row_price,
                col=1,
            )

        # Midpoint scatter
        if self.feature_flags.get("plot_midpoint_scatter", False) and {"High", "Low"}.issubset(df.columns):
            mid = (df["High"] + df["Low"]) / 2.0
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mid,
                    name="Midpoint",
                    mode="markers",
                    marker=dict(size=3),
                ),
                row=row_price,
                col=1,
            )

        # MAs
        for p in [9, 10, 20, 50, 100, 200, 300]:
            col = f"SMA_{p}"
            flag_name = f"no_ma{p}"
            if col in df.columns and not self.feature_flags.get(flag_name, False):
                fig.add_trace(
                    go.Scatter(x=x, y=df[col], mode="lines", name=col, line=dict(width=1), visible=get_vis(col)),
                    row=row_price,
                    col=1,
                )

        # Bollinger
        if not self.feature_flags.get("no_boll", False):
            for col in [
                "BB_Upper_1std",
                "BB_Upper_2std",
                "BB_Lower_1std",
                "BB_Lower_2std",
            ]:
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(x=x, y=df[col], mode="lines", name=col, line=dict(width=1), visible=get_vis(col)),
                        row=row_price,
                        col=1,
                    )

        # Fibonacci levels
        if fib_levels and not self.feature_flags.get("no_fib", False) and len(df) > 0:
            x_min = df.index.min()
            x_max = df.index.max()
            for label, level in fib_levels.items():
                fib_name = f"Fib {label}"
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[level, level],
                        mode="lines",
                        name=fib_name,
                        line=dict(width=1, dash="dot"),
                        visible=get_vis(fib_name) if self.smart_legends else get_vis(label) # check both variants
                    ),
                    row=row_price,
                    col=1,
                )

        # Regressions - numeric & multi-timeframe (unless focusing solely on log or rolling)
        if not only_log_channels and not only_rolling_lr and not self.feature_flags.get("no_lrc", False):
            if "Reg_Num_Center" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=df["Reg_Num_Center"],
                        mode="lines",
                        name="Reg_Num_Center",
                        line=dict(width=1),
                    ),
                    row=row_price,
                    col=1,
                )
            for col in df.columns:
                name = str(col)
                if name.startswith("Reg_Num_High_") or name.startswith("Reg_Num_Low_"):
                    fig.add_trace(
                        go.Scatter(x=x, y=df[col], mode="lines", name=name, line=dict(width=1), visible=get_vis(name)),
                        row=row_price,
                        col=1,
                    )
            # Multi-timeframe
            for prefix in self.feature_flags.get("reg_timeframe_names", []):
                center_col = f"{prefix}_Linear_Reg"
                if center_col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=df[center_col],
                            mode="lines",
                            name=center_col,
                            line=dict(width=1),
                        ),
                        row=row_price,
                        col=1,
                    )

        # Log channels view
        if only_log_channels:
            for col in df.columns:
                if col.startswith("LogReg"):
                    fig.add_trace(
                        go.Scatter(x=x, y=df[col], mode="lines", name=col, line=dict(width=1), visible=get_vis(col)),
                        row=row_price,
                        col=1,
                    )

        # Rolling LR view
        if only_rolling_lr:
            for span in [12, 20, 50, 100, 150]:
                col = f"LRcurve_{span}"
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(x=x, y=df[col], mode="lines", name=col, line=dict(width=1)),
                        row=row_price,
                        col=1,
                    )

        # Geometry lines
        for line_info in geometry.get("lines", []):
            name = line_info.get("name")
            if name and name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=df[name],
                        mode="lines",
                        name=name,
                        line=dict(width=1, dash="dash"),
                    ),
                    row=row_price,
                    col=1,
                )

        # Extrema markers if wanted
        if highlight_extrema and "Close" in df.columns:
            # last 200 and last 30 day extrema
            tail200 = df["Close"].tail(200)
            if not tail200.empty:
                max_idx = tail200.idxmax()
                min_idx = tail200.idxmin()
                fig.add_trace(
                    go.Scatter(
                        x=[max_idx],
                        y=[df.loc[max_idx, "Close"]],
                        mode="markers",
                        name="200d_max",
                        marker=dict(symbol="triangle-up", size=10),
                    ),
                    row=row_price,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[min_idx],
                        y=[df.loc[min_idx, "Close"]],
                        mode="markers",
                        name="200d_min",
                        marker=dict(symbol="triangle-down", size=10),
                    ),
                    row=row_price,
                    col=1,
                )
            tail30 = df["Close"].tail(30)
            if not tail30.empty:
                max_idx = tail30.idxmax()
                min_idx = tail30.idxmin()
                fig.add_trace(
                    go.Scatter(
                        x=[max_idx],
                        y=[df.loc[max_idx, "Close"]],
                        mode="markers",
                        name="30d_max",
                        marker=dict(symbol="triangle-up", size=9),
                    ),
                    row=row_price,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[min_idx],
                        y=[df.loc[min_idx, "Close"]],
                        mode="markers",
                        name="30d_min",
                        marker=dict(symbol="triangle-down", size=9),
                    ),
                    row=row_price,
                    col=1,
                )

        # Events: buy/sell/stars/spikes/touches
        if events:
            # buy
            buy_dates = [e["date"] for e in events if e.get("event_type") == "buy"]
            buy_prices = [e["price"] for e in events if e.get("event_type") == "buy"]
            if buy_dates:
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode="markers",
                        name="Buy",
                        marker=dict(symbol="triangle-up", size=10),
                    ),
                    row=row_price,
                    col=1,
                )
            # sell
            sell_dates = [e["date"] for e in events if e.get("event_type") == "sell"]
            sell_prices = [e["price"] for e in events if e.get("event_type") == "sell"]
            if sell_dates:
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode="markers",
                        name="Sell",
                        marker=dict(symbol="triangle-down", size=10),
                    ),
                    row=row_price,
                    col=1,
                )
            
            # Legacy Touches (Wicks)
            if not self.feature_flags.get("no_wicks", False):
                wick_dates = [e["date"] for e in events if e.get("event_type") in ("regression_touch", "legacy_lrc_touch")]
                wick_prices = [e["price"] for e in events if e.get("event_type") in ("regression_touch", "legacy_lrc_touch")]
                if wick_dates:
                    fig.add_trace(
                         go.Scatter(
                            x=wick_dates,
                            y=wick_prices,
                            mode="markers",
                            name="Reg Wick Touch",
                            marker=dict(symbol="x", size=8, color="blue"),
                        ),
                        row=row_price,
                        col=1
                    )

            # Legacy Body Touches (if enabled)
            if self.feature_flags.get("detect_body_touches", False):
                body_dates = [e["date"] for e in events if e.get("event_type") == "regression_body_touch"]
                body_prices = [e["price"] for e in events if e.get("event_type") == "regression_body_touch"]
                if body_dates:
                    fig.add_trace(
                         go.Scatter(
                            x=body_dates,
                            y=body_prices,
                            mode="markers",
                            name="Reg Body Touch",
                            marker=dict(symbol="cross", size=8, color="orange"),
                        ),
                        row=row_price,
                        col=1
                    )

            # Option B: Advanced Physics Traces
            # 1. Elephant
            ele_dates = [e["date"] for e in events if e.get("event_type") == "physics_elephant"]
            ele_prices = [e["price"] for e in events if e.get("event_type") == "physics_elephant"]
            if ele_dates:
                fig.add_trace(
                    go.Scatter(
                        x=ele_dates, y=ele_prices, mode="markers", name="Elephant",
                        marker=dict(symbol="square", size=8, color="purple", line=dict(width=1, color="black"))
                    ), row=row_price, col=1
                )
            
            # 2. Pinbar
            pin_dates = [e["date"] for e in events if e.get("event_type") == "physics_pinbar"]
            pin_prices = [e["price"] for e in events if e.get("event_type") == "physics_pinbar"]
            if pin_dates:
                fig.add_trace(
                    go.Scatter(
                        x=pin_dates, y=pin_prices, mode="markers", name="Pinbar",
                        marker=dict(symbol="diamond", size=7, color="gold", line=dict(width=1, color="black"))
                    ), row=row_price, col=1
                )
            
            # 3. Streaks (9-count)
            # Up
            s9u_dates = [e["date"] for e in events if e.get("event_type") == "streak_9_up"]
            s9u_prices = [e["price"] for e in events if e.get("event_type") == "streak_9_up"]
            if s9u_dates:
                 fig.add_trace(
                    go.Scatter(
                        x=s9u_dates, y=s9u_prices, mode="text", name="9-Up",
                        text=["9"] * len(s9u_dates), textposition="top center",
                        textfont=dict(color="green", size=14, weight="bold")
                    ), row=row_price, col=1
                )
            # Down
            s9d_dates = [e["date"] for e in events if e.get("event_type") == "streak_9_down"]
            s9d_prices = [e["price"] for e in events if e.get("event_type") == "streak_9_down"]
            if s9d_dates:
                 fig.add_trace(
                    go.Scatter(
                        x=s9d_dates, y=s9d_prices, mode="text", name="9-Down",
                        text=["9"] * len(s9d_dates), textposition="bottom center",
                        textfont=dict(color="red", size=14, weight="bold")
                    ), row=row_price, col=1
                )

            # 4. Regimes
            nr7_dates = [e["date"] for e in events if e.get("event_type") == "pattern_nr7"]
            nr7_prices = [e["price"] for e in events if e.get("event_type") == "pattern_nr7"]
            if nr7_dates:
                fig.add_trace(
                    go.Scatter(
                        x=nr7_dates, y=nr7_prices, mode="markers", name="NR7",
                        marker=dict(symbol="circle", size=6, color="blue", opacity=0.7)
                    ), row=row_price, col=1
                )
            
            # 5. Gaps
            gap_dates = [e["date"] for e in events if e.get("event_type") in ("gap_bull", "gap_bear")]
            gap_prices = [e["price"] for e in events if e.get("event_type") in ("gap_bull", "gap_bear")]
            if gap_dates:
                 fig.add_trace(
                    go.Scatter(
                        x=gap_dates, y=gap_prices, mode="markers", name="Gaps",
                        marker=dict(symbol="line-ew", size=10, color="gray", line=dict(width=2))
                    ), row=row_price, col=1
                )


            # stars (with color by direction)
            if not self.feature_flags.get("no_stars", False):
                up_dates: List[Any] = []
                up_prices: List[float] = []
                up_sizes: List[float] = []
                down_dates: List[Any] = []
                down_prices: List[float] = []
                down_sizes: List[float] = []

                for e in events:
                    if e.get("event_type") != "star":
                        continue
                    lbl = str(e.get("label", ""))
                    try:
                        extra = json.loads(e.get("extra") or "{}")
                        size = extra.get("size", 10)
                    except Exception:
                        size = 10
                    dt = e.get("date")
                    price = e.get("price")
                    if lbl.startswith("up_"):
                        up_dates.append(dt)
                        up_prices.append(price)
                        up_sizes.append(size)
                    elif lbl.startswith("down_"):
                        down_dates.append(dt)
                        down_prices.append(price)
                        down_sizes.append(size)

                if up_dates and not self.feature_flags.get("no_green_stars", False):
                    fig.add_trace(
                        go.Scatter(
                            x=up_dates,
                            y=up_prices,
                            mode="markers",
                            name="Up Stars",
                            marker=dict(symbol="star", size=up_sizes, color="green"),
                        ),
                        row=row_price,
                        col=1,
                    )
                if down_dates and not self.feature_flags.get("no_red_stars", False):
                    fig.add_trace(
                        go.Scatter(
                            x=down_dates,
                            y=down_prices,
                            mode="markers",
                            name="Down Stars",
                            marker=dict(symbol="star", size=down_sizes, color="red"),
                        ),
                        row=row_price,
                        col=1,
                    )

            # spikes
            if not self.feature_flags.get("no_volume_spikes", False):
                spike_dates = [e["date"] for e in events if e.get("event_type") == "volume_price_spike"]
                spike_prices = [e["price"] for e in events if e.get("event_type") == "volume_price_spike"]
                if spike_dates:
                    fig.add_trace(
                        go.Scatter(
                            x=spike_dates,
                            y=spike_prices,
                            mode="markers",
                            name="Spikes",
                            marker=dict(symbol="circle-open", size=9),
                        ),
                        row=row_price,
                        col=1,
                    )

        # Volume
        current_row = row_price + 1
        if has_volume and "Volume" in df.columns:
            # simple colored bars based on up/down
            if {"Open", "Close"}.issubset(df.columns):
                colors = np.where(df["Close"] >= df["Open"], "rgba(0,200,0,0.6)", "rgba(200,0,0,0.6)")
            else:
                colors = "rgba(0,0,150,0.6)"
            fig.add_trace(
                go.Bar(x=x, y=df["Volume"], name="Volume", marker=dict(color=colors)),
                row=current_row,
                col=1,
            )
            current_row += 1

        # RSI
        if has_rsi and "RSI_14" in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df["RSI_14"], mode="lines", name="RSI_14"),
                row=current_row,
                col=1,
            )
            fig.add_hline(y=70, line_dash="dot", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dot", row=current_row, col=1)
            current_row += 1

        # MACD
        if has_macd and {"MACD", "MACD_Signal", "MACD_Hist"}.issubset(df.columns):
            fig.add_trace(
                go.Scatter(x=x, y=df["MACD"], mode="lines", name="MACD"),
                row=current_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=df["MACD_Signal"], mode="lines", name="MACD_Signal"),
                row=current_row,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=x, y=df["MACD_Hist"], name="MACD_Hist"),
                row=current_row,
                col=1,
            )

        fig.update_layout(
            xaxis_rangeslider=dict(visible=not self.feature_flags.get("no_rangeslider", False) and not self.clean_plot),
            margin=dict(l=40, r=10, t=40, b=40),
            showlegend=True,
        )
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
        return fig


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Geometry-aware, regression-rich technical analysis dashboard."
    )

    # Core
    p.add_argument(
        "--tickers",
        type=str,
        default="SPY,QQQ,IWM",
        help="Comma-separated tickers or ratios (e.g. 'SPY,QQQ' or 'SPY/QQQ').",
    )
    # Alias so you can keep using --ticker
    p.add_argument(
        "--ticker",
        dest="tickers",
        type=str,
        help="Alias for --tickers.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./market_data",
        help="Root output directory.",
    )
    p.add_argument(
        "--lookback",
        type=float,
        default=1.0,
        help="Lookback in years (minimum history requirement).",
    )
    p.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Annualized risk-free rate for performance metrics.",
    )

    # Optional config
    p.add_argument(
        "--date-range",
        type=str,
        default=None,
        help="Either yfinance period (1y,6mo,3mo,max) or 'YYYY-MM-DD,YYYY-MM-DD'.",
    )
    p.add_argument(
        "--views",
        type=str,
        default="overview,touch_stats,log_channels,regression,last_year,emas_extrema,rolling_lr",
        help="Comma-separated view IDs or 'all'.",
    )
    p.add_argument(
        "--geometry-mode",
        type=str,
        choices=["auto", "prompt"],
        default="auto",
        help="Geometry mode: auto or prompt.",
    )
    p.add_argument(
        "--geometry-intervals",
        "--intervals",
        dest="geometry_intervals",
        type=str,
        default="6m,3m",
        help="Comma-separated geometry intervals like '6m,3m,1m'.",
    )
    p.add_argument(
        "--reg-timeframes",
        type=str,
        default="Long:144,Mid:60,Short:20",
        help="Comma-separated Name:Days mapping for regression timeframes.",
    )
    p.add_argument(
        "--legacy-lrc-days",
        type=int,
        default=0,
        help="Legacy LRC days (0 disables legacy LRC).",
    )
    p.add_argument(
        "--extrema-reg-lines",
        action="store_true",
        help="Enable extrema-based regression overlays (not fully implemented in plotting).",
    )
    p.add_argument(
        "--extrema-windows",
        type=str,
        default="30,90,180",
        help="Comma-separated day windows for extrema-based regressions.",
    )
    p.add_argument(
        "--extrema-n",
        type=int,
        default=7,
        help="Number of extreme points per extrema-based regression.",
    )
    p.add_argument(
        "--extrema-order",
        type=int,
        default=5,
        help="Order for scipy.signal.argrelextrema.",
    )
    p.add_argument(
        "--extrema-project-days",
        type=int,
        default=30,
        help="Forward projection days for extrema regression lines.",
    )
    p.add_argument(
        "--plot-midpoint-scatter",
        action="store_true",
        help="Plot (High+Low)/2 midpoint markers on price panel.",
    )
    p.add_argument(
        "--clean-plot",
        action="store_true",
        help="Single full-screen price panel, no volume/RSI/MACD/rangeslider.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file (CLI overrides file values).",
    )
    
    # --- LEGACY FEATURES RESTORED ---
    p.add_argument(
        "--smart-legends",
        action="store_true",
        help="Restore legacy 'Smart Legend' logic: hides regression lines that were never touched."
    )
    p.add_argument(
        "--detect-body-touches",
        action="store_true",
        help="Enable legacy distinction between 'Body Touches' and 'Wick Touches'."
    )
    p.add_argument(
        "--touch-tolerance",
        type=float,
        default=0.0,
        help="Percentage tolerance for touches (e.g. 0.001 for 0.1%). Defaults to 0.0 (strict).",
    )
    
    # --- OPTION B: NEW PHYSICS FLAGS ---
    p.add_argument("--no-physics", action="store_true", help="Disable Candle Physics (Elephants, Pinbars).")
    p.add_argument("--no-streaks", action="store_true", help="Disable 9-Count Streaks.")
    p.add_argument("--no-patterns", action="store_true", help="Disable NR7/Inside/Outside pattern detection.")
    p.add_argument("--no-gaps", action="store_true", help="Disable Gap detection.")

    # Feature toggle flags (--no-*)
    # Trend / regression / geometry
    p.add_argument("--no-lrc", action="store_true", help="Disable multi-timeframe regression overlays.")
    p.add_argument(
        "--no-legacy-lrc-plot",
        action="store_true",
        help="Disable plotting legacy LRC line & bands (analysis still computed).",
    )
    p.add_argument(
        "--no-legacy-lrc-wicks",
        action="store_true",
        help="Disable legacy LRC wick touch markers (not separately plotted).",
    )
    p.add_argument(
        "--no-geometry",
        action="store_true",
        help="Disable geometry lines/projections/intersections entirely.",
    )

    # Fib / Bollinger / MAs
    p.add_argument("--no-fib", action="store_true", help="Disable Fib levels.")
    p.add_argument("--no-fib-wicks", action="store_true", help="Disable Fib wick touch markers.")
    p.add_argument("--no-boll", action="store_true", help="Disable Bollinger bands.")
    p.add_argument("--no-ma-touches", action="store_true", help="Disable MA body touch markers.")
    p.add_argument("--no-candlesticks", action="store_true", help="Use line instead of candlesticks.")
    p.add_argument("--no-open", action="store_true", help="Disable open markers (not separately plotted).")
    p.add_argument("--no-high", action="store_true", help="Disable high markers (not separately plotted).")
    p.add_argument("--no-low", action="store_true", help="Disable low markers (not separately plotted).")
    p.add_argument("--no-close", action="store_true", help="Disable close markers (not separately plotted).")

    for period in [10, 20, 50, 100, 200, 300]:
        p.add_argument(
            f"--no-ma{period}",
            action="store_true",
            help=f"Disable SMA_{period} overlay.",
        )

    p.add_argument("--no-wicks", action="store_true", help="Disable regression wick-touch markers.")

    # Stars & colors
    p.add_argument("--no-stars", action="store_true", help="Disable all consecutive-sequence star markers.")
    p.add_argument("--no-red-stars", action="store_true", help="Disable red down-stars.")
    p.add_argument("--no-green-stars", action="store_true", help="Disable green up-stars.")

    # Volume / subplots
    p.add_argument("--no-volume-bars", action="store_true", help="Disable volume bar subplot.")
    p.add_argument("--no-volume-spikes", action="store_true", help="Disable volume/price spike markers.")

    # Indicators / UX
    p.add_argument("--no-rsi-subplot", action="store_true", help="Disable RSI subplot.")
    p.add_argument("--no-macd-subplot", action="store_true", help="Disable MACD subplot.")
    p.add_argument("--no-rangeslider", action="store_true", help="Disable x-axis rangeslider.")

    # Logging / UX
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )
    p.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the main dashboard HTML in default browser when done.",
    )

    return p


def merge_config_with_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """If --config is given, merge JSON config values into args unless overridden on CLI."""
    if not getattr(args, "config", None):
        return args
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        return args
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return args

    defaults = parser.parse_args([])

    for key, value in cfg.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == getattr(defaults, key):
            setattr(args, key, value)
    return args


def configure_logging(level_str: str) -> logging.Logger:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("geom_dashboard")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args = merge_config_with_args(args, parser)

    logger = configure_logging(args.log_level)

    tickers = parse_tickers_string(args.tickers)
    if not tickers:
        logger.error("No tickers specified.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_start, date_end, period_from_range = parse_date_range_arg(args.date_range)
    if date_start and date_end and date_start >= date_end:
        logger.error("Invalid --date-range start >= end.")
        sys.exit(1)

    if period_from_range:
        period = period_from_range
    else:
        period = default_period_for_lookback(float(args.lookback))

    reg_timeframes = parse_reg_timeframes(args.reg_timeframes)
    geometry_intervals = parse_geometry_intervals(args.geometry_intervals)
    extrema_windows = parse_extrema_windows(args.extrema_windows)

    feature_flags: Dict[str, Any] = {
        "no_lrc": args.no_lrc,
        "no_legacy_lrc_plot": args.no_legacy_lrc_plot,
        "no_legacy_lrc_wicks": args.no_legacy_lrc_wicks,
        "no_geometry": args.no_geometry,
        "no_fib": args.no_fib,
        "no_fib_wicks": args.no_fib_wicks,
        "no_boll": args.no_boll,
        "no_ma_touches": args.no_ma_touches,
        "no_candlesticks": args.no_candlesticks,
        "no_open": args.no_open,
        "no_high": args.no_high,
        "no_low": args.no_low,
        "no_close": args.no_close,
        "no_wicks": args.no_wicks,
        "no_stars": args.no_stars,
        "no_red_stars": args.no_red_stars,
        "no_green_stars": args.no_green_stars,
        "no_volume_bars": args.no_volume_bars,
        "no_volume_spikes": args.no_volume_spikes,
        "no_rsi_subplot": args.no_rsi_subplot,
        "no_macd_subplot": args.no_macd_subplot,
        "no_rangeslider": args.no_rangeslider,
        "plot_midpoint_scatter": args.plot_midpoint_scatter,
        "reg_timeframe_names": list(reg_timeframes.keys()),
        # Legacy Feature Flags
        "smart_legends": args.smart_legends,
        "detect_body_touches": args.detect_body_touches,
        "touch_tolerance": args.touch_tolerance,
        # Option B Flags
        "no_physics": args.no_physics,
        "no_streaks": args.no_streaks,
        "no_patterns": args.no_patterns,
        "no_gaps": args.no_gaps,
    }

    data_ingestion = DataIngestion(
        output_dir=output_dir,
        lookback_years=float(args.lookback),
        date_start=date_start,
        date_end=date_end,
        period=period,
        logger=logger,
    )

    analysis = FinancialAnalysis(
        risk_free_rate=float(args.risk_free_rate),
        reg_timeframes=reg_timeframes,
        legacy_lrc_days=int(args.legacy_lrc_days),
        geometry_mode=args.geometry_mode,
        geometry_intervals=geometry_intervals,
        extrema_windows=extrema_windows,
        extrema_n=int(args.extrema_n),
        extrema_order=int(args.extrema_order),
        extrema_project_days=int(args.extrema_project_days),
        feature_flags=feature_flags,
        logger=logger,
    )

    renderer = DashboardRenderer(
        output_dir=output_dir,
        views=args.views,
        feature_flags=feature_flags,
        clean_plot=args.clean_plot,
        logger=logger,
    )

    for inst in tickers:
        logger.info("Processing instrument %s", inst)
        df = data_ingestion.get_instrument_df(inst)
        if df.empty:
            logger.warning("No data for %s, skipping.", inst)
            continue
        result = analysis.run_for_instrument(inst, df)
        renderer.register_instrument(inst, result)

    main_html = renderer.render_all_dashboards()

    if args.open_browser:
        try:
            path_str = str(main_html.resolve())
            if sys.platform.startswith("darwin"):
                os.system(f'open "{path_str}"')
            elif sys.platform.startswith("linux"):
                os.system(f'xdg-open "{path_str}"')
            elif sys.platform.startswith("win"):
                os.system(f'start "" "{path_str}"')
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to open browser: %s", exc)


if __name__ == "__main__":
    main()
