#!/usr/bin/env python3
# SCRIPTNAME: mas.lrc.vwap.multifan.peak.troughs.ath.v35.py
# AUTHOR: Michael Derby
# DATE:   November 20, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK
"""
Adaptive Anchored VWAP engine (Wick-Anchored & Noise Filtered)
PLUS Multi Linear Regression Channel (LRC)
PLUS All Time High (ATH) Detector (Top 3 Highs -> Max Close)
PLUS Price Frequency Distribution (Most Common Price Levels)
PLUS Multi Moving Averages (SMA, EMA, SMMA, WMA, VWMA)
PLUS Dual Bollinger Band Modes (Short Term & Long Term)
PLUS Ichimoku Cloud
PLUS OHLC Price Dots (Classic Overlay)
PLUS GROUPED LEGEND (Collapsible & Draggable)

UPDATES (v35):
- Added --show-ohlc-dots flag.
- Implements "classic" OHLC markers from ichimoku.py:
  Open(Cyan), High(Green), Low(Yellow), Close(White), Midpoint(Orange).
"""

import argparse
import logging
import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd

# Try importing local dependencies
try:
    import data_retrieval as dr
    import options_data_retrieval as odr
except ImportError as e:
    sys.exit(f"CRITICAL ERROR: Could not import local dependencies. {e}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def make_default_config() -> Dict[str, Any]:
    return {
        "peak_order": 10,
        "peaks_only": False,
        "slope_window": 20, "slope_confirm_bars": 3, 
        "pivot_window": 50, "pivot_z_thresh": 1.5,
        "vol_short_window": 20, "vol_long_window": 60, 
        "vol_up_ratio": 1.25, "vol_down_ratio": 0.80,
        "min_bars_between_anchors": 5, 
        "vol_filter_window": 60, "vol_filter_percentile": 0.6,
        "require_volume": True,
        # LRC Config
        "show_lrc": False,
        "lrc_length": 252,
        # ATH Config
        "show_ath": False,
        # Freq Config
        "show_freq": False,
        "freq_lookback": 252,
        "freq_bins": 100,
        # MA Config
        "show_ma": False,
        "ma_type": "EMA",
        # BB Config
        "show_long_bbs": False,
        "show_short_bbs": False,
        # Ichimoku Config
        "show_ichimoku": False,
        # OHLC Dots Config
        "show_ohlc_dots": False
    }

def sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys_to_int = ["peak_order", "slope_window", "slope_confirm_bars", "pivot_window", 
                   "vol_short_window", "vol_long_window", "min_bars_between_anchors", 
                   "vol_filter_window", "lrc_length", "freq_lookback", "freq_bins"]
    clean_cfg = cfg.copy()
    for k in keys_to_int:
        if k in clean_cfg:
            try: clean_cfg[k] = int(clean_cfg[k])
            except: clean_cfg[k] = 10
    return clean_cfg

# ---------------------------------------------------------------------------
# Color Helpers (Hex -> RGBA)
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hex_to_rgba_string(hex_color: str, opacity: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {opacity})"

# ---------------------------------------------------------------------------
# Math Helpers
# ---------------------------------------------------------------------------

def find_peaks_troughs_ohlc(df: pd.DataFrame, order: int = 5):
    highs = df['High'].to_numpy()
    lows  = df['Low'].to_numpy()
    peaks, troughs = [], []
    n = len(df)
    if order < 1: order = 1
    for i in range(order, n - order):
        if highs[i] == np.max(highs[i - order : i + order + 1]):
            peaks.append((i, highs[i]))
        if lows[i] == np.min(lows[i - order : i + order + 1]):
            troughs.append((i, lows[i]))
    return peaks, troughs

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    x = np.arange(window, dtype=float)
    def _slope(arr):
        if np.any(np.isnan(arr)): return np.nan
        y = arr.astype(float)
        cov = np.sum((x - x.mean()) * (y - y.mean()))
        var = np.sum((x - x.mean()) ** 2)
        return 0.0 if var == 0 else cov / var
    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)

def _realized_vol(closes: pd.Series, window: int) -> pd.Series:
    window = int(window)
    lr = np.log(closes).diff().replace([np.inf, -np.inf], np.nan)
    return lr.rolling(window=window, min_periods=window).std()

# ---------------------------------------------------------------------------
# Shared MA Logic
# ---------------------------------------------------------------------------

def _calc_ma_series(series: pd.Series, length: int, ma_type: str, vol_series: Optional[pd.Series] = None) -> pd.Series:
    if ma_type == "SMA":
        return series.rolling(window=length).mean()
    elif ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "SMMA": # RMA
        return series.ewm(alpha=1.0/length, adjust=False).mean()
    elif ma_type == "WMA":
        def wma_calc(x):
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window=length).apply(wma_calc, raw=True)
    elif ma_type == "VWMA":
        if vol_series is None: return series.rolling(window=length).mean()
        pv = (series * vol_series).rolling(window=length).sum()
        v = vol_series.rolling(window=length).sum()
        return pv / v
    else:
        return series.rolling(window=length).mean()

# ---------------------------------------------------------------------------
# Multi MA Logic
# ---------------------------------------------------------------------------

def calculate_moving_averages(df: pd.DataFrame, ma_type: str) -> Dict[int, pd.Series]:
    lengths = [5, 9, 20, 50, 100, 200]
    results = {}
    for length in lengths:
        results[length] = _calc_ma_series(df['Close'], length, ma_type, df.get('Volume'))
    return results

# ---------------------------------------------------------------------------
# Dual Mode Bollinger Bands Logic
# ---------------------------------------------------------------------------

def calculate_bollinger_bands(df: pd.DataFrame, mode: str = "long") -> List[Dict[str, Any]]:
    configs = []
    results = []
    
    if mode == "long":
        # === LONG TERM (Fan of 50) ===
        colors = ["#4FC3F7", "#4DB6AC", "#81C784", "#AED581", "#81C784", "#4DB6AC", "#4FC3F7"]
        multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        close = df['Close']
        for i, mult in enumerate(multipliers):
            base_col = colors[i % len(colors)]
            fill_col = hex_to_rgba_string(base_col, 0.10)  
            line_col = "rgba(255, 255, 255, 0.5)"
            basis = _calc_ma_series(close, 50, "SMA") 
            std_dev = close.rolling(window=50).std()
            upper = basis + (mult * std_dev)
            lower = basis - (mult * std_dev)
            results.append({
                "name": f"L_BB{i+1}_{mult}",
                "len": 50, "mult": mult, 
                "fill_color": fill_col,
                "line_color": line_col, 
                "basis": basis, "upper": upper, "lower": lower
            })
            
    elif mode == "short":
        # === SHORT TERM (Strictly TradingView Dialogue Only) ===
        cfg1 = { "len": 5, "mult": 0.5, "src": df['Close'], "type": "EMA", "line_col": "#000000", "fill_col": "#FFF9C4", "name": "S_BB_5_Close" }
        cfg2 = { "len": 20, "mult": 0.5, "src": df['Low'], "type": "EMA", "line_col": "#2962FF", "fill_col": "#FFCC80", "name": "S_BB_20_Low" }
        custom_sets = [cfg1, cfg2]
        for c in custom_sets:
            basis = _calc_ma_series(c['src'], c['len'], c['type'])
            std = c['src'].rolling(window=c['len']).std() 
            results.append({
                "name": c['name'], "len": c['len'], "mult": c['mult'],
                "line_color": c['line_col'], 
                "fill_color": hex_to_rgba_string(c['fill_col'], 0.50), 
                "basis": basis, 
                "upper": basis + (c['mult'] * std), 
                "lower": basis - (c['mult'] * std)
            })

    return results

# ---------------------------------------------------------------------------
# Ichimoku Cloud Logic
# ---------------------------------------------------------------------------

def calculate_ichimoku(df: pd.DataFrame) -> Dict[str, Any]:
    high = df['High']
    low = df['Low']
    close = df['Close']
    high_9 = high.rolling(window=9).max()
    low_9 = low.rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    high_26 = high.rolling(window=26).max()
    low_26 = low.rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    high_52 = high.rolling(window=52).max()
    low_52 = low.rolling(window=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)
    chikou_span = close.shift(-26)
    return {
        "tenkan_sen": tenkan_sen, "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a, "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span
    }

# ---------------------------------------------------------------------------
# ATH Detector Logic
# ---------------------------------------------------------------------------

def calculate_ath_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    top_3_highs = df.nlargest(3, 'High')
    target_close = top_3_highs['Close'].max()
    return {
        "target_close": target_close,
        "top_days": top_3_highs.index.tolist()
    }

# ---------------------------------------------------------------------------
# Price Frequency Logic
# ---------------------------------------------------------------------------

def _get_top_two_modes(data: np.ndarray, step: float) -> List[float]:
    if len(data) == 0: return []
    binned = np.round(data / step) * step
    counts = Counter(binned)
    most_common = counts.most_common(2)
    return [mc[0] for mc in most_common]

def calculate_price_frequency(df: pd.DataFrame, lookback: int, bins: int) -> Dict[str, Any]:
    if len(df) < 10: return {}
    subset = df.iloc[-lookback:].copy()
    period_high = subset['High'].max()
    period_low = subset['Low'].min()
    period_range = period_high - period_low
    if period_range == 0: step = 0.01
    else: step = period_range / bins
    
    modes = {}
    modes['High'] = _get_top_two_modes(subset['High'].values, step)
    modes['Low'] = _get_top_two_modes(subset['Low'].values, step)
    modes['Open'] = _get_top_two_modes(subset['Open'].values, step)
    modes['Close'] = _get_top_two_modes(subset['Close'].values, step)
    
    all_vals = np.concatenate([
        subset['High'].values, subset['Low'].values,
        subset['Open'].values, subset['Close'].values
    ])
    modes['Combined'] = _get_top_two_modes(all_vals, step)
    
    return {
        "modes": modes, "step": step,
        "start_date": subset.index[0], "end_date": subset.index[-1]
    }

# ---------------------------------------------------------------------------
# LRC Logic
# ---------------------------------------------------------------------------

def calculate_lrc_stats(df: pd.DataFrame, length: int) -> Dict[str, Any]:
    if len(df) < length: return {}
    subset = df.iloc[-length:].copy()
    y = subset['Close'].values
    x = np.arange(length)
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    reg_line = m * x + b
    residuals = y - reg_line
    std_dev = np.sqrt(np.sum(residuals**2) / (length if length > 0 else 1))
    pearson_r = np.corrcoef(x, y)[0, 1]
    return {
        "m": m, "b": b, "std_dev": std_dev, "pearson_r": pearson_r,
        "reg_line": reg_line, "subset_index": subset.index
    }

# ---------------------------------------------------------------------------
# VWAP Logic
# ---------------------------------------------------------------------------

def detect_triggers(df, cfg):
    closes = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=df.index)
    peaks_list, troughs_list = find_peaks_troughs_ohlc(df, order=int(cfg["peak_order"]))
    is_major_peak = pd.Series(False, index=df.index)
    is_major_trough = pd.Series(False, index=df.index)
    for i, _ in peaks_list: is_major_peak.iloc[i] = True
    for i, _ in troughs_list: is_major_trough.iloc[i] = True

    slope = _rolling_slope(closes, int(cfg["slope_window"]))
    sign = np.sign(slope)
    is_inflection = pd.Series(False, index=closes.index)
    conf = int(cfg["slope_confirm_bars"])
    if len(closes) > (int(cfg["slope_window"]) + conf):
        for i in range(int(cfg["slope_window"]) + conf, len(closes)):
            prev = sign.iloc[i - conf - 1]
            sl = sign.iloc[int(i - conf + 1) : int(i + 1)] 
            if prev != 0 and not np.isnan(prev):
                if np.all(sl == sl.iloc[-1]) and sl.iloc[-1] != prev:
                    is_inflection.iloc[i] = True

    slope_std = slope.rolling(int(cfg["pivot_window"])).std()
    z_score = (slope.diff().abs() / slope_std).replace([np.inf, -np.inf], np.nan)
    is_pivot = (z_score >= cfg["pivot_z_thresh"]) & z_score.notna()

    s_vol = _realized_vol(closes, int(cfg["vol_short_window"]))
    l_vol = _realized_vol(closes, int(cfg["vol_long_window"]))
    ratio = s_vol / l_vol
    is_up = (ratio > cfg["vol_up_ratio"]) & (ratio.shift(1) <= cfg["vol_up_ratio"])
    is_down = (ratio < cfg["vol_down_ratio"]) & (ratio.shift(1) >= cfg["vol_down_ratio"])

    pct = cfg["vol_filter_percentile"]
    if pct <= 0: gate = pd.Series(True, index=volume.index)
    else:
        q = volume.rolling(int(cfg["vol_filter_window"]), min_periods=1).quantile(pct)
        gate = volume >= q

    return {
        "is_major_peak": is_major_peak, "is_major_trough": is_major_trough,
        "slope": slope, "is_inflection": is_inflection, "is_pivot": is_pivot,
        "is_regime_up": is_up, "is_regime_down": is_down, 
        "vol_gate": gate, "vol_ratio": ratio
    }

def compute_adaptive_avwap(df: pd.DataFrame, cfg: Dict[str, Any]):
    cfg = sanitize_config(cfg)
    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if 'Volume' not in df.columns: df['Volume'] = 1.0
    triggers = detect_triggers(df, cfg)
    px_close = df["Close"].values
    px_high = df["High"].values
    px_low = df["Low"].values
    vol = df["Volume"].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    cum_pv = np.cumsum(px_close * vol)
    cum_v = np.cumsum(vol)
    n = len(px_close)
    avwap_list = [None] * n
    anchors = []
    current_anchor_pos = 0
    
    anchors.append({"idx": df.index[0], "position": 0, "reason": "initial", "price": float(px_close[0])})
    avwap_list[0] = px_close[0]
    
    last_anchor_pos = 0
    min_bars = int(cfg["min_bars_between_anchors"])
    peaks_only = bool(cfg.get("peaks_only", False))
    
    for i, 1 in range(1, n):
        reasons = []
        if triggers["is_major_peak"].iloc[i]: reasons.append("major_peak")
        if triggers["is_major_trough"].iloc[i]: reasons.append("major_trough")
        if not peaks_only:
            if triggers["is_inflection"].iloc[i]: reasons.append("slope_inflection")
            if triggers["is_pivot"].iloc[i]: reasons.append("regression_pivot")
            if triggers["is_regime_up"].iloc[i]: reasons.append("vol_regime_up")
            if triggers["is_regime_down"].iloc[i]: reasons.append("vol_regime_down")
        
        can_anchor = False
        if reasons:
            if (i - last_anchor_pos) >= min_bars:
                if "major_peak" in reasons or "major_trough" in reasons: can_anchor = True
                elif not peaks_only and triggers["vol_gate"].iloc[i]: can_anchor = True
        
        if can_anchor:
            current_anchor_pos = i
            last_anchor_pos = i
            if "major_peak" in reasons: anchor_p = float(px_high[i])
            elif "major_trough" in reasons: anchor_p = float(px_low[i])
            else: anchor_p = float(px_close[i])
            anchors.append({"idx": df.index[i], "position": i, "reason": ",".join(reasons), "price": anchor_p})
            
        if current_anchor_pos == 0:
            v_sum = cum_v[i]
            pv_sum = cum_pv[i]
        else:
            prev_idx = current_anchor_pos - 1
            v_sum = cum_v[i] - cum_v[prev_idx]
            pv_sum = cum_pv[i] - cum_pv[prev_idx]
        val = pv_sum / v_sum if v_sum > 0 else np.nan
        avwap_list[i] = val

    df_out = df.copy()
    df_out["AVWAP"] = avwap_list
    return df_out, anchors

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def calculate_single_anchor_vwap(df, start_idx_int):
    px = df["Close"].values
    vol = df["Volume"].fillna(0).values
    px_slice = px[start_idx_int:]
    vol_slice = vol[start_idx_int:]
    cum_pv = np.cumsum(px_slice * vol_slice)
    cum_v = np.cumsum(vol_slice)
    vwap_slice = np.full_like(cum_pv, np.nan)
    mask = cum_v > 0
    vwap_slice[mask] = cum_pv[mask] / cum_v[mask]
    full_vwap = np.full(len(df), np.nan)
    full_vwap[start_idx_int:] = vwap_slice
    return full_vwap

def plot_interactive(df: pd.DataFrame, ticker: str, anchors: List[Dict[str, Any]], 
                     out_dir: str,  # CONSTRAINT: Pass output directory explicitly
                     lrc_data: Optional[Dict] = None, 
                     ath_data: Optional[Dict] = None,
                     freq_data: Optional[Dict] = None,
                     ma_data: Optional[Dict] = None,
                     ma_type: str = "EMA",
                     bb_long: Optional[List[Dict]] = None,
                     bb_short: Optional[List[Dict]] = None,
                     ichimoku_data: Optional[Dict] = None,
                     show_ohlc_dots: bool = False):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logging.warning("Plotly not installed.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC', showlegend=False
    ), row=1, col=1)

    # 1a. OHLC Dots (Classic Overlay)
    if show_ohlc_dots:
        print("Adding OHLC Dots (Open=Cyan, Close=White, High=Green, Low=Yellow)...")
        fig.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='markers', name='Open', marker=dict(color='cyan', size=4), legendgroup="OHLC"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='markers', name='Close', marker=dict(color='white', size=4), legendgroup="OHLC"))
        fig.add_trace(go.Scatter(x=df.index, y=df['High'], mode='markers', name='High', marker=dict(color='green', size=4), legendgroup="OHLC"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Low'], mode='markers', name='Low', marker=dict(color='yellow', size=4), legendgroup="OHLC"))
        midpoint = (df['High'] + df['Low']) / 2.0
        fig.add_trace(go.Scatter(x=df.index, y=midpoint, mode='markers', name='Midpoint', marker=dict(color='orange', size=4), legendgroup="OHLC"))

    # 2. VWAP Fan
    active_anchors = anchors[-30:] 
    colors = ["#FFA500", "#00FFFF", "#FF00FF", "#00FF00", "#FFFF00", "#FF4500", "#1E90FF"]
    print(f"Generating fan plots for {len(active_anchors)} anchors...")
    
    first_vwap = True
    for i, anchor in enumerate(active_anchors):
        start_pos = anchor['position']
        vwap_series = calculate_single_anchor_vwap(df, start_pos)
        color = colors[i % len(colors)]
        symbol = 'triangle-down' if "major_peak" in anchor['reason'] else 'triangle-up' if "major_trough" in anchor['reason'] else 'circle'
        
        fig.add_trace(go.Scatter(
            x=df.index, y=vwap_series, mode='lines',
            line=dict(color=color, width=1.2), opacity=0.8,
            name=f"AVWAP {str(anchor['idx'].date())}",
            legendgroup="VWAP Fan",
            legendgrouptitle_text="VWAP Fan" if first_vwap else None
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[anchor['idx']], y=[anchor['price']], mode='markers',
            marker=dict(symbol=symbol, size=10, color=color, line=dict(width=1, color='black')),
            showlegend=False, hoverinfo='text', text=f"{anchor['reason']}",
            legendgroup="VWAP Fan"
        ), row=1, col=1)
        first_vwap = False

    # 3. Multi Moving Averages
    if ma_data:
        ma_styles = {
            5:   {"color": "black", "width": 1},
            9:   {"color": "red",   "width": 1},
            20:  {"color": "cyan",  "width": 1},
            50:  {"color": "blue",  "width": 2},
            100: {"color": "orange","width": 1},
            200: {"color": "green", "width": 2}
        }
        first_ma = True
        for length, series in ma_data.items():
            style = ma_styles.get(length, {"color": "gray", "width": 1})
            fig.add_trace(go.Scatter(
                x=df.index, y=series, mode='lines',
                line=dict(color=style["color"], width=style["width"]),
                name=f"{ma_type} {length}",
                legendgroup="Moving Averages",
                legendgrouptitle_text="Moving Averages" if first_ma else None
            ), row=1, col=1)
            first_ma = False
        print(f"Added {len(ma_data)} Moving Averages ({ma_type}).")

    # 4. Bollinger Bands Plotter
    def plot_bb_set(bbs, group_name):
        if not bbs: return
        print(f"Adding {len(bbs)} {group_name} Bollinger Band sets...")
        first_bb = True
        for bb in bbs:
            fig.add_trace(go.Scatter(
                x=df.index, y=bb['upper'], mode='lines',
                line=dict(width=0.5, color=bb['line_color']),
                showlegend=False, hoverinfo='skip',
                legendgroup=group_name
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=bb['lower'], mode='lines',
                line=dict(width=0.5, color=bb['line_color']),
                fill='tonexty', fillcolor=bb['fill_color'], 
                showlegend=False, hoverinfo='skip',
                legendgroup=group_name
            ), row=1, col=1)
            
            should_plot_basis = True
            if "Long" in group_name and bb['mult'] != 2.0: should_plot_basis = False
            
            if should_plot_basis:
                fig.add_trace(go.Scatter(
                    x=df.index, y=bb['basis'], mode='lines',
                    line=dict(width=1, color=bb['line_color'], dash='dot'),
                    name=f"{bb['name']}",
                    legendgroup=group_name,
                    legendgrouptitle_text=group_name if first_bb else None
                ), row=1, col=1)
            first_bb = False

    plot_bb_set(bb_long, "Long BB Fan")
    plot_bb_set(bb_short, "Short BB Set")

    # 5. Ichimoku Cloud Plotter
    if ichimoku_data:
        print("Adding Ichimoku Cloud...")
        
        fig.add_trace(go.Scatter(
            x=df.index, y=ichimoku_data['tenkan_sen'], mode='lines',
            line=dict(color='red', width=1.5), name='Tenkan-sen',
            legendgroup='Ichimoku', legendgrouptitle_text='Ichimoku Cloud'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=ichimoku_data['kijun_sen'], mode='lines',
            line=dict(color='blue', width=1.5), name='Kijun-sen',
            legendgroup='Ichimoku'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=ichimoku_data['chikou_span'], mode='lines',
            line=dict(color='green', width=1.5), name='Chikou Span',
            legendgroup='Ichimoku'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=ichimoku_data['senkou_span_a'], mode='lines',
            line=dict(width=0), showlegend=False, hoverinfo='skip',
            legendgroup='Ichimoku'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=ichimoku_data['senkou_span_b'], mode='lines',
            line=dict(width=0), 
            fill='tonexty', fillcolor='rgba(0, 255, 0, 0.75)', 
            name='Kumo (Cloud)', legendgroup='Ichimoku'
        ), row=1, col=1)

    # 6. LRC Overlay
    if lrc_data:
        x_dates = lrc_data['subset_index']
        reg_line = lrc_data['reg_line']
        std_dev = lrc_data['std_dev']
        fig.add_trace(go.Scatter(x=x_dates, y=reg_line, mode='lines', line=dict(color='red', width=2),
                                 name=f"LRC (R={lrc_data['pearson_r']:.2f})",
                                 legendgroup="LRC", legendgrouptitle_text="LinReg Channel"), row=1, col=1)
        for mult in np.arange(0.25, 3.25, 0.25):
            fig.add_trace(go.Scatter(x=x_dates, y=reg_line + std_dev*mult, mode='lines',
                                     line=dict(color='cyan', width=.5), showlegend=False, hoverinfo='skip', legendgroup="LRC"), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_dates, y=reg_line - std_dev*mult, mode='lines',
                                     line=dict(color='magenta', width=.5), showlegend=False, hoverinfo='skip', legendgroup="LRC"), row=1, col=1)

    # 7. ATH Overlay
    if ath_data:
        val = ath_data['target_close']
        fig.add_hline(y=val, line_width=3, line_color="yellow", line_dash="solid",
                      annotation_text=f"ATH Detect: {val:.2f}", annotation_position="top left")
        print(f"ATH Line added at {val:.2f}")

    # 8. Freq Overlay
    if freq_data:
        start_d = freq_data['start_date']
        end_d = freq_data['end_date']
        modes = freq_data['modes']
        c_map = {'High': 'green', 'Low': 'red', 'Open': 'blue', 'Close': 'orange', 'Combined': 'purple'}
        first_freq = True
        for cat, vals in modes.items():
            col = c_map.get(cat, 'gray')
            width = 2 if cat == 'Combined' else 1
            for v in vals:
                fig.add_trace(go.Scatter(
                    x=[start_d, end_d], y=[v, v], mode='lines',
                    line=dict(color=col, width=width, dash='solid'),
                    name=f"Freq {cat}", hoverinfo='name+y',
                    legendgroup="Price Freq",
                    legendgrouptitle_text="Price Frequencies" if first_freq else None
                ), row=1, col=1)
                first_freq = False
        print(f"Frequency Lines added for {len(modes)} categories.")

    # 9. Volume
    cols = ['red' if o > c else 'green' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=cols, name='Volume', opacity=0.3), row=2, col=1)

    # CONFIG: Enable 'editable' to allow dragging legend
    config = dict(
        editable=True,
        scrollZoom=True,
        displayModeBar=True
    )

    fig.update_layout(
        title=dict(text=f"{ticker.upper()} - Integrated Analysis", font=dict(size=24)),
        template="plotly_dark", xaxis_rangeslider_visible=False, height=900,
        legend=dict(
            orientation="v", # Vertical stack
            yanchor="top", y=1,
            xanchor="left", x=1.02, # Start detached on the right side
            groupclick="toggleitem"
        ),
        hovermode="x unified"
    )
    
    # SAVE to HTML before showing
    # CONSTRAINT: Ensure output path is in /dev/shm
    out_file = os.path.join(out_dir, f"{ticker}_analysis.html")
    fig.write_html(out_file, config=config)
    print(f"Interactive chart saved to: {out_file}")
    
    # Show
    fig.show(config=config)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_data(args):
    if args.start and not args.end: args.end = datetime.now().strftime('%Y-%m-%d')
    if args.start and args.end: return dr.load_or_download_ticker(args.ticker, start=args.start, end=args.end)
    return dr.load_or_download_ticker(args.ticker, period=args.period)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--period", default="2y")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--ensure-options", action="store_true")
    
    # Toggles
    p.add_argument("--peak-order", type=int, default=10)
    p.add_argument("--peaks-only", action="store_true")
    p.add_argument("--show-lrc", action="store_true")
    p.add_argument("--lrc-length", type=int, default=252)
    
    # NEW TOGGLES
    p.add_argument("--show-ath", action="store_true", help="Show ATH Detector Line")
    p.add_argument("--show-freq", action="store_true", help="Show Price Frequency Lines")
    p.add_argument("--freq-lookback", type=int, default=252)
    p.add_argument("--freq-bins", type=int, default=100)
    
    # MA Toggles
    p.add_argument("--show-ma", action="store_true", help="Show Multi Moving Averages")
    p.add_argument("--ma-type", default="EMA", choices=["SMA", "EMA", "SMMA", "WMA", "VWMA"], help="Moving Average Type")

    # BB Toggles
    p.add_argument("--show-long-bbs", action="store_true", help="Show Long Term Fan (Length 50, Blue/Green)")
    p.add_argument("--show-short-bbs", action="store_true", help="Show Short Term Bands (5 & 20 Period, Multi-Color)")
    p.add_argument("--show-bb", action="store_true", help="Alias for --show-long-bbs")

    # Ichimoku Toggle
    p.add_argument("--show-ichimoku", action="store_true", help="Show Ichimoku Cloud")

    # OHLC Dots Toggle
    p.add_argument("--show-ohlc-dots", action="store_true", help="Show OHLC Price Dots (Classic Overlay)")

    # Legacy
    p.add_argument("--slope-window", type=int, default=20)
    p.add_argument("--slope-confirm", type=int, default=3)
    p.add_argument("--pivot-window", type=int, default=50)
    p.add_argument("--pivot-z", type=float, default=1.5)
    p.add_argument("--vol-short", type=int, default=20)
    p.add_argument("--vol-long", type=int, default=60)
    
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    try:
        df = load_data(args)
        if df is None or df.empty: raise ValueError("No data")
        
        cfg = make_default_config()
        cfg.update({k: v for k, v in vars(args).items() if k in cfg})
        if args.peaks_only: cfg["peaks_only"] = True
        
        # Handle Alias
        if args.show_bb: cfg["show_long_bbs"] = True
        
        # 1. VWAP
        df_out, anchors = compute_adaptive_avwap(df, cfg)
        
        # 2. LRC
        lrc_data = None
        if args.show_lrc:
            logging.info("Calculating LRC...")
            lrc_data = calculate_lrc_stats(df_out, args.lrc_length)
            
        # 3. ATH
        ath_data = None
        if args.show_ath:
            logging.info("Calculating ATH Detector...")
            ath_data = calculate_ath_stats(df_out)
            
        # 4. Freq
        freq_data = None
        if args.show_freq:
            logging.info(f"Calculating Price Frequencies (Lookback: {args.freq_lookback})...")
            freq_data = calculate_price_frequency(df_out, args.freq_lookback, args.freq_bins)

        # 5. Multi MA
        ma_data = None
        if args.show_ma:
            logging.info(f"Calculating Multi MA ({args.ma_type})...")
            ma_data = calculate_moving_averages(df_out, args.ma_type)

        # 6. Multi BB (Long)
        bb_long = None
        if cfg["show_long_bbs"]:
            logging.info("Calculating Long Term BB Fan (Len 50)...")
            bb_long = calculate_bollinger_bands(df_out, mode="long")

        # 7. Multi BB (Short)
        bb_short = None
        if args.show_short_bbs:
            logging.info("Calculating Short Term BBs (5 & 20)...")
            bb_short = calculate_bollinger_bands(df_out, mode="short")

        # 8. Ichimoku
        ichimoku_data = None
        if args.show_ichimoku:
            logging.info("Calculating Ichimoku Cloud...")
            ichimoku_data = calculate_ichimoku(df_out)

        # Prepare output directory in /dev/shm
        out_dir = dr.create_output_directory(args.ticker)

        # Save CSV
        if not args.no_save:
            path = os.path.join(out_dir, f"{args.ticker}_full_analysis.csv")
            df_out.to_csv(path)
            logging.info(f"Saved to {path}")
            
        if args.ensure_options:
            try: odr.ensure_some_options_cached(args.ticker)
            except: pass
                
        # Plot (Pass output directory explicitly)
        if args.plot:
            plot_interactive(df_out, args.ticker, anchors, out_dir, lrc_data, ath_data, 
                             freq_data, ma_data, args.ma_type, bb_long, bb_short,
                             ichimoku_data, args.show_ohlc_dots)

    except Exception as e:
        logging.error(f"Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
