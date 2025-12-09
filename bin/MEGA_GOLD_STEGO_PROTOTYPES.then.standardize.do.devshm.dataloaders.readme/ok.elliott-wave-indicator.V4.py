#!/usr/bin/env python3
import os
import sys

# CONSTRAINT: Prevent __pycache__ creation on disk
sys.dont_write_bytecode = True

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# CONSTRAINT: Import local data retrieval module
try:
    import data_retrieval as dr  # expects data_retrieval.py on PYTHONPATH or alongside this script
except ImportError:
    print("Error: data_retrieval.py not found.")
    sys.exit(1)

# ---------- Enforced save+reload ----------

def retrieve_to_csv_and_reload(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = dr.load_or_download_ticker(ticker, start=start, end=end)
    df = dr.fix_yfinance_dataframe(df)
    if 'Date' in df.columns:
        to_save = df.copy()
    else:
        idx_name = df.index.name or 'Date'
        to_save = df.reset_index().rename(columns={idx_name: 'Date'})
    
    # CONSTRAINT: Output to /dev/shm via data_retrieval logic
    out_dir = dr.create_output_directory(ticker)
    out_csv = os.path.join(out_dir, f"{ticker}.csv")
    to_save.to_csv(out_csv, index=False)
    
    re = pd.read_csv(out_csv, parse_dates=['Date'])
    re.sort_values('Date', inplace=True)
    return re

# ---------- Helpers ----------

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date'])
        out.set_index('Date', inplace=True)
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        out.index.name = 'Date'
    return out

# ---------- Indicators ----------

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = ensure_datetime_index(data)
    data['20DMA'] = data['Close'].rolling(window=20).mean()
    data['50DMA'] = data['Close'].rolling(window=50).mean()
    data['9DMA']  = data['Close'].rolling(window=9).mean()

    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std  = data['Close'].rolling(window=20).std()
    data['BB_Middle']     = rolling_mean
    data['BB_Upper_2std'] = rolling_mean + (rolling_std * 2.0)
    data['BB_Lower_2std'] = rolling_mean - (rolling_std * 2.0)
    return data

# ---------- Elliott Wave Identification ----------

def identify_elliott_waves(data: pd.DataFrame, price_column='Close', window_size=120) -> pd.DataFrame:
    result = ensure_datetime_index(data).copy()
    prices = result[price_column].values

    peak_indices, _ = find_peaks(prices, distance=window_size//4, prominence=np.std(prices)*0.5)
    trough_indices, _ = find_peaks(-prices, distance=window_size//4, prominence=np.std(prices)*0.5)

    result['is_peak'] = False
    result['is_trough'] = False
    result['wave_point'] = np.nan
    result['wave_label'] = ''
    result['wave_type'] = ''

    if len(result) > 0:
        result.iloc[peak_indices, result.columns.get_loc('is_peak')] = True
        result.iloc[trough_indices, result.columns.get_loc('is_trough')] = True

    combined_idx = np.sort(np.concatenate([peak_indices, trough_indices]))
    kind_by_idx = {idx: ('peak' if idx in peak_indices else 'trough') for idx in combined_idx}
    all_points = [(idx, kind_by_idx[idx]) for idx in combined_idx]

    if len(all_points) >= 9:
        wave_count = 0
        last_type = None
        correction_phase = False

        for idx, point_type in all_points:
            if wave_count >= 8:
                break
            if not correction_phase:
                if wave_count == 0:
                    wave_count = 1
                    result.iloc[idx, result.columns.get_loc('wave_point')] = prices[idx]
                    result.iloc[idx, result.columns.get_loc('wave_label')] = '1'
                    result.iloc[idx, result.columns.get_loc('wave_type')] = 'Start'
                    last_type = point_type
                elif (point_type != last_type) and (wave_count < 5):
                    wave_count += 1
                    label = str(wave_count)
                    result.iloc[idx, result.columns.get_loc('wave_point')] = prices[idx]
                    result.iloc[idx, result.columns.get_loc('wave_label')] = label
                    result.iloc[idx, result.columns.get_loc('wave_type')] = 'Down' if wave_count % 2 == 0 else 'Up'
                    last_type = point_type
                    if wave_count == 5:
                        correction_phase = True
            else:
                if (point_type != last_type) and (wave_count < 8):
                    wave_count += 1
                    label_map = {6: 'A', 7: 'B', 8: 'C'}
                    label = label_map.get(wave_count, '')
                    result.iloc[idx, result.columns.get_loc('wave_point')] = prices[idx]
                    result.iloc[idx, result.columns.get_loc('wave_label')] = label
                    result.iloc[idx, result.columns.get_loc('wave_type')] = 'Down' if label in ('A', 'C') else 'Up'
                    last_type = point_type

    wave1 = result.index[result['wave_label'] == '1']
    wave2 = result.index[result['wave_label'] == '2']
    wave3 = result.index[result['wave_label'] == '3']
    wave4 = result.index[result['wave_label'] == '4']
    wave5 = result.index[result['wave_label'] == '5']

    if not (wave1.empty or wave3.empty or wave5.empty):
        w1 = result.loc[wave1[0], price_column]
        w3 = result.loc[wave3[0], price_column]
        w5 = result.loc[wave5[0], price_column]
        result['rule1_valid'] = (w3 > w1) or (w3 > w5)

    if not (wave1.empty or wave2.empty or wave4.empty):
        w1p = result.loc[wave1[0], price_column]
        w4p = result.loc[wave4[0], price_column]
        trend_up = True
        if not wave3.empty:
            w3p = result.loc[wave3[0], price_column]
            trend_up = (w1p < w3p)
        result['rule2_valid'] = (w4p > w1p) if trend_up else (w4p < w1p)

    return result

# ---------- Plotting ----------

def plot_elliott_waves(data: pd.DataFrame, ticker: str) -> None:
    data = ensure_datetime_index(data)
    candlestick = go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlesticks')
    ma_20 = go.Scatter(x=data.index, y=data['20DMA'], mode='lines', name='20DMA', line=dict(color='blue'))
    ma_50 = go.Scatter(x=data.index, y=data['50DMA'], mode='lines', name='50DMA', line=dict(color='red'))
    bb_mid   = go.Scatter(x=data.index, y=data['BB_Middle'],    mode='lines', name='BB_Middle',    line=dict(color='gray', dash='dash'))
    bb_upper = go.Scatter(x=data.index, y=data['BB_Upper_2std'], mode='lines', name='BB_Upper_2std', line=dict(color='lightgreen', dash='dot'))
    bb_lower = go.Scatter(x=data.index, y=data['BB_Lower_2std'], mode='lines', name='BB_Lower_2std', line=dict(color='pink', dash='dot'))

    wave_points = data[data['wave_label'].astype(str) != '']
    wave_colors = {'1': 'orange', '2': 'purple', '3': 'cyan', '4': 'yellow', '5': 'magenta', 'A': 'red', 'B': 'green', 'C': 'blue'}

    wave_traces = []
    for label, color in wave_colors.items():
        pts = wave_points[wave_points['wave_label'] == label]
        if not pts.empty:
            wave_traces.append(go.Scatter(x=pts.index, y=pts['Close'], mode='markers+text', name=f'Wave {label}',
                                          marker=dict(color=color, size=12, symbol='circle'),
                                          text=label, textposition="top center", textfont=dict(size=14, color=color)))

    if len(wave_points) > 1:
        wave_traces.append(go.Scatter(x=wave_points.index, y=wave_points['Close'], mode='lines', name='Elliott Wave Pattern',
                                      line=dict(color='white', width=1, dash='dot'), showlegend=True))

    traces = [candlestick, ma_20, ma_50, bb_mid, bb_upper, bb_lower] + wave_traces

    y_min = min(data['Low'].min(), data['BB_Lower_2std'].min()) * 0.95
    y_max = max(data['High'].max(), data['BB_Upper_2std'].max()) * 1.05

    layout = go.Layout(
        title=f'{ticker} Stock Price with Elliott Wave Analysis',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price ($)', range=[y_min, y_max]),
        template='plotly_dark',
        legend=dict(x=1.0, y=1.0),
        hovermode='x'
    )

    fig = go.Figure(data=traces, layout=layout)

    wave_points_sorted = wave_points.sort_index()
    if len(wave_points_sorted) > 1:
        for i in range(len(wave_points_sorted) - 1):
            current_point = wave_points_sorted.iloc[i]
            next_point    = wave_points_sorted.iloc[i + 1]
            mid_x = current_point.name + (next_point.name - current_point.name) / 2
            mid_y = (current_point['Close'] + next_point['Close']) / 2
            if current_point['Close'] < next_point['Close']:
                arrow_y, color = -40, "rgba(0, 255, 0, 0.7)"
                direction = "Upwave"
            else:
                arrow_y, color = 40, "rgba(255, 0, 0, 0.7)"
                direction = "Downwave"
            fig.add_annotation(x=mid_x, y=mid_y, text=f"{direction} {current_point['wave_label']}-{next_point['wave_label']}",
                               showarrow=True, arrowhead=1, ax=0, ay=arrow_y, bgcolor=color,
                               font=dict(size=10, color="white", family="Arial Black"), bordercolor="white", borderwidth=1)

    if 'rule1_valid' in data.columns and 'rule2_valid' in data.columns:
        r1 = bool(data['rule1_valid'].dropna().iloc[-1]) if not data['rule1_valid'].dropna().empty else False
        r2 = bool(data['rule2_valid'].dropna().iloc[-1]) if not data['rule2_valid'].dropna().empty else False
        validation_text = (f"Elliott Wave Rules Validation:<br>"
                           f"- Rule 1 (Wave 3 not shortest): {'✓' if r1 else '✗'}<br>"
                           f"- Rule 2 (Wave 4 no overlap with Wave 1): {'✓' if r2 else '✗'}")
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", text=validation_text, showarrow=False,
                           font=dict(size=12, color="white"), align="left", bgcolor="rgba(50,50,50,0.8)",
                           bordercolor="white", borderwidth=1, borderpad=4)

    fig.show()

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description='Analyze stock with Elliott Wave indicators (save+reload daily CSV).')
    parser.add_argument('ticker', type=str, help='Ticker, e.g., AAPL')
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='YYYY-MM-DD')
    parser.add_argument('--end_date',   type=str, default='2025-12-31', help='YYYY-MM-DD')
    parser.add_argument('--window_size', type=int, default=120, help='Peak/trough detection window')
    args = parser.parse_args()

    df = retrieve_to_csv_and_reload(args.ticker.upper(), args.start_date, args.end_date)
    if df is None or df.empty:
        print(f"Failed to load data for {args.ticker} in range [{args.start_date}, {args.end_date}].")
        sys.exit(1)

    df = calculate_indicators(df)
    df = identify_elliott_waves(df, price_column='Close', window_size=args.window_size)
    plot_elliott_waves(df, args.ticker.upper())

if __name__ == "__main__":
    main()
