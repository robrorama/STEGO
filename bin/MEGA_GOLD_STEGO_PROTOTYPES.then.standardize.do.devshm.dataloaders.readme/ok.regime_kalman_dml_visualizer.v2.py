#!/usr/bin/env python3
# SCRIPTNAME: regime_kalman_dml_visualizer.v1b.py
# AUTHOR: Michael Derby
# DATE: November 23, 2025

import argparse
import os
import sys
import logging
import webbrowser
import numpy as np
import pandas as pd

from datetime import datetime

try:
    import data_retrieval as dr
except Exception as e:
    print("ERROR importing data_retrieval.py:", e)
    sys.exit(1)

# -----------------------------
# Gaussian HMM (2 states)
# -----------------------------
def fit_hmm_gaussian(returns: np.ndarray, n_states=2, n_iter=60):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_states, covariance_type="full", max_iter=n_iter)
    gm.fit(returns.reshape(-1,1))
    probs = gm.predict_proba(returns.reshape(-1,1))
    states = gm.predict(returns.reshape(-1,1))
    return probs, states, gm

# -----------------------------
# Simple Kalman Filter
# -----------------------------
def kalman_local_level(y: np.ndarray, Q_ratio=0.1):
    n = len(y)
    level = np.zeros(n)
    P = 1.0
    Q = Q_ratio * np.var(y)
    R = np.var(np.diff(y)) if np.var(np.diff(y))>0 else 1e-4

    level[0] = y[0]
    innov = np.zeros(n)

    for t in range(1, n):
        pred = level[t-1]
        P = P + Q
        K = P / (P + R)
        level[t] = pred + K * (y[t] - pred)
        innov[t] = y[t] - pred
        P = (1 - K) * P
    return level, innov

# -----------------------------
# DML with FULL NAN CLEANING
# -----------------------------
def dml_effect(Y, T, X):
    mask = (
        np.isfinite(Y) &
        np.isfinite(T) &
        np.all(np.isfinite(X), axis=1)
    )

    Yc = Y[mask]
    Tc = T[mask]
    Xc = X[mask]

    if len(Yc) < 200:
        print("WARNING: Not enough clean data for DML. Skipping.")
        return np.nan

    from sklearn.model_selection import KFold
    from sklearn.ensemble import GradientBoostingRegressor

    kf = KFold(n_splits=5, shuffle=False)
    u_list = []
    v_list = []

    for tr, te in kf.split(Xc):
        X_train, X_test = Xc[tr], Xc[te]
        Y_train, Y_test = Yc[tr], Yc[te]
        T_train, T_test = Tc[tr], Tc[te]

        m_y = GradientBoostingRegressor()
        m_y.fit(X_train, Y_train)
        u = Y_test - m_y.predict(X_test)

        m_t = GradientBoostingRegressor()
        m_t.fit(X_train, T_train)
        v = T_test - m_t.predict(X_test)

        u_list.append(u)
        v_list.append(v)

    u_all = np.concatenate(u_list)
    v_all = np.concatenate(v_list)

    theta = np.dot(u_all, v_all) / np.dot(v_all, v_all)
    return theta

# -----------------------------
# Anchored VWAP Engine
# -----------------------------
def compute_avwap(df, anchors):
    out = {}
    px = df["Close"].values
    vol = df["Volume"].values
    n = len(df)

    for a in anchors:
        vwap = np.full(n, np.nan)
        cum_pv = 0
        cum_v = 0
        for i in range(a, n):
            cum_pv += px[i]*vol[i]
            cum_v += vol[i]
            vwap[i] = cum_pv / max(cum_v,1e-9)
        out[a] = vwap
    return out

def detect_anchors(df, lookback=5):
    prices = df["Close"].values
    anchors=[]
    for i in range(lookback, len(prices)-lookback):
        window = prices[i-lookback:i+lookback]
        if prices[i] == np.min(window) or prices[i] == np.max(window):
            anchors.append(i)
    return anchors

# -----------------------------
# PLOTLY CHARTS
# -----------------------------
def plot_regime_chart(df, hmm_probs, outfile):
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[1:],
        y=hmm_probs[:,0],
        mode="lines",
        name="State0 Probability"
    ))
    fig.add_trace(go.Scatter(
        x=df.index[1:],
        y=hmm_probs[:,1],
        mode="lines",
        name="State1 Probability"
    ))
    fig.update_layout(title="HMM Regimes", height=600)
    fig.write_html(outfile, auto_open=True)

def plot_kalman_chart(df, level, innov, outfile):
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=level, mode='lines', name='Kalman Level'))
    fig.add_trace(go.Scatter(x=df.index, y=innov, mode='lines', name='Innovations'))
    fig.update_layout(title="Kalman Filter", height=600)
    fig.write_html(outfile, auto_open=True)

def plot_candles_vwap(df, vwap_dict, outfile):
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="Candles"
    ))
    for a, v in vwap_dict.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=v, mode='lines', name=f"AVWAP {a}"
        ))
    fig.update_layout(title="Candles + Anchored VWAPs", height=800)
    fig.write_html(outfile, auto_open=True)

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    df = dr.load_or_download_ticker(args.ticker, args.period).dropna()

    df['ret'] = np.log(df['Close']).diff()
    ret = df['ret'].dropna().values

    hmm_probs, states, gm = fit_hmm_gaussian(ret)

    level, innov = kalman_local_level(df['Close'].values)

    Y = np.roll(df['ret'].values, -1)
    T = df['ret'].values
    X = np.column_stack([df['Volume'].values, level, innov])

    theta = dml_effect(Y, T, X)
    print("DML theta:", theta)

    anchors = detect_anchors(df)
    vwap_dict = compute_avwap(df, anchors)

    outdir = f"OUTPUT_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(outdir, exist_ok=True)

    plot_regime_chart(df, hmm_probs, os.path.join(outdir,"HMM.html"))
    plot_kalman_chart(df, level, innov, os.path.join(outdir,"Kalman.html"))
    plot_candles_vwap(df, vwap_dict, os.path.join(outdir,"Candles_VWAP.html"))

    print("Charts saved in →", outdir)

if __name__ == "__main__":
    main()

