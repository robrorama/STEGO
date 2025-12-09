#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SCRIPTNAME: gui.launcher.V4.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

"""
A Tkinter-based GUI to launch various financial analysis Python scripts.
V3.1 Updates: COMPLETE Script List, TreeView, Search, Scrollbars.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, font
import os
import subprocess
import threading
import queue

# --- Configuration: Define all scripts, their paths, and arguments ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# STRUCTURE: Nested Dictionary { "Category Name": { "Script Name": { config... } } }
SCRIPTS_CONFIG = {
    "Technical Analysis (Trends & Patterns)": {
        "Peak/Trough I-Bars": {
            "path": os.path.join(BASE_PATH, "PEAK_TROUGH_I_BARS_AND_SUPPORT_RESISTANCE_LINES", "peak_TROUGH_I_Bars.v1.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period (e.g. 1y):", "default": "1y"},
                {"name": "--date-range", "type": "entry", "label": "Date Range (YYYY-MM-DD,YYYY-MM-DD):"},
                {"name": "--order", "type": "entry", "label": "Peak/Trough Order:", "default": "5"},
                {"name": "--sr-last-n", "type": "entry", "label": "S/R Window (days):", "default": "30"},
                {"name": "--save-png", "type": "check", "label": "Save as PNG"},
                {"name": "--save-html", "type": "check", "label": "Save as HTML"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Browser"},
            ]
        },
        "Peak/Trough Circles": {
            "path": os.path.join(BASE_PATH, "PEAK_TROUGH_CIRCLES", "peakTrough.circles.v3.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "1y"},
                {"name": "--dates", "type": "entry", "label": "Date Range (YYYY-MM-DD,YYYY-MM-DD):"},
                {"name": "--order", "type": "entry", "label": "Pivot Order:", "default": "5"},
                {"name": "--short-days", "type": "entry", "label": "Short Trend (days):", "default": "60"},
                {"name": "--medium-days", "type": "entry", "label": "Medium Trend (days):", "default": "120"},
                {"name": "--long-days", "type": "entry", "label": "Long Trend (days):", "default": "252"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Browser"},
            ]
        },
        "Peak/Trough Divergence (RSI/MACD)": {
            "path": os.path.join(BASE_PATH, "PEAK_TROUGH_RSI_MACD", "peak.troughs.v22.macd.rsi.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "1y"},
                {"name": "--date-range", "type": "entry", "label": "Date Range (YYYY-MM-DD,YYYY-MM-DD):"},
                {"name": "--order", "type": "entry", "label": "Pivot Order:", "default": "5"},
                {"name": "--short-days", "type": "entry", "label": "Short Trend (days):", "default": "60"},
                {"name": "--medium-days", "type": "entry", "label": "Medium Trend (days):", "default": "120"},
                {"name": "--long-days", "type": "entry", "label": "Long Trend (days):", "default": "252"},
                {"name": "--save-html", "type": "check", "label": "Save HTML"},
                {"name": "--save-png", "type": "check", "label": "Save PNG"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Browser"},
            ]
        },
        "Ichimoku Chart": {
            "path": os.path.join(BASE_PATH, "ICHIMOKU", "ichimoku.v7.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--start_date", "type": "entry", "label": "Start (YYYY-MM-DD):", "default": "2023-01-01"},
                {"name": "--end_date", "type": "entry", "label": "End (YYYY-MM-DD):", "default": "2023-12-31"},
            ]
        },
        "Darvas Boxes": {
            "path": os.path.join(BASE_PATH, "DARVAS_BOXES", "darvas.v3.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "max"},
                {"name": "--no-browser", "type": "check", "label": "Don't Open Browser"},
            ]
        },
        "Point & Figure Charts": {
            "path": os.path.join(BASE_PATH, "POINT_AND_FIGURE_CHART", "point.figure.charts.v1.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "1y"},
                {"name": "--date-range", "type": "entry", "label": "Date Range (YYYY-MM-DD,YYYY-MM-DD):"},
                {"name": "--method", "type": "combo", "label": "Method:", "values": ["hilo", "close"], "default": "hilo"},
                {"name": "--box-size", "type": "entry", "label": "Fixed Box Size (e.g. 1.0):"},
                {"name": "--box-pct", "type": "entry", "label": "Box Percent (e.g. 1.0):"},
                {"name": "--reversal", "type": "entry", "label": "Reversal (Boxes):", "default": "3"},
                {"name": "--save-png", "type": "check", "label": "Save PNG"},
                {"name": "--save-csv", "type": "check", "label": "Save CSV"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Window"},
            ]
        },
        "Head & Shoulders Pattern": {
            "path": os.path.join(BASE_PATH, "HEAD_AND_SHOULDERS", "head.shoulders.v4.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "1y"},
                {"name": "--date-range", "type": "entry", "label": "Date Range (YYYY-MM-DD,YYYY-MM-DD):"},
                {"name": "--order", "type": "entry", "label": "Pivot Order:", "default": "5"},
                {"name": "--short-days", "type": "entry", "label": "Short Trend (days):", "default": "60"},
                {"name": "--medium-days", "type": "entry", "label": "Medium Trend (days):", "default": "120"},
                {"name": "--long-days", "type": "entry", "label": "Long Trend (days):", "default": "252"},
                {"name": "--save-html", "type": "check", "label": "Save HTML Reports"},
                {"name": "--save-png", "type": "check", "label": "Save PNG Images"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Browser"},
            ]
        },
        "Adaptive Anchored VWAP": {
            "path": os.path.join(BASE_PATH, "Adaptive_Anchored_VWAP", "Adaptive_Anchored_Volume_Weighted_Average_Profile.V34.py"),
            "args": [
                {"name": "--ticker", "type": "entry", "label": "Ticker:", "required": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "2y"},
                {"name": "--start", "type": "entry", "label": "Start (YYYY-MM-DD):"},
                {"name": "--end", "type": "entry", "label": "End (YYYY-MM-DD):"},
                {"name": "--plot", "type": "check", "label": "Generate & Show Plot (Required)", "default": "on"},
                {"name": "--peaks-only", "type": "check", "label": "Anchor to Peaks Only"},
                {"name": "--show-ohlc-dots", "type": "check", "label": "Show OHLC Dots"},
                {"name": "--show-lrc", "type": "check", "label": "Show LRC"},
                {"name": "--show-ath", "type": "check", "label": "Show ATH Line"},
                {"name": "--show-freq", "type": "check", "label": "Show Price Frequencies"},
                {"name": "--show-ma", "type": "check", "label": "Show Moving Averages"},
                {"name": "--show-long-bbs", "type": "check", "label": "Show Long-Term BB Fan"},
                {"name": "--show-short-bbs", "type": "check", "label": "Show Short-Term BBs"},
                {"name": "--show-ichimoku", "type": "check", "label": "Show Ichimoku Cloud"},
                {"name": "--lrc-length", "type": "entry", "label": "LRC Length:", "default": "252"},
                {"name": "--ma-type", "type": "combo", "label": "MA Type:", "values": ["EMA", "SMA", "SMMA", "WMA", "VWMA"], "default": "EMA"},
                {"name": "--peak-order", "type": "entry", "label": "Peak Order:", "default": "10"},
                {"name": "--no-save", "type": "check", "label": "Don't Save CSV"},
            ]
        },
        "Signals Inflection Points": {
            "path": os.path.join(BASE_PATH, "SIGNALS_INFLECTION_POINTS", "identify.signal.inflection.points.V3.plotly.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "6mo"},
                {"name": "--start", "type": "entry", "label": "Start (YYYY-MM-DD):"},
                {"name": "--end", "type": "entry", "label": "End (YYYY-MM-DD):"},
                {"name": "--ma", "type": "entry", "label": "MA Periods (space separated):", "default": "20 50 200"},
                {"name": "--bollinger-period", "type": "entry", "label": "Bollinger Period:", "default": "20"},
                {"name": "--std-dev", "type": "entry", "label": "Std Dev:", "default": "2.0"},
                {"name": "--regression-period", "type": "entry", "label": "Regression Period:", "default": "21"},
            ]
        },
        "PE Candlesticks": {
            "path": os.path.join(BASE_PATH, "PE_CANDLESTICKS", "PE_candlesticks.V3.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period (e.g. max, 5y):", "default": "max"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Browser"},
            ]
        },
        "MICRA Analysis": {
            "path": os.path.join(BASE_PATH, "MICRA.classic.V2", "micra.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--date-range", "type": "entry", "label": "Period (1y) or Range (YYYY-MM-DD,YYYY-MM-DD):", "default": "1y"},
                {"name": "--geometry-mode", "type": "combo", "label": "Geometry Mode:", "values": ["auto", "prompt"], "default": "auto"},
            ]
        },
    },

    "Volume Analysis": {
        "Volume Visualizer": {
            "path": os.path.join(BASE_PATH, "VOLUME_CIRCLES", "volume.visualizer.V2.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "period_or_start", "type": "entry", "label": "Period or Start (YYYY-MM-DD):", "positional": True, "default": "1y"},
                {"name": "end_date", "type": "entry", "label": "End Date (Optional):", "positional": True},
                {"name": "--zscore", "type": "entry", "label": "Z-Score Threshold:", "default": "2.0"},
                {"name": "--hv-mode", "type": "combo", "label": "High Volume Mode:", "values": ["zonly", "strict"], "default": "zonly"},
                {"name": "--rec-threshold", "type": "entry", "label": "Recurrence Threshold:", "default": "3"},
                {"name": "--vol-color", "type": "combo", "label": "Volume Color Rule:", "values": ["up", "down"], "default": "up"},
                {"name": "--no-tabs", "type": "check", "label": "Don't Open Browser Tabs"},
                {"name": "--no-gif", "type": "check", "label": "Don't Generate GIF"},
                {"name": "--extra-tabs", "type": "check", "label": "Show Extra Tabs"},
            ]
        },
        "Volume by Price": {
            "path": os.path.join(BASE_PATH, "VOLUME_BY_PRICE", "VolumeByPrice.V1.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "1y"},
                {"name": "--start", "type": "entry", "label": "Start (YYYY-MM-DD):"},
                {"name": "--end", "type": "entry", "label": "End (YYYY-MM-DD):"},
                {"name": "--bins", "type": "entry", "label": "Bins:", "default": "24"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Browser"},
            ]
        },
    },

    "Options Tools": {
        "Options: All Charts (Report)": {
            "path": os.path.join(BASE_PATH, "OPTIONS_VISUALIZER", "options_visualizer.v1.py"),
            "command": "charts all",
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--quantile", "type": "entry", "label": "Unusual Vol Quantile:", "default": "0.95"},
                {"name": "--iv-min", "type": "entry", "label": "IV Min (Optional):"},
                {"name": "--iv-max", "type": "entry", "label": "IV Max (Optional):"},
                {"name": "--smooth-iv", "type": "check", "label": "Smooth IV Surface"},
            ]
        },
        "Options: Analyze Spreads": {
            "path": os.path.join(BASE_PATH, "OPTIONS_VISUALIZER", "options_visualizer.v1.py"),
            "command": "analyze spreads",
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "weeks_to_expiry", "type": "entry", "label": "Expiry Index (0=Next):", "positional": True, "default": "0"},
                {"name": "--offset", "type": "entry", "label": "Price Offset ($):", "default": "0.0"},
                {"name": "--max-otm-percent", "type": "entry", "label": "Max OTM %:", "default": "5.0"},
                {"name": "--save-csv", "type": "check", "label": "Save to CSV"},
            ]
        },
        "Options: Analyze Streaks": {
            "path": os.path.join(BASE_PATH, "OPTIONS_VISUALIZER", "options_visualizer.v1.py"),
            "command": "analyze streaks",
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "max"},
                {"name": "--save-csv", "type": "check", "label": "Save to CSV"},
            ]
        },
        "Gamma/Vanna Exposure": {
            "path": os.path.join(BASE_PATH, "GAMMA_CHARMM_VISUALIZER", "gamma.charmm.visualizer.v4.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--risk_free", "type": "entry", "label": "Risk Free Rate:", "default": "0.05"},
                {"name": "--div_yield", "type": "entry", "label": "Dividend Yield:", "default": "0.0"},
                {"name": "--contract_size", "type": "entry", "label": "Contract Size:", "default": "100"},
                {"name": "--max_expirations", "type": "entry", "label": "Max Expirations (Int):"},
                {"name": "--strike_slice", "type": "entry", "label": "Term Structure Slice (Price):"},
                {"name": "--ensure_remote", "type": "check", "label": "Force Download Remote Chains"},
                {"name": "--animate", "type": "check", "label": "Generate Animation (MP4)"},
            ]
        },
    },

    "Comparisons & Correlations": {
         "Ratio Comparison (Clickable)": {
            "path": os.path.join(BASE_PATH, "CLICKABLE_RATIO_COMPARISON", "clickable_trends_compare_two.py"),
            "args": [
                {"name": "ticker1", "type": "entry", "label": "Ticker 1:", "positional": True},
                {"name": "ticker2", "type": "entry", "label": "Ticker 2:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period or Range:", "default": "1y"},
                {"name": "--timeframes", "type": "entry", "label": "Timeframes List:", "default": "1mo,3mo,6mo,1y,max"},
                {"name": "--renderer", "type": "entry", "label": "Renderer:", "default": "browser"},
                {"name": "--clickable", "type": "check", "label": "Enable Clickable Lines (Matplotlib)"},
                {"name": "--all-timeframes", "type": "check", "label": "Show All Timeframes"},
                {"name": "--no-normalized", "type": "check", "label": "Skip Normalized Plot"},
                {"name": "--no-ratio", "type": "check", "label": "Skip Ratio Plot"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Charts"},
                {"name": "--no-save", "type": "check", "label": "Don't Save PNGs"},
            ]
        },
        "Correlation & Lag Analysis": {
            "path": os.path.join(BASE_PATH, "CORRELATION_LAG_DETECTOR", "price.correlations.and.lagging.indicator.two.assets.v9.py"),
            "args": [
                {"name": "--ticker1", "type": "entry", "label": "Ticker 1 (Ref):", "required": True},
                {"name": "--ticker2", "type": "entry", "label": "Ticker 2 (Comp):", "required": True},
                {"name": "--no-show", "type": "check", "label": "Don't Open Browser"},
            ]
        },
    },
    
    "Macro & Fundamental": {
         "Commodities Analysis": {
            "path": os.path.join(BASE_PATH, "COMMODITIES", "commodities.v4.py"),
            "args": [
                {"name": "--mode", "type": "combo", "label": "Chart Mode:", "values": ["all", "plotly-grouped", "plotly-lrc", "plotly-combined"], "default": "all"},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "1y"},
                {"name": "--dates", "type": "entry", "label": "Date Range (YYYY-MM-DD,YYYY-MM-DD):"},
                {"name": "--log-channels", "type": "check", "label": "Use Log Scale for LRC"},
                {"name": "--no-open", "type": "check", "label": "Don't Open Browser"},
            ]
        },
        "Market Heatmap (S&P 500)": {
            "path": os.path.join(BASE_PATH, "HEATMAP_TICKERS", "revamp.heatmap.tickers.by.volume.downloads.first.v3.py"),
            "args": [
                {"name": "csv_path", "type": "entry", "label": "CSV Path (Optional):", "positional": True},
                {"name": "out_dir", "type": "entry", "label": "Output Dir (Optional):", "positional": True},
                {"name": "--delay", "type": "entry", "label": "Download Delay (s):", "default": "1.0"},
            ]
        },
         "Analyst Recommendations Pie Chart": {
            "path": os.path.join(BASE_PATH, "PIE_CHART_ANALYST_RECOMMENDATIONS", "unified.pie.charts.v2.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--no-browser", "type": "check", "label": "Don't Open Browser"},
            ]
        },
    },

    "Experimental / Misc": {
        "Mega Charting Suite": {
            "path": os.path.join(BASE_PATH, "MEGA_CHARTING", "mega_charting_suite.v1.py"),
            "args": [
                {"name": "--ticker", "type": "entry", "label": "Ticker:", "required": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "6mo"},
                {"name": "--start", "type": "entry", "label": "Start (YYYY-MM-DD):"},
                {"name": "--end", "type": "entry", "label": "End (YYYY-MM-DD):"},
                {"name": "--ratio-with", "type": "entry", "label": "Ratio Ticker (Compare):"},
                {"name": "--max-tabs", "type": "entry", "label": "Max Tabs:", "default": "128"},
                {"name": "--tab-delay-ms", "type": "entry", "label": "Tab Delay (ms):", "default": "60"},
                {"name": "--outdir", "type": "entry", "label": "Output Dir (Optional):"},
                {"name": "--include-all-patterns", "type": "check", "label": "Include ALL TA-Lib Patterns"},
                {"name": "--no-open", "type": "check", "label": "Don't Open Browser Tabs"},
                {"name": "--no-pdf", "type": "check", "label": "Don't Generate PDF Report"},
            ]
        },
        "Projectile/Trajectory Analysis": {
            "path": os.path.join(BASE_PATH, "PROJECTILES_INFLECTION_POINTS", "projectiles.v5.with.inflection.point.py"),
            "args": [
                {"name": "ticker", "type": "entry", "label": "Ticker:", "positional": True},
                {"name": "--period", "type": "entry", "label": "Period:", "default": "2y"},
                {"name": "--lookback", "type": "entry", "label": "Lookback (days):", "default": "90"},
                {"name": "--no-show", "type": "check", "label": "Don't Open Browser"},
            ]
        },
    }
}

class ScriptLauncherGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Financial Scripts Launcher V3.1")
        self.master.geometry("1100x800")

        self.arg_widgets = {}
        self.output_queue = queue.Queue()

        # --- Style Configuration ---
        style = ttk.Style(self.master)
        style.theme_use('clam')
        style.configure("TFrame", background="#2E2E2E")
        style.configure("TLabel", background="#2E2E2E", foreground="white", font=("Arial", 10))
        style.configure("TButton", background="#4A4A4A", foreground="white", font=("Arial", 10, "bold"))
        style.map("TButton", background=[('active', '#6A6A6A')])
        style.configure("TEntry", fieldbackground="#4A4A4A", foreground="white", insertbackground="white")
        style.configure("TCheckbutton", background="#2E2E2E", foreground="white", indicatorcolor="black")
        style.map("TCheckbutton", indicatorcolor=[('selected', '#007ACC')])
        
        # Treeview Specific Styling
        style.configure("Treeview", 
                        background="#4A4A4A", 
                        foreground="white", 
                        fieldbackground="#4A4A4A", 
                        font=("Arial", 10))
        style.map("Treeview", background=[('selected', '#007ACC')])
        style.configure("Treeview.Heading", background="#333333", foreground="white", font=("Arial", 10, "bold"))
        
        self.master.configure(bg="#2E2E2E")
        self.monospace_font = font.Font(family="Monospace", size=9)

        self._create_widgets()
        self.master.after(100, self.process_queue)

    def _create_widgets(self):
        # --- Main Layout ---
        main_pane = tk.PanedWindow(self.master, orient=tk.VERTICAL, sashrelief=tk.RAISED, bg="#2E2E2E", sashwidth=6)
        main_pane.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(top_frame, height=450) # Allocate more space to top

        bottom_frame = ttk.Frame(main_pane, padding=(10, 0, 10, 10))
        main_pane.add(bottom_frame)

        # --- Top Section (Tree and Arguments) ---
        top_pane = tk.PanedWindow(top_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg="#2E2E2E", sashwidth=6)
        top_pane.pack(fill=tk.BOTH, expand=True)

        # 1. Left Side: Search + TreeView
        left_container = ttk.Frame(top_pane)
        top_pane.add(left_container, width=350)
        
        # Search Bar
        search_frame = ttk.Frame(left_container)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_tree)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Treeview with Scrollbar
        tree_frame = ttk.Frame(left_container)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, columns=("type"), show="tree", selectmode="browse")
        self.tree.heading("#0", text="Scripts Library", anchor="w")
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind("<<TreeviewSelect>>", self.on_script_select)
        
        # Populate Tree initially
        self.populate_tree()

        # 2. Right Side: Arguments
        self.args_frame_container = ttk.Frame(top_pane, padding=(10, 0, 0, 0))
        top_pane.add(self.args_frame_container)

        ttk.Label(self.args_frame_container, text="Script Configuration:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Canvas + Scrollbar for Arguments
        self.args_canvas = tk.Canvas(self.args_frame_container, bg="#2E2E2E", highlightthickness=0)
        self.args_scrollbar = ttk.Scrollbar(self.args_frame_container, orient="vertical", command=self.args_canvas.yview)
        self.scrollable_args_frame = ttk.Frame(self.args_canvas)

        self.scrollable_args_frame.bind("<Configure>", lambda e: self.args_canvas.configure(scrollregion=self.args_canvas.bbox("all")))
        self.args_canvas.create_window((0, 0), window=self.scrollable_args_frame, anchor="nw")
        self.args_canvas.configure(yscrollcommand=self.args_scrollbar.set)

        self.args_canvas.pack(side="left", fill="both", expand=True)
        self.args_scrollbar.pack(side="right", fill="y")
        
        # --- Bottom Section (Controls and Output) ---
        control_frame = ttk.Frame(bottom_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.run_button = ttk.Button(control_frame, text="Run Selected Script", command=self.start_script_run, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="Clear Output", command=self.clear_output).pack(side=tk.LEFT)

        ttk.Label(bottom_frame, text="Process Output:", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
        self.output_text = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, height=12, bg="#1E1E1E", fg="#00FF00", font=self.monospace_font)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.configure(state='disabled')

    def populate_tree(self, filter_text=""):
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        filter_text = filter_text.lower()
        
        for category, scripts in SCRIPTS_CONFIG.items():
            # If filtering, check if category matches OR any script matches
            cat_matches = filter_text in category.lower()
            matching_scripts = []
            
            for script_name in scripts:
                if filter_text in script_name.lower() or cat_matches:
                    matching_scripts.append(script_name)
            
            if matching_scripts:
                # Open categories by default to show scripts
                cat_id = self.tree.insert("", "end", text=category, open=True)
                for script_name in matching_scripts:
                    self.tree.insert(cat_id, "end", text=script_name)

    def filter_tree(self, *args):
        text = self.search_var.get()
        self.populate_tree(text)

    def on_script_select(self, event=None):
        selected_item = self.tree.selection()
        if not selected_item:
            return

        item_text = self.tree.item(selected_item[0], "text")
        parent_id = self.tree.parent(selected_item[0])
        
        # If parent_id is empty, it's a category folder, not a script
        if not parent_id:
            self.run_button.config(state=tk.DISABLED)
            return
            
        parent_text = self.tree.item(parent_id, "text")
        
        # Locate Config
        config = SCRIPTS_CONFIG.get(parent_text, {}).get(item_text, {})
        if not config:
            return

        self.run_button.config(state=tk.NORMAL)
        self._build_arg_ui(config)

    def _build_arg_ui(self, config):
        # Clear existing
        for widget in self.scrollable_args_frame.winfo_children():
            widget.destroy()
        self.arg_widgets.clear()

        for arg_info in config.get("args", []):
            frame = ttk.Frame(self.scrollable_args_frame)
            frame.pack(fill=tk.X, pady=5)
            
            label_text = arg_info.get("label", arg_info["name"])
            label = ttk.Label(frame, text=label_text, width=25, anchor="w")
            label.pack(side=tk.LEFT, padx=(0, 10))

            arg_type = arg_info["type"]
            arg_name = arg_info["name"]

            if arg_type == "entry":
                widget = ttk.Entry(frame, width=40)
                if arg_info.get("default"):
                    widget.insert(0, arg_info["default"])
            elif arg_type == "check":
                var = tk.BooleanVar()
                if arg_info.get("default") == "on":
                    var.set(True)
                widget = ttk.Checkbutton(frame, variable=var)
                self.arg_widgets[arg_name] = var 
            elif arg_type == "combo":
                var = tk.StringVar()
                widget = ttk.Combobox(frame, textvariable=var, values=arg_info["values"], state="readonly")
                widget.set(arg_info.get("default", arg_info["values"][0]))
                self.arg_widgets[arg_name] = var 

            widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            if arg_type != "check" and arg_type != "combo":
                 self.arg_widgets[arg_name] = widget

    def start_script_run(self):
        selected_item = self.tree.selection()
        if not selected_item:
            return
            
        script_name = self.tree.item(selected_item[0], "text")
        parent_text = self.tree.item(self.tree.parent(selected_item[0]), "text")
        
        config = SCRIPTS_CONFIG[parent_text][script_name]
        
        self.run_button.config(state=tk.DISABLED, text="Running...")
        self.clear_output()
        
        thread = threading.Thread(target=self.run_script_in_thread, args=(config,), daemon=True)
        thread.start()

    def run_script_in_thread(self, config):
        command = ["python3", config["path"]]
        
        if "command" in config:
            command.extend(config["command"].split())

        positional_args = []
        named_args = []

        for arg_info in config.get("args", []):
            arg_name = arg_info["name"]
            widget_val = self.arg_widgets[arg_name]
            
            value = ""
            if isinstance(widget_val, (tk.Entry, ttk.Combobox)):
                value = widget_val.get().strip()
            elif isinstance(widget_val, tk.Variable):
                value = widget_val.get()
            
            if not value and not arg_info.get("required") and arg_info["type"] != "check":
                continue

            if arg_info.get("positional"):
                positional_args.append(str(value))
            elif arg_info["type"] == "check":
                 if value: # If checked
                    named_args.append(arg_name)
            else:
                 named_args.extend([arg_name, str(value)])
        
        command.extend(positional_args)
        command.extend(named_args)
        
        self.update_output(f"Executing: {' '.join(command)}\n\n", "info")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                cwd=BASE_PATH 
            )

            for line in iter(process.stdout.readline, ''):
                self.output_queue.put(line)

            process.stdout.close()
            return_code = process.wait()
            self.output_queue.put(f"\n--- Script finished with exit code {return_code} ---\n")

        except FileNotFoundError:
            self.output_queue.put(f"ERROR: Script not found at {config['path']}\n")
        except Exception as e:
            self.output_queue.put(f"An unexpected error occurred: {e}\n")
        
        self.output_queue.put(("__DONE__",))

    def process_queue(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__DONE__":
                    self.run_button.config(state=tk.NORMAL, text="Run Selected Script")
                else:
                    self.update_output(item)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)

    def update_output(self, text, tag=None):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, text, tag)
        self.output_text.config(state='disabled')
        self.output_text.see(tk.END)
        
    def clear_output(self):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')

# --- Main Application Entry ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptLauncherGUI(root)
    root.mainloop()
