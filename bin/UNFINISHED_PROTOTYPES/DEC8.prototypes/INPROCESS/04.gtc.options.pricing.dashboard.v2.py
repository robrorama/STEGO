# SCRIPTNAME: 04.gtc.options.pricing.dashboard.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. THE MATH ENGINE (Universal for any Ticker)
# ==========================================

class OptionMath:
    """
    The engine room. This works for ANY ticker (SPX, AAPL, BTC, Oil).
    It uses standard Black-Scholes to reverse-engineer missing data.
    """
    def __init__(self, S, K, T, r, market_price=None, flag='p'):
        self.S = S          # Spot Price (Stock Price)
        self.K = K          # Strike Price
        self.T = T          # Time to Expiry (in years)
        self.r = r          # Risk-free rate (approx 5% = 0.05)
        self.flag = flag    # 'c' for Call, 'p' for Put
        self.market_price = market_price
        
        # AUTO-CALCULATE MISSING DATA:
        # If we have price but no volatility, we solve for it using Newton-Raphson.
        self.sigma = self.find_implied_volatility() if market_price else 0.4
        
        # Calculate Greeks immediately
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        self.delta = self._delta()
        self.gamma = self._gamma()
        self.vega = self._vega()
        self.theta = self._theta()

    def bs_price(self, spot_override=None, sigma_override=None):
        """Calculates theoretical price based on hypotheticals."""
        S = spot_override if spot_override else self.S
        sigma = sigma_override if sigma_override else self.sigma
        
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        
        if self.flag == 'c':
            return S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def find_implied_volatility(self):
        """Solves the equation: Market Price - BS_Price(vol) = 0"""
        if self.market_price is None: return 0.2
        def objective(sigma):
            return self.bs_price(sigma_override=sigma) - self.market_price
        try:
            return brentq(objective, 0.001, 5.0) # Search range 0.1% to 500% IV
        except:
            return 0.5 # Fallback

    # Greek Formulas
    def _delta(self): return norm.cdf(self.d1) - 1 if self.flag == 'p' else norm.cdf(self.d1)
    def _gamma(self): return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    def _vega(self): return self.S * norm.pdf(self.d1) * np.sqrt(self.T) / 100 
    def _theta(self): 
        t1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        t2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return (t1 + t2) / 365 # Daily decay

# ==========================================
# 2. THE STRATEGY LOGIC (The "Why")
# ==========================================

class TradeNarrator:
    """
    This class translates the math into English explanations for the trader.
    """
    def generate_plan(self, ticker, base_opt, shock_S, shock_vol, volume, avg_vol):
        # 1. Calculate the Shock
        shock_price = base_opt.bs_price(spot_override=shock_S, sigma_override=shock_vol)
        
        # 2. Define our Exit Logic
        # L1: Conservative. Stock bounces 30% of the drop. Vol crush is mild (40%).
        # L2: Aggressive. Stock bounces 50% of the drop. Vol crush is heavy (70%).
        
        drop_magnitude = base_opt.S - shock_S
        vol_spike_magnitude = shock_vol - base_opt.sigma
        
        # Scenario L1
        s_l1 = shock_S + (drop_magnitude * 0.30)
        v_l1 = shock_vol - (vol_spike_magnitude * 0.40)
        price_l1 = base_opt.bs_price(spot_override=s_l1, sigma_override=v_l1)
        
        # Scenario L2
        s_l2 = shock_S + (drop_magnitude * 0.50)
        v_l2 = shock_vol - (vol_spike_magnitude * 0.70)
        price_l2 = base_opt.bs_price(spot_override=s_l2, sigma_override=v_l2)
        
        # Volume/Liquidity Check
        liq_note = ""
        penalty = 0.0
        if volume < avg_vol * 0.8:
            penalty = 0.05 # 5% haircut on price targets
            price_l1 *= (1 - penalty)
            price_l2 *= (1 - penalty)
            liq_note = f"\n[!] WARNING: Volume is {volume/avg_vol:.0%} of average. Targets reduced by 5% to ensure fill."

        # PRINT THE REPORT
        print(f"\n============= STRATEGY REPORT: {ticker} =============")
        print(f"Current State: Stock ${base_opt.S:.2f} | Option ${base_opt.market_price:.2f} | IV {base_opt.sigma:.1%}")
        print(f"Scenario:      Drop to ${shock_S:.2f} (-{(base_opt.S-shock_S)/base_opt.S:.1%}) | IV Spike to {shock_vol:.1%}")
        print(f"The 'Panic' Price (Theoretical): ${shock_price:.2f}")
        print("-" * 50)
        
        print(f"ORDER 1 (Scale Out Half): LIMIT @ ${price_l1:.2f}")
        print(f"   WHY? Assumes a 'Dead Cat Bounce'.")
        print(f"   1. Price recovers 30% of the drop (to ${s_l1:.2f}).")
        print(f"   2. Fear persists (IV only drops 40% of the spike).")
        
        print(f"\nORDER 2 (Exit Remainder): LIMIT @ ${price_l2:.2f}")
        print(f"   WHY? Assumes a 'Mean Reversion'.")
        print(f"   1. Price recovers 50% of the drop (to ${s_l2:.2f}).")
        print(f"   2. Fear evaporates (IV drops 70% of the spike).")
        
        if liq_note: print(liq_note)
        print("=====================================================")
        
        return {
            'L1': price_l1, 'L2': price_l2, 
            'ShockPrice': shock_price, 
            'TargetSpot_L1': s_l1, 'TargetSpot_L2': s_l2
        }

# ==========================================
# 3. VISUALIZATION
# ==========================================

def plot_rationale(ticker, base_opt, report, shock_S):
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#222", "figure.facecolor": "#111", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 1: The "Why" (Price Path) ---
    stages = ['Start', 'The Drop', 'Bounce (L1)', 'Revert (L2)']
    prices = [base_opt.S, shock_S, report['TargetSpot_L1'], report['TargetSpot_L2']]
    
    ax1.plot(stages, prices, marker='o', color='cyan', linewidth=2, linestyle='--')
    ax1.fill_between(stages, prices, min(prices)*0.995, color='cyan', alpha=0.1)
    
    # Annotate the specific logic on the chart
    ax1.text(1, shock_S, " Panic Selling\n (Enter Trade?)", ha='center', va='bottom', color='red', fontweight='bold')
    ax1.text(2, report['TargetSpot_L1'], f" L1 Target\n (Small Bounce)", ha='center', va='bottom', color='yellow')
    ax1.text(3, report['TargetSpot_L2'], f" L2 Target\n (Normalization)", ha='center', va='bottom', color='lime')
    
    ax1.set_title(f"{ticker} Price Path Logic", fontsize=14, color='white')
    ax1.set_ylabel("Underlying Price ($)")
    
    # --- Plot 2: The Action (Option Price Levels) ---
    levels = ['Current Mkt', 'Panic Peak', 'L1 Limit', 'L2 Limit']
    opt_prices = [base_opt.market_price, report['ShockPrice'], report['L1'], report['L2']]
    colors = ['gray', 'red', 'orange', 'green']
    
    bars = ax2.bar(levels, opt_prices, color=colors)
    ax2.set_title(f"Option Premium & Targets", fontsize=14, color='white')
    ax2.bar_label(bars, fmt='$%.2f', padding=3, color='white', fontsize=12, fontweight='bold')
    
    # Draw "Value Gap" lines
    ax2.axhline(report['L1'], color='orange', linestyle=':', alpha=0.5)
    ax2.axhline(report['L2'], color='green', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. USER CONFIGURATION (EDIT THIS PART)
# ==========================================

def run_script():
    # -----------------------------------------------
    # INPUT DATA (Replace these with your Broker Data)
    # -----------------------------------------------
    TICKER = "NVDA"       # Works for ANY ticker
    SPOT_PRICE = 460.00   # Current Stock Price
    STRIKE = 450.00       # Put Strike
    EXPIRY_DAYS = 5       # Days to expiration
    OPTION_PRICE = 8.50   # Current Ask Price of the Option
    RISK_FREE_RATE = 0.05 # 5% interest rate
    
    # Volume data (for liquidity approximation)
    # If you don't have this, just set both to 1000
    TODAY_VOL = 120000 
    AVG_VOL = 150000      
    
    # -----------------------------------------------
    # SCENARIO SETTINGS (The "What If")
    # -----------------------------------------------
    # "I want to prepare for a 3% drop and a 5 point vol spike"
    DROP_PCT = 0.03       # 3% Drop
    VOL_SPIKE = 5.0       # +5 Volatility Points (e.g., 30% -> 35%)
    
    # -----------------------------------------------
    # EXECUTION
    # -----------------------------------------------
    print("Processing Data...")
    
    # 1. Initialize Math Engine
    # Note: We use flag='p' for Puts. Change to 'c' for Calls.
    opt = OptionMath(S=SPOT_PRICE, K=STRIKE, T=EXPIRY_DAYS/365, r=RISK_FREE_RATE, market_price=OPTION_PRICE, flag='p')
    
    # 2. Calculate Shock Variables
    shock_S = SPOT_PRICE * (1 - DROP_PCT)
    shock_vol = opt.sigma + (VOL_SPIKE / 100)
    
    # 3. Generate Narrative & Numbers
    narrator = TradeNarrator()
    report = narrator.generate_plan(TICKER, opt, shock_S, shock_vol, TODAY_VOL, AVG_VOL)
    
    # 4. Visualize
    plot_rationale(TICKER, opt, report, shock_S)

if __name__ == "__main__":
    run_script()
