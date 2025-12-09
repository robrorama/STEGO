# SCRIPTNAME: 05.options.traders.drop.playbook.v2.py
# AUTHOR: Michael Derby
# DATE:   December 08, 2025
# FRAMEWORK: STEGO FINANCIAL FRAMEWORK

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq
import warnings

# Try to import yfinance for real data backfilling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==========================================
# 1. ARGUMENT PARSER (The Listener)
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Hedge Fund Options Execution Dashboard")
    
    # Required: The Basics
    parser.add_argument("--ticker", type=str, required=True, help="The Ticker Symbol (e.g., SPY, NVDA)")
    parser.add_argument("--strike", type=float, required=True, help="The Option Strike Price")
    parser.add_argument("--type", type=str, choices=['c', 'p'], default='p', help="Option Type: 'c' for Call, 'p' for Put (Default: p)")
    
    # Optional: If you don't provide these, we try to fetch them or estimate
    parser.add_argument("--spot", type=float, help="Current Underlying Spot Price (will fetch if omitted)")
    parser.add_argument("--opt_price", type=float, required=True, help="Current Market Price of the Option")
    parser.add_argument("--dte", type=int, default=5, help="Days to Expiry (Default: 5)")
    parser.add_argument("--volume", type=int, help="Today's Volume (will fetch/approximate if omitted)")
    parser.add_argument("--avg_vol", type=int, help="Average Volume (will fetch/approximate if omitted)")
    
    # Scenario Inputs (The "Stress Test")
    parser.add_argument("--drop_pct", type=float, default=2.0, help="Shock: Drop percentage (e.g., 2.0 for 2%)")
    parser.add_argument("--vol_spike", type=float, default=5.0, help="Shock: Volatility Spike in points (e.g., 5.0)")

    return parser.parse_args()

# ==========================================
# 2. DATA GAP FILLER (No Fake Data)
# ==========================================
def fill_missing_data(args):
    """
    If the user didn't provide Spot or Volume, we fetch REAL data.
    We do NOT generate random numbers.
    """
    data = {
        'ticker': args.ticker.upper(),
        'strike': args.strike,
        'type': args.type,
        'spot': args.spot,
        'opt_price': args.opt_price,
        'dte': args.dte,
        'volume': args.volume,
        'avg_vol': args.avg_vol,
        'drop_pct': args.drop_pct / 100.0,
        'vol_spike': args.vol_spike
    }

    # If Spot or Volume is missing, try to fetch real data
    if (data['spot'] is None or data['volume'] is None) and YFINANCE_AVAILABLE:
        print(f"[*] Fetching real-time data for {data['ticker']} to fill missing gaps...")
        try:
            ticker_obj = yf.Ticker(data['ticker'])
            todays_data = ticker_obj.history(period='1d')
            
            if not todays_data.empty:
                current_close = todays_data['Close'].iloc[-1]
                current_vol = todays_data['Volume'].iloc[-1]
                
                # Fill Spot if missing
                if data['spot'] is None:
                    data['spot'] = float(current_close)
                    print(f"    -> Retrieved Spot Price: ${data['spot']:.2f}")
                
                # Fill Volume if missing
                if data['volume'] is None:
                    data['volume'] = int(current_vol)
                    # Rough approx for Avg Volume if not provided
                    data['avg_vol'] = int(current_vol * 1.2) if data['avg_vol'] is None else data['avg_vol']
                    print(f"    -> Retrieved Volume: {data['volume']:,}")
            else:
                print("    [!] API data empty. Using fallbacks.")
        except Exception as e:
            print(f"    [!] Data fetch failed ({e}). Proceeding with available inputs.")

    # HARD STOP: If we still don't have Spot Price, we cannot calc Greeks.
    if data['spot'] is None:
        print("\n[CRITICAL ERROR] Spot Price is missing.")
        print("Please provide --spot PRICE in your command.")
        sys.exit(1)
        
    # Default Volume if completely unavailable (to prevent crash, but warn user)
    if data['volume'] is None:
        data['volume'] = 1000000
        data['avg_vol'] = 1000000
        print("    [!] Warning: Volume data missing. Assuming standard liquidity.")

    return data

# ==========================================
# 3. MATH ENGINE
# ==========================================
class OptionMath:
    def __init__(self, S, K, T, r, market_price, flag):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.flag = flag
        self.market_price = market_price
        
        # Reverse Engineer IV
        self.sigma = self.find_implied_volatility()
        
        # Calc Greeks
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        self.delta = self._delta()

    def bs_price(self, spot_override=None, sigma_override=None):
        S = spot_override if spot_override else self.S
        sigma = sigma_override if sigma_override else self.sigma
        d1 = (np.log(S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        if self.flag == 'c':
            return S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def find_implied_volatility(self):
        if self.market_price <= 0: return 0.01
        def objective(sigma): return self.bs_price(sigma_override=sigma) - self.market_price
        try: return brentq(objective, 0.01, 5.0)
        except: return 0.5

    def _delta(self):
        return norm.cdf(self.d1) - 1 if self.flag == 'p' else norm.cdf(self.d1)

# ==========================================
# 4. VISUALIZATION ENGINE
# ==========================================
def visualize_strategy(data, opt, shock_S, shock_vol, l1_price, l2_price, shock_price):
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e1e", "figure.facecolor": "#121212", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2)
    
    # --- CHART 1: The PnL Target Zones ---
    ax1 = fig.add_subplot(gs[0, :]) # Span top row
    
    scenarios = ['Current', 'SHOCK (-{:.1%})'.format(data['drop_pct']), 'L1 Exit (Bounce)', 'L2 Exit (Revert)']
    prices = [data['opt_price'], shock_price, l1_price, l2_price]
    colors = ['#7f8c8d', '#e74c3c', '#f39c12', '#2ecc71']
    
    bars = ax1.bar(scenarios, prices, color=colors, alpha=0.9)
    ax1.bar_label(bars, fmt='$%.2f', padding=5, color='white', fontsize=12, fontweight='bold')
    
    # Add reference lines
    ax1.axhline(data['opt_price'], color='white', linestyle='--', alpha=0.3, label="Entry Basis")
    
    ax1.set_title(f"EXECUTION PLAN: {data['ticker']} ${data['strike']} {data['type'].upper()}", fontsize=16, fontweight='bold', color='white')
    ax1.set_ylabel("Option Premium ($)")
    
    # --- CHART 2: Implied Volatility Context ---
    ax2 = fig.add_subplot(gs[1, 0])
    
    iv_vals = [opt.sigma * 100, shock_vol * 100]
    iv_labels = ['Current IV', 'Shock IV']
    
    ax2.plot(iv_labels, iv_vals, marker='o', markersize=10, color='cyan', linewidth=2)
    ax2.fill_between(iv_labels, 0, iv_vals, color='cyan', alpha=0.1)
    
    ax2.text(0, opt.sigma*100 + 1, f"{opt.sigma:.1%}", color='white', ha='center')
    ax2.text(1, shock_vol*100 + 1, f"{shock_vol:.1%}", color='white', ha='center')
    
    ax2.set_title("Volatility Expansion (+Vega PnL)", fontsize=12, color='white')
    ax2.set_ylim(bottom=max(0, (opt.sigma*100)-5))
    
    # --- CHART 3: Action Table ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Create a textual table on the plot
    text_str = (
        f"ACTIONABLE LIMITS\n"
        f"-----------------\n"
        f"STOP LOSS (Mid):   ${shock_price*0.80:.2f}\n"
        f"L1 SELL (40% qty): ${l1_price:.2f}\n"
        f"L2 SELL (60% qty): ${l2_price:.2f}\n\n"
        f"MARKET CONTEXT\n"
        f"--------------\n"
        f"Drop Tested: {data['drop_pct']:.1%}\n"
        f"Vol Spike:   +{data['vol_spike']} pts\n"
        f"Liquidity:   {data['volume'] / data['avg_vol']:.0%} of Avg"
    )
    
    ax3.text(0.1, 0.5, text_str, fontsize=14, fontfamily='monospace', color='white', va='center')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    # 1. Parse CLI Args
    args = parse_arguments()
    
    # 2. Fill Data Gaps (Fetch real data if missing)
    data = fill_missing_data(args)
    
    print(f"\n======== {data['ticker']} EXECUTION MATRIX ========")
    print(f"Inputs: Spot ${data['spot']:.2f} | Strike ${data['strike']} | OptPrice ${data['opt_price']:.2f}")
    
    # 3. Initialize Math
    opt = OptionMath(data['spot'], data['strike'], data['dte']/365, 0.05, data['opt_price'], data['type'])
    print(f"Derived Market IV: {opt.sigma:.2%}")
    
    # 4. Run Shock Logic
    shock_S = data['spot'] * (1 - data['drop_pct'])
    shock_vol = opt.sigma + (data['vol_spike'] / 100.0)
    
    # Calculate Theoretical Panic Price
    shock_price = opt.bs_price(spot_override=shock_S, sigma_override=shock_vol)
    
    # Calculate Exit Targets
    # L1: Spot bounces 30% of drop, Vol reverts 40%
    l1_spot = shock_S + ((data['spot'] - shock_S) * 0.30)
    l1_vol = shock_vol - ((shock_vol - opt.sigma) * 0.40)
    l1_price = opt.bs_price(spot_override=l1_spot, sigma_override=l1_vol)
    
    # L2: Spot bounces 50% of drop, Vol reverts 70%
    l2_spot = shock_S + ((data['spot'] - shock_S) * 0.50)
    l2_vol = shock_vol - ((shock_vol - opt.sigma) * 0.70)
    l2_price = opt.bs_price(spot_override=l2_spot, sigma_override=l2_vol)
    
    # Liquidity Adjustment
    if data['volume'] < data['avg_vol'] * 0.8:
        print(f"[!] Low Liquidity Detected ({data['volume']:,}). Adjusting limits down by 5%.")
        l1_price *= 0.95
        l2_price *= 0.95

    # 5. Console Output
    print("-" * 40)
    print(f"SCENARIO: Spot drops to ${shock_S:.2f} | IV hits {shock_vol:.1%}")
    print(f"THEORETICAL PEAK PRICE: ${shock_price:.2f}")
    print("-" * 40)
    print(f" >> SET L1 SELL LIMIT: ${l1_price:.2f}")
    print(f" >> SET L2 SELL LIMIT: ${l2_price:.2f}")
    print("===========================================")

    # 6. Launch Visuals
    visualize_strategy(data, opt, shock_S, shock_vol, l1_price, l2_price, shock_price)

if __name__ == "__main__":
    main()
