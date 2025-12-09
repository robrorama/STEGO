import numpy as np
import pandas as pd
import scipy.stats as si
import itertools
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. CORE MATH ENGINE (Vectorized BSM)
# ==========================================
class BSMEngine:
    """
    High-performance, vectorized Black-Scholes-Merton Calculator.
    Handles standard Greeks plus Vanna and Charm.
    """
    
    @staticmethod
    def d1_d2(S, K, T, r, sigma):
        # Prevent division by zero and sqrt of negative numbers
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-3)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def calculate_metrics(S, K, T, r, sigma, option_type='call'):
        """
        Returns a dictionary of vectors: Price + All Greeks (First, Second, Third order)
        """
        d1, d2 = BSMEngine.d1_d2(S, K, T, r, sigma)
        norm_pdf_d1 = si.norm.pdf(d1)
        norm_cdf_d1 = si.norm.cdf(d1)
        norm_cdf_d2 = si.norm.cdf(d2)
        sqrt_T = np.sqrt(T)

        # 1. Price & Delta
        if option_type == 'call':
            price = S * norm_cdf_d1 - K * np.exp(-r * T) * norm_cdf_d2
            delta = norm_cdf_d1
            theta_carry = - (S * sigma * norm_pdf_d1) / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm_cdf_d2
        else:
            price = K * np.exp(-r * T) * (1 - norm_cdf_d2) - S * (1 - norm_cdf_d1)
            delta = norm_cdf_d1 - 1
            theta_carry = - (S * sigma * norm_pdf_d1) / (2 * sqrt_T) + r * K * np.exp(-r * T) * (1 - norm_cdf_d2)

        # 2. Gamma & Vega (Same for Call/Put usually)
        gamma = norm_pdf_d1 / (S * sigma * sqrt_T)
        vega = S * sqrt_T * norm_pdf_d1 / 100  # Scaled to 1 vol point change
        
        # 3. Theta (Daily)
        theta = theta_carry / 365.0

        # 4. Higher Order Greeks: Vanna & Charm
        # Vanna: dDelta/dSigma -> Sensitivity of Delta to Volatility
        # Formula: -e^(-qT) * N'(d1) * d2 / sigma
        vanna = -norm_pdf_d1 * d2 / sigma 

        # Charm: dDelta/dT -> Time decay of Delta
        # Formula (approx for q=0): -N'(d1) * [1/(2*sigma*sqrt(T)) - d2/(2T)] 
        # Note: This is crucial for GTC orders as delta drifts overnight.
        charm = -norm_pdf_d1 * (1 / (2 * sigma * sqrt_T) - d2 / (2 * T))
        if option_type == 'put':
            charm = charm + (r * np.exp(-r*T) * (1-norm_cdf_d2)) # Adjustment for put

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'vanna': vanna,
            'charm': charm
        }

# ==========================================
# 2. MODULE A: WHAT-IF GRID ENGINE
# ==========================================
class WhatIfGridEngine:
    """
    Generates the massive multi-dimensional scenario analysis grid.
    """
    def __init__(self, base_spot, base_iv, base_r=0.045):
        self.base_spot = base_spot
        self.base_iv = base_iv
        self.base_r = base_r
        
        # Dimensions Definition
        self.shocks_iv = np.array([-3, -2, -1, 0, 1, 2, 3]) / 100.0  # Adjust vol points
        self.shocks_spot_pct = np.array([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
        self.buckets_theta = np.array([0.25, 0.50, 0.75, 1.00]) / 365.0 # Days as years
        
        # Target Deltas to solve for Strikes
        self.target_deltas = [0.50, 0.05, 0.10, 0.15, 0.20] # ATM and Wings
        # Expiries (Days)
        self.expiries_days = np.array([1, 2, 3, 4, 5, 7, 14, 21, 30, 45])

    def _solve_strike_for_delta(self, delta_target, T, is_call=True):
        """Inverse BS to find Strike from Delta."""
        # Approx inversion for grid generation speed
        # d1 = norm.ppf(delta)
        # ln(S/K) ... solve for K
        inv_n = si.norm.ppf(delta_target if is_call else 1 + delta_target)
        sigma_root_t = self.base_iv * np.sqrt(T/365.0)
        # K = S * exp( - (d1 * sigma_root_t - (r + 0.5sigma^2)T ) )
        # Simplified for grid generation:
        return self.base_spot * np.exp(-inv_n * sigma_root_t)

    def generate_grid(self, filepath="gtc_whatif_grid.parquet"):
        print(f"Generating GTC Grid for Spot: {self.base_spot}...")
        
        rows = []
        
        # 1. Build Base Scenario List
        for d_days in self.expiries_days:
            T_base = d_days / 365.0
            
            # Determine Strikes based on Delta buckets
            strikes = []
            for d in self.target_deltas:
                strikes.append(self._solve_strike_for_delta(d, d_days, is_call=True))
                strikes.append(self._solve_strike_for_delta(-d, d_days, is_call=False))
            strikes = np.unique(strikes)

            # Cartesian Product of Shocks
            for K in strikes:
                # Apply Shocks
                for s_shock in self.shocks_spot_pct:
                    S_new = self.base_spot * (1 + s_shock)
                    
                    for v_shock in self.shocks_iv:
                        sigma_new = max(0.01, self.base_iv + v_shock)
                        
                        for t_decay in self.buckets_theta:
                            # T_new = Time remaining after 't_decay' has passed
                            T_new = max(1e-5, T_base - t_decay)
                            
                            rows.append({
                                'T_orig_days': d_days,
                                'Strike': K,
                                'Spot_Shock_Pct': s_shock,
                                'IV_Shock_Vol': v_shock,
                                'Theta_Decay_Year': t_decay,
                                'S_eff': S_new,
                                'Sigma_eff': sigma_new,
                                'T_eff': T_new,
                                'r': self.base_r
                            })

        # 2. Vectorized Calculation
        df = pd.DataFrame(rows)
        
        # Determine Call/Put based on Strike vs Spot (simplified logic or explicit)
        # Assuming Call for calculation demo, ideally split rows
        df['Type'] = np.where(df['Strike'] > df['S_eff'], 'call', 'put') 
        
        metrics = BSMEngine.calculate_metrics(
            df['S_eff'].values, 
            df['Strike'].values, 
            df['T_eff'].values, 
            df['r'].values, 
            df['Sigma_eff'].values,
            option_type='call' # Simplified for vectorization, in prod handle puts
        )
        
        # Assign Metrics
        for k, v in metrics.items():
            df[k] = v
            
        # 3. Compute Derived Metrics
        # NBBO Overlap Prob (Mock from historical stats - usually a logistic regression)
        df['NBBO_Overlap_Prob'] = np.clip(1 - (np.abs(df['delta']) * 2), 0.1, 0.9)
        
        # Theta Efficiency: How much decay do I capture per $ of premium risk?
        df['Theta_Efficiency'] = df['theta'] / df['price']
        
        # Expected Slippage (bps) - Heuristic based on Vega/Gamma risk
        df['Exp_Slippage_Bps'] = 5.0 + (np.abs(df['gamma']) * 100) + (df['vega'] * 2)

        # 4. Save to Disk
        print(f"Grid Complete: {len(df)} scenarios calculated.")
        df.to_parquet(filepath, index=False)
        return df

# ==========================================
# 3. MODULE B: BS/IV OFFLINE CACHE TABLE
# ==========================================
class OfflineGreeksCache:
    """
    A 4-D Cache keyed by buckets for instant intraday inference without recalculation.
    Key: (S-bucket, Vol-bucket, T-bucket, Delta-bucket)
    """
    def __init__(self, df_grid: pd.DataFrame):
        self.grid = df_grid
        self.lookup_table = None

    def build_cache(self):
        """
        Pivots the granular grid into a fast lookup structure.
        We bin the continuous variables into discrete buckets for the key.
        """
        # Create discrete buckets for the keys
        self.grid['S_Bucket'] = pd.cut(self.grid['S_eff'], bins=10, labels=False)
        self.grid['Vol_Bucket'] = pd.cut(self.grid['Sigma_eff'], bins=5, labels=False)
        self.grid['T_Bucket'] = pd.cut(self.grid['T_eff'], bins=5, labels=False)
        self.grid['Delta_Bucket'] = pd.cut(self.grid['delta'], bins=10, labels=False)
        
        # Group by buckets and take the mean of metrics (centroid of the bucket)
        self.lookup_table = self.grid.groupby(
            ['S_Bucket', 'Vol_Bucket', 'T_Bucket', 'Delta_Bucket']
        )[['price', 'delta', 'gamma', 'vega', 'theta', 'vanna', 'charm']].mean().reset_index()
        
        # Create a MultiIndex for O(1) lookup
        self.lookup_table.set_index(
            ['S_Bucket', 'Vol_Bucket', 'T_Bucket', 'Delta_Bucket'], inplace=True
        )
        print("Cache Table Built. Ready for inference.")
        return self.lookup_table

    def get_inference(self, s_idx, v_idx, t_idx, d_idx):
        try:
            return self.lookup_table.loc[(s_idx, v_idx, t_idx, d_idx)]
        except KeyError:
            return None

# ==========================================
# 4. MODULE C: GTC LIMIT PRICE ENGINE
# ==========================================
class GTCLimitEngine:
    """
    Constructs actionable Limit Orders based on the What-If Grid and Edge Models.
    """
    def __init__(self, grid_path="gtc_whatif_grid.parquet"):
        self.df = pd.read_parquet(grid_path)
        
    def compute_candidate_limits(self, current_market_data: Dict):
        """
        1. Filters grid for current market state approximation.
        2. Applies Edge Scoring.
        3. Returns limit prices.
        """
        # Unpack market data
        spot = current_market_data['spot']
        vol = current_market_data['iv']
        
        # Filter Grid for "Near Current State" (0 shock scenarios)
        # In prod, you interpolate. Here we filter.
        candidates = self.df[
            (self.df['Spot_Shock_Pct'] == 0) & 
            (self.df['IV_Shock_Vol'] == 0) &
            (self.df['Theta_Decay_Year'] == 0.25/365.0) # Look 6 hours ahead
        ].copy()
        
        # --- LOGIC CONTINUATION FROM PROMPT CUTOFF ---
        
        # 1. Edge Calculation
        # Definition: Edge = Theoretical Value * (1 + Desired Margin based on Greeks)
        # We penalize high Gamma/Vega risk scenarios
        
        risk_premium = (np.abs(candidates['gamma']) * 50) + (candidates['vega'] * 0.10)
        
        # Limit Price Construction
        # Bid = Model - Risk Premium
        # Ask = Model + Risk Premium
        candidates['Limit_Bid'] = candidates['price'] - risk_premium
        candidates['Limit_Ask'] = candidates['price'] + risk_premium
        
        # Edge Score Calculation
        # (Model - Limit) / Spread_Estimate
        spread_est = candidates['Exp_Slippage_Bps'] * candidates['price'] / 10000
        candidates['Edge_Score_Bid'] = (candidates['price'] - candidates['Limit_Bid']) / spread_est
        
        # Filter for actionable GTCs (High Theta Efficiency, Good Edge)
        actionable = candidates[
            (candidates['Theta_Efficiency'] > 0.1) & 
            (candidates['Edge_Score_Bid'] > 1.5)
        ]
        
        return actionable[['Strike', 'T_orig_days', 'delta', 'price', 'Limit_Bid', 'Limit_Ask', 'vanna', 'charm']]

    def visualize_opportunity_surface(self, actionable_df):
        """
        Create a Visual Surface of Edge Scores vs Delta/Expiry
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=actionable_df['delta'],
            y=actionable_df['T_orig_days'],
            z=actionable_df['vanna'],
            mode='markers',
            marker=dict(
                size=5,
                color=actionable_df['Limit_Bid'], # Color by limit price
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        fig.update_layout(
            title="GTC Opportunity Cloud: Vanna/Delta/Time",
            scene=dict(
                xaxis_title='Delta',
                yaxis_title='Days to Exp',
                zaxis_title='Vanna Exposure'
            )
        )
        return fig

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Initialize
    base_spot_price = 4500.00
    base_vol = 0.18 # 18% IV
    
    # 2. Run Module A: Grid Gen
    grid_engine = WhatIfGridEngine(base_spot_price, base_vol)
    df_grid = grid_engine.generate_grid()
    
    # 3. Run Module B: Cache Build
    cache_engine = OfflineGreeksCache(df_grid)
    lookup = cache_engine.build_cache()
    
    # 4. Run Module C: Limit Construction
    gtc_engine = GTCLimitEngine()
    current_market = {'spot': 4500, 'iv': 0.18}
    limits = gtc_engine.compute_candidate_limits(current_market)
    
    print("\n--- GTC Candidate Limits (Top 5) ---")
    print(limits.head(5).to_string())

    # 5. Visuals (Optional - requires plotly)
    # fig = gtc_engine.visualize_opportunity_surface(limits)
    # fig.show()
