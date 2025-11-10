"""functionSPACE Trading Protocol - Core Simulation Module

Implements the full trading lifecycle (Create, Buy, Sell, Settle) with
mathematical fidelity according to the Trading Spec v2.
"""

import numpy as np
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import root_scalar
from scipy.special import logsumexp
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import pickle
import json
from dataclasses import dataclass, asdict


@dataclass
class Position:
    """Represents a user's position (the NFT)."""
    position_id: str
    belief_p: List[float]  # Belief vector (K+1 dimensional)
    minted_claims_m: float  # Scalar m
    input_collateral_C: float  # Scalar C
    status: str = "open"  # "open" or "closed"
    sold_price: Optional[float] = None  # t* if sold, None if still open
    
    def to_dict(self):
        return asdict(self)


class Market:
    """The central stateful trading market."""
    
    def __init__(self):
        self.alpha_vector: Optional[np.ndarray] = None
        self.positions_db: Dict[str, Position] = {}
        self.market_params: Optional[Dict] = None
        self.is_settled: bool = False
        self.settlement_outcome: Optional[float] = None
        self.settlement_payouts: Optional[Dict[str, float]] = None
        self.title: Optional[str] = None
        
    def create_market(self, L: float, H: float, K: int, P0: float, mu: float, eps_alpha: float, title: str,
                     tau: float = 0.01, gamma: float = 1.0, lambda_s: float = 0.5, lambda_d: float = 0.5,
                     overwrite: bool = False):
        """Initialize the market with uniform prior.
        
        Args:
            L: Lower bound of outcome space
            H: Upper bound of outcome space
            K: Number of internal discretization points (total K+1 points)
            P0: Initial total "pseudocount" mass
            mu: Minting ratio (claims per unit potential change)
            eps_alpha: Minimum alpha floor for numerical stability
            title: Semantic title for the market
            tau: Settlement eligibility threshold (default: 0.01)
            gamma: Settlement temperature parameter (default: 1.0)
            lambda_s: Settlement claim share weight (default: 0.5)
            lambda_d: Settlement accuracy share weight (default: 0.5)
            overwrite: If True, allows overwriting existing market (default: False)
        """
        if self.alpha_vector is not None and not overwrite:
            raise ValueError("Market already created")
            
        # Initialize uniform prior: alpha_k = P0 / (K+1) for all k
        self.alpha_vector = np.full(K + 1, P0 / (K + 1), dtype=np.float64)
        
        self.market_params = {
            'L': L,
            'H': H,
            'K': K,
            'P0': P0,
            'mu': mu,
            'eps_alpha': eps_alpha,
            'tau': tau,
            'gamma': gamma,
            'lambda_s': lambda_s,
            'lambda_d': lambda_d
        }
        
        self.title = title
        self.positions_db = {}
        self.is_settled = False
        
    def _compute_delta_A_difference_form(self, alpha_old: np.ndarray, alpha_new: np.ndarray) -> float:
        """Compute potential change using the difference form (Directive 1).
        
        ΔA = [ln(Γ(P_new)) - ln(Γ(P_old))] - Σ[ln(Γ(α_{k,new})) - ln(Γ(α_{k,old}))]
        
        This prevents subtractive cancellation errors.
        """
        P_old = np.sum(alpha_old)
        P_new = np.sum(alpha_new)
        
        # Compute using gammaln (log-gamma)
        term1 = gammaln(P_new) - gammaln(P_old)
        term2 = np.sum(gammaln(alpha_new) - gammaln(alpha_old))
        
        delta_A = term1 - term2
        return delta_A
    
    def buy(self, collateral_C: float, belief_p: List[float]) -> Position:
        """Execute a buy operation.
        
        Args:
            collateral_C: Input collateral amount
            belief_p: Belief vector (must sum to 1, length K+1)
            
        Returns:
            New Position object
        """
        if self.alpha_vector is None:
            raise ValueError("Market not initialized")
        if self.is_settled:
            raise ValueError("Market is settled, no more trades allowed")
            
        # Validate belief vector
        p = np.array(belief_p, dtype=np.float64)
        if len(p) != len(self.alpha_vector):
            raise ValueError(f"Belief vector must have length {len(self.alpha_vector)}")
        if not np.isclose(np.sum(p), 1.0):
            raise ValueError("Belief vector must sum to 1")
        if np.any(p < 0):
            raise ValueError("Belief vector elements must be non-negative")
            
        # Compute alpha_before and alpha_after
        alpha_before = self.alpha_vector.copy()
        alpha_after = alpha_before + collateral_C * p
        
        # Compute ΔA using difference form (Directive 1)
        delta_A = self._compute_delta_A_difference_form(alpha_before, alpha_after)
        
        # Mint claims
        mu = self.market_params['mu']
        m = mu * delta_A
        
        # Create position
        position_id = f"pos_{len(self.positions_db) + 1:04d}"
        position = Position(
            position_id=position_id,
            belief_p=belief_p,
            minted_claims_m=m,
            input_collateral_C=collateral_C
        )
        
        # Update state
        self.positions_db[position_id] = position
        self.alpha_vector = alpha_after
        
        return position
    
    def _generate_p_from_normal(self, mean: float, std_dev: float) -> List[float]:
        """Generate a belief vector from a normal distribution.
        
        Args:
            mean: Mean of the normal distribution
            std_dev: Standard deviation of the normal distribution
            
        Returns:
            Normalized probability vector (K+1 elements, sums to 1)
        """
        if self.market_params is None:
            raise ValueError("Market not initialized")
        
        L = self.market_params['L']
        H = self.market_params['H']
        K = self.market_params['K']
        
        # Create K+1 discretization points
        x_points = np.linspace(L, H, K + 1)
        
        # Edge case: near-zero std_dev creates a "one-hot" vector
        if std_dev < 1e-6:
            # Find closest index to mean
            closest_idx = np.argmin(np.abs(x_points - mean))
            p_vector = np.zeros(K + 1, dtype=np.float64)
            p_vector[closest_idx] = 1.0
            return p_vector.tolist()
        
        # Standard case: evaluate normal PDF at discretization points
        densities = norm.pdf(x_points, loc=mean, scale=std_dev)
        
        # Normalize to sum to 1
        total = np.sum(densities)
        if total == 0:
            # Fallback to uniform if all densities are zero (shouldn't happen)
            p_vector = np.full(K + 1, 1.0 / (K + 1), dtype=np.float64)
        else:
            p_vector = densities / total
        
        return p_vector.tolist()
    
    def simulate_sell(self, position_id: str) -> Dict:
        """Simulate a sell operation (READ-ONLY).
        
        Implements the Symmetric Redemption algorithm (Spec §7.4).
        
        Args:
            position_id: ID of the position to sell
            
        Returns:
            Dictionary with 't_star' (collateral value) and 'iterations' (solver iterations)
        """
        if self.alpha_vector is None:
            raise ValueError("Market not initialized")
        if self.is_settled:
            raise ValueError("Market is settled")
        if position_id not in self.positions_db:
            raise ValueError(f"Position {position_id} not found")
            
        position = self.positions_db[position_id]
        if position.status == "closed":
            raise ValueError(f"Position {position_id} is already closed")
        p = np.array(position.belief_p, dtype=np.float64)
        m = position.minted_claims_m
        
        alpha_current = self.alpha_vector.copy()
        mu = self.market_params['mu']
        eps_alpha = self.market_params['eps_alpha']
        
        # Step 1: Compute t_max (maximum t before any alpha_k hits eps_alpha)
        # t_max = min_k ((alpha_k - eps_alpha) / p_k) for p_k > 0
        t_max = np.inf
        for k in range(len(alpha_current)):
            if p[k] > 0:
                t_k = (alpha_current[k] - eps_alpha) / p[k]
                t_max = min(t_max, t_k)
        
        if t_max <= 0:
            raise ValueError("Cannot sell: position would violate eps_alpha constraint")
        
        # Step 2: Define g(t) and solve for g(t) = 0
        def g_of_t(t: float) -> float:
            """Root function: g(t) = -ΔA(α, α - t·p) - m/μ"""
            if t < 0 or t > t_max:
                return np.inf  # Invalid region
            
            # Check eps_alpha constraint
            alpha_new = alpha_current - t * p
            if np.any(alpha_new < eps_alpha - 1e-12):
                return np.inf  # Violates constraint
            
            # Compute -ΔA using difference form
            # Note: ΔA(old→new) when new < old gives negative ΔA
            # So -ΔA is positive
            delta_A = self._compute_delta_A_difference_form(alpha_current, alpha_new)
            
            return -delta_A - m / mu
        
        def h_of_t(t: float) -> float:
            """Derivative: h(t) = dg/dt"""
            if t < 0 or t > t_max:
                return 0
            
            alpha_new = alpha_current - t * p
            if np.any(alpha_new < eps_alpha - 1e-12):
                return 0
            
            P_new = np.sum(alpha_new)
            
            # h(t) = digamma(P) - Σ p_k * digamma(α_k)
            result = digamma(P_new) - np.sum(p * digamma(alpha_new))
            return result
        
        # Step 3: Solve for t*
        # Use Brent's method (root_scalar with method='brentq')
        try:
            # Check boundary conditions
            g_0 = g_of_t(0)
            g_max = g_of_t(t_max * 0.9999)  # Slightly inside boundary
            
            if g_0 * g_max > 0:
                # Same sign at boundaries - edge case
                if abs(g_0) < abs(g_max):
                    return {'t_star': 0.0, 'iterations': 0}
                else:
                    return {'t_star': t_max * 0.9999, 'iterations': 0}
            
            result = root_scalar(
                g_of_t,
                method='brentq',
                bracket=[0, t_max * 0.9999],
                xtol=1e-15,
                rtol=1e-15
            )
            
            t_star = result.root
            
            # Ensure t_star is within bounds
            t_star = max(0, min(t_star, t_max))
            
            # Capture iterations from solver
            iterations = result.iterations if hasattr(result, 'iterations') else 0
            
            return {'t_star': t_star, 'iterations': iterations}
            
        except Exception as e:
            raise ValueError(f"Solver failed: {str(e)}")
    
    def execute_sell(self, position_id: str) -> float:
        """Execute a sell operation (MUTATES STATE).
        
        Args:
            position_id: ID of the position to sell
            
        Returns:
            Collateral value t* returned to seller
        """
        if self.is_settled:
            raise ValueError("Market is settled")
            
        # Simulate to get t* (now returns dict)
        result = self.simulate_sell(position_id)
        t_star = result['t_star']
        
        # Update state
        position = self.positions_db[position_id]
        p = np.array(position.belief_p, dtype=np.float64)
        
        self.alpha_vector = self.alpha_vector - t_star * p
        
        # Mark position as closed instead of deleting (maintain history)
        position.status = "closed"
        position.sold_price = t_star
        
        return t_star
    
    def simulate_settle(self, outcome_x: float) -> Dict[str, float]:
        """Simulate final settlement (READ-ONLY).
        
        Implements the 6-step Final Settlement algorithm (Spec §7.5).
        This ensures participant-funded solvency: sum(payouts) = sum(collateral).
        
        Args:
            outcome_x: The realized outcome value
            
        Returns:
            Dictionary mapping position_id to payout amount
        """
        if self.alpha_vector is None:
            raise ValueError("Market not initialized")
        
        if len(self.positions_db) == 0:
            return {}
            
        L = self.market_params['L']
        H = self.market_params['H']
        K = self.market_params['K']
        tau = self.market_params['tau']
        gamma = self.market_params['gamma']
        lambda_s = self.market_params['lambda_s']
        lambda_d = self.market_params['lambda_d']
        
        # Validate outcome
        if not (L <= outcome_x <= H):
            raise ValueError(f"Outcome {outcome_x} must be in [{L}, {H}]")
        
        # STEP 0: Calculate total collateral pool to redistribute
        # CRITICAL: Only include OPEN positions - closed positions already withdrew their funds
        Pool = sum(pos.input_collateral_C for pos in self.positions_db.values() if pos.status == "open")
        
        # Find bracketing bucket indices for outcome_x
        x_values = np.linspace(L, H, K + 1)
        if outcome_x == L:
            j_lower, j_upper = 0, 0
        elif outcome_x == H:
            j_lower, j_upper = K, K
        else:
            j_lower = np.searchsorted(x_values, outcome_x, side='right') - 1
            j_upper = j_lower + 1
        
        delta_x = (H - L) / K
        
        # STEP 1: Compute log-densities (l_i) for each OPEN position
        log_densities = {}
        
        for pos_id, position in self.positions_db.items():
            # Skip closed positions - they already withdrew funds
            if position.status != "open":
                continue
                
            p = np.array(position.belief_p, dtype=np.float64)
            
            # Compute position's density at outcome_x
            if j_lower == j_upper:
                f_p_x = p[j_lower] / delta_x
            else:
                w_lower = (x_values[j_upper] - outcome_x) / (x_values[j_upper] - x_values[j_lower])
                w_upper = 1.0 - w_lower
                p_lower = p[j_lower] / delta_x
                p_upper = p[j_upper] / delta_x
                f_p_x = w_lower * p_lower + w_upper * p_upper
            
            # Compute log-density (handle zero density)
            if f_p_x > 0:
                log_densities[pos_id] = np.log(f_p_x)
            else:
                log_densities[pos_id] = -np.inf
        
        # STEP 2: Eligibility gate (τ threshold)
        l_max = max(log_densities.values())
        threshold = l_max + np.log(tau)
        eligible_set = {pos_id for pos_id, l_i in log_densities.items() if l_i >= threshold}
        
        # If no positions are eligible, return zero payouts
        if len(eligible_set) == 0:
            return {pos_id: 0.0 for pos_id in self.positions_db.keys()}
        
        # STEP 3: Accuracy shares (a_i) - tempered softmax
        z_scores = {}
        for pos_id in eligible_set:
            l_i = log_densities[pos_id]
            z_scores[pos_id] = np.exp(gamma * (l_i - l_max))
        
        Z = sum(z_scores.values())
        accuracy_shares = {pos_id: z_scores[pos_id] / Z for pos_id in eligible_set}
        
        # STEP 4: Claim shares (s_i) - only for OPEN positions
        M_total_claims = sum(pos.minted_claims_m for pos in self.positions_db.values() if pos.status == "open")
        claim_shares = {
            pos_id: self.positions_db[pos_id].minted_claims_m / M_total_claims
            for pos_id, pos in self.positions_db.items() if pos.status == "open"
        }
        
        # STEP 5: Weights & Payouts
        weights = {}
        for pos_id in eligible_set:
            s_i = claim_shares[pos_id]
            a_i = accuracy_shares[pos_id]
            weights[pos_id] = (s_i ** lambda_s) * (a_i ** lambda_d)
        
        W = sum(weights.values())
        
        # STEP 6: Final payouts (normalized to Pool)
        payouts = {}
        for pos_id, position in self.positions_db.items():
            # Closed positions get None (not eligible for settlement)
            if position.status == "closed":
                payouts[pos_id] = None
            elif pos_id in eligible_set and W > 0:
                payout = Pool * (weights[pos_id] / W)
                payouts[pos_id] = round(payout, 18)
            else:
                payouts[pos_id] = 0.0
        
        return payouts
    
    def execute_settle(self, outcome_x: float) -> Dict[str, float]:
        """Execute final settlement (MUTATES STATE).
        
        Args:
            outcome_x: The realized outcome value
            
        Returns:
            Dictionary mapping position_id to payout amount
        """
        payouts = self.simulate_settle(outcome_x)
        
        # Freeze market
        self.is_settled = True
        self.settlement_outcome = outcome_x
        self.settlement_payouts = payouts
        
        return payouts
    
    def get_current_pool(self) -> float:
        """Calculate current collateral pool from open positions only.
        
        Returns:
            Total collateral currently in the market
        """
        if self.alpha_vector is None:
            return 0.0
        return sum(pos.input_collateral_C for pos in self.positions_db.values() if pos.status == "open")
    
    def get_consensus_pdf(self, num_points: int = 100) -> Tuple[List[float], List[float]]:
        """Compute the public consensus PDF for visualization.
        
        Args:
            num_points: Number of points to sample
            
        Returns:
            (x_values, y_values) for plotting
        """
        if self.alpha_vector is None:
            return [], []
            
        L = self.market_params['L']
        H = self.market_params['H']
        K = self.market_params['K']
        
        x_grid = np.linspace(L, H, num_points)
        y_values = []
        
        x_buckets = np.linspace(L, H, K + 1)
        delta_x = (H - L) / K
        
        alpha = self.alpha_vector
        P = np.sum(alpha)
        
        # FIXED: Precompute categorical densities at bucket centers
        # f_α(x_j) = α_j / (P × Δx) for categorical distribution
        densities = alpha / (P * delta_x)
        
        for x in x_grid:
            # Find bracketing buckets
            if x == L:
                j_lower, j_upper = 0, 0
            elif x == H:
                j_lower, j_upper = K, K
            else:
                j_lower = np.searchsorted(x_buckets, x, side='right') - 1
                j_upper = j_lower + 1
            
            # Interpolate
            if j_lower == j_upper:
                f = densities[j_lower]
            else:
                w_lower = (x_buckets[j_upper] - x) / (x_buckets[j_upper] - x_buckets[j_lower])
                w_upper = 1.0 - w_lower
                
                # Simple linear interpolation of densities
                f = w_lower * densities[j_lower] + w_upper * densities[j_upper]
            
            y_values.append(f)
        
        return x_grid.tolist(), y_values
    
    def get_position_pdf(self, position_id: str, num_points: int = 100) -> Tuple[List[float], List[float]]:
        """Compute a position's belief PDF for visualization.
        
        Args:
            position_id: ID of the position
            num_points: Number of points to sample
            
        Returns:
            (x_values, y_values) for plotting
        """
        if position_id not in self.positions_db:
            return [], []
            
        position = self.positions_db[position_id]
        p = np.array(position.belief_p, dtype=np.float64)
        
        L = self.market_params['L']
        H = self.market_params['H']
        K = self.market_params['K']
        
        x_grid = np.linspace(L, H, num_points)
        y_values = []
        
        x_buckets = np.linspace(L, H, K + 1)
        delta_x = (H - L) / K
        
        # p is a probability vector - convert to density
        densities = p / delta_x
        
        for x in x_grid:
            # Find bracketing buckets
            if x == L:
                j_lower, j_upper = 0, 0
            elif x == H:
                j_lower, j_upper = K, K
            else:
                j_lower = np.searchsorted(x_buckets, x, side='right') - 1
                j_upper = j_lower + 1
            
            # Interpolate
            if j_lower == j_upper:
                f = densities[j_lower]
            else:
                w_lower = (x_buckets[j_upper] - x) / (x_buckets[j_upper] - x_buckets[j_lower])
                w_upper = 1.0 - w_lower
                
                f = w_lower * densities[j_lower] + w_upper * densities[j_upper]
            
            y_values.append(f)
        
        return x_grid.tolist(), y_values
    
    def save_state(self, filepath: str):
        """Save market state to file."""
        state = {
            'alpha_vector': self.alpha_vector.tolist() if self.alpha_vector is not None else None,
            'positions_db': {pid: pos.to_dict() for pid, pos in self.positions_db.items()},
            'market_params': self.market_params,
            'is_settled': self.is_settled,
            'settlement_outcome': self.settlement_outcome,
            'settlement_payouts': self.settlement_payouts,
            'title': self.title
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load market state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.alpha_vector = np.array(state['alpha_vector'], dtype=np.float64) if state['alpha_vector'] else None
        
        self.positions_db = {}
        for pid, pos_dict in state['positions_db'].items():
            self.positions_db[pid] = Position(
                position_id=pos_dict['position_id'],
                belief_p=pos_dict['belief_p'],
                minted_claims_m=pos_dict['minted_claims_m'],
                input_collateral_C=pos_dict['input_collateral_C']
            )
        
        self.market_params = state['market_params']
        self.is_settled = state['is_settled']
        self.settlement_outcome = state['settlement_outcome']
        self.settlement_payouts = state['settlement_payouts']
        self.title = state.get('title', None)
