# hull/options/binomial_tree.py
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


# ─── 1. Risk-neutral probabilities ───────────────────────────────────────────

def compute_risk_neutral_prob(u: float, d: float, r: float, T: float, N: int) -> float:
    """
    Compute the risk-neutral probability p of an up move per step.
    Hull eq. 11.2: p = (e^{r Δt} - d) / (u - d)
    """

    delta_t = T / N

    p = (np.exp(r * delta_t) - d) / (u - d)
    
    return p


# ─── 2. Up/down factors ──────────────────────────────────────────────────────

def compute_up_down_factors(sigma: float, T: float, N: int) -> tuple[float, float]:
    """
    Compute Cox-Ross-Rubinstein u and d from volatility.
    Hull eq. 11.7: u = e^{σ√Δt}, d = 1/u
    """
    delta_t = T / N

    u = np.exp(sigma * (delta_t ** (1/2)))
    d = 1 / u
    
    return [u, d]


# ─── 3. Stock price tree ─────────────────────────────────────────────────────

def build_stock_tree(S0: float, u: float, d: float, N: int) -> np.ndarray:
    """
    Build the (N+1) x (N+1) stock price tree.
    Node (i, j) = S0 * u^j * d^{i-j} where j up-moves in i steps.
    Upper triangle is valid; lower triangle is zero/unused.
    """
    tree = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(i + 1):
            tree[i, j] = S0 * (d ** (i - j)) * (u ** (j))
    
    return tree


# ─── 4. Option payoff at expiry ──────────────────────────────────────────────

def compute_terminal_payoffs(stock_tree: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """
    Compute option payoffs at expiry (final column of the tree).
    option_type: 'call' or 'put'
    """
    row = stock_tree[-1]
    if(option_type == 'call'):
        final = np.maximum(row - K, 0)
    elif(option_type == 'put'):
        final = np.maximum(K - row, 0)
    else: 
        final = np.zeros_like(row)
    
    return final


# ─── 5. European option pricing ──────────────────────────────────────────────

def price_european_option(
    S0: float, K: float, T: float, r: float, sigma: float,
    N: int, option_type: str
) -> float:
    """
    Price a European option by backward induction through the binomial tree.
    At each node: f = e^{-r Δt} * [p * f_u + (1-p) * f_d]
    """
    if sigma is None:
        return 0

    u_d = compute_up_down_factors(sigma, T, N)
    
    u = u_d[0]
    d = u_d[1]

    p = compute_risk_neutral_prob(u, d, r, T, N)

    tree = build_stock_tree(S0, u, d, N,)

    payoff = compute_terminal_payoffs(tree, K, option_type)
    
    option = np.zeros((N+1, N+1))
    
    option[N, :] = payoff

    delta_t = T / N

    for i in range(N-1, -1, -1):
        for j in range(i+1):
            option[i,j] = np.exp(-r * delta_t) * (p * option[i+1,j+1] + (1-p) * option[i+1,j])

    return option[0, 0]
    


# ─── 6. American option pricing ──────────────────────────────────────────────

def price_american_option(
    S0: float, K: float, T: float, r: float, sigma: float,
    N: int, option_type: str
) -> float:
    """
    Price an American option; at each node take max(intrinsic value, continuation value).
    This is the key difference from European pricing.
    """

    u_d = compute_up_down_factors(sigma, T, N)
    
    u = u_d[0]
    d = u_d[1]

    p = compute_risk_neutral_prob(u, d, r, T, N)

    tree = build_stock_tree(S0, u, d, N,)

    payoff = compute_terminal_payoffs(tree, K, option_type)
    
    option = np.zeros((N+1, N+1))
    
    option[N, :] = payoff

    delta_t = T / N

    for i in range(N-1, -1, -1):
        for j in range(i+1):
            if(option_type == 'put'):
                intrinsic = max(K - tree[i,j], 0)  # for put
            else:
                intrinsic = max(tree[i,j] - K, 0)  # for put

            continuation = np.exp(-r * delta_t) * (p * option[i+1,j+1] + (1-p) * option[i+1,j])

            option[i,j] = max(continuation, intrinsic)

    return option[0, 0]
    

# ─── 7. Full option tree (for inspection) ────────────────────────────────────

def build_option_tree(
    S0: float, K: float, T: float, r: float, sigma: float,
    N: int, option_type: str, american: bool = False
) -> np.ndarray:
    """
    Return the full option value tree (N+1) x (N+1) so you can inspect every node.
    Calls the appropriate pricing logic depending on american flag.
    """

    u_d = compute_up_down_factors(sigma, T, N)
    
    u = u_d[0]
    d = u_d[1]

    p = compute_risk_neutral_prob(u, d, r, T, N)

    tree = build_stock_tree(S0, u, d, N,)

    payoff = compute_terminal_payoffs(tree, K, option_type)
    
    option = np.zeros((N+1, N+1))
    
    option[N, :] = payoff

    delta_t = T / N

    for i in range(N-1, -1, -1):
        for j in range(i+1):
            if american:
                if(option_type == 'put'):
                    intrinsic = max(K - tree[i,j], 0)  # for put
                else:
                    intrinsic = max(tree[i,j] - K, 0)  # for put

                continuation = np.exp(-r * delta_t) * (p * option[i+1,j+1] + (1-p) * option[i+1,j])

                option[i,j] = max(continuation, intrinsic)
            else:
                option[i,j] = np.exp(-r * delta_t) * (p * option[i+1,j+1] + (1-p) * option[i+1,j])

    return option


# ─── 8. Greeks via finite difference ─────────────────────────────────────────

def binomial_delta(
    S0: float, K: float, T: float, r: float, sigma: float,
    N: int, option_type: str, american: bool = False
) -> float:
    """
    Estimate Delta from the first two nodes of the tree.
    Hull eq. 11.5: Δ = (f_u - f_d) / (S0*u - S0*d)
    """
    tree = build_option_tree(S0, K, T, r, sigma, N, option_type, american)
    
    u_d = compute_up_down_factors(sigma, T, N)
    
    u = u_d[0]
    d = u_d[1]

    delta = (tree[1, 1] - tree[1, 0]) / (S0 * u - S0 * d)
    
    return delta



# ── Test 1: Hull Example 11.1 (p. 242)
# 3-month European call, S0=20, K=21, r=0.12, u=1.1, d=0.9, N=1
# Expected: p ≈ 0.6523, price ≈ 0.633
u, d = 1.1, 0.9
r_test, T_test = 0.12, 3/12
p = compute_risk_neutral_prob(u, d, r_test, T_test, N=1)
print(f"Test 1 - p: {p:.4f}  (expected ~0.6523)")

# ── Test 2: CRR u/d factors
# sigma=0.3, T=1, N=50 → u ≈ 1.0435, d ≈ 0.9583
u2, d2 = compute_up_down_factors(sigma=0.3, T=1, N=50)
print(f"Test 2 - u: {u2:.4f}, d: {d2:.4f}  (expected u≈1.0435, d≈0.9583)")

# ── Test 3: European call convergence to BS
# S0=100, K=100, T=1, r=0.05, sigma=0.2
# BS price ≈ 10.4506
# With N=500 binomial should be within 0.01 of BS
bs = 10.4506
bn = price_european_option(100, 100, 1, 0.05, 0.2, N=500, option_type='call')
print(f"Test 3 - BS: {bs:.4f}, Binomial N=500: {bn:.4f}  (diff < 0.01)")

# ── Test 4: American put > European put (early exercise premium)
# S0=50, K=52, T=2, r=0.05, sigma=0.3
# American put should be strictly greater than European put
eu_put = price_european_option(50, 52, 2, 0.05, 0.3, N=200, option_type='put')
am_put = price_american_option(50, 52, 2, 0.05, 0.3, N=200, option_type='put')
print(f"Test 4 - European put: {eu_put:.4f}, American put: {am_put:.4f}  (American > European)")

# ── Test 5: American call == European call (no early exercise for non-dividend paying)
eu_call = price_european_option(50, 52, 2, 0.05, 0.3, N=200, option_type='call')
am_call = price_american_option(50, 52, 2, 0.05, 0.3, N=200, option_type='call')
print(f"Test 5 - European call: {eu_call:.4f}, American call: {am_call:.4f}  (should be equal)")