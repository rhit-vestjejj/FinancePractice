import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import norm


def black_scholes_index_call(S_0, K, r, q, sigma, T):
    """
    Price a European call on a stock index paying continuous dividend yield q.
    """

    d_1 = (np.log(S_0 / K) + (((r - q) + (sigma ** 2 / 2)) * T)) / (sigma * (T ** (1/2)))

    d_2 = d_1  - sigma * (T ** (1/2))


    c = S_0 * np.exp(-q * T) * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)

    return c


def black_scholes_index_put(S_0, K, r, q, sigma, T):
    """
    Price a European put on a stock index paying continuous dividend yield q.
    """
    d_1 = (np.log(S_0 / K) + (((r - q) + (sigma ** 2 / 2)) * T)) / (sigma * (T ** (1/2)))

    d_2 = d_1  - sigma * (T ** (1/2))

    p = K * np.exp(-r * T) * norm.cdf(-d_2) - S_0 * np.exp(-q * T) * norm.cdf(-d_1)

    return p


def verify_put_call_parity(S_0, K, r, q, T, c, p):
    """
    Check put-call parity for index options: c + K*e^(-rT) = p + S_0*e^(-qT).
    Returns (lhs, rhs) so you can confirm they match.
    """

    lhs = c + K* np.exp(-r * T)
    rhs = p + S_0* np.exp(-q * T)

    return lhs, rhs 

# Hull Example 14.2 (or similar textbook values)
# S_0 = 930, K = 900, r = 0.08, q = 0.03, sigma = 0.20, T = 2/12
S_0, K, r, q, sigma, T = 930, 900, 0.08, 0.03, 0.20, 2/12

c = black_scholes_index_call(S_0, K, r, q, sigma, T)
p = black_scholes_index_put(S_0, K, r, q, sigma, T)
print(f"Call: {c:.2f}")   # Expected: ~51.83
print(f"Put: {p:.2f}")    # Expected: ~14.19

lhs, rhs = verify_put_call_parity(S_0, K, r, q, T, c, p)
print(f"Parity LHS: {lhs:.4f}, RHS: {rhs:.4f}")  # Should match