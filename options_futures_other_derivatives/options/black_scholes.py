import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import norm


def d1(S, K, r, T, sigma):
    """
    Compute d1 from the Black-Scholes-Merton formula.
    
    Parameters
    ----------
    S : float - Current stock price
    K : float - Strike price
    r : float - Risk-free rate (continuous compounding)
    T : float - Time to maturity in years
    sigma : float - Volatility (annualized)
    
    Returns
    -------
    float
    """

    d_1 = (np.log(S / K) + ((r + (sigma ** 2 / 2)) * T)) / (sigma * (T ** (1/2)))

    return d_1


def d2(S, K, r, T, sigma):
    """
    Compute d2 from the Black-Scholes-Merton formula.
    
    Parameters
    ----------
    S : float - Current stock price
    K : float - Strike price
    r : float - Risk-free rate (continuous compounding)
    T : float - Time to maturity in years
    sigma : float - Volatility (annualized)
    
    Returns
    -------
    float
    """

    d_1 = d1(S, K, r, T, sigma)

    d_2 = d_1  - sigma * (T ** (1/2))
    
    return d_2


def bs_call(S, K, r, T, sigma):
    """
    Compute the Black-Scholes price of a European call option.
    
    c = S * N(d1) - K * exp(-rT) * N(d2)
    
    Parameters
    ----------
    S : float - Current stock price
    K : float - Strike price
    r : float - Risk-free rate (continuous compounding)
    T : float - Time to maturity in years
    sigma : float - Volatility (annualized)
    
    Returns
    -------
    float
    """

    d_1 = d1(S, K, r, T, sigma)

    d_2 = d2(S, K, r, T, sigma)

    c = S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)

    return c


def bs_put(S, K, r, T, sigma):
    """
    Compute the Black-Scholes price of a European put option.
    
    p = K * exp(-rT) * N(-d2) - S * N(-d1)
    
    Parameters
    ----------
    S : float - Current stock price
    K : float - Strike price
    r : float - Risk-free rate (continuous compounding)
    T : float - Time to maturity in years
    sigma : float - Volatility (annualized)
    
    Returns
    -------
    float
    """

    d_1 = d1(S, K, r, T, sigma)

    d_2 = d2(S, K, r, T, sigma)

    p = K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

    return p


def bs_call_put_parity_check(S, K, r, T, sigma):
    """
    Verify put-call parity holds: c - p = S - K*exp(-rT).
    Returns the absolute difference between the two sides.
    Should be ~0 if your implementations are correct.
    
    Returns
    -------
    float - Absolute parity error
    """

    c = bs_call(S, K, r, T, sigma)

    p = bs_put(S, K, r, T, sigma)

    return (c - p) - (S - K * np.exp(-r * T))


# Hull Example: S=42, K=40, r=0.10, T=0.5, sigma=0.20
S = 42.0
K = 40.0
r = 0.10
T = 0.5
sigma = 0.20

print("=== Black-Scholes-Merton ===")
print(f"S={S}, K={K}, r={r}, T={T}, sigma={sigma}")
print()

# d1 and d2
# Expected: d1 = 0.7693, d2 = 0.6278
d1_val = d1(S, K, r, T, sigma)
d2_val = d2(S, K, r, T, sigma)
print(f"d1 = {d1_val:.4f}  (expected: 0.7693)")
print(f"d2 = {d2_val:.4f}  (expected: 0.6278)")
print()

# Call price
# Expected: c = 4.76
c = bs_call(S, K, r, T, sigma)
print(f"Call price = {c:.2f}  (expected: 4.76)")

# Put price
# Expected: p = 0.81
p = bs_put(S, K, r, T, sigma)
print(f"Put price  = {p:.2f}  (expected: 0.81)")
print()

# Put-call parity check
parity_err = bs_call_put_parity_check(S, K, r, T, sigma)
print(f"Put-call parity error = {parity_err:.10f}  (expected: ~0.0)")
print()

# Additional test: ATM option, S=K=100, r=0.05, T=1.0, sigma=0.30
S2, K2, r2, T2, sigma2 = 100.0, 100.0, 0.05, 1.0, 0.30
c2 = bs_call(S2, K2, r2, T2, sigma2)
p2 = bs_put(S2, K2, r2, T2, sigma2)
print(f"ATM Call (S=K=100, r=0.05, T=1, sigma=0.30) = {c2:.4f}  (expected: 14.2312)")
print(f"ATM Put  (S=K=100, r=0.05, T=1, sigma=0.30) = {p2:.4f}  (expected: 9.3542)")

# Edge case: very deep ITM call
c3 = bs_call(100, 50, 0.05, 1.0, 0.20)
print(f"Deep ITM Call (S=100, K=50) = {c3:.2f}  (expected: ~51.44)")

# Edge case: very deep OTM call
c4 = bs_call(100, 200, 0.05, 1.0, 0.20)
print(f"Deep OTM Call (S=100, K=200) = {c4:.6f}  (expected: ~0.000000)")