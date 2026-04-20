import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import norm


def black_scholes_currency_call(S_0, K, r, r_f, sigma, T):
    """
    Price a European call on a currency using Garman-Kohlhagen (S_0 is spot exchange rate in domestic per unit foreign).
    """
    d_1 = (np.log(S_0 / K) + (((r - r_f) + (sigma ** 2 / 2)) * T)) / (sigma * (T ** (1/2)))

    d_2 = d_1  - sigma * (T ** (1/2))


    c = S_0 * np.exp(-r_f * T) * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)

    return c


def black_scholes_currency_put(S_0, K, r, r_f, sigma, T):
    """
    Price a European put on a currency using Garman-Kohlhagen.
    """
    d_1 = (np.log(S_0 / K) + (((r - r_f) + (sigma ** 2 / 2)) * T)) / (sigma * (T ** (1/2)))

    d_2 = d_1  - sigma * (T ** (1/2))

    p = K * np.exp(-r * T) * norm.cdf(-d_2) - S_0 * np.exp(-r_f * T) * norm.cdf(-d_1)

    return p


# S_0 = 1.6000 (USD per GBP), K = 1.6000, r = 0.08, r_f = 0.11, sigma = 0.141, T = 4/12
S_0, K, r, r_f, sigma, T = 1.56, 1.6000, 0.06, 0.08, 0.12, 1/2

c = black_scholes_currency_call(S_0, K, r, r_f, sigma, T)
p = black_scholes_currency_put(S_0, K, r, r_f, sigma, T)
print(f"Call: {c:.4f}")   # Expected: ~0.0639
print(f"Put: {p:.4f}")    # Expected: ~0.0799