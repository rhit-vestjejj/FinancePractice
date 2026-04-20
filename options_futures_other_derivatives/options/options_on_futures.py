import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import norm


def blacks_model_call(F_0, K, r, sigma, T):
    """
    Price a European call on a futures contract using Black's model.
    """
    d1=(np.log(F_0/K)+(sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    return np.exp(-r*T)*(F_0*norm.cdf(d1)-K*norm.cdf(d2))


def blacks_model_put(F_0, K, r, sigma, T):
    """
    Price a European put on a futures contract using Black's model.
    """
    d1=(np.log(F_0/K)+(sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    return np.exp(-r*T)*(K*norm.cdf(-d2)-F_0*norm.cdf(-d1))


# F_0 = 240, K = 240, r = 0.09, sigma = 0.25, T = 4/12
F_0, K, r, sigma, T = 240, 240, 0.09, 0.25, 4/12

c = blacks_model_call(F_0, K, r, sigma, T)
p = blacks_model_put(F_0, K, r, sigma, T)
print(f"Call: {c:.2f}")   # Expected: ~12.56
print(f"Put: {p:.2f}")    # Expected: ~12.56  (ATM + same formula structure = symmetric)