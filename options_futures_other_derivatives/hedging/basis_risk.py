import numpy as np
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')

def basis(spot: float, futures: float) -> float:
    """Current basis"""

    basis = spot - futures

    return basis

def expected_payoff(spot_price_t2: float,
                    futures_price_t1: float,
                    futures_price_t2: float,
                    quantity: float) -> float:
    """
    Payoff of hedged position
    t1 = time hedge is opened
    t2 = time hedge is closed
    """

    spot_payoff = spot_price_t2 * quantity
    futures_payoff = (futures_price_t1 - futures_price_t2) * quantity

    total = spot_payoff + futures_payoff

    return total

def hedge_effectiveness(spot_returns, futures_returns) -> float:
    """
    R-squared of regression of spot on futures returns
    Measures what proportion of spot risk is eliminated by hedge
    """

    rho = np.corrcoef(spot_returns, futures_returns)[0, 1]
    
    return rho ** 2

def _scalar(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)



print("Basis: ", basis(679.46, 685.77))

spot_price_t2 = 105
futures_price_t1 = 102
futures_price_t2 = 104
quantity = 1000

total = expected_payoff(spot_price_t2=spot_price_t2, futures_price_t1=futures_price_t1, futures_price_t2=futures_price_t2, quantity=quantity)

print("Expected total: ", total)

spot_prices = yf.download("SPY", period="1y", progress=False)['Close'].squeeze().dropna()
futures_prices = yf.download("ES=F", period="1y", progress=False)['Close'].squeeze().dropna()

spots_returns = spot_prices.pct_change().dropna()
futures_returns = futures_prices.pct_change().dropna()
spots_returns, futures_returns = spots_returns.align(futures_returns, join='inner')

effectiveness = hedge_effectiveness(spot_returns=spots_returns, futures_returns=futures_returns)

print("Effectiveness ", effectiveness)