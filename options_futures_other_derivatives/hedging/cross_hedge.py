import numpy as np
import yfinance as yf

def cross_hedge_ratio(spot_ticker: str, futures_ticker: str, period: str = "1y") -> float:
    # same as hedge_ratio but explicitly for non-identical assets
    # returns h = rho * (sigma_s / sigma_f)

    spot_prices = yf.download(spot_ticker, period=period, progress=False)['Close'].squeeze().dropna()
    futures_prices = yf.download(futures_ticker, period=period, progress=False)['Close'].squeeze().dropna()

    spots_returns = spot_prices.pct_change().dropna()
    futures_returns = futures_prices.pct_change().dropna()
    spots_returns, futures_returns = spots_returns.align(futures_returns, join='inner')


    rho = np.corrcoef(spots_returns, futures_returns)[0, 1]
    sigma_s = np.std(spots_returns)
    sigma_f = np.std(futures_returns)

    h = rho * (sigma_s / sigma_f)

    quality = hedge_quality(spots_returns, futures_returns)

    return h, quality

def contracts_needed(h: float, portfolio_value: float, 
                     futures_price: float, contract_size: float) -> int:
    # number of futures contracts needed
    
    futures_contracts_value = futures_price * contract_size
    
    N = h * (portfolio_value / futures_contracts_value)

    return N

def hedge_quality(spot_returns, futures_returns) -> dict:
    # returns correlation, r_squared, and recommendation
    # if r_squared < 0.5 warn that cross hedge is poor quality
    
    rho = np.corrcoef(spot_returns, futures_returns)[0, 1]

    rho_squared = rho ** 2

    if(rho_squared < 0.5):
        print("Poor hedge")
    
    return {
        'correlation': rho,
        'r_squared': rho_squared,
        'recommendation': 'Poor cross hedge' if rho_squared < 0.5 else 'Acceptable cross hedge'
    }


h, quality = cross_hedge_ratio("QQQ", "SPY")

print(f"QQQ/SPY hedge ratio: ", h)
print(f"Quality: {quality}")

# bad cross hedge
h2, quality2 = cross_hedge_ratio("GLD", "ES=F")
print(f"GLD/ES hedge ratio: ", h2)
print(f"Quality: {quality2}")