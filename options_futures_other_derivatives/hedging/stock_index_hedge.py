import numpy as np
import yfinance as yf
import pandas as pd

def compute_beta(portfolio_ticker: str, index_ticker: str, period: str = "1y") -> float:
    # beta = cov(portfolio, index) / var(index)

    portfolio_prices = yf.download(portfolio_ticker, period=period, progress=False)['Close'].squeeze().dropna()
    index_prices = yf.download(index_ticker, period=period, progress=False)['Close'].squeeze().dropna()

    portfolio_returns = portfolio_prices.pct_change().dropna()
    index_returns = index_prices.pct_change().dropna()
    portfolio_returns, index_returns = portfolio_returns.align(index_returns, join='inner')

    beta = np.cov(portfolio_returns, index_returns)[0, 1] / np.var(index_returns)
    
    return beta

def contracts_to_hedge(beta: float, portfolio_value: float,
                       futures_price: float, contract_size: float) -> int:
    # fully hedge portfolio
    
    futures_contracts_value = futures_price * contract_size
    
    N = beta * (portfolio_value / futures_contracts_value)

    return N

def contracts_to_target_beta(beta_current: float, beta_target: float,
                              portfolio_value: float, futures_price: float,
                              contract_size: float) -> int:
    # change portfolio beta to target

    futures_contract_value = futures_price * contract_size

    N = (beta_target - beta_current) * (portfolio_value / futures_contract_value)

    return N

def _scalar(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)

es = yf.download("MES=F", period="5d", progress=False)
futures_price = _scalar(es['Close'].iloc[-1])

# futures_price = 5263.75  # approximate ES=F Friday close


beta = compute_beta("AAPL", "SPY")
portfolio_value = 1000000
contract_size = 50

N = contracts_to_hedge(beta, portfolio_value, futures_price, contract_size)

print("Beta: ", beta)
print("N: ", N)

target_beta = 0.5
N2 = contracts_to_target_beta(beta, target_beta, portfolio_value, futures_price, contract_size)

print("N2: ", N2)