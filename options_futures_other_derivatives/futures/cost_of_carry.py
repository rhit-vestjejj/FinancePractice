import numpy as np

def carry_stock(r: float, q: float) -> float:
    """
    Net cost of carry for a stock
    r = risk free rate
    q = continuous dividend yield
    """

    cost = r - q

    return cost

def carry_commodity(r: float, u: float, y: float) -> float:
    """
    Net cost of carry for a commodity
    r = risk free rate
    u = storage cost rate
    y = convenience yield
    """

    cost = r + u - y

    return cost

def carry_currency(r: float, rf: float) -> float:
    """
    Net cost of carry for a currency
    r = domestic risk free rate
    rf = foreign risk free rate
    """

    cost = r - rf

    return cost

def forward_carry(S: float, c: float, T: float) -> float:
    """
    Forward price given cost of carry
    S = spot price
    c = net cost of carry
    T = time to maturity in years
    """

    F = S * np.exp(c * T)
    
    return F


S = 679.46
c = carry_stock(0.05, 0.013)
T = 0.25
F = forward_carry(S, c, T)
print(f"SPY Forward via carry: ", F)

c_gold = carry_commodity(0.05, 0.005, 0.0)
print(f"Gold carry: ", c_gold)

c_fx = carry_currency(0.05, 0.03)
print(f"EUR/USD carry: ", c_fx)