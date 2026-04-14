import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import Optional
import math

# ─── CHAPTER 6: INTEREST RATE FUTURES ────────────────────────────────────────


def get_treasury_data(tenor: str = "^TNX") -> pd.Series:
    """Fetch current Treasury yield data from yfinance for use in pricing."""
    pass


def bond_futures_theoretical_price(
    B: float,
    r: float,
    T: float,
    c_i: np.ndarray,
    t_i: np.ndarray
) -> float:
    """
    Compute the theoretical futures price for a Treasury bond futures contract
    using Hull's cost-of-carry model adjusted for known cash flows.

    Parameters
    ----------
    B  : current bond (spot) price (full/dirty price)
    r  : risk-free rate (continuously compounded)
    T  : time to futures delivery (in years)
    c_i : array of coupon payments paid during [0, T]
    t_i : array of times at which each coupon in c_i is paid (in years)

    Returns
    -------
    F_0 : theoretical futures price
    """
    I = sum(c_i * np.exp(-r * t_i))

    F_0 = (B - I) * np.exp(r * T)
    
    return F_0


def conversion_factor(
    coupon_rate: float,
    maturity_years: float,
    face: float = 100.0,
    ytm: float = 0.06,
    freq: int = 2
) -> float:
    """
    Compute the CME conversion factor for a T-bond deliverable,
    which normalises the bond as if it yields 6% with coupons
    rounded to the nearest quarter-year.

    Parameters
    ----------
    coupon_rate   : annual coupon rate of the deliverable bond
    maturity_years: remaining maturity in years (rounded to nearest quarter)
    face          : face value (default 100)
    ytm           : reference yield used by CME (default 0.06)
    freq          : coupon frequency per year (default 2 = semi-annual)

    Returns
    -------
    CF : conversion factor (dimensionless)
    """

    N = round(maturity_years * freq)

    C_period = coupon_rate * face / freq

    k = np.arange(1, N + 1)
    pv_coupons = np.sum(C_period / (1 + ytm/freq)**k)
        
    CF = (1 / face) * (pv_coupons  + face / ((1 + (ytm / freq)) ** N))
    
    return CF


def cheapest_to_deliver(
    quoted_prices: np.ndarray,
    conversion_factors: np.ndarray,
    F_0: float
) -> int:
    """
    Identify the cheapest-to-deliver (CTD) bond from a set of deliverables
    by minimising (Quoted Price - F_0 * CF) across all candidates.

    Parameters
    ----------
    quoted_prices      : array of quoted (clean) prices for each deliverable
    conversion_factors : array of conversion factors, one per deliverable
    F_0                : quoted futures price

    Returns
    -------
    idx : index of the CTD bond in the input arrays
    """

    cheaptest = np.argmin(quoted_prices - F_0 * conversion_factors)
    
    return cheaptest


def eurodollar_futures_rate(
    futures_price: float
) -> float:
    """
    Convert a quoted Eurodollar futures price to its implied 3-month LIBOR rate
    using Hull's standard Z = 100(1 - R) convention.

    Parameters
    ----------
    futures_price : quoted Eurodollar futures price (e.g. 94.50)

    Returns
    -------
    R : implied 3-month LIBOR / SOFR rate (annualised, decimal)
    """
    R = 1 - futures_price / 100

    return R


def convexity_adjustment(
    sigma: float,
    T1: float,
    T2: float
) -> float:
    """
    Compute Hull's convexity adjustment to convert a Eurodollar futures rate
    to a forward rate, accounting for the daily marking-to-market effect.

    Parameters
    ----------
    sigma : annual standard deviation of short-rate changes
    T1    : time to futures expiration (years)
    T2    : time to end of the rate underlying the futures (years, T2 = T1 + 0.25)

    Returns
    -------
    adj : convexity adjustment (to be subtracted from futures rate to get forward rate)
    """
    adj = (1.0/2) * (sigma ** 2) * T1 * T2
    
    return adj


def forward_rate_from_eurodollar(
    futures_price: float,
    sigma: float,
    T1: float,
    T2: Optional[float] = None
) -> float:
    """
    Derive the continuously compounded forward rate from a Eurodollar futures
    price after applying the convexity adjustment.

    Parameters
    ----------
    futures_price : quoted Eurodollar futures price
    sigma         : annual std dev of short-rate changes
    T1            : time to futures expiration (years)
    T2            : time to end of reference period (default T1 + 0.25)

    Returns
    -------
    r_fwd : forward rate (continuously compounded, decimal)
    """

    if T2 is None: T2 = T1 + 0.25

    future_rate = eurodollar_futures_rate(futures_price)
    convexity = convexity_adjustment(sigma, T1, T2)

    future_rate = 4 * np.log(1 + (future_rate / 4))

    forward_rate = future_rate - convexity



    return forward_rate


def duration_based_hedge_ratio(
    P: float,
    D_P: float,
    V_F: float,
    D_F: float
) -> float:
    """
    Compute the number of interest rate futures contracts needed to hedge
    a bond portfolio using Hull's duration-based hedge ratio.

    Parameters
    ----------
    P   : value of the portfolio being hedged
    D_P : duration of the portfolio
    V_F : value of one futures contract (futures price * contract size)
    D_F : duration of the cheapest-to-deliver bond underlying the futures

    Returns
    -------
    N_star : optimal number of futures contracts (positive = short futures)
    """
    N_star = (P * D_P) / (V_F * D_F)
    
    return N_star


def hedge_bond_portfolio(
    ticker: str,
    portfolio_value: float,
    portfolio_duration: float,
    futures_price: float,
    ctd_duration: float,
    contract_size: float = 100_000.0
) -> dict:
    """
    Fetch current yield data and compute a full duration-based hedge for a
    bond portfolio, returning the hedge ratio, contract count, and hedge metrics.

    Parameters
    ----------
    ticker            : yfinance ticker for a proxy bond ETF (e.g. 'TLT')
    portfolio_value   : current market value of the bond portfolio
    portfolio_duration: dollar duration of the portfolio
    futures_price     : current T-bond futures quoted price (per $100 face)
    ctd_duration      : duration of the cheapest-to-deliver bond
    contract_size     : notional per futures contract (default $100,000)

    Returns
    -------
    dict with keys: N_star, V_F, D_F, portfolio_value, hedge_pnl_per_bp
    """
    
    V_F = futures_price * contract_size / 100

    N_star = duration_based_hedge_ratio(portfolio_value, portfolio_duration, V_F, ctd_duration)

    portfolio_dv01 = portfolio_value * portfolio_duration * 0.0001
    futures_dv01 = N_star * V_F * ctd_duration * 0.0001
    hedge_pnl_per_bp = portfolio_dv01 - futures_dv01
        
    return {
        'N_star': N_star,
        'V_F': V_F,
        'D_F': ctd_duration,
        'portfolio_value': portfolio_value,
        'hedge_pnl_per_bp': hedge_pnl_per_bp
    }


print("=" * 60)
print("TEST 1: bond_futures_theoretical_price")
print("=" * 60)
# Hull Example (Ch 6): 
# Spot price B = 115.00, r = 10% cont. compounded
# T = 9/12 = 0.75 years, one coupon of 6.00 paid at t=0.5
B = 121.978 # dirty / cash price)
r = 0.10
T = 0.7397
c_i = np.array([6.00])
t_i = np.array([0.3342])
F0 = bond_futures_theoretical_price(B, r, T, c_i, t_i)
print(f"F_0 = {F0}")
print(f"Expected: ~109.1030\n")  # Hull Table 6.2

print("=" * 60)
print("TEST 2: conversion_factor")
print("=" * 60)
# 10% coupon bond with 20 years to maturity
CF = conversion_factor(coupon_rate=0.10, maturity_years=20.0)
print(f"CF = {CF}")
print(f"Expected: ~1.4623  (bond at premium since 10% > 6% reference)\n")

# 8% coupon bond with 18.5 years → rounds to 18.25 at CME
CF2 = conversion_factor(coupon_rate=0.08, maturity_years=18.5)
print(f"CF (8%, 18.5yr) = {CF2}")
print(f"Expected: ~1.2615\n")

print("=" * 60)
print("TEST 3: cheapest_to_deliver")
print("=" * 60)
quoted = np.array([99.50, 143.50, 119.75])
cfs    = np.array([1.0382, 1.5188, 1.2615])
F_0    = 93.25
idx = cheapest_to_deliver(quoted, cfs, F_0)
print(f"CTD bond index: {idx}")
costs = quoted - F_0 * cfs
print(f"Costs (Q - F*CF): {np.round(costs, 4)}")
print(f"Expected: bond with lowest cost is CTD\n")

print("=" * 60)
print("TEST 4: eurodollar_futures_rate + convexity_adjustment")
print("=" * 60)
futures_price = 94.00
R = eurodollar_futures_rate(futures_price)
print(f"Implied LIBOR rate: {R}  (Expected: 0.06)")

sigma = 0.012
T1, T2 = 2.0, 2.25
adj = convexity_adjustment(sigma, T1, T2)
print(f"Convexity adj: {adj}  (Expected: ~0.000324 for sigma=0.012, T1=2)\n")

print("=" * 60)
print("TEST 5: forward_rate_from_eurodollar")
print("=" * 60)
r_fwd = forward_rate_from_eurodollar(
    futures_price=94.00, sigma=0.012, T1=2.0
)
print(f"Forward rate (cont. compounded): {r_fwd}")
print(f"Expected: slightly below 0.06 after convexity adj and rate conversion\n")

print("=" * 60)
print("TEST 6: duration_based_hedge_ratio")
print("=" * 60)
# Portfolio: $10M, duration 7.1 years
# Futures: price 91-12 (~91.375), contract $100k face, CTD duration 8.8yr
P = 10_000_000
D_P = 7.1
futures_q = 91 + 12/32   # 91-12 in 32nds
V_F = futures_q * 1000   # per $100k contract → multiply by 1000
D_F = 8.8
N = duration_based_hedge_ratio(P, D_P, V_F, D_F)
print(f"N* = {N} contracts")
print(f"Expected: ~88.90 contracts (round to 89)\n")

print("=" * 60)
print("TEST 7: hedge_bond_portfolio (live data)")
print("=" * 60)
result = hedge_bond_portfolio(
    ticker='TLT',
    portfolio_value=10_000_000,
    portfolio_duration=7.1,
    futures_price=91.375,
    ctd_duration=8.8,
    contract_size=100_000
)
print(pd.Series(result).round(4))