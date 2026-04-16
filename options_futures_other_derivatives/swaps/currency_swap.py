# currency_swap.py
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta



# ── core valuation ────────────────────────────────────────────────────────────

def value_currency_swap_bond_method(
    V_domestic: float,
    V_foreign: float,
    S0: float
) -> float:
    """
    Value a pay-domestic / receive-foreign currency swap as the difference
    between two bond values (Hull Ch 7, bond method).

    Parameters
    ----------
    V_domestic : float
        PV of the domestic-currency bond (the leg we pay), in domestic units.
    V_foreign : float
        PV of the foreign-currency bond (the leg we receive), in foreign units.
    S0 : float
        Spot exchange rate in domestic currency per one unit of foreign currency.

    Returns
    -------
    float
        Value of the swap to the party paying domestic and receiving foreign.
    """
    V_swap = S0 * V_foreign - V_domestic
    
    return V_swap


def price_bond_leg(
    principal: float,
    coupon_rate: float,
    r: float,
    payment_times: list[float]
) -> float:
    """
    Price a single bond leg of a currency swap using continuous compounding.

    Parameters
    ----------
    principal    : notional / face value
    coupon_rate  : annual coupon rate (as a decimal)
    r            : continuously compounded discount rate
    payment_times: list of cash-flow times in years, e.g. [0.5, 1.0, 1.5, 2.0]
                   principal is repaid at payment_times[-1]

    Returns
    -------
    float
        PV of all cash flows on this leg.
    """
    PV = np.sum(principal * coupon_rate * np.exp(-r * np.array(payment_times))) + principal * np.exp(-r * payment_times[-1])
    
    return PV


def value_currency_swap_forward_method(
    principal_domestic: float,
    coupon_domestic: float,
    r_domestic: float,
    principal_foreign: float,
    coupon_foreign: float,
    r_foreign: float,
    S0: float,
    rf: float,
    payment_times: list[float]
) -> float:
    """
    Value a currency swap as a portfolio of currency forward contracts,
    one for each net cash flow exchange (Hull Ch 7, forward method).

    Parameters
    ----------
    principal_domestic  : notional on the domestic leg
    coupon_domestic     : annual coupon rate on the domestic leg
    r_domestic          : continuously compounded domestic discount rate
    principal_foreign   : notional on the foreign leg
    coupon_foreign      : annual coupon rate on the foreign leg
    r_foreign           : continuously compounded foreign discount rate
    S0                  : spot exchange rate (domestic per foreign)
    rf                  : foreign continuously compounded risk-free rate
                          (used to build forward rates via covered interest parity)
    payment_times       : list of payment times in years

    Returns
    -------
    float
        Value of the swap to the party receiving foreign, paying domestic.
    """
    V_swap = 0

    for n in range(len(payment_times)):
        F_t = forward_exchange_rate(S0, r_domestic, r_foreign, payment_times[n])
        foreign_cash_flow = principal_foreign * coupon_foreign * F_t
        domestic_cash_flow = principal_domestic * coupon_domestic

        net = foreign_cash_flow - domestic_cash_flow

        discounted = net * np.exp(-r_domestic * payment_times[n])

        V_swap += discounted

    F_t = forward_exchange_rate(S0, r_domestic, r_foreign, payment_times[-1])

    foreign_cash_flow = principal_foreign * F_t
    domestic_cash_flow = principal_domestic

    net = foreign_cash_flow - domestic_cash_flow

    V_swap += net * np.exp(-r_domestic * payment_times[-1])

    return V_swap


def forward_exchange_rate(S0: float, r_d: float, r_f: float, T: float) -> float:
    """
    Compute the forward exchange rate at time T using covered interest rate
    parity (continuous compounding).
    """
    F = S0 * np.exp((r_d - r_f) * T)

    return F


# ── analysis ──────────────────────────────────────────────────────────────────

def swap_cash_flows(
    principal_domestic: float,
    coupon_domestic: float,
    principal_foreign: float,
    coupon_foreign: float,
    S0: float,
    r_d: float,
    r_f: float,
    payment_times: list[float]
) -> pd.DataFrame:
    """
    Build a DataFrame showing each payment date's domestic cash flow, foreign
    cash flow converted to domestic at the forward rate, and the net cash flow.
    """
    rows = []
    for n in range(len(payment_times)):
        domestic_cash_flow = principal_domestic * coupon_domestic
        F_t = forward_exchange_rate(S0, r_d, r_f, payment_times[n])
        foreign_cash_flow = principal_foreign * coupon_foreign * F_t

        net = foreign_cash_flow - domestic_cash_flow
        if n == len(payment_times) - 1:
            domestic_cash_flow += principal_domestic
            foreign_cash_flow += principal_foreign * F_t
            net = foreign_cash_flow - domestic_cash_flow
        rows.append([domestic_cash_flow, foreign_cash_flow, net])

    df = pd.DataFrame(rows, columns=["Domestic Cash Flow", "Foreign Cash Flow (Domestic Terms)", "Net Cash Flow"])

    return df

def duration_of_swap(
    principal: float,
    coupon_rate: float,
    r: float,
    payment_times: list[float]
) -> float:
    """
    Compute the duration (Macaulay) of one bond leg of the swap — used to
    assess interest rate sensitivity of each leg.
    """
    
    D = 0
    for n in range(len(payment_times) - 1):
        CF_t = principal * coupon_rate

        D += payment_times[n] * CF_t * np.exp(-r * payment_times[n])
    
    CF_t = principal * coupon_rate + principal

    D += payment_times[-1] * CF_t * np.exp(-r * payment_times[-1])
    
    D /= price_bond_leg(principal, coupon_rate, r, payment_times)
    
    return D


# ── Test 1: forward_exchange_rate ──────────────────────────────────────
# Hull example: S0=0.65, r_d=0.05 (USD), r_f=0.08 (GBP), T=1
# F = 0.65 * exp((0.05 - 0.08) * 1) = 0.65 * exp(-0.03) ≈ 0.6308
F = forward_exchange_rate(S0=0.65, r_d=0.05, r_f=0.08, T=1.0)
print(f"Test 1 - Forward rate (expect ~0.6308): {F}")

# ── Test 2: price_bond_leg ─────────────────────────────────────────────
# 3-year bond, principal=10M, coupon=8% annual paid annually, r=5%
# PV = 0.8e6*exp(-0.05) + 0.8e6*exp(-0.10) + 10.8e6*exp(-0.15)
#    ≈ 761,386 + 724,664 + 9,267,359 ≈ 10,753,409
pv = price_bond_leg(
    principal=10_000_000,
    coupon_rate=0.08,
    r=0.05,
    payment_times=[1.0, 2.0, 3.0]
)
print(f"Test 2 - Bond leg PV (expect ~10,753,409): {pv}")

# ── Test 3: value_currency_swap_bond_method ────────────────────────────
# Pay USD leg (V_domestic), receive GBP leg (V_foreign)
# V_domestic = 10,753,409 (from test 2 at 5%)
# V_foreign  = price GBP leg: principal=7M, coupon=4%, r=8%, T=[1,2,3]
# PV_gbp = 0.28e6*exp(-0.08) + 0.28e6*exp(-0.16) + 7.28e6*exp(-0.24)
#        ≈ 258,800 + 239,013 + 5,684,075 ≈ 6,181,888 GBP
# S0 = 1.4286 (USD per GBP)
# Swap value = 1.4286 * 6,181,888 - 10,753,409 ≈ -930,000  (approx)
V_d = pv
V_f = price_bond_leg(7_000_000, 0.04, 0.08, [1.0, 2.0, 3.0])
val = value_currency_swap_bond_method(V_d, V_f, S0=1.4286)
print(f"Test 3 - Swap value bond method (expect ~ -930,000): {val}")

# ── Test 4: value_currency_swap_forward_method ────────────────────────
# Same swap as test 3 — both methods must agree to within rounding
val_fwd = value_currency_swap_forward_method(
    principal_domestic=10_000_000, coupon_domestic=0.08, r_domestic=0.05,
    principal_foreign=7_000_000,  coupon_foreign=0.04,  r_foreign=0.08,
    S0=1.4286, rf=0.08,
    payment_times=[1.0, 2.0, 3.0]
)
print(f"Test 4 - Swap value fwd method  (expect ~ -930,000): {val_fwd}")
print(f"         Bond vs Fwd difference: {abs(val - val_fwd)}  (should be ~0)")

# ── Test 5: swap_cash_flows DataFrame ─────────────────────────────────
# Verify net cash flows at each period match what forward method accumulates
df = swap_cash_flows(10_000_000, 0.08, 7_000_000, 0.04, 1.4286, 0.05, 0.08, [1.0, 2.0, 3.0])
print("\nTest 5 - Cash flow schedule:")
print(df.to_string(index=False))

# ── Test 6: duration_of_swap ───────────────────────────────────────────
# 3-year, 8% annual, r=5%, principal=10M
# Macaulay duration should be slightly under 3 years (< 3)
dur = duration_of_swap(10_000_000, 0.08, 0.05, [1.0, 2.0, 3.0])
print(f"\nTest 6 - Duration of 8% bond at 5% (expect < 3.0 yrs): {dur}")