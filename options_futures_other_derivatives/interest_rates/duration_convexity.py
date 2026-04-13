import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────
# FUNCTION SKELETONS
# ─────────────────────────────────────────────

def macaulay_duration(cash_flows: np.ndarray, times: np.ndarray, ytm: float) -> float:
    """
    Compute Macaulay duration: the PV-weighted average time to receive cash flows,
    expressed in years. Hull eq (4.7).

    Parameters
    ----------
    cash_flows : np.ndarray
        Cash flow at each period (coupon + face at maturity).
    times : np.ndarray
        Time in years corresponding to each cash flow.
    ytm : float
        Yield to maturity (continuously compounded).
    face : float
        Face value of the bond (used only for reference, cash_flows already encodes it).

    Returns
    -------
    float
        Macaulay duration in years.
    """
    P = bond_price_from_ytm(cash_flows, times, ytm)

    D = (1/P) * np.sum(times * cash_flows * np.exp(-ytm * times))

    return D


def modified_duration(cash_flows: np.ndarray, times: np.ndarray, ytm: float, m: int = 2) -> float:
    """
    Compute modified duration from Macaulay duration: D* = D / (1 + y/m),
    where y is the yield with compounding frequency m. Hull eq (4.8).

    Parameters
    ----------
    cash_flows : np.ndarray
    times : np.ndarray
    ytm : float
        Yield expressed with compounding frequency m (NOT continuously compounded).
    m : int
        Compounding frequency per year (2 = semiannual, 1 = annual).

    Returns
    -------
    float
        Modified duration.
    """
    B = np.sum(cash_flows * (1 + ytm/m) ** (-times * m))
    D = np.sum(times * cash_flows * (1 + ytm/m) ** (-times * m)) / B
    
    D_mod =  D / (1 + ytm/m)

    return D_mod


def dollar_duration(cash_flows: np.ndarray, times: np.ndarray, ytm: float) -> float:
    """
    Compute dollar duration (DV01 scaled): D* × B, the dollar change in bond price
    per unit change in yield. Also called 'price value of a basis point' when scaled by 0.0001.

    Parameters
    ----------
    cash_flows : np.ndarray
    times : np.ndarray
    ytm : float
        Continuously compounded yield.

    Returns
    -------
    float
        Dollar duration.
    """

    D_star = macaulay_duration(cash_flows, times, ytm)
    B = bond_price_from_ytm(cash_flows, times, ytm)

    DD = D_star * B

    return DD


def convexity(cash_flows: np.ndarray, times: np.ndarray, ytm: float) -> float:
    """
    Compute convexity: the second derivative of bond price w.r.t. yield, divided by price.
    Hull eq (4.12): C = (1/B) * sum[ c_i * t_i^2 * e^{-y*t_i} ]

    Parameters
    ----------
    cash_flows : np.ndarray
    times : np.ndarray
    ytm : float
        Continuously compounded yield.

    Returns
    -------
    float
        Convexity (in years^2).
    """
    B = bond_price_from_ytm(cash_flows, times, ytm)

    C = (1/B) * np.sum((times ** 2) * cash_flows * np.exp(-ytm * times))
    
    return C


def price_change_approximation(duration: float, convexity: float, price: float, delta_y: float) -> float:
    """
    Approximate bond price change using the duration-convexity Taylor expansion.
    Hull eq (4.11): ΔB ≈ -D* × B × Δy + (1/2) × C × B × (Δy)^2

    Parameters
    ----------
    duration : float
        Modified duration D*.
    convexity : float
        Convexity C.
    price : float
        Current bond price B.
    delta_y : float
        Change in yield Δy.

    Returns
    -------
    float
        Approximate dollar change in price ΔB.
    """
    delta_B = price * (-duration * delta_y + (1.0/2) * convexity * (delta_y ** 2))

    return delta_B


def bond_price_from_ytm(cash_flows: np.ndarray, times: np.ndarray, ytm: float) -> float:
    """
    Price a bond by discounting cash flows at a continuously compounded ytm.
    B = sum[ c_i * e^{-y * t_i} ]

    Parameters
    ----------
    cash_flows : np.ndarray
    times : np.ndarray
    ytm : float

    Returns
    -------
    float
        Bond price.
    """
    B = np.sum(cash_flows * np.exp(-ytm * times))

    return B


def duration_matching_hedge(portfolio_duration: float, portfolio_value: float,
                             futures_duration: float, futures_price: float,
                             face: float = 1000.0) -> float:
    """
    Compute the number of futures contracts needed to duration-hedge a bond portfolio.
    Hull eq (4.16): N* = (D_T × P) / (D_F × F)
    Here we're zeroing out duration, so target duration D_T = 0 → short N* contracts.

    Parameters
    ----------
    portfolio_duration : float
        Modified duration of the bond portfolio.
    portfolio_value : float
        Current value of the bond portfolio.
    futures_duration : float
        Duration of the cheapest-to-deliver bond underlying the futures.
    futures_price : float
        Current futures price × face value (i.e., dollar value of one contract).
    face : float

    Returns
    -------
    float
        Number of futures contracts to short (positive = short).
    """
    N_star = (portfolio_duration * portfolio_value) / (futures_duration * futures_price)

    return N_star


# ── Shared bond setup ──────────────────────────────────────────────────
# 3-year bond, 10% annual coupon, face = 1000, ytm = 12% (continuously compounded)
# Hull Example 4.6 (adapted)
face = 1000.0
coupon_rate = 0.10
ytm_cc = 0.12  # continuously compounded

times = np.array([1.0, 2.0, 3.0])
cash_flows = np.array([coupon_rate * face,
                        coupon_rate * face,
                        coupon_rate * face + face])
# cash_flows = [100, 100, 1100]

# ── Test 1: bond_price_from_ytm ────────────────────────────────────────
price = bond_price_from_ytm(cash_flows, times, ytm_cc)
print(f"Bond price: {price}")
# Expected: ~946.49  (sum of discounted cash flows at y=0.12 continuous)

# ── Test 2: macaulay_duration ──────────────────────────────────────────
D_mac = macaulay_duration(cash_flows, times, ytm_cc)
print(f"Macaulay duration: {D_mac}")
# Expected: ~2.6542 years
# Manually: weighted times = sum(t_i * PV_i) / B

# ── Test 3: modified_duration ──────────────────────────────────────────
# Use semiannual compounding ytm = 12% → y = 0.12, m = 2
ytm_semi = 0.12
D_mod = modified_duration(cash_flows, times, ytm_semi, m=2)
print(f"Modified duration: {D_mod}")
# Expected: Macaulay_dur / (1 + 0.12/2) = D_mac / 1.06
# ~2.5040 (note: you'll need to recompute Macaulay with semi compounding internally,
#          OR convert ytm_semi to continuous first — decide which and be consistent)

# ── Test 4: convexity ─────────────────────────────────────────────────
C = convexity(cash_flows, times, ytm_cc)
print(f"Convexity: {C}")
# Expected: ~8.08  (units: years^2)
# sum( c_i * t_i^2 * e^{-y*t_i} ) / B

# ── Test 5: price_change_approximation ────────────────────────────────
delta_y = 0.001  # +10 bps
D_mod_cc = D_mac  # for continuously compounded, modified duration = macaulay duration
delta_B = price_change_approximation(D_mod_cc, C, price, delta_y)
print(f"Approx price change for +10bps: {delta_B}")
# Expected: ~ -2.513  (negative because price falls when yield rises)
# Cross-check: compute bond_price_from_ytm(cash_flows, times, ytm_cc + 0.001) - price

actual_delta_B = bond_price_from_ytm(cash_flows, times, ytm_cc + delta_y) - price
print(f"Actual price change:            {actual_delta_B}")
# Should be close to approx — convexity makes them not identical

# ── Test 6: dollar_duration ───────────────────────────────────────────
DD = dollar_duration(cash_flows, times, ytm_cc)
print(f"Dollar duration: {DD}")
# Expected: D_mac * price ≈ 2.6542 * 946.49 ≈ 2513.7

# ── Test 7: duration_matching_hedge ───────────────────────────────────
# Portfolio: $10M bond, duration 7.5 years
# Futures: duration 5.2 years, price = $93,000 per contract
N_star = duration_matching_hedge(
    portfolio_duration=7.5,
    portfolio_value=10_000_000,
    futures_duration=5.2,
    futures_price=93_000
)
print(f"Futures contracts to short: {N_star}")
# Expected: (7.5 * 10_000_000) / (5.2 * 93_000) ≈ 155.35