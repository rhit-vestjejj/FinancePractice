import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import brentq


# ─────────────────────────────────────────────
# interest_rate_swap.py
# Hull Ch. 7 — Swaps
# ─────────────────────────────────────────────


def bootstrap_swap_curve(maturities: np.ndarray, swap_rates: np.ndarray) -> np.ndarray:
    """
    Bootstrap a zero-rate curve from par swap rates using continuous compounding.
    
    Parameters
    ----------
    maturities : np.ndarray
        Maturities in years, e.g. [0.5, 1.0, 1.5, 2.0].
    swap_rates : np.ndarray
        Observed par swap rates (annualized, continuous), same length as maturities.
    
    Returns
    -------
    np.ndarray
        Zero rates (continuously compounded) for each maturity.
    
    Notes
    -----
    Hull Section 7.6 — par swap rates imply zero rates via bootstrapping.
    A par swap rate is the fixed rate that makes the swap's initial value zero.
    For a 1-year semi-annual swap, use the 6-month zero rate you already
    bootstrapped to solve for the 1-year zero rate.
    """
    boot = []

    for n in range (len(maturities)):
        D = np.sum(np.exp(-np.array(boot[:n]) * maturities[:n]))
        if n == 0:
            delta_t = maturities[n]
        else:
            delta_t = maturities[n] - maturities[n-1]
        S = swap_rates[n]
        t_N = maturities[n]

        z_N = -np.log((1 - S * delta_t * D) / (1 + S * delta_t)) / t_N

        boot.append(z_N)
    
    return boot
    


def fixed_leg_pv(
    notional: float,
    fixed_rate: float,
    payment_times: np.ndarray,
    zero_rates: np.ndarray
) -> float:
    """
    Price the fixed leg of an interest rate swap as a coupon bond.
    
    Parameters
    ----------
    notional : float
        Principal (Q in Hull notation).
    fixed_rate : float
        Annualized fixed coupon rate (k).
    payment_times : np.ndarray
        Coupon payment times in years (t_i).
    zero_rates : np.ndarray
        Continuously compounded zero rates for each t_i.
    
    Returns
    -------
    float
        Present value of the fixed leg (B_fix).
    
    Notes
    -----
    Hull Eq 7.1 — fixed leg is simply a fixed-coupon bond.
    Don't forget the notional repayment at the final payment date.
    """
    x = 0
    for n in range(len(payment_times)):
        if n == 0:
            delta_t = payment_times[n]
        else:
            delta_t = payment_times[n] - payment_times[n - 1]
        x += fixed_rate * notional * delta_t * np.exp(-zero_rates[n] * payment_times[n])

    B_fixed = x + notional * np.exp(-zero_rates[-1] * payment_times[-1])
    
    return B_fixed


def floating_leg_pv(
    notional: float,
    payment_times: np.ndarray,
    zero_rates: np.ndarray,
    last_reset_rate: float,
    time_to_next_payment: float
) -> float:
    """
    Price the floating leg of an interest rate swap as a floating-rate bond.
    
    Parameters
    ----------
    notional : float
        Principal (Q).
    payment_times : np.ndarray
        Remaining payment times in years.
    zero_rates : np.ndarray
        Continuously compounded zero rates for each payment_times.
    last_reset_rate : float
        LIBOR/SOFR rate set at the last reset date (r*).
    time_to_next_payment : float
        Time in years until the next floating payment (t*).
    
    Returns
    -------
    float
        Present value of the floating leg (B_fl).
    
    Notes
    -----
    Hull Section 7.7 — a floating-rate bond is worth par at each reset date.
    Immediately after a reset, B_fl = Q * exp(-r(t*) * t*) * (1 + r* * t*) ... 
    actually think about the simpler argument Hull uses here.
    """

    B_fl = notional * (1 + last_reset_rate * payment_times[0]) * np.exp(-zero_rates[0] * time_to_next_payment)

    return B_fl


def swap_value_bond_approach(
    notional: float,
    fixed_rate: float,
    payment_times: np.ndarray,
    zero_rates: np.ndarray,
    last_reset_rate: float,
    time_to_next_payment: float,
    position: str = 'pay_fixed'
) -> float:
    """
    Value an interest rate swap using the bond decomposition (Hull Eq. 7.4/7.5).
    
    Parameters
    ----------
    notional : float
    fixed_rate : float
    payment_times : np.ndarray
    zero_rates : np.ndarray
    last_reset_rate : float
    time_to_next_payment : float
    position : str
        'pay_fixed' (long float, short fixed) or 'receive_fixed'.
    
    Returns
    -------
    float
        V_swap from the perspective of the given position.
    
    Notes
    -----
    Hull Eq. 7.4: V_swap = B_fl - B_fix  (for pay-fixed party).
    Hull Eq. 7.5: V_swap = B_fix - B_fl  (for receive-fixed party).
    """
    B_fl = floating_leg_pv(notional, payment_times, zero_rates, last_reset_rate, time_to_next_payment)

    B_fix = fixed_leg_pv(notional, fixed_rate, payment_times, zero_rates)

    if position == 'pay_fixed':
        return B_fl - B_fix
    elif position == 'receive_fixed':
        return B_fix - B_fl
    


def forward_rate(
    t1: float,
    t2: float,
    zero_rate_t1: float,
    zero_rate_t2: float
) -> float:
    """
    Compute the continuously compounded forward rate between t1 and t2.
    
    Parameters
    ----------
    t1, t2 : float
        Start and end of forward period in years (t1 < t2).
    zero_rate_t1, zero_rate_t2 : float
        Continuously compounded zero rates at t1 and t2.
    
    Returns
    -------
    float
        Forward rate r_f (continuously compounded).
    
    Notes
    -----
    Hull Eq. 4.5: r_f = (r2*T2 - r1*T1) / (T2 - T1).
    """
    r_f = (zero_rate_t2 * t2 - zero_rate_t1 * t1) / (t2 - t1)

    return r_f


def fra_value(
    notional: float,
    forward_rate_k: float,
    market_forward_rate: float,
    t1: float,
    t2: float,
    zero_rate_t2: float
) -> float:
    """
    Value a single forward rate agreement (FRA).
    
    Parameters
    ----------
    notional : float
    forward_rate_k : float
        The fixed rate locked in by the FRA (R_K).
    market_forward_rate : float
        Current market forward rate for [t1, t2] (R_F).
    t1, t2 : float
        FRA period start and end in years.
    zero_rate_t2 : float
        Zero rate to discount back from t2.
    
    Returns
    -------
    float
        Present value of the FRA (from perspective of receiving floating).
    
    Notes
    -----
    Hull Eq. 7.8: V_FRA = L * (R_F - R_K) * (T2-T1) * exp(-R2*T2).
    This is for a receive-floating FRA; negate for pay-floating.
    """
    V_FRA = notional * (market_forward_rate - forward_rate_k) * (t2 - t1) * np.exp(-zero_rate_t2 * t2)

    return V_FRA


def swap_value_fra_approach(
    notional: float,
    fixed_rate: float,
    payment_times: np.ndarray,
    zero_rates: np.ndarray,
    position: str = 'pay_fixed'
) -> float:
    """
    Value an IRS as a portfolio of FRAs (Hull Section 7.8).
    
    Parameters
    ----------
    notional : float
    fixed_rate : float
        Fixed rate on the swap (R_K for each embedded FRA).
    payment_times : np.ndarray
        Payment dates [t1, t2, ..., tN] with t0=0 implied.
    zero_rates : np.ndarray
        Zero rates for each payment date.
    position : str
        'pay_fixed' or 'receive_fixed'.
    
    Returns
    -------
    float
        V_swap as sum of FRA values.
    
    Notes
    -----
    Hull Section 7.8 — each floating payment is a FRA where the fixed rate
    is the swap's fixed rate R_K. The FRAs don't all have the same value
    (unlike the initial par swap where they collectively sum to zero).
    """
    V_swap = 0
    for n in range(len(payment_times)):
        if n == 0:
            delta_t = payment_times[n]
            r_f = zero_rates[0]
        else:
            delta_t = payment_times[n] - payment_times[n - 1]
            r_f = forward_rate(payment_times[n-1], payment_times[n], zero_rates[n-1], zero_rates[n])

        V_swap += notional * (r_f - fixed_rate) * delta_t * np.exp(-zero_rates[n] * payment_times[n])
        
    if position == 'receive_fixed':
        V_swap = -V_swap
    return V_swap


def par_swap_rate(
    payment_times: np.ndarray,
    zero_rates: np.ndarray
) -> float:
    """
    Compute the par (fair) fixed rate that makes a new swap worth zero.
    
    Parameters
    ----------
    payment_times : np.ndarray
    zero_rates : np.ndarray
    
    Returns
    -------
    float
        Par swap rate s (continuously compounded, annualized).
    
    Notes
    -----
    Hull Eq. 7.3 — solve B_fix = B_fl = notional for the fixed rate s.
    Since B_fl = Q at inception (reset date), you need:
    s * sum(exp(-r_i * t_i) * dt_i) + exp(-r_N * t_N) = 1.
    Solve analytically — no numerical solver needed here.
    """
    x = np.exp(-zero_rates[-1] * payment_times[-1])
    y = 0
    for n in range(len(payment_times)):
        if n == 0:
            dt = payment_times[n]
        else:
            dt = payment_times[n] - payment_times[n - 1]
        y += dt * np.exp(-zero_rates[n] * payment_times[n])
    s = (1 - x) / y

    return s


def swap_dv01(
    notional: float,
    fixed_rate: float,
    payment_times: np.ndarray,
    zero_rates: np.ndarray,
    last_reset_rate: float,
    time_to_next_payment: float,
    position: str = 'pay_fixed',
    bump: float = 0.0001
) -> float:
    """
    Compute the DV01 of the swap via parallel yield curve bump.
    
    Parameters
    ----------
    bump : float
        Yield curve shift in rate units (default 1bp = 0.0001).
    
    Returns
    -------
    float
        DV01: change in swap value for +1bp parallel shift.
    
    Notes
    -----
    Bump all zero rates up by `bump`, revalue swap, take the difference.
    Sign convention: DV01 is positive if the swap gains value when rates rise.
    """
    V_swaps = swap_value_bond_approach(notional, fixed_rate, payment_times, zero_rates, last_reset_rate, time_to_next_payment, position)
    bumped_rates = zero_rates + bump
    V_swaps_bump = swap_value_bond_approach(notional, fixed_rate, payment_times, bumped_rates, last_reset_rate, time_to_next_payment, position) 

    delta = V_swaps_bump - V_swaps
    
    return delta


print("=" * 60)
print("TEST 1: bootstrap_swap_curve")
print("=" * 60)
# Hull Example 7.2 (approximate) — use 6-month swap rates to bootstrap
# Semi-annual payments, continuously compounded zero rates
maturities = np.array([0.5, 1.0, 1.5, 2.0])
swap_rates  = np.array([0.040, 0.042, 0.044, 0.045])
zeros = bootstrap_swap_curve(maturities, swap_rates)
print("Bootstrapped zero rates:", np.round(zeros, 6))
# Expected: zero rates should be close to but not identical to swap rates
# z[0] ≈ 0.040 (first swap rate = first zero rate for a single-period swap)
# z[1] slightly above 0.042, z[2] slightly above 0.044, z[3] slightly above 0.045
# All rates should be monotonically increasing (upward sloping curve)

print("\n" + "=" * 60)
print("TEST 2: fixed_leg_pv and floating_leg_pv")
print("=" * 60)
# Hull Example 7.1: 2-year swap, semi-annual, notional=100M
# Fixed rate = 6%, LIBOR flat zero curve at 5% (continuous)
notional = 100e6
fixed_rate = 0.06
payment_times = np.array([0.5, 1.0, 1.5, 2.0])
zero_rates_flat = np.full(4, 0.05)  # flat 5% zero curve
last_reset = 0.05  # last LIBOR reset was at 5%
t_next = 0.5       # next payment in 6 months

b_fix = fixed_leg_pv(notional, fixed_rate, payment_times, zero_rates_flat)
b_fl  = floating_leg_pv(notional, payment_times, zero_rates_flat, last_reset, t_next)
print(f"B_fix = {b_fix}")
print(f"B_fl  = {b_fl}")
# Expected: B_fix ≈ 98,234,950  (below par because fixed rate > zero rate... 
#   wait — fixed 6% > 5% zero curve, so bond trades above par at ~101.9M? 
#   work it out yourself)
# Expected: B_fl  ≈ 100,000,000 (floating bond ≈ par when reset was at market rate)
# Hint: think carefully about when a fixed bond trades above vs below par

print("\n" + "=" * 60)
print("TEST 3: swap_value_bond_approach")
print("=" * 60)
v_pay_fixed = swap_value_bond_approach(
    notional, fixed_rate, payment_times, zero_rates_flat,
    last_reset, t_next, position='pay_fixed'
)
v_receive_fixed = swap_value_bond_approach(
    notional, fixed_rate, payment_times, zero_rates_flat,
    last_reset, t_next, position='receive_fixed'
)
print(f"V_swap (pay fixed):     {v_pay_fixed}")
print(f"V_swap (receive fixed): {v_receive_fixed}")
# Expected: V_pay_fixed + V_receive_fixed = 0 (zero-sum)
# V_pay_fixed should be negative here (you're paying 6% in a 5% market — bad deal)

print("\n" + "=" * 60)
print("TEST 4: swap_value_fra_approach vs bond approach")
print("=" * 60)
v_fra = swap_value_fra_approach(
    notional, fixed_rate, payment_times, zero_rates_flat,
    position='pay_fixed'
)
print(f"V_swap via FRA approach: {v_fra}")
print(f"V_swap via bond approach: {v_pay_fixed}")
# Expected: Both methods must give the SAME answer (within floating-point error)
# |v_fra - v_pay_fixed| < 1e-4 * notional

print("\n" + "=" * 60)
print("TEST 5: par_swap_rate")
print("=" * 60)
s = par_swap_rate(payment_times, zero_rates_flat)
print(f"Par swap rate: {s}")
# Expected: s ≈ 0.05 (when zero curve is flat at 5%, par swap rate ≈ 5%)
# Verify: swap_value_bond_approach with fixed_rate=s should return ≈ 0

print("\n" + "=" * 60)
print("TEST 6: swap_dv01")
print("=" * 60)
dv01 = swap_dv01(notional, fixed_rate, payment_times, zero_rates_flat, last_reset, t_next)
print(f"DV01 (pay fixed, 1bp): {dv01}")
# Expected: DV01 should be NEGATIVE for pay-fixed position (~-17,000 to -19,000)
# Intuition: if rates rise 1bp, floating payments go up → pay-fixed party GAINS
# So DV01 should be POSITIVE for pay-fixed... think about which leg dominates

print("\n" + "=" * 60)
print("TEST 7: bootstrapped curve → par swap rate round-trip")
print("=" * 60)
zeros_from_bootstrap = bootstrap_swap_curve(maturities, swap_rates)
s_roundtrip = par_swap_rate(payment_times, zeros_from_bootstrap)
print(f"Original 2Y swap rate:    {swap_rates[-1]}")
print(f"Recovered par swap rate:  {s_roundtrip}")
# Expected: |s_roundtrip - swap_rates[-1]| < 1e-8
# This is the key consistency check: bootstrap → par swap rate should round-trip exactly