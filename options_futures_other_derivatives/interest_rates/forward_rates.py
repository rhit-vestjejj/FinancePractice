import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# forward_rates.py  —  Hull Ch. 4
# ---------------------------------------------------------------------------


def bootstrap_zero_rates(
    maturities: np.ndarray,
    par_yields: np.ndarray,
    freq: int = 2
) -> np.ndarray:
    """
    Bootstrap continuously compounded zero rates from par yields using
    Hull's iterative bond-pricing approach (Ch. 4).

    Parameters
    ----------
    maturities : np.ndarray
        Increasing array of maturities in years, e.g. [0.5, 1.0, 1.5, 2.0].
    par_yields : np.ndarray
        Corresponding par yields (as decimals) for each maturity.
    freq : int
        Coupon payment frequency per year (default 2 = semi-annual).

    Returns
    -------
    np.ndarray
        Continuously compounded zero rates for each maturity.
    """

    rates = []
    for i in range(len(maturities)):
        cn = par_yields[i] / freq * 100
        pv_coupons = 0
        coupon_dates = np.arange(1/freq, maturities[i] + 1e-9, 1/freq)
        for j in range(i):
            pv_coupons += np.exp(-rates[j] * coupon_dates[j])
        rates.append((-1 / maturities[i]) * np.log((100 - (cn * pv_coupons)) / (100 + cn)))
    return rates


def forward_rate(
    R1: float,
    R2: float,
    T1: float,
    T2: float
) -> float:
    """
    Compute the continuously compounded forward rate between T1 and T2
    using Hull eq. 4.5.

    Parameters
    ----------
    R1 : float  — zero rate for maturity T1 (continuously compounded)
    R2 : float  — zero rate for maturity T2 (continuously compounded)
    T1 : float  — near maturity in years
    T2 : float  — far maturity in years  (T2 > T1)

    Returns
    -------
    float : forward rate R_F
    """
    Rf = (R2 * T2 - R1 * T1) / (T2 - T1)

    return Rf


def forward_rate_schedule(
    maturities: np.ndarray,
    zero_rates: np.ndarray
) -> pd.DataFrame:
    """
    Build a table of all consecutive forward rates from a zero curve —
    i.e. the forward rate for every adjacent pair (T_i, T_{i+1}).

    Returns
    -------
    pd.DataFrame with columns: T1, T2, R1, R2, forward_rate
    """
    rows = []
    for i in range (len(maturities) - 1):
        fr = forward_rate(zero_rates[i], zero_rates[i + 1], maturities[i], maturities[i+1])
        rows.append({
            "T1": maturities[i],
            "T2": maturities[i + 1],
            "R1": zero_rates[i],
            "R2": zero_rates[i + 1],
            "forward_rate": fr
        })
    df = pd.DataFrame(rows)

    return df


def interpolate_zero_curve(
    maturities: np.ndarray,
    zero_rates: np.ndarray,
    query_maturities: np.ndarray
) -> np.ndarray:
    """
    Cubic-spline interpolate zero rates at arbitrary maturities so you
    can compute forward rates between non-grid points.

    Returns
    -------
    np.ndarray : interpolated zero rates at query_maturities
    """
    cs = CubicSpline(maturities, zero_rates)
    new_points = cs(query_maturities)
    return new_points


def instantaneous_forward_rate(
    maturities: np.ndarray,
    zero_rates: np.ndarray,
    T: float
) -> float:
    """
    Approximate the instantaneous forward rate at T using the derivative
    of the zero curve: f(T) = R(T) + T * dR/dT  (Hull footnote, Ch. 4).

    Returns
    -------
    float : instantaneous forward rate at T
    """
    cs = CubicSpline(maturities, zero_rates)
    R_T = cs(T)          # R(T)
    dR_dT = cs(T, 1)     # dR/dT at T

    f_T = R_T + T * dR_dT
    
    return f_T


def forward_rate_from_treasury(
    T1: float,
    T2: float,
    ticker_short: str = "^IRX",
    ticker_long: str = "^FVX"
) -> dict:
    """
    Pull live Treasury yields via yfinance, convert to continuously
    compounded zero rates, and compute the forward rate between T1 and T2.

    Uses ^IRX (13-week ~0.25yr) and ^FVX (5-year) as defaults.

    Returns
    -------
    dict with keys: R1, R2, T1, T2, forward_rate, source_tickers
    """
    tickr_short = yf.Ticker(ticker_short)
    R1 = np.log(1 + tickr_short.fast_info['lastPrice'] / 100)

    tickr_long = yf.Ticker(ticker_long)
    R2 = np.log(1 + tickr_long.fast_info['lastPrice'] / 100)

    fr = forward_rate(R1, R2, T1, T2)

    rates = {
        'R1': R1,
        'R2': R2,
        'T1': T1,
        'T2': T2,
        'forward_rate': fr,
        'source_tickers': (ticker_short, ticker_long)
    }

    return rates


def forward_rate_agreement_value(
    R_K: float,
    R_F: float,
    T1: float,
    T2: float,
    L: float,
    R: float
) -> float:
    """
    Value an FRA where R_K is the agreed fixed rate, R_F is the current
    forward rate, L is the principal, and R is the T2 zero rate —
    Hull eq. 4.9.

    A positive value means the FRA favors the party receiving R_K.

    Returns
    -------
    float : present value of the FRA
    """

    V_fra = L * (R_K - R_F) * (T2 - T1) * np.exp(-R * T2)

    return V_fra

# -----------------------------------------------------------------------
# T1: forward_rate — Hull Table 4.5 replication
# Zero rates (cont. comp.): 3% @ 1yr, 4% @ 2yr
# Expected forward rate: (0.04*2 - 0.03*1) / (2-1) = 0.05  → 5.0%
# -----------------------------------------------------------------------
rf = forward_rate(R1=0.03, R2=0.04, T1=1.0, T2=2.0)
print(f"T1 forward_rate: {rf}  (expected 0.0500)")

# -----------------------------------------------------------------------
# T2: forward_rate — negative slope case
# Zero rates: 5% @ 1yr, 4.5% @ 2yr (inverted curve)
# Expected: (0.045*2 - 0.05*1) / 1 = 0.04  → 4.0%
# -----------------------------------------------------------------------
rf2 = forward_rate(R1=0.05, R2=0.045, T1=1.0, T2=2.0)
print(f"T2 inverted curve: {rf2}  (expected 0.0400)")

# -----------------------------------------------------------------------
# T3: forward_rate_agreement_value
# Hull Example 4.3: R_K=4%, R_F=5%, T1=1, T2=1.5, L=1e6, R(T2)=4.5%
# V = (0.04-0.05)*0.5*1e6 * exp(-0.045*1.5)
#   = -5000 * exp(-0.0675) ≈ -4,643.56
# -----------------------------------------------------------------------
fra_val = forward_rate_agreement_value(
    R_K=0.04, R_F=0.05, T1=1.0, T2=1.5, L=1_000_000, R=0.045
)
print(f"T3 FRA value: {fra_val}  (expected ≈ -4643.56)")

# -----------------------------------------------------------------------
# T4: bootstrap_zero_rates — Hull Table 4.3 style
# Par yields (semi-annual): 6m→5%, 1yr→5.8%, 1.5yr→6.4%, 2yr→6.8%
# Known bootstrapped zeros (cont. comp.) ≈: 0.0494, 0.0575, 0.0629, 0.0666
# -----------------------------------------------------------------------
mats = np.array([0.5, 1.0, 1.5, 2.0])
pars = np.array([0.05, 0.058, 0.064, 0.068])
zeros = bootstrap_zero_rates(mats, pars, freq=2)
print("T4 bootstrapped zeros:", zeros)
print("   expected approx:    [0.0494, 0.0575, 0.0629, 0.0666]")

# -----------------------------------------------------------------------
# T5: forward_rate_schedule — monotone upward curve
# Zeros: [3%, 3.5%, 4%, 4.5%] at [1, 2, 3, 4] years
# Forward 1→2: (0.035*2 - 0.03*1)/1 = 0.04
# Forward 2→3: (0.04*3 - 0.035*2)/1 = 0.05
# Forward 3→4: (0.045*4 - 0.04*3)/1 = 0.06
# -----------------------------------------------------------------------
sched = forward_rate_schedule(
    maturities=np.array([1.0, 2.0, 3.0, 4.0]),
    zero_rates=np.array([0.03, 0.035, 0.04, 0.045])
)
print("\nT5 forward_rate_schedule:")
print(sched)
print("   expected forward rates: [0.04, 0.05, 0.06]")

# -----------------------------------------------------------------------
# T6: forward_rate_from_treasury — live data smoke test
# Just verify the dict keys exist and rate is a sensible float (0–20%)
# -----------------------------------------------------------------------

result = forward_rate_from_treasury(T1=0.25, T2=5.0)
print(f"\nT6 live Treasury forward rate: {result['forward_rate']}")
print(f"   R1={result['R1']}, R2={result['R2']}")
assert 0.0 < result['forward_rate'] < 0.20, "Rate out of plausible range"
print("   Sanity check passed.")
