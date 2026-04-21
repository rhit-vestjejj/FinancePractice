import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import griddata

from .implied_vol import implied_volatility, fetch_option_chain


# ── 1. Volatility smile for a single expiry ────────────────────────────────

def vol_smile(ticker, expiry, r, option_type='call', moneyness_range=(0.8, 1.2)):
    """
    Compute the implied volatility smile for a single expiry by pulling
    the live option chain, filtering strikes to the given moneyness range
    (K/S), and inverting BS for each strike.

    Parameters
    ----------
    ticker          : str   - e.g. 'AAPL'
    expiry          : str   - 'YYYY-MM-DD'
    r               : float - risk-free rate
    option_type     : str   - 'call' or 'put'
    moneyness_range : tuple - (low, high) bounds on K/S to keep

    Returns
    -------
    pd.DataFrame with columns ['strike', 'moneyness', 'market_price', 'iv']
        sorted by strike ascending; rows with failed IV (nan) dropped
    """

    if option_type == 'call':
        chain, _, S, expiry_str = fetch_option_chain(ticker, expiry)
    else:
        _, chain, S, expiry_str = fetch_option_chain(ticker, expiry)

    T = (pd.to_datetime(expiry_str) - pd.Timestamp.today()).days / 365.0
    
    chain = chain.copy()
    chain['mid'] = np.where(
        (chain['bid'] > 0) & (chain['ask'] > 0),
        (chain['bid'] + chain['ask']) / 2.0,
        chain['lastPrice']
    )
    chain = chain[chain['mid'] > 0].copy()
        
    chain['moneyness'] = chain['strike'] / S
    chain = chain[(chain['moneyness'] >= moneyness_range[0]) &
                  (chain['moneyness'] <= moneyness_range[1])].copy()

    if option_type == 'call':
        intrinsic = np.maximum(S - chain['strike'], 0)
    else:
        intrinsic = np.maximum(chain['strike'] - S, 0)
    chain = chain[chain['mid'] > intrinsic].copy()

    ivs = []
    for _, row in chain.iterrows():
        iv = implied_volatility(row['mid'], S, row['strike'], T, r, option_type)
        ivs.append(iv)


    chain['iv'] = ivs
    chain = chain.dropna(subset=['iv'])
    chain = chain[chain['iv'] > 0]

    result = chain[['strike', 'moneyness', 'mid', 'iv']].copy()
    result.columns = ['strike', 'moneyness', 'market_price', 'iv']
    return result.sort_values('strike').reset_index(drop=True)



# ── 2. Full volatility surface across multiple expiries ────────────────────

def vol_surface(ticker, r, option_type='call', moneyness_range=(0.8, 1.2),
                max_expiries=6):
    """
    Build the volatility surface by calling vol_smile for each of the
    first max_expiries available expiry dates on the ticker.

    Parameters
    ----------
    ticker          : str   - e.g. 'AAPL'
    r               : float - risk-free rate
    option_type     : str   - 'call' or 'put'
    moneyness_range : tuple - moneyness filter passed to vol_smile
    max_expiries    : int   - cap on how many expiries to include

    Returns
    -------
    pd.DataFrame with columns ['strike', 'moneyness', 'T', 'market_price', 'iv']
        T is time to expiry in years; nan rows dropped
    """
    expiries = yf.Ticker(ticker).options
    lists = []
    for expiry in expiries[:max_expiries]:
        smile = vol_smile(ticker, expiry, r, option_type, moneyness_range)
        # compute T, add it as a column, append to a list
        T = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365
        if T <= 0:
            continue
        smile['T'] = T
        lists.append(smile)
    # concat everything
    iv = pd.concat(lists)
    
    return iv.dropna()


# ── 3. Interpolate IV from the surface ─────────────────────────────────────

def interpolate_iv(surface_df, K, T, S, method='cubic'):
    """
    Given a vol surface DataFrame (from vol_surface), interpolate to
    estimate implied volatility at an arbitrary (K, T) point using
    scipy.interpolate.griddata on the (moneyness, T) grid.

    Parameters
    ----------
    surface_df : pd.DataFrame - output of vol_surface()
    K          : float        - target strike
    T          : float        - target time to expiry (years)
    S          : float        - current spot (needed to compute moneyness = K/S)
    method     : str          - 'linear', 'cubic', or 'nearest'

    Returns
    -------
    float - interpolated implied volatility, or np.nan if outside the grid
    """
    points = surface_df[['moneyness', 'T']].values

    values = surface_df['iv'].values

    query_point = (K / S, T)

    interp_vol = griddata(points, values, query_point, method=method)

    return float(interp_vol)


TICKER = 'AAPL'
R = 0.045

# ── Test 1: vol_smile ──────────────────────────────────────────────
# Pull the smile for the nearest expiry >7 days out.
# Expected: DataFrame with columns [strike, moneyness, market_price, iv]
#   - moneyness values all between 0.8 and 1.2
#   - iv values all positive and roughly in [0.05, 1.5]
#   - at least 5 rows after filtering
#   - smile shape: OTM puts / deep ITM calls should have higher IV
#     than ATM (moneyness ~ 1.0) — i.e. not perfectly flat

tk = yf.Ticker(TICKER)
expiries = tk.options
expiry = next(e for e in expiries if (pd.to_datetime(e) - pd.Timestamp.today()).days > 7)

smile = vol_smile(TICKER, expiry, R, option_type='call')
print("=== vol_smile ===")
print(smile.to_string(index=False))
print(f"\nRows: {len(smile)}  (expected >= 5)")
print(f"Moneyness range: [{smile['moneyness'].min():.3f}, {smile['moneyness'].max():.3f}]  (expected within [0.8, 1.2])")
print(f"IV range: [{smile['iv'].min():.4f}, {smile['iv'].max():.4f}]  (expected all > 0)")
atm = smile.iloc[(smile['moneyness'] - 1.0).abs().argsort()[:1]]['iv'].values[0]
wing = smile.iloc[(smile['moneyness'] - 0.85).abs().argsort()[:1]]['iv'].values[0]
print(f"ATM IV: {atm:.4f}, Wing IV (m~0.85): {wing:.4f}  (expected wing >= ATM for equity skew)")

# ── Test 2: vol_surface ────────────────────────────────────────────
# Expected: DataFrame with columns [strike, moneyness, T, market_price, iv]
#   - multiple distinct T values (at least 2)
#   - T values all > 0
#   - iv values all positive
#   - at least 20 total rows

surf = vol_surface(TICKER, R, option_type='call', max_expiries=4)
print("\n=== vol_surface ===")
print(f"Shape: {surf.shape}  (expected rows >= 20)")
print(f"Distinct expiries: {surf['T'].nunique()}  (expected >= 2)")
print(f"T range: [{surf['T'].min():.4f}, {surf['T'].max():.4f}]  (expected all > 0)")
print(f"IV range: [{surf['iv'].min():.4f}, {surf['iv'].max():.4f}]  (expected all > 0)")
print(surf.head(10).to_string(index=False))

# ── Test 3: interpolate_iv ─────────────────────────────────────────
# Pick an (K, T) that sits inside the grid.
# Expected: a finite positive float, roughly in the same ballpark as
#   surrounding grid points.

S = yf.Ticker(TICKER).fast_info['last_price']
mid_T = surf['T'].median()
mid_K = S * 1.0  # ATM

iv_interp = interpolate_iv(surf, K=mid_K, T=mid_T, S=S, method='cubic')
print(f"\n=== interpolate_iv ===")
print(f"Interpolated IV at K={mid_K:.2f}, T={mid_T:.4f}: {iv_interp:.4f}")
print(f"  (expected: finite, positive, roughly near ATM vols above)")

# Sanity: linear vs cubic should be close for interior points
iv_lin = interpolate_iv(surf, K=mid_K, T=mid_T, S=S, method='linear')
print(f"Linear IV: {iv_lin:.4f}, Cubic IV: {iv_interp:.4f}")
print(f"  Difference: {abs(iv_lin - iv_interp):.6f}  (expected < 0.05)")