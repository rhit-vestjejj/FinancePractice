import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import brentq
from scipy.stats import norm


# ─────────────────────────────────────────────
# 1. verify_put_call_parity
# ─────────────────────────────────────────────
def verify_put_call_parity(
    S0: float,
    K: float,
    r: float,
    T: float,
    c: float,
    p: float
) -> dict:
    """
    Checks whether the put-call parity relationship holds for a given
    set of European option prices and returns the parity violation (arbitrage spread).

    Hull Eq: c + K*exp(-r*T) = p + S0

    Returns dict with:
        lhs       : c + K*exp(-r*T)
        rhs       : p + S0
        violation : lhs - rhs
        holds     : bool (True if |violation| < 0.01)
    """
    parity = {
        'lhs': c + K * np.exp(-r * T),
        'rhs': p + S0,
        'violation': (c + K * np.exp(-r * T)) - (p + S0),
        'holds': abs((c + K * np.exp(-r * T)) - (p + S0)) < 0.01
    }
    
    return parity


# ─────────────────────────────────────────────
# 2. synthetic_forward_from_options
# ─────────────────────────────────────────────
def synthetic_forward_from_options(
    S0: float,
    K: float,
    r: float,
    T: float,
    c: float,
    p: float
) -> float:
    """
    Computes the synthetic forward price implied by put-call parity:
        F0 = (c - p)*exp(r*T) + K

    Returns the synthetic forward price F0.
    """
    F0 = (c - p) * np.exp(r * T) + K
    
    return F0


# ─────────────────────────────────────────────
# 3. implied_forward_rate
# ─────────────────────────────────────────────
def implied_forward_rate(
    S0: float,
    K: float,
    T: float,
    c: float,
    p: float
) -> float:
    """
    Backs out the continuously compounded risk-free rate r implied by
    put-call parity, given observed market prices for a call and put
    with the same strike and expiry.

    Solve: c - p = S0 - K*exp(-r*T)  =>  r = -ln((S0 - c + p) / K) / T
    """
    r = np.log((c - p - S0) / -K) / -T
    
    return r


# ─────────────────────────────────────────────
# 4. parity_arbitrage_strategy
# ─────────────────────────────────────────────
def parity_arbitrage_strategy(
    S0: float,
    K: float,
    r: float,
    T: float,
    c: float,
    p: float
) -> dict:
    """
    Identifies which direction of arbitrage exists (if any) when parity
    is violated, and returns the trades required and the arbitrage profit.

    Two cases:
        lhs > rhs: sell call, buy put, buy stock, borrow PV(K)
        lhs < rhs: buy call, sell put, sell stock, lend PV(K)

    Returns dict with:
        direction     : 'sell_call' | 'buy_call' | 'none'
        trades        : list of str describing each leg
        arb_profit    : float
    """

    parity = verify_put_call_parity(S0, K, r, T, c, p)
    lhs = parity['lhs']
    rhs = parity['rhs']

    if(lhs < rhs):
        direction = 'buy call'
        arb_profit = rhs - lhs
        trades = 'buy call, sell put, sell stock, lend PV(K)'

    elif(rhs < lhs):
        direction = 'sell call'
        arb_profit = lhs - rhs
        trades = 'sell call, buy put, buy stock, borrow PV(K)'

    else:
        direction = 'none'
        arb_profit = 0
        trades = 'none'

    return {
        'direction': direction,
        'arb_profit': arb_profit,
        'trades': trades
    }


# ─────────────────────────────────────────────
# 5. fetch_option_chain
# ─────────────────────────────────────────────
def fetch_option_chain(ticker: str, expiry_index: int = 0) -> pd.DataFrame:
    """
    Downloads the option chain for a given ticker from yfinance and
    returns a merged DataFrame of calls and puts at matching strikes,
    along with mid-prices for both.

    Returns DataFrame with columns:
        strike, call_bid, call_ask, call_mid, put_bid, put_ask, put_mid
    """
    ticker = yf.Ticker("SPY")
    ticker.options          # list of expiry date strings e.g. ['2024-01-19', '2024-01-26', ...]
    expiry = ticker.options[expiry_index]
    chain = ticker.option_chain(expiry)
    calls = chain.calls             # DataFrame of calls
    puts = chain.puts              # DataFrame of puts

    calls['call_mid'] = (calls['bid'] + calls['ask']) / 2
    calls['time'] = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365
    puts['put_mid'] = (puts['bid'] + puts['ask']) / 2
    

    final = pd.merge(
        calls[['strike', 'bid', 'ask', 'call_mid', 'lastPrice', 'time']].rename(columns={'bid': 'call_bid', 'ask': 'call_ask'}),
        puts[['strike', 'bid', 'ask', 'put_mid']].rename(columns={'bid': 'put_bid', 'ask': 'put_ask'}),
        on='strike'
    )
    return final


# ─────────────────────────────────────────────
# 6. scan_parity_violations
# ─────────────────────────────────────────────
def scan_parity_violations(
    ticker: str,
    r: float,
    expiry_index: int = 0
) -> pd.DataFrame:
    """
    Fetches the live option chain for a ticker, computes put-call parity
    violation for every strike, and returns a DataFrame sorted by
    |violation| descending.

    Returns DataFrame with columns:
        strike, c, p, lhs, rhs, violation, holds
    """
    option_chain = fetch_option_chain(ticker, expiry_index)
    S0 = option_chain['lastPrice']
    K = option_chain['strike']
    T = option_chain['time']
    c = option_chain['call_mid']
    p = option_chain['put_mid']
    
    parity = verify_put_call_parity(S0, K, r, T, c, p)
    
    lhs = parity['lhs']
    rhs = parity['rhs']
    violation = parity['violation']
    holds = parity['holds']

    option_chain['lhs'] = lhs
    option_chain['rhs'] = rhs
    option_chain['violation'] = violation
    option_chain['holds'] = holds

    result = option_chain[['strike', 'call_mid', 'put_mid', 'lhs', 'rhs', 'violation', 'holds']]
    result = result.rename(columns={'call_mid': 'c', 'put_mid': 'p'})
    result = result.iloc[result['violation'].abs().argsort()[::-1].values]
    return result

# --- Test 1: verify_put_call_parity ---
# Hull Example: S0=42, K=40, r=0.10, T=0.5
# c=3.00 (BS price), compute p from parity: p = c + K*exp(-r*T) - S0
S0, K, r, T = 42.0, 40.0, 0.10, 0.5
c = 3.00
p_fair = c + K * np.exp(-r * T) - S0   # ~1.099
result = verify_put_call_parity(S0, K, r, T, c, p_fair)
print("Test 1 - Parity holds (violation ~0):")
print(result)
# Expected: violation ~ 0.0, holds = True

# Test with a mispriced put
p_cheap = p_fair - 0.50   # underpriced by $0.50
result2 = verify_put_call_parity(S0, K, r, T, c, p_cheap)
print("\nTest 1b - Violation of $0.50:")
print(result2)
# Expected: violation ~ +0.50, holds = False

# --- Test 2: synthetic_forward_from_options ---
F0 = synthetic_forward_from_options(S0, K, r, T, c, p_fair)
F0_direct = S0 * np.exp(r * T)   # ~44.16
print(f"\nTest 2 - Synthetic F0: {F0:.4f}, Direct F0: {F0_direct:.4f}")
# Expected: both ~ 44.16

# --- Test 3: implied_forward_rate ---
r_implied = implied_forward_rate(S0, K, T, c, p_fair)
print(f"\nTest 3 - Implied r: {r_implied:.4f} (expected ~0.10)")

# --- Test 4: parity_arbitrage_strategy ---
# Overpriced call: lhs > rhs
strat = parity_arbitrage_strategy(S0, K, r, T, c + 0.50, p_fair)
print("\nTest 4 - Arbitrage (overpriced call):")
for k, v in strat.items():
    print(f"  {k}: {v}")
# Expected: direction = 'sell_call', arb_profit ~ 0.50

# No arb
strat2 = parity_arbitrage_strategy(S0, K, r, T, c, p_fair)
print("\nTest 4b - No arbitrage:")
print(f"  direction: {strat2['direction']}, arb_profit: {strat2['arb_profit']:.4f}")
# Expected: direction = 'none', arb_profit = 0.0

# --- Test 5: fetch_option_chain ---
print("\nTest 5 - Fetching SPY option chain...")
chain = fetch_option_chain("SPY", expiry_index=1)
print(chain.head())
# Expected: DataFrame with strike, call_mid, put_mid columns, no NaNs

# --- Test 6: scan_parity_violations ---
print("\nTest 6 - Scanning SPY parity violations...")
r_rf = 0.053   # approximate current risk-free rate
violations = scan_parity_violations("SPY", r=r_rf, expiry_index=1)
print(violations.head(10))
# Expected: sorted by |violation|, most liquid strikes near ATM should be close to 0