# hull/options/payoff_diagrams.py

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
from typing import Optional


# ── Underlying payoff ──────────────────────────────────────────────────────────

def long_underlying(S: np.ndarray, S0: float) -> np.ndarray:
    """Return the profit of holding one share bought at S0, across spot prices S."""

    profit = S - S0

    return profit


def short_underlying(S: np.ndarray, S0: float) -> np.ndarray:
    """Return the profit of shorting one share sold at S0, across spot prices S."""

    profit = S0 - S

    return profit


# ── Vanilla option payoffs ─────────────────────────────────────────────────────

def long_call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    """Return the terminal payoff of a long call with strike K, across spot prices S."""

    payoff = np.maximum(S - K, 0)

    return payoff

def short_call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    """Return the terminal payoff of a short call with strike K."""

    payoff = -np.maximum(S - K, 0)

    return payoff


def long_put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    """Return the terminal payoff of a long put with strike K."""

    payoff = np.maximum(K - S, 0)

    return payoff


def short_put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    """Return the terminal payoff of a short put with strike K."""

    payoff = -np.maximum(K - S, 0)
    
    return payoff


# ── Profit (payoff net of premium) ────────────────────────────────────────────

def long_call_profit(S: np.ndarray, K: float, c: float) -> np.ndarray:
    """Return the profit of a long call position (payoff minus premium c)."""

    profit = np.maximum(S - K, 0) - c

    return profit


def short_call_profit(S: np.ndarray, K: float, c: float) -> np.ndarray:
    """Return the profit of a short call position (premium received minus payoff)."""

    profit = -np.maximum(S - K, 0) + c

    return profit


def long_put_profit(S: np.ndarray, K: float, p: float) -> np.ndarray:
    """Return the profit of a long put position (payoff minus premium p)."""

    profit = np.maximum(K - S, 0) - p

    return profit


def short_put_profit(S: np.ndarray, K: float, p: float) -> np.ndarray:
    """Return the profit of a short put position (premium received minus payoff)."""

    profit = -np.maximum(K - S, 0) + p

    return profit


# ── Option strategies ──────────────────────────────────────────────────────────

def bull_call_spread(S: np.ndarray, K1: float, K2: float,
                     c1: float, c2: float) -> np.ndarray:
    """
    Return the profit of a bull call spread:
    long call at K1 (lower strike), short call at K2 (upper strike).
    K1 < K2 required.
    """
    profit = long_call_profit(S, K1, c1) + short_call_profit(S, K2, c2)
    
    return profit


def bear_put_spread(S: np.ndarray, K1: float, K2: float,
                    p1: float, p2: float) -> np.ndarray:
    """
    Return the profit of a bear put spread:
    long put at K2 (higher strike), short put at K1 (lower strike).
    K1 < K2 required.
    """

    profit = long_put_profit(S, K2, p2) + short_put_profit(S, K1, p1)
    
    return profit


def straddle(S: np.ndarray, K: float, c: float, p: float) -> np.ndarray:
    """Return the profit of a long straddle: long call + long put at the same strike K."""

    profit = long_call_profit(S, K, c) + long_put_profit(S, K, p)

    return profit


def strangle(S: np.ndarray, K_put: float, K_call: float,
             p: float, c: float) -> np.ndarray:
    """
    Return the profit of a long strangle:
    long put at K_put, long call at K_call, where K_put < K_call.
    """
    profit = long_put_profit(S, K_put, p) + long_call_profit(S, K_call, c)
    
    return profit


def butterfly_spread(S: np.ndarray, K1: float, K2: float, K3: float,
                     c1: float, c2: float, c3: float) -> np.ndarray:
    """
    Return the profit of a long butterfly spread using calls:
    long K1, short 2x K2, long K3, where K1 < K2 < K3 and K2 = (K1+K3)/2.
    """

    profit = long_call_profit(S, K1, c1) + 2 * short_call_profit(S, K2, c2) + long_call_profit(S, K3, c3)

    return profit


def covered_call(S: np.ndarray, S0: float, K: float, c: float) -> np.ndarray:
    """Return the profit of a covered call: long underlying (bought at S0) + short call at K."""

    profit = long_underlying(S, S0) + short_call_profit(S, K, c)

    return profit


def protective_put(S: np.ndarray, S0: float, K: float, p: float) -> np.ndarray:
    """Return the profit of a protective put: long underlying (bought at S0) + long put at K."""

    profit = long_underlying(S, S0) + long_put_profit(S, K, p)

    return profit


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_payoff(S: np.ndarray, profiles: dict, title: str,
                zero_line: bool = True) -> None:
    """
    Plot one or more payoff/profit profiles on a single axes.
    profiles: dict mapping label (str) -> np.ndarray of values.
    Draws a horizontal zero line if zero_line=True.
    """

    fig, ax = plt.subplots()
    if zero_line:
        ax.axhline(0)
    for label, values in profiles.items():
        ax.plot(S, values, label=label)
    ax.set_title(title)
    ax.legend()
    plt.show()


# ── Market data helpers ────────────────────────────────────────────────────────

def fetch_option_chain(ticker: str, expiry: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch the options chain for ticker using yfinance.
    Returns calls and puts as a combined DataFrame with columns:
    ['strike', 'lastPrice', 'bid', 'ask', 'type'].
    If expiry is None, uses the nearest available expiry.
    """
    ticker = yf.Ticker(ticker)
    ticker.options          # list of expiry date strings e.g. ['2024-01-19', '2024-01-26', ...]
    if expiry == None:
        import datetime
        today = datetime.date.today().isoformat()
        expiry = next(e for e in ticker.options if e > today)
    chain = ticker.option_chain(expiry)
    calls = chain.calls             # DataFrame of calls
    puts = chain.puts   
               # DataFrame of puts
    calls['type'] = 'call'
    puts['type'] = 'put'

    cols = ['strike', 'lastPrice', 'bid', 'ask', 'type']

    final = pd.concat([calls[cols], puts[cols]])

    return final


def atm_strike(ticker: str) -> float:
    """
    Return the at-the-money strike for ticker, defined as the
    available strike closest to the current spot price.
    """
    stock = yf.Ticker(ticker)
    spot = stock.history(period="1d")["Close"].iloc[-1]

    expiry = stock.options[0]
    chain = stock.option_chain(expiry)
    strikes = np.union1d(chain.calls["strike"].values, chain.puts["strike"].values)

    return float(strikes[np.abs(strikes - spot).argmin()])

ticker = "SPY"
S0 = atm_strike(ticker)
S = np.linspace(0.7 * S0, 1.3 * S0, 500)

# --- Test 1: vanilla payoffs ---
profiles = {
    "Long call":  long_call_payoff(S, S0),
    "Short call": short_call_payoff(S, S0),
    "Long put":   long_put_payoff(S, S0),
    "Short put":  short_put_payoff(S, S0),
}
plot_payoff(S, profiles, title="Vanilla option payoffs (K = ATM)")

# --- Test 2: straddle profit (fetch real premiums) ---
chain = fetch_option_chain(ticker)
atm_calls = chain[(chain["type"] == "call") &
                    (chain["strike"] == S0)]
atm_puts  = chain[(chain["type"] == "put") &
                    (chain["strike"] == S0)]

if not atm_calls.empty and not atm_puts.empty:
    c = atm_calls["lastPrice"].values[0]
    p = atm_puts["lastPrice"].values[0]
    profiles2 = {
        "Straddle profit": straddle(S, S0, c, p),
    }
    print(f"ATM strike: {S0}")
    print(f"Call premium: {c}")
    print(f"Put premium: {p}")
    print(f"Breakeven upper: {S0 + c + p}")
    print(f"Breakeven lower: {S0 - c - p}")
    plot_payoff(S, profiles2, title=f"ATM Straddle on {ticker}")
else:
    print("Could not find exact ATM strike in chain — adjust tolerance if needed.")

# --- Test 3: butterfly ---
K1, K2, K3 = S0 * 0.95, S0, S0 * 1.05
# Approximate premiums from chain (nearest strikes)
def nearest_call_price(chain, K):
    calls = chain[chain["type"] == "call"].copy()
    calls["dist"] = (calls["strike"] - K).abs()
    return calls.sort_values("dist").iloc[0]["lastPrice"]

c1 = nearest_call_price(chain, K1)
c2 = nearest_call_price(chain, K2)
c3 = nearest_call_price(chain, K3)
profiles3 = {
    "Butterfly": butterfly_spread(S, K1, K2, K3, c1, c2, c3),
}
print(f"\nButterfly strikes: {K1:.2f} / {K2:.2f} / {K3:.2f}")
print(f"Premiums: {c1:.2f} / {c2:.2f} / {c3:.2f}")
print(f"Max profit (at K2): {butterfly_spread(np.array([K2]), K1, K2, K3, c1, c2, c3)[0]:.2f}")
print(f"Max loss (below K1 or above K3): {butterfly_spread(np.array([K1 * 0.5]), K1, K2, K3, c1, c2, c3)[0]:.2f}")
plot_payoff(S, profiles3, title=f"Butterfly Spread on {ticker}")

plt.show()