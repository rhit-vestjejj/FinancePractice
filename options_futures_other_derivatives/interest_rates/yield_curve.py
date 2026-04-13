import numpy as np

def bootstrap(maturities: list, prices: list,
              coupon_rates: list, face: float = 1000,
              frequency: int = 2) -> dict:
    
    spot_rates = {}
    
    for bond_idx in range(len(maturities)):
        maturity = maturities[bond_idx]
        price = prices[bond_idx]
        coupon_rate = coupon_rates[bond_idx]
        
        C = (coupon_rate / frequency) * face
        
        n_periods = int(maturity * frequency)
        coupon_times = [(i + 1) / frequency for i in range(n_periods)]
        
        pv_coupons = 0
        for t in coupon_times[:-1]:
            pv_coupons += C * np.exp(-spot_rates[t] * t)
        
        remainder = price - pv_coupons
        last_cash_flow = C + face
        
        r_new = -np.log(remainder / last_cash_flow) / maturity
        
        spot_rates[maturity] = r_new
    
    return spot_rates
        

def interpolate_rate(spot_rates: dict, maturity: float) -> float:
    # linear interpolation for maturities between known points
    maturities = sorted(spot_rates.keys())
    rates = [spot_rates[m] for m in maturities]
    return np.interp(maturity, maturities, rates)


maturities = [0.5, 1.0, 1.5, 2.0]
coupon_rates = [0.045, 0.045, 0.045, 0.045]
prices = [1000, 1000, 1000, 1000]

spot_rates = bootstrap(maturities, prices, coupon_rates)
print("Spot rates:")
for m, r in spot_rates.items():
    print(f"  {m} year: {r:.4f}")

print(f"\nInterpolated rate at 1.3 years: {interpolate_rate(spot_rates, 1.3):.4f}")

