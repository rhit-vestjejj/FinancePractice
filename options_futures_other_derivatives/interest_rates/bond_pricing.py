import numpy as np

def bond_price_continuous(face: float, coupon_rate: float, 
                          maturity: float, yield_rate: float,
                          frequency: int = 2) -> float:
    # price using continuous compounding

    price = 0
    C = (coupon_rate / frequency) * face

    for i in range (maturity * frequency):
        price += C * np.exp(-yield_rate * ((i + 1) / frequency))
    price += face * np.exp(-yield_rate * maturity)

    return price

def bond_price_discrete(face: float, coupon_rate: float,
                        maturity: float, yield_rate: float,
                        frequency: int = 2) -> float:
    # price using discrete compounding
    # frequency = coupon payments per year (2 = semiannual)

    price = 0
    C = (coupon_rate / frequency) * face
    y = yield_rate / frequency

    for i in range (maturity * frequency):
        price += C / ((1 + y) ** (i + 1))
    price += face / ((1 + y) ** (maturity * frequency))
    
    return price

price = bond_price_continuous(face=1000, coupon_rate=0.05, maturity=2, yield_rate=0.05)
print(f"Par bond price: {price}")

price_discount = bond_price_continuous(face=1000, coupon_rate=0.05, maturity=2, yield_rate=0.07)
print(f"Discount bond: {price_discount}")

price_premium = bond_price_continuous(face=1000, coupon_rate=0.05, maturity=2, yield_rate=0.03)
print(f"Premium bond: {price_premium}")

price2 = bond_price_discrete(face=1000, coupon_rate=0.05, maturity=2, yield_rate=0.05)
print(f"Par bond price 2: {price2}")

price_discount2 = bond_price_discrete(face=1000, coupon_rate=0.05, maturity=2, yield_rate=0.07)
print(f"Discount bond 2: {price_discount2}")

price_premium2 = bond_price_discrete(face=1000, coupon_rate=0.05, maturity=2, yield_rate=0.03)
print(f"Premium bond 2: {price_premium2}")