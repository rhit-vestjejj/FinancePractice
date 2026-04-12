import numpy as np

def rolling_pnl(spot_prices: list, entry_prices: list, exit_prices:list, 
                roll_dates: list, quantity: float) -> dict:
    # calculate PnL of rolling hedge across multiple roll dates
    # returns total pnl, roll costs, and effective hedge price
    
    total_pnl = 0
    roll_costs = []
    period_pnls = []
    
    for i in range(len(roll_dates) - 1):
        period_pnl = (entry_prices[i] - exit_prices[i]) * quantity
        period_pnls.append(period_pnl)
        total_pnl += period_pnl
        
        if i < len(roll_dates) - 2:
            cost = roll_cost(exit_prices[i], entry_prices[i + 1], quantity)
            roll_costs.append(cost)
            total_pnl -= cost
    
    return {
        'total_pnl': total_pnl,
        'period_pnls': period_pnls,
        'roll_costs': roll_costs,
        'effective_hedge_price': effective_hedge_price(entry_prices[0], roll_costs, quantity)
    }

def roll_cost(futures_price_old: float, futures_price_new: float, 
              quantity: float) -> float:
    # cost of rolling from expiring contract to new contract

    roll_cost = (futures_price_new - futures_price_old) * quantity

    return roll_cost

def effective_hedge_price(initial_futures: float, 
                          roll_costs: list,
                          quantity: float) -> float:
    # total effective price after all rolls

    effective_price = initial_futures + np.sum(roll_costs) / quantity

    return effective_price

entry_prices = [100, 103, 106]
exit_prices = [102, 105, 108]

roll_dates = ['2023-01-01', '2023-04-01', '2023-07-01', '2024-01-01']

quantity = 1000

result = rolling_pnl(None, entry_prices, exit_prices, roll_dates, quantity)
print(f"Total PnL: {result['total_pnl']}")
print(f"Period PnLs: {result['period_pnls']}")
print(f"Roll Costs: {result['roll_costs']}")
print(f"Effective Hedge Price: {result['effective_hedge_price']}")