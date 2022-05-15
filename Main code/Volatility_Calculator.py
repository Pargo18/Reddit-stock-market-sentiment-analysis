import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def bs_call_price(S, K, T, r, vol):
    vol /= 100 # So that the main function works with the percentage of volatility
    d1 = (np.log(S/K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

def calc_volatility(S, K, T, r, C):
    f_obj = lambda x: np.abs(bs_call_price(S, K, T, r, x) - C)
    opt_vol = minimize_scalar(fun=f_obj, bounds=(1e-4, None))
    return opt_vol.x

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    S = 50     # Current option price
    K = 45     # Strike price
    T = 1      # Time to maturity (years)
    r = 0.04   # Risk-free rate
    C = 9      # Call price

    implied_vol = calc_volatility(S, K, T, r, C)

