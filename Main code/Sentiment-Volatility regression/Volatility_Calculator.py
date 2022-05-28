import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar
from datetime import datetime

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def keep_integer_hours(dates):
    idx = [i for i, date in enumerate(dates) if date.minute == 0]
    dates_clean = [pd.Timestamp(datetime(year=dates[i].year, month=dates[i].month, day=dates[i].day, hour=dates[i].hour)) for i in idx]
    return idx, dates_clean

def bs_call_price(S, K, T, r, vol):
    vol /= 100 # So that the main function works with the percentage of volatility
    d1 = (np.log(S/K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

def implied_volatility(S, K, T, r, C):
    f_obj = lambda x: np.abs(bs_call_price(S, K, T, r, x) - C)
    opt_vol = minimize_scalar(fun=f_obj, bounds=(1e-4, None))
    return opt_vol.x

def historical_volatility(df_close, trading_days=252):
    df_return = np.log(df_close / df_close.shift(1))
    df_return.fillna(0, inplace=True)
    df_vol = df_return.rolling(window=trading_days).std() * np.sqrt(trading_days)
    df_vol.dropna(how='all', inplace=True)
    date_idx, dates_clean = keep_integer_hours(df_vol.index.to_list())
    df_vol = df_vol.iloc[date_idx]
    df_vol.index = dates_clean
    return df_vol

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    S = 50     # Current option price
    K = 45     # Strike price
    T = 1      # Time to maturity (years)
    r = 0.04   # Risk-free rate
    C = 9      # Call price

    implied_vol = implied_volatility(S, K, T, r, C)

