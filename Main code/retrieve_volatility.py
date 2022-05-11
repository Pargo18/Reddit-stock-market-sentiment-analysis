import yfinance as yf
import pandas as pd
import time
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")

def keep_integer_hours(dates):
    idx = [i for i, date in enumerate(dates) if date.minute == 0]
    dates_clean = [pd.Timestamp(datetime(year=dates[i].year, month=dates[i].month, day=dates[i].day, hour=dates[i].hour)) for i in idx]
    return idx, dates_clean

df_tickers = pd.read_csv('..\\Data\\wallstreetbets_2021.csv')
ticker_lst = df_tickers['ticker'].tolist()

stock_data = yf.download(ticker_lst, start='2022-04-01', end='2022-05-10', interval='5m')
stock_data = stock_data.stack(level=0).rename_axis(['Date', 'Value']).reset_index(level=1)

stock_data_close = stock_data.loc[stock_data['Value'] == 'Close']
stock_data_close.drop(columns='Value', inplace=True)

stock_data_vol = stock_data_close.pct_change().rolling(12).std()
stock_data_vol.dropna(how='all', inplace=True)
date_idx, dates_clean = keep_integer_hours(stock_data_vol.index.to_list())
stock_data_vol = stock_data_vol.iloc[date_idx]
stock_data_vol.index = dates_clean
# stock_data_vol.plot()

stock_data_vol.to_csv('..\\Data\\Volatility_data.csv', index=True)
