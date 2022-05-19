import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from Volatility_Calculator import historical_volatility

import matplotlib
matplotlib.use("TkAgg")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def download_options(tickers, date_start='2019-1-1', date_end=None, interval='1d'):
    if date_end is None:
        date_end = datetime.date.today()
    option_data = yf.download(tickers, start=date_start, end=date_end, interval=interval)
    option_data = option_data.stack(level=0).rename_axis(['Date', 'Value']).reset_index(level=1)
    return option_data

def compile_df(df_raw, attr='Close'):
    df_close = df_raw.loc[df_raw['Value'] == attr]
    df_close.drop(columns='Value', inplace=True)
    return df_close

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':

    df_tickers = pd.read_csv('..\\Data\\FinalTickers.csv')
    ticker_lst = df_tickers['ticker'].tolist()

    option_data = download_options(ticker_lst)

    option_data_close = compile_df(df_raw=option_data, attr='Close')
    option_data_close.to_csv('..\\Data\\Close_data.csv', index=True)

    option_data_vol = historical_volatility(df_close=option_data_close, trading_days=252)
    option_data_vol.to_csv('..\\Data\\Volatility_data.csv', index=True)

    option_data_volume = compile_df(df_raw=option_data, attr='Volume')
    option_data_volume.to_csv('..\\Data\\Volume_data.csv', index=True)

