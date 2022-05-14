import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import matplotlib
matplotlib.use("TkAgg")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def compile_option_data(sent_data, vol_data, ticker, features):
    df_vol = pd.DataFrame(data=vol_data[['Timestamp', ticker]].to_numpy(), columns=['Timestamp', 'Volatility'])
    df_vol['Timestamp'] = pd.to_datetime(df_vol['Timestamp']).dt.date
    # df_vol['Timestamp'] = [pd.to_datetime(item) for item in df_vol['Timestamp']]
    df_vol['Volatility'] = df_vol['Volatility'].astype(float)

    df = sent_data[sent_data[ticker]]
    df.reset_index(inplace=True, drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.date
    # df.loc['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.date
    # df['Timestamp'] = [pd.to_datetime(item) for item in df['Timestamp']]

    df = df[['Timestamp'] + features]
    df['Volatility'] = 0

    for i, timestep in enumerate(df['Timestamp']):
        if pd.to_datetime(timestep) in [pd.to_datetime(item) for item in df_vol['Timestamp']]:
            df['Volatility'].iloc[i] = df_vol[df_vol['Timestamp'] == pd.to_datetime(timestep)]['Volatility']
        else:
            if pd.to_datetime(timestep) - datetime.timedelta(1) in [pd.to_datetime(item) for item in
                                                                    df_vol['Timestamp']]:
                df['Volatility'].iloc[i] = \
                df_vol[df_vol['Timestamp'] == pd.to_datetime(timestep) - datetime.timedelta(1)]['Volatility']
            elif pd.to_datetime(timestep) - datetime.timedelta(2) in [pd.to_datetime(item) for item in
                                                                      df_vol['Timestamp']]:
                df['Volatility'].iloc[i] = \
                df_vol[df_vol['Timestamp'] == pd.to_datetime(timestep) - datetime.timedelta(2)]['Volatility']
            elif pd.to_datetime(timestep) - datetime.timedelta(3) in [pd.to_datetime(item) for item in
                                                                      df_vol['Timestamp']]:
                df['Volatility'].iloc[i] = \
                df_vol[df_vol['Timestamp'] == pd.to_datetime(timestep) - datetime.timedelta(3)]['Volatility']

    return df
