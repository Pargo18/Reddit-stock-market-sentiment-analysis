import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import matplotlib
# matplotlib.use("TkAgg")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def compile_option_data(df_1, df_2, features, ticker, new_attr='Volatility', lag=0):
    df_vol = pd.DataFrame(data=df_2[['Timestamp', ticker]].to_numpy(), columns=['Timestamp', new_attr])
    df_vol['Timestamp'] = pd.to_datetime(df_vol['Timestamp']).dt.date
    df_vol[new_attr] = df_vol[new_attr].astype(float)
    df_vol[new_attr] = df_vol[new_attr].shift(-lag)
    df_vol.dropna(how='any', inplace=True)

    df = df_1[df_1[ticker]]
    df.reset_index(inplace=True, drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.date

    df = df[['Timestamp'] + features]
    df[new_attr] = 0

    for i, timestep in enumerate(df['Timestamp']):
        if pd.to_datetime(timestep) in [pd.to_datetime(item) for item in df_vol['Timestamp']]:
            df[new_attr].iloc[i] = df_vol[df_vol['Timestamp'] == pd.to_datetime(timestep)][new_attr]
        else:
            if pd.to_datetime(timestep) + datetime.timedelta(1) in [pd.to_datetime(item) for item in df_vol['Timestamp']]:
                df[new_attr].iloc[i] = \
                df_vol[df_vol['Timestamp'] == pd.to_datetime(timestep) + datetime.timedelta(1)][new_attr]
            elif pd.to_datetime(timestep) + datetime.timedelta(2) in [pd.to_datetime(item) for item in df_vol['Timestamp']]:
                df[new_attr].iloc[i] = \
                df_vol[df_vol['Timestamp'] == pd.to_datetime(timestep) + datetime.timedelta(2)][new_attr]
            elif pd.to_datetime(timestep) + datetime.timedelta(3) in [pd.to_datetime(item) for item in df_vol['Timestamp']]:
                df[new_attr].iloc[i] = \
                df_vol[df_vol['Timestamp'] == pd.to_datetime(timestep) + datetime.timedelta(3)][new_attr]

    return df

def fix_features(df, features_sum, features_mean, features_max):
    df_1 = df.groupby(['Timestamp'])[features_sum].apply(lambda x: x.sum())
    df_2 = df.groupby(['Timestamp'])[features_max].apply(lambda x: x.max())
    df_3 = df.groupby(['Timestamp'])[features_mean].apply(lambda x: x.mean())
    df = pd.concat([df_1, df_2, df_3], axis=1)
    df = create_features(df)
    return df

def create_features(df):
    # df['Volatility'] *= df['score'] / df['Volume']
    return df