import numpy as np
import pandas as pd
import copy

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def clean_diff_timelines(df_RV, df_IV):


    first_date, last_date = max(df_RV['Timestamp'].min(), df_IV['Timestamp'].min()), \
                            min(df_RV['Timestamp'].max(), df_IV['Timestamp'].max())

    # tickers = [col for col in df_IV.columns if col!='Timestamp']
    # tickers = [ticker for ticker in tickers if ticker in list(df_RV.columns)]
    common_cols = np.intersect1d(df_RV.columns, df_IV.columns)
    tickers = [col for col in common_cols if col != 'Timestamp']
    common_cols = ['Timestamp'] + tickers

    df_RV = df_RV[(df_RV['Timestamp']>=first_date) & (df_RV['Timestamp']>=first_date)].reset_index(drop=True)
    df_IV = df_IV[(df_IV['Timestamp']>=first_date) & (df_IV['Timestamp']>=first_date)].reset_index(drop=True)

    if len(df_RV) >= len(df_IV):
        df_RV = df_RV[df_RV['Timestamp'].isin(list(df_IV['Timestamp']))].reset_index(drop=True)
        df_IV = df_IV[df_IV['Timestamp'].isin(list(df_RV['Timestamp']))].reset_index(drop=True)
    else:
        df_IV = df_IV[df_IV['Timestamp'].isin(list(df_RV['Timestamp']))].reset_index(drop=True)
        df_RV = df_RV[df_RV['Timestamp'].isin(list(df_IV['Timestamp']))].reset_index(drop=True)

    df_diff = copy.deepcopy(df_RV[common_cols])
    df_diff[tickers] = df_diff[tickers].astype(float)
    df_diff[tickers] -= df_IV[tickers].astype(float)

    return df_diff

def metric_norm(x):
    #TODO
    # return np.linalg.norm(np.log(x), axis=0) / x.shape[0]
    return np.linalg.norm(x, axis=0) / x.shape[0]

def apply_metric(df_diff, func):
    tickers = [col for col in df_diff.columns if col!='Timestamp']
    df_metric = pd.DataFrame(data=[func(df_diff[tickers].values)], columns=tickers)
    return df_metric

def apply_running_metric(df_diff, func):
    tickers = [col for col in df_diff.columns if col!='Timestamp']
    df_metric = pd.DataFrame(data=[], columns=df_diff.columns)
    for idx, row in df_diff.iterrows():
        if idx > 0:
            x = df_diff[tickers].iloc[:idx].values
            data = [[row['Timestamp']] + [i for i in func(x)]]
            df_metric = pd.concat([df_metric, pd.DataFrame(data=data, columns=df_diff.columns)])
    df_metric.reset_index(drop=True, inplace=True)
    return df_metric

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':

    RV_data = pd.read_csv('..\\..\\Data\\volatility_data.csv')
    RV_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
    IV_data = pd.read_csv('..\\..\\Data\\Implied_volatility.csv')
    IV_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)

    df_diff = clean_diff_timelines(RV_data, IV_data)

    df_metric = apply_metric(df_diff, func=metric_norm)

    df_metric = apply_running_metric(df_diff, func=metric_norm)
