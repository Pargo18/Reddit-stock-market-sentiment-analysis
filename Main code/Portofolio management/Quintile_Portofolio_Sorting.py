import numpy as np
import pandas as pd
import copy
import Diff_Volatility_Metric as dvm

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def sort_portfolio_quintiles(df_metric, n_port=5):
    port_sort = list(pd.qcut(df_metric.values.flatten(), q=n_port, labels=[i+1 for i in range(n_port)]).T)
    df_portfolio = pd.DataFrame(data=[port_sort], columns=df_metric.columns)
    return df_portfolio


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


if __name__ == '__main__':

    RV_data = pd.read_csv('..\\..\\Data\\volatility_data.csv')
    RV_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
    IV_data = pd.read_csv('..\\..\\Data\\Implied_volatility.csv')
    IV_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)

    return_data = pd.read_csv('..\\..\\Data\\Return_data.csv')
    return_data.rename(columns={'Date': 'Timestamp'}, inplace=True)
    return_data = return_data.iloc[-1:].reset_index(drop=True)
    return_data.drop(columns=['Timestamp'], axis=1, inplace=True)

    df_diff = dvm.clean_diff_timelines(RV_data, IV_data)

    metric_func = dvm.metric_norm

    df_metric = dvm.apply_metric(df_diff, func=metric_func)

    df_port_sort = sort_portfolio_quintiles(df_metric)

    df_merge = pd.concat([df_port_sort, return_data], axis=0).T
    df_merge.columns = ['Portfolio', 'Return']
    df_port_returns = df_merge.groupby(by=['Portfolio']).mean()
