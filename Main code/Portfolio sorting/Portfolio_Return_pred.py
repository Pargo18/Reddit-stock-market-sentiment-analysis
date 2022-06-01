import numpy as np
import pandas as pd
import copy
import Diff_Volatility_Metric as dvm
import Quintile_Portofolio_Sorting as qps

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':

    RV_data = pd.read_excel('..\\..\\Data\\Portfolio sorting\\Processed RV\\RV_10days.xlsx')
    RV_data['Timestamp'] = RV_data['Timestamp'].dt.date
    RV_data.dropna(axis=0, how='all', inplace=True)
    RV_data.fillna(method='ffill', inplace=True)

    IV_data = pd.read_excel('..\\..\\Data\\Portfolio sorting\\Processed IV\\MeanIV.xlsx')
    IV_data['Timestamp'] = IV_data['Timestamp'].dt.date
    IV_data.dropna(axis=0, how='all', inplace=True)
    IV_data.fillna(method='ffill', inplace=True)

    return_data = pd.read_csv('..\\..\\Data\\Return_data.csv')
    return_data.rename(columns={'Date': 'Timestamp'}, inplace=True)
    return_data['Timestamp'] = pd.to_datetime(return_data['Timestamp']).dt.date
    # return_data = return_data.iloc[-1:].reset_index(drop=True)
    # return_data.drop(columns=['Timestamp'], axis=1, inplace=True)

    df_diff = dvm.clean_diff_timelines(RV_data, IV_data)
    metric_func = dvm.metric_norm
    df_metric = dvm.apply_running_metric(df_diff, func=metric_func)

    n_port = 5
    df_port_sort = qps.sort_running_portfolio_quintiles(metric=copy.deepcopy(df_metric), n_port=n_port)
    df_port_returns = qps.portfolio_running_returns(df_port=copy.deepcopy(df_port_sort), df_returns=copy.deepcopy(return_data))

