import numpy as np
import pandas as pd
import copy
import Diff_Volatility_Metric as dvm

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def sort_portfolio_quintiles(metric, n_port=5):
    port_sort = list(pd.qcut(metric.values.flatten(), q=n_port, labels=[i+1 for i in range(n_port)]).T)
    df_portfolio = pd.DataFrame(data=[port_sort], columns=metric.columns)
    return df_portfolio

def sort_running_portfolio_quintiles(metric, n_port=5):

    timestamps = list(metric['Timestamp'])
    metric.drop(columns=['Timestamp'], inplace=True)
    metric = metric.astype(float)

    df_portfolio = pd.DataFrame(data=[], columns=metric.columns)
    for idx, row in metric.iterrows():
        port_sort = list(pd.qcut(row.values.flatten(), q=n_port, labels=[i+1 for i in range(n_port)]).T)
        df_portfolio = pd.concat([df_portfolio, pd.DataFrame(data=[port_sort], columns=metric.columns)])

    df_portfolio.reset_index(drop=True, inplace=True)
    df_portfolio.insert(0, column='Timestamp', value=timestamps)

    return df_portfolio

def portfolio_returns(df_port, df_returns):
    df_merge = pd.concat([df_port, df_returns], axis=0).T
    df_merge.columns = ['Portfolio', 'Return']
    df_port_returns = df_merge.groupby(by=['Portfolio']).mean()
    return df_port_returns

def portfolio_running_returns(df_port, df_returns):

    tickers = [col for col in df_port.columns if col != 'Timestamp']

    n_port = df_port[tickers].values.max()
    port_cols = [i for i in range(1, n_port+1)]

    df_port_returns = pd.DataFrame(data=[], columns=['Timestamp']+port_cols)

    for idx, row in df_port.iterrows():
        timestamp = row['Timestamp']
        if timestamp in list(df_returns['Timestamp']):
            port = row[tickers].values.flatten()
            returns = df_returns[df_returns['Timestamp']==timestamp][tickers].values.flatten()
            unique, _ = np.unique(port, return_counts=True)
            if unique.max() <= n_port:
                port_returns = [np.nanmean(returns[port==i]) for i in range(1, n_port+1)]
            else:
                port_returns = [np.nanmean(returns[port==i]) for i in unique]
            df_port_returns = pd.concat([df_port_returns, pd.DataFrame(data=[[timestamp] + port_returns], columns=df_port_returns.columns)])
        else:
            continue

    df_port_returns.reset_index(drop=True, inplace=True)

    return df_port_returns

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':

    RV_data = pd.read_csv('..\\..\\Data\\volatility_data.csv')
    RV_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
    IV_data = pd.read_csv('..\\..\\Data\\Implied_volatility.csv')
    IV_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)

    return_data = pd.read_csv('..\\..\\Data\\Return_data.csv')
    return_data.rename(columns={'Date': 'Timestamp'}, inplace=True)
    # return_data.drop(columns=['Timestamp'], axis=1, inplace=True)
    # return_data = return_data.iloc[-1:].reset_index(drop=True)

    df_diff = dvm.clean_diff_timelines(RV_data, IV_data)

    metric_func = dvm.metric_norm

    # df_metric = dvm.apply_metric(df_diff, func=metric_func)
    # df_port_sort = sort_portfolio_quintiles(df_metric)
    # df_port_returns = portfolio_returns(df_port=df_port_sort, df_returns=return_data)

    df_metric = dvm.apply_running_metric(df_diff, func=metric_func)
    df_port_sort = sort_running_portfolio_quintiles(copy.deepcopy(df_metric))
    df_port_returns = portfolio_running_returns(df_port=copy.deepcopy(df_port_sort), df_returns=copy.deepcopy(return_data))
