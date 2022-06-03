import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import Diff_Volatility_Metric as dvm
import Quintile_Portofolio_Sorting as qps
import Portfolio_Return_pred as pr
import Visualizations as vs

from sklearn.linear_model import LinearRegression

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def port_performance(df_port_returns, per_top=0.9):  # per_top<=1 - could be used for log returns as well
    per_bottom = 1 - per_top
    # split columns to top and bottom
    cols = [col for col in df_port_returns.columns if col != 'Timestamp']
    cols_top = cols[:int(per_top * len(cols))]
    cols_bottom = cols[int(per_top * len(cols))]

    # add tpm, bottom and EW columns to df
    if isinstance(cols_top, list):
        df_port_returns['top'] = df_port_returns[cols_top].mean(axis=1)
    else:
        df_port_returns['top'] = df_port_returns[cols_top]
    if isinstance(cols_bottom, list):
        df_port_returns['bottom'] = df_port_returns[cols_bottom].mean(axis=1)
    else:
        df_port_returns['bottom'] = df_port_returns[cols_bottom]
    df_port_returns['EW'] = df_port_returns[cols].mean(axis=1)
    df_port_returns['L/S'] = df_port_returns['top'] - df_port_returns['bottom']

    return df_port_returns


def regression(df_port_returns, df_risk, col):  # specify regressand column
    reg = LinearRegression().fit(df_risk.drop(['date', 'Timestamp', 'RF'], axis=1).to_numpy(),
                                 df_port_returns[col].to_numpy())
    return (reg.coef_, reg.intercept_)


def summary(df_port_returns, df_risk):  # summary table of results
    columns = ['top', 'bottom', 'EW', 'L/S']  # columns for which the regression will be ran
    data = []
    for col in columns:
        coef, intercept = regression(df_port_returns, df_risk, col)
        data.append(np.append(np.append(coef, intercept), col))

    df = pd.DataFrame(data, columns=['CMA', 'RMW', 'HML', 'SMB', 'Mkt-RF', 'Intercept', 'method'])
    df.set_index('method', inplace=True)
    return df