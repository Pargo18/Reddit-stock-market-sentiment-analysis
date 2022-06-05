import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm
import seaborn as sns
import copy
import Diff_Volatility_Metric as dvm
import Portofolio_Sorting as qps
import Portfolio_Return_pred as pr
import Visualizations as vs
from Linear_Regression import linear_regression_portfolio

from sklearn.linear_model import LinearRegression


import matplotlib
matplotlib.use("TkAgg")
# matplotlib.use("Agg")

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

if __name__ == '__main__':

    input_folder = '..\\..\\Data\\Portfolio sorting'

    df_IV = pd.read_excel(input_folder + '\\Processed IV\\' + 'OrganizedIV.xlsx')
    df_IV['Timestamp'] = df_IV['Timestamp'].dt.date
    df_IV.dropna(axis=0, how='all', inplace=True)
    df_IV.fillna(method='ffill', inplace=True)

    df_RV = pd.read_excel(input_folder + '\\Processed RV\\' +'RV_10days.xlsx')
    df_RV['Timestamp'] = df_RV['Timestamp'].dt.date
    df_RV.dropna(axis=0, how='all', inplace=True)
    df_RV.fillna(method='bfill', inplace=True)

    return_data = pd.read_csv('..\\..\\Data\\Return_data.csv')
    return_data.rename(columns={'Date': 'Timestamp'}, inplace=True)
    return_data['Timestamp'] = pd.to_datetime(return_data['Timestamp']).dt.date

    df_diff = dvm.clean_diff_timelines(df_RV, df_IV)
    df_metric = dvm.apply_running_metric(copy.deepcopy(df_diff), func=dvm.metric_norm)
    # df_metric = dvm.apply_running_metric(copy.deepcopy(df_diff), func=dvm.metric_last)
    df_metric.interpolate(method='linear', axis=0, inplace=True)
    df_metric.dropna(how='all', axis=1, inplace=True)

    df_port_sort = qps.sort_running_portfolio(copy.deepcopy(df_metric), n_port=10)  # df metric per ticker and number of portfolios
    df_port_returns = qps.portfolio_running_returns(df_port=copy.deepcopy(df_port_sort), df_returns=copy.deepcopy(return_data))
    df_port_returns = port_performance(df_port_returns)

    df_risk = pd.read_excel('..\\..\\Data\\Portfolio sorting\\' + 'Data_5_Factors_2x3_daily.xlsx')
    df_risk.rename(columns={'date': 'Timestamp', 'Unnamed: 0': 'Timestamp'}, inplace=True)
    df_risk['Timestamp'] = pd.to_datetime(df_risk['Timestamp'].astype(str), format='%Y%m%d')
    df_risk['Timestamp'] = pd.to_datetime(df_risk['Timestamp']).dt.date




    common_timestamps = []
    for timestamp in df_port_returns['Timestamp']:
        if timestamp in list(df_risk['Timestamp']):
            common_timestamps.append(timestamp)

    df_risk.drop(columns=['RF'], inplace=True)

    covariate_names = [col for col in df_risk.columns if col!='Timestamp']
    outcome_names = ['top', 'bottom', 'EW', 'L/S']

    df_covariates = df_risk[df_risk['Timestamp'].isin(common_timestamps)][covariate_names]
    df_outcomes = df_port_returns[df_port_returns['Timestamp'].isin(common_timestamps)][outcome_names]


    df_beta_ols = pd.DataFrame(data=[], columns=outcome_names, index=['Intercept']+covariate_names)
    df_beta_sd = pd.DataFrame(data=[], columns=outcome_names, index=['Intercept']+covariate_names)
    # df_pval = pd.DataFrame(data=[], columns=outcome_names, index=['Intercept']+covariate_names+['ANOVA'])
    df_pval = pd.DataFrame(data=[], columns=outcome_names, index=['Intercept']+covariate_names)
    df_tstudent_dof = pd.DataFrame(data=[], columns=outcome_names, index=[0])

    for outcome in outcome_names:
        beta, beta_sd, p_val, f_val = linear_regression_portfolio(covariates=df_covariates.values, outcome=df_outcomes[outcome].values)
        df_beta_ols[outcome] = beta
        df_beta_sd[outcome] = beta_sd
        # df_pval[outcome] = np.append(p_val, f_val)
        df_pval[outcome] = p_val
        df_tstudent_dof[outcome] = len(df_covariates) - (len(covariate_names) + 1)

    df_beta_ols.to_csv('..\\..\\Data\\Output\\Portfolio sorting\\Beta_OLS.csv')
    df_beta_sd.to_csv('..\\..\\Data\\Output\\Portfolio sorting\\Beta_Sd.csv')
    df_pval.to_csv('..\\..\\Data\\Output\\Portfolio sorting\\p_values.csv')
    df_tstudent_dof.to_csv('..\\..\\Data\\Output\\Portfolio sorting\\dof_tstudent.csv', index=False)



    sns.heatmap(df_pval)
    plt.title('p-value per outcome and risk factor', fontsize=16)
    plt.xlabel('Outcome', fontsize=14)
    plt.ylabel('Risk factors', fontsize=14)
    plt.savefig('..\\..\\Data\\Output\\Portfolio sorting\\p_val_heatmap.png', bbox_inches='tight')


    fig, axs = plt.subplots(df_beta_ols.shape[0], df_beta_ols.shape[1])
    plt.suptitle('Distribution of regression coefficients per outcome and covariate\n'
                 '(coefficient value of zero indicated with a red line)', fontsize=12)
    for j, outcome in enumerate(df_beta_ols.columns):
        dof = df_tstudent_dof[outcome]
        for i, covariate in enumerate(df_beta_ols.index):
            ax = axs[i, j]
            mu = df_beta_ols.loc[covariate, outcome]
            sd = df_beta_sd.loc[covariate, outcome]
            x = np.linspace(mu - 6 * sd, mu + 6 * sd, 300)
            # y = norm.pdf(x, loc=mu, scale=sd)
            y = t.pdf(x, loc=mu, scale=sd, df=dof)

            ax.plot(x, y, color='b')
            ax.axvline(0, color='r')
            ax.set_xlabel(outcome, fontsize=14)
            # if len(covariate.split('_')) > 1:
            #     ax.set_ylabel('\n'.join(covariate.split('_')), fontsize=12)
            # else:
            #     ax.set_ylabel(covariate, fontsize=12)
            ax.set_ylabel(covariate, fontsize=12)

            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
            )

            ax.grid(False)

            if i < axs.shape[0] - 1:
                ax.axes.get_xaxis().set_visible(False)
            if j > 0:
                ax.axes.get_yaxis().set_visible(False)
    plt.savefig('..\\..\\Data\\Output\\Portfolio sorting\\beta_dist.png', bbox_inches='tight')


    # fig, axs = plt.subplots(df_beta_ols.shape[0]-1, df_beta_ols.shape[1])
    # plt.suptitle('Regression model of each outcome projected per covariate-outcome plane', fontsize=12)
    # for j, outcome in enumerate(df_beta_ols.columns):
    #     X = df_covariates.values
    #     X = np.c_[np.ones(X.shape[0]), X]
    #     Y = df_outcomes[outcome].values
    #     XX = np.vstack((X.min(axis=0), X.max(axis=0)))
    #     Y_hat = np.dot(df_beta_ols[outcome].values, XX.T)
    #     beta = df_beta_ols[outcome].values
    #     X = X[:, 1:]
    #     XX = XX[:, 1:]
    #     for i, covariate in enumerate([idx for idx in df_beta_ols.index if idx!='Intercept']):
    #         ax = axs[i, j]
    #         ax.scatter(X[:, i], Y, color='r')
    #         ax.plot(XX[:, i], Y_hat, color='b')
    #         # ax.plot(XX[:, i], beta[0]+XX[:, i]*beta[i+1], color='b')
    #         ax.set_xlabel(outcome, fontsize=14)
    #         ax.set_ylabel(covariate, fontsize=12)
    #
    #         ax.tick_params(
    #             axis='both',
    #             which='both',
    #             bottom=False,
    #             top=False,
    #             left=False,
    #             right=False,
    #             labelbottom=False,
    #             labeltop=False,
    #             labelleft=False,
    #             labelright=False,
    #         )
    #
    #         ax.grid(False)
    #
    #         if i < axs.shape[0] - 1:
    #             ax.axes.get_xaxis().set_visible(False)
    #         if j > 0:
    #             ax.axes.get_yaxis().set_visible(False)
    #
    # plt.savefig('..\\..\\Data\\Output\\Portfolio sorting\\regression.png', bbox_inches='tight')


    outcome = 'top'
    # df = pd.concat([df_covariates, df_outcomes[outcome]], axis=1)
    df = pd.concat([df_covariates, df_outcomes[outcome_names]], axis=1)
    # g = sns.PairGrid(df, y_vars=[outcome], x_vars=covariate_names, height=4)
    g = sns.PairGrid(df, y_vars=outcome_names, x_vars=covariate_names, height=4)
    g.map(sns.regplot, color='blue')
    # g.set(ylim=(-1, 11), yticks=[0, 5, 10])
    plt.savefig('..\\..\\Data\\Output\\Portfolio sorting\\regression.png', bbox_inches='tight')


    reg = LinearRegression().fit(df_covariates, df_outcomes['top'].values.reshape(-1, 1))
    reg.coef_
    reg.intercept_
