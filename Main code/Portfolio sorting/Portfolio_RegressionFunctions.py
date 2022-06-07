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


# import matplotlib
# matplotlib.use("TkAgg")
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
    df_IV.fillna(method='bfill', inplace=True)

    df_RV = pd.read_excel(input_folder + '\\Processed RV\\' +'RV_10days.xlsx')
    df_RV['Timestamp'] = df_RV['Timestamp'].dt.date
    df_RV.dropna(axis=0, how='all', inplace=True)
    df_RV.fillna(method='bfill', inplace=True)

    return_data = pd.read_csv('..\\..\\Data\\Return_data.csv')
    return_data.rename(columns={'Date': 'Timestamp'}, inplace=True)
    return_data['Timestamp'] = pd.to_datetime(return_data['Timestamp']).dt.date
    return_data[[col for col in return_data.columns if col != 'Timestamp']] =\
        return_data[[col for col in return_data.columns if col != 'Timestamp']].shift(-1)


    df_diff = dvm.clean_diff_timelines(df_RV, df_IV)
    # df_metric = dvm.apply_running_metric(copy.deepcopy(df_diff), func=dvm.metric_norm)
    df_metric = dvm.apply_running_metric(copy.deepcopy(df_diff), func=dvm.metric_last)
    df_metric.interpolate(method='linear', axis=0, inplace=True)
    df_metric.dropna(how='all', axis=1, inplace=True)

    df_port_sort = qps.sort_running_portfolio(copy.deepcopy(df_metric), n_port=10)  # df metric per ticker and number of portfolios
    df_port_returns = qps.portfolio_running_returns(df_port=copy.deepcopy(df_port_sort), df_returns=copy.deepcopy(return_data))
    df_port_returns = port_performance(df_port_returns)

    df_risk = pd.read_excel('..\\..\\Data\\Portfolio sorting\\' + 'Data_5_Factors_2x3_daily.xlsx')
    df_risk.rename(columns={'date': 'Timestamp', 'Unnamed: 0': 'Timestamp'}, inplace=True)
    df_risk['Timestamp'] = pd.to_datetime(df_risk['Timestamp'].astype(str), format='%Y%m%d')
    df_risk['Timestamp'] = pd.to_datetime(df_risk['Timestamp']).dt.date
    df_risk[[col for col in df_risk.columns if col != 'Timestamp']] =\
        df_risk[[col for col in df_risk.columns if col != 'Timestamp']] / 100

    common_timestamps = []
    for timestamp in df_port_returns['Timestamp']:
        if timestamp in list(df_risk['Timestamp']):
            common_timestamps.append(timestamp)



    output_folder = '..\\..\\Data\\Output\\Portfolio sorting\\5 factors'

    #TODO
    covariate_names = [col for col in df_risk.columns if col!='Timestamp']
    covariate_names.remove('RF')

    outcome_names = ['top', 'bottom', 'EW', 'L/S']

    df_covariates = df_risk[df_risk['Timestamp'].isin(common_timestamps)][covariate_names]
    df_outcomes = df_port_returns[df_port_returns['Timestamp'].isin(common_timestamps)][outcome_names]


    df_beta_ols = pd.DataFrame(data=[], columns=outcome_names, index=['Intercept']+covariate_names)
    df_beta_sd = pd.DataFrame(data=[], columns=outcome_names, index=['Intercept']+covariate_names)
    df_t_stat = pd.DataFrame(data=[], columns=outcome_names, index=['Intercept']+covariate_names)
    df_pval = pd.DataFrame(data=[], columns=outcome_names, index=['Intercept']+covariate_names)
    df_R_sq = pd.DataFrame(data=[], columns=outcome_names, index=[0])
    df_SE = pd.DataFrame(data=[], columns=outcome_names, index=[0])
    df_tstudent_dof = pd.DataFrame(data=[], columns=outcome_names, index=[0])
    df_alpha = pd.DataFrame(data=[], columns=outcome_names, index=['Estimator', 'Standard error', 'p-value'])

    for outcome in outcome_names:
        beta, beta_sd, p_val, f_val, t_stat, R_sq, SE = linear_regression_portfolio(covariates=df_covariates.values, outcome=df_outcomes[outcome].values)
        df_beta_ols[outcome] = beta
        df_beta_sd[outcome] = beta_sd
        df_t_stat[outcome] = t_stat
        df_pval[outcome] = p_val
        df_R_sq[outcome] = R_sq
        df_SE[outcome] = SE
        df_tstudent_dof[outcome] = len(df_covariates) - (len(covariate_names) + 1)

        t_obs = (beta[0] - 0) / beta_sd[0]
        pval_alpha = 1 - t.cdf(x=np.abs(t_obs), df=df_tstudent_dof[outcome].values, loc=0, scale=1)
        df_alpha[outcome] = [beta[0].item(), beta_sd[0], pval_alpha.item()]

        df_outcome_summary = pd.DataFrame(data=np.vstack((beta.flatten(), beta_sd, t_stat,
                                                          np.append(R_sq, np.ones(len(covariate_names))*np.nan),
                                                          np.append(SE, np.ones(len(covariate_names))*np.nan))),
                                          columns=['Intercept']+covariate_names,
                                          index=['Coefficient', 'Std error', 't-stat', 'R^2', 'SE'])

        df_outcome_summary.to_csv(output_folder + '\\Summary_' + outcome.replace('/', '') + '.csv')

    df_beta_ols.to_csv(output_folder + '\\Beta_OLS.csv')
    df_beta_sd.to_csv(output_folder + '\\Beta_Sd.csv')
    df_t_stat.to_csv(output_folder + '\\t-stat.csv')
    df_pval.to_csv(output_folder + '\\p_values.csv')
    df_R_sq.to_csv(output_folder + '\\R_squared.csv', index=False)
    df_SE.to_csv(output_folder + '\\SE.csv', index=False)
    df_tstudent_dof.to_csv(output_folder + '\\dof_tstudent.csv', index=False)
    df_tstudent_dof.to_csv(output_folder + '\\alpha_results.csv', index=False)



    covariate_names2 = copy.deepcopy(covariate_names)
    covariate_names2.append('RF')
    df_covariates2 = df_risk[df_risk['Timestamp'].isin(common_timestamps)][covariate_names2]
    dummy = df_outcomes - pd.DataFrame(data=np.c_[df_covariates2['RF'].values, df_covariates2['RF'].values,
                                                  np.zeros(len(df_covariates2)), np.zeros(len(df_covariates2))],
                                       columns=['top', 'bottom', 'L/S', 'EW'])
    # dummy_2 = np.log(df_outcomes)
    cum_return = df_outcomes.sum(axis=0)
    annual_return = df_outcomes.mean(axis=0) * 252
    return_std = df_outcomes.std(axis=0)
    sharpe = dummy.mean(axis=0) / return_std
    alpha = df_alpha.loc['Estimator']
    annual_alpha = alpha * 252

    df_performance = pd.concat([cum_return, annual_return, return_std, sharpe, alpha, annual_alpha], axis=1)
    df_performance.columns = ['Cumulative return', 'Annual return', 'STD', 'Sharpe ratio', 'Alpha', 'Annual alpha']
    df_performance = df_performance.transpose()



    sns.heatmap(df_pval)
    plt.title('p-value per outcome and risk factor', fontsize=16)
    plt.xlabel('Outcome', fontsize=14)
    plt.ylabel('Risk factors', fontsize=14)
    plt.savefig(output_folder + '\\p_val_heatmap.png', bbox_inches='tight')


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
    plt.savefig(output_folder + '\\beta_dist.png', bbox_inches='tight')


    outcome = 'top'
    # df = pd.concat([df_covariates, df_outcomes[outcome]], axis=1)
    df = pd.concat([df_covariates, df_outcomes[outcome_names]], axis=1)
    # g = sns.PairGrid(df, y_vars=[outcome], x_vars=covariate_names, height=4)
    g = sns.PairGrid(df, y_vars=outcome_names, x_vars=covariate_names, height=4)
    g.map(sns.regplot, color='blue')
    # g.set(ylim=(-1, 11), yticks=[0, 5, 10])
    plt.savefig(output_folder + '\\regression.png', bbox_inches='tight')

