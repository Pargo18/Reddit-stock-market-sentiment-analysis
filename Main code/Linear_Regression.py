import numpy as np
import pandas as pd
from scipy.stats import t, f
from generate_features import compile_option_data, fix_features
import seaborn as sns

import matplotlib
matplotlib.use("TkAgg")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def check_invertability(X):
    try:
        X_inv = np.linalg.inv(np.dot(X.T, X))
    except:
        return False
    else:
        return True

def solve_OLS(X, Y):
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return beta

def error_estimator(beta, X, Y):
    residuals = Y - np.dot(X, beta)
    RSS = np.dot(residuals.T, residuals)
    sigma_var = RSS.item() / (X.shape[0] - X.shape[1])
    return sigma_var

def calc_beta_var(sigma_var, X):
    beta_covar = sigma_var * np.linalg.inv(np.dot(X.T, X))
    return np.diag(beta_covar)

def calc_p_value(beta, beta_var, dof, beta_Ho=None):
    if beta_Ho is None:
        beta_Ho = np.zeros(beta.shape[0]).reshape(-1, 1)
    p_val = np.repeat(1.0, repeats=beta.shape[0])
    for i, (b, b_Ho, b_var) in enumerate(zip(beta, beta_Ho, beta_var)):
        T_obs = (b - b_Ho) / np.sqrt(b_var)
        p_val[i] = 2 * (1 - t.cdf(x=np.abs(T_obs), df=dof, loc=0, scale=1))
    return p_val

def calc_f_value(beta, X, Y):
    p_1 = 1
    p_2 = beta.shape[0] - 1
    n = X.shape[0]
    d_1 = p_2 - p_1
    d_2 = n-p_2
    residuals = Y - beta[0]
    RSS_1 = np.dot(residuals.T, residuals)
    residuals = Y - np.dot(X, beta)
    RSS_2 = np.dot(residuals.T, residuals)
    f_stat = ((RSS_1 - RSS_2) / d_1) / (RSS_2 / d_2)
    f_stat = f_stat.item()
    f_val = 1 - f.cdf(x=f_stat, dfn=d_1, dfd=d_2)
    return f_val

def linear_regression(df, covariate_col, outcome_col, beta_Ho=None):
    X = df[covariate_col].to_numpy()
    X = np.c_[np.ones(X.shape[0]).reshape(-1, 1), X]
    Y = df[outcome_col].to_numpy().reshape(-1, 1)
    dof = X.shape[0] - X.shape[1]
    beta = solve_OLS(X, Y)
    sigma_var = error_estimator(beta, X, Y)
    beta_var = calc_beta_var(sigma_var, X)
    p_val = calc_p_value(beta, beta_var, dof, beta_Ho=beta_Ho)
    f_val = calc_f_value(beta, X, Y)
    return beta, np.sqrt(beta_var), p_val[1:], f_val

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':

    # vol_data = pd.read_csv('..\\Data\\volatility_data.csv')
    vol_data = pd.read_csv('..\\Data\\Implied_volatility.csv')
    vol_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)

    # tickers = [col for col in vol_data.columns if col != 'Timestamp']
    tickers = ['AMC', 'BB', 'GME', 'SPY', 'TSLA']

    volume_data = pd.read_csv('..\\Data\\Volume_data.csv')
    volume_data.rename(columns={'Date': 'Timestamp'}, inplace=True)

    sent_data = pd.read_csv('..\\Data\\UpdatedSubmissions.csv')
    sent_data.rename(columns={'DateTime': 'Timestamp'}, inplace=True)
    sent_data.drop(columns=['index'], inplace=True)
    sent_data['upvotes'] = sent_data['upvote_rate'] * sent_data['score']

    general_features = [
                        'score',
                        'upvote_rate',
                        'put_comments',
                        'buy_comments',
                        'call_comments',
                        'sell_comments',
                        'compound',
                        'mean_NLTK_comments'
                        # 'LM_Positive',
                        # 'LM_Negative',
                        # 'LM_Polarity',
                        # 'LM_Subjectivity',
                        # 'LM_Positive_comments',
                        # 'LM_Negative_comments',
                        # 'LM_Polarity_comments',
                        # 'LM_Subjectivity_comments',
                        # 'upvotes'
    ]

    features_sum = [
                   'put_comments',
                    'buy_comments',
                    'call_comments',
                    'sell_comments'
                   #  'upvotes'
                    ]
    features_mean = [
                     'upvote_rate',
                     'compound',
                     'mean_NLTK_comments',
                     'Volatility'
                     # 'Volume',
                     # 'LM_Positive',
                     # 'LM_Negative',
                     # 'LM_Polarity',
                     # 'LM_Subjectivity',
                     # 'LM_Positive_comments',
                     # 'LM_Negative_comments',
                     # 'LM_Polarity_comments',
                     # 'LM_Subjectivity_comments'
                     ]
    features_max = [
                    'score'
                    # 'upvotes'
                    ]

    df_beta_ols = pd.DataFrame(data=[], columns=tickers, index=general_features + ['ticker_comments'])
    df_beta_sd = pd.DataFrame(data=[], columns=tickers, index=general_features + ['ticker_comments'])
    df_pval = pd.DataFrame(data=[], columns=tickers, index=general_features + ['ticker_comments'] + ['ANOVA'])
    df_tstudent_dof = pd.DataFrame(data=[], columns=tickers, index=[0])

    lag = 0

    for ticker in tickers:

        if ticker in sent_data.columns:

            features = general_features + [ticker+'_comments']

            df = compile_option_data(df_1=sent_data,
                                     df_2=vol_data,
                                     ticker=ticker,
                                     features=features,
                                     new_attr='Volatility',
                                     lag=lag)

            # compiled_volume = compile_option_data(df_1=sent_data,
            #                                       df_2=volume_data,
            #                                       ticker=ticker,
            #                                       features=features,
            #                                       new_attr='Volume',
            #                                       lag=lag)
            # df['Volume'] = compiled_volume['Volume']

            df = fix_features(df=df,
                              features_sum=features_sum+[ticker+'_comments'],
                              features_mean=features_mean,
                              features_max=features_max)

            # df['TIS'] = (df['buy_comments'] + df['call_comments']) / \
            #             (df['buy_comments'] + df['call_comments'] + df['sell_comments'] + df['put_comments'])
            # features = [f for f in features if f not in ['put_comments', 'buy_comments', 'call_comments', 'sell_comments']]
            # features += ['TIS']

            # df['Volatility'] = np.log(df['Volatility'].to_numpy())

            inv = check_invertability(df[features].to_numpy())
            if inv:
                if len(df) > len(features) + 1:
                    beta, beta_sd, p_val, f_val = linear_regression(df, covariate_col=features, outcome_col='Volatility')
                    df_beta_ols[ticker] = beta[1:]
                    df_beta_sd[ticker] = beta_sd[1:]
                    df_pval[ticker] = np.append(p_val, f_val)
                    df_tstudent_dof[ticker] = len(df) - (len(features) + 1)


        df_beta_ols.to_csv('..\\Data\\Output\\Beta_OLS.csv')
        df_beta_sd.to_csv('..\\Data\\Output\\Beta_Sd.csv')
        df_pval.to_csv('..\\Data\\Output\\p_values.csv')
        df_tstudent_dof.to_csv('..\\Data\\Output\\dof_tstudent.csv', index=False)
