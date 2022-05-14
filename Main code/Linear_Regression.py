import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from generate_features import compile_option_data

import matplotlib
matplotlib.use("TkAgg")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def solve_ols(X, Y):
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return beta

def error_estimator(beta, X, Y):
    residuals = Y - np.dot(X, beta)
    residual_sq_sum = np.dot(residuals.T, residuals)
    sigma_var = residual_sq_sum.item() / (X.shape[0] - X.shape[1])
    return sigma_var

def calc_beta_var(sigma_var, X):
    beta_covar = sigma_var * np.linalg.inv(np.dot(X.T, X))
    return np.diag(beta_covar)

def calc_p_value(beta, beta_var, dof, beta_Ho=None):
    if beta_Ho==None:
        beta_Ho = np.zeros(beta.shape[0]).reshape(-1, 1)
    p_val = np.repeat(1.0, repeats=beta.shape[0])
    for i, (b, b_Ho, b_var) in enumerate(zip(beta, beta_Ho, beta_var)):
        T_obs = (b - b_Ho) / np.sqrt(b_var)
        p_val[i] = 2 * (1 - t.cdf(x=np.abs(T_obs), df=dof, loc=0, scale=1))
    return p_val

def regression_analysis(df, covariate_col, outcome_col, beta_Ho=None):
    X = df[covariate_col].to_numpy()
    X = np.c_[np.ones(X.shape[0]).reshape(-1, 1), X]
    Y = df[outcome_col].to_numpy().reshape(-1, 1)
    dof = X.shape[0] - X.shape[1]
    beta = solve_ols(X, Y)
    sigma_var = error_estimator(beta, X, Y)
    beta_var = calc_beta_var(sigma_var, X)
    p_val = calc_p_value(beta, beta_var, dof, beta_Ho=beta_Ho)
    return p_val

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vol_data = pd.read_csv('..\\Data\\volatility_data.csv')
vol_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
tickers = [col for col in vol_data.columns if col != 'Timestamp']

sent_data = pd.read_csv('..\\Data\\UpdatedSubmissions.csv')
sent_data.rename(columns={'DateTime': 'Timestamp'}, inplace=True)
sent_data.drop(columns=['index'], inplace=True)


general_features = ['score', 'upvote_rate', 'put_comments', 'buy_comments', 'call_comments',
                    'sell_comments', 'compound', 'mean_NLTK_comments']

df_pval = pd.DataFrame(data=[], columns=tickers, index=['intercept'] + general_features + ['ticker_comments'])

for ticker in tickers:

    if ticker in sent_data.columns:

        features = general_features + [ticker+'_comments']

        df = compile_option_data(sent_data=sent_data, vol_data=vol_data, ticker=ticker, features=features)

        df = df.groupby(['Timestamp']).sum()

        if len(df) > len(features) + 1:
            p_val = regression_analysis(df, covariate_col=features, outcome_col='Volatility')
            df_pval[ticker] = p_val