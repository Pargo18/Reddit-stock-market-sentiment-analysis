import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import Visualizations as vs

plt.style.use('ggplot')

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def plots(df, name='Differences RV-IV'):  # provide also relevant names for the plots
    # time series
    df.plot(subplots=True, figsize=(16, 12), title='Time series of ' + name)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel('Days of 2021', fontsize=14)

    # correlation plot between tickers
    corrmat = df.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(corrmat, vmax=1., square=True, cmap="rocket_r")
    plt.title('Correlation plot between tickers of ' + name, fontsize=15)
    plt.show()

    # autocorrelation plot of tickers
    for ticker in [col for col in df.columns if col != 'Timestamp']:
        pd.plotting.autocorrelation_plot(df[ticker], label=ticker)
        plt.title('Autocorrelation plot on ' + name)

def diff_return_plots(df_diff, return_data):  # identical Timestamp first
    for ticker in [col for col in df_diff.columns if col != 'Timestamp']:
        fig, axs = plt.subplots(2)
        fig.suptitle(f'Diference-Return correlation: {np.corrcoef(df_diff[ticker], return_data[ticker])[0][1]:.2f}')
        axs[0].plot(df_diff.Timestamp, df_diff[ticker])
        axs[0].set_title('Volatility difference of '+ticker)
        axs[1].plot(return_data.Timestamp, return_data[ticker])
        axs[1].set_title('Return of '+ticker)
        plt.tight_layout()