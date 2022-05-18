import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t

import matplotlib
matplotlib.use("TkAgg")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

df_beta_ols = pd.read_csv('..\\Data\\Output\\Beta_OLS.csv', index_col='Unnamed: 0')
df_beta_sd = pd.read_csv('..\\Data\\Output\\Beta_Sd.csv', index_col='Unnamed: 0')
df_pval = pd.read_csv('..\\Data\\Output\\p_values.csv', index_col='Unnamed: 0')
df_tstudent_dof = pd.read_csv('..\\Data\\Output\\dof_tstudent.csv')
df_tstudent_dof = df_tstudent_dof.reset_index(drop=True)

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

sns.heatmap(df_pval)
plt.title('p-value per ticker and covariate', fontsize=16)
plt.xlabel('Ticker', fontsize=14)
plt.ylabel('Covariate', fontsize=14)
# fig = plt.gcf()
# fig.set_size_inches((6, 15), forward=False)
# fig.savefig('..\\Data\\Output\\p_val_heatmap.jpg', dpi=500)
plt.savefig('..\\Data\\Output\\p_val_heatmap.png', bbox_inches='tight')

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

fig, axs = plt.subplots(df_beta_ols.shape[0], df_beta_ols.shape[1])
plt.suptitle('Distribution of regression coefficients per ticker and covariate\n'
             '(coefficient value of zero indicated with a red line)', fontsize=12)
for j, ticker in enumerate(df_beta_ols.columns):
    dof = df_tstudent_dof[ticker]
    for i, covariate in enumerate(df_beta_ols.index):
        ax = axs[i, j]
        mu = df_beta_ols.loc[covariate, ticker]
        sd = df_beta_sd.loc[covariate, ticker]
        x = np.linspace(mu-6*sd, mu+6*sd, 300)
        # y = norm.pdf(x, loc=mu, scale=sd)
        y = t.pdf(x, loc=mu, scale=sd, df=dof)

        ax.plot(x, y)
        ax.axvline(0, color='r')
        ax.set_xlabel(ticker, fontsize=14)
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

        if i < axs.shape[0] - 1:
            ax.axes.get_xaxis().set_visible(False)
        if j > 0:
            ax.axes.get_yaxis().set_visible(False)
plt.savefig('..\\Data\\Output\\beta_dist.png', bbox_inches='tight')


