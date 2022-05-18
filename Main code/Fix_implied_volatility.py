import pandas as pd
import numpy as np

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

df_raw = pd.read_csv('..\\Data\\OrganizedImpliedVolatility.csv')
df = df_raw[['Date']+[col for col in df_raw.columns if 'implied' in col.lower()]]
df = df.rename(columns={'Date': ''})
df = df.rename(columns={col: col.split('_')[-1] for col in df.columns if col not in ['Timestamp']})
df = df.iloc[::-1]
df.reset_index(inplace=True, drop=True)
df[df[[col for col in df.columns if col not in ['Timestamp']]]==' '] = np.NaN
df = df.fillna(method='ffill')
df.to_csv('..\\Data\\Implied_volatility.csv', index=False)
