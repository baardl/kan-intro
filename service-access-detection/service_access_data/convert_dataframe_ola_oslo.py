

import pandas as pd
from datetime import timedelta

# Les CSV-filen inn i en DataFrame
# df = pd.read_csv('./login_data_backup.csv')
# df.loc[(df['user'] == 'Ola') & (df['service'] == 'brukeranmeldelser') & (df['geography'] != 'oslo'), 'geography'] = 'oslo'
# df.to_csv('login_data_ola_oslo.csv', index=False)
# print(df.head())


df = pd.read_csv('login_data_weekday.csv')
df.loc[(df['user'] == 'Ola') & (df['service'] == 'brukeranmeldelser') & (df['weekday'] != 'Sunday'), 'weekday'] = 'Sunday'
df.to_csv('login_data_ola_oslo_sunday.csv', index=False)
print(df.head())

