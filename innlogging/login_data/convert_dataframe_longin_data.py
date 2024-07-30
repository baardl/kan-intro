

import pandas as pd
from datetime import timedelta

# Les CSV-filen inn i en DataFrame
df = pd.read_csv('login_data.csv')



# Konverter 'date' kolonnen til datetime format
df['date'] = pd.to_datetime(df['date'])

# Legg til en ny kolonne 'weekday' som representerer ukedagen for hver dato som tekst
# df['weekday'] = df['date'].dt.day_name(locale='nb_NO')
df['weekday'] = df['date'].dt.day_name()
# Fjern 'date' og 'time' kolonnene fra DataFrame
df = df.drop(columns=['date', 'time'])

print(df.head())

# Skriv den oppdaterte DataFrame til en ny CSV-fil
df.to_csv('login_data_weekday.csv', index=False)