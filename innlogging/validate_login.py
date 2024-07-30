import logging
import pickle
import sys

import dill
import torch
from joblib import load
from kan import KAN, ckpt
logging.basicConfig(level=logging.DEBUG)
# Write logging config to stout
logging.basicConfig(stream=sys.stdout)

# Definer mappings for kategoriske variabler
geography_mapping = {'Oslo': 0, 'Bergen': 1, 'Trondheim': 2, 'Stavanger': 3, 'Kristiansand': 4}
service_mapping = {'brukeranmeldelser': 0, 'profil': 1, 'meldinger': 2, 'innstillinger': 3}
user_mapping = {'Ola': 0, 'Per': 1, 'Kari': 2, 'Anne': 3, 'Nils': 4}

# Funksjon for å konvertere input
def convert_input(date, hour, geography, service, user):
    geography_num = geography_mapping[geography]
    service_num = service_mapping[service]
    user_num = user_mapping[user]
    return [hour, geography_num, service_num, user_num]

# Eksempel på bruk av konverteringsfunksjonen
date = '2023-01-07'  # en lørdag
hour = 14  # klokka 14
geography = 'Stavanger'
service = 'brukeranmeldelser'
user = 'Nils'

input_data = convert_input(date, hour, geography, service, user)
print(f'Konvertert input data: {input_data}')

# Last inn den trente modellen
# model = dill.load(open('./validate_login.model','rb'))
model = ckpt.loadckpt()
logging.debug("Model is loaded")
logging.debug("Try to prone the model")
# model.prune()
logging.debug("Model is pruned")
# Konverter inputdata til PyTorch tensor
x_input = torch.tensor([input_data], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

# Gjør prediksjon
# model.eval()  # Sett modellen i evalueringsmodus
with torch.no_grad():  # Deaktiver gradientberegning
    output = model(x_input)

# Legg til epsilon stability
epsilon = 1e-8  # small constant
output += epsilon

# Konverter output til en liste
prediction = output.tolist()

print(f'Prediction: {prediction}')
