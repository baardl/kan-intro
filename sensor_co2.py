# Logistic Regression brukes for klassifikasjonsproblemer, hvor målet er å predikere hvilken kategori (klasse) en observasjon tilhører.

#Hovedpunkter:
#Formål: Predikere en kategorisk avhengig variabel (binær eller flerkategorisk) basert på en eller flere uavhengige variabler.

#Modell: Modellen bruker en logistisk funksjon (sigmoid-funksjon) for å kartlegge den lineære kombinasjonen av uavhengige variabler til en sannsynlighet.
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Sample data for CO2 sensor
sensor_values = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 0 for non-CO2, 1 for CO2

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
sensor_values_poly = poly.fit_transform(sensor_values.reshape(-1, 1))

# Model training
model = LogisticRegression()
model.fit(sensor_values_poly, labels)

# Save the model
dump((model, poly), 'sensor_co2.model')
print('CO2 model trained and saved.')


