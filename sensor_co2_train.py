# Logistic Regression brukes for klassifikasjonsproblemer, hvor målet er å predikere hvilken kategori (klasse) en observasjon tilhører.

#Hovedpunkter:
#Formål: Predikere en kategorisk avhengig variabel (binær eller flerkategorisk) basert på en eller flere uavhengige variabler.

#Modell: Modellen bruker en logistisk funksjon (sigmoid-funksjon) for å kartlegge den lineære kombinasjonen av uavhengige variabler til en sannsynlighet.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import argparse
import os

# Argument parser setup
parser = argparse.ArgumentParser(description='Predict CO2 sensor value.')
parser.add_argument('prediction_value', type=float, help='Sensor value to predict CO2 label for')
parser.add_argument('--train', action='store_true', help='Train the model')
args = parser.parse_args()

# File to save the model
model_file = 'co2_model.joblib'

if args.train or not os.path.exists(model_file):
    # Sample data
    sensor_values = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000])
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])  # 0 for non-CO2, 1 for CO2

    # Polynomial transformation
    poly = PolynomialFeatures(degree=2)
    sensor_values_poly = poly.fit_transform(sensor_values.reshape(-1, 1))

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(sensor_values_poly, labels)

    # Save the model
    dump((model, poly), model_file)
    print('Model trained and saved.')
else:
    # Load the model
    model, poly = load(model_file)
    print('Model loaded from file.')

# Predictions
prediction_value = args.prediction_value
prediction_value_poly = poly.transform([[prediction_value]])
prediction = model.predict(prediction_value_poly)

print(f'Predicted label for sensor value {prediction_value}: {prediction[0]}')

# Plotting
if args.train:
    plt.scatter(sensor_values, labels, color='red')
    plt.plot(sensor_values, model.predict(sensor_values_poly), color='blue')
    plt.xlabel('Sensor Value')
    plt.ylabel('Label')
    plt.title('Polynomial Regression for CO2 Sensor')
    plt.show()

