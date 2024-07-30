import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Predict CO2 sensor value.')
parser.add_argument('prediction_value', type=float, help='Sensor value to predict CO2 label for')
args = parser.parse_args()

# Sample data
sensor_values = np.array([100, 200, 300, 350, 400, 500, 600, 700, 800, 900, 1000])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 0 for non-CO2, 1 for CO2

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
sensor_values_poly = poly.fit_transform(sensor_values.reshape(-1, 1))

# Model training
model = LinearRegression()
model.fit(sensor_values_poly, labels)

# Predictions
prediction_value = args.prediction_value
prediction_value_poly = poly.transform([[prediction_value]])
prediction = model.predict(prediction_value_poly)

print(f'Predicted label for sensor value {prediction_value}: {prediction[0]}')
