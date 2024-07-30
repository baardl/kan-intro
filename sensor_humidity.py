import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Sample data for humidity sensor
sensor_values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 0 for non-humidity, 1 for humidity

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
sensor_values_poly = poly.fit_transform(sensor_values.reshape(-1, 1))

# Model training
model = LogisticRegression()
model.fit(sensor_values_poly, labels)

# Save the model
dump((model, poly), 'sensor_humidity.model')
print('Humidity model trained and saved.')

