import argparse
from joblib import load
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_text(s):
    try:
        float(s)
        return False
    except ValueError:
        return True

def load_model(model_path):
    model, poly = load(model_path)
    return model, poly

def predict_probability(model, poly, value):
    value_poly = poly.transform([[value]])
    probability = model.predict_proba(value_poly)
    return probability

def main():
    parser = argparse.ArgumentParser(description='Identify sensor type and value.')
    parser.add_argument('input_value', help='The input value to check')
    args = parser.parse_args()

    input_value = args.input_value

    if is_number(input_value):
        value = float(input_value)
        print(f'{value} is a valid sensor value.')

        model_co2, poly_co2 = load_model('sensor_co2.model')
        model_humidity, poly_humidity = load_model('sensor_humidity.model')

        prob_co2 = predict_probability(model_co2, poly_co2, value)
        prob_humidity = predict_probability(model_humidity, poly_humidity, value)

        threshold = 0.5
        co2_prob = prob_co2[0][1]
        humidity_prob = prob_humidity[0][1]

        if co2_prob > threshold or humidity_prob > threshold:
            if co2_prob > humidity_prob:
                print(f'The input value {value} is more likely a CO2 sensor value with probability {co2_prob:.2f}.')
            else:
                print(f'The input value {value} is more likely a Humidity sensor value with probability {humidity_prob:.2f}.')
        else:
            print(f'The input value {value} does not match any known sensor types with sufficient probability (threshold: {threshold}).')
    elif is_text(input_value):
        print(f'{input_value} is text. This might be a sensor name.')
    else:
        print(f'{input_value} is not recognized as a valid input.')

if __name__ == "__main__":
    main()
