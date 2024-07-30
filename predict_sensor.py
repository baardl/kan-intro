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

def predict(model, poly, value):
    value_poly = poly.transform([[value]])
    return model.predict(value_poly)

def main():
    parser = argparse.ArgumentParser(description='Identify sensor type and value.')
    parser.add_argument('input_value', help='The input value to check')
    args = parser.parse_args()

    input_value = args.input_value

    if is_number(input_value):
        value = float(input_value)
        if 0 <= value <= 4000:
            print(f'{value} is a valid sensor value.')
            
            model_co2, poly_co2 = load_model('co2_model.joblib')
            model_humidity, poly_humidity = load_model('humidity_model.joblib')

            prediction_co2 = predict(model_co2, poly_co2, value)
            prediction_humidity = predict(model_humidity, poly_humidity, value)

            if prediction_co2[0] > prediction_humidity[0]:
                print(f'The input value {value} is more likely a CO2 sensor value.')
            else:
                print(f'The input value {value} is more likely a Humidity sensor value.')
        else:
            print(f'{value} is out of the valid range (0-4000).')
    elif is_text(input_value):
        print(f'{input_value} is text. This might be a sensor name.')
    else:
        print(f'{input_value} is not recognized as a valid input.')

if __name__ == "__main__":
    main()

