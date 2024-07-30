import argparse

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Argument parser setup
parser = argparse.ArgumentParser(description='Check if the input is a number.')
parser.add_argument('input_value', help='The input value to check')
args = parser.parse_args()

if is_number(args.input_value):
    print(f'{args.input_value} is a number.')
else:
    print(f'{args.input_value} is not a number.')

