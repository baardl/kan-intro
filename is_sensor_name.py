import argparse

def is_text(s):
    try:
        float(s)
        return False
    except ValueError:
        return True

# Argument parser setup
parser = argparse.ArgumentParser(description='Check if the input is text.')
parser.add_argument('input_value', help='The input value to check')
args = parser.parse_args()

if is_text(args.input_value):
    print(f'{args.input_value} is text.')
else:
    print(f'{args.input_value} is not text.')

