import argparse


parser = argparse.ArgumentParser(description='Text Classification')

parser.add_argument("--output-dir", help='Base directory to save outputs of the run', required=True)
parser.add_argument("--data-path", help='Location of the CSV with `text` and `label`', required=True)

args = parser.parse_args()