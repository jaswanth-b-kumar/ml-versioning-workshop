import sys
from pathlib import Path

import pandas as pd

TRAIN_PATH = Path('data/processed/train.csv')

EXPECTED_COLUMNS = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)',
    'target',
]

def main():
    if not TRAIN_PATH.exists():
        print(f"Error: {TRAIN_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(TRAIN_PATH)

    cols = list(df.columns)
    if cols != EXPECTED_COLUMNS:
        print(f"Error: Columns do not match expected columns.\nExpected: {EXPECTED_COLUMNS}\nFound: {cols}", file=sys.stderr)
        sys.exit(1)

    print("Data validation passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
