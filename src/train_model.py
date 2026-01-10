import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import subprocess

TRAIN_PATH = Path('data/processed/train.csv')
TEST_PATH = Path('data/processed/test.csv')
MODELS_DIR = Path('models')
METRICS_DIR = Path('metrics')

def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip().decode('utf-8')
    except Exception:
        commit_hash = "unknown"
    return commit_hash

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)   

    print(f"Accuracy: {accuracy}")

    model_path = MODELS_DIR / 'iris_logistic_regression.pkl'
    joblib.dump(model, model_path)

    metrics = {
        'accuracy': accuracy,  
        'git_commit_hash': get_git_commit_hash()
    }

    metrics_path = METRICS_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Git commit hash: {metrics['git_commit_hash']}")
    print("Training complete.")
    print("accuracy:", accuracy)

if __name__ == "__main__":
    main()