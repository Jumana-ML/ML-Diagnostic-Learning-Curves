
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features
        and y is a Series of the target (churned).
    """
    #Load CSV, drop customer_id, separate features and target
    df = pd.read_csv(filepath)
    df.drop(columns="customer_id", inplace=True)
    X = df.drop(columns="churned")
    y = df["churned"]
    return X, y

def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    #Create a ColumnTransformer with StandardScaler for numeric
    #and OneHotEncoder for categorical columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def run_diagnostic():
    x, y = load_and_prepare()
    
    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    train_size = np.linspace(0.1, 1.0, 10)

    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, x, y,
        train_sizes=train_size,
        cv=StratifiedKFold(n_splits=5),
        scoring='f1',
        n_jobs=-1,
        random_state=42
    ) 

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="#E74C3C", label="Training Score", linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#E74C3C")

    plt.plot(train_sizes, test_mean, 'o-', color="#2ECC71", label="Cross-validation Score", linewidth=2)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#2ECC71")

    plt.title("Learning Curves Diagnostic (Logistic Regression)", fontsize=14)
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("learning_diagnostic_plot.png")
    plt.close()

if __name__ == "__main__":
    run_diagnostic()