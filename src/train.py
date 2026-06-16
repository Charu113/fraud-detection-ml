# src/train.py

import time
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocess import (
    preprocess_data,
    split_data,
    apply_smote
)


def train_all_models(X_train, y_train):

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    }

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        start = time.time()

        model.fit(X_train, y_train)

        end = time.time()

        print(f"Completed in {(end-start):.2f} seconds")

        trained_models[name] = model

    joblib.dump(
        trained_models["XGBoost"],
        "models/fraud_model.pkl"
    )

    print("\nModel Saved Successfully!")
    print("models/fraud_model.pkl")

    return trained_models


if __name__ == "__main__":

    print("Starting Training Pipeline...\n")

    df, scaler = preprocess_data(
        "data/creditcard.csv"
    )

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_sm, y_train_sm = apply_smote(
        X_train,
        y_train
    )

    trained_models = train_all_models(
        X_train_sm,
        y_train_sm
    )

    print("\nTraining Completed Successfully!")