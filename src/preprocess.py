# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib


# =====================================================
# Step 1: Load Data & Feature Engineering
# =====================================================

def preprocess_data(filepath="data/creditcard.csv"):

    # Load dataset
    df = pd.read_csv(filepath)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    print("Dataset Shape After Removing Duplicates:")
    print(df.shape)

    # -------------------------------------
    # Feature Engineering
    # -------------------------------------

    # Convert Time into Hour
    df["Hour"] = (df["Time"] / 3600) % 24

    # Log transform Amount
    df["Log_Amount"] = np.log1p(df["Amount"])

    # Remove original columns
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    # -------------------------------------
    # Feature Scaling
    # -------------------------------------

    scaler = StandardScaler()

    cols_to_scale = ["Hour", "Log_Amount"]

    df[cols_to_scale] = scaler.fit_transform(
        df[cols_to_scale]
    )

    # Save scaler for Streamlit app
    joblib.dump(scaler, "models/scaler.pkl")

    print("Scaler saved successfully!")

    return df, scaler


# =====================================================
# Step 2: Train Test Split
# =====================================================

def split_data(df):

    X = df.drop("Class", axis=1)

    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain Test Split Completed")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    return X_train, X_test, y_train, y_test


# =====================================================
# Step 3: Handle Class Imbalance using SMOTE
# =====================================================

def apply_smote(X_train, y_train):

    print("\nBefore SMOTE")
    print("Fraud Transactions:", y_train.sum())
    print("Legitimate Transactions:", (y_train == 0).sum())

    smote = SMOTE(
        random_state=42,
        k_neighbors=5
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE")
    print("Fraud Transactions:", y_resampled.sum())
    print("Legitimate Transactions:", (y_resampled == 0).sum())

    return X_resampled, y_resampled


# =====================================================
# Main Execution
# =====================================================

if __name__ == "__main__":

    print("Starting Preprocessing Pipeline...\n")

    df, scaler = preprocess_data(
        "data/creditcard.csv"
    )

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_sm, y_train_sm = apply_smote(
        X_train,
        y_train
    )

    print("\nPreprocessing Complete Successfully!")