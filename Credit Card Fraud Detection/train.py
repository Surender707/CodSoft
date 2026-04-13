import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings("ignore")


def load_data(filepath):
    print("=" * 70)
    print("  CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    print()
    print("[1/6] Loading dataset...")
    data = pd.read_csv(filepath)
    print(f"  Dataset loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"  Columns: {list(data.columns)}")
    return data


def explore_data(data):
    print()
    print("[2/6] Exploring data distribution...")
    print()
    print("  Class Distribution:")
    class_counts = data["Class"].value_counts()
    total = len(data)
    for label, count in class_counts.items():
        label_name = "Genuine" if label == 0 else "Fraudulent"
        percentage = (count / total) * 100
        print(f"    {label_name} (Class {label}): {count} transactions ({percentage:.4f}%)")

    print()
    print("  Dataset Statistics:")
    print(f"    Missing values: {data.isnull().sum().sum()}")
    print(f"    Duplicate rows: {data.duplicated().sum()}")
    fraud_ratio = class_counts.get(1, 0) / class_counts.get(0, 1)
    print(f"    Fraud-to-Genuine ratio: 1:{int(1/fraud_ratio) if fraud_ratio > 0 else 'N/A'}")
    return data


def preprocess_data(data):
    print()
    print("[3/6] Preprocessing and normalizing data...")

    data = data.drop_duplicates()
    print(f"  After removing duplicates: {data.shape[0]} rows")

    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()

    data["Amount_Scaled"] = scaler_amount.fit_transform(data[["Amount"]])
    data["Time_Scaled"] = scaler_time.fit_transform(data[["Time"]])

    data = data.drop(columns=["Amount", "Time"])

    print("  Normalized 'Amount' and 'Time' features using StandardScaler")
    print(f"  Final feature count: {data.shape[1] - 1}")

    joblib.dump(scaler_amount, "scaler_amount.pkl")
    joblib.dump(scaler_time, "scaler_time.pkl")
    print("  Saved scalers: scaler_amount.pkl, scaler_time.pkl")

    return data


def handle_imbalance(X_train, y_train):
    print()
    print("[4/6] Handling class imbalance with SMOTE oversampling...")
    print(f"  Before SMOTE:")
    print(f"    Genuine: {sum(y_train == 0)}")
    print(f"    Fraudulent: {sum(y_train == 1)}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE:")
    print(f"    Genuine: {sum(y_resampled == 0)}")
    print(f"    Fraudulent: {sum(y_resampled == 1)}")

    return X_resampled, y_resampled


def split_data(data):
    print()
    print("[4.5/6] Splitting data into train/test sets...")

    X = data.drop(columns=["Class"])
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set:  {X_test.shape[0]} samples")
    print(f"  Train fraud ratio: {sum(y_train == 1) / len(y_train) * 100:.4f}%")
    print(f"  Test fraud ratio:  {sum(y_test == 1) / len(y_test) * 100:.4f}%")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print()
    print("[5/6] Training Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    print("  Model training complete.")
    print(f"  Number of trees: {model.n_estimators}")
    print(f"  Max depth: {model.max_depth}")

    return model


def evaluate_model(model, X_test, y_test):
    print()
    print("[6/6] Evaluating model performance...")
    print()

    y_pred = model.predict(X_test)

    precision_genuine = precision_score(y_test, y_pred, pos_label=0)
    recall_genuine = recall_score(y_test, y_pred, pos_label=0)
    f1_genuine = f1_score(y_test, y_pred, pos_label=0)

    precision_fraud = precision_score(y_test, y_pred, pos_label=1)
    recall_fraud = recall_score(y_test, y_pred, pos_label=1)
    f1_fraud = f1_score(y_test, y_pred, pos_label=1)

    print("  " + "-" * 55)
    print(f"  {'Metric':<20} {'Genuine':>15} {'Fraudulent':>15}")
    print("  " + "-" * 55)
    print(f"  {'Precision':<20} {precision_genuine:>15.4f} {precision_fraud:>15.4f}")
    print(f"  {'Recall':<20} {recall_genuine:>15.4f} {recall_fraud:>15.4f}")
    print(f"  {'F1-Score':<20} {f1_genuine:>15.4f} {f1_fraud:>15.4f}")
    print("  " + "-" * 55)
    print()

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"    True Negatives  (Genuine predicted Genuine):     {cm[0][0]}")
    print(f"    False Positives (Genuine predicted Fraudulent):  {cm[0][1]}")
    print(f"    False Negatives (Fraud predicted Genuine):       {cm[1][0]}")
    print(f"    True Positives  (Fraud predicted Fraudulent):    {cm[1][1]}")
    print()

    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    print(f"  Overall Accuracy: {accuracy * 100:.2f}%")


def save_model(model, feature_names):
    print()
    print("  Saving model and metadata...")
    joblib.dump(model, "fraud_detection_model.pkl")
    joblib.dump(feature_names, "feature_names.pkl")
    print("  Saved: fraud_detection_model.pkl")
    print("  Saved: feature_names.pkl")
    print()
    print("=" * 70)
    print("  Training pipeline complete. Model is ready for deployment.")
    print("=" * 70)


def main():
    filepath = "creditcard.csv"

    try:
        data = load_data(filepath)
    except FileNotFoundError:
        print(f"ERROR: File '{filepath}' not found.")
        print("Please place the creditcard.csv dataset in the current directory.")
        sys.exit(1)

    data = explore_data(data)
    data = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(data)

    feature_names = list(X_train.columns)

    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)

    model = train_model(X_train_balanced, y_train_balanced)

    evaluate_model(model, X_test, y_test)

    save_model(model, feature_names)


if __name__ == "__main__":
    main()
