import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


def load_model_artifacts():
    print("=" * 70)
    print("  CREDIT CARD FRAUD DETECTION - INFERENCE SYSTEM")
    print("=" * 70)
    print()
    print("  Loading model artifacts...")

    try:
        model = joblib.load("fraud_detection_model.pkl")
        scaler_amount = joblib.load("scaler_amount.pkl")
        scaler_time = joblib.load("scaler_time.pkl")
        feature_names = joblib.load("feature_names.pkl")
        print("  Model and scalers loaded successfully.")
        print()
        return model, scaler_amount, scaler_time, feature_names
    except FileNotFoundError as e:
        print(f"  ERROR: Could not load model artifacts - {e}")
        print("  Please run train.py first to train and save the model.")
        sys.exit(1)


def get_transaction_from_user(feature_names):
    print("-" * 70)
    print("  Enter Transaction Details for Fraud Analysis")
    print("-" * 70)
    print()

    transaction = {}

    print("  Enter the Time of the transaction (seconds from first transaction):")
    while True:
        try:
            time_val = float(input("    Time: "))
            transaction["Time"] = time_val
            break
        except ValueError:
            print("    Invalid input. Please enter a numeric value.")

    print()
    print("  Enter the Amount of the transaction:")
    while True:
        try:
            amount_val = float(input("    Amount: "))
            transaction["Amount"] = amount_val
            break
        except ValueError:
            print("    Invalid input. Please enter a numeric value.")

    print()
    print("  Enter the PCA-transformed features (V1 through V28).")
    print("  If you do not have these values, enter 0 for each.")
    print()

    for i in range(1, 29):
        feature_name = f"V{i}"
        while True:
            try:
                val = float(input(f"    {feature_name}: "))
                transaction[feature_name] = val
                break
            except ValueError:
                print("    Invalid input. Please enter a numeric value.")

    return transaction


def preprocess_transaction(transaction, scaler_amount, scaler_time, feature_names):
    amount_scaled = scaler_amount.transform([[transaction["Amount"]]])[0][0]
    time_scaled = scaler_time.transform([[transaction["Time"]]])[0][0]

    processed = {}
    for feat in feature_names:
        if feat == "Amount_Scaled":
            processed[feat] = amount_scaled
        elif feat == "Time_Scaled":
            processed[feat] = time_scaled
        else:
            processed[feat] = transaction.get(feat, 0.0)

    df = pd.DataFrame([processed])
    df = df[feature_names]
    return df


def predict_fraud(model, transaction_df):
    prediction = model.predict(transaction_df)[0]
    probability = model.predict_proba(transaction_df)[0]

    return prediction, probability


def display_result(prediction, probability, transaction):
    print()
    print("=" * 70)
    print("  FRAUD DETECTION RESULT")
    print("=" * 70)
    print()

    if prediction == 1:
        print("  *** ALERT: FRAUDULENT TRANSACTION DETECTED ***")
    else:
        print("  Transaction appears GENUINE.")

    print()
    print(f"  Transaction Amount: ${transaction['Amount']:.2f}")
    print(f"  Transaction Time:   {transaction['Time']:.0f} seconds")
    print()
    print("  Prediction Probabilities:")
    print(f"    Genuine probability:    {probability[0] * 100:.2f}%")
    print(f"    Fraudulent probability: {probability[1] * 100:.2f}%")
    print()
    print(f"  Confidence Level: {max(probability) * 100:.2f}%")
    print()

    risk_level = "LOW"
    if probability[1] > 0.3:
        risk_level = "MEDIUM"
    if probability[1] > 0.6:
        risk_level = "HIGH"
    if probability[1] > 0.85:
        risk_level = "CRITICAL"

    print(f"  Risk Assessment: {risk_level}")
    print("=" * 70)


def evaluate_csv(model, scaler_amount, scaler_time, feature_names, filepath):
    print()
    print(f"  Loading transactions from: {filepath}")

    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"  ERROR: File '{filepath}' not found.")
        return

    print(f"  Loaded {len(data)} transactions.")
    print()

    if "Amount" in data.columns and "Time" in data.columns:
        data["Amount_Scaled"] = scaler_amount.transform(data[["Amount"]])
        data["Time_Scaled"] = scaler_time.transform(data[["Time"]])
        data = data.drop(columns=["Amount", "Time"])

    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        print(f"  WARNING: Missing features: {missing_features}")
        for feat in missing_features:
            data[feat] = 0.0

    data = data[feature_names]

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    fraud_count = sum(predictions == 1)
    genuine_count = sum(predictions == 0)

    print("  " + "-" * 50)
    print("  BATCH EVALUATION RESULTS")
    print("  " + "-" * 50)
    print(f"  Total transactions analyzed: {len(predictions)}")
    print(f"  Genuine transactions:        {genuine_count}")
    print(f"  Fraudulent transactions:     {fraud_count}")
    print(f"  Fraud rate:                  {fraud_count / len(predictions) * 100:.2f}%")
    print("  " + "-" * 50)

    if fraud_count > 0:
        print()
        print("  Flagged Fraudulent Transactions:")
        print(f"  {'Index':<10} {'Fraud Probability':<20} {'Risk Level':<15}")
        print("  " + "-" * 45)
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1:
                risk = "LOW"
                if prob[1] > 0.3:
                    risk = "MEDIUM"
                if prob[1] > 0.6:
                    risk = "HIGH"
                if prob[1] > 0.85:
                    risk = "CRITICAL"
                print(f"  {idx:<10} {prob[1] * 100:<20.2f}% {risk:<15}")

    print()


def main():
    model, scaler_amount, scaler_time, feature_names = load_model_artifacts()

    while True:
        print("-" * 70)
        print("  Select an option:")
        print("    1. Evaluate a single transaction (manual input)")
        print("    2. Evaluate transactions from a CSV file")
        print("    3. Exit")
        print("-" * 70)

        choice = input("  Enter your choice (1/2/3): ").strip()

        if choice == "1":
            print()
            transaction = get_transaction_from_user(feature_names)
            transaction_df = preprocess_transaction(
                transaction, scaler_amount, scaler_time, feature_names
            )
            prediction, probability = predict_fraud(model, transaction_df)
            display_result(prediction, probability, transaction)
            print()

        elif choice == "2":
            print()
            filepath = input("  Enter the path to the CSV file: ").strip()
            evaluate_csv(model, scaler_amount, scaler_time, feature_names, filepath)

        elif choice == "3":
            print()
            print("  Exiting Fraud Detection System. Goodbye.")
            print()
            break

        else:
            print()
            print("  Invalid choice. Please enter 1, 2, or 3.")
            print()


if __name__ == "__main__":
    main()
