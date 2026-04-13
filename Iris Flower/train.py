import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = data[feature_columns]
    y = data["species"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )

    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)
    print("Classification Report:")
    print(report)
    print("=" * 50)

def save_model(model, label_encoder, model_path, encoder_path):
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {encoder_path}")

def main():
    print("=" * 50)
    print("IRIS FLOWER CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 50)

    print("\n[1] Loading dataset...")
    data = load_data("IRIS.csv")
    print(f"    Dataset shape: {data.shape}")
    print(f"    Species found: {list(data['species'].unique())}")

    print("\n[2] Preprocessing data...")
    X, y, label_encoder = preprocess_data(data)
    print(f"    Features shape: {X.shape}")
    print(f"    Labels encoded: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    print("\n[3] Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Testing samples: {X_test.shape[0]}")

    print("\n[4] Training Random Forest Classifier...")
    model = train_model(X_train, y_train)
    print("    Training complete!")

    print("\n[5] Evaluating model...")
    evaluate_model(model, X_test, y_test, label_encoder)

    print("\n[6] Saving model and encoder...")
    save_model(model, label_encoder, "iris_model.pkl", "label_encoder.pkl")

    print("\nTraining pipeline finished successfully!")
    print("You can now run main.py to make predictions.")

if __name__ == "__main__":
    main()
