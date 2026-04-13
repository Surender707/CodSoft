import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def load_dataset(filepath):
    data = pd.read_csv(filepath)
    print("Dataset loaded successfully.")
    print("Shape:", data.shape)
    print()
    print("First 5 rows:")
    print(data.head())
    print()
    print("Column data types:")
    print(data.dtypes)
    print()
    print("Missing values per column:")
    print(data.isnull().sum())
    print()
    return data


def handle_missing_values(data):
    data["Age"] = data["Age"].fillna(data["Age"].median())

    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    data["Cabin"] = data["Cabin"].fillna("Unknown")

    print("Missing values handled successfully.")
    print("Remaining missing values:")
    print(data.isnull().sum())
    print()
    return data


def feature_engineering(data):
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    data["IsAlone"] = 0
    data.loc[data["FamilySize"] == 1, "IsAlone"] = 1

    data["HasCabin"] = data["Cabin"].apply(lambda x: 0 if x == "Unknown" else 1)

    data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    rare_titles = data["Title"].value_counts()
    rare_titles = rare_titles[rare_titles < 10].index.tolist()
    data["Title"] = data["Title"].replace(rare_titles, "Rare")

    print("Feature engineering completed.")
    print("New columns added: FamilySize, IsAlone, HasCabin, Title")
    print()
    return data


def encode_features(data):
    label_encoders = {}

    categorical_columns = ["Sex", "Embarked", "Title"]

    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
        print("Encoded column:", col, "->", list(le.classes_))

    print()
    return data, label_encoders


def select_features(data):
    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "HasCabin",
        "Title",
    ]

    X = data[feature_columns]
    y = data["Survived"]

    print("Selected features:", feature_columns)
    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
    print()
    return X, y, feature_columns


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("Model training completed.")
    print()
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy: {:.4f}".format(accuracy))
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    return accuracy


def save_model(model, label_encoders, feature_columns, filepath):
    model_data = {
        "model": model,
        "label_encoders": label_encoders,
        "feature_columns": feature_columns,
    }
    joblib.dump(model_data, filepath)
    print("Model and metadata saved to:", filepath)
    print()


def display_feature_importance(model, feature_columns):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False)

    print("Feature Importances:")
    print(feature_importance_df.to_string(index=False))
    print()


def main():
    print("=" * 60)
    print("  Titanic Survival Prediction - Training Pipeline")
    print("=" * 60)
    print()

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Titanic-Dataset.csv")
    data = load_dataset(dataset_path)

    data = handle_missing_values(data)

    data = feature_engineering(data)

    data, label_encoders = encode_features(data)

    X, y, feature_columns = select_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Training set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])
    print()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    display_feature_importance(model, feature_columns)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_model.pkl")
    save_model(model, label_encoders, feature_columns, model_path)

    print("=" * 60)
    print("  Training pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
