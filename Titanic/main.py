import pandas as pd
import numpy as np
import joblib
import os
import sys


def load_model(filepath):
    if not os.path.exists(filepath):
        print("Error: Model file not found at:", filepath)
        print("Please run train.py first to train and save the model.")
        sys.exit(1)

    model_data = joblib.load(filepath)
    print("Model loaded successfully from:", filepath)
    print()
    return model_data


def extract_title(name):
    import re
    title_search = re.search(r" ([A-Za-z]+)\.", name)
    if title_search:
        return title_search.group(1)
    return "Unknown"


def get_passenger_input():
    print("-" * 50)
    print("  Enter Passenger Details")
    print("-" * 50)
    print()

    name = input("Passenger Name (e.g., Smith, Mr. John): ").strip()
    if not name:
        name = "Unknown, Mr. Unknown"

    while True:
        try:
            pclass = int(input("Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd): ").strip())
            if pclass in [1, 2, 3]:
                break
            print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number (1, 2, or 3).")

    while True:
        sex = input("Sex (male/female): ").strip().lower()
        if sex in ["male", "female"]:
            break
        print("Please enter 'male' or 'female'.")

    while True:
        try:
            age = float(input("Age: ").strip())
            if 0 <= age <= 120:
                break
            print("Please enter a valid age between 0 and 120.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            sibsp = int(input("Number of Siblings/Spouses aboard: ").strip())
            if sibsp >= 0:
                break
            print("Please enter a non-negative number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            parch = int(input("Number of Parents/Children aboard: ").strip())
            if parch >= 0:
                break
            print("Please enter a non-negative number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            fare = float(input("Fare paid: ").strip())
            if fare >= 0:
                break
            print("Please enter a non-negative number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        embarked = input("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): ").strip().upper()
        if embarked in ["C", "Q", "S"]:
            break
        print("Please enter C, Q, or S.")

    cabin = input("Cabin number (press Enter if unknown): ").strip()
    if not cabin:
        cabin = "Unknown"

    passenger = {
        "Name": name,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Cabin": cabin,
    }

    print()
    return passenger


def preprocess_passenger(passenger, label_encoders):
    family_size = passenger["SibSp"] + passenger["Parch"] + 1

    is_alone = 1 if family_size == 1 else 0

    has_cabin = 0 if passenger["Cabin"] == "Unknown" else 1

    title = extract_title(passenger["Name"])

    title_encoder = label_encoders["Title"]
    known_titles = list(title_encoder.classes_)
    if title not in known_titles:
        if "Rare" in known_titles:
            title = "Rare"
        else:
            title = known_titles[0]

    sex_encoded = label_encoders["Sex"].transform([passenger["Sex"]])[0]
    embarked_encoded = label_encoders["Embarked"].transform([passenger["Embarked"]])[0]
    title_encoded = label_encoders["Title"].transform([title])[0]

    features = pd.DataFrame([{
        "Pclass": passenger["Pclass"],
        "Sex": sex_encoded,
        "Age": passenger["Age"],
        "SibSp": passenger["SibSp"],
        "Parch": passenger["Parch"],
        "Fare": passenger["Fare"],
        "Embarked": embarked_encoded,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "HasCabin": has_cabin,
        "Title": title_encoded,
    }])

    return features


def predict_survival(model, features):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return prediction, probability


def display_result(passenger, prediction, probability):
    print("=" * 50)
    print("  Prediction Result")
    print("=" * 50)
    print()
    print("Passenger Name:", passenger["Name"])
    print("Ticket Class:", passenger["Pclass"])
    print("Sex:", passenger["Sex"].capitalize())
    print("Age:", passenger["Age"])
    print("Fare:", passenger["Fare"])
    print("Port of Embarkation:", passenger["Embarked"])
    print()

    if prediction == 1:
        print("Prediction: SURVIVED")
    else:
        print("Prediction: DID NOT SURVIVE")

    print()
    print("Survival Probability: {:.2f}%".format(probability[1] * 100))
    print("Non-Survival Probability: {:.2f}%".format(probability[0] * 100))
    print()
    print("=" * 50)


def main():
    print()
    print("=" * 60)
    print("  Titanic Survival Prediction - Prediction Engine")
    print("=" * 60)
    print()

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_model.pkl")
    model_data = load_model(model_path)

    model = model_data["model"]
    label_encoders = model_data["label_encoders"]
    feature_columns = model_data["feature_columns"]

    print("Feature columns used by model:", feature_columns)
    print()

    while True:
        passenger = get_passenger_input()

        features = preprocess_passenger(passenger, label_encoders)

        prediction, probability = predict_survival(model, features)

        display_result(passenger, prediction, probability)

        again = input("Would you like to predict for another passenger? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            print()
            print("Thank you for using the Titanic Survival Prediction Engine.")
            print()
            break
        print()


if __name__ == "__main__":
    main()
