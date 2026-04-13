import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import os
import sys


def load_artifacts(model_dir="."):
    """Load the trained model, label encoders, and feature columns."""
    model_path = os.path.join(model_dir, "movie_rating_model.pkl")
    encoders_path = os.path.join(model_dir, "label_encoders.pkl")
    features_path = os.path.join(model_dir, "feature_columns.pkl")

    for path, name in [
        (model_path, "Model"),
        (encoders_path, "Encoders"),
        (features_path, "Feature columns"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found at '{path}'.")
            print("Please run train.py first to train and save the model.")
            sys.exit(1)

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    feature_columns = joblib.load(features_path)

    print("Model and artifacts loaded successfully.")
    return model, encoders, feature_columns


def safe_encode(encoder, value, column_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        print(f"  Note: '{value}' was not seen during training for '{column_name}'. Using fallback encoding.")
        return -1


def get_user_input():
    print("\n" + "-" * 50)
    print("  Enter Movie Details for Rating Prediction")
    print("-" * 50)

    genre = input("  Genre (e.g., Drama, Action, Comedy): ").strip()
    if not genre:
        genre = "Unknown"

    director = input("  Director name: ").strip()
    if not director:
        director = "Unknown"

    actor1 = input("  Actor 1 (lead actor): ").strip()
    if not actor1:
        actor1 = "Unknown"

    actor2 = input("  Actor 2: ").strip()
    if not actor2:
        actor2 = "Unknown"

    actor3 = input("  Actor 3: ").strip()
    if not actor3:
        actor3 = "Unknown"

    year_str = input("  Release Year (e.g., 2020): ").strip()
    try:
        year = int(year_str)
    except (ValueError, TypeError):
        print("  Invalid year. Defaulting to 2020.")
        year = 2020

    duration_str = input("  Duration in minutes (e.g., 120): ").strip()
    try:
        duration = int(duration_str)
    except (ValueError, TypeError):
        print("  Invalid duration. Defaulting to 120.")
        duration = 120

    votes_str = input("  Number of votes (e.g., 5000): ").strip()
    try:
        votes = int(votes_str.replace(",", ""))
    except (ValueError, TypeError):
        print("  Invalid votes. Defaulting to 1000.")
        votes = 1000

    return {
        "Genre": genre,
        "Director": director,
        "Actor 1": actor1,
        "Actor 2": actor2,
        "Actor 3": actor3,
        "Year": year,
        "Duration": duration,
        "Votes": votes,
    }


def predict_rating(model, encoders, feature_columns, movie_data):
    """Build a feature vector from movie_data and predict the rating."""

    current_year = 2026

    primary_genre = movie_data["Genre"].split(",")[0].strip()
    genre_encoded = safe_encode(encoders.get("Primary_Genre"), primary_genre, "Primary_Genre") if "Primary_Genre" in encoders else 0
    director_encoded = safe_encode(encoders.get("Director"), movie_data["Director"], "Director") if "Director" in encoders else 0
    actor1_encoded = safe_encode(encoders.get("Actor 1"), movie_data["Actor 1"], "Actor 1") if "Actor 1" in encoders else 0
    actor2_encoded = safe_encode(encoders.get("Actor 2"), movie_data["Actor 2"], "Actor 2") if "Actor 2" in encoders else 0
    actor3_encoded = safe_encode(encoders.get("Actor 3"), movie_data["Actor 3"], "Actor 3") if "Actor 3" in encoders else 0

    movie_age = current_year - movie_data["Year"]
    if movie_age < 0:
        movie_age = 0

    log_votes = np.log1p(movie_data["Votes"])

    feature_map = {
        "Year": movie_data["Year"],
        "Duration": movie_data["Duration"],
        "Votes": movie_data["Votes"],
        "Log_Votes": log_votes,
        "Movie_Age": movie_age,
        "Primary_Genre_encoded": genre_encoded,
        "Director_encoded": director_encoded,
        "Actor 1_encoded": actor1_encoded,
        "Actor 2_encoded": actor2_encoded,
        "Actor 3_encoded": actor3_encoded,
    }

    feature_values = []
    for col in feature_columns:
        feature_values.append(feature_map.get(col, 0))

    feature_df = pd.DataFrame([feature_values], columns=feature_columns)

    prediction = model.predict(feature_df)[0]

    prediction = max(1.0, min(10.0, prediction))

    return round(prediction, 2)


def display_prediction(movie_data, predicted_rating):
    """Display the prediction result in a formatted output."""
    print("\n" + "=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    print(f"  Movie Details:")
    print(f"    Genre:    {movie_data['Genre']}")
    print(f"    Director: {movie_data['Director']}")
    print(f"    Actor 1:  {movie_data['Actor 1']}")
    print(f"    Actor 2:  {movie_data['Actor 2']}")
    print(f"    Actor 3:  {movie_data['Actor 3']}")
    print(f"    Year:     {movie_data['Year']}")
    print(f"    Duration: {movie_data['Duration']} minutes")
    print(f"    Votes:    {movie_data['Votes']}")
    print(f"\n  >>> Predicted Rating: {predicted_rating} / 10.0 <<<")
    print("=" * 50)


def main():
    print("*" * 60)
    print("  Movie Rating Prediction System")
    print("*" * 60)

    model, encoders, feature_columns = load_artifacts()

    while True:
        movie_data = get_user_input()

        predicted_rating = predict_rating(model, encoders, feature_columns, movie_data)

        display_prediction(movie_data, predicted_rating)

        print("\n" + "-" * 50)
        again = input("  Would you like to predict another movie rating? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  Thank you for using the Movie Rating Prediction System!")
            print("  Goodbye.\n")
            break


if __name__ == "__main__":
    main()
