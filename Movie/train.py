import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import re


def load_dataset(filepath):
    print("=" * 60)
    print("  STEP 1: Loading Dataset")
    print("=" * 60)
    df = pd.read_csv(filepath, encoding="latin-1")
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Columns: {list(df.columns)}")
    return df


def explore_data(df):
    print("\n" + "=" * 60)
    print("  STEP 2: Exploratory Data Analysis")
    print("=" * 60)

    print("\n--- First 5 Rows ---")
    print(df.head().to_string())

    print("\n--- Dataset Info ---")
    print(f"Shape: {df.shape}")
    print(f"Data Types:\n{df.dtypes.to_string()}")

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())

    print("\n--- Statistical Summary (Numeric Columns) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().to_string())

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "Rating" in df.columns:
            rating_data = df["Rating"].dropna()
            axes[0].hist(rating_data, bins=20, color="steelblue", edgecolor="black")
            axes[0].set_title("Distribution of Movie Ratings")
            axes[0].set_xlabel("Rating")
            axes[0].set_ylabel("Frequency")

        if "Year" in df.columns:
            year_data = df["Year"].dropna()
            year_data = pd.to_numeric(
                year_data.astype(str).str.extract(r"(\d{4})", expand=False),
                errors="coerce"
            ).dropna()
            axes[1].hist(year_data, bins=30, color="coral", edgecolor="black")
            axes[1].set_title("Distribution of Release Years")
            axes[1].set_xlabel("Year")
            axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig("eda_distributions.png", dpi=100)
        plt.close()
        print("\nSaved distribution plots to eda_distributions.png")
    except Exception as e:
        print(f"\nCould not generate distribution plots: {e}")

    try:
        if "Genre" in df.columns:
            genre_counts = df["Genre"].dropna().str.split(",").explode().str.strip().value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            genre_counts.plot(kind="barh", color="teal", edgecolor="black", ax=ax)
            ax.set_title("Top 10 Genres")
            ax.set_xlabel("Count")
            ax.set_ylabel("Genre")
            plt.tight_layout()
            plt.savefig("eda_top_genres.png", dpi=100)
            plt.close()
            print("Saved top genres plot to eda_top_genres.png")
    except Exception as e:
        print(f"Could not generate genre plot: {e}")


def clean_data(df):
    print("\n" + "=" * 60)
    print("  STEP 3: Data Cleaning and Preprocessing")
    print("=" * 60)

    df = df.copy()

    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        print("Cleaned 'Year' column - extracted numeric year values.")

    if "Duration" in df.columns:
        df["Duration"] = df["Duration"].astype(str).str.extract(r"(\d+)", expand=False)
        df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
        print("Cleaned 'Duration' column - extracted numeric duration values.")

    if "Votes" in df.columns:
        df["Votes"] = df["Votes"].astype(str).str.replace(",", "", regex=False)
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
        print("Cleaned 'Votes' column - removed commas and converted to numeric.")

    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    initial_rows = len(df)
    df = df.dropna(subset=["Rating"])
    print(f"Dropped rows with missing Rating: {initial_rows - len(df)} rows removed.")

    for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in ["Year", "Duration", "Votes"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing '{col}' with median value: {median_val}")

    print(f"Final dataset shape after cleaning: {df.shape}")
    return df


def engineer_features(df):
    print("\n" + "=" * 60)
    print("  STEP 4: Feature Engineering")
    print("=" * 60)

    df = df.copy()

    if "Genre" in df.columns:
        df["Primary_Genre"] = df["Genre"].astype(str).str.split(",").str[0].str.strip()
        print("Created 'Primary_Genre' from the first genre in the Genre column.")

    encoders = {}
    categorical_features = ["Primary_Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"Label-encoded '{col}' -> '{col}_encoded' ({len(le.classes_)} unique values)")

    if "Year" in df.columns:
        current_year = 2026
        df["Movie_Age"] = current_year - df["Year"]
        df["Movie_Age"] = df["Movie_Age"].clip(lower=0)
        print("Created 'Movie_Age' feature (current year - release year).")

    if "Votes" in df.columns:
        df["Log_Votes"] = np.log1p(df["Votes"])
        print("Created 'Log_Votes' feature (log-transformed vote count).")

    feature_columns = [
        "Year",
        "Duration",
        "Votes",
        "Log_Votes",
        "Movie_Age",
        "Primary_Genre_encoded",
        "Director_encoded",
        "Actor 1_encoded",
        "Actor 2_encoded",
        "Actor 3_encoded",
    ]

    available_features = [col for col in feature_columns if col in df.columns]
    print(f"\nFinal feature set ({len(available_features)} features): {available_features}")

    return df, available_features, encoders


def train_model(df, feature_columns, target_column="Rating"):
    print("\n" + "=" * 60)
    print("  STEP 5: Model Training and Evaluation")
    print("=" * 60)

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size:     {X_test.shape[0]}")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest Regressor...")
    model.fit(X_train, y_train)
    print("Training complete.")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("\n--- Training Metrics ---")
    print(f"MAE:  {mean_absolute_error(y_train, y_pred_train):.4f}")
    print(f"MSE:  {mean_squared_error(y_train, y_pred_train):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print(f"R2:   {r2_score(y_train, y_pred_train):.4f}")

    print("\n--- Test Metrics ---")
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    try:
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        feat_imp.plot(kind="barh", color="darkcyan", edgecolor="black", ax=ax)
        ax.set_title("Feature Importance (Random Forest)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=100)
        plt.close()
        print("\nSaved feature importance plot to feature_importance.png")
    except Exception as e:
        print(f"\nCould not generate feature importance plot: {e}")

    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred_test, alpha=0.4, color="teal", edgecolors="black", s=30)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="red",
            linewidth=2,
            linestyle="--",
        )
        ax.set_xlabel("Actual Rating")
        ax.set_ylabel("Predicted Rating")
        ax.set_title("Actual vs Predicted Ratings")
        plt.tight_layout()
        plt.savefig("actual_vs_predicted.png", dpi=100)
        plt.close()
        print("Saved actual vs predicted plot to actual_vs_predicted.png")
    except Exception as e:
        print(f"Could not generate actual vs predicted plot: {e}")

    return model


def save_artifacts(model, encoders, feature_columns, save_dir="."):
    print("\n" + "=" * 60)
    print("  STEP 6: Saving Model Artifacts")
    print("=" * 60)

    model_path = os.path.join(save_dir, "movie_rating_model.pkl")
    encoders_path = os.path.join(save_dir, "label_encoders.pkl")
    features_path = os.path.join(save_dir, "feature_columns.pkl")

    joblib.dump(model, model_path)
    print(f"Saved trained model to: {model_path}")

    joblib.dump(encoders, encoders_path)
    print(f"Saved label encoders to: {encoders_path}")

    joblib.dump(feature_columns, features_path)
    print(f"Saved feature columns to: {features_path}")

    print("\nAll artifacts saved successfully.")


def main():
    print("*" * 60)
    print("  Movie Rating Prediction - Training Pipeline")
    print("*" * 60)

    dataset_path = "IMDb Movies India.csv"

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file '{dataset_path}' not found.")
        print("Please place the dataset in the current directory and try again.")
        return

    df = load_dataset(dataset_path)

    explore_data(df)

    df_clean = clean_data(df)

    df_featured, feature_columns, encoders = engineer_features(df_clean)

    model = train_model(df_featured, feature_columns)

    save_artifacts(model, encoders, feature_columns)

    print("\n" + "*" * 60)
    print("  Training Pipeline Complete!")
    print("  You can now run main.py to make predictions.")
    print("*" * 60)


if __name__ == "__main__":
    main()
