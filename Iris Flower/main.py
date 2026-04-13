import joblib
import numpy as np
import sys

def load_model(model_path, encoder_path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

def get_user_input():
    print("-" * 50)
    print("Enter the flower measurements below:")
    print("-" * 50)

    try:
        sepal_length = float(input("  Sepal Length (cm): "))
        sepal_width = float(input("  Sepal Width  (cm): "))
        petal_length = float(input("  Petal Length (cm): "))
        petal_width = float(input("  Petal Width  (cm): "))
    except ValueError:
        print("\nError: Please enter valid numeric values.")
        sys.exit(1)

    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

def predict_species(model, label_encoder, features):
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    predicted_class = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(probabilities) * 100

    return predicted_class, confidence, probabilities[0]

def display_results(predicted_class, confidence, probabilities, label_encoder):
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"  Predicted Species : {predicted_class}")
    print(f"  Confidence        : {confidence:.2f}%")
    print("-" * 50)
    print("  Probability Breakdown:")
    for class_name, prob in zip(label_encoder.classes_, probabilities):
        bar_length = int(prob * 30)
        bar = "|" * bar_length
        print(f"    {class_name:20s} : {prob * 100:6.2f}%  {bar}")
    print("=" * 50)

def main():
    print("=" * 50)
    print("IRIS FLOWER CLASSIFICATION - PREDICTION")
    print("=" * 50)

    try:
        model, label_encoder = load_model("iris_model.pkl", "label_encoder.pkl")
        print("Model loaded successfully!\n")
    except FileNotFoundError:
        print("Error: Model files not found.")
        print("Please run train.py first to train and save the model.")
        sys.exit(1)

    while True:
        features = get_user_input()
        predicted_class, confidence, probabilities = predict_species(
            model, label_encoder, features
        )
        display_results(predicted_class, confidence, probabilities, label_encoder)

        print("\nWould you like to classify another flower?")
        choice = input("Enter 'yes' to continue or any other key to exit: ").strip().lower()
        if choice != "yes":
            print("\nThank you for using the Iris Flower Classifier!")
            break

if __name__ == "__main__":
    main()
