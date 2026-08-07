from src.preprocess import load_data, preprocess_data
from src.train_model import train_model
from src.evaluate import evaluate_model

def main():
    print("=" * 50)
    print("Student Suspicious Behavior Detection")
    print("=" * 50)

    # Load dataset
    df = load_data("data/suspicious_behaviour.csv")

    # Preprocess dataset
    X, y = preprocess_data(df)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()