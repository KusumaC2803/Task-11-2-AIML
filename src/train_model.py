import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(X, y):
    """
    Split the dataset and train the Random Forest model.
    """

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "models/random_forest.pkl")

    print("Model trained successfully.")
    print("Model saved in models/random_forest.pkl")

    return model, X_test, y_test