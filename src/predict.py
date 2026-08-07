import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def predict_sample(sample):
    # Load trained model
    model = joblib.load("models/random_forest.pkl")

    sample = pd.DataFrame([sample])

    # Encode text columns
    encoder = LabelEncoder()

    for column in sample.columns:
        if sample[column].dtype == "object":
            sample[column] = encoder.fit_transform(sample[column].astype(str))

    prediction = model.predict(sample)

    if prediction[0] == 1:
        print("\nPrediction: Suspicious Behaviour Detected")
    else:
        print("\nPrediction: Normal Behaviour")