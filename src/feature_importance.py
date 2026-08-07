import joblib
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(data_path):
    # Load trained model
    model = joblib.load("models/random_forest.pkl")

    # Load dataset
    df = pd.read_csv(data_path)

    # Features
    X = df.drop("label", axis=1)

    # Convert categorical columns into numbers
    X = pd.get_dummies(X)

    # Match training features
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    # Feature importance
    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 10 Important Features")
    print(feature_df.head(10))

    plt.figure(figsize=(10,6))
    plt.barh(
        feature_df["Feature"][:10],
        feature_df["Importance"][:10]
    )

    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Top 10 Feature Importance")

    plt.tight_layout()

    plt.savefig("outputs/feature_importance.png")

    plt.show()

    print("\nFeature importance graph saved in outputs folder.")