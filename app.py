import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Student Suspicious Behaviour Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Student Suspicious Behaviour Detection")

st.write(
    """
Upload a CSV file containing student behavioural features.
The trained Random Forest model will classify each record as
**Normal** or **Suspicious**.
"""
)

# ------------------------------
# Load Model
# ------------------------------
try:
    model = joblib.load("models/random_forest.pkl")
except FileNotFoundError:
    st.error("Model file not found! Make sure models/random_forest.pkl exists.")
    st.stop()

# ------------------------------
# Upload CSV
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df)

    prediction_df = df.copy()

    # Remove target column if uploaded
    if "label" in prediction_df.columns:
        prediction_df = prediction_df.drop(columns=["label"])

    # ------------------------------
    # Preprocessing
    # ------------------------------
    for col in prediction_df.columns:

        # Numeric column
        if pd.api.types.is_numeric_dtype(prediction_df[col]):
            prediction_df[col] = prediction_df[col].fillna(
                prediction_df[col].median()
            )

        # Categorical column
        else:
            prediction_df[col] = (
                prediction_df[col]
                .fillna(prediction_df[col].mode()[0])
                .astype("category")
                .cat.codes
            )

    # ------------------------------
    # Match feature order
    # ------------------------------
    prediction_df = prediction_df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # ------------------------------
    # Predict
    # ------------------------------
    predictions = model.predict(prediction_df)

    result = df.copy()

    result["Prediction"] = [
        "Suspicious" if p == 1 else "Normal"
        for p in predictions
    ]

    st.subheader("Prediction Results")
    st.dataframe(result)

    # ------------------------------
    # Summary
    # ------------------------------
    st.subheader("Prediction Summary")

    counts = result["Prediction"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Normal Students",
            int(counts.get("Normal", 0))
        )

    with col2:
        st.metric(
            "Suspicious Students",
            int(counts.get("Suspicious", 0))
        )

    st.bar_chart(counts)

    # ------------------------------
    # Download
    # ------------------------------
    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Predictions",
        csv,
        "prediction_results.csv",
        "text/csv"
    )

else:

    st.info("Please upload a CSV file to begin prediction.")