import streamlit as st
import pandas as pd
import joblib

from src.preprocess import preprocess_data

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Student Suspicious Behaviour Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Student Suspicious Behaviour Detection")
st.write(
    """
This application predicts whether a student's behaviour during an online examination
is **Normal** or **Suspicious** using a trained Random Forest model.
"""
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
model = joblib.load("models/random_forest.pkl")

# ----------------------------------------------------
# Upload CSV
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df)

    # Remove label column if available
    if "label" in df.columns:
        actual_labels = df["label"]
        df = df.drop("label", axis=1)
    else:
        actual_labels = None

    # ------------------------------------------------
    # Apply same preprocessing used during training
    # ------------------------------------------------
    processed_df = df.copy()

    for col in processed_df.columns:

        if processed_df[col].dtype == "object":

            processed_df[col] = (
                processed_df[col]
                .fillna(processed_df[col].mode()[0])
                .astype("category")
                .cat.codes
            )

        else:

            processed_df[col] = (
                processed_df[col]
                .fillna(processed_df[col].median())
            )

    # ------------------------------------------------
    # Match feature names used while training
    # ------------------------------------------------
    processed_df = processed_df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------
    predictions = model.predict(processed_df)

    result = df.copy()

    result["Prediction"] = predictions

    result["Prediction"] = result["Prediction"].replace(
        {
            0: "Normal",
            1: "Suspicious"
        }
    )

    if actual_labels is not None:
        result["Actual Label"] = actual_labels

    st.subheader("Prediction Results")
    st.dataframe(result)

    # ------------------------------------------------
    # Prediction Summary
    # ------------------------------------------------
    st.subheader("Prediction Summary")

    summary = result["Prediction"].value_counts()

    st.bar_chart(summary)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Normal Students",
            int((result["Prediction"] == "Normal").sum())
        )

    with col2:
        st.metric(
            "Suspicious Students",
            int((result["Prediction"] == "Suspicious").sum())
        )

    # ------------------------------------------------
    # Download Results
    # ------------------------------------------------
    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Predictions",
        csv,
        "prediction_results.csv",
        "text/csv"
    )

else:

    st.info("Upload a CSV file to start prediction.")