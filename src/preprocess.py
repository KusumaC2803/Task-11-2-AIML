import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):

    # Make a copy of dataframe
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing values
    for column in df.columns:
        if df[column].dtype == "object":
            df.loc[:, column] = df[column].fillna(df[column].mode()[0])
        else:
            df.loc[:, column] = df[column].fillna(df[column].median())

    # Encode categorical columns
    encoder = LabelEncoder()

    for column in df.select_dtypes(include="object").columns:
        df.loc[:, column] = encoder.fit_transform(df[column].astype(str))

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y