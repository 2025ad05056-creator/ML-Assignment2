import streamlit as st
import pandas as pd
import joblib

st.title("Machine Learning Classification Models")
st.write("Adult Income Prediction using Multiple ML Models")

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    st.subheader("Select Machine Learning Model")

    model_name = st.selectbox(
        "Choose Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest"
        )
    )

    st.write("Selected Model:", model_name)

    if model_name == "Logistic Regression":
        model = joblib.load("models/logistic_regression.pkl")

    elif model_name == "Decision Tree":
        model = joblib.load("models/decision_tree.pkl")

    elif model_name == "KNN":
        model = joblib.load("models/knn.pkl")

    elif model_name == "Naive Bayes":
        model = joblib.load("models/naive_bayes.pkl")

    else:
        model = joblib.load("models/random_forest.pkl")

    if st.button("Predict"):

        predictions = model.predict(data)

        st.subheader("Predictions")

        prediction_df = data.copy()
        prediction_df["Prediction"] = predictions

        st.dataframe(prediction_df)

        st.subheader("Model Evaluation Metrics")

        results = pd.read_csv("model_results.csv")

        selected_result = results[results["ML Model"] == model_name]

        st.dataframe(selected_result)

        st.subheader("All Model Comparison")

        st.dataframe(results)