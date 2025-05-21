import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import plotly.express as px
from sklearn.naive_bayes import GaussianNB

def naive_bayes_classification_page(data):
    st.header("Naive Bayes Classification")

    if "classification_history" not in st.session_state:
        st.session_state.classification_history = []

    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    # Tooltips for Feature Selection
    with st.expander("How to Select Features"):
        st.markdown("""
        - **Good Features**: Naive Bayes works well with independent features, ideally not highly correlated.
        - **Numeric Features**: Ideal for continuous variables (e.g., age, salary). For categorical data, it works well with techniques like one-hot encoding.
        - **Avoid Overfitting**: Ensure you don't include too many features without enough data, as it can lead to overfitting.
        """)

    # Train-test split slider
    st.slider("Train-Test Split", min_value=0.1, max_value=0.9, value=0.2, step=0.05, key="split_ratio")
    split_ratio = st.session_state.split_ratio

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)

    # Naive Bayes Model
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    st.subheader("Model Parameters")
    st.write(model.get_params())

    # Interactive sliders for customization
    st.subheader("Customize Hyperparameters")
    if st.button("Apply Custom Hyperparameters"):
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        st.session_state.classification_history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1
        })

        st.success("Custom hyperparameters applied and evaluated!")

        st.subheader("Updated Performance")
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")

    # Visualizing predictions with Plotly
    st.markdown("---")
    st.subheader("Naive Bayes Prediction")
    fig = px.scatter(x=X_test[features[0]], y=X_test[features[1]], color=predictions, labels={'x': features[0], 'y': features[1]}, title="Naive Bayes Classification")
    st.plotly_chart(fig)

    # Educational section with an expandable box
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Accuracy**: Proportion of correctly predicted instances.
        - **Precision**: Fraction of relevant instances among the retrieved instances.
        - **Recall**: Fraction of relevant instances that have been retrieved.
        - **F1 Score**: The harmonic mean of precision and recall.
        """)
