import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px

MODEL_KEY = "svr_model"

def svr_regression_page(data: pd.DataFrame):
    st.header("Support Vector Regression (SVR)")

    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = {
            "features": [],
            "target": None,
            "test_size": 0.2,
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.1,
            "model": None,
            "scaler_X": None,
            "scaler_y": None,
            "results_df": None,
            "trained": False,
        }

    # Feature and Target Selection
    features = st.multiselect(
        "Select feature column(s):",
        options=data.columns,
        default=st.session_state[MODEL_KEY]["features"]
    )
    target = st.selectbox(
        "Select target column:",
        options=data.columns,
        index=data.columns.get_loc(st.session_state[MODEL_KEY]["target"]) if st.session_state[MODEL_KEY]["target"] in data.columns else 0
    )

    if not features or not target or target in features:
        st.warning("Please select valid features and a target column (target cannot be a feature).")
        return

    # Save selections
    st.session_state[MODEL_KEY]["features"] = features
    st.session_state[MODEL_KEY]["target"] = target

    # Feature selection tooltip
    with st.expander("🔍 How to Choose Good Features"):
        st.markdown("""
        - **SVR is sensitive to feature scaling**, so numerical features are best.
        - Avoid categorical columns unless they are encoded.
        - Choose features that correlate well with the target.
        - Consider checking pairwise plots or correlation matrices before final selection.
        """)

    # Train-Test split slider
    test_size = st.slider(
        "Select test size ratio:",
        min_value=0.1, max_value=0.5,
        value=st.session_state[MODEL_KEY]["test_size"],
        step=0.05
    )
    st.session_state[MODEL_KEY]["test_size"] = test_size

    # SVR Parameters Expander
    with st.expander(" Customize SVR Parameters"):
        kernel = st.selectbox(
            "Kernel",
            ["rbf", "linear", "poly", "sigmoid"],
            index=["rbf", "linear", "poly", "sigmoid"].index(st.session_state[MODEL_KEY]["kernel"])
        )
        C = st.slider(
            "C (Regularization parameter)",
            min_value=0.01, max_value=10.0,
            value=st.session_state[MODEL_KEY]["C"],
            step=0.1
        )
        epsilon = st.slider(
            "Epsilon",
            min_value=0.0, max_value=1.0,
            value=st.session_state[MODEL_KEY]["epsilon"],
            step=0.01
        )

    # Save parameters
    st.session_state[MODEL_KEY]["kernel"] = kernel
    st.session_state[MODEL_KEY]["C"] = C
    st.session_state[MODEL_KEY]["epsilon"] = epsilon

    # Train button
    if st.button("Train SVR Model"):
        # Prepare data
        X = data[features]
        y = data[target]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42
        )

        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X_train, y_train)

        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        results_df = pd.DataFrame({
            "Actual": y_test_actual,
            "Predicted": y_pred
        })

        # Save all to session state
        st.session_state[MODEL_KEY].update({
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "results_df": results_df,
            "trained": True,
        })

        st.success("Model trained successfully!")

    # If model trained, show results
    if st.session_state[MODEL_KEY]["trained"]:
        model = st.session_state[MODEL_KEY]["model"]
        scaler_y = st.session_state[MODEL_KEY]["scaler_y"]
        results_df = st.session_state[MODEL_KEY]["results_df"]

        # Metrics
        mse = mean_squared_error(results_df["Actual"], results_df["Predicted"])
        mae = mean_absolute_error(results_df["Actual"], results_df["Predicted"])
        r2 = r2_score(results_df["Actual"], results_df["Predicted"])

        st.subheader(" Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        col3.metric("R² Score", f"{r2:.4f}")

        with st.expander("ℹ️ What do these metrics mean?"):
            st.markdown("""
            - **Mean Squared Error (MSE)**: Penalizes large errors more than small ones. Lower is better.
            - **Mean Absolute Error (MAE)**: Average of absolute differences between predicted and actual values.
            - **R² Score**: Proportion of variance in target explained by model. Closer to 1 is better.
            """)

        # Plot Actual vs Predicted
        st.subheader("📈 Actual vs Predicted")
        fig = px.scatter(
            results_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Prediction on custom input
        with st.expander("🔮 Predict with Custom Input"):
            input_data = {}
            for feature in features:
                default_val = float(data[feature].mean())
                val = st.number_input(f"Input value for {feature}:", value=default_val)
                input_data[feature] = val

            if st.button("Predict", key="predict_button"):
                input_df = pd.DataFrame([input_data])
                input_scaled = st.session_state[MODEL_KEY]["scaler_X"].transform(input_df)
                pred_scaled = st.session_state[MODEL_KEY]["model"].predict(input_scaled)
                pred = st.session_state[MODEL_KEY]["scaler_y"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
                st.success(f"Predicted value: **{pred:.4f}**")
