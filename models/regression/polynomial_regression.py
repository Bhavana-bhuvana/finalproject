import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import plotly.express as px

def polynomial_regression_page(data):
    st.header("Polynomial Regression Model")

    if "regression_history" not in st.session_state:
        st.session_state.regression_history = []

    # Select features and target
    features = st.multiselect("Select feature columns (X):", options=data.columns)
    target = st.selectbox("Select target column (y):", options=data.columns)

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    # Train-test split slider with tooltip
    split_ratio = st.slider("Test set size", min_value=0.1, max_value=0.9, value=0.2, step=0.05, help="Test set size")

    degree = st.slider("Polynomial Degree", min_value=1, max_value=5, value=2, step=1, help="Polynomial degree")

    X = data[features].values
    y = data[target].values

    # Polynomial transform
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)

    # Get feature names for all poly terms
    poly_feature_names = poly.get_feature_names_out(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=split_ratio, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    intercept = model.intercept_
    coefficients = model.coef_

    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        "Term": poly_feature_names,
        "Coefficient": [round(c, 4) for c in coefficients]
    })
    st.dataframe(coef_df, use_container_width=True)

    # Customize coefficients & intercept sliders (one per poly term)
    st.subheader("Customize Coefficients & Intercept")
    new_intercept = st.number_input("Intercept", value=float(intercept), format="%.4f", step=0.1)
    new_coeffs = []
    for i, name in enumerate(poly_feature_names):
        val = st.number_input(f"Coefficient for {name}", value=float(coefficients[i]), format="%.4f", step=0.1)
        new_coeffs.append(val)

    if st.button("Apply Custom Parameters"):
        y_pred_custom = X_poly @ np.array(new_coeffs) + new_intercept

        r2 = r2_score(y, y_pred_custom)
        mae = mean_absolute_error(y, y_pred_custom)
        mse = mean_squared_error(y, y_pred_custom)
        rmse = np.sqrt(mse)

        st.session_state.regression_history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Intercept": new_intercept,
            **{name: c for name, c in zip(poly_feature_names, new_coeffs)},
            "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse
        })

        st.success("Custom coefficients applied and evaluated!")

        st.subheader("Updated Performance")
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

    if st.session_state.regression_history:
        st.markdown("---")
        st.subheader("Change History")
        hist_df = pd.DataFrame(st.session_state.regression_history)
        st.dataframe(hist_df, use_container_width=True)

    # Predicting new data point
    st.markdown("---")
    st.subheader("Predict New Data Point")

    new_data = {}
    for feature in features:
        val = st.number_input(f"Enter value for {feature}:", value=0.0, format="%.4f")
        new_data[feature] = val

    use_custom = st.checkbox("Use custom coefficients for prediction", value=False)

    if st.button("Predict Target"):
        input_array = np.array([list(new_data.values())])
        input_poly = poly.transform(input_array)

        if use_custom:
            prediction = input_poly @ np.array(new_coeffs) + new_intercept
        else:
            prediction = model.predict(input_poly)

        st.success(f"Predicted {target}: {round(prediction[0], 4)}")

    # Plotting polynomial regression curve only if single feature selected
    if len(features) == 1:
        st.markdown("---")
        st.subheader("Polynomial Regression Plot (Scatter + Curve)")
        x_feature = features[0]
        fig = px.scatter(x=data[x_feature], y=data[target], labels={'x': x_feature, 'y': target}, title="Polynomial Regression Fit")

        sorted_idx = data[x_feature].argsort()
        x_sorted = data[x_feature].values[sorted_idx]
        X_poly_sorted = poly.transform(x_sorted.reshape(-1, 1))
        y_pred_sorted = model.predict(X_poly_sorted)

        fig.add_scatter(x=x_sorted, y=y_pred_sorted, mode='lines', name='Prediction Curve', line=dict(color='red'))
        st.plotly_chart(fig)
    else:
        st.info("Plot available only for single feature polynomial regression.")

    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Intercept**: Expected value of target when all features are zero.
        - **Coefficient**: Impact of each polynomial term on the target.
        - **R² Score**: Fraction of variance explained by the model (1 is perfect).
        - **MAE**: Average absolute error between predictions and true values.
        - **MSE**: Average squared error between predictions and true values.
        - **RMSE**: Square root of MSE, in target units.
        """)

