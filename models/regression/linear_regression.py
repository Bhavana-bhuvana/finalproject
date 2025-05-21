# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from datetime import datetime
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import io

# def linear_regression_page(data):
#     st.header("Linear Regression Model")
#     MODEL_KEY = "regression_linear"

#     # Initialize session state for this model
#     if MODEL_KEY not in st.session_state:
#         st.session_state[MODEL_KEY] = {
#             "features": [],
#             "target": None,
#             "split_ratio": 0.2,
#             "intercept": None,
#             "coefficients": [],
#             "history": [],
#             "new_input": {},
#             "last_prediction": None
#         }

#     if data is None or data.empty or data.columns.empty:
#         st.warning("Please upload and preprocess a dataset with valid columns before using this model.")
#         return

#     features = st.multiselect("Select feature columns (X):", options=data.columns, default=st.session_state[MODEL_KEY]["features"])
#     target = st.selectbox(
#         "Select target column (y):",
#         options=data.columns,
#         index=data.columns.get_loc(st.session_state[MODEL_KEY]["target"]) if st.session_state[MODEL_KEY]["target"] in data.columns else 0
#     )

#     # Validate selections
#     if not features or not target or target in features:
#         st.warning("Please select valid feature(s) and target.")
#         return

#     st.session_state[MODEL_KEY]["features"] = features
#     st.session_state[MODEL_KEY]["target"] = target

#     X = data[features]
#     y = data[target]

#     # Train-test split
#     with st.expander("Train-Test Split Explanation"):
#         st.markdown("""
#         Adjust the ratio to control how much data is used to train vs. test the model.
#         A high test split can reveal overfitting issues.
#         """)

#     split_ratio = st.slider("Train-Test Split (Test Size)", min_value=0.1, max_value=0.9, value=st.session_state[MODEL_KEY]["split_ratio"], step=0.05)
#     st.session_state[MODEL_KEY]["split_ratio"] = split_ratio

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

#     # Train model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)

#     intercept = model.intercept_
#     coefficients = model.coef_

#     st.session_state[MODEL_KEY]["intercept"] = float(intercept)
#     st.session_state[MODEL_KEY]["coefficients"] = list(coefficients)

#     st.subheader("Model Coefficients")
#     st.write("**Intercept:**", round(intercept, 4))
#     coef_df = pd.DataFrame({
#         "Feature": features,
#         "Coefficient": [round(c, 4) for c in coefficients]
#     })
#     st.dataframe(coef_df, use_container_width=True)

#     r2_train = r2_score(y_train, train_pred)
#     r2_test = r2_score(y_test, test_pred)

#     st.subheader("Performance Summary")
#     st.metric("Train R²", f"{r2_train:.4f}")
#     st.metric("Test R²", f"{r2_test:.4f}")

#     if r2_train > 0.9 and r2_test < 0.6:
#         st.warning("Possible Overfitting: Model performs well on training but poorly on test data.")
#     elif r2_train < 0.5 and r2_test < 0.5:
#         st.warning("Possible Underfitting: Model is not learning the pattern from training data.")

#     st.subheader("Customize Coefficients & Intercept")
#     new_intercept = st.number_input("Intercept", value=st.session_state[MODEL_KEY]["intercept"], format="%.4f", step=0.1)
#     new_coeffs = []
#     for i, feature in enumerate(features):
#         coeff_val = st.session_state[MODEL_KEY]["coefficients"][i] if i < len(st.session_state[MODEL_KEY]["coefficients"]) else 0.0
#         coeff = st.number_input(f"Coefficient for {feature}", value=coeff_val, format="%.4f", step=0.1)
#         new_coeffs.append(coeff)

#     if st.button("Apply Custom Parameters"):
#         X_array = X.values
#         y_pred_custom = np.dot(X_array, new_coeffs) + new_intercept

#         r2 = r2_score(y, y_pred_custom)
#         mae = mean_absolute_error(y, y_pred_custom)
#         mse = mean_squared_error(y, y_pred_custom)
#         rmse = np.sqrt(mse)

#         result = {
#             "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "Intercept": new_intercept,
#             **{f: c for f, c in zip(features, new_coeffs)},
#             "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse
#         }

#         st.session_state[MODEL_KEY]["history"].append(result)

#         st.success("Custom coefficients applied and evaluated!")

#         st.subheader("Updated Performance")
#         st.metric("R² Score", f"{r2:.4f}")
#         st.metric("MAE", f"{mae:.4f}")
#         st.metric("MSE", f"{mse:.4f}")
#         st.metric("RMSE", f"{rmse:.4f}")

#     if st.session_state[MODEL_KEY]["history"]:
#         st.markdown("---")
#         st.subheader("Change History")
#         hist_df = pd.DataFrame(st.session_state[MODEL_KEY]["history"])
#         st.dataframe(hist_df, use_container_width=True)

#         # Download history CSV
#         csv = hist_df.to_csv(index=False).encode("utf-8")
#         st.download_button(
#             label="📥 Download History CSV",
#             data=csv,
#             file_name="linear_regression_history.csv",
#             mime="text/csv"
#         )

#     st.markdown("---")
#     st.subheader("Predict New Data Point")
#     new_data = {}
#     for feature in features:
#         val = st.number_input(f"Enter value for {feature}:", value=0.0, format="%.4f", key=f"predict_{feature}")
#         new_data[feature] = val

#     st.session_state[MODEL_KEY]["new_input"] = new_data

#     if st.button("Predict Target"):
#         input_array = np.array([list(new_data.values())])
#         use_custom = st.checkbox("Use custom coefficients for prediction", value=False)
#         if use_custom:
#             prediction = np.dot(input_array, new_coeffs) + new_intercept
#         else:
#             prediction = model.predict(input_array)
#         st.success(f"Predicted {target}: {round(prediction[0], 4)}")
#         st.session_state[MODEL_KEY]["last_prediction"] = float(prediction[0])

#     st.markdown("---")
#     st.subheader("Regression Plot (Scatter + Line)")
#     fig = px.scatter(
#         x=X_test[features[0]], y=y_test,
#         labels={'x': features[0], 'y': target},
#         title="Regression Fit"
#     )
#     fig.add_scatter(x=X_test[features[0]], y=test_pred, mode='lines', name='Prediction Line', line=dict(color='red'))
#     st.plotly_chart(fig)

#     # Download regression plot as PNG
#     img_buffer = io.BytesIO()
#     fig.write_image(img_buffer, format='png')
#     img_buffer.seek(0)
#     st.download_button(
#         label="📥 Download Regression Plot",
#         data=img_buffer,
#         file_name="linear_regression_plot.png",
#         mime="image/png"
#     )

#     st.markdown("---")
#     st.subheader("Correlation Heatmap")
#     with st.expander("Show correlation between features and target"):
#         corr = data[[*features, target]].corr()
#         fig_corr, ax = plt.subplots()
#         sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#         st.pyplot(fig_corr)
#         corr_buffer = io.BytesIO()
#     fig_corr.savefig(corr_buffer, format='png', bbox_inches='tight')
#     corr_buffer.seek(0)

#     st.download_button(
#         label="📥 Download Correlation Heatmap",
#         data=corr_buffer,
#         file_name="correlation_heatmap.png",
#         mime="image/png",
#         key="download_correlation_plot"
#     )


#     with st.expander("What do these metrics mean?"):
#         st.markdown("""
#         - **Intercept**: Predicted value when all features are zero.
#         - **Coefficient**: Effect of one-unit increase in a feature on the target.
#         - **R² Score**: Fraction of variance explained (1 is perfect).
#         - **MAE**: Average of absolute prediction errors.
#         - **MSE**: Average of squared prediction errors.
#         - **RMSE**: Square root of MSE – same units as target.
#         - **Overfitting**: High train performance, poor test performance.
#         - **Underfitting**: Poor performance on both train and test.
#         """)

#     # Optional: Reset button
#     if st.button("Reset Session", key="reset_session_button"):
#             st.session_state.pop(MODEL_KEY, None)
#             st.rerun()
#     # if st.button("Reset Session"):
#     #     st.session_state.pop(MODEL_KEY, None)
#     #     st.rerun()
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64

# Your external logging function
from db import save_log  

def linear_regression_page(data, user_email=None):
    st.header("Linear Regression Model")
    MODEL_KEY = "regression_linear"

    # Initialize session state for this model
    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = {
            "features": [],
            "target": None,
            "split_ratio": 0.2,
            "intercept": None,
            "coefficients": [],
            "history": [],
            "new_input": {},
            "last_prediction": None
        }

    if data is None or data.empty or data.columns.empty:
        st.warning("Please upload and preprocess a dataset with valid columns before using this model.")
        return

    # Feature and target selection
    features = st.multiselect(
        "Select feature columns (X):",
        options=data.columns,
        default=st.session_state[MODEL_KEY]["features"]
    )
    target = st.selectbox(
        "Select target column (y):",
        options=data.columns,
        index=data.columns.get_loc(st.session_state[MODEL_KEY]["target"]) if st.session_state[MODEL_KEY]["target"] in data.columns else 0
    )

    if not features or not target or target in features:
        st.warning("Please select valid feature(s) and target.")
        return

    st.session_state[MODEL_KEY]["features"] = features
    st.session_state[MODEL_KEY]["target"] = target

    X = data[features]
    y = data[target]

    # Train-test split slider with explanation
    with st.expander("Train-Test Split Explanation"):
        st.markdown("""
        Adjust the ratio to control how much data is used to train vs. test the model.
        A high test split can reveal overfitting issues.
        """)

    split_ratio = st.slider(
        "Train-Test Split (Test Size)",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state[MODEL_KEY]["split_ratio"],
        step=0.05
    )
    st.session_state[MODEL_KEY]["split_ratio"] = split_ratio

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    intercept = model.intercept_
    coefficients = model.coef_

    st.session_state[MODEL_KEY]["intercept"] = float(intercept)
    st.session_state[MODEL_KEY]["coefficients"] = list(coefficients)

    st.subheader("Model Coefficients")
    st.write("**Intercept:**", round(intercept, 4))
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": [round(c, 4) for c in coefficients]
    })
    st.dataframe(coef_df, use_container_width=True)

    # Performance summary
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)

    st.subheader("Performance Summary")
    st.metric("Train R²", f"{r2_train:.4f}")
    st.metric("Test R²", f"{r2_test:.4f}")

    if r2_train > 0.9 and r2_test < 0.6:
        st.warning("Possible Overfitting: Model performs well on training but poorly on test data.")
    elif r2_train < 0.5 and r2_test < 0.5:
        st.warning("Possible Underfitting: Model is not learning the pattern from training data.")

    # Customize coefficients and intercept
    st.subheader("Customize Coefficients & Intercept")
    new_intercept = st.number_input(
        "Intercept",
        value=st.session_state[MODEL_KEY]["intercept"],
        format="%.4f",
        step=0.1
    )
    new_coeffs = []
    for i, feature in enumerate(features):
        coeff_val = st.session_state[MODEL_KEY]["coefficients"][i] if i < len(st.session_state[MODEL_KEY]["coefficients"]) else 0.0
        coeff = st.number_input(
            f"Coefficient for {feature}",
            value=coeff_val,
            format="%.4f",
            step=0.1
        )
        new_coeffs.append(coeff)

    if st.button("Apply Custom Parameters"):
        X_array = X.values
        y_pred_custom = np.dot(X_array, new_coeffs) + new_intercept

        r2 = r2_score(y, y_pred_custom)
        mae = mean_absolute_error(y, y_pred_custom)
        mse = mean_squared_error(y, y_pred_custom)
        rmse = np.sqrt(mse)

        result = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Intercept": new_intercept,
            **{f: c for f, c in zip(features, new_coeffs)},
            "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse
        }

        st.session_state[MODEL_KEY]["history"].append(result)

        st.success("Custom coefficients applied and evaluated!")

        st.subheader("Updated Performance")
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

    # Show history of changes if any
    if st.session_state[MODEL_KEY]["history"]:
        st.markdown("---")
        st.subheader("Change History")
        hist_df = pd.DataFrame(st.session_state[MODEL_KEY]["history"])
        st.dataframe(hist_df, use_container_width=True)

        # Download history CSV
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download History CSV",
            data=csv,
            file_name="linear_regression_history.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.subheader("Predict New Data Point")
    new_data = {}
    for feature in features:
        val = st.number_input(
            f"Enter value for {feature}:",
            value=0.0,
            format="%.4f",
            key=f"predict_{feature}"
        )
        new_data[feature] = val

    st.session_state[MODEL_KEY]["new_input"] = new_data

    if st.button("Predict Target"):
        input_array = np.array([list(new_data.values())])
        use_custom = st.checkbox("Use custom coefficients for prediction", value=False)
        if use_custom:
            prediction = np.dot(input_array, new_coeffs) + new_intercept
        else:
            prediction = model.predict(input_array)
        st.success(f"Predicted {target}: {round(prediction[0], 4)}")
        st.session_state[MODEL_KEY]["last_prediction"] = float(prediction[0])

    st.markdown("---")
    st.subheader("Regression Plot (Scatter + Line)")

    fig = px.scatter(
        x=X_test[features[0]], y=y_test,
        labels={'x': features[0], 'y': target},
        title="Regression Fit"
    )
    fig.add_scatter(x=X_test[features[0]], y=test_pred, mode='lines', name='Prediction Line', line=dict(color='red'))
    st.plotly_chart(fig)

    # --- Fix for image buffer handling ---
    img_buffer = io.BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)

    # Base64 encoding for logging
    plot_image_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    img_buffer.seek(0)

    st.download_button(
        label="📥 Download Regression Plot",
        data=img_buffer,
        file_name="linear_regression_plot.png",
        mime="image/png"
    )

    st.markdown("---")
    st.subheader("Correlation Heatmap")

    with st.expander("Show correlation between features and target"):
        corr = data[[*features, target]].corr()
        fig_corr, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)

        corr_buffer = io.BytesIO()
        fig_corr.savefig(corr_buffer, format='png', bbox_inches='tight')
        corr_buffer.seek(0)

        corr_image_bytes = corr_buffer.getvalue()
        corr_image_base64 = base64.b64encode(corr_image_bytes).decode('utf-8')

        st.download_button(
            label="📥 Download Correlation Heatmap",
            data=corr_image_bytes,
            file_name="correlation_heatmap.png",
            mime="image/png",
            key="download_corr"
        )

        corr_buffer.close()

    # Prepare session log data for saving
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "Linear Regression",
        "features": features,
        "target": target,
        "train_test_split": split_ratio,
        "metrics": {
            "train_r2": round(r2_train, 4),
            "test_r2": round(r2_test, 4),
        },
        "custom_params": {
            "intercept": new_intercept,
            "coefficients": new_coeffs
        },
        "prediction_history": st.session_state[MODEL_KEY]["history"],
        "last_prediction": st.session_state[MODEL_KEY]["last_prediction"],
        "plot_image_base64": plot_image_base64,
        "correlation_heatmap_base64": corr_image_base64
    }
    user_email = st.session_state.get("user_email", None)
    if not user_email:
        st.error("User email not found in session.")
        st.stop()
    else:
        save_log(user_email, "Regression", "Linear Regression", session_data)
