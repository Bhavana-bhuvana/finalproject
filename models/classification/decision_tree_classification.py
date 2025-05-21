import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import io
import base64

def decision_tree_classification_page(data):
    st.header(" Decision Tree Classifier")

    if data is None:
        st.warning("Please upload and preprocess a dataset first.")
        return

    # Session state
    if 'dt_model' not in st.session_state:
        st.session_state['dt_model'] = None
    if 'dt_predictions' not in st.session_state:
        st.session_state['dt_predictions'] = None
    if 'dt_report' not in st.session_state:
        st.session_state['dt_report'] = ""

    # Feature/target selection
    target_col = st.selectbox("🎯 Select Target Variable", data.columns)
    feature_cols = st.multiselect(
        " Select Feature Columns", 
        [col for col in data.columns if col != target_col],
        default=[col for col in data.columns if col != target_col]
    )

    if not feature_cols or not target_col:
        st.warning("Select at least one feature and a target column to proceed.")
        return

    X = data[feature_cols]
    y = data[target_col]

    # Train-test split
    test_size = st.slider("Test Size (for splitting data)", 0.1, 0.5, 0.3)

    if st.button("🚀 Train Decision Tree Model"):
        model = DecisionTreeClassifier(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)

        st.success(f"✅ Model trained! Accuracy: {acc:.2f}")
        st.text("📊 Classification Report:")
        st.code(report)

        # Store in session
        st.session_state['dt_model'] = model
        st.session_state['dt_predictions'] = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })
        st.session_state['dt_report'] = report

        # Plot: Feature Importance
        st.subheader("📈 Feature Importance")
        importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig, ax = plt.subplots()
        importance.plot(kind='barh', ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        # Download feature importance as PNG
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("📥 Download Feature Importance Plot", data=buf, file_name="feature_importance.png", mime="image/png")

        # Plot: Decision Tree
        st.subheader("🌳 Decision Tree Visualization")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        plot_tree(model, feature_names=feature_cols, class_names=True, filled=True, ax=ax2)
        st.pyplot(fig2)

        # Download tree plot
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png")
        buf2.seek(0)
        st.download_button("📥 Download Tree Plot", data=buf2, file_name="decision_tree.png", mime="image/png")

        # Download predictions
        csv = st.session_state['dt_predictions'].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

        # Download classification report
        st.download_button("📄 Download Classification Report", data=report, file_name="classification_report.txt", mime="text/plain")

    elif st.session_state['dt_model']:
        st.info("A trained model is already available. Click above to retrain.")

