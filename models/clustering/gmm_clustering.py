import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
import io

def gmm_clustering_page(data):
    st.subheader("Gaussian Mixture Model (GMM) Clustering")

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features = st.multiselect("Select features for clustering:", numeric_cols, key="gmm_features")

    if len(features) < 2:
        st.warning("Please select at least two numeric features.")
        return

    n_components = st.slider("Number of Clusters (Gaussian Components)", 2, 10, 3, key="gmm_n_components")

    # Scale data
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    cluster_labels = gmm.predict(X_scaled)

    # Create clustered DataFrame
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = cluster_labels

    # Store in session
    st.session_state["gmm_clustered_data"] = data_with_clusters

    st.markdown("###  Clustered Data Preview")
    st.dataframe(data_with_clusters)
    st.success(f" Formed {n_components} soft clusters using Gaussian Mixture Model!")

    # Download button
    csv = data_with_clusters.to_csv(index=False).encode("utf-8")
    st.download_button(" Download Clustered Data as CSV", csv, "gmm_clustered_data.csv", "text/csv")

    # Predict new data point
    st.markdown("###  Predict Cluster for a New Data Point")
    new_data = {}
    for feature in features:
        new_data[feature] = st.number_input(f"Value for {feature}", key="gmm_input_" + feature)

    if st.button("Predict Cluster", key="gmm_predict_btn"):
        new_df = pd.DataFrame([new_data])
        new_scaled = scaler.transform(new_df)
        probs = gmm.predict_proba(new_scaled)
        predicted_cluster = int(np.argmax(probs))
        confidence = probs[0][predicted_cluster]

        new_df["Predicted Cluster"] = predicted_cluster
        new_df["Confidence"] = confidence
        st.session_state["gmm_new_point"] = new_df

        st.success(f"Predicted Cluster: {predicted_cluster} (Confidence: {confidence:.2f})")
        st.dataframe(new_df)

    # Cluster Visualization
    st.markdown("###  Cluster Visualization (2D)")
    if len(features) > 2:
        x_axis = st.selectbox("X-axis", features, index=0, key="gmm_xaxis")
        y_axis = st.selectbox("Y-axis", [f for f in features if f != x_axis], index=0, key="gmm_yaxis")
    else:
        x_axis, y_axis = features[0], features[1]

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=data_with_clusters[x_axis],
        y=data_with_clusters[y_axis],
        hue=data_with_clusters["Cluster"],
        palette="tab10",
        ax=ax,
        legend="full"
    )

    # Show new point if predicted
    if "gmm_new_point" in st.session_state:
        new_pt = st.session_state["gmm_new_point"]
        ax.scatter(
            new_pt[x_axis].values[0],
            new_pt[y_axis].values[0],
            s=200, c='red', marker='D', label='New Data Point'
        )

    plt.title("GMM Cluster Visualization (Elliptical Gaussian Distributions)")
    ax.legend()
    st.pyplot(fig)

    # Educational Explanation
    with st.expander("📘 How GMM Works"):
        st.markdown("""
        - **GMM (Gaussian Mixture Model)** models data as a combination of multiple **Gaussian distributions**.
        - It uses **soft clustering** by assigning **probabilities** of belonging to each cluster.
        - Based on **Expectation-Maximization (EM)** algorithm:
          - Step 1: Estimate cluster memberships (E-Step)
          - Step 2: Update distribution parameters (M-Step)
        - Effective for **overlapping, elliptical-shaped clusters**.
        - Can handle **noise and uncertainty** better than hard clustering like K-Means.
        """)

    with st.expander("📌 Tips for Better GMM Results"):
        st.markdown("""
        - Scale your data properly before clustering (already applied).
        - More features can improve separation, but may also add noise.
        - Choose the number of clusters wisely — try multiple and compare results.
        - Use `Confidence` from prediction to assess uncertainty for new points.
        """)
