import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io

def kmeans_clustering_page(data):
    st.subheader("Clustering Analysis")

    MODEL_KEY = "clustering_kmeans"
    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = {
            "features": [],
            "cluster_count": 3,
            "history": [],  # Store new points + predicted clusters
        }

    if data is None or data.empty:
        st.warning("Please upload a valid dataset to continue.")
        return

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    features = st.multiselect(
        "Select features for clustering:", 
        options=numeric_cols,
        default=st.session_state[MODEL_KEY]["features"]
    )

    cluster_count = st.slider(
        "Number of clusters", 2, 10, 
        value=st.session_state[MODEL_KEY]["cluster_count"]
    )

    if len(features) < 2:
        st.warning("Select at least two numeric features.")
        return

    st.session_state[MODEL_KEY]["features"] = features
    st.session_state[MODEL_KEY]["cluster_count"] = cluster_count

    # Prepare and scale data
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = cluster_labels

    st.markdown("### Clustered Data")
    st.dataframe(data_with_clusters, use_container_width=True)

    st.success(f"Formed {cluster_count} clusters!")

    # Provide download for clustered data CSV
    csv_clusters = data_with_clusters.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Clustered Data CSV",
        data=csv_clusters,
        file_name="clustered_data.csv",
        mime="text/csv"
    )

    # Centroids (in original scale)
    centroids = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids)
    centroid_df = pd.DataFrame(centroids_original, columns=features)
    centroid_df.index = [f"Cluster {i}" for i in range(cluster_count)]
    st.markdown("### Centroid Coordinates (Original Scale)")
    st.markdown("Each row below represents the center of a cluster in terms of the original feature values:")
    st.dataframe(centroid_df, use_container_width=True)

    # Provide download for centroids CSV
    csv_centroids = centroid_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Centroid Coordinates CSV",
        data=csv_centroids,
        file_name="centroid_coordinates.csv",
        mime="text/csv"
    )

    # Input new data point
    st.markdown("### Predict Cluster for a New Data Point")
    new_data = {}
    for feature in features:
        new_data[feature] = st.number_input(f"Enter value for {feature}", key=f"new_point_{feature}")

    show_new_point = st.button("Predict Cluster for New Data Point")

    new_point_original = None
    predicted_cluster = None

    if show_new_point and all(f in new_data for f in features):
        new_point_df = pd.DataFrame([new_data])
        new_scaled = scaler.transform(new_point_df)
        predicted_cluster = kmeans.predict(new_scaled)[0]
        new_point_original = new_point_df[features].values[0]

        st.success(f"The new data point belongs to cluster {predicted_cluster}")

        # Show new data in table
        result_df = new_point_df.copy()
        result_df["Predicted Cluster"] = predicted_cluster
        st.markdown("#### New Data Point Assignment")
        st.dataframe(result_df)

        # Append to history in session state
        st.session_state[MODEL_KEY]["history"].append({
            **new_data,
            "Predicted Cluster": predicted_cluster
        })

    # Visualization selection for scatter plot (2D)
    if len(features) > 2:
        st.markdown("### Select Features to Visualize (2D)")
        x_axis = st.selectbox("X-axis", features, index=0)
        y_axis = st.selectbox("Y-axis", [f for f in features if f != x_axis], index=0)
    else:
        x_axis, y_axis = features[0], features[1]

    # Plot clusters with centroids and new point
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=data_with_clusters[x_axis],
        y=data_with_clusters[y_axis],
        hue=data_with_clusters["Cluster"],
        palette="tab10",
        ax=ax
    )
    x_idx, y_idx = features.index(x_axis), features.index(y_axis)
    ax.scatter(
        centroids_original[:, x_idx],
        centroids_original[:, y_idx],
        s=100, c='black', marker='X', label='Centroids'
    )
    if new_point_original is not None:
        ax.scatter(
            new_point_original[x_idx],
            new_point_original[y_idx],
            s=150, c='red', marker='D', label='New Point'
        )

    plt.title("Cluster Visualization with Centroids and New Point")
    plt.legend()
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    st.download_button(
    label="📥 Download Cluster Plot as PNG",
    data=buf,
    file_name="cluster_plot.png",
    mime="image/png"
)

    # Show performance metric
    st.markdown("### Clustering Performance")
    st.info(f"**Inertia (WCSS)**: {kmeans.inertia_:.2f}")

    # Educational info
    st.markdown("### What is K-Means Clustering?")
    st.info("""
    - **K-Means** is an unsupervised machine learning algorithm used for grouping data into `K` clusters.
    - It works by minimizing the distance between points and their respective cluster centers (centroids).
    - The algorithm uses the **inertia** (within-cluster sum of squares) to evaluate performance.
    - Choose features carefully for meaningful clustering results.
    """)

    # Show prediction history table and download button
    if st.session_state[MODEL_KEY]["history"]:
        st.markdown("---")
        st.subheader("Prediction History for New Data Points")
        history_df = pd.DataFrame(st.session_state[MODEL_KEY]["history"])
        st.dataframe(history_df, use_container_width=True)

        csv_history = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prediction History CSV",
            data=csv_history,
            file_name="kmeans_prediction_history.csv",
            mime="text/csv"
        )

    # Reset Session button
    if st.button("🔄 Reset Clustering Session"):
        st.session_state.pop(MODEL_KEY, None)
        st.experimental_rerun()
