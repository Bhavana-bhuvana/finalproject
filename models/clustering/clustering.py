import streamlit as st
from models.clustering.kmeans_clustering import kmeans_clustering_page
# from models.clustering.dbscan_clustering import dbscan_clustering_page
# from models.clustering.gmm_clustering import gmm_clustering_page
# from models.clustering.hierarchical_clustering import hierarchical_clustering_page

def clustering_page(model_name, data):
    # st.subheader("Select Features for Clustering")
    # features = st.multiselect("Select feature columns (X):", options=data.columns)

    # if not features:
    #     st.warning("Please select feature(s) for clustering.")
    #     return

    # X = data[features]

    if model_name == "K-Means":
      # n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10)
        kmeans_clustering_page(data)
    # elif model_name == "DBSCAN":
    #     eps = st.slider("Select epsilon (eps)", min_value=0.1, max_value=5.0, step=0.1)
    #     min_samples = st.slider("Select min samples", min_value=2, max_value=10)
    #     dbscan_clustering_page(X, eps, min_samples)
    # elif model_name == "Gaussian Mixture Model":
    #     n_components = st.slider("Select number of components", min_value=2, max_value=10)
    #     gmm_clustering_page(X, n_components)
    # elif model_name == "Hierarchical":
    #     n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10)
    #    hierarchical_clustering_page(X, n_clusters)
    else:
        st.error("Unsupported Clustering Model")




































# import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# def clustering_page(data):
#     st.subheader(" Clustering Analysis")

#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     features = st.multiselect("Select features for clustering:", numeric_cols)

#     if len(features) < 2:
#         st.warning("Select at least two features.")
#         return

#     cluster_count = st.slider("Number of clusters", 2, 10, 3)
#     X = data[features]
#     X_scaled = StandardScaler().fit_transform(X)

#     kmeans = KMeans(n_clusters=cluster_count)
#     cluster_labels = kmeans.fit_predict(X_scaled)

#     data_with_clusters = data.copy()
#     data_with_clusters["Cluster"] = cluster_labels

#     st.dataframe(data_with_clusters)

#     st.success(f"Formed {cluster_count} clusters!")

#     centroids = kmeans.cluster_centers_
#     centroids_original = StandardScaler().fit(X).inverse_transform(centroids)

#     fig, ax = plt.subplots()
#     sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=cluster_labels, palette="tab10", ax=ax)
#     plt.scatter(
#         centroids_original[:, 0], centroids_original[:, 1],
#         s=200, c='black', marker='X', label='Centroids'
#     )
#     plt.title("Cluster Visualization with Centroids")
#     plt.legend()
#     st.pyplot(fig)
#********************************************************************************************************
#     st.subheader(" Clustering Analysis")

#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     features = st.multiselect("Select features for clustering:", numeric_cols)

#     if len(features) < 2:
#         st.warning("Select at least two numeric features.")
#         return

#     cluster_count = st.slider("Number of clusters", 2, 10, 3)
#     # Prepare and scale data
#     X = data[features]
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Fit KMeans
#     kmeans = KMeans(n_clusters=cluster_count, random_state=42)
#     cluster_labels = kmeans.fit_predict(X_scaled)
#     data_with_clusters = data.copy()
#     data_with_clusters["Cluster"] = cluster_labels

#     st.markdown("###  Clustered Data")
#     st.dataframe(data_with_clusters)

#     st.success(f"Formed {cluster_count} clusters!")

#     # Centroids (in original scale)
#     centroids = kmeans.cluster_centers_
#     centroids_original = scaler.inverse_transform(centroids)
#     centroid_df = pd.DataFrame(centroids_original, columns=features)
#     centroid_df.index = [f"Cluster {i}" for i in range(cluster_count)]
#     st.markdown("###  Centroid Coordinates (Original Scale)")
#     st.markdown("Each row below represents the center of a cluster in terms of the original feature values:")
#     st.dataframe(centroid_df)

#     # Input new data point
#     st.markdown("###  Predict Cluster for a New Data Point")
#     new_data = {}
#     for feature in features:
#         new_data[feature] = st.number_input(f"Enter value for {feature}", key=feature)

#     show_new_point = st.button("Predict Cluster for New Data Point")

#     new_point_df = pd.DataFrame()
#     new_point_original = None
#     predicted_cluster = None

#     if show_new_point and all(f in new_data for f in features):
#         new_point_df = pd.DataFrame([new_data])
#         new_scaled = scaler.transform(new_point_df)
#         predicted_cluster = kmeans.predict(new_scaled)[0]
#         new_point_original = new_point_df[features].values[0]

#         st.success(f"The new data point belongs to cluster {predicted_cluster}")

#         # Show new data in table
#         result_df = new_point_df.copy()
#         result_df["Predicted Cluster"] = predicted_cluster
#         st.markdown("####  New Data Point Assignment")
#         st.dataframe(result_df)
#     if len(features) > 2:
#         st.markdown("###  Select Features to Visualize (2D)")
#         x_axis = st.selectbox("X-axis", features, index=0)
#         y_axis = st.selectbox("Y-axis", [f for f in features if f != x_axis], index=0)
#     else:
#         x_axis, y_axis = features[0], features[1]
#     # Plot clusters with centroids and new point
#     fig, ax = plt.subplots()
#     sns.scatterplot(
#     x=data_with_clusters[x_axis],
#     y=data_with_clusters[y_axis],
#     hue=data_with_clusters["Cluster"],
#     palette="tab10",
#     ax=ax
#     )
# # Update centroid plotting accordingly
#     x_idx, y_idx = features.index(x_axis), features.index(y_axis)
#     ax.scatter(
#     centroids_original[:, x_idx],
#     centroids_original[:, y_idx],
#     s=100, c='black', marker='X', label='Centroids'
#     )
#     if new_point_original is not None:
#         ax.scatter(
#             new_point_original[x_idx],
#             new_point_original[y_idx],
#             s=150, c='red', marker='D', label='New Point'
#     )



#     plt.title("Cluster Visualization with Centroids and New Point")
#     plt.legend()
#     st.pyplot(fig)

#     # Show performance metric
#     st.markdown("###  Clustering Performance")
#     st.info(f"**Inertia (WCSS)**: {kmeans.inertia_:.2f}")

#     # Educational info
#     st.markdown("###  What is K-Means Clustering?")
#     st.info("""
#     - **K-Means** is an unsupervised machine learning algorithm used for grouping data into `K` clusters.
#     - It works by minimizing the distance between points and their respective cluster centers (centroids).
#     - The algorithm uses the **inertia** (within-cluster sum of squares) to evaluate performance.
#     - Choose features carefully for meaningful clustering results.
#     """)
import streamlit as st
from models.clustering.kmeans_clustering import kmeans_clustering_page
from models.clustering.dbscan_clustering import dbscan_clustering_page
from models.clustering.gmm_clustering import gmm_clustering_page
from models.clustering.hierarchical_clustering import hierarchical_clustering_page

def clustering_page(model_name, data):
    # st.subheader("Select Features for Clustering")
    # features = st.multiselect("Select feature columns (X):", options=data.columns)

    # if not features:
    #     st.warning("Please select feature(s) for clustering.")
    #     return

    # X = data[features]

    if model_name == "K-Means":
        kmeans_clustering_page(data)
    elif model_name == "DBSCAN":
        dbscan_clustering_page(data)
    elif model_name == "Gaussian Mixture Model":
        n_components = st.slider("Select number of components", min_value=2, max_value=10)
        gmm_clustering_page(data)
    elif model_name == "Hierarchical":
       hierarchical_clustering_page(data)
    else:
        st.error("Unsupported Clustering Model")




































# import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# def clustering_page(data):
#     st.subheader(" Clustering Analysis")

#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     features = st.multiselect("Select features for clustering:", numeric_cols)

#     if len(features) < 2:
#         st.warning("Select at least two features.")
#         return

#     cluster_count = st.slider("Number of clusters", 2, 10, 3)
#     X = data[features]
#     X_scaled = StandardScaler().fit_transform(X)

#     kmeans = KMeans(n_clusters=cluster_count)
#     cluster_labels = kmeans.fit_predict(X_scaled)

#     data_with_clusters = data.copy()
#     data_with_clusters["Cluster"] = cluster_labels

#     st.dataframe(data_with_clusters)

#     st.success(f"Formed {cluster_count} clusters!")

#     centroids = kmeans.cluster_centers_
#     centroids_original = StandardScaler().fit(X).inverse_transform(centroids)

#     fig, ax = plt.subplots()
#     sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=cluster_labels, palette="tab10", ax=ax)
#     plt.scatter(
#         centroids_original[:, 0], centroids_original[:, 1],
#         s=200, c='black', marker='X', label='Centroids'
#     )
#     plt.title("Cluster Visualization with Centroids")
#     plt.legend()
#     st.pyplot(fig)
#********************************************************************************************************
#     st.subheader(" Clustering Analysis")

#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     features = st.multiselect("Select features for clustering:", numeric_cols)

#     if len(features) < 2:
#         st.warning("Select at least two numeric features.")
#         return

#     cluster_count = st.slider("Number of clusters", 2, 10, 3)
#     # Prepare and scale data
#     X = data[features]
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Fit KMeans
#     kmeans = KMeans(n_clusters=cluster_count, random_state=42)
#     cluster_labels = kmeans.fit_predict(X_scaled)
#     data_with_clusters = data.copy()
#     data_with_clusters["Cluster"] = cluster_labels

#     st.markdown("###  Clustered Data")
#     st.dataframe(data_with_clusters)

#     st.success(f"Formed {cluster_count} clusters!")

#     # Centroids (in original scale)
#     centroids = kmeans.cluster_centers_
#     centroids_original = scaler.inverse_transform(centroids)
#     centroid_df = pd.DataFrame(centroids_original, columns=features)
#     centroid_df.index = [f"Cluster {i}" for i in range(cluster_count)]
#     st.markdown("###  Centroid Coordinates (Original Scale)")
#     st.markdown("Each row below represents the center of a cluster in terms of the original feature values:")
#     st.dataframe(centroid_df)

#     # Input new data point
#     st.markdown("###  Predict Cluster for a New Data Point")
#     new_data = {}
#     for feature in features:
#         new_data[feature] = st.number_input(f"Enter value for {feature}", key=feature)

#     show_new_point = st.button("Predict Cluster for New Data Point")

#     new_point_df = pd.DataFrame()
#     new_point_original = None
#     predicted_cluster = None

#     if show_new_point and all(f in new_data for f in features):
#         new_point_df = pd.DataFrame([new_data])
#         new_scaled = scaler.transform(new_point_df)
#         predicted_cluster = kmeans.predict(new_scaled)[0]
#         new_point_original = new_point_df[features].values[0]

#         st.success(f"The new data point belongs to cluster {predicted_cluster}")

#         # Show new data in table
#         result_df = new_point_df.copy()
#         result_df["Predicted Cluster"] = predicted_cluster
#         st.markdown("####  New Data Point Assignment")
#         st.dataframe(result_df)
#     if len(features) > 2:
#         st.markdown("###  Select Features to Visualize (2D)")
#         x_axis = st.selectbox("X-axis", features, index=0)
#         y_axis = st.selectbox("Y-axis", [f for f in features if f != x_axis], index=0)
#     else:
#         x_axis, y_axis = features[0], features[1]
#     # Plot clusters with centroids and new point
#     fig, ax = plt.subplots()
#     sns.scatterplot(
#     x=data_with_clusters[x_axis],
#     y=data_with_clusters[y_axis],
#     hue=data_with_clusters["Cluster"],
#     palette="tab10",
#     ax=ax
#     )
# # Update centroid plotting accordingly
#     x_idx, y_idx = features.index(x_axis), features.index(y_axis)
#     ax.scatter(
#     centroids_original[:, x_idx],
#     centroids_original[:, y_idx],
#     s=100, c='black', marker='X', label='Centroids'
#     )
#     if new_point_original is not None:
#         ax.scatter(
#             new_point_original[x_idx],
#             new_point_original[y_idx],
#             s=150, c='red', marker='D', label='New Point'
#     )



#     plt.title("Cluster Visualization with Centroids and New Point")
#     plt.legend()
#     st.pyplot(fig)

#     # Show performance metric
#     st.markdown("###  Clustering Performance")
#     st.info(f"**Inertia (WCSS)**: {kmeans.inertia_:.2f}")

#     # Educational info
#     st.markdown("###  What is K-Means Clustering?")
#     st.info("""
#     - **K-Means** is an unsupervised machine learning algorithm used for grouping data into `K` clusters.
#     - It works by minimizing the distance between points and their respective cluster centers (centroids).
#     - The algorithm uses the **inertia** (within-cluster sum of squares) to evaluate performance.
#     - Choose features carefully for meaningful clustering results.
#     """)
