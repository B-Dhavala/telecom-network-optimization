#kmean_clustering.py
from sklearn.cluster import KMeans
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Function to perform KMeans clustering
def kmeans_clustering(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Combined_Label_Encoded']])
    logger.info(f"KMeans clustering completed with {n_clusters} clusters.")
    
    for cluster_id in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster_id]
        combined_labels = cluster_df['Combined_Label'].unique()
        logger.info(f"Cluster {cluster_id}: {sorted(combined_labels)}")

    return kmeans
