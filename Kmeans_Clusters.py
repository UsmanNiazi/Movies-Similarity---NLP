# This file takes these parameters as inputs

# First parameter is the number of clusters and second is the TFIDF Matrix which is the vector representation of plots in the dataset
# Import k-means to perform clusters
from sklearn.cluster import KMeans


def Create_Cluster(number, matrix):
    # Create a KMeans object with 5 clusters and save as km
    km = KMeans(n_clusters=number)

    # Fit the k-means object with tfidf_matrix
    km.fit(matrix)

    clusters = km.labels_.tolist()

    return clusters
