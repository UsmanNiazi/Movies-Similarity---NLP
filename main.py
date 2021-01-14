"""

Author: Usman Khan
"""


from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from tokenize_and_stem import tokenize_and_stem
from Kmeans_Clusters import Create_Cluster

# Import TfidfVectorizer to create TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Import cosine_similarity to calculate similarity of movie plots
from sklearn.metrics.pairwise import cosine_similarity

# Set seed for reproducibility
np.random.seed(5)

# Read in IMDb and Wikipedia movie data (both in same file)
movies_df = pd.read_csv('data/movies.csv')

print("Number of movies loaded: %s " % (len(movies_df)))

# Combine wiki_plot and imdb_plot into a single column
movies_df['plot'] = movies_df['wiki_plot'].astype(str) + "\n" + \
    movies_df['imdb_plot'].astype(str)

# Instantiate TfidfVectorizer object with english stopwords and our own tokenizer from the file tokenize_and_stem.py
# parameters for efficient processing of text

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem,
                                   ngram_range=(1, 3))


# Fit and transform the tfidf_vectorizer with the "plot" of each movie
# to create a vector representation of the plot summaries

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_df["plot"]])

# Create a column cluster to denote the generated cluster for each movie
movies_df["cluster"] = Create_Cluster(5, tfidf_matrix)

# Calculate the similarity distance
similarity_distance = 1 - cosine_similarity(tfidf_matrix)

# Create mergings matrix
mergings = linkage(similarity_distance, method='complete')

# Plot the dendrogram, using title as label column
dendrogram_ = dendrogram(mergings,
                         labels=[x for x in movies_df["title"]],
                         leaf_rotation=90,
                         leaf_font_size=12,
                         )

# Adjust the plot
fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

# Show the plotted dendrogram
plt.show()
