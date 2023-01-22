import json
import random as rand
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def create_tfidf_features(corpus):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,3), stop_words="english", max_features=400)
    train_tfidf = vectorizer.fit_transform(corpus).toarray()
    print(train_tfidf.shape)
    return train_tfidf


def sbert_sentence_encoding(corpus):
    MODEL_NAME = 'all-MiniLM-L6-v2'  #other models:  'all-MiniLM-L12-v2' 'all-mpnet-base-v2'
    model = SentenceTransformer(MODEL_NAME)
    encodings = model.encode(corpus)
    return encodings


def k_means(X_train, K, labels, classes=None):
    # random generate centroids
    centroids = []

    # Extending the list of centroids with K random examples from the training set
    centroids.extend(X_train[np.random.choice(
        X_train.shape[0], K, replace=False), :])

    # Creating K empty clusters
    clusters = [[] for _ in range(K)]

    # Assigning a random cluster to each train example
    for x_train, label in zip(X_train, labels):
        rand.choice(clusters).append((x_train, label))

  

    iteration = 0
    max_iter = 20
    # running for maximum of 20 iteration or until convergence of clusters 
    while iteration < max_iter:
        swaps = 0
        iteration += 1
        # Creating new clusters for this iteration
        new_clusters = [[] for _ in range(K)]
        for cluster_id, cluster in enumerate(clusters):
            for x_train, label in cluster:
                min_distance = None
                centroid_id = 0
                for i, centroid in enumerate(centroids):
                    # Calculating the distance between the example and the centroid
                    current_distance = np.linalg.norm(centroid - x_train)

                    # Checking if this distance is the smallest distance seen so far
                    if not min_distance or min_distance > current_distance:
                        centroid_id = i
                        min_distance = current_distance

                # Assigning the example to the closest cluster
                new_clusters[centroid_id].append((x_train, label))
                # Checking if the example was reassigned to a new cluster
                if centroid_id != cluster_id:
                    swaps += 1

        # Clusters converge - we can break out of this loop
        if swaps == 0:
            break

       # Updating the clusters
        clusters = new_clusters

        # recalculate centroids
        for cluster_id, cluster in enumerate(clusters):
            cluster_pts = [x_train for x_train, _ in cluster]
            cluster_array = np.array(cluster_pts, dtype=np.float32)
            centroids[cluster_id] = np.mean(cluster_array, axis=0)

    print(f'clusters converged after {iteration} iterations')
  

    return clusters


def kmeans_cluster_and_evaluate(data_file, encoding_type):
    print(f'starting kmeans clustering and evaluation with {data_file} and encoding {encoding_type}')
    # Reading data file with pandas
    table_data = pd.read_csv(data_file, sep='\t')
    label_train = table_data[table_data.columns[0]]
    classes = label_train.unique()
    corpus = table_data[table_data.columns[1]].to_numpy()

    # K is the number of clusters in the dataset
    K = classes.size
    print(f'dataset has {K} classes. classes = {classes}')

    # Perform feature extraction from sentences and
    train_features = None
    if encoding_type == "SBERT":
        train_features = sbert_sentence_encoding(corpus)       
    else:
        train_features = create_tfidf_features(corpus)
      

    
    # Averaging test results over 10 iterations of testing
    RI = 0
    ARI = 0
    for _ in range(10):
        clusters = k_means(train_features, K, label_train, classes)
        labels_true = [label for cluster in clusters for x, label in cluster]
        labels_pred = [classes[cluster_id] for cluster_id,
                       cluster in enumerate(clusters) for _ in cluster]
        ri = metrics.rand_score(labels_true, labels_pred)
        ari = metrics.adjusted_rand_score(labels_true, labels_pred)
        RI += ri
        ARI += ari
    RI /= 10.0
    ARI /= 10.0

    evaluation_results = {'mean_RI_score':  RI,
                          'mean_ARI_score': ARI}

    return evaluation_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(
        config['data'], config["encoding_type"])

    for k, v in results.items():
        print(k, v)
