import json
import pandas as pd 
import numpy as np 
from sentence_transformers import SentenceTransformer
from compare_clustering_solutions import evaluate_clustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics


# class Cluster: 
#     def __init__(self, id):
#         self.cluster_name = f'cluster{id}' 
#         self.representative_sentences = [] 
#         self.requests = [] 


# def write_output_file(clusters):
#     #cluster_dicts = [cluster.__dict__ for cluster in clusters]
#     json_object = json.dumps(clusters.__dict__, indent=4)
#     # Writing to sample.json
#     print(clusters[0].__dict__)
#     with open("dataset1-clustering-min-size-10.json", "w") as outfile:
#         for cluster in clusters: 
#             clusterObj = {item for item in cluster.__dict__}
#             outfile.write(clusterObj)

def plotClusters(df,label):
    #Getting unique labels 
    u_labels = np.unique(label)    
    #plotting the results    
    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
    plt.legend()
    plt.show()


def find_reps_for_cluster(clusters):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    kmeans = KMeans(n_clusters=3, n_init=5)
    for cluster in clusters: 
        data = [request for request in cluster["requests"]]
        embeddings = model.encode(data)
        kmeans.fit(embeddings)
        pred = kmeans.predict(embeddings)       

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        cluster["representative_sentences"].extend([data[idx] for idx in closest])        





model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    df = pd.read_csv(data_file)
    data = df["request"].to_numpy()
    embeddings = model.encode(data)


    print('shape of embeddings:',embeddings.shape)

    #algorithm used: 
    db = DBSCAN(eps=5, min_samples=8).fit(embeddings)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print("Estimated number of clusters: %d" % n_clusters)
    print("Estimated number of noise points: %d" % n_noise_)

    clusters = [{"cluster_name": f'cluster{i+1}', "representative_sentences" : [], "requests" : []} for i in range(n_clusters)]
    for i,label in enumerate(labels): 
        if label == -1:
            continue 
        clusters[label]['requests'].append(data[i])

    find_reps_for_cluster(clusters)    

    unclustered = [data[i] for i,label in enumerate(labels) if label == -1]
    outputObj = {"cluster_list" : clusters, "unclustered" : unclustered}
    json_object = json.dumps(outputObj, indent=4)
    with open(output_file, "w+") as outfile:
        outfile.write(json_object)
    

    evaluate_clustering(config['example_solution_file'], config['output_file'])  # invocation example
 
    # for testing / plotting 
    # pca = PCA(4)         # dimensionality reduction 
    # embeddings = pca.fit_transform(embeddings)    
    # print(embeddings.shape)

    # limit = int((embeddings.shape[0]//2)**0.5)
    # print('limit =', limit)
    # for k in range(10, limit+1):
    #     kmeans = KMeans(n_clusters=k, n_init=5)
    #     kmeans.fit(embeddings)
    #     pred = kmeans.predict(embeddings)
    #     score = silhouette_score(embeddings, pred)
    #     print('Silhouette Score for k = {}: {:<.3f}'.format(k, score))
        

    


if __name__ == '__main__':
    with open('./Project/config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])


    # todo: evaluate your clustering solution against the provided one
    #evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])
