import json
import pandas as pd 
import numpy as np 
from sentence_transformers import SentenceTransformer
from compare_clustering_solutions import evaluate_clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



pca = PCA(2)



def plotClusters(df,label):
    #Getting unique labels 
    u_labels = np.unique(label)
    
    #plotting the results    
    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
    plt.legend()
    plt.show()

# todo: implement this function
#  you are encouraged to break the functionality into multiple functions,
#  but don't split your code into multiple *.py files
#
#  todo: the final outcome is the json file with clustering results saved as output_file
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    df = pd.read_csv(data_file)
    data = df["request"].to_numpy()
    embeddings = model.encode(data)
    
    embeddings = pca.fit_transform(embeddings)
    
    limit = int((embeddings.shape[0]//2)**0.5)
    print('limit =', limit)
    for k in range(limit, limit+1):
        kmeans = KMeans(n_clusters=k, n_init=5)
        kmeans.fit(embeddings)
        pred = kmeans.predict(embeddings)
        score = silhouette_score(embeddings, pred)
        print('Silhouette Score for k = {}: {:<.3f}'.format(k, score))
        plotClusters(embeddings, pred)

    


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
