import json
import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
from compare_clustering_solutions import evaluate_clustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import umap 
import hdbscan

def find_reps_for_cluster(clusters, num_reps):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    kmeans = KMeans(n_clusters=num_reps, n_init=5)
    for cluster in clusters: 
        data = [request for request in cluster["requests"]]
        embeddings = model.encode(data)
        kmeans.fit(embeddings)
        kmeans.predict(embeddings)     
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        cluster["representative_sentences"].extend([data[idx] for idx in closest])   


def suggest_topic(cluster):
    requests = [request for request in cluster["requests"]]
    stopwords = ['me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
    possible_pos_tags = {('VBN', 'IN', 'NN'), ('VB', 'TO', 'DT', 'NN'), ('JJ', 'VBP', 'DT'), ('VB', 'IN', 'DT', 'NN'), ('VB', 'DT', 'NN', 'NN'), ('VBN', 'IN', 'DT', 'NN'), ('WP', 'PRP', 'VBP', '.'),('WP','PRP','VBP'), ('VB', 'RB', 'TO', 'NN'), ('RB', 'NNS'), ('JJ', 'NNS'), ('VBN', 'NN'), ('NN', 'TO', 'DT', 'NNS'), ('VBN', 'TO', 'NN'), ('NN', 'NN'), ('VB', 'NNS'), ('NN', 'IN', 'NN'), ('NNS', 'NN'), ('VB', 'DT', 'NN'), ('NN', 'NNS'), ('VBG', 'NN')}
    c_vec = CountVectorizer(ngram_range=(2,4))

    # matrix of ngrams
    ngrams = c_vec.fit_transform(requests)

    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_    

    # sort ngrams by their respective count
    ngram_sorted = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
 
    # heading is prioritized based on num of appearnces in requests and length of the topics - prefer longer topic names over shorter
    heading = None  # holds the current best heading 
    for count, ngram in ngram_sorted:
        if heading is not None and float(count)/heading[0] < 0.7:
            print(f'chosen heading "{heading[1]}" | existed after seeing "{ngram}" count ratio {float(count)/heading[0]}')
            return

        tokenized_ngram = nltk.word_tokenize(ngram)
        word_count = len(tokenized_ngram)
        pos_tags = [tagging[1] for tagging in nltk.pos_tag(tokenized_ngram)]
        if tuple(pos_tags) in possible_pos_tags:    
            if heading is None:     
                print(f'possible heading "{ngram}" counted {count} times. Tokens:{tokenized_ngram} POS tags: {pos_tags}')
                cluster["cluster_name"] = ngram        
                heading = (count,ngram,word_count)
            elif word_count > heading[2]: 
                print(f'heading replaced with "{ngram}" counted {count} times. Tokens:{tokenized_ngram} POS tags: {pos_tags}')
                cluster["cluster_name"] = ngram        
                heading = (count,ngram,word_count)

    # heading was not matched to POS tagging              
    if heading is None: 
        #print(ngram_sorted[:25])        
        ngram_sorted.sort(key = lambda x : len(x[1].split()), reverse=True)
        #print(ngram_sorted[:25])
        cluster["cluster_name"] = ngram_sorted[0][1]   
        print('heading was not found by common method heading=', cluster["cluster_name"])        



model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    df = pd.read_csv(data_file)
    data = df["request"].to_numpy()
  
   
    embeddings = model.encode(data)
    embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, 
                            metric='cosine').fit_transform(embeddings)

   
    print('embeddings:',embeddings.shape)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_size),
                         metric='euclidean',                      
                         cluster_selection_method='eom').fit(embeddings)

    #algorithm used: 
    #clusterer = DBSCAN(eps=6, min_samples=int(min_size)).fit(embeddings)
    labels = clusterer.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    
    print("Estimated number of clusters: %d" % n_clusters)
    print("Estimated number of noise points: %d" % n_outliers)

    clusters = [{"cluster_name": f'cluster{i+1}', "representative_sentences" : [], "requests" : []} for i in range(n_clusters)]
    for i,label in enumerate(labels): 
        if label == -1:
            continue 
        clusters[label]['requests'].append(data[i])
    
    # finding representative requests for each cluster
    find_reps_for_cluster(clusters, int(num_rep)) 

    # suggest topic for cluster 
    for cluster in clusters: 
        suggest_topic(cluster)

    # create list of unclustered 
    unclustered = [data[i] for i,label in enumerate(labels) if label == -1]

    # create JSON output and write to file 
    outputObj = {"cluster_list" : clusters, "unclustered" : unclustered}
    json_object = json.dumps(outputObj, indent=4)
    with open(output_file, "w+") as outfile:
        outfile.write(json_object)  
  

if __name__ == '__main__':
    with open('./Project/config.json', 'r') as json_file:
        config = json.load(json_file)

    # find pos-tagging for topics 
    tags = set()    
    with open(config['example_solution_file'], 'r') as output: 
        solution = json.load(output)
        for cluster in solution["cluster_list"]:
            clust_name = cluster["cluster_name"]
            pos_tags = tuple([tagging[1] for tagging in nltk.pos_tag(nltk.word_tokenize(clust_name))])
            if pos_tags not in tags: 
                print(pos_tags,clust_name)  
            tags.add(pos_tags)
                
    print(f'number of different pos-tagging found in solution {len(tags)}')          

    # cluster unrecognized requests to chatbots and analyze
    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])


    evaluate_clustering(config['example_solution_file'], config['output_file'])
