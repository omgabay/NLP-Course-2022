import json
import pandas as pd 
import numpy as np 
from sentence_transformers import SentenceTransformer, util
from compare_clustering_solutions import evaluate_clustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('punkt')
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, normalize
nltk.download('averaged_perceptron_tagger')


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
    requests = [request.strip() for request in cluster["requests"] if len(request.split()) > 1]
    stopwords = ['please','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
    stopwords = ['please']
    #possible_pos_tags = {('VBN', 'IN', 'NN'), ('VB', 'TO', 'DT', 'NN'), ('JJ', 'VBP', 'DT'), ('VB', 'IN', 'DT', 'NN'), ('VB', 'DT', 'NN', 'NN'), ('VBN', 'IN', 'DT', 'NN'), ('WP', 'PRP', 'VBP', '.'),('WP','PRP','VBP'), ('VB', 'RB', 'TO', 'NN'), ('RB', 'NNS'), ('JJ', 'NNS'), ('VBN', 'NN'), ('NN', 'TO', 'DT', 'NNS'), ('VBN', 'TO', 'NN'), ('NN', 'NN'), ('VB', 'NNS'), ('NN', 'IN', 'NN'), ('NNS', 'NN'), ('VB', 'DT', 'NN'), ('NN', 'NNS'), ('VBG', 'NN')}
    possible_pos_tags = {('NNS', 'NN'), ('VB', 'PRP', 'DT', 'NN'), ('VB', 'PRP', 'RB', '.'), ('NN', 'TO', 'PRP$', 'NN'), ('WP', 'VBP', 'PRP', 'JJ', 'IN'), ('WRB', 'MD', 'PRP', 'VB', 'PRP'), ('NN', 'PRP$', 'NN'), ('VBN', 'NN'), ('VBD', 'PRP', 'PRP$', 'NN'), ('WP', 'MD', 'PRP', 'VB'), ('NN', 'NN', 'TO', 'NNS'), ('NN', 'IN', 'DT', 'NN'), ('VBG', 'IN', 'PRP$', 'NN'), ('NN', 'NNS', 'JJ'), ('VB', 'PRP', 'NNS'), ('VB', 'DT', 'NNS'), ('VB', 'JJR'), ('JJ', 'IN', 'NN'), ('VB', 'TO', 'VB'), ('JJ', 'NN'), ('VBN', 'PRP$', 'NN'), ('VB', 'RP', 'PRP$', 'NN'), ('NN', 'DT', 'NN'), ('VB', 'PRP', 'DT', 'NNS'), ('VB', 'IN', 'PRP$', 'NN'), ('VBG', 'IN', 'MD', 'CD'), ('VBD', 'PRP$', 'NN'), ('NN', 'IN', 'NN'), ('VBG', 'NN'), ('VB', 'WP', 'NN'), ('JJR', 'NN'), ('VB', 'PRP$', 'NN'), ('VBG', 'IN', 'IN', 'NN'), ('VB', 'VBG', 'IN'), ('WP', 'VBZ', 'CD', 'NNS', 'CD'), ('JJ', 'NNS'), ('RB', 'NN'), ('WP', 'VBZ', 'RP', 'IN', 'PRP$', 'NN'), ('VB', 'WRB', 'PRP$', 'NN'), ('PRP', 'VBP', 'VBG', 'RB', 'RB'), ('VBG', 'PRP$', 'NN'), ('WP', 'VBZ', 'CD', 'CD'), ('VBN', 'IN', 'PRP$', 'NN'), ('VBZ', 'NN', 'NN'), ('NN', 'JJ'), ('NN', 'NN'), ('NN', 'IN', 'PRP$', 'NN'), ('VB', 'NN'), ('NN', 'IN', 'NNS'), ('VB', 'PRP', 'PRP$', 'NN'), ('WP', 'VBZ', 'DT', 'NN', 'CD', 'NNS', 'IN', 'RB'), ('VB', 'PRP$', 'NNS'), ('IN', 'DT', 'NNS'), ('VB', 'IN', 'DT', 'NNS'), ('VB', 'PRP', 'NN'), ('VB', 'DT', 'JJ'), ('VB', 'DT', 'NN'), ('NN', 'VBG'), ('VB', 'WRB', 'JJ'), ('NN', 'NNS')} 
    c_vec = CountVectorizer(ngram_range=(2,4), stop_words=stopwords)

    # matrix of ngrams
    ngrams = c_vec.fit_transform(requests)

    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_    

    # sort ngrams by their respective count
    ngram_sorted = sorted([(count_values[i],ngram) for ngram,i in vocab.items()], key= lambda req:len(req[1].split()), reverse=True)
    ngram_sorted.sort(key=lambda req : req[0], reverse=True)
    print(ngram_sorted[:5])
 
    # heading is prioritized based on num of appearnces in requests and length of the topics - prefer longer topic names over shorter
    heading = None  # holds the current best heading 
    for count, ngram in ngram_sorted:
        if heading is not None and float(count)/heading[0] < 0.7:
            #print(f'chosen heading "{heading[1]}" | existed after seeing "{ngram}" count ratio {float(count)/heading[0]}')
            return

        tokenized_ngram = nltk.word_tokenize(ngram)
        word_count = len(tokenized_ngram)
        pos_tags = [tagging[1] for tagging in nltk.pos_tag(tokenized_ngram)]
        if tuple(pos_tags) in possible_pos_tags:    
            if heading is None:     
                #print(f'possible heading "{ngram}" counted {count} times. Tokens:{tokenized_ngram} POS tags: {pos_tags}')
                cluster["cluster_name"] = ngram        
                heading = (count,ngram,word_count)
            elif word_count > heading[2]: 
                #print(f'heading replaced with "{ngram}" counted {count} times. Tokens:{tokenized_ngram} POS tags: {pos_tags}')
                cluster["cluster_name"] = ngram        
                heading = (count,ngram,word_count)

    # heading was not matched to POS tagging              
    if heading is None:            
        cluster["cluster_name"] = ngram_sorted[0][1]   
        print('heading was not found by common method heading=', cluster["cluster_name"])             

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    df = pd.read_csv(data_file)
    data = df["request"].to_numpy()
    min_size = int(min_size)
   

    embeddings = model.encode(data, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    #embeddings = embeddings /  np.linalg.norm(embeddings, axis=1, keepdims=True)

    print('shape of embeddings:',embeddings.shape)
    #thresholds = [0.6,0.61,0.62,0.64,0.65,0.66,0.67,0.68,0.69] 
    thresholds = [0.66]
    best_ari = 0 
    best_threshold = None 
    for th in thresholds:  
        #Two parameters to tune:
        #min_cluster_size: Only consider cluster that have at least 25 elements
        #threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
        print("================================================================")
        print(f"Running community detection with threshold {th}")
        clusters_output = util.community_detection(embeddings, min_community_size=min_size, threshold=th)

        clustered_pts = [] 
        for i, cluster in enumerate(clusters_output):           
            clustered_pts.extend(cluster)           
        
        unclustered = [request for i,request in enumerate(data) if i not in clustered_pts]

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(clusters_output)
        n_outliers = len(unclustered)
    
        
        print("Estimated number of clusters: %d" % n_clusters)
        print("Estimated number of noise points: %d" % n_outliers)

        clusters = [{"cluster_name": f'cluster{i+1}', "representative_sentences" : [], "requests" : []} for i in range(n_clusters)]
        for i,cluster in enumerate(clusters_output): 
            for request_id in cluster:        
                clusters[i]['requests'].append(data[request_id])
        #suggest topic for cluster 
        for cluster in clusters: 
            suggest_topic(cluster)        

        # finding representative requests for each cluster
        find_reps_for_cluster(clusters, int(num_rep))  

        # create output JSON object and write to a file 
        outputObj = {"cluster_list" : clusters, "unclustered" : unclustered}
        json_object = json.dumps(outputObj, indent=4)
        with open(output_file, "w+") as outfile:
            outfile.write(json_object)

        ri, ari = evaluate_clustering(config['example_solution_file'], config['output_file'])
        if ari > best_ari: 
            best_ari = ari 
            best_threshold = th
    print("============================") 
    print(f'Best result with threshold {best_threshold}')      


if __name__ == '__main__':
    with open('./Project/config.json', 'r') as json_file:
        config = json.load(json_file)

    # find pos-tagging for topics 
    tags = set()    
    with open(config['example_solution_file'], 'r') as output: 
        solution = json.load(output)
        for cluster in solution["cluster_list"]:
            clust_name = cluster["cluster_name"]
            pos_tags = [tagging[1] for tagging in nltk.pos_tag(nltk.word_tokenize(clust_name))]
            tags.add(tuple(pos_tags))
            print(pos_tags,clust_name)  
                
    print(f'number of different pos-tagging found in solution {len(tags)}')
    print('{',f'{tags}','}')          

    # cluster unrecognized requests to chatbots and analyze
    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])


    #evaluate_clustering(config['example_solution_file'], config['output_file'])