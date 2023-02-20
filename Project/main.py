import json
import pandas as pd 
from sentence_transformers import SentenceTransformer, util
from compare_clustering_solutions import evaluate_clustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def find_reps_for_cluster(clusters, num_reps):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    kmeans = KMeans(n_clusters=num_reps, n_init=3)
    for cluster in clusters: 
        data = [request for request in cluster["requests"]]
        embeddings = model.encode(data)
        kmeans.fit(embeddings)
        kmeans.predict(embeddings)     
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        cluster["representative_sentences"].extend([data[idx] for idx in closest])   

def suggest_topic_updated(cluster, debug=False): 
    # pull out requests from cluster
    requests = [request.strip() for request in cluster["requests"] if len(request.split()) > 1]
    
    # count n-grams in the list of requests 
    c_vec = CountVectorizer(ngram_range=(2,5))
    ngrams = c_vec.fit_transform(requests)
    # counts ngrams
    count_values = ngrams.toarray().sum(axis=0)

    # list of ngrams
    vocab = c_vec.vocabulary_    

  
    # model 
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # fetch repsentative of cluster 
    repsentative_sents = cluster["representative_sentences"]
    reps_embedded = model.encode(repsentative_sents,convert_to_tensor=True)  


    # sort ngrams by their respective count
    ngram_sorted = sorted([(count_values[i],ngram) for ngram,i in vocab.items()], key= lambda req:len(req[1].split()), reverse=True)
    ngram_sorted.sort(key=lambda req : req[0], reverse=True)

    best_score = 0.0 
    heading = None     
    for _,ngram in ngram_sorted[:20]:
        score = 0.0
        ngram_embedded = model.encode(ngram,convert_to_tensor=True)
        for rep_sentence in reps_embedded: 
            score = max(score, util.cos_sim(rep_sentence,ngram_embedded))
            #score += util.cos_sim(rep_sentence,ngram_embedded)
        #score /= len(repsentative_sents)
        if score > best_score:
            heading, best_score = ngram, score      

    if heading is not None:  
        cluster["cluster_name"] = heading 
        if debug:
            print(f'heading "{cluster["cluster_name"]}" was chosen')
    else:           
        cluster["cluster_name"] = ngram_sorted[0][1]   
        if debug:
            print('heading was not found by common method heading=', cluster["cluster_name"])       
   
    

def suggest_topic(cluster, debug=False):
    requests = [request.strip() for request in cluster["requests"] if len(request.split()) > 1]
    stopwords = ['please']
    possible_pos_tags = {('NNS', 'NN'), ('VB', 'PRP', 'DT', 'NN'), ('VB', 'PRP', 'RB', '.'), ('NN', 'TO', 'PRP$', 'NN'), ('WP', 'VBP', 'PRP', 'JJ', 'IN'), ('WRB','JJ','NNS'), ('WRB', 'MD', 'PRP', 'VB', 'PRP'), ('NN', 'PRP$', 'NN'), ('VBN', 'NN'),('VBN', 'NN', 'NN'), ('VBD', 'PRP', 'PRP$', 'NN'), ('WP', 'MD', 'PRP', 'VB'), ('NN', 'NN', 'TO', 'NNS'), ('NN', 'IN', 'DT', 'NN'), ('VBG', 'IN', 'PRP$', 'NN'), ('VBG','IN','RB'), ('NN', 'NNS', 'JJ'), ('VB', 'PRP', 'NNS'), ('VB', 'DT', 'NNS'), ('VB', 'JJR'), ('JJ', 'IN', 'NN'), ('VB', 'TO', 'VB'), ('JJ', 'NN'), ('VBN', 'PRP$', 'NN'), ('VB', 'RP', 'PRP$', 'NN'), ('NN', 'DT', 'NN'), ('VB', 'PRP', 'DT', 'NNS'), ('VB', 'IN', 'PRP$', 'NN'), ('VBG', 'IN', 'MD', 'CD'), ('VBD', 'PRP$', 'NN'), ('NN', 'IN', 'NN'), ('VBG', 'NN'), ('VB', 'WP', 'NN'), ('JJR', 'NN'), ('VB', 'PRP$', 'NN'), ('VBG', 'IN', 'IN', 'NN'), ('VB', 'VBG', 'IN'), ('WP', 'VBZ', 'CD', 'NNS', 'CD'), ('JJ', 'NNS'), ('RB', 'NN'), ('WP', 'VBZ', 'RP', 'IN', 'PRP$', 'NN'), ('VB', 'WRB', 'PRP$', 'NN'), ('PRP', 'VBP', 'VBG', 'RB', 'RB'), ('VBG', 'PRP$', 'NN'), ('WP', 'VBZ', 'CD', 'CD'), ('VBN', 'IN', 'PRP$', 'NN'), ('VBZ', 'NN', 'NN'), ('NN', 'JJ'), ('NN', 'NN'), ('NN', 'IN', 'PRP$', 'NN'), ('VB', 'NN'), ('NN', 'IN', 'NNS'), ('VB', 'PRP', 'PRP$', 'NN'), ('WP', 'VBZ', 'DT', 'NN', 'CD', 'NNS', 'IN', 'RB'), ('VB', 'PRP$', 'NNS'), ('IN', 'DT', 'NNS'), ('VB', 'IN', 'DT', 'NNS'), ('VB', 'PRP', 'NN'), ('VB', 'DT', 'JJ'), ('VB', 'DT', 'NN'), ('NN', 'VBG'), ('VB', 'WRB', 'JJ'), ('NN', 'NNS')}
    c_vec = CountVectorizer(ngram_range=(2,5), stop_words=stopwords)

    # matrix of ngrams
    ngrams = c_vec.fit_transform(requests)

    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_    

    # sort ngrams by their respective count
    ngram_sorted = sorted([(count_values[i],ngram) for ngram,i in vocab.items()], key= lambda req:len(req[1].split()), reverse=True)
    ngram_sorted.sort(key=lambda req : req[0], reverse=True)

    if debug: 
        print(ngram_sorted[:5])

 
    # heading is prioritized based on num of appearnces in requests and length of the topics - prefer longer topic names over shorter
    heading = None  # holds the current best heading 
    for count, ngram in ngram_sorted:
        if heading is not None and float(count)/heading['count'] < 0.65:
            title = heading['title']
            if debug: 
                print(f'heading "{title}" was chosen | existed after seeing "{ngram}" count ratio {float(count)/heading["count"]}')
            return title

        tokenized_ngram = nltk.word_tokenize(ngram)
        word_count = len(tokenized_ngram)
        pos_tags = [tagging[1] for tagging in nltk.pos_tag(tokenized_ngram)]
        if tuple(pos_tags) in possible_pos_tags:    
            if heading is None:     
                cluster['cluster_name'] = ngram        
                heading ={'title': ngram, 'word_count': word_count, 'count': count}
                if debug: 
                    print(f'possible heading "{ngram}" counted {count} times. Tokens:{tokenized_ngram} POS tags: {pos_tags}')

            elif word_count > heading['word_count']: 
                cluster["cluster_name"] = ngram        
                heading = {'title': ngram, 'word_count': word_count, 'count': count}
                if debug: 
                    print(f'heading replaced with "{ngram}" counted {count} times. Tokens:{tokenized_ngram} POS tags: {pos_tags}')


    if heading is None:    
        # heading was not matched to one of the POS-tags           
        cluster["cluster_name"] = ngram_sorted[0][1]   
        if debug: 
            print('heading was not found by common method heading=', cluster["cluster_name"])             

#model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #2nd best
model = SentenceTransformer('all-MiniLM-L12-v2')  # best


def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    df = pd.read_csv(data_file)
    data = df["request"].to_numpy()
    min_size = int(min_size)    
    
    # encode our requests dataset using sentence transformer 
    embeddings = model.encode(data, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    print('shape of embeddings:',embeddings.shape)


    '''
    Two parameters to tune in Community Detection:
    min_cluster_size: Only consider cluster that have at least min_size elements
    threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    '''       
    clusters_output = util.community_detection(embeddings, min_community_size=min_size, threshold=0.66)

    # we add to clustered_pts all the requests that'd been clustered   
    clustered_pts = [] 
    for i, cluster in enumerate(clusters_output):                   
        clustered_pts.extend(cluster)           
    
    # create a list of unclustered requests 
    unclustered = [request for i,request in enumerate(data) if i not in clustered_pts]

    # Number of clusters found 
    n_clusters = len(clusters_output)    

    
    clusters = [{"cluster_name": f'cluster{i+1}', "representative_sentences" : [], "requests" : []} for i in range(n_clusters)]
    for i,cluster in enumerate(clusters_output): 
        # add request data to cluster 
        for request_id in cluster:        
            clusters[i]['requests'].append(data[request_id])
        

    # Part 2 - finding representative requests for each cluster
    find_reps_for_cluster(clusters, int(num_rep))  

    for cluster in clusters: 
        #Part 3 - suggest topic for cluster 
        suggest_topic_updated(cluster,True)      

    # create output JSON object and write to a file 
    outputObj = {"cluster_list" : clusters, "unclustered" : unclustered}
    json_object = json.dumps(outputObj, indent=4)
    with open(output_file, "w+") as outfile:
        outfile.write(json_object)
     



if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)        
    
    # cluster unrecognized requests to chatbots and analyze
    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])


    evaluate_clustering(config['example_solution_file'], config['output_file'])