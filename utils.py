import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
import codecs
import networkx as nx
from tqdm import tqdm
from bs4 import BeautifulSoup
from bs4.element import Comment
from markdown import markdown
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
import re
from collections import Counter
import signal
from contextlib import contextmanager
import json
import operator
import mistune
import ijson


def loss_function(y_true, predictions):
    label_bins = LabelBinarizer()
    y_values = label_bins.fit_transform(y_true)
    return - 1 / len(y_true) * np.sum(y_values * np.log(predictions + np.finfo(np.float32).eps))
    


def visualize_embeddings(embeddings, n, n_features):

    nodes = embeddings.wv.index2entity[:n]
    vecs = np.empty(shape=(n, n_features))
    for i in range(n):
        vecs[i] = embeddings.wv[nodes[i]]

    my_pca = PCA(n_components=20)
    my_tsne = TSNE(n_components=2)

    vecs_pca = my_pca.fit_transform(vecs)
    vecs_tsne = my_tsne.fit_transform(vecs_pca)

    fig, ax = plt.subplots()
    ax.scatter(vecs_tsne[:, 0], vecs_tsne[:, 1], s=3)
    for x, y, node in zip(vecs_tsne[:, 0], vecs_tsne[:, 1], nodes):
        ax.annotate(node, xy=(x, y), size=8)
    fig.suptitle('t-SNE visualization of node embeddings', fontsize=30)
    fig.set_size_inches(20, 15)
    plt.show()

def load_data(path='./', text=False, max_time_per_doc=30):
    with open(os.path.join(path, 'train_noduplicates.csv'), 'r') as f:
        train_data = f.read().splitlines()

    train_hosts = list()
    y_train = list()
    for row in train_data:
        host, label = row.split(",")
        train_hosts.append(host)
        y_train.append(label.lower())

    labels = np.unique(y_train)

    # Read test data
    with open(os.path.join(path, 'test.csv'), 'r') as f:
        test_hosts = f.read().splitlines()

    # Create a directed, weighted graph
    G = nx.read_weighted_edgelist(os.path.join(path, 'edgelist.txt'), create_using=nx.DiGraph())

    if text==True:
        documents = dict()
        filenames = os.listdir(os.path.join(path, 'text'))

        tokenizer = TweetTokenizer()
        print('Loading text documents')
        for filename in tqdm(filenames):
                # try:
                #     with codecs.open(os.path.join(path, 'text/', filename), encoding='UTF-8') as f:
                #         try:
                #             with time_limit(max_time_per_doc):
                #                 html = mistune.html(f.read())
                #                 print(BeautifulSoup(html))
                #                 documents[filename] = tokenizer.tokenize(re.sub(' +', ' ', text_from_html(html)))
                #         except TimeoutException:
                #             print(filename)
                #             documents[filename] = ['']
            with codecs.open(os.path.join(path, 'text/', filename), encoding='latin-1') as f: 
                try:
                    with time_limit(max_time_per_doc):
                        html = mistune.markdown(f.read())
                        documents[filename] = tokenizer.tokenize(re.sub(' +', ' ', text_from_html(html)))
                except TimeoutException:
                    print(filename)
                    try:
                        with time_limit(max_time_per_doc):
                            documents[filename] = tokenizer.tokenize(f.read().replace("\n", "").strip().lower())
                    except TimeoutException:
                        print(filename)
                        documents[filename] = ['']
        return train_hosts, test_hosts, y_train, G, documents
    return train_hosts, test_hosts, y_train, G

def get_raw_data(encoding="utf-8", errors='ignore'):  #'latin-1'
    # Read training data
    with open("train.csv", 'r') as f:
        train_data = f.read().splitlines()

    train_hosts = list()
    y_train = list()
    for row in tqdm(train_data):
        host, label = row.split(",")
        train_hosts.append(host)
        y_train.append(label.lower())

    # Read test data
    with open("test.csv", 'r') as f:
        test_hosts = f.read().splitlines()

    # Create a directed, weighted graph
    G = nx.read_weighted_edgelist('edgelist.txt', create_using=nx.DiGraph())

    # Load the textual content of a set of webpages for each host into the dictionary "text". 
    # The encoding parameter is required since the majority of our text is french.
    text = dict()
    filenames = os.listdir('text/text')
    for filename in tqdm(filenames):
        with codecs.open(os.path.join('text/text/', filename), encoding=encoding, errors=errors) as f: 
            text[filename] = f.read().replace("\n", "").lower()

    train_data = list()
    for host in tqdm(train_hosts):
        if host in text:
            train_data.append(text[host])
        else:
            train_data.append('')

    # Get textual content of web hosts of the test set
    test_data = list()
    for host in tqdm(test_hosts):
        if host in text:
            test_data.append(text[host])
        else:
            test_data.append('')

    return train_data, test_data, y_train, G, train_hosts, test_hosts


def dump_prediction(clf, y_pred, test_hosts, name="baseline"):
    with open('{}.csv'.format(name), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = clf.classes_.tolist()
        lst.insert(0, "Host")
        writer.writerow(lst)
        for i,test_host in enumerate(test_hosts):
            lst = y_pred[i,:].tolist()
            lst.insert(0, test_host)
            writer.writerow(lst)


def make_vocab(documents, min_freq=15, path_write='../data'):
    tokens = [token for tokens in documents.values() for token in tokens]

    counts = dict(Counter(tokens))
    stop_words = list(stopwords.words('english')) + list(stopwords.words('french'))

    ### fill the gap (filter the dictionary 'counts' by retaining only the words that appear at least 'min_freq' times)
    print('making counts')
    counts = {k:v for k,v in tqdm(counts.items()) if v>=min_freq and k not in stop_words}

    with open(os.path.join(path_write, 'counts_stemmed.json'), 'w') as file:
        json.dump(counts, file, sort_keys=True, indent=4)

    print('counts saved to disk')

    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

    # assign to each word an index based on its frequency in the corpus
    # the most frequent word will get index equal to 1
    # 0 is reserved for out-of-vocabulary words
    word_to_index = dict([(my_tuple[0],idx) for idx,my_tuple in enumerate(sorted_counts,1)])

    with open(os.path.join(path_write, 'vocab_stemmed.json'), 'w') as file:
        json.dump(word_to_index, file, sort_keys=True, indent=4)

    print('vocab saved to disk')
    return word_to_index

def clean_documents(path, nodes, stemmer=PorterStemmer()):

    with open(os.path.join(path, 'vocab.json')) as file:
        word_to_idx = json.load(file)

    idx_to_word = dict((y, x) for x, y in word_to_idx.items())
    documents_tokens = {}

    print('stemming')
    with open(os.path.join(path, 'doc_ints.json')) as file:
        parser = ijson.parse(file)

        for  prefix, event, value in parser:
            if event == 'start_array':
                node = prefix
                documents_tokens[node] = []
            elif event == 'number':
                if value == 0:
                    documents_tokens[node].append('<OOV>')
                else:
                    documents_tokens[node].append(stemmer.stem(idx_to_word[value]))
    
    word_to_idx = make_vocab(documents_tokens)
    documents_to_idx(documents_tokens, word_to_idx)


def documents_to_idx(documents, word_to_idx, oov_idx=0, path_write='../data'):
    documents_ints = {}
    print('making doc to ints')
    for i, doc in tqdm(documents.items()):
        sublist = []
        ### fill the gaps (for the tokens that are not in 'word_to_index', use 'oov_token') ###
        for word in doc:
            if word in word_to_idx.keys():
                sublist.append(word_to_idx[word])
            else:
                sublist.append(oov_idx)
        documents_ints[i] = sublist
    
    with open(os.path.join(path_write, 'doc_ints_stemmed.json'), 'w') as file:
        json.dump(documents_ints, file)
    
    print('documents ints saved to disk')

    return documents_ints


def tag_usefull(element):
    if element.parent.name in ['style', 'script', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_usefull, texts)
    return u" ".join(t.strip() for t in visible_texts)

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class TokenToIndex:
    def __init__(self):
        self.vocab_to_index = dict()
        self.index_to_index = dict()

    def fit(self, X, y=None):
        counter = 0
        for tokens in X:
            for token in tokens:
                pass
        return self

    def transform(self, X, y):
        return X