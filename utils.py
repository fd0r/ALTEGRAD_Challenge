import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
import codecs
import networkx as nx
from tqdm import tqdm


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

def load_data(path='./', text=False):
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
        filenames = os.listdir(os.path.join(path, 'text/text'))

        print('Loading text documents')
        for filename in tqdm(filenames):
            with codecs.open(os.path.join(path, 'text/text/', filename), encoding='latin-1') as f: 
                documents[filename] = f.read().replace("\n", "").lower()

        for host in train_hosts + test_hosts:
            if not host in documents.keys():
                documents[host] = ''
        
        return train_hosts, test_hosts, y_train, G, documents
    return train_hosts, test_hosts, y_train, G

def get_raw_data(encoding="utf-8"):  #'latin-1'
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
        with codecs.open(path.join('text/text/', filename), encoding=encoding) as f: 
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


def dump_prediction(y_pred, name="baseline"):
    with open('{}.csv'.format(name), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = clf.classes_.tolist()
        lst.insert(0, "Host")
        writer.writerow(lst)
        for i,test_host in enumerate(test_hosts):
            lst = y_pred[i,:].tolist()
            lst.insert(0, test_host)
            writer.writerow(lst)
