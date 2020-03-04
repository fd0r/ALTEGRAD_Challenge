
import csv
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from visualization import visualize_embeddings
from deepwalk import deepwalk

with open("../../train_noduplicates.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())

# Read test data
with open("../../test.csv", 'r') as f:
    test_hosts = f.read().splitlines()

# Create a directed, weighted graph
G = nx.read_weighted_edgelist('../../edgelist.txt', create_using=nx.DiGraph())

print(G.number_of_nodes())
print(G.number_of_edges())

make_embeddings = True
if make_embeddings:
    n_features = 128
    n_walks = 50
    walk_length = 20
    embeddings = deepwalk(G, n_walks, walk_length, n_features)
    _visualize_embeddings = True
    _save_embeddings = True
    X_train = np.zeros((len(train_hosts), n_features))
    notinwalks = []
    notinwalks_children = []
    for i in range(len(train_hosts)):
        try:
            X_train[i] = embeddings[train_hosts[i]]
        except KeyError:
            notinwalks.append(train_hosts[i])
            notinwalks_children.append(G[train_hosts[i]])

    print(notinwalks)
    print(notinwalks_children)
    print(len(notinwalks))
    notinwalks_parents = []
    for node in notinwalks:
        for parent in G.nodes():
            if node in G[str(parent)].keys():
                notinwalks_parents.append(G[str(parent)][node])
    print(notinwalks_parents)

    if _visualize_embeddings:
        visualize_embeddings(embeddings, 500, n_features)

    if _save_embeddings:
        embeddings.save('embeddings.model')
else:
    embeddings = Word2Vec.load('embeddings.model')
