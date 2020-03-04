import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from karateclub.node_embedding.structural import Role2Vec, GraphWave
from karateclub.node_embedding.neighbourhood import DeepWalk, GraRep, Walklets, Diff2Vec, NodeSketch, NetMF, BoostNE, HOPE, LaplacianEigenmaps, NMFADMM
from karateclub.node_embedding.meta import NEU
from karateclub.node_embedding.attributed import MUSAE, BANE, TENE, TADW, FSCNMF, SINE
#from karateclub import Deepwalk, GreRep


def loss_function(y_true, predictions, clf):
    y_values = np.zeros((len(y_true), len(clf.classes_)))
    for i, label in enumerate(y_true):
        y_values[i, np.where(clf.classes_ == label)] = 1
    return - 1 / len(y_true) * np.sum(y_values * np.log(predictions + np.finfo(np.float32).eps))

with open("../../train_noduplicates.csv", 'r') as f:
    train_data = f.read().splitlines()

hosts = list()
y = list()
for row in train_data:
    host, label = row.split(",")
    hosts.append(host)
    y.append(label.lower())

G = nx.read_weighted_edgelist('../../edgelist.txt', create_using=nx.DiGraph())

train_hosts, test_hosts, y_train_labels, y_test_labels = train_test_split(hosts, y, test_size=0.2)

models = [
    Role2Vec(walk_number=50, walk_length=80, dimensions=128, workers=8, window_size=2,
            epochs=1, learning_rate=0.05, down_sampling=0.0001, min_count=10, wl_iterations=2),
    #GraphWave(sample_number=100),
    DeepWalk(walk_number=50, walk_length=30, dimensions=128, workers=8,
             window_size=8, epochs=5, learning_rate=0.05, min_count=0),
    #Walklets(),
    #Diff2Vec(),
    #NodeSketch(),
    #NetMF(),
    #BoostNE(),
    #GraRep(),
    HOPE(dimensions=128),
    #LaplacianEigenmaps(),
    #NMFADMM(),
    #NEU(),
    # MUSAE(),
    # BANE(),
    # TENE(),
    # TADW(),
    # FSCNMF(),
    # SINE()
]

scores_train = {}
scores = {}
for model in tqdm(models):
    model.fit(G)
    embeddings = model.get_embedding()

    X_train = np.zeros((len(train_hosts), model.dimensions))
    for i in range(len(train_hosts)):
        X_train[i] = embeddings[int(train_hosts[i])]
    
    X_test = np.zeros((len(test_hosts), model.dimensions))
    for i in range(len(test_hosts)):
        X_test[i] = embeddings[int(test_hosts[i])]

    clf = LogisticRegression(C=1e-2, max_iter=4000)
    clf.fit(X_train, y_train_labels)

    y_train = np.zeros((len(y_train_labels), len(clf.classes_)))
    y_test = np.zeros((len(y_test_labels), len(clf.classes_)))
    for i, label in enumerate(y_train_labels):
        y_train[i, list(clf.classes_).index(label)] = 1
    for i, label in enumerate(y_test_labels):
        y_test[i, list(clf.classes_).index(label)] = 1

    scores_train[model.__class__.__name__] = loss_function(
        y_train, clf.predict_proba(X_train))
    scores[model.__class__.__name__] = loss_function(
        y_test, clf.predict_proba(X_test))

print(scores_train)
print(scores)
