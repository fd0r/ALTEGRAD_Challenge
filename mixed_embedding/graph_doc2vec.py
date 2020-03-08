import csv
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import make_scorer
from utils import loss_function, visualize_embeddings, load_data, make_vocab, documents_to_idx, clean_documents
from graph_embedding import DeepWalk, Node2Vec
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
from joblib import dump, load
from utils import get_raw_data

def warn(*args, **kwargs):
    pass

import os
print(os.getcwd())

warnings.warn = warn

train_data, test_data, y_train, G, train_hosts, test_hosts = get_raw_data()

print(G.number_of_nodes())
print(G.number_of_edges())


# Node embeddings
make_embeddings = False
n_features = 256
n_walks = 150
walk_length = 100
p = 0.
if make_embeddings:
    embedder = DeepWalk(walk_length, n_walks, p, n_features, training_method=1, window=4, verbose=True)
    embedder.fit(G, save_path='graph_embedding/models/deepwalk.model')
    
else:
    embedder = DeepWalk(walk_length, n_walks, p, n_features, load_path='graph_embedding/models/deepwalk.model')

_visualize_embeddings = False

if _visualize_embeddings:
    visualize_embeddings(embedder.model, 500, n_features)


doc_embeddings_train = np.load("doc2vec_xtrain.pkl.npy")
X_train = embedder.transform(train_hosts)
print(X_train.shape, doc_embeddings_train.shape)
X_train = np.concatenate((X_train, doc_embeddings_train), axis=1)  
print(X_train.shape)

doc_embeddings_test = np.load("doc2vec_xtest.pkl.npy")
X_test = embedder.transform(test_hosts)
print(X_test.shape, doc_embeddings_test.shape)
X_test = np.concatenate((X_test, doc_embeddings_test), axis=1)
print(X_test.shape)

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

label_bins = LabelBinarizer()
y_values = label_bins.fit_transform(y_train)
class_weights = y_values.sum(axis=0)/len(y_values)
print('Proportions of classes in the training set : ', dict(zip(list(label_bins.classes_), class_weights.tolist())))


### Grid search for the GLM
# Parameters values 
solvers = [
    'lbfgs',
    'newton-cg',
    'liblinear',
]

all_losses = ['ovr', 'multinomial']
multi_classes = {
    'lbfgs': all_losses,
    'newton-cg': all_losses,
    'liblinear': ['ovr'],
}

penalties = {
    'lbfgs': ['l2'],
    'newton-cg': ['l2'],
    'liblinear': ['l2', 'l1'],
}

# max_iters = [5, 10, 30, 50]

# tols = [5e-1, 1e-1, 1e-2]

# Cs = [0.7, 0.5]

max_iters = [50, 300, 800, 1500]

tols = [1e-1, 1e-2, 1e-3]

Cs = [1, 1e-1, 5e-2, 1e-2]


multi_classes_and_penalties = {}
for solver in solvers:
    multi_classes_and_penalties[solver] = []
    for multi_classe in multi_classes[solver]:
        for penalty in penalties[solver]:
            multi_classes_and_penalties[solver].append((multi_classe, penalty))

classifiers = []
for solver in solvers:
    for multi_classe, penalty in multi_classes_and_penalties[solver]:
        for max_iter in max_iters:
            for tol in tols:
                for C in Cs:
                    classifiers.append(CalibratedClassifierCV(
                        LogisticRegression(
                        penalty=penalty,
                        tol=tol, 
                        C=C,
                        solver=solver,
                        max_iter=max_iter,
                        multi_class=multi_classe), cv=7))


print(len(classifiers))


### Catch exception to use best model if interrupted
scores = {}
output = {}
try:
    for i, clf in enumerate(tqdm(classifiers)):
        output = cross_validate(clf,
                                X_train,
                                y_train,
                                scoring=make_scorer(loss_function, needs_proba=True),
                                return_train_score=True)
        scores[i] = {'train_score': output['train_score'].mean(), 'score': output['test_score'].mean()}

    best_clf_index = np.argmin(list(map(lambda x: x['score'], list(scores.values()))))

    print(best_clf_index)
    clf = clone(classifiers[best_clf_index])
    print(clf)
    print(scores[best_clf_index])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    with open('../benchmark_graph.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            lst = clf.classes_.tolist()
            lst.insert(0, "Host")
            writer.writerow(lst)
            for i, test_host in enumerate(test_hosts):
                lst = y_pred[i, :].tolist()
                lst.insert(0, test_host)
                writer.writerow(lst)

except:
    best_clf_index = np.argmin(list(map(lambda x: x['score'], list(scores.values()))))

    print(best_clf_index)
    clf = clone(classifiers[best_clf_index])
    print(clf)
    print(scores[best_clf_index])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    with open('../benchmark_graph.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = clf.classes_.tolist()
        lst.insert(0, "Host")
        writer.writerow(lst)
        for i, test_host in enumerate(test_hosts):
            lst = y_pred[i, :].tolist()
            lst.insert(0, test_host)
            writer.writerow(lst)
