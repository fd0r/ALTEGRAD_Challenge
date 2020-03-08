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
from graph_models.node_embedding import DeepWalk, Node2Vec
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import warnings
from joblib import dump, load

def warn(*args, **kwargs):
    pass

warnings.warn = warn

train_hosts, test_hosts, y_train, G = load_data()

nodes = [str(node) for node in G.nodes()]
filenames = os.listdir('../data/text')

print([node for node in nodes if node not in filenames])
print([node for node in filenames if node not in nodes])

# word_to_idx = make_vocab(documents)

# documents_ints = documents_to_idx(documents, word_to_idx)

# clean_documents('../data', [str(node) for node in G.nodes()])

# print(max([len(doc) for doc in documents.values()]))
# print(np.mean([len(doc) for doc in documents.values()]))


print(G.number_of_nodes())
print(G.number_of_edges())


# Node embeddings
make_embeddings = Falsepuyt
n_features = 256
n_walks = 150
walk_length = 100
p = 0.
if make_embeddings:
    embedder = DeepWalk(walk_length, n_walks, p, n_features, training_method=1, window=4, verbose=True)
    # embedder = Node2Vec(walk_length, n_walks, n_features, p=5, q=1, verbose=True)
    embedder.fit(G, save_path='graph_models/node_embedding/models/deepwalk.model')
    
else:
    embedder = DeepWalk(walk_length, n_walks, p, n_features, load_path='graph_models/node_embedding/models/deepwalk.model')

_visualize_embeddings = False

if _visualize_embeddings:
    visualize_embeddings(embedder.model, 500, n_features)

# Document embeddings
# n_features_text = 128
# _make_text_embeddings = True
# if _make_text_embeddings:
#     vec = TfidfVectorizer(decode_error='ignore', strip_accents='unicode', encoding='latin-1', min_df=20, max_df=1000)
#     vec.fit(list(documents.values()))
#     doc_embeddings_train = vec.transform([text for node, text in documents.items() if node in train_hosts])
#     doc_embeddings_test = vec.transform([text for node, text in documents.items() if node in test_hosts])
#     # tagged_documents = [TaggedDocument(doc, [i]) for i, doc in documents.items()]
#     # doc_embeddings = Doc2Vec(tagged_documents, vector_size=n_features_text, window=2, min_count=0, workers=8, )
#     # doc_embeddings.save('graph_models/node_embedding/doc_embeddings.model')
# else:
#     doc_embeddings = Doc2Vec.load('graph_models/node_embedding/doc_embeddings.model')

X_train = embedder.transform(train_hosts)


# X_train = np.concatenate((X_train, doc_embeddings_train.toarray()), axis=1)  

X_test = embedder.transform(test_hosts)
# X_test = np.concatenate((X_test, doc_embeddings_test.toarray()), axis=1)


print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

label_bins = LabelBinarizer()
y_values = label_bins.fit_transform(y_train)
class_weights = y_values.sum(axis=0)/len(y_values)
print('Proportions of classes in the training set : ', dict(zip(list(label_bins.classes_), class_weights.tolist())))

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

max_iters = [5, 10, 30, 50]

tols = [5e-1]#, 1e-1, 1e-2]

Cs = [0.7, 0.5]

# max_iters = [50, 300, 800, 1500]

# tols = [1e-1, 1e-2, 1e-3]

# Cs = [1, 1e-1, 5e-2, 1e-2]


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
                    classifiers.append(LogisticRegression(
                        penalty=penalty,
                        tol=tol, 
                        C=C,
                        solver=solver,
                        max_iter=max_iter,
                        multi_class=multi_classe))

print(len(classifiers))

# classifiers = [
#     ExtraTreesClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=10, bootstrap=True, max_features=int(np.sqrt(n_features)), criterion='gini', n_estimators=500),
#     RandomForestClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=10, bootstrap=True, max_features=int(np.sqrt(n_features)), criterion='gini', n_estimators=500),
#     MLPClassifier(solver='sgd', learning_rate='adaptive', max_iter=2000, early_stopping=True, batch_size=64, hidden_layer_sizes=[128, 64, 64]),
#     LogisticRegression(C=1e-1, max_iter=3000),
# ]

scores = {}
output = {}
scores_scaled = {}
output_scaled = {}
try:
    for i, clf in enumerate(tqdm(classifiers)):
        output = cross_validate(clf,
                                X_train,
                                y_train,
                                scoring=make_scorer(loss_function, needs_proba=True),
                                return_train_score=True)
        scores[i] = {'train_score': output['train_score'].mean(), 'score': output['test_score'].mean()}

        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # output_scaled = cross_validate(clf,
        #                                X_train_scaled,
        #                                y_train,
        #                                scoring=make_scorer(loss_function, needs_proba=True),
        #                                return_train_score=True)
        # scores_scaled[i] = {'train_score': output_scaled['train_score'].mean(), 'score': output_scaled['test_score'].mean()}

    best_clf_index = np.argmin(list(map(lambda x: x['score'], list(scores.values()))))
    # best_clf_scaled_index = np.argmin(list(map(lambda x: x['score'], list(scores_scaled.values()))))

    print(best_clf_index)
    # print(best_clf_scaled_index)
    clf = clone(classifiers[best_clf_index])
    # clf_scaled = clone(classifiers[best_clf_scaled_index])
    print(clf)
    # print(clf_scaled)
    print(scores[best_clf_index])
    # print(scores_scaled[best_clf_scaled_index])
    dump(clf, 'graph_models/best_logreg.joblib')
    # dump(clf_scaled, 'graph_models/best_logreg_scaled.joblib')
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    # Write predictions to a file
    with open('../benchmark_graph.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            lst = clf.classes_.tolist()
            lst.insert(0, "Host")
            writer.writerow(lst)
            for i, test_host in enumerate(test_hosts):
                lst = y_pred[i, :].tolist()
                lst.insert(0, test_host)
                writer.writerow(lst)

    # scaler = StandardScaler()
    # clf_scaled.fit(scaler.fit_transform(X_train), y_train)
    # scaler = StandardScaler()
    # y_pred = clf_scaled.predict_proba(scaler.fit_transform(X_test))
    # # Write predictions to a file
    # with open('../benchmark_graph_scaled.csv', 'w') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         lst = clf_scaled.classes_.tolist()
    #         lst.insert(0, "Host")
    #         writer.writerow(lst)
    #         for i, test_host in enumerate(test_hosts):
    #             lst = y_pred[i, :].tolist()
    #             lst.insert(0, test_host)
    #             writer.writerow(lst)

except:
    best_clf_index = np.argmin(list(map(lambda x: x['score'], list(scores.values()))))
    # best_clf_scaled_index = np.argmin(list(map(lambda x: x['score'], list(scores_scaled.values()))))

    print(best_clf_index)
    # print(best_clf_scaled_index)
    clf = clone(classifiers[best_clf_index])
    # clf_scaled = clone(classifiers[best_clf_scaled_index])
    print(clf)
    # print(clf_scaled)
    print(scores[best_clf_index])
    # print(scores_scaled[best_clf_scaled_index])
    dump(clf, 'graph_models/best_logreg.joblib')
    # dump(clf_scaled, 'graph_models/best_logreg_scaled.joblib')
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    # Write predictions to a file
    with open('../benchmark_graph.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = clf.classes_.tolist()
        lst.insert(0, "Host")
        writer.writerow(lst)
        for i, test_host in enumerate(test_hosts):
            lst = y_pred[i, :].tolist()
            lst.insert(0, test_host)
            writer.writerow(lst)

    # scaler = StandardScaler()
    # clf_scaled.fit(scaler.fit_transform(X_train), y_train)
    # scaler = StandardScaler()
    # y_pred = clf_scaled.predict_proba(scaler.fit_transform(X_test))
    # # Write predictions to a file
    # with open('../benchmark_graph_scaled.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     lst = clf_scaled.classes_.tolist()
    #     lst.insert(0, "Host")
    #     writer.writerow(lst)
    #     for i, test_host in enumerate(test_hosts):
    #         lst = y_pred[i, :].tolist()
    #         lst.insert(0, test_host)
    #         writer.writerow(lst)
