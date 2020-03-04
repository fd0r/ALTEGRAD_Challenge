import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import codecs
from os import path
import csv
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def get_raw_data():
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
        with codecs.open(path.join('text/text/', filename), encoding='latin-1') as f: 
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

