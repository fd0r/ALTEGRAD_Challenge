import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm


class DeepWalk():

    def __init__(self, walk_length=80, n_walks=50, n_features=128, window=8, epochs=5, skipgram_method=1, workers=8, load_path=None, verbose=True):
        self.walk_length = walk_length
        self.n_walks = n_walks
        self.n_features = n_features
        self.skipgram_method = skipgram_method
        self.workers = workers
        self.window = window
        self.epochs = epochs
        self.walks = None
        self.model = None
        self.verbose = verbose
        if not load_path is None:
            self.model = Word2Vec.load(load_path)

    def fit(self, G, save_path=None):
        self.walks = self.generate_walks(G)
        self.model = Word2Vec(
            size=self.n_features,
            window=self.workers,
            min_count=0,
            sg=self.skipgram_method,
            workers=8)
        self.model.build_vocab(self.walks)
        self.model.train(self.walks, total_examples=self.model.corpus_count, epochs=self.epochs)
        
        if not save_path is None:
            self.model.save(save_path)
        return self

    def transform(self, nodes):
        if self.model is None:
            raise Exception('You must fit the model first')
        
        X = np.zeros((len(nodes), self.n_features))
        for i, node in enumerate(nodes):
            X[i] = self.model.wv[node]

        return X

    def fit_transform(self, G, nodes):
        self.fit(G)
        return self.transform(nodes)

    def random_walk(self, G, node):
        node = str(node)
        walk = [node]
        neighs = list(dict(G[node]).keys())
        t = 1
        while len(neighs) > 0 and t < self.walk_length:
            #weights = np.array(list(map(lambda x: x['weight'], list(dict(G[node]).values()))))
            #node = neighs[np.random.multinomial(1, weights/weights.sum()).argmax()]
            node = neighs[int(np.random.rand()*len(neighs))]
            walk.append(node)
            neighs = list(dict(G[node]).keys())
            t += 1
        
        return walk


    def generate_walks(self, G):
        walks = []
        nodes = list(G.nodes())
        if self.verbose:
            print('Generating walks')

        max_len = 0
        walk_mean_length = 0
        for node in tqdm(nodes, disable=not self.verbose):
            for i in range(self.n_walks):
                walk = self.random_walk(G, node)
                walk_length = len(walk)
                walks.append(walk)
                if walk_length > max_len:
                    max_len = walk_length
                walk_mean_length += walk_length
        walk_mean_length /= len(walks)

        # print('max len : ', max_len)
        # print('mean len : ', walk_mean_length)
        # print('num of walks above 20:', sum([1 for walk in walks if len(walk) > 20]))
        # print('num of walks above 30:', sum([1 for walk in walks if len(walk) > 30]))
        # print('num of walks above 50:', sum([1 for walk in walks if len(walk) > 50]))
        # print('num of walks above 60:', sum([1 for walk in walks if len(walk) > 60]))
        # print('num of walks above 80:', sum([1 for walk in walks if len(walk) >= 80]))
        # print('num of walks above 100:', sum([1 for walk in walks if len(walk) >= 100]))
        # print('num of walks : ', len(walks))
        return walks
    
