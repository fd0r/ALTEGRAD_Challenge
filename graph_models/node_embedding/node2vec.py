import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec
from tqdm import tqdm


class Node2Vec():
	def __init__(self, walk_length=80, n_walks=50, n_features=128, p=1, q=1, window=8, epochs=5, skipgram_method=1, workers=8, load_path=None, verbose=True):
		self.walk_length = walk_length
		self.n_walks = n_walks
		self.n_features = n_features
		self.p = p
		self.q = q
		self.skipgram_method = skipgram_method
		self.workers = workers
		self.window = window
		self.epochs = epochs
		self.walks = None
		self.model = None
		self.verbose = verbose
		if not load_path is None:
			self.model = Word2Vec.load(load_path)
	
	def fit(self, G, is_directed=True, save_path=None):
		self.preprocess_transition_probs(G, is_directed)
		self.walks = self.generate_walks(G)
		self.model = Word2Vec(
			size=self.n_features,
			window=self.workers,
			min_count=0,
			sg=self.skipgram_method,
			workers=8)
		self.model.build_vocab(self.walks)
		self.model.train(
			self.walks, total_examples=self.model.corpus_count, epochs=self.epochs)

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

	def fit_transform(self, G, nodes, is_directed=True, save_path=None):
		self.fit(G, is_directed, save_path)
		return self.transform(nodes)
		

	def node2vec_walk(self, G, node):
		'''
		Simulate a random walk starting from start node.
		'''
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [node]

		while len(walk) < self.walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def generate_walks(self, G):
		'''
		Repeatedly simulate random walks from each node.
		'''
		walks = []
		nodes = list(G.nodes())
		if self.verbose:
			print('Generating walks')
		for walk_iter in tqdm(range(self.n_walks), disable=not self.verbose):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(G, node))

		return walks

	def get_alias_edge(self, G, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/self.p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/self.q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self, G, is_directed):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''

		alias_nodes = {}
		if self.verbose:
			print('Preprocessing transition probs')
		for node in tqdm(G.nodes(), disable=not self.verbose):
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}
		
		if self.verbose:
			print('Preprocessing edges')
		if is_directed:
			for edge in tqdm(G.edges(), disable=not self.verbose):
				alias_edges[edge] = self.get_alias_edge(G, edge[0], edge[1])
		else:
			for edge in tqdm(G.edges(), disable=not self.verbose):
				alias_edges[edge] = self.get_alias_edge(G, edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(G, edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

