import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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