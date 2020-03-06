import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch.nntorch.nn  as  nn
import torch
import numpy as np
import json
import random
from graph_models.node_embedding import DeepWalk

train_hosts, test_hosts, y_train, G = load_data('../data')

n_nodes = G.number_of_nodes()

context_tuple_list = []
w = 4

with open('../data/vocab.json') as file:
    vocab = json.load(file)

vocab_size = len(vocab)

with open('../data/docs_ints.json') as file:
    docs_ints = json.load(file)

n_walks = 150
walk_length = 100

embedder = DeepWalk(walk_length, n_walks, verbose=True)
embedder.generate_walks(G)

walks = embedder.walks


def sample_negative(sample_size):
    sample_probability = {}
    node_counts = dict(Counter(list(itertools.chain.from_iterable(walks))))
    normalizing_factor = sum([v**0.75 for v in word_counts.values()])
    for node in node_counts:
        sample_probability[node] = node_counts[node]**0.75 / normalizing_factor
    nodes = np.array(list(node_counts.keys()))
    while True:
        node_list = []
        sampled_index = np.array(np.random.multinomial(
            sample_size, list(sample_probability.values())))
        for index, count in enumerate(sampled_index):
            for _ in range(count):
                 node_list.append(nodes[index])
        yield node_list

window = 4
negative_samples = sample_negative(8)

for walk in walks:
    for i, node in enumerate(walk):
        first_context_node_index = max(0, i-w)
        last_context_node_index = min(i+w, len(walk))
        for j in range(first_context_node_index, last_context_node_index):
            if i != j:
                context_tuple_list.append((node, walk[j], next(negative_samples)))




class NodeDoc2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.embeddings_target = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)

    def forward(self, target_node, context_node, negative_example):
        emb_target = self.embeddings_target(target_node)
        emb_context = self.embeddings_context(context_node)
        emb_product = torch.mul(emb_target, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out = torch.sum(F.logsigmoid(emb_product))
        emb_negative = self.embeddings_context(negative_example)
        emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2))
        emb_product = torch.sum(emb_product, dim=1)
        out += torch.sum(F.logsigmoid(-emb_product))
        return -out

class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_percent_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop_training(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / \
            max(self.loss_list)
        print("Loss gain: {}%".format(round(100*gain, 2)))
        if gain < self.min_percent_gain:
            return True
        else:
            return False


def get_batches(context_tuple_list, batch_size=100):
    random.shuffle(context_tuple_list)
    batches = []
    batch_target, batch_context, batch_negative = [], [], []
    for i in range(len(context_tuple_list)):
        batch_target.append(context_tuple_list[i][0])
        batch_context.append(docs_to_ints[context_tuple_list[i][1]])
        batch_negative.append([docs_to_ints[node] for node in context_tuple_list[i][2]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = autograd.Variable(torch.from_numpy(np.array(batch_target)).long())
            tensor_context = autograd.Variable(torch.from_numpy(np.array(batch_context)).long())
            tensor_negative = autograd.Variable(torch.from_numpy(np.array(batch_negative)).long())
            batches.append((tensor_target, tensor_context, tensor_negative))
            batch_target, batch_context, batch_negative = [], [], []
    return batches

    

net = NodeDoc2Vec(embedding_size=256, vocab_size=vocab_size)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
early_stopping = EarlyStopping(patience=5, min_percent_gain=1)
context_tensor_list = []

while True:
    losses = []
    context_tuple_batches = get_batches(context_tuple_list, batch_size=2000)
    for i in range(len(context_tuple_batches)):
        net.zero_grad()
        target_tensor, context_tensor, negative_tensor = context_tuple_batches[i]
        loss = net(target_tensor, context_tensor, negative_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
    print("Loss: ", np.mean(losses))
    early_stopping.update_loss(np.mean(losses))
    if early_stopping.stop_training():
        break
