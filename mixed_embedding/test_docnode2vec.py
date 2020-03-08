import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch.nn  as  nn
import torch
import numpy as np
import json
import random
from tqdm import tqdm
from graph_models.node_embedding import DeepWalk
from collections import Counter
from utils import load_data
import ijson
import itertools
import time

train_hosts, test_hosts, y_train, G = load_data('../data')

n_nodes = G.number_of_nodes()

print(n_nodes)

context_tuple_list = []
w = 2

with open('../data/vocab_stemmed.json') as file:
    vocab = json.load(file)

vocab_size = len(vocab)
print(vocab_size)

docs_ints = {}
docs_lengths = []
with open('../data/doc_ints_stemmed.json', 'r') as file:
    # t = time.time()
    docs_ints = json.loads(file.read())
    
print(len(docs_ints))
docs_lengths = [len(x) for x in docs_ints.values()]
print(np.mean(docs_lengths))
print(np.median(docs_lengths))
print(max(docs_lengths))

for node in train_hosts + test_hosts:
    if not node in docs_ints.keys():
        docs_ints[node] = []

n_walks = 2
walk_length = 5

embedder = DeepWalk(walk_length, n_walks, verbose=True)
walks = embedder.generate_walks(G)

# def sample_negative(sample_size):
#     sample_probability = {}
#     node_counts = dict(Counter(list(itertools.chain.from_iterable(walks))))
#     normalizing_factor = sum([v**0.75 for v in node_counts.values()])
#     for node in node_counts:
#         sample_probability[node] = node_counts[node]**0.75 / normalizing_factor
#     nodes = np.array(list(node_counts.keys()))
#     while True:
#         node_list = []
#         sampled_index = np.array(np.random.multinomial(
#             sample_size, list(sample_probability.values())))
#         for index, count in enumerate(sampled_index):
#             for _ in range(count):
#                  node_list.append(nodes[index])
#         yield node_list

# negative_samples = sample_negative(8)

for walk in tqdm(walks):
    for i, node in enumerate(walk):
        first_context_node_index = max(0, i-w)
        last_context_node_index = min(i+w, len(walk))
        for j in range(first_context_node_index, last_context_node_index):
            if i != j:
                context_tuple_list.append((node, walk[j]))#, next(negative_samples)))


print(len(context_tuple_list))

# class NodeDoc2Vec(nn.Module):

#     def __init__(self, embedding_size, vocab_size):
#         super(NodeDoc2Vec, self).__init__()
#         self.embeddings_target = nn.Embedding(vocab_size, embedding_size)
#         self.embeddings_context = nn.Embedding(vocab_size, embedding_size)

#     def forward(self, target_node, context_node, negative_example):
#         emb_target = self.embeddings_target(target_node)
#         emb_context = self.embeddings_context(context_node)
#         emb_product = torch.mul(emb_target, emb_context)
#         emb_product = torch.sum(emb_product, dim=1)
#         out = torch.sum(F.logsigmoid(emb_product))
#         emb_negative = self.embeddings_context(negative_example)
#         emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2))
#         emb_product = torch.sum(emb_product, dim=1)
#         out += torch.sum(F.logsigmoid(-emb_product))
#         return -out


class DocNode2Vec(nn.Module):

    def __init__(self, node_embedding_size, nodes_size, word_embedding_size, vocab_size):
        super(NodeDoc2Vec, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.linear_1 = nn.Linear(word_embedding_size, nodes_size)
        self.node_embedding = nn.Embedding(nodes_size, node_embedding_size)
        self.linear = nn.Linear(node_embedding_size, nodes_size)

    def forward(self, context_word):
        emb_docs = self.word_embedding(context_word)
        hidden_docs = self.linear_1(emb_docs)
        out_docs = F.relu(hidden_docs)
        node_emb = self.node_embedding(out_docs)
        hidden = self.linear(node_emb)
        out = F.log_softmax(hidden)
        return out


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


def get_batches(context_tuple_list, batch_size=100, n_words_per_doc=8000):
    random.shuffle(context_tuple_list)
    batches = []
    batch_target, batch_context, batch_negative = [], [], []
    for i in range(len(context_tuple_list)):
        batch_target.append(int(context_tuple_list[i][0]))

        context_words = get_words_idx_from_node(context_tuple_list[i][1], n_words_per_doc)
        
        batch_context.append(context_words)
        # batch_negative.append([get_words_idx_from_node(node) for node in context_tuple_list[i][2]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = autograd.Variable(torch.from_numpy(np.array(batch_target)).long())
            tensor_context = autograd.Variable(torch.from_numpy(np.array(batch_context)).long())
            # tensor_negative = autograd.Variable(torch.from_numpy(np.array(batch_negative)).long())
            batches.append((tensor_target, tensor_context))#, tensor_negative))
            batch_target, batch_context, batch_negative = [], [], []
    return batches

def get_words_idx_from_node(node, n_words_per_doc, oov_idx=1):
    words = docs_ints[node]
    if len(words) > n_words_per_doc:
        words = words[:n_words_per_doc]
    else:
        words = words + [oov_idx]*(n_words_per_doc - len(words))
    return words
    

n_words_per_doc = 1000
net = DocNode2Vec(node_embedding_size=128, nodes_size=n_nodes,
                  word_embedding_size=128, vocab_size=vocab_size)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
early_stopping = EarlyStopping(patience=5, min_percent_gain=1)
context_tensor_list = []
n_epochs = 10

# while True:
#     losses = []
#     context_tuple_batches = get_batches(context_tuple_list, batch_size=2000)
#     for i in range(len(context_tuple_batches)):
#         net.zero_grad()
#         target_tensor, context_tensor, negative_tensor = context_tuple_batches[i]
#         loss = net(target_tensor, context_tensor, negative_tensor)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.data)
#     print("Loss: ", np.mean(losses))
#     early_stopping.update_loss(np.mean(losses))
#     if early_stopping.stop_training():
#         break

for epoch in range(n_epochs):
    losses = []
    context_tuple_batches = get_batches(context_tuple_list, batch_size=2000, n_words_per_doc=n_words_per_doc)

    with tqdm(total=len(context_tuple_batches),unit_scale=True,postfix={'loss':0.0},desc="Epoch : %i/%i" % (epoch+1, n_epochs),ncols=50) as pbar:
        for j in range(len(context_tuple_batches)):
            net.zero_grad()
            target_tensor, context_tensor = context_tuple_batches[j]
            log_probs = net(context_tensor)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
            pbar.set_postfix({'loss': sum(losses)/(j+1)})
            pbar.update(1)
    # print("Loss: ", np.mean(losses))
    early_stopping.update_loss(np.mean(losses))
    if early_stopping.stop_training():
        break
