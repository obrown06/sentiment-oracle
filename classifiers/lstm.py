import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from torch.autograd import Variable

torch.manual_seed(1)

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_labels):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_label = nn.Linear(hidden_dim, num_labels)
        self.hidden = self.init_hidden(1)

    def init_hidden(self, batch_size):

        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def forward(self, x, lengths):
        embeds = self.word_embeddings(x)
        packed_data = pack_padded_sequence(embeds, lengths)
        lstm_out, self.hidden = self.lstm(packed_data, self.hidden)
        # USED TO HAVE embeds.view(len(document), 1, -1),
        padded_out = pad_packed_sequence(lstm_out, batch_first=True)
        print("padded_out", padded_out)
        outputs = self.hidden_to_label(padded_out)
        label_scores = F.log_softmax(outputs, dim=1)
        return label_scores

    def set_embedding_weights(self, weights):
        """ Given a numpy array of weights, it sets the embedding values """
        self.word_embeddings.weight.data = torch.Tensor(weights)


def batch_and_pad(data, targets, NBATCHES):
    batched_data, batched_targets = batch(data, targets, NBATCHES)
    #print("batched_data", batched_data)
    #print("batched_targets", batched_targets)

    data, targets, lengths = [], [], []
    for i in range(len(batched_targets)):
        batch_data, batch_targets, batch_lengths = pad_and_sort(batched_data[i], batched_targets[i])
        print("batch_lengths type:", type(batch_targets))
        targets.append(batch_targets)
        data.append(batch_data)
        lengths.append(batch_lengths)

    return data, targets, lengths

def pad_and_sort(batch, targets):

    # sort batch and targets by length
    batch_to_target = dict()
    for i in range(len(batch)):
        batch_to_target[tuple(batch[i])] = targets[i]

    batch = sorted(batch, key=len, reverse=True)

    targets = []

    for i in range(len(batch)):
        targets.append(batch_to_target[tuple(batch[i])])

    # pad arrays with zeros

    max_len = batch[0].shape[0]
    lengths = []

    for i in range(len(batch)):
        lengths.append(batch[i].shape[0])
        batch[i] = np.pad(batch[i], (0, max_len - batch[i].shape[0]), 'constant')

    return np.asarray(batch).T, np.asarray(targets).T, lengths


def batch(data, targets, NBATCHES):
    #print("targets: ", targets)
    n_samples = len(data)
    #print("n_samples:", n_samples)
    batch_size = n_samples // NBATCHES
    #print("batch_size: ", batch_size)
    batched_data = []
    batched_targets = []

    for i in range(NBATCHES):
        batched_data.append(data[batch_size * i:batch_size * (i + 1)])
        batched_targets.append(targets[batch_size * i:batch_size * (i + 1)])
    #print("batched_targets: ", batched_targets)

    return batched_data, batched_targets


def train(net, data, target, ALPHA = 0.01, EPOCHS = 10, NBATCHES = 10):
    optimizer = optim.SGD(net.parameters(), lr=ALPHA, momentum=0.9)
    criterion = nn.NLLLoss()

    batched_data, batched_targets, batched_lengths = batch_and_pad(data, target, NBATCHES)

    for epoch in range(EPOCHS):
        for i in range(NBATCHES):
            print("batched_targets[i] type", type(batched_targets[i]))
            print("batched_data[i] type", type(batched_data[i]))
            print("batched_lengths[i] type", type(batched_lengths[i]))
            print("batched_lenghts[i]: ", batched_lengths[i])
            batch, target, lengths = Variable(torch.from_numpy(batched_data[i])), Variable(torch.from_numpy(batched_targets[i]).long()), batched_lengths[i]
            print("lengths size: ", len(lengths))
            print("batch size", batch.size())
            net.hidden = net.init_hidden(len(lengths))
            optimizer.zero_grad()
            net_out = net(batch, lengths)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("Epoch: ", epoch)
            print("Loss: ", loss.data[0])
