import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from torch.autograd import Variable

torch.manual_seed(1)

class PyTorchLSTMClassifier(nn.Module):

    def __init__(self, data_info, classifier_info):
        super(PyTorchLSTMClassifier, self).__init__()
        self.data_info = data_info
        self.classifier_info = classifier_info
        self.class_labels = data_info["class_labels"]
        self.nepochs = classifier_info["nepochs"]
        self.nbatches = classifier_info["nbatches"]

        self.hidden_dim = classifier_info["hidden_dim"]
        self.word_embeddings = nn.Embedding(classifier_info["nfeatures"], classifier_info["embed_size"])
        self.lstm = nn.LSTM(classifier_info["embed_size"], classifier_info["hidden_dim"])
        self.hidden_to_label = nn.Linear(classifier_info["hidden_dim"], len(data_info["class_labels"]))
        self.hidden = self.init_hidden(1)

    def train(self, data, target, embeddings):
        self.set_embedding_weights(embeddings)
        optimizer = optim.Adam(self.parameters())
        criterion = nn.NLLLoss()

        batched_data, batched_targets, batched_lengths = self.batch_and_pad(data, target, self.nbatches)

        for epoch in range(self.nepochs):
            for i in range(self.nbatches):
                batch, target, lengths = Variable(torch.from_numpy(batched_data[i])), Variable(torch.from_numpy(batched_targets[i]).long()), batched_lengths[i]
                self.hidden = self.init_hidden(len(lengths))
                optimizer.zero_grad()
                out = self.forward(batch, lengths)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                if i % 2 == 0:
                    print("Batch: ", i)

            if epoch % 2 == 0:
                print("Epoch: ", epoch)
                print("Loss: ", loss.data[0])


    def init_hidden(self, batch_size):

        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def forward(self, x, lengths):
        embeds = self.word_embeddings(x)
        packed_data = pack_padded_sequence(embeds, lengths)
        lstm_out, self.hidden = self.lstm(packed_data, self.hidden)
        padded_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lengths = [l - 1 for l in lengths]
        padded_out = padded_out[torch.arange(0, len(lengths)).long(), torch.LongTensor(lengths), :]
        outputs = self.hidden_to_label(padded_out)
        label_scores = F.log_softmax(outputs, dim=1)
        return label_scores

    def set_embedding_weights(self, weights):
        """ Given a numpy array of weights, it sets the embedding values """
        self.word_embeddings.weight.data = torch.Tensor(weights)


    def batch_and_pad(self, data, targets, NBATCHES):
        batched_data, batched_targets = self.batch(data, targets, NBATCHES)

        data, targets, lengths = [], [], []
        for i in range(len(batched_targets)):
            batch_data, batch_targets, batch_lengths = self.pad_and_sort(batched_data[i], batched_targets[i])
            targets.append(batch_targets)
            data.append(batch_data)
            lengths.append(batch_lengths)

        return data, targets, lengths

    def pad_and_sort(self, batch, targets):

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

        return np.asarray(batch).T, np.asarray(targets), lengths


    def batch(self, data, targets, NBATCHES):
        n_samples = len(data)
        batch_size = n_samples // NBATCHES
        batched_data = []
        batched_targets = []

        for i in range(NBATCHES):
            batched_data.append(data[batch_size * i:batch_size * (i + 1)])
            batched_targets.append(targets[batch_size * i:batch_size * (i + 1)])

        return batched_data, batched_targets


    def test(self, data, target):
        predictions = np.array([])

        for i in range(target.shape[0]):
            doc_data = data[i]
            predictions = np.append(predictions, self.classify(doc_data))

        predictions = predictions.reshape(target.shape)

        return predictions, target

    def classify(self, data):
        class_index = self.predict(data)
        return self.class_labels[class_index]

    def predict(self, data):
        lengths = [len(data)]
        self.hidden = self.init_hidden(len(lengths))
        data = torch.from_numpy(data)
        data = Variable(data.view(data.size(0), 1))
        out = self.forward(data, lengths)
        return out.data.max(1)[1][0]
