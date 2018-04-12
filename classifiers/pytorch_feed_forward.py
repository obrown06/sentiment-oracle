import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

class FeedForwardClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_labels):
        super(FeedForwardClassifier, self).__init__()
        self.input_layer = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        label_scores = F.log_softmax(x, dim=1)
        return label_scores

def train(net, data, target, ALPHA = 0.01, EPOCHS = 10, NBATCHES = 10):
    data, target = torch.from_numpy(data), torch.from_numpy(target)
    data = torch.transpose(data, 0, 1)
    data, target = data.type(torch.FloatTensor), target.type(torch.LongTensor)
    optimizer = optim.SGD(net.parameters(), lr=ALPHA, momentum=0.9)
    criterion = nn.NLLLoss()

    batched_data, batched_targets = batch(data, target, NBATCHES)


    for epoch in range(EPOCHS):
        for i in range(NBATCHES):
            data_batch, target_batch = Variable(batched_data[i]), Variable(batched_targets[i])
            optimizer.zero_grad()
            net_out = net(data_batch)
            loss = criterion(net_out, target_batch)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("Epoch: ", epoch)
            print("Loss: ", loss.data[0])

def batch(data, targets, NBATCHES):
    n_samples = data.size()[0]
    batch_size = n_samples // NBATCHES
    batched_data = []
    batched_targets = []

    for i in range(NBATCHES):
        batched_data.append(data.narrow(0, batch_size * i, batch_size))
        batched_targets.append(targets.narrow(0, batch_size * i, batch_size))

    return batched_data, batched_targets



def test(net, class_list, data, target):
    """
    Arguments:
    net : the stored representation of a feed-forward pytorch neural net
    class_list : a python list containing the possible output class labels
    data : a numpy array of dimension [number of features] x [number of test examples], containing the
        values of every feature in every document in the test set; includes a row of 1s which pair
        with the w[0] (the bias)
    target -  a numpy array containing the actual class labels for each document in the test set

    Returns:
    predictions : a numpy array containing the predicted class labels for each document
                  in the test set (mapped by self.classify())
    Y : a numpy array containing the actual class labels for each document in the test set
    """
    predictions = np.array([])

    for i in range(data.shape[1]):
        doc_data = data[:,i].reshape(data[:,i].shape[0], 1)
        predictions = np.append(predictions, classify(net, doc_data, class_list))

    predictions = predictions.reshape(target.shape)

    return predictions, target

def classify(net, data, class_list):
    """
    Arguments:
    net : the stored representation of a feed-forward pytorch neural net
    class_list : a python list containing the possible output class labels
    data     : a numpy array of dimension [number of features] x [1], containing the
               values of every feature in a single document; includes a row of 1s which pair
               with the w[0] (the bias)

    Returns:
    max : a scalar containing the predicted class label for the given document
    """
    class_index = predict(net, data)
    return class_list[class_index]

def predict(net, data):
    """
    Arguments:
    net : the stored representation of a feed-forward pytorch neural net
    data     : a numpy array of dimension [number of features] x [1], containing the
               values of every feature in a single document

    Returns:
    h : the output of the classifier when applied to the given set of data
    """
    data = torch.from_numpy(data)
    data = torch.transpose(data, 0, 1)
    data = Variable(data.type(torch.FloatTensor))
    net_out = net(data)
    return net_out.data.max(1)[1][0]