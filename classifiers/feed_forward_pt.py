import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

class FeedForwardClassifier(nn.Module):

    def __init__(self, data_info, classifier_info):
        super(FeedForwardClassifier, self).__init__()
        self.data_info = data_info
        self.classifier_info = classifier_info

        self.alpha = classifier_info["alpha"]
        self.nbatches = classifier_info["nbatches"]
        self.nepochs = classifier_info["nepochs"]
        self.class_labels = data_info["class_labels"]

        self.input_layer = nn.Linear(classifier_info["embedding_dim"], classifier_info["hidden_dim"])
        self.hidden_layer = nn.Linear(classifier_info["hidden_dim"], classifier_info["hidden_dim"])
        self.output_layer = nn.Linear(classifier_info["hidden_dim"], classifier_info["output_dim"])

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        label_scores = F.log_softmax(x, dim=1)
        return label_scores

    def train(self, data, target):
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data = torch.transpose(data, 0, 1)
        data, target = data.type(torch.FloatTensor), target.type(torch.LongTensor)
        optimizer = optim.SGD(self.parameters(), lr=self.alpha, momentum=0.9)
        criterion = nn.NLLLoss()

        batched_data, batched_targets = self.batch(data, target, self.nbatches)


        for epoch in range(self.nepochs):
            for i in range(self.nbatches):
                data_batch, target_batch = Variable(batched_data[i]), Variable(batched_targets[i])
                optimizer.zero_grad()
                out = self.forward(data_batch)
                loss = criterion(out, target_batch)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print("Epoch: ", epoch)
                print("Loss: ", loss.data[0])

    def batch(self, data, targets, NBATCHES):
        n_samples = data.size()[0]
        batch_size = n_samples // NBATCHES
        batched_data = []
        batched_targets = []

        for i in range(NBATCHES):
            batched_data.append(data.narrow(0, batch_size * i, batch_size))
            batched_targets.append(targets.narrow(0, batch_size * i, batch_size))

        return batched_data, batched_targets



    def test(self, data, target):
        """
        Arguments:
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
            predictions = np.append(predictions, self.classify(doc_data))

        #predictions = predictions.reshape(target.shape)

        return predictions, target

    def classify(self, data):
        """
        Arguments:
        data     : a numpy array of dimension [number of features] x [1], containing the
               values of every feature in a single document; includes a row of 1s which pair
               with the w[0] (the bias)

        Returns:
            max : a scalar containing the predicted class label for the given document
        """
        class_index = self.predict(data)
        return self.class_labels[class_index]

    def predict(self, data):
        """
        Arguments:
        data     : a numpy array of dimension [number of features] x [1], containing the
               values of every feature in a single document

        Returns:
        h : the output of the classifier when applied to the given set of data
        """
        data = torch.from_numpy(data)
        data = torch.transpose(data, 0, 1)
        data = Variable(data.type(torch.FloatTensor))
        out = self.forward(data)
        return out.data.max(1)[1][0]
