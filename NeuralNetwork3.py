from copy import deepcopy

import numpy as np
import pandas as pd
import sys

"""
Hyper parameters
"""
N_in = 784
N_h1 = 256
N_h2 = 256
N_out = 10
lr = 0.001
momentum = 0.3


def get_mini_batch(X, Y, batch_size, N, testing=False):
    """
    Function to convert data to batches
    :param X: image array
    :param Y: label array or null
    :param batch_size: maximum size of each batch
    :param N: size of data
    :param testing: boolean to indicate if data is test data
    :return: list of minibatches
    """
    minibatches = []
    number_of_minibatches = int(N / batch_size)
    if not testing:
        randomize = np.arange(N)
        np.random.shuffle(randomize)
        shuffled_X = X[randomize]
        shuffled_Y = Y[randomize]
        k = 0
        while k < number_of_minibatches:
            minibatch_X = shuffled_X[k * batch_size: (k + 1) * batch_size]
            minibatch_Y = shuffled_Y[k * batch_size: (k + 1) * batch_size]
            minibatch_pair = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch_pair)
            k += 1

        if N % batch_size != 0:
            last_minibatch_X = shuffled_X[k * batch_size: N]
            last_minibatch_Y = shuffled_Y[k * batch_size: N]
            last_minibatch_pair = (last_minibatch_X, last_minibatch_Y)
            minibatches.append(last_minibatch_pair)
    else:
        k = 0
        while k < number_of_minibatches:
            minibatch_X = X[k * batch_size: (k + 1) * batch_size]
            minibatches.append(minibatch_X)
            k += 1

        if N % batch_size != 0:
            last_minibatch_X = X[k * batch_size: N]
            minibatches.append(last_minibatch_X)
    return minibatches


class FullyConnected:
    """
    Class for fully connected neural network
    """
    def __init__(self, N_in, N_h1, N_h2, N_out):
        """
        Function to initialize the neural network
        :param N_in: Number of neurons in input layer
        :param N_h1: Number of neurons in first hidden layer
        :param N_h2: Number of neurons in second hidden layer
        :param N_out: Number of output neurons
        """
        self.N_in = N_in
        self.N_h1 = N_h1
        self.N_h2 = N_h2
        self.N_out = N_out

        r1a = -1 / (N_h1 ** 0.5)
        r1b = 1 / (N_h1 ** 0.5)
        w1 = (r1a - r1b) * np.random.rand(N_h1, N_in) + r1b

        r2a = -1 / (N_h2 ** 0.5)
        r2b = 1 / (N_h2 ** 0.5)
        w2 = (r2a - r2b) * np.random.rand(N_h2, N_h1) + r2b

        r3a = -1 / (N_out ** 0.5)
        r3b = 1 / (N_out ** 0.5)
        w3 = (r3a - r3b) * np.random.rand(N_out, N_h2) + r3b

        self.weights = {'w1': w1, 'w2': w2, 'w3': w3}
        b1 = (r1a - r1b) * np.random.rand(N_h1) + r1b
        b2 = (r2a - r2b) * np.random.rand(N_h2) + r2b
        b3 = (r3a - r3b) * np.random.rand(N_out) + r3b

        self.biases = {'b1': b1, 'b2': b2, 'b3': b3}
        z1 = 0
        z2 = 0
        z3 = 0
        self.cache = {'z1': z1, 'z2': z2, 'z3': z3}
        self.gradients = {'dw1': 0, 'dw2': 0, 'dw3': 0, 'db1': 0, 'db2': 0, 'db3': 0}

    def train(self, inputs, labels, lr=0.001, momentum=0.3):
        """
        Function to train the neural net
        :param inputs: image array
        :param labels: label array
        :param lr: learning rate of the neural net
        :param momentum: momentum of the gradient descent algorithm
        """
        outputs = self.forward(inputs)
        self.backward(inputs, labels, outputs, lr, momentum)
        self.weights, self.biases = self.update_weights(self.weights, self.biases)
        return

    def predict(self, inputs):
        """
        Function to predict value for test data
        :param inputs: test image array
        :return: The score and the prediction of the test data
        """
        outputs = self.forward(inputs)
        score = np.max(outputs, axis=1)
        idx = np.argmax(outputs, axis=1)
        return score, idx

    def forward(self, inputs):
        """
        Function to perform forward propagation
        :param inputs: image array
        :return: output array with score for each output
        """
        self.cache['z1'] = self.weighted_sum(inputs, self.weights['w1'], self.biases['b1'])
        a1 = self.sigmoid(self.cache['z1'])

        self.cache['z2'] = self.weighted_sum(a1, self.weights['w2'], self.biases['b2'])
        a2 = self.sigmoid(self.cache['z2'])

        self.cache['z3'] = self.weighted_sum(a2, self.weights['w3'], self.biases['b3'])
        outputs = self.softmax(self.cache['z3'])
        return outputs

    def backward(self, inputs, labels, outputs, lr, momentum):
        """
        Function to perform backward propagation
        :param inputs: image array
        :param labels: label array
        :param outputs: predictions array
        :param lr: learning rate
        :param momentum: momentum of gradient descent algorithm
        """
        dout = self.delta_cross_entropy_softmax(outputs, labels)

        d2 = np.matmul(dout, self.weights['w3']) * (self.delta_sigmoid(self.cache['z2']))

        d1 = np.matmul(d2, self.weights['w2']) * (self.delta_sigmoid(self.cache['z1']))

        self.calculate_grad(inputs, d1, d2, dout, lr, momentum)
        return

    def calculate_grad(self, inputs, d1, d2, dout, lr, momentum):
        """
        Function to calculate the gradient at each layer
        :param inputs: image array
        :param d1: Differential at layer 1
        :param d2: Differential at layer 2
        :param dout: Differential at output layer
        :param lr: Learning rate
        :param momentum: Momentum for gradient descent
        """
        self.gradients['dw3'] = np.matmul(dout.transpose(), self.sigmoid(self.cache['z2'])) * lr + \
            momentum * self.gradients['dw3']
        self.gradients['dw2'] = np.matmul(d2.transpose(), self.sigmoid(self.cache['z1'])) * lr + \
            momentum * self.gradients['dw2']
        self.gradients['dw1'] = np.matmul(d1.transpose(), inputs) * lr + momentum * self.gradients['dw1']

        self.gradients['db3'] = np.sum(dout, axis=0) * lr + momentum * self.gradients['db3']
        self.gradients['db2'] = np.sum(d2, axis=0) * lr + momentum * self.gradients['db2']
        self.gradients['db1'] = np.sum(d1, axis=0) * lr + momentum * self.gradients['db1']
        return

    def update_weights(self, weights, biases):
        """
        Function to perform weight updates
        :param weights: previous weights
        :param biases: previous biases
        :return: updated weights and biases
        """
        n_weights = deepcopy(weights)
        n_biases = deepcopy(biases)
        n_weights['w1'] = weights['w1'] - self.gradients['dw1']
        n_weights['w2'] = weights['w2'] - self.gradients['dw2']
        n_weights['w3'] = weights['w3'] - self.gradients['dw3']
        n_biases['b1'] = biases['b1'] - self.gradients['db1']
        n_biases['b2'] = biases['b2'] - self.gradients['db2']
        n_biases['b3'] = biases['b3'] - self.gradients['db3']
        return n_weights, n_biases

    def cross_entropy_loss(self, outputs, labels):
        """
        Function to calculate cross entropy loss
        :param outputs: predictions array
        :param labels: label array
        :return: cross entropy loss for given predictions and labels
        """
        c = 0
        out = np.zeros(outputs.shape)
        for i in labels:
            out[c][i] = 1.
            c = c + 1
        N = outputs.shape[0]
        creloss = -np.sum(out * np.log(outputs)) / N
        return creloss

    def sigmoid(self, z):
        """
        Function to calculate sigmoid
        :param z: array
        :return: sigmoid of the array
        """
        e_x = np.exp(-z)
        result = 1 / (1 + e_x)
        return result

    def delta_sigmoid(self, z):
        """
        Function to calculate derivative of sigmoid
        :param z: array
        :return: derivative of sigmoid of the array
        """
        grad_sigmoid = self.sigmoid(z) * (1 - self.sigmoid(z))
        return grad_sigmoid

    def softmax(self, x):
        """
        Function to calculate softmax
        :param x: output array
        :return: softmax of the output array
        """
        maxes = np.max(x, 1)
        e_x = np.exp(x - maxes[:, None])
        cumulative_sum = np.sum(e_x, axis=1)
        return e_x / cumulative_sum[:, None]

    def weighted_sum(self, X, w, b):
        """
        Function to perform matrix multiplication
        :param X: input array
        :param w: weight array
        :param b: bias array
        :return: weighted sum array
        """
        result = np.matmul(X, w.transpose()) + b
        return result

    def delta_cross_entropy_softmax(self, outputs, labels):
        """
        Function to calculate derivative of cross entropy
        :param outputs: output array
        :param labels: label array
        :return: derivative array of cross entropy
        """
        k = 0
        N = outputs.shape[0]
        out = np.zeros(outputs.shape)
        for j in labels:
            out[k][j] = 1.
            k += 1
        avg_grads = (outputs - out) / N
        return avg_grads


train_image, train_label, test_image = sys.argv[1:]
net = FullyConnected(N_in, N_h1, N_h2, N_out)
N_epochs = 7
inputs = pd.read_csv(train_image, header=None)
labels = pd.read_csv(train_label, header=None)
inputs = np.asarray(inputs)
N = inputs.shape[0]
labels = np.asarray(labels).reshape(N, )
batch_size = max(1, N//2500)

for epoch in range(N_epochs):
    mini_batches = get_mini_batch(inputs, labels, batch_size, N)
    for (batch_input, batch_labels) in mini_batches:
        net.train(batch_input, batch_labels, lr, momentum)

test_inputs = pd.read_csv(test_image, header=None)
M = test_inputs.shape[0]
test_inputs = np.asarray(test_inputs)

predictions = []
mini_batches = get_mini_batch(test_inputs, None, batch_size, M, testing=True)
for batch_input in mini_batches:
    score, idx = net.predict(batch_input)
    predictions.append(idx)

df = pd.DataFrame(predictions, dtype='Int64')
df.to_csv("test_predictions.csv", header=None, index=None, sep='\n')
