# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:15:50 2017

@author: Gongwei Chen
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from datetime import datetime
import numpy as np
import sys
from activation import tanh as act
from output import softmax_cross_entropy

class RNNModel:
    
    def __init__(self, input_dim, hidden_dim, output_dim, bptt_truncate=14):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/input_dim),
                                   (hidden_dim, input_dim))
        self.W = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim),
                                   (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1/output_dim), np.sqrt(1/hidden_dim),
                                   (output_dim, hidden_dim))
        
    def forward_prop(self, x):
        """
        forward propagation (predicting word probabilities)
        x is 2-D array, [time_step, input_dim]
        """
        T = len(x)
        self.S = np.zeros([T+1, self.hidden_dim])
        
        # For each time step
        activation = act()
        for t in range(T):
            h = np.dot(self.W, self.S[t]) + np.dot(self.U, x[t])
            self.S[t+1] = activation.forward(h)
        
            # one image only have one label
        # so we only need output of last time step
        return np.dot(self.V, self.S[-1])
        
    def predict(self, x):
        o = self.forward_prop(x)
        return np.argmax(o)
        
    def calculate_loss_and_accuracy(self, X, Y):
        # calculate Loss of batch example
        assert len(X) == len(Y)
        cost_function = softmax_cross_entropy()
        loss = 0.0
        acc = 0.0
        for i in range(len(Y)):
            # one image only have one label
            # so we only need output of last time step
            o = self.forward_prop(X[i])
            loss += cost_function.loss(o, Y[i])
            if np.argmax(o) == np.argmax(Y[i]):
                acc += 1.
        return loss / len(Y), acc / len(Y)
        
    def bptt(self, x, y):
        T = len(x)
        # Perform forward propagation
        o = self.forward_prop(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        # calculate the gradient of Loss with respect to 
        # output in forward propagation
        cost_function = softmax_cross_entropy()
        delta_o = cost_function.diff_x(o, y)
        # one image only have one label
        # Just need last output backwards...
        dLdV = np.outer(delta_o, self.S[T])
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o) * (1 - (self.S[T] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(T, max(0, T-self.bptt_truncate), -1):
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, self.S[bptt_step-1])
            dLdU += np.outer(delta_t, x[bptt_step-1])
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - self.S[bptt_step-1] ** 2)
        
        return [dLdU, dLdV, dLdW]
    
    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        
    def train_with_sgd(self,X_train, y_train, X_test, y_test,
                       learning_rate=0.01, nepoch=40, evaluate_loss_after=1):
        # X_train: The training data set
        # y_train: The training data labels
        # learning_rate: Initial learning rate for SGD
        # nepoch: Number of times to iterate through the complete dataset
        # evaluate_loss_after: Evaluate the loss after this many epochs
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss, acc = self.calculate_loss_and_accuracy(X_test, y_test)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{}: Loss after epoch={}, loss: {}, accuracy: {}".format(
                      time, epoch, loss, acc))
                # Adjust the learning rate if loss increases
#                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                if epoch > 1 and epoch % 10 == 0:
                    learning_rate = learning_rate * 0.5 
                    print("Setting learning rate to {}".format(learning_rate))
                sys.stdout.flush()
            # For each training example...
            assert len(X_train) == len(y_train)
            for i in range(len(y_train)):
                # One SGD step
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1