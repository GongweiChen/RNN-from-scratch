# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:06:39 2017

@author: Gongwei Chen
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

#from datetime import datetime
#import numpy as np
#import sys
from rnn import RNNModel
import MNIST

data_dir = 'D:\\Bashfiles\\lha-k40-files\\data\\MNIST'

def get_train_data(data_dir):
    dataset = MNIST.load_mnist(data_dir)
    return dataset.train, dataset.test
    
def main():
    train, test = get_train_data(data_dir)
    input_dim = train.images.shape[2]
    hidden_dim = 30
    output_dim = train.labels.shape[1]
    RNN = RNNModel(input_dim, hidden_dim, output_dim)
    RNN.train_with_sgd(train.images, train.labels, test.images, test.labels)
    
if __name__ == '__main__':
    main()
