# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:11:11 2017

@author: Gongwei Chen
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

class softmax_cross_entropy(object):
    
    def loss(self, x, y):
        # x is 1-D array, weighted inputs
        # y is the target
        self.output = np.exp(x)
        self.output = self.output / self.output.sum()
        return -np.log(self.output).dot(y)
        
    def diff_x(self, x, y):
        self.loss(x, y)
        return y.sum()*self.output - y
        