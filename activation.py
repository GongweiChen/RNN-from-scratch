# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:13:13 2017

@author: Gongwei Chen
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

class sigmoid(object):
    
    def forward(self, x):
        output = 1.0 / (1.0 + np.exp(-x))
        return output
        
    def diff_x(self, x):
        # The differentiation of output with respect to x
        output = self.forward(x)
        return output * (1.0 - output)
        

class tanh(object):
    
    def forward(self, x):
        output = np.tanh(x)
        return output
        
    def diff_x(self, x):
        # The differentiation of output with respect to x
        output = self.forward(x)
        return 1.0 - output ** 2
        
