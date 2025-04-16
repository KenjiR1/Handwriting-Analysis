# -*- coding: utf-8 -*-
# Code for creating model object
# input_dim = number of features (dependent variables)
# Contains 3 neural layers, 2 activation functions

import torch.nn as nn

class TernaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3)
            )
    
    def forward(self, x):
        return self.net(x)



    
