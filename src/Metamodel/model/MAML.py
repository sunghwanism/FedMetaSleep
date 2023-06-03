import os

import torch

class MataModel:
    
    def __init__(self, num_updates=10, n_tasks=9, nmodals=2, K=10):
        self.ntasks = n_tasks
        self.num_update = num_updates
        self.nmodals = nmodals
        self.nsamples_per_class = K
    
    
    def init_model(self):
        
        self.num_update = self.num_updates
        ntasks = self.ntasks
        
        self.weights = self.model.construct_weights()
        
        
        
    def cond(outputa, outputb, lossa, lossb, acca, accb, i, iters):
        return torch.lt(i, iters)
    
    
    def forward(self, x, weights):
        
        