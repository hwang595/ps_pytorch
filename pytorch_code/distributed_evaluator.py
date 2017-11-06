from __future__ import print_function
from mpi4py import MPI
import numpy as np

from nn_ops import NN_Trainer

import torch
from torch.autograd import Variable

import time
from datetime import datetime
import copy

class DistributedEvaluator(NN_Trainer):
    '''
    The DistributedEvaluator aims at providing a seperate node in the distributed cluster to evaluate
    the model on validation/test set and return the results
    In this version, the DistributedEvaluator will only load the model from the dir where the master
    save the model and do the evaluation task based on a user defined frequency 
    '''
    def __init__(self, **kwargs):
        self._cur_step = 0
        self._model_dir = kwargs['model_dir']
        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []

    def evaluate(self, validation_loader):

    def load_model(self, file_path):
        with open(file_path, "rb") as f_:
            self.network = torch.load(f_)
        return self.network

if __name__ == "__main__":
    # this is only a simple test case
    evaluator_nn = DistributedEvaluator()
    print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size))