from __future__ import print_function
import os.path
import time
import argparse
from datetime import datetime
import copy

from mpi4py import MPI
import numpy as np

from nn_ops import NN_Trainer

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms

#for tmp solution
from mnist import mnist
from datasets import MNISTDataset
from cifar10 import cifar10
from datasets import Cifar10Dataset

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Validation settings
    parser.add_argument('--eval-batch-size', type=int, default=10000, metavar='N',
                        help='the batch size when doing model validation, complete at once on default')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--model-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    args = parser.parse_args()
    return args

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
        self._eval_freq = int(kwargs['eval_freq'])
        self._eval_batch_size = kwargs['eval_batch_size']
        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []

    def evaluate(self, validation_loader):
        # init objective to fetch at the begining
        self._next_step_to_fetch = self._cur_step + self._eval_freq
        self._num_batch_per_epoch = len(validation_loader) / self._eval_batch_size
        self._epoch_counter = validation_loader.dataset.epochs_completed
        # check if next temp model exsits, if not we wait here else
        # we continue to do the model evaluation
        while True:
            model_dir_=self._model_dir_generator(self._next_step_to_fetch)
            if os.path.isfile(model_dir_):
                self._load_model(model_dir_)
                print("Evaluator evaluating results on step {}".format(self._next_step_to_fetch))
                self._evaluate_model(validation_loader)
                self._next_step_to_fetch += self._eval_freq
            else:
                # TODO(hwang): sleep appropriate period of time make sure to tune this parameter
                time.sleep(10)

    def _evaluate_model(self, validation_loader):
        self.network.eval()
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        # which indicate an epoch based validation is done
        while validation_loader.dataset.epochs_completed <= self._epoch_counter:
            eval_image_batch, eval_label_batch = validation_loader.next_batch(batch_size=self._eval_batch_size)
            X_batch, y_batch = Variable(eval_image_batch.float()), Variable(eval_label_batch.long())
            output = self.network(X_batch)
            prec1_tmp, prec5_tmp = accuracy(output.data, eval_label_batch.long(), topk=(1, 5))
            prec1_counter_ += prec1_tmp
            prec5_counter_ += prec5_tmp
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        self._epoch_counter = validation_loader.dataset.epochs_completed
        print('Testset Performance: Cur Step:{} Prec@1: {} Prec@5: {}'.format(self._next_step_to_fetch, prec1.numpy()[0], prec5.numpy()[0]))

    def _load_model(self, file_path):
        with open(file_path, "rb") as f_:
            self.network = torch.load(f_)
        return self.network

    def _model_dir_generator(self, next_step_to_fetch):
        return self._model_dir+"model_step_"+str(next_step_to_fetch)

if __name__ == "__main__":
    # this is only a simple test case
    args = add_fit_args(argparse.ArgumentParser(description='PyTorch Distributed Evaluator'))
    # fetch dataset
    if args.dataset == "MNIST":
        mnist_data = mnist.read_data_sets(train_dir='./mnist_data', reshape=True)
        validation_set = MNISTDataset(dataset=mnist_data.validation, transform=transforms.ToTensor())
    elif args.dataset == "Cifar10":
        cifar10_data = cifar10.read_data_sets(padding_size=0, reshape=True)
        validation_set = Cifar10Dataset(dataset=cifar10_data.validation, transform=transforms.ToTensor())
    
    kwargs_evaluator={'model_dir':args.model_dir, 'eval_freq':args.eval_freq, 'eval_batch_size':args.eval_batch_size}
    evaluator_nn = DistributedEvaluator(**kwargs_evaluator)
    evaluator_nn.evaluate(validation_loader=validation_set)
    print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size))