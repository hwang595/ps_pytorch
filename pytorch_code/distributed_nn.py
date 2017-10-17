from __future__ import print_function

import sys
import math
import threading
import argparse
import time

import numpy as np
from mpi4py import MPI

import torch
from torch.autograd import Variable
from torch import nn
from distributed_functions.distributed_backward import backward
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply
import torch.nn.functional as F

from torchvision import datasets, transforms

from nn_ops import NN_Trainer, accuracy
from data_loader_ops.my_data_loader import DataLoader

# normal version
from distributed_worker import *
from sync_replicas_master_nn import *

#for tmp solution
from mnist import mnist
from datasets import MNISTDataset
from cifar10 import cifar10
from datasets import Cifar10Dataset

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--network', type=str, default='LeNet', metavar='N',
                        help='which kind of network we are going to use, support LeNet and ResNet currently')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--comm-type', type=str, default='Bcast', metavar='N',
                        help='which kind of method we use during the mode fetching stage')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # this is only a simple test case
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args = add_fit_args(argparse.ArgumentParser(description='PyTorch MNIST Single Machine Test'))

    # fetch dataset
    if args.dataset == "MNIST":
        mnist_data = mnist.read_data_sets(train_dir='./mnist_data', reshape=True)
        train_set = MNISTDataset(dataset=mnist_data.train, transform=transforms.ToTensor())
    elif args.dataset == "Cifar10":
        cifar10_data = cifar10.read_data_sets(padding_size=0, reshape=True)
        train_set = Cifar10Dataset(dataset=cifar10_data.train, transform=transforms.ToTensor())

    kwargs = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'momentum':args.momentum, 'network':args.network,
                'comm_method':args.comm_type}

    if rank == 0:
        master_fc_nn = SyncReplicasMaster_NN(comm=comm, **kwargs)
        master_fc_nn.build_model()
        print("I am the master: the world size is {}, cur step: {}".format(master_fc_nn.world_size, master_fc_nn.cur_step))
        master_fc_nn.train()
        print("Done sending messages to workers!")
    else:
        worker_fc_nn = DistributedWorker(comm=comm, **kwargs)
        worker_fc_nn.build_model()
        print("I am worker: {} in all {} workers, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size-1, worker_fc_nn.next_step))
        worker_fc_nn.train(train_loader=train_set)
        print("Now the next step is: {}".format(worker_fc_nn.next_step))