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
import torch.nn.functional as F

from torchvision import datasets, transforms

from nn_ops import NN_Trainer, accuracy
from data_loader_ops.my_data_loader import DataLoader

from distributed_worker import *
from sync_replicas_master_nn import *


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--max-steps', type=int, default=10000, metavar='N',
                        help='the maximum number of iterations')
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
    parser.add_argument('--mode', type=str, default='normal', metavar='N',
                        help='determine if we kill the stragglers or just implement normal training')
    parser.add_argument('--kill-threshold', type=float, default=7.0, metavar='KT',
                        help='timeout threshold which triggers the killing process (default: 7s)')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--comm-type', type=str, default='Bcast', metavar='N',
                        help='which kind of method we use during the mode fetching stage')
    parser.add_argument('--num-aggregate', type=int, default=5, metavar='N',
                        help='how many number of gradients we wish to gather at each iteration')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--train-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--compress-grad', type=str, default='compress', metavar='N',
                        help='compress/none indicate if we compress the gradient matrix before communication')
    parser.add_argument('--enable-gpu', type=bool, default=False, help='whether to use gradient approx method')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # this is only a simple test case
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args = add_fit_args(argparse.ArgumentParser(description='PyTorch MNIST Single Machine Test'))

    # load training and test set here:
    if args.dataset == "MNIST":
        training_set = datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.test_batch_size, shuffle=True)
    elif args.dataset == "Cifar10":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True)
        testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False)

    kwargs_master = {'batch_size':args.batch_size, 
                'learning_rate':args.lr, 
                'max_epochs':args.epochs, 
                'momentum':args.momentum, 
                'network':args.network, 
                'comm_method':args.comm_type, 
                'kill_threshold': args.num_aggregate, 
                'timeout_threshold':args.kill_threshold, 
                'eval_freq':args.eval_freq, 
                'train_dir':args.train_dir, 
                'max_steps':args.max_steps, 
                'compress_grad':args.compress_grad}

    kwargs_worker = {'batch_size':args.batch_size, 
                'learning_rate':args.lr, 
                'max_epochs':args.epochs, 
                'momentum':args.momentum, 
                'network':args.network,
                'comm_method':args.comm_type, 
                'kill_threshold':args.kill_threshold, 
                'eval_freq':args.eval_freq, 
                'train_dir':args.train_dir, 
                'max_steps':args.max_steps, 
                'compress_grad':args.compress_grad, 
                'enable_gpu':args.enable_gpu}

    if rank == 0:
        master_fc_nn = SyncReplicasMaster_NN(comm=comm, **kwargs_master)
        master_fc_nn.build_model()
        print("I am the master: the world size is {}, cur step: {}".format(master_fc_nn.world_size, master_fc_nn.cur_step))
        master_fc_nn.start()
        print("Done sending messages to workers!")
    else:
        worker_fc_nn = DistributedWorker(comm=comm, **kwargs_worker)
        worker_fc_nn.build_model()
        print("I am worker: {} in all {} workers, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size-1, worker_fc_nn.next_step))
        worker_fc_nn.train(train_loader=train_loader, test_loader=test_loader)
        print("Now the next step is: {}".format(worker_fc_nn.next_step))