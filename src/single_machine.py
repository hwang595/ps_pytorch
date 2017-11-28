import sys
import math
import threading
import argparse
import time

import torch
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors
from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl

import torch.nn as nn
from distributed_functions.distributed_backward import backward
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply
import torch.nn.functional as F

from torchvision import datasets, transforms

from nn_ops import NN_Trainer, accuracy
from data_loader_ops.my_data_loader import DataLoader

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
    args = parser.parse_args()
    return args

# we use LeNet here for our simple case
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x, target):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        loss = self.ceriation(x, target)
        return x, loss
    def name(self):
        return 'lenet'

class LeNetLearner:
    """a deprecated class, please don't call this one in any time"""
    def __init__(self, rank, world_size, args):
        self._step_changed = False
        self._update_step = False
        self._new_step_queued = 0
        self._rank = rank
        self._world_size = world_size
        self._cur_step = 0
        self._next_step = self._cur_step + 1
        self._step_fetch_request = False
        self.max_num_epochs = args.epochs
        self.lr = args.lr
        self.momentum = args.momentum

    def build_model(self):
        self.network = LeNet()

        # only for test use
        self.module = self.network

        # this is only used for test
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)

    def test_model(self):
        '''this is only for test, please don't call this function'''
        from copy import deepcopy
        self._module_copies = [deepcopy(self.module)]
        self.device_ids = []

        t = None
        for p in self.module.parameters():
            tp = type(p.data)
            if t is not None and t is not tp:
                raise ValueError("DistributedDataParallel requires all parameters' data to be of the same type")
            t = tp

        self.bucket_sizes = []
        self.bucket_map = {}
        MB = 1024 * 1024
        self.broadcast_bucket_size = 10 * MB  # used for param sync before forward
        bucket_bytes_cap = 1 * MB
        bucket_bytes = bucket_bytes_cap  # to init the first bucket immediately
        for param_tuple in zip(*map(lambda m: m.parameters(), self._module_copies)):
            if bucket_bytes >= bucket_bytes_cap:
                self.bucket_sizes.append(0)
                bucket_bytes = 0
            self.bucket_sizes[-1] += 1
            for p in param_tuple:
                self.bucket_map[p] = len(self.bucket_sizes) - 1
            bucket_bytes += p.numel() * p.element_size()

        self.buckets = [[[] for _ in range(len(self.device_ids))] for _ in range(len(self.bucket_sizes))]
        self.bucket_events = [[None] * len(self.device_ids) for _ in range(len(self.bucket_sizes))]
        self.reduced = [False] * len(self.bucket_sizes)

    def train(self, train_loader=None):
        self.network.train()

        # iterate of epochs
        for i in range(self.max_num_epochs):            
            for batch_idx, (data, y_batch) in enumerate(train_loader):
                iter_start_time = time.time()
                data, target = Variable(data), Variable(y_batch)
                self.optimizer.zero_grad()
                logits, loss = self.network(data, target)
                tmp_time_0 = time.time()
                loss.backward()

                for params in self.network.parameters():
                    print(params.grad.data.numpy())
                    print('**********************************************************')

                if batch_idx == 5:
                    self.update_state_dict()
                
                duration_backward = time.time()-tmp_time_0

                tmp_time_1 = time.time()
                self.optimizer.step()
                duration_update = time.time()-tmp_time_1

                print("backward duration: {}".format(duration_backward))
                print("update duration: {}".format(duration_update))
                # calculate training accuracy
                prec1, prec5 = accuracy(logits.data, y_batch, topk=(1, 5))
                # load the training info
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Prec@1: {}  Prec@5: {}  Time Cost: {}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0], 
                    prec1.numpy()[0], 
                    prec5.numpy()[0], time.time()-iter_start_time))

    def update_state_dict(self):
        """for this test version, we set all params to zeros here"""
        # we need to build a state dict first
        new_state_dict = {}
        for key_name, param in self.network.state_dict().items():
            tmp_dict = {key_name: torch.FloatTensor(param.size()).zero_()}
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)


if __name__ == "__main__":
    args = add_fit_args(argparse.ArgumentParser(description='PyTorch MNIST Single Machine Test'))

    kwargs = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'momentum':args.momentum, 'network':args.network}

    # load training and test set here:
    if args.dataset == "MNIST":
        training_set = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.test_batch_size, shuffle=True)
    elif args.dataset == "Cifar10":
        trainset = datasets.CIFAR10(root='./cifar10_data', train=True,
                                                download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./cifar10_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])), batch_size=args.test_batch_size, shuffle=True)

    nn_learner = NN_Trainer(**kwargs)
    nn_learner.build_model()
    nn_learner.train_and_validate(train_loader=train_loader, test_loader=test_loader)    