from __future__ import print_function

from mpi4py import MPI
import numpy as np

import time
import sys
import math
import threading
import argparse
import time

import torch
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors

import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

'''this is a trial example, we use MNIST on LeNet for simple test here'''
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

def receive_grads(comm=None):
	'''this is just a test function, we assume there are only two threads in our program'''
	# initialize bunch of buffers to receive the grads
	grad_buffer = []

	#grad_buffer.append(np.zeros((20, 1, 5, 5)))
	grad_buffer.append(np.zeros((500, )))
	grad_buffer.append(np.zeros((20,)))
	grad_buffer.append(np.zeros((50, 20, 5, 5)))
	grad_buffer.append(np.zeros((50,)))
	grad_buffer.append(np.zeros((500, 800)))
	grad_buffer.append(np.zeros((500, )))
	grad_buffer.append(np.zeros((10, 500)))
	grad_buffer.append(np.zeros((10, )))


	grad_recv_req_list = []

	for buffer_idx, gf in enumerate(grad_buffer):
		req = comm.Irecv([gf, MPI.DOUBLE], source=1, tag=buffer_idx)
		grad_recv_req_list.append(req)

	for req in grad_recv_req_list:
		req.wait()

	print("All grads are received, start printing what I received!")
	print(grad_buffer[0])
	for grad in grad_buffer:
		print(grad)
		print('----------------------------------------------------')

	print("Done!")

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
    def __init__(self, comm, args):
    	self._comm = comm

        self._rank = comm.Get_rank()
        self._world_size = comm.Get_size()
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

                req_send_check = []
                for param_idx, param in enumerate(self.network.parameters()):
                    # get gradient from layers here
                    # in this version we fetch weights at once
                    # remember to change type here, which is essential
                    grads = param.grad.data.numpy().astype(np.float64)

                   
                    if len(req_send_check) != 0:
                    	req_send_check[-1].wait()

                    print(grads)
                    print("===============================================================")
                    req_isend = self._comm.Isend([grads, MPI.DOUBLE], dest=0, tag=param_idx)
                    req_send_check.append(req_isend)
                   
                print("Worker {} Done sending grads!".format(self._rank))
                exit()

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


if __name__ == "__main__":
    # initialize communication world
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
    	# action as master node
    	print("I'm the master")
    	receive_grads(comm=comm)
    else:
		print("I'm worker {}".format(rank))
		# act as worker nodes
		args = add_fit_args(argparse.ArgumentParser(description='PyTorch MNIST Example'))

		# load training and test set here:
		train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../../data', train=True, download=True,
		               transform=transforms.Compose([
		                   transforms.ToTensor(),
		                   transforms.Normalize((0.1307,), (0.3081,))
		               ])), batch_size=args.batch_size, shuffle=True)

		test_loader = torch.utils.data.DataLoader(
		    datasets.MNIST('../../data', train=False, transform=transforms.Compose([
		                       transforms.ToTensor(),
		                       transforms.Normalize((0.1307,), (0.3081,))
		                   ])), batch_size=args.test_batch_size, shuffle=True)

		dist_worker = LeNetLearner(comm=comm, args=args)
		dist_worker.build_model()
		dist_worker.train(train_loader=train_loader)