import sys
import math
import threading

import torch
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors
from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
import torch.distributed as dist

from torch.nn import Module
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from torchvision import datasets

'''this is a trial example, we use MNIST on LeNet for simple test here'''

# communication functions come in here:
def asynchronous_fetch_weights():
	''' Fetch all layer weights asynchronously. (from master) '''
	pass


def synchronous_fetch_step():
	''''synchronously fetch global step from master'''
	pass


def asynchronous_fetch_step_update():
	'''asynchronously fetch model from master'''
	pass


def asynchronous_fetch_step():
	'''synchronously fetch global step from master'''
	pass

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

class DistributedWorker:
	def __init__(rank, world_size):
		self._step_changed = False
		self._update_step = False
		self._new_step_queued = 0
		self._rank = rank
		self._world_size = world_size
		self._cur_step = 0
		self._next_step = self._cur_step + 1
		self._step_fetch_request = False

	def train(self):
		pass




