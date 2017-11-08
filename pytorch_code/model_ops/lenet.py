import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.autograd import Variable

from mpi4py import MPI

# we use LeNet here for our simple case
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        #loss = self.ceriation(x, target)
        return x
    def name(self):
        return 'lenet'

class LeNetSplit(nn.Module):
    '''
    this is a module that we split the module and do backward process layer by layer
    please don't call this module for normal uses, this is a hack and run slower than
    the automatic chain rule version
    '''
    def __init__(self):
        super(LeNetSplit, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()

        self.full_modules = [self.conv1, self.conv2, self.fc1, self.fc2]
        self._init_channel_index = len(self.full_modules)*2

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        self.output = []
        self.input = []
        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.conv1(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.maxpool2d(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.relu(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.conv2(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.maxpool2d(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.relu(x)
        self.output.append(x)

        x = x.view(-1, 4*4*50)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.fc1(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.fc2(x)
        self.output.append(x)
        return x

    @property
    def fetch_init_channel_index(self):
        return self._init_channel_index

    def backward_normal(self, g, communicator, req_send_check, cur_step):
        mod_avail_index = len(self.full_modules)-1
        #channel_index = len(self.full_modules)*2-2
        channel_index = self._init_channel_index - 2
        mod_counters_ = [0]*len(self.full_modules)
        for i, output in reversed(list(enumerate(self.output))):
            req_send_check[-1].wait()
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
                # get gradient here after some sanity checks:
                tmp_grad = self.full_modules[mod_avail_index].weight.grad
                if not pd.isnull(tmp_grad):
                    grads = tmp_grad.data.numpy().astype(np.float64)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_send_check.append(req_isend)
                    # update counters
                    mod_avail_index-=1
                    channel_index-=1
                else:
                    continue
            else:
                output.backward(self.input[i+1].grad.data)
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad
                if not pd.isnull(tmp_grad_weight) and not pd.isnull(tmp_grad_bias):
                    # we always send bias first
                    if mod_counters_[mod_avail_index] == 0:
                        grads = tmp_grad_bias.data.numpy().astype(np.float64)
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_send_check.append(req_isend)
                        channel_index-=1
                        mod_counters_[mod_avail_index]+=1
                    elif mod_counters_[mod_avail_index] == 1:
                        grads = tmp_grad_weight.data.numpy().astype(np.float64)
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_send_check.append(req_isend)
                        channel_index-=1
                        mod_counters_[mod_avail_index]+=1
                        # update counters
                        mod_avail_index-=1
                else:
                    continue
        if mod_counters_[0] == 1:
            req_send_check[-1].wait()
            grads = tmp_grad_weight.data.numpy().astype(np.float64)
            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
            req_send_check.append(req_isend)
        return req_send_check

    def backward_signal_kill(self, g, communicator, req_send_check, cur_step):
        '''
        This killer is triggered by signals bcasting from master, channel of 
        signal is kept checking by each worker to determine if they're the 
        straggler
        '''
        mod_avail_index = len(self.full_modules)-1
        channel_index = self._init_channel_index - 2
        mod_counters_ = [0]*len(self.full_modules)

        # should kill flag
        should_kill = False

        for i, output in reversed(list(enumerate(self.output))):
        ############################ killing process on workers #####################################
            for _ in range(10000):
                status = MPI.Status()
                communicator.Iprobe(0, 77, status)
                if status.Get_source() == 0:
                    print("Worker {}, Cur Step: {} I'm the straggler, killing myself!".format(communicator.Get_rank(), cur_step))
                    tmp = communicator.recv(source=0, tag=77)
                    should_kill = True
                    break
            if should_kill:
                break
        ############################################################################################

            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
                # get gradient here after some sanity checks:
                tmp_grad = self.full_modules[mod_avail_index].weight.grad
                if not pd.isnull(tmp_grad):
                    grads = tmp_grad.data.numpy().astype(np.float64)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_send_check.append(req_isend)
                    # update counters
                    mod_avail_index-=1
                    channel_index-=1
                else:
                    continue
            else:
                output.backward(self.input[i+1].grad.data)
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad
                if not pd.isnull(tmp_grad_weight) and not pd.isnull(tmp_grad_bias):
                    # we always send bias first
                    if mod_counters_[mod_avail_index] == 0:
                        grads = tmp_grad_bias.data.numpy().astype(np.float64)
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_send_check.append(req_isend)
                        channel_index-=1
                        mod_counters_[mod_avail_index]+=1
                    elif mod_counters_[mod_avail_index] == 1:
                        grads = tmp_grad_weight.data.numpy().astype(np.float64)
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_send_check.append(req_isend)
                        channel_index-=1
                        mod_counters_[mod_avail_index]+=1
                        # update counters
                        mod_avail_index-=1
                else:
                    continue
        if mod_counters_[0] == 1:
            grads = tmp_grad_weight.data.numpy().astype(np.float64)
            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
            req_send_check.append(req_isend)
        return req_send_check

    def backward_timeout_kill(self, g, communicator, req_send_check):
        """do we even need this?"""
        pass