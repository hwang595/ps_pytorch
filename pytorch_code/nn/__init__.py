from __future__ import print_function

import time

import numpy as np
import torch
from torch.autograd import Variable

from model_ops.lenet import LeNet

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

class NN_Trainer(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.lr = kwargs['learning_rate']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']

    def build_model(self):
        # build network
        self.network=LeNet()
        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)

    def train(self, train_loader):
        self.network.train()

        # iterate of epochs
        for i in range(self.max_epochs):            
            for batch_idx, (data, y_batch) in enumerate(train_loader):
                iter_start_time = time.time()
                data, target = Variable(data), Variable(y_batch)
                self.optimizer.zero_grad()
                logits, loss = self.network(data, target)
                tmp_time_0 = time.time()
                loss.backward()

                for layer_idx, layer in enumerate(self.network.parameters()):
                    print(layer.data.numpy())
                    print("===========================================================")

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