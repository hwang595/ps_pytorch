from __future__ import print_function

import time

import numpy as np
import torch
from torch.autograd import Variable

from model_ops.lenet import LeNet
from model_ops.resnet import *
from model_ops.resnet_split import *

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
        self.network_config = kwargs['network']

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNet()
        elif self.network_config == "ResNet":
            #self.network=ResNet18()
            self.network=ResNetSplit18(1)
        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_and_validate(self, train_loader, test_loader):
        # iterate of epochs
        for i in range(self.max_epochs):
            # change back to training mode
            self.network.train()      
            for batch_idx, (data, y_batch) in enumerate(train_loader):
                iter_start_time = time.time()
                data, target = Variable(data), Variable(y_batch)
                self.optimizer.zero_grad()
                ################# backward on normal model ############################
                '''
                logits = self.network(data)
                loss = self.criterion(logits, target)
                '''
                #######################################################################

                ################ backward on splitted model ###########################
                logits = self.network(data)
                logits_1 = Variable(logits.data, requires_grad=True)
                loss = self.criterion(logits_1, target)
                loss.backward()
                self.network.backward_single(logits_1.grad)
                #######################################################################
                tmp_time_0 = time.time()
                
                for param in self.network.parameters():
                    # get gradient from layers here
                    # in this version we fetch weights at once
                    # remember to change type here, which is essential
                    grads = param.grad.data.numpy().astype(np.float64)

                duration_backward = time.time()-tmp_time_0

                tmp_time_1 = time.time()
                self.optimizer.step()
                duration_update = time.time()-tmp_time_1

                # calculate training accuracy
                prec1, prec5 = accuracy(logits.data, y_batch, topk=(1, 5))
                # load the training info
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Prec@1: {}  Prec@5: {}  Time Cost: {}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0], 
                    prec1.numpy()[0], 
                    prec5.numpy()[0], time.time()-iter_start_time))
            # we evaluate the model performance on end of each epoch
            self.validate(test_loader)

    def validate(self, test_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        for data, y_batch in test_loader:
            data, target = Variable(data, volatile=True), Variable(y_batch)
            output = self.network(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            prec1_tmp, prec5_tmp = accuracy(output.data, y_batch, topk=(1, 5))
            prec1_counter_ += prec1_tmp.numpy()[0]
            prec5_counter_ += prec5_tmp.numpy()[0]
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Prec@1: {} Prec@5: {}'.format(test_loss, prec1, prec5))