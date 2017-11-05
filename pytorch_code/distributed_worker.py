from __future__ import print_function
from mpi4py import MPI
import numpy as np

from nn_ops import NN_Trainer
from timeout import Timeout, TimeoutError
from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *

import torch
from torch.autograd import Variable

import time
from datetime import datetime
import copy

STEP_START_ = 1

TAG_LIST_ = [i*30 for i in range(50000)]

LAYER_DIGITS= int(1e+3)

def prepare_grad_list(params):
    grad_list = []
    for param_idx, param in enumerate(params):
        # get gradient from layers here
        # in this version we fetch weights at once
        # remember to change type here, which is essential
        #grads = param.grad.data.numpy().astype(np.float64)
        grads = param.grad.data.numpy().astype(np.float64)
        grad_list.append((param_idx, grads))
    return grad_list


class ModelBuffer(object):
    def __init__(self, network):
        """
        this class is used to save model weights received from parameter server
        current step for each layer of model will also be updated here to make sure
        the model is always up-to-date
        """
        self.recv_buf = []
        self.layer_cur_step = []
        '''initialize space to receive model from parameter server'''
        #for key_name, param in network.state_dict().items():
        #    self.recv_buf.append(np.zeros(param.size()))
        #    self.layer_cur_step.append(0)

        # consider we don't want to update the param of `BatchNorm` layer right now
        # we temporirially deprecate the foregoing version and only update the model
        # parameters
        for param_idx, param in enumerate(network.parameters()):
            self.recv_buf.append(np.zeros(param.size()))
            self.layer_cur_step.append(0)


class DistributedWorker(NN_Trainer):
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.rank = comm.Get_rank() # rank of this Worker
        #self.status = MPI.Status()
        self.cur_step = 0
        self.next_step = 0 # we will fetch this one from parameter server

        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.lr = kwargs['learning_rate']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self.kill_threshold = kwargs['kill_threshold']

        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNetSplit()
        elif self.network_config == "ResNet18":
            self.network=ResNetSplit18(self.kill_threshold)
        elif self.network_config == "ResNet34":
            self.network=ResNetSplit34()

        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf()
        #self._param_idx = len(self.network.full_modules)*2-1
        self._param_idx = self.network.fetch_init_channel_index-1

    def train(self, train_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1
        self.sync_fetch_step()
        # do some sync check here
        assert(self.update_step())
        assert(self.cur_step == STEP_START_)

        # number of batches in one epoch
        self._num_batch_per_epoch = len(train_loader) / self.batch_size
        self._batch_idx = -1
        self._epoch_idx = 0
        self._epoch_avg_loss = 0
        self._iteration_last_step=0
        self._iter_start_time=0

        # init time costs here in case non of these are assigned
        fetch_weight_duration=0
        forward_duration=0
        backward_duration=0

        first = True

        print("Worker {}: starting training".format(self.rank))
        # start the training process
        while True:
            # the worker shouldn't know the current global step
            # except received the message from parameter server
            self.async_fetch_step()

            # the only way every worker know which step they're currently on is to check the cur step variable
            updated = self.update_step()

            if (not updated) and (not first):
                # wait here unitl enter next step
                continue

            # the real start point of this iteration
            iter_start_time = time.time()
            first = False
            print("Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))


            # only consider triggering the killing process after the first iteration
            if self.cur_step > 1:
                try:
                    with Timeout(seconds=self.kill_threshold):
                        fetch_weight_duration, forward_duration, backward_duration=self.training_process(train_loader)
                except TimeoutError:
                    print("Worker: {} Timeout".format(self.rank))
            else:
                fetch_weight_duration, forward_duration, backward_duration=self.training_process(train_loader)

            # on the end of a certain iteration
            print('Worker: {}, Cur Step: {}, Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.4f}, Time Cost: {:.4f}, FetchWeight: {:.4f}, Forward: {:.4f}, Backward: {:.4f}'.format(self.rank,
                    self.cur_step, self._epoch_idx, self._batch_idx * self.batch_size, self.batch_size*self._num_batch_per_epoch, 
                    (100. * (self._batch_idx * self.batch_size) / (self.batch_size*self._num_batch_per_epoch)), self._loss_data, time.time()-iter_start_time, fetch_weight_duration, forward_duration, backward_duration))
    
    def training_process(self, train_loader):
        fetch_weight_start_time = time.time()
        if self.comm_type == "Bcast":
            self.async_fetch_weights_bcast()
        elif self.comm_type == "Async":
            self.async_fetch_weights_async()

        fetch_weight_duration = time.time() - fetch_weight_start_time
        time_point_log = datetime.now()

        # start the normal training process
        train_image_batch, train_label_batch = train_loader.next_batch(batch_size=self.batch_size)
        X_batch, y_batch = Variable(train_image_batch.float()), Variable(train_label_batch.long())

        # manage batch index manually
        self._batch_idx += 1
        self.optimizer.zero_grad()

        if self._batch_idx == self._num_batch_per_epoch - 1:
            self._batch_idx = 0
            self._epoch_avg_loss /= self._num_batch_per_epoch
            print("Average for epoch {} is {}".format(self._epoch_idx, self._epoch_avg_loss))
            self._epoch_idx += 1
            self._epoch_avg_loss = 0

        # forward step
        forward_start_time = time.time()
        logits = self.network(X_batch)
        logits_1 = Variable(logits.data, requires_grad=True)

        loss = self.criterion(logits_1, y_batch)
        self._loss_data = loss.data[0]

        self._epoch_avg_loss += loss.data[0]
        forward_duration = time.time()-forward_start_time

        # backward step
        backward_start_time = time.time()
        loss.backward()
        # we can send the grad of this very first layer to parameter server right here before
        # the chain rule is begining
        req_send_check = []
        init_grad_data = logits_1.grad.data.numpy()
        init_grad_data = np.sum(init_grad_data, axis=0).astype(np.float64)
        # send grad to parameter server
        #req_isend = self.comm.Isend([init_grad_data, MPI.DOUBLE], dest=0, tag=88+self._param_idx)
        req_isend = self.comm.Isend([init_grad_data, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+self._param_idx, step_token=self.cur_step))
        req_send_check.append(req_isend)
        # Try signal killing method here:
        #req_send_check, killed=self.network.backward_signal_kill(logits_1.grad, communicator=self.comm, req_send_check=req_send_check, cur_step=self.cur_step)
        
        # Try Timeout killing strategy this time:
        '''
        if self.cur_step == 1:
            req_send_check=self.network.backward(logits_1.grad, communicator=self.comm, req_send_check=req_send_check, cur_step=self.cur_step)
            req_send_check[-1].wait()
        else:
            try:
                req_send_check = self.network.backward_timeout_kill(logits_1.grad, communicator=self.comm, req_send_check=req_send_check, cur_step=self.cur_step)
                req_send_check[-1].wait()
            except StopIteration:
                print("Worker: {} Timeout".format(self.rank))
        '''
        # Normal backward
        req_send_check=self.network.backward(logits_1.grad, communicator=self.comm, req_send_check=req_send_check, cur_step=self.cur_step)
        req_send_check[-1].wait()
        backward_duration = time.time()-backward_start_time
        return fetch_weight_duration, forward_duration, backward_duration

    def init_recv_buf(self):
        self.model_recv_buf = ModelBuffer(self.network)

    def sync_fetch_step(self):
        '''fetch the first step from the parameter server'''
        self.next_step = self.comm.recv(source=0, tag=10)

    def async_fetch_step(self):
        req = self.comm.irecv(source=0, tag=10)
        self.next_step = req.wait()

    def async_fetch_weights_async(self):
        request_layers = []
        layers_to_update = []
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            if self.model_recv_buf.layer_cur_step[layer_idx] < self.cur_step:
                layers_to_update.append(layer_idx)
                req = self.comm.Irecv([self.model_recv_buf.recv_buf[layer_idx], MPI.DOUBLE], source=0, tag=11+layer_idx)
                request_layers.append(req)

        assert (len(layers_to_update) == len(request_layers))
        weights_to_update = []
        for req_idx, req_l in enumerate(request_layers):
            req_l.wait()
            weights = self.model_recv_buf.recv_buf[req_idx]
            weights_to_update.append(weights)
            # we also need to update the layer cur step here:
            self.model_recv_buf.layer_cur_step[req_idx] = self.cur_step
        self.model_update(weights_to_update)
    
    def async_fetch_weights_bcast(self):
        layers_to_update = []
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            if self.model_recv_buf.layer_cur_step[layer_idx] < self.cur_step:
                layers_to_update.append(layer_idx)
                #req = self.comm.Irecv([self.model_recv_buf.recv_buf[layer_idx], MPI.DOUBLE], source=0, tag=11+layer_idx)
                #request_layers.append(req)
                self.comm.Bcast([self.model_recv_buf.recv_buf[layer_idx], MPI.DOUBLE], root=0)
        weights_to_update = []
        for req_idx, layer_idx in enumerate(layers_to_update):
            weights = self.model_recv_buf.recv_buf[req_idx]
            weights_to_update.append(weights)
            # we also need to update the layer cur step here:
            self.model_recv_buf.layer_cur_step[req_idx] = self.cur_step
        self.model_update(weights_to_update)
    
    def update_step(self):
        '''update local (global) step on worker'''
        changed = (self.cur_step != self.next_step)
        self.cur_step = self.next_step
        return changed

    def model_update(self, weights_to_update):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        model_counter_ = 0
        for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
            # handle the case that `running_mean` and `running_var` contained in `BatchNorm` layer
            if "running_mean" in key_name or "running_var" in key_name:
                tmp_dict={key_name: param}
            else:
                assert param.size() == weights_to_update[model_counter_].shape
                tmp_dict = {key_name: torch.from_numpy(weights_to_update[model_counter_])}
                model_counter_ += 1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

if __name__ == "__main__":
    # this is only a simple test case
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    worker_fc_nn = WorkerFC_NN(comm=comm, world_size=world_size, rank=rank)
    print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size))