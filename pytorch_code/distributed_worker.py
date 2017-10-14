from __future__ import print_function
from mpi4py import MPI
import numpy as np

from nn_ops import NN_Trainer
from model_ops.lenet import LeNet
from model_ops.resnet import *

import torch
from torch.autograd import Variable

import time

STEP_START_ = 1

TAG_LIST_ = [i*30 for i in range(50000)]

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

        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNet()
        elif self.network_config == "ResNet":
            self.network=ResNet18()

        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf()

    def train(self, train_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1
        self.sync_fetch_step()
        # do some sync check here
        assert(self.update_step())
        assert(self.cur_step == STEP_START_)

        # number of batches in one epoch
        num_batch_per_epoch = len(train_loader) / self.batch_size
        batch_idx = -1
        epoch_idx = 0
        epoch_avg_loss = 0
        iteration_last_step=0
        iter_start_time=0

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
            iteration_last_step = time.time() - iter_start_time
            iter_start_time = time.time()
            first = False
            print("Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))

            # TODO(hwang): return layer request here and do weight before the forward step begins, rather than implement
            # the wait() in the fetch function
            fetch_weight_start_time = time.time()
            self.async_fetch_weights()
            fetch_weight_duration = time.time() - fetch_weight_start_time

            # start the normal training process
            train_image_batch, train_label_batch = train_loader.next_batch(batch_size=self.batch_size)
            X_batch, y_batch = Variable(train_image_batch.float()), Variable(train_label_batch.long())

            # manage batch index manually
            batch_idx += 1
            self.optimizer.zero_grad()

            if batch_idx == num_batch_per_epoch - 1:
                batch_idx = 0
                epoch_avg_loss /= num_batch_per_epoch
                print("Average for epoch {} is {}".format(epoch_idx, epoch_avg_loss))
                epoch_idx += 1
                epoch_avg_loss = 0

            # forward step
            forward_start_time = time.time()
            logits = self.network(X_batch)
            loss = self.criterion(logits, y_batch)
            epoch_avg_loss += loss.data[0]
            forward_duration = time.time()-forward_start_time

            # backward step
            backward_start_time = time.time()
            loss.backward()
            backward_duration = time.time()-backward_start_time
            # TODO(hwang): figure out the killing process in pytorch framework asap
            req_send_check = []
            grad_list = prepare_grad_list(self.network.parameters())

            send_grad_start_time = time.time()
            for grad_index_tuple in reversed(grad_list):
                param_idx = grad_index_tuple[0]
                grads = grad_index_tuple[1]
                if len(req_send_check) != 0:
                    req_send_check[-1].wait()
                req_isend = self.comm.Isend([grads, MPI.DOUBLE], dest=0, tag=88+param_idx)
                #req_isend = self.comm.Isend([grads, MPI.FLOAT], dest=0, tag=88+param_idx)
                req_send_check.append(req_isend)
            send_grad_duration = time.time() - send_grad_start_time
            # on the end of a certain iteration
            print('Worker: {}, Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.4f}, Time Cost: {:.4f}, FetchWeight: {:.4f}, Forward: {:.4f}, Backward: {:.4f}, SendGrad:{}'.format(self.rank,
                    epoch_idx, batch_idx * self.batch_size, self.batch_size*num_batch_per_epoch, 
                    (100. * (batch_idx * self.batch_size) / (self.batch_size*num_batch_per_epoch)), loss.data[0], time.time()-iter_start_time, fetch_weight_duration, forward_duration, backward_duration, send_grad_duration))

    def init_recv_buf(self):
        self.model_recv_buf = ModelBuffer(self.network)

    def sync_fetch_step(self):
        '''fetch the first step from the parameter server'''
        self.next_step = self.comm.recv(source=0, tag=10)

    def async_fetch_step(self):
        req = self.comm.irecv(source=0, tag=10)
        self.next_step = req.wait()

    def async_fetch_weights_old(self):
        '''
        this is the original MPI code, deprecated
        TODO(hwang): remember to remove it after testing
        '''
        request_layers = []
        # keep in mind that this list saved the mapped layer index
        layers_to_update = []
        for layer_idx, layer in enumerate(self.module):
            if layer.is_fc_layer:
                # do a weird layer index map here:
                layer_map_idx = self.fc_layer_counts.index(layer_idx)
                # Check if we have already fetched the weights for this
                # particular step. If so, don't fetch it.
                if self._layer_cur_step[layer_map_idx] < self.cur_step:
                    layers_to_update.append(layer_map_idx)
                    req = self.comm.Irecv([layer.recv_buf, MPI.DOUBLE], source=0, tag=11+layer_idx)
                    request_layers.append(req)

        assert (len(layers_to_update) == len(request_layers))
        for req_idx, req_l in enumerate(request_layers):
            fc_layer_idx = self.fc_layer_counts[req_idx]
            req_l.wait() # got the weights
            weights = self.module[fc_layer_idx].recv_buf
            # we also need to update the layer cur step here:
            self._layer_cur_step[layers_to_update[req_idx]] = self.cur_step

            # assign fetched weights to weights of module containers
            assert (self.module[fc_layer_idx].W.shape == weights[0:self.module[fc_layer_idx].get_shape[0], :].shape)
            self.module[fc_layer_idx].W = weights[0:self.module[fc_layer_idx].get_shape[0], :]
            assert (self.module[fc_layer_idx].b.shape == weights[self.module[fc_layer_idx].get_shape[0], :].shape)
            self.module[fc_layer_idx].b = weights[self.module[fc_layer_idx].get_shape[0], :]

    def async_fetch_weights(self):
        request_layers = []
        layers_to_update = []
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            if self.model_recv_buf.layer_cur_step[layer_idx] < self.cur_step:
                layers_to_update.append(layer_idx)
                #print(self.model_recv_buf.recv_buf[layer_idx].shape)
                #print('----------------------------------------------------------------------------')
                req = self.comm.Irecv([self.model_recv_buf.recv_buf[layer_idx], MPI.DOUBLE], source=0, tag=11+layer_idx)
                #req = self.comm.Irecv([self.model_recv_buf.recv_buf[layer_idx], MPI.FLOAT], source=0, tag=11+layer_idx)
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