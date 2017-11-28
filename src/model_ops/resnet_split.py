'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385


Please Note that, this version is a hack, it's super hacky, never call this one for normal use
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import pandas as pd
import numpy as np
import timeout_decorator

from mpi4py import MPI

import sys
sys.path.insert(0, '../compress_gradient')
from compress_gradient import compress

LAYER_DIGITS= int(1e+3)
TIMEOUT_THRESHOLD_=10

def generate_tag(layer_tag, step_token):
    '''
    Tag component [current-step-token (which help to recogize stale gradient)
                   +layer-tag]
    we only limit the digits for layer tag here since step token can be 
    extremely large e.g. 10k steps

    :param layer_tag
    :param step token
    :return:
    '''
    tag = step_token * LAYER_DIGITS \
          + layer_tag
    tag = int(tag)
    return tag

class BasicBlockSplit(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockSplit, self).__init__()
        self.full_modules = []

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.full_modules.append(self.conv1)

        self.bn1 = nn.BatchNorm2d(planes)
        self.full_modules.append(self.bn1)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.full_modules.append(self.conv2)

        self.bn2 = nn.BatchNorm2d(planes)
        self.full_modules.append(self.bn2)

        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            self.full_modules.append(self.shortcut[0])
            self.full_modules.append(self.shortcut[1])

    def forward(self, x, input_list, output_list):
        '''
        the input_list and output_list here is similar to input/output in ResNet class
        '''
        # we skip the detach and append operation on the very first x here 
        # since that's done outside of this function
        out = self.conv1(x)
        output_list.append(out)

        out = Variable(out.data, requires_grad=True)
        input_list.append(out)
        out = self.bn1(out)
        output_list.append(out)

        out = Variable(out.data, requires_grad=True)
        input_list.append(out)
        out = self.relu(out)
        output_list.append(out)

        out = Variable(out.data, requires_grad=True)
        input_list.append(out)
        out = self.conv2(out)
        output_list.append(out)

        out = Variable(out.data, requires_grad=True)
        input_list.append(out)
        out = self.bn2(out)
        output_list.append(out)

        # TODO(hwang): figure out if this part also need hack
        out += self.shortcut(x)

        out = Variable(out.data, requires_grad=True)
        input_list.append(out)
        out = self.relu(out)
        output_list.append(out)
        return out, input_list, output_list


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # we skip the detach operation on the very first x here since that's done outside of this function
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetSplit(nn.Module):
    def __init__(self, block, num_blocks, kill_threshold, num_classes=10):
        super(ResNetSplit, self).__init__()
        global TIMEOUT_THRESHOLD_
        self.in_planes = 64
        self.full_modules = []

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.full_modules.append(self.conv1)

        self.bn1 = nn.BatchNorm2d(64)
        self.full_modules.append(self.bn1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.full_modules.append(self.linear)

        self.relu = nn.ReLU()
        self.avg_pool2d = nn.AvgPool2d(kernel_size=4)
        self._init_channel_index = self.count_channel_index()
        self._timeout_threshold = kill_threshold
        TIMEOUT_THRESHOLD_ = self._timeout_threshold
        self.killed_request_list = []

    @property
    def fetch_init_channel_index(self):
        return self._init_channel_index

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            block_layers = block(self.in_planes, planes, stride)
            layers.append(block_layers)
            for m in block_layers.full_modules:
                self.full_modules.append(m)

            self.in_planes = planes * block.expansion
        layers_split = nn.ModuleList(layers)

        return layers_split

    def forward(self, x):
        # use these containers to save intermediate variables
        self.output = []
        self.input = []

        # start the forward process right here implement the following logic to every intermediate var:
        # detach from previous history
        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.conv1(x)
        # add to list of outputs
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.bn1(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.relu(x)
        self.output.append(x)

        # start to handle blocks
        for layer in self.layer1:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)

        for layer in self.layer2:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)

        for layer in self.layer3:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)

        for layer in self.layer4:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.avg_pool2d(x)
        self.output.append(x)

        x = x.view(x.size(0), -1)
        
        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.linear(x)
        self.output.append(x)
        return x

    def count_channel_index(self):
        channel_index_ = 0
        for k, v in self.state_dict().items():
            if "running_mean" in k or "running_var" in k:
                continue
            else:
                channel_index_ += 1
        return channel_index_

    def backward(self, g, communicator, req_send_check, cur_step):
        mod_avail_index = len(self.full_modules)-1
        channel_index = self._init_channel_index-2
        mod_counters_ = [0]*len(self.full_modules)
        for i, output in reversed(list(enumerate(self.output))):
            # send layer only after the last layer is received
            req_send_check[-1].wait()
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
                # get gradient here after some sanity checks:
                tmp_grad = self.full_modules[mod_avail_index].weight.grad
                if not pd.isnull(tmp_grad):
                    grads = tmp_grad.data.numpy().astype(np.float64)
                    #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                    req_send_check.append(req_isend)
                    # update counters
                    mod_avail_index-=1
                    channel_index-=1
                else:
                    continue
            else:
                if output.size() == self.input[i+1].grad.size():
                    output.backward(self.input[i+1].grad.data)
                else:
                    tmp_grad_output = self.input[i+1].grad.view(output.size())
                    output.backward(tmp_grad_output)

                # since in resnet we do not use bias weight for conv layer
                if pd.isnull(self.full_modules[mod_avail_index].bias):
                    tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad

                    if not pd.isnull(tmp_grad_weight):
                        grads = tmp_grad_weight.data.numpy().astype(np.float64)
                        #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                        req_send_check.append(req_isend)
                        channel_index-=1
                        mod_counters_[mod_avail_index]=2
                        # update counters
                        mod_avail_index-=1
                    else:
                        continue
                else:
                    tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                    tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad

                    if not pd.isnull(tmp_grad_weight) and not pd.isnull(tmp_grad_bias):
                        # we always send bias first
                        if mod_counters_[mod_avail_index] == 0:
                            grads = tmp_grad_bias.data.numpy().astype(np.float64)
                            #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                            req_send_check.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                        elif mod_counters_[mod_avail_index] == 1:
                            grads = tmp_grad_weight.data.numpy().astype(np.float64)
                            #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                            req_send_check.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                            # update counters
                            mod_avail_index-=1
                    else:
                        continue
        # handle the remaining gradients here to send to parameter server
        while channel_index >= 0:
            req_send_check[-1].wait()
            if pd.isnull(self.full_modules[mod_avail_index].bias):
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                grads = tmp_grad_weight.data.numpy().astype(np.float64)
                #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                req_send_check.append(req_isend)
                channel_index-=1
                mod_counters_[mod_avail_index]=2
                # update counters
                mod_avail_index-=1
            else:
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad
                # we always send bias first
                if mod_counters_[mod_avail_index] == 0:
                    grads = tmp_grad_bias.data.numpy().astype(np.float64)
                    #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                    req_send_check.append(req_isend)
                    channel_index-=1
                    mod_counters_[mod_avail_index]+=1
                elif mod_counters_[mod_avail_index] == 1:
                    grads = tmp_grad_weight.data.numpy().astype(np.float64)
                    #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                    req_send_check.append(req_isend)
                    channel_index-=1
                    mod_counters_[mod_avail_index]+=1
                    # update counters
                    mod_avail_index-=1
        return req_send_check

    def backward_normal(self, g, communicator, req_send_check, cur_step, compress_grad):
        mod_avail_index = len(self.full_modules)-1
        channel_index = self._init_channel_index-2
        mod_counters_ = [0]*len(self.full_modules)
        for i, output in reversed(list(enumerate(self.output))):
            # send layer only after the last layer is received
            req_send_check[-1].wait()
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
                # get gradient here after some sanity checks:
                tmp_grad = self.full_modules[mod_avail_index].weight.grad
                if not pd.isnull(tmp_grad):
                    grads = tmp_grad.data.numpy().astype(np.float64)
                    ######################################################################################
                    if compress_grad == 'compress':
                        _compressed_grad = compress(grads)
                        req_isend = communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                    else:
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    ######################################################################################
                    req_send_check.append(req_isend)
                    # update counters
                    mod_avail_index-=1
                    channel_index-=1
                else:
                    continue
            else:
                if output.size() == self.input[i+1].grad.size():
                    output.backward(self.input[i+1].grad.data)
                else:
                    tmp_grad_output = self.input[i+1].grad.view(output.size())
                    output.backward(tmp_grad_output)

                # since in resnet we do not use bias weight for conv layer
                if pd.isnull(self.full_modules[mod_avail_index].bias):
                    tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad

                    if not pd.isnull(tmp_grad_weight):
                        grads = tmp_grad_weight.data.numpy().astype(np.float64)
                        ######################################################################################
                        if compress_grad == 'compress':
                            _compressed_grad = compress(grads)
                            req_isend = communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                        else:
                            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        ######################################################################################
                        req_send_check.append(req_isend)
                        channel_index-=1
                        mod_counters_[mod_avail_index]=2
                        # update counters
                        mod_avail_index-=1
                    else:
                        continue
                else:
                    tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                    tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad

                    if not pd.isnull(tmp_grad_weight) and not pd.isnull(tmp_grad_bias):
                        # we always send bias first
                        if mod_counters_[mod_avail_index] == 0:
                            grads = tmp_grad_bias.data.numpy().astype(np.float64)
                            ######################################################################################
                            if compress_grad == 'compress':
                                _compressed_grad = compress(grads)
                                req_isend = communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                            else:
                                req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            ######################################################################################
                            req_send_check.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                        elif mod_counters_[mod_avail_index] == 1:
                            grads = tmp_grad_weight.data.numpy().astype(np.float64)
                            ######################################################################################
                            if compress_grad == 'compress':
                                _compressed_grad = compress(grads)
                                req_isend = communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                            else:
                                req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            ######################################################################################
                            req_send_check.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                            # update counters
                            mod_avail_index-=1
                    else:
                        continue
        # handle the remaining gradients here to send to parameter server
        while channel_index >= 0:
            req_send_check[-1].wait()
            if pd.isnull(self.full_modules[mod_avail_index].bias):
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                grads = tmp_grad_weight.data.numpy().astype(np.float64)
                ######################################################################################
                if compress_grad == 'compress':
                    _compressed_grad = compress(grads)
                    req_isend = communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                else:
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                ######################################################################################
                req_send_check.append(req_isend)
                channel_index-=1
                mod_counters_[mod_avail_index]=2
                # update counters
                mod_avail_index-=1
            else:
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad
                # we always send bias first
                if mod_counters_[mod_avail_index] == 0:
                    grads = tmp_grad_bias.data.numpy().astype(np.float64)
                    ######################################################################################
                    if compress_grad == 'compress':
                        _compressed_grad = compress(grads)
                        req_isend = communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                    else:
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    ######################################################################################
                    req_send_check.append(req_isend)
                    channel_index-=1
                    mod_counters_[mod_avail_index]+=1
                elif mod_counters_[mod_avail_index] == 1:
                    grads = tmp_grad_weight.data.numpy().astype(np.float64)
                    ######################################################################################
                    if compress_grad == 'compress':
                        _compressed_grad = compress(grads)
                        req_isend = communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                    else:
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    ######################################################################################
                    req_send_check.append(req_isend)
                    channel_index-=1
                    mod_counters_[mod_avail_index]+=1
                    # update counters
                    mod_avail_index-=1
        return req_send_check

    def backward_signal_kill(self, g, communicator, req_send_check, cur_step):
        mod_avail_index = len(self.full_modules)-1
        channel_index = self._init_channel_index-2
        mod_counters_ = [0]*len(self.full_modules)

        # should kill flag
        should_kill = False
        
        for i, output in reversed(list(enumerate(self.output))):
        ############################ killing process on workers #####################################
            for _ in range(100):
                status = MPI.Status()
                communicator.Iprobe(0, 77, status)
                if status.Get_source() == 0:
                    print("Worker {}, Cur Step: {} I'm the straggler, killing myself!".format(communicator.Get_rank(), cur_step))
                    tmp = communicator.recv(source=0, tag=77)
                    should_kill = True
                    break
            if should_kill:
                channel_index=-5
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
                if output.size() == self.input[i+1].grad.size():
                    output.backward(self.input[i+1].grad.data)
                else:
                    tmp_grad_output = self.input[i+1].grad.view(output.size())
                    output.backward(tmp_grad_output)

                # since in resnet we do not use bias weight for conv layer
                if pd.isnull(self.full_modules[mod_avail_index].bias):
                    tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad

                    if not pd.isnull(tmp_grad_weight):
                        grads = tmp_grad_weight.data.numpy().astype(np.float64)
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_send_check.append(req_isend)
                        channel_index-=1
                        mod_counters_[mod_avail_index]=2
                        # update counters
                        mod_avail_index-=1
                    else:
                        continue
                else:
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
        # handle the remaining gradients here to send to parameter server
        while channel_index >= 0:
            if pd.isnull(self.full_modules[mod_avail_index].bias):
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                grads = tmp_grad_weight.data.numpy().astype(np.float64)
                req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                req_send_check.append(req_isend)
                channel_index-=1
                mod_counters_[mod_avail_index]=2
                # update counters
                mod_avail_index-=1
            else:
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad
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
        if channel_index == -1:
            killed = False
        elif channel_index == -5:
            killed = True
        return req_send_check, killed

    @timeout_decorator.timeout(10.5, timeout_exception=StopIteration)
    def backward_timeout_kill(self, g, communicator, req_send_check, cur_step):
        mod_avail_index = len(self.full_modules)-1
        channel_index = self._init_channel_index-2
        mod_counters_ = [0]*len(self.full_modules)

        # meset request list of killed workers
        self.killed_request_list = []
        for i, output in reversed(list(enumerate(self.output))):
            # send layer only after the last layer is received
            req_send_check[-1].wait()
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
                # get gradient here after some sanity checks:
                tmp_grad = self.full_modules[mod_avail_index].weight.grad
                if not pd.isnull(tmp_grad):
                    grads = tmp_grad.data.numpy().astype(np.float64)
                    #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                    req_send_check.append(req_isend)
                    #self.killed_request_list.append(req_isend)
                    # update counters
                    mod_avail_index-=1
                    channel_index-=1
                else:
                    continue
            else:
                if output.size() == self.input[i+1].grad.size():
                    output.backward(self.input[i+1].grad.data)
                else:
                    tmp_grad_output = self.input[i+1].grad.view(output.size())
                    output.backward(tmp_grad_output)

                # since in resnet we do not use bias weight for conv layer
                if pd.isnull(self.full_modules[mod_avail_index].bias):
                    tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad

                    if not pd.isnull(tmp_grad_weight):
                        grads = tmp_grad_weight.data.numpy().astype(np.float64)
                        #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                        req_send_check.append(req_isend)
                        #self.killed_request_list.append(req_isend)
                        channel_index-=1
                        mod_counters_[mod_avail_index]=2
                        # update counters
                        mod_avail_index-=1
                    else:
                        continue
                else:
                    tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                    tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad

                    if not pd.isnull(tmp_grad_weight) and not pd.isnull(tmp_grad_bias):
                        # we always send bias first
                        if mod_counters_[mod_avail_index] == 0:
                            grads = tmp_grad_bias.data.numpy().astype(np.float64)
                            #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                            req_send_check.append(req_isend)
                            #self.killed_request_list.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                        elif mod_counters_[mod_avail_index] == 1:
                            grads = tmp_grad_weight.data.numpy().astype(np.float64)
                            #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                            req_send_check.append(req_isend)
                            #self.killed_request_list.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                            # update counters
                            mod_avail_index-=1
                    else:
                        continue
        # handle the remaining gradients here to send to parameter server
        while channel_index >= 0:
            req_send_check[-1].wait()
            if pd.isnull(self.full_modules[mod_avail_index].bias):
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                grads = tmp_grad_weight.data.numpy().astype(np.float64)
                #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                req_send_check.append(req_isend)
                #self.killed_request_list.append(req_isend)
                channel_index-=1
                mod_counters_[mod_avail_index]=2
                # update counters
                mod_avail_index-=1
            else:
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad
                # we always send bias first
                if mod_counters_[mod_avail_index] == 0:
                    grads = tmp_grad_bias.data.numpy().astype(np.float64)
                    #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                    req_send_check.append(req_isend)
                    #self.killed_request_list.append(req_isend)
                    channel_index-=1
                    mod_counters_[mod_avail_index]+=1
                elif mod_counters_[mod_avail_index] == 1:
                    grads = tmp_grad_weight.data.numpy().astype(np.float64)
                    #req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                    req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=generate_tag(layer_tag=88+channel_index, step_token=cur_step))
                    req_send_check.append(req_isend)
                    #self.killed_request_list.append(req_isend)
                    channel_index-=1
                    mod_counters_[mod_avail_index]+=1
                    # update counters
                    mod_avail_index-=1
        return req_send_check
    
    def backward_single(self, g):
        for i, output in reversed(list(enumerate(self.output))):
            #print("Backward processing, step {}".format(i))
            #print("--------------------------------------------------------")
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
            else:
                
                #print(output.size())
                #print(self.input[i+1].grad.size())
                #tmp = self.input[i+1].grad.view(output.size())
                #print(tmp.size())
                #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                
                if output.size() == self.input[i+1].grad.size():
                    output.backward(self.input[i+1].grad.data)
                else:
                    tmp_grad_output = self.input[i+1].grad.view(output.size())
                    output.backward(tmp_grad_output)

def ResNetSplit18(kill_threshold):
    return ResNetSplit(BasicBlockSplit, [2,2,2,2], kill_threshold=kill_threshold)

def ResNetSplit34():
    return ResNetSplit(BasicBlockSplit, [3,4,6,3])

def ResNetSplit50():
    return ResNetSplit(Bottleneck, [3,4,6,3])

def ResNetSplit101():
    return ResNetSplit(Bottleneck, [3,4,23,3])

def ResNetSplit152():
    return ResNetSplit(Bottleneck, [3,8,36,3])

if __name__ == "__main__":
    a = ResNetSplit18(1)
    print("Done!")